# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import traceback
from collections import defaultdict
from io import BytesIO
from typing import Any, Optional

import datasets
import numpy as np
import torch
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image
from torch.utils.data import Dataset, Sampler
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.import_utils import load_extern_object
from verl.utils.tokenizer import normalize_token_ids

logger = logging.getLogger(__name__)

_OVERLONG_FILTER_CACHE_FORMAT_VERSION = 1
_OVERLONG_FILTER_CACHE_SUBDIR = "overlong_filter_cache"

try:
    from torchdata.stateful_dataloader.stateful import Stateful
except Exception:  # pragma: no cover - torchdata is available in normal trainer environments

    class Stateful:  # type: ignore[no-redef]
        pass


def _to_jsonable(value):
    if isinstance(value, (DictConfig, ListConfig)):
        value = OmegaConf.to_container(value, resolve=True)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _to_jsonable(val) for key, val in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def collate_fn(data_list: list[dict]) -> dict:
    """
    Collate a batch of sample dicts into batched tensors and arrays.

    Args:
        data_list: List of dicts mapping feature names to torch.Tensor or other values.

    Returns:
        Dict where tensor entries are stacked into a torch.Tensor of shape
        (batch_size, \\*dims) and non-tensor entries are converted to
        np.ndarray of dtype object with shape (batch_size,).
    """
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.fromiter(val, dtype=object, count=len(val))

    return {**tensors, **non_tensors}


def _parse_per_batch_domain_weights(raw):
    if raw is None:
        return None
    if isinstance(raw, DictConfig):
        raw = dict(raw)
    elif isinstance(raw, ListConfig):
        raw = list(raw)

    if isinstance(raw, dict):
        weights = dict(raw)
    elif isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return None
        if raw.startswith("{") or raw.startswith("["):
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                weights = parsed
            elif isinstance(parsed, list):
                weights = {}
                for item in parsed:
                    if isinstance(item, dict):
                        name = item.get("name") or item.get("domain")
                        weight = item.get("weight") or item.get("value")
                        if name is None or weight is None:
                            raise ValueError(f"Invalid domain weight entry: {item!r}")
                        weights[str(name)] = float(weight)
                    elif isinstance(item, (list, tuple)) and len(item) == 2:
                        weights[str(item[0])] = float(item[1])
                    else:
                        raise ValueError(f"Invalid domain weight entry: {item!r}")
            else:
                raise ValueError("per_batch_domain_weights JSON must be dict or list")
        else:
            weights = {}
            for part in raw.split(","):
                part = part.strip()
                if not part:
                    continue
                if ":" not in part:
                    raise ValueError(
                        "per_batch_domain_weights must be 'name:weight' pairs separated by commas"
                    )
                name, value = part.split(":", 1)
                name = name.strip()
                value = value.strip()
                if not name:
                    raise ValueError("per_batch_domain_weights contains empty domain name")
                weights[name] = float(value)
    elif isinstance(raw, (list, tuple)):
        weights = {}
        for item in raw:
            if isinstance(item, DictConfig):
                item = dict(item)
            elif isinstance(item, ListConfig):
                item = list(item)

            if isinstance(item, dict):
                name = item.get("name") or item.get("domain")
                weight = item.get("weight") or item.get("value")
                if name is None or weight is None:
                    raise ValueError(f"Invalid domain weight entry: {item!r}")
                weights[str(name)] = float(weight)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                weights[str(item[0])] = float(item[1])
            elif isinstance(item, str):
                if ":" not in item:
                    raise ValueError(f"Invalid domain weight entry: {item!r}")
                name, value = item.split(":", 1)
                weights[name.strip()] = float(value.strip())
            else:
                raise ValueError(f"Invalid domain weight entry: {item!r}")
    else:
        raise TypeError(f"Unsupported per_batch_domain_weights type: {type(raw).__name__}")

    filtered = {}
    for name, value in weights.items():
        try:
            weight = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"Invalid weight for domain '{name}': {value!r}") from None
        if not np.isfinite(weight):
            raise ValueError(f"Non-finite weight for domain '{name}': {weight!r}")
        if weight > 0:
            filtered[str(name)] = weight
    if not filtered:
        raise ValueError("per_batch_domain_weights must contain at least one positive weight")
    total = sum(filtered.values())
    return {name: weight / total for name, weight in filtered.items()}


class PerBatchDomainSampler(Sampler[int], Stateful):
    def __init__(self, domain2indices, domain_weights, batch_size, total_size, seed=18, shuffle=True):
        self.domain_names = list(domain_weights.keys())
        self.domain_weights = [domain_weights[name] for name in self.domain_names]
        self.domain2indices = {
            name: np.array(domain2indices[name], dtype=np.int64, copy=True) for name in self.domain_names
        }
        self.batch_size = int(batch_size)
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.rng = np.random.default_rng(self.seed)
        self._batch_counts = self._compute_batch_counts()
        self._domain_perm = {}
        self._domain_pos = {}
        self.epoch = 0
        self._batch_cursor = 0
        self._epoch_started = False
        for name in self.domain_names:
            self._reshuffle(name)
        self.total_size = int(total_size)
        self._num_samples = (self.total_size // self.batch_size) * self.batch_size

    def _compute_batch_counts(self):
        raw = np.array(self.domain_weights, dtype=np.float64) * self.batch_size
        counts = np.floor(raw).astype(np.int64)
        remainder = raw - counts
        leftover = int(self.batch_size - counts.sum())
        if leftover > 0:
            order = np.argsort(-remainder)
            for i in range(leftover):
                counts[order[i % len(order)]] += 1
        return counts.tolist()

    def _reshuffle(self, name):
        if self.shuffle:
            self._domain_perm[name] = self.rng.permutation(self.domain2indices[name])
        else:
            self._domain_perm[name] = self.domain2indices[name].copy()
        self._domain_pos[name] = 0

    def _take(self, name, count):
        out = []
        while len(out) < count:
            perm = self._domain_perm[name]
            pos = self._domain_pos[name]
            remaining = len(perm) - pos
            take = min(count - len(out), remaining)
            if take > 0:
                out.extend(perm[pos : pos + take])
                pos += take
            if pos >= len(perm):
                self._reshuffle(name)
                pos = self._domain_pos[name]
            self._domain_pos[name] = pos
        return out

    def _start_epoch(self):
        if self.shuffle:
            self.rng = np.random.default_rng(self.seed + self.epoch)
        for name in self.domain_names:
            self._reshuffle(name)
        self._batch_cursor = 0
        self._epoch_started = True

    def __iter__(self):
        if not self._epoch_started:
            self._start_epoch()
        num_batches = self._num_samples // self.batch_size
        for _ in range(self._batch_cursor, num_batches):
            batch = []
            for name, count in zip(self.domain_names, self._batch_counts, strict=True):
                if count:
                    batch.extend(self._take(name, int(count)))
            if self.shuffle:
                self.rng.shuffle(batch)
            self._batch_cursor += 1
            for idx in batch:
                yield int(idx)
        self.epoch += 1
        self._epoch_started = False

    def __len__(self):
        return self._num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)
        self._batch_cursor = 0
        self._epoch_started = False

    def state_dict(self):
        return {
            "domain_names": list(self.domain_names),
            "epoch": self.epoch,
            "batch_cursor": self._batch_cursor,
            "epoch_started": self._epoch_started,
            "rng_state": self.rng.bit_generator.state,
            "domain_pos": dict(self._domain_pos),
            "domain_perm": {k: v.copy() for k, v in self._domain_perm.items()},
            "shuffle": self.shuffle,
            "seed": self.seed,
            "batch_size": self.batch_size,
        }

    def _validate_state_dict(self, state_dict):
        if list(state_dict.get("domain_names", [])) != list(self.domain_names):
            raise ValueError("PerBatchDomainSampler domain names do not match the saved state")
        saved_batch_size = int(state_dict.get("batch_size", self.batch_size))
        if saved_batch_size != self.batch_size:
            raise ValueError(
                f"PerBatchDomainSampler batch size changed from {saved_batch_size} to {self.batch_size}"
            )
        domain_pos = state_dict.get("domain_pos", {})
        domain_perm = state_dict.get("domain_perm", {})
        restored_domain_perm = {}
        restored_domain_pos = {}
        for name in self.domain_names:
            if name in domain_perm:
                perm = np.array(domain_perm[name], dtype=np.int64, copy=True)
                expected_len = len(self.domain2indices[name])
                if len(perm) != expected_len:
                    raise ValueError(
                        f"PerBatchDomainSampler domain '{name}' size changed from {len(perm)} to {expected_len}"
                    )
                if perm.size and (int(perm.min()) < 0 or int(perm.max()) >= self.total_size):
                    raise ValueError(
                        f"PerBatchDomainSampler domain '{name}' saved indices are incompatible with "
                        f"current dataset size {self.total_size}"
                    )
                restored_domain_perm[name] = perm
            if name in domain_pos:
                pos = int(domain_pos[name])
                perm_len = len(restored_domain_perm.get(name, self._domain_perm[name]))
                if pos < 0 or pos > perm_len:
                    raise ValueError(
                        f"PerBatchDomainSampler domain '{name}' saved position {pos} is outside "
                        f"permutation length {perm_len}"
                    )
                restored_domain_pos[name] = pos
        return restored_domain_perm, restored_domain_pos

    def validate_state_dict(self, state_dict):
        self._validate_state_dict(state_dict)

    def load_state_dict(self, state_dict):
        restored_domain_perm, restored_domain_pos = self._validate_state_dict(state_dict)
        self.epoch = int(state_dict.get("epoch", 0))
        self._batch_cursor = int(state_dict.get("batch_cursor", 0))
        self._epoch_started = bool(state_dict.get("epoch_started", False))
        self.shuffle = bool(state_dict.get("shuffle", self.shuffle))
        self.seed = int(state_dict.get("seed", self.seed))
        rng_state = state_dict.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state
        for name in self.domain_names:
            if name in restored_domain_perm:
                self._domain_perm[name] = restored_domain_perm[name]
            if name in restored_domain_pos:
                self._domain_pos[name] = restored_domain_pos[name]


class RLHFDataset(Dataset):
    """
    Load and preprocess RLHF data from Parquet files.

    - Caches files locally.
    - Reads into a HuggingFace Dataset and tokenizes prompts.
    - Optionally handles images/videos via a ProcessorMixin.
    - Filters prompts over a max length.
    - Supports resuming from checkpoints.

    Args:
        data_files (str or list): Path(s) to Parquet file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        if not isinstance(data_files, list | ListConfig):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.tokenizer = tokenizer
        self.processor = processor
        self.max_samples = max_samples
        self.config = config

        self.cache_dir = os.path.expanduser(config.get("cache_dir", "~/.cache/verl/rlhf"))
        self.prompt_key = config.get("prompt_key", "prompt")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get("image_patch_size", 14)
        self.image_root = config.get("image_root", None)
        self.min_pixels = config.get("min_pixels", None)
        self.max_pixels = config.get("max_pixels", None)
        self.normalize_bbox_to_1000 = config.get("normalize_bbox_to_1000", False)
        self.add_fixation_post_prompt = config.get("add_fixation_post_prompt", False)
        self.max_prompt_length = config.get("max_prompt_length", 1024)
        self.return_raw_chat = config.get("return_raw_chat", False)
        self.return_full_prompt = config.get("return_full_prompt", False)
        self.truncation = config.get("truncation", "error")
        self.filter_overlong_prompts = config.get("filter_overlong_prompts", True)
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.system_prompt = config.get("system_prompt", None)
        if isinstance(self.system_prompt, str) and os.path.exists(self.system_prompt):
            with open(self.system_prompt, encoding="utf-8") as f:
                self.system_prompt = f.read()

        self.tool_config_path = config.get("tool_config_path", None)
        self.tool_schemas = None
        if self.tool_config_path:
            try:
                from verl.tools.utils.tool_registry import initialize_tools_from_config

                tool_list = initialize_tools_from_config(self.tool_config_path)
                # match ToolAgentLoop behaviour: model_dump to plain dicts
                self.tool_schemas = [
                    tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list
                ]
            except Exception as e:
                logger.warning("Failed to initialize tools from %s: %s", self.tool_config_path, e)
                self.tool_schemas = None

        self.num_workers = config.get("filter_overlong_prompts_workers", max(1, os.cpu_count() // 4))
        self.num_workers = min(self.num_workers, os.cpu_count()) if self.num_workers is not None else None
        self.use_shm = config.get("use_shm", False)
        self.chat_template_func = config.get("chat_template_func", None)
        self.need_tools_kwargs = config.get("need_tools_kwargs", False)
        self.filter_prompts = config.get("filter_prompts", True)
        self.serialize_dataset = False
        self.return_multi_modal_inputs = config.get("return_multi_modal_inputs", True)
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.per_batch_domain_weights = _parse_per_batch_domain_weights(
            config.get("per_batch_domain_weights", None)
        )
        self.domain2indices = None
        self.domain_weights = None

        self._download()
        self._read_files_and_tokenize()
        self._prepare_domain_sampling()

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local

        data_files = self.data_files if not use_origin_parquet else self.original_data_files
        for i, parquet_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir, use_shm=self.use_shm)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.data_files:
            # read files and cache
            if parquet_file.endswith(".parquet"):
                dataframe = datasets.load_dataset("parquet", data_files=parquet_file, cache_dir=self.cache_dir)["train"]
            elif parquet_file.endswith(".json"):
                dataframe = datasets.load_dataset("json", data_files=parquet_file, cache_dir=self.cache_dir)["train"]
            elif parquet_file.endswith(".jsonl"):
                try:
                    dataframe = datasets.load_dataset("json", data_files=parquet_file, cache_dir=self.cache_dir)["train"]
                except Exception as e:
                    logger.warning("Falling back to native JSONL loader for %s: %s", parquet_file, e)
                    dataframe = self._load_native_jsonl(parquet_file)
            else:
                raise ValueError(f"Unsupported file format: {parquet_file}")
            dataframes.append(dataframe)
        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(dataframes)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.select(indices.tolist())
            print(f"selected {self.max_samples} random samples out of {total}")

        self.dataframe = self.maybe_filter_out_long_prompts(self.dataframe)

    def _prepare_domain_sampling(self):
        if not self.per_batch_domain_weights:
            return
        if "ability" not in self.dataframe.column_names:
            raise KeyError("per_batch_domain_weights requires an 'ability' column in the dataset")
        abilities = np.asarray(self.dataframe["ability"], dtype=object)
        if abilities.size == 0:
            raise ValueError("Dataset is empty; cannot build per-batch domain sampler")

        available_domains = set(np.unique(abilities))
        filtered_weights = {}
        for domain, weight in self.per_batch_domain_weights.items():
            if domain not in available_domains:
                logger.warning("per_batch_domain_weights domain '%s' not found in dataset", domain)
                continue
            filtered_weights[domain] = weight
        if not filtered_weights:
            raise ValueError("No per_batch_domain_weights domains were found in the dataset")

        total = sum(filtered_weights.values())
        self.domain_weights = {name: weight / total for name, weight in filtered_weights.items()}
        self.domain2indices = {
            domain: np.where(abilities == domain)[0] for domain in self.domain_weights
        }

        missing = available_domains - set(self.domain_weights.keys())
        if missing:
            logger.warning(
                "per_batch_domain_weights does not include domains %s; they will be ignored",
                sorted(missing),
            )

    def build_domain_sampler(self, batch_size, seed=18, shuffle=True):
        if self.domain2indices is None or self.domain_weights is None:
            raise RuntimeError("per_batch_domain_weights is not configured for this dataset")
        return PerBatchDomainSampler(
            domain2indices=self.domain2indices,
            domain_weights=self.domain_weights,
            batch_size=batch_size,
            total_size=len(self.dataframe),
            seed=seed,
            shuffle=shuffle,
        )

    def _load_native_jsonl(self, data_file: str) -> datasets.Dataset:
        records = []
        with open(data_file, encoding="utf-8") as f:
            for line_number, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL row in {data_file}:{line_number}") from e
                records.append(self._prepare_jsonl_record_for_arrow(record))

        return datasets.Dataset.from_list(records)

    def _prepare_jsonl_record_for_arrow(self, record: dict) -> dict:
        record = dict(record)
        if self.prompt_key in record:
            record[self.prompt_key] = self._normalize_prompt_for_arrow(record[self.prompt_key])
        return record

    def _normalize_prompt_for_arrow(self, prompt: Any) -> list[dict]:
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        if isinstance(prompt, dict):
            return [self._ensure_message_dict(prompt)]

        if isinstance(prompt, (list, tuple, ListConfig)):
            if all(isinstance(item, str) for item in prompt):
                return [{"role": "user", "content": "\n".join(prompt)}]
            messages = []
            for idx, item in enumerate(prompt):
                default_role = "user" if idx == 0 else "assistant"
                if isinstance(item, dict):
                    messages.append(self._ensure_message_dict(item, default_role=default_role))
                else:
                    messages.append({"role": default_role, "content": str(item)})
            return messages

        return [{"role": "user", "content": str(prompt)}]

    def _get_overlong_filter_cache_root(self) -> Optional[str]:
        cache_root = os.environ.get("HF_DATASETS_CACHE")
        if not cache_root:
            logger.warning("HF_DATASETS_CACHE is not set; skipping overlong prompt filter cache.")
            print("Overlong filter cache bypassed: HF_DATASETS_CACHE is not set.", file=sys.stderr, flush=True)
            return None
        try:
            cache_root = os.path.join(os.path.expanduser(cache_root), _OVERLONG_FILTER_CACHE_SUBDIR)
            os.makedirs(cache_root, exist_ok=True)
            return cache_root
        except OSError as exc:
            logger.warning("Failed to prepare overlong prompt filter cache at %s: %s", cache_root, exc)
            print(
                f"Overlong filter cache bypassed: failed to prepare cache directory '{cache_root}': {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    @staticmethod
    def _build_file_cache_descriptor(requested_path: str, resolved_path: str) -> dict:
        stat_result = os.stat(resolved_path)
        return {
            "requested_path": requested_path,
            "resolved_path": resolved_path,
            "size": int(stat_result.st_size),
            "mtime_ns": int(stat_result.st_mtime_ns),
        }

    @staticmethod
    def _build_processing_identity(obj) -> Optional[dict]:
        if obj is None:
            return None

        init_kwargs = getattr(obj, "init_kwargs", None)
        tokenizer = getattr(obj, "tokenizer", None)
        image_processor = getattr(obj, "image_processor", None)
        identity = {
            "class_name": obj.__class__.__name__,
            "name_or_path": getattr(obj, "name_or_path", None),
            "init_name_or_path": init_kwargs.get("name_or_path") if isinstance(init_kwargs, dict) else None,
            "chat_template": getattr(obj, "chat_template", None),
        }
        if tokenizer is not None:
            identity["tokenizer"] = {
                "class_name": tokenizer.__class__.__name__,
                "name_or_path": getattr(tokenizer, "name_or_path", None),
                "chat_template": getattr(tokenizer, "chat_template", None),
            }
        if image_processor is not None:
            identity["image_processor"] = {
                "class_name": image_processor.__class__.__name__,
            }
        return _to_jsonable(identity)

    def _build_overlong_filter_cache_spec(self, dataframe: datasets.Dataset) -> dict:
        data_files = []
        for requested_path, resolved_path in zip(self.original_data_files, self.data_files, strict=True):
            data_files.append(
                self._build_file_cache_descriptor(requested_path=requested_path, resolved_path=resolved_path)
            )

        return {
            "format_version": _OVERLONG_FILTER_CACHE_FORMAT_VERSION,
            "data_files": data_files,
            "max_prompt_length": int(self.max_prompt_length),
            "min_pixels": None if self.min_pixels is None else int(self.min_pixels),
            "max_pixels": None if self.max_pixels is None else int(self.max_pixels),
            "prompt_key": self.prompt_key,
            "image_key": self.image_key,
            "video_key": self.video_key,
            "add_fixation_post_prompt": bool(self.add_fixation_post_prompt),
            "system_prompt": self.system_prompt,
            "apply_chat_template_kwargs": _to_jsonable(self.apply_chat_template_kwargs),
            "image_root": self.image_root,
            "image_patch_size": None if self.image_patch_size is None else int(self.image_patch_size),
            "tool_schemas": _to_jsonable(self.tool_schemas),
            "has_processor": self.processor is not None,
            "tokenizer": self._build_processing_identity(self.tokenizer),
            "processor": self._build_processing_identity(self.processor),
            "source_length": int(len(dataframe)),
            "dataframe_fingerprint": getattr(dataframe, "_fingerprint", None),
            "max_samples": None if self.max_samples is None else int(self.max_samples),
            "shuffle": bool(self.shuffle),
            "seed": self.seed,
        }

    @staticmethod
    def _hash_overlong_filter_cache_spec(filter_spec: dict) -> str:
        payload = json.dumps(_to_jsonable(filter_spec), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _get_overlong_filter_cache_paths(self, cache_key: str) -> Optional[dict[str, str]]:
        cache_root = self._get_overlong_filter_cache_root()
        if cache_root is None:
            return None
        cache_dir = os.path.join(cache_root, cache_key)
        return {
            "cache_dir": cache_dir,
            "meta_path": os.path.join(cache_dir, "meta.json"),
            "indices_path": os.path.join(cache_dir, "kept_indices.npy"),
            "lock_path": os.path.join(cache_root, f"{cache_key}.lock"),
        }

    @staticmethod
    def _validate_kept_indices(kept_indices: np.ndarray, source_length: int) -> np.ndarray:
        if kept_indices.ndim != 1:
            raise ValueError(f"Expected kept indices to be 1D, got shape {kept_indices.shape!r}")
        if kept_indices.dtype.kind not in {"i", "u"}:
            raise ValueError(f"Expected integer kept indices, got dtype {kept_indices.dtype}")
        if kept_indices.size:
            if kept_indices.min() < 0 or kept_indices.max() >= source_length:
                raise ValueError("Kept indices are out of bounds for the source dataset")
            if not np.all(kept_indices[1:] >= kept_indices[:-1]):
                raise ValueError("Kept indices must be sorted in non-decreasing order")
        return kept_indices.astype(np.int64, copy=False)

    def _load_overlong_filter_cache(
        self,
        cache_paths: Optional[dict[str, str]],
        filter_spec: dict,
    ) -> Optional[np.ndarray]:
        if cache_paths is None:
            return None

        meta_path = cache_paths["meta_path"]
        indices_path = cache_paths["indices_path"]
        if not os.path.exists(meta_path) or not os.path.exists(indices_path):
            return None

        try:
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if meta.get("format_version") != _OVERLONG_FILTER_CACHE_FORMAT_VERSION:
                raise ValueError(f"Unsupported cache format version: {meta.get('format_version')!r}")
            if meta.get("filter_spec") != filter_spec:
                raise ValueError("Stored filter spec does not match the current configuration")

            kept_indices = np.load(indices_path, allow_pickle=False)
            kept_indices = self._validate_kept_indices(kept_indices, int(filter_spec["source_length"]))

            if int(meta.get("source_length", -1)) != int(filter_spec["source_length"]):
                raise ValueError("Stored source length does not match current dataset length")
            if int(meta.get("kept_count", -1)) != int(kept_indices.size):
                raise ValueError("Stored kept_count does not match cached kept indices")

            return kept_indices
        except Exception as exc:
            logger.warning("Ignoring invalid overlong prompt filter cache at %s: %s", cache_paths["cache_dir"], exc)
            print(
                f"Overlong filter cache bypassed: invalid cache at '{cache_paths['cache_dir']}': {exc}",
                file=sys.stderr,
                flush=True,
            )
            return None

    def _save_overlong_filter_cache(
        self,
        cache_paths: dict[str, str],
        filter_spec: dict,
        cache_key: str,
        kept_indices: np.ndarray,
    ) -> None:
        os.makedirs(cache_paths["cache_dir"], exist_ok=True)

        meta = {
            "format_version": _OVERLONG_FILTER_CACHE_FORMAT_VERSION,
            "cache_key": cache_key,
            "source_length": int(filter_spec["source_length"]),
            "kept_count": int(kept_indices.size),
            "filter_spec": filter_spec,
        }

        fd, meta_tmp_path = tempfile.mkstemp(dir=cache_paths["cache_dir"], prefix="meta.", suffix=".tmp")
        os.close(fd)
        try:
            with open(meta_tmp_path, "w", encoding="utf-8") as handle:
                json.dump(meta, handle, sort_keys=True)
            os.replace(meta_tmp_path, cache_paths["meta_path"])
        finally:
            if os.path.exists(meta_tmp_path):
                os.remove(meta_tmp_path)

        fd, indices_tmp_path = tempfile.mkstemp(dir=cache_paths["cache_dir"], prefix="kept_indices.", suffix=".tmp")
        os.close(fd)
        try:
            with open(indices_tmp_path, "wb") as handle:
                np.save(handle, kept_indices, allow_pickle=False)
            os.replace(indices_tmp_path, cache_paths["indices_path"])
        finally:
            if os.path.exists(indices_tmp_path):
                os.remove(indices_tmp_path)

    @staticmethod
    def _get_overlong_filter_temp_index_column(dataframe: datasets.Dataset) -> str:
        column_name = "__overlong_filter_cache_idx__"
        while column_name in dataframe.column_names:
            column_name = f"_{column_name}"
        return column_name

    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if not self.filter_overlong_prompts:
            return dataframe

        tokenizer = self.tokenizer
        processor = self.processor
        image_key = self.image_key
        video_key = self.video_key
        filter_spec = None
        cache_key = None
        cache_paths = None

        try:
            filter_spec = self._build_overlong_filter_cache_spec(dataframe)
            cache_key = self._hash_overlong_filter_cache_spec(filter_spec)
            cache_paths = self._get_overlong_filter_cache_paths(cache_key)
            cached_indices = self._load_overlong_filter_cache(cache_paths, filter_spec)
            if cached_indices is not None:
                print(
                    f"Overlong filter cache hit: using kept indices from '{cache_paths['cache_dir']}'.",
                    file=sys.stderr,
                    flush=True,
                )
                return dataframe.select(cached_indices.tolist())

            if cache_paths is not None:
                print(
                    f"Overlong filter cache miss: computing kept indices for key {cache_key}.",
                    file=sys.stderr,
                    flush=True,
                )
        except Exception as exc:
            logger.warning("Failed to prepare overlong prompt filter cache: %s", exc)
            print(
                f"Overlong filter cache bypassed: failed to prepare cache metadata: {exc}",
                file=sys.stderr,
                flush=True,
            )
            filter_spec = None
            cache_key = None
            cache_paths = None

        if processor is not None:
            from verl.utils.dataset.vision_utils import process_image, process_video

            def doc2len(doc) -> int:
                try:
                    messages = self._build_messages(doc, remove_prompt=False)
                    # pass tool schemas if available so the processor can format prompts
                    apply_kwargs = dict(**self.apply_chat_template_kwargs)
                    if self.tool_schemas is not None:
                        apply_kwargs["tools"] = self.tool_schemas

                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False, **apply_kwargs
                    )
                    if image_key in doc and doc[image_key]:
                        images = [
                            process_image(
                                self._image_to_content_block(image),
                                image_patch_size=self.image_patch_size,
                            )
                            for image in doc[image_key]
                        ]
                    else:
                        images = None

                    if video_key in doc and doc[video_key]:
                        videos, video_metadata = zip(
                            *[
                                process_video(
                                    video, image_patch_size=self.image_patch_size, return_video_metadata=True
                                )
                                for video in doc[video_key]
                            ],
                            strict=True,
                        )
                        videos = list(videos)
                        video_metadata = list(video_metadata)
                        videos_kwargs = {"video_metadata": video_metadata, "do_sample_frames": False}
                    else:
                        videos = None
                        videos_kwargs = {}

                    if images is None and videos is None:
                        # only text prompt
                        return len(
                            processor.tokenizer(
                                text=raw_prompt,
                                add_special_tokens=False,  # avoid adding special tokens
                                return_attention_mask=False,
                            )["input_ids"]
                        )
                    else:
                        # multi-modal prompt
                        return len(
                            processor(text=[raw_prompt], images=images, videos=videos, videos_kwargs=videos_kwargs)[
                                "input_ids"
                            ][0]
                        )
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    traceback.print_exc()
                    return self.max_prompt_length + 1

        else:

            def doc2len(doc) -> int:
                try:
                    apply_kwargs = dict(**self.apply_chat_template_kwargs)
                    if self.tool_schemas is not None:
                        apply_kwargs["tools"] = self.tool_schemas

                    # Keep explicit tokenization to avoid transformers version default changes.
                    apply_kwargs.pop("tokenize", None)
                    apply_kwargs.pop("return_dict", None)
                    apply_kwargs.pop("return_tensors", None)

                    messages = self._build_messages(doc, remove_prompt=False)
                    tokenized_prompt = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=True, **apply_kwargs
                    )
                    return len(normalize_token_ids(tokenized_prompt))
                except Exception:
                    print("Error processing one of the samples, skipping...")
                    traceback.print_exc()
                    return self.max_prompt_length + 1

        print("Start filtering overlong prompts...", file=sys.stderr, flush=True)
        len_before = len(dataframe)
        index_column = self._get_overlong_filter_temp_index_column(dataframe)
        dataframe_with_index = dataframe.add_column(index_column, list(range(len_before)))

        dataframe = dataframe_with_index.filter(
            lambda doc: doc2len(doc) <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        kept_indices = np.asarray(dataframe[index_column], dtype=np.int64)
        dataframe = dataframe.remove_columns([index_column])

        if cache_paths is not None and filter_spec is not None and cache_key is not None:
            try:
                from filelock import FileLock

                with FileLock(cache_paths["lock_path"], timeout=60):
                    cached_indices = self._load_overlong_filter_cache(cache_paths, filter_spec)
                    if cached_indices is not None:
                        print(
                            f"Overlong filter cache hit after compute: reusing kept indices from '{cache_paths['cache_dir']}'.",
                            file=sys.stderr,
                            flush=True,
                        )
                        return dataframe_with_index.remove_columns([index_column]).select(cached_indices.tolist())

                    self._save_overlong_filter_cache(cache_paths, filter_spec, cache_key, kept_indices)
                    print(
                        f"Overlong filter cache saved: wrote kept indices to '{cache_paths['cache_dir']}'.",
                        file=sys.stderr,
                        flush=True,
                    )
            except Exception as exc:
                logger.warning("Failed to save overlong prompt filter cache at %s: %s", cache_paths["cache_dir"], exc)
                print(
                    f"Overlong filter cache bypassed: failed to save cache at '{cache_paths['cache_dir']}': {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        print(f"filter dataset len: {len(dataframe)}")
        print(f"Filtered {len_before - len(dataframe)} overlong prompts.", file=sys.stderr, flush=True)
        return dataframe

    def resume_dataset_state(self):
        self.serialize_dataset = not hasattr(self, "original_data_files")
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r"old dataloader ckpt file is used, please train from scratch for better ckpt performance")

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if "dataframe" in state:
                del state["dataframe"]
            return state

        return self.__dict__.copy()

    def __len__(self):
        return len(self.dataframe)

    def _resolve_image_path(self, image: str) -> str:
        if self.image_root and not os.path.isabs(image):
            return os.path.join(self.image_root, image)
        return image

    def _image_to_content_block(self, image: Any) -> dict:
        if isinstance(image, Image.Image):
            block = {"type": "image", "image": image.convert("RGB")}
            if self.min_pixels is not None:
                block["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                block["max_pixels"] = self.max_pixels
            return block
        if isinstance(image, str):
            block = {"type": "image", "image": self._resolve_image_path(image)}
            if self.min_pixels is not None:
                block["min_pixels"] = self.min_pixels
            if self.max_pixels is not None:
                block["max_pixels"] = self.max_pixels
            return block
        if isinstance(image, dict):
            block = dict(image)
            if "bytes" in block:
                block["image"] = Image.open(BytesIO(block["bytes"]))
            block.setdefault("type", "image")
            if self.min_pixels is not None:
                block.setdefault("min_pixels", self.min_pixels)
            if self.max_pixels is not None:
                block.setdefault("max_pixels", self.max_pixels)
            return block
        raise TypeError(f"image must be str, dict or PIL.Image, unsupported image type: {type(image)}")

    def _normalize_messages(self, raw_messages) -> list[dict]:
        if isinstance(raw_messages, dict):
            messages = [self._ensure_message_dict(raw_messages)]
        elif isinstance(raw_messages, str):
            messages = [self._ensure_message_dict({"role": "user", "content": raw_messages})]
        elif isinstance(raw_messages, (list, tuple, ListConfig)):
            messages = []
            for idx, message in enumerate(raw_messages):
                default_role = "user" if idx == 0 else "assistant"
                if isinstance(message, dict):
                    messages.append(self._ensure_message_dict(message, default_role=default_role))
                elif isinstance(message, str):
                    messages.append(self._ensure_message_dict({"role": default_role, "content": message}))
                else:
                    messages.append(self._ensure_message_dict({"role": default_role, "content": str(message)}))
        else:
            raise TypeError(
                "Unsupported prompt payload type. Expected dict, list, tuple, or str, "
                f"but received {type(raw_messages).__name__}"
            )

        normalized = [copy.deepcopy(message) for message in messages]
        if self.system_prompt and (not normalized or normalized[0].get("role") != "system"):
            normalized.insert(0, {"role": "system", "content": self.system_prompt})
        return normalized

    @staticmethod
    def _ensure_message_dict(message: dict, default_role: str = "user") -> dict:
        msg = dict(message)
        msg.setdefault("role", default_role)
        if "content" not in msg:
            raise KeyError("Message dict must contain a 'content' field")
        return msg

    def _maybe_append_fixation_post_prompt(self, messages: list[dict], example: dict) -> None:
        if not self.add_fixation_post_prompt:
            return

        fixation_post_prompt = self._format_fixation_post_prompt(example.get("fixations"))
        if fixation_post_prompt is None:
            return

        for message in messages:
            if message.get("role") != "user":
                continue
            message["content"] = self._append_text_to_message_content(message.get("content"), fixation_post_prompt)
            return

    @staticmethod
    def _format_fixation_post_prompt(fixations) -> Optional[str]:
        if not isinstance(fixations, (list, tuple)) or len(fixations) == 0:
            return None

        formatted_points = []
        for idx, point in enumerate(fixations, start=1):
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                continue
            try:
                x = int(round(float(point[0])))
                y = int(round(float(point[1])))
            except (TypeError, ValueError):
                continue
            formatted_points.append(f"{idx}. [{x}, {y}]")

        if not formatted_points:
            return None

        return (
            "Here are some image regions that you may want to examine to help you answer:\n"
            + "\n".join(formatted_points)
        )

    @staticmethod
    def _append_text_to_message_content(content, extra_text: str):
        if isinstance(content, str):
            separator = "\n\n" if content else ""
            return f"{content}{separator}{extra_text}"

        if isinstance(content, list):
            updated_content = []
            last_text_idx = None
            for idx, item in enumerate(content):
                if isinstance(item, dict):
                    copied_item = dict(item)
                    updated_content.append(copied_item)
                    if copied_item.get("type") == "text" and isinstance(copied_item.get("text"), str):
                        last_text_idx = idx
                else:
                    updated_content.append(item)

            if last_text_idx is not None:
                base_text = updated_content[last_text_idx].get("text", "")
                separator = "\n\n" if base_text else ""
                updated_content[last_text_idx]["text"] = f"{base_text}{separator}{extra_text}"
            else:
                updated_content.append({"type": "text", "text": extra_text})
            return updated_content

        if content is None:
            return extra_text

        return f"{str(content)}\n\n{extra_text}"

    def _prepare_image_metadata(self, row_dict: dict) -> None:
        row_images = row_dict.get(self.image_key) or []
        if not row_images:
            return

        extra_info = row_dict.setdefault("extra_info", {})
        if extra_info is None:
            extra_info = {}
            row_dict["extra_info"] = extra_info
        extra_info.setdefault("images", row_images)

        image_sizes = []
        for image in row_images:
            image_size = None
            try:
                if isinstance(image, Image.Image):
                    image_size = (image.width, image.height)
                elif isinstance(image, str):
                    with Image.open(self._resolve_image_path(image)) as img:
                        image_size = (img.width, img.height)
                elif isinstance(image, dict):
                    raw_image = image.get("image")
                    if isinstance(raw_image, Image.Image):
                        image_size = (raw_image.width, raw_image.height)
                    elif "bytes" in image:
                        with Image.open(BytesIO(image["bytes"])) as img:
                            image_size = (img.width, img.height)
            except Exception:
                image_size = None
            image_sizes.append(image_size)

        if any(image_sizes):
            extra_info.setdefault("image_sizes", image_sizes)
        if self.normalize_bbox_to_1000:
            extra_info.setdefault("normalize_bbox_to_1000", True)

    def _build_messages(self, example: dict, *, remove_prompt: bool = True):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", **image}
        - <video>: {"type": "video", **video}

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        if self.prompt_key not in example:
            raise KeyError(f"Prompt key '{self.prompt_key}' not found in example")

        raw_messages = example.pop(self.prompt_key) if remove_prompt else example[self.prompt_key]
        messages = self._normalize_messages(raw_messages)
        self._maybe_append_fixation_post_prompt(messages, example)

        # When concatenating image and video datasets, pop will return None for image or video sample
        images = example.pop(self.image_key, None) if remove_prompt else example.get(self.image_key, None)
        videos = example.pop(self.video_key, None) if remove_prompt else example.get(self.video_key, None)
        images = images or []
        videos = videos or []

        image_offset, video_offset = 0, 0
        for message in messages:
            if not images and not videos:
                continue

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), f"image_offset {image_offset} >= len(images) {len(images)}"
                    content_list.append(self._image_to_content_block(images[image_offset]))
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    content_list.append({"type": "video", **videos[video_offset]})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        """For rollout, apply_chat_template has been moved to AgentLoop, so we only return raw_prompt here."""
        row_dict: dict = self.dataframe[item]
        self._prepare_image_metadata(row_dict)
        row_dict["raw_prompt"] = self._build_messages(row_dict)

        # TODO(wuxibin): We still need a dummy tensor to make sure DataProto.batch is not empty.
        # Remove this after deprecate DataProto by TensorDict.
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)

        # add index for each prompt
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = dict()
        index = row_dict.get("extra_info", {}).get("index", 0)
        tools_kwargs = row_dict.get("extra_info", {}).get("tools_kwargs", {})
        interaction_kwargs = row_dict.get("extra_info", {}).get("interaction_kwargs", {})
        need_tools_kwargs = row_dict.get("extra_info", {}).get("need_tools_kwargs", self.need_tools_kwargs)
        if need_tools_kwargs and not tools_kwargs:
            logger.warning("tools_kwargs is empty for index %s, data source: %s", index, row_dict["data_source"])
        row_dict["index"] = index
        row_dict["tools_kwargs"] = tools_kwargs
        row_dict["interaction_kwargs"] = interaction_kwargs
        return row_dict

    @staticmethod
    def _apply_default_image_pixel_bounds(
        messages: list[dict], min_pixels: Optional[int], max_pixels: Optional[int]
    ) -> list[dict]:
        if min_pixels is None and max_pixels is None:
            return messages

        updated_messages = []
        for message in messages:
            updated_message = dict(message)
            content = updated_message.get("content")
            if isinstance(content, list):
                updated_content = []
                for block in content:
                    if isinstance(block, dict):
                        block = dict(block)
                        if block.get("type") == "image" or "image" in block:
                            if min_pixels is not None:
                                block.setdefault("min_pixels", min_pixels)
                            if max_pixels is not None:
                                block.setdefault("max_pixels", max_pixels)
                    updated_content.append(block)
                updated_message["content"] = updated_content
            elif isinstance(content, dict) and (content.get("type") == "image" or "image" in content):
                content = dict(content)
                if min_pixels is not None:
                    content.setdefault("min_pixels", min_pixels)
                if max_pixels is not None:
                    content.setdefault("max_pixels", max_pixels)
                updated_message["content"] = content
            updated_messages.append(updated_message)
        return updated_messages

    @classmethod
    async def process_vision_info(
        cls,
        messages: list[dict],
        image_patch_size,
        config: DictConfig,
    ) -> tuple[list[Image.Image], list[tuple[torch.Tensor, dict]]]:
        """Extract images and videos from messages.

        This method is called by AgentLoop (e.g SingleTurnAgentLoop) before apply_chat_template to
        the `raw_prompt` from dataset. User may customize RLHFDataset and override this method to
        support custom vision extraction.

        >>> messages = kwargs["raw_prompt"]
        >>> images, videos = RLHFDataset.process_vision_info(messages, image_patch_size)
        >>> videos, video_metadatas = zip(*videos)
        >>> raw_prompt = processor.apply_chat_template(messages, tokenize=False)
        >>> inputs = processor(text=[raw_prompt], images=images, videos=videos,
        ...                    video_metadata=video_metadatas, do_sample_frames=False)

        Args:
            messages: List of messages from dataset `raw_prompt`.
            image_patch_size: Image patch size for processor.
            config: Config for dataset.

        Returns:
            images: List of images.
            videos: List of videos, each video is a tuple of (video_tensor, video_metadata).
        """
        from qwen_vl_utils import process_vision_info

        min_pixels = config.get("min_pixels", None) if config is not None else None
        max_pixels = config.get("max_pixels", None) if config is not None else None
        messages = cls._apply_default_image_pixel_bounds(messages, min_pixels, max_pixels)
        images, videos = process_vision_info(messages, image_patch_size=image_patch_size, return_video_metadata=True)
        return images, videos

    def split(self, num_splits: int):
        """
        split the dataset into num_splits sub-datasets
        Args:
            num_splits: specified number of splits
        Returns:
            List[RLHFDataset]: list of RLHFDataset splits
        Raises:
            ValueError: if num_splits is not a positive integer
        """
        if not isinstance(num_splits, int) or num_splits <= 0:
            raise ValueError(f"num_splits must be a positive integer, got {num_splits}")

        if not hasattr(self, "dataframe"):
            raise AttributeError(
                "dataframe not found in RLHFDataset\n"
                "reason: _read_files_and_tokenize() not called or Parquet file loading failed"
            )
        if self.dataframe is None:
            raise ValueError("RLHFDataset dataframe 为 None!")

        total_samples = len(self.dataframe)
        print(f"total_samples: {total_samples}")
        if total_samples == 0:
            raise ValueError("Cannot split an empty dataset")

        # Calculate effective sample count after dropping remainders if needed
        if total_samples % num_splits != 0:
            total_samples = total_samples - (total_samples % num_splits)
            logging.warning(f"Dropping {len(self.dataframe) % num_splits} samples, effective samples: {total_samples}")

        split_size = total_samples // num_splits
        splits = []

        for i in range(num_splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < num_splits - 1 else total_samples

            split_dataframe = self.dataframe.select(range(start_idx, end_idx))

            split_dataset = RLHFDataset(
                data_files=self.data_files,
                tokenizer=self.tokenizer,
                config=self.config,
                processor=self.processor,
                max_samples=self.max_samples,
            )
            split_dataset.dataframe = split_dataframe
            split_dataset.serialize_dataset = self.serialize_dataset
            split_dataset.original_data_files = self.original_data_files

            splits.append(split_dataset)

        return splits


def get_dataset_class(data_config: DictConfig):
    """Get RLHF dataset class.

    Args:
        data_config: The data config.

    Returns:
        dataset_cls: The dataset class.
    """

    # Check if a custom dataset class is specified in the data configuration
    # and if the path to the custom class is provided
    if "custom_cls" in data_config and data_config.custom_cls.get("path", None) is not None:
        # Dynamically load the custom dataset class
        dataset_cls = load_extern_object(data_config.custom_cls.path, data_config.custom_cls.name)
        # Verify that the custom dataset class inherits from torch.utils.data.Dataset
        if not issubclass(dataset_cls, Dataset):
            raise TypeError(
                f"The custom dataset class '{data_config.custom_cls.name}' from "
                f"'{data_config.custom_cls.path}' must inherit from torch.utils.data.Dataset"
            )
    else:
        # Use the default RLHFDataset class if no custom class is specified
        dataset_cls = RLHFDataset
    print(f"Using dataset class: {dataset_cls.__name__}")

    return dataset_cls
