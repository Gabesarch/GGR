# Adopted from lmms-eval from https://github.com/EvolvingLMMs-Lab/lmms-eval.
# Copyright (c) 2025 LMMs-Lab
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
# Modified for GGR: retained only the required task dependency closure;
# imports and resource paths are local, and judging uses the Qwen3 backend.

from __future__ import annotations

import os

from functools import lru_cache

from typing import Optional

from transformers import AutoTokenizer

DEFAULT_MAX_RESPONSE_TOKENS = 8096

DEFAULT_HF_TOKENIZER = os.getenv("LMMS_EVAL_HF_TOKENIZER", "Qwen/Qwen3-8B")

DEFAULT_HF_USE_FAST = os.getenv("LMMS_EVAL_HF_TOKENIZER_USE_FAST", "1").lower() not in {
    "0",
    "false",
    "no",
}

DEFAULT_HF_TRUST_REMOTE_CODE = os.getenv("LMMS_EVAL_HF_TRUST_REMOTE_CODE", "0").lower() in {
    "1",
    "true",
    "yes",
}

def _hf_local_files_only() -> bool:
    override = os.getenv("LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY")
    if override is not None:
        return override.lower() not in {"0", "false", "no"}
    return os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes"} or os.getenv(
        "TRANSFORMERS_OFFLINE", ""
    ).lower() in {"1", "true", "yes"}

def _resolve_hf_model_path(name: str) -> str:
    """If *name* looks like an HF repo id (e.g. ``Qwen/Qwen3-8B``) and a
    matching directory exists under ``$HF_HOME``, return the local path so
    that offline loading works with ``snapshot_download``-cached models.

    Only activates when running in offline mode (``HF_HUB_OFFLINE=1`` or
    ``TRANSFORMERS_OFFLINE=1``) to avoid shadowing remote repos during
    online runs.  Also validates that the candidate directory contains
    tokenizer artifacts before accepting it.
    """
    if os.path.isabs(name) or os.path.isdir(name):
        return name
    if not _hf_local_files_only():
        return name
    hf_home = os.path.expanduser(os.getenv("HF_HOME", ""))
    if hf_home:
        local = os.path.join(hf_home, name)
        if os.path.isdir(local) and (os.path.isfile(os.path.join(local, "tokenizer_config.json")) or os.path.isfile(os.path.join(local, "tokenizer.json"))):
            return local
    return name

@lru_cache(maxsize=4)
def _get_hf_tokenizer(name: str):
    return AutoTokenizer.from_pretrained(
        _resolve_hf_model_path(name),
        use_fast=DEFAULT_HF_USE_FAST,
        trust_remote_code=DEFAULT_HF_TRUST_REMOTE_CODE,
        local_files_only=_hf_local_files_only(),
    )

def truncate_response_tail_tiktoken(
    text: Optional[str],
    *,
    max_tokens: int = DEFAULT_MAX_RESPONSE_TOKENS,
    encoding_name: Optional[str] = None,
) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    if max_tokens <= 0:
        return ""
    tokenizer_name = encoding_name or DEFAULT_HF_TOKENIZER
    tokenizer = _get_hf_tokenizer(tokenizer_name)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    if len(token_ids) <= max_tokens:
        return text
    return tokenizer.decode(
        token_ids[-max_tokens:],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
