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

import gc
import atexit

import json

import os

from dataclasses import dataclass

from typing import Any, Dict, Optional


_DEFAULT_SYSTEM_PROMPT = ""

def _resolve_model_path(model_hint: Optional[str]) -> str:
    """Resolve the model path for the local vLLM judge."""
    for env_var in ("CHARXIV_JUDGE_MODEL_PATH", "VLLM_JUDGE_MODEL_PATH"):
        candidate = os.getenv(env_var)
        if candidate:
            return candidate
    if model_hint:
        return model_hint
    env_model = os.getenv("MODEL_VERSION")
    if env_model:
        return env_model
    raise RuntimeError(
        "Unable to resolve judge model path. Set CHARXIV_JUDGE_MODEL_PATH or MODEL_VERSION."
    )

@dataclass(frozen=True)
class JudgeConfig:
    model_path: str
    tensor_parallel_size: int
    dtype: str
    max_model_len: int
    max_num_seqs: Optional[int]
    temperature: float
    top_p: float
    top_k: int
    presence_penalty: float
    max_tokens: int
    gpu_memory_utilization: float
    system_prompt: Optional[str]

def _load_config(model_hint: Optional[str]) -> JudgeConfig:
    model_path = _resolve_model_path(model_hint)
    tensor_parallel = int(os.getenv("CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE", os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")))
    dtype = os.getenv("CHARXIV_JUDGE_DTYPE", os.getenv("VLLM_JUDGE_DTYPE", "bfloat16"))
    max_model_len = int(os.getenv("CHARXIV_JUDGE_MAX_MODEL_LEN", "18384"))
    max_num_seqs = int(os.getenv("VLLM_JUDGE_MAX_NUM_SEQS", "256"))
    temperature = float(os.getenv("CHARXIV_JUDGE_TEMPERATURE", "0.0"))
    top_p = float(os.getenv("CHARXIV_JUDGE_TOP_P", "0.8"))
    top_k = int(os.getenv("CHARXIV_JUDGE_TOP_K", "20"))
    presence_penalty = float(os.getenv("CHARXIV_JUDGE_PRESENCE_PENALTY", "0.0"))
    max_tokens = int(os.getenv("CHARXIV_JUDGE_MAX_TOKENS", os.getenv("VLLM_JUDGE_MAX_TOKENS", "2048")))
    gpu_memory_utilization = float(
        os.getenv("CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION", os.getenv("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", "0.9"))
    )
    system_prompt = os.getenv("CHARXIV_JUDGE_SYSTEM_PROMPT", os.getenv("VLLM_JUDGE_SYSTEM_PROMPT", _DEFAULT_SYSTEM_PROMPT))
    return JudgeConfig(
        model_path=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype=dtype,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        presence_penalty=presence_penalty,
        max_tokens=max_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        system_prompt=system_prompt,
    )

class VLLMJudgeEngineQwen3:
    """Thin wrapper around vLLM for JSON generation with batched prompts."""

    def __init__(self, config: JudgeConfig) -> None:
        from vllm import LLM, SamplingParams
        self._config = config
        template_path = os.path.join(os.path.dirname(__file__), "qwen3_nonthinking.jinja")
        self._chat_template = None
        if os.path.isfile(template_path):
            with open(template_path, "r", encoding="utf-8") as handle:
                self._chat_template = handle.read()
        self._llm = LLM(
            model=config.model_path,
            tensor_parallel_size=config.tensor_parallel_size,
            dtype=config.dtype,
            max_model_len=config.max_model_len,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_seqs=config.max_num_seqs,
        )
        self.supports_multimodal = hasattr(self._llm, "chat")
        self._tokenizer = self._llm.get_tokenizer()
        if self._chat_template and hasattr(self._tokenizer, "chat_template"):
            self._tokenizer.chat_template = self._chat_template
        sampling_kwargs = {
            "temperature": 0.7,
            "top_p": 0.8,
            "max_tokens": 2048,
            "top_k": 20,
            "min_p": 0.0,
        }
        self._base_sampling_kwargs = dict(sampling_kwargs)
        self._sampling_params = SamplingParams(**self._base_sampling_kwargs)
        # raw_prompt = config.system_prompt or _DEFAULT_SYSTEM_PROMPT
        # self._system_prompt = raw_prompt.strip()

    def _build_chat_prompt(self, prompt: str) -> str:
        messages = []
        # if self._system_prompt:
        #     messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def _resolve_sampling_params(self, max_tokens: Optional[int]) -> SamplingParams:
        from vllm import SamplingParams
        if max_tokens is None or max_tokens == self._base_sampling_kwargs.get("max_tokens"):
            return self._sampling_params
        override_kwargs = dict(self._base_sampling_kwargs)
        override_kwargs["max_tokens"] = max_tokens
        return SamplingParams(**override_kwargs)

    def generate_json_batch(
        self,
        prompts: list[str],
        *,
        max_tokens: Optional[int] = None,
        use_tqdm: bool = False,
    ) -> list[str]:
        if not prompts:
            return []
        chat_prompts = [self._build_chat_prompt(prompt) for prompt in prompts]
        sampling_params = self._resolve_sampling_params(max_tokens)
        outputs = self._llm.generate(
            chat_prompts,
            sampling_params=sampling_params,
            use_tqdm=use_tqdm,
        )
        responses: list[str] = []
        for output in outputs:
            if not output.outputs:
                responses.append("")
                continue
            responses.append(output.outputs[0].text.strip())
        return responses

    def generate_json(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        batched = self.generate_json_batch([prompt], max_tokens=max_tokens)
        return batched[0] if batched else ""

    def generate_json_messages(self, messages: list[dict[str, Any]], *, max_tokens: Optional[int] = None) -> str:
        if not self.supports_multimodal:
            raise RuntimeError("This local judge backend does not support multimodal message inputs.")
        sampling_params = self._resolve_sampling_params(max_tokens)
        chat_kwargs: dict[str, Any] = {}
        if self._chat_template:
            chat_kwargs["chat_template"] = self._chat_template
        outputs = self._llm.chat(
            messages=[list(messages or [])],
            sampling_params=sampling_params,
            use_tqdm=False,
            **chat_kwargs,
        )
        if not outputs or not outputs[0].outputs:
            return ""
        return outputs[0].outputs[0].text.strip()

    def shutdown(self) -> None:
        """Gracefully shut down the vLLM engine to avoid 'died unexpectedly' errors."""
        llm = getattr(self, "_llm", None)
        if llm is None:
            return
        try:
            engine = getattr(llm, "llm_engine", None)
            if engine is not None:
                core = getattr(engine, "engine_core", None)
                if core is not None and hasattr(core, "shutdown"):
                    core.shutdown()
                elif hasattr(engine, "shutdown"):
                    engine.shutdown()
            elif hasattr(llm, "shutdown"):
                llm.shutdown()
        except Exception:
            pass
        try:
            from vllm.distributed.parallel_state import destroy_model_parallel
            destroy_model_parallel()
        except Exception:
            pass
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        gc.collect()
        self._llm = None

def extract_json_candidate(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    candidates = [text]
    if "{" in text and "}" in text:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None

_ENGINE_CACHE = {}

def get_judge_engine(model_hint=None):
    config = _load_config(model_hint)
    if config.model_path not in _ENGINE_CACHE:
        _ENGINE_CACHE[config.model_path] = VLLMJudgeEngineQwen3(config)
        atexit.register(_ENGINE_CACHE[config.model_path].shutdown)
    return _ENGINE_CACHE[config.model_path]
