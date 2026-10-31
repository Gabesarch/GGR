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
"""Image/chat preparation for grounded reasoning evaluation."""
import base64
from io import BytesIO

RESPONSE_TOKENS = 14336


def make_messages(images, question, system_prompt, max_pixels, min_pixels):
    from qwen_vl_utils import fetch_image
    content = []
    if not images:
        raise ValueError('Benchmark example has no decodable image')
    for image in images:
        image = fetch_image({'image': image, 'max_pixels': max_pixels, 'min_pixels': min_pixels})
        buffer = BytesIO()
        image.save(buffer, format='PNG')
        content.append({'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode('utf-8')}})
    content.append({'type': 'text', 'text': question})
    return [{'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]}, {'role': 'user', 'content': content}]


def actor_kwargs(model, tensor_parallel_size, gpu_memory_utilization, seed=1):
    return dict(model=model, tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization, trust_remote_code=True,
                max_model_len=32768, max_num_batched_tokens=8196, mm_processor_cache_gb=0,
                enable_prefix_caching=False, seed=seed)


def sampling_kwargs(max_tokens):
    return dict(max_tokens=max_tokens, temperature=1., top_p=.95, top_k=20, presence_penalty=1.5)
