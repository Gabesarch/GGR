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

from collections.abc import Iterable
from typing import Any


def _get_nested_value(obj: Any, keys: tuple[str, ...], default: Any = None) -> Any:
    cur = obj
    for key in keys:
        if cur is None:
            return default
        if isinstance(cur, dict):
            cur = cur.get(key, default)
            continue
        get = getattr(cur, "get", None)
        if callable(get):
            cur = get(key, default)
            continue
        cur = getattr(cur, key, default)
    return cur


def get_response_display_prefix(config: Any) -> str:
    """Return the configured response display prefix without touching rollout tensors."""
    prefix = _get_nested_value(config, ("reward", "reward_kwargs", "reward_response_prefix"), "")
    return prefix if isinstance(prefix, str) else ""


def apply_response_display_prefix(text: str, prefix: str) -> str:
    """Prefix decoded response text for display only, idempotently."""
    if not prefix or text.startswith(prefix):
        return text
    return f"{prefix}{text}"


def apply_response_display_prefixes(texts: Iterable[str], prefix: str) -> list[str]:
    return [apply_response_display_prefix(text, prefix) for text in texts]
