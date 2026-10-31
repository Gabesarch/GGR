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

from typing import Any

import re

def _extract_last_boxed_content(text: str) -> str | None:
    r"""Return content of the last balanced \boxed{...}, if present."""
    last_content: str | None = None
    start_idx = 0
    while True:
        idx = text.find("\\boxed{", start_idx)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            last_content = text[start : pos - 1]
            start_idx = pos
        else:
            break
    return last_content

def _extract_last_square_boxed_content(text: str) -> str | None:
    r"""Return the last balanced ``\boxed[...]`` as a list-like value.

    Some chart prompts request list answers using square brackets. Models can
    consequently combine that instruction with the boxed-answer instruction
    and emit ``\boxed["answer"]``.  Preserve the brackets so downstream chart
    evaluators can distinguish scalar and multi-answer list syntax.
    """
    last_content: str | None = None
    start_idx = 0
    while True:
        idx = text.find("\\boxed[", start_idx)
        if idx == -1:
            break
        start = idx + len("\\boxed[")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "[":
                depth += 1
            elif text[pos] == "]":
                depth -= 1
            pos += 1
        if depth == 0:
            last_content = f"[{text[start : pos - 1]}]"
            start_idx = pos
        else:
            break
    return last_content

def _extract_safe_terminal_bracket(text: str) -> str | None:
    """Return a defensible answer-like bracket at the end of a response.

    This is intentionally narrower than generic ``[...]`` extraction because
    the reasoning traces frequently end in image coordinates.  It accepts a
    quoted value/list, a single numeric value, or the explicit unanswerable
    sentinel, but not an unquoted numeric pair such as ``[421, 317]``.
    """
    match = re.search(r"(\[[^\[\]\n]*\])\s*$", text)
    if match is None:
        return None
    bracketed = match.group(1)
    inner = bracketed[1:-1].strip()
    if not inner:
        return None
    if re.fullmatch(r"(?i:unanswerable)", inner):
        return bracketed
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?%?", inner):
        return bracketed
    if re.search(r"['\"]", inner):
        return bracketed
    return None

def _strip_boxed(text: str) -> str:
    r"""Remove \boxed{...} wrappers, keeping inner content. Handles nested braces."""
    result = text
    while True:
        idx = result.find("\\boxed{")
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            inner = result[start : pos - 1]
            result = result[:idx] + inner + result[pos:]
        else:
            break  # Unbalanced braces — stop
    return result

def _strip_tex_text_wrapper(text: str) -> str:
    r"""If text is exactly \text{...}, unwrap it once (or repeatedly)."""
    result = text.strip()
    while result.startswith("\\text{") and result.endswith("}"):
        start = len("\\text{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0 or pos != len(result):
            break
        result = result[start : pos - 1].strip()
    return result

def extract_final_answer(
    raw_response: Any,
    parse_boxed: bool = True,
    strip_latex_wrappers: bool = False,
    parse_square_boxed: bool = False,
    parse_terminal_bracket: bool = False,
) -> str:
    """Extract the final answer, removing thinking reasoning.

    - If <answer>...</answer> exists, prefer its content.
    - Otherwise, drop content up to </think> if present.
    - Trim common answer prefixes.
    - *parse_boxed*: extract content of the last \\boxed{...} (for math tasks).
    - *parse_square_boxed*: also accept chart-style ``\\boxed[...]`` output.
    - *parse_terminal_bracket*: accept a safe answer-like final ``[...]`` when
      a reasoning trace is truncated before ``</think>``.
    - *strip_latex_wrappers*: remove \\boxed{...} and \\text{...} wrappers
      inline, keeping surrounding text intact (for instruction-following tasks
      where these are just noise from reasoning models).
    """
    if not isinstance(raw_response, str):
        return "" if raw_response is None else str(raw_response)

    text = raw_response.strip()
    if not text:
        return ""

    if "</think>" in text and "<think>" not in text:
        text = f"<think>\n{text}"
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # if "Answer:" in text:
    #     text = text.rsplit("Answer:", 1)[-1].strip()
    # elif "answer:" in text:
    #     text = text.rsplit("answer:", 1)[-1].strip()

    if parse_boxed:
        boxed_content = _extract_last_boxed_content(text)
        if parse_square_boxed:
            square_content = _extract_last_square_boxed_content(text)
            if square_content is not None and (
                boxed_content is None
                or text.rfind("\\boxed[") > text.rfind("\\boxed{")
            ):
                boxed_content = square_content
        if boxed_content is not None:
            text = boxed_content.strip()
        else:
            if parse_terminal_bracket:
                terminal_bracket = _extract_safe_terminal_bracket(text)
                if terminal_bracket is not None:
                    text = terminal_bracket
            stripped = _strip_boxed(text)
            if stripped != text:
                text = stripped.strip()

        text = _strip_tex_text_wrapper(text)
    elif strip_latex_wrappers:
        text = _strip_boxed(text)
        text = _strip_tex_text_wrapper(text)

    return text.strip()
