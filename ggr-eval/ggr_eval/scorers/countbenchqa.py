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

import re

from typing import Any, Dict, Optional

from PIL import Image as PILImage

from ggr_eval.scorers.answer_extraction import extract_final_answer

from ggr_eval.scorers.count_extraction import count_matches, last_number_word_value

def _get_first_image(doc: Dict[str, Any]):
    # Common HF Image feature key is 'image'
    if "image" in doc:
        img = doc["image"]
        try:
            return img.convert("RGB")
        except Exception:
            return img
    # Some datasets use 'images' as a list
    if "images" in doc and isinstance(doc["images"], (list, tuple)) and len(doc["images"]) > 0:
        img = doc["images"][0]
        try:
            return img.convert("RGB")
        except Exception:
            return img
    # If an image path is provided
    if "image_path" in doc and isinstance(doc["image_path"], str):
        try:
            return PILImage.open(doc["image_path"]).convert("RGB")
        except Exception:
            return PILImage.open(doc["image_path"])  # best effort
    # If nothing is found, return None to let caller decide
    return None

def countbenchqa_doc_to_visual(doc: Dict[str, Any]):
    img = _get_first_image(doc)
    return [img] if img is not None else []

_QUESTION_KEYS = [
    "question",
    "prompt",
    "query",
    "instruction",
    "text",
    "caption",
]

def countbenchqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"]:
        pre = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"]:
        post = lmms_eval_specific_kwargs["post_prompt"]

    question = None
    for k in _QUESTION_KEYS:
        if k in doc and isinstance(doc[k], str) and doc[k].strip():
            question = doc[k].strip()
            break

    if question is None:
        # Fallback generic prompt if schema differs; still returns a valid string
        question = "How many objects are requested?"

    return f"{pre}{question}{post}"

_NUM_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
}

def _extract_int_from_text(s: str) -> Optional[int]:
    if not isinstance(s, str):
        try:
            return int(s)
        except Exception:
            return None
    t = s.strip().lower()
    # First try to find an integer in the string
    m = re.search(r"[-+]?\d+", t)
    if m:
        try:
            return int(m.group(0))
        except Exception:
            pass
    # Try number words (simple 0-20)
    for w, v in _NUM_WORDS.items():
        if w in t:
            return v
    return None

def _question_text(doc: Dict[str, Any]) -> str:
    for key in _QUESTION_KEYS:
        value = doc.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return ""

def _get_ground_truth_count(doc: Dict[str, Any]) -> Optional[int]:
    # Common field names for answers (CountBenchQA uses 'number')
    for k in ["number", "answer", "count", "label", "gt", "target"]:
        if k in doc:
            return _extract_int_from_text(doc[k])
    return None

def countbenchqa_cot_process_results(doc: Dict[str, Any], results):
    # When using CoT, prefer the last number in the response to avoid capturing intermediate counts
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred = extract_final_answer(pred)
    gt_int = _get_ground_truth_count(doc)
    score = 1.0 if count_matches(pred, gt_int, _question_text(doc)) else 0.0
    return {"exact_match": score}
