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

from typing import Any, Dict, List, Optional

from datasets import Dataset, Features, Value

from PIL import Image as PILImage

from ggr_eval.scorers.answer_extraction import extract_final_answer

from ggr_eval.scorers.count_extraction import count_matches, last_number_word_value

def countqa_process_docs(dataset: Dataset) -> Dataset:
    """Expand a single example with multiple QA pairs into individual rows."""
    questions: List[str] = []
    answers: List[str] = []
    question_ids: List[str] = []
    images: List[Any] = []
    objects: List[Any] = []
    categories: List[Any] = []
    focused_flags: List[bool] = []
    full_configs: List[Any] = []

    for example_idx, example in enumerate(dataset):
        q_list = example.get("questions") or []
        a_list = example.get("answers") or []
        if not q_list:
            continue

        for qa_idx, question in enumerate(q_list):
            answer = a_list[qa_idx] if qa_idx < len(a_list) else ""
            images.append(example.get("image"))
            objects.append(example.get("objects"))
            categories.append(example.get("categories"))
            focused_flags.append(bool(example.get("is_focused", False)))
            full_configs.append(example.get("full_config"))
            questions.append(str(question).strip())
            answers.append(str(answer).strip())
            question_ids.append(f"{example_idx}_{qa_idx}")

    if not questions:
        return dataset

    features = Features(
        {
            "image": dataset.features["image"],
            "objects": dataset.features["objects"],
            "categories": dataset.features["categories"],
            "is_focused": dataset.features["is_focused"],
            "full_config": dataset.features["full_config"],
            "question": Value("string"),
            "answer": Value("string"),
            "question_id": Value("string"),
        }
    )

    flat_dataset = Dataset.from_dict(
        {
            "image": images,
            "objects": objects,
            "categories": categories,
            "is_focused": focused_flags,
            "full_config": full_configs,
            "question": questions,
            "answer": answers,
            "question_id": question_ids,
        },
        features=features,
    )
    return flat_dataset

def _get_first_image(doc: Dict[str, Any]):
    img = doc.get("image")
    if img is None:
        return None
    if isinstance(img, PILImage.Image):
        return img.convert("RGB")
    if hasattr(img, "convert"):
        try:
            return img.convert("RGB")
        except Exception:
            return img
    if isinstance(img, str):
        try:
            return PILImage.open(img).convert("RGB")
        except Exception:
            return None
    return None

def countqa_doc_to_visual(doc: Dict[str, Any]):
    img = _get_first_image(doc)
    return [img] if img is not None else []

def countqa_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    if lmms_eval_specific_kwargs.get("pre_prompt"):
        pre = lmms_eval_specific_kwargs["pre_prompt"]
    if lmms_eval_specific_kwargs.get("post_prompt"):
        post = lmms_eval_specific_kwargs["post_prompt"]

    question = doc.get("question")
    if not question:
        q_list = doc.get("questions") or []
        if q_list:
            question = q_list[0]
    question = str(question).strip() if question else "How many objects are present?"
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

def _extract_int_from_text(value: Any) -> Optional[int]:
    if isinstance(value, (int, float)):
        try:
            return int(value)
        except Exception:
            return None
    if not isinstance(value, str):
        return None
    text = value.strip().lower()
    match = re.search(r"[-+]?\d+", text)
    if match:
        try:
            return int(match.group(0))
        except Exception:
            pass
    for word, number in _NUM_WORDS.items():
        if word in text:
            return number
    return None

def _question_text(doc: Dict[str, Any]) -> str:
    question = doc.get("question")
    if question:
        return str(question)
    questions = doc.get("questions") or []
    return str(questions[0]) if questions else ""

def _get_ground_truth_count(doc: Dict[str, Any]) -> Optional[int]:
    answer = doc.get("answer")
    if answer is not None:
        return _extract_int_from_text(answer)
    if "answers" in doc and doc["answers"]:
        return _extract_int_from_text(doc["answers"][0])
    for key in ["number", "count", "target"]:
        if key in doc and doc[key] is not None:
            return _extract_int_from_text(doc[key])
    return None

def countqa_cot_process_results(doc: Dict[str, Any], results):
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred = extract_final_answer(pred)
    gt_int = _get_ground_truth_count(doc)
    score = 1.0 if count_matches(pred, gt_int, _question_text(doc)) else 0.0
    return {"exact_match": score}
