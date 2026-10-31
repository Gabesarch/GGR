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

import unicodedata

from loguru import logger as eval_logger

from ggr_eval.scorers.answer_extraction import extract_final_answer

TASKS = [
    "Reasoning",
    "Perception",
]

SUBTASKS = [
    "Monitoring",
    "Autonomous_Driving",
    "OCR with Complex Context",
    "Diagram and Table",
    "Remote Sensing",
]

def mme_realworld_doc_to_visual(doc):
    img = decode_base64_to_image(doc["bytes"])
    return [img.convert("RGB")]

import base64

import io

from PIL import Image

def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data))
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image

def mme_realworld_doc_to_text_cot(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"]
    option_prompt = "The choices are listed below:\n" + "\n".join(doc["multi-choice options"]) + "\n"

    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")

    return f"{pre}{question} {option_prompt}{post}"

def _strip_reasoning_tags(text: str) -> str:
    final_answer = extract_final_answer(text)
    if final_answer or not isinstance(text, str) or "</think>" not in text:
        return final_answer

    # Some reasoning models put their only final answer immediately before
    # ``</think>`` and then stop.  The shared extractor correctly prefers
    # post-thinking content, but that leaves these otherwise unambiguous MME
    # responses empty.  Fall back narrowly to the last boxed option letter in
    # the reasoning block; do not scan arbitrary reasoning prose for letters.
    reasoning = text.rsplit("</think>", 1)[0]
    boxed_letters = re.findall(
        r"\\boxed\s*\{\s*([A-E])\s*\}", reasoning, flags=re.IGNORECASE
    )
    return boxed_letters[-1].upper() if boxed_letters else ""

def _normalize_choice_text(value):
    value = unicodedata.normalize("NFKD", str(value or "")).casefold()
    return " ".join(re.sub(r"[^a-z0-9]+", " ", value).split())

def _choice_text_by_letter(choices):
    result = {}
    for choice in choices or []:
        match = re.match(r"^\s*\(([A-E])\)\s*(.*?)\s*$", str(choice), re.IGNORECASE)
        if match:
            result[match.group(1).upper()] = match.group(2)
    return result

def extract_characters_regex(s, choices=None):
    """Extract an unambiguous option letter, including exact option-text answers."""
    if type(s) is dict:
        s = ""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = _strip_reasoning_tags(s).strip()
    choice_map = _choice_text_by_letter(
        choices or ["(A)", "(B)", "(C)", "(D)", "(E)"]
    )
    allowed_letters = set(choice_map) or set("ABCDE")

    # Prefer a complete answer over searching arbitrary prose for isolated
    # letters. The latter incorrectly treats the English article "a" as A.
    match = re.fullmatch(r"\s*\(?\s*([A-E])\s*\)?[\s.]*", s, re.IGNORECASE)
    if match and match.group(1).upper() in allowed_letters:
        return match.group(1).upper()

    normalized = _normalize_choice_text(s)
    text_matches = [
        letter
        for letter, choice_text in choice_map.items()
        if choice_text and _normalize_choice_text(choice_text) == normalized
    ]
    if len(text_matches) == 1:
        return text_matches[0]

    match = re.match(r"^\s*\(([A-E])\)(?:\s|$)", s, re.IGNORECASE)
    if match and match.group(1).upper() in allowed_letters:
        return match.group(1).upper()

    declarations = re.findall(
        r"(?i)(?:answer|option|choice)\s*(?:(?:is\s*)?:\s*|is\s+)?\(?([A-E])\)?\b",
        s,
    )
    if declarations and declarations[-1].upper() in allowed_letters:
        return declarations[-1].upper()
    return ""

def mme_realworld_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme_realworld score), value: metric value
    """
    pred = results[0] if isinstance(results, (list, tuple)) else results
    if isinstance(pred, (list, tuple)):
        pred = pred[-1] if len(pred) > 0 else ""
    pred_ans = extract_characters_regex(pred, doc.get("multi-choice options"))
    # gt_ans = doc["answer"].lower().strip().replace(".", "")

    category = "Perception" if "perception" in doc["category"].lower() else "Reasoning"
    sub_category = doc["category"].split("/")[-1]
    task_category = doc["l2-category"]
    pred_ans_norm = str(pred_ans).strip().lower()
    answer_norm = str(doc["answer"]).strip().lower()
    correct = bool(pred_ans_norm) and (pred_ans_norm == answer_norm or answer_norm in pred_ans_norm)
    data_dict = {
        "question_id": doc["index"],
        "category": category,
        "sub_category": sub_category,
        "task_category": task_category,
        "pred_answer": pred_ans,
        "answer": doc["answer"],
        "correct": correct,
    }

    # return {f"mme_realworld_perception_score": data_dict for metric in matrices}
    return {f"mme_realworld_score": data_dict}

def mme_realworld_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """

    metrics = {}
    for task in TASKS:
        metrics[f"{task}"] = {}
        for subtask in SUBTASKS:
            metrics[f"{task}"][f"{subtask}"] = {}

    for i in range(len(results)):
        result = results[i]
        Task = result["category"]
        Subtask = result["sub_category"]
        Category = result["task_category"].lower()
        if "attribute" in Category.lower():
            Category = Category.split("/")[0] + "/attribute"
        pred_answer = result["pred_answer"]
        pred_answer_text = pred_answer if isinstance(pred_answer, str) else " ".join(str(item) for item in pred_answer)
        cnt = result.get("correct")
        if cnt is None:
            cnt = pred_answer_text.lower() == result["answer"].lower() or result["answer"].lower() in pred_answer_text.lower()
        if Category not in metrics[Task][Subtask].keys():
            metrics[Task][Subtask][f"{Category}"] = {"true": cnt, "false": 1 - cnt, "is_E": pred_answer_text == "E"}
        else:
            metrics[Task][Subtask][f"{Category}"]["true"] += cnt
            metrics[Task][Subtask][f"{Category}"]["false"] += 1 - cnt
            metrics[Task][Subtask][f"{Category}"]["is_E"] += pred_answer_text == "E"

    sum_all, succ_all = 0, 0
    for task, tasks_values in metrics.items():
        eval_logger.info(f"*" * 32 + f"{task} (Task Start)")
        cnt_task, cnt_E, sum_task = 0, 0, 0
        for substask, subtask_value in tasks_values.items():
            eval_logger.info(f"+" * 16 + f"{substask} (Subtask Start)")
            cnt_subtask, sum_subtask, e_subtask = 0, 0, 0
            for category, category_dict in subtask_value.items():
                cnt_subtask += category_dict["true"]
                sum_subtask += category_dict["false"] + category_dict["true"]
                e_subtask += category_dict["is_E"]
                acc = category_dict["true"] / (category_dict["false"] + category_dict["true"])
                eval_logger.info(f"-" * 4 + f"\t" + "Acc " + "{:.4f}".format(acc) + f"\t{category.capitalize()} ({category_dict['false'] + category_dict['true']} items)")

            if sum_subtask == 0:
                acc_subtasks = 0
                e_subtask = 0
            else:
                acc_subtasks = cnt_subtask / sum_subtask
            eval_logger.info(f"+" * 16 + f"\t Acc " + "{:.4f}".format(acc_subtasks) + f"\t E choice {e_subtask} \t{substask} ({sum_subtask} items)")
            cnt_task += cnt_subtask
            sum_task += sum_subtask
            cnt_E += e_subtask

        if sum_task == 0:
            acc_task = 0
        else:
            acc_task = cnt_task / sum_task
        succ_all += cnt_task
        sum_all += sum_task
        eval_logger.info(f"*" * 32 + f"Acc " + "{:.4f}".format(acc_task) + f"\t E choice {cnt_E} \t{task} ({sum_task} items)\n")
    eval_logger.info(f"*" * 32 + f"Overall Acc " + "{:.4f}".format(succ_all / sum_all))
    return succ_all / sum_all
