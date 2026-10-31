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

from typing import Any

NUMBER_WORDS = {
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

_NUMBER_WORD_PATTERN = "|".join(
    sorted((re.escape(word) for word in NUMBER_WORDS), key=len, reverse=True)
)

_NUMBER_TOKEN = rf"(?:[-+]?\d+|{_NUMBER_WORD_PATTERN})"

_NUMBER_RE = re.compile(
    rf"(?<![\w.])({_NUMBER_TOKEN})(?!\w)(?!\.\d)", re.IGNORECASE
)

_DIGIT_RE = re.compile(r"[-+]?\d+")

_COORDINATE_RE = re.compile(
    r"\[\s*[-+]?\d+(?:\.\d+)?\s*,\s*[-+]?\d+(?:\.\d+)?\s*\]"
)

_PAREN_RE = re.compile(r"\([^()]*\)")

_EXPLICIT_TOTAL_PATTERNS = (
    re.compile(
        rf"\b(?:total(?:\s+(?:is|of))?|altogether|in\s+all|count\s+is)\s*[:=]?\s*({_NUMBER_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b({_NUMBER_TOKEN})\s+(?:items?\s+)?(?:in\s+total|total|altogether|in\s+all)\b",
        re.IGNORECASE,
    ),
    re.compile(rf"=\s*({_NUMBER_TOKEN})\b", re.IGNORECASE),
    re.compile(
        rf"\b(?:the\s+)?(?:total\s+)?(?:number|count)\s+of\s+"
        rf"[^.!?]{{0,100}}?\s+(?:is|are|equals?)\s*[:=]?\s*({_NUMBER_TOKEN})\b",
        re.IGNORECASE,
    ),
)

_LEADING_COUNT_RE = re.compile(
    rf"^\s*(?:there\s+(?:are|is|appears?\s+to\s+be|seems?\s+to\s+be)\s+)?"
    rf"({_NUMBER_TOKEN})\b",
    re.IGNORECASE,
)

_COMPONENT_RE = re.compile(
    rf"(?<![\w.])({_NUMBER_TOKEN})(?![\w.])\s+"
    rf"([a-z][a-z-]*(?:\s+(?!(?:and|plus)\b)[a-z][a-z-]*){{0,8}}?)"
    rf"(?=\s*(?:,|\band\b|\bplus\b|\+|$))",
    re.IGNORECASE,
)

_QUESTION_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "do",
    "does",
    "how",
    "image",
    "in",
    "many",
    "of",
    "on",
    "picture",
    "the",
    "there",
    "with",
    "you",
}

def _number_value(token: str) -> int | None:
    normalized = token.strip().lower()
    if normalized in NUMBER_WORDS:
        return NUMBER_WORDS[normalized]
    try:
        return int(normalized)
    except (TypeError, ValueError):
        return None

def _numbers_in_text(text: str) -> list[int]:
    values = []
    for match in _NUMBER_RE.finditer(text):
        value = _number_value(match.group(1))
        if value is not None:
            values.append(value)
    return values

def _question_terms(question: str | None) -> set[str]:
    if not question:
        return set()
    return {
        _singularize(token)
        for token in re.findall(r"[a-z][a-z-]*", question.lower())
        if token not in _QUESTION_STOPWORDS
    }

def _singularize(token: str) -> str:
    if token.endswith("ies") and len(token) > 3:
        return token[:-3] + "y"
    if token.endswith(("ches", "shes", "xes", "zes", "ses")) and len(token) > 2:
        return token[:-2]
    if token.endswith("s") and not token.endswith("ss") and len(token) > 1:
        return token[:-1]
    return token

def _component_values(text: str, question: str | None) -> list[int]:
    question_terms = _question_terms(question)
    values = []
    for match in _COMPONENT_RE.finditer(text):
        noun_tokens = {
            _singularize(token)
            for token in re.findall(r"[a-z][a-z-]*", match.group(2).lower())
        }
        if question_terms and not (question_terms & noun_tokens):
            continue
        value = _number_value(match.group(1))
        if value is not None:
            values.append(value)
    return values

def extract_count_candidates(value: Any, question: str | None = None) -> set[int]:
    """Return defensible count interpretations from a final answer.

    The old evaluator always selected the last integer.  This keeps that
    interpretation while also recognizing an explicit total, a leading count,
    question-matched component counts, and component sums such as
    ``8 red sockets and 8 blue sockets``. Coordinates and parenthetical
    breakdowns are excluded from component summation.
    """
    if isinstance(value, bool):
        return set()
    if isinstance(value, (int, float)):
        try:
            return {int(value)}
        except (TypeError, ValueError, OverflowError):
            return set()
    if not isinstance(value, str):
        return set()

    text = value.strip().lower()
    if not text:
        return set()

    without_coordinates = _COORDINATE_RE.sub(" ", text)

    # Retain the legacy COT interpretation after removing coordinate artifacts.
    # The legacy implementation preferred the last digit; without digits, it
    # effectively preferred the largest recognized number word because
    # NUMBER_WORDS is ordered from zero through twenty.
    candidates = set()
    legacy_digits = _DIGIT_RE.findall(without_coordinates)
    if legacy_digits:
        candidates.add(int(legacy_digits[-1]))
    else:
        legacy_words = {
            number
            for word, number in NUMBER_WORDS.items()
            if re.search(rf"\b{re.escape(word)}\b", without_coordinates)
        }
        if legacy_words:
            candidates.add(max(legacy_words))

    all_numbers = _numbers_in_text(without_coordinates)
    if not all_numbers:
        return candidates

    candidates.add(all_numbers[-1])
    explicit_totals = set()
    for pattern in _EXPLICIT_TOTAL_PATTERNS:
        for match in pattern.finditer(without_coordinates):
            parsed = _number_value(match.group(1))
            if parsed is not None:
                explicit_totals.add(parsed)
    candidates.update(explicit_totals)

    without_parentheticals = _PAREN_RE.sub(" ", without_coordinates)
    components = _component_values(without_parentheticals, question)
    if len(components) >= 2 and re.search(r"\b(?:and|plus)\b|\+", without_parentheticals):
        component_sum = sum(components)
        if not explicit_totals or component_sum in explicit_totals:
            candidates.add(component_sum)
    else:
        leading = _LEADING_COUNT_RE.match(without_coordinates)
        if leading:
            parsed = _number_value(leading.group(1))
            if parsed is not None:
                candidates.add(parsed)

    return candidates

def count_matches(value: Any, target: int | None, question: str | None = None) -> bool:
    return target is not None and target in extract_count_candidates(value, question)

def last_number_word_value(text: str) -> int | None:
    """Return the last numeric token in textual order, with word boundaries."""
    values = _numbers_in_text(text.lower()) if isinstance(text, str) else []
    return values[-1] if values else None
