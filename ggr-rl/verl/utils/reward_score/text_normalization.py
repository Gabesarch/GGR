"""Text normalization helpers for string matching."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_NON_ALNUM_PATTERN = re.compile(r"[\W_]+")
_COMPACT_NON_ALNUM_PATTERN = re.compile(r"[^a-z0-9]+")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_THOUSANDS_SEP_PATTERN = re.compile(r"(?<=\d),(?=\d{3}\b)")
_TRAILING_DECIMAL_PATTERN = re.compile(r"\b(\d+)\.0+\b(?!\.)")
_ANSWER_OPTION_DELIMITER_PATTERN = re.compile(r"[,;/|]")

_ARTICLES = {"a", "an", "the"}
_NUMBER_WORDS = {
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

_PUNCT_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00a0": " ",
    }
)


def _normalize_unicode(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text)
    normalized = unicodedata.normalize("NFD", normalized)
    return "".join(
        ch for ch in normalized if unicodedata.category(ch) != "Mn"
    )


def _normalize_numbers(text: str) -> str:
    text = _THOUSANDS_SEP_PATTERN.sub("", text)
    return _TRAILING_DECIMAL_PATTERN.sub(r"\1", text)


def _strip_articles(text: str) -> str:
    tokens = text.split()
    if not tokens:
        return text
    kept = [token for token in tokens if token not in _ARTICLES]
    if not kept:
        return text
    # Avoid stripping when it would drop meaningful single-letter tokens.
    if not any(len(token) > 1 for token in kept):
        return text
    return " ".join(kept)


def normalize_text_for_match(text: Any) -> str:
    normalized = str(text)
    normalized = _normalize_unicode(normalized)
    normalized = normalized.translate(_PUNCT_TRANSLATION)
    normalized = _normalize_numbers(normalized)
    normalized = normalized.lower()
    normalized = _NON_ALNUM_PATTERN.sub(" ", normalized)
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized).strip()
    return _strip_articles(normalized)


def _compact_normalize_text(text: Any) -> str:
    normalized = _normalize_unicode(str(text))
    normalized = normalized.translate(_PUNCT_TRANSLATION).lower()
    return _COMPACT_NON_ALNUM_PATTERN.sub("", normalized)


def _parse_simple_integer(text: Any) -> int | None:
    normalized = normalize_text_for_match(text)
    if not normalized:
        return None
    compact = normalized.replace(" ", "")
    if re.fullmatch(r"[+-]?\d+", compact):
        return int(compact)
    return _NUMBER_WORDS.get(normalized.replace("-", " "))


def _tokens_for_unordered_match(text: Any) -> list[str]:
    normalized = normalize_text_for_match(text)
    tokens = [token for token in normalized.split() if token != "and"]
    if 2 <= len(tokens) <= 5:
        return tokens
    return []


def _levenshtein_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    if not right:
        return len(left)

    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (left_char != right_char)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def _has_answer_stuffing(raw_pred: str, normalized_pred: str) -> bool:
    if len(_ANSWER_OPTION_DELIMITER_PATTERN.findall(raw_pred)) >= 4:
        return True

    option_parts = []
    for part in re.split(r"[,;/|]+", raw_pred):
        normalized_part = normalize_text_for_match(part)
        if normalized_part:
            option_parts.append(normalized_part)
    if len(option_parts) >= 4:
        return True

    tokens = normalized_pred.split()
    if len(tokens) >= 4 and len(set(tokens)) == 1:
        return True

    half = len(normalized_pred) // 2
    return bool(half and normalized_pred[:half] == normalized_pred[half:])


def search_string_reward(predicted: Any, truth: Any, **_: Any) -> float:
    """Tolerant scorer for visual search/OCR answers with conservative partial credit."""
    raw_pred = str(predicted).strip()
    raw_truth = str(truth).strip()
    if not raw_pred or not raw_truth:
        return 0.0

    if raw_pred == raw_truth:
        return 1.0

    normalized_pred = normalize_text_for_match(raw_pred)
    normalized_truth = normalize_text_for_match(raw_truth)
    if normalized_pred and normalized_pred == normalized_truth:
        return 1.0

    compact_pred = _compact_normalize_text(raw_pred)
    compact_truth = _compact_normalize_text(raw_truth)
    if compact_pred and compact_pred == compact_truth:
        return 1.0

    pred_int = _parse_simple_integer(raw_pred)
    truth_int = _parse_simple_integer(raw_truth)
    if pred_int is not None and truth_int is not None and pred_int == truth_int:
        return 1.0

    pred_tokens = _tokens_for_unordered_match(raw_pred)
    truth_tokens = _tokens_for_unordered_match(raw_truth)
    if pred_tokens and truth_tokens and sorted(pred_tokens) == sorted(truth_tokens):
        return 1.0

    compare_pred = normalized_pred or compact_pred
    compare_truth = normalized_truth or compact_truth
    if not compare_pred or not compare_truth:
        return 0.0

    max_len = max(len(compare_pred), len(compare_truth))
    if max_len < 3:
        return 0.0

    distance = _levenshtein_distance(compare_pred, compare_truth)
    similarity = 1.0 - (distance / max_len)

    shorter, longer = sorted((compare_pred, compare_truth), key=len)
    compact_shorter, compact_longer = sorted((compact_pred, compact_truth), key=len)
    strict_substring = shorter != longer and shorter in longer
    compact_substring = bool(
        compact_shorter and compact_shorter != compact_longer and compact_shorter in compact_longer
    )
    strict_substring = strict_substring or compact_substring
    if strict_substring:
        similarity = max(similarity, 0.75)

    if similarity < 0.5:
        return 0.0

    score = min(similarity, 0.95)
    if strict_substring:
        score = min(score, 0.75)
    if len(compare_pred) > 2.5 * len(compare_truth):
        score = min(score, 0.70)
    if len(compare_pred.split()) >= len(compare_truth.split()) + 3:
        score = min(score, 0.70)
    if _has_answer_stuffing(raw_pred, compare_pred):
        score = min(score, 0.60)
    return max(0.0, score)
