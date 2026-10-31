"""Reward helpers for fixation outputs with think-text formatting."""

from __future__ import annotations

import re
from typing import Any

from math_verify import parse
from math_verify.errors import TimeoutException
from math_verify.parser import StringExtractionConfig

from verl.utils.reward_score.math_verify_reward_type_boxed import (
    _extract_grounding_clicking_answer,
    _grounding_reward,
    _math_verify_timeout,
    _number_list_reward,
    _numeric_reward,
)
from verl.utils.reward_score.text_normalization import normalize_text_for_match, search_string_reward

__all__ = [
    "format_reward",
    "acc_reward",
    "compute_score",
    "compute_score_from_data_source",
    "async_compute_score_from_data_source",
]


_PARSE_TIMEOUT = 3
_THINK_BLOCK_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
# Historical SFT datasets use both matched delimiters. Keep the same integer
# point contract for either spelling, including in mixed-format traces.
_POINT_LITERAL = r"(?:\[\s*-?\d+\s*,\s*-?\d+\s*\]|\(\s*-?\d+\s*,\s*-?\d+\s*\))"
_INLINE_POINT_PATTERN = re.compile(_POINT_LITERAL)
_POINT_RANGE_PATTERN = re.compile(
    _POINT_LITERAL
    + r"\s*(?:to|through|[-\u2013\u2014])\s*"
    + _POINT_LITERAL,
    re.IGNORECASE,
)
_BRACKET_CONTENT_PATTERN = re.compile(r"\[([^\[\]]*)\]")
_NUMERIC_TOKEN_PATTERN = re.compile(r"-?(?:\d+(?:\.\d+)?|\.\d+)")
_INTEGER_TOKEN_PATTERN = re.compile(r"-?\d+")
_NAMED_COORDINATE_PATTERN = re.compile(r"\b(?:x|y)_(?:min|max)\s*=\s*-?\d+\b", re.IGNORECASE)
_ALPHANUMERIC_PATTERN = re.compile(r"[A-Za-z0-9]")


def _strip_tex_text_wrapper(text: str) -> str:
    if not text:
        return text
    stripped = text.lstrip()
    if not stripped.startswith(r"\text"):
        return text
    idx = len(r"\text")
    while idx < len(stripped) and stripped[idx].isspace():
        idx += 1
    if idx >= len(stripped) or stripped[idx] != "{":
        return text
    idx += 1
    depth = 1
    content_start = idx
    while idx < len(stripped) and depth > 0:
        ch = stripped[idx]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        idx += 1
    if depth != 0:
        return text
    content = stripped[content_start : idx - 1].strip()
    trailing = stripped[idx:].strip()
    if trailing:
        return text
    return content


def _extract_boxed_contents(text: str) -> list[str]:
    if not text:
        return []

    contents: list[str] = []
    idx = 0
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            idx = cursor
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            content = text[content_start : cursor - 1].strip()
            content = _strip_tex_text_wrapper(content)
            contents.append(content)
            idx = cursor
        else:
            break
    return contents


def _extract_last_boxed_bracket_content(text: str) -> str | None:
    """Return payload from the last malformed '\\boxed[...]' block, if present."""
    if not text:
        return None

    idx = 0
    last_content: str | None = None
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text):
            break
        if text[cursor] != "[":
            idx = cursor + 1
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_content = text[content_start : cursor - 1].strip()
            idx = cursor
        else:
            break
    return last_content


def _extract_boxed_answer(text: str) -> str:
    candidate = text.strip()
    if not candidate:
        return candidate

    boxed_values = _extract_boxed_contents(candidate)
    if boxed_values and boxed_values[-1]:
        return boxed_values[-1].strip()

    if r"\boxed" in candidate:
        malformed_bracket = _extract_last_boxed_bracket_content(candidate)
        if malformed_bracket:
            return malformed_bracket.strip()

    return candidate


def _extract_reward_type(extra_info: Any) -> str | None:
    if not isinstance(extra_info, dict):
        return None
    reward_type = extra_info.get("reward_type")
    if not isinstance(reward_type, str):
        return None
    reward_type = reward_type.strip().lower()
    if reward_type in {
        "short_string",
        "string_match",
        "multiple_choice",
        "number_list",
        "numeric",
        "counting",
        "search",
        "grounding",
    }:
        return reward_type
    return None


def _extract_think_blocks_and_answer(text: str) -> tuple[list[str] | None, str | None]:
    if not text:
        return None, None

    matches = list(_THINK_BLOCK_PATTERN.finditer(text))
    if not matches:
        return None, None
    if text[: matches[0].start()].strip():
        return None, None

    think_blocks: list[str] = []
    cursor = 0
    for match in matches:
        if text[cursor : match.start()].strip():
            return None, None
        think_blocks.append(match.group(1))
        cursor = match.end()

    return think_blocks, text[cursor:]


def _has_disallowed_coordinate_formatting(text: str) -> bool:
    if not text:
        return False
    if _NAMED_COORDINATE_PATTERN.search(text):
        return True

    # Parentheses can also contain ordinary arithmetic/list prose. Only matched
    # integer pairs count as points; do not newly reject unrelated prose here.
    for match in _BRACKET_CONTENT_PATTERN.finditer(text):
        parts = [part.strip() for part in match.group(1).split(",")]
        if len(parts) < 2:
            continue
        if not all(_NUMERIC_TOKEN_PATTERN.fullmatch(part) for part in parts):
            continue
        if len(parts) != 2:
            return True
        if not all(_INTEGER_TOKEN_PATTERN.fullmatch(part) for part in parts):
            return True

    return False


def _has_interleaved_non_point_text(think_text: str) -> bool:
    point_matches = list(_INLINE_POINT_PATTERN.finditer(think_text))
    if len(point_matches) < 2:
        return False

    for left, right in zip(point_matches, point_matches[1:]):
        between_points = think_text[left.end() : right.start()]
        if _ALPHANUMERIC_PATTERN.search(between_points):
            return True

    return False


def _parse_and_validate_fixation_think_text(
    predict_str: str,
    reject_point_ranges: bool = False,
) -> tuple[bool, str | None]:
    think_blocks, answer_text = _extract_think_blocks_and_answer(str(predict_str))
    if not think_blocks:
        return False, None

    think_text = "\n".join(think_blocks)
    if _has_disallowed_coordinate_formatting(think_text):
        return False, None
    if reject_point_ranges and _POINT_RANGE_PATTERN.search(think_text):
        return False, None
    if not _INLINE_POINT_PATTERN.search(think_text):
        return False, None
    if not _has_interleaved_non_point_text(think_text):
        return False, None

    final_answer = answer_text.strip()
    return True, final_answer


def _parse_and_validate_fixation_output(
    predict_str: str,
    reject_point_ranges: bool = False,
) -> tuple[bool, str | None]:
    return _parse_and_validate_fixation_think_text(
        predict_str,
        reject_point_ranges=reject_point_ranges,
    )


def _short_string_reward(predicted: str, truth: str, **_: Any) -> float:
    normalized_pred = normalize_text_for_match(predicted)
    normalized_truth = normalize_text_for_match(truth)
    if not normalized_pred or not normalized_truth:
        return 0.0
    return 1.0 if normalized_pred == normalized_truth else 0.0


def _parse_strict_integer_count(text: str) -> int | None:
    candidate = str(text).strip()
    if not _INTEGER_TOKEN_PATTERN.fullmatch(candidate):
        return None
    return int(candidate)


# For the ``exponential_half`` curve, absolute errors at or beyond this cutoff
# receive zero reward instead of the vanishing ``0.5**error`` tail. This keeps
# exact answers sharply rewarded (1.0), "close" answers rewarded but much lower
# (0.5, 0.25, 0.125 for errors 1/2/3), and collapses the noisy tail to a clean
# zero so GRPO group std/advantages are not driven by meaningless distinctions
# between e.g. "off by 6" and "off by 7".
_DEFAULT_COUNTING_REWARD_CUTOFF = 4


def _counting_reward(
    predicted: str,
    truth: str,
    counting_reward_curve: str | None = None,
    counting_reward_cutoff: int | None = _DEFAULT_COUNTING_REWARD_CUTOFF,
    **_: Any,
) -> float:
    predicted_count = _parse_strict_integer_count(predicted)
    truth_count = _parse_strict_integer_count(truth)
    if predicted_count is None or truth_count is None:
        return 0.0
    absolute_error = abs(predicted_count - truth_count)
    if counting_reward_curve == "exponential_half":
        if counting_reward_cutoff is not None and absolute_error >= counting_reward_cutoff:
            return 0.0
        return 0.5**absolute_error
    return max(0.0, 1.0 - absolute_error / 5.0)


def _extract_reward_answer(answer_text: str, reward_type: str) -> str | None:
    candidate = str(answer_text).strip()
    if not candidate:
        return None
    if reward_type == "grounding":
        return _extract_grounding_clicking_answer(candidate)
    extracted = _extract_boxed_answer(candidate)
    return extracted or None


def _multiple_choice_reward(predicted: str, truth: str, **_: Any) -> float:
    try:
        parsed = parse(
            truth,
            extraction_config=[
                StringExtractionConfig(
                    strings=(
                        "A",
                        "B",
                        "C",
                        "D",
                        "E",
                        "F",
                        "G",
                        "H",
                        "I",
                        "J",
                        "K",
                        "L",
                        "M",
                        "N",
                        "O",
                        "P",
                        "Q",
                        "R",
                        "S",
                        "T",
                        "U",
                        "V",
                        "W",
                        "X",
                        "Y",
                        "Z",
                    )
                )
            ],
            parsing_timeout=_math_verify_timeout(_PARSE_TIMEOUT),
        )
    except TimeoutException:
        parsed = None
    except Exception:
        parsed = None

    if parsed is None:
        parsed_values = []
    elif isinstance(parsed, (list, tuple, set)):
        parsed_values = list(parsed)
    else:
        parsed_values = [parsed]

    normalized_truth = str(parsed_values[0]) if parsed_values else truth
    return _short_string_reward(predicted, normalized_truth)


def format_reward(
    predict_str: str,
    extra_info: Any | None = None,  # noqa: ARG001
    reject_point_ranges: bool = False,
) -> float:
    is_valid, _ = _parse_and_validate_fixation_output(
        predict_str,
        reject_point_ranges=reject_point_ranges,
    )
    return 1.0 if is_valid else 0.0


def acc_reward(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,  # noqa: ARG001
    extra_info=None,
    counting_reward_curve: str | None = None,
    counting_reward_cutoff: int | None = _DEFAULT_COUNTING_REWARD_CUTOFF,
    reject_point_ranges: bool = False,
) -> float:
    reward_type = _extract_reward_type(extra_info)
    if reward_type is None:
        return 0.0

    is_valid, final_answer_text = _parse_and_validate_fixation_output(
        predict_str,
        reject_point_ranges=reject_point_ranges,
    )
    if not is_valid or not final_answer_text:
        return 0.0

    final_answer = _extract_reward_answer(final_answer_text, reward_type)
    if final_answer is None:
        return 0.0

    truth = str(ground_truth).strip()
    reward_handlers = {
        "short_string": _short_string_reward,
        "string_match": _short_string_reward,
        "multiple_choice": _multiple_choice_reward,
        "number_list": _number_list_reward,
        "numeric": _numeric_reward,
        "counting": _counting_reward,
        "search": search_string_reward,
        "grounding": _grounding_reward,
    }
    handler = reward_handlers.get(reward_type)
    if handler is None:
        return 0.0
    return handler(
        final_answer,
        truth,
        extra_info=extra_info,
        counting_reward_curve=counting_reward_curve,
        counting_reward_cutoff=counting_reward_cutoff,
    )


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,
    format_score: float = 0.5,
    extra_info=None,
    counting_reward_curve: str | None = None,
    counting_reward_cutoff: int | None = _DEFAULT_COUNTING_REWARD_CUTOFF,
    reject_point_ranges: bool = False,
) -> dict[str, float]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)

    formatting = format_reward(
        predict_str,
        extra_info=extra_info,
        reject_point_ranges=reject_point_ranges,
    )
    if formatting == 0.0:
        accuracy = 0.0
    else:
        accuracy = acc_reward(
            predict_str,
            ground_truth,
            use_boxed=use_boxed,
            extra_info=extra_info,
            counting_reward_curve=counting_reward_curve,
            counting_reward_cutoff=counting_reward_cutoff,
            reject_point_ranges=reject_point_ranges,
        )

    score = (1.0 - format_score) * accuracy + format_score * formatting
    return {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
    }


def compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    counting_reward_curve: str | None = None,
    counting_reward_cutoff: int | None = _DEFAULT_COUNTING_REWARD_CUTOFF,
    reject_point_ranges: bool = False,
    **_: dict,
) -> dict[str, float]:
    """Adapter for custom_reward_function configs that expect default signature."""
    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        extra_info=extra_info,
        counting_reward_curve=counting_reward_curve,
        counting_reward_cutoff=counting_reward_cutoff,
        reject_point_ranges=reject_point_ranges,
    )


async def async_compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    counting_reward_curve: str | None = None,
    counting_reward_cutoff: int | None = _DEFAULT_COUNTING_REWARD_CUTOFF,
    reject_point_ranges: bool = False,
    **_: dict,
) -> dict[str, float]:
    """Async adapter for DAPO reward loops.

    DAPO executes synchronous custom rewards in a ThreadPoolExecutor. The
    math_verify timeout helper disables signal-based timeouts off the main
    thread, so this remains safe in Ray async reward workers.
    """

    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        extra_info=extra_info,
        counting_reward_curve=counting_reward_curve,
        counting_reward_cutoff=counting_reward_cutoff,
        reject_point_ranges=reject_point_ranges,
    )
