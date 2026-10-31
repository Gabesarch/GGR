"""Single-stage annotation formatting and balancing routines."""
from __future__ import annotations
import hashlib
import math
import re
from collections import Counter, defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any

MEDIA_PLACEHOLDERS = ("<image>", "<video>", "<audio>")
THINK_OPEN_RE = re.compile(r"<think\s*>", re.I)
THINK_CLOSE_RE = re.compile(r"</think\s*>", re.I)
CATEGORY_ORDER = ("chart", "counting", "search")
DEFAULT_MAX_CATEGORY_RATIO = 1.25
DEFAULT_COUNT_PRIORITY_MIN = 6
DEFAULT_COUNT_PRIORITY_MAX = 20
DEFAULT_COUNT_PRIORITY_WEIGHT = 3

def canonical_question(value: Any) -> str:
    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<image>", " ", text, flags=re.IGNORECASE)
    return " ".join(text.split())

def question_image_key(stimulus_path: str, question: Any) -> tuple[str, str]:
    stimulus = str(stimulus_path or "").strip()
    cleaned_question = canonical_question(question)
    if not stimulus or not cleaned_question:
        raise ValueError("trial lacks a stimulus path or question")
    return stimulus, cleaned_question

def key_digest(key: tuple[str, str]) -> str:
    return hashlib.sha256((key[0] + "\n" + key[1]).encode("utf-8")).hexdigest()

def find_boxed_answer_spans(text: str) -> list[tuple[int, int, str]]:
    """Find well-formed ``\\boxed{...}`` spans, including nested braces."""
    spans: list[tuple[int, int, str]] = []
    prefix = r"\boxed{"
    search_start = 0
    while True:
        start = text.find(prefix, search_start)
        if start < 0:
            return spans
        content_start = start + len(prefix)
        depth = 1
        index = content_start
        while index < len(text):
            char = text[index]
            if char == "\\":
                index += 2
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    spans.append((start, index + 1, text[content_start:index]))
                    search_start = index + 1
                    break
            index += 1
        else:
            return spans

def format_assistant_content(sample_text: str) -> str | None:
    """Apply the legacy visible-trace-to-SFT formatting exactly."""
    text = sample_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    for placeholder in MEDIA_PLACEHOLDERS:
        text = text.replace(placeholder, " ")
    if not text:
        return None
    boxed_spans = find_boxed_answer_spans(text)
    if not boxed_spans:
        return None
    boxed_start, boxed_end, _ = boxed_spans[-1]
    reasoning = text[:boxed_start].strip()
    boxed_answer = text[boxed_start:boxed_end].strip()
    return f"<think>\n{reasoning}\n</think>\n\n{boxed_answer}"

def assistant_think_tag_structure(text: str) -> tuple[bool, int, int]:
    """Return whether *text* has exactly one ordered think-tag pair and counts."""
    open_matches = list(THINK_OPEN_RE.finditer(text))
    close_matches = list(THINK_CLOSE_RE.finditer(text))
    valid = (
        len(open_matches) == 1
        and len(close_matches) == 1
        and open_matches[0].start() == 0
        and open_matches[0].end() <= close_matches[0].start()
    )
    return valid, len(open_matches), len(close_matches)

def format_user_content(question: Any) -> str:
    """Apply the legacy one-image user-message formatting."""
    text = str(question or "").replace("\r\n", "\n").replace("\r", "\n")
    for placeholder in MEDIA_PLACEHOLDERS:
        text = text.replace(placeholder, " ")
    text = "\n".join(" ".join(line.split()) for line in text.split("\n"))
    text = "\n".join(line for line in text.split("\n") if line).strip()
    return f"<image>\n{text}"

def parse_count_target(answer: Any) -> int | None:
    """Recover a numeric or English-word counting answer for reuse priority."""
    raw = str(answer or "").strip().replace(",", "")
    if not raw:
        return None
    try:
        value = Decimal(raw)
    except InvalidOperation:
        value = None
    if value is not None:
        if not value.is_finite() or value < 0 or value != value.to_integral_value():
            return None
        return int(value)

    units = {
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
    }
    tens = {
        "twenty": 20,
        "thirty": 30,
        "forty": 40,
        "fifty": 50,
        "sixty": 60,
        "seventy": 70,
        "eighty": 80,
        "ninety": 90,
    }
    tokens = raw.casefold().replace("-", " ").split()
    if not tokens:
        return None
    total = 0
    current = 0
    saw_number = False
    for token in tokens:
        if token == "and":
            if not saw_number:
                return None
            continue
        if token in units:
            current += units[token]
            saw_number = True
        elif token in tens:
            current += tens[token]
            saw_number = True
        elif token == "hundred":
            if not saw_number:
                return None
            current *= 100
        elif token == "thousand":
            if not saw_number:
                return None
            total += current * 1000
            current = 0
        else:
            return None
    return total + current if saw_number else None

def category_for_row(row: dict[str, Any]) -> str:
    category = str((row.get("metadata") or {}).get("category") or "")
    if category not in CATEGORY_ORDER:
        raise ValueError(f"row {row.get('id')!r} has invalid category {category!r}")
    return category

def row_diversity_group(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    existing = metadata.get("question_image_key_sha256")
    if existing:
        return str(existing)
    messages = row.get("messages") or []
    user_text = "\n".join(
        str(message.get("content") or "")
        for message in messages
        if message.get("role") == "user"
    )
    return hashlib.sha256(
        (
            str(metadata.get("source_path") or "")
            + "\n"
            + canonical_question(user_text)
        ).encode("utf-8")
    ).hexdigest()

def stable_order_key(seed: int, *parts: Any) -> str:
    payload = "\n".join([str(seed), *(str(part) for part in parts)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

def diversity_order(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    category: str,
    count_priority_min: int | None = None,
    count_priority_max: int | None = None,
    count_priority_weight: int = 1,
) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row_diversity_group(row)].append(row)
    groups = sorted(
        buckets, key=lambda group: stable_order_key(seed, category, "group", group)
    )
    for group, bucket in buckets.items():
        bucket.sort(
            key=lambda row: stable_order_key(
                seed, category, group, "row", row.get("id")
            )
        )
    prioritize_counting = (
        category == "counting"
        and count_priority_min is not None
        and count_priority_max is not None
    )
    if prioritize_counting:
        if isinstance(count_priority_weight, bool) or count_priority_weight < 1:
            raise ValueError("count priority weight must be a positive integer")
        priority_groups = [
            group
            for group in groups
            if any(
                is_priority_count_row(
                    row,
                    minimum=count_priority_min,
                    maximum=count_priority_max,
                )
                for row in buckets[group]
            )
        ]
        priority_group_set = set(priority_groups)
        regular_groups = [
            group for group in groups if group not in priority_group_set
        ]
        coverage_groups = priority_groups + regular_groups
        ordered = [buckets[group][0] for group in coverage_groups]

        def repeated_rows(selected_groups: list[str]) -> list[dict[str, Any]]:
            repeated: list[dict[str, Any]] = []
            for offset in range(
                1,
                max((len(buckets[group]) for group in selected_groups), default=0),
            ):
                for group in selected_groups:
                    if offset < len(buckets[group]):
                        repeated.append(buckets[group][offset])
            return repeated

        priority_repeats = repeated_rows(priority_groups)
        regular_repeats = repeated_rows(regular_groups)
        priority_index = 0
        regular_index = 0
        while (
            priority_index < len(priority_repeats)
            or regular_index < len(regular_repeats)
        ):
            priority_stop = min(
                len(priority_repeats), priority_index + count_priority_weight
            )
            ordered.extend(priority_repeats[priority_index:priority_stop])
            priority_index = priority_stop
            if regular_index < len(regular_repeats):
                ordered.append(regular_repeats[regular_index])
                regular_index += 1
    else:
        ordered = []
        for offset in range(
            max((len(bucket) for bucket in buckets.values()), default=0)
        ):
            for group in groups:
                bucket = buckets[group]
                if offset < len(bucket):
                    ordered.append(bucket[offset])
    if len(ordered) != len(rows):
        raise AssertionError("diversity ordering changed row membership")
    return ordered

def count_rows_by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        metadata = row.get("metadata") or {}
        counts[category_for_row(row)][str(metadata.get("dataset") or "unknown")] += 1
    return {
        category: dict(sorted(category_counts.items()))
        for category, category_counts in sorted(counts.items())
    }

def is_priority_count_row(row: dict[str, Any], *, minimum: int, maximum: int) -> bool:
    if category_for_row(row) != "counting":
        return False
    target = (row.get("metadata") or {}).get("count_target")
    return (
        not isinstance(target, bool)
        and isinstance(target, int)
        and minimum <= target <= maximum
    )

def balance_unique_samples_by_category(
    rows: list[dict[str, Any]],
    *,
    seed: int,
    max_samples_per_question: int,
    max_category_ratio: float = DEFAULT_MAX_CATEGORY_RATIO,
    count_priority_min: int = DEFAULT_COUNT_PRIORITY_MIN,
    count_priority_max: int = DEFAULT_COUNT_PRIORITY_MAX,
    count_priority_weight: int = DEFAULT_COUNT_PRIORITY_WEIGHT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cap groups, retain diverse rows, enforce a max category ratio, interleave."""

    if (
        isinstance(max_samples_per_question, bool)
        or not isinstance(max_samples_per_question, int)
        or max_samples_per_question < 1
    ):
        raise ValueError("max_samples_per_question must be a positive integer")
    if count_priority_min < 0:
        raise ValueError("count priority minimum must be nonnegative")
    if count_priority_max < count_priority_min:
        raise ValueError("count priority maximum must be >= minimum")
    if isinstance(count_priority_weight, bool) or count_priority_weight < 1:
        raise ValueError("count priority weight must be a positive integer")
    if (
        isinstance(max_category_ratio, bool)
        or not isinstance(max_category_ratio, (int, float))
        or not math.isfinite(float(max_category_ratio))
        or float(max_category_ratio) < 1.0
    ):
        raise ValueError("max_category_ratio must be a finite number >= 1.0")
    max_category_ratio = float(max_category_ratio)

    ids = [str(row.get("id") or "") for row in rows]
    if any(not row_id for row_id in ids):
        raise ValueError("all rows must have a nonempty ID before balancing")
    if len(ids) != len(set(ids)):
        raise ValueError("row IDs must be unique before balancing")

    by_category: dict[str, list[dict[str, Any]]] = {
        category: [] for category in CATEGORY_ORDER
    }
    for row in rows:
        by_category[category_for_row(row)].append(row)
    original_counts = {
        category: len(by_category[category]) for category in CATEGORY_ORDER
    }
    empty = [category for category, count in original_counts.items() if not count]
    if empty:
        raise ValueError(f"cannot balance empty categories: {empty}")

    original_group_counts: dict[str, Counter[str]] = {}
    capped_by_category: dict[str, list[dict[str, Any]]] = {}
    for category in CATEGORY_ORDER:
        original_group_counts[category] = Counter(
            row_diversity_group(row) for row in by_category[category]
        )
        ordered = diversity_order(
            by_category[category],
            seed=seed,
            category=category,
            count_priority_min=count_priority_min,
            count_priority_max=count_priority_max,
            count_priority_weight=count_priority_weight,
        )
        used: Counter[str] = Counter()
        capped: list[dict[str, Any]] = []
        for row in ordered:
            group = row_diversity_group(row)
            if used[group] >= max_samples_per_question:
                continue
            used[group] += 1
            capped.append(row)
        capped_by_category[category] = capped

    capped_counts = {
        category: len(capped_by_category[category]) for category in CATEGORY_ORDER
    }
    target = min(capped_counts.values())
    maximum_target = int(Decimal(str(max_category_ratio)) * target)
    target_counts = {
        category: min(capped_counts[category], maximum_target)
        for category in CATEGORY_ORDER
    }
    selected = {
        category: capped_by_category[category][: target_counts[category]]
        for category in CATEGORY_ORDER
    }
    selected_group_counts = {
        category: Counter(row_diversity_group(row) for row in selected[category])
        for category in CATEGORY_ORDER
    }

    def group_stats(counts: Counter[str]) -> dict[str, int | float]:
        sizes = list(counts.values())
        return {
            "groups": len(sizes),
            "rows": sum(sizes),
            "minimum_rows_per_group": min(sizes, default=0),
            "maximum_rows_per_group": max(sizes, default=0),
            "mean_rows_per_group": (
                round(sum(sizes) / len(sizes), 6) if sizes else 0.0
            ),
        }

    for category in CATEGORY_ORDER:
        for row in selected[category]:
            group = row_diversity_group(row)
            metadata = row.setdefault("metadata", {})
            metadata.update(
                {
                    "category_balance_policy": "unique-downsample",
                    "category_balance_category": category,
                    "category_balance_is_duplicate": False,
                    "category_balance_original_id": str(row["id"]),
                    "category_balance_duplicate_of": None,
                    "category_balance_copy_index": 0,
                    "category_balance_original_count": original_counts[category],
                    "category_balance_capped_available_count": capped_counts[category],
                    "category_balance_target_count": target_counts[category],
                    "category_balance_smallest_category_count": target,
                    "category_balance_max_category_ratio": max_category_ratio,
                    "category_balance_seed": seed,
                    "category_balance_max_samples_per_question": (
                        max_samples_per_question
                    ),
                    "category_balance_question_original_count": (
                        original_group_counts[category][group]
                    ),
                    "category_balance_question_selected_count": (
                        selected_group_counts[category][group]
                    ),
                    "category_balance_count_priority": is_priority_count_row(
                        row,
                        minimum=count_priority_min,
                        maximum=count_priority_max,
                    ),
                }
            )

    # Proportional interleaving retains each category's deterministic internal
    # order while spreading smaller categories across the complete dataset.
    positions = {category: 0 for category in CATEGORY_ORDER}
    combined: list[dict[str, Any]] = []
    while len(combined) < sum(target_counts.values()):
        available = [
            category
            for category in CATEGORY_ORDER
            if positions[category] < target_counts[category]
        ]
        category = min(
            available,
            key=lambda candidate: (
                (positions[candidate] + 1) / target_counts[candidate],
                CATEGORY_ORDER.index(candidate),
            ),
        )
        combined.append(selected[category][positions[category]])
        positions[category] += 1
    for global_idx, row in enumerate(combined):
        metadata = row.setdefault("metadata", {})
        metadata["prebalance_global_idx"] = metadata.get("global_idx")
        metadata["global_idx"] = global_idx

    final_ids = [str(row["id"]) for row in combined]
    if len(final_ids) != len(set(final_ids)):
        raise AssertionError("unique downsampling produced duplicate row IDs")
    balanced_counts = {category: len(selected[category]) for category in CATEGORY_ORDER}
    if balanced_counts != target_counts:
        raise AssertionError(
            f"category balancing failed: {balanced_counts} != {target_counts}"
        )
    achieved_ratio = max(balanced_counts.values()) / min(balanced_counts.values())
    if achieved_ratio > max_category_ratio + 1e-12:
        raise AssertionError(
            f"category ratio {achieved_ratio} exceeds {max_category_ratio}"
        )
    if any(
        count > max_samples_per_question
        for counts in selected_group_counts.values()
        for count in counts.values()
    ):
        raise AssertionError("question group cap was exceeded")

    audit = {
        "policy_name": "unique-downsample",
        "policy": (
            "retain only unique rollout rows; cap each question-image group; "
            "retain the smallest capped category and downsample larger "
            "categories to the configured maximum ratio; "
            "select one-per-question-image before repeats; use weighted repeat "
            "retention for counting targets in the configured range; "
            "interleave chart/counting/search"
        ),
        "seed": seed,
        "category_order": list(CATEGORY_ORDER),
        "question_group_definition": "normalized question plus source image",
        "max_samples_per_question": max_samples_per_question,
        "original_category_counts": original_counts,
        "original_question_group_stats": {
            category: group_stats(original_group_counts[category])
            for category in CATEGORY_ORDER
        },
        "capped_available_category_counts": capped_counts,
        "rows_removed_by_question_cap": {
            category: original_counts[category] - capped_counts[category]
            for category in CATEGORY_ORDER
        },
        "target_rows_per_category": (
            target if len(set(target_counts.values())) == 1 else None
        ),
        "minimum_capped_category_count": target,
        "maximum_category_ratio": max_category_ratio,
        "maximum_rows_per_category": maximum_target,
        "target_category_counts": target_counts,
        "achieved_largest_to_smallest_category_ratio": round(achieved_ratio, 8),
        "rows_removed_by_category_balance": {
            category: capped_counts[category] - target_counts[category]
            for category in CATEGORY_ORDER
        },
        "balanced_category_counts": balanced_counts,
        "selected_question_group_stats": {
            category: group_stats(selected_group_counts[category])
            for category in CATEGORY_ORDER
        },
        "all_question_groups_retained": {
            category: (
                len(selected_group_counts[category])
                == len(original_group_counts[category])
            )
            for category in CATEGORY_ORDER
        },
        "duplicates_added": {category: 0 for category in CATEGORY_ORDER},
        "exact_duplicate_rows_added": False,
        "original_source_dataset_counts": count_rows_by_source(rows),
        "balanced_source_dataset_counts": count_rows_by_source(combined),
        "counting_retention_priority": {
            "applied": True,
            "target_range_inclusive": [count_priority_min, count_priority_max],
            "repeat_weight": count_priority_weight,
            "policy": (
                "retain one row per selected question-image group before any "
                "repeat, then retain in-range versus other counting repeats "
                f"with a {count_priority_weight}:1 weighted preference"
            ),
            "original_priority_rows": sum(
                is_priority_count_row(
                    row,
                    minimum=count_priority_min,
                    maximum=count_priority_max,
                )
                for row in by_category["counting"]
            ),
            "original_priority_groups": len(
                {
                    row_diversity_group(row)
                    for row in by_category["counting"]
                    if is_priority_count_row(
                        row,
                        minimum=count_priority_min,
                        maximum=count_priority_max,
                    )
                }
            ),
            "capped_priority_rows": sum(
                is_priority_count_row(
                    row,
                    minimum=count_priority_min,
                    maximum=count_priority_max,
                )
                for row in capped_by_category["counting"]
            ),
            "selected_priority_rows": sum(
                is_priority_count_row(
                    row,
                    minimum=count_priority_min,
                    maximum=count_priority_max,
                )
                for row in selected["counting"]
            ),
            "selected_priority_groups": len(
                {
                    row_diversity_group(row)
                    for row in selected["counting"]
                    if is_priority_count_row(
                        row,
                        minimum=count_priority_min,
                        maximum=count_priority_max,
                    )
                }
            ),
        },
    }
    return combined, audit
