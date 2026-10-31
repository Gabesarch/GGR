#!/usr/bin/env python3
"""Normalize point syntax and discourse-label text in a ShareGPT SFT dataset.

Only message ``content`` strings are changed. Structured metadata such as
coordinate arrays is deliberately preserved.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
AXIS = r"(?:[- ]?coord(?:inate)?)?"
BOX_SEPARATOR = r"\s*[,;]\s*"


def _opening_wrapper_pattern(opening: str) -> str:
    """Match a wrapper without consuming LaTeX command argument braces.

    A final answer such as ``\\boxed{2004, 2006}`` is a two-item answer, not a
    curly-braced coordinate.  Requiring a non-letter immediately before ``{``
    preserves command arguments while continuing to normalize standalone
    VLM-emitted points such as ``{10, 20}``.
    """
    if opening == "{":
        return r"(?<![A-Za-z])\{"
    return re.escape(opening)


# Convert bounding boxes before converting ordinary points.  Otherwise a
# two-corner box such as ``[[10, 20], [30, 40]]`` would first become two
# independent points and lose its box semantics.  Flat box variants cover the
# delimiters commonly emitted by VLMs, while the labeled forms cover
# ``x1/y1/x2/y2`` and ``left/top/right/bottom`` spellings.
NUMERIC_BOX_BODY = (
    rf"(?P<x1>{NUMBER}){BOX_SEPARATOR}(?P<y1>{NUMBER})"
    rf"{BOX_SEPARATOR}(?P<x2>{NUMBER}){BOX_SEPARATOR}(?P<y2>{NUMBER})"
)
FLAT_NUMERIC_BOX_RES = tuple(
    re.compile(
        rf"{_opening_wrapper_pattern(opening)}\s*{NUMERIC_BOX_BODY}\s*"
        rf"{re.escape(closing)}"
    )
    for opening, closing in (("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"))
)
NESTED_NUMERIC_BOX_RE = re.compile(
    rf"[\[(]\s*[\[(]\s*(?P<x1>{NUMBER}){BOX_SEPARATOR}(?P<y1>{NUMBER})"
    rf"\s*[\])]\s*[,;]\s*[\[(]\s*(?P<x2>{NUMBER})"
    rf"{BOX_SEPARATOR}(?P<y2>{NUMBER})\s*[\])\]]\s*[\])\]]"
)

# Two explicit corner points joined by "to" describe a rectangular extent,
# not two independent evidence points. Handle same-wrapper pairs before the
# ordinary point pass, including parentheses, square brackets, braces, and
# angle brackets. For example, "(786, 444) to (816, 491)" and
# "[786, 444] to [816, 491]" both become the single center point "(801, 468)"
# (with the usual half-up coordinate rounding).
POINT_RANGE_NUMERIC_BOX_RES = tuple(
    re.compile(
        rf"{_opening_wrapper_pattern(opening)}\s*"
        rf"(?P<x1>{NUMBER}){BOX_SEPARATOR}(?P<y1>{NUMBER})\s*"
        rf"{re.escape(closing)}\s+\bto\b\s+"
        rf"{_opening_wrapper_pattern(opening)}\s*"
        rf"(?P<x2>{NUMBER}){BOX_SEPARATOR}(?P<y2>{NUMBER})\s*"
        rf"{re.escape(closing)}",
        re.IGNORECASE,
    )
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"))
)

# Three or more points joined by ``to`` describe a path/polyline rather than a
# two-corner rectangular extent. Protect the separators while the two-corner
# box rules run, but continue normalizing every point inside the chain.
CHAIN_NUMERIC_POINT = (
    rf"(?:\(\s*{NUMBER}\s*[,;]\s*{NUMBER}\s*\)"
    rf"|\[\s*{NUMBER}\s*[,;]\s*{NUMBER}\s*\]"
    rf"|\{{\s*{NUMBER}\s*[,;]\s*{NUMBER}\s*\}}"
    rf"|<\s*{NUMBER}\s*[,;]\s*{NUMBER}\s*>)"
)
NUMERIC_POINT_CHAIN_RE = re.compile(
    rf"{CHAIN_NUMERIC_POINT}(?:\s+\bto\b\s+{CHAIN_NUMERIC_POINT}){{2,}}",
    re.IGNORECASE,
)

BOX_X1_LABEL = r"(?:x\s*(?:[_-]?\s*1|[_-]?\s*min)|left)"
BOX_Y1_LABEL = r"(?:y\s*(?:[_-]?\s*1|[_-]?\s*min)|top)"
BOX_X2_LABEL = r"(?:x\s*(?:[_-]?\s*2|[_-]?\s*max)|right)"
BOX_Y2_LABEL = r"(?:y\s*(?:[_-]?\s*2|[_-]?\s*max)|bottom)"
BOX_FIELD_SEPARATOR = r"\s*(?:[,;]|\band\b)\s*"
LABELED_NUMERIC_BOX_BODY = (
    rf"['\"]?{BOX_X1_LABEL}['\"]?\s*[:=]\s*(?P<x1>{NUMBER})"
    rf"{BOX_FIELD_SEPARATOR}['\"]?{BOX_Y1_LABEL}['\"]?\s*[:=]\s*(?P<y1>{NUMBER})"
    rf"{BOX_FIELD_SEPARATOR}['\"]?{BOX_X2_LABEL}['\"]?\s*[:=]\s*(?P<x2>{NUMBER})"
    rf"{BOX_FIELD_SEPARATOR}['\"]?{BOX_Y2_LABEL}['\"]?\s*[:=]\s*(?P<y2>{NUMBER})"
)
LABELED_NUMERIC_BOX_WRAPPED_RES = tuple(
    re.compile(
        rf"{_opening_wrapper_pattern(opening)}\s*{LABELED_NUMERIC_BOX_BODY}\s*"
        rf"{re.escape(closing)}",
        re.IGNORECASE,
    )
    for opening, closing in (("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"))
)
LABELED_NUMERIC_BOX_RE = re.compile(LABELED_NUMERIC_BOX_BODY, re.IGNORECASE)

# A wrapper-free four-number sequence is treated as a box only when it has an
# explicit box label, avoiding changes to unrelated lists in ordinary prose.
PREFIXED_BARE_NUMERIC_BOX_RE = re.compile(
    rf"(?P<prefix>\b(?:bounding\s+box|bbox|box)\b"
    rf"(?:\s+coordinates?)?\s*(?:(?:is|at)\s+|[:=]\s*)?)"
    rf"(?P<x1>{NUMBER})(?:\s*[,;]\s*|\s+)"
    rf"(?P<y1>{NUMBER})(?:\s*[,;]\s*|\s+)"
    rf"(?P<x2>{NUMBER})(?:\s*[,;]\s*|\s+)"
    rf"(?P<y2>{NUMBER})",
    re.IGNORECASE,
)

SYMBOLIC_BOX_AXIS_1 = r"(?:x\s*(?:_?\{?1\}?|_?min)|left)"
SYMBOLIC_BOX_ORDINATE_1 = r"(?:y\s*(?:_?\{?1\}?|_?min)|top)"
SYMBOLIC_BOX_AXIS_2 = r"(?:x\s*(?:_?\{?2\}?|_?max)|right)"
SYMBOLIC_BOX_ORDINATE_2 = r"(?:y\s*(?:_?\{?2\}?|_?max)|bottom)"
SYMBOLIC_BOX_BODY = (
    rf"{SYMBOLIC_BOX_AXIS_1}{BOX_SEPARATOR}{SYMBOLIC_BOX_ORDINATE_1}"
    rf"{BOX_SEPARATOR}{SYMBOLIC_BOX_AXIS_2}{BOX_SEPARATOR}{SYMBOLIC_BOX_ORDINATE_2}"
)
SYMBOLIC_BOX_RES = tuple(
    re.compile(
        rf"{_opening_wrapper_pattern(opening)}\s*{SYMBOLIC_BOX_BODY}\s*"
        rf"{re.escape(closing)}",
        re.IGNORECASE,
    )
    for opening, closing in (("[", "]"), ("(", ")"), ("{", "}"), ("<", ">"))
)

# The grounding traces predominantly use literal pixel points such as
# ``[421, 307]`` and symbolic instructions such as ``[x, y]``.  Normalize all
# common wrappers, including already-parenthesized points, so decimal values
# cannot bypass integer rounding merely because their syntax is canonical.
NUMERIC_POINT_BODY = rf"(?P<x>{NUMBER})\s*[,;]\s*(?P<y>{NUMBER})"
PAREN_NUMERIC_POINT_RE = re.compile(rf"\(\s*{NUMERIC_POINT_BODY}\s*\)")
SQUARE_NUMERIC_POINT_RE = re.compile(rf"\[\s*{NUMERIC_POINT_BODY}\s*\]")
CURLY_NUMERIC_POINT_RE = re.compile(
    rf"{_opening_wrapper_pattern('{')}\s*{NUMERIC_POINT_BODY}\s*\}}"
)
ANGLE_NUMERIC_POINT_RE = re.compile(rf"<\s*{NUMERIC_POINT_BODY}\s*>")
NUMERIC_POINT_RES = (
    ("parenthesized_numeric", PAREN_NUMERIC_POINT_RE),
    ("square_numeric", SQUARE_NUMERIC_POINT_RE),
    ("curly_numeric", CURLY_NUMERIC_POINT_RE),
    ("angle_numeric", ANGLE_NUMERIC_POINT_RE),
)

PAREN_SYMBOLIC_POINT_RE = re.compile(r"\(\s*x\s*[,;]\s*y\s*\)", re.IGNORECASE)
SQUARE_SYMBOLIC_POINT_RE = re.compile(r"\[\s*x\s*[,;]\s*y\s*\]", re.IGNORECASE)
CURLY_SYMBOLIC_POINT_RE = re.compile(r"\{\s*x\s*[,;]\s*y\s*\}", re.IGNORECASE)
ANGLE_SYMBOLIC_POINT_RE = re.compile(r"<\s*x\s*[,;]\s*y\s*>", re.IGNORECASE)
SYMBOLIC_POINT_RES = (
    ("parenthesized_symbolic", PAREN_SYMBOLIC_POINT_RE),
    ("square_symbolic", SQUARE_SYMBOLIC_POINT_RE),
    ("curly_symbolic", CURLY_SYMBOLIC_POINT_RE),
    ("angle_symbolic", ANGLE_SYMBOLIC_POINT_RE),
)

# Normalize directly paired axis-value forms while avoiding unrelated prose
# that happens to mention x and y in separate clauses. Paired wrappers are
# handled before the bare expression to avoid producing double parentheses.
XY_BODY = (
    rf"x{AXIS}\s*[:=]\s*(?P<x>{NUMBER})"
    rf"\s*[*_$`]*\s*(?:[,;:]|\band\b|\bis\s+at\b|\bat\b)\s*"
    rf"[*_$`]*\s*y{AXIS}\s*[:=]\s*(?P<y>{NUMBER})"
)
XY_WRAPPED_RES = tuple(
    re.compile(
        rf"{_opening_wrapper_pattern(opening)}\s*{XY_BODY}\s*" rf"{re.escape(closing)}",
        re.IGNORECASE,
    )
    for opening, closing in (("(", ")"), ("[", "]"), ("{", "}"), ("<", ">"))
)
XY_BARE_RE = re.compile(rf"\b{XY_BODY}", re.IGNORECASE)

MARKDOWN_EMPHASIS = r"(?:\*{1,3}|_{1,3})"
DISCOURSE_LABEL = r"(?:planning|exploration|evaluation|evalation|reflection)"
# Remove a label only when it is the leading paragraph scaffold, never when it
# is part of ordinary content such as ``Self-Reflection``, ``Initial Planning``,
# ``Lunar Exploration``, or ``Evaluation Metrics``. Punctuation may sit inside
# or outside Markdown emphasis (``**Planning:**`` and ``**Planning**:``).
PARAGRAPH_DISCOURSE_LABEL_RE = re.compile(
    rf"^[ \t]*(?:(?:\#{{1,6}}|[-+*])\s+|\d+[.)]\s+)?"
    rf"(?:{MARKDOWN_EMPHASIS})?[ \t]*{DISCOURSE_LABEL}"
    rf"(?:{MARKDOWN_EMPHASIS})?[ \t]*[:：.\-–—][ \t]*"
    rf"(?:{MARKDOWN_EMPHASIS})?[ \t]*",
    re.IGNORECASE | re.MULTILINE,
)
# Also remove an exact standalone Markdown heading/emphasized scaffold. Plain
# unformatted ``Evaluation`` is intentionally retained because it can be a
# real chart/diagram stage name.
STANDALONE_MARKED_DISCOURSE_LABEL_RE = re.compile(
    rf"^[ \t]*(?:"
    rf"\#{{1,6}}\s+(?:{MARKDOWN_EMPHASIS})?[ \t]*{DISCOURSE_LABEL}"
    rf"(?:{MARKDOWN_EMPHASIS})?"
    rf"|(?:[-+*]\s+)?{MARKDOWN_EMPHASIS}[ \t]*{DISCOURSE_LABEL}"
    rf"[ \t]*{MARKDOWN_EMPHASIS}"
    rf")[ \t]*(?=\n|$)",
    re.IGNORECASE | re.MULTILINE,
)


def _parenthesize(match: re.Match[str]) -> str:
    return (
        f"({_round_coordinate(Decimal(match.group('x')))}, "
        f"{_round_coordinate(Decimal(match.group('y')))})"
    )


def _round_coordinate(value: Decimal) -> str:
    rounded = value.to_integral_value(rounding=ROUND_HALF_UP)
    if rounded == 0:
        return "0"
    return format(rounded, "f")


def _bbox_center(match: re.Match[str]) -> str:
    center_x = (Decimal(match.group("x1")) + Decimal(match.group("x2"))) / 2
    center_y = (Decimal(match.group("y1")) + Decimal(match.group("y2"))) / 2
    return f"({_round_coordinate(center_x)}, {_round_coordinate(center_y)})"


def _prefixed_bbox_center(match: re.Match[str]) -> str:
    return f"{match.group('prefix')}{_bbox_center(match)}"


def _protect_numeric_point_chains(text: str) -> tuple[str, str, int]:
    sentinel = "__SFT_POINT_CHAIN_TO__"
    while sentinel in text:
        sentinel += "_"

    def protect(match: re.Match[str]) -> str:
        return re.sub(
            r"\s+\bto\b\s+", f" {sentinel} ", match.group(0), flags=re.IGNORECASE
        )

    return (*NUMERIC_POINT_CHAIN_RE.subn(protect, text), sentinel)


def normalize_text(text: str) -> tuple[str, Counter]:
    counts: Counter = Counter()
    text, label_count = PARAGRAPH_DISCOURSE_LABEL_RE.subn("", text)
    counts["paragraph_discourse_label"] += label_count
    text, standalone_label_count = STANDALONE_MARKED_DISCOURSE_LABEL_RE.subn("", text)
    counts["standalone_marked_discourse_label"] += standalone_label_count
    # Removing a standalone scaffold can leave three or more line breaks where
    # two paragraph separators are sufficient.
    if standalone_label_count:
        text = re.sub(r"\n{3,}", "\n\n", text)
    text, chain_count, point_chain_sentinel = _protect_numeric_point_chains(text)
    counts["numeric_point_chain_preserved"] += chain_count
    for pattern in POINT_RANGE_NUMERIC_BOX_RES:
        text, count = pattern.subn(_bbox_center, text)
        counts["numeric_bbox_point_range_to_center"] += count
        counts["numeric_bbox_to_center"] += count
    text, count = NESTED_NUMERIC_BOX_RE.subn(_bbox_center, text)
    counts["numeric_bbox_nested_to_center"] += count
    counts["numeric_bbox_to_center"] += count
    for index, pattern in enumerate(LABELED_NUMERIC_BOX_WRAPPED_RES):
        text, count = pattern.subn(_bbox_center, text)
        counts[f"numeric_bbox_labeled_wrapped_{index}_to_center"] += count
        counts["numeric_bbox_to_center"] += count
    text, count = LABELED_NUMERIC_BOX_RE.subn(_bbox_center, text)
    counts["numeric_bbox_labeled_bare_to_center"] += count
    counts["numeric_bbox_to_center"] += count
    for index, pattern in enumerate(FLAT_NUMERIC_BOX_RES):
        text, count = pattern.subn(_bbox_center, text)
        counts[f"numeric_bbox_flat_{index}_to_center"] += count
        counts["numeric_bbox_to_center"] += count
    text, count = PREFIXED_BARE_NUMERIC_BOX_RE.subn(_prefixed_bbox_center, text)
    counts["numeric_bbox_prefixed_bare_to_center"] += count
    counts["numeric_bbox_to_center"] += count
    for index, pattern in enumerate(SYMBOLIC_BOX_RES):
        text, count = pattern.subn("(x, y)", text)
        counts[f"symbolic_bbox_flat_{index}_to_point"] += count
        counts["symbolic_bbox_to_point"] += count
    for index, pattern in enumerate(XY_WRAPPED_RES):
        text, count = pattern.subn(_parenthesize, text)
        counts[f"xy_labeled_wrapped_{index}"] += count
    text, count = XY_BARE_RE.subn(_parenthesize, text)
    counts["xy_labeled_bare"] += count
    text, newly_exposed_chain_count, second_point_chain_sentinel = (
        _protect_numeric_point_chains(text)
    )
    counts["numeric_point_chain_preserved"] += newly_exposed_chain_count
    # Labeled points can expose a range only after x/y normalization, as in
    # "x: 516, y: 601 to x: 687, y: 795". Collapse that range before the
    # ordinary point pass.
    for pattern in POINT_RANGE_NUMERIC_BOX_RES:
        text, count = pattern.subn(_bbox_center, text)
        counts["numeric_bbox_point_range_to_center"] += count
        counts["numeric_bbox_to_center"] += count
    for label, pattern in NUMERIC_POINT_RES:
        text, count = pattern.subn(_parenthesize, text)
        counts[label] += count
        counts["numeric_points_integer_normalized"] += count
    for label, pattern in SYMBOLIC_POINT_RES:
        text, count = pattern.subn("(x, y)", text)
        counts[label] += count
    text = text.replace(point_chain_sentinel, "to")
    text = text.replace(second_point_chain_sentinel, "to")
    return text, counts


def remaining_noncanonical(text: str) -> Counter:
    counts: Counter = Counter()
    counts["paragraph_discourse_label"] = len(
        PARAGRAPH_DISCOURSE_LABEL_RE.findall(text)
    )
    counts["standalone_marked_discourse_label"] = len(
        STANDALONE_MARKED_DISCOURSE_LABEL_RE.findall(text)
    )
    # Nested two-corner boxes are converted before flat four-number boxes in
    # ``normalize_text``.  A list of two flat boxes can therefore legitimately
    # become ``((x1, y1), (x2, y2))``: two canonical center points that happen
    # to match the original nested-box surface grammar.  Do not flag that
    # post-normalization representation or collapse two evidence points into
    # one.  Any original nested box was already handled by the first pass.
    # Same-wrapper two-corner ranges are already collapsed before ordinary
    # point normalization. A range still visible here consists entirely of
    # canonical points and may be a polyline or a data-point/image-point pair
    # whose mixed wrappers only became identical during normalization.
    counts["numeric_bbox_labeled_bare"] = len(LABELED_NUMERIC_BOX_RE.findall(text))
    counts["numeric_bbox_prefixed_bare"] = len(
        PREFIXED_BARE_NUMERIC_BOX_RE.findall(text)
    )
    for index, pattern in enumerate(FLAT_NUMERIC_BOX_RES):
        counts[f"numeric_bbox_flat_{index}"] = len(pattern.findall(text))
    for index, pattern in enumerate(LABELED_NUMERIC_BOX_WRAPPED_RES):
        counts[f"numeric_bbox_labeled_wrapped_{index}"] = len(pattern.findall(text))
    for index, pattern in enumerate(SYMBOLIC_BOX_RES):
        counts[f"symbolic_bbox_flat_{index}"] = len(pattern.findall(text))
    for label, pattern in NUMERIC_POINT_RES:
        matches = list(pattern.finditer(text))
        if label == "parenthesized_numeric":
            counts["noninteger_parenthesized_numeric"] = sum(
                not re.fullmatch(r"-?(?:0|[1-9]\d*)", match.group(axis))
                for match in matches
                for axis in ("x", "y")
            )
        else:
            counts[label] = len(matches)
    for label, pattern in SYMBOLIC_POINT_RES:
        if label != "parenthesized_symbolic":
            counts[label] = len(pattern.findall(text))
    counts["xy_labeled_bare"] = len(XY_BARE_RE.findall(text))
    for index, pattern in enumerate(XY_WRAPPED_RES):
        counts[f"xy_labeled_wrapped_{index}"] = len(pattern.findall(text))
    return +counts
