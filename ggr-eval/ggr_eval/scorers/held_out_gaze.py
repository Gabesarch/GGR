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

import ast

import base64

import io

import math

import os

from pathlib import Path

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image as PILImage

COORD_MIN = 0.0

COORD_MAX = 1000.0

def _coordinate_format() -> str:
    coordinate_format = os.getenv("HELD_OUT_GAZE_COORD_FORMAT", "normalized_xy")
    if coordinate_format not in {"normalized_xy", "absolute_xy", "normalized_yx"}:
        raise ValueError(f"Unsupported HELD_OUT_GAZE_COORD_FORMAT={coordinate_format!r}")
    return coordinate_format

def _coerce_image(image: Any) -> Optional[PILImage.Image]:
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if isinstance(image, bytes):
        try:
            return PILImage.open(io.BytesIO(image)).convert("RGB")
        except Exception:
            return None
    if isinstance(image, str):
        path = Path(image)
        if path.exists():
            try:
                return PILImage.open(path).convert("RGB")
            except Exception:
                return None
        payload = image.strip()
        if payload.startswith("data:"):
            _, _, payload = payload.partition(",")
        payload = "".join(payload.split())
        if payload:
            payload += "=" * (-len(payload) % 4)
            try:
                return PILImage.open(io.BytesIO(base64.b64decode(payload, validate=False))).convert("RGB")
            except Exception:
                return None
    return None

def held_out_gaze_doc_to_visual(doc: Dict[str, Any]) -> List[PILImage.Image]:
    image = _coerce_image(doc.get("image"))
    return [image] if image is not None else []

def held_out_gaze_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    pre = ""
    post = ""
    if lmms_eval_specific_kwargs:
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")
    coordinate_format = _coordinate_format()
    if coordinate_format != "normalized_xy":
        if coordinate_format == "absolute_xy":
            image = _coerce_image(doc.get("image"))
            width = image.width if image is not None else doc.get("original_image_width")
            height = image.height if image is not None else doc.get("original_image_height")
            if not width or not height or width <= 0 or height <= 0:
                raise ValueError("absolute_xy coordinates require the original image")
            convention = f"absolute (x, y) pixel coordinates in the original {width} × {height} image"
        else:
            convention = "normalized (y, x) coordinates in the 0 to 1000 coordinate system, vertical first"
        pre = (
            "You are given an image and a visual question. Answer the question by first writing a grounded "
            "reasoning trace inside <think> and </think> tags. In the trace, include the image locations "
            f"you inspect as {convention}. After </think>, give the final answer, preferably in "
            "\\boxed{...}. Question: "
        )
    return f"{pre}{str(doc.get('question', '')).strip()}{post}"

def apply_bounds(points: Iterable[Sequence[float]], mode: str = "discard") -> List[List[float]]:
    bounded: List[List[float]] = []
    if mode not in {"discard", "clip"}:
        raise ValueError(f"Unsupported out-of-bounds mode {mode!r}; expected 'discard' or 'clip'.")
    for point in points:
        if len(point) != 2:
            continue
        x, y = float(point[0]), float(point[1])
        if not (math.isfinite(x) and math.isfinite(y)):
            continue
        in_bounds = COORD_MIN <= x <= COORD_MAX and COORD_MIN <= y <= COORD_MAX
        if in_bounds:
            bounded.append([x, y])
        elif mode == "clip":
            bounded.append([min(max(x, COORD_MIN), COORD_MAX), min(max(y, COORD_MIN), COORD_MAX)])
    return bounded

def _maybe_parse_literal(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped:
        return []
    try:
        import json

        return json.loads(stripped)
    except Exception:
        try:
            return ast.literal_eval(stripped)
        except Exception:
            return value

def validate_fixations(value: Any, mode: str = "discard") -> List[List[float]]:
    parsed = _maybe_parse_literal(value)
    if parsed is None:
        return []
    points: List[List[float]] = []
    if not isinstance(parsed, (list, tuple)):
        return []
    for item in parsed:
        if isinstance(item, dict):
            if "x" not in item or "y" not in item:
                continue
            pair = (item["x"], item["y"])
        else:
            pair = item
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        try:
            x = float(pair[0])
            y = float(pair[1])
        except Exception:
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append([x, y])
    return apply_bounds(points, mode=mode)
