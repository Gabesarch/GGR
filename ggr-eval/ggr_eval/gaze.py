"""Condition-fitted gaze alignment; equations and grid from paper Appendix A.6."""

from __future__ import annotations

import hashlib
import math
import re
import numpy as np

SIGMAS = (25., 35., 50., 70., 100., 140., 200.)

EPSILONS = (.01, .03, .1, .2, .4, .7, .9)

SPLIT_SEED = 314159

UNIFORM_LPD = -math.log(1_000_000.)

NUMBER = r'[+-]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[eE][+-]?\d+)?'

PAIR = re.compile(r'\[\s*(' + NUMBER + r')\s*,\s*(' + NUMBER + r')\s*\]|\(\s*(' + NUMBER + r')\s*,\s*(' + NUMBER + r')\s*\)')

THINK = re.compile(r'<think>(.*?)</think>', re.I | re.S)

def calibration_ids(n: int, seed: int = SPLIT_SEED) -> set[int]:
    return {i for i in range(n) if int.from_bytes(
        hashlib.sha256(f'{seed}:{i}'.encode()).digest()[:8], 'big') / (1 << 64) < .5}

def valid_points(points) -> list[list[float]]:
    result = []
    for pair in points:
        if len(pair) != 2:
            raise ValueError('Malformed fixation')
        x, y = map(float, pair)
        if math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1000 and 0 <= y <= 1000:
            result.append([x, y])
    return result

def parse_predictions(raw: str, coordinate_format: str, image_size=None) -> list[list[float]]:
    """Reparse brackets and parentheses, retaining repeated emitted coordinates."""
    if coordinate_format not in ('normalized_xy', 'normalized_yx', 'absolute_xy'):
        raise ValueError('Unknown coordinate format')
    if '</think>' in raw.lower() and not raw.lstrip().lower().startswith('<think>'):
        raw = '<think>\n' + raw
    blocks = [m.group(1) for m in THINK.finditer(raw)] or [raw]
    points = []
    for block in blocks:
        for match in PAIR.finditer(block):
            a, b, c, d = match.groups()
            x, y = float(a if a is not None else c), float(b if b is not None else d)
            if coordinate_format == 'normalized_yx':
                x, y = y, x
            elif coordinate_format == 'absolute_xy':
                if image_size is None or min(image_size) <= 0:
                    raise ValueError('Absolute coordinates require original image dimensions')
                x, y = x * 1000 / image_size[0], y * 1000 / image_size[1]
            points.append([x, y])
    return valid_points(points)

def score_grid(human, predicted, sigmas=SIGMAS, epsilons=EPSILONS) -> np.ndarray:
    """Mean log density over human fixations for every sigma/epsilon pair."""
    human = np.asarray(human, dtype=float).reshape(-1, 2)
    predicted = np.asarray(predicted, dtype=float).reshape(-1, 2)
    if not len(human):
        raise ValueError('No human fixations')
    sigma = np.asarray(sigmas, dtype=float)
    epsilon = np.asarray(epsilons, dtype=float)
    if np.any(sigma <= 0) or not np.isfinite(sigma).all():
        raise ValueError('Sigma must be positive and finite')
    if np.any(epsilon < 0) or np.any(epsilon > 1) or not np.isfinite(epsilon).all():
        raise ValueError('Epsilon must be in [0, 1]')
    if not len(predicted):
        return np.full((len(sigma), len(epsilon)), UNIFORM_LPD).ravel()
    d2 = ((human[:, None, :] - predicted[None, :, :]) ** 2).sum(axis=-1)
    kernel = -d2[None, :, :] / (2 * sigma[:, None, None] ** 2)
    kernel -= np.log(2 * np.pi * sigma[:, None, None] ** 2)
    peak = kernel.max(axis=-1, keepdims=True)
    logp = peak[:, :, 0] + np.log(np.exp(kernel - peak).sum(axis=-1)) - math.log(len(predicted))
    with np.errstate(divide='ignore'):
        mixed = np.logaddexp(np.log1p(-epsilon)[None, :, None] + logp[:, None, :],
                             np.log(epsilon)[None, :, None] + UNIFORM_LPD)
    return mixed.mean(axis=-1).ravel()

def select_fit(scores: np.ndarray, calibration: set[int], sigmas=SIGMAS, epsilons=EPSILONS) -> int:
    """Select only on calibration images, with the established near-tie rule."""
    means = scores[sorted(calibration)].mean(axis=0)
    candidates = [(i, s, e) for i, (s, e) in enumerate((s, e) for s in sigmas for e in epsilons)
                  if means[i] >= means.max() - 1e-12]
    return min(candidates, key=lambda v: (abs(math.log(v[1] / 50)), abs(v[2] - .1), v[1], v[2]))[0]

def sem(values) -> float | None:
    return float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else None
