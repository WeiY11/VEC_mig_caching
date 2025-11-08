#!/usr/bin/env python3
"""
Normalization helpers used across training/evaluation scripts.

The helpers read scaling factors from `config.normalization`, falling back
to the legacy hard-coded divisors when overrides are not provided. This
keeps previous behaviour intact while allowing experiments to tune the
normalization strategy via environment variables or config patches.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np

from config import config


def _get_norm_cfg():
    return getattr(config, "normalization", None)


def _get_eps() -> float:
    cfg = _get_norm_cfg()
    eps = getattr(cfg, "metric_epsilon", 1e-6) if cfg else 1e-6
    try:
        return float(eps)
    except (TypeError, ValueError):
        return 1e-6


def _get_scale(attr: str, fallback: float) -> float:
    cfg = _get_norm_cfg()
    value = fallback
    if cfg is not None and hasattr(cfg, attr):
        try:
            value = float(getattr(cfg, attr))
        except (TypeError, ValueError):
            value = fallback
    return max(float(value), _get_eps())


def normalize_scalar(
    value: Optional[float],
    attr: str,
    fallback: float,
    *,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    clip: bool = True,
) -> float:
    """Normalize a scalar with an attribute-driven scale."""
    safe_value = 0.0 if value is None else float(value)
    normalized = safe_value / _get_scale(attr, fallback)
    if not clip:
        return float(normalized)
    return float(np.clip(normalized, clip_min, clip_max))


def normalize_ratio(
    numerator: Optional[float],
    denominator: Optional[float],
    *,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
    default: float = 0.0,
) -> float:
    """Safely divide two numbers and clamp the result."""
    if denominator is None:
        denominator = 0.0
    denom = float(denominator)
    eps = _get_eps()
    if denom <= eps:
        return float(np.clip(default, clip_min, clip_max))
    numer = 0.0 if numerator is None else float(numerator)
    value = numer / denom
    return float(np.clip(value, clip_min, clip_max))


def normalize_feature_vector(
    values: Optional[Sequence[float]],
    length: int,
    *,
    clip: bool = True,
    pad_value: float = 0.0,
) -> List[float]:
    """Pad/clamp feature vectors to a fixed length."""
    if length <= 0:
        return []
    arr = np.full(length, pad_value, dtype=float)
    if values is not None:
        vec = np.asarray(list(values), dtype=float).reshape(-1)
        take = min(length, vec.size)
        if take > 0:
            arr[:take] = vec[:take]
    if clip:
        arr = np.clip(arr, 0.0, 1.0)
    else:
        arr = np.maximum(arr, 0.0)
    return arr.tolist()


def normalize_distribution(
    values: Optional[Sequence[float]],
    *,
    smoothing: Optional[float] = None,
) -> List[float]:
    """Convert a vector into a probability distribution (sum=1)."""
    if values is None:
        return []
    arr = np.asarray(list(values), dtype=float).reshape(-1)
    if arr.size == 0:
        return []
    arr = np.clip(arr, 0.0, None)
    cfg = _get_norm_cfg()
    smooth = smoothing
    if smooth is None:
        smooth = getattr(cfg, "distribution_smoothing", 0.0) if cfg else 0.0
    total = float(arr.sum())
    if total <= 0.0:
        return (np.full(arr.size, 1.0 / arr.size, dtype=float)).tolist()
    if smooth and smooth > 0.0:
        arr = (arr + smooth) / (total + smooth * arr.size)
    else:
        arr = arr / total
    return arr.tolist()


def apply_smoothing(
    values: Iterable[float],
    smoothing: Optional[float] = None,
) -> List[float]:
    """Lightweight helper to add Laplace smoothing to non-negative vectors."""
    vec = np.asarray(list(values), dtype=float)
    vec = np.clip(vec, 0.0, None)
    cfg = _get_norm_cfg()
    smooth = smoothing
    if smooth is None:
        smooth = getattr(cfg, "distribution_smoothing", 0.0) if cfg else 0.0
    if smooth and smooth > 0.0:
        vec = vec + smooth
    return vec.tolist()
