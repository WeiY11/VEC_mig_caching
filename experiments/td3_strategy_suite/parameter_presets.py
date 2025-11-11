#!/usr/bin/env python3
"""Shared helpers for building comparison presets from the active config."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

from config import config

_DATA_SIZE_LABELS = ["Light", "Standard", "Heavy", "Very Heavy", "Extreme"]


def _scale_from_base(
    base_value: float,
    factors: Sequence[float],
    *,
    digits: int = 3,
    min_value: float = 0.0,
) -> List[float]:
    """Return rounded values obtained by scaling a base value."""

    if base_value <= 0:
        base_value = 1.0
    values: List[float] = []
    seen: set[float] = set()
    for factor in factors:
        scaled = max(base_value * factor, min_value)
        rounded = round(scaled, digits)
        if rounded not in seen:
            seen.add(rounded)
            values.append(rounded)
    return values


def default_arrival_rates() -> List[float]:
    """Five representative arrival rates derived from the current config."""

    base = float(getattr(config.task, "arrival_rate", 1.0) or 1.0)
    # Cover light → overload regimes while keeping per-vehicle rate positive.
    return _scale_from_base(base, (0.6, 0.8, 1.0, 1.2, 1.4), digits=2, min_value=0.05)


def default_vehicle_compute_levels() -> List[float]:
    """Five total vehicle compute presets (GHz)."""

    base = float(getattr(config.compute, "total_vehicle_compute", 6e9))
    base_ghz = base / 1e9
    return _scale_from_base(base_ghz, (0.5, 0.75, 1.0, 1.25, 1.5), digits=2, min_value=0.5)


def default_rsu_compute_levels() -> List[float]:
    """Five RSU compute presets (GHz)."""

    base = float(getattr(config.compute, "total_rsu_compute", 40e9))
    base_ghz = base / 1e9
    return _scale_from_base(base_ghz, (0.6, 0.8, 1.0, 1.2, 1.4), digits=1, min_value=5.0)


def default_uav_compute_levels() -> List[float]:
    """Five UAV compute presets (GHz)."""

    base = float(getattr(config.compute, "total_uav_compute", 6e9))
    base_ghz = base / 1e9
    return _scale_from_base(base_ghz, (0.5, 0.75, 1.0, 1.25, 1.5), digits=2, min_value=1.0)


def default_bandwidths_mhz() -> List[float]:
    """Five bandwidth presets (MHz)."""

    base = float(getattr(config.communication, "total_bandwidth", 50e6))
    base_mhz = base / 1e6
    return _scale_from_base(base_mhz, (0.4, 0.6, 0.8, 1.0, 1.2), digits=1, min_value=5.0)


def default_data_size_configs(num_segments: int = 3) -> List[Tuple[int, int, str]]:
    """Build evenly spaced task data-size ranges (KB) from the config."""

    data_range = getattr(config.task, "data_size_range", None) or getattr(
        config.task, "task_data_size_range", (0.5e6 / 8, 15e6 / 8)
    )
    min_bytes = float(data_range[0])
    max_bytes = float(data_range[1])
    if max_bytes <= min_bytes:
        max_bytes = min_bytes + 256 * 1024  # fallback 256KB span

    min_kb = min_bytes / 1024.0
    max_kb = max_bytes / 1024.0
    span = max_kb - min_kb
    segments = max(1, num_segments)
    step = span / segments

    configs: List[Tuple[int, int, str]] = []
    for idx in range(segments):
        start = min_kb + step * idx
        end = min(start + step, max_kb)
        label_base = _DATA_SIZE_LABELS[idx] if idx < len(_DATA_SIZE_LABELS) else f"Range {idx + 1}"
        label = f"{label_base} ({int(round(start))}-{int(round(end))}KB)"
        configs.append((int(round(start)), int(round(end)), label))
    return configs


__all__ = [
    "default_arrival_rates",
    "default_vehicle_compute_levels",
    "default_rsu_compute_levels",
    "default_uav_compute_levels",
    "default_bandwidths_mhz",
    "default_data_size_configs",
]
