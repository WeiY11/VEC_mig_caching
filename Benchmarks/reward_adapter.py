#!/usr/bin/env python3
"""
Reward adapter to reuse VEC unified_reward_calculator inside Benchmarks training loops.

Call get_reward_wrapper(sim_info) to map environment info into (reward, metrics).
"""
from __future__ import annotations

from typing import Dict, Any, Tuple

from utils.unified_reward_calculator import calculate_reward


def compute_reward_from_info(info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    # Expect info["system_metrics"] structure from simulator
    metrics = info.get("system_metrics", {}) if isinstance(info, dict) else {}
    reward = calculate_reward(
        avg_delay=metrics.get("avg_task_delay", 0.0),
        total_energy=metrics.get("total_energy_consumption", 0.0),
        cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
        migration_success_rate=metrics.get("migration_success_rate", 0.0),
        task_completion_rate=metrics.get("task_completion_rate", 0.0),
        dropped_rate=metrics.get("dropped_rate", 0.0),
    )
    return reward, {k: float(v) for k, v in metrics.items()}


__all__ = ["compute_reward_from_info"]
