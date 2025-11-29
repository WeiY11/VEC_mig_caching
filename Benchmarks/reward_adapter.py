#!/usr/bin/env python3
"""
Reward adapter to reuse VEC unified_reward_calculator inside Benchmarks training loops.

Call get_reward_wrapper(sim_info) to map environment info into (reward, metrics).
"""
from __future__ import annotations

from typing import Dict, Any, Tuple
import numpy as np

from utils.unified_reward_calculator import calculate_unified_reward


def compute_reward_from_info(info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    # Expect info["system_metrics"] structure from simulator
    metrics = info.get("system_metrics", {}) if isinstance(info, dict) else {}
    
    # Construct system_metrics dict expected by calculate_unified_reward
    # It expects keys like 'avg_task_delay', 'total_energy_consumption', etc.
    # The metrics dict from info should already have these keys.
    
    reward = calculate_unified_reward(system_metrics=metrics)
    
    # Filter and convert metrics to float, ignoring non-numeric values (lists, dicts, etc.)
    clean_metrics = {}
    for k, v in metrics.items():
        try:
            if isinstance(v, (int, float, np.number)):
                 clean_metrics[k] = float(v)
        except (ValueError, TypeError):
            pass
            
    return reward, clean_metrics


__all__ = ["compute_reward_from_info"]
