#!/usr/bin/env python3
"""
Offload bias diagnosis helper.

Compares reward gaps under different offload ratios using the unified reward
calculator (same logic as training), and shows the effect of hand-tuned
RSU/UAV/local coefficients.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from utils.unified_reward_calculator import calculate_reward


def _reward_from_ratios(local: float, rsu: float, uav: float, cache_hit: float = 0.1) -> float:
    """Mock a set of metrics and feed them to the unified reward calculator."""
    avg_delay = 0.4 * (0.5 * rsu + 0.8 * uav + 1.2 * local)
    total_energy = 0.5 * (0.6 * rsu + 0.9 * uav + 1.1 * local)
    completion = np.clip(0.9 + 0.05 * rsu - 0.05 * local, 0.0, 1.0)
    migration = np.clip(0.6 * uav + 0.8 * rsu, 0.0, 1.0)
    dropped = max(0.0, 1.0 - completion)
    return calculate_reward(
        avg_delay=avg_delay,
        total_energy=total_energy,
        cache_hit_rate=cache_hit,
        migration_success_rate=migration,
        task_completion_rate=completion,
        dropped_rate=dropped,
    )


def calculate_reward_comparison() -> List[Tuple[str, float]]:
    scenarios = {
        "bad_local70_uav20_rsu10": (0.70, 0.10, 0.20),
        "mid_local50_uav30_rsu20": (0.50, 0.20, 0.30),
        "target_rsu60_uav20_local20": (0.20, 0.60, 0.20),
        "better_rsu70_uav15_local15": (0.15, 0.70, 0.15),
        "ideal_rsu80_uav10_local10": (0.10, 0.80, 0.10),
    }

    results: List[Tuple[str, float]] = []
    for name, (local, rsu, uav) in scenarios.items():
        r = _reward_from_ratios(local, rsu, uav)
        results.append((name, r))
        print(f"{name:32s} reward={r:+.3f}  rsu={rsu:.2f} uav={uav:.2f} local={local:.2f}")

    best = max(results, key=lambda x: x[1])[1]
    worst = min(results, key=lambda x: x[1])[1]
    print("\nReward gaps:")
    for name, r in results:
        gap = r - worst
        pct = gap / (best - worst) * 100 if best != worst else 0.0
        print(f"{name:32s} gap={gap:+.3f} ({pct:5.1f}%)")
    print(f"\nbest - worst = {best - worst:.3f}")
    return results


def compare_old_vs_new() -> None:
    """Compare illustrative old vs new hand-crafted coefficients."""
    local, rsu, uav = 0.20, 0.60, 0.20
    cache_bonus = 0.15
    cost = 0.88
    old_reward = -cost + cache_bonus + rsu * 25.0 + uav * 3.0 - local * 10.0
    new_reward = -cost + cache_bonus + rsu * 50.0 + max(0, rsu - 0.5) * 20.0 + uav * 1.5 - local * 15.0
    print("\nOld vs New (hand-tuned coefficients at rsu60/uav20/local20):")
    print(f"old_reward={old_reward:+.2f} new_reward={new_reward:+.2f} delta={new_reward - old_reward:+.2f}")
    print("Key changes: RSU coef doubled (25->50), RSU priority added, UAV coef halved, local penalty increased.")


if __name__ == "__main__":
    print("=" * 80)
    print("Offload Bias Diagnosis")
    print("=" * 80)
    calculate_reward_comparison()
    compare_old_vs_new()
    print("\nGuidance: rsu_offload_ratio should be >0.5 (ideal >0.6); uav/local <0.25.")
