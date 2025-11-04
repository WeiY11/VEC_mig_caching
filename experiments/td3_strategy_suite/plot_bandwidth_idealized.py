#!/usr/bin/env python3
"""
Generate an idealised bandwidth comparison for the six TD3 strategies.

This script synthesises “best case” results showing how each strategy
could behave as channel bandwidth increases. The data is heuristic and
intended for visual explanation rather than empirical validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

STRATEGIES = [
    "local-only",
    "remote-only",
    "offloading-only",
    "resource-only",
    "comprehensive-no-migration",
    "comprehensive-migration",
]

STRATEGY_LABELS = {
    "local-only": "Local-Only",
    "remote-only": "Edge-Only",
    "offloading-only": "Offloading-Only",
    "resource-only": "Resource-Only",
    "comprehensive-no-migration": "Comprehensive (No Migration)",
    "comprehensive-migration": "Comprehensive (TD3)",
}

BANDWIDTHS = [10, 20, 30, 40, 50]


@dataclass
class IdealisedMetrics:
    raw_cost: float
    avg_delay: float
    avg_energy: float
    completion_rate: float


def generate_metrics(bandwidth: float) -> Dict[str, IdealisedMetrics]:
    """Return heuristic metrics for each strategy at the given bandwidth."""
    def clamp(value: float, low: float, high: float) -> float:
        return max(low, min(high, value))

    results: Dict[str, IdealisedMetrics] = {}

    COST_SHAPES = {
        "local-only": [6.3, 6.35, 6.4, 6.45, 6.475],
        "remote-only": [9.4, 8.2, 7.4, 6.6, 6.0],
        "offloading-only": [6.8, 6.1, 5.65, 5.2, 4.95],
        "resource-only": [6.5, 5.7, 5.3, 4.9, 4.6],
        "comprehensive-no-migration": [5.9, 5.4, 4.8, 4.2, 3.8],
        "comprehensive-migration": [5.4, 4.6, 4.1, 3.6, 3.2],
    }

    def metrics_from_cost(cost: float) -> IdealisedMetrics:
        weighted = cost / 10.0
        avg_delay = 0.18 + 0.6 * weighted
        avg_energy = 250.0 + 140.0 * weighted
        completion = 0.55 + 0.4 * (1.0 - weighted)
        return IdealisedMetrics(
            raw_cost=clamp(cost, 1.0, 10.0),
            avg_delay=avg_delay,
            avg_energy=avg_energy,
            completion_rate=clamp(completion, 0.5, 0.95),
        )

    idx = BANDWIDTHS.index(bandwidth)
    for strat, series in COST_SHAPES.items():
        cost_value = series[idx]
        results[strat] = metrics_from_cost(cost_value)

    return results


def normalise_costs(data: Dict[str, IdealisedMetrics]) -> Dict[str, float]:
    costs = {key: metrics.raw_cost for key, metrics in data.items()}
    min_cost = min(costs.values())
    max_cost = max(costs.values())
    span = max(max_cost - min_cost, 1e-6)
    return {key: (value - min_cost) / span for key, value in costs.items()}


def main() -> None:
    output_root = Path("results/parameter_sensitivity/idealised_bandwidth")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_path = output_root / f"ideal_bandwidth_{timestamp}"
    suite_path.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    chart_data: Dict[str, List[float]] = {key: [] for key in STRATEGIES}
    normalized_chart: Dict[str, List[float]] = {key: [] for key in STRATEGIES}

    for bw in BANDWIDTHS:
        metrics = generate_metrics(bw)
        normalized = normalise_costs(metrics)

        summary[str(bw)] = {
            key: {
                **asdict(metrics[key]),
                "normalized_cost": normalized[key],
            }
            for key in STRATEGIES
        }

        for key in STRATEGIES:
            chart_data[key].append(metrics[key].raw_cost)
            normalized_chart[key].append(normalized[key])

        path = suite_path / f"{bw}mhz.json"
        path.write_text(json.dumps(summary[str(bw)], indent=2, ensure_ascii=False), encoding="utf-8")

    summary_path = suite_path / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    # Raw cost chart
    plt.figure(figsize=(10, 6))
    for key in STRATEGIES:
        plt.plot(BANDWIDTHS, chart_data[key], marker="o", linewidth=2, label=STRATEGY_LABELS[key])
    plt.xlabel("Channel Bandwidth (MHz)")
    plt.ylabel("Average Weighted Cost")
    plt.title("Idealised Average Weighted Cost vs Bandwidth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    raw_chart_path = suite_path / "idealised_bandwidth_cost.png"
    plt.savefig(raw_chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Normalised cost chart
    plt.figure(figsize=(10, 6))
    for key in STRATEGIES:
        plt.plot(BANDWIDTHS, normalized_chart[key], marker="o", linewidth=2, label=STRATEGY_LABELS[key])
    plt.xlabel("Channel Bandwidth (MHz)")
    plt.ylabel("Normalized Cost (0=best)")
    plt.title("Normalised Cost Projection across Bandwidth")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    norm_chart_path = suite_path / "idealised_bandwidth_cost_normalized.png"
    plt.savefig(norm_chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Idealised dataset generated.")
    print(f"Summary JSON   : {summary_path}")
    print(f"Raw cost chart : {raw_chart_path}")
    print(f"Norm chart     : {norm_chart_path}")


if __name__ == "__main__":
    main()
