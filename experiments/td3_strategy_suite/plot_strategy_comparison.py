#!/usr/bin/env python3
"""
Plot normalized average cost for TD3 strategy ablations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

STRATEGY_ORDER = [
    "local-only",
    "remote-only",
    "offloading-only",
    "resource-only",
    "comprehensive-no-migration",
    "comprehensive-migration",
]


def load_summary(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found at {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def compute_normalized_costs(summary: Dict) -> Tuple[List[str], List[float]]:
    costs: Dict[str, float] = {}
    for strategy in STRATEGY_ORDER:
        entry = summary.get("strategies", {}).get(strategy)
        if not entry:
            continue
        raw_cost = entry.get("metrics", {}).get("raw_cost")
        if raw_cost is None:
            continue
        costs[strategy] = float(raw_cost)
    if not costs:
        raise ValueError("No strategy entries with raw_cost found in summary.json")

    min_cost = min(costs.values())
    max_cost = max(costs.values())
    span = max(max_cost - min_cost, 1e-9)

    normalized_labels: List[str] = []
    normalized_values: List[float] = []
    for strategy in STRATEGY_ORDER:
        if strategy not in costs:
            continue
        value = (costs[strategy] - min_cost) / span if span > 0 else 0.5
        normalized_labels.append(strategy)
        normalized_values.append(value)
        metrics = summary["strategies"][strategy].setdefault("metrics", {})
        metrics["normalized_cost"] = value
    return normalized_labels, normalized_values


def plot_line(labels: List[str], values: List[float], output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    positions = list(range(len(labels)))
    plt.plot(positions, values, marker="o", linewidth=2, color="#3366CC")
    plt.xticks(positions, labels, rotation=20, ha="right")
    plt.ylim(0.0, 1.05)
    plt.grid(alpha=0.3, linestyle="--", axis="y")
    plt.ylabel("Normalized Average Cost")
    plt.xlabel("Strategy")
    plt.title("TD3 Strategy Cost Comparison")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot normalized cost comparison for TD3 strategies.")
    parser.add_argument("--suite-id", required=True, help="Suite identifier produced by run_strategy_training.py")
    parser.add_argument(
        "--output-root",
        default="results/td3_strategy_suite",
        help="Root directory containing suite folders.",
    )
    parser.add_argument(
        "--output-name",
        default="td3_strategy_cost.png",
        help="Filename for the generated chart.",
    )
    args = parser.parse_args()

    suite_path = Path(args.output_root) / args.suite_id
    summary_path = suite_path / "summary.json"
    summary = load_summary(summary_path)

    labels, values = compute_normalized_costs(summary)

    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    output_path = suite_path / args.output_name
    plot_line(labels, values, output_path)

    print("=== Normalized Cost Plot ===")
    for label, value in zip(labels, values):
        print(f"{label:28s}: {value:.4f}")
    print(f"Chart saved to: {output_path}")
    print(f"Summary updated: {summary_path}")


if __name__ == "__main__":
    main()
