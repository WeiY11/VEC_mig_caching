#!/usr/bin/env python3
"""
CAMTD3 车辆数量敏感性实验
========================

目标
----
- 对比不同网络规模下六种策略（或子集）的成本、时延与能耗表现
- 分析系统可扩展性，为容量规划提供依据
- 输出标准化图表与 JSON 汇总
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
)
from experiments.camtd3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_VEHICLE_COUNTS = [8, 12, 16]


def parse_vehicle_counts(value: str) -> List[int]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_VEHICLE_COUNTS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    vehicle_counts = [int(record["num_vehicles"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(vehicle_counts, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Number of Vehicles")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Vehicle Count on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "vehicle_count_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "vehicle_count_vs_delay.png")
    make_chart("avg_energy", "Average Energy (J)", "vehicle_count_vs_energy.png")
    make_chart("normalized_cost", "Normalized Cost", "vehicle_count_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "vehicle_count_vs_cost.png",
        "vehicle_count_vs_delay.png",
        "vehicle_count_vs_energy.png",
        "vehicle_count_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strategy performance across different vehicle counts."
    )
    parser.add_argument(
        "--vehicle-counts",
        type=str,
        default="default",
        help="Comma-separated vehicle counts or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="vehicle_count",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="vehicle_count",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    vehicle_counts = parse_vehicle_counts(args.vehicle_counts)
    configs: List[Dict[str, object]] = []
    for count in vehicle_counts:
        overrides = {
            "num_vehicles": count,
            "num_rsus": 4,
            "num_uavs": 2,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{count}veh",
                "label": f"{count} Vehicles",
                "overrides": overrides,
                "num_vehicles": count,
            }
        )

    suite_dir = build_suite_path(common)
    results = evaluate_configs(
        configs=configs,
        episodes=common.episodes,
        seed=common.seed,
        silent=common.silent,
        suite_path=suite_dir,
        strategies=strategy_keys,
    )

    summary = {
        "experiment_type": "vehicle_count_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    print("\nVehicle Count Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Vehicles':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['num_vehicles']:<12}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
