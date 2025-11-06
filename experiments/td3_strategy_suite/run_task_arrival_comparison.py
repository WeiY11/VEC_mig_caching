#!/usr/bin/env python3
"""
TD3 任务到达率敏感性实验
==========================

目标
----
- 对比不同任务到达率下各策略的成本、时延与完成率表现
- 支持策略子集与自定义到达率配置
- 生成可直接用于论文的折线图与汇总 JSON
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

from experiments.td3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
)
from experiments.td3_strategy_suite.visualization_utils import (
    add_line_charts,
    print_chart_summary,
)
from experiments.td3_strategy_suite.suite_cli import (
    add_common_experiment_args,
    format_strategy_list,
    resolve_common_args,
    resolve_strategy_keys,
    suite_path as build_suite_path,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_ARRIVAL_RATES = [1.0, 1.5, 2.0, 2.5]


def parse_arrival_rates(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_ARRIVAL_RATES)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    arrival_rates = [float(record["arrival_rate"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(arrival_rates, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Task Arrival Rate (tasks/s)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Arrival Rate on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "arrival_rate_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "arrival_rate_vs_delay.png")
    make_chart("completion_rate", "Completion Rate", "arrival_rate_vs_completion.png")
    make_chart("normalized_cost", "Normalized Cost", "arrival_rate_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "arrival_rate_vs_cost.png",
        "arrival_rate_vs_delay.png",
        "arrival_rate_vs_completion.png",
        "arrival_rate_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strategy performance across task arrival rates."
    )
    parser.add_argument(
        "--arrival-rates",
        type=str,
        default="default",
        help="Comma-separated arrival rates (tasks/s) or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="arrival_rate",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="arrival_rate",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    arrival_rates = parse_arrival_rates(args.arrival_rates)
    configs: List[Dict[str, object]] = []
    for rate in arrival_rates:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "task_arrival_rate": float(rate),
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{rate:.2f}",
                "label": f"{rate:.2f} tasks/s",
                "overrides": overrides,
                "arrival_rate": rate,
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
        "experiment_type": "task_arrival_sensitivity",
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

    print("\nArrival Rate Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Arrival Rate':<18}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (18 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['arrival_rate']:<18.2f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
