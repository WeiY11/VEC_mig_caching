#!/usr/bin/env python3
"""
TD3 任务复杂度对比实验
========================

研究目标
--------
- 通过调整任务复杂度（计算密度与复杂度乘子），评估各策略在高计算压力下的表现
- 观察平均成本、时延、能耗与任务完成率的变化趋势
- 支撑论文中关于“不同任务计算强度”场景的性能对比
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt

# ========== 添加项目根目录到 Python 路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    evaluate_configs,
    strategy_label,
    tail_mean,
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
DEFAULT_COMPLEXITY_LEVELS = [0.8, 1.0, 1.2, 1.5, 1.8]


def parse_complexity_levels(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_COMPLEXITY_LEVELS)
    parsed: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("Complexity level list cannot be empty.")
    return parsed


def clamp_density(value: float) -> float:
    return max(200.0, min(value, 800.0))


def complexity_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["data_loss_ratio"] = max(tail_mean(episode_metrics.get("data_loss_ratio_bytes", [])), 0.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    levels = [float(record["complexity_level"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(levels, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Task Complexity Multiplier")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Task Complexity on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "task_complexity_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "task_complexity_vs_delay.png")
    make_chart("avg_energy", "Average Energy (J)", "task_complexity_vs_energy.png")
    make_chart("completion_rate", "Task Completion Rate", "task_complexity_vs_completion.png")

    print("\nCharts saved:")
    for name in [
        "task_complexity_vs_cost.png",
        "task_complexity_vs_delay.png",
        "task_complexity_vs_energy.png",
        "task_complexity_vs_completion.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare TD3 strategies under different task complexity levels."
    )
    parser.add_argument(
        "--complexity-levels",
        type=str,
        default="default",
        help="Comma-separated complexity multipliers or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="task_complexity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="task_complexity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    complexity_levels = parse_complexity_levels(args.complexity_levels)
    configs: List[Dict[str, object]] = []
    for level in complexity_levels:
        density = clamp_density(400.0 * level)
        overrides = {
            "task_complexity_multiplier": float(level),
            "task_compute_density": float(density),
            "high_load_mode": True,
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": f"complex_{level:.2f}",
                "label": f"{level:.2f}× Complexity",
                "overrides": overrides,
                "complexity_level": float(level),
                "task_compute_density": density,
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
        per_strategy_hook=complexity_metrics_hook,
    )

    summary = {
        "experiment_type": "task_complexity_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "complexity_levels": complexity_levels,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    print("\nTask Complexity Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Multiplier':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['complexity_level']:<12.2f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

