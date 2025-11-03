#!/usr/bin/env python3
"""
CAMTD3 带宽敏感性对比实验
========================

目标
----
- 检验不同通信带宽设置下，各策略的成本、时延与吞吐表现
- 生成论文可用的四类对比图（成本/时延/归一化成本/吞吐）
- 允许快速子集实验以缩短调试时间
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
from experiments.camtd3_strategy_suite.visualization_utils import (
    add_line_charts,
    print_chart_summary,
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
DEFAULT_BANDWIDTHS = [10, 20, 30, 40, 50]


def parse_bandwidths(value: str) -> List[int]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_BANDWIDTHS)
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def bandwidth_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    throughput_series = episode_metrics.get("throughput_mbps") or episode_metrics.get("avg_throughput_mbps")
    avg_throughput = 0.0
    if throughput_series:
        values = list(map(float, throughput_series))
        if values:
            half = values[len(values) // 2 :] if len(values) >= 100 else values
            avg_throughput = float(sum(half) / max(len(half), 1))

    if avg_throughput <= 0:
        avg_task_size_mb = 0.35  # 约 350KB
        num_tasks_per_step = config.get("assumed_tasks_per_step", 12)
        avg_delay = metrics.get("avg_delay", 0.0)
        if avg_delay > 0:
            avg_throughput = (avg_task_size_mb * num_tasks_per_step) / avg_delay

    metrics["avg_throughput_mbps"] = max(avg_throughput, 0.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    labels = [record["label"] for record in results]
    x_positions = range(len(results))

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(x_positions, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xticks(x_positions, labels)
        plt.ylabel(ylabel)
        plt.title(f"Impact of Bandwidth on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "bandwidth_vs_total_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "bandwidth_vs_delay.png")
    make_chart("normalized_cost", "Normalized Cost", "bandwidth_vs_normalized_cost.png")
    make_chart("avg_throughput_mbps", "Average Throughput (Mbps)", "bandwidth_vs_throughput.png")

    print("\nCharts saved:")
    for name in [
        "bandwidth_vs_total_cost.png",
        "bandwidth_vs_delay.png",
        "bandwidth_vs_normalized_cost.png",
        "bandwidth_vs_throughput.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under varied channel bandwidth."
    )
    parser.add_argument(
        "--bandwidths",
        type=str,
        default="default",
        help="Comma-separated bandwidth list in MHz or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="bandwidth",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="bandwidth",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    bandwidths = parse_bandwidths(args.bandwidths)
    configs: List[Dict[str, object]] = []
    for bw in bandwidths:
        overrides = {
            "bandwidth": float(bw),
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{bw}mhz",
                "label": f"{bw} MHz",
                "overrides": overrides,
                "bandwidth_mhz": bw,
                "assumed_tasks_per_step": 12,
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
        per_strategy_hook=bandwidth_hook,
    )

    summary = {
        "experiment_type": "bandwidth_cost_sensitivity",
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

    print("\nBandwidth Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Bandwidth':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['bandwidth_mhz']:<12}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
