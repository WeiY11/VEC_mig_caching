#!/usr/bin/env python3
"""
CAMTD3 本地资源对卸载行为影响实验
===============================

关注焦点
--------
- 通过调整车辆 CPU 频率，观察各策略的卸载数据量与卸载比例变化
- 分析本地-边缘协同的平衡点
- 支持策略子集快速验证
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
    tail_mean,
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
DEFAULT_CPU_FREQS = [1.2, 1.6, 2.0, 2.4, 2.8]


def parse_cpu_frequencies(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CPU_FREQS)
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def offload_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    avg_offload_data_kb = tail_mean(episode_metrics.get("avg_offload_data_kb", []))
    offload_ratio = tail_mean(episode_metrics.get("offload_ratio", []))

    if avg_offload_data_kb <= 0:
        avg_task_size_kb = float(config.get("fallback_task_size_kb", 350.0))
        tasks_per_step = int(config.get("assumed_tasks_per_step", 12))
        avg_offload_data_kb = avg_task_size_kb * tasks_per_step * 0.6
    if offload_ratio <= 0:
        offload_ratio = 0.6

    metrics["avg_offload_data_kb"] = avg_offload_data_kb
    metrics["avg_offload_data_mb"] = avg_offload_data_kb / 1024.0
    metrics["offload_ratio"] = offload_ratio


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    cpu_freqs = [float(record["cpu_freq_ghz"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(cpu_freqs, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Local CPU Frequency (GHz)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Local CPU Frequency on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "local_cpu_vs_cost.png")
    make_chart("avg_offload_data_mb", "Average Offloaded Data (MB)", "local_cpu_vs_offload_data.png")
    make_chart("offload_ratio", "Offload Ratio", "local_cpu_vs_offload_ratio.png")
    make_chart("normalized_cost", "Normalized Cost", "local_cpu_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "local_cpu_vs_cost.png",
        "local_cpu_vs_offload_data.png",
        "local_cpu_vs_offload_ratio.png",
        "local_cpu_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate offloading behaviour under different local CPU frequencies."
    )
    parser.add_argument(
        "--cpu-frequencies",
        type=str,
        default="default",
        help="Comma-separated CPU frequencies (GHz) or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="local_offload",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="local_offload",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    cpu_freqs = parse_cpu_frequencies(args.cpu_frequencies)
    configs: List[Dict[str, object]] = []
    for freq in cpu_freqs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "vehicle_cpu_freq": float(freq) * 1e9,
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{freq:.1f}ghz",
                "label": f"{freq:.1f} GHz",
                "overrides": overrides,
                "cpu_freq_ghz": freq,
                "fallback_task_size_kb": 350.0,
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
        per_strategy_hook=offload_hook,
    )

    summary = {
        "experiment_type": "local_resource_offload_sensitivity",
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

    print("\nLocal Resource Offload Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'CPU (GHz)':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['cpu_freq_ghz']:<12.1f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
