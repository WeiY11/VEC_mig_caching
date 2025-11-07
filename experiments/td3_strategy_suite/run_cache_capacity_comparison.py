#!/usr/bin/env python3
"""
TD3 协同缓存容量对比实验
=============================

研究目标
--------
- 扫描协同缓存容量（RSU/UAV 共享缓存）对六类策略性能的影响
- 关注缓存命中率、缓存利用率、数据丢失比例等指标的变化趋势
- 为论文图表或报告提供可复现的灵敏度分析脚本
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
DEFAULT_CACHE_LEVELS_MB = [256, 512, 768, 1024, 1536]  # MB (极小/小/中/大/极大)


def parse_cache_levels(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CACHE_LEVELS_MB)
    parsed: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("Cache capacity list cannot be empty.")
    return parsed


def cache_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    hit_rate = tail_mean(episode_metrics.get("cache_hit_rate", []))
    utilization = tail_mean(episode_metrics.get("cache_utilization", []))
    eviction_rate = tail_mean(episode_metrics.get("cache_eviction_rate", []))
    loss_ratio = tail_mean(episode_metrics.get("data_loss_ratio_bytes", []))

    metrics["cache_hit_rate"] = max(hit_rate, 0.0)
    metrics["cache_utilization"] = max(utilization, 0.0)
    metrics["cache_eviction_rate"] = max(eviction_rate, 0.0)
    metrics["data_loss_ratio"] = max(loss_ratio, 0.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    capacities = [float(record["cache_capacity_mb"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(capacities, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Collaborative Cache Capacity (MB)")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Cache Capacity on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "cache_capacity_vs_cost.png")
    make_chart("cache_hit_rate", "Cache Hit Rate", "cache_capacity_vs_hit_rate.png")
    make_chart("cache_utilization", "Cache Utilization", "cache_capacity_vs_utilization.png")
    make_chart("data_loss_ratio", "Data Loss Ratio", "cache_capacity_vs_loss_ratio.png")

    print("\nCharts saved:")
    for name in [
        "cache_capacity_vs_cost.png",
        "cache_capacity_vs_hit_rate.png",
        "cache_capacity_vs_utilization.png",
        "cache_capacity_vs_loss_ratio.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate TD3 strategy performance under different collaborative cache capacities."
    )
    parser.add_argument(
        "--cache-levels",
        type=str,
        default="default",
        help="Comma-separated cache capacity values (MB) or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="cache_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="cache_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    cache_levels = parse_cache_levels(args.cache_levels)
    configs: List[Dict[str, object]] = []
    for capacity_mb in cache_levels:
        overrides = {
            "cache_capacity": float(capacity_mb),
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": f"{int(capacity_mb)}mb",
                "label": f"{int(capacity_mb)} MB",
                "overrides": overrides,
                "cache_capacity_mb": float(capacity_mb),
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
        per_strategy_hook=cache_metrics_hook,
    )

    summary = {
        "experiment_type": "cache_capacity_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "cache_levels_mb": cache_levels,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    print("\nCache Capacity Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Capacity (MB)':<16}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (16 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['cache_capacity_mb']:<16.0f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

