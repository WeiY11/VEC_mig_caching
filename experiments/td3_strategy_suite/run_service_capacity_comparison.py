#!/usr/bin/env python3
"""
TD3 边缘服务能力对比实验
==========================

研究目标
--------
- 调整 RSU/UAV 的服务处理能力，观察各策略在排队压力下的表现差异
- 关注平均成本、时延、队列负载 (ρ) 以及过载事件等关键指标
- 帮助确定部署时的节点算力配置与冗余度
"""

from __future__ import annotations

import argparse
import json
import math
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
    get_default_scenario_overrides,
)

DEFAULT_EPISODES = 500
DEFAULT_SEED = 42
DEFAULT_SERVICE_FACTORS = [0.6, 0.8, 1.0, 1.2, 1.4]  # (极低/低/基准/高/极高)


def parse_service_factors(value: str) -> List[float]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_SERVICE_FACTORS)
    parsed: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        parsed.append(float(item))
    if not parsed:
        raise ValueError("Service factor list cannot be empty.")
    return parsed


def _scale_capacity(base: float, factor: float, minimum: float) -> float:
    return max(minimum, base * factor)


def derive_service_parameters(factor: float) -> Dict[str, float]:
    rsu_base = max(1, int(round(4 * factor)))
    rsu_max = max(rsu_base + 1, int(round(9 * factor)))
    uav_base = max(1, int(round(3 * factor)))
    uav_max = max(uav_base + 1, int(round(6 * factor)))
    rsu_work_capacity = _scale_capacity(2.5, factor, 1.0)
    uav_work_capacity = _scale_capacity(1.7, factor, 0.5)
    return {
        "rsu_base_service": rsu_base,
        "rsu_max_service": rsu_max,
        "uav_base_service": uav_base,
        "uav_max_service": uav_max,
        "rsu_work_capacity": rsu_work_capacity,
        "uav_work_capacity": uav_work_capacity,
    }


def service_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    metrics["queue_rho_max"] = max(tail_mean(episode_metrics.get("queue_rho_max", [])), 0.0)
    metrics["queue_overload_events"] = max(tail_mean(episode_metrics.get("queue_overload_events", [])), 0.0)
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    factors = [float(record["service_factor"]) for record in results]

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(factors, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xlabel("Service Capacity Scaling Factor")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Service Capacity on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "service_capacity_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "service_capacity_vs_delay.png")
    make_chart("queue_rho_max", "Peak Queue Utilisation (ρ)", "service_capacity_vs_queue_rho.png")
    make_chart("completion_rate", "Task Completion Rate", "service_capacity_vs_completion.png")

    print("\nCharts saved:")
    for name in [
        "service_capacity_vs_cost.png",
        "service_capacity_vs_delay.png",
        "service_capacity_vs_queue_rho.png",
        "service_capacity_vs_completion.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate how RSU/UAV service capacity scaling affects TD3 strategy performance."
    )
    parser.add_argument(
        "--service-factors",
        type=str,
        default="default",
        help="Comma-separated scaling factors (e.g., '0.8,1.0,1.2') or 'default'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="service_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="service_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    service_factors = parse_service_factors(args.service_factors)
    configs: List[Dict[str, object]] = []
    for factor in service_factors:
        params = derive_service_parameters(factor)
        overrides = get_default_scenario_overrides(**params)
        configs.append(
            {
                "key": f"svc_{factor:.2f}",
                "label": f"{factor:.2f}× Capacity",
                "overrides": overrides,
                "service_factor": float(factor),
                "service_parameters": params,
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
        per_strategy_hook=service_metrics_hook,
    )

    summary = {
        "experiment_type": "service_capacity_scaling",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "service_factors": service_factors,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    print("\nService Capacity Scaling Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'Factor':<10}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (10 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['service_factor']:<10.2f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

