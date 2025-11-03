#!/usr/bin/env python3
"""
CAMTD3 边缘节点配置对比实验
==========================

主要目标
--------
- 探索不同 RSU/UAV 组合对系统性能的影响
- 对比六种策略（或指定子集）在基础设施扩展下的收益
- 评估单位节点成本 (cost per node) 等衍生指标
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

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
DEFAULT_CONFIGS: List[Tuple[int, int, str]] = [
    (2, 0, "2 RSU, 0 UAV"),      # 优化: 小规模
    (4, 2, "4 RSU, 2 UAV"),      # 优化: 中规模 (基准)
    (6, 3, "6 RSU, 3 UAV"),      # 优化: 大规模
]  # 优化: 5配置→3配置


def parse_configurations(value: str) -> List[Tuple[int, int, str]]:
    if not value or value.strip().lower() == "default":
        return list(DEFAULT_CONFIGS)
    configs: List[Tuple[int, int, str]] = []
    for item in value.split(";"):
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) < 2:
            raise ValueError(f"Invalid edge node specification: {item}")
        num_rsus = int(parts[0])
        num_uavs = int(parts[1])
        label = parts[2] if len(parts) > 2 else f"{num_rsus} RSU, {num_uavs} UAV"
        configs.append((num_rsus, num_uavs, label))
    return configs


def edge_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    total_nodes = int(config["num_rsus"]) + int(config["num_uavs"])
    metrics["total_nodes"] = total_nodes
    metrics["cost_per_node"] = metrics["raw_cost"] / max(total_nodes, 1)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    labels = [record["label"] for record in results]
    x_positions = range(len(results))

    def make_chart(metric: str, ylabel: str, filename: str) -> None:
        plt.figure(figsize=(10, 6))
        for strat_key in strategy_keys:
            values = [record["strategies"][strat_key][metric] for record in results]
            plt.plot(x_positions, values, marker="o", linewidth=2, label=strategy_label(strat_key))
        plt.xticks(x_positions, labels, rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Impact of Edge Node Configuration on {ylabel}")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(suite_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()

    make_chart("raw_cost", "Average Cost", "edge_config_vs_cost.png")
    make_chart("avg_delay", "Average Delay (s)", "edge_config_vs_delay.png")
    make_chart("cost_per_node", "Cost per Node", "edge_config_vs_cost_per_node.png")
    make_chart("normalized_cost", "Normalized Cost", "edge_config_vs_normalized_cost.png")

    print("\nCharts saved:")
    for name in [
        "edge_config_vs_cost.png",
        "edge_config_vs_delay.png",
        "edge_config_vs_cost_per_node.png",
        "edge_config_vs_normalized_cost.png",
    ]:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate strategy performance across edge node configurations."
    )
    parser.add_argument(
        "--configurations",
        type=str,
        default="default",
        help="Semicolon-separated list like '4,2,Label;6,2,Label'.",
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="edge_node",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="edge_node",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    node_configs = parse_configurations(args.configurations)
    configs: List[Dict[str, object]] = []
    for num_rsus, num_uavs, label in node_configs:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": int(num_rsus),
            "num_uavs": int(num_uavs),
            "override_topology": True,
        }
        configs.append(
            {
                "key": f"{num_rsus}rsu_{num_uavs}uav",
                "label": label,
                "overrides": overrides,
                "num_rsus": int(num_rsus),
                "num_uavs": int(num_uavs),
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
        per_strategy_hook=edge_hook,
    )

    summary = {
        "experiment_type": "edge_node_configuration",
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

    print("\nEdge Node Configuration Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"{'RSU/UAV':<12}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (12 + 18 * len(strategy_keys)))
    for record in results:
        label = f"{record['num_rsus']} / {record['num_uavs']}"
        print(f"{label:<12}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()
