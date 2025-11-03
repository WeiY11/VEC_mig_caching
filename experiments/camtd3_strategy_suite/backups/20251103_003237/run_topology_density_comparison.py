#!/usr/bin/env python3
"""
CAMTD3 网络拓扑密度对比实验
============================

研究目标
--------
- 评估不同RSU部署密度对系统性能的影响
- 分析稀疏、适中、密集三种拓扑配置下各策略的表现
- 研究基础设施投资与性能收益的关系（论文经济性分析）
- 验证算法在不同网络覆盖条件下的鲁棒性

学术价值
--------
- 支撑论文中关于"基础设施部署优化"的讨论
- 提供RSU密度与系统性能的量化关系
- 为实际部署提供理论指导（成本-性能权衡）
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

# 拓扑密度配置：(num_rsus, num_uavs, label, description)
TOPOLOGY_CONFIGS = [
    (2, 1, "Sparse", "低密度（农村/郊区）"),
    (3, 1, "Low", "中低密度（小城市）"),
    (4, 2, "Medium", "适中密度（标准）"),
    (6, 2, "High", "高密度（城市）"),
    (8, 3, "Dense", "密集部署（CBD）"),
]


def topology_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算拓扑相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = max(tail_mean(episode_metrics.get("offload_ratio", [])), 0.0)
    
    # 计算资源利用率（归一化）
    num_nodes = config.get("num_rsus", 4) + config.get("num_uavs", 2)
    metrics["resource_efficiency"] = metrics["completion_rate"] / max(num_nodes, 1)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图"""
    
    # 提取拓扑标签（用于横坐标）
    topology_labels = [record["topology_label"] for record in results]
    x_pos = range(len(topology_labels))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # ========== 子图1: 平均成本 ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(x_pos, values, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topology_labels, rotation=15)
    ax.set_xlabel("Topology Density")
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost vs. RSU Density")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(x_pos, values, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topology_labels, rotation=15)
    ax.set_xlabel("Topology Density")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion vs. RSU Density")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # ========== 子图3: 卸载率 ==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        ax.plot(x_pos, values, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topology_labels, rotation=15)
    ax.set_xlabel("Topology Density")
    ax.set_ylabel("Offload Ratio (%)")
    ax.set_title("Offloading Behavior vs. RSU Density")
    ax.grid(alpha=0.3)
    ax.legend()
    
    # ========== 子图4: 资源效率 ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["resource_efficiency"] for record in results]
        ax.plot(x_pos, values, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(topology_labels, rotation=15)
    ax.set_xlabel("Topology Density")
    ax.set_ylabel("Resource Efficiency")
    ax.set_title("Resource Utilization Efficiency")
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    output_path = suite_dir / "topology_density_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under different network topology densities."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="topology_density",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="topology_density",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for num_rsus, num_uavs, label, description in TOPOLOGY_CONFIGS:
        overrides = {
            "override_topology": True,
            "num_vehicles": 12,  # 固定车辆数
            "num_rsus": num_rsus,
            "num_uavs": num_uavs,
        }
        configs.append(
            {
                "key": f"topo_{label.lower()}",
                "label": f"{label} ({num_rsus}R+{num_uavs}U)",
                "overrides": overrides,
                "topology_label": label,
                "topology_description": description,
                "num_rsus": num_rsus,
                "num_uavs": num_uavs,
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
        per_strategy_hook=topology_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "topology_density_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "topology_configs": [
            {"label": label, "rsus": rsus, "uavs": uavs, "description": desc}
            for rsus, uavs, label, desc in TOPOLOGY_CONFIGS
        ],
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nTopology Density Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Topology':<15} {'Nodes':>8}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (23 + 18 * len(strategy_keys)))
    for record in results:
        nodes_str = f"{record['num_rsus']}R+{record['num_uavs']}U"
        print(f"{record['topology_label']:<15} {nodes_str:>8}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

