#!/usr/bin/env python3
"""
CAMTD3 资源异构性对比实验
==========================

研究目标
--------
- 评估不同计算资源分布对系统性能的影响
- 模拟同构、轻度异构、中度异构、重度异构四种场景
- 分析算法在资源不均衡环境下的适应能力
- 验证负载均衡策略的有效性

资源配置场景
------------
1. 同构配置 (Homogeneous): 所有节点能力相同
2. 轻度异构 (Light-Hetero): RSU能力差异小，UAV统一
3. 中度异构 (Medium-Hetero): RSU能力差异中等，UAV有高低配
4. 重度异构 (Heavy-Hetero): RSU能力差异大，UAV差异明显
5. 极端异构 (Extreme-Hetero): 少数超强节点+多数弱节点（现实场景）

学术价值
--------
- 支撑论文中关于"异构资源调度优化"的讨论
- 验证算法的公平性和负载均衡能力
- 为实际混合部署场景提供理论指导
- 与资源感知算法的对比基准
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

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

# 资源异构性配置
# 基准: RSU=10 GHz, UAV=8 GHz, Vehicle=2 GHz
HETEROGENEITY_CONFIGS = [
    {
        "key": "homogeneous",
        "label": "Homogeneous",
        "description": "同构配置（理想情况）",
        "rsu_compute_range": (10.0, 10.0),  # (min, max) GHz
        "uav_compute_range": (8.0, 8.0),
        "vehicle_compute_range": (2.0, 2.0),
        "heterogeneity_level": 0.0,
    },
    {
        "key": "light_hetero",
        "label": "Light-Hetero",
        "description": "轻度异构（±20%）",
        "rsu_compute_range": (8.0, 12.0),
        "uav_compute_range": (7.0, 9.0),
        "vehicle_compute_range": (1.8, 2.2),
        "heterogeneity_level": 0.2,
    },
    {
        "key": "medium_hetero",
        "label": "Medium-Hetero",
        "description": "中度异构（±40%）",
        "rsu_compute_range": (6.0, 14.0),
        "uav_compute_range": (5.0, 11.0),
        "vehicle_compute_range": (1.5, 2.5),
        "heterogeneity_level": 0.4,
    },
    {
        "key": "heavy_hetero",
        "label": "Heavy-Hetero",
        "description": "重度异构（±60%）",
        "rsu_compute_range": (4.0, 16.0),
        "uav_compute_range": (3.5, 12.5),
        "vehicle_compute_range": (1.0, 3.0),
        "heterogeneity_level": 0.6,
    },
    {
        "key": "extreme_hetero",
        "label": "Extreme-Hetero",
        "description": "极端异构（2x~5x差异）",
        "rsu_compute_range": (3.0, 20.0),
        "uav_compute_range": (2.0, 15.0),
        "vehicle_compute_range": (0.5, 4.0),
        "heterogeneity_level": 0.8,
    },
]


def hetero_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算异构性相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 计算负载均衡度（时延标准差，越小越均衡）
    delay_list = episode_metrics.get("avg_delay", [])
    if delay_list:
        metrics["load_balance_score"] = 1.0 / (1.0 + float(np.std(delay_list)))
    else:
        metrics["load_balance_score"] = 0.0
    
    # 计算资源利用率方差（模拟，实际需要从环境获取）
    metrics["resource_utilization_variance"] = config.get("heterogeneity_level", 0.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图"""
    
    hetero_labels = [record["hetero_label"] for record in results]
    hetero_levels = [record["heterogeneity_level"] for record in results]
    x_pos = range(len(hetero_labels))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== 子图1: 平均成本 vs 异构度 ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(hetero_levels, values, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("Heterogeneity Level")
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost vs. Resource Heterogeneity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(x_pos, values, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hetero_labels, rotation=20, ha="right")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion Across Heterogeneity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图3: 负载均衡分数 ==========
    ax = axes[0, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["load_balance_score"] for record in results]
        ax.plot(x_pos, values, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hetero_labels, rotation=20, ha="right")
    ax.set_ylabel("Load Balance Score")
    ax.set_title("Load Balancing Performance")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图4: 平均时延 ==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["avg_delay"] for record in results]
        ax.plot(x_pos, values, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hetero_labels, rotation=20, ha="right")
    ax.set_ylabel("Average Delay (s)")
    ax.set_title("Delay Across Heterogeneity Levels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图5: 卸载率 ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        ax.plot(x_pos, values, marker="v", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hetero_labels, rotation=20, ha="right")
    ax.set_ylabel("Offload Ratio (%)")
    ax.set_title("Offloading Decision Adaptation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图6: 相对性能下降 ==========
    ax = axes[1, 2]
    # 计算相对于同构配置的性能下降百分比
    baseline_costs = {}
    for strat_key in strategy_keys:
        baseline_costs[strat_key] = results[0]["strategies"][strat_key]["raw_cost"]
    
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        degradation = [(v - baseline_costs[strat_key]) / baseline_costs[strat_key] * 100 
                      for v in values]
        ax.plot(x_pos, degradation, marker="*", linewidth=2, label=strategy_label(strat_key))
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(hetero_labels, rotation=20, ha="right")
    ax.set_ylabel("Performance Degradation (%)")
    ax.set_title("Robustness to Heterogeneity")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    output_path = suite_dir / "resource_heterogeneity_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under different resource heterogeneity levels."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="resource_hetero",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="resource_hetero",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for hetero_config in HETEROGENEITY_CONFIGS:
        # 注意: 这里需要与SystemConfig配合，实际实现中可能需要扩展配置系统
        # 目前作为示例，使用hetero_level参数
        overrides = {
            "heterogeneity_level": hetero_config["heterogeneity_level"],
            # 未来可扩展: rsu_compute_range, uav_compute_range等
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": hetero_config["key"],
                "label": f"{hetero_config['label']} (Level={hetero_config['heterogeneity_level']:.1f})",
                "overrides": overrides,
                "hetero_label": hetero_config["label"],
                "hetero_description": hetero_config["description"],
                "heterogeneity_level": hetero_config["heterogeneity_level"],
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
        per_strategy_hook=hetero_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "resource_heterogeneity_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "heterogeneity_configs": HETEROGENEITY_CONFIGS,
        "results": results,
        "note": "Heterogeneity levels are simulated. Full implementation requires SystemConfig extension.",
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nResource Heterogeneity Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Heterogeneity':<18} {'Level':>8}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (26 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['hetero_label']:<18} {record['heterogeneity_level']:>8.1f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")
    print("\nNote: This experiment simulates heterogeneity. Full resource distribution")
    print("      control requires extending SystemConfig with per-node capabilities.")


if __name__ == "__main__":
    main()

