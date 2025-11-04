#!/usr/bin/env python3
"""
CAMTD3 边缘计算能力成本对比实验
================================

研究目标
--------
- 评估不同边缘计算能力配置对系统成本的影响
- 对比低配、标准、高配、超高配四种计算资源配置
- 分析计算能力提升的边际收益递减效应
- 为基础设施投资决策提供量化依据

计算能力配置
------------
配置维度：
1. RSU计算频率：5 GHz → 20 GHz
2. UAV计算频率：4 GHz → 16 GHz
3. 保持车辆计算能力不变（2 GHz）

配置等级：
- 低配 (Low): RSU=5GHz, UAV=4GHz (成本最低，性能较弱)
- 标准 (Standard): RSU=10GHz, UAV=8GHz (基准配置)
- 高配 (High): RSU=15GHz, UAV=12GHz (高性能)
- 超高配 (Ultra): RSU=20GHz, UAV=16GHz (顶级配置)
- 极限配 (Extreme): RSU=25GHz, UAV=20GHz (超出典型范围)

学术价值
--------
- 支撑论文中关于"计算资源配置优化"的讨论
- 展示边际收益递减规律
- 为实际部署提供成本-性能最优配置建议
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

# 边缘计算能力配置
# 格式: (RSU_GHz, UAV_GHz, label, description, relative_cost)
COMPUTE_CAPACITY_CONFIGS = [{
        "key": "low", "rsu_compute_ghz": 15.0, }]


def compute_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算边缘计算能力相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 计算成本效率（完成率/成本）
    if metrics["raw_cost"] > 0:
        metrics["cost_efficiency"] = metrics["completion_rate"] / metrics["raw_cost"]
    else:
        metrics["cost_efficiency"] = 0.0
    
    # 计算边际收益（相对于低配的性能提升）
    # 这个在plot阶段计算更合适


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图，包括边际收益分析"""
    
    config_labels = [record["config_label"] for record in results]
    compute_levels = [record["rsu_compute_ghz"] for record in results]
    relative_costs = [record["relative_cost"] for record in results]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # ========== 子图1: 平均成本 vs 计算能力 ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(compute_levels, values, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("RSU Compute Capacity (GHz)")
    ax.set_ylabel("Average Cost")
    ax.set_title("Cost vs. Edge Computing Capacity")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    x_pos = range(len(config_labels))
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(x_pos, values, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_ylabel("Completion Rate (%)")
    ax.set_title("Task Completion Across Compute Levels")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图3: 成本效率（完成率/成本）==========
    ax = axes[0, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["cost_efficiency"] for record in results]
        ax.plot(x_pos, values, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_ylabel("Cost Efficiency (Completion/Cost)")
    ax.set_title("Cost-Effectiveness Analysis")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图4: 边际收益分析（成本下降率）==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        # 计算相对于第一个配置的成本改善百分比
        baseline = costs[0]
        improvements = [(baseline - c) / baseline * 100 for c in costs]
        ax.plot(compute_levels, improvements, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("RSU Compute Capacity (GHz)")
    ax.set_ylabel("Cost Reduction (%)")
    ax.set_title("Marginal Benefit Analysis")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图5: 卸载率变化 ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        ax.plot(x_pos, values, marker="v", linewidth=2, label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=20, ha="right")
    ax.set_ylabel("Offload Ratio (%)")
    ax.set_title("Offloading Decision Adaptation")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    # ========== 子图6: 投资回报率（ROI）分析 ==========
    ax = axes[1, 2]
    # ROI = (性能提升) / (成本增加)
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        baseline_cost = costs[0]
        
        # 计算ROI：成本改善 / 硬件成本增加
        roi_values = []
        for i, (cost, rel_cost) in enumerate(zip(costs, relative_costs)):
            cost_improvement = baseline_cost - cost
            hardware_cost_increase = rel_cost - 1.0  # 相对于低配的硬件成本增加
            if hardware_cost_increase > 0:
                roi = cost_improvement / hardware_cost_increase
            else:
                roi = 0.0
            roi_values.append(roi)
        
        ax.plot(compute_levels, roi_values, marker="*", linewidth=2, label=strategy_label(strat_key))
    ax.set_xlabel("RSU Compute Capacity (GHz)")
    ax.set_ylabel("ROI (Cost Saving / HW Investment)")
    ax.set_title("Return on Investment Analysis")
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    output_path = suite_dir / "edge_compute_capacity_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")
    
    # ========== 生成配置推荐报告 ==========
    recommendation_path = suite_dir / "compute_capacity_recommendation.txt"
    with open(recommendation_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("边缘计算能力配置推荐报告\n")
        f.write("=" * 80 + "\n\n")
        
        for compute_config in COMPUTE_CAPACITY_CONFIGS:
            f.write(f"【{compute_config['label']}】\n")
            f.write(f"  配置: RSU={compute_config['rsu_compute_ghz']}GHz, "
                   f"UAV={compute_config['uav_compute_ghz']}GHz\n")
            f.write(f"  硬件成本: {compute_config['relative_cost']:.1f}x (相对于低配)\n")
            f.write(f"  适用场景: {compute_config['description']}\n")
            f.write(f"  性能表现:\n")
            
            # 找到对应的结果
            matching_result = next((r for r in results if r["config_key"] == compute_config["key"]), None)
            if matching_result:
                for strat_key in strategy_keys:
                    metrics = matching_result["strategies"][strat_key]
                    f.write(f"    - {strategy_label(strat_key)}: "
                           f"成本={metrics['raw_cost']:.4f}, "
                           f"完成率={metrics['completion_rate']*100:.1f}%, "
                           f"效率={metrics['cost_efficiency']:.4f}\n")
            f.write("\n")
        
        # 添加总结建议
        f.write("=" * 80 + "\n")
        f.write("配置建议总结\n")
        f.write("=" * 80 + "\n")
        f.write("1. 成本敏感型场景: 推荐 Low-Config 或 Standard-Config\n")
        f.write("2. 性能优先型场景: 推荐 High-Config\n")
        f.write("3. 平衡配置: Standard-Config 通常性价比最高\n")
        f.write("4. 注意: Ultra/Extreme配置可能出现边际收益递减\n")
    
    print(f"Recommendation report saved: {recommendation_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under different edge computing capacity configurations."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="edge_compute_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="edge_compute_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for compute_config in COMPUTE_CAPACITY_CONFIGS:
        overrides = {
            "rsu_compute_ghz": compute_config["rsu_compute_ghz"],
            "uav_compute_ghz": compute_config["uav_compute_ghz"],
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": compute_config["key"],
                "label": f"{compute_config['label']} ({compute_config['rsu_compute_ghz']}GHz/{compute_config['uav_compute_ghz']}GHz)",
                "overrides": overrides,
                "config_key": compute_config["key"],
                "config_label": compute_config["label"],
                "config_description": compute_config["description"],
                "rsu_compute_ghz": compute_config["rsu_compute_ghz"],
                "uav_compute_ghz": compute_config["uav_compute_ghz"],
                "relative_cost": compute_config["relative_cost"],
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
        per_strategy_hook=compute_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "edge_compute_capacity_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "compute_capacity_configs": COMPUTE_CAPACITY_CONFIGS,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nEdge Computing Capacity Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Config':<18} {'RSU(GHz)':>10} {'UAV(GHz)':>10} {'HW Cost':>8}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (46 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['config_label']:<18} {record['rsu_compute_ghz']:>10.1f} "
              f"{record['uav_compute_ghz']:>10.1f} {record['relative_cost']:>8.1f}x", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

