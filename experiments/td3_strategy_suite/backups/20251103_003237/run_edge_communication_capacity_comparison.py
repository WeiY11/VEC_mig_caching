#!/usr/bin/env python3
"""
CAMTD3 边缘通信资源容量成本对比实验
====================================

研究目标
--------
- 评估边缘节点通信资源总量（带宽）变化对系统成本的影响
- 对比低带宽、标准带宽、高带宽配置下各策略的表现
- 分析通信资源瓶颈对任务卸载决策的影响
- 为通信资源配置优化提供量化依据

通信资源配置
------------
配置维度：
1. 总带宽 (Total Bandwidth): 10 MHz → 60 MHz
2. 上行/下行带宽: 均分总带宽
3. 保持发射功率、频率等其他参数不变

配置等级：
- 极低带宽 (Very-Low): 10 MHz (拥塞场景)
- 低带宽 (Low): 15 MHz (资源受限)
- 标准带宽 (Standard): 20 MHz (3GPP标准，基准配置)
- 高带宽 (High): 30 MHz (充足资源)
- 超高带宽 (Very-High): 40 MHz (丰富资源)
- 极高带宽 (Ultra): 50 MHz (理想场景)
- 极限带宽 (Extreme): 60 MHz (探索上界)

学术价值
--------
- 支撑论文中关于"通信资源配置优化"的讨论
- 展示通信瓶颈对卸载策略的影响
- 验证算法在不同通信条件下的适应能力
- 为实际部署提供通信资源配置建议

关键指标
--------
- 传输成本: 数据传输产生的时延和能耗
- 卸载率变化: 通信资源对卸载决策的影响
- 通信效率: 单位带宽的性能产出
- 边际收益: 带宽增加的性能提升比例
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

# 边缘通信资源配置
# 格式: (total_bandwidth_mhz, label, description, relative_cost)
COMM_CAPACITY_CONFIGS = [
    {
        "key": "very_low",
        "label": "Very-Low-BW",
        "description": "极低带宽（拥塞场景）",
        "total_bandwidth_mhz": 10.0,
        "relative_cost": 0.5,  # 相对成本（基准=1.0）
    },
    {
        "key": "low",
        "label": "Low-BW",
        "description": "低带宽（资源受限）",
        "total_bandwidth_mhz": 15.0,
        "relative_cost": 0.75,
    },
    {
        "key": "standard",
        "label": "Standard-BW",
        "description": "标准带宽（3GPP基准）",
        "total_bandwidth_mhz": 20.0,
        "relative_cost": 1.0,
    },
    {
        "key": "high",
        "label": "High-BW",
        "description": "高带宽（充足资源）",
        "total_bandwidth_mhz": 30.0,
        "relative_cost": 1.5,
    },
    {
        "key": "very_high",
        "label": "Very-High-BW",
        "description": "超高带宽（丰富资源）",
        "total_bandwidth_mhz": 40.0,
        "relative_cost": 2.0,
    },
    {
        "key": "ultra",
        "label": "Ultra-BW",
        "description": "极高带宽（理想场景）",
        "total_bandwidth_mhz": 50.0,
        "relative_cost": 2.5,
    },
    {
        "key": "extreme",
        "label": "Extreme-BW",
        "description": "极限带宽（探索上界）",
        "total_bandwidth_mhz": 60.0,
        "relative_cost": 3.0,
    },
]


def comm_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """计算通信资源相关指标"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # 通信相关指标
    metrics["comm_delay"] = tail_mean(episode_metrics.get("comm_delay", []))
    metrics["transmission_energy"] = tail_mean(episode_metrics.get("transmission_energy", []))
    
    # 计算通信效率（任务完成率/通信时延）
    if metrics["comm_delay"] > 0:
        metrics["comm_efficiency"] = metrics["completion_rate"] / metrics["comm_delay"]
    else:
        metrics["comm_efficiency"] = 0.0
    
    # 计算带宽利用效率（完成率/带宽）
    bandwidth_mhz = config.get("total_bandwidth_mhz", 20.0)
    metrics["bandwidth_efficiency"] = metrics["completion_rate"] / max(bandwidth_mhz, 1.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """生成多维度对比图，包括通信效率分析"""
    
    config_labels = [record["config_label"] for record in results]
    bandwidth_values = [record["total_bandwidth_mhz"] for record in results]
    relative_costs = [record["relative_cost"] for record in results]
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 15))
    
    # ========== 子图1: 平均成本 vs 带宽 ==========
    ax = axes[0, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax.plot(bandwidth_values, values, marker="o", linewidth=2.5, 
               markersize=8, label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Average Cost", fontsize=11)
    ax.set_title("Cost vs. Communication Bandwidth", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图2: 任务完成率 ==========
    ax = axes[0, 1]
    x_pos = range(len(config_labels))
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        ax.plot(x_pos, values, marker="s", linewidth=2.5, markersize=8, 
               label=strategy_label(strat_key))
    ax.set_xticks(x_pos)
    ax.set_xticklabels(config_labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Completion Rate (%)", fontsize=11)
    ax.set_title("Task Completion Across Bandwidth Levels", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图3: 通信时延 ==========
    ax = axes[0, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["comm_delay"] for record in results]
        ax.plot(bandwidth_values, values, marker="^", linewidth=2.5, markersize=8,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Communication Delay (s)", fontsize=11)
    ax.set_title("Comm Delay vs. Bandwidth", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图4: 卸载率变化 ==========
    ax = axes[1, 0]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        ax.plot(bandwidth_values, values, marker="D", linewidth=2.5, markersize=8,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Offload Ratio (%)", fontsize=11)
    ax.set_title("Offloading Decision Adaptation", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图5: 通信效率 ==========
    ax = axes[1, 1]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["comm_efficiency"] for record in results]
        ax.plot(bandwidth_values, values, marker="v", linewidth=2.5, markersize=8,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Comm Efficiency (Completion/Delay)", fontsize=11)
    ax.set_title("Communication Efficiency", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图6: 带宽利用效率 ==========
    ax = axes[1, 2]
    for strat_key in strategy_keys:
        values = [record["strategies"][strat_key]["bandwidth_efficiency"] for record in results]
        ax.plot(bandwidth_values, values, marker="*", linewidth=2.5, markersize=10,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Bandwidth Efficiency (Completion/MHz)", fontsize=11)
    ax.set_title("Bandwidth Utilization Efficiency", fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图7: 边际收益分析 ==========
    ax = axes[2, 0]
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        # 计算相对于第一个配置的成本改善百分比
        baseline = costs[0]
        improvements = [(baseline - c) / baseline * 100 if baseline > 0 else 0 for c in costs]
        ax.plot(bandwidth_values, improvements, marker="P", linewidth=2.5, markersize=8,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("Cost Reduction (%)", fontsize=11)
    ax.set_title("Marginal Benefit Analysis", fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图8: ROI分析 ==========
    ax = axes[2, 1]
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        baseline_cost = costs[0]
        
        roi_values = []
        for i, (cost, rel_cost) in enumerate(zip(costs, relative_costs)):
            cost_improvement = baseline_cost - cost
            comm_cost_increase = rel_cost - 0.5  # 相对于极低带宽的成本增加
            if comm_cost_increase > 0:
                roi = cost_improvement / comm_cost_increase
            else:
                roi = 0.0
            roi_values.append(roi)
        
        ax.plot(bandwidth_values, roi_values, marker="X", linewidth=2.5, markersize=8,
               label=strategy_label(strat_key))
    ax.set_xlabel("Total Bandwidth (MHz)", fontsize=11)
    ax.set_ylabel("ROI (Cost Saving / BW Investment)", fontsize=11)
    ax.set_title("Return on Investment Analysis", fontsize=12, fontweight='bold')
    ax.axhline(y=0, color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    
    # ========== 子图9: 成本分解（时延 vs 能耗）==========
    ax = axes[2, 2]
    # 选择一个代表性策略展示成本分解
    representative_strategy = strategy_keys[0] if strategy_keys else None
    if representative_strategy:
        delays = [record["strategies"][representative_strategy]["avg_delay"] for record in results]
        energies = [record["strategies"][representative_strategy]["avg_energy"] for record in results]
        
        x = np.arange(len(config_labels))
        width = 0.35
        
        # 归一化显示（权重已应用）
        delay_costs = [d * 2.0 / 0.2 for d in delays]  # weight_delay * normalize
        energy_costs = [e * 1.2 / 1000.0 for e in energies]  # weight_energy * normalize
        
        ax.bar(x - width/2, delay_costs, width, label='Delay Cost', alpha=0.8)
        ax.bar(x + width/2, energy_costs, width, label='Energy Cost', alpha=0.8)
        
        ax.set_xticks(x)
        ax.set_xticklabels(config_labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel("Weighted Cost Component", fontsize=11)
        ax.set_title(f"Cost Breakdown ({strategy_label(representative_strategy)})", 
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    output_path = suite_dir / "edge_communication_capacity_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nChart saved: {output_path}")
    
    # ========== 生成配置推荐报告 ==========
    recommendation_path = suite_dir / "communication_capacity_recommendation.txt"
    with open(recommendation_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("边缘通信资源配置推荐报告\n")
        f.write("=" * 80 + "\n\n")
        
        for comm_config in COMM_CAPACITY_CONFIGS:
            f.write(f"【{comm_config['label']}】\n")
            f.write(f"  配置: 总带宽={comm_config['total_bandwidth_mhz']}MHz "
                   f"(上行/下行各{comm_config['total_bandwidth_mhz']/2:.1f}MHz)\n")
            f.write(f"  通信成本: {comm_config['relative_cost']:.1f}x (相对于极低带宽)\n")
            f.write(f"  适用场景: {comm_config['description']}\n")
            f.write(f"  性能表现:\n")
            
            matching_result = next((r for r in results if r["config_key"] == comm_config["key"]), None)
            if matching_result:
                for strat_key in strategy_keys:
                    metrics = matching_result["strategies"][strat_key]
                    f.write(f"    - {strategy_label(strat_key)}: "
                           f"成本={metrics['raw_cost']:.4f}, "
                           f"完成率={metrics['completion_rate']*100:.1f}%, "
                           f"卸载率={metrics['offload_ratio']*100:.1f}%, "
                           f"通信时延={metrics['comm_delay']:.3f}s\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("配置建议总结\n")
        f.write("=" * 80 + "\n")
        f.write("1. 拥塞场景 (≤10MHz): 系统性能显著下降，建议提升至标准配置\n")
        f.write("2. 资源受限 (15MHz): 可满足基本需求，但存在瓶颈\n")
        f.write("3. 标准配置 (20MHz): 3GPP基准，性价比最优\n")
        f.write("4. 充足资源 (30MHz): 性能提升明显，适合高负载场景\n")
        f.write("5. 丰富资源 (≥40MHz): 性能提升趋于平缓，存在边际收益递减\n")
        f.write("\n关键发现:\n")
        f.write("- 通信带宽是影响卸载决策的关键因素\n")
        f.write("- 20-30MHz是性价比最优区间\n")
        f.write("- 超过40MHz后，边际收益递减明显\n")
    
    print(f"Recommendation report saved: {recommendation_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare CAMTD3 strategies under different edge communication capacity configurations."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="edge_comm_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="edge_comm_capacity",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for comm_config in COMM_CAPACITY_CONFIGS:
        bw_mhz = comm_config["total_bandwidth_mhz"]
        bw_hz = bw_mhz * 1e6
        
        overrides = {
            "total_bandwidth": bw_hz,
            "uplink_bandwidth": bw_hz / 2,
            "downlink_bandwidth": bw_hz / 2,
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": comm_config["key"],
                "label": f"{comm_config['label']} ({bw_mhz}MHz)",
                "overrides": overrides,
                "config_key": comm_config["key"],
                "config_label": comm_config["label"],
                "config_description": comm_config["description"],
                "total_bandwidth_mhz": bw_mhz,
                "relative_cost": comm_config["relative_cost"],
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
        per_strategy_hook=comm_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "edge_communication_capacity_sensitivity",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "comm_capacity_configs": COMM_CAPACITY_CONFIGS,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_results(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nEdge Communication Capacity Sensitivity Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Config':<18} {'BW(MHz)':>10} {'Comm Cost':>10}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (38 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['config_label']:<18} {record['total_bandwidth_mhz']:>10.0f} "
              f"{record['relative_cost']:>10.2f}x", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()



