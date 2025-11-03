#!/usr/bin/env python3
"""
CAMTD3 多目标权重Pareto前沿分析
================================

研究目标
--------
- 系统性分析时延-能耗权重组合的性能空间
- 绘制Pareto前沿曲线，展示优化目标的权衡关系
- 为不同应用场景推荐最优权重配置
- 验证算法在多目标优化中的有效性

实验设计
--------
权重组合策略：
1. 时延优先 (Latency-Priority): ω_T=3.0, ω_E=0.5
2. 偏时延 (Latency-Biased): ω_T=2.5, ω_E=1.0
3. 均衡 (Balanced): ω_T=2.0, ω_E=1.2（默认）
4. 偏能耗 (Energy-Biased): ω_T=1.5, ω_E=2.0
5. 能耗优先 (Energy-Priority): ω_T=0.5, ω_E=3.0
6. 极端时延 (Extreme-Latency): ω_T=4.0, ω_E=0.2
7. 极端能耗 (Extreme-Energy): ω_T=0.2, ω_E=4.0

学术价值
--------
- 支撑论文中关于"多目标优化框架"的理论分析
- 展示算法的灵活性和可配置性
- 为实际部署提供权重选择指导
- 与多目标优化算法（NSGA-II等）的对比基准
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

# 权重配置
WEIGHT_CONFIGS = [
    {
        "key": "extreme_latency",
        "label": "Extreme-Latency",
        "weight_delay": 4.0,
        "weight_energy": 0.2,
        "scenario": "超低时延（自动驾驶）",
    },
    {
        "key": "latency_priority",
        "label": "Latency-Priority",
        "weight_delay": 3.0,
        "weight_energy": 0.5,
        "scenario": "时延敏感（远程医疗）",
    },
    {
        "key": "latency_biased",
        "label": "Latency-Biased",
        "weight_delay": 2.5,
        "weight_energy": 1.0,
        "scenario": "偏时延（在线游戏）",
    },
    {
        "key": "balanced",
        "label": "Balanced",
        "weight_delay": 2.0,
        "weight_energy": 1.2,
        "scenario": "均衡（标准配置）",
    },
    {
        "key": "energy_biased",
        "label": "Energy-Biased",
        "weight_delay": 1.5,
        "weight_energy": 2.0,
        "scenario": "偏能耗（物联网）",
    },
    {
        "key": "energy_priority",
        "label": "Energy-Priority",
        "weight_delay": 0.5,
        "weight_energy": 3.0,
        "scenario": "能耗优先（电池设备）",
    },
    {
        "key": "extreme_energy",
        "label": "Extreme-Energy",
        "weight_delay": 0.2,
        "weight_energy": 4.0,
        "scenario": "极限节能（传感器）",
    },
]


def pareto_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """提取时延和能耗的原始值（用于Pareto分析）"""
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    
    # 保存权重信息
    metrics["weight_delay"] = config.get("weight_delay", 2.0)
    metrics["weight_energy"] = config.get("weight_energy", 1.2)


def plot_pareto_frontier(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """绘制Pareto前沿图和多维分析图"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # ========== 主图: Pareto前沿 (占据左上大空间) ==========
    ax_pareto = plt.subplot2grid((3, 3), (0, 0), rowspan=2, colspan=2)
    
    for strat_key in strategy_keys:
        delays = [record["strategies"][strat_key]["avg_delay"] for record in results]
        energies = [record["strategies"][strat_key]["avg_energy"] for record in results]
        labels = [record["weight_label_short"] for record in results]
        
        # 绘制曲线
        ax_pareto.plot(delays, energies, marker="o", linewidth=2, 
                      markersize=8, label=strategy_label(strat_key), alpha=0.7)
        
        # 标注权重点
        for i, (d, e, lbl) in enumerate(zip(delays, energies, labels)):
            if i % 2 == 0:  # 每隔一个标注，避免拥挤
                ax_pareto.annotate(lbl, (d, e), fontsize=7, alpha=0.6,
                                 xytext=(3, 3), textcoords='offset points')
    
    ax_pareto.set_xlabel("Average Delay (s)", fontsize=12)
    ax_pareto.set_ylabel("Average Energy (J)", fontsize=12)
    ax_pareto.set_title("Pareto Frontier: Delay-Energy Trade-off", fontsize=14, fontweight='bold')
    ax_pareto.grid(alpha=0.3)
    ax_pareto.legend(fontsize=9)
    
    # ========== 子图1: 加权成本 vs 权重比例 ==========
    ax1 = plt.subplot2grid((3, 3), (0, 2))
    weight_ratios = [r["weight_delay"] / (r["weight_delay"] + r["weight_energy"]) for r in results]
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        ax1.plot(weight_ratios, costs, marker="s", linewidth=2, label=strategy_label(strat_key))
    ax1.set_xlabel("Delay Weight Ratio")
    ax1.set_ylabel("Weighted Cost")
    ax1.set_title("Cost vs. Weight Ratio")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)
    
    # ========== 子图2: 完成率 vs 权重配置 ==========
    ax2 = plt.subplot2grid((3, 3), (1, 2))
    x_pos = range(len(results))
    for strat_key in strategy_keys:
        completion_rates = [record["strategies"][strat_key]["completion_rate"] * 100 
                          for record in results]
        ax2.plot(x_pos, completion_rates, marker="^", linewidth=2, label=strategy_label(strat_key))
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([r["weight_label_short"] for r in results], rotation=45, fontsize=7)
    ax2.set_ylabel("Completion Rate (%)")
    ax2.set_title("Completion Rate Stability")
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=7)
    
    # ========== 子图3: 时延分布（箱线图）==========
    ax3 = plt.subplot2grid((3, 3), (2, 0))
    for strat_key in strategy_keys:
        delays = [record["strategies"][strat_key]["avg_delay"] for record in results]
        ax3.plot(range(len(delays)), delays, marker="o", linewidth=2, label=strategy_label(strat_key))
    ax3.set_xlabel("Weight Config Index")
    ax3.set_ylabel("Avg Delay (s)")
    ax3.set_title("Delay Across Weight Configs")
    ax3.grid(alpha=0.3)
    ax3.legend(fontsize=7)
    
    # ========== 子图4: 能耗分布 ==========
    ax4 = plt.subplot2grid((3, 3), (2, 1))
    for strat_key in strategy_keys:
        energies = [record["strategies"][strat_key]["avg_energy"] for record in results]
        ax4.plot(range(len(energies)), energies, marker="D", linewidth=2, label=strategy_label(strat_key))
    ax4.set_xlabel("Weight Config Index")
    ax4.set_ylabel("Avg Energy (J)")
    ax4.set_title("Energy Across Weight Configs")
    ax4.grid(alpha=0.3)
    ax4.legend(fontsize=7)
    
    # ========== 子图5: 归一化对比（雷达图思路的线性版）==========
    ax5 = plt.subplot2grid((3, 3), (2, 2))
    # 计算每个策略的归一化性能（成本）
    all_costs = []
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        all_costs.extend(costs)
    cost_min, cost_max = min(all_costs), max(all_costs)
    
    for strat_key in strategy_keys:
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        normalized = [(c - cost_min) / (cost_max - cost_min + 1e-6) for c in costs]
        ax5.plot(range(len(normalized)), normalized, marker="*", linewidth=2, 
                label=strategy_label(strat_key))
    ax5.set_xlabel("Weight Config Index")
    ax5.set_ylabel("Normalized Cost (0-1)")
    ax5.set_title("Normalized Performance")
    ax5.grid(alpha=0.3)
    ax5.legend(fontsize=7)
    
    plt.tight_layout()
    output_path = suite_dir / "pareto_weight_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\nPareto frontier chart saved: {output_path}")
    
    # ========== 额外: 生成推荐表格 ==========
    recommendations_path = suite_dir / "weight_recommendations.txt"
    with open(recommendations_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("多目标权重配置推荐指南\n")
        f.write("=" * 80 + "\n\n")
        
        for weight_config in WEIGHT_CONFIGS:
            f.write(f"【{weight_config['label']}】\n")
            f.write(f"  权重配置: ω_T={weight_config['weight_delay']:.1f}, "
                   f"ω_E={weight_config['weight_energy']:.1f}\n")
            f.write(f"  适用场景: {weight_config['scenario']}\n")
            f.write(f"  性能表现:\n")
            
            # 找到对应的结果
            matching_result = next((r for r in results if r["weight_config_key"] == weight_config["key"]), None)
            if matching_result:
                for strat_key in strategy_keys:
                    metrics = matching_result["strategies"][strat_key]
                    f.write(f"    - {strategy_label(strat_key)}: "
                           f"成本={metrics['raw_cost']:.4f}, "
                           f"时延={metrics['avg_delay']:.3f}s, "
                           f"能耗={metrics['avg_energy']:.2f}J\n")
            f.write("\n")
    
    print(f"Weight recommendations saved: {recommendations_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pareto frontier analysis for delay-energy weight trade-offs."
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="pareto_weight",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )

    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="pareto_weight",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)

    # 构建配置列表
    configs: List[Dict[str, object]] = []
    for weight_config in WEIGHT_CONFIGS:
        overrides = {
            "weight_delay": weight_config["weight_delay"],
            "weight_energy": weight_config["weight_energy"],
            "override_topology": True,
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
        }
        configs.append(
            {
                "key": weight_config["key"],
                "label": f"{weight_config['label']} (ω_T={weight_config['weight_delay']}, ω_E={weight_config['weight_energy']})",
                "overrides": overrides,
                "weight_config_key": weight_config["key"],
                "weight_label_short": f"{weight_config['weight_delay']:.1f}/{weight_config['weight_energy']:.1f}",
                "weight_delay": weight_config["weight_delay"],
                "weight_energy": weight_config["weight_energy"],
                "scenario": weight_config["scenario"],
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
        per_strategy_hook=pareto_metrics_hook,
    )

    # 保存结果
    summary = {
        "experiment_type": "pareto_weight_analysis",
        "suite_id": common.suite_id,
        "created_at": datetime.now().isoformat(),
        "num_configs": len(results),
        "episodes_per_config": common.episodes,
        "seed": common.seed,
        "strategy_keys": strategy_keys,
        "weight_configs": WEIGHT_CONFIGS,
        "results": results,
    }
    summary_path = suite_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_pareto_frontier(results, suite_dir, strategy_keys)

    # 打印结果表格
    print("\nPareto Weight Analysis Completed")
    print(f"Suite ID             : {common.suite_id}")
    print(f"Strategies           : {format_strategy_list(common.strategies)}")
    print(f"Configurations tested: {len(results)}")
    print(f"\n{'Weight Config':<20} {'ω_T':>6} {'ω_E':>6}", end="")
    for strat_key in strategy_keys:
        print(f"{strategy_label(strat_key):>18}", end="")
    print()
    print("-" * (32 + 18 * len(strategy_keys)))
    for record in results:
        print(f"{record['weight_config_key']:<20} {record['weight_delay']:>6.1f} {record['weight_energy']:>6.1f}", end="")
        for strat_key in strategy_keys:
            print(f"{record['strategies'][strat_key]['raw_cost']:<18.4f}", end="")
        print()
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()

