#!/usr/bin/env python3
"""
CAMTD3 边缘基础设施综合对比实验
==============================

【合并说明】
本实验合并了两个原实验：
1. run_edge_compute_capacity_comparison.py - 边缘计算能力对比
2. run_edge_communication_capacity_comparison.py - 边缘通信资源对比

【研究目标】
- 综合评估边缘计算和通信资源配置对系统性能的影响
- 识别计算瓶颈vs通信瓶颈场景
- 分析资源均衡配置的重要性
- 为基础设施投资决策提供量化依据

【实验场景】
选择5种代表性场景：
1. 计算+通信均低（最差场景）
2. 高计算+低通信（通信瓶颈）
3. 低计算+高通信（计算瓶颈）
4. 计算+通信均衡（标准场景，基准）
5. 计算+通信均高（最优场景）

【核心指标】
- 总成本、时延、能耗
- 任务完成率
- 卸载比例
- 计算效率、通信效率
- 瓶颈识别

【论文对应】
- 边缘基础设施配置优化
- 资源瓶颈分析
- 成本-效益权衡

【使用示例】
```bash
# 快速测试（10轮）
python experiments/camtd3_strategy_suite/run_edge_infrastructure_comparison.py \\
    --episodes 10 --suite-id edge_infra_quick

# 完整实验（500轮）
python experiments/camtd3_strategy_suite/run_edge_infrastructure_comparison.py--episodes 500 --seed 42
```
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

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

# 边缘基础设施配置场景
# 格式: (rsu_compute_ghz, uav_compute_ghz, bandwidth_mhz, label, description)
INFRASTRUCTURE_SCENARIOS = [
    {
        "key": "low_both",
        "rsu_compute_ghz": 10.0,
        "uav_compute_ghz": 8.0,
        "bandwidth_mhz": 15.0,
        "label": "低计算+低通信",
        "description": "资源受限场景（最差）",
        "compute_level": 0.6,
        "comm_level": 0.6,
    },
    {
        "key": "high_compute_low_comm",
        "rsu_compute_ghz": 20.0,
        "uav_compute_ghz": 16.0,
        "bandwidth_mhz": 15.0,
        "label": "高计算+低通信",
        "description": "通信瓶颈场景",
        "compute_level": 1.4,
        "comm_level": 0.6,
    },
    {
        "key": "low_compute_high_comm",
        "rsu_compute_ghz": 10.0,
        "uav_compute_ghz": 8.0,
        "bandwidth_mhz": 40.0,
        "label": "低计算+高通信",
        "description": "计算瓶颈场景",
        "compute_level": 0.6,
        "comm_level": 1.4,
    },
    {
        "key": "balanced",
        "rsu_compute_ghz": 15.0,
        "uav_compute_ghz": 12.0,
        "bandwidth_mhz": 20.0,
        "label": "均衡配置",
        "description": "标准场景（基准）",
        "compute_level": 1.0,
        "comm_level": 1.0,
    },
    {
        "key": "high_both",
        "rsu_compute_ghz": 20.0,
        "uav_compute_ghz": 16.0,
        "bandwidth_mhz": 40.0,
        "label": "高计算+高通信",
        "description": "资源充足场景（最优）",
        "compute_level": 1.4,
        "comm_level": 1.4,
    },
]


def infrastructure_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """
    计算边缘基础设施综合指标
    
    【功能】
    1. 提取基础性能指标
    2. 计算资源利用效率
    3. 识别瓶颈类型
    """
    # ========== 基础指标 ==========
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # ========== 通信指标 ==========
    metrics["comm_delay"] = tail_mean(episode_metrics.get("comm_delay", []))
    metrics["transmission_energy"] = tail_mean(episode_metrics.get("transmission_energy", []))
    
    # ========== 计算效率 ==========
    raw_cost = metrics.get("raw_cost", 1.0)
    if raw_cost > 0:
        metrics["cost_efficiency"] = metrics["completion_rate"] / raw_cost
    else:
        metrics["cost_efficiency"] = 0.0
    
    # ========== 通信效率 ==========
    if metrics["comm_delay"] > 0:
        metrics["comm_efficiency"] = metrics["completion_rate"] / metrics["comm_delay"]
    else:
        metrics["comm_efficiency"] = 0.0
    
    # ========== 带宽效率 ==========
    bandwidth_mhz = config.get("bandwidth_mhz", 20.0)
    metrics["bandwidth_efficiency"] = metrics["completion_rate"] / max(bandwidth_mhz, 1.0)
    
    # ========== 瓶颈识别（简单启发式）==========
    compute_level = config.get("compute_level", 1.0)
    comm_level = config.get("comm_level", 1.0)
    
    if compute_level < comm_level:
        metrics["bottleneck_type"] = "compute"  # 计算瓶颈
    elif comm_level < compute_level:
        metrics["bottleneck_type"] = "communication"  # 通信瓶颈
    else:
        metrics["bottleneck_type"] = "balanced"  # 均衡


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """
    生成综合对比图表
    
    【图表清单】
    1. 总成本对比（柱状图）
    2. 完成率对比
    3. 时延vs能耗散点图
    4. 卸载比例对比
    5. 资源效率对比
    6. 瓶颈识别热力图
    """
    scenario_labels = [record["scenario_label"] for record in results]
    n_scenarios = len(scenario_labels)
    
    # ========== 图1: 总成本对比（分组柱状图）==========
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(n_scenarios)
    width = 0.15
    
    for i, strat_key in enumerate(strategy_keys[:6]):  # 最多显示6个策略
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, costs, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Infrastructure Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Cost", fontsize=12, fontweight='bold')
    ax.set_title("Total Cost Comparison Across Edge Infrastructure Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "edge_infra_cost_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图2: 完成率对比 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat_key in enumerate(strategy_keys[:6]):
        completion_rates = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, completion_rates, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Infrastructure Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Task Completion Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Completion Rate Across Edge Infrastructure Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(suite_dir / "edge_infra_completion_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图3: 时延vs能耗（散点图）==========
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_keys)))
    
    for i, strat_key in enumerate(strategy_keys):
        delays = [record["strategies"][strat_key]["avg_delay"] for record in results]
        energies = [record["strategies"][strat_key]["avg_energy"] for record in results]
        ax.scatter(delays, energies, s=150, alpha=0.7, c=[colors[i]], label=strategy_label(strat_key))
        
        # 连线显示场景变化
        ax.plot(delays, energies, alpha=0.3, c=colors[i], linestyle='--')
    
    ax.set_xlabel("Average Delay (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Energy (J)", fontsize=12, fontweight='bold')
    ax.set_title("Delay-Energy Trade-off Across Infrastructure Scenarios", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "edge_infra_delay_energy_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图4: 卸载比例对比 ==========
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, strat_key in enumerate(strategy_keys[:6]):
        offload_ratios = [record["strategies"][strat_key]["offload_ratio"] * 100 for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, offload_ratios, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Infrastructure Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Offload Ratio (%)", fontsize=12, fontweight='bold')
    ax.set_title("Offloading Behavior Across Edge Infrastructure Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=15, ha='right')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "edge_infra_offload_ratio.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图5: 资源效率对比（雷达图）==========
    # 选择一个代表性策略（如CAMTD3）
    representative_strategy = strategy_keys[0] if strategy_keys else "comprehensive-migration"
    
    categories = scenario_labels
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 归一化指标
    cost_efficiencies = [record["strategies"][representative_strategy]["cost_efficiency"] for record in results]
    comm_efficiencies = [record["strategies"][representative_strategy]["comm_efficiency"] for record in results]
    completion_rates = [record["strategies"][representative_strategy]["completion_rate"] for record in results]
    
    max_cost_eff = max(cost_efficiencies) if max(cost_efficiencies) > 0 else 1.0
    max_comm_eff = max(comm_efficiencies) if max(comm_efficiencies) > 0 else 1.0
    
    norm_cost_eff = [x / max_cost_eff for x in cost_efficiencies]
    norm_comm_eff = [x / max_comm_eff for x in comm_efficiencies]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    norm_cost_eff += norm_cost_eff[:1]
    norm_comm_eff += norm_comm_eff[:1]
    completion_rates += completion_rates[:1]
    angles += angles[:1]
    
    ax.plot(angles, norm_cost_eff, 'o-', linewidth=2, label='Cost Efficiency (norm)')
    ax.fill(angles, norm_cost_eff, alpha=0.25)
    ax.plot(angles, norm_comm_eff, 's-', linewidth=2, label='Comm Efficiency (norm)')
    ax.fill(angles, norm_comm_eff, alpha=0.25)
    ax.plot(angles, completion_rates, '^-', linewidth=2, label='Completion Rate')
    ax.fill(angles, completion_rates, alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Resource Efficiency Profile: {strategy_label(representative_strategy)}", 
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(suite_dir / "edge_infra_efficiency_radar.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*70)
    print("图表已保存:")
    print("="*70)
    chart_list = [
        "edge_infra_cost_comparison.png",
        "edge_infra_completion_rate.png",
        "edge_infra_delay_energy_tradeoff.png",
        "edge_infra_offload_ratio.png",
        "edge_infra_efficiency_radar.png",
    ]
    for name in chart_list:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of edge infrastructure (compute + communication) impact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="edge_infrastructure",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    
    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="edge_infrastructure",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)
    
    print("="*70)
    print("CAMTD3 边缘基础设施综合对比实验")
    print("="*70)
    print(f"场景数量: {len(INFRASTRUCTURE_SCENARIOS)}")
    print(f"策略数量: {len(strategy_keys)}")
    print(f"每配置训练轮数: {common.episodes}")
    print(f"随机种子: {common.seed}")
    print(f"总训练次数: {len(INFRASTRUCTURE_SCENARIOS)} × {len(strategy_keys)} = {len(INFRASTRUCTURE_SCENARIOS) * len(strategy_keys)}")
    print("="*70)
    
    # ========== 构建配置列表 ==========
    configs: List[Dict[str, object]] = []
    for scenario in INFRASTRUCTURE_SCENARIOS:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "rsu_cpu_freq": scenario["rsu_compute_ghz"] * 1e9,  # Hz
            "uav_cpu_freq": scenario["uav_compute_ghz"] * 1e9,  # Hz
            "bandwidth": scenario["bandwidth_mhz"] * 1e6,  # Hz
            "override_topology": True,
        }
        configs.append(
            {
                "key": scenario["key"],
                "label": scenario["label"],
                "override_scenario": overrides,
                "scenario_label": scenario["label"],
                "description": scenario["description"],
                "rsu_compute_ghz": scenario["rsu_compute_ghz"],
                "uav_compute_ghz": scenario["uav_compute_ghz"],
                "bandwidth_mhz": scenario["bandwidth_mhz"],
                "compute_level": scenario["compute_level"],
                "comm_level": scenario["comm_level"],
            }
        )
    
    # ========== 运行实验 ==========
    results = evaluate_configs(
        configs=configs,
        episodes=common.episodes,
        seed=common.seed,
        suite_prefix=common.suite_prefix,
        suite_id=common.suite_id,
        output_root=common.output_root,
        silent=common.silent,
        strategies=strategy_keys,
        metrics_hook=infrastructure_metrics_hook,
    )
    
    # ========== 生成图表 ==========
    suite_dir = build_suite_path(
        output_root=common.output_root,
        prefix=common.suite_prefix,
        suite_id=common.suite_id,
    )
    plot_results(results, suite_dir, strategy_keys)
    
    # ========== 保存详细结果 ==========
    summary_path = suite_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "edge_infrastructure_comparison",
                "description": "边缘基础设施综合对比（合并实验）",
                "timestamp": datetime.now().isoformat(),
                "scenarios": INFRASTRUCTURE_SCENARIOS,
                "strategies": format_strategy_list(strategy_keys),
                "episodes": common.episodes,
                "seed": common.seed,
                "results": results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    
    print(f"\n汇总结果: {summary_path}")
    print("\n" + "="*70)
    print("实验完成!")
    print("="*70)
    
    # ========== 打印关键发现 ==========
    print("\n关键发现:")
    for record in results:
        label = record["scenario_label"]
        print(f"\n场景: {label}")
        for strat_key in strategy_keys[:3]:  # 只显示前3个策略
            metrics = record["strategies"][strat_key]
            print(f"  {strategy_label(strat_key)}:")
            print(f"    - 总成本: {metrics['raw_cost']:.3f}")
            print(f"    - 完成率: {metrics['completion_rate']:.2%}")
            print(f"    - 卸载比例: {metrics['offload_ratio']:.2%}")


if __name__ == "__main__":
    main()

