#!/usr/bin/env python3
"""
TD3 网络环境与拓扑综合对比实验
==============================

【合并说明】
本实验合并了三个原实验：
1. run_bandwidth_cost_comparison.py - 带宽敏感性
2. run_channel_quality_comparison.py - 信道质量
3. run_topology_density_comparison.py - 拓扑密度

【研究目标】
- 综合评估网络环境和拓扑配置对系统性能的影响
- 识别不同维度的瓶颈场景
- 分析基础设施部署与通信质量的协同效应
- 为实际部署提供量化指导

【实验场景】
选择6种代表性场景，涵盖三个维度：
1. 最差场景：低带宽 + 差信道 + 稀疏拓扑
2. 带宽瓶颈：低带宽 + 好信道 + 密集拓扑
3. 信道瓶颈：高带宽 + 差信道 + 稀疏拓扑
4. 拓扑瓶颈：高带宽 + 好信道 + 稀疏拓扑
5. 标准场景：中带宽 + 中信道 + 标准拓扑（基准）
6. 最优场景：高带宽 + 好信道 + 密集拓扑

【核心指标】
- 总成本、时延、能耗
- 任务完成率
- 卸载比例
- 通信效率
- 资源利用率

【论文对应】
- 网络环境敏感性分析
- 基础设施部署优化
- 通信-计算协同优化

【使用示例】
```bash
# 快速测试（10轮）
python experiments/td3_strategy_suite/run_network_topology_comparison.py \\
    --episodes 10 --suite-id network_quick

# 完整实验（500轮）
python experiments/td3_strategy_suite/run_network_topology_comparison.py \\
    --episodes 500 --seed 42
```
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

# ========== 添加项目根目录到Python路径 ==========
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

# 网络与拓扑综合场景配置
# 格式: 带宽(MHz), 信道质量, RSU数量, UAV数量, 标签, 描述
NETWORK_TOPOLOGY_SCENARIOS = [
    {
        "key": "worst",
        "bandwidth_mhz": 10,
        "channel_quality": "poor",
        "noise_power_dbm": -90,  # 差信道
        "path_loss_exponent": 4.0,
        "num_rsus": 2,
        "num_uavs": 1,
        "label": "最差场景",
        "description": "低带宽 + 差信道 + 稀疏拓扑",
    },
    {
        "key": "bandwidth_bottleneck",
        "bandwidth_mhz": 10,
        "channel_quality": "good",
        "noise_power_dbm": -110,  # 好信道
        "path_loss_exponent": 3.0,
        "num_rsus": 6,
        "num_uavs": 3,
        "label": "带宽瓶颈",
        "description": "低带宽 + 好信道 + 密集拓扑",
    },
    {
        "key": "channel_bottleneck",
        "bandwidth_mhz": 50,
        "channel_quality": "poor",
        "noise_power_dbm": -90,  # 差信道
        "path_loss_exponent": 4.0,
        "num_rsus": 2,
        "num_uavs": 1,
        "label": "信道瓶颈",
        "description": "高带宽 + 差信道 + 稀疏拓扑",
    },
    {
        "key": "topology_bottleneck",
        "bandwidth_mhz": 50,
        "channel_quality": "good",
        "noise_power_dbm": -110,  # 好信道
        "path_loss_exponent": 3.0,
        "num_rsus": 2,
        "num_uavs": 1,
        "label": "拓扑瓶颈",
        "description": "高带宽 + 好信道 + 稀疏拓扑",
    },
    {
        "key": "standard",
        "bandwidth_mhz": 30,
        "channel_quality": "medium",
        "noise_power_dbm": -100,  # 中等信道
        "path_loss_exponent": 3.5,
        "num_rsus": 4,
        "num_uavs": 2,
        "label": "标准场景",
        "description": "中带宽 + 中信道 + 标准拓扑",
    },
    {
        "key": "best",
        "bandwidth_mhz": 50,
        "channel_quality": "good",
        "noise_power_dbm": -110,  # 好信道
        "path_loss_exponent": 3.0,
        "num_rsus": 6,
        "num_uavs": 3,
        "label": "最优场景",
        "description": "高带宽 + 好信道 + 密集拓扑",
    },
]


def network_topology_metrics_hook(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """
    计算网络与拓扑综合指标
    
    【功能】
    1. 提取基础性能指标
    2. 计算通信效率
    3. 计算资源利用率
    4. 识别瓶颈类型
    """
    # ========== 基础指标 ==========
    metrics["avg_delay"] = metrics.get("avg_delay", 0.0)
    metrics["avg_energy"] = metrics.get("avg_energy", 0.0)
    metrics["completion_rate"] = metrics.get("completion_rate", 0.0)
    metrics["offload_ratio"] = tail_mean(episode_metrics.get("offload_ratio", []))
    
    # ========== 通信效率 ==========
    comm_delay = tail_mean(episode_metrics.get("comm_delay", []))
    metrics["comm_delay"] = comm_delay
    
    if comm_delay > 0:
        metrics["comm_efficiency"] = metrics["completion_rate"] / comm_delay
    else:
        metrics["comm_efficiency"] = 0.0
    
    # ========== 吞吐量估算 ==========
    avg_task_size_mb = 0.35  # 约350KB
    num_tasks_per_step = config.get("assumed_tasks_per_step", 12)
    if metrics["avg_delay"] > 0:
        metrics["avg_throughput_mbps"] = (avg_task_size_mb * num_tasks_per_step) / metrics["avg_delay"]
    else:
        metrics["avg_throughput_mbps"] = 0.0
    
    # ========== 资源利用效率 ==========
    num_nodes = config.get("num_rsus", 4) + config.get("num_uavs", 2)
    metrics["resource_efficiency"] = metrics["completion_rate"] / max(num_nodes, 1)
    
    # ========== 带宽效率 ==========
    bandwidth_mhz = config.get("bandwidth_mhz", 30.0)
    metrics["bandwidth_efficiency"] = metrics["completion_rate"] / max(bandwidth_mhz, 1.0)


def plot_results(results: List[Dict[str, object]], suite_dir: Path, strategy_keys: List[str]) -> None:
    """
    生成综合对比图表
    
    【图表清单】
    1. 总成本对比（柱状图）
    2. 完成率对比
    3. 时延vs能耗（散点图）
    4. 通信效率对比
    5. 资源利用率对比
    6. 综合雷达图
    """
    scenario_labels = [record["scenario_label"] for record in results]
    n_scenarios = len(scenario_labels)
    
    # ========== 图1: 总成本对比（分组柱状图）==========
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(n_scenarios)
    width = 0.13
    
    for i, strat_key in enumerate(strategy_keys[:6]):  # 最多6个策略
        costs = [record["strategies"][strat_key]["raw_cost"] for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, costs, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Network & Topology Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Cost", fontsize=12, fontweight='bold')
    ax.set_title("Cost Comparison Across Network & Topology Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "network_topology_cost_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图2: 完成率对比 ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat_key in enumerate(strategy_keys[:6]):
        completion_rates = [record["strategies"][strat_key]["completion_rate"] * 100 for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, completion_rates, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Network & Topology Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Task Completion Rate (%)", fontsize=12, fontweight='bold')
    ax.set_title("Completion Rate Across Network & Topology Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim([0, 105])
    plt.tight_layout()
    plt.savefig(suite_dir / "network_topology_completion_rate.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图3: 时延-能耗权衡（散点图）==========
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_keys)))
    
    for i, strat_key in enumerate(strategy_keys):
        delays = [record["strategies"][strat_key]["avg_delay"] for record in results]
        energies = [record["strategies"][strat_key]["avg_energy"] for record in results]
        ax.scatter(delays, energies, s=150, alpha=0.7, c=[colors[i]], label=strategy_label(strat_key))
        ax.plot(delays, energies, alpha=0.3, c=colors[i], linestyle='--')
    
    ax.set_xlabel("Average Delay (s)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Average Energy (J)", fontsize=12, fontweight='bold')
    ax.set_title("Delay-Energy Trade-off Across Network Scenarios", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "network_topology_delay_energy_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图4: 通信效率对比（折线图）==========
    fig, ax = plt.subplots(figsize=(12, 6))
    for strat_key in strategy_keys[:6]:
        comm_effs = [record["strategies"][strat_key]["comm_efficiency"] for record in results]
        ax.plot(x, comm_effs, marker="o", linewidth=2, markersize=8, label=strategy_label(strat_key))
    
    ax.set_xlabel("Network & Topology Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Communication Efficiency", fontsize=12, fontweight='bold')
    ax.set_title("Communication Efficiency Across Network Scenarios", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "network_topology_comm_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # ========== 图5: 资源利用率对比 ==========
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, strat_key in enumerate(strategy_keys[:6]):
        resource_effs = [record["strategies"][strat_key]["resource_efficiency"] for record in results]
        offset = width * (i - len(strategy_keys[:6])/2)
        ax.bar(x + offset, resource_effs, width, label=strategy_label(strat_key))
    
    ax.set_xlabel("Network & Topology Scenario", fontsize=12, fontweight='bold')
    ax.set_ylabel("Resource Efficiency", fontsize=12, fontweight='bold')
    ax.set_title("Resource Utilization Efficiency", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(suite_dir / "network_topology_resource_efficiency.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("\n" + "="*70)
    print("图表已保存:")
    print("="*70)
    chart_list = [
        "network_topology_cost_comparison.png",
        "network_topology_completion_rate.png",
        "network_topology_delay_energy_tradeoff.png",
        "network_topology_comm_efficiency.png",
        "network_topology_resource_efficiency.png",
    ]
    for name in chart_list:
        print(f"  - {suite_dir / name}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of network environment & topology impact.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add_common_experiment_args(
        parser,
        default_suite_prefix="network_topology",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    
    args = parser.parse_args()
    common = resolve_common_args(
        args,
        default_suite_prefix="network_topology",
        default_output_root="results/parameter_sensitivity",
        default_episodes=DEFAULT_EPISODES,
        default_seed=DEFAULT_SEED,
        allow_strategies=True,
    )
    strategy_keys = resolve_strategy_keys(common.strategies)
    
    print("="*70)
    print("TD3 网络环境与拓扑综合对比实验")
    print("="*70)
    print(f"场景数量: {len(NETWORK_TOPOLOGY_SCENARIOS)}")
    print(f"策略数量: {len(strategy_keys)}")
    print(f"每配置训练轮数: {common.episodes}")
    print(f"随机种子: {common.seed}")
    print(f"总训练次数: {len(NETWORK_TOPOLOGY_SCENARIOS)} × {len(strategy_keys)} = {len(NETWORK_TOPOLOGY_SCENARIOS) * len(strategy_keys)}")
    print("="*70)
    
    # ========== 构建配置列表 ==========
    configs: List[Dict[str, object]] = []
    for scenario in NETWORK_TOPOLOGY_SCENARIOS:
        overrides = {
            "num_vehicles": 12,
            "num_rsus": scenario["num_rsus"],
            "num_uavs": scenario["num_uavs"],
            "bandwidth": scenario["bandwidth_mhz"] * 1e6,  # Hz
            "noise_power_dbm": scenario["noise_power_dbm"],
            "path_loss_exponent": scenario["path_loss_exponent"],
            "override_topology": True,
            "assumed_tasks_per_step": 12,
        }
        configs.append(
            {
                "key": scenario["key"],
                "label": scenario["label"],
                "override_scenario": overrides,
                "scenario_label": scenario["label"],
                "description": scenario["description"],
                "bandwidth_mhz": scenario["bandwidth_mhz"],
                "channel_quality": scenario["channel_quality"],
                "num_rsus": scenario["num_rsus"],
                "num_uavs": scenario["num_uavs"],
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
        metrics_hook=network_topology_metrics_hook,
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
                "experiment": "network_topology_comparison",
                "description": "网络环境与拓扑综合对比（合并实验）",
                "timestamp": datetime.now().isoformat(),
                "scenarios": NETWORK_TOPOLOGY_SCENARIOS,
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
        for strat_key in strategy_keys[:3]:
            metrics = record["strategies"][strat_key]
            print(f"  {strategy_label(strat_key)}:")
            print(f"    - 总成本: {metrics['raw_cost']:.3f}")
            print(f"    - 完成率: {metrics['completion_rate']:.2%}")
            print(f"    - 通信效率: {metrics['comm_efficiency']:.3f}")


if __name__ == "__main__":
    main()

