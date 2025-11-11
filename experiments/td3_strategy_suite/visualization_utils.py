#!/usr/bin/env python3
"""
TD3策略对比实验 - 通用可视化工具
======================================

【功能】
为所有对比实验提供统一的离散折线图生成函数

【使用示例】
```python
from experiments.td3_strategy_suite.visualization_utils import add_line_charts

# 在plot_results函数中调用
add_line_charts(
    results=results,
    suite_dir=suite_dir,
    strategy_keys=strategy_keys,
    x_label="Vehicle Count",
    file_prefix="vehicle_count"
)
```
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from experiments.td3_strategy_suite.strategy_runner import strategy_label


def add_line_charts(
    results: List[Dict],
    suite_dir: Path,
    strategy_keys: List[str],
    x_label: str,
    file_prefix: str,
    include_multiline: bool = True,
) -> List[str]:
    """
    为对比实验生成离散折线图
    
    【功能】
    自动生成4-5张离散折线对比图：
    1. 时延折线图
    2. 能耗折线图
    3. 成本折线图
    4. 完成率折线图
    5. 多指标综合图（可选）
    
    【参数】
    - results: List[Dict] - 实验结果列表
    - suite_dir: Path - 图表保存目录
    - strategy_keys: List[str] - 策略键列表
    - x_label: str - X轴标签（如 "Vehicle Count", "Cache Capacity (MB)"）
    - file_prefix: str - 文件名前缀（如 "vehicle", "cache_capacity"）
    - include_multiline: bool - 是否生成多指标综合图（默认True）
    
    【返回值】
    - List[str] - 生成的图表文件名列表
    
    【使用示例】
    >>> charts = add_line_charts(
    ...     results=results,
    ...     suite_dir=Path("results/vehicle_comparison"),
    ...     strategy_keys=["comprehensive-no-migration", "local-only"],
    ...     x_label="Number of Vehicles",
    ...     file_prefix="vehicle"
    ... )
    >>> print(f"生成了 {len(charts)} 张图表")
    """
    
    scenario_labels = [record["scenario_label"] for record in results]
    n_scenarios = len(scenario_labels)
    x = np.arange(n_scenarios)
    
    # 颜色方案
    colors = plt.cm.tab10(np.linspace(0, 1, len(strategy_keys)))
    
    generated_charts = []
    
    # ========== 图1: 时延折线对比 ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, strat_key in enumerate(strategy_keys):
        delays = [record["strategies"][strat_key]["avg_delay"] for record in results]
        ax.plot(x, delays, marker='o', linewidth=2.5, markersize=8, 
                label=strategy_label(strat_key), color=colors[i], alpha=0.8)
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Delay (s)", fontsize=13, fontweight='bold')
    ax.set_title(f"Average Delay - {x_label} Comparison", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    chart_name = f"{file_prefix}_delay_line.png"
    plt.savefig(suite_dir / chart_name, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(chart_name)
    
    # ========== 图2: 能耗折线对比 ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, strat_key in enumerate(strategy_keys):
        energies = [record["strategies"][strat_key]["avg_energy"] for record in results]
        ax.plot(x, energies, marker='s', linewidth=2.5, markersize=8, 
                label=strategy_label(strat_key), color=colors[i], alpha=0.8)
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel("Average Energy (J)", fontsize=13, fontweight='bold')
    ax.set_title(f"Average Energy Consumption - {x_label} Comparison", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    chart_name = f"{file_prefix}_energy_line.png"
    plt.savefig(suite_dir / chart_name, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(chart_name)
    
    # ========== 图3: 成本折线对比 ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, strat_key in enumerate(strategy_keys):
        # 兼容total_cost和raw_cost
        costs = [
            record["strategies"][strat_key].get(
                "total_cost", 
                record["strategies"][strat_key].get("raw_cost", 0)
            ) 
            for record in results
        ]
        ax.plot(x, costs, marker='^', linewidth=2.5, markersize=8, 
                label=strategy_label(strat_key), color=colors[i], alpha=0.8)
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel("Total Cost", fontsize=13, fontweight='bold')
    ax.set_title(f"Total Cost - {x_label} Comparison", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    plt.tight_layout()
    chart_name = f"{file_prefix}_cost_line.png"
    plt.savefig(suite_dir / chart_name, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(chart_name)
    
    # ========== 图4: 任务完成率折线对比 ==========
    fig, ax = plt.subplots(figsize=(12, 7))
    for i, strat_key in enumerate(strategy_keys):
        completion_rates = [
            record["strategies"][strat_key]["completion_rate"] * 100 
            for record in results
        ]
        ax.plot(x, completion_rates, marker='D', linewidth=2.5, markersize=8, 
                label=strategy_label(strat_key), color=colors[i], alpha=0.8)
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel("Task Completion Rate (%)", fontsize=13, fontweight='bold')
    ax.set_title(f"Task Completion Rate - {x_label} Comparison", fontsize=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
    ax.legend(fontsize=10, loc='best', framealpha=0.9)
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_ylim([0, 105])  # 完成率范围0-100%
    plt.tight_layout()
    chart_name = f"{file_prefix}_completion_line.png"
    plt.savefig(suite_dir / chart_name, dpi=300, bbox_inches="tight")
    plt.close()
    generated_charts.append(chart_name)
    
    # ========== 图5: 多指标综合折线对比（可选）==========
    if include_multiline and strategy_keys:
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # 选择代表性策略（通常是第一个，即TD3）
        representative_strategy = strategy_keys[0]
        
        # 提取各指标
        delays = [record["strategies"][representative_strategy]["avg_delay"] for record in results]
        energies = [record["strategies"][representative_strategy]["avg_energy"] for record in results]
        costs = [
            record["strategies"][representative_strategy].get(
                "total_cost",
                record["strategies"][representative_strategy].get("raw_cost", 0)
            )
            for record in results
        ]
        completion = [record["strategies"][representative_strategy]["completion_rate"] for record in results]
        offload = [record["strategies"][representative_strategy]["offload_ratio"] for record in results]
        
        # 归一化函数
        def normalize(data):
            min_val, max_val = min(data), max(data)
            if max_val - min_val < 1e-6:
                return [0.5] * len(data)
            return [(v - min_val) / (max_val - min_val) for v in data]
        
        norm_delay = normalize(delays)
        norm_energy = normalize(energies)
        norm_cost = normalize(costs)
        
        # 绘制多条折线
        ax.plot(x, norm_delay, marker='o', linewidth=2.5, markersize=8, 
                label='Delay (norm)', alpha=0.8)
        ax.plot(x, norm_energy, marker='s', linewidth=2.5, markersize=8, 
                label='Energy (norm)', alpha=0.8)
        ax.plot(x, norm_cost, marker='^', linewidth=2.5, markersize=8, 
                label='Cost (norm)', alpha=0.8)
        ax.plot(x, completion, marker='D', linewidth=2.5, markersize=8, 
                label='Completion Rate', alpha=0.8)
        ax.plot(x, offload, marker='v', linewidth=2.5, markersize=8, 
                label='Offload Ratio', alpha=0.8)
        
        ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
        ax.set_ylabel("Normalized Value / Rate", fontsize=13, fontweight='bold')
        ax.set_title(
            f"Multi-metric Performance: {strategy_label(representative_strategy)} - {x_label}", 
            fontsize=15, fontweight='bold'
        )
        ax.set_xticks(x)
        ax.set_xticklabels(scenario_labels, rotation=20, ha='right', fontsize=10)
        ax.legend(fontsize=11, loc='best', framealpha=0.9, ncol=2)
        ax.grid(alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_ylim([-0.05, 1.1])
        plt.tight_layout()
        chart_name = f"{file_prefix}_multiline.png"
        plt.savefig(suite_dir / chart_name, dpi=300, bbox_inches="tight")
        plt.close()
        generated_charts.append(chart_name)
    
    return generated_charts


def print_chart_summary(
    original_charts: List[str],
    line_charts: List[str],
    suite_dir: Path,
) -> None:
    """
    打印图表生成摘要
    
    【参数】
    - original_charts: List[str] - 原有图表文件名列表
    - line_charts: List[str] - 新增折线图文件名列表
    - suite_dir: Path - 图表保存目录
    """
    print("\n" + "="*70)
    print("图表已保存:")
    print("="*70)
    
    # 原有图表
    for chart in original_charts:
        print(f"  - {suite_dir / chart}")
    
    # 新增折线图
    if line_charts:
        print(f"\n  【新增离散折线图 ({len(line_charts)}张)】:")
        for chart in line_charts:
            desc = ""
            if "delay_line" in chart:
                desc = " (时延折线对比)"
            elif "energy_line" in chart:
                desc = " (能耗折线对比)"
            elif "cost_line" in chart:
                desc = " (成本折线对比)"
            elif "completion_line" in chart:
                desc = " (完成率折线对比)"
            elif "multiline" in chart:
                desc = " (多指标综合对比)"
            print(f"  - {suite_dir / chart}{desc}")


