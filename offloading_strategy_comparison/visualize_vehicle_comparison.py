#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆数量对比结果可视化
生成学术论文风格的对比图表
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 学术风格设置
plt.style.use('seaborn-v0_8-whitegrid')
matplotlib.rcParams['figure.dpi'] = 100
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams['font.size'] = 11
matplotlib.rcParams['axes.labelsize'] = 12
matplotlib.rcParams['axes.titlesize'] = 14
matplotlib.rcParams['xtick.labelsize'] = 10
matplotlib.rcParams['ytick.labelsize'] = 10
matplotlib.rcParams['legend.fontsize'] = 10


def load_latest_results(results_dir: str = "results/offloading_comparison") -> Dict:
    """加载最新的车辆扫描结果"""
    results_path = Path(results_dir)
    
    # 查找最新的vehicle_sweep文件
    vehicle_files = list(results_path.glob("vehicle_sweep_*.json"))
    if not vehicle_files:
        print("未找到车辆扫描结果文件")
        return None
    
    # 选择最新的文件
    latest_file = max(vehicle_files, key=lambda p: p.stat().st_mtime)
    print(f"加载结果文件: {latest_file}")
    
    with open(latest_file, 'r') as f:
        return json.load(f)


def plot_vehicle_comparison(data: Dict, save_dir: str = "academic_figures/vehicle_comparison"):
    """生成车辆数量对比图"""
    
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    if not data or 'results' not in data:
        print("数据格式错误")
        return
    
    vehicle_counts = data['values']  # [8, 12, 16, 20, 24]
    results = data['results']
    
    # 策略名称映射（更学术的命名）
    name_mapping = {
        'LocalOnly': 'Local-Only',
        'RSUOnly': 'RSU-Only',
        'LoadBalance': 'Load Balance',
        'Random': 'Random',
        'TD3': 'TD3 (Proposed)',
        'TD3-NoMig': 'TD3 w/o Migration'
    }
    
    # 颜色方案（学术配色）
    colors = {
        'Local-Only': '#FF6B6B',     # 红色系
        'RSU-Only': '#4ECDC4',       # 青色系
        'Load Balance': '#45B7D1',   # 蓝色系
        'Random': '#95A5A6',         # 灰色系
        'TD3 (Proposed)': '#2ECC71', # 绿色系（突出显示）
        'TD3 w/o Migration': '#27AE60'  # 深绿色系
    }
    
    # 线型
    linestyles = {
        'Local-Only': '--',
        'RSU-Only': '-.',
        'Load Balance': ':',
        'Random': '--',
        'TD3 (Proposed)': '-',      # 实线（突出显示）
        'TD3 w/o Migration': '--'
    }
    
    # 标记
    markers = {
        'Local-Only': 's',
        'RSU-Only': '^',
        'Load Balance': 'd',
        'Random': 'x',
        'TD3 (Proposed)': 'o',     # 圆形（突出显示）
        'TD3 w/o Migration': 'v'
    }
    
    # 创建4子图布局
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison with Varying Number of Vehicles', fontsize=16, y=1.02)
    
    # 准备数据
    strategy_metrics = {}
    for strategy_name, strategy_results in results.items():
        display_name = name_mapping.get(strategy_name, strategy_name)
        
        costs = []
        delays = []
        energies = []
        completions = []
        
        for result in strategy_results:
            # 重新计算加权成本（确保使用TD3归一化）
            cost = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 1000.0)
            costs.append(cost)
            delays.append(result['avg_delay'])
            energies.append(result['avg_energy'])
            completions.append(result['avg_completion_rate'] * 100)
        
        strategy_metrics[display_name] = {
            'costs': costs,
            'delays': delays,
            'energies': energies,
            'completions': completions
        }
    
    # 1. 加权成本对比
    ax = axes[0, 0]
    for strategy_name, metrics in strategy_metrics.items():
        ax.plot(vehicle_counts, metrics['costs'], 
                label=strategy_name,
                color=colors.get(strategy_name, 'black'),
                linestyle=linestyles.get(strategy_name, '-'),
                marker=markers.get(strategy_name, 'o'),
                markersize=8,
                linewidth=2 if 'TD3' in strategy_name and 'w/o' not in strategy_name else 1.5)
    
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Weighted Cost')
    ax.set_title('(a) Weighted Cost (2.0×Delay + 1.2×Energy/1000)')
    ax.set_xticks(vehicle_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # 2. 平均时延对比
    ax = axes[0, 1]
    for strategy_name, metrics in strategy_metrics.items():
        ax.plot(vehicle_counts, metrics['delays'],
                label=strategy_name,
                color=colors.get(strategy_name, 'black'),
                linestyle=linestyles.get(strategy_name, '-'),
                marker=markers.get(strategy_name, 'o'),
                markersize=8,
                linewidth=2 if 'TD3' in strategy_name and 'w/o' not in strategy_name else 1.5)
    
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Average Delay (s)')
    ax.set_title('(b) Average Task Delay')
    ax.set_xticks(vehicle_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # 3. 总能耗对比
    ax = axes[1, 0]
    for strategy_name, metrics in strategy_metrics.items():
        # 能耗转换为kJ显示
        energies_kj = [e/1000 for e in metrics['energies']]
        ax.plot(vehicle_counts, energies_kj,
                label=strategy_name,
                color=colors.get(strategy_name, 'black'),
                linestyle=linestyles.get(strategy_name, '-'),
                marker=markers.get(strategy_name, 'o'),
                markersize=8,
                linewidth=2 if 'TD3' in strategy_name and 'w/o' not in strategy_name else 1.5)
    
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Total Energy (kJ)')
    ax.set_title('(c) Total Energy Consumption')
    ax.set_xticks(vehicle_counts)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    # 4. 任务完成率对比
    ax = axes[1, 1]
    for strategy_name, metrics in strategy_metrics.items():
        ax.plot(vehicle_counts, metrics['completions'],
                label=strategy_name,
                color=colors.get(strategy_name, 'black'),
                linestyle=linestyles.get(strategy_name, '-'),
                marker=markers.get(strategy_name, 'o'),
                markersize=8,
                linewidth=2 if 'TD3' in strategy_name and 'w/o' not in strategy_name else 1.5)
    
    ax.set_xlabel('Number of Vehicles')
    ax.set_ylabel('Task Completion Rate (%)')
    ax.set_title('(d) Task Completion Rate')
    ax.set_xticks(vehicle_counts)
    ax.set_ylim([0, 105])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = save_path / "vehicle_comparison_main.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    output_file_png = save_path / "vehicle_comparison_main.png"
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"已保存主要对比图: {output_file}")
    
    # 生成专门的加权成本突出图
    plt.figure(figsize=(10, 6))
    
    # 只显示加权成本，突出TD3的优势
    for strategy_name, metrics in strategy_metrics.items():
        if 'TD3' in strategy_name:
            plt.plot(vehicle_counts, metrics['costs'],
                    label=strategy_name,
                    color=colors.get(strategy_name, 'black'),
                    linestyle=linestyles.get(strategy_name, '-'),
                    marker=markers.get(strategy_name, 'o'),
                    markersize=10,
                    linewidth=2.5)
        else:
            plt.plot(vehicle_counts, metrics['costs'],
                    label=strategy_name,
                    color=colors.get(strategy_name, 'black'),
                    linestyle=linestyles.get(strategy_name, '-'),
                    marker=markers.get(strategy_name, 'o'),
                    markersize=8,
                    linewidth=1.5,
                    alpha=0.7)
    
    plt.xlabel('Number of Vehicles', fontsize=14)
    plt.ylabel('Weighted Cost', fontsize=14)
    plt.title('Weighted Cost Comparison: TD3 vs. Baselines', fontsize=16)
    plt.xticks(vehicle_counts)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=12, framealpha=0.9)
    
    # 添加改进百分比标注（TD3相对于最佳baseline）
    td3_costs = strategy_metrics.get('TD3 (Proposed)', {}).get('costs', [])
    if td3_costs:
        # 找出每个车辆数下的最佳baseline（不含TD3）
        for i, vehicles in enumerate(vehicle_counts):
            baseline_costs = []
            for name, metrics in strategy_metrics.items():
                if 'TD3' not in name:
                    baseline_costs.append(metrics['costs'][i])
            
            if baseline_costs:
                best_baseline = min(baseline_costs)
                td3_cost = td3_costs[i]
                improvement = (best_baseline - td3_cost) / best_baseline * 100
                
                # 在图上标注改进百分比
                plt.annotate(f'{improvement:.1f}%',
                           xy=(vehicles, td3_cost),
                           xytext=(vehicles, td3_cost - 0.5),
                           fontsize=9,
                           color='green',
                           ha='center')
    
    plt.tight_layout()
    
    output_file = save_path / "weighted_cost_highlight.pdf"
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    output_file_png = save_path / "weighted_cost_highlight.png"
    plt.savefig(output_file_png, bbox_inches='tight', dpi=300)
    print(f"已保存加权成本突出图: {output_file}")
    
    # 生成性能改进表格
    generate_performance_table(strategy_metrics, vehicle_counts, save_path)
    
    plt.show()


def generate_performance_table(strategy_metrics: Dict, vehicle_counts: List[int], save_path: Path):
    """生成性能对比表格"""
    
    # 创建Markdown表格
    md_content = "# Performance Comparison Table\n\n"
    md_content += "## Weighted Cost Comparison\n\n"
    md_content += "| Strategy | " + " | ".join([f"N={v}" for v in vehicle_counts]) + " | Average |\n"
    md_content += "|----------|" + "--------|" * (len(vehicle_counts) + 1) + "\n"
    
    for strategy_name in ['Local-Only', 'RSU-Only', 'Load Balance', 'Random', 'TD3 (Proposed)', 'TD3 w/o Migration']:
        if strategy_name in strategy_metrics:
            costs = strategy_metrics[strategy_name]['costs']
            avg_cost = np.mean(costs)
            row = f"| {strategy_name} | "
            row += " | ".join([f"{c:.2f}" for c in costs])
            row += f" | {avg_cost:.2f} |\n"
            md_content += row
    
    # TD3改进百分比
    if 'TD3 (Proposed)' in strategy_metrics:
        td3_costs = strategy_metrics['TD3 (Proposed)']['costs']
        
        # 计算相对于每个baseline的改进
        md_content += "\n## TD3 Improvement Over Baselines (%)\n\n"
        md_content += "| Baseline | " + " | ".join([f"N={v}" for v in vehicle_counts]) + " | Average |\n"
        md_content += "|----------|" + "--------|" * (len(vehicle_counts) + 1) + "\n"
        
        for baseline_name in ['Local-Only', 'RSU-Only', 'Load Balance', 'Random']:
            if baseline_name in strategy_metrics:
                baseline_costs = strategy_metrics[baseline_name]['costs']
                improvements = []
                for td3_cost, baseline_cost in zip(td3_costs, baseline_costs):
                    improvement = (baseline_cost - td3_cost) / baseline_cost * 100
                    improvements.append(improvement)
                
                avg_improvement = np.mean(improvements)
                row = f"| vs {baseline_name} | "
                row += " | ".join([f"{imp:.1f}%" for imp in improvements])
                row += f" | {avg_improvement:.1f}% |\n"
                md_content += row
    
    # 保存表格
    table_file = save_path / "performance_table.md"
    with open(table_file, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"已保存性能表格: {table_file}")


if __name__ == "__main__":
    # 加载并可视化结果
    data = load_latest_results()
    if data:
        plot_vehicle_comparison(data)
    else:
        print("未能加载数据，请先运行实验")
