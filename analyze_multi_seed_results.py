#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析多个随机种子的Vehicle Sweep实验结果
计算均值、标准差、置信区间
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict
import glob

def load_all_results(results_dir: Path) -> Dict[int, List[Dict]]:
    """
    加载所有实验结果，按车辆数分组
    
    Returns:
        {num_vehicles: [result1, result2, ...]}
    """
    results_by_vehicles = {}
    
    # 查找所有summary文件
    summary_files = sorted(results_dir.glob("td3_vehicle_sweep_summary_*.json"))
    
    print(f"Found {len(summary_files)} experiment files")
    
    for file in summary_files:
        print(f"  - {file.name}")
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
            for config in data:
                num_vehicles = config['num_vehicles']
                if num_vehicles not in results_by_vehicles:
                    results_by_vehicles[num_vehicles] = []
                results_by_vehicles[num_vehicles].append(config)
    
    return results_by_vehicles

def calculate_statistics(results: List[Dict], metric_name: str) -> Dict:
    """计算统计指标"""
    values = [r[metric_name] for r in results]
    
    return {
        'mean': np.mean(values),
        'std': np.std(values, ddof=1) if len(values) > 1 else 0.0,
        'min': np.min(values),
        'max': np.max(values),
        'n': len(values),
        'values': values
    }

def main():
    results_dir = Path("results/experiments/td3_vehicle_sweep")
    
    print("=" * 80)
    print("Multi-Seed Vehicle Sweep Analysis")
    print("=" * 80)
    print()
    
    # 加载所有结果
    results_by_vehicles = load_all_results(results_dir)
    
    if not results_by_vehicles:
        print("[ERROR] No results found!")
        return
    
    print("\n" + "=" * 80)
    print("Statistical Analysis")
    print("=" * 80)
    
    # 分析每个车辆配置
    vehicle_counts = sorted(results_by_vehicles.keys())
    
    # 准备绘图数据
    plot_data = {
        'vehicles': [],
        'delay_mean': [],
        'delay_std': [],
        'completion_mean': [],
        'completion_std': [],
        'reward_mean': [],
        'reward_std': []
    }
    
    for num_vehicles in vehicle_counts:
        results = results_by_vehicles[num_vehicles]
        print(f"\n{num_vehicles} vehicles ({len(results)} experiments):")
        
        # 计算时延统计
        delay_stats = calculate_statistics(results, 'avg_delay')
        print(f"  Avg Delay: {delay_stats['mean']:.4f} ± {delay_stats['std']:.4f}s")
        print(f"    Range: [{delay_stats['min']:.4f}, {delay_stats['max']:.4f}]")
        
        # 计算完成率统计
        completion_stats = calculate_statistics(results, 'avg_completion')
        print(f"  Completion: {completion_stats['mean']*100:.2f} ± {completion_stats['std']*100:.2f}%")
        
        # 计算奖励统计
        reward_stats = calculate_statistics(results, 'avg_step_reward')
        print(f"  Avg Reward: {reward_stats['mean']:.4f} ± {reward_stats['std']:.4f}")
        
        # 检查异常
        if delay_stats['std'] > 0.02 and len(results) > 1:
            print(f"  [WARNING] High delay variance across seeds!")
            print(f"    Individual values: {[f'{v:.4f}' for v in delay_stats['values']]}")
        
        # 保存绘图数据
        plot_data['vehicles'].append(num_vehicles)
        plot_data['delay_mean'].append(delay_stats['mean'])
        plot_data['delay_std'].append(delay_stats['std'])
        plot_data['completion_mean'].append(completion_stats['mean'] * 100)
        plot_data['completion_std'].append(completion_stats['std'] * 100)
        plot_data['reward_mean'].append(reward_stats['mean'])
        plot_data['reward_std'].append(reward_stats['std'])
    
    # 生成综合图表
    print("\n" + "=" * 80)
    print("Generating charts...")
    print("=" * 80)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 时延图（带误差棒）
    ax = axes[0]
    ax.errorbar(plot_data['vehicles'], plot_data['delay_mean'], 
                yerr=plot_data['delay_std'], fmt='o-', capsize=5, 
                linewidth=2.5, markersize=8, color='#D55E00')
    ax.set_xlabel('Number of Vehicles', fontsize=12)
    ax.set_ylabel('Average Delay (s)', fontsize=12)
    ax.set_title('(a) Average Delay vs Vehicle Count\n(Mean ± Std Dev)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 标注均值
    for i, (x, y, std) in enumerate(zip(plot_data['vehicles'], 
                                         plot_data['delay_mean'], 
                                         plot_data['delay_std'])):
        label = f'{y:.4f}'
        if std > 0.001:
            label += f'\n±{std:.4f}'
        ax.text(x, y, label, ha='center', va='bottom', fontsize=9)
    
    # 完成率图
    ax = axes[1]
    ax.errorbar(plot_data['vehicles'], plot_data['completion_mean'],
                yerr=plot_data['completion_std'], fmt='o-', capsize=5,
                linewidth=2.5, markersize=8, color='#029E73')
    ax.set_xlabel('Number of Vehicles', fontsize=12)
    ax.set_ylabel('Task Completion Rate (%)', fontsize=12)
    ax.set_title('(b) Completion Rate vs Vehicle Count\n(Mean ± Std Dev)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_ylim([95, 100])
    
    # 奖励图
    ax = axes[2]
    ax.errorbar(plot_data['vehicles'], plot_data['reward_mean'],
                yerr=plot_data['reward_std'], fmt='o-', capsize=5,
                linewidth=2.5, markersize=8, color='#0173B2')
    ax.set_xlabel('Number of Vehicles', fontsize=12)
    ax.set_ylabel('Average Step Reward', fontsize=12)
    ax.set_title('(c) Reward vs Vehicle Count\n(Mean ± Std Dev)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.suptitle('TD3 Vehicle Sweep - Multi-Seed Analysis', 
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    output_path = results_dir / "multi_seed_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Chart saved: {output_path}")
    
    # 生成Markdown报告
    md_path = results_dir / "multi_seed_analysis_report.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# TD3 Vehicle Sweep - Multi-Seed Analysis Report\n\n")
        f.write("## Statistical Summary\n\n")
        f.write("| Vehicles | Avg Delay (s) | Completion (%) | Avg Reward | N |\n")
        f.write("|----------|---------------|----------------|------------|---|\n")
        
        for i, nv in enumerate(plot_data['vehicles']):
            f.write(f"| {nv} | {plot_data['delay_mean'][i]:.4f} ± {plot_data['delay_std'][i]:.4f} | ")
            f.write(f"{plot_data['completion_mean'][i]:.2f} ± {plot_data['completion_std'][i]:.2f} | ")
            f.write(f"{plot_data['reward_mean'][i]:.4f} ± {plot_data['reward_std'][i]:.4f} | ")
            f.write(f"{len(results_by_vehicles[nv])} |\n")
        
        f.write("\n## Key Findings\n\n")
        
        # 检查时延趋势
        delays = plot_data['delay_mean']
        if all(delays[i] <= delays[i+1] + 0.01 for i in range(len(delays)-1)):
            f.write("- ✅ **Delay trend is monotonically increasing or stable** (expected behavior)\n")
        else:
            f.write("- ⚠️ **Delay trend shows non-monotonic behavior** (needs investigation)\n")
            
            # 找出异常点
            for i in range(1, len(delays)):
                if delays[i] < delays[i-1] - 0.02:
                    f.write(f"  - {plot_data['vehicles'][i]} vehicles: "
                           f"Delay decreased from {delays[i-1]:.4f}s to {delays[i]:.4f}s\n")
        
        # 检查方差
        high_var_configs = [nv for i, nv in enumerate(plot_data['vehicles']) 
                           if plot_data['delay_std'][i] > 0.02]
        if high_var_configs:
            f.write(f"\n- ⚠️ **High variance (>0.02s) detected** for: {high_var_configs}\n")
            f.write("  - Suggests strong seed dependency or insufficient training\n")
        else:
            f.write("\n- ✅ **Low variance across seeds** indicates consistent performance\n")
        
        # 完成率分析
        avg_completion = np.mean(plot_data['completion_mean'])
        f.write(f"\n- Average completion rate across all configurations: **{avg_completion:.2f}%**\n")
        
        if avg_completion > 97.0:
            f.write("  - ✅ System maintains high task completion (>97%)\n")
        
        f.write("\n## Recommendations\n\n")
        
        if len(results_by_vehicles[vehicle_counts[0]]) < 3:
            f.write("- ⚠️ **Run more seeds**: Current analysis based on <3 seeds. ")
            f.write("Recommend at least 3 seeds for statistical significance.\n")
        
        if any(plot_data['delay_std'][i] > 0.02 for i in range(len(plot_data['vehicles']))):
            f.write("- ⚠️ **Increase training episodes**: High variance suggests need for longer training (1000+ episodes).\n")
        
        f.write("\n---\n")
        f.write(f"\n*Generated from {sum(len(v) for v in results_by_vehicles.values())} experiments*\n")
    
    print(f"[OK] Report saved: {md_path}")
    
    print("\n" + "=" * 80)
    print("Analysis Complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()

