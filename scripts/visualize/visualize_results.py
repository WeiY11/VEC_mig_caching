#!/usr/bin/env python3
"""
实验结果可视化脚本
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def load_results():
    """加载实验结果"""
    with open('results/full_experiment_results.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def create_performance_comparison():
    """创建性能对比图表"""
    results = load_results()
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    scenarios = ['standard', 'high_load', 'large_scale']
    algorithms = ['MATD3-MIG', 'Random', 'Greedy', 'Round_Robin', 'Load_Aware']
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('MATD3-MIG算法性能对比', fontsize=16, fontweight='bold')
    
    # 1. 平均时延对比
    ax1 = axes[0, 0]
    delays = []
    for scenario in scenarios:
        scenario_delays = [results[scenario][alg]['avg_delay'] for alg in algorithms]
        delays.append(scenario_delays)
    
    x = np.arange(len(scenarios))
    width = 0.15
    
    for i, alg in enumerate(algorithms):
        alg_delays = [delays[j][i] for j in range(len(scenarios))]
        ax1.bar(x + i*width, alg_delays, width, label=alg)
    
    ax1.set_xlabel('实验场景')
    ax1.set_ylabel('平均时延 (秒)')
    ax1.set_title('平均时延对比')
    ax1.set_xticks(x + width * 2)
    ax1.set_xticklabels(['标准', '高负载', '大规模'])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 完成率对比
    ax2 = axes[0, 1]
    completion_rates = []
    for scenario in scenarios:
        scenario_rates = [results[scenario][alg]['completion_rate'] * 100 for alg in algorithms]
        completion_rates.append(scenario_rates)
    
    for i, alg in enumerate(algorithms):
        alg_rates = [completion_rates[j][i] for j in range(len(scenarios))]
        ax2.bar(x + i*width, alg_rates, width, label=alg)
    
    ax2.set_xlabel('实验场景')
    ax2.set_ylabel('完成率 (%)')
    ax2.set_title('任务完成率对比')
    ax2.set_xticks(x + width * 2)
    ax2.set_xticklabels(['标准', '高负载', '大规模'])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 能耗对比
    ax3 = axes[1, 0]
    energies = []
    for scenario in scenarios:
        scenario_energies = [results[scenario][alg]['total_energy'] / 1e6 for alg in algorithms]  # 转换为MJ
        energies.append(scenario_energies)
    
    for i, alg in enumerate(algorithms):
        alg_energies = [energies[j][i] for j in range(len(scenarios))]
        ax3.bar(x + i*width, alg_energies, width, label=alg)
    
    ax3.set_xlabel('实验场景')
    ax3.set_ylabel('总能耗 (MJ)')
    ax3.set_title('总能耗对比')
    ax3.set_xticks(x + width * 2)
    ax3.set_xticklabels(['标准', '高负载', '大规模'])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 缓存命中率对比
    ax4 = axes[1, 1]
    cache_rates = []
    for scenario in scenarios:
        scenario_cache = [results[scenario][alg]['cache_hit_rate'] * 100 for alg in algorithms]
        cache_rates.append(scenario_cache)
    
    for i, alg in enumerate(algorithms):
        alg_cache = [cache_rates[j][i] for j in range(len(scenarios))]
        ax4.bar(x + i*width, alg_cache, width, label=alg)
    
    ax4.set_xlabel('实验场景')
    ax4.set_ylabel('缓存命中率 (%)')
    ax4.set_title('缓存命中率对比')
    ax4.set_xticks(x + width * 2)
    ax4.set_xticklabels(['标准', '高负载', '大规模'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_improvement_chart():
    """创建改进效果图表"""
    results = load_results()
    
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # MATD3-MIG相对于其他算法的改进百分比
    scenarios = ['standard', 'high_load', 'large_scale']
    baseline_algs = ['Random', 'Greedy', 'Round_Robin', 'Load_Aware']
    metrics = ['delay_improvement', 'energy_improvement', 'completion_improvement']
    metric_names = ['时延改进', '能耗改进', '完成率改进']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MATD3-MIG相对其他算法的改进效果', fontsize=16, fontweight='bold')
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        improvements = []
        for scenario in scenarios:
            scenario_improvements = [results[scenario]['improvements'][alg][metric] for alg in baseline_algs]
            improvements.append(scenario_improvements)
        
        x = np.arange(len(scenarios))
        width = 0.2
        
        for j, alg in enumerate(baseline_algs):
            alg_improvements = [improvements[k][j] for k in range(len(scenarios))]
            ax.bar(x + j*width, alg_improvements, width, label=f'vs {alg}')
        
        ax.set_xlabel('实验场景')
        ax.set_ylabel('改进百分比 (%)')
        ax.set_title(f'{metric_name}对比')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(['标准', '高负载', '大规模'])
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/improvement_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_summary():
    """打印实验结果摘要"""
    results = load_results()
    
    print("=" * 80)
    print("🎯 MATD3-MIG算法实验结果摘要")
    print("=" * 80)
    
    for scenario in ['standard', 'high_load', 'large_scale']:
        scenario_names = {
            'standard': '标准场景',
            'high_load': '高负载场景', 
            'large_scale': '大规模场景'
        }
        
        print(f"\n📊 {scenario_names[scenario]}:")
        print("-" * 50)
        
        matd3_results = results[scenario]['MATD3-MIG']
        print(f"  ✅ 平均时延: {matd3_results['avg_delay']:.3f}s")
        print(f"  ✅ 任务完成率: {matd3_results['completion_rate']*100:.1f}%")
        print(f"  ✅ 总能耗: {matd3_results['total_energy']/1e6:.1f}MJ")
        print(f"  ✅ 缓存命中率: {matd3_results['cache_hit_rate']*100:.0f}%")
        
        print(f"\n  🚀 最佳改进效果:")
        improvements = results[scenario]['improvements']
        best_delay = max(improvements.values(), key=lambda x: x['delay_improvement'])
        best_energy = max(improvements.values(), key=lambda x: x['energy_improvement'])
        best_completion = max(improvements.values(), key=lambda x: x['completion_improvement'])
        
        print(f"    • 时延改进: {best_delay['delay_improvement']:.1f}%")
        print(f"    • 能耗改进: {best_energy['energy_improvement']:.1f}%")
        print(f"    • 完成率改进: {best_completion['completion_improvement']:.1f}%")

if __name__ == "__main__":
    print("🚀 开始生成实验结果可视化...")
    
    # 确保结果目录存在
    Path('results').mkdir(exist_ok=True)
    
    # 打印摘要
    print_summary()
    
    # 生成图表
    try:
        print("\n📊 生成性能对比图表...")
        create_performance_comparison()
        
        print("📈 生成改进效果图表...")
        create_improvement_chart()
        
        print("\n✅ 可视化完成！图表已保存到 results/ 目录")
        print("📁 查看文件:")
        print("  - results/performance_comparison.png")
        print("  - results/improvement_comparison.png")
        
    except ImportError as e:
        print(f"\n⚠️ 缺少可视化依赖: {e}")
        print("💡 安装建议: pip install matplotlib pandas")
        print("📊 但实验数据已成功保存到 results/ 目录")