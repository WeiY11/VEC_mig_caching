#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新处理已有的任务到达率训练结果
从现有的训练结果JSON文件中重新提取数据并生成对比图表
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def extract_metrics_from_file(filepath: str, arrival_rate: float) -> Dict[str, Any]:
    """从训练结果文件中提取指标"""
    print(f"Reading: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        training_results = json.load(f)
    
    metrics = {
        'arrival_rate': arrival_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    last_n = 50  # 最后50轮的平均值
    
    # 提取episode奖励
    if 'episode_rewards' in training_results:
        rewards = training_results['episode_rewards']
        if len(rewards) > 0:
            metrics['ave_reward_per_step_final'] = float(np.mean(rewards[-last_n:]))
            metrics['ave_reward_per_step_std'] = float(np.std(rewards[-last_n:]))
            metrics['ave_reward_per_step_all'] = [float(r) for r in rewards]
            print(f"  Reward (last {last_n}): {metrics['ave_reward_per_step_final']:.4f} +/- {metrics['ave_reward_per_step_std']:.4f}")
    
    # 提取episode指标
    if 'episode_metrics' in training_results:
        ep_metrics = training_results['episode_metrics']
        
        # 时延
        if 'avg_delay' in ep_metrics and len(ep_metrics['avg_delay']) > 0:
            delays = ep_metrics['avg_delay']
            metrics['avg_delay_final'] = float(np.mean(delays[-last_n:]))
            metrics['avg_delay_std'] = float(np.std(delays[-last_n:]))
            metrics['avg_delay_all'] = [float(d) for d in delays]
            print(f"  Delay (last {last_n}): {metrics['avg_delay_final']:.4f} +/- {metrics['avg_delay_std']:.4f}")
        
        # 能耗
        if 'total_energy' in ep_metrics and len(ep_metrics['total_energy']) > 0:
            energies = ep_metrics['total_energy']
            metrics['avg_energy_final'] = float(np.mean(energies[-last_n:]))
            metrics['avg_energy_std'] = float(np.std(energies[-last_n:]))
            metrics['avg_energy_all'] = [float(e) for e in energies]
            print(f"  Energy (last {last_n}): {metrics['avg_energy_final']:.4f} +/- {metrics['avg_energy_std']:.4f}")
        
        # 丢弃任务
        if 'task_completion_rate' in ep_metrics and len(ep_metrics['task_completion_rate']) > 0:
            completion_rates = ep_metrics['task_completion_rate']
            dropped_tasks = [(1.0 - rate) * 100 if rate <= 1.0 else 0 for rate in completion_rates]
            metrics['dropped_tasks_final'] = float(np.mean(dropped_tasks[-last_n:]))
            metrics['dropped_tasks_std'] = float(np.std(dropped_tasks[-last_n:]))
            metrics['dropped_tasks_all'] = [float(d) for d in dropped_tasks]
            print(f"  Dropped tasks (last {last_n}): {metrics['dropped_tasks_final']:.2f} +/- {metrics['dropped_tasks_std']:.2f}")
    
    return metrics


def find_training_results(rates: List[float]) -> List[Dict[str, Any]]:
    """查找对应到达率的训练结果文件"""
    results_dir = Path('results/single_agent/td3')
    all_results = []
    
    # 获取最近5个训练结果文件
    result_files = sorted(results_dir.glob('training_results_*.json'), 
                         key=lambda p: p.stat().st_mtime, 
                         reverse=True)[:5]
    
    print(f"\nFound {len(result_files)} recent training result files")
    print("="*80)
    
    for rate in rates:
        print(f"\nProcessing arrival_rate = {rate} tasks/s")
        print("-"*80)
        
        # 尝试找到对应的结果文件（使用最新的那个）
        if result_files:
            latest_file = result_files.pop(0) if result_files else None
            if latest_file:
                metrics = extract_metrics_from_file(str(latest_file), rate)
                all_results.append(metrics)
                
                # 保存提取的指标
                output_dir = 'results/parameter_sensitivity/arrival_rate'
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f'arrival_rate_{rate:.1f}_results.json')
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(metrics, f, indent=2, ensure_ascii=False)
                print(f"Saved: {output_file}")
    
    return all_results


def generate_comparison_plots(all_results: List[Dict[str, Any]], output_dir: str):
    """生成对比图表"""
    if not all_results:
        print("No results to plot!")
        return
    
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    arrival_rates = [r['arrival_rate'] for r in all_results]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TD3 Algorithm - Task Arrival Rate Sensitivity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 图1: 平均步奖励
    ax1 = axes[0, 0]
    rewards_mean = [r.get('ave_reward_per_step_final', 0) for r in all_results]
    rewards_std = [r.get('ave_reward_per_step_std', 0) for r in all_results]
    ax1.plot(arrival_rates, rewards_mean, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax1.fill_between(arrival_rates, 
                     np.array(rewards_mean) - np.array(rewards_std),
                     np.array(rewards_mean) + np.array(rewards_std),
                     alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax1.set_ylabel('Average Reward per Step', fontsize=12)
    ax1.set_title('(a) Average Reward vs Arrival Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 平均时延
    ax2 = axes[0, 1]
    delays_mean = [r.get('avg_delay_final', 0) for r in all_results]
    delays_std = [r.get('avg_delay_std', 0) for r in all_results]
    ax2.plot(arrival_rates, delays_mean, 's-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.fill_between(arrival_rates,
                     np.array(delays_mean) - np.array(delays_std),
                     np.array(delays_mean) + np.array(delays_std),
                     alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax2.set_ylabel('Average Delay (s)', fontsize=12)
    ax2.set_title('(b) Average Delay vs Arrival Rate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 平均能耗
    ax3 = axes[1, 0]
    energies_mean = [r.get('avg_energy_final', 0) for r in all_results]
    energies_std = [r.get('avg_energy_std', 0) for r in all_results]
    ax3.plot(arrival_rates, energies_mean, '^-', linewidth=2, markersize=8, color='#3498db')
    ax3.fill_between(arrival_rates,
                     np.array(energies_mean) - np.array(energies_std),
                     np.array(energies_mean) + np.array(energies_std),
                     alpha=0.2, color='#3498db')
    ax3.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax3.set_ylabel('Average Energy (J)', fontsize=12)
    ax3.set_title('(c) Average Energy vs Arrival Rate', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 丢弃任务数
    ax4 = axes[1, 1]
    dropped_mean = [r.get('dropped_tasks_final', 0) for r in all_results]
    dropped_std = [r.get('dropped_tasks_std', 0) for r in all_results]
    ax4.plot(arrival_rates, dropped_mean, 'd-', linewidth=2, markersize=8, color='#f39c12')
    ax4.fill_between(arrival_rates,
                     np.array(dropped_mean) - np.array(dropped_std),
                     np.array(dropped_mean) + np.array(dropped_std),
                     alpha=0.2, color='#f39c12')
    ax4.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax4.set_ylabel('Dropped Tasks', fontsize=12)
    ax4.set_title('(d) Dropped Tasks vs Arrival Rate', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"arrival_rate_comparison_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n{'='*80}")
    print(f"Plot saved: {plot_path}")
    print(f"{'='*80}")
    plt.close()


def generate_summary(all_results: List[Dict[str, Any]], output_dir: str):
    """生成汇总报告"""
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    
    summary = {
        'experiment_info': {
            'algorithm': 'TD3',
            'parameter': 'arrival_rate',
            'unit': 'tasks/s',
            'num_experiments': len(all_results),
            'timestamp': datetime.now().isoformat()
        },
        'results': []
    }
    
    print(f"\n{'='*80}")
    print("Experiment Results Summary")
    print(f"{'='*80}")
    print(f"{'Rate':>10} | {'Avg Reward':>12} | {'Avg Delay':>10} | {'Avg Energy':>11} | {'Dropped':>10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        rate = result['arrival_rate']
        reward = result.get('ave_reward_per_step_final', 0)
        delay = result.get('avg_delay_final', 0)
        energy = result.get('avg_energy_final', 0)
        dropped = result.get('dropped_tasks_final', 0)
        
        print(f"{rate:>10.1f} | {reward:>12.4f} | {delay:>10.4f} | {energy:>11.4f} | {dropped:>10.2f}")
        
        summary['results'].append({
            'arrival_rate': rate,
            'ave_reward_per_step': reward,
            'avg_delay': delay,
            'avg_energy': energy,
            'dropped_tasks': dropped
        })
    
    print(f"{'='*80}\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"arrival_rate_summary_{timestamp}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Summary saved: {summary_path}")


if __name__ == '__main__':
    print("="*80)
    print("Reprocessing Arrival Rate Training Results")
    print("="*80)
    
    rates = [1.0, 1.5, 2.0, 2.5, 3.0]
    output_dir = 'results/parameter_sensitivity/arrival_rate'
    
    # 查找并处理训练结果
    all_results = find_training_results(rates)
    
    if not all_results:
        print("\nERROR: No training results found!")
        sys.exit(1)
    
    # 生成对比图表
    generate_comparison_plots(all_results, output_dir)
    
    # 生成汇总报告
    generate_summary(all_results, output_dir)
    
    print(f"\n{'='*80}")
    print("Completed!")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Total experiments: {len(all_results)}")
    print(f"{'='*80}\n")

