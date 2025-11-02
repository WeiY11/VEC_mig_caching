#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权重对比实验可视化工具

生成对比图表：
1. 综合性能雷达图
2. 各指标对比柱状图
3. 收敛曲线对比
4. Pareto前沿分析
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from typing import List, Dict
import seaborn as sns

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def load_experiment_results(results_dir: str = "results/weight_comparison") -> List[Dict]:
    """加载所有实验结果"""
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return []
    
    results = []
    
    for config_name in os.listdir(results_dir):
        config_path = os.path.join(results_dir, config_name)
        if not os.path.isdir(config_path):
            continue
        
        # 查找训练结果文件
        result_files = [f for f in os.listdir(config_path) 
                       if f.startswith('training_results') and f.endswith('.json')]
        
        if not result_files:
            continue
        
        # 读取最新结果
        result_file = sorted(result_files)[-1]
        result_path = os.path.join(config_path, result_file)
        
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            
            # 读取权重配置
            config_file = os.path.join(config_path, "weights_config.json")
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    weights = json.load(f)
            else:
                weights = {}
            
            metrics = data.get('episode_metrics', {})
            
            results.append({
                'name': config_name,
                'weights': weights,
                'metrics': metrics,
                'data': data
            })
            
        except Exception as e:
            print(f"读取失败 {config_name}: {e}")
    
    return results


def plot_radar_comparison(results: List[Dict], output_file: str = None):
    """绘制综合性能雷达图"""
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # 定义指标
    categories = ['完成率', '缓存命中率', '能效\n(归一化)', '时延\n(归一化)', '稳定性']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=12)
    
    # 为每个配置绘制雷达图
    for result in results[:5]:  # 最多显示5个配置
        metrics = result['metrics']
        last_100 = min(100, len(metrics.get('total_energy', [])))
        
        if last_100 == 0:
            continue
        
        # 计算各项指标（归一化到0-1）
        completion = np.mean(metrics['task_completion_rate'][-last_100:])
        cache_hit = np.mean(metrics['cache_hit_rate'][-last_100:])
        
        # 能效：能耗越低越好
        avg_energy = np.mean(metrics['total_energy'][-last_100:])
        energy_eff = 1 - (avg_energy - 3000) / 5000  # 假设3000-8000J范围
        energy_eff = np.clip(energy_eff, 0, 1)
        
        # 时延：时延越低越好
        avg_delay = np.mean(metrics['avg_delay'][-last_100:])
        delay_perf = 1 - (avg_delay - 0.25) / 0.3  # 假设0.25-0.55s范围
        delay_perf = np.clip(delay_perf, 0, 1)
        
        # 稳定性：标准差越小越好
        std_completion = np.std(metrics['task_completion_rate'][-last_100:])
        stability = 1 - std_completion / 0.05  # 假设最大标准差0.05
        stability = np.clip(stability, 0, 1)
        
        values = [completion, cache_hit, energy_eff, delay_perf, stability]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['name'])
        ax.fill(angles, values, alpha=0.15)
    
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.set_title('权重配置综合性能对比\n(归一化雷达图)', size=16, pad=20)
    ax.grid(True)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"雷达图已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_metrics_comparison(results: List[Dict], output_file: str = None):
    """绘制各指标对比柱状图"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('权重配置详细指标对比', fontsize=18, fontweight='bold')
    
    names = []
    energies = []
    cache_hits = []
    completions = []
    delays = []
    losses = []
    stds_energy = []
    
    for result in results:
        metrics = result['metrics']
        last_100 = min(100, len(metrics.get('total_energy', [])))
        
        if last_100 == 0:
            continue
        
        names.append(result['name'][:15])  # 截断长名称
        energies.append(np.mean(metrics['total_energy'][-last_100:]))
        cache_hits.append(np.mean(metrics['cache_hit_rate'][-last_100:]) * 100)
        completions.append(np.mean(metrics['task_completion_rate'][-last_100:]) * 100)
        delays.append(np.mean(metrics['avg_delay'][-last_100:]))
        losses.append(np.mean(metrics['data_loss_ratio_bytes'][-last_100:]) * 100)
        stds_energy.append(np.std(metrics['total_energy'][-last_100:]))
    
    x = np.arange(len(names))
    
    # 1. 能耗对比
    axes[0, 0].bar(x, energies, color='steelblue', alpha=0.8)
    axes[0, 0].set_title('平均能耗 (J)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('能耗 (J)', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # 2. 缓存命中率
    axes[0, 1].bar(x, cache_hits, color='orange', alpha=0.8)
    axes[0, 1].set_title('缓存命中率 (%)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('命中率 (%)', fontsize=12)
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. 任务完成率
    axes[0, 2].bar(x, completions, color='green', alpha=0.8)
    axes[0, 2].set_title('任务完成率 (%)', fontsize=14, fontweight='bold')
    axes[0, 2].set_ylabel('完成率 (%)', fontsize=12)
    axes[0, 2].set_xticks(x)
    axes[0, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[0, 2].axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95%目标')
    axes[0, 2].legend()
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. 平均时延
    axes[1, 0].bar(x, delays, color='purple', alpha=0.8)
    axes[1, 0].set_title('平均时延 (s)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('时延 (s)', fontsize=12)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 0].axhline(y=0.40, color='red', linestyle='--', alpha=0.5, label='0.4s目标')
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. 数据丢失率
    axes[1, 1].bar(x, losses, color='red', alpha=0.8)
    axes[1, 1].set_title('数据丢失率 (%)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('丢失率 (%)', fontsize=12)
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. 能耗稳定性
    axes[1, 2].bar(x, stds_energy, color='brown', alpha=0.8)
    axes[1, 2].set_title('能耗标准差 (J)', fontsize=14, fontweight='bold')
    axes[1, 2].set_ylabel('标准差 (J)', fontsize=12)
    axes[1, 2].set_xticks(x)
    axes[1, 2].set_xticklabels(names, rotation=45, ha='right')
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"对比图已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_convergence_comparison(results: List[Dict], output_file: str = None):
    """绘制收敛曲线对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('收敛曲线对比', fontsize=18, fontweight='bold')
    
    for result in results[:5]:  # 最多显示5条曲线
        metrics = result['metrics']
        
        if len(metrics.get('total_energy', [])) == 0:
            continue
        
        episodes = range(1, len(metrics['total_energy']) + 1)
        
        # 计算移动平均
        window = 20
        
        def moving_average(data, window):
            return np.convolve(data, np.ones(window)/window, mode='valid')
        
        # 1. 能耗收敛
        energy_ma = moving_average(metrics['total_energy'], window)
        axes[0, 0].plot(range(window, len(metrics['total_energy'])+1), energy_ma, 
                       label=result['name'], linewidth=2, alpha=0.8)
        
        # 2. 缓存命中率收敛
        cache_ma = moving_average(metrics['cache_hit_rate'], window)
        axes[0, 1].plot(range(window, len(metrics['cache_hit_rate'])+1), 
                       np.array(cache_ma)*100, label=result['name'], linewidth=2, alpha=0.8)
        
        # 3. 完成率收敛
        completion_ma = moving_average(metrics['task_completion_rate'], window)
        axes[1, 0].plot(range(window, len(metrics['task_completion_rate'])+1), 
                       np.array(completion_ma)*100, label=result['name'], linewidth=2, alpha=0.8)
        
        # 4. 时延收敛
        delay_ma = moving_average(metrics['avg_delay'], window)
        axes[1, 1].plot(range(window, len(metrics['avg_delay'])+1), delay_ma, 
                       label=result['name'], linewidth=2, alpha=0.8)
    
    # 设置子图
    axes[0, 0].set_title('能耗收敛', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('能耗 (J)', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    axes[0, 1].set_title('缓存命中率收敛', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('命中率 (%)', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    
    axes[1, 0].set_title('完成率收敛', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('完成率 (%)', fontsize=12)
    axes[1, 0].axhline(y=95, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    axes[1, 1].set_title('时延收敛', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Episode', fontsize=12)
    axes[1, 1].set_ylabel('时延 (s)', fontsize=12)
    axes[1, 1].axhline(y=0.40, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"收敛曲线已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_cost_comparison(results: List[Dict], output_file: str = None):
    """绘制平均成本对比图（核心目标函数）"""
    
    fig, axes = plt.subplots(2, 1, figsize=(16, 14))
    fig.suptitle('平均成本对比分析\n(Objective = ω_T × (Delay/Target) + ω_E × (Energy/Target))', 
                 fontsize=18, fontweight='bold')
    
    names = []
    total_costs = []
    delay_costs = []
    energy_costs = []
    cache_costs = []
    
    for result in results:
        metrics = result['metrics']
        weights = result['weights']
        last_100 = min(100, len(metrics.get('total_energy', [])))
        
        if last_100 == 0:
            continue
        
        # 获取平均指标
        avg_delay = np.mean(metrics['avg_delay'][-last_100:])
        avg_energy = np.mean(metrics['total_energy'][-last_100:])
        avg_cache_miss = 1 - np.mean(metrics['cache_hit_rate'][-last_100:])
        
        # 获取权重配置
        w_delay = weights.get('reward_weight_delay', 2.0)
        w_energy = weights.get('reward_weight_energy', 1.2)
        w_cache = weights.get('reward_weight_cache', 0.15)
        target_delay = weights.get('latency_target', 0.40)
        target_energy = weights.get('energy_target', 1200.0)
        
        # 计算归一化成本
        norm_delay = avg_delay / target_delay
        norm_energy = avg_energy / target_energy
        
        # 计算各部分成本
        delay_cost = w_delay * norm_delay
        energy_cost = w_energy * norm_energy
        cache_cost = w_cache * avg_cache_miss
        total_cost = delay_cost + energy_cost + cache_cost
        
        names.append(result['name'][:15])
        total_costs.append(total_cost)
        delay_costs.append(delay_cost)
        energy_costs.append(energy_cost)
        cache_costs.append(cache_cost)
    
    # 排序（按总成本）
    sorted_indices = np.argsort(total_costs)
    names = [names[i] for i in sorted_indices]
    total_costs = [total_costs[i] for i in sorted_indices]
    delay_costs = [delay_costs[i] for i in sorted_indices]
    energy_costs = [energy_costs[i] for i in sorted_indices]
    cache_costs = [cache_costs[i] for i in sorted_indices]
    
    x = np.arange(len(names))
    
    # 1. 总成本对比（柱状图）
    bars = axes[0].bar(x, total_costs, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # 标注最小值
    min_idx = np.argmin(total_costs)
    bars[min_idx].set_color('gold')
    bars[min_idx].set_edgecolor('red')
    bars[min_idx].set_linewidth(3)
    
    axes[0].set_title('总成本对比 (越低越好)', fontsize=16, fontweight='bold', pad=20)
    axes[0].set_ylabel('总成本', fontsize=14, fontweight='bold')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上标注数值
    for i, (cost, bar) in enumerate(zip(total_costs, bars)):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{cost:.2f}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 标注最优配置
    axes[0].text(min_idx, total_costs[min_idx] * 1.15, 
                '★ 最优配置', ha='center', fontsize=12, 
                fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))
    
    # 2. 成本组成堆叠图
    width = 0.6
    p1 = axes[1].bar(x, delay_costs, width, label='时延成本', color='purple', alpha=0.8)
    p2 = axes[1].bar(x, energy_costs, width, bottom=delay_costs, label='能耗成本', color='orange', alpha=0.8)
    p3 = axes[1].bar(x, cache_costs, width, 
                    bottom=np.array(delay_costs) + np.array(energy_costs),
                    label='缓存成本', color='green', alpha=0.8)
    
    axes[1].set_title('成本组成分析 (堆叠图)', fontsize=16, fontweight='bold', pad=20)
    axes[1].set_ylabel('成本值', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('配置方案', fontsize=14, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=45, ha='right', fontsize=11)
    axes[1].legend(loc='upper left', fontsize=12, framealpha=0.9)
    axes[1].grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"成本对比图已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()
    
    # 打印详细成本分析
    print("\n" + "="*80)
    print("详细成本分析")
    print("="*80)
    print(f"{'配置':15s} | {'总成本':>8s} | {'时延成本':>8s} | {'能耗成本':>8s} | {'缓存成本':>8s} | {'时延占比':>8s} | {'能耗占比':>8s}")
    print("-"*80)
    for i in range(len(names)):
        delay_pct = delay_costs[i] / total_costs[i] * 100
        energy_pct = energy_costs[i] / total_costs[i] * 100
        marker = "★" if i == min_idx else " "
        print(f"{marker} {names[i]:14s} | {total_costs[i]:8.2f} | {delay_costs[i]:8.2f} | "
              f"{energy_costs[i]:8.2f} | {cache_costs[i]:8.2f} | {delay_pct:7.1f}% | {energy_pct:7.1f}%")
    print("="*80)
    print(f"最优配置: {names[min_idx]} (总成本 = {total_costs[min_idx]:.2f})")
    print("="*80 + "\n")


def plot_reward_curves(results: List[Dict], output_file: str = None):
    """绘制奖励曲线对比"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('奖励曲线对比 (Reward per Episode)', fontsize=18, fontweight='bold')
    
    # 移动平均窗口
    window = 20
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 1. 前5个配置的奖励曲线
    for i, result in enumerate(results[:5]):
        metrics = result['metrics']
        if 'episode_rewards' not in result['data']:
            continue
        
        rewards = result['data']['episode_rewards']
        if len(rewards) == 0:
            continue
        
        episodes = range(1, len(rewards) + 1)
        rewards_ma = moving_average(rewards, window)
        
        axes[0, 0].plot(range(window, len(rewards)+1), rewards_ma, 
                       label=result['name'], linewidth=2, alpha=0.8)
    
    axes[0, 0].set_title('奖励曲线 (配置1-5)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Average Reward', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 后5个配置的奖励曲线
    for i, result in enumerate(results[5:10]):
        metrics = result['metrics']
        if 'episode_rewards' not in result['data']:
            continue
        
        rewards = result['data']['episode_rewards']
        if len(rewards) == 0:
            continue
        
        episodes = range(1, len(rewards) + 1)
        rewards_ma = moving_average(rewards, window)
        
        axes[0, 1].plot(range(window, len(rewards)+1), rewards_ma, 
                       label=result['name'], linewidth=2, alpha=0.8)
    
    axes[0, 1].set_title('奖励曲线 (配置6-10)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Average Reward', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 最后4个配置的奖励曲线
    for i, result in enumerate(results[10:]):
        metrics = result['metrics']
        if 'episode_rewards' not in result['data']:
            continue
        
        rewards = result['data']['episode_rewards']
        if len(rewards) == 0:
            continue
        
        episodes = range(1, len(rewards) + 1)
        rewards_ma = moving_average(rewards, window)
        
        axes[1, 0].plot(range(window, len(rewards)+1), rewards_ma, 
                       label=result['name'], linewidth=2, alpha=0.8)
    
    axes[1, 0].set_title('奖励曲线 (配置11-14)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Average Reward', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 最终奖励对比（后100轮平均）
    final_rewards = []
    config_names = []
    
    for result in results:
        if 'episode_rewards' not in result['data']:
            continue
        
        rewards = result['data']['episode_rewards']
        if len(rewards) == 0:
            continue
        
        last_100 = min(100, len(rewards))
        avg_reward = np.mean(rewards[-last_100:])
        
        final_rewards.append(avg_reward)
        config_names.append(result['name'][:12])
    
    if final_rewards:
        x = np.arange(len(config_names))
        bars = axes[1, 1].bar(x, final_rewards, color='steelblue', alpha=0.8, edgecolor='black')
        
        # 标注最高奖励
        max_idx = np.argmax(final_rewards)
        bars[max_idx].set_color('gold')
        bars[max_idx].set_edgecolor('red')
        bars[max_idx].set_linewidth(2)
        
        axes[1, 1].set_title('最终奖励对比 (后100轮平均)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('Average Reward', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 标注数值
        for bar in bars:
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{height:.2f}',
                          ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"奖励曲线对比图已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_cost_curves(results: List[Dict], output_file: str = None):
    """绘制成本曲线对比（随训练进度变化）"""
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('成本曲线对比 (Cost = ω_T × (Delay/Target) + ω_E × (Energy/Target))', 
                 fontsize=18, fontweight='bold')
    
    # 移动平均窗口
    window = 20
    
    def moving_average(data, window):
        if len(data) < window:
            return data
        return np.convolve(data, np.ones(window)/window, mode='valid')
    
    # 计算成本曲线
    def calculate_cost_curve(result):
        metrics = result['metrics']
        weights = result['weights']
        
        if len(metrics.get('avg_delay', [])) == 0:
            return None
        
        w_delay = weights.get('reward_weight_delay', 2.0)
        w_energy = weights.get('reward_weight_energy', 1.2)
        w_cache = weights.get('reward_weight_cache', 0.15)
        target_delay = weights.get('latency_target', 0.40)
        target_energy = weights.get('energy_target', 1200.0)
        
        delays = np.array(metrics['avg_delay'])
        energies = np.array(metrics['total_energy'])
        cache_hits = np.array(metrics['cache_hit_rate'])
        
        # 计算每个episode的成本
        norm_delays = delays / target_delay
        norm_energies = energies / target_energy
        cache_misses = 1 - cache_hits
        
        costs = w_delay * norm_delays + w_energy * norm_energies + w_cache * cache_misses
        
        return costs
    
    # 1. 前5个配置的成本曲线
    for result in results[:5]:
        costs = calculate_cost_curve(result)
        if costs is None:
            continue
        
        costs_ma = moving_average(costs, window)
        episodes = range(window, len(costs)+1)
        
        axes[0, 0].plot(episodes, costs_ma, label=result['name'], linewidth=2, alpha=0.8)
    
    axes[0, 0].set_title('成本曲线 (配置1-5)', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Episode', fontsize=12)
    axes[0, 0].set_ylabel('Total Cost', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.3)
    
    # 2. 后5个配置的成本曲线
    for result in results[5:10]:
        costs = calculate_cost_curve(result)
        if costs is None:
            continue
        
        costs_ma = moving_average(costs, window)
        episodes = range(window, len(costs)+1)
        
        axes[0, 1].plot(episodes, costs_ma, label=result['name'], linewidth=2, alpha=0.8)
    
    axes[0, 1].set_title('成本曲线 (配置6-10)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Episode', fontsize=12)
    axes[0, 1].set_ylabel('Total Cost', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(alpha=0.3)
    
    # 3. 最后4个配置的成本曲线
    for result in results[10:]:
        costs = calculate_cost_curve(result)
        if costs is None:
            continue
        
        costs_ma = moving_average(costs, window)
        episodes = range(window, len(costs)+1)
        
        axes[1, 0].plot(episodes, costs_ma, label=result['name'], linewidth=2, alpha=0.8)
    
    axes[1, 0].set_title('成本曲线 (配置11-14)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Episode', fontsize=12)
    axes[1, 0].set_ylabel('Total Cost', fontsize=12)
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.3)
    
    # 4. 成本下降率对比
    cost_reductions = []
    config_names = []
    
    for result in results:
        costs = calculate_cost_curve(result)
        if costs is None:
            continue
        
        # 计算成本下降率（前20% vs 后20%）
        split_point = len(costs) // 5
        early_cost = np.mean(costs[:split_point])
        late_cost = np.mean(costs[-split_point:])
        
        if early_cost > 0:
            reduction = (early_cost - late_cost) / early_cost * 100
            cost_reductions.append(reduction)
            config_names.append(result['name'][:12])
    
    if cost_reductions:
        x = np.arange(len(config_names))
        colors = ['green' if r > 0 else 'red' for r in cost_reductions]
        bars = axes[1, 1].bar(x, cost_reductions, color=colors, alpha=0.8, edgecolor='black')
        
        axes[1, 1].set_title('成本改善率 (前20% vs 后20%)', fontsize=14, fontweight='bold')
        axes[1, 1].set_ylabel('改善率 (%)', fontsize=12)
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(config_names, rotation=45, ha='right', fontsize=10)
        axes[1, 1].axhline(y=0, color='black', linestyle='--', linewidth=1)
        axes[1, 1].grid(axis='y', alpha=0.3)
        
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars, cost_reductions)):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height,
                          f'{val:.1f}%',
                          ha='center', va='bottom' if val > 0 else 'top', 
                          fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"成本曲线对比图已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def plot_pareto_frontier(results: List[Dict], output_file: str = None):
    """绘制Pareto前沿（时延-能耗权衡）"""
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    delays = []
    energies = []
    names = []
    
    for result in results:
        metrics = result['metrics']
        last_100 = min(100, len(metrics.get('total_energy', [])))
        
        if last_100 == 0:
            continue
        
        avg_delay = np.mean(metrics['avg_delay'][-last_100:])
        avg_energy = np.mean(metrics['total_energy'][-last_100:])
        
        delays.append(avg_delay)
        energies.append(avg_energy)
        names.append(result['name'])
    
    # 绘制散点
    scatter = ax.scatter(delays, energies, s=200, c=range(len(names)), 
                        cmap='viridis', alpha=0.7, edgecolors='black', linewidth=2)
    
    # 标注配置名称
    for i, name in enumerate(names):
        ax.annotate(name, (delays[i], energies[i]), 
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))
    
    # 找到Pareto前沿
    pareto_front = []
    for i in range(len(delays)):
        is_pareto = True
        for j in range(len(delays)):
            if i != j:
                # 如果存在其他点同时在时延和能耗上都优于当前点，则当前点不在Pareto前沿上
                if delays[j] <= delays[i] and energies[j] <= energies[i]:
                    if delays[j] < delays[i] or energies[j] < energies[i]:
                        is_pareto = False
                        break
        if is_pareto:
            pareto_front.append(i)
    
    # 绘制Pareto前沿
    if pareto_front:
        pareto_delays = [delays[i] for i in sorted(pareto_front, key=lambda x: delays[x])]
        pareto_energies = [energies[i] for i in sorted(pareto_front, key=lambda x: delays[x])]
        ax.plot(pareto_delays, pareto_energies, 'r--', linewidth=2, 
               label='Pareto前沿', alpha=0.5)
    
    ax.set_xlabel('平均时延 (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('平均能耗 (J)', fontsize=14, fontweight='bold')
    ax.set_title('时延-能耗权衡分析\n(Pareto前沿)', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Pareto分析已保存: {output_file}")
    else:
        plt.show()
    
    plt.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="权重对比可视化工具")
    parser.add_argument("--results-dir", type=str, default="results/weight_comparison",
                       help="实验结果目录")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="图表输出目录")
    
    args = parser.parse_args()
    
    # 加载结果
    print("加载实验结果...")
    results = load_experiment_results(args.results_dir)
    
    if not results:
        print("未找到有效的实验结果！")
        return
    
    print(f"找到 {len(results)} 个实验结果")
    
    # 设置输出目录
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/weight_comparison/comparison_{timestamp}"
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成所有图表
    print("\n生成对比图表...")
    
    # 1. 成本对比图（最重要）⭐
    print("  [1/7] 生成成本对比图...")
    plot_cost_comparison(results, 
                        os.path.join(args.output_dir, "cost_comparison.png"))
    
    # 2. 成本曲线对比 ⭐ 新增
    print("  [2/7] 生成成本曲线对比图...")
    plot_cost_curves(results,
                    os.path.join(args.output_dir, "cost_curves.png"))
    
    # 3. 奖励曲线对比 ⭐ 新增
    print("  [3/7] 生成奖励曲线对比图...")
    plot_reward_curves(results,
                      os.path.join(args.output_dir, "reward_curves.png"))
    
    # 4. 雷达图
    print("  [4/7] 生成综合性能雷达图...")
    plot_radar_comparison(results, 
                         os.path.join(args.output_dir, "radar_comparison.png"))
    
    # 5. 详细指标对比
    print("  [5/7] 生成详细指标对比图...")
    plot_metrics_comparison(results, 
                           os.path.join(args.output_dir, "metrics_comparison.png"))
    
    # 6. 收敛曲线
    print("  [6/7] 生成收敛曲线对比图...")
    plot_convergence_comparison(results, 
                               os.path.join(args.output_dir, "convergence_comparison.png"))
    
    # 7. Pareto前沿
    print("  [7/7] 生成Pareto前沿分析图...")
    plot_pareto_frontier(results, 
                        os.path.join(args.output_dir, "pareto_frontier.png"))
    
    print(f"\n✓ 所有图表已保存到: {args.output_dir}/")
    print("\n生成的图表列表:")
    print("  1. cost_comparison.png   - 成本对比（柱状图+堆叠图）⭐")
    print("  2. cost_curves.png       - 成本曲线（训练过程）⭐")
    print("  3. reward_curves.png     - 奖励曲线（训练过程）⭐")
    print("  4. radar_comparison.png  - 综合性能雷达图")
    print("  5. metrics_comparison.png - 6指标详细对比")
    print("  6. convergence_comparison.png - 4维收敛曲线")
    print("  7. pareto_frontier.png   - 时延-能耗Pareto前沿")


if __name__ == "__main__":
    main()

