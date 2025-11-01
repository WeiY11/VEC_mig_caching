#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成任务到达率 vs 成本(Cost)的对比图表
Cost = -Reward (因为Reward是负值)
"""

import sys
import os

# 修复Windows编码问题
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 读取汇总数据
with open('results/parameter_sensitivity/arrival_rate/arrival_rate_summary_20251031_203223.json', 'r', encoding='utf-8') as f:
    summary = json.load(f)

results = summary['results']

# 提取数据
arrival_rates = [r['arrival_rate'] for r in results]
rewards = [r['ave_reward_per_step'] for r in results]
costs = [-r for r in rewards]  # Cost = -Reward
delays = [r['avg_delay'] for r in results]
energies = [r['avg_energy'] for r in results]
dropped = [r['dropped_tasks'] for r in results]

# 读取详细数据以获取标准差
costs_std = []
for rate in arrival_rates:
    detail_file = f'results/parameter_sensitivity/arrival_rate/arrival_rate_{rate:.1f}_results.json'
    with open(detail_file, 'r', encoding='utf-8') as f:
        detail = json.load(f)
        # Cost的标准差 = Reward的标准差（因为只是取负）
        std = detail.get('ave_reward_per_step_std', 0)
        costs_std.append(std)

# 创建图表
fig, ax = plt.subplots(figsize=(12, 7))

# 绘制成本曲线
ax.plot(arrival_rates, costs, 'o-', linewidth=3, markersize=10, 
        color='#e74c3c', label='Average Cost per Step')
ax.fill_between(arrival_rates, 
                np.array(costs) - np.array(costs_std),
                np.array(costs) + np.array(costs_std),
                alpha=0.2, color='#e74c3c')

# 标注数值
for i, (rate, cost) in enumerate(zip(arrival_rates, costs)):
    ax.annotate(f'{cost:.1f}', 
                xy=(rate, cost), 
                xytext=(0, 10),
                textcoords='offset points',
                ha='center',
                fontsize=10,
                fontweight='bold',
                color='#e74c3c')

ax.set_xlabel('Task Arrival Rate (tasks/s)', fontsize=14, fontweight='bold')
ax.set_ylabel('Average Cost per Step', fontsize=14, fontweight='bold')
ax.set_title('TD3 Algorithm - Average Cost per Step vs Task Arrival Rate', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(fontsize=12, loc='upper right')

# 设置y轴从0开始，让趋势更明显
ax.set_ylim(bottom=0)

plt.tight_layout()

# 保存图表
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
output_path = f'results/parameter_sensitivity/arrival_rate/cost_comparison_{timestamp}.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Cost comparison plot saved: {output_path}")
plt.close()

# 创建详细的成本对比表格
print("\n" + "="*80)
print("TD3 Algorithm - Cost Analysis vs Task Arrival Rate")
print("="*80)
print(f"{'Arrival Rate':>15} | {'Avg Cost/Step':>15} | {'Std Dev':>12} | {'Trend':>10}")
print("-"*80)

for i, (rate, cost, std) in enumerate(zip(arrival_rates, costs, costs_std)):
    if i == 0:
        trend = "Baseline"
    else:
        change = cost - costs[i-1]
        change_pct = (change / costs[i-1]) * 100
        if change < 0:
            trend = f"↓ {abs(change_pct):.1f}%"
        else:
            trend = f"↑ {abs(change_pct):.1f}%"
    
    print(f"{rate:>15.1f} | {cost:>15.2f} | {std:>12.2f} | {trend:>10}")

print("="*80)

# 额外分析
print("\n" + "="*80)
print("Key Insights")
print("="*80)

best_idx = np.argmin(costs)
worst_idx = np.argmax(costs)

print(f"[BEST] Best Performance (Lowest Cost):")
print(f"   - Arrival Rate: {arrival_rates[best_idx]:.1f} tasks/s")
print(f"   - Average Cost: {costs[best_idx]:.2f}")
print(f"   - Delay: {delays[best_idx]:.4f}s")
print(f"   - Energy: {energies[best_idx]:.2f}J")
print(f"   - Dropped Tasks: {dropped[best_idx]:.2f}")

print(f"\n[WORST] Worst Performance (Highest Cost):")
print(f"   - Arrival Rate: {arrival_rates[worst_idx]:.1f} tasks/s")
print(f"   - Average Cost: {costs[worst_idx]:.2f}")
print(f"   - Delay: {delays[worst_idx]:.4f}s")
print(f"   - Energy: {energies[worst_idx]:.2f}J")
print(f"   - Dropped Tasks: {dropped[worst_idx]:.2f}")

improvement = ((costs[worst_idx] - costs[best_idx]) / costs[worst_idx]) * 100
print(f"\n[IMPROVEMENT] Cost Reduction: {improvement:.1f}% from worst to best")

print("="*80)

# 生成成本组成分析
print("\n" + "="*80)
print("Cost Composition Analysis (Cost = 2.0×Delay + 1.2×Energy)")
print("="*80)
print(f"{'Arrival Rate':>15} | {'Delay Cost':>12} | {'Energy Cost':>13} | {'Total Cost':>12}")
print("-"*80)

for rate, delay, energy in zip(arrival_rates, delays, energies):
    delay_cost = 2.0 * delay
    energy_cost = 1.2 * energy
    total_cost_computed = delay_cost + energy_cost
    print(f"{rate:>15.1f} | {delay_cost:>12.2f} | {energy_cost:>13.2f} | {total_cost_computed:>12.2f}")

print("="*80)
print("Note: The above computed cost may differ slightly from ave_cost_per_step")
print("      because ave_cost_per_step includes per-step averaging effects.")
print("="*80)

print(f"\n[OK] Analysis complete! Plot saved to: {output_path}")

