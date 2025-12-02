#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析奖励组成，诊断权重校准问题
"""

import json
import numpy as np
from pathlib import Path

# 读取最新训练结果
result_file = Path("results/single_agent/optimized_td3/training_results_20251202_020552.json")
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

rewards = np.array(data['episode_rewards'])
metrics = data.get('episode_metrics', {})
delays = np.array(metrics.get('avg_delay', []))
# 能耗数据可能在metrics中也可能直接在根节点
energies = np.array(metrics.get('avg_energy', metrics.get('total_energy', data.get('episode_metrics', {}).get('total_energy', []))))
completions = np.array(metrics.get('completion_rate', []))

print("=" * 80)
print("奖励组成分析 - 诊断权重校准问题")
print("=" * 80)

# 当前配置（从修复后的配置）
latency_target = 2.0
energy_target = 1500.0
weight_delay = 0.5
weight_energy = 0.5

print("\n【当前配置】")
print(f"归一化目标: latency_target={latency_target}s, energy_target={energy_target}J")
print(f"核心权重: weight_delay={weight_delay}, weight_energy={weight_energy}")

# 系统性能统计
print("\n【系统性能统计】")
print(f"平均延迟:   {np.mean(delays):.4f}s (min={np.min(delays):.4f}, max={np.max(delays):.4f})")
if len(energies) > 0 and np.sum(energies) > 0:
    print(f"平均能耗:   {np.mean(energies):.2f}J (min={np.min(energies):.2f}, max={np.max(energies):.2f})")
else:
    print(f"平均能耗:   数据缺失")
print(f"平均完成率: {np.mean(completions)*100:.2f}%")

# 计算归一化后的成本（模拟奖励计算）
print("\n【归一化分析】")

# 延迟归一化
norm_delays = np.maximum(0, (delays - latency_target) / latency_target)
avg_norm_delay = np.mean(norm_delays)
print(f"归一化延迟: 均值={avg_norm_delay:.4f}")
print(f"  - 低于目标(<{latency_target}s): {np.sum(delays < latency_target)} episodes ({np.sum(delays < latency_target)/len(delays)*100:.1f}%)")
print(f"  - 高于目标(>{latency_target}s): {np.sum(delays >= latency_target)} episodes ({np.sum(delays >= latency_target)/len(delays)*100:.1f}%)")

# 能耗归一化（如果有数据）
if len(energies) > 0 and np.sum(energies) > 0:
    norm_energies = np.maximum(0, (energies - energy_target) / energy_target)
    avg_norm_energy = np.mean(norm_energies)
    print(f"\n归一化能耗: 均值={avg_norm_energy:.4f}")
    print(f"  - 低于目标(<{energy_target}J): {np.sum(energies < energy_target)} episodes ({np.sum(energies < energy_target)/len(energies)*100:.1f}%)")
    print(f"  - 高于目标(>{energy_target}J): {np.sum(energies >= energy_target)} episodes ({np.sum(energies >= energy_target)/len(energies)*100:.1f}%)")
else:
    print("\n能耗数据缺失（显示为0）")
    norm_energies = np.zeros_like(delays)
    avg_norm_energy = 0.0

# 估算延迟成本（使用tanh平滑）
delay_costs = np.tanh(norm_delays) * weight_delay
avg_delay_cost = np.mean(delay_costs)
print(f"\n延迟成本: 均值={avg_delay_cost:.4f} (权重={weight_delay})")

# 估算能耗成本
if len(energies) > 0 and np.sum(energies) > 0:
    energy_costs = np.tanh(norm_energies) * weight_energy
    avg_energy_cost = np.mean(energy_costs)
    print(f"能耗成本: 均值={avg_energy_cost:.4f} (权重={weight_energy})")
else:
    energy_costs = np.zeros_like(delays)
    avg_energy_cost = 0.0
    print(f"能耗成本: 数据缺失")

# 估算完成率惩罚
completion_gaps = 1.0 - completions
avg_completion_gap = np.mean(completion_gaps)
print(f"完成率缺口: 均值={avg_completion_gap:.4f} ({np.mean(completions)*100:.2f}%完成)")

# 估算总成本
estimated_costs = delay_costs + energy_costs
avg_estimated_cost = np.mean(estimated_costs)
print(f"\n估算基础成本: 均值={avg_estimated_cost:.4f}")
print(f"实际奖励均值: {np.mean(rewards):.4f}")
print(f"差值: {np.mean(rewards) - (-avg_estimated_cost):.4f}")

# 问题诊断
print("\n" + "=" * 80)
print("【问题诊断】")
print("=" * 80)

# 检查1: 延迟成本过高
if avg_norm_delay > 0.5:
    print(f"\n⚠️ 问题1: 延迟归一化过高 ({avg_norm_delay:.4f})")
    print(f"   原因: 实际延迟({np.mean(delays):.2f}s) 显著高于目标({latency_target}s)")
    print(f"   建议: 提高latency_target到 {np.mean(delays)*1.2:.2f}s")

# 检查2: 能耗成本
if avg_norm_energy > 0.5:
    print(f"\n⚠️ 问题2: 能耗归一化过高 ({avg_norm_energy:.4f})")
    print(f"   原因: 实际能耗({np.mean(energies):.0f}J) 显著高于目标({energy_target}J)")
    print(f"   建议: 提高energy_target到 {np.mean(energies)*1.2:.0f}J")

# 检查3: 奖励范围
reward_range = np.max(rewards) - np.min(rewards)
print(f"\n⚠️ 问题3: 奖励范围过大 ({reward_range:.2f})")
print(f"   最小值: {np.min(rewards):.4f}")
print(f"   最大值: {np.max(rewards):.4f}")
if reward_range > 5.0:
    print(f"   建议: 收紧reward_clip_range或降低权重")

# 检查4: 与优化前对比
baseline_reward = -1.87
current_reward = np.mean(rewards)
print(f"\n⚠️ 问题4: 性能不如优化前")
print(f"   优化前: {baseline_reward:.4f}")
print(f"   当前: {current_reward:.4f}")
print(f"   退化: {(current_reward/baseline_reward - 1)*100:.2f}%")

# 权重校准建议
print("\n" + "=" * 80)
print("【权重校准建议】")
print("=" * 80)

# 建议1: 调整归一化目标
suggested_latency_target = np.percentile(delays, 60)  # 60分位数
suggested_energy_target = np.percentile(energies, 60) if (len(energies) > 0 and np.sum(energies) > 0) else 1500.0

print(f"\n✅ 建议1: 调整归一化目标（让60%的episode低于目标）")
print(f"   latency_target: {latency_target:.2f} → {suggested_latency_target:.2f}s")
print(f"   energy_target: {energy_target:.0f} → {suggested_energy_target:.0f}J")

# 建议2: 调整权重平衡
if avg_delay_cost > avg_energy_cost * 2:
    print(f"\n✅ 建议2: 降低延迟权重（延迟成本{avg_delay_cost:.4f} >> 能耗成本{avg_energy_cost:.4f}）")
    suggested_weight_delay = 0.3
    suggested_weight_energy = 0.7
    print(f"   weight_delay: {weight_delay} → {suggested_weight_delay}")
    print(f"   weight_energy: {weight_energy} → {suggested_weight_energy}")
elif avg_energy_cost > avg_delay_cost * 2:
    print(f"\n✅ 建议2: 降低能耗权重（能耗成本{avg_energy_cost:.4f} >> 延迟成本{avg_delay_cost:.4f}）")
    suggested_weight_delay = 0.7
    suggested_weight_energy = 0.3
    print(f"   weight_delay: {weight_delay} → {suggested_weight_delay}")
    print(f"   weight_energy: {weight_energy} → {suggested_weight_energy}")
else:
    print(f"\n✅ 建议2: 权重平衡合理")
    print(f"   延迟成本: {avg_delay_cost:.4f}")
    print(f"   能耗成本: {avg_energy_cost:.4f}")

# 建议3: 检查其他惩罚项
print(f"\n✅ 建议3: 检查其他惩罚项")
print(f"   完成率缺口: {avg_completion_gap:.4f} (如果>0.01需要检查)")
print(f"   估算成本-实际奖励差值: {abs(np.mean(rewards) - (-avg_estimated_cost)):.4f}")
if abs(np.mean(rewards) - (-avg_estimated_cost)) > 1.0:
    print(f"   ⚠️ 差值过大，可能存在其他未考虑的惩罚项")
    print(f"   建议检查: cache惩罚、queue惩罚、remote_reject惩罚")

print("\n" + "=" * 80)
