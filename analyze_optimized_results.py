#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
分析OPTIMIZED_TD3优化后的训练结果 (500 Episodes)
对比优化前（800轮）与优化后的改进效果
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 读取训练结果
result_file = Path("results/single_agent/optimized_td3/training_results_20251202_020552.json")
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

rewards = data['episode_rewards']
num_episodes = len(rewards)

print("=" * 80)
print(f"OPTIMIZED_TD3 训练结果分析 (优化后 - {num_episodes} Episodes)")
print("=" * 80)

# 提取其他数据（如果有）
metrics = data.get('episode_metrics', {})
delays = metrics.get('avg_delay', [0] * num_episodes)
energies = metrics.get('avg_energy', [0] * num_episodes)
completions = metrics.get('completion_rate', [1.0] * num_episodes)

# 基本统计
print("\n【1. 基本统计】")
print(f"平均奖励:   {np.mean(rewards):.4f}")
print(f"标准差:     {np.std(rewards):.4f}")
print(f"变异系数:   {np.std(rewards)/abs(np.mean(rewards)):.4f}")
print(f"最小值:     {np.min(rewards):.4f}")
print(f"最大值:     {np.max(rewards):.4f}")
print(f"中位数:     {np.median(rewards):.4f}")

# 阶段性表现
print("\n【2. 阶段性表现分析】")
stages = {
    'P1 (0-125)': (0, 125),
    'P2 (125-250)': (125, 250),
    'P3 (250-375)': (250, 375),
    'P4 (375-500)': (375, 500)
}

for name, (start, end) in stages.items():
    stage_rewards = rewards[start:end]
    print(f"{name}: 均值={np.mean(stage_rewards):.4f}, "
          f"标准差={np.std(stage_rewards):.4f}")

# 前后期对比
early_100 = np.mean(rewards[:100])
late_100 = np.mean(rewards[-100:])
improvement = late_100 - early_100
improvement_pct = (improvement / abs(early_100)) * 100

print("\n【3. 收敛性分析】")
print(f"前100轮均值:  {early_100:.4f}")
print(f"后100轮均值:  {late_100:.4f}")
print(f"绝对改进:     {improvement:.4f}")
print(f"相对改进:     {improvement_pct:+.2f}%")

# 异常值分析
print("\n【4. 异常值分析】")
threshold_low = -1.0  # 优化后的异常阈值
abnormal_low = [r for r in rewards if r < threshold_low]
print(f"异常低奖励(<{threshold_low})数量: {len(abnormal_low)} ({len(abnormal_low)/num_episodes*100:.2f}%)")
if abnormal_low:
    print(f"异常值均值: {np.mean(abnormal_low):.4f}")
    print(f"最低值: {np.min(abnormal_low):.4f}")

# 系统指标统计
print("\n【5. 系统性能指标】")
if delays and delays[0] != 0:
    print(f"平均延迟:     {np.mean(delays):.4f}s")
    print(f"平均能耗:     {np.mean(energies):.2f}J")
    print(f"平均完成率:   {np.mean(completions)*100:.2f}%")
else:
    print("  系统指标数据未记录（仅记录了奖励值）")

# 与优化前对比
print("\n" + "=" * 80)
print("【对比分析: 优化前 vs 优化后】")
print("=" * 80)

print("\n优化前（800轮训练）:")
print("  - 平均奖励:   -1.8667")
print("  - 标准差:     0.5552")
print("  - 变异系数:   0.2974")
print("  - 前后期改进: -3.19%")
print("  - 异常值频率: 3.88%")

print(f"\n优化后（{num_episodes}轮训练）:")
print(f"  - 平均奖励:   {np.mean(rewards):.4f}")
print(f"  - 标准差:     {np.std(rewards):.4f}")
print(f"  - 变异系数:   {np.std(rewards)/abs(np.mean(rewards)):.4f}")
print(f"  - 前后期改进: {improvement_pct:+.2f}%")
print(f"  - 异常值频率: {len(abnormal_low)/num_episodes*100:.2f}%")

# 计算总改进
reward_improvement = ((-1.8667 - np.mean(rewards)) / -1.8667) * 100
cv_improvement = ((0.2974 - np.std(rewards)/abs(np.mean(rewards))) / 0.2974) * 100
convergence_improvement = improvement_pct - (-3.19)

print("\n【总体改进】")
print(f"奖励改进:     {reward_improvement:+.2f}%")
print(f"稳定性改进:   {cv_improvement:+.2f}% (变异系数降低)")
print(f"收敛性改进:   {convergence_improvement:+.2f}% (前后期改进提升)")

# 判断优化效果
print("\n" + "=" * 80)
print("【优化效果评估】")
print("=" * 80)

if reward_improvement > 50:
    print("✅ 奖励改进: 优秀 (>50%)")
elif reward_improvement > 30:
    print("✅ 奖励改进: 良好 (30-50%)")
else:
    print("⚠️ 奖励改进: 一般 (<30%)")

if improvement_pct > 10:
    print("✅ 收敛性: 优秀 (前后期改进>10%)")
elif improvement_pct > 5:
    print("✅ 收敛性: 良好 (前后期改进5-10%)")
else:
    print("⚠️ 收敛性: 需改进 (前后期改进<5%)")

if np.std(rewards)/abs(np.mean(rewards)) < 0.15:
    print("✅ 稳定性: 优秀 (CV<0.15)")
elif np.std(rewards)/abs(np.mean(rewards)) < 0.25:
    print("✅ 稳定性: 良好 (CV 0.15-0.25)")
else:
    print("⚠️ 稳定性: 需改进 (CV>0.25)")

# 关键优化点回顾
print("\n" + "=" * 80)
print("【应用的优化措施】")
print("=" * 80)
print("""
✅ 阶段1.1: 探索噪声优化
   - exploration_noise: 0.15 → 0.08
   - noise_decay: 0.998 → 0.995
   - min_noise: 0.02 → 0.01

✅ 阶段1.2: 权重优化
   - reward_weight_remote_reject: 0.5 → 0.15 (-70%)
   - reward_weight_cache: 0.4 → 0.2 (-50%)
   - reward_weight_cache_bonus: 0.5 → 0.3 (-40%)
   - reward_weight_completion_gap: 0.1 → 0.05 (-50%)
   - reward_clip_range: (-50, 0) → (-10, 0)

✅ 阶段2: 学习率与批量优化
   - actor_lr: 3e-5 → 6e-5 (提升2倍)
   - critic_lr: 8e-5 → 1.5e-4 (提升1.875倍)
   - batch_size: 768 → 512 (-33%)
   - warmup_steps: 2000 → 5000 (+150%)

✅ 阶段3: 归一化目标修复
   - latency_target: 0.4 → 2.0 (提升5倍)
   - energy_target: 3500 → 1500 (降低57%)
   - 修复奖励尺度爆炸问题（-60 → -3范围）
   - 启用update_reward_targets()确保全局单例同步
""")

print("=" * 80)
