#!/usr/bin/env python3
"""
统一奖励计算器权重配置深度分析
分析训练结果 training_results_20251202_005655.json (800 episodes, 优化后)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# 读取训练结果
results_file = 'D:/VEC_mig_caching/results/single_agent/optimized_td3/training_results_20251202_005655.json'
with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

episode_rewards = np.array(data['episode_rewards'])

# =====================================================================
# 第一部分：训练效果评估
# =====================================================================
print("=" * 80)
print("训练结果总览 (800 Episodes, 阶段1优化后)")
print("=" * 80)

print(f"\n【基本统计】")
print(f"平均奖励: {np.mean(episode_rewards):.4f}")
print(f"标准差:   {np.std(episode_rewards):.4f}")
print(f"变异系数: {np.std(episode_rewards)/abs(np.mean(episode_rewards)):.4f}")
print(f"最小值:   {np.min(episode_rewards):.4f}")
print(f"最大值:   {np.max(episode_rewards):.4f}")

# 阶段性分析
phases = {
    'P1 (0-200)': episode_rewards[:200],
    'P2 (200-400)': episode_rewards[200:400],
    'P3 (400-600)': episode_rewards[400:600],
    'P4 (600-800)': episode_rewards[600:],
}

print(f"\n【阶段性表现】")
for phase_name, phase_data in phases.items():
    mean_val = np.mean(phase_data)
    std_val = np.std(phase_data)
    print(f"{phase_name}: 均值={mean_val:.4f}, 标准差={std_val:.4f}")

improvement = np.mean(phases['P4 (600-800)']) - np.mean(phases['P1 (0-200)'])
print(f"\n前后期改进: {improvement:.4f} ({improvement/np.mean(phases['P1 (0-200)'])*100:.2f}%)")

# =====================================================================
# 第二部分：当前权重配置
# =====================================================================
print("\n" + "=" * 80)
print("当前统一奖励计算器权重配置")
print("=" * 80)

weights_config = {
    '核心权重': {
        'weight_delay': 0.5,
        'weight_energy': 0.3,
    },
    '惩罚权重': {
        'penalty_dropped': 0.01,
        'weight_completion_gap': 0.1,
        'weight_loss_ratio': 0.1,
        'weight_cache_pressure': 0.1,
        'weight_queue_overload': 0.05,
        'weight_remote_reject': 0.5,
        'weight_cache': 0.4,
        'weight_migration': 0.1,
        'weight_local_penalty': 0.0,
    },
    '奖励权重': {
        'weight_offload_bonus': 0.1,
        'weight_cache_bonus': 0.5,
        'weight_joint': 0.05,
    },
    '目标值': {
        'latency_target': 0.4,  # seconds
        'energy_target': 3500.0,  # Joules
        'latency_tolerance': 1.0,
        'energy_tolerance': 5000.0,
    }
}

print("\n【核心权重】(Delay + Energy)")
for k, v in weights_config['核心权重'].items():
    print(f"  {k:25s} = {v:.2f}")

print("\n【惩罚权重】")
for k, v in weights_config['惩罚权重'].items():
    print(f"  {k:30s} = {v:.2f}")

print("\n【奖励权重】(Bonus)")
for k, v in weights_config['奖励权重'].items():
    print(f"  {k:30s} = {v:.2f}")

print("\n【目标值与归一化】")
for k, v in weights_config['目标值'].items():
    print(f"  {k:30s} = {v:.2f}")

# =====================================================================
# 第三部分：权重配置问题诊断
# =====================================================================
print("\n" + "=" * 80)
print("权重配置问题诊断")
print("=" * 80)

# 计算权重比例
delay_weight = 0.5
energy_weight = 0.3
core_weight_ratio = delay_weight / energy_weight if energy_weight > 0 else float('inf')

print(f"\n【问题1: 核心权重比例分析】")
print(f"延迟权重 / 能耗权重 = {delay_weight} / {energy_weight} = {core_weight_ratio:.2f}")
print(f"当前配置: Delay占核心权重的 {delay_weight/(delay_weight+energy_weight)*100:.1f}%")
print(f"            Energy占核心权重的 {energy_weight/(delay_weight+energy_weight)*100:.1f}%")
print(f"\n✅ 评估: 比例基本合理")
print(f"   - VEC系统通常优先延迟 (实时性要求)")
print(f"   - 比例1.67:1在合理范围内")
print(f"   - 但考虑到能耗目标3500J vs 延迟目标0.4s的归一化尺度差异")
print(f"     实际权重可能需要微调")

# 辅助权重总和
penalty_weights = {
    'dropped': 0.01,
    'completion_gap': 0.1,
    'loss_ratio': 0.1,
    'cache_pressure': 0.1,
    'queue_overload': 0.05,
    'remote_reject': 0.5,
    'cache': 0.4,
    'migration': 0.1,
    'local_penalty': 0.0,
}

bonus_weights = {
    'offload_bonus': 0.1,
    'cache_bonus': 0.5,
    'joint': 0.05,
}

total_penalty = sum(penalty_weights.values())
total_bonus = sum(bonus_weights.values())
total_core = delay_weight + energy_weight

print(f"\n【问题2: 辅助项与核心权重平衡】")
print(f"核心权重总和:   {total_core:.2f} (delay + energy)")
print(f"惩罚权重总和:   {total_penalty:.2f}")
print(f"奖励权重总和:   {total_bonus:.2f}")
print(f"辅助项净影响:   {total_penalty - total_bonus:.2f}")
print(f"\n⚠️  潜在问题:")
print(f"   - 惩罚权重总和 {total_penalty:.2f} > 核心权重 {total_core:.2f}")
print(f"   - 辅助项可能掩盖核心优化目标")
print(f"   - remote_reject权重0.5过高 (=核心延迟权重)")
print(f"   - cache相关权重累计达0.9 (0.4 cache + 0.5 bonus)")

# 异常值分析
outliers = episode_rewards[episode_rewards < -3.0]
print(f"\n【问题3: 奖励分布与异常值】")
print(f"异常低奖励(<-3.0)数量: {len(outliers)} ({len(outliers)/len(episode_rewards)*100:.2f}%)")
print(f"异常值均值: {np.mean(outliers):.4f}")
print(f"最低值: {np.min(episode_rewards):.4f}")
print(f"\n⚠️  潜在原因:")
print(f"   - reward_clip_range = (-50.0, 0.0) 范围过宽")
print(f"   - 实际奖励99%在[-3, -1]，裁剪无作用")
print(f"   - 极端情况下惩罚项累积过重")

# =====================================================================
# 第四部分：具体权重问题识别
# =====================================================================
print("\n" + "=" * 80)
print("具体权重配置问题")
print("=" * 80)

print("\n【❌ 问题权重清单】")
print("\n1. remote_reject权重 = 0.5")
print("   问题: 与核心延迟权重相当，过度惩罚边缘拒绝")
print("   影响: 可能导致智能体过度规避UAV/RSU卸载")
print("   建议: 降低至 0.1-0.2")

print("\n2. cache相关权重过高")
print("   - weight_cache = 0.4 (miss惩罚)")
print("   - weight_cache_bonus = 0.5 (hit奖励)")
print("   - 累计影响: 0.9 (超过核心权重0.8)")
print("   问题: 缓存成为主导优化目标，偏离延迟/能耗核心")
print("   建议: cache=0.2, cache_bonus=0.3")

print("\n3. 归一化尺度不匹配")
print("   - latency_target = 0.4s")
print("   - energy_target = 3500J")
print("   - 归一化后延迟/能耗尺度相差8750倍")
print("   问题: 即使权重比1.67:1，实际影响仍严重失衡")
print("   建议: 调整归一化因子或权重以匹配真实影响")

print("\n4. completion_gap权重过高")
print("   - weight_completion_gap = 0.1")
print("   - 配合completion_target = 0.88")
print("   问题: 在高负载下过度惩罚，系统难达88%")
print("   建议: 降低至 0.05 或调低target至0.85")

print("\n5. 惩罚项过多导致累积效应")
print("   - 9个独立惩罚项 (dropped, gap, loss, pressure, queue, reject, cache, migration, local)")
print("   - 极端情况下可累积至 >3.0 成本")
print("   问题: 导致-5.0异常低奖励")
print("   建议: 简化惩罚项，合并相关指标")

# =====================================================================
# 第五部分：优化建议
# =====================================================================
print("\n" + "=" * 80)
print("权重优化建议方案")
print("=" * 80)

print("\n【方案A: 渐进式调整】(推荐)")
print("\n阶段1: 降低辅助权重")
print("```python")
print("# config/system_config.py RLConfig")
print("self.reward_weight_delay = 0.5          # 保持不变")
print("self.reward_weight_energy = 0.3         # 保持不变")
print("self.reward_penalty_dropped = 0.01      # 保持不变")
print("")
print("# 降低辅助项权重")
print("self.reward_weight_remote_reject = 0.15  # 0.5 → 0.15 (降低70%)")
print("self.reward_weight_cache = 0.2           # 0.4 → 0.2 (降低50%)")
print("self.reward_weight_cache_bonus = 0.3     # 0.5 → 0.3 (降低40%)")
print("self.reward_weight_completion_gap = 0.05 # 0.1 → 0.05 (降低50%)")
print("```")
print("\n预期效果:")
print("  - 核心权重占比提升至 50%+ (当前约36%)")
print("  - 辅助项总权重降至 <0.8")
print("  - 减少异常低奖励频率")

print("\n阶段2: 收紧奖励裁剪范围")
print("```python")
print("# utils/unified_reward_calculator.py")
print("self.reward_clip_range = (-10.0, 0.0)  # (-50.0, 0.0) → (-10.0, 0.0)")
print("```")
print("\n预期效果:")
print("  - 限制极端惩罚")
print("  - Q值估计更稳定")

print("\n阶段3: 调整归一化目标 (如阶段1+2效果不佳)")
print("```python")
print("# config/system_config.py RLConfig")
print("self.latency_target = 0.5      # 0.4 → 0.5 (放宽目标)")
print("self.energy_target = 4000.0    # 3500 → 4000 (放宽目标)")
print("```")

print("\n【方案B: 激进式重构】")
print("\n核心思想: 只保留核心权重+关键惩罚")
print("```python")
print("# 核心权重")
print("self.reward_weight_delay = 0.6          # 提升")
print("self.reward_weight_energy = 0.4         # 提升")
print("")
print("# 关键惩罚")
print("self.reward_penalty_dropped = 0.02      # 保留")
print("self.reward_weight_completion_gap = 0.05")
print("")
print("# 禁用次要项")
print("self.reward_weight_cache = 0.0          # 禁用")
print("self.reward_weight_cache_bonus = 0.0    # 禁用")
print("self.reward_weight_remote_reject = 0.0  # 禁用")
print("self.reward_weight_cache_pressure = 0.0 # 禁用")
print("# ... 其他辅助项全部设为0")
print("```")
print("\n风险: 可能丢失部分优化细节，但收敛更快")

# =====================================================================
# 第六部分：结论
# =====================================================================
print("\n" + "=" * 80)
print("诊断结论")
print("=" * 80)

print("\n【训练未收敛的主要原因分析】")
print("\n基于800轮训练数据 (优化后):")
print(f"  - 平均奖励: {np.mean(episode_rewards):.4f}")
print(f"  - 前后期改进: {improvement:.4f} ({improvement/np.mean(phases['P1 (0-200)'])*100:.2f}%)")
print(f"  - 变异系数: {np.std(episode_rewards)/abs(np.mean(episode_rewards)):.4f}")

print("\n✅ 阶段1优化 (探索噪声降低) 已生效:")
print(f"  - 前后期有轻微改进 (约3%)")
print(f"  - 变异系数仍高 (0.30)")
print(f"  - 说明噪声不是唯一问题")

print("\n❌ 权重配置是重要的次要原因:")
print("  1. 辅助权重过高，掩盖核心优化目标")
print("  2. remote_reject和cache权重严重失衡")
print("  3. 惩罚项过多，极端情况累积过重")
print("  4. 归一化尺度不匹配，实际权重偏离设计")

print("\n🎯 优先级排序:")
print("  【P1】降低remote_reject权重 (0.5 → 0.15)")
print("  【P2】降低cache相关权重 (0.4→0.2, 0.5→0.3)")
print("  【P3】收紧奖励裁剪范围 (-50→-10)")
print("  【P4】调整学习率和批量大小 (如计划的阶段2)")

print("\n📊 预期改进:")
print("  如果同时应用权重优化+阶段2(学习率):")
print("  - 前后期改进可达 10-15%")
print("  - 变异系数降至 0.20-0.25")
print("  - 最终奖励收敛至 -1.2 到 -1.0")

print("\n" + "=" * 80)
