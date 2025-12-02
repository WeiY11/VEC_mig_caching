"""快速检查奖励权重配置"""
from config.system_config import config

print("=" * 60)
print("🔍 奖励权重配置检查")
print("=" * 60)

# 核心权重
print("\n【核心权重】")
print(f"  weight_delay  = {config.rl.reward_weight_delay}")
print(f"  weight_energy = {config.rl.reward_weight_energy}")
print(f"  penalty_dropped = {config.rl.reward_penalty_dropped}")

# Bonus权重
print("\n【Bonus权重】")
print(f"  weight_offload_bonus = {config.rl.reward_weight_offload_bonus}")
print(f"  weight_cache_bonus   = {config.rl.reward_weight_cache_bonus}")
print(f"  weight_joint         = {config.rl.reward_weight_joint}")
print(f"  weight_local_penalty = {config.rl.reward_weight_local_penalty}")

# 目标值
print("\n【目标值】")
print(f"  latency_target = {config.rl.latency_target}s")
print(f"  energy_target  = {config.rl.energy_target}J")

# 归一化配置
print("\n【归一化配置】")
if hasattr(config, 'normalization'):
    print(f"  delay_normalizer_value  = {config.normalization.delay_normalizer_value}")
    print(f"  energy_normalizer_value = {config.normalization.energy_normalizer_value}")
else:
    print("  [未找到normalization配置]")

print("=" * 60)

# 模拟奖励计算
print("\n【模拟奖励计算】")
delay = 1.513
energy = 919.9
norm_delay = delay / 1.5
norm_energy = energy / 900
core_cost = 1.0 * norm_delay + 1.0 * norm_energy

# 假设RSU卸载率50%，UAV卸载率30%，本地20%
offload_bonus = 0.5 * (1.5 * 0.5 + 0.8 * 0.3 - 0.2)  # 使用默认值0.5
import numpy as np
offload_bonus_clipped = np.clip(offload_bonus, -1.0, 1.0)

total_cost_with_bonus = core_cost - offload_bonus_clipped
total_cost_without_bonus = core_cost

reward_with_bonus = -total_cost_with_bonus
reward_without_bonus = -total_cost_without_bonus

print(f"  假设：delay={delay}s, energy={energy}J")
print(f"  假设：RSU=50%, UAV=30%, Local=20%")
print(f"  ")
print(f"  norm_delay  = {norm_delay:.4f}")
print(f"  norm_energy = {norm_energy:.4f}")
print(f"  core_cost   = {core_cost:.4f}")
print(f"  ")
print(f"  offload_bonus (weight=0.5) = {offload_bonus_clipped:.4f}")
print(f"  ")
print(f"  total_cost (有bonus)  = {total_cost_with_bonus:.4f}")
print(f"  total_cost (无bonus)  = {total_cost_without_bonus:.4f}")
print(f"  ")
print(f"  reward (有bonus)  = {reward_with_bonus:.4f}")
print(f"  reward (无bonus)  = {reward_without_bonus:.4f}")
print(f"  ")
print(f"  差距 = {abs(reward_with_bonus - reward_without_bonus):.4f}")
print(f"  倍数 = {abs(reward_without_bonus / reward_with_bonus):.2f}x")

print("=" * 60)
