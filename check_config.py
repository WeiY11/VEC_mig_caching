import json

with open('results/single_agent/optimized_td3/training_results_20251202_014158.json', 'r') as f:
    data = json.load(f)

cfg = data.get('system_config', {})

print("=" * 60)
print("训练配置检查")
print("=" * 60)

print("\n【奖励权重】")
print(f"  delay:           {cfg.get('reward_weight_delay')}")
print(f"  energy:          {cfg.get('reward_weight_energy')}")
print(f"  remote_reject:   {cfg.get('reward_weight_remote_reject')}")
print(f"  cache:           {cfg.get('reward_weight_cache')}")
print(f"  cache_bonus:     {cfg.get('reward_weight_cache_bonus')}")
print(f"  completion_gap:  {cfg.get('reward_weight_completion_gap')}")

print("\n【归一化目标】")
print(f"  latency_target:  {cfg.get('latency_target')}")
print(f"  energy_target:   {cfg.get('energy_target')}")
print(f"  use_dynamic_norm: {cfg.get('use_dynamic_reward_normalization')}")

print("\n【裁剪范围】")
print(f"  reward_clip_range: {cfg.get('reward_clip_range')}")

print("\n【训练配置】")
train_cfg = data.get('training_config', {})
print(f"  actor_lr:   {train_cfg.get('actor_lr')}")
print(f"  critic_lr:  {train_cfg.get('critic_lr')}")
print(f"  batch_size: {train_cfg.get('batch_size')}")
print(f"  warmup_steps: {train_cfg.get('warmup_steps')}")

print("\n【奖励统计】")
rewards = data['episode_rewards']
print(f"  前10轮: {rewards[:10]}")
print(f"  后10轮: {rewards[-10:]}")
