#!/usr/bin/env python3
"""检查训练结果中的NaN值"""

import json
import numpy as np
from pathlib import Path

# 加载最新的训练结果
result_file = Path('results/single_agent/td3/training_results_20251122_041137.json')

if not result_file.exists():
    print(f"❌ 文件不存在: {result_file}")
    exit(1)

print(f"📖 加载训练结果: {result_file}")
with open(result_file, 'r') as f:
    data = json.load(f)

# 检查episode_rewards
if 'episode_rewards' not in data:
    print("❌ 训练结果中没有 episode_rewards")
    exit(1)

rewards = data['episode_rewards']
print(f"\n📊 奖励数据统计:")
print(f"  总Episode数: {len(rewards)}")

# 检查NaN值
nan_count = sum(1 for r in rewards if not np.isfinite(r))
print(f"  NaN/Inf数量: {nan_count}")

if nan_count > 0:
    nan_indices = [i for i, r in enumerate(rewards) if not np.isfinite(r)]
    print(f"\n⚠️ 发现 {nan_count} 个NaN/Inf值")
    print(f"  首次出现在Episode: {nan_indices[0] + 1}")
    print(f"  前20个NaN位置: {[i+1 for i in nan_indices[:20]]}")
    
    # 显示NaN前后的正常值
    first_nan = nan_indices[0]
    print(f"\n  Episode {first_nan} (NaN之前): {rewards[first_nan-1] if first_nan > 0 else 'N/A'}")
    print(f"  Episode {first_nan+1} (NaN): {rewards[first_nan]}")
    print(f"  Episode {first_nan+2} (NaN之后): {rewards[first_nan+1] if first_nan < len(rewards)-1 else 'N/A'}")
else:
    print("\n✅ 没有发现NaN/Inf值")

# 显示奖励范围
finite_rewards = [r for r in rewards if np.isfinite(r)]
if finite_rewards:
    print(f"\n📈 有效奖励统计:")
    print(f"  最小值: {min(finite_rewards):.4f}")
    print(f"  最大值: {max(finite_rewards):.4f}")
    print(f"  平均值: {np.mean(finite_rewards):.4f}")
    print(f"  中位数: {np.median(finite_rewards):.4f}")
    
    # 显示前10和后10个episode
    print(f"\n  前10个episode奖励: {[f'{r:.4f}' for r in rewards[:10]]}")
    print(f"  后10个episode奖励: {[f'{r:.4f}' if np.isfinite(r) else 'NaN' for r in rewards[-10:]]}")

print("\n" + "=" * 60)
