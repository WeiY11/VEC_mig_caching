#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超简单测试 - 验证环境是否正常
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

print("="*60)
print("TD3消融实验环境 - 超简单测试")
print("="*60)

# 测试1: 配置加载
print("\n[1/3] 测试配置加载...")
try:
    from ablation_experiments.ablation_configs import get_all_ablation_configs
    configs = get_all_ablation_configs()
    print(f"    成功: 加载 {len(configs)} 个配置")
    for cfg in configs:
        print(f"      - {cfg.name}")
except Exception as e:
    print(f"    失败: {e}")
    sys.exit(1)

# 测试2: 训练环境创建
print("\n[2/3] 测试训练环境...")
try:
    from train_single_agent import SingleAgentTrainingEnvironment
    print("    正在创建TD3训练环境...")
    training_env = SingleAgentTrainingEnvironment("TD3")
    print(f"    成功: TD3训练环境创建正常")
except Exception as e:
    print(f"    失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试3: 运行一个超快速训练
print("\n[3/3] 测试训练流程（2轮）...")
try:
    from ablation_experiments.ablation_configs import get_config_by_name
    
    # 应用Full-System配置
    full_config = get_config_by_name('Full-System')
    full_config.apply_to_system()
    
    # 运行2轮训练
    for ep in [1, 2]:
        result = training_env.run_episode(ep, max_steps=20)  # 每轮只20步
        print(f"    Episode {ep}: Reward={result['avg_reward']:.3f}")
    
    print("    成功: 训练流程正常")
    
except Exception as e:
    print(f"    失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 全部通过
print("\n" + "="*60)
print("全部测试通过！")
print("="*60)
print("\n实验环境准备就绪，可以开始消融实验：")
print("  1. 快速测试：python run_ablation_td3.py --episodes 10")
print("  2. 标准实验：python run_ablation_td3.py --episodes 200")
print("="*60)

