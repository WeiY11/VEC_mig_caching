#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM-TD3优化配置
解决奖励无法收敛的问题

关键优化：
1. 大幅降低启发式融合权重，让RL主导学习
2. 增强探索能力，提高噪声参数
3. 放宽梯度裁剪，加速参数更新
4. 降低保守正则强度，减少过度约束
5. 提高学习率，加快策略优化
"""
import os

# ========== CAM-TD3 特定配置 ==========

# 融合策略优化：完全禁用启发式融合，让RL完全主导
os.environ['CAM_FUSION_TRAIN_WEIGHT'] = '0.0'  # 完全禁用训练融合
os.environ['CAM_FUSION_EVAL_WEIGHT'] = '0.0'   # 完全禁用评估融合

# ========== TD3 核心参数优化 ==========

# 探索参数
os.environ['TD3_EXPLORATION_NOISE'] = '0.20'    # 初始探索噪声：20%
os.environ['TD3_NOISE_DECAY'] = '0.9995'        # 噪声衰减：更缓慢
os.environ['TD3_MIN_NOISE'] = '0.05'            # 最小噪声：5%

# 学习率优化
os.environ['TD3_ACTOR_LR'] = '2e-4'             # Actor学习率：2e-4
os.environ['TD3_CRITIC_LR'] = '3e-4'            # Critic学习率：3e-4

# 梯度裁剪
os.environ['TD3_GRADIENT_CLIP'] = '1.0'         # 放宽到1.0

# 保守正则
os.environ['TD3_CQL_ALPHA'] = '0.08'            # CQL正则：0.08
os.environ['TD3_UNCERTAINTY_WEIGHT'] = '0.02'   # 不确定性权重：0.02

# 启发式融合（禁用）
os.environ['TD3_HEURISTIC_BLEND'] = '0'         # 禁用启发式融合
os.environ['TD3_BLEND_RATIO'] = '0.0'           # 融合比例：0

# 训练参数
os.environ['TD3_BATCH_SIZE'] = '256'            # 批大小
os.environ['TD3_POLICY_DELAY'] = '2'            # 策略延迟
os.environ['TD3_TAU'] = '0.005'                 # 软更新系数

# 中央资源模式（必须启用）
os.environ['CENTRAL_RESOURCE'] = '1'

print("✅ CAM-TD3优化配置已加载")
print(f"   - 训练融合权重: {os.environ['CAM_FUSION_TRAIN_WEIGHT']}")
print(f"   - 探索噪声: {os.environ['TD3_EXPLORATION_NOISE']}")
print(f"   - Actor学习率: {os.environ['TD3_ACTOR_LR']}")
print(f"   - 梯度裁剪: {os.environ['TD3_GRADIENT_CLIP']}")
print(f"   - CQL正则: {os.environ['TD3_CQL_ALPHA']}")
print(f"   - 启发式融合: {'禁用' if os.environ['TD3_HEURISTIC_BLEND'] == '0' else '启用'}")
