#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的TD3算法 - 针对VEC系统优化

核心改进（相比CAMTD3）：
1. 移除启发式融合机制（实验证明干扰学习）
2. 优化探索策略（adaptive noise）
3. 增强奖励塑形（progress-based reward shaping）
4. 改进网络结构（更适合VEC任务）

性能目标：
- 时延 < 0.20s（比Random的0.56s提升60%+）
- 能耗 < 6500J（比Random的6763J降低5%+）
- 完成率 > 98%（比Random的99.1%持平或略高）
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Optional
from single_agent.td3 import TD3Environment, TD3Config


class ImprovedTD3Config(TD3Config):
    """改进的TD3配置"""
    
    def __init__(self):
        super().__init__()
        # 优化学习率（更激进的学习）
        self.actor_lr = 2e-4  # 提高至2e-4
        self.critic_lr = 3e-4  # 提高至3e-4
        
        # 优化探索参数（自适应探索）
        self.exploration_noise = 0.2  # 初始探索噪声
        self.noise_decay = 0.9995  # 缓慢衰减
        self.min_noise = 0.08  # 保持适度探索
        
        # 优化TD3参数
        self.policy_delay = 2  # 标准延迟
        self.tau = 0.005  # 标准软更新
        
        # 优化训练参数
        self.batch_size = 256
        self.warmup_steps = 2000  # 减少预热时间
        
        # 启用进度奖励塑形
        self.use_progress_shaping = True
        self.progress_alpha = 0.1  # 进度奖励权重


class ImprovedTD3Environment(TD3Environment):
    """改进的TD3环境 - 移除启发式融合，专注优化学习"""
    
    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2):
        super().__init__(num_vehicles, num_rsus, num_uavs)
        self.algorithm_label = "Improved-TD3"
        
        # 使用改进的配置
        self.config = ImprovedTD3Config()
        
        # 进度追踪（用于奖励塑形）
        self.best_delay = float('inf')
        self.best_energy = float('inf')
        self.episode_count = 0
        
        print(f"\n🚀 Improved TD3 已启用")
        print(f"   核心改进:")
        print(f"   ✓ 移除启发式融合（避免干扰学习）")
        print(f"   ✓ 自适应探索噪声（更好的探索-利用平衡）")
        print(f"   ✓ 进度奖励塑形（加速收敛）")
        print(f"   ✓ 优化超参数（更快学习）\n")
    
    def calculate_reward(self, system_metrics: Dict, 
                        cache_metrics: Optional[Dict] = None,
                        migration_metrics: Optional[Dict] = None) -> float:
        """
        改进的奖励函数 - 添加进度奖励塑形
        
        核心思想：
        1. 基础奖励：统一奖励计算器（确保一致性）
        2. 进度奖励：鼓励持续改进（避免早期振荡）
        """
        from utils.unified_reward_calculator import calculate_unified_reward
        
        # ========== 1. 基础奖励 ==========
        base_reward = calculate_unified_reward(
            system_metrics, 
            cache_metrics, 
            migration_metrics, 
            algorithm="general"
        )
        
        # ========== 2. 进度奖励塑形（可选）==========
        if not self.config.use_progress_shaping:
            return base_reward
        
        # 提取当前性能
        current_delay = max(0.0, float(system_metrics.get('avg_task_delay', 0)))
        current_energy = max(0.0, float(system_metrics.get('total_energy_consumption', 0)))
        
        # 计算进度奖励（相对于历史最佳）
        progress_reward = 0.0
        
        if current_delay < self.best_delay:
            # 时延改进
            improvement = (self.best_delay - current_delay) / max(self.best_delay, 0.1)
            progress_reward += improvement * 5.0  # 时延改进奖励
            self.best_delay = current_delay
        
        if current_energy < self.best_energy:
            # 能耗改进
            improvement = (self.best_energy - current_energy) / max(self.best_energy, 1000.0)
            progress_reward += improvement * 3.0  # 能耗改进奖励
            self.best_energy = current_energy
        
        # ========== 3. 最终奖励 ==========
        final_reward = base_reward + self.config.progress_alpha * progress_reward
        
        # 裁剪到合理范围
        final_reward = np.clip(final_reward, -30.0, 5.0)
        
        return final_reward
    
    def reset_episode(self):
        """Episode重置时调用"""
        self.episode_count += 1
        
        # 每50个episode重置最佳记录（允许重新探索）
        if self.episode_count % 50 == 0:
            self.best_delay = float('inf')
            self.best_energy = float('inf')
            print(f"   [Episode {self.episode_count}] 重置最佳记录，鼓励新探索")

