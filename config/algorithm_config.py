#!/usr/bin/env python3
"""
算法配置
"""

from typing import Dict, Any

class AlgorithmConfig:
    """算法配置类"""
    
    def __init__(self):
        # MATD3配置
        self.matd3_config = {
            'actor_lr': 0.0003,
            'critic_lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,
            'policy_delay': 2,
            'noise_std': 0.05,      # 降低噪声
            'noise_clip': 0.3,      # 降低噪声裁剪
            'hidden_dim': 256,
            'batch_size': 128,
            'memory_size': 100000
        }
        
        # MADDPG配置
        self.maddpg_config = {
            'actor_lr': 0.0003,
            'critic_lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.01,
            'noise_std': 0.05,      # 降低噪声
            'hidden_dim': 256,
            'batch_size': 128,
            'memory_size': 100000
        }
        
        # 单智能体DDPG配置 - 🔧 深度优化版本（2025-09-30更新）
        self.ddpg_config = {
            'actor_lr': 3e-5,      # 🔧 优化：降低70%提高稳定性（原1e-4）
            'critic_lr': 1e-4,     # 🔧 优化：降低67%防止过拟合（原3e-4）
            'gamma': 0.99,
            'tau': 0.003,          # 🔧 优化：更稳定的软更新（原0.005）
            'noise_std': 0.15,     # 🔧 优化：降低初始噪声（原0.2）
            'noise_decay': 0.99995, # 🔧 新增：更慢的噪声衰减
            'min_noise': 0.05,     # 🔧 新增：最小噪声水平
            'hidden_dim': 256,
            'batch_size': 256,     # 🔧 优化：加倍批次大小（原64）
            'buffer_size': 200000, # 🔧 优化：加倍缓冲区（原50000）
            'memory_size': 200000, # 保持兼容性
            'warmup_steps': 2000,  # 🔧 新增：预热步数
            'update_freq': 2,      # 🔧 新增：更新频率
            # PER参数
            'use_per': True,       # 🔧 新增：启用优先经验回放
            'per_alpha': 0.6,
            'per_beta_start': 0.4,
            'gradient_clip': 0.5,  # 🔧 新增：梯度裁剪
            'reward_normalize': True  # 🔧 新增：奖励归一化
        }
        
        # PPO配置
        self.ppo_config = {
            'lr': 0.0003,
            'gamma': 0.99,
            'eps_clip': 0.2,
            'k_epochs': 4,
            'hidden_dim': 256,
            'batch_size': 64
        }
    
    def get_algorithm_config(self, algorithm: str) -> Dict[str, Any]:
        """获取指定算法的配置"""
        config_map = {
            'MATD3': self.matd3_config,
            'MADDPG': self.maddpg_config,
            'DDPG': self.ddpg_config,
            'PPO': self.ppo_config
        }
        
        return config_map.get(algorithm.upper(), {})
    
    def update_algorithm_config(self, algorithm: str, **kwargs):
        """更新算法配置"""
        config = self.get_algorithm_config(algorithm)
        config.update(kwargs)