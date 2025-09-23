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
            'noise_std': 0.1,
            'noise_clip': 0.5,
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
            'noise_std': 0.1,
            'hidden_dim': 256,
            'batch_size': 128,
            'memory_size': 100000
        }
        
        # 单智能体DDPG配置
        self.ddpg_config = {
            'actor_lr': 0.0003,
            'critic_lr': 0.0003,
            'gamma': 0.99,
            'tau': 0.005,  # 统一为0.005，与实现一致
            'noise_std': 0.2,
            'hidden_dim': 256,
            'batch_size': 64,
            'memory_size': 50000
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