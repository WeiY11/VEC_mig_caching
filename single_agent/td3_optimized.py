"""
优化的TD3算法实现 - 核心网络组件

本模块提供TD3算法的核心网络结构：
- OptimizedTD3Config: TD3超参数配置
- OptimizedTD3Actor: Actor网络（策略网络）
- OptimizedTD3Critic: Twin Critic网络（Q值网络）

注意：
- 状态空间和动作空间定义请参考 common_state_action.py
- 完整的训练环境请使用 optimized_td3_wrapper.py 中的 OptimizedTD3Wrapper
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class OptimizedTD3Config:
    """优化的TD3配置 - 修复版本"""
    # 网络结构
    hidden_dim: int = 256        # 适当增加网络容量以学习复杂策略
    actor_lr: float = 3e-4       # 提高学习率，加速收敛
    critic_lr: float = 3e-4      # 提高学习率，加速收敛
    
    # 训练参数
    batch_size: int = 256        # 适中批次大小
    buffer_size: int = 500000    # 适中缓冲区
    tau: float = 0.005           # 标准软更新速率
    gamma: float = 0.99          # 标准折扣因子
    
    # TD3特有参数
    policy_delay: int = 2        # 标准策略延迟
    target_noise: float = 0.2    # 标准目标噪声
    noise_clip: float = 0.5      # 标准噪声裁剪
    
    # 探索参数
    exploration_noise: float = 0.25   # 适中的初始探索
    noise_decay: float = 0.9998       # 更缓慢的衰减速度
    min_noise: float = 0.08           # 保留更多探索
    
    # 训练控制
    warmup_steps: int = 10000        # 减少预热步数，约50个episode
    update_freq: int = 1             # 每步都更新
    
    # 正则化参数
    weight_decay: float = 1e-5       # 更小的L2正则化
    grad_clip: float = 1.0           # 适度梯度裁剪


# =============================================================================
# 注意：VECActionSpace 和 VECStateSpace 已移至 common_state_action.py
# 使用 UnifiedStateActionSpace 类获取统一的状态/动作空间定义
# =============================================================================


class OptimizedTD3Actor(nn.Module):
    """优化的TD3 Actor网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512, max_action: float = 1.0):
        super(OptimizedTD3Actor, self).__init__()
        
        self.max_action = max_action
        
        # 更深的网络结构
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用较小的权重初始化
        nn.init.uniform_(self.network[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-2].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.network(state)


class OptimizedTD3Critic(nn.Module):
    """优化的TD3 Twin Critic网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        super(OptimizedTD3Critic, self).__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Q2网络
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """改进的权重初始化"""
        for network in [self.q1_network, self.q2_network]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
            
            # 最后一层使用较小的权重初始化
            nn.init.uniform_(network[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(network[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 返回两个Q值"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1值"""
        sa = torch.cat([state, action], dim=1)
        return self.q1_network(sa)


# =============================================================================
# 注意：完整的训练环境请使用 optimized_td3_wrapper.py 中的 OptimizedTD3Wrapper
# 以下类已废弃，保留仅供参考
# =============================================================================
