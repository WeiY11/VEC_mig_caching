#!/usr/bin/env python3
"""
TD3增强探索版本 - 借鉴SAC的探索优势

核心改进：
1. 自适应探索噪声（基于状态）
2. 针对缓存动作维度的额外探索
3. 早期高探索，后期逐渐降低

使用：
    export TD3_ENHANCED_EXPLORATION=1
    python train_single_agent.py --algorithm TD3 --episodes 800
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from single_agent.td3 import TD3Agent, TD3Config


class EnhancedExplorationTD3Agent(TD3Agent):
    """增强探索版TD3 - 借鉴SAC的探索策略"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 🔥 增强探索参数
        self.base_exploration_noise = 0.3  # 提高基础噪声（从0.2）
        self.cache_exploration_bonus = 0.15  # 缓存维度额外探索
        self.exploration_noise = self.base_exploration_noise
        
        # 自适应探索（基于性能）
        self.recent_cache_hits = []
        self.adaptive_exploration = True
        
        print("🔥 TD3增强探索版本已启用")
        print(f"   基础噪声: {self.base_exploration_noise}")
        print(f"   缓存探索: +{self.cache_exploration_bonus}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作 - 增强探索版"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        if training:
            # 🔥 自适应噪声：早期高，后期低
            # 动态调整基于训练进度
            progress = min(1.0, self.step_count / 100000.0)
            adaptive_noise = self.base_exploration_noise * (1.0 - 0.7 * progress)
            
            # 基础探索噪声
            noise = np.random.normal(0, adaptive_noise, size=action.shape)
            
            # 🎯 缓存/迁移维度额外探索（后8维）
            # 这部分对延迟影响大，需要充分探索
            cache_start = len(action) - 8
            if cache_start > 0:
                cache_noise = np.random.normal(
                    0, 
                    self.cache_exploration_bonus, 
                    size=8
                )
                noise[cache_start:] += cache_noise
            
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update_exploration(self, cache_hit_rate: float):
        """基于缓存命中率动态调整探索"""
        if not self.adaptive_exploration:
            return
        
        self.recent_cache_hits.append(cache_hit_rate)
        if len(self.recent_cache_hits) > 50:
            self.recent_cache_hits.pop(0)
        
        # 如果缓存命中率持续低，增加探索
        if len(self.recent_cache_hits) >= 20:
            avg_hit_rate = np.mean(self.recent_cache_hits[-20:])
            if avg_hit_rate < 0.45:  # 低于45%
                self.cache_exploration_bonus = min(0.25, self.cache_exploration_bonus * 1.05)
                print(f"🔍 低缓存命中率({avg_hit_rate:.1%})，增加探索: {self.cache_exploration_bonus:.3f}")


# 便捷创建函数
def create_enhanced_td3_env(num_vehicles=12, num_rsus=4, num_uavs=2):
    """创建增强探索版TD3环境"""
    from single_agent.td3 import TD3Environment
    from single_agent.common_state_action import UnifiedStateActionSpace
    
    # 创建标准TD3环境
    env = TD3Environment(num_vehicles, num_rsus, num_uavs)
    
    # 替换agent为增强版
    state_dim = env.state_dim
    action_dim = env.action_dim
    config = env.config
    
    enhanced_agent = EnhancedExplorationTD3Agent(
        state_dim, action_dim, config,
        num_vehicles, num_rsus, num_uavs
    )
    
    env.agent = enhanced_agent
    
    return env

