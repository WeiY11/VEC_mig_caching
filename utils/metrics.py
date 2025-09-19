#!/usr/bin/env python3
"""
性能指标计算工具
"""

import numpy as np
from typing import List, Dict, Any
from collections import deque

class Metrics:
    """性能指标计算器"""
    
    def __init__(self):
        self.data = {}
    
    def add_metric(self, name: str, value: float):
        """添加指标数据"""
        if name not in self.data:
            self.data[name] = []
        self.data[name].append(value)
    
    def get_mean(self, name: str) -> float:
        """获取平均值"""
        if name not in self.data or not self.data[name]:
            return 0.0
        return np.mean(self.data[name])
    
    def get_std(self, name: str) -> float:
        """获取标准差"""
        if name not in self.data or len(self.data[name]) < 2:
            return 0.0
        return np.std(self.data[name])
    
    def get_latest(self, name: str, n: int = 1) -> List[float]:
        """获取最新的n个值"""
        if name not in self.data:
            return []
        return self.data[name][-n:]
    
    def get_moving_average(self, name: str, window: int = 10) -> float:
        """获取移动平均值"""
        if name not in self.data or not self.data[name]:
            return 0.0
        
        recent_data = self.data[name][-window:]
        return np.mean(recent_data)
    
    def reset(self):
        """重置所有数据"""
        self.data.clear()
    
    def summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有指标的摘要"""
        summary = {}
        for name in self.data:
            if self.data[name]:
                summary[name] = {
                    'mean': self.get_mean(name),
                    'std': self.get_std(name),
                    'min': min(self.data[name]),
                    'max': max(self.data[name]),
                    'latest': self.data[name][-1] if self.data[name] else 0.0,
                    'count': len(self.data[name])
                }
        return summary

class MovingAverage:
    """移动平均计算器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.sum = 0.0
    
    def update(self, value: float) -> float:
        """更新移动平均值"""
        if len(self.values) == self.window_size:
            # 移除最旧的值
            self.sum -= self.values[0]
        
        # 添加新值
        self.values.append(value)
        self.sum += value
        
        return self.sum / len(self.values)
    
    def get_average(self) -> float:
        """获取当前平均值"""
        if not self.values:
            return 0.0
        return self.sum / len(self.values)
    
    def reset(self):
        """重置"""
        self.values.clear()
        self.sum = 0.0

class PerformanceTracker:
    """性能跟踪器"""
    
    def __init__(self):
        self.episode_rewards = MovingAverage(100)
        self.episode_lengths = MovingAverage(100)
        self.actor_losses = MovingAverage(50)
        self.critic_losses = MovingAverage(50)
        self.q_values = MovingAverage(100)
        
        self.total_episodes = 0
        self.total_steps = 0
    
    def update_episode(self, reward: float, length: int):
        """更新回合数据"""
        self.episode_rewards.update(reward)
        self.episode_lengths.update(length)
        self.total_episodes += 1
        self.total_steps += length
    
    def update_training(self, actor_loss: float = None, 
                       critic_loss: float = None, q_value: float = None):
        """更新训练数据"""
        if actor_loss is not None:
            self.actor_losses.update(actor_loss)
        if critic_loss is not None:
            self.critic_losses.update(critic_loss)
        if q_value is not None:
            self.q_values.update(q_value)
    
    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        return {
            'avg_episode_reward': self.episode_rewards.get_average(),
            'avg_episode_length': self.episode_lengths.get_average(),
            'avg_actor_loss': self.actor_losses.get_average(),
            'avg_critic_loss': self.critic_losses.get_average(),
            'avg_q_value': self.q_values.get_average(),
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps
        }
    
    def reset(self):
        """重置所有数据"""
        self.episode_rewards.reset()
        self.episode_lengths.reset()
        self.actor_losses.reset()
        self.critic_losses.reset()
        self.q_values.reset()
        self.total_episodes = 0
        self.total_steps = 0