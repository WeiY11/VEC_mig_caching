#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一个简化的奖励计算器，严格遵循成本最小化原则。
奖励 = - (加权成本 + 惩罚项)
确保所有算法使用一致、专注的奖励信号。
"""

import numpy as np
from typing import Dict
from config import config

class SimpleRewardCalculator:
    """
    简化的奖励计算器。
    - 移除了所有正向奖励(bonus)
    - 奖励严格为负值
    - 引入对丢弃任务的直接惩罚
    """

    def __init__(self):
        # 从配置加载权重
        self.weight_delay = config.rl.reward_weight_delay
        self.weight_energy = config.rl.reward_weight_energy
        self.weight_loss = config.rl.reward_weight_loss
        
        # 引入对丢弃任务（通常是超时）的直接惩罚
        self.penalty_weight_dropped = config.rl.reward_penalty_dropped
        
        # 归一化因子 - 更稳定的设置，避免过度敏感
        self.delay_normalizer = 1.0      # 恢复标准归一化，提高稳定性
        self.energy_normalizer = 8000.0  # 适中敏感度，避免训练不稳定
        self.loss_normalizer = 1.0       # MB, 丢失数据按MB计算
        
        # 奖励范围限制，确保为负值
        self.reward_clip_range = (-25.0, -0.01)  # 适中惩罚范围
        
        print("✅ 简化奖励函数初始化 (SimpleRewardCalculator)")
        print(f"   权重: Delay={self.weight_delay}, Energy={self.weight_energy}, Loss={self.weight_loss}")
        print(f"   惩罚: DroppedTaskPenalty={self.penalty_weight_dropped}")

    def calculate_reward(self, system_metrics: Dict) -> float:
        """
        计算简化后的奖励值。
        """
        # 提取核心指标
        avg_delay = max(0.0, float(system_metrics.get('avg_task_delay', 0.0)))
        total_energy = max(0.0, float(system_metrics.get('total_energy_consumption', 0.0)))
        
        # 使用修正后的数据丢失量（bytes），并转换为MB
        data_loss_bytes = max(0.0, float(system_metrics.get('data_loss_bytes', 0.0)))
        data_loss_mb = data_loss_bytes / 1e6
        
        # 提取丢弃任务数量
        dropped_tasks = int(system_metrics.get('dropped_tasks', 0))
        
        # 归一化
        norm_delay = avg_delay / self.delay_normalizer
        norm_energy = total_energy / self.energy_normalizer
        norm_loss = data_loss_mb / self.loss_normalizer
        
        # 计算三项基本成本
        base_cost = (self.weight_delay * norm_delay +
                     self.weight_energy * norm_energy +
                     self.weight_loss * norm_loss)
                     
        # 计算惩罚项
        penalty = self.penalty_weight_dropped * dropped_tasks
        
        # 极致时延控制：目标时延<0.22s
        delay_penalty = max(0, (avg_delay - 0.22) * 5.0) if avg_delay > 0.22 else 0  # 时延超过0.22s时极强惩罚
        energy_penalty = max(0, (total_energy - 3500) / 1000.0) if total_energy > 3500 else 0  # 能耗超过3500J时强力惩罚
        
        # 总成本 = 基本成本 + 基本惩罚 + 温和阈值惩罚
        total_cost = base_cost + penalty + delay_penalty + energy_penalty
        
        # 奖励是总成本的负值
        reward = -total_cost
        
        # 裁剪奖励值
        clipped_reward = np.clip(reward, self.reward_clip_range[0], self.reward_clip_range[1])
        
        return clipped_reward

# 全局实例
_simple_reward_calculator = SimpleRewardCalculator()

def calculate_simple_reward(system_metrics: Dict) -> float:
    """
    供外部调用的接口
    """
    return _simple_reward_calculator.calculate_reward(system_metrics)
