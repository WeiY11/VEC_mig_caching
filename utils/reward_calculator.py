#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一的奖励函数计算模块
对应论文目标函数的标准化实现
确保所有算法使用一致的奖励计算逻辑
"""

import numpy as np
from typing import Dict, Optional
from config import config


class UnifiedRewardCalculator:
    """
    统一奖励计算器 - 对应论文式(24)目标函数
    目标: min(ω_T * delay + ω_E * energy + ω_D * data_loss)
    奖励: reward = -cost
    """
    
    def __init__(self):
        # 奖励权重配置 - 从配置文件获取
        self.weight_delay = config.rl.reward_weight_delay     # ω_T
        self.weight_energy = config.rl.reward_weight_energy   # ω_E  
        self.weight_loss = config.rl.reward_weight_loss       # ω_D
        
        # 归一化参数 - 确保数值稳定性
        self.delay_normalizer = 1.0      # 延迟归一化因子 (秒)
        self.energy_normalizer = 1000.0  # 能耗归一化因子 (J)
        self.loss_normalizer = 1.0       # 数据丢失率归一化因子
        
        # 奖励范围限制
        self.min_reward = -10.0
        self.max_reward = 5.0
        
        # 移动平均参数
        self.alpha = 0.1
        self.reward_history = []
        self.reward_mean = 0.0
        self.reward_std = 1.0
    
    def calculate_base_reward(self, system_metrics: Dict) -> float:
        """
        计算基础奖励 - 基于论文目标函数
        
        Args:
            system_metrics: 系统性能指标字典
            
        Returns:
            基础奖励值
        """
        # 提取系统指标并添加默认值保护
        avg_delay = system_metrics.get('avg_task_delay', 0.0)
        total_energy = system_metrics.get('total_energy_consumption', 0.0)
        data_loss_rate = system_metrics.get('data_loss_rate', 0.0)
        
        # 数值有效性检查
        avg_delay = max(0.0, float(avg_delay)) if np.isfinite(avg_delay) else 0.0
        total_energy = max(0.0, float(total_energy)) if np.isfinite(total_energy) else 0.0
        data_loss_rate = np.clip(float(data_loss_rate), 0.0, 1.0) if np.isfinite(data_loss_rate) else 0.0
        
        # 归一化指标
        normalized_delay = avg_delay / self.delay_normalizer
        normalized_energy = total_energy / self.energy_normalizer
        normalized_loss = data_loss_rate / self.loss_normalizer
        
        # 计算成本函数 - 对应论文式(24)
        cost = (self.weight_delay * normalized_delay + 
                self.weight_energy * normalized_energy + 
                self.weight_loss * normalized_loss)
        
        # 转换为奖励 (成本的负值)
        base_reward = -cost
        
        return base_reward
    
    def calculate_performance_bonus(self, system_metrics: Dict) -> float:
        """
        计算性能奖励 - 鼓励高性能表现
        
        Args:
            system_metrics: 系统性能指标字典
            
        Returns:
            性能奖励值
        """
        # 任务完成率奖励
        completion_rate = system_metrics.get('task_completion_rate', 0.0)
        completion_rate = np.clip(completion_rate, 0.0, 1.0)
        completion_bonus = 0.1 * completion_rate
        
        # 缓存命中率奖励
        cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
        cache_hit_rate = np.clip(cache_hit_rate, 0.0, 1.0)
        cache_bonus = 0.05 * np.tanh(cache_hit_rate * 2.0)  # 使用tanh平滑
        
        # 迁移成功率奖励
        migration_rate = system_metrics.get('migration_success_rate', 0.0)
        migration_rate = np.clip(migration_rate, 0.0, 1.0)
        migration_bonus = 0.02 * migration_rate
        
        return completion_bonus + cache_bonus + migration_bonus
    
    def calculate_unified_reward(self, system_metrics: Dict, 
                               agent_type: Optional[str] = None) -> float:
        """
        计算统一奖励 - 标准化的奖励计算接口
        
        Args:
            system_metrics: 系统性能指标字典
            agent_type: 智能体类型 (可选，用于特定化奖励)
            
        Returns:
            最终奖励值
        """
        # 基础奖励
        base_reward = self.calculate_base_reward(system_metrics)
        
        # 性能奖励
        performance_bonus = self.calculate_performance_bonus(system_metrics)
        
        # 智能体特定奖励 (可选)
        agent_bonus = self._calculate_agent_specific_bonus(system_metrics, agent_type)
        
        # 组合奖励
        total_reward = base_reward + performance_bonus + agent_bonus
        
        # 应用奖励范围限制
        clipped_reward = np.clip(total_reward, self.min_reward, self.max_reward)
        
        # 更新移动统计
        self._update_reward_statistics(clipped_reward)
        
        return clipped_reward
    
    def _calculate_agent_specific_bonus(self, system_metrics: Dict, 
                                      agent_type: Optional[str]) -> float:
        """
        计算智能体特定奖励
        
        Args:
            system_metrics: 系统性能指标字典
            agent_type: 智能体类型
            
        Returns:
            智能体特定奖励值
        """
        if agent_type is None:
            return 0.0
        
        bonus = 0.0
        
        if agent_type == 'vehicle_agent':
            # 车辆智能体关注本地处理效率
            local_ratio = system_metrics.get('local_processing_ratio', 0.0)
            bonus = 0.05 * np.clip(local_ratio, 0.0, 1.0)
            
        elif agent_type == 'rsu_agent':
            # RSU智能体关注缓存和负载均衡
            cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
            rsu_utilization = system_metrics.get('avg_rsu_utilization', 0.5)
            
            cache_bonus = 0.03 * np.clip(cache_hit_rate, 0.0, 1.0)
            # 负载均衡奖励 (利用率接近0.7时奖励最高)
            load_balance_bonus = 0.02 * (1.0 - abs(0.7 - rsu_utilization))
            bonus = cache_bonus + load_balance_bonus
            
        elif agent_type == 'uav_agent':
            # UAV智能体关注能效和电池管理
            uav_battery = system_metrics.get('avg_uav_battery', 1.0)
            uav_energy_efficiency = system_metrics.get('uav_energy_efficiency', 0.0)
            
            battery_bonus = 0.03 * np.clip(uav_battery, 0.0, 1.0)
            efficiency_bonus = 0.02 * np.clip(uav_energy_efficiency, 0.0, 1.0)
            bonus = battery_bonus + efficiency_bonus
        
        return bonus
    
    def _update_reward_statistics(self, reward: float):
        """
        更新奖励统计信息 - 用于归一化和分析
        
        Args:
            reward: 当前奖励值
        """
        self.reward_history.append(reward)
        
        # 保持历史记录长度
        if len(self.reward_history) > 1000:
            self.reward_history.pop(0)
        
        # 更新移动平均和标准差
        if len(self.reward_history) > 10:
            recent_rewards = self.reward_history[-100:]  # 最近100个奖励
            self.reward_mean = self.alpha * reward + (1 - self.alpha) * self.reward_mean
            self.reward_std = np.std(recent_rewards)
    
    def normalize_reward(self, reward: float) -> float:
        """
        奖励归一化 - 可选的归一化处理
        
        Args:
            reward: 原始奖励值
            
        Returns:
            归一化后的奖励值
        """
        if self.reward_std > 1e-6:
            normalized = (reward - self.reward_mean) / self.reward_std
            return np.clip(normalized, -3.0, 3.0)  # 限制在3个标准差内
        else:
            return reward
    
    def get_reward_statistics(self) -> Dict:
        """
        获取奖励统计信息
        
        Returns:
            奖励统计字典
        """
        return {
            'mean': self.reward_mean,
            'std': self.reward_std,
            'min': min(self.reward_history) if self.reward_history else 0.0,
            'max': max(self.reward_history) if self.reward_history else 0.0,
            'count': len(self.reward_history)
        }


# 全局奖励计算器实例
reward_calculator = UnifiedRewardCalculator()


def calculate_unified_reward(system_metrics: Dict, agent_type: Optional[str] = None) -> float:
    """
    统一奖励计算接口 - 供所有算法调用
    
    Args:
        system_metrics: 系统性能指标字典
        agent_type: 智能体类型 (可选)
        
    Returns:
        统一计算的奖励值
    """
    return reward_calculator.calculate_unified_reward(system_metrics, agent_type)