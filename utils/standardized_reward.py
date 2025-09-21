#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一奖励函数修复模块
确保所有算法使用相同的奖励计算逻辑
"""

import numpy as np
from typing import Dict, Optional
from config import config

class StandardizedRewardFunction:
    """
    标准化奖励函数 - 严格按照论文目标函数实现
    论文目标: min(ω_T * delay + ω_E * energy + ω_D * data_loss)
    奖励函数: reward = -cost
    """
    
    def __init__(self):
        # 严格按照论文权重配置
        self.weight_delay = config.rl.reward_weight_delay     # ω_T
        self.weight_energy = config.rl.reward_weight_energy   # ω_E  
        self.weight_loss = config.rl.reward_weight_loss       # ω_D
        
        # 统一的归一化参数 - 基于论文和实际数据范围
        self.delay_normalizer = 1.0      # 延迟归一化因子 (秒)
        self.energy_normalizer = 1000.0  # 能耗归一化因子 (J)
        self.loss_normalizer = 1.0       # 数据丢失率归一化因子
        
        # 奖励范围限制 - 确保数值稳定
        self.min_reward = -10.0
        self.max_reward = 5.0
        
        print(f"✅ 标准化奖励函数初始化")
        print(f"   权重配置: 延迟={self.weight_delay}, 能耗={self.weight_energy}, 丢失={self.weight_loss}")
    
    def calculate_paper_reward(self, system_metrics: Dict) -> float:
        """
        修复版本：因为之前的奖励信号过弱，需要增强信号强度
        对应论文式(24): min(ω_T * T + ω_E * E + ω_D * D)
        """
        # 提取系统指标
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
        
        # 计算成本函数 - 严格对应论文式(24)
        cost = (self.weight_delay * normalized_delay + 
                self.weight_energy * normalized_energy + 
                self.weight_loss * normalized_loss)
        
        # 转换为奖励 (成本的负值)
        base_reward = -cost
        
        # 🔧 修复：放大信号强度解决诊断发现的信号过弱问题
        amplified_reward = base_reward * 8.0  # 8倍放大，使奖励变化更显著
        
        # 应用放大后的奖励范围限制
        clipped_reward = np.clip(amplified_reward, -40.0, 20.0)  # 扩大范围保持信号强度
        
        return clipped_reward
    
    def calculate_with_performance_bonus(self, system_metrics: Dict, 
                                       agent_type: Optional[str] = None) -> float:
        """
        修复版本：在论文奖励基础上添加性能激励，解决相关性问题
        """
        # 基础论文奖励 (已放大)
        base_reward = self.calculate_paper_reward(system_metrics)
        
        # 🔧 修复：强化性能激励解决相关性问题
        completion_rate = system_metrics.get('task_completion_rate', 0.0)
        cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
        
        # 显著增强性能奖励，确保与性能指标强相关
        performance_bonus = 5.0 * completion_rate + 3.0 * cache_hit_rate  # 显著增强相关性
        
        # 智能体特定奖励 (针对多智能体场景)
        agent_bonus = 0.0
        if agent_type:
            if agent_type == 'vehicle_agent':
                local_efficiency = system_metrics.get('local_processing_ratio', 0.0)
                agent_bonus = 1.0 * local_efficiency  # 增强智能体奖励
            elif agent_type == 'rsu_agent':
                load_balance = 1.0 - abs(0.7 - system_metrics.get('avg_rsu_utilization', 0.7))
                agent_bonus = 1.0 * load_balance
            elif agent_type == 'uav_agent':
                battery_level = system_metrics.get('avg_uav_battery', 1.0)
                agent_bonus = 1.0 * battery_level
        
        final_reward = base_reward + performance_bonus + agent_bonus
        
        # 放大后的范围限制
        return np.clip(final_reward, -80.0, 50.0)


# 创建全局标准化奖励函数实例
_standardized_reward_function = StandardizedRewardFunction()


def calculate_standardized_reward(system_metrics: Dict, agent_type: Optional[str] = None, 
                                 use_paper_only: bool = False) -> float:
    """
    标准化奖励计算接口 - 供所有算法调用
    
    Args:
        system_metrics: 系统性能指标字典
        agent_type: 智能体类型 (可选)
        use_paper_only: 是否只使用论文奖励函数 (默认False，会添加轻量激励)
        
    Returns:
        标准化计算的奖励值
    """
    if use_paper_only:
        return _standardized_reward_function.calculate_paper_reward(system_metrics)
    else:
        return _standardized_reward_function.calculate_with_performance_bonus(system_metrics, agent_type)


def validate_reward_consistency():
    """验证奖励函数一致性"""
    # 测试用例
    test_metrics = {
        'avg_task_delay': 0.1,
        'total_energy_consumption': 500.0,
        'data_loss_rate': 0.05,
        'task_completion_rate': 0.9,
        'cache_hit_rate': 0.8
    }
    
    # 计算奖励
    paper_reward = calculate_standardized_reward(test_metrics, use_paper_only=True)
    full_reward = calculate_standardized_reward(test_metrics, use_paper_only=False)
    
    print(f"✅ 奖励函数一致性验证:")
    print(f"   论文奖励: {paper_reward:.4f}")
    print(f"   完整奖励: {full_reward:.4f}")
    print(f"   权重验证: 延迟={config.rl.reward_weight_delay}, 能耗={config.rl.reward_weight_energy}, 丢失={config.rl.reward_weight_loss}")
    
    return paper_reward, full_reward


if __name__ == "__main__":
    validate_reward_consistency()