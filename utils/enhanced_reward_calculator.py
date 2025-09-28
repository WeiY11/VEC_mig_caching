#!/usr/bin/env python3
"""
增强的奖励计算器
针对缓存、卸载、迁移三个子问题提供专门的奖励信号
"""

import numpy as np
from typing import Dict, Optional
from config import config

class EnhancedRewardCalculator:
    """
    增强奖励计算器
    为DRL提供针对性的奖励信号，指导缓存和迁移策略学习
    """
    
    def __init__(self):
        # 从配置加载基础权重
        self.weight_delay = config.rl.reward_weight_delay
        self.weight_energy = config.rl.reward_weight_energy
        self.weight_loss = config.rl.reward_weight_loss
        
        # 🔧 新增：子系统奖励权重
        self.weight_cache = 0.3        # 缓存性能权重
        self.weight_migration = 0.2    # 迁移性能权重
        self.weight_coordination = 0.1 # 协调奖励权重
        
        # 归一化因子
        self.delay_normalizer = 1.0
        self.energy_normalizer = 1000.0  # 修正为合理值
        self.cache_normalizer = 1.0
        
        # 奖励范围
        self.reward_clip_range = (-10.0, 2.0)  # 允许少量正奖励
        
        print("✅ 增强奖励计算器初始化完成")
        print(f"   基础权重: Delay={self.weight_delay}, Energy={self.weight_energy}, Loss={self.weight_loss}")
        print(f"   子系统权重: Cache={self.weight_cache}, Migration={self.weight_migration}")
    
    def calculate_enhanced_reward(self, system_metrics: Dict, 
                                cache_metrics: Optional[Dict] = None,
                                migration_metrics: Optional[Dict] = None) -> Dict[str, float]:
        """
        计算增强奖励，包含总奖励和分解奖励
        
        Returns:
            {
                'total_reward': float,
                'delay_reward': float,
                'energy_reward': float, 
                'cache_reward': float,
                'migration_reward': float,
                'coordination_reward': float
            }
        """
        # 基础指标奖励
        delay_reward = self._calculate_delay_reward(system_metrics)
        energy_reward = self._calculate_energy_reward(system_metrics)
        loss_reward = self._calculate_loss_reward(system_metrics)
        
        # 子系统专门奖励
        cache_reward = self._calculate_cache_reward(system_metrics, cache_metrics)
        migration_reward = self._calculate_migration_reward(system_metrics, migration_metrics)
        
        # 协调奖励（奖励系统间的协作）
        coordination_reward = self._calculate_coordination_reward(
            system_metrics, cache_metrics, migration_metrics
        )
        
        # 总奖励
        total_reward = (
            self.weight_delay * delay_reward +
            self.weight_energy * energy_reward +
            self.weight_loss * loss_reward +
            self.weight_cache * cache_reward +
            self.weight_migration * migration_reward +
            self.weight_coordination * coordination_reward
        )
        
        # 限制奖励范围
        total_reward = np.clip(total_reward, *self.reward_clip_range)
        
        return {
            'total_reward': total_reward,
            'delay_reward': delay_reward,
            'energy_reward': energy_reward,
            'loss_reward': loss_reward,
            'cache_reward': cache_reward,
            'migration_reward': migration_reward,
            'coordination_reward': coordination_reward
        }
    
    def _calculate_delay_reward(self, system_metrics: Dict) -> float:
        """计算时延奖励"""
        avg_delay = max(0.0, float(system_metrics.get('avg_task_delay', 0.0)))
        
        # 非线性惩罚：时延越高惩罚越重
        delay_penalty = -(avg_delay / self.delay_normalizer) ** 1.5
        
        # 时延目标奖励：低于0.2秒给予奖励
        if avg_delay < 0.2:
            delay_bonus = 0.1 * (0.2 - avg_delay) / 0.2
        else:
            delay_bonus = 0.0
        
        return delay_penalty + delay_bonus
    
    def _calculate_energy_reward(self, system_metrics: Dict) -> float:
        """计算能耗奖励"""
        total_energy = max(0.0, float(system_metrics.get('total_energy_consumption', 0.0)))
        
        # 能耗惩罚
        energy_penalty = -(total_energy / self.energy_normalizer)
        
        # 能效奖励：能耗低于800焦耳给予奖励
        if total_energy < 800.0:
            energy_bonus = 0.05 * (800.0 - total_energy) / 800.0
        else:
            energy_bonus = 0.0
        
        return energy_penalty + energy_bonus
    
    def _calculate_loss_reward(self, system_metrics: Dict) -> float:
        """计算数据丢失奖励"""
        completion_rate = max(0.0, min(1.0, float(system_metrics.get('task_completion_rate', 0.0))))
        
        # 完成率奖励
        completion_bonus = completion_rate * 0.2  # 最高0.2奖励
        
        # 数据丢失惩罚
        loss_rate = 1.0 - completion_rate
        loss_penalty = -(loss_rate ** 2) * 2.0  # 非线性惩罚
        
        return completion_bonus + loss_penalty
    
    def _calculate_cache_reward(self, system_metrics: Dict, cache_metrics: Optional[Dict]) -> float:
        """
        🔧 新增：计算缓存专门奖励
        """
        if not cache_metrics:
            return 0.0
        
        cache_hit_rate = cache_metrics.get('hit_rate', 0.0)
        cache_utilization = cache_metrics.get('utilization', 0.0)
        
        # 缓存命中率奖励
        hit_rate_reward = cache_hit_rate * 0.3  # 最高0.3奖励
        
        # 缓存利用率奖励（鼓励合理利用）
        if 0.6 <= cache_utilization <= 0.9:
            utilization_reward = 0.1
        elif cache_utilization > 0.9:
            utilization_reward = -0.1  # 过度利用惩罚
        else:
            utilization_reward = 0.0
        
        # 缓存效率奖励
        effectiveness = cache_metrics.get('effectiveness', 0.0)
        efficiency_reward = effectiveness * 0.2
        
        return hit_rate_reward + utilization_reward + efficiency_reward
    
    def _calculate_migration_reward(self, system_metrics: Dict, migration_metrics: Optional[Dict]) -> float:
        """
        🔧 新增：计算迁移专门奖励
        """
        if not migration_metrics:
            return 0.0
        
        migration_success_rate = migration_metrics.get('success_rate', 0.0)
        avg_delay_saved = migration_metrics.get('avg_delay_saved', 0.0)
        migration_frequency = migration_metrics.get('frequency', 0.0)
        
        # 迁移成功率奖励
        success_reward = migration_success_rate * 0.15
        
        # 时延节省奖励
        delay_saved_reward = min(0.1, avg_delay_saved * 0.1)
        
        # 迁移频率平衡（过多或过少都不好）
        optimal_frequency = 0.1  # 每10步1次迁移为理想
        frequency_penalty = -abs(migration_frequency - optimal_frequency) * 0.5
        
        return success_reward + delay_saved_reward + frequency_penalty
    
    def _calculate_coordination_reward(self, system_metrics: Dict, 
                                     cache_metrics: Optional[Dict],
                                     migration_metrics: Optional[Dict]) -> float:
        """
        🔧 新增：计算协调奖励，鼓励子系统间协作
        """
        coordination_reward = 0.0
        
        if cache_metrics and migration_metrics:
            cache_hit_rate = cache_metrics.get('hit_rate', 0.0)
            migration_success_rate = migration_metrics.get('success_rate', 0.0)
            
            # 双高协调奖励：缓存和迁移都表现好
            if cache_hit_rate > 0.7 and migration_success_rate > 0.8:
                coordination_reward += 0.1
            
            # 负载均衡协调：如果迁移有效降低了延迟且缓存命中率稳定
            avg_delay = system_metrics.get('avg_task_delay', 1.0)
            if avg_delay < 0.3 and cache_hit_rate > 0.6:
                coordination_reward += 0.05
        
        return coordination_reward
    
    def get_reward_breakdown(self, system_metrics: Dict,
                           cache_metrics: Optional[Dict] = None,
                           migration_metrics: Optional[Dict] = None) -> str:
        """
        获取奖励分解的可读报告
        """
        rewards = self.calculate_enhanced_reward(system_metrics, cache_metrics, migration_metrics)
        
        breakdown = f"""
奖励分解报告:
├── 总奖励: {rewards['total_reward']:.3f}
├── 时延奖励: {rewards['delay_reward']:.3f}
├── 能耗奖励: {rewards['energy_reward']:.3f}  
├── 数据丢失: {rewards['loss_reward']:.3f}
├── 缓存奖励: {rewards['cache_reward']:.3f}
├── 迁移奖励: {rewards['migration_reward']:.3f}
└── 协调奖励: {rewards['coordination_reward']:.3f}
        """
        
        return breakdown.strip()

# 全局增强奖励计算器
_enhanced_reward_calculator = EnhancedRewardCalculator()

def calculate_enhanced_reward(system_metrics: Dict,
                            cache_metrics: Optional[Dict] = None,
                            migration_metrics: Optional[Dict] = None) -> float:
    """
    供外部调用的增强奖励接口
    """
    result = _enhanced_reward_calculator.calculate_enhanced_reward(
        system_metrics, cache_metrics, migration_metrics
    )
    return result['total_reward']

def get_reward_breakdown(system_metrics: Dict,
                        cache_metrics: Optional[Dict] = None, 
                        migration_metrics: Optional[Dict] = None) -> str:
    """
    获取奖励分解报告
    """
    return _enhanced_reward_calculator.get_reward_breakdown(
        system_metrics, cache_metrics, migration_metrics
    )
