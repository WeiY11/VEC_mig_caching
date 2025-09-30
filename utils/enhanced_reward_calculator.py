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
        self.weight_energy = config.rl.reward_weight_energy * 1.5  # 🔧 增加能耗权重50%，防止过拟合到高能耗策略
        self.weight_loss = config.rl.reward_weight_loss
        
        # 🔧 新增：子系统奖励权重
        self.weight_cache = 0.3        # 缓存性能权重
        self.weight_migration = 0.2    # 迁移性能权重
        self.weight_coordination = 0.1 # 协调奖励权重
        
        # 归一化因子
        self.delay_normalizer = 1.0
        self.energy_normalizer = 1000.0  # 修正为合理值
        self.cache_normalizer = 1.0
        
        # 🔧 修复：奖励必须始终为负值，符合VEC成本最小化原则
        self.reward_clip_range = (-15.0, -0.01)  # 确保奖励始终为负值
        
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
        """
        🔧 修复：计算时延成本（纯负值）
        """
        avg_delay = max(0.0, float(system_metrics.get('avg_task_delay', 0.0)))
        
        # 时延成本：时延越高成本越高
        # 使用平方惩罚，鼓励更低时延
        delay_cost = -(avg_delay / self.delay_normalizer) ** 1.2
        
        # 🔧 移除正向奖励，改为成本减免
        if avg_delay < 0.2:
            # 低时延时成本减免，但仍为负值
            cost_reduction = delay_cost * 0.5  # 减免50%成本，但总体仍为负
        else:
            cost_reduction = 0.0
        
        return delay_cost + cost_reduction  # 仍然为负值
    
    def _calculate_energy_reward(self, system_metrics: Dict) -> float:
        """
        🔧 修复：计算能耗成本（纯负值）
        """
        total_energy = max(0.0, float(system_metrics.get('total_energy_consumption', 0.0)))
        
        # 能耗成本：能耗越高成本越高
        energy_cost = -(total_energy / self.energy_normalizer)
        
        # 🔧 移除正向奖励，改为成本减免
        if total_energy < 800.0:
            # 低能耗时成本减免，但仍为负值
            cost_reduction = energy_cost * 0.3  # 减免30%成本，但总体仍为负
        else:
            cost_reduction = 0.0
        
        return energy_cost + cost_reduction  # 仍然为负值
    
    def _calculate_loss_reward(self, system_metrics: Dict) -> float:
        """
        🔧 修复：计算数据丢失成本（纯负值）
        """
        completion_rate = max(0.0, min(1.0, float(system_metrics.get('task_completion_rate', 0.0))))
        
        # 数据丢失成本：丢失率越高成本越高
        loss_rate = 1.0 - completion_rate
        loss_cost = -(loss_rate ** 2) * 3.0  # 非线性成本
        
        # 🔧 移除正向奖励，改为基于完成率的成本减免
        if completion_rate > 0.9:
            # 高完成率时成本减免，但仍为负值
            cost_reduction = loss_cost * 0.4  # 减免40%成本
        else:
            cost_reduction = 0.0
        
        return loss_cost + cost_reduction  # 仍然为负值
    
    def _calculate_cache_reward(self, system_metrics: Dict, cache_metrics: Optional[Dict]) -> float:
        """
        🔧 修复：计算缓存成本（纯负值）
        """
        if not cache_metrics:
            return -0.1  # 无缓存数据时的默认成本
        
        cache_hit_rate = cache_metrics.get('hit_rate', 0.0)
        cache_utilization = cache_metrics.get('utilization', 0.0)
        
        # 缓存miss成本：命中率越低成本越高
        cache_miss_rate = 1.0 - cache_hit_rate
        cache_miss_cost = -(cache_miss_rate ** 1.5) * 0.5
        
        # 缓存管理成本
        if cache_utilization > 0.9:
            management_cost = -0.2  # 过度利用额外成本
        elif cache_utilization < 0.3:
            management_cost = -0.1  # 利用不足的机会成本
        else:
            management_cost = -0.05  # 正常管理成本
        
        return cache_miss_cost + management_cost  # 总是负值
    
    def _calculate_migration_reward(self, system_metrics: Dict, migration_metrics: Optional[Dict]) -> float:
        """
        🔧 修复：计算迁移成本（纯负值）
        """
        if not migration_metrics:
            return -0.05  # 无迁移数据时的默认成本
        
        migration_success_rate = migration_metrics.get('success_rate', 0.0)
        migration_frequency = migration_metrics.get('frequency', 0.0)
        
        # 迁移失败成本：失败率越高成本越高
        migration_failure_rate = 1.0 - migration_success_rate
        migration_failure_cost = -(migration_failure_rate ** 2) * 0.3
        
        # 迁移操作成本：频率过高有额外成本
        if migration_frequency > 0.15:  # 频繁迁移
            operation_cost = -migration_frequency * 0.2
        else:
            operation_cost = -0.02  # 基础迁移管理成本
        
        return migration_failure_cost + operation_cost  # 总是负值
    
    def _calculate_coordination_reward(self, system_metrics: Dict, 
                                     cache_metrics: Optional[Dict],
                                     migration_metrics: Optional[Dict]) -> float:
        """
        🔧 修复：计算系统协调成本（纯负值）
        """
        if not cache_metrics or not migration_metrics:
            return -0.03  # 缺乏协调数据的成本
        
        cache_hit_rate = cache_metrics.get('hit_rate', 0.0)
        migration_success_rate = migration_metrics.get('success_rate', 0.0)
        avg_delay = system_metrics.get('avg_task_delay', 1.0)
        
        # 系统协调不良成本
        coordination_cost = -0.1  # 基础协调管理成本
        
        # 🔧 基于系统协调效果的成本减免（但仍为负值）
        if cache_hit_rate > 0.7 and migration_success_rate > 0.7:
            # 双系统协调良好时，减免部分成本
            coordination_cost *= 0.5  # 减免50%协调成本
        
        if avg_delay < 0.3:
            # 低延迟时，证明协调有效，进一步减免成本
            coordination_cost *= 0.7  # 再减免30%
        
        return coordination_cost  # 始终为负值
    
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
