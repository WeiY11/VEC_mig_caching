#!/usr/bin/env python3
"""
TD3 + SAC奖励bonus机制

核心思想：
- 保持TD3的确定性策略和稳定性
- 借鉴SAC的正向奖励机制
- 鼓励低延迟和高缓存命中

使用：
    在train_single_agent.py中将algorithm改为'TD3_BONUS'
"""

import numpy as np
from typing import Dict, Optional
from single_agent.td3 import TD3Environment


class TD3BonusEnvironment(TD3Environment):
    """TD3 + 奖励bonus机制"""
    
    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2):
        super().__init__(num_vehicles, num_rsus, num_uavs)
        
        print("🎁 TD3 Bonus版本已启用")
        print("   - 低延迟奖励")
        print("   - 高缓存命中奖励")
        print("   - 高完成率奖励")
    
    def calculate_reward(self, system_metrics: Dict, 
                        cache_metrics: Optional[Dict] = None,
                        migration_metrics: Optional[Dict] = None) -> float:
        """
        增强奖励函数 - 借鉴SAC的bonus机制
        
        核心公式：
        cost = 2.0 × delay + 1.2 × energy + 0.02 × dropped
        bonus = 低延迟奖励 + 高缓存奖励 + 高完成率奖励
        reward = bonus - cost
        """
        from utils.unified_reward_calculator import UnifiedRewardCalculator
        
        # 基础成本（与标准TD3相同）
        calc = UnifiedRewardCalculator(algorithm="general")
        
        # 提取指标
        avg_delay = max(0.0, float(system_metrics.get('avg_task_delay', 0)))
        total_energy = max(0.0, float(system_metrics.get('total_energy_consumption', 0)))
        dropped_tasks = max(0, int(system_metrics.get('dropped_tasks', 0)))
        completion_rate = max(0.0, float(system_metrics.get('task_completion_rate', 0)))
        cache_hit_rate = max(0.0, float(system_metrics.get('cache_hit_rate', 0)))
        
        # 归一化
        norm_delay = avg_delay / 0.2
        norm_energy = total_energy / 1000.0
        
        # 核心成本
        core_cost = 2.0 * norm_delay + 1.2 * norm_energy + 0.02 * dropped_tasks
        
        # 🎁 Bonus机制（借鉴SAC）
        bonus = 0.0
        
        # 1. 低延迟奖励
        if avg_delay < 0.3:
            bonus += (0.3 - avg_delay) * 5.0
        
        # 2. 高缓存命中奖励（关键！）
        if cache_hit_rate > 0.4:
            bonus += (cache_hit_rate - 0.4) * 8.0
        
        # 3. 高完成率奖励
        if completion_rate > 0.9:
            bonus += (completion_rate - 0.9) * 10.0
        
        # 最终奖励
        reward = bonus - core_cost
        
        # 裁剪范围（比SAC稍宽）
        reward = float(np.clip(reward, -20.0, 5.0))
        
        return reward

