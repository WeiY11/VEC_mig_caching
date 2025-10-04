#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一奖励计算器 (Unified Reward Calculator)
适用于所有单智能体DRL算法（DDPG, TD3, DQN, PPO, SAC）

设计原则：
1. 核心优化目标：时延 + 能耗双目标加权和
2. 辅助约束：通过丢弃任务惩罚保证完成率
3. 成本最小化：奖励严格为负值（成本）
4. 算法适配：SAC保留轻微调整以适应最大熵框架
"""

import numpy as np
from typing import Dict, Optional
from config import config


class UnifiedRewardCalculator:
    """
    统一奖励计算器 - 所有算法共享核心逻辑
    """

    def __init__(self, algorithm: str = "general"):
        """
        初始化统一奖励计算器
        
        Args:
            algorithm: 算法类型 ("general", "sac")
                - "general": 通用版本（DDPG, TD3, DQN, PPO）
                - "sac": SAC专用版本（考虑最大熵特性）
        """
        self.algorithm = algorithm.upper()
        
        # 从配置加载核心权重
        self.weight_delay = config.rl.reward_weight_delay      # 默认 2.0
        self.weight_energy = config.rl.reward_weight_energy    # 默认 1.2
        self.penalty_dropped = config.rl.reward_penalty_dropped # 默认 0.02
        
        # 🎯 核心设计：归一化因子（确保时延和能耗在相同数量级）
        # 目标：delay=0.2s 和 energy=600J 归一化后贡献相当
        self.delay_normalizer = 1.0      # 0.2s → 0.2
        self.energy_normalizer = 600.0   # 🔧 调整：突出能耗反馈
        
        # 🔧 SAC专用调整：更激进的归一化以平衡探索
        if self.algorithm == "SAC":
            self.delay_normalizer = 0.3      # 0.2s → 0.67（更敏感）
            self.energy_normalizer = 1500.0  # 1000J → 0.67（更敏感）
        
        # 奖励范围限制
        if self.algorithm == "SAC":
            # SAC允许小幅正值奖励（最大熵需要明确激励）
            self.reward_clip_range = (-15.0, 3.0)
        else:
            # 通用版本：纯成本最小化
            self.reward_clip_range = (-25.0, -0.01)
        
        print(f"[OK] 统一奖励计算器初始化 ({self.algorithm})")
        print(f"   核心权重: Delay={self.weight_delay:.1f}, Energy={self.weight_energy:.1f}")
        print(f"   归一化: Delay/{self.delay_normalizer:.1f}, Energy/{self.energy_normalizer:.0f}")
        print(f"   奖励范围: {self.reward_clip_range}")
        print(f"   优化目标: 最小化 {self.weight_delay}*Delay + {self.weight_energy}*Energy")

    def calculate_reward(self, 
                        system_metrics: Dict,
                        cache_metrics: Optional[Dict] = None,
                        migration_metrics: Optional[Dict] = None) -> float:
        """
        计算统一奖励（支持缓存和迁移指标，但不影响核心奖励）
        
        Args:
            system_metrics: 系统性能指标
            cache_metrics: 缓存指标（可选，用于未来扩展）
            migration_metrics: 迁移指标（可选，用于未来扩展）
        
        Returns:
            reward: 标量奖励值
        """
        # 1️⃣ 提取核心指标（安全处理None值）
        def safe_float(value, default=0.0):
            """安全转换为float，处理None和异常值"""
            if value is None:
                return default
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return default
        
        def safe_int(value, default=0):
            """安全转换为int，处理None和异常值"""
            if value is None:
                return default
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default
        
        avg_delay = safe_float(system_metrics.get('avg_task_delay'), 0.0)
        total_energy = safe_float(system_metrics.get('total_energy_consumption'), 0.0)
        dropped_tasks = safe_int(system_metrics.get('dropped_tasks'), 0)
        
        # 2️⃣ 归一化
        norm_delay = avg_delay / self.delay_normalizer
        norm_energy = total_energy / self.energy_normalizer
        
        # 3️⃣ 计算基础成本（双目标加权和）
        base_cost = (self.weight_delay * norm_delay + 
                     self.weight_energy * norm_energy)
        
        # 4️⃣ 丢弃任务惩罚（保证完成率约束）
        dropped_penalty = self.penalty_dropped * dropped_tasks
        
        # 5️⃣ 自适应阈值惩罚（防止极端情况）
        delay_threshold_penalty = 0.0
        energy_threshold_penalty = 0.0
        
        if self.algorithm == "SAC":
            # SAC：更激进的阈值惩罚
            if avg_delay > 0.25:
                delay_threshold_penalty = (avg_delay - 0.25) * 8.0
            if total_energy > 2000:
                energy_threshold_penalty = (total_energy - 2000) / 1000.0
        else:
            # 通用算法：温和的阈值惩罚
            if avg_delay > 0.30:
                delay_threshold_penalty = (avg_delay - 0.30) * 5.0
            if total_energy > 3000:
                energy_threshold_penalty = (total_energy - 3000) / 1500.0
        
        # 6️⃣ 总成本
        total_cost = (base_cost + 
                     dropped_penalty + 
                     delay_threshold_penalty + 
                     energy_threshold_penalty)
        
        # 7️⃣ SAC专用：正向激励机制（最大熵框架需要明确"好"的信号）
        bonus = 0.0
        if self.algorithm == "SAC":
            completion_rate = safe_float(system_metrics.get('task_completion_rate'), 0.0)
            
            # 延迟优秀奖励
            if avg_delay < 0.20:
                bonus += (0.20 - avg_delay) * 3.0
            
            # 完成率优秀奖励
            if completion_rate > 0.95:
                bonus += (completion_rate - 0.95) * 15.0
        
        # 8️⃣ 最终奖励
        if self.algorithm == "SAC":
            reward = bonus - total_cost  # SAC: bonus可能为正
        else:
            reward = -total_cost  # 通用: 纯负值成本
        
        # 9️⃣ 裁剪到合理范围
        clipped_reward = np.clip(reward, *self.reward_clip_range)
        
        return clipped_reward
    
    def get_reward_breakdown(self, system_metrics: Dict) -> str:
        """获取奖励分解的可读报告"""
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return default
        
        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default
        
        avg_delay = safe_float(system_metrics.get('avg_task_delay'), 0.0)
        total_energy = safe_float(system_metrics.get('total_energy_consumption'), 0.0)
        dropped_tasks = safe_int(system_metrics.get('dropped_tasks'), 0)
        completion_rate = safe_float(system_metrics.get('task_completion_rate'), 0.0)
        
        reward = self.calculate_reward(system_metrics)
        
        breakdown = f"""
奖励分解报告 ({self.algorithm}):
├── 总奖励: {reward:.3f}
├── 核心指标:
│   ├── 时延: {avg_delay:.3f}s (归一化: {avg_delay/self.delay_normalizer:.3f})
│   ├── 能耗: {total_energy:.1f}J (归一化: {total_energy/self.energy_normalizer:.3f})
│   └── 完成率: {completion_rate:.1%}
├── 成本贡献:
│   ├── 时延成本: {self.weight_delay * avg_delay/self.delay_normalizer:.3f}
│   ├── 能耗成本: {self.weight_energy * total_energy/self.energy_normalizer:.3f}
│   └── 丢弃惩罚: {self.penalty_dropped * dropped_tasks:.3f} ({dropped_tasks}个任务)
└── 优化方向: {'最大化奖励（含bonus）' if self.algorithm == 'SAC' else '最小化成本'}
        """
        
        return breakdown.strip()


# ==================== 全局实例和便捷接口 ====================

# 通用版本（DDPG, TD3, DQN, PPO）
_general_reward_calculator = UnifiedRewardCalculator(algorithm="general")

# SAC专用版本
_sac_reward_calculator = UnifiedRewardCalculator(algorithm="sac")


def calculate_unified_reward(system_metrics: Dict,
                             cache_metrics: Optional[Dict] = None,
                             migration_metrics: Optional[Dict] = None,
                             algorithm: str = "general") -> float:
    """
    统一奖励计算接口（所有算法调用）
    
    Args:
        system_metrics: 系统性能指标
        cache_metrics: 缓存指标（可选）
        migration_metrics: 迁移指标（可选）
        algorithm: 算法类型 ("general" 或 "sac")
    
    Returns:
        reward: 标量奖励值
    """
    if algorithm.upper() == "SAC":
        calculator = _sac_reward_calculator
    else:
        calculator = _general_reward_calculator
    
    return calculator.calculate_reward(system_metrics, cache_metrics, migration_metrics)


def get_reward_breakdown(system_metrics: Dict, algorithm: str = "general") -> str:
    """获取奖励分解报告"""
    if algorithm.upper() == "SAC":
        calculator = _sac_reward_calculator
    else:
        calculator = _general_reward_calculator
    
    return calculator.get_reward_breakdown(system_metrics)


# ==================== 向后兼容接口 ====================

def calculate_enhanced_reward(system_metrics: Dict,
                             cache_metrics: Optional[Dict] = None,
                             migration_metrics: Optional[Dict] = None) -> float:
    """向后兼容接口（供现有代码调用）"""
    return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, "general")


def calculate_sac_reward(system_metrics: Dict) -> float:
    """SAC专用接口（向后兼容）"""
    return calculate_unified_reward(system_metrics, algorithm="sac")


def calculate_simple_reward(system_metrics: Dict) -> float:
    """简化接口（向后兼容）"""
    return calculate_unified_reward(system_metrics, algorithm="general")

