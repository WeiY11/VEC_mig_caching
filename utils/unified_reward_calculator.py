#!/usr/bin/env python3
"""
统一奖励计算器，供所有单智能体强化学习算法使用。

核心理念是成本最小化：更低的延迟和更低的能耗会带来更高（更少负值）的奖励。
某些算法（如SAC）期望正向奖励，因此我们为这种情况保留了一个小的可选奖励。

Unified reward calculator used by all single-agent RL algorithms.
The philosophy is cost-minimisation: lower latency and lower energy
lead to higher (less negative) rewards. Some algorithms (SAC) expect
positive rewards, so we keep a small optional bonus for that case.
"""

from __future__ import annotations

from typing import Dict, Optional, List

import numpy as np

from config import config


class UnifiedRewardCalculator:
    """
    可复用的奖励计算器，用于单智能体训练器。
    
    该类实现了统一的奖励计算逻辑，支持不同算法（如SAC、TD3等）的特定需求。
    采用成本最小化方法：延迟越低、能耗越低、任务丢弃越少，奖励越高。
    
    Reusable reward calculator for the single-agent trainers.
    """

    def __init__(self, algorithm: str = "general") -> None:
        """
        初始化统一奖励计算器。
        
        Args:
            algorithm: 算法名称，用于特定算法的调整（如"SAC"、"TD3"等）
                      不同算法可能有不同的归一化因子和奖励范围
        """
        self.algorithm = algorithm.upper()

        # 从配置中获取核心权重参数
        # Core weights taken from configuration.
        self.weight_delay = float(config.rl.reward_weight_delay)  # 延迟权重
        self.weight_energy = float(config.rl.reward_weight_energy)  # 能耗权重
        self.penalty_dropped = float(config.rl.reward_penalty_dropped)  # 任务丢弃惩罚
        self.weight_cache = float(getattr(config.rl, "reward_weight_cache", 0.0))  # 缓存权重
        self.weight_migration = float(getattr(config.rl, "reward_weight_migration", 0.0))  # 迁移权重

        # 归一化任务优先级权重（如果存在）
        # Normalise priority weights if they exist.
        priority_weights = getattr(config, "task", None)
        priority_weights = getattr(priority_weights, "type_priority_weights", None)
        if isinstance(priority_weights, dict) and priority_weights:
            # 计算权重总和并归一化
            total = sum(float(v) for v in priority_weights.values()) or 1.0
            self.task_priority_weights = {
                int(task_type): float(value) / total
                for task_type, value in priority_weights.items()
            }
        else:
            # 默认所有任务类型权重相等
            self.task_priority_weights = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}

        # 归一化因子（200毫秒时隙，约1000焦耳典型能耗）
        # Normalisation factors (200 ms slot, ~1000 J typical energy).
        self.delay_normalizer = 0.2  # 延迟归一化因子（秒）
        self.energy_normalizer = 1000.0  # 能耗归一化因子（焦耳）
        
        # SAC算法使用不同的归一化参数
        if self.algorithm == "SAC":
            self.delay_normalizer = 0.25
            self.energy_normalizer = 1200.0

        # 设置奖励裁剪范围，防止奖励值过大或过小
        if self.algorithm == "SAC":
            self.reward_clip_range = (-15.0, 3.0)  # SAC期望较小的奖励范围
        else:
            self.reward_clip_range = (-80.0, -0.005)  # 其他算法使用负值范围

        print(f"[OK] Unified reward calculator ({self.algorithm})")
        print(
            f"   Core weights: delay={self.weight_delay:.2f}, "
            f"energy={self.weight_energy:.2f}, drop={self.penalty_dropped:.3f}"
        )
        print(
            f"   Normalisation: delay/{self.delay_normalizer:.2f}, "
            f"energy/{self.energy_normalizer:.0f}"
        )

    # ------------------------------------------------------------------ #
    # 辅助方法 / Helpers

    @staticmethod
    def _safe_float(value: Optional[float], default: float = 0.0) -> float:
        """
        安全地将值转换为浮点数。
        
        Args:
            value: 待转换的值
            default: 转换失败时的默认值
            
        Returns:
            转换后的浮点数或默认值
        """
        if value is None:
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _safe_int(value: Optional[int], default: int = 0) -> int:
        """
        安全地将值转换为整数。
        
        Args:
            value: 待转换的值
            default: 转换失败时的默认值
            
        Returns:
            转换后的整数或默认值
        """
        if value is None:
            return default
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_float_list(source) -> List[float]:
        """
        将输入转换为浮点数列表。
        
        支持numpy数组、列表、元组等多种输入类型。
        如果某个元素无法转换，则使用0.0作为默认值。
        
        Args:
            source: 待转换的数据源
            
        Returns:
            浮点数列表
        """
        if isinstance(source, np.ndarray):
            iterable = source.tolist()
        elif isinstance(source, (list, tuple)):
            iterable = list(source)
        else:
            return []
        result: List[float] = []
        for item in iterable:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                result.append(0.0)
        return result

    # ------------------------------------------------------------------ #
    # 公共API / Public API

    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> float:
        """
        为强化学习智能体计算标量奖励。
        
        奖励计算基于系统性能指标，采用成本最小化原则：
        - 延迟越低，奖励越高
        - 能耗越低，奖励越高
        - 任务丢弃越少，奖励越高
        - 可选：缓存命中率越高，奖励越高
        - 可选：迁移成本越低，奖励越高
        
        Args:
            system_metrics: 系统性能指标字典，包含：
                - avg_task_delay: 平均任务延迟（秒）
                - total_energy_consumption: 总能耗（焦耳）
                - dropped_tasks: 丢弃的任务数量
                - task_completion_rate: 任务完成率
            cache_metrics: 可选的缓存指标字典
                - miss_rate: 缓存未命中率
            migration_metrics: 可选的迁移指标字典
                - migration_cost: 迁移成本
                
        Returns:
            标量奖励值，范围由reward_clip_range定义
        """

        # 从系统指标中提取关键性能数据，确保非负值
        avg_delay = max(0.0, self._safe_float(system_metrics.get("avg_task_delay")))
        total_energy = max(
            0.0, self._safe_float(system_metrics.get("total_energy_consumption"))
        )
        dropped_tasks = max(0, self._safe_int(system_metrics.get("dropped_tasks")))
        completion_rate = max(
            0.0, self._safe_float(system_metrics.get("task_completion_rate"))
        )

        # 归一化核心成本指标
        # Normalised core costs.
        norm_delay = avg_delay / max(1e-9, self.delay_normalizer)
        norm_energy = total_energy / max(1e-9, self.energy_normalizer)

        # 计算总成本：延迟成本 + 能耗成本 + 任务丢弃惩罚
        total_cost = self.weight_delay * norm_delay + self.weight_energy * norm_energy
        total_cost += self.penalty_dropped * dropped_tasks

        # 可选的缓存和迁移惩罚
        # Optional cache/migration penalties.
        if cache_metrics and self.weight_cache:
            miss_rate = self._safe_float(cache_metrics.get("miss_rate"), 0.0)
            total_cost += self.weight_cache * miss_rate

        if migration_metrics and self.weight_migration:
            migration_cost = self._safe_float(migration_metrics.get("migration_cost"), 0.0)
            total_cost += self.weight_migration * migration_cost

        # 当延迟/能耗超过配置的软限制时，添加温和的惩罚
        # Add gentle penalties when latency/energy exceed configured soft limits.
        latency_target = self._safe_float(getattr(config.rl, "latency_target", 0.0), 0.0)
        latency_tolerance = self._safe_float(
            getattr(config.rl, "latency_upper_tolerance", latency_target), latency_target
        )
        if latency_tolerance > 0 and avg_delay > latency_tolerance:
            # 超出容忍范围的延迟会产生额外惩罚
            total_cost += self.weight_delay * (avg_delay - latency_tolerance) / latency_tolerance

        energy_target = self._safe_float(getattr(config.rl, "energy_target", 0.0), 0.0)
        energy_tolerance = self._safe_float(
            getattr(config.rl, "energy_upper_tolerance", energy_target), energy_target
        )
        if energy_tolerance > 0 and total_energy > energy_tolerance:
            # 超出容忍范围的能耗会产生额外惩罚
            total_cost += self.weight_energy * (total_energy - energy_tolerance) / energy_tolerance

        # SAC期望正向奖励 -> 允许小的奖励加成
        # SAC expects positive reward -> allow small bonus.
        bonus = 0.0
        if self.algorithm == "SAC":
            # 延迟低于归一化阈值时给予奖励
            if avg_delay < self.delay_normalizer:
                bonus += (self.delay_normalizer - avg_delay) * 3.0
            # 任务完成率高于95%时给予额外奖励
            if completion_rate > 0.95:
                bonus += (completion_rate - 0.95) * 15.0
            reward = bonus - total_cost
        else:
            # 其他算法使用负成本作为奖励
            reward = -total_cost

        # 裁剪奖励值到指定范围，防止梯度爆炸或消失
        reward = float(np.clip(reward, *self.reward_clip_range))
        return reward

    def get_reward_breakdown(self, system_metrics: Dict) -> str:
        """
        获取奖励组成的人类可读分解报告。
        
        用于调试和分析，显示奖励的各个组成部分及其贡献。
        
        Args:
            system_metrics: 系统性能指标字典
            
        Returns:
            格式化的多行字符串，包含奖励详细分解
        """
        avg_delay = self._safe_float(system_metrics.get("avg_task_delay"), 0.0)
        total_energy = self._safe_float(system_metrics.get("total_energy_consumption"), 0.0)
        dropped_tasks = self._safe_int(system_metrics.get("dropped_tasks"), 0)
        completion_rate = self._safe_float(system_metrics.get("task_completion_rate"), 0.0)

        reward = self.calculate_reward(system_metrics)
        lines = [
            f"Reward report ({self.algorithm}):",
            f"  total reward        : {reward:.3f}",
            f"  latency             : {avg_delay:.3f}s (norm {avg_delay / max(1e-9, self.delay_normalizer):.3f})",
            f"  energy              : {total_energy:.2f}J (norm {total_energy / max(1e-9, self.energy_normalizer):.3f})",
            f"  completion rate     : {completion_rate:.1%}",
            f"  dropped tasks       : {dropped_tasks}",
            f"  core cost estimate  : {self.weight_delay * avg_delay / max(1e-9, self.delay_normalizer) + self.weight_energy * total_energy / max(1e-9, self.energy_normalizer):.3f}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
# 便捷的单例对象，在整个项目中使用
# Convenience singletons used across the project.

_general_reward_calculator = UnifiedRewardCalculator(algorithm="general")
_sac_reward_calculator = UnifiedRewardCalculator(algorithm="sac")


def calculate_unified_reward(
    system_metrics: Dict,
    cache_metrics: Optional[Dict] = None,
    migration_metrics: Optional[Dict] = None,
    algorithm: str = "general",
) -> float:
    """
    统一奖励计算的便捷函数。
    
    根据指定算法选择相应的奖励计算器，计算并返回奖励值。
    
    Args:
        system_metrics: 系统性能指标
        cache_metrics: 可选的缓存指标
        migration_metrics: 可选的迁移指标
        algorithm: 算法名称（"SAC"或"general"）
        
    Returns:
        计算得到的奖励值
    """
    calculator = _sac_reward_calculator if algorithm.upper() == "SAC" else _general_reward_calculator
    return calculator.calculate_reward(system_metrics, cache_metrics, migration_metrics)


def get_reward_breakdown(system_metrics: Dict, algorithm: str = "general") -> str:
    """
    获取奖励分解报告的便捷函数。
    
    Args:
        system_metrics: 系统性能指标
        algorithm: 算法名称（"SAC"或"general"）
        
    Returns:
        格式化的奖励分解报告字符串
    """
    calculator = _sac_reward_calculator if algorithm.upper() == "SAC" else _general_reward_calculator
    return calculator.get_reward_breakdown(system_metrics)


# ---------------------------------------------------------------------- #
# 向后兼容的辅助函数名称
# Backwards-compatible helper names.

def calculate_enhanced_reward(
    system_metrics: Dict,
    cache_metrics: Optional[Dict] = None,
    migration_metrics: Optional[Dict] = None,
) -> float:
    """
    增强奖励计算（向后兼容）。
    
    这是calculate_unified_reward的别名，使用"general"算法。
    保留此函数以确保与旧代码的兼容性。
    
    Args:
        system_metrics: 系统性能指标
        cache_metrics: 可选的缓存指标
        migration_metrics: 可选的迁移指标
        
    Returns:
        计算得到的奖励值
    """
    return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, "general")


def calculate_sac_reward(system_metrics: Dict) -> float:
    """
    SAC算法专用奖励计算（向后兼容）。
    
    为SAC算法提供正向奖励空间的便捷函数。
    
    Args:
        system_metrics: 系统性能指标
        
    Returns:
        计算得到的奖励值（可能为正值）
    """
    return calculate_unified_reward(system_metrics, algorithm="sac")


def calculate_simple_reward(system_metrics: Dict) -> float:
    """
    简单奖励计算（向后兼容）。
    
    这是calculate_unified_reward的简化版本，使用"general"算法。
    
    Args:
        system_metrics: 系统性能指标
        
    Returns:
        计算得到的奖励值
    """
    return calculate_unified_reward(system_metrics, algorithm="general")

