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
        self.weight_joint = float(getattr(config.rl, "reward_weight_joint", 0.05))  # 缓存-迁移联动权重
        self.completion_target = float(getattr(config.rl, "completion_target", 0.95))
        self.weight_completion_gap = float(getattr(config.rl, "reward_weight_completion_gap", 0.0))
        self.weight_loss_ratio = float(getattr(config.rl, "reward_weight_loss_ratio", 0.0))
        self.cache_pressure_threshold = float(getattr(config.rl, "cache_pressure_threshold", 0.85))
        self.weight_cache_pressure = float(getattr(config.rl, "reward_weight_cache_pressure", 0.0))
        self.weight_queue_overload = float(getattr(config.rl, "reward_weight_queue_overload", 0.0))
        self.weight_remote_reject = float(getattr(config.rl, "reward_weight_remote_reject", 0.0))
        self.latency_target = float(getattr(config.rl, "latency_target", 0.4))
        self.energy_target = float(getattr(config.rl, "energy_target", 1200.0))
        self.latency_tolerance = float(getattr(config.rl, "latency_upper_tolerance", self.latency_target * 2.0))
        self.energy_tolerance = float(getattr(config.rl, "energy_upper_tolerance", self.energy_target * 1.5))

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

        # 归一化因子（基于典型延迟0.2s和能耗1000J）
        # Normalisation factors (based on typical delay 0.2s and energy 1000J).
        # 注：时隙已改为100ms，但归一化因子基于实际延迟范围，无需改变
        self.delay_normalizer = 0.2  # 延迟归一化因子（秒）- 典型延迟参考值
        self.energy_normalizer = 1000.0  # 能耗归一化因子（焦耳）
        self.delay_bonus_scale = max(1e-6, self.latency_target)
        self.energy_bonus_scale = max(1e-6, self.energy_target)
        
        # SAC算法使用不同的归一化参数
        if self.algorithm == "SAC":
            self.delay_normalizer = 0.25
            self.energy_normalizer = 1200.0

        norm_cfg = getattr(config, "normalization", None)
        if norm_cfg is not None:
            self.latency_target = float(getattr(norm_cfg, "delay_reference", self.latency_target))
            self.latency_tolerance = float(getattr(norm_cfg, "delay_upper_reference", self.latency_tolerance))
            self.energy_target = float(getattr(norm_cfg, "energy_reference", self.energy_target))
            self.energy_tolerance = float(getattr(norm_cfg, "energy_upper_reference", self.energy_tolerance))
            self.delay_normalizer = float(
                getattr(norm_cfg, "delay_normalizer_value", self.delay_normalizer)
            )
            self.energy_normalizer = float(
                getattr(norm_cfg, "energy_normalizer_value", self.energy_normalizer)
            )
            self.delay_bonus_scale = max(1e-6, self.latency_target)
            self.energy_bonus_scale = max(1e-6, self.energy_target)

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
        data_loss_ratio = max(
            0.0, self._safe_float(system_metrics.get("data_loss_ratio_bytes"))
        )
        cache_utilization = max(
            0.0, self._safe_float(system_metrics.get("cache_utilization"))
        )
        queue_overload_events = max(
            0.0, self._safe_float(system_metrics.get("queue_overload_events"))
        )
        remote_rejection_rate = max(
            0.0, self._safe_float(system_metrics.get("remote_rejection_rate"))
        )

        # ========== 核心归一化：统一使用目标值归一化 ==========
        # Objective = w_T × (delay/target) + w_E × (energy/target)
        # 这样可以确保两个指标在同一尺度上，权重才有意义
        
        norm_delay = avg_delay / max(self.latency_target, 1e-6)
        norm_energy = total_energy / max(self.energy_target, 1e-6)

        # 计算核心成本：归一化后的加权和
        core_cost = self.weight_delay * norm_delay + self.weight_energy * norm_energy
        
        # 任务丢弃惩罚（轻微惩罚，主要用于保证完成率）
        drop_penalty = self.penalty_dropped * dropped_tasks
        
        # 总成本
        total_cost = core_cost + drop_penalty

        # 完成率差距惩罚
        if self.weight_completion_gap > 0.0:
            completion_gap = max(0.0, self.completion_target - completion_rate)
            total_cost += self.weight_completion_gap * completion_gap

        # 数据丢失率惩罚
        if self.weight_loss_ratio > 0.0:
            total_cost += self.weight_loss_ratio * data_loss_ratio

        # 缓存压力惩罚（超过阈值才惩罚）
        if self.weight_cache_pressure > 0.0 and cache_utilization > self.cache_pressure_threshold:
            total_cost += self.weight_cache_pressure * (cache_utilization - self.cache_pressure_threshold)

        # 队列过载事件惩罚
        if self.weight_queue_overload > 0.0 and queue_overload_events > 0.0:
            total_cost += self.weight_queue_overload * queue_overload_events

        if self.weight_remote_reject > 0.0 and remote_rejection_rate > 0.0:
            total_cost += self.weight_remote_reject * remote_rejection_rate

        # ========== 辅助指标（可选，权重较小）==========
        # 注意：缓存和迁移是手段，不是优化目标，所以权重设为0
        
        # 可选的缓存惩罚（通常权重为0）
        if cache_metrics and self.weight_cache > 0:
            miss_rate = np.clip(self._safe_float(cache_metrics.get("miss_rate"), 0.0), 0.0, 1.0)
            total_cost += self.weight_cache * miss_rate

        # 可选的迁移惩罚（通常权重为0）
        if migration_metrics and self.weight_migration > 0:
            migration_cost = self._safe_float(migration_metrics.get("migration_cost"), 0.0)
            total_cost += self.weight_migration * migration_cost

        if cache_metrics and migration_metrics and self.weight_joint > 0:
            hit_rate = np.clip(self._safe_float(cache_metrics.get("hit_rate"), 0.0), 0.0, 1.0)
            effectiveness = np.clip(self._safe_float(migration_metrics.get("effectiveness"), 0.0), 0.0, 1.0)
            joint_bonus = hit_rate * effectiveness

            cache_joint = cache_metrics.get("joint_params", {}) if isinstance(cache_metrics, dict) else {}
            migration_joint = migration_metrics.get("joint_params", {}) if isinstance(migration_metrics, dict) else {}
            prefetch_lead = self._safe_float(cache_joint.get("prefetch_lead_time"), 0.0)
            backoff_factor = np.clip(self._safe_float(migration_joint.get("migration_backoff"), 0.0), 0.0, 1.0)

            total_requests = max(1.0, self._safe_float(cache_metrics.get("total_requests"), 1.0))
            prefetch_events = max(0.0, self._safe_float(cache_metrics.get("prefetch_events"), 0.0))
            prefetch_ratio = np.clip(prefetch_events / total_requests, 0.0, 1.0)

            coupling_penalty = max(0.0, 0.3 - prefetch_ratio) * 0.5
            coupling_penalty += abs(prefetch_lead - 1.5) * 0.05
            coupling_penalty += backoff_factor * 0.1

            total_cost -= self.weight_joint * joint_bonus
            total_cost += self.weight_joint * coupling_penalty

        # ========== 计算最终奖励 ==========
        # 奖励 = -成本（成本越低，奖励越高）
        
        if self.algorithm == "SAC":
            # SAC需要适度正向奖励，添加基础奖励
            base_reward = 5.0  # 基础奖励
            # 完成率高时额外奖励
            completion_bonus = 0.0
            if completion_rate > 0.95:
                completion_bonus = (completion_rate - 0.95) * 10.0
            reward = base_reward + completion_bonus - total_cost
        else:
            # TD3/DDPG/PPO等算法：直接使用负成本作为奖励
            reward = -total_cost

        # 裁剪奖励值（归一化后范围更小）
        # 预期范围：-10 到 0（如果一切正常）
        if self.algorithm == "SAC":
            reward = float(np.clip(reward, -15.0, 10.0))
        else:
            reward = float(np.clip(reward, -20.0, 0.0))
        
        return reward

    def update_targets(
        self,
        latency_target: Optional[float] = None,
        energy_target: Optional[float] = None,
    ) -> None:
        """动态更新目标值，使奖励函数可以在训练中自适应拓扑变化。"""
        if latency_target is not None:
            self.latency_target = float(latency_target)
            self.latency_tolerance = float(
                getattr(config.rl, "latency_upper_tolerance", self.latency_target * 2.0)
            )
            self.delay_bonus_scale = max(1e-6, self.latency_target)
        if energy_target is not None:
            self.energy_target = float(energy_target)
            self.energy_tolerance = float(
                getattr(config.rl, "energy_upper_tolerance", self.energy_target * 1.5)
            )
            self.energy_bonus_scale = max(1e-6, self.energy_target)

    def get_reward_breakdown(self, system_metrics: Dict) -> str:
        """
        获取奖励组成的人类可读分解报告。
        
        用于调试和分析，显示奖励的各个组成部分及其贡献。
        
        Args:
            system_metrics: 系统性能指标字典
            
        Returns:
            格式化的多行字符串，包含奖励详细分解
        """
        avg_delay = max(0.0, self._safe_float(system_metrics.get("avg_task_delay"), 0.0))
        total_energy = max(0.0, self._safe_float(system_metrics.get("total_energy_consumption"), 0.0))
        dropped_tasks = max(0, self._safe_int(system_metrics.get("dropped_tasks"), 0))
        completion_rate = max(0.0, self._safe_float(system_metrics.get("task_completion_rate"), 0.0))

        # 计算归一化值
        norm_delay = avg_delay / max(self.latency_target, 1e-6)
        norm_energy = total_energy / max(self.energy_target, 1e-6)
        
        # 计算加权成本
        weighted_delay = self.weight_delay * norm_delay
        weighted_energy = self.weight_energy * norm_energy
        core_cost = weighted_delay + weighted_energy
        
        reward = self.calculate_reward(system_metrics)
        lines = [
            f"Reward report ({self.algorithm}):",
            f"  Total Reward        : {reward:.3f}",
            f"  ----------------------------------------",
            f"  Delay               : {avg_delay:.4f}s (target: {self.latency_target}s)",
            f"  Normalized Delay    : {norm_delay:.4f}",
            f"  Weighted Delay      : {weighted_delay:.4f} (w={self.weight_delay})",
            f"  ----------------------------------------",
            f"  Energy              : {total_energy:.2f}J (target: {self.energy_target:.0f}J)",
            f"  Normalized Energy   : {norm_energy:.4f}",
            f"  Weighted Energy     : {weighted_energy:.4f} (w={self.weight_energy})",
            f"  ----------------------------------------",
            f"  Core Cost (D+E)     : {core_cost:.4f}",
            f"  Dropped Tasks       : {dropped_tasks} (penalty: {self.penalty_dropped * dropped_tasks:.4f})",
            f"  Completion Rate     : {completion_rate:.1%}",
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


def update_reward_targets(
    latency_target: Optional[float] = None,
    energy_target: Optional[float] = None,
) -> None:
    """
    动态更新全局奖励目标，确保单例计算器与全局config保持同步。
    """
    if latency_target is not None:
        config.rl.latency_target = float(latency_target)
    if energy_target is not None:
        config.rl.energy_target = float(energy_target)
    _general_reward_calculator.update_targets(latency_target, energy_target)
    _sac_reward_calculator.update_targets(latency_target, energy_target)


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

