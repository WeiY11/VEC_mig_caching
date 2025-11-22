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

from dataclasses import dataclass, field
from typing import Dict, Optional, List

import numpy as np

from config import config


@dataclass
class RewardMetrics:
    """提取后的原始指标，便于后续统一计算。"""
    avg_delay: float = 0.0
    total_energy: float = 0.0
    dropped_tasks: int = 0
    completion_rate: float = 0.0
    data_loss_ratio: float = 0.0
    cache_utilization: float = 0.0
    queue_overload_events: float = 0.0
    remote_rejection_rate: float = 0.0
    rsu_offload_ratio: float = 0.0
    uav_offload_ratio: float = 0.0
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    migration_cost: float = 0.0
    migration_effectiveness: float = 0.0
    prefetch_events: float = 0.0
    total_cache_requests: float = 1.0
    prefetch_lead: float = 0.0
    migration_backoff: float = 0.0


@dataclass
class RewardComponents:
    """分解后的成本和奖励组成，便于调试与扩展。"""
    norm_delay: float
    norm_energy: float
    core_cost: float
    drop_penalty: float = 0.0
    completion_gap_penalty: float = 0.0
    data_loss_penalty: float = 0.0
    cache_pressure_penalty: float = 0.0
    queue_penalty: float = 0.0
    remote_reject_penalty: float = 0.0
    offload_bonus: float = 0.0
    cache_penalty: float = 0.0
    cache_bonus: float = 0.0
    migration_penalty: float = 0.0
    joint_coupling_penalty: float = 0.0
    joint_bonus: float = 0.0
    total_cost: float = 0.0
    reward_pre_clip: float = 0.0
    reward: float = 0.0


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
        self.weight_cache_bonus = float(getattr(config.rl, "reward_weight_cache_bonus", 0.0))
        self.weight_migration = float(getattr(config.rl, "reward_weight_migration", 0.0))  # 迁移权重
        self.weight_joint = float(getattr(config.rl, "reward_weight_joint", 0.05))  # 缓存-迁移联动权重
        # 🔧 新增：远程卸载激励权重
        self.weight_offload_bonus = float(getattr(config.rl, "reward_weight_offload_bonus", 0.15))  # 边缘计算利用奖励
        self.completion_target = float(getattr(config.rl, "completion_target", 0.95))
        self.weight_completion_gap = float(getattr(config.rl, "reward_weight_completion_gap", 0.0))
        self.weight_loss_ratio = float(getattr(config.rl, "reward_weight_loss_ratio", 0.0))
        self.cache_pressure_threshold = float(getattr(config.rl, "cache_pressure_threshold", 0.85))
        self.weight_cache_pressure = float(getattr(config.rl, "reward_weight_cache_pressure", 0.0))
        self.weight_queue_overload = float(getattr(config.rl, "reward_weight_queue_overload", 0.0))
        self.weight_remote_reject = float(getattr(config.rl, "reward_weight_remote_reject", 0.0))
        self.latency_target = float(getattr(config.rl, "latency_target", 0.4))
        self.energy_target = float(getattr(config.rl, "energy_target", 2200.0))
        self.latency_tolerance = float(getattr(config.rl, "latency_upper_tolerance", self.latency_target * 2.0))
        self.energy_tolerance = float(getattr(config.rl, "energy_upper_tolerance", self.energy_target * 1.5))
        # 分段容错/钳位
        self.total_cost_clip = float(getattr(config.rl, "reward_total_cost_clip", 120.0))
        self.component_clip = float(getattr(config.rl, "reward_component_clip", 25.0))

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

        # 🔧 修复：归一化因子必须与优化目标值对齐
        # Normalisation factors MUST align with optimization targets (latency_target and energy_target).
        # 根据训练结果（Episode 1000+）：延迟稳定在~0.05s，能耗稳定在~5000J
        # 目标值设置为：latency_target=0.4s, energy_target=1200J
        # 因此归一化基准应该直接使用这些目标值，而非硬编码的0.2s和1000J
        self.delay_normalizer = self.latency_target  # 与目标值对齐
        self.energy_normalizer = self.energy_target  # 与目标值对齐
        self.delay_bonus_scale = max(1e-6, self.latency_target)
        self.energy_bonus_scale = max(1e-6, self.energy_target)
        
        # SAC算法使用不同的归一化参数（但仍需与target对齐）
        if self.algorithm == "SAC":
            self.delay_normalizer = self.latency_target  # 对齐
            self.energy_normalizer = self.energy_target  # 对齐

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
            val = float(value)
            if not np.isfinite(val):
                return default
            return val
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
            if isinstance(value, float) and not np.isfinite(value):
                return default
            return int(value)
        except (TypeError, ValueError, OverflowError):
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

    @staticmethod
    def _piecewise_ratio(value: float, target: float, tolerance: float) -> float:
        """
        分段容错的归一化比例：低于目标时半幅惩罚，目标-容差线性，超容差超线性。
        """
        v = max(0.0, float(value))
        t = max(1e-6, float(target))
        tol = max(t, float(tolerance))
        if v <= t:
            return 0.5 * (v / t)
        if v <= tol:
            return 1.0 + (v - t) / max(tol - t, 1e-6)
        return 2.0 + (v - tol) / max(t, 1e-6)

    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    # ??API / Public API

    def _extract_metrics(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict],
        migration_metrics: Optional[Dict],
    ) -> RewardMetrics:
        """?????????????????????"""
        metrics = RewardMetrics()
        metrics.avg_delay = max(0.0, self._safe_float(system_metrics.get("avg_task_delay")))
        metrics.total_energy = max(0.0, self._safe_float(system_metrics.get("total_energy_consumption")))
        metrics.dropped_tasks = max(0, self._safe_int(system_metrics.get("dropped_tasks")))
        metrics.completion_rate = max(0.0, self._safe_float(system_metrics.get("task_completion_rate")))
        metrics.data_loss_ratio = max(0.0, self._safe_float(system_metrics.get("data_loss_ratio_bytes")))
        metrics.cache_utilization = max(0.0, self._safe_float(system_metrics.get("cache_utilization")))
        metrics.queue_overload_events = max(0.0, self._safe_float(system_metrics.get("queue_overload_events")))
        metrics.remote_rejection_rate = max(0.0, self._safe_float(system_metrics.get("remote_rejection_rate")))
        metrics.rsu_offload_ratio = max(0.0, self._safe_float(system_metrics.get("rsu_offload_ratio")))
        metrics.uav_offload_ratio = max(0.0, self._safe_float(system_metrics.get("uav_offload_ratio")))

        if cache_metrics:
            metrics.cache_hit_rate = float(max(0.0, min(1.0, self._safe_float(cache_metrics.get("hit_rate"), 0.0))))
            metrics.cache_miss_rate = float(max(0.0, min(1.0, self._safe_float(cache_metrics.get("miss_rate"), 0.0))))
            metrics.prefetch_events = max(0.0, self._safe_float(cache_metrics.get("prefetch_events"), 0.0))
            metrics.total_cache_requests = max(1.0, self._safe_float(cache_metrics.get("total_requests"), 1.0))
            cache_joint = cache_metrics.get("joint_params", {}) if isinstance(cache_metrics, dict) else {}
            metrics.prefetch_lead = self._safe_float(cache_joint.get("prefetch_lead_time"), 0.0)
        if migration_metrics:
            metrics.migration_cost = max(0.0, self._safe_float(migration_metrics.get("migration_cost"), 0.0))
            metrics.migration_effectiveness = float(max(0.0, min(1.0, self._safe_float(migration_metrics.get("effectiveness"), 0.0))))
            migration_joint = migration_metrics.get("joint_params", {}) if isinstance(migration_metrics, dict) else {}
            metrics.migration_backoff = float(max(0.0, min(1.0, self._safe_float(migration_joint.get("migration_backoff"), 0.0))))
        return metrics

    def _compute_components(self, m: RewardMetrics) -> RewardComponents:
        """????/????????"""
        norm_delay = self._piecewise_ratio(m.avg_delay, self.latency_target, self.latency_tolerance)
        norm_energy = self._piecewise_ratio(m.total_energy, self.energy_target, self.energy_tolerance)
        core_cost = self.weight_delay * norm_delay + self.weight_energy * norm_energy

        drop_penalty = self.penalty_dropped * m.dropped_tasks
        completion_gap_penalty = self.weight_completion_gap * max(0.0, self.completion_target - m.completion_rate) if self.weight_completion_gap > 0.0 else 0.0
        data_loss_penalty = self.weight_loss_ratio * m.data_loss_ratio if self.weight_loss_ratio > 0.0 else 0.0
        cache_pressure_penalty = 0.0
        if self.weight_cache_pressure > 0.0 and m.cache_utilization > self.cache_pressure_threshold:
            cache_pressure_penalty = self.weight_cache_pressure * (m.cache_utilization - self.cache_pressure_threshold)
        queue_penalty = self.weight_queue_overload * m.queue_overload_events if self.weight_queue_overload > 0.0 else 0.0
        remote_reject_penalty = self.weight_remote_reject * m.remote_rejection_rate if self.weight_remote_reject > 0.0 else 0.0

        offload_bonus = self.weight_offload_bonus * (m.rsu_offload_ratio + m.uav_offload_ratio) if self.weight_offload_bonus > 0.0 else 0.0
        cache_penalty = self.weight_cache * m.cache_miss_rate if self.weight_cache > 0.0 else 0.0
        cache_bonus = self.weight_cache_bonus * m.cache_hit_rate if self.weight_cache_bonus > 0.0 else 0.0
        migration_penalty = self.weight_migration * m.migration_cost if self.weight_migration > 0.0 else 0.0

        joint_bonus = 0.0
        joint_coupling_penalty = 0.0
        if self.weight_joint > 0.0:
            joint_bonus = self.weight_joint * (m.cache_hit_rate * m.migration_effectiveness)
            prefetch_ratio = np.clip(m.prefetch_events / max(1.0, m.total_cache_requests), 0.0, 1.0)
            coupling_penalty = max(0.0, 0.3 - prefetch_ratio) * 0.5
            coupling_penalty += abs(m.prefetch_lead - 1.5) * 0.05
            coupling_penalty += m.migration_backoff * 0.1
            joint_coupling_penalty = self.weight_joint * coupling_penalty

        def _clip(x: float) -> float:
            return float(np.clip(x, -self.component_clip, self.component_clip))

        total_cost = (
            core_cost
            + _clip(drop_penalty)
            + _clip(completion_gap_penalty)
            + _clip(data_loss_penalty)
            + _clip(cache_pressure_penalty)
            + _clip(queue_penalty)
            + _clip(remote_reject_penalty)
            + _clip(cache_penalty)
            + _clip(migration_penalty)
            + _clip(joint_coupling_penalty)
            - _clip(offload_bonus)
            - _clip(cache_bonus)
            - _clip(joint_bonus)
        )
        total_cost = float(np.clip(total_cost, -self.total_cost_clip, self.total_cost_clip))

        return RewardComponents(
            norm_delay=norm_delay,
            norm_energy=norm_energy,
            core_cost=core_cost,
            drop_penalty=drop_penalty,
            completion_gap_penalty=completion_gap_penalty,
            data_loss_penalty=data_loss_penalty,
            cache_pressure_penalty=cache_pressure_penalty,
            queue_penalty=queue_penalty,
            remote_reject_penalty=remote_reject_penalty,
            offload_bonus=offload_bonus,
            cache_penalty=cache_penalty,
            cache_bonus=cache_bonus,
            migration_penalty=migration_penalty,
            joint_coupling_penalty=joint_coupling_penalty,
            joint_bonus=joint_bonus,
            total_cost=total_cost,
            reward_pre_clip=-total_cost,
            reward=-total_cost,
        )

    def _compose_reward(self, components: RewardComponents, completion_rate: float) -> RewardComponents:
        """????????????????"""
        if self.algorithm == "SAC":
            base_reward = 5.0
            completion_bonus = (completion_rate - 0.95) * 10.0 if completion_rate > 0.95 else 0.0
            reward_raw = base_reward + completion_bonus - components.total_cost
            reward_clipped = float(np.clip(reward_raw, -15.0, 10.0))
        else:
            reward_raw = -components.total_cost
            reward_clipped = float(np.clip(reward_raw, -20.0, 0.0))
        components.reward_pre_clip = reward_raw
        components.reward = reward_clipped if np.isfinite(reward_clipped) else 0.0
        return components

    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> float:
        """???????????????"""
        metrics = self._extract_metrics(system_metrics, cache_metrics, migration_metrics)
        components = self._compute_components(metrics)
        components = self._compose_reward(components, metrics.completion_rate)
        return components.reward if np.isfinite(components.reward) else 0.0

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

    def get_reward_breakdown(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> str:
        """生成奖励分解，便于快速诊断各成本来源。"""
        metrics = self._extract_metrics(system_metrics, cache_metrics, migration_metrics)
        components = self._compute_components(metrics)
        components = self._compose_reward(components, metrics.completion_rate)

        lines = [
            f"Reward report ({self.algorithm}):",
            f"  Total Reward        : {components.reward:.4f} (pre-clip {components.reward_pre_clip:.4f})",
            f"  Core Cost (D+E)     : {components.core_cost:.4f}",
            f"    - Delay (norm/w)  : {components.norm_delay:.4f} / w={self.weight_delay}",
            f"    - Energy (norm/w) : {components.norm_energy:.4f} / w={self.weight_energy}",
            f"  Drop Penalty        : {components.drop_penalty:.4f}",
            f"  Completion Gap      : {components.completion_gap_penalty:.4f}",
            f"  Data Loss Penalty   : {components.data_loss_penalty:.4f}",
            f"  Cache Pressure      : {components.cache_pressure_penalty:.4f}",
            f"  Queue Penalty       : {components.queue_penalty:.4f}",
            f"  Remote Reject       : {components.remote_reject_penalty:.4f}",
            f"  Cache Penalty/Bonus : {components.cache_penalty:.4f} / -{components.cache_bonus:.4f}",
            f"  Migration Penalty   : {components.migration_penalty:.4f}",
            f"  Joint Penalty/Bonus : {components.joint_coupling_penalty:.4f} / -{components.joint_bonus:.4f}",
            f"  Offload Bonus       : -{components.offload_bonus:.4f}",
            f"  ----------------------------------------",
            f"  Total Cost          : {components.total_cost:.4f}",
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

