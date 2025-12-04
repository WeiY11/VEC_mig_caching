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


class RunningMeanStd:
    """Tracks the running mean and variance of a stream of data."""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        if np.isscalar(x) or (isinstance(x, np.ndarray) and x.ndim == 0):
            batch_mean = float(x)
            batch_var = 0.0
            batch_count = 1
        else:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


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
    local_offload_ratio: float = 0.0  # 本地处理占比
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
    local_penalty: float = 0.0  # 本地处理额外惩罚
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
        self.weight_cache = float(getattr(config.rl, "reward_weight_cache", 0.0))
        self.weight_cache_bonus = float(getattr(config.rl, "reward_weight_cache_bonus", 0.0))
        self.weight_migration = float(getattr(config.rl, "reward_weight_migration", 0.0))
        # 🔧 P0修复：将fallback默认值从0.05改为0.0，防止bonus抵消core_cost
        self.weight_joint = float(getattr(config.rl, "reward_weight_joint", 0.0))
        # 边缘计算卸载奖励：适度激励远程处理（默认0.0，避免干扰核心优化）
        # 🔧 P0修复：将fallback默认值从0.5改为0.0，防止bonus抵消core_cost
        self.weight_offload_bonus = float(getattr(config.rl, "reward_weight_offload_bonus", 0.0))
        # 本地处理惩罚：移除额外惩罚（默认0.0）
        self.weight_local_penalty = float(getattr(config.rl, "reward_weight_local_penalty", 0.0))
        self.completion_target = float(getattr(config.rl, "completion_target", 0.95))
        self.weight_completion_gap = float(getattr(config.rl, "reward_weight_completion_gap", 0.0))
        self.weight_loss_ratio = float(getattr(config.rl, "reward_weight_loss_ratio", 0.0))
        self.cache_pressure_threshold = float(getattr(config.rl, "cache_pressure_threshold", 0.85))
        self.weight_cache_pressure = float(getattr(config.rl, "reward_weight_cache_pressure", 0.0))
        self.weight_queue_overload = float(getattr(config.rl, "reward_weight_queue_overload", 0.0))
        self.weight_remote_reject = float(getattr(config.rl, "reward_weight_remote_reject", 0.0))
        self.latency_target = float(getattr(config.rl, "latency_target", 0.1))  # 🔧 v16: 0.1s
        self.energy_target = float(getattr(config.rl, "energy_target", 10000.0))  # 🔧 v16: 10000J
        self.latency_tolerance = float(getattr(config.rl, "latency_upper_tolerance", self.latency_target * 2.0))
        self.energy_tolerance = float(getattr(config.rl, "energy_upper_tolerance", self.energy_target * 1.5))
        # 分段容错/钳位
        # 🔧 v12修复：扩大裁剪范围配合更大的权重
        # 核心权重增加到(5.0, 3.0)，需要更大的裁剪范围
        self.total_cost_clip = float(getattr(config.rl, "reward_total_cost_clip", 50.0))  # 🔧 v12: 10 → 50
        self.component_clip = float(getattr(config.rl, "reward_component_clip", 10.0))    # 🔧 v12: 3 → 10
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

        # 🔧 v27优化：使用min-max归一化，确保延迟和能耗在相同量级[0,1]
        # 问题：简单除以target会导致范围差异巨大
        # 解决：使用(value - min) / (max - min)归一化到[0,1]
        # 这样延迟和能耗都有相同的信号强度
        self.delay_min = float(getattr(config.rl, "latency_min", 0.05))  # 最优延迟(RSU处理简单任务)
        self.delay_max = float(getattr(config.rl, "latency_upper_tolerance", 2.0))  # 最差延迟(本地处理复杂任务)
        self.energy_min = float(getattr(config.rl, "energy_min", 1000.0))  # 最优能耗(episode)
        self.energy_max = float(getattr(config.rl, "energy_upper_tolerance", 25000.0))  # 最差能耗(episode)
        
        # 归一化范围
        self.delay_range = max(self.delay_max - self.delay_min, 1e-6)
        self.energy_range = max(self.energy_max - self.energy_min, 1e-6)
        
        # 保留旧参数兼容性
        self.delay_normalizer = self.latency_target
        self.energy_normalizer = self.energy_target
        self.delay_bonus_scale = max(1e-6, self.latency_target)
        self.energy_bonus_scale = max(1e-6, self.energy_target)
        
        # 🆕 动态归一化配置
        # ⚠️ 当前禁用以改善收敛性（config.rl.use_dynamic_reward_normalization=False）
        # 如果未来启用，需要充分测试动态归一化对训练稳定性的影响
        self.use_dynamic_normalization = getattr(config.rl, "use_dynamic_reward_normalization", False)
        if self.use_dynamic_normalization:
            self.delay_rms = RunningMeanStd(shape=())
            self.energy_rms = RunningMeanStd(shape=())
            # 初始化为目标值，避免初期波动过大
            self.delay_rms.mean = self.latency_target
            self.energy_rms.mean = self.energy_target
            print(f"   ⚠️ Dynamic Normalization: ENABLED (Experimental)")
            print(f"      Initial: delay={self.latency_target:.2f}s, energy={self.energy_target:.0f}J")
        else:
            print(f"   Dynamic Normalization: DISABLED (Recommended)")

        # 已移除SAC的特殊归一化参数，所有算法现在使用统一的归一化逻辑

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

        # 🔧 v29修复：收紧裁剪范围，避免奖励信号过于分散
        # 原来-50到0范围太宽，不同策略的差异被稀释
        # 收紧到-10到0，放大策略改进的反馈信号
        self.reward_clip_range = (-10.0, 0.0)  # 🔧 v29: -50 → -10 (收紧5倍)

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
        metrics.local_offload_ratio = max(0.0, self._safe_float(system_metrics.get("local_offload_ratio", 0.0)))

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
        """
        🔧 v27优化：使用min-max归一化，延迟和能耗都在[0,1]范围
        
        设计理念：
        1. norm = (value - min) / (max - min) → [0, 1]
        2. 当value=min时，norm=0（最优，无惩罚）
        3. 当value=max时，norm=1（最差，最大惩罚）
        4. 超出范围时线性外推（允许norm>1）
        
        这样智能体直接优化：minimize(0.5*norm_delay + 0.5*norm_energy)
        """
        # --- 核心成本：min-max归一化到[0,1] ---
        # 延迟归一化
        delay_val = max(0.0, m.avg_delay)
        norm_delay = (delay_val - self.delay_min) / self.delay_range
        norm_delay = max(0.0, norm_delay)  # 不允许负值
        
        # 能耗归一化
        energy_val = max(0.0, m.total_energy)
        norm_energy = (energy_val - self.energy_min) / self.energy_range
        norm_energy = max(0.0, norm_energy)  # 不允许负值
        
        # 🔧 v29优化：纯成本最小化，不混入bonus
        # 加权和（使用配置的权重）
        delay_penalty = self.weight_delay * norm_delay
        energy_penalty = self.weight_energy * norm_energy
        core_cost = delay_penalty + energy_penalty

        # --- 🔧 v29优化：简化惩罚项，保持清晰语义 ---
        # 1. 丢弃惩罚 - 每丢一个任务扣分
        drop_penalty = self.penalty_dropped * m.dropped_tasks
        
        # 2. 完成率惩罚 - 只在低于目标时才惩罚，使用平方项放大差距
        # 🔧 v29: 使用soft惩罚，避免过度保守
        completion_gap = max(0.0, self.completion_target - m.completion_rate)
        completion_gap_penalty = self.weight_completion_gap * (completion_gap ** 1.5)  # 平方根放大
        
        # 3. 数据丢失惩罚 - 直接使用丢失比例
        data_loss_penalty = self.weight_loss_ratio * m.data_loss_ratio
        
        # 🔧 v29：移除offload_bonus，避免bonus/cost语义混淆
        # 原来的offload_bonus会让智能体困惑：到底是最小化cost还是最大化bonus？
        # 现在纯粹做成本最小化，语义更清晰
        offload_bonus = 0.0
        delay_improvement_bonus = 0.0
        
        # 其他惩罚项（保持为0，需要时再启用）
        cache_pressure_penalty = 0.0
        queue_penalty = 0.0
        remote_reject_penalty = 0.0
        local_penalty = 0.0
        cache_penalty = 0.0
        migration_penalty = 0.0
        cache_bonus = 0.0
        joint_bonus = 0.0
        joint_coupling_penalty = 0.0

        # 🔧 v29：简化总成本计算
        total_cost = core_cost + drop_penalty + completion_gap_penalty + data_loss_penalty
        total_cost = float(np.clip(total_cost, 0.0, self.total_cost_clip))

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
            local_penalty=local_penalty,
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
        """组装最终奖励
        
        🔧 MDP优化v2.0: 转为正向奖励空间 [0, 10]
        
        优化点:
        1. 使用正向奖励（基线奖励 - 成本）替代负向奖励
        2. 增加任务完成率bonus，提供即时反馈
        3. 放大奖励差异，让策略改进更明显
        
        奖励范围: [0, 10]，越接近10表示性能越好
        - 0: 最差情况（高延迟、高能耗、低完成率）
        - 10: 理想情况（低延迟、低能耗、高完成率）
        """
        import os
        
        # 检查是否启用正向奖励模式
        # 🔧 修复：默认禁用正向奖励，回归传统的负值惩罚模式
        # 用户反馈正向奖励导致曲线平坦，无法观察到收敛趋势
        use_positive_reward = os.environ.get('MDP_POSITIVE_REWARD', '0') == '1'
        
        if use_positive_reward:
            # === 正向奖励模式 ===
            # 基线奖励：表示理想情况的最高分
            baseline_reward = 10.0
            
            # 核心成本归一化到[0, baseline]范围
            # total_cost 通常在 [0, total_cost_clip] 范围
            normalized_cost = min(components.total_cost / max(self.total_cost_clip, 1.0), 1.0)
            
            # 基础奖励 = 基线 - 成本
            base_reward = baseline_reward * (1.0 - normalized_cost * 0.8)  # 成本最多扣除80%
            
            # 任务完成率bonus：提供即时正向反馈
            completion_bonus = completion_rate * 2.0  # 最高+2分
            
            # 低延迟奖励：鼓励快速处理
            if components.norm_delay < 0.5:  # 延迟低于目标的50%
                delay_bonus = (0.5 - components.norm_delay) * 2.0  # 最高+1分
            else:
                delay_bonus = 0.0
            
            # 组合奖励
            reward_raw = base_reward + completion_bonus + delay_bonus
            
            # 裁剪到[0, 12]范围
            reward_clipped = float(np.clip(reward_raw, 0.0, 12.0))
        else:
            # === 原始负向奖励模式（保留兼容性） ===
            reward_scale = float(getattr(config.rl, 'reward_scale', 1.0))
            reward_raw = -abs(components.total_cost) * reward_scale
            reward_clipped = float(np.clip(reward_raw, 
                                           self.reward_clip_range[0] * reward_scale, 
                                           self.reward_clip_range[1]))
        
        components.reward_pre_clip = reward_raw
        components.reward = reward_clipped if np.isfinite(reward_clipped) else 0.0
        return components

    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> tuple[float, Dict[str, float]]:
        """计算奖励并返回标量值和组件字典
        
        Returns:
            tuple: (reward, reward_components)
                - reward: 总奖励标量值
                - reward_components: 包含各分量的字典
        """
        # 🔧 修复: 只在主进程打印调试信息，避免并行环境日志混乱
        import os
        is_main_process = os.environ.get('WORKER_ID', '0') == '0'
        
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0

        metrics = self._extract_metrics(system_metrics, cache_metrics, migration_metrics)
        components = self._compute_components(metrics)
        components = self._compose_reward(components, metrics.completion_rate)
        
        # 仅主进程打印前2次调试信息
        if is_main_process and self._debug_count < 2:
            print(f"[RewardDebug] delay={system_metrics.get('avg_task_delay', 0):.4f}, energy={system_metrics.get('total_energy_consumption', 0):.1f}")
            print(f"  reward={components.reward:.4f}, core_cost={components.core_cost:.4f}")
            self._debug_count += 1
        
        # 构造奖励组件字典供调试使用
        reward_components = {
            'delay': -components.norm_delay * self.weight_delay,
            'energy': -components.norm_energy * self.weight_energy,
            'cache': -components.cache_penalty + components.cache_bonus,
            'penalty': -(components.drop_penalty + components.completion_gap_penalty + 
                        components.queue_penalty + components.remote_reject_penalty),
            'core_cost': -components.core_cost,
            'total': components.reward
        }
        
        final_reward = components.reward if np.isfinite(components.reward) else 0.0
        return final_reward, reward_components
    
    def calculate_instant_reward(
        self,
        step_metrics: Dict,
        prev_metrics: Dict = None,
    ) -> tuple[float, Dict[str, float]]:
        """
        🆕 MDP优化: 计算即时奖励（基于单步增量）
        
        优化点:
        1. 使用单步增量而非累积值
        2. 提供即时任务完成奖励
        3. 奖励与动作直接相关，增强因果关系
        
        Args:
            step_metrics: 本步的增量指标
                - step_completed: 本步完成任务数
                - step_dropped: 本步丢弃任务数
                - step_energy: 本步能耗(J)
                - step_delay: 本步平均延迟(s)
                - cache_hits: 本步缓存命中数
            prev_metrics: 上一步的指标（用于计算改善）
        
        Returns:
            (instant_reward, reward_breakdown)
        """
        prev_metrics = prev_metrics or {}
        
        # 提取本步指标
        step_completed = int(step_metrics.get('step_completed', 0))
        step_dropped = int(step_metrics.get('step_dropped', 0))
        step_energy = float(step_metrics.get('step_energy', 0.0))
        step_delay = float(step_metrics.get('step_delay', 0.0))
        cache_hits = int(step_metrics.get('cache_hits', 0))
        step_total = step_completed + step_dropped
        
        # === 即时奖励组件 ===
        
        # 1. 任务完成奖励（每完成一个任务+1分）
        completion_reward = float(step_completed) * 1.0
        
        # 2. 任务丢弃惩罚（每丢弃一个任务-2分）
        drop_penalty = float(step_dropped) * 2.0
        
        # 3. 能耗惩罚（归一化后的能耗）
        energy_penalty = min(1.0, step_energy / 100.0) * 1.0  # 每步100J扣分1分
        
        # 4. 延迟惩罚（仅当延迟超过目标时）
        delay_target = 0.5  # 目标延迟0.5s
        if step_delay > delay_target:
            delay_penalty = (step_delay - delay_target) * 2.0  # 每超过0.1s扣分0.2分
        else:
            delay_penalty = 0.0
            # 低延迟奖励
            delay_bonus = (delay_target - step_delay) / delay_target * 0.5
            completion_reward += delay_bonus
        
        # 5. 缓存命中奖励
        cache_reward = float(cache_hits) * 0.5
        
        # 6. 改善奖励（如果有上一步数据）
        improvement_bonus = 0.0
        if prev_metrics:
            prev_delay = float(prev_metrics.get('step_delay', step_delay))
            prev_energy = float(prev_metrics.get('step_energy', step_energy))
            
            # 延迟改善
            if step_delay < prev_delay:
                improvement_bonus += (prev_delay - step_delay) * 1.0
            
            # 能耗改善
            if step_energy < prev_energy:
                improvement_bonus += (prev_energy - step_energy) / 100.0 * 0.5
        
        # === 组合最终奖励 ===
        # 基线奖励 + 完成奖励 - 惩罚
        baseline = 2.0  # 每步基线奖励
        instant_reward = (
            baseline
            + completion_reward
            + cache_reward
            + improvement_bonus
            - drop_penalty
            - energy_penalty
            - delay_penalty
        )
        
        # 裁剪到合理范围
        instant_reward = float(np.clip(instant_reward, -5.0, 10.0))
        
        # 构造奖励分解
        reward_breakdown = {
            'baseline': baseline,
            'completion': completion_reward,
            'cache': cache_reward,
            'improvement': improvement_bonus,
            'drop_penalty': -drop_penalty,
            'energy_penalty': -energy_penalty,
            'delay_penalty': -delay_penalty if step_delay > delay_target else 0.0,
            'total': instant_reward,
        }
        
        return instant_reward, reward_breakdown

    def update_targets(
        self,
        latency_target: Optional[float] = None,
        energy_target: Optional[float] = None,
    ) -> None:
        """动态更新目标值，使奖励函数可以在训练中自适应拓扑变化。
        
        🔧 P0修复：同步更新归一化因子，确保奖励计算与目标值一致
        """
        if latency_target is not None:
            self.latency_target = float(latency_target)
            self.latency_tolerance = float(
                getattr(config.rl, "latency_upper_tolerance", self.latency_target * 2.0)
            )
            # 🔧 P0修复：同步归一化因子
            self.delay_normalizer = self.latency_target
            self.delay_bonus_scale = max(1e-6, self.latency_target)
        if energy_target is not None:
            self.energy_target = float(energy_target)
            self.energy_tolerance = float(
                getattr(config.rl, "energy_upper_tolerance", self.energy_target * 1.5)
            )
            # 🔧 P0修复：同步归一化因子
            self.energy_normalizer = self.energy_target
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
    reward, _ = calculator.calculate_reward(system_metrics, cache_metrics, migration_metrics)
    return reward


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
