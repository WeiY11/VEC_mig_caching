"""
TD3延时-能耗协同优化变体

在标准TD3基础上，引入自适应奖励塑形器，以动态加权方式更强地压制时延与能耗，
同时保持任务完成率约束。该环境与原TD3共享相同的网络结构与训练流程，仅调整奖励设计。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from config import config
from .td3 import TD3Environment


def _safe_float(value, default: float = 0.0) -> float:
    """防御式转换为float。"""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value, default: int = 0) -> int:
    """防御式转换为int。"""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass
class LatencyEnergyRewardShaper:
    """
    延时-能耗自适应奖励塑形器

    通过指数滑动平均追踪指标，并根据偏离程度动态调整权重。同时结合完成率约束与突增惩罚，
    达到“时延与能耗协同最小化”的目标。
    """

    delay_target: float = field(default_factory=lambda: getattr(config.rl, "latency_target", 0.20))
    energy_target: float = field(default_factory=lambda: getattr(config.rl, "energy_target", 2200.0))
    delay_tolerance: float = field(default_factory=lambda: getattr(config.rl, "latency_upper_tolerance", 0.30))
    energy_tolerance: float = field(default_factory=lambda: getattr(config.rl, "energy_upper_tolerance", 3200.0))

    base_delay_weight: float = 2.4
    base_energy_weight: float = 1.6
    adaptive_gain: float = 1.8
    balance_gain: float = 0.8
    balance_tolerance: float = 0.12
    completion_target: float = 0.95
    completion_penalty_gain: float = 12.0
    dropped_penalty_gain: float = 0.05
    spike_penalty_gain: float = 6.5
    ema_alpha: float = 0.12

    reward_clip_low: float = -40.0
    reward_clip_high: float = -1e-3

    _delay_ema: Optional[float] = None
    _energy_ema: Optional[float] = None
    _last_breakdown: Dict[str, float] = field(default_factory=dict, init=False)

    def _update_ema(self, current: float, ema_value: Optional[float]) -> float:
        if ema_value is None:
            return current
        return self.ema_alpha * current + (1.0 - self.ema_alpha) * ema_value

    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> float:
        """根据系统指标计算奖励。"""
        avg_delay = max(0.0, _safe_float(system_metrics.get("avg_task_delay"), 0.0))
        total_energy = max(0.0, _safe_float(system_metrics.get("total_energy_consumption"), 0.0))
        completion_rate = np.clip(_safe_float(system_metrics.get("task_completion_rate"), 0.0), 0.0, 1.0)
        dropped_tasks = max(0, _safe_int(system_metrics.get("dropped_tasks"), 0))

        # 更新指数滑动平均，捕捉趋势用于突增惩罚
        self._delay_ema = self._update_ema(avg_delay, self._delay_ema)
        self._energy_ema = self._update_ema(total_energy, self._energy_ema)

        delay_ratio = avg_delay / max(self.delay_target, 1e-6)
        energy_ratio = total_energy / max(self.energy_target, 1e-6)

        adaptive_delay_weight = self.base_delay_weight * (1.0 + self.adaptive_gain * max(0.0, delay_ratio - 1.0))
        adaptive_energy_weight = self.base_energy_weight * (1.0 + self.adaptive_gain * max(0.0, energy_ratio - 1.0))

        base_cost = adaptive_delay_weight * delay_ratio + adaptive_energy_weight * energy_ratio

        # 约束项：完成率与丢弃任务
        completion_penalty = self.completion_penalty_gain * max(0.0, self.completion_target - completion_rate)
        dropped_penalty = self.dropped_penalty_gain * dropped_tasks

        # 平衡惩罚：鼓励同时降低
        balance_penalty = self.balance_gain * max(0.0, abs(delay_ratio - energy_ratio) - self.balance_tolerance)

        # 突增惩罚：时延或能耗突然飙升
        delay_spike = max(0.0, avg_delay - max(self._delay_ema or avg_delay, self.delay_target))
        energy_spike = max(0.0, total_energy - max(self._energy_ema or total_energy, self.energy_target))
        spike_penalty = self.spike_penalty_gain * (
            delay_spike / max(self.delay_target, 1e-6) + 0.5 * energy_spike / max(self.energy_target, 1e-6)
        )

        # 阈值惩罚：超过容忍范围时加速惩罚
        delay_threshold_penalty = 0.0
        if avg_delay > self.delay_tolerance:
            delay_threshold_penalty = (avg_delay - self.delay_tolerance) / max(self.delay_target, 1e-6) * adaptive_delay_weight

        energy_threshold_penalty = 0.0
        if total_energy > self.energy_tolerance:
            energy_threshold_penalty = (
                (total_energy - self.energy_tolerance) / max(self.energy_target, 1e-6) * adaptive_energy_weight
            )

        total_cost = (
            base_cost
            + completion_penalty
            + dropped_penalty
            + balance_penalty
            + spike_penalty
            + delay_threshold_penalty
            + energy_threshold_penalty
        )

        reward = -float(total_cost)
        clipped_reward = float(np.clip(reward, self.reward_clip_low, self.reward_clip_high))

        self._last_breakdown = {
            "reward": clipped_reward,
            "base_cost": base_cost,
            "adaptive_delay_weight": adaptive_delay_weight,
            "adaptive_energy_weight": adaptive_energy_weight,
            "delay_ratio": delay_ratio,
            "energy_ratio": energy_ratio,
            "completion_penalty": completion_penalty,
            "dropped_penalty": dropped_penalty,
            "balance_penalty": balance_penalty,
            "spike_penalty": spike_penalty,
            "delay_threshold_penalty": delay_threshold_penalty,
            "energy_threshold_penalty": energy_threshold_penalty,
            "completion_rate": completion_rate,
            "avg_delay": avg_delay,
            "total_energy": total_energy,
        }

        return clipped_reward

    def get_last_breakdown(self) -> Dict[str, float]:
        """返回上一轮计算的奖励分解。"""
        return dict(self._last_breakdown)

    def format_breakdown(self, system_metrics: Dict) -> str:
        """格式化可读的奖励分解报告。"""
        breakdown = self.get_last_breakdown()
        if not breakdown:
            # 确保先计算
            self.calculate_reward(system_metrics)
            breakdown = self.get_last_breakdown()

        lines = [
            "TD3-LE 奖励分解:",
            f"  Total Reward: {breakdown.get('reward', 0.0):.4f}",
            f"  Base Cost: {breakdown.get('base_cost', 0.0):.4f}",
            f"  Delay Ratio: {breakdown.get('delay_ratio', 0.0):.3f}  (Weight={breakdown.get('adaptive_delay_weight', 0.0):.2f})",
            f"  Energy Ratio: {breakdown.get('energy_ratio', 0.0):.3f} (Weight={breakdown.get('adaptive_energy_weight', 0.0):.2f})",
            f"  Completion Penalty: {breakdown.get('completion_penalty', 0.0):.3f}",
            f"  Dropped Penalty: {breakdown.get('dropped_penalty', 0.0):.3f}",
            f"  Balance Penalty: {breakdown.get('balance_penalty', 0.0):.3f}",
            f"  Spike Penalty: {breakdown.get('spike_penalty', 0.0):.3f}",
            f"  Delay Threshold Penalty: {breakdown.get('delay_threshold_penalty', 0.0):.3f}",
            f"  Energy Threshold Penalty: {breakdown.get('energy_threshold_penalty', 0.0):.3f}",
            f"  Completion Rate: {breakdown.get('completion_rate', 0.0):.3f}",
            f"  Avg Delay: {breakdown.get('avg_delay', 0.0):.3f}s",
            f"  Total Energy: {breakdown.get('total_energy', 0.0):.1f}J",
        ]
        return "\n".join(lines)


class TD3LatencyEnergyEnvironment(TD3Environment):
    """
    TD3延时-能耗优化环境。

    继承自标准TD3环境，仅重写奖励计算逻辑，以便在训练过程中更强地驱动延时和能耗同步下降。
    """

    def __init__(self, num_vehicles: int, num_rsus: int, num_uavs: int):
        super().__init__(num_vehicles, num_rsus, num_uavs)
        self._reward_shaper = LatencyEnergyRewardShaper()
        self.last_reward_breakdown: Dict[str, float] = {}
        print("⚡ TD3-LE 奖励塑形器已启用：重点最小化时延与能耗")

    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None,
    ) -> float:
        reward = self._reward_shaper.calculate_reward(system_metrics, cache_metrics, migration_metrics)
        self.last_reward_breakdown = self._reward_shaper.get_last_breakdown()
        return reward

    def get_reward_breakdown(self, system_metrics: Dict) -> str:
        return self._reward_shaper.format_breakdown(system_metrics)
