#!/usr/bin/env python3
"""Shared helpers for reorganised TD3 strategy comparison experiments."""

from __future__ import annotations

import contextlib
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from experiments.td3_strategy_suite.strategy_runner import compute_cost

# 配置日志记录器
logger = logging.getLogger(__name__)

OverrideBuilder = Callable[[float], Tuple[Dict[str, Any], Dict[str, Any]]]


@dataclass(frozen=True)
class ModeSpec:
    """Description of a comparison mode/strategy."""

    name: str
    key: str
    description: str
    color: str
    marker: str
    linestyle: str = "-"
    linewidth: float = 2.0
    central_resource: bool = True
    resource_init: Optional[str] = "learned"
    disable_migration: bool = False
    enforce_offload_mode: Optional[str] = None
    extra_env: Dict[str, str] = field(default_factory=dict)

    def env_overrides(self) -> Dict[str, str]:
        """Environment variables that should be active while running this mode."""

        overrides = dict(self.extra_env)
        overrides["CENTRAL_RESOURCE"] = "1" if self.central_resource else "0"
        if self.resource_init:
            overrides["RESOURCE_ALLOCATION_MODE"] = self.resource_init
        return overrides


@dataclass(frozen=True)
class DimensionSpec:
    """Definition of a sweep dimension (e.g. arrival rate, compute budget)."""

    key: str
    name: str
    values: Sequence[float]
    result_key: str
    override_builder: OverrideBuilder
    description: str = ""


@contextlib.contextmanager
def _temporary_environ(overrides: Dict[str, Optional[str]]):
    """Temporarily set environment variables."""

    previous: Dict[str, Optional[str]] = {}
    for key, value in overrides.items():
        previous[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


class ComparisonSuite:
    """Coordinator that executes sweeps for a set of modes."""

    def __init__(
        self,
        *,
        modes: Sequence[ModeSpec],
        train_fn: Callable[..., Dict[str, Any]],
        seed_fn: Callable[[], None],
        reward_config: Any,
    ) -> None:
        self.modes = list(modes)
        self.train_fn = train_fn
        self.seed_fn = seed_fn
        self.reward_config = reward_config

    def run_dimension(
        self,
        dimension: DimensionSpec,
        *,
        episodes: int,
        silent: bool,
        seed: int,
    ) -> List[Dict[str, Any]]:
        """Sweep a single dimension and return aggregated metrics."""

        print(f"\n{'=' * 80}")
        print(f"{dimension.name} 对比实验")
        if dimension.description:
            print(dimension.description)
        print("=" * 80)

        total_runs = len(dimension.values) * len(self.modes)
        counter = 0
        results: List[Dict[str, Any]] = []

        for value in dimension.values:
            override_config, metadata = dimension.override_builder(value)
            metadata = dict(metadata or {})
            display = metadata.get("label") or f"{value}"
            print(f"\n配置: {dimension.name} = {display}")
            print("-" * 80)

            entry: Dict[str, Any] = {
                dimension.result_key: value,
                "modes": {},
                **metadata,
            }

            for mode in self.modes:
                counter += 1
                print(f"[{counter}/{total_runs}] {mode.name} ...")
                entry["modes"][mode.key] = self._run_mode(
                    mode=mode,
                    override_config=override_config,
                    episodes=episodes,
                    silent=silent,
                    seed=seed,
                )

            results.append(entry)

        return results

    def _run_mode(
        self,
        *,
        mode: ModeSpec,
        override_config: Dict[str, Any],
        episodes: int,
        silent: bool,
        seed: int,
    ) -> Dict[str, Any]:
        print(f"  运行: {mode.name}")
        logger.info(f"Starting mode: {mode.name}, episodes={episodes}, seed={seed}")
        
        env_overrides = mode.env_overrides()
        env_overrides["RANDOM_SEED"] = str(seed)
        logger.debug(f"Environment overrides: {env_overrides}")

        # 类型转换：Dict[str, str] -> Dict[str, Optional[str]]
        env_overrides_optional: Dict[str, Optional[str]] = {k: v for k, v in env_overrides.items()}
        
        with _temporary_environ(env_overrides_optional):
            try:
                self.seed_fn()
                scenario_override = dict(override_config) if override_config else None
                result = self.train_fn(
                    algorithm="TD3",
                    num_episodes=episodes,
                    silent_mode=silent,
                    override_scenario=scenario_override,
                    disable_migration=mode.disable_migration,
                    enforce_offload_mode=mode.enforce_offload_mode,
                )
            except ValueError as exc:
                # 配置错误（如无效的参数范围）
                error_msg = f"配置错误: {exc}"
                logger.error(f"Mode {mode.name} failed with ValueError: {exc}", exc_info=True)
                print(f"  ❌ 模式 {mode.name} 配置错误: {exc}")
                return {"success": False, "error": error_msg, "error_type": "ValueError"}
            except RuntimeError as exc:
                # 运行时错误（如训练失败、GPU内存不足）
                error_msg = f"运行时错误: {exc}"
                logger.error(f"Mode {mode.name} failed with RuntimeError: {exc}", exc_info=True)
                print(f"  ❌ 模式 {mode.name} 运行失败: {exc}")
                return {"success": False, "error": error_msg, "error_type": "RuntimeError"}
            except KeyError as exc:
                # 缺少必要的配置项或指标
                error_msg = f"缺少配置项: {exc}"
                logger.error(f"Mode {mode.name} failed with KeyError: {exc}", exc_info=True)
                print(f"  ❌ 模式 {mode.name} 缺少配置: {exc}")
                return {"success": False, "error": error_msg, "error_type": "KeyError"}
            except Exception as exc:
                # 其他未知错误
                error_msg = f"未知错误: {type(exc).__name__}: {exc}"
                logger.error(f"Mode {mode.name} failed with unexpected error: {exc}", exc_info=True)
                print(f"  ⚠️ 模式 {mode.name} 运行失败: {exc}")
                return {"success": False, "error": error_msg, "error_type": type(exc).__name__}

        return self._summarise_training(result, mode.name)

    def _summarise_training(self, result: Dict[str, Any], mode_name: str) -> Dict[str, Any]:
        episode_metrics = result.get("episode_metrics") or {}

        def tail_mean(values: Iterable[float]) -> float:
            values = list(values or [])
            if not values:
                return 0.0
            tail = values[len(values) // 2 :]
            return float(np.mean(tail)) if tail else float(np.mean(values))

        avg_delay = tail_mean(episode_metrics.get("avg_delay", []))
        avg_energy = tail_mean(episode_metrics.get("total_energy", []))
        completion_rate = tail_mean(episode_metrics.get("task_completion_rate", []))
        cache_hit_rate = tail_mean(episode_metrics.get("cache_hit_rate", []))

        avg_cost = compute_cost(avg_delay, avg_energy)

        print(
            f"    ✅ {mode_name} 完成 - 成本:{avg_cost:.3f} "
            f"(时延 {avg_delay:.3f}s, 能耗 {avg_energy:.1f}J) "
            f"完成率 {completion_rate * 100:.1f}%"
        )

        return {
            "success": True,
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "avg_cost": avg_cost,
            "completion_rate": completion_rate,
            "cache_hit_rate": cache_hit_rate,
            "final_reward": result.get("final_episode_reward", 0.0),
        }
