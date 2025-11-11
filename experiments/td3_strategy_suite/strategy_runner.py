#!/usr/bin/env python3
"""
Shared utilities for evaluating the six strategy presets across arbitrary scenarios.
"""

from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config  # noqa: E402
from train_single_agent import _apply_global_seed_from_env, train_single_algorithm  # noqa: E402
from experiments.td3_strategy_suite.run_strategy_training import (  # noqa: E402
    STRATEGY_PRESETS,
    STRATEGY_ORDER,
    _run_heuristic_strategy,
)
from utils.unified_reward_calculator import UnifiedRewardCalculator  # noqa: E402
# 缓存系统已禁用
# from experiments.td3_strategy_suite.strategy_model_cache import get_global_cache  # noqa: E402

STRATEGY_KEYS: List[str] = list(STRATEGY_ORDER)
for extra_key in STRATEGY_PRESETS.keys():
    if extra_key not in STRATEGY_KEYS:
        STRATEGY_KEYS.append(extra_key)


def strategy_group(strategy_key: str) -> str:
    preset = STRATEGY_PRESETS[strategy_key]
    return str(preset.get("group", "baseline"))


def strategies_for_group(group_name: str) -> List[str]:
    target = group_name.strip().lower()
    return [key for key in STRATEGY_KEYS if strategy_group(key).lower() == target]


STRATEGY_GROUPS: List[str] = sorted({strategy_group(key) for key in STRATEGY_PRESETS})

# ========== 初始化统一奖励计算器 ==========
# 使用统一奖励计算器确保与训练时的奖励函数一致
_reward_calculator = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    """获取全局奖励计算器实例（延迟初始化）"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator


def tail_mean(values: Iterable[float]) -> float:
    seq = list(map(float, values or []))
    if not seq:
        return 0.0
    if len(seq) >= 100:
        seq = seq[len(seq) // 2 :]
    return float(np.mean(seq))


def compute_cost(avg_delay: float, avg_energy: float) -> float:
    """
    计算统一代价函数值（与训练时的奖励函数一致）
    
    【功能】
    使用统一奖励计算器计算归一化的加权代价，确保与训练时使用的
    奖励函数完全一致。该函数用于策略对比实验的性能评估。
    
    【参数】
    avg_delay: float - 平均任务时延（秒）
    avg_energy: float - 平均总能耗（焦耳）
    
    【返回值】
    float - 归一化的加权代价（越小越好）
    
    【计算公式】
    Cost = ω_T · (T / T_target) + ω_E · (E / E_target)
    其中：
    - ω_T = 2.0（时延权重）
    - ω_E = 1.2（能耗权重）
    - T_target = 0.4s（时延目标值，用于归一化）
    - E_target = 1200J（能耗目标值，用于归一化）
    
    【修复说明】
    ✅ 修复后：使用latency_target和energy_target，与训练时的奖励计算完全一致
    ✅ 修复前：错误使用了delay_normalizer(0.2)和energy_normalizer(1000)
    ✅ 确保评估指标与训练指标可比
    """
    weight_delay = float(config.rl.reward_weight_delay)
    weight_energy = float(config.rl.reward_weight_energy)
    
    # ✅ 修复：使用与训练时完全一致的归一化因子
    calc = _get_reward_calculator()
    delay_normalizer = calc.latency_target  # 0.4（与训练一致）
    energy_normalizer = calc.energy_target  # 1200.0（与训练一致）
    
    return (
        weight_delay * (avg_delay / max(delay_normalizer, 1e-6))
        + weight_energy * (avg_energy / max(energy_normalizer, 1e-6))
    )


def normalize_costs(cost_map: Dict[str, float]) -> Dict[str, float]:
    if not cost_map:
        return {}
    min_cost = min(cost_map.values())
    max_cost = max(cost_map.values())
    span = max(max_cost - min_cost, 1e-12)
    return {k: (v - min_cost) / span for k, v in cost_map.items()}


def enrich_with_normalized_costs(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    normalized = normalize_costs({k: v["raw_cost"] for k, v in results.items()})
    enriched: Dict[str, Dict[str, float]] = {}
    for key, metrics in results.items():
        enriched[key] = dict(metrics)
        enriched[key]["normalized_cost"] = normalized[key]
        enriched[key]["strategy_label"] = strategy_label(key)
        enriched[key]["strategy_group"] = strategy_group(key)
    return enriched


def run_strategy_suite(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]] = None,
    central_resource: bool = False,
) -> Dict[str, Dict[str, float]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=False,
        central_resource=central_resource,
    )


def run_strategy_suite_with_details(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]] = None,
    central_resource: bool = False,
) -> Dict[str, Dict[str, object]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=True,
        central_resource=central_resource,
    )


def _run_strategy_suite_internal(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]],
    include_episode_metrics: bool,
    central_resource: bool = False,
) -> Dict[str, Dict[str, float]]:
    keys = list(strategies) if strategies is not None else STRATEGY_KEYS
    results: Dict[str, Dict[str, float]] = {}

    for key in keys:
        preset = STRATEGY_PRESETS[key]
        preset_override = preset.get("override_scenario")
        if preset_override:
            merged_override = copy.deepcopy(preset_override)
        else:
            merged_override = {}
        if override_scenario:
            merged_override.update(override_scenario)
        if merged_override:
            merged_override.setdefault("override_topology", True)

        env_options = dict(preset.get("env_options") or {})
        preset_central = bool(preset.get("central_resource", False))
        effective_central = bool(central_resource or preset_central)

        os.environ["RANDOM_SEED"] = str(seed)
        _apply_global_seed_from_env()

        if effective_central:
            os.environ['CENTRAL_RESOURCE'] = '1'
        else:
            os.environ.pop('CENTRAL_RESOURCE', None)

        algorithm_kind = str(preset["algorithm"]).lower()
        if algorithm_kind == "heuristic":
            outcome = _run_heuristic_strategy(
                preset=preset,
                episodes=episodes,
                seed=seed,
                extra_override=merged_override,
                env_options=env_options,
            )
        else:
            outcome = train_single_algorithm(
                preset["algorithm"],
                num_episodes=episodes,
                silent_mode=silent,
                override_scenario=merged_override,
                use_enhanced_cache=preset["use_enhanced_cache"],
                disable_migration=preset["disable_migration"],
                enforce_offload_mode=preset["enforce_offload_mode"],
                joint_controller=env_options.get("joint_controller", False),
            )

        episode_metrics = outcome.get("episode_metrics", {})
        avg_delay = tail_mean(episode_metrics.get("avg_delay", []))
        avg_energy = tail_mean(episode_metrics.get("total_energy", []))
        completion_rate = tail_mean(episode_metrics.get("task_completion_rate", []))
        raw_cost = compute_cost(avg_delay, avg_energy)

        results[key] = {
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "completion_rate": completion_rate,
            "raw_cost": raw_cost,
            "episodes": episodes,
            "seed": seed,
            "from_cache": False,
        }
        if include_episode_metrics:
            results[key]["episode_metrics"] = episode_metrics

    return results

def attach_normalized_costs(result_list: List[Dict[str, object]]) -> None:
    for item in result_list:
        strategies = item.get("strategies", {})
        costs = {k: v.get("raw_cost", 0.0) for k, v in strategies.items()}
        normalized = normalize_costs(costs)
        for key, value in normalized.items():
            strategies[key]["normalized_cost"] = value


def strategy_label(strategy_key: str) -> str:
    return STRATEGY_PRESETS[strategy_key]["description"]


def dump_json(path: os.PathLike[str] | str, data: Dict[str, object]) -> None:
    path_obj = Path(path)
    path_obj.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def evaluate_configs(
    configs: List[Dict[str, object]],
    episodes: int,
    seed: int,
    silent: bool,
    suite_path: Path,
    strategies: Optional[Iterable[str]] = None,
    per_strategy_hook: Optional[Callable[[str, Dict[str, float], Dict[str, object], Dict[str, List[float]]], None]] = None,
    central_resource: bool = False,
) -> List[Dict[str, object]]:
    suite_path.mkdir(parents=True, exist_ok=True)
    evaluated: List[Dict[str, object]] = []
    keys = list(strategies) if strategies is not None else STRATEGY_KEYS

    for index, cfg in enumerate(configs, 1):
        cfg_copy = dict(cfg)
        cfg_key = str(cfg_copy.get("key") or f"config_{index}")
        label = str(cfg_copy.get("label", cfg_key))
        overrides = dict(cfg_copy.get("overrides", {}))
        config_dir = suite_path / cfg_key
        config_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*72}")
        print(f"Configuration: {label} ({cfg_key})")
        print(f"Overrides: {json.dumps(overrides, ensure_ascii=False)}")
        print('=' * 72)

        raw = run_strategy_suite_with_details(
            override_scenario=overrides,
            episodes=episodes,
            seed=seed,
            silent=silent,
            strategies=keys,
            central_resource=central_resource,
        )
        enriched = enrich_with_normalized_costs(raw)

        reference_key = "comprehensive-no-migration"
        target_completion = enriched.get(reference_key, {}).get("completion_rate", 0.0)
        if target_completion <= 0:
            target_completion = max(
                (metrics.get("completion_rate", 0.0) for metrics in enriched.values()),
                default=0.0,
            )

        for strat_key in keys:
            metrics = enriched[strat_key]
            episode_metrics = metrics.pop("episode_metrics", None)
            completion_rate = max(float(metrics.get("completion_rate", 0.0)), 1e-6)
            if target_completion > 0:
                multiplier = max(1.0, target_completion / completion_rate)
            else:
                multiplier = 1.0
            metrics["resource_multiplier_required"] = multiplier

            if per_strategy_hook:
                per_strategy_hook(strat_key, metrics, cfg_copy, episode_metrics or {})
            detail_path = config_dir / f"{strat_key}.json"
            metrics_to_save = dict(cfg_copy)
            metrics_to_save.update(
                {
                    "strategy": strat_key,
                    "strategy_group": strategy_group(strat_key),
                    **metrics,
                }
            )
            detail_path.write_text(json.dumps(metrics_to_save, indent=2, ensure_ascii=False), encoding="utf-8")
            print(
                f"  - {strategy_label(strat_key)}: "
                f"Cost={metrics['raw_cost']:.4f} Delay={metrics['avg_delay']:.4f}s "
                f"Energy={metrics['avg_energy']:.2f}J"
            )

        cfg_entry = {
            **cfg_copy,
            "key": cfg_key,
            "label": label,
            "strategies": enriched,
            "episodes": episodes,
            "seed": seed,
            "strategy_groups": sorted({strategy_group(k) for k in keys}),
        }
        evaluated.append(cfg_entry)

    return evaluated
