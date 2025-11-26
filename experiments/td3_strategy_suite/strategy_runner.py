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
from experiments.td3_strategy_suite.metrics_enrichment import enrich_strategy_metrics  # noqa: E402
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


def compute_cost(avg_delay: float, avg_energy: float, avg_reward: Optional[float] = None, 
                completion_rate: Optional[float] = None) -> float:
    """
    计算统一代价函数值（与训练时的奖励函数一致）
    
    【核心原理】
    训练时: reward = -cost (成本越低，奖励越高)
    因此:   raw_cost = -reward
    
    【参数】
    avg_delay: float - 平均任务时延（秒，保留用于日志输出）
    avg_energy: float - 平均总能耗（焦耳，保留用于日志输出）
    avg_reward: float - 平均奖励（必须提供）
    completion_rate: float - 任务完成率（可选，已在训练时的奖励函数中考虑）
    
    【返回值】
    float - 归一化的加权代价（越小越好）
    
    【🔧 修复说明】
    完成率惩罚已在训练时的UnifiedRewardCalculator中通过completion_gap_penalty计算，
    这里不再额外添加乘法惩罚，避免双重惩罚导致对比失真。
    """
    # 🎯 直接从奖励计算成本（与train_single_agent.py完全一致）
    if avg_reward is None:
        raise ValueError(
            "avg_reward is required for cost calculation. "
            "Ensure training results include episode_rewards."
        )
    
    # 🔧 修复：不再额外添加完成率惩罚，避免与训练时的completion_gap_penalty双重计算
    base_cost = -avg_reward
    return base_cost


def normalize_costs(cost_map: Dict[str, float]) -> Dict[str, float]:
    if not cost_map:
        return {}
    min_cost = min(cost_map.values())
    max_cost = max(cost_map.values())
    span = max(max_cost - min_cost, 1e-12)
    return {k: (v - min_cost) / span for k, v in cost_map.items()}


def enrich_with_normalized_costs(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, object]]:
    from typing import cast
    normalized = normalize_costs({k: v["raw_cost"] for k, v in results.items()})
    enriched: Dict[str, Dict[str, object]] = {}
    for key, metrics in results.items():
        enriched[key] = cast(Dict[str, object], dict(metrics))
        enriched[key]["normalized_cost"] = float(normalized[key])
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
    strategy_episode_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=False,
        central_resource=central_resource,
        strategy_episode_overrides=strategy_episode_overrides,
    )


def run_strategy_suite_with_details(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]] = None,
    central_resource: bool = False,
    strategy_episode_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=True,
        central_resource=central_resource,
        strategy_episode_overrides=strategy_episode_overrides,
    )


def _run_strategy_suite_internal(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]],
    include_episode_metrics: bool,
    central_resource: bool = False,
    strategy_episode_overrides: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    keys = list(strategies) if strategies is not None else STRATEGY_KEYS
    results: Dict[str, Dict[str, float]] = {}

    for key in keys:
        episodes_for_strategy = (
            strategy_episode_overrides.get(key, episodes)
            if strategy_episode_overrides
            else episodes
        )
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
                episodes=episodes_for_strategy,
                seed=seed,
                extra_override=merged_override,
                env_options=env_options,
            )
        else:
            outcome = train_single_algorithm(
                preset["algorithm"],
                num_episodes=episodes_for_strategy,
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
        
        # 🎯 优先从奖励计算raw_cost（与train_single_agent.py一致）
        episode_rewards = outcome.get("episode_rewards", [])
        avg_reward: Optional[float] = None
        if episode_rewards and len(episode_rewards) > 0:
            # 使用后50%数据（收敛后）
            if len(episode_rewards) >= 100:
                half_point = len(episode_rewards) // 2
                avg_reward = float(np.mean(episode_rewards[half_point:]))
            elif len(episode_rewards) >= 50:
                avg_reward = float(np.mean(episode_rewards[-30:]))
            else:
                avg_reward = float(np.mean(episode_rewards))
        raw_cost = compute_cost(avg_delay, avg_energy, avg_reward, completion_rate)

        results[key] = {
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "completion_rate": completion_rate,
            "raw_cost": raw_cost,
            "episodes": episodes_for_strategy,
            "seed": seed,
            "from_cache": False,
        }
        if avg_reward is not None:
            results[key]["avg_reward"] = avg_reward
        if include_episode_metrics:
            results[key]["episode_metrics"] = episode_metrics

    return results

def attach_normalized_costs(result_list: List[Dict[str, object]]) -> None:
    """使用全局归一化确保跨配置可比性"""
    # 🎯 修复：收集所有配置点的所有策略成本,计算全局min/max
    all_costs: List[float] = []
    for item in result_list:
        strategies_obj = item.get("strategies", {})
        from typing import cast
        strategies = cast(Dict[str, Dict[str, float]], strategies_obj)
        for v in strategies.values():
            all_costs.append(v.get("raw_cost", 0.0))
    
    # 计算全局归一化基准
    global_min = min(all_costs) if all_costs else 0.0
    global_max = max(all_costs) if all_costs else 1.0
    global_span = max(global_max - global_min, 1e-12)
    
    # 应用全局归一化
    for item in result_list:
        strategies_obj = item.get("strategies", {})
        from typing import cast
        strategies = cast(Dict[str, Dict[str, float]], strategies_obj)
        for key, metrics in strategies.items():
            raw = metrics.get("raw_cost", 0.0)
            strategies[key]["normalized_cost"] = (raw - global_min) / global_span


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
    strategy_episode_overrides: Optional[Dict[str, int]] = None,
) -> List[Dict[str, object]]:
    suite_path.mkdir(parents=True, exist_ok=True)
    evaluated: List[Dict[str, object]] = []
    keys = list(strategies) if strategies is not None else STRATEGY_KEYS

    for index, cfg in enumerate(configs, 1):
        cfg_copy = dict(cfg)
        cfg_key = str(cfg_copy.get("key") or f"config_{index}")
        label = str(cfg_copy.get("label", cfg_key))
        from typing import cast
        overrides_val = cfg_copy.get("overrides", {})
        overrides = cast(Dict[str, object], overrides_val if isinstance(overrides_val, dict) else {})
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
            strategy_episode_overrides=strategy_episode_overrides,
        )
        from typing import cast
        enriched = enrich_with_normalized_costs(cast(Dict[str, Dict[str, float]], raw))

        reference_key = "comprehensive-no-migration"
        from typing import cast
        ref_data = enriched.get(reference_key, {})
        ref_completion_val = ref_data.get("completion_rate") if ref_data else None
        target_completion = float(cast(float, ref_completion_val)) if ref_completion_val is not None else 0.0
        if target_completion <= 0:
            completion_values = []
            for metrics_obj in enriched.values():
                metrics_dict = cast(Dict[str, object], metrics_obj)
                cr_val = metrics_dict.get("completion_rate")
                if cr_val is not None:
                    completion_values.append(float(cast(float, cr_val)))
            target_completion = max(completion_values, default=0.0)

        for strat_key in keys:
            metrics = cast(Dict[str, object], enriched[strat_key])
            episode_metrics = metrics.pop("episode_metrics", None)
            cr_val = metrics.get("completion_rate")
            completion_rate = max(float(cast(float, cr_val)) if cr_val is not None else 0.0, 1e-6)
            if target_completion > 0:
                multiplier = max(1.0, float(target_completion) / completion_rate)
            else:
                multiplier = 1.0
            metrics["resource_multiplier_required"] = multiplier
            
            # 🎯 默认启用指标增强（如果没有自定义hook）
            if not per_strategy_hook and episode_metrics:
                from typing import cast
                ep_metrics = cast(Dict[str, List[float]], episode_metrics)
                enrich_strategy_metrics(strat_key, cast(Dict[str, float], metrics), cfg_copy, ep_metrics)

            if per_strategy_hook:
                from typing import cast
                ep_metrics = cast(Dict[str, List[float]], episode_metrics or {})
                per_strategy_hook(strat_key, cast(Dict[str, float], metrics), cfg_copy, ep_metrics)
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
            
            # 🎯 显示成本计算来源（检查是否有reward数据）
            use_reward_cost = metrics.get("avg_reward") is not None
            cost_source = "(-reward)" if use_reward_cost else "(delay+energy)"
            print(
                f"  - {strategy_label(strat_key)}: "
                f"Cost={metrics['raw_cost']:.4f} {cost_source} "
                f"Delay={metrics['avg_delay']:.4f}s "
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

    if evaluated:
        attach_normalized_costs(evaluated)

    return evaluated
