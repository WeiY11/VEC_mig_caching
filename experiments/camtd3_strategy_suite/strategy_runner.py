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
from experiments.camtd3_strategy_suite.run_strategy_training import STRATEGY_PRESETS  # noqa: E402

STRATEGY_KEYS: List[str] = list(STRATEGY_PRESETS.keys())


def tail_mean(values: Iterable[float]) -> float:
    seq = list(map(float, values or []))
    if not seq:
        return 0.0
    if len(seq) >= 100:
        seq = seq[len(seq) // 2 :]
    return float(np.mean(seq))


def compute_cost(avg_delay: float, avg_energy: float) -> float:
    weight_delay = float(config.rl.reward_weight_delay)
    weight_energy = float(config.rl.reward_weight_energy)
    return weight_delay * avg_delay + weight_energy * (avg_energy / 1000.0)


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
    return enriched


def run_strategy_suite(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, float]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=False,
    )


def run_strategy_suite_with_details(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, object]]:
    return _run_strategy_suite_internal(
        override_scenario=override_scenario,
        episodes=episodes,
        seed=seed,
        silent=silent,
        strategies=strategies,
        include_episode_metrics=True,
    )


def _run_strategy_suite_internal(
    override_scenario: Dict[str, object],
    episodes: int,
    seed: int,
    silent: bool,
    strategies: Optional[Iterable[str]],
    include_episode_metrics: bool,
) -> Dict[str, Dict[str, float]]:
    keys = list(strategies) if strategies is not None else STRATEGY_KEYS
    results: Dict[str, Dict[str, float]] = {}
    for key in keys:
        preset = STRATEGY_PRESETS[key]
        merged_override = copy.deepcopy(preset.get("override_scenario", {}))
        merged_override.update(override_scenario or {})
        merged_override.setdefault("override_topology", True)

        os.environ["RANDOM_SEED"] = str(seed)
        _apply_global_seed_from_env()

        outcome = train_single_algorithm(
            preset["algorithm"],
            num_episodes=episodes,
            silent_mode=silent,
            override_scenario=merged_override,
            use_enhanced_cache=preset["use_enhanced_cache"],
            disable_migration=preset["disable_migration"],
            enforce_offload_mode=preset["enforce_offload_mode"],
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
        )
        enriched = enrich_with_normalized_costs(raw)

        for strat_key in keys:
            metrics = enriched[strat_key]
            episode_metrics = metrics.pop("episode_metrics", None)
            if per_strategy_hook:
                per_strategy_hook(strat_key, metrics, cfg_copy, episode_metrics or {})
            detail_path = config_dir / f"{strat_key}.json"
            metrics_to_save = dict(cfg_copy)
            metrics_to_save.update({"strategy": strat_key, **metrics})
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
        }
        evaluated.append(cfg_entry)

    return evaluated
