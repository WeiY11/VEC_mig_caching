#!/usr/bin/env python3
"""
Unified multi-algorithm comparison workflow for the VEC environment.

This module builds a configurable bridge between reinforcement learning (DRL),
heuristic, and meta-heuristic approaches so they can be assessed under the same
vehicular edge computing scenario. The runner handles per-algorithm execution,
metric extraction, aggregation across random seeds, and persistence of results
that downstream visualisation or reporting tools can reuse.
"""

from __future__ import annotations

import csv
import json
import os
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for seeding
    torch = None  # type: ignore

from config import config
from train_single_agent import SingleAgentTrainingEnvironment, train_single_algorithm

try:
    from baseline_comparison.improved_baseline_algorithms import create_baseline_algorithm
except ImportError:  # pragma: no cover - fallback to legacy baselines
    try:
        from baseline_comparison.baseline_algorithms import create_baseline_algorithm  # type: ignore
    except ImportError:  # pragma: no cover - final fallback within this repository
        from experiments.fallback_baselines import create_baseline_algorithm  # type: ignore

try:
    from baseline_comparison.individual_runners.metaheuristic import GABaseline, PSOBaseline
    FALLBACK_SIMULATED_ANNEALING = None
except ImportError:  # pragma: no cover - metaheuristics optional
    GABaseline = PSOBaseline = None  # type: ignore
    try:
        from experiments.fallback_baselines import SimulatedAnnealingPolicy as FALLBACK_SIMULATED_ANNEALING  # type: ignore
    except ImportError:  # pragma: no cover - fallback unavailable
        FALLBACK_SIMULATED_ANNEALING = None

try:
    from baseline_comparison.individual_runners.drl.run_td3_xuance import run_td3_xuance  # type: ignore
    HAS_TD3_XUANCE = True
except ImportError:  # pragma: no cover - optional dependency
    run_td3_xuance = None  # type: ignore
    HAS_TD3_XUANCE = False


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses describing experiment inputs
# ----------------------------------------------------------------------------------------------------------------------


@dataclass
class ComparisonScenario:
    """Defines the network topology overrides shared by all compared algorithms."""

    num_vehicles: int = 12
    num_rsus: Optional[int] = None
    num_uavs: Optional[int] = None
    max_steps_per_episode: Optional[int] = None
    use_enhanced_cache: bool = True
    extra: Dict[str, Any] = field(default_factory=dict)

    def build_overrides(self) -> Dict[str, Any]:
        """Translate the scenario into the override payload understood by the simulator."""
        overrides: Dict[str, Any] = dict(self.extra)
        if self.num_vehicles is not None:
            overrides["num_vehicles"] = self.num_vehicles
        if self.num_rsus is not None:
            overrides["num_rsus"] = self.num_rsus
        if self.num_uavs is not None:
            overrides["num_uavs"] = self.num_uavs
        if overrides:
            overrides["override_topology"] = True
        overrides.pop("use_enhanced_cache", None)
        overrides.pop("max_steps_per_episode", None)
        return overrides

    def to_dict(self) -> Dict[str, Any]:
        data = dict(self.extra)
        data.update(
            {
                "num_vehicles": self.num_vehicles,
                "num_rsus": self.num_rsus,
                "num_uavs": self.num_uavs,
                "max_steps_per_episode": self.max_steps_per_episode,
                "use_enhanced_cache": self.use_enhanced_cache,
            }
        )
        return data

    def with_updates(self, updates: Dict[str, Any]) -> "ComparisonScenario":
        data = self.to_dict()
        data.update(updates)
        return ComparisonScenario.from_dict(data)

    def with_parameter(self, parameter: str, value: Any) -> "ComparisonScenario":
        return self.with_updates({parameter: value})

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ComparisonScenario":
        known_keys = {"num_vehicles", "num_rsus", "num_uavs", "max_steps_per_episode", "use_enhanced_cache"}
        known = {key: data.get(key) for key in known_keys if key in data}
        extra = {key: value for key, value in data.items() if key not in known_keys}
        return cls(extra=extra, **known)


@dataclass
class AlgorithmSpec:
    """Configuration for a single algorithm in the comparison suite."""

    name: str
    category: str
    episodes: Optional[int] = None
    seeds: Optional[List[int]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    label: Optional[str] = None

    @property
    def id(self) -> str:
        """Short identifier used for directory and file naming."""
        return (self.label or self.name).strip().replace(" ", "_")

    @property
    def category_key(self) -> str:
        return self.category.lower()


@dataclass
class ScenarioSweepSpec:
    """Describes a parameter sweep across multiple scenario settings."""

    name: str
    parameter: str
    values: List[Any]
    label: Optional[str] = None
    unit: Optional[str] = None
    metrics: Optional[List[str]] = None
    episodes: Optional[Dict[str, int]] = None
    seeds: Optional[List[int]] = None
    scenario_overrides: Dict[str, Any] = field(default_factory=dict)
    value_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def get_value_override(self, value: Any) -> Dict[str, Any]:
        return self.value_overrides.get(str(value), {})

# ----------------------------------------------------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------------------------------------------------


def _set_random_seed(seed: int) -> None:
    """Apply deterministic seeds across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["RANDOM_SEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def _mean_last(series: Iterable[float], window: int) -> Optional[float]:
    """Compute the mean over the trailing window for a sequence."""
    values = list(series)
    if not values:
        return None
    if window > 0 and len(values) > window:
        values = values[-window:]
    return float(np.mean(values))


# ----------------------------------------------------------------------------------------------------------------------
# Core runner
# ----------------------------------------------------------------------------------------------------------------------


class AlgorithmComparisonRunner:
    """Coordinator that executes and aggregates multi-category algorithm experiments."""

    def __init__(
        self,
        scenario: ComparisonScenario,
        output_root: str = "results/algorithm_comparison",
        timestamp: Optional[str] = None,
    ) -> None:
        self.scenario = scenario
        self.timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_root = Path(output_root)
        self.output_dir = self.output_root / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------------------------------------------
    # Public API
    # --------------------------------------------------------------------------------------------------

    def run_all(
        self,
        algorithm_specs: List[AlgorithmSpec],
        default_episode_map: Dict[str, int],
        default_seeds: List[int],
        metrics: List[str],
    ) -> Dict[str, Any]:
        """Iterate through the configured algorithms and compile aggregated results."""
        aggregated: Dict[str, Any] = {}
        per_seed_records: Dict[str, List[Dict[str, Any]]] = {}

        self._write_json(
            {
                "timestamp": self.timestamp,
                "scenario": self.scenario.to_dict(),
                "defaults": {
                    "episodes": default_episode_map,
                    "seeds": default_seeds,
                    "metrics": metrics,
                },
            },
            "comparison_config_snapshot.json",
        )

        for spec in algorithm_specs:
            episodes = self._resolve_episode_count(spec, default_episode_map)
            seeds = spec.seeds or default_seeds
            run_details: List[Dict[str, Any]] = []

            for seed in seeds:
                detail = self._execute_algorithm(spec, episodes, seed)
                run_details.append(detail)

            aggregated_result = self._aggregate_results(spec, run_details, metrics, episodes)
            aggregated[spec.id] = aggregated_result
            per_seed_records[spec.id] = run_details

        summary_path = self._write_json(
            {
                "timestamp": self.timestamp,
                "results": aggregated,
            },
            "aggregated_results.json",
        )
        runs_path = self._write_json(
            {
                "timestamp": self.timestamp,
                "per_seed": per_seed_records,
            },
            "per_seed_runs.json",
        )

        self._write_summary_csv(aggregated, metrics)

        try:
            from visualization.algorithm_comparison import generate_metric_overview_chart

            generate_metric_overview_chart(aggregated, metrics, self.output_dir)
        except Exception as exc:  # pragma: no cover - plotting is best effort
            print(f"[WARN] Unable to generate comparison overview charts: {exc}")

        return {
            "output_dir": str(self.output_dir),
            "summary_file": str(summary_path),
            "per_seed_file": str(runs_path),
            "results": aggregated,
        }

    # --------------------------------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------------------------------

    def _resolve_episode_count(self, spec: AlgorithmSpec, defaults: Dict[str, int]) -> int:
        if spec.episodes is not None:
            return int(spec.episodes)
        category_key = spec.category_key
        if category_key in defaults:
            return int(defaults[category_key])
        if "default" in defaults:
            return int(defaults["default"])
        # Fallback to a conservative default if nothing provided.
        return 200

    def _execute_algorithm(self, spec: AlgorithmSpec, episodes: int, seed: int) -> Dict[str, Any]:
        category = spec.category_key
        if category == "drl":
            return self._run_drl_algorithm(spec, episodes, seed)
        if category in {"heuristic", "baseline"}:
            return self._run_baseline_algorithm(spec, episodes, seed)
        if category in {"meta", "metaheuristic"}:
            return self._run_metaheuristic_algorithm(spec, episodes, seed)
        raise ValueError(f"Unsupported algorithm category: {spec.category}")

    @contextmanager
    def _scenario_context(self, seed: int) -> Dict[str, Any]:
        """Apply shared scenario overrides and seed handling for each run."""
        _set_random_seed(seed)
        overrides = self.scenario.build_overrides()

        previous_override = os.environ.get("TRAINING_SCENARIO_OVERRIDES")
        if overrides:
            os.environ["TRAINING_SCENARIO_OVERRIDES"] = json.dumps(overrides)

        original_max_steps = config.experiment.max_steps_per_episode
        if self.scenario.max_steps_per_episode:
            config.experiment.max_steps_per_episode = self.scenario.max_steps_per_episode

        network_backup: Dict[str, Any] = {}
        communication_backup: Dict[str, Any] = {}

        def _capture_network(attr: str, value: Any) -> None:
            if attr not in network_backup:
                network_backup[attr] = getattr(config.network, attr, None)
            setattr(config.network, attr, value)

        def _capture_comm(attr: str, value: Any) -> None:
            if attr not in communication_backup:
                communication_backup[attr] = getattr(config.communication, attr, None)
            setattr(config.communication, attr, value)

        if "num_vehicles" in overrides:
            _capture_network("num_vehicles", overrides["num_vehicles"])
        if "num_rsus" in overrides:
            _capture_network("num_rsus", overrides["num_rsus"])
        if "num_uavs" in overrides:
            _capture_network("num_uavs", overrides["num_uavs"])
        if "bandwidth" in overrides:
            _capture_network("bandwidth", overrides["bandwidth"])
        if "coverage_radius" in overrides:
            _capture_network("coverage_radius", overrides["coverage_radius"])
        if "thermal_noise_density" in overrides:
            _capture_comm("thermal_noise_density", overrides["thermal_noise_density"])
        if "noise_figure" in overrides:
            _capture_comm("noise_figure", overrides["noise_figure"])

        try:
            yield overrides
        finally:
            if self.scenario.max_steps_per_episode:
                config.experiment.max_steps_per_episode = original_max_steps
            for attr, value in network_backup.items():
                if value is not None:
                    setattr(config.network, attr, value)
            for attr, value in communication_backup.items():
                if value is not None:
                    setattr(config.communication, attr, value)
            if overrides:
                if previous_override is None:
                    os.environ.pop("TRAINING_SCENARIO_OVERRIDES", None)
                else:
                    os.environ["TRAINING_SCENARIO_OVERRIDES"] = previous_override

    def _run_drl_algorithm(self, spec: AlgorithmSpec, episodes: int, seed: int) -> Dict[str, Any]:
        name_upper = spec.name.upper()
        disable_migration_flag = bool((spec.params or {}).get("disable_migration", False))
        base_algorithm_override = (spec.params or {}).get("base_algorithm") if spec.params else None
        train_algorithm_name = base_algorithm_override or spec.name
        fallback_used: Optional[str] = None
        if name_upper == "TD3_XUANCE" and HAS_TD3_XUANCE:
            with self._scenario_context(seed) as overrides:
                num_vehicles = overrides.get("num_vehicles", self.scenario.num_vehicles)
                max_steps = (
                    overrides.get("max_steps_per_episode")
                    or self.scenario.max_steps_per_episode
                    or config.experiment.max_steps_per_episode
                )

                args = SimpleNamespace(
                    episodes=episodes,
                    seed=seed,
                    num_vehicles=num_vehicles,
                    max_steps=max_steps,
                    save_dir=None,
                    verbose=False,
                )

                start_time = time.time()
                result = run_td3_xuance(args)  # type: ignore[misc]
                wall_time_hours = (time.time() - start_time) / 3600.0

            summary = self._summarise_xuance_output(result, episodes)
            summary.setdefault("training_time_hours", wall_time_hours)
        elif name_upper == "TD3_XUANCE":
            fallback_algorithm = spec.params.get("fallback_algorithm") if spec.params else None
            fallback_algorithm = fallback_algorithm or "TD3"
            print(
                "[WARN] TD3_Xuance implementation is unavailable; "
                f"falling back to {fallback_algorithm} for reproducibility."
            )
            fallback_used = fallback_algorithm
            with self._scenario_context(seed) as overrides:
                start_time = time.time()
                result = train_single_algorithm(
                    fallback_algorithm,
                    num_episodes=episodes,
                    silent_mode=True,
                    override_scenario=overrides,
                    use_enhanced_cache=self.scenario.use_enhanced_cache,
                    disable_migration=disable_migration_flag,
                )
                wall_time_hours = (time.time() - start_time) / 3600.0

            summary = self._summarise_training_output(result, episodes)
            summary.setdefault("training_time_hours", wall_time_hours)
        else:
            with self._scenario_context(seed) as overrides:
                start_time = time.time()
                result = train_single_algorithm(
                    train_algorithm_name,
                    num_episodes=episodes,
                    silent_mode=True,
                    override_scenario=overrides,
                    use_enhanced_cache=self.scenario.use_enhanced_cache,
                    disable_migration=disable_migration_flag,
                )
                wall_time_hours = (time.time() - start_time) / 3600.0

            summary = self._summarise_training_output(result, episodes)
            summary.setdefault("training_time_hours", wall_time_hours)

        record = {
            "seed": seed,
            "algorithm": spec.name,
            "category": spec.category_key,
            "episodes": episodes,
            "window_size": summary.get("window_size"),
            "summary": summary,
        }
        if fallback_used:
            record["fallback_algorithm"] = fallback_used
        record_path = self._dump_run_record(spec, seed, record)
        record["local_record_path"] = str(record_path)
        return record

    def _run_baseline_algorithm(self, spec: AlgorithmSpec, episodes: int, seed: int) -> Dict[str, Any]:
        def factory() -> Any:
            params = dict(spec.params)
            if params:
                print(f"[INFO] Baseline parameters for {spec.name}: {params}")
            if getattr(create_baseline_algorithm, "__module__", "").startswith("experiments.fallback_baselines"):
                params.setdefault("seed", seed)
            try:
                return create_baseline_algorithm(spec.name, **params)
            except TypeError:
                if params:
                    print(f"[WARN] Baseline factory for {spec.name} ignored parameters (signature mismatch).")
                return create_baseline_algorithm(spec.name)

        return self._run_baseline_family(spec, episodes, seed, factory)

    def _run_metaheuristic_algorithm(self, spec: AlgorithmSpec, episodes: int, seed: int) -> Dict[str, Any]:
        meta_map = {
            "ga": GABaseline,
            "genetic": GABaseline,
            "genetic_algorithm": GABaseline,
            "pso": PSOBaseline,
            "particle_swarm": PSOBaseline,
        }
        if FALLBACK_SIMULATED_ANNEALING is not None:
            meta_map.update(
                {
                    "simulatedannealing": FALLBACK_SIMULATED_ANNEALING,
                    "simulated_annealing": FALLBACK_SIMULATED_ANNEALING,
                    "sa": FALLBACK_SIMULATED_ANNEALING,
                }
            )
        key = spec.name.lower()
        if key not in meta_map:
            raise ValueError(f"Meta-heuristic '{spec.name}' is not available.")

        cls = meta_map[key]
        if cls is None:  # pragma: no cover - guard when optional dependency missing
            raise RuntimeError("Meta-heuristic modules are not installed.")

        def factory() -> Any:
            kwargs = dict(spec.params)
            if FALLBACK_SIMULATED_ANNEALING is not None and cls is FALLBACK_SIMULATED_ANNEALING:
                kwargs.setdefault("seed", seed)
            return cls(**kwargs)

        return self._run_baseline_family(spec, episodes, seed, factory)

    def _run_baseline_family(self, spec: AlgorithmSpec, episodes: int, seed: int, factory) -> Dict[str, Any]:
        with self._scenario_context(seed) as overrides:
            env = SingleAgentTrainingEnvironment(
                "TD3",
                override_scenario=overrides,
                use_enhanced_cache=self.scenario.use_enhanced_cache,
            )
            algorithm = factory()
            if hasattr(algorithm, "update_environment"):
                algorithm.update_environment(env)

            max_steps = config.experiment.max_steps_per_episode
            window_size = max(10, episodes // 5)

            episode_rewards: List[float] = []
            delays: List[float] = []
            energies: List[float] = []
            completions: List[float] = []
            cache_rates: List[float] = []
            migration_rates: List[float] = []

            start_time = time.time()
            for _ in range(episodes):
                state = env.reset_environment()
                if hasattr(algorithm, "reset"):
                    algorithm.reset()

                total_reward = 0.0
                steps = 0
                last_info: Dict[str, Any] = {}

                for _ in range(max_steps):
                    action_vec = algorithm.select_action(state)
                    actions_dict = env._build_actions_from_vector(action_vec)
                    next_state, reward, done, info = env.step(action_vec, state, actions_dict)
                    total_reward += float(reward)
                    steps += 1
                    state = next_state
                    last_info = info
                    if done:
                        break

                avg_reward = total_reward / max(1, steps)
                metrics = last_info.get("system_metrics", {})

                episode_rewards.append(float(avg_reward))
                delays.append(float(metrics.get("avg_task_delay", 0.0)))
                energies.append(float(metrics.get("total_energy_consumption", 0.0)))
                completions.append(float(metrics.get("task_completion_rate", 0.0)))
                cache_rates.append(float(metrics.get("cache_hit_rate", 0.0)))
                migration_rates.append(float(metrics.get("migration_success_rate", 0.0)))

            wall_time_hours = (time.time() - start_time) / 3600.0

        summary = {
            "avg_reward": _mean_last(episode_rewards, window_size),
            "avg_delay": _mean_last(delays, window_size),
            "avg_energy": _mean_last(energies, window_size),
            "avg_completion_rate": _mean_last(completions, window_size),
            "cache_hit_rate": _mean_last(cache_rates, window_size),
            "migration_success_rate": _mean_last(migration_rates, window_size),
            "training_time_hours": wall_time_hours,
            "window_size": window_size,
        }

        record = {
            "seed": seed,
            "algorithm": spec.name,
            "category": spec.category_key,
            "episodes": episodes,
            "window_size": window_size,
            "summary": summary,
        }
        record_path = self._dump_run_record(spec, seed, record)
        record["local_record_path"] = str(record_path)
        return record

    def _summarise_training_output(self, result: Dict[str, Any], episodes: int) -> Dict[str, Any]:
        window_size = max(10, episodes // 5)
        episode_rewards = result.get("episode_rewards", [])
        episode_metrics = result.get("episode_metrics", {})

        summary = {
            "avg_reward": _mean_last(episode_rewards, window_size),
            "avg_delay": _mean_last(episode_metrics.get("avg_delay", []), window_size),
            "avg_energy": _mean_last(episode_metrics.get("total_energy", []), window_size),
            "avg_completion_rate": _mean_last(episode_metrics.get("task_completion_rate", []), window_size),
            "cache_hit_rate": _mean_last(episode_metrics.get("cache_hit_rate", []), window_size),
            "migration_success_rate": _mean_last(episode_metrics.get("migration_success_rate", []), window_size),
            "training_time_hours": result.get("training_config", {}).get("training_time_hours"),
            "window_size": window_size,
        }
        return summary

    def _summarise_xuance_output(self, result: Dict[str, Any], episodes: int) -> Dict[str, Any]:
        window_size = max(10, episodes // 5)
        summary = {
            "avg_reward": _mean_last(result.get("episode_rewards", []), window_size),
            "avg_delay": _mean_last(result.get("episode_delays", []), window_size),
            "avg_energy": _mean_last(result.get("episode_energies", []), window_size),
            "avg_completion_rate": _mean_last(result.get("episode_completion_rates", []), window_size),
            "cache_hit_rate": None,
            "migration_success_rate": None,
            "training_time_hours": None,
            "window_size": window_size,
        }
        return summary

    def _aggregate_results(
        self,
        spec: AlgorithmSpec,
        run_details: List[Dict[str, Any]],
        metrics: List[str],
        episodes: int,
    ) -> Dict[str, Any]:
        summary_stats: Dict[str, Dict[str, Optional[float]]] = {}

        for metric in metrics:
            values = [
                run["summary"].get(metric)
                for run in run_details
                if run["summary"].get(metric) is not None
            ]
            if values:
                numeric = [float(v) for v in values]
                summary_stats[metric] = {
                    "mean": float(np.mean(numeric)),
                    "std": float(np.std(numeric, ddof=0)),
                    "min": float(np.min(numeric)),
                    "max": float(np.max(numeric)),
                    "count": len(numeric),
                }
            else:
                summary_stats[metric] = {
                    "mean": None,
                    "std": None,
                    "min": None,
                    "max": None,
                    "count": 0,
                }

        return {
            "algorithm": spec.name,
            "label": spec.label or spec.name,
            "category": spec.category_key,
            "episodes": episodes,
            "seeds": [run["seed"] for run in run_details],
            "window_size": run_details[0]["summary"].get("window_size") if run_details else None,
            "summary": summary_stats,
        }

    def _dump_run_record(self, spec: AlgorithmSpec, seed: int, record: Dict[str, Any]) -> Path:
        algo_dir = self.output_dir / spec.id.lower()
        algo_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{spec.id.lower()}_seed_{seed}.json"
        path = algo_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=2, ensure_ascii=False)
        return path

    def _write_json(self, payload: Dict[str, Any], filename: str) -> Path:
        path = self.output_dir / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        return path

    def _write_summary_csv(self, aggregated: Dict[str, Any], metrics: List[str]) -> Path:
        path = self.output_dir / "summary.csv"
        headers = ["algorithm", "category", "episodes"] + metrics
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(headers)
            for key, data in aggregated.items():
                row = [
                    data.get("label", key),
                    data.get("category"),
                    data.get("episodes"),
                ]
                for metric in metrics:
                    metric_block = data.get("summary", {}).get(metric, {})
                    row.append(metric_block.get("mean"))
                writer.writerow(row)
        return path


__all__ = [
    "AlgorithmComparisonRunner",
    "AlgorithmSpec",
    "ComparisonScenario",
    "ScenarioSweepSpec",
    "ScenarioSweepExecutor",
]


class ScenarioSweepExecutor:
    """Coordinate scenario sweeps and generate per-metric line charts."""

    def __init__(
        self,
        base_scenario: ComparisonScenario,
        algorithm_specs: List[AlgorithmSpec],
        default_episode_map: Dict[str, int],
        default_seeds: List[int],
        default_metrics: List[str],
        base_output_dir: Path,
        base_timestamp: Optional[str] = None,
    ) -> None:
        self.base_scenario = base_scenario
        self.algorithm_specs = algorithm_specs
        self.default_episode_map = default_episode_map
        self.default_seeds = default_seeds
        self.default_metrics = default_metrics
        self.base_output_dir = Path(base_output_dir)
        self.base_timestamp = base_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")

    def run_sweeps(self, sweeps: List[ScenarioSweepSpec]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        for sweep in sweeps:
            results[sweep.name] = self._run_single_sweep(sweep)
        return results

    def _run_single_sweep(self, sweep: ScenarioSweepSpec) -> Dict[str, Any]:
        metrics = sweep.metrics or self.default_metrics
        episodes_map = sweep.episodes or self.default_episode_map
        seeds = sweep.seeds or self.default_seeds

        sweep_root = self.base_output_dir / "sweeps" / sweep.name
        sweep_root.mkdir(parents=True, exist_ok=True)

        value_results: Dict[str, Dict[str, Any]] = {}
        for idx, value in enumerate(sweep.values):
            scenario = (
                self.base_scenario.with_parameter(sweep.parameter, value)
                .with_updates(sweep.scenario_overrides)
                .with_updates(sweep.get_value_override(value))
            )

            timestamp = f"{self.base_timestamp}_{sweep.name}_{idx}"
            runner = AlgorithmComparisonRunner(
                scenario,
                output_root=str(sweep_root),
                timestamp=timestamp,
            )
            run_output = runner.run_all(self.algorithm_specs, episodes_map, seeds, metrics)
            value_results[str(value)] = run_output

        sweep_data = self._build_sweep_dataset(sweep, value_results, metrics)

        dataset_path = sweep_root / f"{sweep.name}_line_data.json"
        with dataset_path.open("w", encoding="utf-8") as fh:
            json.dump(sweep_data, fh, indent=2, ensure_ascii=False)

        plots: Dict[str, str] = {}
        try:
            from visualization.algorithm_comparison import generate_sweep_line_plots

            plots = generate_sweep_line_plots(sweep_data, sweep_root)
        except Exception as exc:  # pragma: no cover - plotting optional
            print(f"[WARN] Unable to generate sweep plots for {sweep.name}: {exc}")

        csv_entries: List[Dict[str, Any]] = []
        for value in sweep.values:
            aggregated = value_results[str(value)]["results"]
            for alg_id, data in aggregated.items():
                label = data.get("label", alg_id)
                for metric in metrics:
                    metric_block = data.get("summary", {}).get(metric, {})
                    csv_entries.append(
                        {
                            "parameter": sweep.parameter,
                            "value": value,
                            "algorithm": label,
                            "metric": metric,
                            "mean": metric_block.get("mean"),
                            "std": metric_block.get("std"),
                        }
                    )

        csv_path = sweep_root / f"{sweep.name}_summary.csv"
        if csv_entries:
            with csv_path.open("w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["parameter", "value", "algorithm", "metric", "mean", "std"],
                )
                writer.writeheader()
                writer.writerows(csv_entries)
        else:
            csv_path = None

        combined_summary = {
            "sweep": {
                "name": sweep.name,
                "label": sweep.label or sweep.name,
                "parameter": sweep.parameter,
                "unit": sweep.unit,
                "values": sweep.values,
                "metrics": metrics,
            },
            "aggregated": {
                str(value): value_results[str(value)]["results"]
                for value in sweep.values
            },
        }
        summary_path = sweep_root / f"{sweep.name}_aggregated.json"
        with summary_path.open("w", encoding="utf-8") as fh:
            json.dump(combined_summary, fh, indent=2, ensure_ascii=False)

        return {
            "data_file": str(dataset_path),
            "summary_file": str(summary_path),
            "plot_files": plots,
            "csv_file": str(csv_path) if csv_path else None,
            "values": sweep.values,
            "metrics": metrics,
        }

    def _build_sweep_dataset(
        self,
        sweep: ScenarioSweepSpec,
        value_results: Dict[str, Dict[str, Any]],
        metrics: List[str],
    ) -> Dict[str, Any]:
        dataset: Dict[str, Any] = {
            "name": sweep.name,
            "label": sweep.label or sweep.name,
            "parameter": sweep.parameter,
            "unit": sweep.unit,
            "values": sweep.values,
            "metrics": metrics,
            "algorithms": {},
        }

        for spec in self.algorithm_specs:
            metric_series: Dict[str, List[Dict[str, Any]]] = {}
            for metric in metrics:
                series: List[Dict[str, Any]] = []
                for value in sweep.values:
                    aggregated = value_results[str(value)]["results"].get(spec.id)
                    metric_block = (aggregated or {}).get("summary", {}).get(metric, {})
                    series.append(
                        {
                            "value": value,
                            "mean": metric_block.get("mean"),
                            "std": metric_block.get("std"),
                        }
                    )
                metric_series[metric] = series

            dataset["algorithms"][spec.id] = {
                "algorithm": spec.name,
                "label": spec.label or spec.name,
                "category": spec.category_key,
                "metrics": metric_series,
            }

        return dataset
