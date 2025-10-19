#!/usr/bin/env python3
"""
Command line entry-point for the unified algorithm comparison experiment.

内置对比算法：CAM-TD3、TD3_xuance、Random、RoundRobin、LocalOnly、RSUOnly、SimulatedAnnealing。
用法示例：
  python run_algorithm_comparison.py
  python run_algorithm_comparison.py --include CAM-TD3 SimulatedAnnealing --metrics avg_reward avg_delay
  python run_algorithm_comparison.py --output-dir results/custom --seeds 42 2025 9527
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from experiments.algorithm_comparison import (
    AlgorithmComparisonRunner,
    AlgorithmSpec,
    ComparisonScenario,
    ScenarioSweepExecutor,
    ScenarioSweepSpec,
)


def _load_config(config_path: Path) -> Dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _build_scenario(data: Dict) -> ComparisonScenario:
    if not data:
        return ComparisonScenario()
    return ComparisonScenario.from_dict(data)


def _build_algorithm_specs(entries: List[Dict], include: Optional[List[str]] = None) -> List[AlgorithmSpec]:
    specs: List[AlgorithmSpec] = []
    include_normalised = {name.lower() for name in include} if include else None

    for entry in entries:
        name = entry["name"]
        if include_normalised and name.lower() not in include_normalised:
            continue
        specs.append(
            AlgorithmSpec(
                name=name,
                category=entry["category"],
                episodes=entry.get("episodes"),
                seeds=entry.get("seeds"),
                params=entry.get("params", {}),
                label=entry.get("label"),
            )
        )
    if not specs:
        raise ValueError("No algorithms selected for execution. Check configuration or --include filter.")
    return specs


def _build_sweeps(entries: Optional[List[Dict]]) -> List[ScenarioSweepSpec]:
    sweeps: List[ScenarioSweepSpec] = []
    for entry in entries or []:
        sweeps.append(
            ScenarioSweepSpec(
                name=entry["name"],
                parameter=entry["parameter"],
                values=entry.get("values", []),
                label=entry.get("label"),
                unit=entry.get("unit"),
                metrics=entry.get("metrics"),
                episodes=entry.get("episodes"),
                seeds=entry.get("seeds"),
                scenario_overrides=entry.get("scenario_overrides", {}),
                value_overrides=entry.get("value_overrides", {}),
            )
        )
    return sweeps


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified DRL/heuristic meta-heuristic comparisons in VEC.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/algorithm_comparison_config.json"),
        help="Path to the comparison configuration JSON file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/algorithm_comparison"),
        help="Directory where comparison artefacts will be stored.",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Optional list of algorithm names to run (case-insensitive filter).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        help="Override metric list used for aggregation/visualisation.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Override random seeds applied to every algorithm run.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override default episode count for algorithms without explicit episode settings.",
    )
    args = parser.parse_args()

    config_data = _load_config(args.config)
    scenario = _build_scenario(config_data.get("scenario", {}))
    defaults = config_data.get("defaults", {})

    default_episode_map = defaults.get("episodes", {})
    if args.episodes:
        default_episode_map = dict(default_episode_map)  # shallow copy
        default_episode_map["default"] = args.episodes

    default_seeds = args.seeds or defaults.get("seeds", [42])
    metrics = args.metrics or defaults.get(
        "metrics",
        ["avg_reward", "avg_delay", "avg_energy", "avg_completion_rate", "training_time_hours"],
    )

    specs = _build_algorithm_specs(config_data.get("algorithms", []), include=args.include)
    sweeps = _build_sweeps(config_data.get("sweeps"))

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = AlgorithmComparisonRunner(scenario, output_root=str(args.output_dir), timestamp=base_timestamp)
    results = runner.run_all(specs, default_episode_map, default_seeds, metrics)

    print("\n=== Comparison Completed ===")
    print(f"Results stored under: {results['output_dir']}")
    for alg_id, summary in results["results"].items():
        label = summary.get("label", alg_id)
        print(f"\n[{label}] ({summary.get('category')}) Episodes={summary.get('episodes')} Seeds={summary.get('seeds')}")
        for metric in metrics:
            metric_stats = summary.get("summary", {}).get(metric, {})
            mean_value = metric_stats.get("mean")
            std_value = metric_stats.get("std")
            if mean_value is None:
                continue
            print(f"  - {metric}: {mean_value:.4f} (+/- {std_value:.4f})")

    if sweeps:
        print("\n=== Scenario Sweeps ===")
        sweep_executor = ScenarioSweepExecutor(
            base_scenario=scenario,
            algorithm_specs=specs,
            default_episode_map=default_episode_map,
            default_seeds=default_seeds,
            default_metrics=metrics,
            base_output_dir=Path(results["output_dir"]),
            base_timestamp=base_timestamp,
        )
        sweep_results = sweep_executor.run_sweeps(sweeps)
        for name, info in sweep_results.items():
            print(f"\n[Sweep: {name}] data -> {info['data_file']}")
            csv_file = info.get('csv_file')
            if csv_file:
                print(f"  - csv: {csv_file}")
            plot_files = info.get("plot_files", {})
            if plot_files:
                for metric, path in plot_files.items():
                    print(f"  - {metric}: {path}")
            else:
                print("  (no plot generated)")


if __name__ == "__main__":
    main()
