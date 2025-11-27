#!/usr/bin/env python3
"""
Run a local-only benchmark comparison aligned with the methods referenced in
the Benchmarks papers. It reuses the unified algorithm comparison runner but
pins to a dedicated config that maps each paper to an available local algorithm.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from experiments.algorithm_comparison import (
    AlgorithmComparisonRunner,
    ScenarioSweepExecutor,
)
from scripts.run import run_algorithm_comparison as rac

DEFAULT_CONFIG = Path("config/benchmark_local_comparison.json")
DEFAULT_OUTPUT = Path("results/benchmark_local")


def _load_defaults(
    config_data: Dict,
    seeds_arg: Optional[List[int]],
    episode_override: Optional[int],
):
    defaults = config_data.get("defaults", {})
    default_episode_map = defaults.get("episodes", {})
    if episode_override:
        default_episode_map = dict(default_episode_map)
        default_episode_map["default"] = episode_override

    default_seeds = seeds_arg or defaults.get("seeds", [42])
    metrics = defaults.get(
        "metrics",
        [
            "avg_reward",
            "avg_delay",
            "avg_energy",
            "avg_completion_rate",
            "cache_hit_rate",
            "migration_success_rate",
            "training_time_hours",
        ],
    )
    return default_episode_map, default_seeds, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Local benchmark suite mapped to Benchmarks papers.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config JSON to load.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Directory where comparison artefacts will be stored.",
    )
    parser.add_argument("--include", nargs="+", help="Optional subset of algorithms to run.")
    parser.add_argument("--seeds", nargs="+", type=int, help="Override random seeds.")
    parser.add_argument(
        "--episodes",
        type=int,
        help="Override default episode count for algorithms without explicit settings.",
    )
    parser.add_argument("--skip-sweeps", action="store_true", help="Skip the scenario sweeps.")
    args = parser.parse_args()

    config_data = rac._load_config(args.config)
    scenario = rac._build_scenario(config_data.get("scenario", {}))

    default_episode_map, default_seeds, metrics = _load_defaults(config_data, args.seeds, args.episodes)
    specs = rac._build_algorithm_specs(config_data.get("algorithms", []), include=args.include)
    sweeps = [] if args.skip_sweeps else rac._build_sweeps(config_data.get("sweeps"))

    base_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    runner = AlgorithmComparisonRunner(scenario, output_root=str(args.output_dir), timestamp=base_timestamp)
    results = runner.run_all(specs, default_episode_map, default_seeds, metrics)

    print("\nCompleted core comparison.")
    print(f"Results stored under: {results['output_dir']}")

    for alg_id, summary in results["results"].items():
        label = summary.get("label", alg_id)
        print(f"[{label}] ({summary.get('category')}) Episodes={summary.get('episodes')} Seeds={summary.get('seeds')}")
        for metric in metrics:
            metric_stats = summary.get("summary", {}).get(metric, {})
            mean_value = metric_stats.get("mean")
            std_value = metric_stats.get("std")
            if mean_value is None:
                continue
            print(f"  - {metric}: {mean_value:.4f} (+/- {std_value:.4f})")

    if sweeps:
        print("\nRunning configured sweeps...")
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
            print(f"[Sweep: {name}] data -> {info['data_file']}")
            csv_file = info.get("csv_file")
            if csv_file:
                print(f"  - csv: {csv_file}")
            plot_files = info.get("plot_files", {})
            if plot_files:
                for metric, path in plot_files.items():
                    print(f"  - {metric}: {path}")
            else:
                print("  (no plot generated)")
    else:
        print("Sweeps skipped.")


if __name__ == "__main__":
    main()
