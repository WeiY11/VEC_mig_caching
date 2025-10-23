#!/usr/bin/env python3
"""
Batch launcher for extreme-condition comparison experiments.

This utility wraps `run_algorithm_comparison.py` to execute the high-load and
low-bandwidth scenarios defined for the paper experiments. It ensures the
correct configs, output directories and algorithm selections are used so the
user only needs to invoke a single command.

Example usage:
    python tools/run_extreme_conditions.py --scenario high_load
    python tools/run_extreme_conditions.py --scenario low_bw
    python tools/run_extreme_conditions.py --scenario all
    python tools/run_extreme_conditions.py --list
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNNER = PROJECT_ROOT / "run_algorithm_comparison.py"

SCENARIOS: Dict[str, Dict[str, object]] = {
    "high_load": {
        "description": "High task arrival rate (+40%) with larger payloads across 12/16/20 vehicles.",
        "config": PROJECT_ROOT / "config/paper_extreme_high_load.json",
        "include": ["CAM-TD3", "TD3", "TD3_Xuance", "SimulatedAnnealing"],
        "output_dir": PROJECT_ROOT / "results/paper_comparison/high_load",
    },
    "low_bw": {
        "description": "Bandwidth-limited environment (12 MHz, reduced coverage) for 12 and 20 vehicles.",
        "config": PROJECT_ROOT / "config/paper_extreme_low_bw.json",
        "include": ["CAM-TD3", "TD3", "RoundRobin", "RSUOnly"],
        "output_dir": PROJECT_ROOT / "results/paper_comparison/low_bw",
    },
}


def list_scenarios() -> None:
    print("Available scenarios:")
    for name, meta in SCENARIOS.items():
        config = meta["config"]
        include = ", ".join(meta["include"])
        description = meta["description"]
        print(f"  - {name}: {description}")
        print(f"      config: {config}")
        print(f"      algorithms: {include}")


def run_scenario(name: str, dry_run: bool = False) -> None:
    if name not in SCENARIOS:
        raise SystemExit(f"Unknown scenario '{name}'. Use --list to inspect available options.")

    metadata = SCENARIOS[name]
    config_path: Path = metadata["config"]  # type: ignore[assignment]
    include: List[str] = metadata["include"]  # type: ignore[assignment]
    output_dir: Path = metadata["output_dir"]  # type: ignore[assignment]

    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")
    if not RUNNER.exists():
        raise SystemExit(f"Runner script not found: {RUNNER}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(config_path),
        "--output-dir",
        str(output_dir),
    ]
    if include:
        cmd.append("--include")
        cmd.extend(include)

    print(f"[INFO] Launching scenario '{name}': {' '.join(cmd)}")
    if dry_run:
        return

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        raise SystemExit(f"Scenario '{name}' exited with code {result.returncode}.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute extreme-condition comparison experiments.")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["high_load", "low_bw", "all"],
        help="Which scenario to run. 'all' runs both sequentially.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Only list available scenarios.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list:
        list_scenarios()
        return

    scenarios = SCENARIOS.keys() if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        run_scenario(scenario, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

