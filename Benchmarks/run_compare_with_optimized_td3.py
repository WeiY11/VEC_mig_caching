#!/usr/bin/env python3
"""
Quick runner to produce a reference result with OPTIMIZED_TD3 (VEC environment)
for side-by-side comparison against the standalone Benchmarks baselines.

Usage:
    python Benchmarks/run_compare_with_optimized_td3.py --episodes 400 --seed 42
Options:
    --episodes    Training episodes (default 400)
    --seed        RNG seed (default 42)
    --vehicles    Override vehicle count (default 12)
    --rsus        Override RSU count (default 4)
    --uavs        Override UAV count (default 2)
    --extra-args  Extra args passed through to train_single_agent.py

This script simply shells out to:
    python train_single_agent.py --algorithm OPTIMIZED_TD3 ...
and prints the log and where the result JSON lands.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import json
from datetime import datetime


def run_optimized_td3(args, override_scenario: dict | None = None) -> str | None:
    cmd = [
        sys.executable,
        "train_single_agent.py",
        "--algorithm",
        "OPTIMIZED_TD3",
        "--episodes",
        str(args.episodes),
        "--seed",
        str(args.seed),
        "--num-vehicles",
        str(args.vehicles),
        "--num-rsus",
        str(args.rsus),
        "--num-uavs",
        str(args.uavs),
        "--silent-mode",
    ]
    if getattr(args, "extra_args", None):
        cmd.extend(args.extra_args)

    log_dir = "results/benchmark_refs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"optimized_td3_{timestamp}.log")

    print("Running:", " ".join(cmd))
    print("Log:", log_path)
    env = os.environ.copy()
    if override_scenario:
        env["TRAINING_SCENARIO_OVERRIDES"] = json.dumps(override_scenario)
    with open(log_path, "w", encoding="utf-8") as fh:
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, text=True, env=env)
    proc.wait()
    if proc.returncode != 0:
        print(f"[ERROR] train_single_agent exited with {proc.returncode}")
        return None

    # Find latest result JSON
    result_glob = "results/single_agent/optimized_td3/training_results_*.json"
    matches = glob.glob(result_glob)
    if not matches:
        print("[WARN] No result JSON found under results/single_agent/optimized_td3/")
        return None
    latest = max(matches, key=os.path.getmtime)
    print("Latest OPTIMIZED_TD3 result:", latest)
    return latest


def main():
    parser = argparse.ArgumentParser(description="Run OPTIMIZED_TD3 as reference for Benchmarks comparison.")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--rsus", type=int, default=4)
    parser.add_argument("--uavs", type=int, default=2)
    parser.add_argument("--extra-args", nargs=argparse.REMAINDER, help="Extra args passed to train_single_agent.py")
    args = parser.parse_args()

    latest = run_optimized_td3(args)
    if latest:
        print("\nReference ready. Compare this JSON with outputs from the Benchmarks baselines.")


if __name__ == "__main__":
    main()
