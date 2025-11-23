#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM-TD3 Comprehensive Comparison Automation Script
==================================================
Automates the execution of 4 key experiments:
1. Core Performance Comparison (CAM-TD3 vs Baselines)
2. Load Analysis (Varying Arrival Rate)
3. Scalability Analysis (Varying Vehicle Count)
4. Ablation Study (Component Analysis)

Usage:
    python experiments/run_comprehensive_comparison.py --dry-run
    python experiments/run_comprehensive_comparison.py --experiment core
    python experiments/run_comprehensive_comparison.py --all
"""

import os
import sys
import argparse
import subprocess
import time
import json
from datetime import datetime
from typing import List, Dict, Any

# Configuration
DEFAULT_EPISODES = 200  # Default for full run
DRY_RUN_EPISODES = 2    # For testing
PYTHON_EXEC = sys.executable
SCRIPT_PATH = "train_single_agent.py"

def run_command(cmd: List[str], env: Dict[str, str], log_file: str, dry_run: bool = False):
    """Execute a command with specific environment variables."""
    cmd_str = " ".join(cmd)
    print(f"🚀 Running: {cmd_str}")
    if dry_run:
        print(f"   [DRY RUN] Would execute with env: {json.dumps(env, indent=2)}")
        return

    with open(log_file, "w", encoding='utf-8') as f:
        process = subprocess.Popen(
            cmd,
            env={**os.environ, **env},
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8'
        )
        
    print(f"   PID: {process.pid}, Log: {log_file}")
    process.wait()
    if process.returncode != 0:
        print(f"❌ Command failed with return code {process.returncode}")
        return False
    else:
        print(f"✅ Command finished successfully")
        return True

def move_results(algorithm: str, dest_folder: str, dest_name: str):
    """Move the latest result files to the experiment folder."""
    import shutil
    import glob
    
    # Map algorithm name to folder name (train_single_agent.py uses lowercase)
    algo_dir = f"results/single_agent/{algorithm.lower()}"
    
    # Find latest JSON result
    json_files = glob.glob(f"{algo_dir}/training_results_*.json")
    if not json_files:
        print(f"⚠️  No result file found in {algo_dir}")
        return
    
    latest_json = max(json_files, key=os.path.getmtime)
    
    # Find corresponding chart (usually has same timestamp)
    timestamp = latest_json.split('_')[-1].replace('.json', '')
    chart_files = glob.glob(f"{algo_dir}/*{timestamp}.png")
    
    # Destination paths
    dest_json = f"{dest_folder}/{dest_name}.json"
    dest_chart = f"{dest_folder}/{dest_name}.png"
    
    print(f"   📦 Moving results to {dest_json}")
    shutil.copy2(latest_json, dest_json)
    if chart_files:
        shutil.copy2(chart_files[0], dest_chart)

def get_base_cmd(episodes: int, seed: int = 42) -> List[str]:
    return [
        PYTHON_EXEC, SCRIPT_PATH,
        "--episodes", str(episodes),
        "--seed", str(seed),
        "--silent-mode"
    ]

def run_experiment_core(episodes: int, dry_run: bool):
    """Experiment 1: Core Performance Comparison"""
    print("\n🧪 Starting Experiment 1: Core Performance Comparison")
    print("=" * 60)
    
    # Defined baselines
    algorithms = [
        ("CAM_TD3", ["--algorithm", "CAM_TD3"]),
        ("TD3", ["--algorithm", "TD3"]),
        ("DDPG", ["--algorithm", "DDPG"]),
        ("SAC", ["--algorithm", "SAC"]),
        ("Greedy", ["--algorithm", "TD3", "--fixed-offload-policy", "greedy"]),
        ("Local-Only", ["--algorithm", "TD3", "--fixed-offload-policy", "local_only"]),
        ("Remote-Only", ["--algorithm", "TD3", "--fixed-offload-policy", "rsu_only"]),
        ("Random", ["--algorithm", "TD3", "--fixed-offload-policy", "random"]),
    ]
    
    dest_folder = "results/comparison/exp1_core"
    os.makedirs(dest_folder, exist_ok=True)
    
    for name, args in algorithms:
        print(f"\n👉 Testing Algorithm: {name}")
        cmd = get_base_cmd(episodes) + args
        env = {} 
        log_file = f"{dest_folder}/{name.lower()}.log"
        
        if run_command(cmd, env, log_file, dry_run):
            if not dry_run:
                # Extract algorithm name for folder lookup
                # If using fixed policy, the algo is usually TD3 (as per cmd args)
                algo_arg = "TD3" # Default fallback
                if "--algorithm" in args:
                    algo_arg = args[args.index("--algorithm") + 1]
                
                move_results(algo_arg, dest_folder, name.lower())

def run_experiment_load(episodes: int, dry_run: bool):
    """Experiment 2: Impact of Task Load"""
    print("\n🧪 Starting Experiment 2: Load Analysis (Congestion)")
    print("=" * 60)
    
    arrival_rates = [1.5, 2.5, 3.5]
    targets = ["CAM_TD3", "TD3"]
    
    dest_folder = "results/comparison/exp2_load"
    os.makedirs(dest_folder, exist_ok=True)
    
    for rate in arrival_rates:
        print(f"\n📊 Testing Arrival Rate: {rate}")
        for algo in targets:
            print(f"   👉 Algorithm: {algo}")
            cmd = get_base_cmd(episodes) + ["--algorithm", algo]
            env = {"TASK_ARRIVAL_RATE": str(rate)}
            log_file = f"{dest_folder}/{algo.lower()}_rate_{rate}.log"
            
            if run_command(cmd, env, log_file, dry_run):
                if not dry_run:
                    move_results(algo, dest_folder, f"{algo.lower()}_rate_{rate}")
            
        # Greedy baseline
        print(f"   👉 Algorithm: Greedy")
        cmd = get_base_cmd(episodes) + ["--algorithm", "TD3", "--fixed-offload-policy", "greedy"]
        env = {"TASK_ARRIVAL_RATE": str(rate)}
        log_file = f"{dest_folder}/greedy_rate_{rate}.log"
        if run_command(cmd, env, log_file, dry_run):
            if not dry_run:
                move_results("TD3", dest_folder, f"greedy_rate_{rate}")

def run_experiment_scalability(episodes: int, dry_run: bool):
    """Experiment 3: Scalability Analysis"""
    print("\n🧪 Starting Experiment 3: Scalability Analysis")
    print("=" * 60)
    
    vehicle_counts = [10, 20, 30]
    targets = ["CAM_TD3", "TD3"]
    
    dest_folder = "results/comparison/exp3_scale"
    os.makedirs(dest_folder, exist_ok=True)
    
    for count in vehicle_counts:
        print(f"\n🚗 Testing Vehicle Count: {count}")
        for algo in targets:
            print(f"   👉 Algorithm: {algo}")
            cmd = get_base_cmd(episodes) + ["--algorithm", algo, "--num-vehicles", str(count)]
            env = {}
            log_file = f"{dest_folder}/{algo.lower()}_veh_{count}.log"
            
            if run_command(cmd, env, log_file, dry_run):
                if not dry_run:
                    move_results(algo, dest_folder, f"{algo.lower()}_veh_{count}")
            
        # Greedy baseline
        print(f"   👉 Algorithm: Greedy")
        cmd = get_base_cmd(episodes) + ["--algorithm", "TD3", "--fixed-offload-policy", "greedy", "--num-vehicles", str(count)]
        env = {}
        log_file = f"{dest_folder}/greedy_veh_{count}.log"
        if run_command(cmd, env, log_file, dry_run):
            if not dry_run:
                move_results("TD3", dest_folder, f"greedy_veh_{count}")

def run_experiment_ablation(episodes: int, dry_run: bool):
    """Experiment 4: Ablation Study"""
    print("\n🧪 Starting Experiment 4: Ablation Study")
    print("=" * 60)
    
    variants = [
        ("CAM_TD3_Full", ["--algorithm", "CAM_TD3"], {}),
        ("No_Migration", ["--algorithm", "CAM_TD3"], {"DISABLE_MIGRATION": "1"}),
        ("No_Cache", ["--algorithm", "CAM_TD3", "--no-enhanced-cache"], {}),
    ]
    
    dest_folder = "results/comparison/exp4_ablation"
    os.makedirs(dest_folder, exist_ok=True)
    
    for name, args, extra_env in variants:
        print(f"\n👉 Testing Variant: {name}")
        cmd = get_base_cmd(episodes) + args
        env = extra_env
        log_file = f"{dest_folder}/{name.lower()}.log"
        
        if run_command(cmd, env, log_file, dry_run):
            if not dry_run:
                algo_arg = "CAM_TD3"
                move_results(algo_arg, dest_folder, name.lower())

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES, help="Number of episodes per run")
    parser.add_argument("--experiment", type=str, choices=["core", "load", "scale", "ablation"], help="Run specific experiment")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    
    args = parser.parse_args()
    
    episodes = DRY_RUN_EPISODES if args.dry_run else args.episodes
    
    if args.all:
        run_experiment_core(episodes, args.dry_run)
        run_experiment_load(episodes, args.dry_run)
        run_experiment_scalability(episodes, args.dry_run)
        run_experiment_ablation(episodes, args.dry_run)
    elif args.experiment == "core":
        run_experiment_core(episodes, args.dry_run)
    elif args.experiment == "load":
        run_experiment_load(episodes, args.dry_run)
    elif args.experiment == "scale":
        run_experiment_scalability(episodes, args.dry_run)
    elif args.experiment == "ablation":
        run_experiment_ablation(episodes, args.dry_run)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
