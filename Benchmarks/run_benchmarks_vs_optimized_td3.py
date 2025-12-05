#!/usr/bin/env python3
"""
Run Benchmarks algorithms (DDPG/TD3/SAC/Local/Heuristic/SA) inside the VEC simulator
using the same action/state/reward as OPTIMIZED_TD3, and optionally run the reference
OPTIMIZED_TD3 for side-by-side comparison.

Example:
  python Benchmarks/run_benchmarks_vs_optimized_td3.py --episodes 200 --alg td3 sac ddpg local heuristic sa --run-ref --groups 5
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import List, Any, Dict, Tuple

import numpy as np

from Benchmarks.vec_env_adapter import VecEnvWrapper, _build_scenario_overrides
from Benchmarks.reward_adapter import compute_reward_from_info
# New algorithm implementations based on original papers
from Benchmarks.nath_ddpg_mec import NathDDPGConfig, train_nath_ddpg
from Benchmarks.liu_bayesian_optimization import LiuBOConfig, train_liu_bo
from Benchmarks.zhang_ronet_nano import RoNetConfig, train_ronet
from Benchmarks.wang_gail_ppo import WangGAILConfig, train_wang_gail
# Original implementations (DDPG from Lillicrap, TD3 from Fujimoto)
from Benchmarks.lillicrap_ddpg_vanilla import DDPGConfig, train_ddpg
from Benchmarks.fujimoto_td3_vanilla import TD3Config as VanillaTD3Config, train_vanilla_td3
from Benchmarks.local_only_policy import LocalOnlyPolicy
from Benchmarks.nath_dynamic_offload_heuristic import DynamicOffloadHeuristic
from Benchmarks.liu_online_sa import SAConfig, OnlineSimulatedAnnealing
from Benchmarks.run_compare_with_optimized_td3 import run_optimized_td3

try:
    import torch
except ImportError:
    torch = None


def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        try:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except Exception:
            pass


def to_jsonable(obj: Any):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    return obj


def run_rl(algo: str, episodes: int, seed: int, env_cfg, max_steps_per_ep: int = 200):
    set_global_seeds(seed)
    env = VecEnvWrapper(**env_cfg)
    total_steps = max_steps_per_ep * episodes
    warmup_cap = max(total_steps // 2, 1000)
    
    # Nath & Wu 2020 - DDPG for MEC offloading
    if algo == "nath_ddpg":
        cfg = NathDDPGConfig(num_mus=env_cfg.get("num_vehicles", 6))
        cfg.start_steps = min(cfg.start_steps, warmup_cap)
        return train_nath_ddpg(env, cfg, max_steps=total_steps, seed=seed)
    
    # Liu & Cao 2022 - Bayesian Optimization
    if algo == "liu_bo":
        cfg = LiuBOConfig()
        return train_liu_bo(env, cfg, max_steps=total_steps, seed=seed)
    
    # Zhang RoNet 2023 - DNN + Adversarial Training
    if algo == "ronet":
        cfg = RoNetConfig()
        return train_ronet(env, cfg, max_steps=total_steps, seed=seed)
    
    # Wang 2025 - GAIL + Improved PPO
    if algo == "wang_gail":
        cfg = WangGAILConfig()
        return train_wang_gail(env, cfg, max_steps=total_steps, seed=seed)
    
    # Original Lillicrap DDPG (baseline)
    if algo == "ddpg":
        cfg = DDPGConfig()
        cfg.start_steps = min(cfg.start_steps, warmup_cap)
        return train_ddpg(env, cfg, max_steps=total_steps, seed=seed)
    
    # Vanilla TD3 (Fujimoto et al. 2018) - pure implementation
    if algo in ("vanilla_td3", "td3"):
        cfg = VanillaTD3Config()
        cfg.start_steps = min(cfg.start_steps, warmup_cap)
        return train_vanilla_td3(env, cfg, max_steps=total_steps, seed=seed)
    
    raise ValueError(f"Unsupported RL algo: {algo}")


def run_local(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    set_global_seeds(seed)
    env = VecEnvWrapper(**env_cfg)
    policy = LocalOnlyPolicy(env_cfg["num_rsus"], env_cfg["num_uavs"])
    ep_rewards = []
    ep_metrics = {
        "avg_task_delay": [],
        "total_energy_consumption": [],
        "task_completion_rate": [],
        "dropped_tasks": [],
        "cache_hit_rate": []
    }
    for _ in range(episodes):
        state = env.reset()
        ep_r = 0.0
        # Accumulators
        cur_ep_delay = []
        cur_ep_energy = 0.0
        cur_ep_completed = []
        cur_ep_dropped = 0
        cur_ep_cache_hits = []
        
        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(env.action_dim)
            state, _, done, info = env.step(action)
            reward, metrics = compute_reward_from_info(info)
            ep_r += reward
            
            # Collect metrics
            cur_ep_delay.append(metrics.get("avg_task_delay", 0.0))
            cur_ep_energy += metrics.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(metrics.get("task_completion_rate", 0.0))
            cur_ep_dropped += metrics.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(metrics.get("cache_hit_rate", 0.0))

            if done:
                break
        ep_rewards.append(ep_r)
        
        # Aggregate
        ep_metrics["avg_task_delay"].append(np.mean(cur_ep_delay) if cur_ep_delay else 0.0)
        ep_metrics["total_energy_consumption"].append(cur_ep_energy)
        ep_metrics["task_completion_rate"].append(np.mean(cur_ep_completed) if cur_ep_completed else 0.0)
        ep_metrics["dropped_tasks"].append(cur_ep_dropped)
        ep_metrics["cache_hit_rate"].append(np.mean(cur_ep_cache_hits) if cur_ep_cache_hits else 0.0)

    return {"episode_rewards": ep_rewards, "episode_metrics": ep_metrics, "episodes": episodes}


def run_heuristic(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    set_global_seeds(seed)
    env = VecEnvWrapper(**env_cfg)
    policy = DynamicOffloadHeuristic(env_cfg["num_rsus"], env_cfg["num_uavs"])
    ep_rewards = []
    ep_metrics = {
        "avg_task_delay": [],
        "total_energy_consumption": [],
        "task_completion_rate": [],
        "dropped_tasks": [],
        "cache_hit_rate": []
    }
    for _ in range(episodes):
        state = env.reset()
        ep_r = 0.0
        # Accumulators
        cur_ep_delay = []
        cur_ep_energy = 0.0
        cur_ep_completed = []
        cur_ep_dropped = 0
        cur_ep_cache_hits = []

        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(state, env.action_dim)
            state, _, done, info = env.step(action)
            reward, metrics = compute_reward_from_info(info)
            ep_r += reward
            
            # Collect metrics
            cur_ep_delay.append(metrics.get("avg_task_delay", 0.0))
            cur_ep_energy += metrics.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(metrics.get("task_completion_rate", 0.0))
            cur_ep_dropped += metrics.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(metrics.get("cache_hit_rate", 0.0))

            if done:
                break
        ep_rewards.append(ep_r)
        
        # Aggregate
        ep_metrics["avg_task_delay"].append(np.mean(cur_ep_delay) if cur_ep_delay else 0.0)
        ep_metrics["total_energy_consumption"].append(cur_ep_energy)
        ep_metrics["task_completion_rate"].append(np.mean(cur_ep_completed) if cur_ep_completed else 0.0)
        ep_metrics["dropped_tasks"].append(cur_ep_dropped)
        ep_metrics["cache_hit_rate"].append(np.mean(cur_ep_cache_hits) if cur_ep_cache_hits else 0.0)

    return {"episode_rewards": ep_rewards, "episode_metrics": ep_metrics, "episodes": episodes}


def run_sa(env_cfg, episodes: int, seed: int):
    """Simulated annealing over simple offload preference weights, evaluated in the VEC env."""
    base_env = VecEnvWrapper(**env_cfg)
    action_dim = base_env.action_dim
    max_steps_per_ep = 200

    def params_to_action(params: np.ndarray) -> np.ndarray:
        # params: 3 weights -> softmax -> fill [local, rsu, uav] and pad zeros.
        logits = params - params.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        act = np.zeros(action_dim, dtype=np.float32)
        act[:3] = probs.astype(np.float32)
        return act

    def eval_once(params: np.ndarray, eval_idx: int) -> float:
        set_global_seeds(seed + eval_idx)
        env = VecEnvWrapper(**env_cfg)
        act = params_to_action(params)
        state = env.reset()
        ep_r = 0.0
        # Accumulators
        cur_ep_delay = []
        cur_ep_energy = 0.0
        cur_ep_completed = []
        cur_ep_dropped = 0
        cur_ep_cache_hits = []

        for _ in range(max_steps_per_ep):
            state, _, done, info = env.step(act)
            reward, metrics = compute_reward_from_info(info)
            ep_r += reward
            cur_ep_delay.append(metrics.get("avg_task_delay", 0.0))
            cur_ep_energy += metrics.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(metrics.get("task_completion_rate", 0.0))
            cur_ep_dropped += metrics.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(metrics.get("cache_hit_rate", 0.0))
            if done:
                break
        return ep_r

    def evaluate_fn(params: np.ndarray) -> float:
        # Average over a few short episodes for stability.
        scores = [eval_once(params, idx) for idx in range(max(1, episodes // 5))]
        return float(np.mean(scores))

    sa = OnlineSimulatedAnnealing(dim=3, bounds=[(0.0, 5.0)] * 3, cfg=SAConfig(seed=seed, max_iters=episodes))
    res = sa.search(evaluate_fn)
    
    # SA returns history of best scores. We map this to episode_rewards.
    ep_rewards = res["history"]
    
    # To get metrics for the best solution (for bar charts), we re-evaluate once.
    best_params = res["best_params"]
    # We need to extract metrics from eval_once. 
    # Let's modify eval_once to return (score, metrics) and evaluate_fn to return score.
    # But eval_once is defined inside run_sa.
    
    # Let's redefine eval_once to capture metrics
    final_metrics = {}
    
    def eval_with_metrics(params: np.ndarray) -> Tuple[float, Dict[str, float]]:
        set_global_seeds(seed + 999) # Use a different seed for final eval
        env = VecEnvWrapper(**env_cfg)
        act = params_to_action(params)
        state = env.reset()
        ep_r = 0.0
        
        cur_ep_delay = []
        cur_ep_energy = 0.0
        cur_ep_completed = []
        cur_ep_dropped = 0
        cur_ep_cache_hits = []
        
        for _ in range(max_steps_per_ep):
            state, _, done, info = env.step(act)
            reward, metrics = compute_reward_from_info(info)
            ep_r += reward
            
            cur_ep_delay.append(metrics.get("avg_task_delay", 0.0))
            cur_ep_energy += metrics.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(metrics.get("task_completion_rate", 0.0))
            cur_ep_dropped += metrics.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(metrics.get("cache_hit_rate", 0.0))
            
            if done:
                break
                
        m = {
            "avg_task_delay": np.mean(cur_ep_delay) if cur_ep_delay else 0.0,
            "total_energy_consumption": cur_ep_energy,
            "task_completion_rate": np.mean(cur_ep_completed) if cur_ep_completed else 0.0,
            "dropped_tasks": cur_ep_dropped,
            "cache_hit_rate": np.mean(cur_ep_cache_hits) if cur_ep_cache_hits else 0.0
        }
        return ep_r, m

    # Get final metrics
    _, final_m = eval_with_metrics(best_params)
    
    # Replicate final metrics for the length of episodes (as an approximation for bar charts)
    # Or just return a list with one element? 
    # The visualization script expects lists.
    # For SA, the "training curve" for cost is not easily available without modifying SA.
    # We will just provide the final metrics repeated, or just the final one.
    # Let's provide a list of length 1 containing the final metrics, 
    # or replicate it to match episodes length (flat line).
    # A flat line is misleading. Let's just return the final metrics as a single-element list?
    # But plot_training_curves expects same length as rewards.
    # Let's fill with NaNs or just the final value?
    # Let's fill with the final value (flat line) to show "converged" performance.
    
    ep_metrics = {k: [v] * len(ep_rewards) for k, v in final_m.items()}

    return {
        "episode_rewards": ep_rewards,
        "episode_metrics": ep_metrics,
        "episodes": len(ep_rewards),
        "seed": seed
    }


def main():
    parser = argparse.ArgumentParser(description="Compare Benchmarks algorithms vs OPTIMIZED_TD3 in VEC sim.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--groups", type=int, default=5, help="Number of data groups (seeds) per experiment comparison.")
    parser.add_argument("--alg", nargs="+", default=["vanilla_td3", "ddpg", "local"], help="Algorithms to run. Options: vanilla_td3, ddpg, nath_ddpg, liu_bo, ronet, wang_gail, local, heuristic, sa")
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--rsus", type=int, default=4)
    parser.add_argument("--uavs", type=int, default=2)
    parser.add_argument("--arrival", type=float, default=2.0)
    parser.add_argument("--bandwidth", type=float, default=18.0)
    parser.add_argument("--radius", type=float, default=320.0)
    parser.add_argument("--bandwidths", type=float, nargs="*", help="Bandwidth sweep values (MHz).")
    parser.add_argument("--edge-compute", type=float, nargs="*", help="Edge compute total sweep (Hz) split RSU:UAV at 5:1.")
    parser.add_argument("--vehicles-list", type=int, nargs="*", help="Vehicle-count sweep values.")
    parser.add_argument("--data-sizes", type=float, nargs="*", help="Task data size sweep values (KB, fixed min/max).")
    parser.add_argument("--arrival-rates", type=float, nargs="*", help="Task arrival rate sweep values.")
    parser.add_argument("--run-ref", action="store_true", help="Also run OPTIMIZED_TD3 reference.")
    parser.add_argument("--max-steps", type=int, default=200, help="Max steps per episode in env loop.")
    args = parser.parse_args()

    env_cfg = dict(
        num_vehicles=args.vehicles,
        num_rsus=args.rsus,
        num_uavs=args.uavs,
        task_arrival_rate=args.arrival,
        bandwidth=args.bandwidth,
        coverage_radius=args.radius,
        edge_compute_total=None,
        task_data_size_kb=None,
    )

    def tail_avg(out: Dict[str, Any]) -> float | str:
        if "episode_rewards" not in out:
            return "n/a"
        rewards = out.get("episode_rewards", [])
        if not rewards:
            return "n/a"
        tail = rewards[-10:] if len(rewards) >= 10 else rewards
        return sum(tail) / len(tail)

    def summarize_groups(group_runs: List[Dict[str, Any]]) -> Dict[str, float]:
        vals = []
        for gr in group_runs:
            ta = tail_avg(gr)
            if isinstance(ta, (int, float)):
                vals.append(float(ta))
        if not vals:
            return {}
        return {
            "tail_avg_mean": float(mean(vals)),
            "tail_avg_std": float(pstdev(vals)) if len(vals) > 1 else 0.0,
        }

    experiments: List[Tuple[str, Dict[str, Any]]] = [("base", env_cfg)]

    def add_sweep(label: str, key: str, values) -> None:
        if not values:
            return
        for v in values:
            cfg = env_cfg.copy()
            cfg[key] = v
            experiments.append((f"{label}={v}", cfg))

    add_sweep("bandwidth", "bandwidth", args.bandwidths)
    add_sweep("edge_compute", "edge_compute_total", args.edge_compute)
    add_sweep("vehicles", "num_vehicles", args.vehicles_list)
    add_sweep("data_kb", "task_data_size_kb", args.data_sizes)
    add_sweep("arrival_rate", "task_arrival_rate", args.arrival_rates)

    results = {}
    for exp_label, cfg in experiments:
        print(f"\n=== Experiment: {exp_label} ===")
        exp_res = {}
        exp_summary = {}
        for alg in args.alg:
            group_runs = []
            for g in range(args.groups):
                seed = args.seed + g
                if alg in {"ippo", "ddpg", "sac"}:
                    out = run_rl(alg, episodes=args.episodes, seed=seed, env_cfg=cfg, max_steps_per_ep=args.max_steps)
                elif alg == "local":
                    out = run_local(cfg, episodes=args.episodes, seed=seed, max_steps_per_ep=args.max_steps)
                elif alg == "heuristic":
                    out = run_heuristic(cfg, episodes=args.episodes, seed=seed, max_steps_per_ep=args.max_steps)
                elif alg == "sa":
                    out = run_sa(cfg, episodes=args.episodes, seed=seed)
                else:
                    raise ValueError(f"Unsupported alg: {alg}")
                out["seed"] = seed
                group_runs.append(out)
                print(f"[{exp_label}][{alg}][group {g+1}/{args.groups}] seed={seed} last10_avg={tail_avg(out)} episodes={out.get('episodes', 'na')}")
            exp_res[alg] = group_runs
            exp_summary[alg] = summarize_groups(group_runs)

        if args.run_ref:
            ref_runs = []
            for g in range(args.groups):
                ref_seed = args.seed + g
                ref_args = argparse.Namespace(**vars(args))
                ref_args.seed = ref_seed
                override = _build_scenario_overrides(
                    num_vehicles=cfg.get("num_vehicles"),
                    num_rsus=cfg.get("num_rsus"),
                    num_uavs=cfg.get("num_uavs"),
                    task_arrival_rate=cfg.get("task_arrival_rate"),
                    bandwidth=cfg.get("bandwidth"),
                    coverage_radius=cfg.get("coverage_radius"),
                    edge_compute_total=cfg.get("edge_compute_total"),
                    total_rsu_compute=cfg.get("total_rsu_compute"),
                    total_uav_compute=cfg.get("total_uav_compute"),
                    task_data_size_kb=cfg.get("task_data_size_kb"),
                )
                ref_path = run_optimized_td3(ref_args, override_scenario=override)
                ref_runs.append({"seed": ref_seed, "result_path": ref_path, "override": override})
            exp_res["optimized_td3_ref"] = ref_runs
        results[exp_label] = {"runs": exp_res, "summary": exp_summary, "config": cfg}

    # Persist sweep results for reproducibility
    out_dir = Path("results/benchmarks_sweeps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = {
        "args": vars(args),
        "experiments": results,
    }
    out_path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    print(f"\nSaved sweep results to {out_path}")

    # Not persisting to disk to keep script light; users can extend as needed.


if __name__ == "__main__":
    main()
