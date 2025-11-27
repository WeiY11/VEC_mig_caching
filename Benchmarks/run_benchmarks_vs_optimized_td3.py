#!/usr/bin/env python3
"""
Run Benchmarks algorithms (DDPG/TD3/SAC/Local/Heuristic/SA) inside the VEC simulator
using the same action/state/reward as OPTIMIZED_TD3, and optionally run the reference
OPTIMIZED_TD3 for side-by-side comparison.

Example:
  python Benchmarks/run_benchmarks_vs_optimized_td3.py --episodes 200 --alg td3 sac ddpg local heuristic sa --run-ref
"""
from __future__ import annotations

import argparse
from typing import List

from Benchmarks.vec_env_adapter import VecEnvWrapper
from Benchmarks.reward_adapter import compute_reward_from_info
from Benchmarks.cam_td3_uav_mec import CAMTD3Config, train_cam_td3
from Benchmarks.lillicrap_ddpg_vanilla import DDPGConfig, train_ddpg
from Benchmarks.zhang_robust_sac import RobustSACConfig, train_robust_sac
from Benchmarks.local_only_policy import LocalOnlyPolicy
from Benchmarks.nath_dynamic_offload_heuristic import DynamicOffloadHeuristic
from Benchmarks.liu_online_sa import SAConfig, OnlineSimulatedAnnealing
from Benchmarks.run_compare_with_optimized_td3 import run_optimized_td3


def run_rl(algo: str, episodes: int, seed: int, env_cfg, max_steps_per_ep: int = 200):
    env = VecEnvWrapper(**env_cfg)
    total_steps = max_steps_per_ep * episodes
    if algo == "td3":
        cfg = CAMTD3Config(num_rsus=env_cfg["num_rsus"], num_uavs=env_cfg["num_uavs"])
        return train_cam_td3(env, cfg, max_steps=total_steps, seed=seed)
    if algo == "ddpg":
        cfg = DDPGConfig()
        return train_ddpg(env, cfg, max_steps=total_steps, seed=seed)
    if algo == "sac":
        cfg = RobustSACConfig()
        return train_robust_sac(env, cfg, max_steps=total_steps, seed=seed)
    raise ValueError(f"Unsupported RL algo: {algo}")


def run_local(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    import numpy as np
    env = VecEnvWrapper(**env_cfg)
    policy = LocalOnlyPolicy(env_cfg["num_rsus"], env_cfg["num_uavs"])
    ep_rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_r = 0.0
        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(env.action_dim)
            state, _, done, info = env.step(action)
            reward, _ = compute_reward_from_info(info)
            ep_r += reward
            if done:
                break
        ep_rewards.append(ep_r)
    return {"episode_rewards": ep_rewards, "episodes": episodes}


def run_heuristic(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    import numpy as np
    env = VecEnvWrapper(**env_cfg)
    policy = DynamicOffloadHeuristic(env_cfg["num_rsus"], env_cfg["num_uavs"])
    ep_rewards = []
    for _ in range(episodes):
        state = env.reset()
        ep_r = 0.0
        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(state, env.action_dim)
            state, _, done, info = env.step(action)
            reward, _ = compute_reward_from_info(info)
            ep_r += reward
            if done:
                break
        ep_rewards.append(ep_r)
    return {"episode_rewards": ep_rewards, "episodes": episodes}


def run_sa(env_cfg, episodes: int, seed: int):
    """Simulated annealing over simple offload preference weights, evaluated in the VEC env."""
    import numpy as np

    env = VecEnvWrapper(**env_cfg)
    action_dim = env.action_dim
    max_steps_per_ep = 200

    def params_to_action(params: np.ndarray) -> np.ndarray:
        # params: 3 weights -> softmax -> fill [local, rsu, uav] and pad zeros.
        logits = params - params.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        act = np.zeros(action_dim, dtype=np.float32)
        act[:3] = probs.astype(np.float32)
        return act

    def eval_once(params: np.ndarray) -> float:
        act = params_to_action(params)
        state = env.reset()
        ep_r = 0.0
        for _ in range(max_steps_per_ep):
            state, _, done, info = env.step(act)
            reward, _ = compute_reward_from_info(info)
            ep_r += reward
            if done:
                break
        return ep_r

    def evaluate_fn(params: np.ndarray) -> float:
        # Average over a few short episodes for stability.
        scores = [eval_once(params) for _ in range(max(1, episodes // 5))]
        return float(np.mean(scores))

    sa = OnlineSimulatedAnnealing(dim=3, bounds=[(0.0, 5.0)] * 3, cfg=SAConfig(seed=seed))
    return sa.search(evaluate_fn)


def main():
    parser = argparse.ArgumentParser(description="Compare Benchmarks algorithms vs OPTIMIZED_TD3 in VEC sim.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alg", nargs="+", default=["td3", "ddpg", "sac", "local", "heuristic"], help="Algorithms to run.")
    parser.add_argument("--vehicles", type=int, default=12)
    parser.add_argument("--rsus", type=int, default=4)
    parser.add_argument("--uavs", type=int, default=2)
    parser.add_argument("--arrival", type=float, default=2.0)
    parser.add_argument("--bandwidth", type=float, default=18.0)
    parser.add_argument("--radius", type=float, default=320.0)
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
    )

    results = {}
    for alg in args.alg:
        if alg in {"td3", "ddpg", "sac"}:
            out = run_rl(alg, episodes=args.episodes, seed=args.seed, env_cfg=env_cfg, max_steps_per_ep=args.max_steps)
        elif alg == "local":
            out = run_local(env_cfg, episodes=args.episodes, seed=args.seed, max_steps_per_ep=args.max_steps)
        elif alg == "heuristic":
            out = run_heuristic(env_cfg, episodes=args.episodes, seed=args.seed, max_steps_per_ep=args.max_steps)
        elif alg == "sa":
            out = run_sa(env_cfg, episodes=args.episodes, seed=args.seed)
        else:
            raise ValueError(f"Unsupported alg: {alg}")
        results[alg] = out
        print(f"[{alg}] episodes={out.get('episodes', 'na')} last10_avg={sum(out.get('episode_rewards', [])[-10:]) / max(1, min(10, len(out.get('episode_rewards', [])))) if 'episode_rewards' in out else 'n/a'}")

    if args.run_ref:
        ref = run_optimized_td3(args)
        if ref:
            results["optimized_td3_ref"] = ref

    # Not persisting to disk to keep script light; users can extend as needed.


if __name__ == "__main__":
    main()
