#!/usr/bin/env python3
"""
Adapters to run Benchmarks algorithms inside the VEC single-agent simulator
with the same action/state conventions as train_single_agent.py (OPTIMIZED_TD3).

Usage sketch:
    from Benchmarks.vec_env_adapter import VecEnvWrapper
    from Benchmarks.wang_ippo_uav_mec import IPPOConfig, train_ippo
    env = VecEnvWrapper()
    cfg = IPPOConfig()
    train_ippo(env, cfg, max_steps=..., seed=42)
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from train_single_agent import SingleAgentTrainingEnvironment


def _build_scenario_overrides(
    num_vehicles: Optional[int],
    num_rsus: Optional[int],
    num_uavs: Optional[int],
    task_arrival_rate: Optional[float],
    bandwidth: Optional[float],
    coverage_radius: Optional[float],
    edge_compute_total: Optional[float] = None,
    total_rsu_compute: Optional[float] = None,
    total_uav_compute: Optional[float] = None,
    task_data_size_kb: Optional[float] = None,
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if num_vehicles is not None:
        overrides["num_vehicles"] = num_vehicles
    if num_rsus is not None:
        overrides["num_rsus"] = num_rsus
    if num_uavs is not None:
        overrides["num_uavs"] = num_uavs
    if task_arrival_rate is not None:
        overrides["task_arrival_rate"] = task_arrival_rate
    if bandwidth is not None:
        overrides["bandwidth"] = bandwidth
        overrides["total_bandwidth"] = bandwidth
    if coverage_radius is not None:
        overrides["coverage_radius"] = coverage_radius
    if edge_compute_total is not None:
        # Split edge compute total across RSU/UAV using the default 5:1 ratio (50GHz:10GHz).
        total = float(edge_compute_total)
        overrides.setdefault("total_rsu_compute", total * (5.0 / 6.0))
        overrides.setdefault("total_uav_compute", total * (1.0 / 6.0))
    if total_rsu_compute is not None:
        overrides["total_rsu_compute"] = float(total_rsu_compute)
    if total_uav_compute is not None:
        overrides["total_uav_compute"] = float(total_uav_compute)
    if task_data_size_kb is not None:
        size = float(task_data_size_kb)
        overrides["task_data_size_kb"] = size
        overrides.setdefault("task_data_size_min_kb", size)
        overrides.setdefault("task_data_size_max_kb", size)
    if overrides:
        overrides["override_topology"] = True
    return overrides


class VecEnvWrapper:
    """
    Gym-like wrapper around SingleAgentTrainingEnvironment using the OPTIMIZED_TD3 action/state layout.

    - observation_space: flattened state vector (np.array)
    - action_space: continuous box; high/low inferred from environment action_dim
    """

    def __init__(
        self,
        algorithm: str = "OPTIMIZED_TD3",
        num_vehicles: int = 12,
        num_rsus: int = 4,
        num_uavs: int = 2,
        task_arrival_rate: Optional[float] = 2.0,
        bandwidth: Optional[float] = 18.0,
        coverage_radius: Optional[float] = 320.0,
        use_enhanced_cache: bool = True,
        edge_compute_total: Optional[float] = None,
        total_rsu_compute: Optional[float] = None,
        total_uav_compute: Optional[float] = None,
        task_data_size_kb: Optional[float] = None,
    ):
        overrides = _build_scenario_overrides(
            num_vehicles=num_vehicles,
            num_rsus=num_rsus,
            num_uavs=num_uavs,
            task_arrival_rate=task_arrival_rate,
            bandwidth=bandwidth,
            coverage_radius=coverage_radius,
            edge_compute_total=edge_compute_total,
            total_rsu_compute=total_rsu_compute,
            total_uav_compute=total_uav_compute,
            task_data_size_kb=task_data_size_kb,
        )
        prev_override_env = os.environ.get("TRAINING_SCENARIO_OVERRIDES")
        if overrides:
            os.environ["TRAINING_SCENARIO_OVERRIDES"] = json.dumps(overrides)
        self.env = SingleAgentTrainingEnvironment(
            algorithm=algorithm,
            override_scenario=overrides or None,
            use_enhanced_cache=use_enhanced_cache,
            disable_migration=False,
        )
        # Prevent override leakage to subsequent runs in the same shell
        if prev_override_env is None:
            os.environ.pop("TRAINING_SCENARIO_OVERRIDES", None)
        else:
            os.environ["TRAINING_SCENARIO_OVERRIDES"] = prev_override_env
        # Build dummy spaces
        from types import SimpleNamespace

        self.action_dim = getattr(self.env.agent_env, "action_dim", 18)
        high = np.ones(self.action_dim, dtype=np.float32) * 5.0
        class BoxSpace:
            def __init__(self, low, high, shape):
                self.low = low
                self.high = high
                self.shape = shape
            
            def sample(self):
                return np.random.uniform(self.low, self.high, self.shape).astype(np.float32)

        self.action_space = BoxSpace(high=high, low=-high, shape=(self.action_dim,))
        self.observation_space = SimpleNamespace(shape=(self.env.agent_env.state_dim,))

    def reset(self):
        state = self.env.reset_environment()
        return np.array(state, dtype=np.float32)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        # Ensure action shape
        action = np.array(action, dtype=np.float32).reshape(-1)
        if action.size < self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: action.size] = action
            action = padded
        else:
            action = action[: self.action_dim]

        actions_dict = self.env._build_actions_from_vector(action)
        next_state, reward, done, info = self.env.step(action, None, actions_dict)
        return (
            np.array(next_state, dtype=np.float32),
            float(reward),
            bool(done),
            info,
        )
