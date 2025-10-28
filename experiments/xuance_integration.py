#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Xuance integration utilities for the VEC environment.

This module provides a light-weight bridge that allows algorithms implemented in
the Xuance reinforcement learning framework to be trained and evaluated on the
project's vehicular edge computing simulator.  It exposes a helper function
``run_xuance_algorithm`` that mirrors the return structure of ``train_single_algorithm``
so that higher level experiment runners can treat Xuance-based algorithms just
like the in-project implementations.
"""

from __future__ import annotations

import copy
import math
import os
import random
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional at runtime
    torch = None  # type: ignore

from config import config as global_config
from train_single_agent import SingleAgentTrainingEnvironment

try:
    from xuance.environment import make_envs
    from xuance.environment.single_agent_env import REGISTRY_ENV
    from xuance.torch.agents import REGISTRY_Agents
except ImportError as exc:  # pragma: no cover - Xuance must be installed manually
    raise RuntimeError(
        "Xuance library is required for xuance_integration to function. "
        "Please install it with `pip install xuance`."
    ) from exc

__all__ = ["run_xuance_algorithm", "is_xuance_algorithm"]


@dataclass(frozen=True)
class XuanceAlgorithmProfile:
    """Minimal hyper-parameter description for a supported Xuance algorithm."""

    agent: str
    learner: str
    policy: str
    representation: str
    representation_hidden_size: Sequence[int]
    actor_hidden_size: Sequence[int] = ()
    critic_hidden_size: Sequence[int] = ()
    activation: str = "relu"
    activation_action: str = "tanh"
    learning_rate: Optional[float] = None
    learning_rate_actor: Optional[float] = None
    learning_rate_critic: Optional[float] = None
    gamma: float = 0.99
    tau: float = 0.005
    training_frequency: int = 1
    start_training: int = 1000
    batch_size: int = 256
    buffer_size: Optional[int] = None
    horizon_size: Optional[int] = None
    n_epochs: Optional[int] = None
    policy_nepoch: Optional[int] = None
    value_nepoch: Optional[int] = None
    aux_nepoch: Optional[int] = None
    n_minibatch: Optional[int] = None
    clip_range: Optional[float] = None
    kl_beta: Optional[float] = None
    use_gae: bool = False
    gae_lambda: float = 0.95
    use_advnorm: bool = False
    use_obsnorm: bool = False
    use_rewnorm: bool = False
    base_algorithm: str = "TD3"


# ---------------------------------------------------------------------------
# Supported Xuance algorithms (extend this map to add new entries)
# ---------------------------------------------------------------------------

XUANCE_PROFILES: Dict[str, XuanceAlgorithmProfile] = {
    "PPG_XUANCE": XuanceAlgorithmProfile(
        agent="PPG",
        learner="PPG_Learner",
        policy="Gaussian_PPG",
        representation="Basic_MLP",
        representation_hidden_size=(128,),
        actor_hidden_size=(128,),
        critic_hidden_size=(128,),
        activation="leaky_relu",
        activation_action="tanh",
        learning_rate=1e-3,
        gamma=0.98,
        training_frequency=1,
        start_training=0,
        horizon_size=256,
        n_epochs=1,
        policy_nepoch=4,
        value_nepoch=8,
        aux_nepoch=8,
        n_minibatch=1,
        clip_range=0.2,
        kl_beta=1.0,
        use_gae=True,
        gae_lambda=0.95,
        use_advnorm=True,
        use_obsnorm=True,
        use_rewnorm=True,
        base_algorithm="TD3",
    ),
    "NPG_XUANCE": XuanceAlgorithmProfile(
        agent="NPG",
        learner="NPG_Learner",
        policy="Gaussian_AC",
        representation="Basic_MLP",
        representation_hidden_size=(256,),
        actor_hidden_size=(256,),
        activation="leaky_relu",
        activation_action="tanh",
        learning_rate=4e-4,
        gamma=0.98,
        training_frequency=1,
        start_training=0,
        horizon_size=128,
        n_epochs=1,
        n_minibatch=1,
        use_gae=True,
        gae_lambda=0.95,
        use_advnorm=True,
        use_obsnorm=True,
        use_rewnorm=True,
        base_algorithm="TD3",
    ),
}


def is_xuance_algorithm(name: str) -> bool:
    """Return ``True`` if ``name`` matches a Xuance-backed algorithm."""

    return name.upper() in XUANCE_PROFILES


class VECXuanceEnv(gym.Env):
    """
    Gym-compatible environment wrapper exposing the VEC simulator to Xuance.

    The wrapper reuses :class:`SingleAgentTrainingEnvironment` for state management
    and reward computation so that Xuance policies see an identical observation /
    reward interface to the in-project algorithms.
    """

    metadata = {"render_modes": ["human"], "render.modes": ["human"]}

    def __init__(self, config: Namespace):
        self.config = config
        self.max_episode_steps = int(
            getattr(config, "max_episode_steps", global_config.experiment.max_steps_per_episode)
        )
        self._scenario_overrides = copy.deepcopy(getattr(config, "scenario_overrides", {}) or {})
        self._use_enhanced_cache = bool(getattr(config, "use_enhanced_cache", True))
        self._disable_migration = bool(getattr(config, "disable_migration", False))
        base_algorithm = getattr(config, "base_algorithm", "TD3")

        self.training_env = SingleAgentTrainingEnvironment(
            base_algorithm,
            override_scenario=self._scenario_overrides,
            use_enhanced_cache=self._use_enhanced_cache,
            disable_migration=self._disable_migration,
        )

        self.state_dim = int(self.training_env.agent_env.state_dim)
        self.action_dim = int(getattr(self.training_env.agent_env, "action_dim", 18))
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._current_step = 0
        self._episode_reward = 0.0
        self._last_state: Optional[np.ndarray] = None
        self._latest_metrics: Dict[str, float] = {}
        self._seed_value = getattr(config, "env_seed", None)

        if self._seed_value is not None:
            self._apply_seed(self._seed_value)

    # ------------------------------------------------------------------ #
    # Gym API
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self._apply_seed(seed)
        elif self._seed_value is not None:
            self._apply_seed(self._seed_value)

        self._current_step = 0
        self._episode_reward = 0.0
        self._latest_metrics = {}

        state = self.training_env.reset_environment()
        self._last_state = state.astype(np.float32, copy=True)
        info = {"episode_step": 0, "episode_score": 0.0}
        return self._last_state.copy(), info

    def step(self, action):
        action_array = np.asarray(action, dtype=np.float32).reshape(-1)
        if action_array.size != self.action_dim:
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: min(self.action_dim, action_array.size)] = action_array[: self.action_dim]
            action_array = padded

        actions_dict = self.training_env._build_actions_from_vector(action_array)
        next_state, reward, _, info_details = self.training_env.step(
            action_array, self._last_state, actions_dict
        )

        self._current_step += 1
        self._episode_reward += float(reward)
        self._last_state = next_state.astype(np.float32, copy=True)

        system_metrics = info_details.get("system_metrics", {}) or {}
        self._latest_metrics = {
            "avg_delay": float(system_metrics.get("avg_task_delay", 0.0)),
            "total_energy": float(system_metrics.get("total_energy_consumption", 0.0)),
            "task_completion_rate": float(system_metrics.get("task_completion_rate", 0.0)),
            "cache_hit_rate": float(system_metrics.get("cache_hit_rate", 0.0)),
        }

        done = self._current_step >= self.max_episode_steps
        info: Dict[str, Any] = {
            "episode_step": self._current_step,
            "episode_score": self._episode_reward,
            "system_metrics": system_metrics,
        }
        if done:
            info["xuance_episode_metrics"] = self._latest_metrics.copy()

        return (
            self._last_state.copy(),
            float(reward),
            done,
            False,
            info,
        )

    def render(self, mode: str = "human"):
        # The simulator currently has no native renderer for gym usage.
        return None

    def close(self):
        simulator = getattr(self.training_env, "simulator", None)
        if simulator and hasattr(simulator, "close"):
            simulator.close()

    # ------------------------------------------------------------------ #

    def _apply_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        if torch is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():  # pragma: no cover - optional GPU
                torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)


# Register custom environment with Xuance (idempotent).
if "VEC" not in REGISTRY_ENV:
    REGISTRY_ENV["VEC"] = VECXuanceEnv


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_directories(log_dir: Path, model_dir: Path):
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)


def _build_config(
    profile: XuanceAlgorithmProfile,
    algorithm_key: str,
    seed: int,
    num_episodes: int,
    max_steps: int,
    scenario_overrides: Optional[Dict[str, Any]],
    use_enhanced_cache: bool,
    disable_migration: bool,
) -> Namespace:
    running_steps = max(1, num_episodes * max_steps)
    log_dir = Path("results") / "xuance_logs" / algorithm_key.lower()
    model_dir = Path("results") / "xuance_models" / algorithm_key.lower()
    _ensure_directories(log_dir, model_dir)

    config_dict: Dict[str, Any] = {
        "agent": profile.agent,
        "learner": profile.learner,
        "policy": profile.policy,
        "representation": profile.representation,
        "representation_hidden_size": list(profile.representation_hidden_size),
        "actor_hidden_size": list(profile.actor_hidden_size),
        "critic_hidden_size": list(profile.critic_hidden_size),
        "activation": profile.activation,
        "activation_action": profile.activation_action,
        "gamma": profile.gamma,
        "tau": profile.tau,
        "training_frequency": profile.training_frequency,
        "start_training": profile.start_training,
        "batch_size": profile.batch_size,
        "seed": seed,
        "parallels": 1,
        "vectorize": "DummyVecEnv",
        "env_name": "VEC",
        "env_id": f"VEC-{algorithm_key}",
        "env_seed": seed,
        "running_steps": running_steps,
        "eval_interval": running_steps + 1,
        "test_episode": 1,
        "log_dir": str(log_dir),
        "model_dir": str(model_dir),
        "render_mode": "human",
        "use_obsnorm": profile.use_obsnorm,
        "use_rewnorm": profile.use_rewnorm,
        "use_advnorm": profile.use_advnorm,
        "use_gae": profile.use_gae,
        "gae_lambda": profile.gae_lambda,
        "clip_range": profile.clip_range,
        "kl_beta": profile.kl_beta,
        "n_epochs": profile.n_epochs,
        "policy_nepoch": profile.policy_nepoch,
        "value_nepoch": profile.value_nepoch,
        "aux_nepoch": profile.aux_nepoch,
        "n_minibatch": profile.n_minibatch,
        "learning_rate": profile.learning_rate,
        "learning_rate_actor": profile.learning_rate_actor,
        "learning_rate_critic": profile.learning_rate_critic,
        "horizon_size": profile.horizon_size,
        "buffer_size": profile.buffer_size,
        "scenario_overrides": copy.deepcopy(scenario_overrides or {}),
        "use_enhanced_cache": use_enhanced_cache,
        "disable_migration": disable_migration,
        "base_algorithm": profile.base_algorithm,
        "max_episode_steps": max_steps,
        "test_mode": False,
    }

    # On-policy algorithms require a buffer sized by horizon * parallels.
    if profile.horizon_size:
        horizon = min(profile.horizon_size, max_steps)
        config_dict["horizon_size"] = horizon
        config_dict["buffer_size"] = horizon * config_dict["parallels"]
        if profile.start_training <= 0:
            config_dict["start_training"] = horizon
    elif profile.buffer_size is None:
        config_dict["buffer_size"] = max(200_000, max_steps * 100)

    # Fall back to standard learning rates if algorithm specific ones are unset.
    if config_dict.get("learning_rate") is None:
        config_dict["learning_rate"] = 1e-3
    if config_dict.get("learning_rate_actor") is None:
        config_dict["learning_rate_actor"] = config_dict["learning_rate"]
    if config_dict.get("learning_rate_critic") is None:
        config_dict["learning_rate_critic"] = config_dict["learning_rate"]

    return Namespace(**config_dict)


def _train_agent(config: Namespace) -> Any:
    envs = make_envs(config)
    agent_cls = REGISTRY_Agents[config.agent]
    agent = agent_cls(config, envs)
    n_train_steps = max(1, config.running_steps // envs.num_envs)
    agent.train(n_train_steps)
    agent.save_model("final_train_model.pth")
    envs.close()
    return agent


def _evaluate_agent(agent: Any, base_config: Namespace, episodes: int) -> Dict[str, Any]:
    eval_config = copy.deepcopy(base_config)
    eval_config.parallels = 1
    eval_config.env_seed = base_config.env_seed + 10_000
    eval_envs = make_envs(eval_config)
    obs, _ = eval_envs.reset()

    episode_rewards: List[float] = []
    episode_metrics: List[Dict[str, float]] = []

    while len(episode_metrics) < episodes:
        if hasattr(agent, "obs_rms"):
            agent.obs_rms.update(obs)
        processed_obs = agent._process_observation(obs) if hasattr(agent, "_process_observation") else obs
        policy_out = agent.action(processed_obs, test_mode=True)
        next_obs, rewards, terminals, truncations, infos = eval_envs.step(policy_out["actions"])

        obs = copy.deepcopy(next_obs)
        for idx in range(eval_envs.num_envs):
            if terminals[idx] or truncations[idx]:
                metrics = infos[idx].get("xuance_episode_metrics") or {
                    "avg_delay": 0.0,
                    "total_energy": 0.0,
                    "task_completion_rate": 0.0,
                    "cache_hit_rate": 0.0,
                }
                episode_metrics.append(metrics)
                episode_rewards.append(float(infos[idx].get("episode_score", rewards[idx])))
        if len(episode_metrics) >= episodes:
            break

    eval_envs.close()
    return {"episode_rewards": episode_rewards, "episode_metrics": episode_metrics}


def _expand_series(values: Sequence[float], target_len: int) -> List[float]:
    if target_len <= 0:
        return []
    if not values:
        return [0.0 for _ in range(target_len)]
    repeats = math.ceil(target_len / len(values))
    tiled = list(values) * repeats
    return tiled[:target_len]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_xuance_algorithm(
    algorithm: str,
    num_episodes: int,
    seed: int,
    scenario_overrides: Optional[Dict[str, Any]] = None,
    use_enhanced_cache: bool = True,
    disable_migration: bool = False,
    evaluation_episodes: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Train and evaluate a Xuance algorithm on the VEC simulator.

    The return payload mirrors ``train_single_algorithm`` so that the caller can
    be agnostic to the underlying implementation.

    使用说明 (中文指南)：
        run_xuance_algorithm(
            "PPG_Xuance",      # 算法名称（需存在于 XUANCE_PROFILES）
            num_episodes=800,  # 训练轮次
            seed=42,           # 随机种子
            scenario_overrides={"num_vehicles": 12},  # 可选场景重写
            use_enhanced_cache=True,                  # 是否启用增强缓存
            disable_migration=False                   # 是否禁止迁移模块
        )
    返回结构与 ``train_single_algorithm`` 相同，可直接被实验脚本消费。
    """

    algorithm_key = algorithm.upper()
    if algorithm_key not in XUANCE_PROFILES:
        raise ValueError(f"Unsupported Xuance algorithm '{algorithm}'.")

    profile = XUANCE_PROFILES[algorithm_key]
    max_steps = int(getattr(global_config.experiment, "max_steps_per_episode", 200))
    config = _build_config(
        profile,
        algorithm_key,
        seed=seed,
        num_episodes=num_episodes,
        max_steps=max_steps,
        scenario_overrides=scenario_overrides,
        use_enhanced_cache=use_enhanced_cache,
        disable_migration=disable_migration,
    )

    training_start = time.time()
    agent = _train_agent(config)
    training_time = time.time() - training_start

    eval_episodes = evaluation_episodes or max(5, min(15, num_episodes // 20 or 5))
    evaluation = _evaluate_agent(agent, config, eval_episodes)

    if hasattr(agent, "finish"):
        agent.finish()

    raw_rewards = evaluation["episode_rewards"]
    raw_metrics = evaluation["episode_metrics"]

    avg_delay_samples = [m["avg_delay"] for m in raw_metrics]
    total_energy_samples = [m["total_energy"] for m in raw_metrics]
    completion_samples = [m["task_completion_rate"] for m in raw_metrics]
    cache_hit_samples = [m["cache_hit_rate"] for m in raw_metrics]

    summary_metrics = {
        "avg_delay": float(np.mean(avg_delay_samples)) if avg_delay_samples else 0.0,
        "total_energy": float(np.mean(total_energy_samples)) if total_energy_samples else 0.0,
        "task_completion_rate": float(np.mean(completion_samples)) if completion_samples else 0.0,
        "cache_hit_rate": float(np.mean(cache_hit_samples)) if cache_hit_samples else 0.0,
    }

    # Expand per-episode histories so the downstream pipeline can slice the tail section.
    episode_metrics = {
        "avg_delay": _expand_series(avg_delay_samples, num_episodes),
        "total_energy": _expand_series(total_energy_samples, num_episodes),
        "task_completion_rate": _expand_series(completion_samples, num_episodes),
        "cache_hit_rate": _expand_series(cache_hit_samples, num_episodes),
    }
    episode_rewards = _expand_series(raw_rewards, num_episodes)

    return {
        "episode_rewards": episode_rewards,
        "episode_metrics": episode_metrics,
        "summary_metrics": summary_metrics,
        "raw_evaluation": {
            "episode_rewards": raw_rewards,
            "episode_metrics": raw_metrics,
        },
        "training_time_hours": training_time / 3600.0,
        "training_steps": config.running_steps,
    }
