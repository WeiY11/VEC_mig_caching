#!/usr/bin/env python3
"""
Liu & Cao 2022 - Bayesian Optimization for Video Analytics Offloading (IEEE TWC).

Paper: "Deep Learning Video Analytics Through Online Learning Based Edge Computing"

Key features from the paper:
- Contextual Multi-Armed Bandit formulation
- Gaussian Process (GP) as prior for unknown reward function
- GP-UCB acquisition function for action selection
- Adaptive Bayesian Optimization for time-varying environments
- Ornstein-Uhlenbeck temporal kernel for handling dynamics

Self-contained implementation compatible with the VEC environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Optional

import numpy as np

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


@dataclass
class LiuBOConfig:
    """Configuration for Liu Bayesian Optimization (paper parameters)."""
    beta_init: float = 2.0  # UCB exploration parameter
    length_scale: float = 1.0  # GP kernel length scale
    noise_level: float = 0.1  # Observation noise
    decay_factor: float = 0.95  # λ for time-varying environments (Eq. 17)
    window_size: int = 100  # Max samples to keep (sliding window)
    n_random_samples: int = 1000  # Action samples for searching (paper: thousands)
    alpha_weight: float = 1.0  # Weight between processing rate and accuracy
    device: str = "cpu"


class GaussianProcessBandit:
    """
    GP-UCB bandit for continuous action space.
    
    Implements the adaptive Bayesian Optimization from Liu & Cao 2022.
    """
    def __init__(self, a_dim: int, cfg: LiuBOConfig):
        self.a_dim = a_dim
        self.cfg = cfg
        
        # Collected observations D_t = {(z_1, y_1), ..., (z_t, y_t)}
        self.contexts: List[np.ndarray] = []  # action-context pairs z
        self.rewards: List[float] = []  # observed rewards y
        self.timestamps: List[int] = []  # for time-decay weighting
        
        self.step = 0
        self._gp = None
        self._needs_refit = True
    
    def _compute_temporal_weights(self) -> np.ndarray:
        """
        Compute temporal weights using Ornstein-Uhlenbeck process (Eq. 17-18).
        
        w_i = (1 - λ)^((t - t_i) / 2)
        """
        if not self.timestamps:
            return np.array([])
        
        current_t = self.step
        weights = []
        for t_i in self.timestamps:
            delta = (current_t - t_i) / 2.0
            w = (1 - self.cfg.decay_factor) ** delta
            weights.append(max(w, 1e-6))  # Avoid zero weights
        return np.array(weights)
    
    def _fit_gp(self):
        """Fit Gaussian Process to collected observations."""
        if not HAS_SKLEARN or len(self.contexts) < 5:
            self._gp = None
            return
        
        X = np.array(self.contexts)
        y = np.array(self.rewards)
        weights = self._compute_temporal_weights()
        
        # Weight samples by recency
        # Note: sklearn GP doesn't natively support sample weights,
        # so we approximate by weighting the y values
        y_weighted = y * np.sqrt(weights)
        
        kernel = RBF(length_scale=self.cfg.length_scale) + WhiteKernel(noise_level=self.cfg.noise_level)
        
        try:
            self._gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=1e-6,
                n_restarts_optimizer=2,
                normalize_y=True
            )
            self._gp.fit(X, y_weighted)
            self._needs_refit = False
        except Exception:
            self._gp = None
    
    def _ucb_score(self, actions: np.ndarray) -> np.ndarray:
        """
        Compute UCB scores for candidate actions (Eq. 16).
        
        UCB(a) = μ(a) + β^(1/2) * σ(a)
        """
        if self._gp is None:
            # No GP fitted, return random scores
            return np.random.randn(len(actions))
        
        # β_t as specified in paper (Eq. 9)
        t = max(self.step, 1)
        d = self.a_dim
        delta = 0.1
        beta = 2 * np.log(t**2 * np.pi**2 / (3 * delta)) + 2 * d * np.log(t * np.sqrt(np.log(1/delta)))
        beta = max(beta, self.cfg.beta_init)
        
        try:
            mu, sigma = self._gp.predict(actions, return_std=True)
            return mu + np.sqrt(beta) * sigma
        except Exception:
            return np.random.randn(len(actions))
    
    def select_action(self, context: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Select action using GP-UCB (Algorithm: Randomized Action Searching).
        
        1. Sample thousands of actions from action space
        2. Compute UCB score for each
        3. Select action with maximum UCB
        """
        self.step += 1
        
        # Refit GP if needed
        if self._needs_refit and len(self.contexts) >= 5:
            self._fit_gp()
        
        # Sample candidate actions uniformly from [0, 1]^d
        candidates = np.random.uniform(0, 1, size=(self.cfg.n_random_samples, self.a_dim))
        
        # If context provided, concatenate (for contextual bandit)
        if context is not None:
            context_repeated = np.tile(context, (self.cfg.n_random_samples, 1))
            candidates_with_context = np.hstack([candidates, context_repeated])
        else:
            candidates_with_context = candidates
        
        # Compute UCB scores
        ucb_scores = self._ucb_score(candidates_with_context)
        
        # Select best action
        best_idx = np.argmax(ucb_scores)
        return candidates[best_idx]
    
    def update(self, action: np.ndarray, reward: float, context: Optional[np.ndarray] = None):
        """
        Update GP with new observation.
        
        Implements sliding window to handle time-varying environments.
        """
        # Create observation (action-context pair)
        if context is not None:
            z = np.concatenate([action, context])
        else:
            z = action
        
        self.contexts.append(z)
        self.rewards.append(reward)
        self.timestamps.append(self.step)
        
        # Sliding window: keep only recent observations
        if len(self.contexts) > self.cfg.window_size:
            self.contexts = self.contexts[-self.cfg.window_size:]
            self.rewards = self.rewards[-self.cfg.window_size:]
            self.timestamps = self.timestamps[-self.cfg.window_size:]
        
        self._needs_refit = True


class LiuBayesianOptOffloader:
    """
    Bayesian Optimization based offloading (Liu & Cao 2022).
    
    Uses GP-UCB to select server and resolution for video analytics.
    Adapted for VEC task offloading.
    """
    def __init__(self, a_dim: int, cfg: LiuBOConfig):
        self.cfg = cfg
        self.a_dim = a_dim
        self.bandit = GaussianProcessBandit(a_dim, cfg)
        self.last_action = None
    
    def select_action(self, state: Optional[np.ndarray] = None) -> np.ndarray:
        """Select action using GP-UCB."""
        # Use state as context for contextual bandit
        action = self.bandit.select_action(context=state)
        self.last_action = action
        return action
    
    def select_action_with_dim(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """Select action matching the required dimension."""
        if action_dim != self.a_dim:
            # Reinitialize if dimension changed
            self.a_dim = action_dim
            self.bandit = GaussianProcessBandit(action_dim, self.cfg)
        
        action = self.select_action(state)
        
        # Scale to [-1, 1] for compatibility
        action = action * 2 - 1
        return action.astype(np.float32)
    
    def update(self, reward: float, state: Optional[np.ndarray] = None):
        """Update GP with observed reward."""
        if self.last_action is not None:
            self.bandit.update(self.last_action, reward, context=state)


def _reset_env(env):
    res = env.reset()
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0]
    return res


def _step_env(env, action):
    res = env.step(action)
    if isinstance(res, tuple) and len(res) == 5:
        s2, r, terminated, truncated, info = res
        done = bool(terminated or truncated)
    else:
        s2, r, done, info = res
    return s2, r, done, info


def train_liu_bo(
    env,
    cfg: LiuBOConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train Liu Bayesian Optimization agent on a gym-style environment.
    
    Implements the OLSO algorithm from Liu & Cao 2022 TWC paper.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    a_dim = env.action_space.shape[0]
    a_low = env.action_space.low
    a_high = env.action_space.high
    
    agent = LiuBayesianOptOffloader(a_dim, cfg)
    
    ep_rewards = []
    ep_metrics = {
        "avg_task_delay": [],
        "total_energy_consumption": [],
        "task_completion_rate": [],
        "dropped_tasks": [],
        "cache_hit_rate": []
    }
    ep_r = 0.0
    
    # Per-episode accumulators
    cur_ep_delay = []
    cur_ep_energy = 0.0
    cur_ep_completed = []
    cur_ep_dropped = 0
    cur_ep_cache_hits = []
    
    s = _reset_env(env)
    episode = 0
    
    for step in range(1, max_steps + 1):
        # Select action using GP-UCB
        a = agent.select_action(state=s)
        
        # Scale to environment action range
        a_env = a_low + a * (a_high - a_low)
        a_env = np.clip(a_env, a_low, a_high)
        
        s2, r, done, info = _step_env(env, a_env)
        
        # Update GP with observed reward
        agent.update(r, state=s)
        
        # Collect metrics
        if "system_metrics" in info:
            m = info["system_metrics"]
            cur_ep_delay.append(m.get("avg_task_delay", 0.0))
            cur_ep_energy += m.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(m.get("task_completion_rate", 0.0))
            cur_ep_dropped += m.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(m.get("cache_hit_rate", 0.0))
        
        s = s2
        ep_r += r
        
        if done:
            ep_rewards.append(ep_r)
            
            # Aggregate episode metrics
            ep_metrics["avg_task_delay"].append(np.mean(cur_ep_delay) if cur_ep_delay else 0.0)
            ep_metrics["total_energy_consumption"].append(cur_ep_energy)
            ep_metrics["task_completion_rate"].append(np.mean(cur_ep_completed) if cur_ep_completed else 0.0)
            ep_metrics["dropped_tasks"].append(cur_ep_dropped)
            ep_metrics["cache_hit_rate"].append(np.mean(cur_ep_cache_hits) if cur_ep_cache_hits else 0.0)
            
            # Reset accumulators
            cur_ep_delay = []
            cur_ep_energy = 0.0
            cur_ep_completed = []
            cur_ep_dropped = 0
            cur_ep_cache_hits = []
            
            if progress:
                progress(step, np.mean(ep_rewards[-10:]), ep_r)
            
            s, ep_r = _reset_env(env), 0.0
            episode += 1
    
    return {
        "episode_rewards": ep_rewards,
        "episode_metrics": ep_metrics,
        "episodes": episode,
        "config": cfg.__dict__,
    }


__all__ = ["LiuBOConfig", "LiuBayesianOptOffloader", "train_liu_bo"]
