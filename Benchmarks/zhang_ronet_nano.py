#!/usr/bin/env python3
"""
Zhang et al. 2023 - RoNet: Robust Neural Assisted Network Configuration (IEEE ICC).

Paper: "RoNet: Toward Robust Neural Assisted Mobile Network Configuration"

Key features from the paper:
1. Normal Training Stage:
   - DNN predictor for Performance Efficiency (PE)
   - Randomized action searching to find optimal configuration
2. Learn-to-Attack Stage:
   - Bayesian learning with Gaussian Process
   - GP-UCB acquisition function to find adversarial attacks
3. Robust Defense Stage:
   - Model retraining on attacked samples
   - Probabilistic action selection (κ-percentile truncation)

Self-contained implementation adapted for VEC environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class PEPredictor(nn.Module):
    """
    DNN predictor for Performance Efficiency (Section III).
    
    Paper specifies: [128]x[256]x[128] full-connected layers with ReLU.
    Input: state + action
    Output: predicted PE
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianProcessAttacker:
    """
    Learn-to-Attack using Bayesian Learning (Section IV-A).
    
    Uses Gaussian Process to find adversarial perturbations on state.
    """
    def __init__(self, attack_dim: int, epsilon: float = 0.2):
        self.attack_dim = attack_dim
        self.epsilon = epsilon  # Maximum attack scale (l∞ norm constraint)
        
        # Collected attack-performance pairs
        self.attacks: List[np.ndarray] = []
        self.neg_performances: List[float] = []  # Negative PE (we minimize PE)
        
        self.length_scale = 1.0
    
    def _rbf_kernel(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """RBF (squared exponential) kernel."""
        dist = np.sum((v1 - v2) ** 2)
        return np.exp(-dist / (2 * self.length_scale ** 2))
    
    def _compute_kernel_matrix(self, attacks: List[np.ndarray]) -> np.ndarray:
        """Compute kernel matrix K."""
        n = len(attacks)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._rbf_kernel(attacks[i], attacks[j])
        return K
    
    def suggest_attack(self, n_candidates: int = 100) -> np.ndarray:
        """
        Suggest next attack using GP-UCB (Eq. 8).
        
        v_next = argmax μ(v) + β_t * σ(v)
        """
        if len(self.attacks) < 3:
            # Random exploration initially
            return np.random.uniform(-self.epsilon, self.epsilon, self.attack_dim)
        
        # Generate candidates
        candidates = np.random.uniform(-self.epsilon, self.epsilon, 
                                       size=(n_candidates, self.attack_dim))
        
        # Compute GP posterior for each candidate
        X = np.array(self.attacks)
        y = np.array(self.neg_performances)
        
        K = self._compute_kernel_matrix(self.attacks)
        K_inv = np.linalg.pinv(K + 1e-6 * np.eye(len(K)))
        
        ucb_scores = []
        t = len(self.attacks) + 1
        beta = 2 * np.log(t ** 2 * np.pi ** 2 / 6)  # Simplified β_t
        
        for v in candidates:
            # k vector
            k = np.array([self._rbf_kernel(v, x) for x in self.attacks])
            
            # Posterior mean and variance
            mu = k @ K_inv @ y
            sigma2 = self._rbf_kernel(v, v) - k @ K_inv @ k
            sigma = np.sqrt(max(sigma2, 0))
            
            ucb = mu + np.sqrt(beta) * sigma
            ucb_scores.append(ucb)
        
        best_idx = np.argmax(ucb_scores)
        return candidates[best_idx]
    
    def update(self, attack: np.ndarray, neg_performance: float):
        """Update with observed attack and resulting negative PE."""
        self.attacks.append(attack.copy())
        self.neg_performances.append(neg_performance)


@dataclass
class RoNetConfig:
    """Configuration for RoNet."""
    pe_lr: float = 1e-3  # Learning rate for PE predictor
    n_action_samples: int = 1000  # Samples for action searching
    attack_epsilon: float = 0.2  # Attack scale (paper default)
    kappa: float = 0.99  # Percentile for probabilistic selection (paper default)
    retrain_epochs: int = 50  # Retraining epochs (paper default)
    attack_iterations: int = 50  # Attack iterations per episode
    defense_enabled: bool = True  # Enable robust defense
    device: str = "cpu"


class RoNetAgent:
    """
    RoNet Agent for robust network configuration (Zhang et al. 2023).
    
    Three integrated stages:
    1. Normal training with DNN predictor
    2. Learn-to-attack with GP
    3. Robust defense with retraining and probabilistic selection
    """
    def __init__(self, s_dim: int, a_dim: int, cfg: RoNetConfig):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # DNN predictor π_θ
        self.predictor = PEPredictor(s_dim + a_dim).to(self.device)
        self.predictor_opt = optim.Adam(self.predictor.parameters(), lr=cfg.pe_lr)
        
        # Training data D = {(s, a, PE)}
        self.states: List[np.ndarray] = []
        self.actions: List[np.ndarray] = []
        self.performances: List[float] = []
        
        # Attack data for robust defense
        self.attacked_states: List[np.ndarray] = []
        self.attacked_performances: List[float] = []
        
        # Attacker (for learn-to-attack stage)
        self.attacker = GaussianProcessAttacker(s_dim, cfg.attack_epsilon)
        
        self.is_robust = False  # Whether robust defense is active
        self.step_count = 0
    
    def _train_predictor(self, states, actions, targets, epochs: int = 10):
        """Train DNN predictor on collected data."""
        if len(states) < 10:
            return
        
        X = np.hstack([np.array(states), np.array(actions)])
        y = np.array(targets).reshape(-1, 1)
        
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        
        for _ in range(epochs):
            preds = self.predictor(X_t)
            loss = nn.functional.mse_loss(preds, y_t)
            
            self.predictor_opt.zero_grad()
            loss.backward()
            self.predictor_opt.step()
    
    def _randomized_action_search(self, state: np.ndarray) -> np.ndarray:
        """
        Randomized Action Searching (Eq. 4).
        
        a* = argmax π_θ(s, a)
        """
        # Sample candidate actions
        candidates = np.random.uniform(0, 1, size=(self.cfg.n_action_samples, self.a_dim))
        
        state_repeated = np.tile(state, (self.cfg.n_action_samples, 1))
        X = np.hstack([state_repeated, candidates])
        
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            preds = self.predictor(X_t).squeeze().cpu().numpy()
        
        if self.is_robust and self.cfg.defense_enabled:
            # Probabilistic Selection (Section IV-B)
            # Keep top κ percentile, then randomly select
            k = int(self.cfg.kappa * self.cfg.n_action_samples)
            top_indices = np.argsort(preds)[-k:]
            best_idx = np.random.choice(top_indices)
        else:
            best_idx = np.argmax(preds)
        
        return candidates[best_idx]
    
    def act(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action using trained DNN predictor."""
        self.step_count += 1
        
        if len(self.states) < 50:
            # Exploration phase
            return np.random.uniform(0, 1, self.a_dim)
        
        return self._randomized_action_search(state)
    
    def update(self, state: np.ndarray, action: np.ndarray, pe: float):
        """
        Update agent with observed Performance Efficiency.
        
        PE = Prob(f(s,a) < H) / |a| (Eq. 1)
        
        In our VEC context, PE is derived from reward.
        """
        self.states.append(state.copy())
        self.actions.append(action.copy())
        self.performances.append(pe)
        
        # Periodically retrain predictor
        if len(self.states) % 50 == 0:
            self._train_predictor(self.states, self.actions, self.performances, epochs=10)
    
    def learn_to_attack(self, eval_fn: Callable[[np.ndarray, np.ndarray], float]):
        """
        Learn-to-Attack Stage (Section IV-A).
        
        Find adversarial perturbations on state using GP-UCB.
        """
        for _ in range(self.cfg.attack_iterations):
            # Suggest attack
            attack = self.attacker.suggest_attack()
            
            # Pick a random state to attack
            if len(self.states) == 0:
                continue
            idx = random.randint(0, len(self.states) - 1)
            original_state = self.states[idx]
            attacked_state = original_state + attack
            
            # Evaluate attacked performance
            action = self._randomized_action_search(attacked_state)
            attacked_pe = eval_fn(attacked_state, action)
            
            # Update attacker (minimize PE = maximize negative PE)
            self.attacker.update(attack, -attacked_pe)
            
            # Save for robust defense
            self.attacked_states.append(attacked_state)
            self.attacked_performances.append(attacked_pe)
    
    def robust_defense(self):
        """
        Robust Defense Stage (Section IV-B).
        
        Retrain predictor on attacked data and enable probabilistic selection.
        """
        if len(self.attacked_states) < 10:
            return
        
        # Model retraining (Eq. 10)
        # Train on both original and attacked data
        all_states = self.states + self.attacked_states
        all_actions = self.actions + [np.zeros(self.a_dim)] * len(self.attacked_states)
        all_performances = self.performances + self.attacked_performances
        
        self._train_predictor(all_states, all_actions, all_performances, 
                            epochs=self.cfg.retrain_epochs)
        
        self.is_robust = True


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


def train_ronet(
    env,
    cfg: RoNetConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train RoNet agent on a gym-style environment.
    
    Implements the three-stage framework from Zhang et al. 2023.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_low = env.action_space.low
    a_high = env.action_space.high
    
    agent = RoNetAgent(s_dim, a_dim, cfg)
    
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
        # Select action
        a = agent.act(s, deterministic=False)
        
        # Scale to environment action range
        a_env = a_low + a * (a_high - a_low)
        a_env = np.clip(a_env, a_low, a_high)
        
        s2, r, done, info = _step_env(env, a_env)
        
        # Use reward as Performance Efficiency proxy
        pe = r
        agent.update(s, a, pe)
        
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
            
            # Trigger robust defense periodically
            if episode > 0 and episode % 10 == 0 and cfg.defense_enabled:
                # Simple evaluation function for learn-to-attack
                def eval_fn(state, action):
                    a_env = a_low + action * (a_high - a_low)
                    a_env = np.clip(a_env, a_low, a_high)
                    _, r, _, _ = _step_env(env, a_env)
                    return r
                
                agent.learn_to_attack(eval_fn)
                agent.robust_defense()
            
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


__all__ = ["RoNetConfig", "RoNetAgent", "train_ronet"]
