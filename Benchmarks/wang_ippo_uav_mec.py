#!/usr/bin/env python3
"""
IPPO (Independent PPO) for UAV-enabled MEC (Wang et al. 2025 style).

Key features from the paper:
- Independent PPO for multi-agent coordination (UAVs + RSUs)
- PPO Clipping for stable policy updates
- GAE (Generalized Advantage Estimation) for variance reduction
- Joint task offloading and migration optimization

Self-contained implementation compatible with the VEC environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


def mlp(in_dim: int, out_dim: int, hidden: Tuple[int, int] = (256, 256)) -> nn.Sequential:
    """Simple MLP with ReLU activations."""
    h1, h2 = hidden
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, out_dim),
    )


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.
    
    Actor outputs mean and log_std for Gaussian policy.
    Critic outputs state value V(s).
    """
    def __init__(self, s_dim: int, a_dim: int, hidden: Tuple[int, int] = (256, 256)):
        super().__init__()
        h1, h2 = hidden
        
        # Shared feature extractor (optional, can be separate)
        self.shared = nn.Sequential(
            nn.Linear(s_dim, h1),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, a_dim),
            nn.Tanh(),  # Bound actions to [-1, 1], then scale
        )
        self.actor_log_std = nn.Parameter(torch.zeros(a_dim))  # Learnable log_std
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Returns:
            mean: Action mean
            std: Action std
            value: State value V(s)
        """
        features = self.shared(s)
        mean = self.actor_mean(features)
        std = torch.exp(self.actor_log_std.clamp(-20, 2))
        value = self.critic(features)
        return mean, std, value
    
    def get_action(self, s: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value
        """
        mean, std, value = self(s)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(s.shape[0], 1, device=s.device)
        else:
            dist = Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate(self, s: torch.Tensor, a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for PPO update.
        
        Returns:
            log_prob: Log probability of actions
            value: State values
            entropy: Policy entropy
        """
        mean, std, value = self(s)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, value, entropy


class TrajectoryBuffer:
    """
    Trajectory buffer for on-policy PPO.
    
    Stores complete episodes/trajectories for batch updates.
    """
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
        self.next_states = []
    
    def add(self, s, a, r, v, log_prob, done, s_next):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.next_states.append(s_next)
    
    def compute_gae(self, last_value: float, gamma: float, gae_lambda: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute GAE (Generalized Advantage Estimation).
        
        Returns:
            advantages: GAE advantages
            returns: Discounted returns (advantages + values)
        """
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32).squeeze()
        dones = np.array(self.dones, dtype=np.float32)
        
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        
        # Append last value for bootstrapping
        values_ext = np.append(values, last_value)
        
        gae = 0.0
        for t in reversed(range(T)):
            delta = rewards[t] + gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae = delta + gamma * gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae
        
        returns = advantages + values
        return advantages, returns
    
    def get_batch(self, last_value: float, gamma: float, gae_lambda: float) -> Dict[str, np.ndarray]:
        """Get batch data for PPO update."""
        advantages, returns = self.compute_gae(last_value, gamma, gae_lambda)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            "states": np.array(self.states, dtype=np.float32),
            "actions": np.array(self.actions, dtype=np.float32),
            "log_probs": np.array(self.log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
        }
    
    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        self.dones.clear()
        self.next_states.clear()
    
    def __len__(self):
        return len(self.states)


@dataclass
class IPPOConfig:
    """IPPO configuration (Wang et al. 2025 style)."""
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2  # PPO clipping parameter
    vf_coef: float = 0.5       # Value function loss coefficient
    ent_coef: float = 0.01     # Entropy bonus coefficient
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    n_epochs: int = 10         # PPO update epochs per batch
    batch_size: int = 64       # Mini-batch size for PPO updates
    rollout_steps: int = 2048  # Steps per rollout before update
    device: str = "cpu"
    num_rsus: int = 4
    num_uavs: int = 2
    hidden_dims: Tuple[int, int] = (256, 256)


class IPPOAgent:
    """
    IPPO Agent for UAV-MEC task offloading and migration.
    
    In the independent PPO setting, each agent (or the central controller)
    maintains its own policy but shares the same algorithm structure.
    """
    def __init__(self, s_dim: int, a_dim: int, act_limit: float, cfg: IPPOConfig):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)
        
        # Actor-Critic network
        self.ac = ActorCritic(s_dim, a_dim, hidden=cfg.hidden_dims).to(self.device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=cfg.learning_rate)
        
        # Trajectory buffer
        self.buffer = TrajectoryBuffer()
        self.total_steps = 0
    
    def act(self, s: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float, float]:
        """
        Select action given state.
        
        Returns:
            action: Scaled action
            log_prob: Log probability
            value: State value
        """
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, value = self.ac.get_action(st, deterministic=deterministic)
        
        action = action.cpu().numpy()[0] * self.act_limit
        log_prob = log_prob.cpu().numpy()[0, 0]
        value = value.cpu().numpy()[0, 0]
        
        return action, log_prob, value
    
    def store(self, s, a, r, v, log_prob, done, s_next):
        """Store transition in trajectory buffer."""
        self.buffer.add(s, a / self.act_limit, r, v, log_prob, done, s_next)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        PPO update using collected trajectories.
        
        Returns:
            Dictionary of training metrics
        """
        if len(self.buffer) == 0:
            return {}
        
        # Get last value for GAE bootstrap
        last_s = self.buffer.next_states[-1]
        last_st = torch.as_tensor(last_s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            _, _, last_value = self.ac(last_st)
            last_value = last_value.cpu().numpy()[0, 0]
        
        # Get batch data
        batch = self.buffer.get_batch(last_value, self.cfg.gamma, self.cfg.gae_lambda)
        self.buffer.clear()
        
        # Convert to tensors
        states = torch.as_tensor(batch["states"], device=self.device)
        actions = torch.as_tensor(batch["actions"], device=self.device)
        old_log_probs = torch.as_tensor(batch["log_probs"], device=self.device).unsqueeze(-1)
        advantages = torch.as_tensor(batch["advantages"], device=self.device).unsqueeze(-1)
        returns = torch.as_tensor(batch["returns"], device=self.device).unsqueeze(-1)
        
        # PPO update epochs
        total_loss = 0.0
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_ent_loss = 0.0
        n_updates = 0
        
        dataset_size = len(states)
        indices = np.arange(dataset_size)
        
        for epoch in range(self.cfg.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.cfg.batch_size):
                end = min(start + self.cfg.batch_size, dataset_size)
                batch_idx = indices[start:end]
                
                s_batch = states[batch_idx]
                a_batch = actions[batch_idx]
                old_lp_batch = old_log_probs[batch_idx]
                adv_batch = advantages[batch_idx]
                ret_batch = returns[batch_idx]
                
                # Evaluate current policy
                log_prob, value, entropy = self.ac.evaluate(s_batch, a_batch)
                
                # PPO clipped objective
                ratio = torch.exp(log_prob - old_lp_batch)
                surr1 = ratio * adv_batch
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_epsilon, 1 + self.cfg.clip_epsilon) * adv_batch
                pg_loss = -torch.min(surr1, surr2).mean()
                
                # Value function loss
                vf_loss = nn.functional.mse_loss(value, ret_batch)
                
                # Entropy bonus
                ent_loss = -entropy.mean()
                
                # Total loss
                loss = pg_loss + self.cfg.vf_coef * vf_loss + self.cfg.ent_coef * ent_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_ent_loss += ent_loss.item()
                n_updates += 1
        
        return {
            "loss": total_loss / max(n_updates, 1),
            "pg_loss": total_pg_loss / max(n_updates, 1),
            "vf_loss": total_vf_loss / max(n_updates, 1),
            "ent_loss": total_ent_loss / max(n_updates, 1),
        }


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


def train_ippo(
    env,
    cfg: IPPOConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train IPPO agent on a gym-style environment.
    
    Wang et al. 2025 style: uses PPO with GAE for joint task offloading
    and migration optimization in UAV-enabled MEC networks.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_lim = float(env.action_space.high[0])
    
    agent = IPPOAgent(s_dim, a_dim, a_lim, cfg)
    
    ep_rewards = []
    ep_metrics = {
        "avg_task_delay": [],
        "total_energy_consumption": [],
        "task_completion_rate": [],
        "dropped_tasks": [],
        "cache_hit_rate": []
    }
    ep_r = 0.0
    
    # Accumulators for current episode
    cur_ep_delay = []
    cur_ep_energy = 0.0
    cur_ep_completed = []
    cur_ep_dropped = 0
    cur_ep_cache_hits = []
    
    s = _reset_env(env)
    episode = 0
    
    for step in range(1, max_steps + 1):
        # Select action
        a, log_prob, value = agent.act(s, deterministic=False)
        
        # Clip action to valid range
        a = np.clip(a, -a_lim, a_lim)
        
        s2, r, done, info = _step_env(env, a)
        
        # Collect metrics from info
        if "system_metrics" in info:
            m = info["system_metrics"]
            cur_ep_delay.append(m.get("avg_task_delay", 0.0))
            cur_ep_energy += m.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(m.get("task_completion_rate", 0.0))
            cur_ep_dropped += m.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(m.get("cache_hit_rate", 0.0))
        
        # Store transition
        agent.store(s, a, r, value, log_prob, done, s2)
        s = s2
        ep_r += r
        
        # PPO update every rollout_steps or at episode end
        if len(agent.buffer) >= cfg.rollout_steps or done:
            agent.update()
        
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


__all__ = ["IPPOConfig", "IPPOAgent", "train_ippo"]
