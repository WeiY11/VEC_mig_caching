#!/usr/bin/env python3
"""
Nath & Wu 2020 - DDPG for Dynamic Computation Offloading (GLOBECOM 2020).

Paper: "Dynamic Computation Offloading and Resource Allocation for Multi-user Mobile Edge Computing"

Key features from the paper:
- DDPG with Actor-Critic architecture
- Hidden layers: 8N and 4N (where N = number of MUs)
- OU noise for exploration
- Sigmoid output layer for bounded actions
- State: task request vector k_t + channel matrix H_t
- Action: offloading decision x_t + power allocation p_t + MEC resource allocation f_t

Self-contained implementation compatible with the VEC environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class OrnsteinUhlenbeckNoise:
    """OU noise for exploration (as specified in the paper)."""
    def __init__(self, size: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.state = self.mu.copy()
    
    def reset(self):
        self.state = self.mu.copy()
    
    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state = self.state + dx
        return self.state.astype(np.float32)


class Actor(nn.Module):
    """
    Actor network for Nath DDPG.
    
    Paper specifies: 4-layer fully connected with hidden layers 8N and 4N.
    Output layer uses sigmoid to bound actions.
    """
    def __init__(self, s_dim: int, a_dim: int, num_mus: int = 6):
        super().__init__()
        h1 = max(8 * num_mus, 64)  # 8N as specified in paper
        h2 = max(4 * num_mus, 32)  # 4N as specified in paper
        
        self.fc1 = nn.Linear(s_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, a_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        # Paper specifies uniform initialization U[-3e-3, 3e-3] for output layer
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)
    
    def forward(self, s: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(s))
        x = torch.relu(self.fc2(x))
        return torch.sigmoid(self.out(x))  # Sigmoid to bound actions in [0,1]


class Critic(nn.Module):
    """
    Critic network for Nath DDPG.
    
    Same architecture: 4-layer with hidden layers 8N and 4N.
    Outputs Q(s, a).
    """
    def __init__(self, s_dim: int, a_dim: int, num_mus: int = 6):
        super().__init__()
        h1 = max(8 * num_mus, 64)
        h2 = max(4 * num_mus, 32)
        
        self.fc1 = nn.Linear(s_dim + a_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.out.bias, -3e-3, 3e-3)
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    def __init__(self, capacity: int = 50000):
        self.capacity = capacity
        self.buffer = []
        self.idx = 0
    
    def push(self, s, a, r, s2, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.idx] = (s, a, r, s2, done)
        self.idx = (self.idx + 1) % self.capacity
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s2, done = zip(*batch)
        return (
            np.array(s, dtype=np.float32),
            np.array(a, dtype=np.float32),
            np.array(r, dtype=np.float32).reshape(-1, 1),
            np.array(s2, dtype=np.float32),
            np.array(done, dtype=np.float32).reshape(-1, 1),
        )
    
    def __len__(self):
        return len(self.buffer)


@dataclass
class NathDDPGConfig:
    """Configuration for Nath DDPG (paper parameters)."""
    gamma: float = 0.99
    tau: float = 1e-3  # Soft update rate (paper specifies τ=10^-3)
    actor_lr: float = 1e-4  # Paper: αμ = 10^-4
    critic_lr: float = 1e-3  # Paper: αQ = 10^-3
    buffer_size: int = 50000  # Paper: |R| = 50000
    batch_size: int = 128  # Paper: B = 128
    start_steps: int = 1000  # Random exploration before learning
    num_mus: int = 6  # Number of mobile users (paper default)
    device: str = "cpu"


class NathDDPGAgent:
    """
    DDPG Agent for MEC offloading (Nath & Wu 2020).
    
    Solves: min E[Σ(energy + ω*delay)]
    Actions: offloading decisions + power allocation + MEC resource allocation
    """
    def __init__(self, s_dim: int, a_dim: int, cfg: NathDDPGConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.a_dim = a_dim
        
        # Networks
        self.actor = Actor(s_dim, a_dim, cfg.num_mus).to(self.device)
        self.actor_target = Actor(s_dim, a_dim, cfg.num_mus).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic(s_dim, a_dim, cfg.num_mus).to(self.device)
        self.critic_target = Critic(s_dim, a_dim, cfg.num_mus).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers (paper specifies Adam)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(cfg.buffer_size)
        
        # OU noise (as specified in paper)
        self.noise = OrnsteinUhlenbeckNoise(a_dim)
        
        self.total_steps = 0
    
    def act(self, s: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given state."""
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a = self.actor(st).cpu().numpy()[0]
        
        if not deterministic and self.total_steps >= self.cfg.start_steps:
            a = a + self.noise.sample()
            a = np.clip(a, 0, 1)
        
        return a
    
    def store(self, s, a, r, s2, done):
        """Store transition in replay buffer."""
        self.buffer.push(s, a, r, s2, done)
        self.total_steps += 1
    
    def update(self) -> Dict[str, float]:
        """
        DDPG update step.
        
        Paper Algorithm 1:
        - Sample mini-batch from replay buffer
        - Update critic using TD error
        - Update actor using policy gradient
        - Soft update target networks
        """
        if len(self.buffer) < self.cfg.batch_size:
            return {}
        
        # Sample mini-batch (Step 10 in paper)
        s, a, r, s2, done = self.buffer.sample(self.cfg.batch_size)
        s = torch.as_tensor(s, device=self.device)
        a = torch.as_tensor(a, device=self.device)
        r = torch.as_tensor(r, device=self.device)
        s2 = torch.as_tensor(s2, device=self.device)
        done = torch.as_tensor(done, device=self.device)
        
        # Critic update (Step 11 in paper - Eq. 20)
        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = r + self.cfg.gamma * (1 - done) * self.critic_target(s2, a2)
        
        q_current = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q_current, q_target)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        # Actor update (Step 12 in paper - Eq. 21)
        actor_loss = -self.critic(s, self.actor(s)).mean()
        
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()
        
        # Soft update target networks (Step 13 in paper)
        for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
            pt.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * pt.data)
        for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
            pt.data.copy_(self.cfg.tau * p.data + (1 - self.cfg.tau) * pt.data)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
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


def train_nath_ddpg(
    env,
    cfg: NathDDPGConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train Nath DDPG agent on a gym-style environment.
    
    Implements Algorithm 1 from Nath & Wu 2020 GLOBECOM paper.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    
    agent = NathDDPGAgent(s_dim, a_dim, cfg)
    
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
    agent.noise.reset()
    episode = 0
    
    for step in range(1, max_steps + 1):
        # Random action for initial exploration (Step 7 in paper)
        if step < cfg.start_steps:
            a = env.action_space.sample()
            # Normalize to [0, 1] if needed
            a_low = env.action_space.low
            a_high = env.action_space.high
            a = (a - a_low) / (a_high - a_low + 1e-8)
        else:
            a = agent.act(s, deterministic=False)
        
        # Scale action to environment range
        a_low = env.action_space.low
        a_high = env.action_space.high
        a_env = a_low + a * (a_high - a_low)
        
        s2, r, done, info = _step_env(env, a_env)
        
        # Collect metrics
        if "system_metrics" in info:
            m = info["system_metrics"]
            cur_ep_delay.append(m.get("avg_task_delay", 0.0))
            cur_ep_energy += m.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(m.get("task_completion_rate", 0.0))
            cur_ep_dropped += m.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(m.get("cache_hit_rate", 0.0))
        
        # Store transition (Step 9 in paper)
        agent.store(s, a, r, s2, done)
        s = s2
        ep_r += r
        
        # Update networks
        if step >= cfg.start_steps:
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
            agent.noise.reset()
            episode += 1
    
    return {
        "episode_rewards": ep_rewards,
        "episode_metrics": ep_metrics,
        "episodes": episode,
        "config": cfg.__dict__,
    }


__all__ = ["NathDDPGConfig", "NathDDPGAgent", "train_nath_ddpg"]
