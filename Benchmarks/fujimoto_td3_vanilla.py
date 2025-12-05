#!/usr/bin/env python3
"""
Vanilla TD3 implementation aligned with Fujimoto et al. (2018).
"Addressing Function Approximation Error in Actor-Critic Methods"

This is a PURE TD3 implementation WITHOUT any project-specific modifications.
No attention mechanisms, no CQL, no PER, no heuristic blending.

Core TD3 innovations over DDPG:
1. Twin Critic (Clipped Double Q-learning) - take min(Q1, Q2) to reduce overestimation
2. Delayed Policy Updates - update actor every d steps (default d=2)
3. Target Policy Smoothing - add clipped noise to target actions

Design choices to match the original paper:
- 2-layer MLPs (400 -> 300) with ReLU
- Actor output uses tanh scaled by action_limit
- Gaussian noise for exploration (not OU noise)
- Soft target update with tau=0.005
- Policy delay = 2
- Target noise = 0.2, noise clip = 0.5

This is standalone and expects a Gym-style continuous-action environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#


def fanin_init(layer: nn.Linear, scale: float = 1.0) -> None:
    """Fan-in initialization as per original TD3/DDPG papers."""
    fan_in = layer.weight.size(1)
    bound = scale / (fan_in ** 0.5)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class ReplayBuffer:
    """Simple uniform replay buffer."""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, d) -> None:
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.reward[self.ptr] = r
        self.next_state[self.ptr] = s2
        self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            self.state[idx],
            self.action[idx],
            self.reward[idx],
            self.next_state[idx],
            self.done[idx],
        )


# -----------------------------------------------------------------------------#
# Networks (Vanilla MLP - No Attention, No LayerNorm, No Dropout)
# -----------------------------------------------------------------------------#


class TD3Actor(nn.Module):
    """Vanilla TD3 Actor: Simple 2-layer MLP as in the original paper."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: Tuple[int, int] = (400, 300),
        act_limit: float = 1.0
    ):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )
        # Fan-in initialization
        for m in self.net:
            if isinstance(m, nn.Linear):
                fanin_init(m)
        # Final layer with smaller weights
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)
        
        self.act_limit = act_limit

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(state)) * self.act_limit


class TD3Critic(nn.Module):
    """
    Vanilla TD3 Twin Critic: Two independent Q-networks.
    This is the core innovation of TD3 - Clipped Double Q-learning.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: Tuple[int, int] = (400, 300)
    ):
        super().__init__()
        h1, h2 = hidden
        
        # Q1 network
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        
        # Q2 network (independent)
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        
        # Initialize weights
        for net in [self.q1_net, self.q2_net]:
            for m in net:
                if isinstance(m, nn.Linear):
                    fanin_init(m)
            nn.init.uniform_(net[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(net[-1].bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return both Q1 and Q2 values."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa), self.q2_net(sa)

    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return only Q1 value (for actor update)."""
        sa = torch.cat([state, action], dim=-1)
        return self.q1_net(sa)


# -----------------------------------------------------------------------------#
# TD3 Configuration
# -----------------------------------------------------------------------------#


@dataclass
class TD3Config:
    """
    Vanilla TD3 configuration - defaults from the original paper.
    
    Reference: Fujimoto et al. "Addressing Function Approximation Error 
    in Actor-Critic Methods" (2018)
    """
    # Discount and soft update
    gamma: float = 0.99
    tau: float = 0.005  # soft update rate (paper uses 0.005)
    
    # Learning rates (paper uses 1e-3 for both, but 3e-4 is common)
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    
    # Training parameters
    batch_size: int = 256  # paper uses 256
    buffer_size: int = 1_000_000
    start_steps: int = 25_000  # random exploration steps (paper: 25000)
    train_after: int = 1_000
    train_freq: int = 1  # update every step after warmup
    
    # TD3-specific parameters
    policy_delay: int = 2  # delayed policy update (core TD3 innovation)
    target_noise: float = 0.2  # noise added to target actions
    noise_clip: float = 0.5  # clip range for target noise
    
    # Exploration noise (Gaussian, not OU)
    exploration_noise: float = 0.1  # paper uses 0.1
    
    # Network architecture
    hidden_dims: Tuple[int, int] = (400, 300)  # paper uses (400, 300)
    
    # Device
    device: str = "cpu"


# -----------------------------------------------------------------------------#
# TD3 Agent
# -----------------------------------------------------------------------------#


class TD3Agent:
    """
    Vanilla TD3 Agent - Pure implementation without modifications.
    
    Core innovations:
    1. Twin Critic: min(Q1, Q2) reduces overestimation
    2. Delayed Policy Update: actor updated every 'policy_delay' steps
    3. Target Policy Smoothing: add clipped noise to target actions
    """
    
    def __init__(self, state_dim: int, action_dim: int, act_limit: float, cfg: TD3Config):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)
        self.action_dim = action_dim
        
        # Actor networks
        self.actor = TD3Actor(state_dim, action_dim, cfg.hidden_dims, act_limit).to(self.device)
        self.actor_target = TD3Actor(state_dim, action_dim, cfg.hidden_dims, act_limit).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks (Twin Q)
        self.critic = TD3Critic(state_dim, action_dim, cfg.hidden_dims).to(self.device)
        self.critic_target = TD3Critic(state_dim, action_dim, cfg.hidden_dims).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        
        # Replay buffer
        self.buffer = ReplayBuffer(cfg.buffer_size, state_dim, action_dim)
        
        # Counters
        self.total_steps = 0
        self.update_count = 0

    def store(self, transition) -> None:
        """Store transition in replay buffer."""
        s, a, r, s2, d = transition
        self.buffer.add(s, a, r, s2, d)
        self.total_steps += 1

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        """Select action with optional Gaussian exploration noise."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        
        if noise:
            # Gaussian noise (TD3 uses Gaussian, not OU noise)
            action += self.cfg.exploration_noise * np.random.randn(self.action_dim)
        
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, batch_size: int) -> Dict[str, float]:
        """
        Perform one TD3 update step.
        
        Key TD3 differences from DDPG:
        1. Use min(Q1, Q2) for target computation
        2. Add clipped noise to target actions (target policy smoothing)
        3. Only update actor every policy_delay steps (delayed policy update)
        """
        # Sample batch
        s, a, r, s2, d = self.buffer.sample(batch_size)
        device = self.device
        
        state = torch.as_tensor(s, device=device)
        action = torch.as_tensor(a, device=device)
        reward = torch.as_tensor(r, device=device)
        next_state = torch.as_tensor(s2, device=device)
        done = torch.as_tensor(d, device=device)
        
        with torch.no_grad():
            # === Target Policy Smoothing (TD3 Innovation #3) ===
            # Add clipped noise to target actions
            noise = torch.randn_like(action) * self.cfg.target_noise
            noise = noise.clamp(-self.cfg.noise_clip, self.cfg.noise_clip)
            
            next_action = self.actor_target(next_state)
            next_action = (next_action + noise).clamp(-self.act_limit, self.act_limit)
            
            # === Clipped Double Q-learning (TD3 Innovation #1) ===
            # Use min(Q1, Q2) to reduce overestimation
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1.0 - done) * self.cfg.gamma * target_q
        
        # Update Critic
        current_q1, current_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()
        
        self.update_count += 1
        actor_loss_value = 0.0
        
        # === Delayed Policy Update (TD3 Innovation #2) ===
        # Update actor only every policy_delay steps
        if self.update_count % self.cfg.policy_delay == 0:
            # Actor loss: maximize Q1
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            self.actor_opt.zero_grad()
            actor_loss.backward()
            self.actor_opt.step()
            
            actor_loss_value = actor_loss.item()
            
            # Soft update target networks
            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)
        
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss_value,
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        """Soft update: target = tau * source + (1 - tau) * target"""
        with torch.no_grad():
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * sp.data)


# -----------------------------------------------------------------------------#
# Training Loop
# -----------------------------------------------------------------------------#


def _reset_env(env):
    """Handle both old and new Gym API."""
    res = env.reset()
    if isinstance(res, tuple) and len(res) >= 1:
        return res[0]
    return res


def _step_env(env, action):
    """Handle both old and new Gym API."""
    res = env.step(action)
    if isinstance(res, tuple) and len(res) == 5:
        s2, r, terminated, truncated, info = res
        done = bool(terminated or truncated)
    else:
        s2, r, done, info = res
    return s2, r, done, info


def train_vanilla_td3(
    env,
    cfg: TD3Config,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> Dict[str, Any]:
    """
    Train vanilla TD3 on the given environment.
    
    Args:
        env: Gym-style environment
        cfg: TD3 configuration
        max_steps: Total training steps
        seed: Random seed
        progress: Optional callback for progress reporting
    
    Returns:
        Dictionary with training results
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Get dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    
    # Create agent
    agent = TD3Agent(state_dim, action_dim, act_limit, cfg)
    
    # Metrics tracking
    ep_rewards: List[float] = []
    ep_metrics = {
        "avg_task_delay": [],
        "total_energy_consumption": [],
        "task_completion_rate": [],
        "dropped_tasks": [],
        "cache_hit_rate": [],
    }
    
    # Episode accumulators
    ep_reward = 0.0
    cur_ep_delay = []
    cur_ep_energy = 0.0
    cur_ep_completed = []
    cur_ep_dropped = 0
    cur_ep_cache_hits = []
    
    state = _reset_env(env)
    episode = 0
    
    for step in range(1, max_steps + 1):
        # Random exploration for start_steps
        if agent.total_steps < cfg.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise=True)
        
        next_state, reward, done, info = _step_env(env, action)
        
        # Collect VEC metrics if available
        if "system_metrics" in info:
            m = info["system_metrics"]
            cur_ep_delay.append(m.get("avg_task_delay", 0.0))
            cur_ep_energy += m.get("total_energy_consumption", 0.0)
            cur_ep_completed.append(m.get("task_completion_rate", 0.0))
            cur_ep_dropped += m.get("dropped_tasks", 0)
            cur_ep_cache_hits.append(m.get("cache_hit_rate", 0.0))
        
        # Store transition
        agent.store((state, action, [reward], next_state, [float(done)]))
        state = next_state
        ep_reward += reward
        
        # Training updates
        if agent.buffer.size >= max(cfg.train_after, cfg.batch_size) and step % cfg.train_freq == 0:
            agent.update(cfg.batch_size)
        
        # Episode done
        if done:
            ep_rewards.append(ep_reward)
            
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
            
            # Progress callback
            if progress:
                avg_reward = np.mean(ep_rewards[-10:]) if ep_rewards else ep_reward
                progress(step, avg_reward, ep_reward)
            
            # Reset for next episode
            state = _reset_env(env)
            ep_reward = 0.0
            episode += 1
    
    return {
        "episode_rewards": ep_rewards,
        "episode_metrics": ep_metrics,
        "episodes": episode,
        "config": cfg.__dict__,
        "algorithm": "Vanilla_TD3",
    }


__all__ = ["TD3Config", "TD3Agent", "train_vanilla_td3"]
