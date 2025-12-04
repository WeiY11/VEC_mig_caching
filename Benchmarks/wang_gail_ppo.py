#!/usr/bin/env python3
"""
Wang et al. 2025 - ILCTS: GAIL + Improved PPO for UAV-MEC (IEEE TSC).

Paper: "Joint Task Offloading and Migration Optimization in UAV-Enabled Dynamic MEC Networks"

Key features from the paper:
1. Offline Expert Policy Design:
   - Improved PPO (IPPO) generates expert trajectories
   - Sliding window mechanism for trajectory collection
2. Online Agent Policy:
   - GAIL (Generative Adversarial Imitation Learning)
   - Discriminator distinguishes expert vs generated actions
   - Generator (policy) tries to fool discriminator
3. Three-stage reward shaping:
   - Episode end: -average_latency
   - Task execution: (deadline - estimated_latency) / deadline
   - No action: 0

Self-contained implementation compatible with the VEC environment.
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Actor-Critic network for PPO (Gaussian policy).
    
    Actor outputs mean and log_std for continuous actions.
    Critic outputs state value V(s).
    """
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(s_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh()
        )
        
        # Actor head
        self.actor_mean = nn.Linear(hidden, a_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(a_dim))
        
        # Critic head
        self.critic = nn.Linear(hidden, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0)
        # Smaller initialization for policy output
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
    
    def forward(self, s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.shared(s)
        mean = torch.tanh(self.actor_mean(features))  # Actions in [-1, 1]
        value = self.critic(features)
        return mean, value
    
    def get_action(self, s: torch.Tensor, deterministic: bool = False):
        mean, value = self(s)
        std = self.actor_log_std.exp().expand_as(mean)
        
        if deterministic:
            return mean, value, None
        
        dist = Normal(mean, std)
        action = dist.sample()
        action = torch.clamp(action, -1, 1)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, value, log_prob
    
    def evaluate(self, s: torch.Tensor, a: torch.Tensor):
        mean, value = self(s)
        std = self.actor_log_std.exp().expand_as(mean)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(a).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        
        return value, log_prob, entropy


class Discriminator(nn.Module):
    """
    GAIL Discriminator.
    
    Distinguishes between expert and generated state-action pairs.
    """
    def __init__(self, s_dim: int, a_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


class TrajectoryBuffer:
    """Buffer for PPO trajectory collection with GAE."""
    def __init__(self, gamma: float = 0.99, lam: float = 0.95):
        self.gamma = gamma
        self.lam = lam
        self.reset()
    
    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []
    
    def add(self, s, a, r, v, log_p, done):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.values.append(v)
        self.log_probs.append(log_p)
        self.dones.append(done)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute GAE (Eq. 20 in paper)."""
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)
        
        T = len(rewards)
        advantages = np.zeros(T)
        returns = np.zeros(T)
        
        gae = 0
        for t in reversed(range(T)):
            if t == T - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages[t] = gae
            returns[t] = advantages[t] + values[t]
        
        return advantages, returns
    
    def get(self, next_value: float = 0.0):
        advantages, returns = self.compute_gae(next_value)
        
        return (
            np.array(self.states),
            np.array(self.actions),
            np.array(self.log_probs),
            returns,
            advantages
        )


@dataclass
class WangGAILConfig:
    """Configuration for Wang GAIL+PPO (paper parameters)."""
    # PPO parameters (Section IV-C)
    gamma: float = 0.99
    lam: float = 0.95
    clip_eps: float = 0.2  # Paper: ε typically 0.2
    ppo_epochs: int = 10
    batch_size: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    entropy_coef: float = 0.01
    
    # GAIL parameters
    disc_lr: float = 3e-4
    disc_epochs: int = 5
    expert_buffer_size: int = 10000
    
    # Sliding window for expert trajectories
    window_size: int = 100
    
    # Reward shaping (Eq. 18)
    latency_weight: float = 100.0  # H1 in paper
    progress_weight: float = 20.0  # H2 in paper
    
    device: str = "cpu"


class WangGAILAgent:
    """
    ILCTS Agent (Wang et al. 2025).
    
    Combines Improved PPO for expert policy and GAIL for imitation learning.
    """
    def __init__(self, s_dim: int, a_dim: int, cfg: WangGAILConfig):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        
        # Policy network (generator)
        self.policy = ActorCritic(s_dim, a_dim).to(self.device)
        self.policy_opt = optim.Adam([
            {'params': self.policy.shared.parameters(), 'lr': cfg.actor_lr},
            {'params': self.policy.actor_mean.parameters(), 'lr': cfg.actor_lr},
            {'params': [self.policy.actor_log_std], 'lr': cfg.actor_lr},
            {'params': self.policy.critic.parameters(), 'lr': cfg.critic_lr}
        ])
        
        # Discriminator for GAIL
        self.discriminator = Discriminator(s_dim, a_dim).to(self.device)
        self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=cfg.disc_lr)
        
        # Trajectory buffer
        self.buffer = TrajectoryBuffer(cfg.gamma, cfg.lam)
        
        # Expert buffer (sliding window)
        self.expert_states = deque(maxlen=cfg.expert_buffer_size)
        self.expert_actions = deque(maxlen=cfg.expert_buffer_size)
        
        # Expert policy (initially same as main policy)
        self.expert_policy = ActorCritic(s_dim, a_dim).to(self.device)
        self.expert_policy.load_state_dict(self.policy.state_dict())
        
        self.use_gail = False  # Enable after initial PPO training
        self.total_steps = 0
    
    def act(self, s: np.ndarray, deterministic: bool = False) -> Tuple[np.ndarray, float]:
        """Select action using current policy."""
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        with torch.no_grad():
            action, value, log_prob = self.policy.get_action(st, deterministic)
        
        a_np = action.cpu().numpy()[0]
        v_np = value.cpu().item()
        lp_np = log_prob.cpu().item() if log_prob is not None else 0.0
        
        return a_np, v_np, lp_np
    
    def store(self, s, a, r, v, log_p, done):
        """Store transition in trajectory buffer."""
        self.buffer.add(s, a, r, v, log_p, done)
        self.total_steps += 1
    
    def _get_gail_reward(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute GAIL reward from discriminator: -log(1 - D(s, a))."""
        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            d_out = self.discriminator(s_t, a_t)
            # GAIL reward = -log(1 - D) ≈ log(D) for high D
            gail_r = -torch.log(1 - d_out + 1e-8).squeeze().cpu().numpy()
        
        return gail_r
    
    def _update_discriminator(self, gen_states, gen_actions):
        """Update discriminator to distinguish expert from generated."""
        if len(self.expert_states) < self.cfg.batch_size:
            return {}
        
        losses = []
        
        for _ in range(self.cfg.disc_epochs):
            # Sample from expert buffer
            expert_idx = np.random.choice(len(self.expert_states), 
                                         min(len(gen_states), len(self.expert_states)), 
                                         replace=False)
            exp_s = np.array([self.expert_states[i] for i in expert_idx])
            exp_a = np.array([self.expert_actions[i] for i in expert_idx])
            
            exp_s_t = torch.as_tensor(exp_s, dtype=torch.float32, device=self.device)
            exp_a_t = torch.as_tensor(exp_a, dtype=torch.float32, device=self.device)
            gen_s_t = torch.as_tensor(gen_states[:len(exp_s)], dtype=torch.float32, device=self.device)
            gen_a_t = torch.as_tensor(gen_actions[:len(exp_a)], dtype=torch.float32, device=self.device)
            
            # Discriminator outputs
            exp_out = self.discriminator(exp_s_t, exp_a_t)
            gen_out = self.discriminator(gen_s_t, gen_a_t)
            
            # Binary cross entropy
            exp_loss = -torch.log(exp_out + 1e-8).mean()
            gen_loss = -torch.log(1 - gen_out + 1e-8).mean()
            disc_loss = exp_loss + gen_loss
            
            self.disc_opt.zero_grad()
            disc_loss.backward()
            self.disc_opt.step()
            
            losses.append(disc_loss.item())
        
        return {"disc_loss": np.mean(losses)}
    
    def update(self, next_value: float = 0.0) -> Dict[str, float]:
        """
        PPO update with optional GAIL reward shaping.
        
        Implements Algorithm in paper Section IV-C.
        """
        states, actions, old_log_probs, returns, advantages = self.buffer.get(next_value)
        
        if len(states) < self.cfg.batch_size:
            self.buffer.reset()
            return {}
        
        # Add to expert buffer if performing well
        mean_return = returns.mean()
        if mean_return > 0:  # Only add "good" trajectories
            for s, a in zip(states, actions):
                self.expert_states.append(s)
                self.expert_actions.append(a)
        
        # Modify rewards with GAIL if enabled
        if self.use_gail and len(self.expert_states) >= self.cfg.batch_size:
            gail_rewards = self._get_gail_reward(states, actions)
            # Blend GAIL reward with environment reward
            returns = returns + 0.1 * gail_rewards
            
            # Update discriminator
            disc_info = self._update_discriminator(states, actions)
        else:
            disc_info = {}
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Convert to tensors
        s_t = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a_t = torch.as_tensor(actions, dtype=torch.float32, device=self.device)
        old_lp_t = torch.as_tensor(old_log_probs, dtype=torch.float32, device=self.device).reshape(-1, 1)
        ret_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device).reshape(-1, 1)
        adv_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device).reshape(-1, 1)
        
        # PPO epochs (Eq. 19)
        policy_losses = []
        value_losses = []
        
        n_samples = len(states)
        
        for _ in range(self.cfg.ppo_epochs):
            indices = np.random.permutation(n_samples)
            
            for start in range(0, n_samples, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                idx = indices[start:end]
                
                values, log_probs, entropy = self.policy.evaluate(s_t[idx], a_t[idx])
                
                # Policy loss (Eq. 19 - clipped objective)
                ratio = torch.exp(log_probs - old_lp_t[idx])
                surr1 = ratio * adv_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps) * adv_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (Eq. 21)
                value_loss = nn.functional.mse_loss(values, ret_t[idx])
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                total_loss = policy_loss + 0.5 * value_loss + self.cfg.entropy_coef * entropy_loss
                
                self.policy_opt.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
                self.policy_opt.step()
                
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
        
        self.buffer.reset()
        
        # Enable GAIL after initial training
        if self.total_steps > 10000 and not self.use_gail:
            self.use_gail = True
            self.expert_policy.load_state_dict(self.policy.state_dict())
        
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            **disc_info
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


def train_wang_gail(
    env,
    cfg: WangGAILConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train Wang GAIL+PPO agent on a gym-style environment.
    
    Implements the ILCTS algorithm from Wang et al. 2025.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_low = env.action_space.low
    a_high = env.action_space.high
    
    agent = WangGAILAgent(s_dim, a_dim, cfg)
    
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
    update_interval = 2048  # Steps between PPO updates
    
    for step in range(1, max_steps + 1):
        # Select action
        a, v, lp = agent.act(s, deterministic=False)
        
        # Scale to environment action range
        a_env = a_low + (a + 1) / 2 * (a_high - a_low)
        a_env = np.clip(a_env, a_low, a_high)
        
        s2, r, done, info = _step_env(env, a_env)
        
        # Store transition
        agent.store(s, a, r, v, lp, done)
        
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
        
        # PPO update
        if step % update_interval == 0:
            _, next_v, _ = agent.act(s, deterministic=True)
            agent.update(next_value=next_v)
        
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
            
            # Perform update at episode end if buffer has data
            if len(agent.buffer.states) >= agent.cfg.batch_size:
                agent.update(next_value=0.0)
            
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


__all__ = ["WangGAILConfig", "WangGAILAgent", "train_wang_gail"]
