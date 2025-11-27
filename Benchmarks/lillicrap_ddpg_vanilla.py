#!/usr/bin/env python3
"""
DDPG implementation aligned with Lillicrap et al. (2015/2019).

Design choices to match the reference:
- 2-layer MLPs (400 -> 300) with ReLU.
- Actor output uses tanh scaled by action_limit.
- Critic L2 regularisation (weight decay) on parameters.
- Soft target update with tau=0.001.
- OU noise for exploration; early steps use random actions.

This is standalone and expects a Gym-style continuous-action environment.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# -----------------------------------------------------------------------------#
# Utilities
# -----------------------------------------------------------------------------#


def fanin_init(layer: nn.Linear, scale: float = 1.0) -> None:
    bound = scale / (layer.weight.size(0) ** 0.5)
    nn.init.uniform_(layer.weight, -bound, bound)
    nn.init.uniform_(layer.bias, -bound, bound)


class ReplayBuffer:
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


class OUNoise:
    def __init__(self, action_dim: int, theta: float = 0.15, sigma: float = 0.2):
        self.theta = theta
        self.sigma = sigma
        self.action_dim = action_dim
        self.state = np.zeros(action_dim, dtype=np.float32)

    def reset(self) -> None:
        self.state.fill(0.0)

    def sample(self) -> np.ndarray:
        dx = self.theta * (-self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (400, 300), act_limit: float = 1.0):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, action_dim),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                fanin_init(m)
        self.act_limit = act_limit

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(s)) * self.act_limit


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: Tuple[int, int] = (400, 300)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                fanin_init(m)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s, a], dim=-1))


# -----------------------------------------------------------------------------#
# DDPG core
# -----------------------------------------------------------------------------#


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.001  # soft update rate
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    critic_l2: float = 1e-2  # weight decay on critic (as in original DDPG)
    batch_size: int = 128
    buffer_size: int = 1_000_000
    start_steps: int = 10_000
    exploration_noise: float = 0.2
    use_ou_noise: bool = True
    train_after: int = 1_000
    train_freq: int = 1
    device: str = "cpu"
    hidden_dims: Tuple[int, int] = (400, 300)


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, act_limit: float, cfg: DDPGConfig):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)

        self.actor = Actor(state_dim, action_dim, hidden=cfg.hidden_dims, act_limit=act_limit).to(self.device)
        self.critic = Critic(state_dim, action_dim, hidden=cfg.hidden_dims).to(self.device)
        self.actor_t = Actor(state_dim, action_dim, hidden=cfg.hidden_dims, act_limit=act_limit).to(self.device)
        self.critic_t = Critic(state_dim, action_dim, hidden=cfg.hidden_dims).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic_t.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.critic_l2)

        self.buffer = ReplayBuffer(cfg.buffer_size, state_dim, action_dim)
        self.total_steps = 0
        self.ou_noise = OUNoise(action_dim) if cfg.use_ou_noise else None

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy()[0]
        if noise:
            if self.ou_noise is not None:
                action += self.ou_noise.sample()
            else:
                action += self.cfg.exploration_noise * np.random.randn(*action.shape)
        return np.clip(action, -self.act_limit, self.act_limit)

    def update(self, batch_size: int) -> None:
        s, a, r, s2, d = self.buffer.sample(batch_size)
        device = self.device
        s = torch.as_tensor(s, device=device)
        a = torch.as_tensor(a, device=device)
        r = torch.as_tensor(r, device=device)
        s2 = torch.as_tensor(s2, device=device)
        d = torch.as_tensor(d, device=device)

        with torch.no_grad():
            a2 = self.actor_t(s2)
            q_target = self.critic_t(s2, a2)
            y = r + self.cfg.gamma * (1.0 - d) * q_target

        q = self.critic(s, a)
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(s, self.actor(s)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for tp, p in zip(self.actor_t.parameters(), self.actor.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for tp, p in zip(self.critic_t.parameters(), self.critic.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)


# -----------------------------------------------------------------------------#
# Training loop
# -----------------------------------------------------------------------------#


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


def train_ddpg(
    env,
    cfg: DDPGConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, act_limit, cfg)

    ep_rewards = []
    ep_reward = 0.0
    state = _reset_env(env)
    episode = 0

    for step in range(1, max_steps + 1):
        explore = agent.total_steps < cfg.start_steps
        if explore:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, noise=True)

        next_state, reward, done, _ = _step_env(env, action)
        agent.store((state, action, [reward], next_state, [float(done)]))
        state = next_state
        ep_reward += reward

        if agent.buffer.size >= max(cfg.train_after, cfg.batch_size) and step % cfg.train_freq == 0:
            agent.update(cfg.batch_size)

        if done:
            ep_rewards.append(ep_reward)
            if progress:
                progress(step, np.mean(ep_rewards[-10:]), ep_reward)
            state, ep_reward = _reset_env(env), 0.0
            if agent.ou_noise is not None:
                agent.ou_noise.reset()
            episode += 1

    return {
        "episode_rewards": ep_rewards,
        "episodes": episode,
        "config": cfg.__dict__,
    }


__all__ = ["DDPGConfig", "DDPGAgent", "train_ddpg"]
