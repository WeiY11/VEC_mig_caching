#!/usr/bin/env python3
"""
Minimal, self-contained DDPG implementation aligned with Lillicrap et al. (2019/2015).

Design goals:
- No dependency on the project's existing RL stacks.
- Runs on any OpenAI Gym–style environment (reset(), step()) with continuous actions.
- Clear defaults; all hyperparameters exposed via DDPGConfig.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

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
    """Ornstein-Uhlenbeck process (closer to Lillicrap et al.)."""

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


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, output_dim),
        )
        # Fannin init for stability
        for m in self.net:
            if isinstance(m, nn.Linear):
                fanin_init(m, scale=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# -----------------------------------------------------------------------------#
# DDPG core
# -----------------------------------------------------------------------------#


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 128
    buffer_size: int = 200_000
    start_steps: int = 5_000
    exploration_noise: float = 0.1
    use_ou_noise: bool = True
    train_after: int = 1_000
    train_freq: int = 1
    device: str = "cpu"


class DDPGAgent:
    def __init__(self, state_dim: int, action_dim: int, act_limit: float, cfg: DDPGConfig):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)

        self.actor = MLP(state_dim, action_dim).to(self.device)
        self.critic = MLP(state_dim + action_dim, 1).to(self.device)
        self.actor_target = MLP(state_dim, action_dim).to(self.device)
        self.critic_target = MLP(state_dim + action_dim, 1).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

        self.buffer = ReplayBuffer(cfg.buffer_size, state_dim, action_dim)
        self.total_steps = 0
        self.ou_noise = OUNoise(action_dim) if cfg.use_ou_noise else None

    def select_action(self, state: np.ndarray, noise: bool = True) -> np.ndarray:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = torch.tanh(self.actor(state_t)).cpu().numpy()[0]
        if noise:
            if self.ou_noise is not None:
                action += self.ou_noise.sample()
            else:
                action += self.cfg.exploration_noise * np.random.randn(*action.shape)
        return np.clip(action * self.act_limit, -self.act_limit, self.act_limit)

    def update(self, batch_size: int) -> None:
        s, a, r, s2, d = self.buffer.sample(batch_size)
        device = self.device
        s = torch.as_tensor(s, device=device)
        a = torch.as_tensor(a, device=device)
        r = torch.as_tensor(r, device=device)
        s2 = torch.as_tensor(s2, device=device)
        d = torch.as_tensor(d, device=device)

        with torch.no_grad():
            a2 = self.actor_target(s2)
            q_target = self.critic_target(torch.cat([s2, a2], dim=-1))
            y = r + self.cfg.gamma * (1.0 - d) * q_target

        q = self.critic(torch.cat([s, a], dim=-1))
        critic_loss = nn.functional.mse_loss(q, y)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(torch.cat([s, self.actor(s)], dim=-1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        with torch.no_grad():
            for target, online in [
                (self.actor_target, self.actor),
                (self.critic_target, self.critic),
            ]:
                for tp, p in zip(target.parameters(), online.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

    def store(self, transition) -> None:
        self.buffer.add(*transition)
        self.total_steps += 1


# -----------------------------------------------------------------------------#
# Training entrypoint
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
    """
    Train DDPG on a gym-style environment.

    Args:
        env: object exposing reset() -> state, step(action) -> (next_state, reward, done, info)
        cfg: hyperparameters
        max_steps: total environment steps
        seed: RNG seed
        progress: optional callback(step, avg_reward, last_reward)
    Returns:
        dict with episode_rewards list and config snapshot.
    """
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
        noise_phase = agent.total_steps < cfg.start_steps
        if noise_phase:
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
