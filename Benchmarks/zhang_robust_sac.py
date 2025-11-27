#!/usr/bin/env python3
"""
Robust SAC variant inspired by Zhang et al. (RoNet 2023):
- Adds action noise during training and adversarial observation noise.
- L2 regularisation on policy for stability.
- Self-contained; no reliance on existing SAC code in the repo.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mlp(in_dim: int, out_dim: int, hidden: tuple[int, int] = (256, 256)) -> nn.Sequential:
    h1, h2 = hidden
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, out_dim),
    )


class TanhGaussianPolicy(nn.Module):
    def __init__(self, s_dim: int, a_dim: int, log_std_min=-20, log_std_max=2, hidden: tuple[int, int] = (256, 256)):
        super().__init__()
        self.net = mlp(s_dim, 2 * a_dim, hidden=hidden)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_logstd = self.net(s)
        mean, log_std = torch.chunk(mean_logstd, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        return mean, std

    def sample(self, s: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean, std = self(s)
        noise = torch.randn_like(mean)
        z = mean + std * noise
        base_action = torch.tanh(z)
        log_prob = (-(noise ** 2) / 2 - std.log() - 0.5 * np.log(2 * np.pi)).sum(dim=-1, keepdim=True)
        log_prob -= torch.log(1 - base_action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        action = base_action * scale
        log_prob -= torch.log(scale).sum(dim=-1, keepdim=True)
        return action, log_prob


class DoubleQ(nn.Module):
    def __init__(self, s_dim: int, a_dim: int):
        super().__init__()
        self.q1 = mlp(s_dim + a_dim, 1)
        self.q2 = mlp(s_dim + a_dim, 1)

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, cap: int, s_dim: int, a_dim: int):
        self.cap = cap
        self.s = np.zeros((cap, s_dim), dtype=np.float32)
        self.a = np.zeros((cap, a_dim), dtype=np.float32)
        self.r = np.zeros((cap, 1), dtype=np.float32)
        self.s2 = np.zeros((cap, s_dim), dtype=np.float32)
        self.d = np.zeros((cap, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, s, a, r, s2, d):
        self.s[self.ptr] = s
        self.a[self.ptr] = a
        self.r[self.ptr] = r
        self.s2[self.ptr] = s2
        self.d[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch: int) -> tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


@dataclass
class RobustSACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_alpha: bool = True
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    buffer_size: int = 300_000
    start_steps: int = 5_000
    train_after: int = 1_000
    train_freq: int = 1
    obs_noise: float = 0.05  # adversarial-style noise on obs during critic update
    act_noise: float = 0.05  # extra noise in sampled actions
    l2_reg: float = 1e-4
    device: str = "cpu"
    adv_epsilon: float = 0.02  # FGSM-like perturbation magnitude
    latency_target: float = 1.0
    energy_target: float = 4000.0
    qos_penalty_weight: float = 0.05  # penalty per unit violation
    hidden_dims: tuple[int, int] = (256, 256)


class RobustSACAgent:
    def __init__(self, s_dim: int, a_dim: int, act_limit: float, cfg: RobustSACConfig):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)
        self.scale_tensor = torch.as_tensor(np.full((1, a_dim), act_limit, dtype=np.float32), device=self.device)

        self.policy = TanhGaussianPolicy(s_dim, a_dim, hidden=cfg.hidden_dims).to(self.device)
        self.policy_t = TanhGaussianPolicy(s_dim, a_dim, hidden=cfg.hidden_dims).to(self.device)
        self.q = DoubleQ(s_dim, a_dim).to(self.device)
        self.q_t = DoubleQ(s_dim, a_dim).to(self.device)
        self.policy_t.load_state_dict(self.policy.state_dict())
        self.q_t.load_state_dict(self.q.state_dict())

        self.opt_p = optim.Adam(self.policy.parameters(), lr=cfg.actor_lr, weight_decay=cfg.l2_reg)
        self.opt_q = optim.Adam(self.q.parameters(), lr=cfg.critic_lr, weight_decay=cfg.l2_reg)
        if cfg.auto_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.opt_alpha = optim.Adam([self.log_alpha], lr=cfg.actor_lr)
        else:
            self.log_alpha = None
            self.opt_alpha = None

        self.buf = ReplayBuffer(cfg.buffer_size, s_dim, a_dim)
        self.total_steps = 0
        self.total_updates = 0
        self.scale = act_limit

    def act(self, s: np.ndarray, noise: bool = True) -> np.ndarray:
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            a, _ = self.policy.sample(st, self.scale_tensor)
        a = a.cpu().numpy()[0]
        if noise:
            a += np.random.normal(0, self.cfg.act_noise, size=a.shape)
        return np.clip(a, -self.scale, self.scale)

    def _current_alpha(self) -> float:
        if self.cfg.auto_alpha and self.log_alpha is not None:
            return float(self.log_alpha.exp().item())
        return self.cfg.alpha

    def store(self, transition) -> None:
        self.buf.add(*transition)
        self.total_steps += 1

    def update(self) -> None:
        s, a, r, s2, d = self.buf.sample(self.cfg.batch_size)
        dev = self.device
        s = torch.as_tensor(s, device=dev)
        a = torch.as_tensor(a, device=dev)
        r = torch.as_tensor(r, device=dev)
        s2 = torch.as_tensor(s2, device=dev)
        d = torch.as_tensor(d, device=dev)

        # Adversarial observation noise (random + FGSM-like sign)
        noise = torch.randn_like(s) * self.cfg.obs_noise
        sign = torch.sign(noise)
        s_noisy = s + noise + self.cfg.adv_epsilon * sign

        with torch.no_grad():
            a2, logp2 = self.policy_t.sample(s2, self.scale_tensor)
            q1_t, q2_t = self.q_t(s2, a2)
            alpha = self._current_alpha()
            q_t = torch.min(q1_t, q2_t) - alpha * logp2
            y = r + self.cfg.gamma * (1.0 - d) * q_t

        q1, q2 = self.q(s_noisy, a)
        loss_q = nn.functional.mse_loss(q1, y) + nn.functional.mse_loss(q2, y)
        self.opt_q.zero_grad()
        loss_q.backward()
        self.opt_q.step()

        a_pi, logp_pi = self.policy.sample(s, self.scale_tensor)
        q1_pi, q2_pi = self.q(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        alpha = self._current_alpha()
        loss_pi = (alpha * logp_pi - q_pi).mean()
        self.opt_p.zero_grad()
        loss_pi.backward()
        self.opt_p.step()

        if self.cfg.auto_alpha and self.log_alpha is not None and self.opt_alpha is not None:
            target_ent = -float(a.shape[-1])
            alpha_loss = -(self.log_alpha * (logp_pi + target_ent).detach()).mean()
            self.opt_alpha.zero_grad()
            alpha_loss.backward()
            self.opt_alpha.step()

        with torch.no_grad():
            for tgt, src in [(self.policy_t, self.policy), (self.q_t, self.q)]:
                for tp, p in zip(tgt.parameters(), src.parameters()):
                    tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
        self.total_updates += 1


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


def train_robust_sac(
    env,
    cfg: RobustSACConfig,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_lim = float(env.action_space.high[0])
    agent = RobustSACAgent(s_dim, a_dim, a_lim, cfg)

    ep_rewards = []
    ep_r = 0.0
    s = _reset_env(env)
    episode = 0

    for step in range(1, max_steps + 1):
        explore = agent.total_steps < cfg.start_steps
        if explore:
            a = env.action_space.sample()
        else:
            a = agent.act(s, noise=True)
        s2, r, done, info = _step_env(env, a)
        # QoS-style penalty to align with robust configuration emphasis
        metrics = info.get("system_metrics", {}) if isinstance(info, dict) else {}
        delay = float(metrics.get("avg_task_delay", 0.0))
        energy = float(metrics.get("total_energy_consumption", 0.0))
        penalty = cfg.qos_penalty_weight * max(0.0, delay - cfg.latency_target)
        penalty += cfg.qos_penalty_weight * 1e-4 * max(0.0, energy - cfg.energy_target)
        r = r - penalty
        agent.store((s, a, [r], s2, [float(done)]))
        s = s2
        ep_r += r

        if agent.buf.size >= max(cfg.train_after, cfg.batch_size) and step % cfg.train_freq == 0:
            agent.update()

        if done:
            ep_rewards.append(ep_r)
            if progress:
                progress(step, np.mean(ep_rewards[-10:]), ep_r)
            s, ep_r = _reset_env(env), 0.0
            episode += 1

    return {
        "episode_rewards": ep_rewards,
        "episodes": episode,
        "config": cfg.__dict__,
    }


__all__ = ["RobustSACConfig", "RobustSACAgent", "train_robust_sac"]
