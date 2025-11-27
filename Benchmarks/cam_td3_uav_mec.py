#!/usr/bin/env python3
"""
Cache-aware TD3 variant for UAV-enabled MEC (Wang et al. 2025 style).

Self-contained implementation:
- Twin critics + delayed actor update.
- Action head can output joint vector: [offload_pref(3), rsu_slots, uav_slots, cache/migration params].
- Does not depend on the repository's previous TD3 code.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def mlp(in_dim: int, out_dim: int, hidden: Tuple[int, int] = (400, 300)) -> nn.Sequential:
    h1, h2 = hidden
    return nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, out_dim),
    )


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

    def sample(self, batch: int) -> Tuple[np.ndarray, ...]:
        idx = np.random.randint(0, self.size, size=batch)
        return self.s[idx], self.a[idx], self.r[idx], self.s2[idx], self.d[idx]


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


@dataclass
class CAMTD3Config:
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 100
    buffer_size: int = 1_000_000
    start_steps: int = 25_000
    noise_std: float = 0.2
    noise_clip: float = 0.5
    use_ou_noise: bool = False
    train_after: int = 1_000
    train_freq: int = 1
    device: str = "cpu"
    num_rsus: int = 4
    num_uavs: int = 2
    cache_ctrl_dim: int = 10  # cache/migration control tail length
    hidden_dims: Tuple[int, int] = (400, 300)


class CAMTD3Agent:
    def __init__(self, s_dim: int, a_dim: int, act_limit: float, cfg: CAMTD3Config):
        self.cfg = cfg
        self.act_limit = act_limit
        self.device = torch.device(cfg.device)
        self.num_rsus = cfg.num_rsus
        self.num_uavs = cfg.num_uavs
        self.cache_ctrl_dim = cfg.cache_ctrl_dim

        self.actor = mlp(s_dim, a_dim, hidden=cfg.hidden_dims).to(self.device)
        self.actor_t = mlp(s_dim, a_dim, hidden=cfg.hidden_dims).to(self.device)
        self.critic1 = mlp(s_dim + a_dim, 1, hidden=cfg.hidden_dims).to(self.device)
        self.critic2 = mlp(s_dim + a_dim, 1, hidden=cfg.hidden_dims).to(self.device)
        self.critic1_t = mlp(s_dim + a_dim, 1, hidden=cfg.hidden_dims).to(self.device)
        self.critic2_t = mlp(s_dim + a_dim, 1, hidden=cfg.hidden_dims).to(self.device)
        self.actor_t.load_state_dict(self.actor.state_dict())
        self.critic1_t.load_state_dict(self.critic1.state_dict())
        self.critic2_t.load_state_dict(self.critic2.state_dict())

        self.opt_a = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
        self.opt_c1 = optim.Adam(self.critic1.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)
        self.opt_c2 = optim.Adam(self.critic2.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

        self.buf = ReplayBuffer(cfg.buffer_size, s_dim, a_dim)
        self.total_updates = 0
        self.total_steps = 0
        self.ou_noise = OUNoise(a_dim) if cfg.use_ou_noise else None

    def _structure_action_np(self, raw: np.ndarray) -> np.ndarray:
        """
        Map raw actor output to structured action vector:
        [offload(3 softmax), rsu softmax, uav softmax, cache/migration tanh...]
        """
        raw = raw.reshape(-1)
        a_dim = self.act_limit * np.ones_like(raw)
        off_len = 3
        rsu_len = min(self.num_rsus, max(0, raw.size - off_len))
        uav_len = min(self.num_uavs, max(0, raw.size - off_len - rsu_len))
        ctrl_len = min(self.cache_ctrl_dim, max(0, raw.size - off_len - rsu_len - uav_len))
        out = np.zeros_like(raw)

        # offload
        off_logits = raw[:off_len]
        off = np.exp(off_logits - off_logits.max())
        off = off / (off.sum() + 1e-8)
        out[:off_len] = off

        # rsu
        if rsu_len > 0:
            rsu_logits = raw[off_len : off_len + rsu_len]
            rsu = np.exp(rsu_logits - rsu_logits.max())
            rsu = rsu / (rsu.sum() + 1e-8)
            out[off_len : off_len + rsu_len] = rsu

        # uav
        if uav_len > 0:
            uav_start = off_len + rsu_len
            uav_logits = raw[uav_start : uav_start + uav_len]
            uav = np.exp(uav_logits - uav_logits.max())
            uav = uav / (uav.sum() + 1e-8)
            out[uav_start : uav_start + uav_len] = uav

        # cache/migration ctrl (bounded)
        if ctrl_len > 0:
            ctrl_start = off_len + rsu_len + uav_len
            ctrl = np.tanh(raw[ctrl_start : ctrl_start + ctrl_len])
            out[ctrl_start : ctrl_start + ctrl_len] = ctrl

        return np.clip(out, -self.act_limit, self.act_limit)

    def _structure_action_torch(self, raw: torch.Tensor) -> torch.Tensor:
        raw = raw.reshape(raw.shape[0], -1)
        batch, dim = raw.shape
        off_len = 3
        rsu_len = min(self.num_rsus, max(0, dim - off_len))
        uav_len = min(self.num_uavs, max(0, dim - off_len - rsu_len))
        ctrl_len = min(self.cache_ctrl_dim, max(0, dim - off_len - rsu_len - uav_len))

        out = torch.zeros_like(raw)
        off_logits = raw[:, :off_len]
        off = torch.softmax(off_logits, dim=-1)
        out[:, :off_len] = off

        if rsu_len > 0:
            rsu_logits = raw[:, off_len : off_len + rsu_len]
            rsu = torch.softmax(rsu_logits, dim=-1)
            out[:, off_len : off_len + rsu_len] = rsu

        if uav_len > 0:
            start = off_len + rsu_len
            uav_logits = raw[:, start : start + uav_len]
            uav = torch.softmax(uav_logits, dim=-1)
            out[:, start : start + uav_len] = uav

        if ctrl_len > 0:
            start = off_len + rsu_len + uav_len
            ctrl = torch.tanh(raw[:, start : start + ctrl_len])
            out[:, start : start + ctrl_len] = ctrl

        return torch.clamp(out, -self.act_limit, self.act_limit)

    def act(self, s: np.ndarray, noise: bool = True) -> np.ndarray:
        st = torch.as_tensor(s, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            raw = self.actor(st).cpu().numpy()[0]
        if noise:
            if self.ou_noise is not None:
                raw += self.ou_noise.sample()
            else:
                raw += np.random.normal(0, self.cfg.noise_std, size=raw.shape)
        return self._structure_action_np(raw)

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

        with torch.no_grad():
            noise = torch.clamp(
                torch.randn_like(a) * self.cfg.noise_std * self.act_limit,
                -self.cfg.noise_clip * self.act_limit,
                self.cfg.noise_clip * self.act_limit,
            )
            a2_raw = self.actor_t(s2) + noise
            a2 = self._structure_action_torch(a2_raw)
            q1_t = self.critic1_t(torch.cat([s2, a2], dim=-1))
            q2_t = self.critic2_t(torch.cat([s2, a2], dim=-1))
            q_t = torch.min(q1_t, q2_t)
            y = r + self.cfg.gamma * (1.0 - d) * q_t

        q1 = self.critic1(torch.cat([s, a], dim=-1))
        q2 = self.critic2(torch.cat([s, a], dim=-1))
        loss_c1 = nn.functional.mse_loss(q1, y)
        loss_c2 = nn.functional.mse_loss(q2, y)
        self.opt_c1.zero_grad()
        loss_c1.backward()
        self.opt_c1.step()
        self.opt_c2.zero_grad()
        loss_c2.backward()
        self.opt_c2.step()

        if self.total_updates % self.cfg.policy_delay == 0:
            a_pi_raw = self.actor(s)
            a_pi = self._structure_action_torch(a_pi_raw)
            act_loss = -self.critic1(torch.cat([s, a_pi], dim=-1)).mean()
            self.opt_a.zero_grad()
            act_loss.backward()
            self.opt_a.step()

            with torch.no_grad():
                for tgt, src in [
                    (self.actor_t, self.actor),
                    (self.critic1_t, self.critic1),
                    (self.critic2_t, self.critic2),
                ]:
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


def train_cam_td3(
    env,
    cfg: CAMTD3Config,
    max_steps: int,
    seed: int = 42,
    progress: Callable[[int, float, float], None] | None = None,
) -> dict:
    """
    Train cache-aware TD3 on a gym-style environment.

    The environment should expose continuous actions; if your action vector encodes
    offload + resource allocations, simply size the action space accordingly.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    s_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    a_lim = float(env.action_space.high[0])
    agent = CAMTD3Agent(s_dim, a_dim, a_lim, cfg)

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
        s2, r, done, _ = _step_env(env, a)
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
            if agent.ou_noise is not None:
                agent.ou_noise.reset()
            episode += 1

    return {
        "episode_rewards": ep_rewards,
        "episodes": episode,
        "config": cfg.__dict__,
    }


__all__ = ["CAMTD3Config", "CAMTD3Agent", "train_cam_td3"]
