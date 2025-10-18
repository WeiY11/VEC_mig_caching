"""
TD3 分层策略扩展（TD3-Hier）

特性：
1. 高层决策：针对任务分配、本地/RSU/UAV 卸载偏好，输出概率权重（软约束）。
2. 低层细化：结合高层输出与环境状态，生成频率/缓存/迁移等连续控制参数。
3. 保留 TD3-LE 的延时能耗奖励塑形，同时记录层级诊断信息便于分析。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import torch
import torch.nn as nn

from .td3 import TD3Config, TD3Agent
from .td3_latency_energy import TD3LatencyEnergyEnvironment


@dataclass
class HierarchicalTD3Config(TD3Config):
    """
    分层策略 TD3 配置：
    - 隐藏层调小以降低低层过拟合风险，同时保留Critic结构。
    - high_level_hidden: 高层策略隐藏层规模
    - low_level_hidden: 低层策略隐藏层规模
    - gating_temperature: 软max温度，越低越尖锐
    - residual_low_level: 是否启用低层残差连接（有助于稳定）
    """

    hidden_dim: int = 384
    actor_lr: float = 8e-5
    critic_lr: float = 7e-5
    batch_size: int = 192
    warmup_steps: int = 1500

    high_level_hidden: int = 256
    low_level_hidden: int = 512
    gating_temperature: float = 0.7
    residual_low_level: bool = True

    def __post_init__(self):
        super().__post_init__()


class HierarchicalTD3Actor(nn.Module):
    """双层结构 Actor：高层输出概率，低层输出连续控制。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        *,
        num_rsus: int,
        num_uavs: int,
        config_obj: Optional[HierarchicalTD3Config] = None,
    ):
        super().__init__()
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.cfg = config_obj or HierarchicalTD3Config()

        self.high_dim = 3 + num_rsus + num_uavs
        self.low_dim = action_dim - self.high_dim
        if self.low_dim <= 0:
            raise ValueError("动作维度不足以构建分层策略，请确认拓扑设置")

        # 高层：全局感知 -> 软max权重
        self.high_trunk = nn.Sequential(
            nn.Linear(state_dim, self.cfg.high_level_hidden),
            nn.LayerNorm(self.cfg.high_level_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.high_level_hidden, self.cfg.high_level_hidden),
            nn.ReLU(),
        )
        self.high_head = nn.Linear(self.cfg.high_level_hidden, self.high_dim)

        # 低层：融合高层输出 + 状态
        low_input_dim = state_dim + self.high_dim
        self.low_trunk = nn.Sequential(
            nn.Linear(low_input_dim, self.cfg.low_level_hidden),
            nn.LayerNorm(self.cfg.low_level_hidden),
            nn.ReLU(),
            nn.Linear(self.cfg.low_level_hidden, hidden_dim),
            nn.ReLU(),
        )
        self.state_adapter = nn.Linear(state_dim, hidden_dim)
        self.low_head = nn.Linear(hidden_dim, self.low_dim)

        self._cached_outputs: Dict[str, Any] = {}
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        state = state.float()

        high_features = self.high_trunk(state)
        gating_logits = self.high_head(high_features)

        cursor = 0
        temp = max(1e-3, self.cfg.gating_temperature)

        task_logits = gating_logits[:, cursor:cursor + 3] / temp
        task_alloc = torch.softmax(task_logits, dim=-1)
        cursor += 3

        if self.num_rsus > 0:
            rsu_logits = gating_logits[:, cursor:cursor + self.num_rsus] / temp
            rsu_alloc = torch.softmax(rsu_logits, dim=-1)
            cursor += self.num_rsus
        else:
            rsu_alloc = torch.zeros(state.size(0), 0, device=state.device)

        if self.num_uavs > 0:
            uav_logits = gating_logits[:, cursor:cursor + self.num_uavs] / temp
            uav_alloc = torch.softmax(uav_logits, dim=-1)
        else:
            uav_alloc = torch.zeros(state.size(0), 0, device=state.device)

        high_concat = torch.cat([task_alloc, rsu_alloc, uav_alloc], dim=-1)

        low_input = torch.cat([state, high_concat], dim=-1)
        low_features = self.low_trunk(low_input)

        if self.cfg.residual_low_level:
            residual = torch.relu(self.state_adapter(state))
            low_features = low_features + residual

        control_raw = torch.tanh(self.low_head(low_features))

        full_action = torch.cat([high_concat, control_raw], dim=-1)

        self._cached_outputs = {
            "task_alloc": task_alloc.detach().cpu().numpy(),
            "rsu_alloc": rsu_alloc.detach().cpu().numpy() if self.num_rsus > 0 else None,
            "uav_alloc": uav_alloc.detach().cpu().numpy() if self.num_uavs > 0 else None,
            "control_raw": control_raw.detach().cpu().numpy(),
        }

        return full_action

    def get_last_outputs(self) -> Dict[str, Any]:
        return self._cached_outputs


class HierarchicalTD3Agent(TD3Agent):
    """使用分层 Actor 的 TD3 Agent。"""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: Optional[HierarchicalTD3Config],
        num_rsus: int,
        num_uavs: int,
    ):
        self.hier_config = config or HierarchicalTD3Config()
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        super().__init__(
            state_dim,
            action_dim,
            self.hier_config,
            actor_cls=HierarchicalTD3Actor,
            actor_kwargs={
                "num_rsus": num_rsus,
                "num_uavs": num_uavs,
                "config_obj": self.hier_config,
            },
        )

    def get_last_hierarchy(self) -> Dict[str, Any]:
        actor = getattr(self, "actor", None)
        if actor is None or not hasattr(actor, "get_last_outputs"):
            return {}
        return actor.get_last_outputs()


class TD3HierarchicalEnvironment(TD3LatencyEnergyEnvironment):
    """延时-能耗奖励 + 分层策略的 TD3 环境封装。"""

    def __init__(self, num_vehicles: int, num_rsus: int, num_uavs: int):
        super().__init__(num_vehicles, num_rsus, num_uavs)
        self.config = HierarchicalTD3Config()
        self.agent = HierarchicalTD3Agent(
            self.state_dim,
            self.action_dim,
            self.config,
            num_rsus,
            num_uavs,
        )
        self._last_hierarchy: Dict[str, Any] = {}
        print("🏗️ TD3-Hierarchical 分层策略已启用")

    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        actions = super().get_actions(state, training)
        if hasattr(self.agent, "get_last_hierarchy"):
            self._last_hierarchy = self.agent.get_last_hierarchy()
        return actions

    def get_hierarchy_diagnostics(self) -> Dict[str, Any]:
        """返回最近一次动作的分层信息，用于可视化或记录。"""
        return self._last_hierarchy or {}
