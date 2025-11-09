"""
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 单智能体算法实现
专门适配MATD3-MIG系统的VEC环境

主要特点:
1. Twin Critic网络减少过估计
2. 延迟策略更新提高稳定性
3. 目标策略平滑化减少方差
4. 改进的探索策略

对应论文: Addressing Function Approximation Error in Actor-Critic Methods
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'TD3': 128}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math
from collections import deque
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

from config import config
from .common_state_action import UnifiedStateActionSpace


@dataclass
class TD3Config:
    """TD3算法配置 - 🎯 v2.0基线版本（稳定可靠）"""
    # 网络结构
    hidden_dim: int = 512  # 🔧 统一使用512，确保所有车辆数配置都有充足容量  
    actor_lr: float = 2e-4  # 🔧 Actor学习率（提升以加快收敛）
    critic_lr: float = 3e-4  # 🔧 Critic学习率（提升以更好评估）
    graph_embed_dim: int = 128  # 图编码器输出维度
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000
    tau: float = 0.005  # 目标网络软更新系数
    gamma: float = 0.99  
    
    # TD3特有参数
    policy_delay: int = 2  # 策略延迟更新
    target_noise: float = 0.05  # 目标策略平滑噪声
    noise_clip: float = 0.2  # 噪声裁剪范围
    
    # 探索参数（优化：更快收敛到稳定策略）
    exploration_noise: float = 0.12  # 初始探索噪声
    noise_decay: float = 0.9992  # 🔧 噪声衰减率（更慢衰减，训练中后期更稳）
    min_noise: float = 0.02  # 🔧 最小探索噪声（更低底限，提升稳定性）
    
    # 🔧 新增：梯度裁剪防止过拟合
    gradient_clip_norm: float = 0.7  # 🔧 放宽梯度裁剪，允许适度更新
    use_gradient_clip: bool = True   # 启用梯度裁剪
    use_reward_normalization: bool = True
    
    def __post_init__(self):
        """从环境变量读取配置，用于固定拓扑优化"""
        import os
        
        # 读取固定拓扑优化器设置的环境变量
        if 'TD3_HIDDEN_DIM' in os.environ:
            self.hidden_dim = int(os.environ['TD3_HIDDEN_DIM'])
            print(f"[TD3Config] 从环境变量读取 hidden_dim: {self.hidden_dim}")
            
        if 'TD3_ACTOR_LR' in os.environ:
            self.actor_lr = float(os.environ['TD3_ACTOR_LR'])
            print(f"[TD3Config] 从环境变量读取 actor_lr: {self.actor_lr}")
            
        if 'TD3_CRITIC_LR' in os.environ:
            self.critic_lr = float(os.environ['TD3_CRITIC_LR'])
            print(f"[TD3Config] 从环境变量读取 critic_lr: {self.critic_lr}")
            
        if 'TD3_BATCH_SIZE' in os.environ:
            self.batch_size = int(os.environ['TD3_BATCH_SIZE'])
            print(f"[TD3Config] 从环境变量读取 batch_size: {self.batch_size}")
            
        if 'TD3_TAU' in os.environ:
            self.tau = float(os.environ['TD3_TAU'])
            print(f"[TD3Config] 从环境变量读取 tau: {self.tau}")
            
        if 'TD3_EXPLORATION_NOISE' in os.environ:
            self.exploration_noise = float(os.environ['TD3_EXPLORATION_NOISE'])
            print(f"[TD3Config] 从环境变量读取 exploration_noise: {self.exploration_noise}")
            
        if 'TD3_POLICY_DELAY' in os.environ:
            self.policy_delay = int(os.environ['TD3_POLICY_DELAY'])
            print(f"[TD3Config] 从环境变量读取 policy_delay: {self.policy_delay}")
            
        if 'TD3_GRADIENT_CLIP' in os.environ:
            self.gradient_clip_norm = float(os.environ['TD3_GRADIENT_CLIP'])
            print(f"[TD3Config] 从环境变量读取 gradient_clip_norm: {self.gradient_clip_norm}")
    
    # PER 参数（优化以减少低质量样本影响）
    per_alpha: float = 0.6  # 🔧 回调优先级指数，减轻早期过度关注
    per_beta_start: float = 0.4  # 🔧 回调IS起点，平衡样本权重
    per_beta_frames: int = 400000  # 🔧 放缓beta增长，稳定学习

    # 后期稳定策略参数
    late_stage_start_updates: int = 60000  # 约800轮更新步内提前稳定
    late_stage_start_updates: int = 60000
    late_stage_tau: float = 0.003
    late_stage_policy_delay: int = 3
    late_stage_noise_floor: float = 0.03
    td_error_clip: float = 4.0
    
    # 训练频率
    update_freq: int = 1
    warmup_steps: int = 4000


class GraphFeatureExtractor(nn.Module):
    """
    轻量图特征编码器：将车辆/RSU/UAV状态映射为增强的全局表示。
    通过显式的距离/缓存注意力，让Actor学习权衡策略。
    """

    def __init__(
        self,
        num_vehicles: int,
        num_rsus: int,
        num_uavs: int,
        node_feature_dim: int = 5,
        global_feature_dim: int = 8,
        embed_dim: int = 128,
    ):
        super().__init__()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.node_feature_dim = node_feature_dim
        self.global_feature_dim = global_feature_dim
        self.embed_dim = embed_dim

        self.node_encoder = nn.Sequential(
            nn.Linear(node_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.group_proj = nn.ModuleDict(
            {
                "vehicles": nn.Linear(embed_dim, embed_dim),
                "rsus": nn.Linear(embed_dim, embed_dim),
                "uavs": nn.Linear(embed_dim, embed_dim),
            }
        )
        self.distance_proj = nn.Sequential(
            nn.Linear(3, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.cache_proj = nn.Sequential(
            nn.Linear(2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )
        self.tradeoff_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
        )

        # 群组嵌入(3) + 全局attention(1) + 距离/缓存上下文(2) + 权衡上下文(1) + 全局特征
        self.output_dim = embed_dim * 7 + global_feature_dim

        self._last_outputs: Dict[str, torch.Tensor] = {}

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        batch = state.size(0)
        total_nodes = self.num_vehicles + self.num_rsus + self.num_uavs
        dynamic_len = total_nodes * self.node_feature_dim
        if dynamic_len > state.size(1):
            raise ValueError("状态向量长度不足以拆解节点特征")

        dynamic_segment = state[:, :dynamic_len]
        global_segment = state[:, -self.global_feature_dim :]

        offset = 0

        def slice_group(count: int) -> torch.Tensor:
            nonlocal offset
            if count == 0:
                return torch.zeros(batch, 0, self.node_feature_dim, device=state.device)
            chunk = dynamic_segment[:, offset : offset + count * self.node_feature_dim]
            offset += count * self.node_feature_dim
            return chunk.view(batch, count, self.node_feature_dim)

        vehicle_feats = slice_group(self.num_vehicles)
        rsu_feats = slice_group(self.num_rsus)
        uav_feats = slice_group(self.num_uavs)

        all_nodes = torch.cat([vehicle_feats, rsu_feats, uav_feats], dim=1)
        if all_nodes.numel() == 0:
            zeros = torch.zeros(batch, self.output_dim - self.global_feature_dim, device=state.device)
            self._last_outputs = {}
            return torch.cat([zeros, global_segment], dim=1)

        encoded = self.node_encoder(all_nodes)  # [B, N, E]
        attn_logits = self.attn_proj(encoded).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attention_embed = torch.sum(attn_weights.unsqueeze(-1) * encoded, dim=1)

        def group_pool(feats: torch.Tensor, proj_layer: nn.Linear) -> torch.Tensor:
            if feats.size(1) == 0:
                return torch.zeros(batch, proj_layer.out_features, device=state.device)
            encoded_group = self.node_encoder(feats)
            pooled = encoded_group.mean(dim=1)
            return proj_layer(pooled)

        vehicle_embed = group_pool(vehicle_feats, self.group_proj["vehicles"])
        rsu_embed = group_pool(rsu_feats, self.group_proj["rsus"])
        uav_embed = group_pool(uav_feats, self.group_proj["uavs"])

        distance_inputs = []
        cache_inputs = []

        if vehicle_feats.size(1) > 0:
            pad = vehicle_feats.new_zeros(batch, vehicle_feats.size(1), 1)
            distance_inputs.append(torch.cat([vehicle_feats[:, :, :2], pad], dim=-1))
            cache_inputs.append(
                torch.stack([vehicle_feats[:, :, 3], vehicle_feats[:, :, 4]], dim=-1)
            )

        if rsu_feats.size(1) > 0:
            pad = rsu_feats.new_zeros(batch, rsu_feats.size(1), 1)
            distance_inputs.append(torch.cat([rsu_feats[:, :, :2], pad], dim=-1))
            cache_inputs.append(
                torch.stack([rsu_feats[:, :, 2], rsu_feats[:, :, 3]], dim=-1)
            )

        if uav_feats.size(1) > 0:
            uav_dist = uav_feats[:, :, :3]
            if uav_dist.size(-1) < 3:
                pad = uav_feats.new_zeros(batch, uav_feats.size(1), 3 - uav_dist.size(-1))
                uav_dist = torch.cat([uav_dist, pad], dim=-1)
            distance_inputs.append(uav_dist)
            cache_inputs.append(
                torch.stack([uav_feats[:, :, 3], uav_feats[:, :, 4]], dim=-1)
            )

        distance_context = state.new_zeros(batch, self.embed_dim)
        cache_context = state.new_zeros(batch, self.embed_dim)
        distance_weights = None
        cache_weights = None

        if distance_inputs:
            distance_inputs_tensor = torch.cat(distance_inputs, dim=1)
            distance_logits = self.distance_proj(distance_inputs_tensor).squeeze(-1)
            distance_weights = torch.softmax(distance_logits, dim=1)
            distance_context = torch.sum(distance_weights.unsqueeze(-1) * encoded, dim=1)

        if cache_inputs:
            cache_inputs_tensor = torch.cat(cache_inputs, dim=1)
            cache_logits = self.cache_proj(cache_inputs_tensor).squeeze(-1)
            cache_weights = torch.softmax(cache_logits, dim=1)
            cache_context = torch.sum(cache_weights.unsqueeze(-1) * encoded, dim=1)

        if distance_inputs and cache_inputs:
            tradeoff_logits = self.tradeoff_head(
                torch.cat([distance_context, cache_context], dim=1)
            )
            tradeoff_weights = torch.softmax(tradeoff_logits, dim=1)
        else:
            tradeoff_weights = state.new_full((batch, 2), 0.5)

        tradeoff_embed = (
            tradeoff_weights[:, :1] * distance_context
            + tradeoff_weights[:, 1:] * cache_context
        )

        fused = torch.cat(
            [
                vehicle_embed,
                rsu_embed,
                uav_embed,
                attention_embed,
                distance_context,
                cache_context,
                tradeoff_embed,
                global_segment,
            ],
            dim=1,
        )

        self._last_outputs = {
            "vehicle_embed": vehicle_embed,
            "rsu_embed": rsu_embed,
            "uav_embed": uav_embed,
            "attention_embed": attention_embed,
            "attention_weights": attn_weights,
            "distance_context": distance_context,
            "cache_context": cache_context,
            "tradeoff_weights": tradeoff_weights,
            "distance_weights": distance_weights,
            "cache_weights": cache_weights,
            "global_segment": global_segment,
        }

        return fused

    def get_last_outputs(self) -> Dict[str, torch.Tensor]:
        return getattr(self, "_last_outputs", {})



class TD3Actor(nn.Module):
    """多头结构TD3 Actor，引入注意力权衡距离与缓存，输出策略与引导分布。"""

    def __init__(
        self,
        state_dim: int,
        offload_dim: int,
        cache_dim: int,
        hidden_dim: int,
        num_vehicles: int,
        num_rsus: int,
        num_uavs: int,
        global_dim: int = 8,
        graph_embed_dim: int = 128,
        max_action: float = 1.0,
    ):
        super().__init__()
        self.max_action = max_action
        self.offload_dim = offload_dim
        self.cache_dim = cache_dim
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs

        self.encoder = GraphFeatureExtractor(
            num_vehicles=num_vehicles,
            num_rsus=num_rsus,
            num_uavs=num_uavs,
            node_feature_dim=5,
            global_feature_dim=global_dim,
            embed_dim=graph_embed_dim,
        )

        fused_dim = self.encoder.output_dim
        self.shared = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        head_hidden = max(hidden_dim // 2, 64)
        self.offload_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, offload_dim),
        )
        self.cache_head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, cache_dim),
        )

        self.distance_gate = nn.Sequential(
            nn.Linear(graph_embed_dim, hidden_dim),
            nn.ReLU(),
        )
        self.cache_gate = nn.Sequential(
            nn.Linear(graph_embed_dim, hidden_dim),
            nn.ReLU(),
        )

        self._init_weights(self.shared)
        self._init_weights(self.offload_head)
        self._init_weights(self.cache_head)
        self._init_weights(self.distance_gate)
        self._init_weights(self.cache_gate)

        self._last_offload = None
        self._last_cache = None
        self._latest_guidance: Dict[str, np.ndarray] = {}

    @staticmethod
    def _init_weights(module: nn.Module):
        for layer in module:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        fused = self.encoder(state)
        encoder_ctx = self.encoder.get_last_outputs()
        shared_feat = self.shared(fused)

        modulated_feat = shared_feat
        if encoder_ctx:
            distance_ctx = encoder_ctx.get("distance_context")
            cache_ctx = encoder_ctx.get("cache_context")
            tradeoff_weights = encoder_ctx.get("tradeoff_weights")
            if (
                distance_ctx is not None
                and cache_ctx is not None
                and tradeoff_weights is not None
            ):
                distance_adjust = self.distance_gate(distance_ctx)
                cache_adjust = self.cache_gate(cache_ctx)
                modulated_feat = (
                    shared_feat
                    + tradeoff_weights[:, :1] * distance_adjust
                    + tradeoff_weights[:, 1:] * cache_adjust
                )

        offload_raw = torch.tanh(self.offload_head(modulated_feat))
        cache_raw = torch.tanh(self.cache_head(modulated_feat))

        self._last_offload = offload_raw
        self._last_cache = cache_raw

        if not torch.is_grad_enabled():
            self._cache_guidance(offload_raw, cache_raw, encoder_ctx)
        else:
            self._latest_guidance = {}

        combined = torch.cat([offload_raw, cache_raw], dim=1)
        return self.max_action * combined

    def _cache_guidance(
        self,
        offload_raw: torch.Tensor,
        cache_raw: torch.Tensor,
        encoder_ctx: Optional[Dict[str, torch.Tensor]],
    ) -> None:
        offload = offload_raw.detach()
        cache = cache_raw.detach()

        start = 0
        task_logits = offload[:, start : start + 3]
        start += 3
        rsu_logits = (
            offload[:, start : start + self.num_rsus]
            if self.num_rsus > 0
            else None
        )
        start += self.num_rsus
        uav_logits = (
            offload[:, start : start + self.num_uavs]
            if self.num_uavs > 0
            else None
        )

        guidance: Dict[str, np.ndarray] = {}
        guidance["offload_prior"] = (
            torch.softmax(task_logits, dim=1).cpu().numpy()
        )
        if rsu_logits is not None and rsu_logits.size(1) > 0:
            guidance["rsu_prior"] = (
                torch.softmax(rsu_logits, dim=1).cpu().numpy()
            )
        if uav_logits is not None and uav_logits.size(1) > 0:
            guidance["uav_prior"] = (
                torch.softmax(uav_logits, dim=1).cpu().numpy()
            )

        guidance["cache_bias"] = (0.5 * (cache + 1.0)).cpu().numpy()

        if encoder_ctx:
            tradeoff_weights = encoder_ctx.get("tradeoff_weights")
            if tradeoff_weights is not None:
                guidance["tradeoff_weights"] = (
                    tradeoff_weights.detach().cpu().numpy()
                )
            distance_weights = encoder_ctx.get("distance_weights")
            cache_weights = encoder_ctx.get("cache_weights")
            distance_group = self._aggregate_group_weights(distance_weights)
            cache_group = self._aggregate_group_weights(cache_weights)
            if distance_group is not None:
                guidance["distance_focus"] = (
                    distance_group.detach().cpu().numpy()
                )
            if cache_group is not None:
                guidance["cache_focus"] = (
                    cache_group.detach().cpu().numpy()
                )

        cleaned: Dict[str, np.ndarray] = {}
        for key, value in guidance.items():
            if isinstance(value, np.ndarray):
                cleaned[key] = value.squeeze(0).copy() if value.shape[0] == 1 else value.copy()
        self._latest_guidance = cleaned

    def _aggregate_group_weights(
        self, weights: Optional[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        if weights is None:
            return None
        counts = [
            self.encoder.num_vehicles,
            self.encoder.num_rsus,
            self.encoder.num_uavs,
        ]
        idx = 0
        segments = []
        for count in counts:
            if count > 0:
                segments.append(weights[:, idx : idx + count].sum(dim=1, keepdim=True))
            idx += count
        if not segments:
            return None
        stacked = torch.cat(segments, dim=1)
        total = stacked.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return stacked / total

    def get_latest_guidance(self) -> Dict[str, np.ndarray]:
        if not self._latest_guidance:
            return {}
        return {key: value.copy() for key, value in self._latest_guidance.items()}


class TD3Critic(nn.Module):
    """TD3 Twin Critic网络 - 双Q网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(TD3Critic, self).__init__()
        
        # Q1网络
        self.q1_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Q2网络
        self.q2_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for network in [self.q1_network, self.q2_network]:
            for layer in network:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
            
            # 最后一层使用较小的权重初始化
            nn.init.uniform_(network[-1].weight, -3e-3, 3e-3)
            nn.init.uniform_(network[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播 - 返回两个Q值"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        
        return q1, q2
    
    def q1(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """只返回Q1值 (用于策略更新)"""
        sa = torch.cat([state, action], dim=1)
        return self.q1_network(sa)


class TD3ReplayBuffer:
    """TD3 Prioritized Experience Replay 缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        # 优先级数组
        self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def __len__(self):
        return self.size
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        max_prio = self.priorities.max() if self.size > 0 else 1.0
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.priorities[self.ptr] = max_prio
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float):
        """按优先级采样经验, 返回样本及重要性权重和索引"""
        if self.size == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.size]
        probs = prios ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()  # 归一化到[0,1]
        weights = weights.astype(np.float32)
        
        batch_states = torch.FloatTensor(self.states[indices])
        batch_actions = torch.FloatTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights_tensor
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """根据新的TD误差更新优先级"""
        self.priorities[indices] = priorities


class TD3Agent:
    """TD3智能体"""
    
    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 config: TD3Config,
                 num_vehicles: Optional[int] = None,
                 num_rsus: Optional[int] = None,
                 num_uavs: Optional[int] = None,
                 global_dim: int = 8,
                 actor_cls: Optional[Any] = None,
                 actor_kwargs: Optional[Dict[str, Any]] = None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        if num_vehicles is None:
            num_vehicles = 12
        if num_rsus is None:
            num_rsus = 4
        if num_uavs is None:
            num_uavs = 2
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.global_dim = global_dim
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('TD3', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        offload_dim = 3 + num_rsus + num_uavs
        cache_dim = action_dim - offload_dim
        if cache_dim <= 0:
            raise ValueError("动作维度不足以拆出缓存/迁移控制参数")

        actor_cls = actor_cls or TD3Actor
        base_actor_kwargs = {
            "state_dim": state_dim,
            "offload_dim": offload_dim,
            "cache_dim": cache_dim,
            "hidden_dim": config.hidden_dim,
            "num_vehicles": num_vehicles,
            "num_rsus": num_rsus,
            "num_uavs": num_uavs,
            "global_dim": global_dim,
            "graph_embed_dim": config.graph_embed_dim,
        }
        if actor_kwargs:
            base_actor_kwargs.update(actor_kwargs)

        # 创建网络
        self.actor = actor_cls(**dict(base_actor_kwargs)).to(self.device)
        self.critic = TD3Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 目标网络
        self.target_actor = actor_cls(**dict(base_actor_kwargs)).to(self.device)
        self.target_critic = TD3Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        # 🔧 暂时禁用学习率调度器，避免短期训练中学习率过快衰减
        # self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.995)
        # self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.995)
        
        # 经验回放缓冲区
        # PER beta参数
        self.beta = config.per_beta_start
        self.beta_increment = (1.0 - config.per_beta_start) / max(1, config.per_beta_frames)
        self.replay_buffer = TD3ReplayBuffer(config.buffer_size, state_dim, action_dim, alpha=config.per_alpha)
        
        # 探索噪声
        self.exploration_noise = config.exploration_noise
        self.step_count = 0
        self.update_count = 0
        self.late_stage_applied = False
        self.latest_guidance: Dict[str, np.ndarray] = {}
        self.guidance_ema: Dict[str, np.ndarray] = {}
        self.guidance_ema_decay = 0.6
        self.guidance_temperature_bounds = (0.7, 1.1)
        # 🔧 从全局config对象获取目标值（不是TD3Config）
        from config import config as global_config
        self.latency_target = float(getattr(global_config.rl, "latency_target", 0.4))
        self.energy_target = float(getattr(global_config.rl, "energy_target", 1200.0))
        self.guidance_feedback_beta = 0.12
        self.energy_ema = self.energy_target
        self.delay_ema = self.latency_target
        self.energy_excess_ema = 0.0
        self.delay_excess_ema = 0.0
        
        self.normalize_rewards = bool(getattr(config, 'use_reward_normalization', False))
        self.reward_rms = RunningMeanStd()

        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
            raw_guidance = self.actor.get_latest_guidance()
        processed_guidance = self._process_guidance(raw_guidance)
        self.latest_guidance = processed_guidance
        action = action_tensor.cpu().numpy()[0]
        
        # 添加探索噪声
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def get_latest_guidance(self) -> Dict[str, np.ndarray]:
        if not self.latest_guidance:
            return {}
        return {key: (value.copy() if isinstance(value, np.ndarray) else value)
                for key, value in self.latest_guidance.items()}
    
    def _process_guidance(self, guidance: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if not guidance:
            return {}
        smoothed = self._smooth_guidance(guidance)
        adjusted = self._apply_guidance_temperature(smoothed)
        return adjusted

    def _smooth_guidance(self, guidance: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        result: Dict[str, np.ndarray] = {}
        decay = float(np.clip(self.guidance_ema_decay, 0.0, 0.99))
        for key, value in guidance.items():
            arr = np.array(value, dtype=np.float32)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr.astype(np.float32)
            prev = self.guidance_ema.get(key)
            if prev is None or prev.shape != arr.shape:
                ema = arr
            else:
                ema = decay * prev + (1.0 - decay) * arr
            self.guidance_ema[key] = ema
            result[key] = ema.copy()
        return result

    def _apply_guidance_temperature(self, guidance: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        processed: Dict[str, np.ndarray] = {}
        energy_ratio = float(self.energy_ema / max(self.energy_target, 1e-6))
        delay_ratio = float(self.delay_ema / max(self.latency_target, 1e-6))
        base_pressure = energy_ratio / max(delay_ratio, 0.7)
        pressure_adjust = 0.8 * self.energy_excess_ema - 0.3 * self.delay_excess_ema
        energy_pressure = float(base_pressure + pressure_adjust)
        temperature_min, temperature_max = self.guidance_temperature_bounds
        energy_pressure = float(np.clip(energy_pressure, temperature_min, temperature_max))
        temperature = float(np.clip(1.0 / max(energy_pressure, 1e-6), temperature_min, temperature_max))

        for key, value in guidance.items():
            arr = np.array(value, dtype=np.float32)
            arr = arr.reshape(1, -1) if arr.ndim == 1 else arr
            if key in ("offload_prior", "rsu_prior", "uav_prior"):
                logits = np.log(np.clip(arr, 1e-6, None))
                logits = logits / max(temperature, 1e-6)
                logits = logits - logits.max(axis=1, keepdims=True)
                arr = np.exp(logits)
                arr /= arr.sum(axis=1, keepdims=True)
                if key == "offload_prior":
                    energy_scale = float(np.clip(energy_pressure, 0.85, 1.15))
                    local_weight = 1.0 + (energy_scale - 1.0) * 0.6
                    rsu_weight = 1.0 - (energy_scale - 1.0) * 0.35
                    uav_weight = rsu_weight
                    energy_weights = np.array([local_weight, rsu_weight, uav_weight], dtype=float)
                    energy_weights = np.clip(energy_weights, 0.35, 1.4)
                    arr = np.clip(arr * energy_weights.reshape(1, -1), 1e-4, None)
                    arr /= arr.sum(axis=1, keepdims=True)
            elif key == "tradeoff_weights":
                arr = np.clip(arr, 1e-6, None)
                arr[:, 1:2] *= energy_pressure
                arr[:, :1] /= energy_pressure
                arr /= arr.sum(axis=1, keepdims=True)
            processed[key] = arr[0] if arr.shape[0] == 1 else arr

        processed["energy_pressure"] = np.array([energy_pressure], dtype=np.float32)
        return processed

    def update_guidance_feedback(
        self,
        system_metrics: Dict[str, float],
        cache_metrics: Optional[Dict[str, float]] = None,
        migration_metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        energy = float(system_metrics.get("total_energy_consumption", 0.0))
        delay = float(system_metrics.get("avg_task_delay", 0.0))
        beta = float(np.clip(self.guidance_feedback_beta, 0.0, 1.0))
        self.energy_ema = (1.0 - beta) * self.energy_ema + beta * max(energy, 0.0)
        self.delay_ema = (1.0 - beta) * self.delay_ema + beta * max(delay, 0.0)
        energy_ratio = energy / max(self.energy_target, 1e-6) if self.energy_target > 0 else 0.0
        delay_ratio = delay / max(self.latency_target, 1e-6) if self.latency_target > 0 else 0.0
        self.energy_excess_ema = (1.0 - beta) * self.energy_excess_ema + beta * max(0.0, energy_ratio - 1.0)
        self.delay_excess_ema = (1.0 - beta) * self.delay_excess_ema + beta * max(0.0, delay_ratio - 1.0)

    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        self.step_count += 1
        
        # 预热期不更新
        if self.step_count < self.config.warmup_steps:
            return {}
        
        self.update_count += 1
        
        # 采样经验批次 (含索引与IS权重)
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights = \
            self.replay_buffer.sample(self.config.batch_size, self.beta)
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 将数据移动到设备
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        weights = weights.to(self.device)

        if self.normalize_rewards:
            rewards_np = batch_rewards.detach().cpu().numpy()
            self.reward_rms.update(rewards_np)
            reward_mean = torch.as_tensor(self.reward_rms.mean, device=self.device, dtype=batch_rewards.dtype)
            reward_std = torch.as_tensor(max(math.sqrt(self.reward_rms.var), 1e-6), device=self.device, dtype=batch_rewards.dtype)
            batch_rewards = torch.clamp((batch_rewards - reward_mean) / reward_std, -5.0, 5.0)

        # 更新Critic并获取TD误差
        critic_loss, td_errors = self._update_critic(batch_states, batch_actions, batch_rewards, 
                                        batch_next_states, batch_dones, weights)
        # 根据TD误差更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy() + 1e-6)

        training_info = {'critic_loss': critic_loss}
        
        # 后期稳定策略：动态调整
        if not self.late_stage_applied and self.update_count >= self.config.late_stage_start_updates:
            self._apply_late_stage_strategy()
            self.late_stage_applied = True

        # 延迟策略更新
        if self.update_count % self.config.policy_delay == 0:
            # 更新Actor
            actor_loss = self._update_actor(batch_states)
            training_info['actor_loss'] = actor_loss
            
            # 软更新目标网络
            self.soft_update(self.target_actor, self.actor, self.config.tau)
            self.soft_update(self.target_critic, self.critic, self.config.tau)
        
        # 衰减噪声
        self.exploration_noise = max(self.config.min_noise, 
                                   self.exploration_noise * self.config.noise_decay)
        
        training_info['exploration_noise'] = self.exploration_noise
        
        return training_info

    def _apply_late_stage_strategy(self):
        """应用后期稳定策略，防止奖励崩溃"""
        print("🔧 启用后期稳定策略：调整tau/policy_delay/噪声下限/TD误差裁剪")
        self.config.tau = self.config.late_stage_tau
        self.config.policy_delay = self.config.late_stage_policy_delay
        self.config.min_noise = max(self.config.min_noise, self.config.late_stage_noise_floor)
        # 限制现有噪声不低于新下限
        self.exploration_noise = max(self.exploration_noise, self.config.min_noise)
    
    def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                      rewards: torch.Tensor, next_states: torch.Tensor, 
                      dones: torch.Tensor, weights: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """更新Critic网络"""
        with torch.no_grad():
            # 目标策略平滑化
            next_actions = self.target_actor(next_states)
            
            # 添加裁剪噪声
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            # 计算目标Q值 (取两个Q网络的最小值)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic损失 (两个Q网络的损失之和)
        # TD误差
        td_errors = (current_q1 - target_q)
        # 加权MSE损失
        critic_loss = (weights * td_errors.pow(2)).mean() + (weights * (current_q2 - target_q).pow(2)).mean()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 🔧 使用配置的梯度裁剪参数
        if self.config.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip_norm)
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        return critic_loss.item(), td_errors.abs().squeeze()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """更新Actor网络"""
        # 计算策略损失 (只使用Q1网络)
        actions = self.actor(states)
        actor_loss = -self.critic.q1(states, actions).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 🔧 使用配置的梯度裁剪参数
        if self.config.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip_norm)
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        # 🔧 暂时禁用学习率调度器
        # self.actor_lr_scheduler.step()
        # self.critic_lr_scheduler.step()
        return actor_loss.item()
    
    def soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """硬更新网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'exploration_noise': self.exploration_noise,
            'step_count': self.step_count,
            'update_count': self.update_count
        }, f"{filepath}_td3.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_td3.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.exploration_noise = checkpoint['exploration_noise']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']


class TD3Environment:
    """TD3训练环境"""
    
    def __init__(
        self,
        num_vehicles: int = 12,
        num_rsus: int = 4,
        num_uavs: int = 2,
        use_central_resource: bool = False,
    ):
        self.config = TD3Config()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.use_central_resource = bool(use_central_resource)
        
        # 🔧 优化后的状态维度：所有节点统一为5维 + 全局状态16维（包含任务类型扩展）
        # 车辆状态: N×5维 + RSU状态: M×5维 + UAV状态: K×5维 + 全局: 16维
        (
            self.local_state_dim,
            self.global_state_dim,
            base_state_dim,
        ) = UnifiedStateActionSpace.calculate_state_dim(num_vehicles, num_rsus, num_uavs)
        self.state_dim = base_state_dim
        self.central_state_dim = 0
        if self.use_central_resource:
            # 车辆: 带宽/计算分配/使用率，各3*N
            # RSU/UAV: 分配/使用率，各2*节点数 + 3个聚合指标
            self.central_state_dim = (
                3 * self.num_vehicles
                + 2 * self.num_rsus
                + 2 * self.num_uavs
                + 3
            )
            self.state_dim += self.central_state_dim
        
        # 🔧 优化后的动作空间：动态适配网络拓扑
        # 3(任务分配) + num_rsus(RSU选择) + num_uavs(UAV选择) + 10(控制参数)
        self.control_param_dim = 10
        self.base_action_dim = 3 + num_rsus + num_uavs + self.control_param_dim
        self.central_resource_action_dim = 0
        if self.use_central_resource:
            # 带宽(车辆) + 车辆计算 + RSU计算 + UAV计算
            self.central_resource_action_dim = (
                self.num_vehicles * 2 + self.num_rsus + self.num_uavs
            )
        self.action_dim = self.base_action_dim + self.central_resource_action_dim
        
        # 创建智能体
        self.agent = TD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=self.config,
            num_vehicles=self.num_vehicles,
            num_rsus=self.num_rsus,
            num_uavs=self.num_uavs,
            global_dim=self.global_state_dim
        )
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"TD3环境初始化完成（优化版）")
        print(f"网络拓扑: {num_vehicles}辆车 + {num_rsus}个RSU + {num_uavs}个UAV")
        if self.use_central_resource:
            print(
                f"状态维度: {self.state_dim} = 基础{base_state_dim} + 中央资源{self.central_state_dim}"
            )
        else:
            print(
                f"状态维度: {self.state_dim} = 局部{self.local_state_dim} ({num_vehicles}×5 + {num_rsus}×5 + {num_uavs}×5) + 全局{self.global_state_dim}"
            )
        base_action_descr = f"3+{num_rsus}+{num_uavs}+{self.control_param_dim}"
        if self.use_central_resource:
            extra_descr = (
                f" + 中央资源({self.num_vehicles}×2+{self.num_rsus}+{self.num_uavs})"
            )
        else:
            extra_descr = ""
        print(f"动作维度: {self.action_dim} (动态适配: {base_action_descr}{extra_descr})")
        print(f"策略延迟更新: {self.config.policy_delay}")
        print(f"优化特性: 移除控制参数冗余, 添加全局状态, 统一归一化")
    
    def get_state_vector(
        self,
        node_states: Dict,
        system_metrics: Dict,
        resource_state: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        🔧 优化版状态向量构建
        状态组成: 车辆(N×5) + RSU(M×5) + UAV(K×5) + 全局(8) 维
        """
        state_components = []
        
        # ========== 1. 局部节点状态 ==========
        
        # 车辆状态 (N×5维)
        for i in range(self.num_vehicles):
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key][:5]  # 只取前5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in vehicle_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        # RSU状态 (M×5维) - 统一为5维
        for i in range(self.num_rsus):
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key][:5]  # 只取前5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in rsu_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        # UAV状态 (K×5维) - 统一为5维
        for i in range(self.num_uavs):
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key][:5]  # 只取前5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in uav_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.5, 0.0, 0.0])
        
        # ========== 2. 全局系统状态 (基础8维 + 任务类型8维) ==========
        global_state = self._build_global_state(node_states, system_metrics)
        state_components.extend(global_state)
        
        # ========== 3. 最终处理 ==========
        state_vector = np.array(state_components[:self.state_dim], dtype=np.float32)
        
        # 维度不足时补齐
        if len(state_vector) < self.state_dim:
            padding_needed = self.state_dim - len(state_vector)
            state_vector = np.pad(
                state_vector, (0, padding_needed), mode='constant', constant_values=0.5
            )
        
        if self.use_central_resource:
            central_state = self._build_central_resource_state(resource_state)
            state_vector[-self.central_state_dim :] = central_state
        
        # 数值安全检查
        state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        state_vector = np.clip(state_vector, 0.0, 1.0)  # 确保所有值在[0,1]
        
        return state_vector
    
    def _build_central_resource_state(
        self, resource_state: Optional[Dict]
    ) -> np.ndarray:
        if not self.use_central_resource or not resource_state:
            return np.full(self.central_state_dim, 0.0, dtype=np.float32)
        
        def _safe_vector(key: str, expected_len: int) -> np.ndarray:
            values = resource_state.get(key)
            if values is None:
                return np.full(expected_len, 1.0 / max(expected_len, 1), dtype=np.float32)
            arr = np.array(values, dtype=np.float32).reshape(-1)
            if arr.size < expected_len:
                arr = np.pad(arr, (0, expected_len - arr.size), constant_values=0.0)
            elif arr.size > expected_len:
                arr = arr[:expected_len]
            return np.clip(arr, 0.0, 1.0)
        
        segments = [
            _safe_vector('bandwidth_allocation', self.num_vehicles),
            _safe_vector('vehicle_compute_allocation', self.num_vehicles),
            _safe_vector('vehicle_compute_usage', self.num_vehicles),
            _safe_vector('rsu_compute_allocation', self.num_rsus),
            _safe_vector('rsu_compute_usage', self.num_rsus),
            _safe_vector('uav_compute_allocation', self.num_uavs),
            _safe_vector('uav_compute_usage', self.num_uavs),
        ]
        
        utilities = np.array(
            [
                float(resource_state.get('vehicle_utilization', 0.0)),
                float(resource_state.get('rsu_utilization', 0.0)),
                float(resource_state.get('uav_utilization', 0.0)),
            ],
            dtype=np.float32,
        )
        utilities = np.clip(np.nan_to_num(utilities, nan=0.0), 0.0, 1.0)
        segments.append(utilities)
        
        central_state = np.concatenate(segments).astype(np.float32, copy=False)
        if central_state.size < self.central_state_dim:
            central_state = np.pad(
                central_state,
                (0, self.central_state_dim - central_state.size),
                constant_values=0.0,
            )
        elif central_state.size > self.central_state_dim:
            central_state = central_state[: self.central_state_dim]
        return central_state
    
    def _build_global_state(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """构建包含任务类型统计的全局状态向量。"""
        return UnifiedStateActionSpace.build_global_state(
            node_states,
            system_metrics,
            self.num_vehicles,
            self.num_rsus
        )
    
    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        动作分解：3(任务分配) + RSU选择 + UAV选择 + control_param_dim(联动控制) + 中央资源段
        """
        if not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        if action.size < self.action_dim:
            action = np.pad(action, (0, self.action_dim - action.size), mode='constant')
        else:
            action = action.astype(np.float32)[: self.action_dim]

        actions: Dict[str, np.ndarray] = {}
        base_segment = action[: self.base_action_dim]

        idx = 0
        task_allocation = base_segment[idx : idx + 3]
        idx += 3

        rsu_selection = base_segment[idx : idx + self.num_rsus]
        idx += self.num_rsus

        uav_selection = base_segment[idx : idx + self.num_uavs]
        idx += self.num_uavs

        control_params = base_segment[idx : idx + self.control_param_dim]

        actions['vehicle_agent'] = action.copy()
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params

        return actions
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        actions = self.decompose_action(global_action)
        guidance = self.agent.get_latest_guidance()
        if guidance:
            actions['guidance'] = guidance
        return actions
    
    def calculate_reward(self, system_metrics: Dict, 
                       cache_metrics: Optional[Dict] = None,
                       migration_metrics: Optional[Dict] = None) -> float:
        """
        使用统一奖励计算器
        """
        from utils.unified_reward_calculator import calculate_unified_reward
        return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, algorithm="general")
    
    def train_step(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """执行一步训练"""
        # TD3需要numpy数组，如果是整数则转换
        if isinstance(action, int):
            action = np.array([action], dtype=np.float32)
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # 存储经验
        self.agent.store_experience(state, action, reward, next_state, done)
        
        # 更新网络
        training_info = self.agent.update()
        
        self.step_count += 1
        
        return training_info
    
    def save_models(self, filepath: str):
        """保存模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        self.agent.save_model(filepath)
        print(f"✓ TD3模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ TD3模型已加载: {filepath}")
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        """存储经验到缓冲区 - 支持PPO兼容性"""
        # TD3只使用前5个参数，log_prob和value被忽略
        self.agent.store_experience(state, action, reward, next_state, done)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict:
        """更新网络参数 - 支持PPO兼容性"""
        # TD3不使用last_value参数
        return self.agent.update()
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'actor_loss_avg': float(np.mean(self.agent.actor_losses[-100:])) if self.agent.actor_losses else 0.0,
            'critic_loss_avg': float(np.mean(self.agent.critic_losses[-100:])) if self.agent.critic_losses else 0.0,
            'exploration_noise': self.agent.exploration_noise,
            'buffer_size': len(self.agent.replay_buffer),
            'step_count': self.step_count,
            'update_count': self.agent.update_count,
            'policy_delay': self.config.policy_delay
        }
class RunningMeanStd:
    """跟踪标量的运行均值和方差，用于奖励归一化。"""

    def __init__(self, epsilon: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        if x.size == 0:
            return
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = x.size

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta * delta * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count
