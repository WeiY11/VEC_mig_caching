"""
队列感知的优先经验回放缓冲区

扩展标准的prioritized replay buffer，增加队列指标跟踪：
1. 队列占用率（queue occupancy）
2. 丢包率（packet loss rate）
3. 迁移拥塞度（migration congestion）

通过队列稀缺度加权，优先采样高负载、高拥塞的样本，加速学习拥塞应对策略。

作者：VEC_mig_caching Team
"""

import numpy as np
import torch
from typing import Tuple, Dict, Optional


class QueueAwareReplayBuffer:
    """
    队列感知的优先经验回放缓冲区
    
    核心改进：
    1. 追踪每个transition的队列指标
    2. 结合TD误差和队列稀缺度计算优先级
    3. 高队列/高丢包样本获得更高采样概率
    4. 低负载样本降权，避免过拟合轻载场景
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        queue_priority_weight: float = 0.3,
        queue_metrics_ema_decay: float = 0.9,
    ):
        """
        Args:
            capacity: 缓冲区容量
            state_dim: 状态维度
            action_dim: 动作维度
            alpha: 优先级指数（PER中的α参数）
            queue_priority_weight: 队列因素在优先级中的权重
            queue_metrics_ema_decay: 队列指标的EMA平滑系数
        """
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        self.queue_priority_weight = queue_priority_weight
        self.queue_metrics_ema_decay = queue_metrics_ema_decay
        
        # 标准transition存储
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # TD误差优先级（标准PER）
        self.td_priorities = np.zeros(capacity, dtype=np.float32)
        
        # 队列指标存储
        self.queue_occupancies = np.zeros(capacity, dtype=np.float32)
        self.packet_losses = np.zeros(capacity, dtype=np.float32)
        self.migration_congestions = np.zeros(capacity, dtype=np.float32)
        
        # 队列指标的全局统计（用于归一化）
        self.queue_occ_ema = 0.5
        self.packet_loss_ema = 0.0
        self.migration_cong_ema = 0.0
        
        # 队列稀缺度权重系数
        self.queue_occ_coef = 0.5
        self.packet_loss_coef = 0.3
        self.migration_cong_coef = 0.2
    
    def __len__(self):
        return self.size
    
    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        queue_metrics: Optional[Dict[str, float]] = None,
    ):
        """
        添加经验到缓冲区
        
        Args:
            state, action, reward, next_state, done: 标准transition
            queue_metrics: 队列指标字典，包含：
                - 'queue_occupancy': 队列占用率 [0, 1]
                - 'packet_loss': 丢包率 [0, 1]
                - 'migration_congestion': 迁移拥塞度 [0, 1]
        """
        # 存储标准transition
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # 初始化TD优先级为最大值（新样本高优先级）
        max_prio = self.td_priorities.max() if self.size > 0 else 1.0
        self.td_priorities[self.ptr] = max_prio
        
        # 存储队列指标
        if queue_metrics is not None:
            queue_occ = float(queue_metrics.get('queue_occupancy', 0.5))
            packet_loss = float(queue_metrics.get('packet_loss', 0.0))
            migration_cong = float(queue_metrics.get('migration_congestion', 0.0))
        else:
            # 如果未提供，使用中性值
            queue_occ = 0.5
            packet_loss = 0.0
            migration_cong = 0.0
        
        # 裁剪到合理范围
        queue_occ = np.clip(queue_occ, 0.0, 1.0)
        packet_loss = np.clip(packet_loss, 0.0, 1.0)
        migration_cong = np.clip(migration_cong, 0.0, 1.0)
        
        self.queue_occupancies[self.ptr] = queue_occ
        self.packet_losses[self.ptr] = packet_loss
        self.migration_congestions[self.ptr] = migration_cong
        
        # 更新全局EMA统计
        decay = self.queue_metrics_ema_decay
        self.queue_occ_ema = decay * self.queue_occ_ema + (1 - decay) * queue_occ
        self.packet_loss_ema = decay * self.packet_loss_ema + (1 - decay) * packet_loss
        self.migration_cong_ema = decay * self.migration_cong_ema + (1 - decay) * migration_cong
        
        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _compute_queue_scarcity(self, indices: np.ndarray) -> np.ndarray:
        """
        计算队列稀缺度分数
        
        稀缺度反映样本的"拥塞程度"，高稀缺度 = 高队列占用 + 高丢包 + 高迁移拥塞
        
        Args:
            indices: 样本索引
            
        Returns:
            scarcity_scores: [len(indices)] 稀缺度分数，归一化到[0, 1]
        """
        # 提取队列指标
        queue_occ = self.queue_occupancies[indices]
        packet_loss = self.packet_losses[indices]
        migration_cong = self.migration_congestions[indices]
        
        # 归一化到相对于全局EMA的比例
        # 高于平均值的样本被放大，低于平均值的被缩小
        queue_occ_norm = queue_occ / (self.queue_occ_ema + 1e-6)
        packet_loss_norm = packet_loss / (self.packet_loss_ema + 1e-6)
        migration_cong_norm = migration_cong / (self.migration_cong_ema + 1e-6)
        
        # 加权组合
        scarcity = (
            self.queue_occ_coef * queue_occ_norm +
            self.packet_loss_coef * packet_loss_norm +
            self.migration_cong_coef * migration_cong_norm
        )
        
        # 裁剪到合理范围
        scarcity = np.clip(scarcity, 0.0, 3.0)  # 最多3倍放大
        
        return scarcity
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray, torch.Tensor]:
        """
        按优先级采样经验批次
        
        优先级 = TD误差^α * (1 + λ * 队列稀缺度)
        其中λ = queue_priority_weight
        
        Args:
            batch_size: 批次大小
            beta: 重要性采样权重的指数
            
        Returns:
            batch_states: [batch_size, state_dim]
            batch_actions: [batch_size, action_dim]
            batch_rewards: [batch_size, 1]
            batch_next_states: [batch_size, state_dim]
            batch_dones: [batch_size, 1]
            indices: [batch_size] 采样索引（用于更新优先级）
            weights: [batch_size, 1] 重要性采样权重
        """
        # 获取有效范围内的TD优先级
        if self.size == self.capacity:
            td_prios = self.td_priorities
        else:
            td_prios = self.td_priorities[:self.size]
        
        # 清理异常值
        td_prios = np.nan_to_num(td_prios, nan=1.0, posinf=1.0, neginf=1.0)
        td_prios = np.maximum(td_prios, 1e-6)
        
        # 计算队列稀缺度加权
        all_indices = np.arange(self.size)
        queue_scarcity = self._compute_queue_scarcity(all_indices)
        
        # 融合TD优先级和队列稀缺度
        # Formula: p = td_prio^α * (1 + λ * scarcity)
        td_prios_powered = td_prios ** self.alpha
        queue_weight_factor = 1.0 + self.queue_priority_weight * queue_scarcity
        combined_prios = td_prios_powered * queue_weight_factor
        
        # 归一化为概率分布
        combined_prios = np.nan_to_num(combined_prios, nan=0.0, posinf=0.0, neginf=0.0)
        prob_sum = combined_prios.sum()
        
        if prob_sum <= 0 or not np.isfinite(prob_sum):
            # 退化为均匀分布
            probs = np.ones(self.size, dtype=np.float32) / self.size
        else:
            probs = combined_prios / prob_sum
        
        # 再次清理并确保概率有效
        probs = np.nan_to_num(probs, nan=1.0/self.size, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)
        
        # 最终归一化
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(self.size, dtype=np.float32) / self.size
        
        # 采样
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        
        # 计算重要性采样权重
        # w_i = (N * P(i))^(-β) / max_w
        weights = (self.size * probs[indices]) ** (-beta)
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        
        max_weight = weights.max()
        if max_weight > 0 and np.isfinite(max_weight):
            weights /= max_weight
        else:
            weights = np.ones_like(weights)
        
        weights = weights.astype(np.float32)
        
        # 提取批次数据
        batch_states = torch.FloatTensor(self.states[indices])
        batch_actions = torch.FloatTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        weights_tensor = torch.FloatTensor(weights).unsqueeze(1)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights_tensor
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        更新TD误差优先级
        
        Args:
            indices: 样本索引
            td_errors: TD误差（绝对值）
        """
        # 裁剪异常值
        td_errors = np.clip(td_errors, 0.0, 100.0)
        td_errors = np.nan_to_num(td_errors, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 更新优先级（加上小的epsilon避免零优先级）
        self.td_priorities[indices] = td_errors + 1e-6
    
    def get_queue_statistics(self) -> Dict[str, float]:
        """
        获取队列指标的统计信息（用于监控）
        
        Returns:
            stats: 队列统计字典
        """
        if self.size == 0:
            return {
                'avg_queue_occupancy': 0.0,
                'avg_packet_loss': 0.0,
                'avg_migration_congestion': 0.0,
                'max_queue_occupancy': 0.0,
                'max_packet_loss': 0.0,
            }
        
        valid_indices = slice(None) if self.size == self.capacity else slice(0, self.size)
        
        return {
            'avg_queue_occupancy': float(self.queue_occupancies[valid_indices].mean()),
            'avg_packet_loss': float(self.packet_losses[valid_indices].mean()),
            'avg_migration_congestion': float(self.migration_congestions[valid_indices].mean()),
            'max_queue_occupancy': float(self.queue_occupancies[valid_indices].max()),
            'max_packet_loss': float(self.packet_losses[valid_indices].max()),
            'queue_occ_ema': float(self.queue_occ_ema),
            'packet_loss_ema': float(self.packet_loss_ema),
            'migration_cong_ema': float(self.migration_cong_ema),
        }
