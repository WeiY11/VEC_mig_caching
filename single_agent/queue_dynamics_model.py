"""
模型化队列动态预测器 - Dreamer/MBPO风格

轻量级MLP模型，预测队列状态转移，用于短期rollout生成合成经验。
通过模拟未来3-5步的队列演化，提前识别队列溢出风险，加速策略稳定。

核心思想：
1. 学习环境动力学模型：s' = f(s, a)
2. 使用当前策略执行短期rollout
3. 生成合成transition并加入replay buffer
4. 计算想象奖励（基于预测的队列溢出概率）

作者：VEC_mig_caching Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import deque


class QueueDynamicsModel(nn.Module):
    """
    队列动态模型 - 预测状态转移
    
    输入: [state, action]
    输出: [next_state_delta, reward, queue_overflow_prob]
    
    使用状态差分建模（delta model）提高稳定性
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [512, 512, 256],
        queue_feature_indices: Optional[List[int]] = None,
    ):
        """
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dims: 隐藏层维度列表
            queue_feature_indices: 状态中队列相关特征的索引（可选）
        """
        super(QueueDynamicsModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.queue_feature_indices = queue_feature_indices
        
        # 共享主干网络
        input_dim = state_dim + action_dim
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
            ])
            prev_dim = hidden_dim
        
        self.shared_trunk = nn.Sequential(*layers)
        
        # 状态差分预测头
        self.delta_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.ReLU(),
            nn.Linear(prev_dim // 2, state_dim),
        )
        
        # 奖励预测头
        self.reward_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 4),
            nn.ReLU(),
            nn.Linear(prev_dim // 4, 1),
        )
        
        # 队列溢出概率预测头（二分类logits）
        self.overflow_head = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 4),
            nn.ReLU(),
            nn.Linear(prev_dim // 4, 1),
            nn.Sigmoid(),  # 输出概率
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: [batch_size, state_dim]
            action: [batch_size, action_dim]
            
        Returns:
            next_state: [batch_size, state_dim] 预测的下一状态
            reward: [batch_size, 1] 预测的奖励
            overflow_prob: [batch_size, 1] 队列溢出概率
        """
        # 拼接输入
        x = torch.cat([state, action], dim=-1)
        
        # 共享特征提取
        features = self.shared_trunk(x)
        
        # 预测状态差分
        delta = self.delta_head(features)
        next_state = state + delta  # 残差连接
        
        # 预测奖励
        reward = self.reward_head(features)
        
        # 预测队列溢出概率
        overflow_prob = self.overflow_head(features)
        
        return next_state, reward, overflow_prob
    
    def compute_loss(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        next_states: torch.Tensor,
        rewards: torch.Tensor,
        overflow_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        计算模型训练损失
        
        Args:
            states: [batch_size, state_dim]
            actions: [batch_size, action_dim]
            next_states: [batch_size, state_dim] 真实下一状态
            rewards: [batch_size, 1] 真实奖励
            overflow_labels: [batch_size, 1] 队列溢出标签（0/1），可选
            
        Returns:
            losses: 损失字典
        """
        pred_next_states, pred_rewards, pred_overflow_probs = self.forward(states, actions)
        
        # 状态预测损失（MSE）
        state_loss = F.mse_loss(pred_next_states, next_states)
        
        # 奖励预测损失（MSE）
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # 队列溢出分类损失（BCE）
        if overflow_labels is not None:
            overflow_loss = F.binary_cross_entropy(pred_overflow_probs, overflow_labels)
        else:
            # 如果没有标签，使用启发式：奖励<-5认为可能溢出
            heuristic_labels = (rewards < -5.0).float()
            overflow_loss = F.binary_cross_entropy(pred_overflow_probs, heuristic_labels)
        
        # 总损失（加权组合）
        total_loss = state_loss + 0.5 * reward_loss + 0.3 * overflow_loss
        
        return {
            'total_loss': total_loss,
            'state_loss': state_loss,
            'reward_loss': reward_loss,
            'overflow_loss': overflow_loss,
        }


class ModelBasedRollout:
    """
    基于模型的rollout管理器
    
    功能：
    1. 从当前状态执行策略的想象rollout
    2. 生成合成transitions
    3. 计算想象奖励（包含队列溢出惩罚）
    """
    
    def __init__(
        self,
        dynamics_model: QueueDynamicsModel,
        rollout_horizon: int = 5,
        imagined_reward_weight: float = 0.3,
        overflow_penalty: float = -10.0,
        device: str = 'cpu',
    ):
        """
        Args:
            dynamics_model: 队列动态模型
            rollout_horizon: rollout步数（3-5步）
            imagined_reward_weight: 想象奖励在总奖励中的权重
            overflow_penalty: 队列溢出的额外惩罚
            device: 计算设备
        """
        self.model = dynamics_model
        self.rollout_horizon = rollout_horizon
        self.imagined_reward_weight = imagined_reward_weight
        self.overflow_penalty = overflow_penalty
        self.device = device
        
        # 统计信息
        self.rollout_count = 0
        self.total_imagined_transitions = 0
    
    def rollout(
        self,
        initial_states: torch.Tensor,
        actor: nn.Module,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        """
        执行想象rollout
        
        Args:
            initial_states: [batch_size, state_dim] 初始状态
            actor: 策略网络（用于选择动作）
            
        Returns:
            transitions: 合成的(s, a, r, s', done)列表
        """
        self.model.eval()
        transitions = []
        
        with torch.no_grad():
            current_states = initial_states.to(self.device)
            batch_size = current_states.size(0)
            
            for step in range(self.rollout_horizon):
                # 使用当前策略选择动作
                actions = actor(current_states)
                
                # 预测下一状态、奖励和溢出概率
                next_states, pred_rewards, overflow_probs = self.model(current_states, actions)
                
                # 计算想象奖励（基础奖励 + 溢出惩罚）
                imagined_rewards = self._compute_imagined_reward(
                    pred_rewards, overflow_probs
                )
                
                # 启发式终止条件：如果溢出概率>0.8，认为episode结束
                dones = (overflow_probs > 0.8).float()
                
                # 存储transitions
                for i in range(batch_size):
                    s = current_states[i].cpu().numpy()
                    a = actions[i].cpu().numpy()
                    r = float(imagined_rewards[i].cpu().item())
                    s_next = next_states[i].cpu().numpy()
                    done = bool(dones[i].cpu().item())
                    
                    transitions.append((s, a, r, s_next, done))
                
                # 如果所有episode都结束，提前停止
                if dones.all():
                    break
                
                # 更新状态
                current_states = next_states
        
        self.model.train()
        self.rollout_count += 1
        self.total_imagined_transitions += len(transitions)
        
        return transitions
    
    def _compute_imagined_reward(
        self,
        pred_rewards: torch.Tensor,
        overflow_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算想象奖励
        
        Formula: r_imagined = r_pred + penalty * overflow_prob
        
        Args:
            pred_rewards: [batch_size, 1] 预测的基础奖励
            overflow_probs: [batch_size, 1] 队列溢出概率
            
        Returns:
            imagined_rewards: [batch_size, 1] 想象奖励
        """
        # 溢出惩罚项（概率加权）
        overflow_penalty = self.overflow_penalty * overflow_probs
        
        # 组合奖励
        imagined_rewards = pred_rewards + overflow_penalty
        
        return imagined_rewards
    
    def generate_synthetic_transitions(
        self,
        real_states: torch.Tensor,
        actor: nn.Module,
        num_rollouts_per_state: int = 1,
    ) -> List[Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]]:
        """
        从真实状态生成合成transitions
        
        Args:
            real_states: [num_states, state_dim] 真实状态
            actor: 策略网络
            num_rollouts_per_state: 每个状态执行的rollout次数
            
        Returns:
            synthetic_transitions: 合成transition列表
        """
        all_transitions = []
        
        for _ in range(num_rollouts_per_state):
            transitions = self.rollout(real_states, actor)
            all_transitions.extend(transitions)
        
        return all_transitions


class ModelTrainer:
    """
    动态模型训练器
    
    负责从replay buffer采样数据训练dynamics model
    """
    
    def __init__(
        self,
        model: QueueDynamicsModel,
        learning_rate: float = 3e-4,
        batch_size: int = 256,
        train_iterations: int = 10,
        device: str = 'cpu',
    ):
        """
        Args:
            model: 队列动态模型
            learning_rate: 学习率
            batch_size: 训练批次大小
            train_iterations: 每次训练的迭代次数
            device: 计算设备
        """
        self.model = model
        self.batch_size = batch_size
        self.train_iterations = train_iterations
        self.device = device
        
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # 训练统计
        self.train_count = 0
        self.recent_losses = deque(maxlen=100)
    
    def train(
        self,
        replay_buffer,
        min_buffer_size: int = 1000,
    ) -> Dict[str, float]:
        """
        从replay buffer采样数据训练模型
        
        Args:
            replay_buffer: 经验回放缓冲区
            min_buffer_size: 开始训练的最小buffer大小
            
        Returns:
            training_stats: 训练统计字典
        """
        if len(replay_buffer) < min_buffer_size:
            return {}
        
        self.model.train()
        total_losses = {'total_loss': 0, 'state_loss': 0, 'reward_loss': 0, 'overflow_loss': 0}
        
        for _ in range(self.train_iterations):
            # 采样batch（注意：这里使用标准采样，不使用queue-aware采样）
            try:
                # 假设replay buffer有sample方法
                batch = replay_buffer.sample(self.batch_size, beta=0.4)
                states, actions, rewards, next_states, dones, _, _ = batch
            except:
                # 如果replay buffer接口不同，回退到简单采样
                indices = np.random.choice(len(replay_buffer), self.batch_size)
                states = torch.FloatTensor(replay_buffer.states[indices])
                actions = torch.FloatTensor(replay_buffer.actions[indices])
                rewards = torch.FloatTensor(replay_buffer.rewards[indices]).unsqueeze(1)
                next_states = torch.FloatTensor(replay_buffer.next_states[indices])
            
            # 移动到设备
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_states = next_states.to(self.device)
            
            # 计算损失
            losses = self.model.compute_loss(
                states, actions, next_states, rewards
            )
            
            # 反向传播
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # 累加损失
            for key, value in losses.items():
                total_losses[key] += value.item()
        
        # 平均损失
        avg_losses = {k: v / self.train_iterations for k, v in total_losses.items()}
        
        # 更新统计
        self.train_count += 1
        self.recent_losses.append(avg_losses['total_loss'])
        
        # 添加额外统计信息
        avg_losses['model_train_count'] = self.train_count
        avg_losses['avg_recent_loss'] = float(np.mean(self.recent_losses))
        
        return avg_losses
    
    def get_stats(self) -> Dict[str, float]:
        """获取训练统计"""
        return {
            'model_train_count': self.train_count,
            'avg_recent_loss': float(np.mean(self.recent_losses)) if self.recent_losses else 0.0,
        }
