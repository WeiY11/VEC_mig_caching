"""
SAC (Soft Actor-Critic) 单智能体算法实现
专门适配MATD3-MIG系统的VEC环境

主要特点:
1. 最大熵强化学习框架
2. 自动温度参数调节
3. 双Q网络减少过估计
4. 高样本效率

对应论文: Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'SAC': 256}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Normal
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class SACConfig:
    """SAC算法配置"""
    # 网络结构
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    
    # SAC参数
    initial_temperature: float = 0.2
    target_entropy_ratio: float = -1.0  # 目标熵比例
    tau: float = 0.005  # 软更新系数
    gamma: float = 0.99  # 折扣因子
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000
    update_freq: int = 1
    target_update_freq: int = 1
    warmup_steps: int = 1000
    
    # 其他参数
    auto_entropy_tuning: bool = True


class SACActor(nn.Module):
    """SAC Actor网络 - 随机策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SACActor, self).__init__()
        
        self.action_dim = action_dim
        
        # 共享特征层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 均值和对数标准差输出
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        
        # 对数标准差的范围限制
        self.log_std_min = -20
        self.log_std_max = 2
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.xavier_uniform_(self.mean_layer.weight)
        nn.init.constant_(self.mean_layer.bias, 0.0)
        nn.init.xavier_uniform_(self.log_std_layer.weight)
        nn.init.constant_(self.log_std_layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor):
        """前向传播"""
        features = self.feature_layers(state)
        
        mean = self.mean_layer(features)
        log_std = self.log_std_layer(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state: torch.Tensor, reparam: bool = True):
        """采样动作"""
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        
        if reparam:
            # 重参数化技巧
            x_t = normal.rsample()
        else:
            x_t = normal.sample()
        
        # tanh变换确保动作在[-1, 1]范围内
        action = torch.tanh(x_t)
        
        # 计算对数概率，需要考虑tanh变换的雅可比行列式
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        mean = torch.tanh(mean)
        
        return action, log_prob, mean


class SACCritic(nn.Module):
    """SAC Critic网络 - 双Q函数网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(SACCritic, self).__init__()
        
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
    
    def forward(self, state: torch.Tensor, action: torch.Tensor):
        """前向传播"""
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_network(sa)
        q2 = self.q2_network(sa)
        
        return q1, q2


class SACReplayBuffer:
    """SAC优先级经验回放缓冲区"""
    
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
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        # 新经验使用最大优先级
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """优先级采样经验批次"""
        if self.size == 0:
            raise ValueError("Buffer is empty")
        
        # 计算采样概率
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        # 计算重要性采样权重
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights = weights / weights.max()  # 归一化
        
        batch_states = torch.FloatTensor(self.states[indices])
        batch_actions = torch.FloatTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = abs(td_error) + 1e-6
            self.max_priority = max(self.max_priority, self.priorities[idx])
    
    def __len__(self):
        return self.size


class SACAgent:
    """SAC智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: SACConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('SAC', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = SACActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = SACCritic(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic = SACCritic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 学习率调度器 - 提高收敛稳定性
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=10000, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=10000, gamma=0.9)
        
        # 温度参数 (自动调节熵)
        if config.auto_entropy_tuning:
            self.target_entropy = config.target_entropy_ratio * action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.alpha_scheduler = optim.lr_scheduler.StepLR(self.alpha_optimizer, step_size=10000, gamma=0.9)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([config.initial_temperature])).to(self.device)
            self.alpha_scheduler = None
        
        # 经验回放缓冲区 - 支持优先级经验回放
        self.replay_buffer = SACReplayBuffer(config.buffer_size, state_dim, action_dim, alpha=0.6)
        
        # PER beta参数
        self.beta = 0.4
        self.beta_increment = (1.0 - 0.4) / max(1, 100000)  # 100k步内从0.4增加到1.0
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
        self.step_count = 0
    
    @property
    def alpha(self):
        """获取当前温度参数"""
        return self.log_alpha.exp()
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        if training:
            with torch.no_grad():
                action, _, _ = self.actor.sample(state_tensor)
        else:
            # 评估模式使用确定性策略
            with torch.no_grad():
                _, _, action = self.actor.sample(state_tensor)
        
        return action.cpu().numpy()[0]
    
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
        
        # 采样经验批次 - 支持优先级经验回放
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights = \
            self.replay_buffer.sample(self.config.batch_size, self.beta)
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        weights = weights.to(self.device)
        
        # 更新Critic并获取TD误差
        critic_loss, td_errors = self._update_critic(batch_states, batch_actions, batch_rewards, 
                                        batch_next_states, batch_dones, weights)
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # 更新Actor和温度参数
        actor_loss, alpha_loss = self._update_actor_and_alpha(batch_states)
        
        # 软更新目标网络
        if self.step_count % self.config.target_update_freq == 0:
            self.soft_update(self.target_critic, self.critic, self.config.tau)
        
        # 更新学习率调度器
        if self.step_count % 100 == 0:  # 每100步更新一次
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            if self.alpha_scheduler is not None:
                self.alpha_scheduler.step()
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'alpha_loss': alpha_loss,
            'alpha': self.alpha.item(),
            'actor_lr': self.actor_optimizer.param_groups[0]['lr'],
            'critic_lr': self.critic_optimizer.param_groups[0]['lr']
        }
    
    def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                      rewards: torch.Tensor, next_states: torch.Tensor, 
                      dones: torch.Tensor, weights: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """更新Critic网络并返回TD误差"""
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        
        # 计算TD误差
        td_errors1 = torch.abs(current_q1 - target_q)
        td_errors2 = torch.abs(current_q2 - target_q)
        td_errors = torch.max(td_errors1, td_errors2).squeeze()
        
        # 使用重要性采样权重
        critic_loss1 = (weights * F.mse_loss(current_q1, target_q, reduction='none')).mean()
        critic_loss2 = (weights * F.mse_loss(current_q2, target_q, reduction='none')).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 添加梯度裁剪提高稳定性
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        return critic_loss.item(), td_errors
    
    def _update_actor_and_alpha(self, states: torch.Tensor) -> Tuple[float, float]:
        """更新Actor网络和温度参数"""
        # 计算策略损失
        actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 添加梯度裁剪提高稳定性
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        
        # 更新温度参数
        alpha_loss = 0.0
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            
            # 温度参数也需要梯度裁剪
            torch.nn.utils.clip_grad_norm_([self.log_alpha], max_norm=1.0)
            
            self.alpha_optimizer.step()
            
            self.alpha_losses.append(alpha_loss.item())
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
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
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha,
            'alpha_optimizer_state_dict': self.alpha_optimizer.state_dict() if self.config.auto_entropy_tuning else None,
            'step_count': self.step_count
        }, f"{filepath}_sac.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_sac.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        self.step_count = checkpoint['step_count']
        
        if self.config.auto_entropy_tuning and checkpoint['alpha_optimizer_state_dict'] is not None:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])


class SACEnvironment:
    """SAC训练环境"""
    
    def __init__(self):
        self.config = SACConfig()
        
        # 环境配置 - 整合VEC系统状态
        self.state_dim = 60  # 整合所有节点状态
        self.action_dim = 30  # 整合所有节点动作
        
        # 创建智能体
        self.agent = SACAgent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"✓ SAC环境初始化完成")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
        print(f"✓ 自动熵调节: {self.config.auto_entropy_tuning}")
        print(f"✓ 目标熵: {self.config.target_entropy_ratio * self.action_dim}")
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """构建全局状态向量"""
        # 基础系统状态
        base_state = np.array([
            system_metrics.get('avg_task_delay', 0.0) / 1.0,
            system_metrics.get('total_energy_consumption', 0.0) / 1000.0,
            system_metrics.get('data_loss_rate', 0.0),
            system_metrics.get('cache_hit_rate', 0.0),
            system_metrics.get('migration_success_rate', 0.0),
        ])
        
        # 节点特定状态 (简化实现)
        node_states_flat = np.random.randn(self.state_dim - len(base_state))
        
        return np.concatenate([base_state, node_states_flat])
    
    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """将全局动作分解为各节点动作"""
        actions = {}
        start_idx = 0
        
        # 为每个智能体类型分配动作
        for agent_type in ['vehicle_agent', 'rsu_agent', 'uav_agent']:
            end_idx = start_idx + 10  # 每个智能体10个动作维度
            actions[agent_type] = action[start_idx:end_idx]
            start_idx = end_idx
        
        return actions
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        return self.decompose_action(global_action)
    
    def calculate_reward(self, system_metrics: Dict) -> float:
        """计算奖励 - 使用标准化奖励函数"""
        from utils.standardized_reward import calculate_standardized_reward
        return calculate_standardized_reward(system_metrics, agent_type='single_agent')
    
    def train_step(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """执行一步训练"""
        # SAC需要numpy数组，如果是整数则转换
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
        print(f"✓ SAC模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ SAC模型已加载: {filepath}")
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        """存储经验到缓冲区 - 支持PPO兼容性"""
        # SAC只使用前5个参数，log_prob和value被忽略
        self.agent.store_experience(state, action, reward, next_state, done)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict:
        """更新网络参数 - 支持PPO兼容性"""
        # SAC不使用last_value参数
        return self.agent.update()
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'actor_loss_avg': float(np.mean(self.agent.actor_losses[-100:])) if self.agent.actor_losses else 0.0,
            'critic_loss_avg': float(np.mean(self.agent.critic_losses[-100:])) if self.agent.critic_losses else 0.0,
            'alpha_loss_avg': float(np.mean(self.agent.alpha_losses[-100:])) if self.agent.alpha_losses else 0.0,
            'alpha': self.agent.alpha.item(),
            'buffer_size': len(self.agent.replay_buffer),
            'step_count': self.step_count,
            'auto_entropy_tuning': self.config.auto_entropy_tuning
        }