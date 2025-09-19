"""
DDPG (Deep Deterministic Policy Gradient) 单智能体算法实现
专门适配MATD3-MIG系统的VEC环境

主要特点:
1. Actor-Critic架构处理连续动作空间
2. 经验回放机制提高样本效率
3. 目标网络稳定训练过程
4. 噪声探索策略

对应论文: Continuous Control with Deep Reinforcement Learning
"""
# 性能优化 - 必须在其他导入之前
try:
    from performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'DDPG': 128}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class DDPGConfig:
    """DDPG算法配置 - 优化收敛性"""
    # 网络结构
    hidden_dim: int = 128  # 减小网络复杂度
    actor_lr: float = 3e-4  # 提高学习率
    critic_lr: float = 3e-4  # 统一学习率
    
    # 训练参数
    batch_size: int = 64   # 减小批次大小，提高更新频率
    buffer_size: int = 50000  # 减小缓冲区大小
    tau: float = 0.01      # 增加软更新系数，加快目标网络更新
    gamma: float = 0.95    # 减小折扣因子，更关注短期奖励
    
    # 探索参数
    noise_scale: float = 0.2   # 增加初始探索
    noise_decay: float = 0.995  # 加快噪声衰减
    min_noise: float = 0.05    # 保持最小探索
    
    # 训练频率
    update_freq: int = 1
    warmup_steps: int = 500    # 减少预热步数


class DDPGActor(nn.Module):
    """DDPG Actor网络 - 确定性策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(DDPGActor, self).__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用较小的权重初始化
        nn.init.uniform_(self.network[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.network[-2].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.max_action * self.network(state)


class DDPGCritic(nn.Module):
    """DDPG Critic网络 - Q函数网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(DDPGCritic, self).__init__()
        
        # 状态编码器
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 状态-动作融合网络
        self.fusion_network = nn.Sequential(
            nn.Linear(hidden_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in [self.state_encoder, self.fusion_network]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用较小的权重初始化
        nn.init.uniform_(self.fusion_network[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fusion_network[-1].bias, -3e-3, 3e-3)
    
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        state_features = self.state_encoder(state)
        fusion_input = torch.cat([state_features, action], dim=1)
        return self.fusion_network(fusion_input)


class DDPGReplayBuffer:
    """DDPG经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """采样经验批次"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch_states = torch.FloatTensor(self.states[indices])
        batch_actions = torch.FloatTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return self.size


class DDPGAgent:
    """DDPG智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: DDPGConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('DDPG', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = DDPGActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = DDPGCritic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 目标网络
        self.target_actor = DDPGActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic = DDPGCritic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 经验回放缓冲区
        self.replay_buffer = DDPGReplayBuffer(config.buffer_size, state_dim, action_dim)
        
        # 探索噪声
        self.noise_scale = config.noise_scale
        self.step_count = 0
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # 添加探索噪声
        if training:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
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
        
        # 采样经验批次
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
            self.replay_buffer.sample(self.config.batch_size)
        
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        
        # 更新Critic
        critic_loss = self._update_critic(batch_states, batch_actions, batch_rewards, 
                                        batch_next_states, batch_dones)
        
        # 更新Actor
        actor_loss = self._update_actor(batch_states)
        
        # 软更新目标网络
        self.soft_update(self.target_actor, self.actor, self.config.tau)
        self.soft_update(self.target_critic, self.critic, self.config.tau)
        
        # 衰减噪声
        self.noise_scale = max(self.config.min_noise, 
                              self.noise_scale * self.config.noise_decay)
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
            'noise_scale': self.noise_scale
        }
    
    def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                      rewards: torch.Tensor, next_states: torch.Tensor, 
                      dones: torch.Tensor) -> float:
        """更新Critic网络"""
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        return critic_loss.item()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """更新Actor网络"""
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
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
            'noise_scale': self.noise_scale,
            'step_count': self.step_count
        }, f"{filepath}_ddpg.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_ddpg.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']
        self.step_count = checkpoint['step_count']


class DDPGEnvironment:
    """DDPG训练环境"""
    
    def __init__(self):
        self.config = DDPGConfig()
        
        # 环境配置 - 整合VEC系统状态
        self.state_dim = 60  # 整合所有节点状态
        self.action_dim = 30  # 整合所有节点动作
        
        # 创建智能体
        self.agent = DDPGAgent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"✓ DDPG环境初始化完成")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
    
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
        """改进的奖励函数 - 更好的收敛性和学习稳定性"""
        import numpy as np
        
        # 安全获取指标
        def safe_get_metric(key: str, default: float = 0.0) -> float:
            value = system_metrics.get(key, default)
            if np.isnan(value) or np.isinf(value):
                return default
            return max(0.0, value)

        # 获取关键指标
        task_completion_rate = safe_get_metric('task_completion_rate', 0.0)
        avg_delay = safe_get_metric('avg_task_delay', 0.0)
        data_loss_rate = safe_get_metric('data_loss_rate', 0.0)
        cache_hit_rate = safe_get_metric('cache_hit_rate', 0.0)
        
        # 1. 基础性能奖励 (0-10分)
        # 任务完成率是最重要的指标
        completion_reward = task_completion_rate * 10.0
        
        # 2. 延迟惩罚 (-5到0分)
        # 延迟越小越好，使用指数衰减
        if avg_delay > 0:
            delay_penalty = -5.0 * min(1.0, avg_delay / 0.1)  # 0.1秒为基准
        else:
            delay_penalty = 0.0
        
        # 3. 数据丢失惩罚 (-10到0分)
        loss_penalty = -10.0 * data_loss_rate
        
        # 4. 缓存效率奖励 (0-2分)
        cache_reward = 2.0 * cache_hit_rate
        
        # 5. 组合奖励
        total_reward = completion_reward + delay_penalty + loss_penalty + cache_reward
        
        # 6. 添加稳定性机制
        # 如果性能很好，给额外奖励
        if task_completion_rate > 0.8 and data_loss_rate < 0.1:
            total_reward += 5.0  # 稳定性奖励
        
        # 如果性能很差，给额外惩罚
        if task_completion_rate < 0.5 or data_loss_rate > 0.3:
            total_reward -= 5.0  # 不稳定惩罚
        
        # 7. 限制奖励范围，提高学习稳定性
        final_reward = np.clip(total_reward, -20.0, 20.0)
        
        return final_reward
    
    def train_step(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """执行一步训练"""
        # DDPG需要numpy数组，如果是整数则转换
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
        print(f"✓ DDPG模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ DDPG模型已加载: {filepath}")
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        """存储经验到缓冲区 - 支持PPO兼容性"""
        # DDPG只使用前5个参数，log_prob和value被忽略
        self.agent.store_experience(state, action, reward, next_state, done)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict:
        """更新网络参数 - 支持PPO兼容性"""
        # DDPG不使用last_value参数
        return self.agent.update()
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'actor_loss_avg': float(np.mean(self.agent.actor_losses[-100:])) if self.agent.actor_losses else 0.0,
            'critic_loss_avg': float(np.mean(self.agent.critic_losses[-100:])) if self.agent.critic_losses else 0.0,
            'noise_scale': self.agent.noise_scale,
            'buffer_size': len(self.agent.replay_buffer),
            'step_count': self.step_count
        }