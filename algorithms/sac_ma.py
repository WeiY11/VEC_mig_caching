"""
SAC-MA (Multi-Agent Soft Actor-Critic) 算法实现
专门适配MATD3-MIG系统的多智能体环境

主要特点:
1. 基于最大熵强化学习的多智能体算法
2. 支持连续动作空间
3. 自动温度调节机制
4. 高样本效率

对应论文: Soft Actor-Critic for Discrete and Continuous Settings
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'SAC-MA': 256}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from torch.distributions import Normal
from collections import deque
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from config import config


@dataclass
class SACMAConfig:
    """SAC-MA算法配置"""
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
    
    # 其他参数
    auto_entropy_tuning: bool = True
    use_automatic_entropy_tuning: bool = True


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
    """SAC Critic网络 - Q函数网络"""
    
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
        
        # Q2网络 (Twin Critic)
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


class SACMAReplayBuffer:
    """SAC-MA经验回放缓冲区"""
    
    def __init__(self, capacity: int, num_agents: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
             rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
             dones: Dict[str, bool]):
        """添加经验到缓冲区"""
        experience = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict:
        """采样经验批次"""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        batch = random.sample(self.buffer, batch_size)
        
        # 重新组织数据格式
        batch_states = {}
        batch_actions = {}
        batch_rewards = {}
        batch_next_states = {}
        batch_dones = {}
        
        agent_ids = list(batch[0]['states'].keys())
        
        for agent_id in agent_ids:
            batch_states[agent_id] = np.array([exp['states'][agent_id] for exp in batch])
            batch_actions[agent_id] = np.array([exp['actions'][agent_id] for exp in batch])
            batch_rewards[agent_id] = np.array([exp['rewards'][agent_id] for exp in batch])
            batch_next_states[agent_id] = np.array([exp['next_states'][agent_id] for exp in batch])
            batch_dones[agent_id] = np.array([exp['dones'][agent_id] for exp in batch])
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'next_states': batch_next_states,
            'dones': batch_dones
        }
    
    def __len__(self):
        return len(self.buffer)


class SACMAAgent:
    """SAC-MA智能体"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, config: SACMAConfig):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
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
        
        # 温度参数 (自动调节熵)
        if config.auto_entropy_tuning:
            self.target_entropy = config.target_entropy_ratio * action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
        else:
            self.log_alpha = torch.log(torch.FloatTensor([config.initial_temperature])).to(self.device)
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.alpha_losses = []
    
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
    
    def update_critic(self, batch: Dict, other_agents: List['SACMAAgent']) -> float:
        """更新Critic网络"""
        states = torch.FloatTensor(batch['states'][self.agent_id]).to(self.device)
        actions = torch.FloatTensor(batch['actions'][self.agent_id]).to(self.device)
        rewards = torch.FloatTensor(batch['rewards'][self.agent_id]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(batch['next_states'][self.agent_id]).to(self.device)
        dones = torch.FloatTensor(batch['dones'][self.agent_id]).unsqueeze(1).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_probs
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # 当前Q值
        current_q1, current_q2 = self.critic(states, actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        
        return critic_loss.item()
    
    def update_actor_and_alpha(self, batch: Dict) -> Tuple[float, float]:
        """更新Actor网络和温度参数"""
        states = torch.FloatTensor(batch['states'][self.agent_id]).to(self.device)
        
        # 计算策略损失
        actions, log_probs, _ = self.actor.sample(states)
        q1, q2 = self.critic(states, actions)
        q = torch.min(q1, q2)
        
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        
        # 更新温度参数
        alpha_loss = 0.0
        if self.config.auto_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha_losses.append(alpha_loss.item())
            alpha_loss = alpha_loss.item()
        
        return actor_loss.item(), alpha_loss
    
    def soft_update(self):
        """软更新目标网络"""
        self._soft_update(self.target_critic, self.critic, self.config.tau)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新网络参数"""
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
        }, f"{filepath}_{self.agent_id}.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_{self.agent_id}.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.log_alpha = checkpoint['log_alpha']
        
        if self.config.auto_entropy_tuning and checkpoint['alpha_optimizer_state_dict'] is not None:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])


class SACMAEnvironment:
    """SAC-MA多智能体环境"""
    
    def __init__(self):
        self.config = SACMAConfig()
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('SAC-MA', self.config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境配置
        self.num_agents = 3
        self.state_dim = 20
        self.action_dim = 10  # 连续动作空间
        
        # 创建智能体
        self.agents = {}
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        for agent_id in agent_ids:
            self.agents[agent_id] = SACMAAgent(
                agent_id=agent_id,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                config=self.config
            )
        
        # 经验回放缓冲区
        self.replay_buffer = SACMAReplayBuffer(
            capacity=self.config.buffer_size,
            num_agents=self.num_agents
        )
        
        # 训练统计
        self.episode_count = 0
        self.update_count = 0
        
        print(f"✓ SAC-MA环境初始化完成")
        print(f"✓ 智能体数量: {self.num_agents}")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
    
    def get_actions(self, states: Dict[str, np.ndarray], training: bool = True) -> Dict[str, np.ndarray]:
        """获取所有智能体的动作"""
        actions = {}
        for agent_id, agent in self.agents.items():
            if agent_id in states:
                actions[agent_id] = agent.select_action(states[agent_id], training)
            else:
                actions[agent_id] = np.zeros(self.action_dim)
        
        return actions
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> Dict[str, np.ndarray]:
        """构建状态向量"""
        states = {}
        
        # 基础系统状态
        base_state = np.array([
            system_metrics.get('avg_task_delay', 0.0) / 1.0,
            system_metrics.get('total_energy_consumption', 0.0) / 1000.0,
            system_metrics.get('data_loss_rate', 0.0),
            system_metrics.get('cache_hit_rate', 0.0),
            system_metrics.get('migration_success_rate', 0.0),
        ])
        
        # 为每个智能体构建状态
        for agent_id in self.agents.keys():
            agent_specific = np.random.randn(self.state_dim - len(base_state))
            states[agent_id] = np.concatenate([base_state, agent_specific])
        
        return states
    
    def train_step(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                   rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                   dones: Dict[str, bool]) -> Dict:
        """执行一步训练"""
        # 存储经验
        self.replay_buffer.push(states, actions, rewards, next_states, dones)
        
        training_info = {}
        
        # 如果缓冲区有足够经验，开始训练
        if len(self.replay_buffer) >= self.config.batch_size:
            # 采样经验批次
            batch = self.replay_buffer.sample(self.config.batch_size)
            
            # 为每个智能体更新网络
            for agent_id, agent in self.agents.items():
                other_agents = [a for aid, a in self.agents.items() if aid != agent_id]
                
                # 更新Critic
                critic_loss = agent.update_critic(batch, other_agents)
                
                # 更新Actor和温度参数
                actor_loss, alpha_loss = agent.update_actor_and_alpha(batch)
                
                # 软更新目标网络
                if self.update_count % self.config.target_update_freq == 0:
                    agent.soft_update()
                
                training_info[agent_id] = {
                    'critic_loss': critic_loss,
                    'actor_loss': actor_loss,
                    'alpha_loss': alpha_loss,
                    'alpha': agent.alpha.item()
                }
            
            self.update_count += 1
        
        return training_info
    
    def save_models(self, filepath: str):
        """保存所有智能体模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save_model(f"{filepath}/sac_ma")
        
        print(f"✓ SAC-MA模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载所有智能体模型"""
        for agent_id, agent in self.agents.items():
            agent.load_model(f"{filepath}/sac_ma")
        
        print(f"✓ SAC-MA模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'actor_loss_avg': np.mean(agent.actor_losses[-100:]) if agent.actor_losses else 0.0,
                'critic_loss_avg': np.mean(agent.critic_losses[-100:]) if agent.critic_losses else 0.0,
                'alpha_loss_avg': np.mean(agent.alpha_losses[-100:]) if agent.alpha_losses else 0.0,
                'alpha': agent.alpha.item()
            }
        
        return stats
    
    def reset_hidden_states(self):
        """重置隐藏状态 (为SAC-MA不需要，但保持接口兼容性)"""
        pass
    
    def get_global_state(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        """获取全局状态"""
        global_state = []
        for agent_id in sorted(states.keys()):
            if agent_id in states:
                global_state.append(states[agent_id])
        return np.concatenate(global_state) if global_state else np.array([])
    
    def store_experience(self, states: Dict, actions: Dict, log_probs: Dict,
                        rewards: Dict, dones: Dict, global_state: Optional[np.ndarray] = None):
        """存储经验 (为SAC-MA不需要log_probs，但保持接口兼容性)"""
        # 对于SAC-MA，直接在train_step中存储经验
        pass
    
    def update(self):
        """更新所有智能体 (为SAC-MA的更新在train_step中执行)"""
        return {}