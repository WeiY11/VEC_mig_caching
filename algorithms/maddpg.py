"""
MADDPG (Multi-Agent Deep Deterministic Policy Gradient) 算法实现
专门适配MATD3-MIG系统的多智能体环境

主要特点:
1. 集中式训练，分布式执行 (Centralized Training, Decentralized Execution)
2. 每个智能体有独立的Actor-Critic网络
3. Critic网络可以访问全局状态信息
4. 支持连续动作空间

对应论文: Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'MADDPG': 256}  # 默认值

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
class MADDPGConfig:
    """MADDPG算法配置"""
    # 网络结构
    hidden_dim: int = 256
    actor_lr: float = 1e-4
    critic_lr: float = 2e-4
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000
    tau: float = 0.01  # 软更新系数
    gamma: float = 0.99  # 折扣因子
    
    # 探索参数
    noise_scale: float = 0.1
    noise_decay: float = 0.9999
    min_noise: float = 0.01
    
    # 训练频率
    update_freq: int = 1
    target_update_freq: int = 1


class MADDPGActor(nn.Module):
    """MADDPG Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(MADDPGActor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        # 输出层使用较小的初始化
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        # 使用tanh激活确保动作在[-1, 1]范围内
        action = torch.tanh(self.output(x))
        
        return action


class MADDPGCritic(nn.Module):
    """MADDPG Critic网络 - 价值网络 (可访问全局信息)"""
    
    def __init__(self, global_state_dim: int, global_action_dim: int, hidden_dim: int = 256):
        super(MADDPGCritic, self).__init__()
        
        # 状态编码器
        self.state_encoder = nn.Linear(global_state_dim, hidden_dim)
        
        # 动作编码器  
        self.action_encoder = nn.Linear(global_action_dim, hidden_dim)
        
        # 融合网络
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in [self.state_encoder, self.action_encoder, self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.uniform_(self.output.weight, -3e-3, 3e-3)
        nn.init.constant_(self.output.bias, 0.0)
    
    def forward(self, global_states: torch.Tensor, global_actions: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            global_states: 全局状态 [batch_size, global_state_dim]
            global_actions: 全局动作 [batch_size, global_action_dim]
        """
        # 编码状态和动作
        state_emb = F.relu(self.state_encoder(global_states))
        action_emb = F.relu(self.action_encoder(global_actions))
        
        # 融合特征
        x = torch.cat([state_emb, action_emb], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        q_value = self.output(x)
        
        return q_value


class MADDPGReplayBuffer:
    """MADDPG经验回放缓冲区"""
    
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


class MADDPGAgent:
    """MADDPG智能体"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 global_state_dim: int, global_action_dim: int,
                 config: MADDPGConfig):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim  # 添加全局状态维度
        self.global_action_dim = global_action_dim  # 添加全局动作维度
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = MADDPGActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = MADDPGCritic(global_state_dim, global_action_dim, config.hidden_dim).to(self.device)
        
        # 目标网络
        self.target_actor = MADDPGActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic = MADDPGCritic(global_state_dim, global_action_dim, config.hidden_dim).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 噪声参数
        self.noise_scale = config.noise_scale
        
        # 训练统计
        self.actor_loss_history = []
        self.critic_loss_history = []
    
    def select_action(self, state: Optional[np.ndarray], training: bool = True) -> np.ndarray:
        """选择动作"""
        # 处理空状态的情况
        if state is None:
            return np.zeros(self.action_dim, dtype=np.float32)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
            # 确保tensor不为None且进行安全转换
            if action_tensor is not None:
                action = action_tensor.cpu().detach().numpy()[0]
            else:
                action = np.zeros(self.action_dim, dtype=np.float32)
        
        # 添加探索噪声
        if training:
            noise = np.random.normal(0, self.noise_scale, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action.astype(np.float32)
    
    def update_critic(self, batch: Dict, other_agents: List['MADDPGAgent']) -> float:
        """更新Critic网络"""
        # 准备全局状态和动作
        global_states = self._get_global_states(batch['states'])
        global_actions = self._get_global_actions(batch['actions'])
        global_next_states = self._get_global_states(batch['next_states'])
        
        # 计算目标Q值
        next_global_actions = self._get_target_actions(batch['next_states'], other_agents)
        target_q = self.target_critic(global_next_states, next_global_actions)
        
        rewards = torch.FloatTensor(batch['rewards'][self.agent_id]).unsqueeze(1).to(self.device)
        dones = torch.FloatTensor(batch['dones'][self.agent_id]).unsqueeze(1).to(self.device)
        
        target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        # 当前Q值
        current_q = self.critic(global_states, global_actions)
        
        # Critic损失
        critic_loss = F.mse_loss(current_q, target_q.detach())
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_loss_history.append(float(critic_loss.item()))
        
        return float(critic_loss.item())
    
    def update_actor(self, batch: Dict, other_agents: List['MADDPGAgent']) -> float:
        """更新Actor网络"""
        # 准备状态
        states = torch.FloatTensor(batch['states'][self.agent_id]).to(self.device)
        
        # 计算策略梯度
        actions = self.actor(states)
        
        # 构建全局动作 (当前智能体使用新动作，其他智能体使用原动作)
        global_actions = self._get_global_actions_with_new_action(batch['actions'], actions, other_agents)
        global_states = self._get_global_states(batch['states'])
        
        # Actor损失 = -Q值 (策略梯度)
        actor_loss = -self.critic(global_states, global_actions).mean()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()
        
        self.actor_loss_history.append(float(actor_loss.item()))
        
        return float(actor_loss.item())
    
    def soft_update(self):
        """软更新目标网络"""
        self._soft_update(self.target_actor, self.actor, self.config.tau)
        self._soft_update(self.target_critic, self.critic, self.config.tau)
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """硬更新网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def _get_global_states(self, states_batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """获取全局状态"""
        global_states = []
        for agent_id in sorted(states_batch.keys()):
            if agent_id in states_batch and states_batch[agent_id] is not None:
                global_states.append(states_batch[agent_id])
        
        if global_states:
            global_states_array = np.concatenate(global_states, axis=1)
            return torch.FloatTensor(global_states_array).to(self.device)
        else:
            # 创建零填充的tensor
            batch_size = 1  # 默认批次大小
            if states_batch:
                first_state = next(iter(states_batch.values()))
                if first_state is not None:
                    batch_size = first_state.shape[0]
            
            zero_states = np.zeros((batch_size, self.global_state_dim), dtype=np.float32)
            return torch.FloatTensor(zero_states).to(self.device)
    
    def _get_global_actions(self, actions_batch: Dict[str, np.ndarray]) -> torch.Tensor:
        """获取全局动作"""
        global_actions = []
        for agent_id in sorted(actions_batch.keys()):
            if agent_id in actions_batch and actions_batch[agent_id] is not None:
                global_actions.append(actions_batch[agent_id])
        
        if global_actions:
            global_actions_array = np.concatenate(global_actions, axis=1)
            return torch.FloatTensor(global_actions_array).to(self.device)
        else:
            # 创建零填充的tensor
            batch_size = 1  # 默认批次大小
            if actions_batch:
                first_action = next(iter(actions_batch.values()))
                if first_action is not None:
                    batch_size = first_action.shape[0]
            
            zero_actions = np.zeros((batch_size, self.global_action_dim), dtype=np.float32)
            return torch.FloatTensor(zero_actions).to(self.device)
    
    def _get_target_actions(self, next_states_batch: Dict[str, np.ndarray], 
                           other_agents: List['MADDPGAgent']) -> torch.Tensor:
        """获取目标动作"""
        target_actions = []
        
        for agent_id in sorted(next_states_batch.keys()):
            if agent_id in next_states_batch and next_states_batch[agent_id] is not None:
                next_states = torch.FloatTensor(next_states_batch[agent_id]).to(self.device)
                
                if agent_id == self.agent_id:
                    actions = self.target_actor(next_states)
                else:
                    # 查找对应智能体
                    agent = next((agent for agent in other_agents if agent.agent_id == agent_id), None)
                    if agent is not None:
                        actions = agent.target_actor(next_states)
                    else:
                        # 如果找不到智能体，创建零动作
                        batch_size = next_states.shape[0]
                        actions = torch.zeros(batch_size, self.action_dim).to(self.device)
                
                target_actions.append(actions)
        
        if target_actions:
            return torch.cat(target_actions, dim=1)
        else:
            # 返回零动作
            batch_size = 1
            if next_states_batch:
                first_state = next(iter(next_states_batch.values()))
                if first_state is not None:
                    batch_size = first_state.shape[0]
            return torch.zeros(batch_size, self.global_action_dim).to(self.device)
    
    def _get_global_actions_with_new_action(self, actions_batch: Dict[str, np.ndarray], 
                                          new_actions: torch.Tensor,
                                          other_agents: List['MADDPGAgent']) -> torch.Tensor:
        """构建包含新动作的全局动作"""
        global_actions = []
        
        for agent_id in sorted(actions_batch.keys()):
            if agent_id == self.agent_id:
                global_actions.append(new_actions)
            else:
                if agent_id in actions_batch and actions_batch[agent_id] is not None:
                    actions = torch.FloatTensor(actions_batch[agent_id]).to(self.device)
                    global_actions.append(actions)
                else:
                    # 如果动作不存在，创建零动作
                    batch_size = new_actions.shape[0]
                    zero_actions = torch.zeros(batch_size, self.action_dim).to(self.device)
                    global_actions.append(zero_actions)
        
        if global_actions:
            return torch.cat(global_actions, dim=1)
        else:
            # 如果没有动作，返回新动作扩展的版本
            batch_size = new_actions.shape[0]
            return torch.cat([new_actions, torch.zeros(batch_size, self.global_action_dim - self.action_dim).to(self.device)], dim=1)
    
    def decay_noise(self):
        """衰减探索噪声"""
        self.noise_scale = max(self.config.min_noise, 
                              self.noise_scale * self.config.noise_decay)
    
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
        }, f"{filepath}_{self.agent_id}.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_{self.agent_id}.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']


class MADDPGEnvironment:
    """MADDPG多智能体环境"""
    
    def __init__(self):
        self.config = MADDPGConfig()
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('MADDPG', self.config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境配置
        self.num_agents = 3  # vehicle_agent, rsu_agent, uav_agent
        self.state_dim = 20  # 单个智能体状态维度
        self.action_dim = 10  # 单个智能体动作维度
        
        # 全局维度
        self.global_state_dim = self.state_dim * self.num_agents
        self.global_action_dim = self.action_dim * self.num_agents
        
        # 创建智能体
        self.agents = {}
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        for agent_id in agent_ids:
            self.agents[agent_id] = MADDPGAgent(
                agent_id=agent_id,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                global_state_dim=self.global_state_dim,
                global_action_dim=self.global_action_dim,
                config=self.config
            )
        
        # 经验回放缓冲区
        self.replay_buffer = MADDPGReplayBuffer(
            capacity=self.config.buffer_size,
            num_agents=self.num_agents
        )
        
        # 训练统计
        self.episode_count = 0
        self.update_count = 0
        
        print(f"✓ MADDPG环境初始化完成")
        print(f"✓ 智能体数量: {self.num_agents}")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
        print(f"✓ 全局状态维度: {self.global_state_dim}")
    
    def get_actions(self, states: Dict[str, np.ndarray], training: bool = True) -> Dict[str, np.ndarray]:
        """获取所有智能体的动作"""
        actions = {}
        for agent_id, agent in self.agents.items():
            if agent_id in states and states[agent_id] is not None:
                actions[agent_id] = agent.select_action(states[agent_id], training)
            else:
                # 如果状态不存在或为None，返回零动作
                actions[agent_id] = np.zeros(self.action_dim, dtype=np.float32)
        
        return actions
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> Dict[str, np.ndarray]:
        """
        构建状态向量 (简化版本，与MATD3保持兼容)
        """
        states = {}
        
        # 基础系统状态 (所有智能体共享)
        base_state = np.array([
            system_metrics.get('avg_task_delay', 0.0) / 1.0,  # 归一化时延
            system_metrics.get('total_energy_consumption', 0.0) / 1000.0,  # 归一化能耗
            system_metrics.get('data_loss_rate', 0.0),  # 数据丢失率
            system_metrics.get('cache_hit_rate', 0.0),  # 缓存命中率
            system_metrics.get('migration_success_rate', 0.0),  # 迁移成功率
        ], dtype=np.float32)
        
        # 为每个智能体构建状态
        for agent_id in self.agents.keys():
            # 基础状态 + 智能体特定状态
            remaining_dim = max(0, self.state_dim - len(base_state))
            agent_specific = np.random.randn(remaining_dim).astype(np.float32)  # 简化实现
            states[agent_id] = np.concatenate([base_state, agent_specific]).astype(np.float32)
        
        return states
    
    def train_step(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                   rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                   dones: Dict[str, bool]) -> Dict[str, Dict]:
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
                
                # 更新Actor
                actor_loss = agent.update_actor(batch, other_agents)
                
                # 软更新目标网络
                if self.update_count % self.config.target_update_freq == 0:
                    agent.soft_update()
                
                # 衰减噪声
                agent.decay_noise()
                
                training_info[agent_id] = {
                    'critic_loss': critic_loss,
                    'actor_loss': actor_loss,
                    'noise_scale': agent.noise_scale
                }
            
            self.update_count += 1
        
        return training_info
    
    def save_models(self, filepath: str):
        """保存所有智能体模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            agent.save_model(f"{filepath}/maddpg")
        
        print(f"✓ MADDPG模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载所有智能体模型"""
        for agent_id, agent in self.agents.items():
            agent.load_model(f"{filepath}/maddpg")
        
        print(f"✓ MADDPG模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'actor_loss_avg': np.mean(agent.actor_loss_history[-100:]) if agent.actor_loss_history else 0.0,
                'critic_loss_avg': np.mean(agent.critic_loss_history[-100:]) if agent.critic_loss_history else 0.0,
                'noise_scale': agent.noise_scale
            }
        
        return stats
    
    def reset_hidden_states(self):
        """重置隐藏状态 (为MADDPG不需要，但保持接口兼容性)"""
        pass
    
    def get_global_state(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        """获取全局状态"""
        global_state = []
        for agent_id in sorted(states.keys()):
            if agent_id in states and states[agent_id] is not None:
                global_state.append(states[agent_id])
        
        if global_state:
            return np.concatenate(global_state, axis=0).astype(np.float32)
        else:
            # 返回正确维度的零数组
            total_dim = self.global_state_dim if hasattr(self, 'global_state_dim') else len(self.agents) * 20
            return np.zeros(total_dim, dtype=np.float32)
    
    def store_experience(self, states: Dict, actions: Dict, log_probs: Dict,
                        rewards: Dict, dones: Dict, global_state: Optional[np.ndarray] = None):
        """存储经验 (为MADDPG不需要log_probs，但保持接口兼容性)"""
        # 对于MADDPG，直接在train_step中存储经验
        pass
    
    def update(self):
        """更新所有智能体 (为MADDPG的更新在train_step中执行)"""
        return {}