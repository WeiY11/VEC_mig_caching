"""
QMIX (Monotonic Value Function Factorization) 算法实现
专门适配MATD3-MIG系统的多智能体环境

主要特点:
1. 基于值函数分解的多智能体Q学习
2. 单调性约束确保全局最优性
3. 集中式训练，分布式执行
4. 支持部分可观测环境

对应论文: QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'QMIX': 32}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import Dict, Tuple, Optional
from dataclasses import dataclass



@dataclass 
class QMIXConfig:
    """QMIX算法配置"""
    # 网络结构
    hidden_dim: int = 128
    rnn_hidden_dim: int = 64
    mixer_hidden_dim: int = 32
    lr: float = 5e-4
    
    # 训练参数
    batch_size: int = 32
    buffer_size: int = 50000
    target_update_freq: int = 200
    gamma: float = 0.99
    
    # 探索参数
    epsilon: float = 1.0
    epsilon_decay: float = 0.9995
    min_epsilon: float = 0.05
    
    # 梯度裁剪
    grad_clip: float = 10.0


class QMIXAgent(nn.Module):
    """QMIX个体智能体网络 (使用RNN处理部分可观测)"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, rnn_hidden_dim: int = 64):
        super(QMIXAgent, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        
        # 输入层
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        
        # RNN层 (处理序列信息)
        self.rnn = nn.GRU(hidden_dim, rnn_hidden_dim, batch_first=True)
        
        # 输出层
        self.fc2 = nn.Linear(rnn_hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in [self.fc1, self.fc2, self.output]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0.0)
    
    def forward(self, inputs: torch.Tensor, hidden_state: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            inputs: [batch_size, seq_len, input_dim] 或 [batch_size, input_dim]
            hidden_state: RNN隐藏状态
        
        Returns:
            q_values: [batch_size, seq_len, output_dim] 或 [batch_size, output_dim]
            new_hidden_state: 新的隐藏状态
        """
        batch_size = inputs.shape[0]
        
        # 处理输入维度
        if len(inputs.shape) == 2:
            # 单步输入，转换为序列格式
            inputs = inputs.unsqueeze(1)  # [batch_size, 1, input_dim]
            single_step = True
        else:
            single_step = False
        
        seq_len = inputs.shape[1]
        
        # 展开为 [batch_size * seq_len, input_dim]
        inputs_flat = inputs.view(-1, self.input_dim)
        
        # 第一层全连接
        x = F.relu(self.fc1(inputs_flat))
        
        # 重新形状为序列格式
        x = x.view(batch_size, seq_len, self.hidden_dim)
        
        # RNN层
        if hidden_state is None:
            rnn_out, new_hidden = self.rnn(x)
        else:
            rnn_out, new_hidden = self.rnn(x, hidden_state)
        
        # 展开RNN输出
        rnn_out_flat = rnn_out.view(-1, self.rnn_hidden_dim)
        
        # 输出层
        x = F.relu(self.fc2(rnn_out_flat))
        q_values = self.output(x)
        
        # 重新形状
        q_values = q_values.view(batch_size, seq_len, self.output_dim)
        
        if single_step:
            q_values = q_values.squeeze(1)  # [batch_size, output_dim]
        
        return q_values, new_hidden
    
    def init_hidden(self, batch_size: int = 1) -> torch.Tensor:
        """初始化隐藏状态"""
        return torch.zeros(1, batch_size, self.rnn_hidden_dim)


class QMIXMixer(nn.Module):
    """QMIX混合网络 - 将个体Q值混合为全局Q值"""
    
    def __init__(self, num_agents: int, state_dim: int, hidden_dim: int = 32):
        super(QMIXMixer, self).__init__()
        
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        
        # 权重生成网络 (超网络)
        self.hyper_w1 = nn.Linear(state_dim, num_agents * hidden_dim)
        self.hyper_w2 = nn.Linear(state_dim, hidden_dim)
        
        # 偏置生成网络
        self.hyper_b1 = nn.Linear(state_dim, hidden_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            agent_qs: [batch_size, num_agents] 个体Q值
            states: [batch_size, state_dim] 全局状态
        
        Returns:
            mixed_q: [batch_size, 1] 混合后的全局Q值
        """
        batch_size = agent_qs.shape[0]
        
        # 生成第一层权重和偏置
        w1 = torch.abs(self.hyper_w1(states))  # 确保权重为正 (单调性约束)
        w1 = w1.view(batch_size, self.num_agents, self.hidden_dim)
        
        b1 = self.hyper_b1(states)  # [batch_size, hidden_dim]
        
        # 第一层计算
        agent_qs = agent_qs.unsqueeze(-1)  # [batch_size, num_agents, 1]
        hidden = F.elu(torch.bmm(agent_qs.transpose(1, 2), w1).squeeze(1) + b1)
        
        # 生成第二层权重和偏置
        w2 = torch.abs(self.hyper_w2(states))  # [batch_size, hidden_dim]
        b2 = self.hyper_b2(states)  # [batch_size, 1]
        
        # 第二层计算
        mixed_q = torch.sum(hidden * w2, dim=1, keepdim=True) + b2
        
        return mixed_q


class QMIXReplayBuffer:
    """QMIX经验回放缓冲区"""
    
    def __init__(self, capacity: int, num_agents: int):
        self.capacity = capacity
        self.num_agents = num_agents
        self.buffer = deque(maxlen=capacity)
    
    def push(self, states: Dict[str, np.ndarray], actions: Dict[str, int],
             rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
             dones: Dict[str, bool], global_state: np.ndarray, next_global_state: np.ndarray):
        """添加经验到缓冲区"""
        experience = {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
            'global_state': global_state,
            'next_global_state': next_global_state
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
        
        batch_global_states = np.array([exp['global_state'] for exp in batch])
        batch_next_global_states = np.array([exp['next_global_state'] for exp in batch])
        
        return {
            'states': batch_states,
            'actions': batch_actions,
            'rewards': batch_rewards,
            'next_states': batch_next_states,
            'dones': batch_dones,
            'global_states': batch_global_states,
            'next_global_states': batch_next_global_states
        }
    
    def __len__(self):
        return len(self.buffer)


class QMIXEnvironment:
    """QMIX多智能体环境"""
    
    def __init__(self):
        self.config = QMIXConfig()
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('QMIX', self.config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        # 环境配置
        self.num_agents = 3  # vehicle_agent, rsu_agent, uav_agent
        self.state_dim = 20  # 单个智能体状态维度
        self.action_dim = 5   # 离散动作空间 (QMIX使用离散动作)
        self.global_state_dim = self.state_dim * self.num_agents
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建智能体网络
        self.agent_networks = {}
        self.target_agent_networks = {}
        
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        # 为兼容性添加agents属性 (指向agent_networks)
        self.agents = {}
        
        for agent_id in agent_ids:
            # 主网络
            self.agent_networks[agent_id] = QMIXAgent(
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                hidden_dim=self.config.hidden_dim,
                rnn_hidden_dim=self.config.rnn_hidden_dim
            ).to(self.device)
            
            # 目标网络
            self.target_agent_networks[agent_id] = QMIXAgent(
                input_dim=self.state_dim,
                output_dim=self.action_dim,
                hidden_dim=self.config.hidden_dim,
                rnn_hidden_dim=self.config.rnn_hidden_dim
            ).to(self.device)
            
            # 初始化目标网络
            self.target_agent_networks[agent_id].load_state_dict(
                self.agent_networks[agent_id].state_dict()
            )
            
            # 为兼容性，agents指向agent_networks
            self.agents[agent_id] = self.agent_networks[agent_id]
        
        # 混合网络
        self.mixer = QMIXMixer(
            num_agents=self.num_agents,
            state_dim=self.global_state_dim,
            hidden_dim=self.config.mixer_hidden_dim
        ).to(self.device)
        
        self.target_mixer = QMIXMixer(
            num_agents=self.num_agents,
            state_dim=self.global_state_dim,
            hidden_dim=self.config.mixer_hidden_dim
        ).to(self.device)
        
        # 初始化目标混合网络
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # 优化器
        all_parameters = []
        for network in self.agent_networks.values():
            all_parameters.extend(network.parameters())
        all_parameters.extend(self.mixer.parameters())
        
        self.optimizer = optim.RMSprop(
            all_parameters,
            lr=self.config.lr
        )
        
        # 经验回放缓冲区
        self.replay_buffer = QMIXReplayBuffer(
            capacity=self.config.buffer_size,
            num_agents=self.num_agents
        )
        
        # 训练统计
        self.episode_count = 0
        self.update_count = 0
        self.epsilon = self.config.epsilon
        
        # 隐藏状态
        self.hidden_states = {}
        for agent_id in agent_ids:
            self.hidden_states[agent_id] = self.agent_networks[agent_id].init_hidden().to(self.device)
        
        print("✓ QMIX环境初始化完成")
        print(f"✓ 智能体数量: {self.num_agents}")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
    
    def get_actions(self, states: Dict[str, np.ndarray], training: bool = True) -> Dict[str, int]:
        """获取所有智能体的动作 (离散动作)"""
        actions = {}
        
        for agent_id, agent_network in self.agent_networks.items():
            if agent_id in states:
                state_tensor = torch.FloatTensor(states[agent_id]).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    q_values, new_hidden = agent_network(state_tensor, self.hidden_states[agent_id])
                    self.hidden_states[agent_id] = new_hidden
                
                # ε-贪婪策略
                if training and random.random() < self.epsilon:
                    action = random.randint(0, self.action_dim - 1)
                else:
                    action = q_values.argmax().item()
                
                actions[agent_id] = action
            else:
                actions[agent_id] = 0  # 默认动作
        
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
        for agent_id in self.agent_networks.keys():
            # 基础状态 + 智能体特定状态
            agent_specific = np.random.randn(self.state_dim - len(base_state))
            states[agent_id] = np.concatenate([base_state, agent_specific])
        
        return states
    
    def get_global_state(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        """获取全局状态"""
        global_state = []
        for agent_id in sorted(states.keys()):
            global_state.append(states[agent_id])
        
        return np.concatenate(global_state)
    
    def train_step(self, states: Dict[str, np.ndarray], actions: Dict[str, int],
                   rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                   dones: Dict[str, bool]) -> Dict:
        """执行一步训练"""
        # 构建全局状态
        global_state = self.get_global_state(states)
        next_global_state = self.get_global_state(next_states)
        
        # 存储经验
        self.replay_buffer.push(states, actions, rewards, next_states, dones,
                               global_state, next_global_state)
        
        training_info = {}
        
        # 如果缓冲区有足够经验，开始训练
        if len(self.replay_buffer) >= self.config.batch_size:
            loss = self._update_networks()
            training_info['total_loss'] = loss
            
            # 更新目标网络
            if self.update_count % self.config.target_update_freq == 0:
                self._update_target_networks()
            
            # 衰减探索率
            self.epsilon = max(self.config.min_epsilon, 
                             self.epsilon * self.config.epsilon_decay)
            
            training_info['epsilon'] = self.epsilon
            self.update_count += 1
        
        return training_info
    
    def _update_networks(self) -> float:
        """更新网络参数"""
        # 采样经验批次
        batch = self.replay_buffer.sample(self.config.batch_size)
        
        # 转换为张量
        batch_size = len(batch['global_states'])
        
        global_states = torch.FloatTensor(batch['global_states']).to(self.device)
        next_global_states = torch.FloatTensor(batch['next_global_states']).to(self.device)
        
        # 计算当前Q值
        agent_qs = []
        agent_ids = sorted(batch['states'].keys())
        
        for agent_id in agent_ids:
            states = torch.FloatTensor(batch['states'][agent_id]).to(self.device)
            actions = torch.LongTensor(batch['actions'][agent_id]).to(self.device)
            
            q_values, _ = self.agent_networks[agent_id](states)
            chosen_q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            agent_qs.append(chosen_q_values)
        
        agent_qs = torch.stack(agent_qs, dim=1)  # [batch_size, num_agents]
        
        # 混合当前Q值
        current_mixed_q = self.mixer(agent_qs, global_states)
        
        # 计算目标Q值
        next_agent_qs = []
        
        for agent_id in agent_ids:
            next_states = torch.FloatTensor(batch['next_states'][agent_id]).to(self.device)
            next_q_values, _ = self.target_agent_networks[agent_id](next_states)
            max_next_q = next_q_values.max(1)[0]
            next_agent_qs.append(max_next_q)
        
        next_agent_qs = torch.stack(next_agent_qs, dim=1)
        
        # 混合目标Q值
        target_mixed_q = self.target_mixer(next_agent_qs, next_global_states)
        
        # 计算团队奖励 (简化：使用平均奖励)
        team_rewards = []
        for i in range(batch_size):
            team_reward = sum(batch['rewards'][agent_id][i] for agent_id in agent_ids) / len(agent_ids)
            team_rewards.append(team_reward)
        
        team_rewards = torch.FloatTensor(team_rewards).unsqueeze(1).to(self.device)
        
        # 计算done掩码 (团队done)
        team_dones = []
        for i in range(batch_size):
            team_done = any(batch['dones'][agent_id][i] for agent_id in agent_ids)
            team_dones.append(team_done)
        
        team_dones = torch.BoolTensor(team_dones).unsqueeze(1).to(self.device)
        
        # 计算目标值
        targets = team_rewards + (1 - team_dones.float()) * self.config.gamma * target_mixed_q
        
        # 计算损失
        loss = F.mse_loss(current_mixed_q, targets.detach())
        
        # 更新网络
        self.optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        total_norm = 0
        for p in self.agent_networks.values():
            param_norm = torch.nn.utils.clip_grad_norm_(p.parameters(), self.config.grad_clip)
            total_norm += param_norm
        
        torch.nn.utils.clip_grad_norm_(self.mixer.parameters(), self.config.grad_clip)
        
        self.optimizer.step()
        
        return loss.item()
    
    def _update_target_networks(self):
        """更新目标网络"""
        for agent_id in self.agent_networks.keys():
            self.target_agent_networks[agent_id].load_state_dict(
                self.agent_networks[agent_id].state_dict()
            )
        
        self.target_mixer.load_state_dict(self.mixer.state_dict())
    
    def reset_hidden_states(self):
        """重置隐藏状态 (每个episode开始时调用)"""
        for agent_id in self.agent_networks.keys():
            self.hidden_states[agent_id] = self.agent_networks[agent_id].init_hidden().to(self.device)
    
    def save_models(self, filepath: str):
        """保存所有模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        # 保存智能体网络
        for agent_id, network in self.agent_networks.items():
            torch.save(network.state_dict(), f"{filepath}/qmix_agent_{agent_id}.pth")
        
        # 保存混合网络
        torch.save(self.mixer.state_dict(), f"{filepath}/qmix_mixer.pth")
        
        # 保存训练状态
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'update_count': self.update_count
        }, f"{filepath}/qmix_training_state.pth")
        
        print(f"✓ QMIX模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载所有模型"""
        # 加载智能体网络
        for agent_id, network in self.agent_networks.items():
            network.load_state_dict(torch.load(f"{filepath}/qmix_agent_{agent_id}.pth", 
                                              map_location=self.device))
            self.target_agent_networks[agent_id].load_state_dict(network.state_dict())
        
        # 加载混合网络
        self.mixer.load_state_dict(torch.load(f"{filepath}/qmix_mixer.pth", 
                                             map_location=self.device))
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        
        # 加载训练状态
        training_state = torch.load(f"{filepath}/qmix_training_state.pth", 
                                   map_location=self.device)
        self.optimizer.load_state_dict(training_state['optimizer_state_dict'])
        self.epsilon = training_state['epsilon']
        self.update_count = training_state['update_count']
        
        print(f"✓ QMIX模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'epsilon': self.epsilon,
            'update_count': self.update_count,
            'buffer_size': len(self.replay_buffer)
        }
    
    def store_experience(self, states: Dict, actions: Dict, log_probs: Dict,
                        rewards: Dict, dones: Dict, global_state: Optional[np.ndarray] = None):
        """存储经验 (为QMIX不需要log_probs，但保持接口兼容性)"""
        # 对于QMIX，直接在train_step中存储经验
        pass
    
    def update(self):
        """更新所有智能体 (为QMIX的更新在train_step中执行)"""
        return {}