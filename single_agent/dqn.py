"""
DQN (Deep Q-Network) 单智能体算法实现
专门适配MATD3-MIG系统的VEC环境

主要特点:
1. 深度Q网络处理离散动作空间
2. 经验回放机制提高样本效率
3. 目标网络稳定训练过程
4. ε-贪婪探索策略

对应论文: Human-level control through deep reinforcement learning
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'DQN': 32}  # 默认值

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
class DQNConfig:
    """DQN算法配置"""
    # 网络结构
    hidden_dim: int = 256
    lr: float = 1e-4
    
    # 训练参数 - 🔧 修复：增加批次大小解决梯度估计问题
    batch_size: int = 128  # 从32增加到128，解决诊断发现的批次过小问题
    buffer_size: int = 50000
    target_update_freq: int = 1000
    gamma: float = 0.99
    
    # 探索参数
    epsilon: float = 1.0
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05
    
    # 训练频率
    update_freq: int = 4
    warmup_steps: int = 1000
    
    # DQN变种选择
    double_dqn: bool = True
    dueling_dqn: bool = True


class DQNNetwork(nn.Module):
    """DQN网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, dueling: bool = True):
        super(DQNNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.dueling = dueling
        
        # 共享特征层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if dueling:
            # Dueling DQN架构
            # 价值流
            self.value_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # 优势流
            self.advantage_stream = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, action_dim)
            )
        else:
            # 标准DQN架构
            self.q_network = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        def init_layer(layer):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.constant_(layer.bias, 0.0)
        
        self.feature_layers.apply(init_layer)
        
        if self.dueling:
            self.value_stream.apply(init_layer)
            self.advantage_stream.apply(init_layer)
        else:
            self.q_network.apply(init_layer)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        features = self.feature_layers(state)
        
        if self.dueling:
            # Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
            values = self.value_stream(features)
            advantages = self.advantage_stream(features)
            
            # 减去优势的均值以确保可识别性
            q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
            
            return q_values
        else:
            # 标准DQN
            return self.q_network(features)


class DQNReplayBuffer:
    """DQN经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
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
        batch_actions = torch.LongTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices])
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices])
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def __len__(self):
        return self.size


class DQNAgent:
    """DQN智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: DQNConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('DQN', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.q_network = DQNNetwork(
            state_dim, action_dim, config.hidden_dim, config.dueling_dqn
        ).to(self.device)
        
        self.target_q_network = DQNNetwork(
            state_dim, action_dim, config.hidden_dim, config.dueling_dqn
        ).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_q_network, self.q_network)
        
        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=config.lr)
        
        # 经验回放缓冲区
        self.replay_buffer = DQNReplayBuffer(config.buffer_size, state_dim)
        
        # 探索参数
        self.epsilon = config.epsilon
        self.step_count = 0
        self.update_count = 0
        
        # 训练统计
        self.losses = []
        self.q_values = []
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """选择动作 - ε-贪婪策略"""
        if training and random.random() < self.epsilon:
            # 随机探索
            return random.randrange(self.action_dim)
        else:
            # 贪婪选择
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                self.q_values.append(q_values.max().item())
                return q_values.argmax().item()
    
    def store_experience(self, state: np.ndarray, action: int, reward: float,
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
        
        # 更新频率控制
        if self.step_count % self.config.update_freq != 0:
            return {}
        
        self.update_count += 1
        
        # 采样经验批次
        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = \
            self.replay_buffer.sample(self.config.batch_size)
        
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        
        # 计算损失并更新
        loss = self._compute_loss(batch_states, batch_actions, batch_rewards, 
                                batch_next_states, batch_dones)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # 更新目标网络
        if self.update_count % self.config.target_update_freq == 0:
            self.hard_update(self.target_q_network, self.q_network)
        
        # 衰减探索率
        self.epsilon = max(self.config.min_epsilon, 
                          self.epsilon * self.config.epsilon_decay)
        
        self.losses.append(loss.item())
        
        return {
            'loss': loss.item(),
            'epsilon': self.epsilon,
            'q_value_avg': float(np.mean(self.q_values[-100:])) if self.q_values else 0.0
        }
    
    def _compute_loss(self, states: torch.Tensor, actions: torch.Tensor, 
                     rewards: torch.Tensor, next_states: torch.Tensor, 
                     dones: torch.Tensor) -> torch.Tensor:
        """计算DQN损失"""
        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # 计算目标Q值
        with torch.no_grad():
            if self.config.double_dqn:
                # Double DQN
                next_actions = self.q_network(next_states).argmax(dim=1)
                next_q_values = self.target_q_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # 标准DQN
                next_q_values = self.target_q_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + (1 - dones) * self.config.gamma * next_q_values
        
        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)
        
        return loss
    
    def hard_update(self, target: nn.Module, source: nn.Module):
        """硬更新网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_q_network_state_dict': self.target_q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'update_count': self.update_count
        }, f"{filepath}_dqn.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_dqn.pth", map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']


class DQNEnvironment:
    """DQN训练环境 - 优化版"""
    
    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2):
        from single_agent.common_state_action import UnifiedStateActionSpace
        
        self.config = DQNConfig()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
        # 🔧 使用统一的状态维度计算
        self.local_state_dim, self.global_state_dim, self.state_dim = \
            UnifiedStateActionSpace.calculate_state_dim(num_vehicles, num_rsus, num_uavs)
        
        # DQN使用离散动作：简化为125个组合（5^3）
        self.action_dim = 125
        
        # 创建智能体
        self.agent = DQNAgent(self.state_dim, self.action_dim, self.config)
        
        # 动作映射
        self.action_map = self._build_action_map()
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"DQN环境初始化完成（优化版v2.0）")
        print(f"网络拓扑: {num_vehicles}辆车 + {num_rsus}个RSU + {num_uavs}个UAV")
        print(f"状态维度: {self.state_dim} = 局部{self.local_state_dim} + 全局{self.global_state_dim}")
        print(f"动作维度: {self.action_dim} (离散)")
        print(f"Double DQN: {self.config.double_dqn}")
        print(f"Dueling DQN: {self.config.dueling_dqn}")
        print(f"优化特性: 统一状态空间, 动态拓扑适配, 全局状态")
    
    def _build_action_map(self) -> Dict[int, Dict[str, int]]:
        """构建离散动作映射"""
        action_map = {}
        action_idx = 0
        
        # 为每个节点类型的5个动作选择创建组合
        for vehicle_action in range(5):
            for rsu_action in range(5):
                for uav_action in range(5):
                    action_map[action_idx] = {
                        'vehicle_agent': vehicle_action,
                        'rsu_agent': rsu_action,
                        'uav_agent': uav_action
                    }
                    action_idx += 1
        
        return action_map
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """使用统一方法构建状态向量"""
        from single_agent.common_state_action import UnifiedStateActionSpace
        return UnifiedStateActionSpace.build_state_vector(
            node_states,
            system_metrics,
            self.num_vehicles,
            self.num_rsus,
            self.num_uavs,
            self.state_dim
        )
    
    def decompose_action(self, action_idx: int) -> Dict[str, int]:
        """将离散动作索引转换为各节点动作"""
        return self.action_map[action_idx]
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, int]:
        """获取动作"""
        discrete_action = self.agent.select_action(state, training)
        return self.decompose_action(discrete_action)
    
    def calculate_reward(self, system_metrics: Dict,
                       cache_metrics: Optional[Dict] = None,
                       migration_metrics: Optional[Dict] = None) -> float:
        """
        使用统一奖励计算器
        """
        from utils.unified_reward_calculator import calculate_unified_reward
        return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, algorithm="general")
    
    def train_step(self, state: np.ndarray, action: int, reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """执行一步训练"""
        # DQN需要整数动作，如果是numpy数组则转换
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action_int = int(action.item())
            else:
                action_int = int(action[0])
        else:
            action_int = int(action)
        
        # 存储经验
        self.agent.store_experience(state, action_int, reward, next_state, done)
        
        # 更新网络
        training_info = self.agent.update()
        
        self.step_count += 1
        
        return training_info
    
    def save_models(self, filepath: str):
        """保存模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        self.agent.save_model(filepath)
        print(f"✓ DQN模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ DQN模型已加载: {filepath}")
    
    def store_experience(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        """存储经验到缓冲区 - 支持PPO兼容性"""
        # DQN需要整数动作，如果是numpy数组则转换
        if isinstance(action, np.ndarray):
            if action.size == 1:
                action_int = int(action.item())
            else:
                action_int = int(action[0])
        else:
            action_int = int(action)
        
        # DQN只使用前5个参数，log_prob和value被忽略
        self.agent.store_experience(state, action_int, reward, next_state, done)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict:
        """更新网络参数 - 支持PPO兼容性"""
        # DQN不使用last_value参数
        return self.agent.update()
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'loss_avg': float(np.mean(self.agent.losses[-100:])) if self.agent.losses else 0.0,
            'q_value_avg': float(np.mean(self.agent.q_values[-100:])) if self.agent.q_values else 0.0,
            'epsilon': self.agent.epsilon,
            'buffer_size': len(self.agent.replay_buffer),
            'step_count': self.step_count,
            'update_count': self.agent.update_count,
            'double_dqn': self.config.double_dqn,
            'dueling_dqn': self.config.dueling_dqn
        }