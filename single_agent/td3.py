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
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class TD3Config:
    """TD3算法配置 - 🎯 优化版v2.0（减少收敛后振荡）"""
    # 网络结构
    hidden_dim: int = 400  
    actor_lr: float = 1e-4  # 🔧 提高Actor学习率，增强策略更新力度
    critic_lr: float = 8e-5  # 🔧 适度提高Critic学习率，追踪更精确
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000
    tau: float = 0.005  # 🔧 回调至稳定值，平衡目标网络跟随速度
    gamma: float = 0.99  
    
    # TD3特有参数
    policy_delay: int = 2  # 🔧 缩短策略延迟，减少策略落后现象
    target_noise: float = 0.05
    noise_clip: float = 0.2
    
    # 探索参数
    exploration_noise: float = 0.2
    noise_decay: float = 0.9997  # 🔧 放慢噪声衰减，避免后期探索不足
    min_noise: float = 0.05  # 🔧 提高最小噪声，保持长期探索
    
    # 🔧 新增：梯度裁剪防止过拟合
    gradient_clip_norm: float = 0.7  # 🔧 放宽梯度裁剪，允许适度更新
    use_gradient_clip: bool = True   # 启用梯度裁剪
    
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
    late_stage_start_updates: int = 90000  # 🔧 约等于800轮更新步
    late_stage_tau: float = 0.003
    late_stage_policy_delay: int = 3
    late_stage_noise_floor: float = 0.03
    td_error_clip: float = 4.0
    
    # 训练频率
    update_freq: int = 1
    warmup_steps: int = 1000


class TD3Actor(nn.Module):
    """TD3 Actor网络 - 确定性策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: float = 1.0):
        super(TD3Actor, self).__init__()
        
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
    
    def __init__(self, state_dim: int, action_dim: int, config: TD3Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('TD3', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = TD3Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = TD3Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 目标网络
        self.target_actor = TD3Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
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
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
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
        # TD误差裁剪，防止极端值主导PER
        if self.config.td_error_clip is not None:
            td_errors = td_errors.clamp(-self.config.td_error_clip, self.config.td_error_clip)
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
    
    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2):
        self.config = TD3Config()
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
        # 🔧 优化后的状态维度：所有节点统一为5维 + 全局状态8维
        # 车辆状态: N×5维 + RSU状态: M×5维 + UAV状态: K×5维 + 全局: 8维
        self.local_state_dim = num_vehicles * 5 + num_rsus * 5 + num_uavs * 5
        self.global_state_dim = 8
        self.state_dim = self.local_state_dim + self.global_state_dim
        
        # 🔧 优化后的动作空间：动态适配网络拓扑
        # 3(任务分配) + num_rsus(RSU选择) + num_uavs(UAV选择) + 7(控制参数)
        self.action_dim = 3 + num_rsus + num_uavs + 7
        
        # 创建智能体
        self.agent = TD3Agent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"TD3环境初始化完成（优化版）")
        print(f"网络拓扑: {num_vehicles}辆车 + {num_rsus}个RSU + {num_uavs}个UAV")
        print(f"状态维度: {self.state_dim} = 局部{self.local_state_dim} ({num_vehicles}×5 + {num_rsus}×5 + {num_uavs}×5) + 全局{self.global_state_dim}")
        print(f"动作维度: {self.action_dim} (动态适配: 3+{num_rsus}+{num_uavs}+7)")
        print(f"策略延迟更新: {self.config.policy_delay}")
        print(f"优化特性: 移除控制参数冗余, 添加全局状态, 统一归一化")
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
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
        
        # ========== 2. 全局系统状态 (8维) ==========
        global_state = self._build_global_state(node_states, system_metrics)
        state_components.extend(global_state)
        
        # ========== 3. 最终处理 ==========
        state_vector = np.array(state_components[:self.state_dim], dtype=np.float32)
        
        # 维度不足时补齐
        if len(state_vector) < self.state_dim:
            padding_needed = self.state_dim - len(state_vector)
            state_vector = np.pad(state_vector, (0, padding_needed), mode='constant', constant_values=0.5)
        
        # 数值安全检查
        state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        state_vector = np.clip(state_vector, 0.0, 1.0)  # 确保所有值在[0,1]
        
        return state_vector
    
    def _build_global_state(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """
        构建全局系统状态（8维）
        提供系统级别的整体信息，辅助智能体进行全局协调决策
        """
        # 收集所有节点的队列信息（从局部状态中提取）
        all_queues = []
        for i in range(self.num_vehicles):
            v_state = node_states.get(f'vehicle_{i}')
            if v_state is not None and len(v_state) > 3:
                all_queues.append(v_state[3])  # 队列维度
        for i in range(self.num_rsus):
            r_state = node_states.get(f'rsu_{i}')
            if r_state is not None and len(r_state) > 3:
                all_queues.append(r_state[3])
        
        # 计算全局指标
        avg_queue = np.mean(all_queues) if all_queues else 0.0
        congestion_ratio = len([q for q in all_queues if q > 0.5]) / max(1, len(all_queues))
        
        # 从system_metrics获取系统级指标
        completion_rate = system_metrics.get('task_completion_rate', 0.5)
        avg_energy = system_metrics.get('total_energy_consumption', 0.0) / max(1, self.num_vehicles + self.num_rsus + self.num_uavs)
        cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
        
        # 构建全局状态向量
        global_state = np.array([
            np.clip(avg_queue, 0.0, 1.0),           # 平均队列占用率
            np.clip(congestion_ratio, 0.0, 1.0),    # 拥塞节点比例
            np.clip(completion_rate, 0.0, 1.0),     # 任务完成率
            np.clip(avg_energy / 1000.0, 0.0, 1.0), # 平均能耗
            np.clip(cache_hit_rate, 0.0, 1.0),      # 缓存命中率
            0.0,  # episode进度（需要从外部传入）
            np.clip(len([q for q in all_queues if q > 0]) / max(1, len(all_queues)), 0.0, 1.0),  # 活跃节点比例
            np.clip(sum(all_queues) / max(1, len(all_queues)), 0.0, 1.0)  # 网络总负载
        ], dtype=np.float32)
        
        return global_state
    
    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        🔧 优化版动作分解：动态适配网络拓扑
        动作空间：3(分配) + num_rsus(RSU选择) + num_uavs(UAV选择) + 7(控制)
        """
        actions = {}
        
        # 确保action长度足够
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)), mode='constant')
        
        # 动态分解动作
        idx = 0
        
        # 1. 任务分配偏好（3维）
        task_allocation = action[idx:idx+3]
        idx += 3
        
        # 2. RSU选择权重（num_rsus维）
        rsu_selection = action[idx:idx+self.num_rsus]
        idx += self.num_rsus
        
        # 3. UAV选择权重（num_uavs维）
        uav_selection = action[idx:idx+self.num_uavs]
        idx += self.num_uavs
        
        # 4. 控制参数（7维）
        control_params = action[idx:idx+7]
        
        # 构建vehicle_agent的完整动作（用于仿真器）
        actions['vehicle_agent'] = np.concatenate([
            task_allocation,   # 3维
            rsu_selection,     # num_rsus维
            uav_selection,     # num_uavs维
            control_params     # 7维
        ])
        
        # RSU和UAV agent的动作（用于选择概率计算）
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        
        return actions
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        return self.decompose_action(global_action)
    
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