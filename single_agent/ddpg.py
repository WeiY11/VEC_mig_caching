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
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
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
    """🔧 DDPG算法配置 - 修复关键问题，对标TD3稳定性"""
    # 网络结构 - 对标TD3容量
    hidden_dim: int = 400      # 🔧 提升到400，与TD3一致
    actor_lr: float = 5e-5     # 🔧 与TD3一致的学习率
    critic_lr: float = 1e-4    # 保持critic学习率
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 100000  # 🔧 与TD3一致
    tau: float = 0.003         # 软更新系数
    gamma: float = 0.99
    
    # 探索参数
    noise_scale: float = 0.2   # 🔧 与TD3的exploration_noise一致
    noise_decay: float = 0.9998 # 🔧 与TD3一致的衰减率
    min_noise: float = 0.05
    
    # 🔧 关键修复：引入策略延迟更新（借鉴TD3）
    policy_delay: int = 4      # 🔧 每4次更新一次Actor，与TD3一致
    update_freq: int = 1       # 改回每步都采样
    warmup_steps: int = 1000   # 🔧 与TD3一致
    
    # 🔧 新增：目标策略平滑化（借鉴TD3）
    target_noise: float = 0.05  # 目标动作噪声
    noise_clip: float = 0.2     # 噪声裁剪范围
    
    # PER参数
    use_per: bool = True
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 500000
    
    # 梯度处理
    gradient_clip: float = 1.0  # 🔧 放宽到1.0，与TD3一致
    reward_scale: float = 1.0
    reward_normalize: bool = False


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
    """🔧 DDPG优先经验回放缓冲区 (支持PER)"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6, use_per: bool = True):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.use_per = use_per
        self.alpha = alpha  # PER优先级指数
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        # 🔧 PER优先级数组
        if self.use_per:
            self.priorities = np.zeros(capacity, dtype=np.float32)
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, 
             next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        # 🔧 新样本使用最大优先级
        if self.use_per:
            max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
            self.priorities[self.ptr] = max_prio
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[torch.Tensor, ...]:
        """🔧 采样经验批次（支持PER）"""
        if self.use_per:
            # PER采样
            prios = self.priorities[:self.size]
            probs = prios ** self.alpha
            probs /= probs.sum()
            
            indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
            
            # 计算重要性权重
            weights = (self.size * probs[indices]) ** (-beta)
            weights /= weights.max()
            weights = torch.FloatTensor(weights).unsqueeze(1)
        else:
            # 均匀采样
            indices = np.random.choice(self.size, batch_size, replace=False)
            weights = torch.ones(batch_size, 1)
        
        batch_states = torch.FloatTensor(self.states[indices])
        batch_actions = torch.FloatTensor(self.actions[indices])
        batch_rewards = torch.FloatTensor(self.rewards[indices]).unsqueeze(1)
        batch_next_states = torch.FloatTensor(self.next_states[indices])
        batch_dones = torch.FloatTensor(self.dones[indices]).unsqueeze(1)
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """🔧 更新样本优先级"""
        if self.use_per:
            self.priorities[indices] = priorities
    
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
        
        # 🔧 优化的经验回放缓冲区（支持PER）
        self.replay_buffer = DDPGReplayBuffer(
            config.buffer_size, state_dim, action_dim, 
            alpha=config.per_alpha if hasattr(config, 'per_alpha') else 0.6,
            use_per=config.use_per if hasattr(config, 'use_per') else False
        )
        
        # 🔧 PER beta参数
        self.beta = config.per_beta_start if hasattr(config, 'per_beta_start') else 0.4
        self.beta_increment = (1.0 - self.beta) / (config.per_beta_frames if hasattr(config, 'per_beta_frames') else 500000)
        
        # 探索噪声
        self.noise_scale = config.noise_scale
        self.step_count = 0
        self.update_count = 0  # 🔧 添加更新计数
        
        # 🔧 奖励统计（用于归一化）
        self.reward_stats = {'mean': 0.0, 'std': 1.0}
        self.reward_history = deque(maxlen=10000)
        
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
        """🔧 优化的更新网络参数 - 引入策略延迟更新（借鉴TD3）"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        self.step_count += 1
        
        # 预热期不更新
        if self.step_count < self.config.warmup_steps:
            return {}
        
        # 🔧 更新频率控制
        if self.step_count % self.config.update_freq != 0:
            return {}
        
        self.update_count += 1
        
        # 🔧 采样经验批次（支持PER）
        sample_result = self.replay_buffer.sample(self.config.batch_size, self.beta)
        if len(sample_result) == 7:  # PER模式
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones, indices, weights = sample_result
        else:  # 普通模式（兼容）
            batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = sample_result[:5]
            indices = np.arange(len(batch_states))
            weights = torch.ones(len(batch_states), 1)
        
        # 🔧 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch_states = batch_states.to(self.device)
        batch_actions = batch_actions.to(self.device)
        batch_rewards = batch_rewards.to(self.device)
        batch_next_states = batch_next_states.to(self.device)
        batch_dones = batch_dones.to(self.device)
        weights = weights.to(self.device)
        
        # 🔧 奖励归一化（可选）
        if self.config.reward_normalize if hasattr(self.config, 'reward_normalize') else False:
            self.reward_history.extend(batch_rewards.cpu().numpy().flatten())
            if len(self.reward_history) > 100:
                self.reward_stats['mean'] = np.mean(self.reward_history)
                self.reward_stats['std'] = np.std(self.reward_history) + 1e-8
                batch_rewards = (batch_rewards - self.reward_stats['mean']) / self.reward_stats['std']
        
        # 🔧 更新Critic并获取TD误差
        critic_loss, td_errors = self._update_critic(
            batch_states, batch_actions, batch_rewards, 
            batch_next_states, batch_dones, weights
        )
        
        # 🔧 更新PER优先级
        if self.config.use_per if hasattr(self.config, 'use_per') else False:
            priorities = td_errors.detach().cpu().numpy() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        training_info = {'critic_loss': critic_loss}
        
        # 🔧 策略延迟更新（借鉴TD3）- 每policy_delay次更新一次Actor
        if self.update_count % self.config.policy_delay == 0:
            actor_loss = self._update_actor(batch_states)
            training_info['actor_loss'] = actor_loss
            
            # 软更新目标网络（只在更新Actor时更新）
            self.soft_update(self.target_actor, self.actor, self.config.tau)
            self.soft_update(self.target_critic, self.critic, self.config.tau)
        
        # 🔧 优化的噪声衰减
        self.noise_scale = max(self.config.min_noise, self.noise_scale * self.config.noise_decay)
        training_info['noise_scale'] = self.noise_scale
        training_info['buffer_size'] = len(self.replay_buffer)
        training_info['beta'] = self.beta
        training_info['update_count'] = self.update_count
        
        return training_info
    
    def _update_critic(self, states: torch.Tensor, actions: torch.Tensor, 
                      rewards: torch.Tensor, next_states: torch.Tensor, 
                      dones: torch.Tensor, weights: torch.Tensor) -> Tuple[float, torch.Tensor]:
        """🔧 优化的Critic更新 - 添加目标策略平滑化（借鉴TD3）"""
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            
            # 🔧 目标策略平滑化：添加裁剪噪声（借鉴TD3）
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q = self.critic(states, actions)
        
        # 🔧 计算TD误差（用于PER）
        td_errors = torch.abs(current_q - target_q)
        
        # 🔧 加权MSE损失
        critic_loss = (weights * F.mse_loss(current_q, target_q, reduction='none')).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 🔧 使用配置的梯度裁剪
        grad_clip = self.config.gradient_clip if hasattr(self.config, 'gradient_clip') else 1.0
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), grad_clip)
        
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        return critic_loss.item(), td_errors.squeeze()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """🔧 优化的Actor更新"""
        actions = self.actor(states)
        actor_loss = -self.critic(states, actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        
        # 🔧 使用配置的梯度裁剪
        grad_clip = self.config.gradient_clip if hasattr(self.config, 'gradient_clip') else 1.0
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), grad_clip)
        
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
        
        # 🔧 修复：正确计算状态维度，与TD3保持一致
        self.state_dim = 130  # 车辆60 + RSU54 + UAV16 = 130维
        self.action_dim = 18  # 支持自适应缓存迁移控制，与TD3保持一致
        
        # 创建智能体
        self.agent = DDPGAgent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"✓ DDPG环境初始化完成 (已优化)")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
        print(f"✓ 策略延迟更新: {self.config.policy_delay} (借鉴TD3)")
        print(f"✓ 目标策略平滑化: 已启用 (target_noise={self.config.target_noise})")
        print(f"✓ 网络容量: hidden_dim={self.config.hidden_dim}")
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """
        🔧 修复：构建准确的130维状态向量，与TD3完全一致
        状态组成: 车辆60维 + RSU54维 + UAV16维 = 130维
        """
        state_components = []
        
        # 1. 车辆状态 (12×5=60维) - 与TD3一致
        for i in range(12):
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key]
                # 确保数值有效性
                valid_state = []
                for val in vehicle_state[:5]:
                    if np.isfinite(val):
                        valid_state.append(float(val))
                    else:
                        valid_state.append(0.5)
                state_components.extend(valid_state)
                
                # 补齐到5维
                while len(state_components) % 5 != 0:
                    state_components.append(0.0)
            else:
                # 默认车辆状态: [位置x, 位置y, 速度, 队列, 能耗]
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        # 2. RSU状态 (6×9=54维) - 与TD3一致
        for i in range(6):
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key]
                # 确保数值有效性和维度正确
                valid_rsu_state = []
                for j, val in enumerate(rsu_state[:9]):
                    if np.isfinite(val):
                        # 对缓存利用率(第2维)进行特殊检查
                        if j == 2:  # 缓存利用率维度
                            valid_rsu_state.append(min(1.0, max(0.0, float(val))))
                        else:
                            valid_rsu_state.append(float(val))
                    else:
                        valid_rsu_state.append(0.5 if j < 2 else 0.0)
                
                # 补齐到9维
                while len(valid_rsu_state) < 9:
                    valid_rsu_state.append(0.0)
                
                state_components.extend(valid_rsu_state)
            else:
                # 默认RSU状态: [位置x, 位置y, 缓存利用率, 队列, 能耗, 缓存参数4维]
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0, 0.7, 0.35, 0.05, 0.3])
        
        # 3. UAV状态 (2×8=16维) - 与TD3一致
        for i in range(2):
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key]
                # 确保数值有效性
                valid_uav_state = []
                for j, val in enumerate(uav_state[:8]):
                    if np.isfinite(val):
                        # 对缓存利用率(第3维)进行特殊处理
                        if j == 3:  # 缓存利用率维度
                            valid_uav_state.append(min(1.0, max(0.0, float(val))))
                        else:
                            valid_uav_state.append(float(val))
                    else:
                        valid_uav_state.append(0.5 if j < 3 else 0.0)
                
                # 补齐到8维
                while len(valid_uav_state) < 8:
                    valid_uav_state.append(0.0)
                
                state_components.extend(valid_uav_state)
            else:
                # 默认UAV状态: [位置x, 位置y, 位置z, 缓存利用率, 能耗, 迁移参数3维]
                state_components.extend([0.5, 0.5, 0.5, 0.0, 0.0, 0.75, 1.0, 0.3])
        
        # 确保状态向量正好是130维
        state_vector = np.array(state_components[:130], dtype=np.float32)
        
        # 如果维度不足130，补齐
        if len(state_vector) < 130:
            padding_needed = 130 - len(state_vector)
            state_vector = np.pad(state_vector, (0, padding_needed), mode='constant', constant_values=0.5)
        
        # 数值安全检查
        state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        
        return state_vector
    
    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        将全局动作分解为各节点动作
        🤖 更新支持18维动作空间，与TD3保持一致：
        - vehicle_agent: 18维 (11维原有 + 7维缓存迁移控制)
        """
        actions = {}
        
        # 确保action长度足够
        if len(action) < 18:
            action = np.pad(action, (0, 18-len(action)), mode='constant')
        
        # 🤖 vehicle_agent 获得所有18维动作
        # 前11维：任务分配(3) + RSU选择(6) + UAV选择(2)
        # 后7维：缓存控制(4) + 迁移控制(3)
        actions['vehicle_agent'] = action[:18]
        
        # 🔧 关键修复：从vehicle_agent中提取RSU和UAV选择
        # 训练框架需要从rsu_agent和uav_agent获取选择概率
        actions['rsu_agent'] = action[3:9]   # RSU选择（6维）
        actions['uav_agent'] = action[9:11]  # UAV选择（2维）
        
        return actions
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        return self.decompose_action(global_action)
    
    def calculate_reward(self, system_metrics: Dict, 
                       cache_metrics: Optional[Dict] = None,
                       migration_metrics: Optional[Dict] = None) -> float:
        """
        🔧 修复：使用增强奖励计算器，与TD3保持一致
        支持缓存和迁移子系统的综合奖励计算
        """
        try:
            from utils.enhanced_reward_calculator import calculate_enhanced_reward
            return calculate_enhanced_reward(system_metrics, cache_metrics, migration_metrics)
        except ImportError:
            # 回退到简单奖励
            from utils.simple_reward_calculator import calculate_simple_reward
            return calculate_simple_reward(system_metrics)
    
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