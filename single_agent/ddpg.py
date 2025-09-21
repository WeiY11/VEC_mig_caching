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
    """DDPG算法配置 - 优化收敛性（根据诊断结果调整）"""
    # 网络结构 - 增加容量提高表现力
    hidden_dim: int = 256      # 增加网络容量（从128到256）
    actor_lr: float = 1e-4     # 降低actor学习率提高稳定性
    critic_lr: float = 3e-4    # 提高critic学习率加快学习
    
    # 训练参数 - 优化收敛性
    batch_size: int = 128      # 增加批次大小（从64到128）
    buffer_size: int = 100000  # 增加缓冲区大小
    tau: float = 0.005         # 减小软更新系数（从0.01到0.005）
    gamma: float = 0.99        # 提高折扣因子（从0.95到0.99）
    
    # 探索参数 - 加强探索
    noise_scale: float = 0.3   # 增加初始探索（从0.2到0.3）
    noise_decay: float = 0.9999 # 更慢的噪声衰减（从0.995到0.9999）
    min_noise: float = 0.1     # 提高最小探索（从0.05到0.1）
    
    # 训练频率
    update_freq: int = 1
    warmup_steps: int = 1000   # 增加预热步数（从500到1000）


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
        """构建全局状态向量 - 修复状态表示问题"""
        state_components = []
        
        # 1. 基础系统状态 (8维) - 增加更多动态特征
        base_state = [
            system_metrics.get('avg_task_delay', 0.0) / 1.0,
            system_metrics.get('total_energy_consumption', 0.0) / 1000.0,
            system_metrics.get('data_loss_rate', 0.0),
            system_metrics.get('cache_hit_rate', 0.0),
            system_metrics.get('migration_success_rate', 0.0),
            # 🔧 修复：添加变化性更强的系统特征
            system_metrics.get('task_completion_rate', 0.0),  # 任务完成率
            min(1.0, system_metrics.get('avg_task_delay', 0.15) / 0.5),  # 延迟负载指标
            min(1.0, system_metrics.get('total_energy_consumption', 600.0) / 1500.0),  # 能耗负载指标
        ]
        state_components.extend(base_state)
        
        # 2. 车辆状态 (12车辆 × 4维 = 48维) - 使用真实状态而非随机数
        vehicle_count = 0
        for i in range(12):  # 支持最多12个车辆
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key]
                # 提取车辆的关键状态特征
                if len(vehicle_state) >= 5:
                    vehicle_features = [
                        float(vehicle_state[0]),  # 位置x (已归一化)
                        float(vehicle_state[1]),  # 位置y (已归一化)  
                        float(vehicle_state[2]),  # 速度 (已归一化)
                        float(vehicle_state[3]),  # 任务数 (已归一化)
                    ]
                else:
                    # 如果状态维度不足，使用默认值
                    vehicle_features = [0.5, 0.5, 0.5, 0.0]
                vehicle_count += 1
            else:
                # 车辆不存在，使用默认状态
                vehicle_features = [0.0, 0.0, 0.0, 0.0]
            
            state_components.extend(vehicle_features)
        
        # 3. RSU状态 (6个RSU × 3维 = 18维)  
        for i in range(6):  # 支持最多6个RSU
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key]
                if len(rsu_state) >= 5:
                    rsu_features = [
                        float(rsu_state[2]),  # 缓存利用率
                        float(rsu_state[3]),  # 队列长度 (已归一化)
                        float(rsu_state[4]),  # 能耗 (已归一化)
                    ]
                else:
                    rsu_features = [0.5, 0.5, 0.5]
            else:
                rsu_features = [0.0, 0.0, 0.0]
            
            state_components.extend(rsu_features)
        
        # 4. UAV状态 (2个UAV × 4维 = 8维)
        for i in range(2):  # 支持最多2个UAV
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key]
                if len(uav_state) >= 5:
                    uav_features = [
                        float(uav_state[2]),  # 高度 (已归一化)
                        float(uav_state[3]),  # 缓存利用率
                        float(uav_state[4]),  # 能耗 (已归一化)
                        1.0,  # 电池电量 (简化为固定值)
                    ]
                else:
                    uav_features = [0.8, 0.5, 0.5, 1.0]
            else:
                uav_features = [0.0, 0.0, 0.0, 0.5]
            
            state_components.extend(uav_features)
        
        # 🔧 修复：确保状态向量长度为60维，用有意义的特征填充
        current_length = len(state_components)
        if current_length < self.state_dim:
            padding_size = self.state_dim - current_length
            
            # 添加有意义的派生特征而非周期性填充
            for i in range(padding_size):
                if i < 4:  # 系统负载分布特征
                    load_factor = system_metrics.get('total_energy_consumption', 600.0) / 1000.0
                    feature_val = 0.3 + 0.4 * np.sin(load_factor * np.pi + i)
                elif i < 8:  # 延迟分布特征
                    delay_factor = system_metrics.get('avg_task_delay', 0.15)
                    feature_val = 0.4 + 0.3 * np.cos(delay_factor * 10 + i)
                elif i < 12:  # 完成率相关特征
                    completion_factor = system_metrics.get('task_completion_rate', 0.9)
                    feature_val = completion_factor * (0.5 + 0.3 * np.sin(i * 0.5))
                else:  # 缓存效率特征
                    cache_factor = system_metrics.get('cache_hit_rate', 0.3)
                    feature_val = cache_factor * (0.6 + 0.2 * np.cos(i * 0.7))
                
                # 确保特征值在合理范围内
                feature_val = np.clip(feature_val, 0.0, 1.0)
                state_components.append(float(feature_val))
        elif current_length > self.state_dim:
            # 如果维度过多，截断
            state_components = state_components[:self.state_dim]
        
        # 转换为numpy数组并进行数值稳定性检查
        state_vector = np.array(state_components, dtype=np.float32)
        
        # 检查并处理NaN/Inf值
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            print(f"⚠️ 警告: 状态向量包含无效值，进行修复")
            state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 确保状态值在合理范围内
        state_vector = np.clip(state_vector, -5.0, 5.0)
        
        return state_vector
    
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
        """计算奖励 - 修复版本，解决相关性和单调性问题"""
        # 提取指标并进行数值稳定性检查
        delay = float(system_metrics.get('avg_task_delay', 0.15))
        energy = float(system_metrics.get('total_energy_consumption', 600.0)) / 1000.0  # 归一化
        loss_rate = float(system_metrics.get('data_loss_rate', 0.05))
        completion_rate = float(system_metrics.get('task_completion_rate', 0.9))
        cache_hit_rate = float(system_metrics.get('cache_hit_rate', 0.3))
        
        # 数值安全检查和约束
        delay = np.clip(delay, 0.01, 2.0) if np.isfinite(delay) else 0.15
        energy = np.clip(energy, 0.1, 3.0) if np.isfinite(energy) else 0.6
        loss_rate = np.clip(loss_rate, 0.0, 1.0) if np.isfinite(loss_rate) else 0.05
        completion_rate = np.clip(completion_rate, 0.0, 1.0) if np.isfinite(completion_rate) else 0.9
        cache_hit_rate = np.clip(cache_hit_rate, 0.0, 1.0) if np.isfinite(cache_hit_rate) else 0.3
        
        # 🔧 修复：强化奖励函数，确保强相关性和单调性
        # 1. 强化惩罚项 - 确保与优化目标强负相关
        delay_penalty = -15.0 * delay        # 强化延迟惩罚，确保负相关
        energy_penalty = -8.0 * energy       # 强化能耗惩罚
        loss_penalty = -25.0 * loss_rate     # 强化丢失率惩罚
        
        # 2. 强化奖励项 - 确保与性能指标强正相关
        completion_reward = 20.0 * completion_rate  # 强化完成率奖励，解决相关性问题
        cache_reward = 10.0 * cache_hit_rate        # 强化缓存命中率奖励
        
        # 3. 线性组合确保单调性（去除非线性函数避免非单调性）
        base_reward = delay_penalty + energy_penalty + loss_penalty + completion_reward + cache_reward
        
        # 4. 大幅放大信号强度（解决信号过弱问题）
        amplified_reward = base_reward * 3.0  # 3倍放大，增强学习信号
        
        # 5. 适当的奖励范围（保持信号强度的同时避免数值问题）
        final_reward = np.clip(amplified_reward, -80.0, 50.0)
        
        return float(final_reward)
    
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