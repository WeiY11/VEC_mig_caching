"""
PPO (Proximal Policy Optimization) 单智能体算法实现
专门适配MATD3-MIG系统的VEC环境

主要特点:
1. 策略梯度方法处理连续动作空间
2. 裁剪代理目标防止过大策略更新
3. GAE (Generalized Advantage Estimation) 减少方差
4. 自适应KL散度约束

对应论文: Proximal Policy Optimization Algorithms
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'PPO': 64}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
from collections import deque
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from config import config


@dataclass
class PPOConfig:
    """PPO算法配置"""
    # 网络结构
    hidden_dim: int = 256
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    
    # PPO参数
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # 训练参数
    batch_size: int = 64
    buffer_size: int = 2048
    ppo_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 其他参数
    normalize_advantages: bool = True
    use_gae: bool = True
    target_kl: float = 0.01  # 自适应KL散度约束


class PPOActor(nn.Module):
    """PPO Actor网络 - 随机策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(PPOActor, self).__init__()
        
        self.action_dim = action_dim
        
        # 共享特征层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
        
        # 动作均值层
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        
        # 动作标准差层 (可学习的参数)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=int(np.sqrt(2)))
                nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.mean_layer.weight, gain=1)  # 使用整数gain
        nn.init.constant_(self.mean_layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        features = self.feature_layers(state)
        
        # 动作分布参数
        mean = self.mean_layer(features)
        std = torch.exp(self.log_std.clamp(-20, 2))
        
        return mean, std
    
    def get_action_and_logprob(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """获取动作和对数概率"""
        mean, std = self.forward(state)
        dist = Normal(mean, std)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        
        return action, log_prob, entropy


class PPOCritic(nn.Module):
    """PPO Critic网络 - 价值网络"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super(PPOCritic, self).__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.value_network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=int(np.sqrt(2)))
                nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用单位增益
        nn.init.orthogonal_(self.value_network[-1].weight, gain=1)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.value_network(state)


class PPOBuffer:
    """PPO经验缓冲区"""
    
    def __init__(self, buffer_size: int, state_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        # 缓冲区数据
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(buffer_size, dtype=np.float32)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)
        self.values = np.zeros(buffer_size, dtype=np.float32)
        self.advantages = np.zeros(buffer_size, dtype=np.float32)
        self.returns = np.zeros(buffer_size, dtype=np.float32)
    
    def store(self, state: np.ndarray, action: np.ndarray, log_prob: float,
              reward: float, done: bool, value: float):
        """存储一步经验"""
        if self.size < self.buffer_size:
            self.size += 1
        
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        
        self.ptr = (self.ptr + 1) % self.buffer_size
    
    def compute_advantages_and_returns(self, last_value: float, gamma: float, gae_lambda: float, use_gae: bool = True):
        """计算优势函数和回报"""
        if use_gae:
            # 使用GAE计算优势函数
            advantages = np.zeros(self.size, dtype=np.float32)
            last_gae = 0
            
            for t in reversed(range(self.size)):
                if t == self.size - 1:
                    next_value = last_value
                    next_done = 0
                else:
                    next_value = self.values[t + 1]
                    next_done = self.dones[t + 1]
                
                delta = self.rewards[t] + gamma * next_value * (1 - next_done) - self.values[t]
                last_gae = delta + gamma * gae_lambda * (1 - next_done) * last_gae
                advantages[t] = last_gae
            
            returns = advantages + self.values[:self.size]
        else:
            # 直接计算回报
            returns = np.zeros(self.size, dtype=np.float32)
            running_return = last_value
            
            for t in reversed(range(self.size)):
                if self.dones[t]:
                    running_return = 0
                running_return = self.rewards[t] + gamma * running_return
                returns[t] = running_return
            
            advantages = returns - self.values[:self.size]
        
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = returns
    
    def get_batch(self, batch_size: int):
        """获取训练批次"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),
            torch.FloatTensor(self.actions[indices]),
            torch.FloatTensor(self.log_probs[indices]),
            torch.FloatTensor(self.advantages[indices]),
            torch.FloatTensor(self.returns[indices]),
            torch.FloatTensor(self.values[indices])
        )
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """PPO智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: PPOConfig):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('PPO', config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = PPOActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = PPOCritic(state_dim, config.hidden_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(config.buffer_size, state_dim, action_dim)
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
        self.kl_divergences = []
        
        # 其他参数
        self.step_count = 0
    
    def select_action(self, state: np.ndarray, training: bool = True):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action_and_logprob(state_tensor)
            value = self.critic(state_tensor)
        
        action = torch.clamp(action, -1.0, 1.0)  # 限制动作范围
        
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, log_prob: float,
                        reward: float, done: bool, value: float):
        """存储经验"""
        self.buffer.store(state, action, log_prob, reward, done, value)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """PPO更新"""
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # 计算优势函数和回报
        self.buffer.compute_advantages_and_returns(
            last_value, self.config.gamma, self.config.gae_lambda, self.config.use_gae
        )
        
        # 标准化优势函数
        if self.config.normalize_advantages and self.buffer.size > 1:
            advantages = self.buffer.advantages[:self.buffer.size]
            self.buffer.advantages[:self.buffer.size] = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            )
        
        # PPO更新循环
        actor_losses = []
        critic_losses = []
        entropy_losses = []
        kl_divs = []
        
        for epoch in range(self.config.ppo_epochs):
            # 获取训练批次
            batch_states, batch_actions, batch_old_log_probs, batch_advantages, batch_returns, batch_old_values = \
                self.buffer.get_batch(self.config.batch_size)
            
            batch_states = batch_states.to(self.device)
            batch_actions = batch_actions.to(self.device)
            batch_old_log_probs = batch_old_log_probs.to(self.device)
            batch_advantages = batch_advantages.to(self.device)
            batch_returns = batch_returns.to(self.device)
            
            # 计算新的策略分布
            _, new_log_probs, entropy = self.actor.get_action_and_logprob(batch_states, batch_actions)
            new_values = self.critic(batch_states).squeeze()
            
            # 计算比率
            ratio = torch.exp(new_log_probs - batch_old_log_probs)
            
            # PPO裁剪损失
            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_ratio, 1.0 + self.config.clip_ratio) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # 熵损失
            entropy_loss = -entropy.mean()
            
            # 总Actor损失
            total_actor_loss = actor_loss + self.config.entropy_coef * entropy_loss
            
            # 更新Actor
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
            self.actor_optimizer.step()
            
            # Critic损失
            critic_loss = F.mse_loss(new_values, batch_returns)
            
            # 更新Critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()
            
            # 计算KL散度
            kl_div = (batch_old_log_probs - new_log_probs).mean()
            
            # 记录损失
            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropy_losses.append(entropy_loss.item())
            kl_divs.append(kl_div.item())
            
            # 早停检查 (如果KL散度过大)
            if kl_div > self.config.target_kl * 4:
                break
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 记录统计信息
        avg_actor_loss = float(np.mean(actor_losses))
        avg_critic_loss = float(np.mean(critic_losses))
        avg_entropy_loss = float(np.mean(entropy_losses))
        avg_kl_div = float(np.mean(kl_divs))
        
        self.actor_losses.append(avg_actor_loss)
        self.critic_losses.append(avg_critic_loss)
        self.entropy_losses.append(avg_entropy_loss)
        self.kl_divergences.append(avg_kl_div)
        
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
            'entropy_loss': avg_entropy_loss,
            'kl_divergence': avg_kl_div,
            'ppo_epochs': len(actor_losses)
        }
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'step_count': self.step_count
        }, f"{filepath}_ppo.pth")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(f"{filepath}_ppo.pth", map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.step_count = checkpoint['step_count']


class PPOEnvironment:
    """PPO训练环境"""
    
    def __init__(self):
        self.config = PPOConfig()
        
        # 🔧 修复：正确计算状态维度，与TD3保持一致
        self.state_dim = 130  # 车辆60 + RSU54 + UAV16 = 130维
        self.action_dim = 30  # 整合所有节点动作
        
        # 创建智能体
        self.agent = PPOAgent(self.state_dim, self.action_dim, self.config)
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print(f"✓ PPO环境初始化完成")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
        print(f"✓ 缓冲区大小: {self.config.buffer_size}")
        print(f"✓ PPO轮次: {self.config.ppo_epochs}")
    
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
    
    def get_actions(self, state: np.ndarray, training: bool = True):
        """获取动作"""
        action, log_prob, value = self.agent.select_action(state, training)
        return self.decompose_action(action), log_prob, value
    
    def calculate_reward(self, system_metrics: Dict) -> float:
        """计算奖励 - 使用标准化奖励函数"""
        from utils.standardized_reward import calculate_standardized_reward
        return calculate_standardized_reward(system_metrics, agent_type='single_agent')
    
    def save_models(self, filepath: str):
        """保存模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        self.agent.save_model(filepath)
        print(f"✓ PPO模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
        print(f"✓ PPO模型已加载: {filepath}")
    
    def train_step(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                   next_state: np.ndarray, done: bool) -> Dict:
        """占位符train_step方法 - PPO不使用此方法"""
        # PPO不使用train_step，而是使用store_experience和update
        # 这里提供空实现以保持接口统一
        return {'message': 'PPO does not use train_step method'}
    
    def store_experience(self, state: np.ndarray, action: Union[np.ndarray, int], reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float = 0.0, value: float = 0.0):
        """存储经验到缓冲区 - 支持统一接口"""
        # 确保action是numpy数组
        if isinstance(action, int):
            action = np.array([action], dtype=np.float32)
        elif not isinstance(action, np.ndarray):
            action = np.array(action, dtype=np.float32)
        
        # PPO Agent的store_experience参数顺序: (state, action, log_prob, reward, done, value)
        # 注意: PPO不使用next_state参数
        self.agent.store_experience(state, action, log_prob, reward, done, value)
        self.step_count += 1
    
    def update(self, last_value: float = 0.0) -> Dict:
        """PPO更新 (在episode结束后调用) - 支持统一接口"""
        return self.agent.update(last_value)
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'actor_loss_avg': float(np.mean(self.agent.actor_losses[-10:])) if self.agent.actor_losses else 0.0,
            'critic_loss_avg': float(np.mean(self.agent.critic_losses[-10:])) if self.agent.critic_losses else 0.0,
            'entropy_loss_avg': float(np.mean(self.agent.entropy_losses[-10:])) if self.agent.entropy_losses else 0.0,
            'kl_divergence_avg': float(np.mean(self.agent.kl_divergences[-10:])) if self.agent.kl_divergences else 0.0,
            'buffer_size': self.agent.buffer.size,
            'step_count': self.step_count
        }