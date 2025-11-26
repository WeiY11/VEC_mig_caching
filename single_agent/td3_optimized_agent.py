"""
优化的TD3智能体实现
修复了原始实现中的关键问题
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple
from collections import deque

from single_agent.td3_optimized import OptimizedTD3Actor, OptimizedTD3Critic, OptimizedTD3Config


class OptimizedTD3ReplayBuffer:
    """优化的TD3经验回放缓冲区"""
    
    def __init__(self, capacity: int, state_dim: int, action_dim: int, alpha: float = 0.6):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.alpha = alpha
        
        # 预分配内存
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        # 优先级相关
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def __len__(self):
        return self.size
    
    def push(self, state: np.ndarray, action: np.ndarray, reward: float, next_state: np.ndarray, done: bool):
        """添加经验到缓冲区"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        # 新经验使用最大优先级
        self.priorities[self.ptr] = self.max_priority
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """采样经验"""
        if self.size < batch_size:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size]
        
        # 检查并处理NaN值
        if np.any(np.isnan(priorities)) or np.any(np.isinf(priorities)):
            print(f"警告: 发现NaN或Inf优先级值，重置为默认值")
            priorities = np.ones_like(priorities)
            self.priorities[:self.size] = priorities
            self.max_priority = 1.0
        
        # 确保优先级为正值
        priorities = np.maximum(priorities, 1e-8)
        
        probs = priorities ** self.alpha
        probs_sum = probs.sum()
        
        # 检查概率和是否有效
        if probs_sum <= 0 or np.isnan(probs_sum) or np.isinf(probs_sum):
            print(f"警告: 概率和无效 ({probs_sum})，使用均匀分布")
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum
        
        # 采样索引
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # 计算重要性采样权重
        weights = (self.size * probs[indices]) ** (-beta)
        max_weight = weights.max()
        
        # 检查权重是否有效
        if max_weight <= 0 or np.isnan(max_weight) or np.isinf(max_weight):
            weights = np.ones_like(weights)
        else:
            weights /= max_weight
        
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
            indices,
            weights
        )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        # 确保priorities是一维数组
        if priorities.ndim > 1:
            priorities = priorities.flatten()
        
        # 检查并处理NaN/Inf值
        if np.any(np.isnan(priorities)) or np.any(np.isinf(priorities)):
            print(f"警告: 更新优先级时发现NaN或Inf值，使用默认值替换")
            priorities = np.where(np.isnan(priorities) | np.isinf(priorities), 1.0, priorities)
        
        # 确保优先级为正值
        priorities = np.maximum(priorities, 1e-8)
        
        self.priorities[indices] = priorities
        valid_max = np.max(priorities[np.isfinite(priorities)])
        self.max_priority = max(self.max_priority, valid_max)


class OptimizedTD3Agent:
    """优化的TD3智能体"""
    
    def __init__(self, state_dim: int, action_dim: int, config: OptimizedTD3Config):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"✓ 使用设备: {self.device}")
        
        # 创建网络
        self.actor = OptimizedTD3Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = OptimizedTD3Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 目标网络
        self.target_actor = OptimizedTD3Actor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.target_critic = OptimizedTD3Critic(state_dim, action_dim, config.hidden_dim).to(self.device)
        
        # 初始化目标网络
        self.hard_update(self.target_actor, self.actor)
        self.hard_update(self.target_critic, self.critic)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # 学习率调度器
        self.actor_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.actor_optimizer, gamma=0.9995)
        self.critic_lr_scheduler = optim.lr_scheduler.ExponentialLR(self.critic_optimizer, gamma=0.9995)
        
        # 经验回放缓冲区
        self.beta = 0.4
        self.beta_increment = (1.0 - 0.4) / 500000
        self.replay_buffer = OptimizedTD3ReplayBuffer(config.buffer_size, state_dim, action_dim)
        
        # 训练统计
        self.total_it = 0
        self.update_count = 0
        self.exploration_noise = config.exploration_noise
        self.initial_exploration_noise = config.exploration_noise  # 记录初始噪声
        
        # 损失记录
        self.actor_losses = deque(maxlen=1000)
        self.critic_losses = deque(maxlen=1000)
        
        # 避免局部最优的参数
        self.episode_count = 0  # 追踪episode数
        self.recent_rewards = deque(maxlen=50)  # 追踪最近50个episode的奖励
        self.last_improvement_episode = 0  # 上次改善的episode
        self.exploration_reset_interval = 100  # 每100个episode重启一次探索
        
        # 🔥 新增：提前终止和噪声退火机制 (600轮后启用)
        self.early_stop_episode = 600  # 在600轮后开始检测是否提前终止
        self.noise_annealing_start = 600  # 在600轮后开始噪声退火
        self.noise_annealing_rate = 0.995  # 噪声退火率 (每个episode乘以0.995)
        self.reward_std_threshold = 0.5  # 奖励方差阈值，低于此值认为收敛
        
        print(f"✓ 优化TD3智能体初始化完成")
        print(f"✓ 网络隐藏维度: {config.hidden_dim}")
        print(f"✓ 缓冲区大小: {config.buffer_size}")
        print(f"✓ 启用避免局部最优机制 (周期重启间隔: {self.exploration_reset_interval}ep)")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().data.numpy().flatten()
        
        if training:
            # 添加探索噪声
            noise = np.random.normal(0, self.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1.0, 1.0)
            
            # 噪声衰减
            self.exploration_noise = max(
                self.config.min_noise,
                self.exploration_noise * self.config.noise_decay
            )
        
        return action
    
    def set_episode_count(self, episode: int, recent_reward: float = None):
        """
        更新episode计数，实现自适应探索重启机制
        """
        self.episode_count = episode
        
        if recent_reward is not None:
            self.recent_rewards.append(recent_reward)
        
        # 🔥 新增：噪声退火 (600轮后)
        if episode >= self.noise_annealing_start:
            # 逐渐降低噪声，但保持最小噪声
            self.exploration_noise = max(
                self.config.min_noise,
                self.exploration_noise * self.noise_annealing_rate
            )
            if episode % 50 == 0:  # 每50轮报告一次
                print(f"💨 Episode {episode}: 噪声退火 -> {self.exploration_noise:.4f}")
        
        # 周期性重启探索：每100个episode检查一次是否陷入局部最优
        if episode % self.exploration_reset_interval == 0 and episode > 100:
            # 计算最近50个episode的平均奖励
            if len(self.recent_rewards) >= 30:
                recent_avg = np.mean(list(self.recent_rewards)[-30:])
                earlier_avg = np.mean(list(self.recent_rewards)[:30])
                
                # 如果没有显著改善，重启探索
                improvement_ratio = (earlier_avg - recent_avg) / (abs(earlier_avg) + 1e-6)
                
                if improvement_ratio < 0.05:  # 改善少于5%
                    # 重启噪声
                    old_noise = self.exploration_noise
                    self.exploration_noise = self.initial_exploration_noise * 0.5  # 重启为初始值的50%
                    print(f"🔄 Episode {episode}: 检测到局部最优,重启探索")
                    print(f"   (改善率: {improvement_ratio*100:.2f}% < 5%)")
                    print(f"   探索噪声: {old_noise:.4f} → {self.exploration_noise:.4f}")
                    print(f"   最近平均奖励: {recent_avg:.4f} (早期: {earlier_avg:.4f})")
    
    def check_early_stopping(self) -> bool:
        """
        🔥 新增：检查是否应该提前终止训练 (600轮后)
        基于最近50个episode奖励方差判断收敛
        """
        if self.episode_count < self.early_stop_episode:
            return False
        
        if len(self.recent_rewards) < 50:
            return False
        
        # 计算最近50个episode奖励的标准差
        reward_std = np.std(list(self.recent_rewards))
        
        if reward_std < self.reward_std_threshold:
            print(f"✅ Episode {self.episode_count}: 训练收敛，提前终止")
            print(f"   奖励标准差: {reward_std:.4f} < {self.reward_std_threshold}")
            print(f"   最近50轮平均奖励: {np.mean(list(self.recent_rewards)):.4f}")
            return True
        
        return False
    
    def store_experience(self, state: np.ndarray, action: np.ndarray, reward: float,
                        next_state: np.ndarray, done: bool):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        self.total_it += 1
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # 采样经验
        batch = self.replay_buffer.sample(self.config.batch_size, self.beta)
        if batch is None:
            return {}
        
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # 更新Critic
        critic_loss, td_errors = self._update_critic(states, actions, rewards, next_states, dones, weights)
        
        # 更新优先级
        td_errors_np = td_errors.cpu().data.numpy()
        
        # 检查TD误差是否包含NaN/Inf
        if np.any(np.isnan(td_errors_np)) or np.any(np.isinf(td_errors_np)):
            print(f"警告: TD误差包含NaN或Inf值，使用默认优先级")
            priorities = np.ones_like(td_errors_np) * 1e-6
        else:
            priorities = np.abs(td_errors_np) + 1e-6
        
        self.replay_buffer.update_priorities(indices, priorities)
        
        # 延迟策略更新
        actor_loss = 0.0
        if self.total_it % self.config.policy_delay == 0:
            actor_loss = self._update_actor(states)
            
            # 软更新目标网络
            self.soft_update(self.target_actor, self.actor, self.config.tau)
            self.soft_update(self.target_critic, self.critic, self.config.tau)
        
        self.update_count += 1
        
        # 学习率调度
        if self.update_count % 1000 == 0:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()
        
        return {
            'critic_loss': critic_loss,
            'actor_loss': actor_loss,
            'exploration_noise': self.exploration_noise,
            'buffer_size': len(self.replay_buffer),
            'update_count': self.update_count,
            'beta': self.beta
        }
    
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
        
        # 检查Q值是否包含NaN
        if torch.any(torch.isnan(current_q1)) or torch.any(torch.isnan(current_q2)):
            print("警告: 当前Q值包含NaN")
            current_q1 = torch.nan_to_num(current_q1, nan=0.0)
            current_q2 = torch.nan_to_num(current_q2, nan=0.0)
        
        if torch.any(torch.isnan(target_q)):
            print("警告: 目标Q值包含NaN")
            target_q = torch.nan_to_num(target_q, nan=0.0)
        
        # TD误差
        td_error1 = target_q - current_q1
        td_error2 = target_q - current_q2
        
        # 检查TD误差是否包含NaN
        if torch.any(torch.isnan(td_error1)) or torch.any(torch.isnan(td_error2)):
            print("警告: TD误差包含NaN")
            td_error1 = torch.nan_to_num(td_error1, nan=0.0)
            td_error2 = torch.nan_to_num(td_error2, nan=0.0)
        
        # 加权损失
        critic_loss1 = (weights * td_error1.pow(2)).mean()
        critic_loss2 = (weights * td_error2.pow(2)).mean()
        critic_loss = critic_loss1 + critic_loss2
        
        # 检查损失是否包含NaN
        if torch.isnan(critic_loss):
            print("警告: Critic损失为NaN，跳过此次更新")
            return 0.0, torch.zeros_like(td_error1).detach()
        
        # 更新Critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        
        # 返回平均TD误差用于优先级更新
        td_errors = (td_error1 + td_error2) / 2
        
        return critic_loss.item(), td_errors.detach()
    
    def _update_actor(self, states: torch.Tensor) -> float:
        """更新Actor网络"""
        # 计算策略损失 (只使用Q1网络)
        actions = self.actor(states)
        actor_loss = -self.critic.q1(states, actions).mean()
        
        # 更新Actor
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
        import os
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'exploration_noise': self.exploration_noise,
            'update_count': self.update_count,
            'total_it': self.total_it,
        }, os.path.join(filepath, 'optimized_td3_agent.pth'))
    
    def load_model(self, filepath: str):
        """加载模型"""
        import os
        checkpoint = torch.load(os.path.join(filepath, 'optimized_td3_agent.pth'), 
                               map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        self.exploration_noise = checkpoint.get('exploration_noise', self.config.exploration_noise)
        self.update_count = checkpoint.get('update_count', 0)
        self.total_it = checkpoint.get('total_it', 0)