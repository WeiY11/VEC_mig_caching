#!/usr/bin/env python3
"""
修复版训练脚本
解决了数值稳定性和收敛问题的训练脚本
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os
from pathlib import Path

class FixedAgent(nn.Module):
    """修复版智能体"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(FixedAgent, self).__init__()
        
        # 改进的网络架构
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),  # 添加层归一化
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加dropout防止过拟合
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh()
        )
        
        self.critic = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, state):
        """前向传播"""
        return self.actor(state)
    
    def get_q_value(self, state, action):
        """获取Q值"""
        return self.critic(torch.cat([state, action], dim=1))

class FixedTrainer:
    """修复版训练器"""
    
    def __init__(self, state_dim=20, action_dim=5, lr=0.0003):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建网络
        self.actor = FixedAgent(state_dim, action_dim)
        self.critic = FixedAgent(state_dim, action_dim)
        self.target_actor = FixedAgent(state_dim, action_dim)
        self.target_critic = FixedAgent(state_dim, action_dim)
        
        # 初始化目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        
        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=1000, gamma=0.95)
        
        # 训练参数
        self.batch_size = 128
        self.memory_size = 50000
        self.memory = []
        self.tau = 0.005  # 软更新参数
        self.gamma = 0.99  # 折扣因子
        self.noise_std = 0.1  # 噪声标准差
        
        # 梯度裁剪
        self.max_grad_norm = 1.0
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'q_values': []
        }
    
    def add_noise(self, action):
        """添加探索噪声"""
        noise = torch.normal(0, self.noise_std, size=action.shape)
        return torch.clamp(action + noise, -1, 1)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = (
            state.cpu().numpy(),
            action.cpu().numpy(),
            reward,
            next_state.cpu().numpy(),
            done
        )
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
    
    def sample_batch(self):
        """采样批次"""
        if len(self.memory) < self.batch_size:
            return None
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        batch = [self.memory[i] for i in indices]
        
        states = torch.FloatTensor([exp[0] for exp in batch])
        actions = torch.FloatTensor([exp[1] for exp in batch])
        rewards = torch.FloatTensor([exp[2] for exp in batch]).unsqueeze(1)
        next_states = torch.FloatTensor([exp[3] for exp in batch])
        dones = torch.BoolTensor([exp[4] for exp in batch]).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones
    
    def update_target_networks(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_step(self):
        """训练步骤"""
        batch = self.sample_batch()
        if batch is None:
            return 0, 0
        
        states, actions, rewards, next_states, dones = batch
        
        # 训练Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            next_actions = self.add_noise(next_actions)  # 目标策略平滑
            target_q = self.target_critic.get_q_value(next_states, next_actions)
            target_q = rewards + self.gamma * target_q * (~dones)
        
        current_q = self.critic.get_q_value(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()
        
        # 训练Actor (延迟更新)
        actor_loss = 0
        if len(self.training_stats['critic_losses']) % 2 == 0:  # 每2步更新一次Actor
            predicted_actions = self.actor(states)
            actor_loss = -self.critic.get_q_value(states, predicted_actions).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()
            
            # 更新目标网络
            self.update_target_networks()
            
            actor_loss = actor_loss.item()
        
        return actor_loss, critic_loss.item()
    
    def generate_environment_data(self):
        """生成环境数据"""
        # 模拟车联网环境状态
        vehicle_states = np.random.uniform(0, 1, 5)  # 车辆状态
        rsu_states = np.random.uniform(0, 1, 5)      # RSU状态
        uav_states = np.random.uniform(0, 1, 5)      # UAV状态
        network_states = np.random.uniform(0, 1, 5)  # 网络状态
        
        state = np.concatenate([vehicle_states, rsu_states, uav_states, network_states])
        return torch.FloatTensor(state).unsqueeze(0)
    
    def calculate_reward(self, action, state):
        """计算奖励"""
        # 基于动作和状态计算奖励
        # 这里使用简化的奖励函数
        
        # 时延奖励 (动作越小时延越低)
        delay_reward = -torch.sum(torch.abs(action)) * 0.1
        
        # 能耗奖励 (动作平衡性)
        energy_reward = -torch.var(action) * 0.05
        
        # 缓存命中奖励 (基于动作的第一个维度)
        cache_reward = torch.sigmoid(action[0, 0]) * 2.0
        
        total_reward = delay_reward + energy_reward + cache_reward
        return total_reward.item()
    
    def train_episode(self):
        """训练一个回合"""
        state = self.generate_environment_data()
        episode_reward = 0
        episode_q_values = []
        
        for step in range(200):  # 每回合200步
            # 选择动作
            with torch.no_grad():
                action = self.actor(state)
                if len(self.memory) < self.batch_size * 10:  # 初期增加探索
                    action = self.add_noise(action)
            
            # 计算奖励
            reward = self.calculate_reward(action, state)
            
            # 生成下一状态
            next_state = self.generate_environment_data()
            done = step == 199
            
            # 存储经验
            self.store_experience(state, action, reward, next_state, done)
            
            # 训练
            if len(self.memory) >= self.batch_size:
                actor_loss, critic_loss = self.train_step()
                
                if actor_loss > 0:  # 只有当Actor更新时才记录
                    self.training_stats['actor_losses'].append(actor_loss)
                self.training_stats['critic_losses'].append(critic_loss)
                
                # 记录Q值
                with torch.no_grad():
                    q_value = self.critic.get_q_value(state, action).item()
                    episode_q_values.append(q_value)
            
            episode_reward += reward
            state = next_state
        
        # 更新学习率
        self.actor_scheduler.step()
        self.critic_scheduler.step()
        
        # 更新统计
        self.training_stats['episodes'] += 1
        self.training_stats['rewards'].append(episode_reward)
        if episode_q_values:
            self.training_stats['q_values'].append(np.mean(episode_q_values))
        
        return episode_reward
    
    def train(self, num_episodes=2000):
        """训练主循环"""
        print("🚀 开始修复版训练...")
        print(f"训练参数: episodes={num_episodes}, batch_size={self.batch_size}")
        
        for episode in range(num_episodes):
            episode_reward = self.train_episode()
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.training_stats['rewards'][-100:])
                avg_actor_loss = np.mean(self.training_stats['actor_losses'][-50:]) if self.training_stats['actor_losses'] else 0
                avg_critic_loss = np.mean(self.training_stats['critic_losses'][-100:]) if self.training_stats['critic_losses'] else 0
                avg_q_value = np.mean(self.training_stats['q_values'][-100:]) if self.training_stats['q_values'] else 0
                
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  平均奖励: {avg_reward:.3f}")
                print(f"  Actor损失: {avg_actor_loss:.6f}")
                print(f"  Critic损失: {avg_critic_loss:.6f}")
                print(f"  平均Q值: {avg_q_value:.3f}")
                print(f"  学习率: {self.actor_optimizer.param_groups[0]['lr']:.6f}")
        
        print("✅ 修复版训练完成！")
        return self.training_stats
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"✅ 模型已保存: {filepath}")

def main():
    """主函数"""
    print("🎯 修复版训练系统")
    print("=" * 50)
    
    # 确保结果目录存在
    Path("results/fixed_training").mkdir(parents=True, exist_ok=True)
    
    # 创建训练器
    trainer = FixedTrainer(state_dim=20, action_dim=5, lr=0.0003)
    
    # 开始训练
    results = trainer.train(num_episodes=1000)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"results/fixed_training/fixed_model_{timestamp}.pth"
    trainer.save_model(model_path)
    
    # 保存训练结果
    results_path = f"results/fixed_training/fixed_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json_results = {
            'episodes': results['episodes'],
            'actor_losses': results['actor_losses'],
            'critic_losses': results['critic_losses'],
            'rewards': results['rewards'],
            'q_values': results['q_values'],
            'timestamp': timestamp,
            'final_performance': {
                'avg_reward': np.mean(results['rewards'][-100:]) if results['rewards'] else 0,
                'final_actor_loss': results['actor_losses'][-1] if results['actor_losses'] else 0,
                'final_critic_loss': results['critic_losses'][-1] if results['critic_losses'] else 0,
                'convergence_episode': len(results['rewards'])
            }
        }
        json.dump(json_results, f, indent=2)
    
    print(f"📊 训练结果已保存: {results_path}")
    
    # 显示最终统计
    print("\n📈 最终训练统计:")
    print(f"  总回合数: {results['episodes']}")
    if results['rewards']:
        print(f"  最终平均奖励: {np.mean(results['rewards'][-100:]):.3f}")
    if results['actor_losses']:
        print(f"  最终Actor损失: {results['actor_losses'][-1]:.6f}")
    if results['critic_losses']:
        print(f"  最终Critic损失: {results['critic_losses'][-1]:.6f}")

if __name__ == "__main__":
    main()