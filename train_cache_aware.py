#!/usr/bin/env python3
"""
缓存感知训练脚本
专门针对缓存优化的训练方法
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os

class CacheAwareAgent(nn.Module):
    """缓存感知智能体"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CacheAwareAgent, self).__init__()
        
        # 缓存状态编码器
        self.cache_encoder = nn.Sequential(
            nn.Linear(state_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 网络状态编码器
        self.network_encoder = nn.Sequential(
            nn.Linear(state_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # 融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()
        )
    
    def forward(self, state):
        """前向传播"""
        # 分离缓存状态和网络状态
        cache_state = state[:, :state.size(1)//2]
        network_state = state[:, state.size(1)//2:]
        
        # 编码
        cache_features = self.cache_encoder(cache_state)
        network_features = self.network_encoder(network_state)
        
        # 融合
        combined_features = torch.cat([cache_features, network_features], dim=1)
        action = self.fusion_layer(combined_features)
        
        return action

class CacheAwareTrainer:
    """缓存感知训练器"""
    
    def __init__(self, state_dim=20, action_dim=5, lr=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # 创建智能体
        self.agent = CacheAwareAgent(state_dim, action_dim)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=lr)
        
        # 训练参数
        self.batch_size = 64
        self.memory_size = 10000
        self.memory = []
        
        # 缓存相关参数
        self.cache_hit_reward = 10.0
        self.cache_miss_penalty = -2.0
        self.cache_update_cost = -0.5
        
        # 训练统计
        self.training_stats = {
            'episodes': 0,
            'total_reward': 0,
            'cache_hit_rate': 0,
            'avg_delay': 0
        }
    
    def simulate_cache_environment(self, action):
        """模拟缓存环境"""
        # 模拟缓存命中/未命中
        cache_hit_prob = torch.sigmoid(action[0]).item()
        cache_hit = np.random.random() < cache_hit_prob
        
        # 计算奖励
        if cache_hit:
            reward = self.cache_hit_reward
            delay = np.random.exponential(0.01)  # 缓存命中，低延迟
        else:
            reward = self.cache_miss_penalty
            delay = np.random.exponential(0.1)   # 缓存未命中，高延迟
        
        # 缓存更新成本
        if action[1] > 0.5:  # 决定更新缓存
            reward += self.cache_update_cost
        
        return reward, delay, cache_hit
    
    def generate_state(self):
        """生成状态"""
        # 缓存状态 (前一半)
        cache_state = np.random.random(self.state_dim // 2)
        
        # 网络状态 (后一半)
        network_state = np.random.random(self.state_dim // 2)
        
        state = np.concatenate([cache_state, network_state])
        return torch.FloatTensor(state).unsqueeze(0)
    
    def store_experience(self, state, action, reward, next_state, done):
        """存储经验"""
        experience = (state, action, reward, next_state, done)
        
        if len(self.memory) >= self.memory_size:
            self.memory.pop(0)
        
        self.memory.append(experience)
    
    def sample_batch(self):
        """采样批次"""
        if len(self.memory) < self.batch_size:
            return None
        
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        experiences = [self.memory[i] for i in batch]
        
        states = torch.cat([exp[0] for exp in experiences])
        actions = torch.cat([exp[1] for exp in experiences])
        rewards = torch.FloatTensor([exp[2] for exp in experiences])
        next_states = torch.cat([exp[3] for exp in experiences])
        dones = torch.BoolTensor([exp[4] for exp in experiences])
        
        return states, actions, rewards, next_states, dones
    
    def train_step(self):
        """训练步骤"""
        batch = self.sample_batch()
        if batch is None:
            return 0
        
        states, actions, rewards, next_states, dones = batch
        
        # 计算当前Q值
        current_actions = self.agent(states)
        
        # 计算目标Q值 (简化版，实际应该用Critic网络)
        with torch.no_grad():
            next_actions = self.agent(next_states)
            target_q = rewards + 0.99 * torch.sum(next_actions, dim=1) * (~dones)
        
        # 计算损失
        current_q = torch.sum(current_actions, dim=1)
        loss = nn.MSELoss()(current_q, target_q)
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_episode(self):
        """训练一个回合"""
        state = self.generate_state()
        episode_reward = 0
        episode_cache_hits = 0
        episode_delays = []
        steps = 0
        
        for step in range(100):  # 每回合100步
            # 选择动作
            with torch.no_grad():
                action = self.agent(state)
            
            # 环境交互
            reward, delay, cache_hit = self.simulate_cache_environment(action[0])
            
            # 生成下一状态
            next_state = self.generate_state()
            done = step == 99
            
            # 存储经验
            self.store_experience(state, action, reward, next_state, done)
            
            # 更新统计
            episode_reward += reward
            if cache_hit:
                episode_cache_hits += 1
            episode_delays.append(delay)
            
            # 训练
            if len(self.memory) >= self.batch_size:
                self.train_step()
            
            state = next_state
            steps += 1
        
        # 更新统计
        self.training_stats['episodes'] += 1
        self.training_stats['total_reward'] += episode_reward
        self.training_stats['cache_hit_rate'] = episode_cache_hits / steps
        self.training_stats['avg_delay'] = np.mean(episode_delays)
        
        return episode_reward, episode_cache_hits / steps, np.mean(episode_delays)
    
    def train(self, num_episodes=1000):
        """训练主循环"""
        print("🚀 开始缓存感知训练...")
        
        episode_rewards = []
        cache_hit_rates = []
        avg_delays = []
        
        for episode in range(num_episodes):
            reward, hit_rate, delay = self.train_episode()
            
            episode_rewards.append(reward)
            cache_hit_rates.append(hit_rate)
            avg_delays.append(delay)
            
            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_hit_rate = np.mean(cache_hit_rates[-100:])
                avg_delay = np.mean(avg_delays[-100:])
                
                print(f"Episode {episode+1}/{num_episodes}")
                print(f"  平均奖励: {avg_reward:.2f}")
                print(f"  缓存命中率: {avg_hit_rate:.2%}")
                print(f"  平均时延: {avg_delay:.4f}s")
        
        print("✅ 缓存感知训练完成！")
        
        return {
            'episode_rewards': episode_rewards,
            'cache_hit_rates': cache_hit_rates,
            'avg_delays': avg_delays,
            'final_stats': self.training_stats
        }
    
    def save_model(self, filepath):
        """保存模型"""
        torch.save({
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, filepath)
        print(f"✅ 模型已保存: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint['training_stats']
        print(f"✅ 模型已加载: {filepath}")

def main():
    """主函数"""
    print("🎯 缓存感知训练系统")
    print("=" * 50)
    
    # 创建训练器
    trainer = CacheAwareTrainer(state_dim=20, action_dim=5, lr=0.001)
    
    # 开始训练
    results = trainer.train(num_episodes=500)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存模型
    model_path = f"results/cache_aware_model_{timestamp}.pth"
    trainer.save_model(model_path)
    
    # 保存训练结果
    results_path = f"results/cache_aware_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = {
            'episode_rewards': [float(x) for x in results['episode_rewards']],
            'cache_hit_rates': [float(x) for x in results['cache_hit_rates']],
            'avg_delays': [float(x) for x in results['avg_delays']],
            'final_stats': results['final_stats'],
            'timestamp': timestamp
        }
        json.dump(json_results, f, indent=2)
    
    print(f"📊 训练结果已保存: {results_path}")
    
    # 显示最终统计
    print("\n📈 最终训练统计:")
    print(f"  总回合数: {results['final_stats']['episodes']}")
    print(f"  最终缓存命中率: {results['cache_hit_rates'][-1]:.2%}")
    print(f"  最终平均时延: {results['avg_delays'][-1]:.4f}s")

if __name__ == "__main__":
    main()