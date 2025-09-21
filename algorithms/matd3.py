"""
MATD3深度强化学习算法 - 对应论文核心算法
多智能体Twin Delayed DDPG算法实现
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'MATD3': 256}  # 默认值

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from collections import deque
import random

from config import config


class Actor(nn.Module):
    """Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出范围[-1, 1]
        return x


class Critic(nn.Module):
    """Critic网络 - 价值网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(Critic, self).__init__()
        
        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2网络 (Twin网络)
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        
        # Q1
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        
        # Q2
        q2 = F.relu(self.fc1_q2(x))
        q2 = F.relu(self.fc2_q2(q2))
        q2 = self.fc3_q2(q2)
        
        return q1, q2
    
    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1


class ReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class MATD3Agent:
    """
    多智能体TD3算法实现
    """
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 网络参数
        self.hidden_dim = config.rl.hidden_dim
        self.actor_lr = config.rl.actor_lr
        self.critic_lr = config.rl.critic_lr
        self.tau = config.rl.tau
        self.gamma = config.rl.gamma
        
        # 优化后的批次大小 - 对应论文中的批量处理参数
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('MATD3', config.rl.batch_size)
        print(f"🚀 {agent_id} 使用优化批次大小: {self.optimized_batch_size}")
        
        # 噪声参数
        self.policy_noise = config.rl.policy_noise
        self.noise_clip = config.rl.noise_clip
        self.policy_delay = config.rl.policy_delay
        
        # 网络初始化 - 使用GPU加速（如果可用）
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.actor = Actor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target = Actor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = Critic(state_dim, action_dim, self.hidden_dim).to(self.device)
        
        # 复制目标网络权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 经验回放
        self.replay_buffer = ReplayBuffer(config.rl.buffer_size)
        
        # 训练计数
        self.total_it = 0
    
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """选择动作 - 使用GPU加速"""
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        
        with torch.no_grad():
            action_tensor = self.actor(state_tensor)
            action = action_tensor.cpu().data.numpy().flatten()
        
        if add_noise:
            noise = np.random.normal(0, config.rl.exploration_noise, size=self.action_dim)
            action = action + noise
            action = np.clip(action, -1, 1)
        
        return action
    
    def train(self, batch_size: Optional[int] = None):
        """训练智能体 - 使用优化后的批次大小"""
        # 使用优化后的批次大小，提高GPU利用率
        batch_size_val = batch_size if batch_size is not None else self.optimized_batch_size
        if len(self.replay_buffer) < batch_size_val:
            return {}
        
        self.total_it += 1
        
        # 采样经验
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size_val)
        
        # 移动到GPU进行加速计算
        state = torch.FloatTensor(state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        done = torch.FloatTensor(done).unsqueeze(1).to(self.device)
        
        # 计算目标Q值
        with torch.no_grad():
            # 目标动作加噪声
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)
            
            # 计算目标Q值 (Twin网络取最小值)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q
        
        # 更新Critic网络
        current_q1, current_q2 = self.critic(state, action)
        
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新Actor网络
        if self.total_it % self.policy_delay == 0:
            # 更新Actor网络
            actor_loss = -self.critic.q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            self._soft_update(self.actor_target, self.actor, self.tau)
            self._soft_update(self.critic_target, self.critic, self.tau)
            
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': actor_loss.item()
            }
        else:
            return {
                'critic_loss': critic_loss.item(),
                'actor_loss': 0.0
            }
    
    def _soft_update(self, target, source, tau):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def store_transition(self, state, action, reward, next_state, done):
        """存储经验"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])


class MATD3Environment:
    """
    MATD3多智能体环境
    管理多个智能体的训练过程
    """
    
    def __init__(self):
        # 智能体配置
        self.num_agents = config.rl.num_agents
        
        # 状态和动作维度 (简化设计)
        self.state_dim = 20  # 节点状态特征维度
        self.action_dim = 10  # 决策动作维度
        
        # 创建智能体
        self.agents: Dict[str, MATD3Agent] = {}
        agent_types = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        for i, agent_type in enumerate(agent_types):
            if i < self.num_agents:
                self.agents[agent_type] = MATD3Agent(
                    agent_id=agent_type,
                    state_dim=self.state_dim,
                    action_dim=self.action_dim
                )
        
        # 奖励权重 - 对应论文目标函数
        self.reward_weights = {
            'delay': config.rl.reward_weight_delay,     # ω_T
            'energy': config.rl.reward_weight_energy,   # ω_E  
            'loss': config.rl.reward_weight_loss        # ω_D
        }
        
        # 训练统计
        self.episode_rewards = []
        self.episode_losses = []
    
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> Dict[str, np.ndarray]:
        """
        构建智能体状态向量 - 改进版本，包含更丰富的状态信息
        每个智能体观察不同的状态特征
        """
        states = {}
        
        # 车辆智能体状态 - 关注本地计算和卸载决策
        if 'vehicle_agent' in self.agents:
            # 基础系统状态
            vehicle_features = [
                np.clip(system_metrics.get('avg_task_delay', 0.0) / 2.0, 0, 1),  # 归一化时延
                np.clip(system_metrics.get('total_energy_consumption', 0.0) / 500.0, 0, 1),  # 归一化能耗
                system_metrics.get('data_loss_rate', 0.0),  # 数据丢失率
                system_metrics.get('task_completion_rate', 0.0),  # 任务完成率
            ]
            
            # 车辆特定状态
            vehicle_nodes = [s for s in node_states.values() if hasattr(s, 'node_type') and s.node_type.value == 'vehicle']
            if vehicle_nodes:
                avg_vehicle_load = np.mean([s.load_factor for s in vehicle_nodes])
                avg_vehicle_queue = np.mean([len(getattr(s, 'task_queue', [])) for s in vehicle_nodes])
                vehicle_count = len(vehicle_nodes)
            else:
                avg_vehicle_load = 0.0
                avg_vehicle_queue = 0.0
                vehicle_count = 0
            
            vehicle_features.extend([
                np.clip(avg_vehicle_load, 0, 1),  # 平均车辆负载
                np.clip(avg_vehicle_queue / 10.0, 0, 1),  # 平均队列长度
                np.clip(vehicle_count / 20.0, 0, 1),  # 车辆数量比例
            ])
            
            # RSU可用性状态
            rsu_nodes = [s for s in node_states.values() if hasattr(s, 'node_type') and s.node_type.value == 'rsu']
            if rsu_nodes:
                avg_rsu_load = np.mean([s.load_factor for s in rsu_nodes])
                rsu_availability = sum(1 for s in rsu_nodes if s.load_factor < 0.8) / len(rsu_nodes)
            else:
                avg_rsu_load = 0.0
                rsu_availability = 0.0
            
            vehicle_features.extend([
                np.clip(avg_rsu_load, 0, 1),  # RSU平均负载
                rsu_availability,  # RSU可用性
                system_metrics.get('cache_hit_rate', 0.0),  # 缓存命中率
            ])
            
            # UAV可用性状态
            uav_nodes = [s for s in node_states.values() if hasattr(s, 'node_type') and s.node_type.value == 'uav']
            if uav_nodes:
                avg_uav_battery = np.mean([getattr(s, 'battery_level', 1.0) for s in uav_nodes])
                avg_uav_load = np.mean([s.load_factor for s in uav_nodes])
                uav_availability = sum(1 for s in uav_nodes if getattr(s, 'battery_level', 1.0) > 0.3) / len(uav_nodes)
            else:
                avg_uav_battery = 1.0
                avg_uav_load = 0.0
                uav_availability = 0.0
            
            vehicle_features.extend([
                avg_uav_battery,  # UAV平均电量
                np.clip(avg_uav_load, 0, 1),  # UAV平均负载
                uav_availability,  # UAV可用性
            ])
            
            # 网络状态
            vehicle_features.extend([
                system_metrics.get('avg_bandwidth_utilization', 0.0),  # 带宽利用率
                system_metrics.get('migration_success_rate', 0.0),  # 使用真实的迁移成功率
                np.random.uniform(0.8, 1.0),  # 信道质量（简化）
                np.clip(system_metrics.get('system_load_ratio', 0.0), 0, 1),  # 系统负载比
            ])
            
            # 确保维度正确
            while len(vehicle_features) < self.state_dim:
                vehicle_features.append(0.0)
            
            states['vehicle_agent'] = np.array(vehicle_features[:self.state_dim], dtype=np.float32)
        
        # RSU智能体状态
        if 'rsu_agent' in self.agents:
            # 安全计算RSU平均负载，避免空列表导致的None返回值
            rsu_load_factors = [s.load_factor for s in node_states.values() if s.node_type.value == 'rsu']
            avg_rsu_load = np.mean(rsu_load_factors) if rsu_load_factors else 0.0
            
            rsu_features = [
                len([s for s in node_states.values() if s.node_type.value == 'rsu']) / 10.0,  # RSU数量
                float(avg_rsu_load),  # 平均负载
                system_metrics.get('cache_hit_rate', 0.0),  # 缓存命中率
                system_metrics.get('migration_success_rate', 0.0),  # 迁移成功率
            ]
            
            while len(rsu_features) < self.state_dim:
                rsu_features.append(0.0)
            
            states['rsu_agent'] = np.array(rsu_features[:self.state_dim], dtype=np.float32)
        
        # UAV智能体状态
        if 'uav_agent' in self.agents:
            # 安全计算UAV平均电量和负载，避免空列表导致的None返回值
            uav_battery_levels = [getattr(s, 'battery_level', 1.0) for s in node_states.values() if s.node_type.value == 'uav']
            avg_uav_battery = np.mean(uav_battery_levels) if uav_battery_levels else 1.0
            
            uav_load_factors = [s.load_factor for s in node_states.values() if s.node_type.value == 'uav']
            avg_uav_load = np.mean(uav_load_factors) if uav_load_factors else 0.0
            
            uav_features = [
                len([s for s in node_states.values() if s.node_type.value == 'uav']) / 5.0,  # UAV数量
                float(avg_uav_battery),  # 平均电量
                float(avg_uav_load),  # 平均负载
                system_metrics.get('avg_bandwidth_utilization', 0.0),  # 带宽利用率
            ]
            
            while len(uav_features) < self.state_dim:
                uav_features.append(0.0)
            
            states['uav_agent'] = np.array(uav_features[:self.state_dim], dtype=np.float32)
        
        return states
    
    def calculate_rewards(self, prev_metrics: Dict, current_metrics: Dict) -> Dict[str, float]:
        """
        计算智能体奖励 - 使用标准化奖励函数
        确保奖励计算的一致性，严格对应论文目标函数
        """
        from utils.standardized_reward import calculate_standardized_reward
        
        rewards = {}
        
        # 为每个智能体计算标准化的基础奖励
        for agent_id in self.agents.keys():
            # 使用标准化奖励计算函数
            base_reward = calculate_standardized_reward(current_metrics, agent_id)
            
            # 计算性能变化奖励 (可选的额外奖励)
            change_bonus = self._calculate_performance_change_bonus(
                prev_metrics, current_metrics, agent_id)
            
            # 组合最终奖励
            final_reward = base_reward + change_bonus
            
            # 确保奖励在合理范围内
            rewards[agent_id] = np.clip(final_reward, -10.0, 5.0)
        
        return rewards
    
    def _calculate_performance_change_bonus(self, prev_metrics: Dict, 
                                          current_metrics: Dict, 
                                          agent_id: str) -> float:
        """
        计算性能变化奖励 - 奖励性能改善
        
        Args:
            prev_metrics: 前一步的系统指标
            current_metrics: 当前步的系统指标  
            agent_id: 智能体ID
            
        Returns:
            性能变化奖励
        """
        # 计算关键指标的变化
        delay_change = (prev_metrics.get('avg_task_delay', 0.0) - 
                       current_metrics.get('avg_task_delay', 0.0))
        energy_change = (prev_metrics.get('total_energy_consumption', 0.0) - 
                        current_metrics.get('total_energy_consumption', 0.0))
        loss_change = (prev_metrics.get('data_loss_rate', 0.0) - 
                      current_metrics.get('data_loss_rate', 0.0))
        
        # 归一化变化 (改善为正，恶化为负)
        delay_bonus = np.tanh(delay_change / 0.1) * 0.1   # 延迟减少奖励
        energy_bonus = np.tanh(energy_change / 50.0) * 0.1  # 能耗减少奖励
        loss_bonus = np.tanh(loss_change / 0.05) * 0.1    # 丢失率减少奖励
        
        # 智能体特定的变化奖励权重
        if agent_id == 'vehicle_agent':
            # 车辆智能体更关注本地处理效率
            return 0.8 * delay_bonus + 0.6 * energy_bonus + 0.4 * loss_bonus
        elif agent_id == 'rsu_agent':
            # RSU智能体更关注整体系统性能
            return 0.6 * delay_bonus + 0.4 * energy_bonus + 0.8 * loss_bonus
        elif agent_id == 'uav_agent':
            # UAV智能体更关注能效
            return 0.5 * delay_bonus + 0.9 * energy_bonus + 0.3 * loss_bonus
        else:
            # 默认权重
            return 0.6 * delay_bonus + 0.6 * energy_bonus + 0.6 * loss_bonus
    
    def train_step(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                  rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                  dones: Dict[str, bool]) -> Dict[str, Dict]:
        """训练所有智能体"""
        training_info = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id in states and agent_id in actions:
                # 存储经验
                agent.store_transition(
                    states[agent_id],
                    actions[agent_id],
                    rewards.get(agent_id, 0.0),
                    next_states[agent_id],
                    dones.get(agent_id, False)
                )
                
                # 训练智能体
                train_info = agent.train()
                training_info[agent_id] = train_info
        
        return training_info
    
    def get_actions(self, states: Dict[str, np.ndarray], training: bool = True) -> Dict[str, np.ndarray]:
        """获取所有智能体的动作"""
        actions = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id in states:
                action = agent.select_action(states[agent_id], add_noise=training)
                actions[agent_id] = action
        
        return actions
    
    def save_models(self, directory: str):
        """保存所有智能体模型"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_id}_model.pth")
            agent.save_model(filepath)
    
    def load_models(self, directory: str):
        """加载所有智能体模型"""
        import os
        
        for agent_id, agent in self.agents.items():
            filepath = os.path.join(directory, f"{agent_id}_model.pth")
            if os.path.exists(filepath):
                agent.load_model(filepath)
    
    def reset_hidden_states(self):
        """重置隐藏状态 (为TD3不需要，但保持接口兼容性)"""
        pass
    
    def get_global_state(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        """获取全局状态"""
        global_state = []
        for agent_id in sorted(states.keys()):
            if agent_id in states:
                global_state.append(states[agent_id])
        return np.concatenate(global_state) if global_state else np.array([])
    
    def store_experience(self, states: Dict, actions: Dict, log_probs: Dict,
                        rewards: Dict, dones: Dict, global_state: Optional[np.ndarray] = None):
        """存储经验 (为TD3不需要log_probs，但保持接口兼容性)"""
        # 对于TD3，直接在train_step中存储经验
        pass
    
    def update(self):
        """更新所有智能体 (为TD3的更新在train_step中执行)"""
        return {}