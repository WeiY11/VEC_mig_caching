"""
MAPPO (Multi-Agent Proximal Policy Optimization) 算法实现
专门适配MATD3-MIG系统的多智能体环境

主要特点:
1. 基于PPO的多智能体策略梯度方法
2. 集中式训练，分布式执行
3. 支持连续和离散动作空间
4. 稳定的策略更新机制

对应论文: The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
"""
# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import OPTIMIZED_BATCH_SIZES
except ImportError:
    OPTIMIZED_BATCH_SIZES = {'MAPPO': 256}  # 默认值

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical, Normal
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MAPPOConfig:
    """MAPPO算法配置"""
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
    batch_size: int = 256
    buffer_size: int = 4000
    ppo_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # 其他参数
    normalize_advantages: bool = True
    use_gae: bool = True
    action_space: str = "continuous"  # "continuous" or "discrete"


class MAPPOActor(nn.Module):
    """MAPPO Actor网络 - 策略网络"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, 
                 action_space: str = "continuous"):
        super(MAPPOActor, self).__init__()
        
        self.action_space = action_space
        self.action_dim = action_dim
        
        # 共享特征层
        self.feature_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        if action_space == "continuous":
            # 连续动作空间 - 输出均值和标准差
            self.mean_layer = nn.Linear(hidden_dim, action_dim)
            self.log_std_layer = nn.Linear(hidden_dim, action_dim)
        else:
            # 离散动作空间 - 输出动作概率
            self.action_layer = nn.Linear(hidden_dim, action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.feature_layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        if self.action_space == "continuous":
            nn.init.orthogonal_(self.mean_layer.weight, gain=1)
            nn.init.constant_(self.mean_layer.bias, 0.0)
            nn.init.orthogonal_(self.log_std_layer.weight, gain=1)
            nn.init.constant_(self.log_std_layer.bias, 0.0)
        else:
            nn.init.orthogonal_(self.action_layer.weight, gain=1)
            nn.init.constant_(self.action_layer.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.distributions.Distribution:
        """前向传播"""
        features = self.feature_layers(state)
        
        if self.action_space == "continuous":
            # 连续动作分布
            mean = self.mean_layer(features)
            log_std = self.log_std_layer(features)
            std = torch.exp(log_std.clamp(-20, 2))
            
            return Normal(mean, std)
        else:
            # 离散动作分布
            logits = self.action_layer(features)
            return Categorical(logits=logits)
    
    def get_action_and_logprob(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """获取动作和对数概率"""
        dist = self.forward(state)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        if self.action_space == "continuous":
            # 对于连续动作，需要求和所有维度的log_prob
            log_prob = log_prob.sum(dim=-1)
        
        entropy = dist.entropy()
        if self.action_space == "continuous":
            entropy = entropy.sum(dim=-1)
        
        return action, log_prob, entropy


class MAPPOCritic(nn.Module):
    """MAPPO Critic网络 - 价值网络 (集中式训练)"""
    
    def __init__(self, global_state_dim: int, hidden_dim: int = 256):
        super(MAPPOCritic, self).__init__()
        
        self.value_network = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for layer in self.value_network:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0.0)
        
        # 最后一层使用较小的初始化
        nn.init.orthogonal_(self.value_network[-1].weight, gain=1)
    
    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.value_network(global_state)


class MAPPOBuffer:
    """MAPPO经验缓冲区"""
    
    def __init__(self, buffer_size: int, num_agents: int, state_dim: int, 
                 action_dim: int, global_state_dim: int):
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.ptr = 0
        self.size = 0
        
        # 为每个智能体存储数据
        self.states = {}
        self.actions = {}
        self.log_probs = {}
        self.rewards = {}
        self.dones = {}
        self.values = {}
        self.advantages = {}
        self.returns = {}
        
        # 全局状态
        self.global_states = np.zeros((buffer_size, global_state_dim), dtype=np.float32)
        
        # 初始化缓冲区
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        for agent_id in agent_ids:
            self.states[agent_id] = np.zeros((buffer_size, state_dim), dtype=np.float32)
            self.actions[agent_id] = np.zeros((buffer_size, action_dim), dtype=np.float32)
            self.log_probs[agent_id] = np.zeros(buffer_size, dtype=np.float32)
            self.rewards[agent_id] = np.zeros(buffer_size, dtype=np.float32)
            self.dones[agent_id] = np.zeros(buffer_size, dtype=np.float32)
            self.values[agent_id] = np.zeros(buffer_size, dtype=np.float32)
            self.advantages[agent_id] = np.zeros(buffer_size, dtype=np.float32)
            self.returns[agent_id] = np.zeros(buffer_size, dtype=np.float32)
    
    def store(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
              log_probs: Dict[str, float], rewards: Dict[str, float],
              dones: Dict[str, bool], values: Dict[str, float], global_state: np.ndarray):
        """存储一步经验"""
        if self.size < self.buffer_size:
            self.size += 1
        
        self.global_states[self.ptr] = global_state
        
        for agent_id in states.keys():
            self.states[agent_id][self.ptr] = states[agent_id]
            self.actions[agent_id][self.ptr] = actions[agent_id]
            self.log_probs[agent_id][self.ptr] = log_probs[agent_id]
            self.rewards[agent_id][self.ptr] = rewards[agent_id]
            self.dones[agent_id][self.ptr] = float(dones[agent_id])
            self.values[agent_id][self.ptr] = values[agent_id]
        
        self.ptr = (self.ptr + 1) % self.buffer_size
    
    def compute_advantages_and_returns(self, gamma: float, gae_lambda: float, use_gae: bool = True):
        """计算优势函数和回报"""
        for agent_id in self.states.keys():
            rewards = self.rewards[agent_id][:self.size]
            values = self.values[agent_id][:self.size]
            dones = self.dones[agent_id][:self.size]
            
            if use_gae:
                # 使用GAE计算优势函数
                advantages = np.zeros_like(rewards)
                last_gae = 0
                
                for t in reversed(range(self.size - 1)):
                    delta = rewards[t] + gamma * values[t + 1] * (1 - dones[t]) - values[t]
                    last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
                    advantages[t] = last_gae
                
                returns = advantages + values
            else:
                # 直接计算回报
                returns = np.zeros_like(rewards)
                running_return = 0
                
                for t in reversed(range(self.size)):
                    if dones[t]:
                        running_return = 0
                    running_return = rewards[t] + gamma * running_return
                    returns[t] = running_return
                
                advantages = returns - values
            
            self.advantages[agent_id][:self.size] = advantages
            self.returns[agent_id][:self.size] = returns
    
    def get_batch(self, batch_size: int) -> Dict:
        """获取训练批次"""
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        batch = {
            'global_states': self.global_states[indices],
            'states': {},
            'actions': {},
            'log_probs': {},
            'advantages': {},
            'returns': {},
            'values': {}
        }
        
        for agent_id in self.states.keys():
            batch['states'][agent_id] = self.states[agent_id][indices]
            batch['actions'][agent_id] = self.actions[agent_id][indices]
            batch['log_probs'][agent_id] = self.log_probs[agent_id][indices]
            batch['advantages'][agent_id] = self.advantages[agent_id][indices]
            batch['returns'][agent_id] = self.returns[agent_id][indices]
            batch['values'][agent_id] = self.values[agent_id][indices]
        
        return batch
    
    def clear(self):
        """清空缓冲区"""
        self.ptr = 0
        self.size = 0


class MAPPOAgent:
    """MAPPO智能体"""
    
    def __init__(self, agent_id: str, state_dim: int, action_dim: int, 
                 global_state_dim: int, mappo_config: MAPPOConfig):
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = mappo_config
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建网络
        self.actor = MAPPOActor(
            state_dim, action_dim, mappo_config.hidden_dim, mappo_config.action_space
        ).to(self.device)
        
        self.critic = MAPPOCritic(global_state_dim, mappo_config.hidden_dim).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=mappo_config.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=mappo_config.critic_lr)
        
        # 训练统计
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_losses = []
    
    def select_action(self, state: np.ndarray, training: bool = True):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, _ = self.actor.get_action_and_logprob(state_tensor)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def evaluate_state(self, global_state: np.ndarray) -> float:
        """评估状态价值"""
        global_state_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            value = self.critic(global_state_tensor)
        
        return value.cpu().item()


class MAPPOEnvironment:
    """MAPPO多智能体环境"""
    
    def __init__(self, action_space: str = "continuous"):
        self.config = MAPPOConfig()
        self.config.action_space = action_space
        
        # 性能优化 - 使用优化的批次大小
        self.optimized_batch_size = OPTIMIZED_BATCH_SIZES.get('MAPPO', self.config.batch_size)
        self.config.batch_size = self.optimized_batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 环境配置
        self.num_agents = 3
        self.state_dim = 20
        self.action_dim = 10 if action_space == "continuous" else 5
        self.global_state_dim = self.state_dim * self.num_agents
        
        # 创建智能体
        self.agents = {}
        agent_ids = ['vehicle_agent', 'rsu_agent', 'uav_agent']
        
        for agent_id in agent_ids:
            self.agents[agent_id] = MAPPOAgent(
                agent_id=agent_id,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                global_state_dim=self.global_state_dim,
                mappo_config=self.config
            )
        
        # 经验缓冲区
        self.buffer = MAPPOBuffer(
            buffer_size=self.config.buffer_size,
            num_agents=self.num_agents,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            global_state_dim=self.global_state_dim
        )
        
        # 训练统计
        self.episode_count = 0
        self.step_count = 0
        
        print("✓ MAPPO环境初始化完成")
        print(f"✓ 智能体数量: {self.num_agents}")
        print(f"✓ 动作空间: {action_space}")
        print(f"✓ 状态维度: {self.state_dim}")
        print(f"✓ 动作维度: {self.action_dim}")
    
    def get_actions(self, states: Dict[str, np.ndarray], training: bool = True) -> Tuple[Dict[str, np.ndarray], Dict[str, float]]:
        """获取所有智能体的动作"""
        actions = {}
        log_probs = {}
        
        for agent_id, agent in self.agents.items():
            if agent_id in states:
                action, log_prob = agent.select_action(states[agent_id], training)
                actions[agent_id] = action
                log_probs[agent_id] = log_prob
            else:
                # 默认动作
                if self.config.action_space == "continuous":
                    actions[agent_id] = np.zeros(self.action_dim)
                else:
                    actions[agent_id] = np.array([0])
                log_probs[agent_id] = 0.0
        
        return actions, log_probs
    
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
        for agent_id in self.agents.keys():
            agent_specific = np.random.randn(self.state_dim - len(base_state))
            states[agent_id] = np.concatenate([base_state, agent_specific])
        
        return states
    
    def get_global_state(self, states: Dict[str, np.ndarray]) -> np.ndarray:
        """获取全局状态"""
        global_state = []
        for agent_id in sorted(states.keys()):
            global_state.append(states[agent_id])
        return np.concatenate(global_state)
    
    def store_experience(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                        log_probs: Dict[str, float], rewards: Dict[str, float],
                        dones: Dict[str, bool], global_state: np.ndarray):
        """存储经验到缓冲区"""
        # 计算状态价值
        values = {}
        for agent_id, agent in self.agents.items():
            values[agent_id] = agent.evaluate_state(global_state)
        
        # 存储到缓冲区
        self.buffer.store(states, actions, log_probs, rewards, dones, values, global_state)
        self.step_count += 1
    
    def train_step(self, states: Dict[str, np.ndarray], actions: Dict[str, np.ndarray],
                   rewards: Dict[str, float], next_states: Dict[str, np.ndarray],
                   dones: Dict[str, bool]) -> Dict:
        """执行一步训练"""
        # 这里暂时不实现，因为MAPPO使用不同的训练模式
        # 实际使用中会调用update方法进行批量更新
        return {}
    
    def update(self) -> Dict:
        """PPO更新 (在episode结束后调用)"""
        if self.buffer.size < self.config.batch_size:
            return {}
        
        # 计算优势函数和回报
        self.buffer.compute_advantages_and_returns(
            self.config.gamma, self.config.gae_lambda, self.config.use_gae
        )
        
        # 标准化优势函数
        if self.config.normalize_advantages:
            for agent_id in self.agents.keys():
                advantages = self.buffer.advantages[agent_id][:self.buffer.size]
                self.buffer.advantages[agent_id][:self.buffer.size] = (
                    (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                )
        
        training_info = {}
        
        # PPO更新循环
        for epoch in range(self.config.ppo_epochs):
            batch = self.buffer.get_batch(self.config.batch_size)
            
            # 为每个智能体更新网络
            for agent_id, agent in self.agents.items():
                actor_loss, critic_loss, entropy_loss = self._update_agent(agent, batch, agent_id)
                
                if agent_id not in training_info:
                    training_info[agent_id] = {
                        'actor_loss': [],
                        'critic_loss': [],
                        'entropy_loss': []
                    }
                
                training_info[agent_id]['actor_loss'].append(actor_loss)
                training_info[agent_id]['critic_loss'].append(critic_loss)
                training_info[agent_id]['entropy_loss'].append(entropy_loss)
        
        # 清空缓冲区
        self.buffer.clear()
        
        # 计算平均损失
        for agent_id in training_info.keys():
            for key in training_info[agent_id].keys():
                training_info[agent_id][key] = np.mean(training_info[agent_id][key])
        
        return training_info
    
    def _update_agent(self, agent: MAPPOAgent, batch: Dict, agent_id: str) -> Tuple[float, float, float]:
        """更新单个智能体"""
        # 准备数据
        states = torch.FloatTensor(batch['states'][agent_id]).to(agent.device)
        actions = torch.FloatTensor(batch['actions'][agent_id]).to(agent.device)
        old_log_probs = torch.FloatTensor(batch['log_probs'][agent_id]).to(agent.device)
        advantages = torch.FloatTensor(batch['advantages'][agent_id]).to(agent.device)
        returns = torch.FloatTensor(batch['returns'][agent_id]).to(agent.device)
        global_states = torch.FloatTensor(batch['global_states']).to(agent.device)
        
        # 计算新的策略分布
        _, new_log_probs, entropy = agent.actor.get_action_and_logprob(states, actions)
        
        # 计算比率
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 熵损失
        entropy_loss = -entropy.mean()
        
        # 总Actor损失
        total_actor_loss = actor_loss + self.config.entropy_coef * entropy_loss
        
        # 更新Actor
        agent.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), self.config.max_grad_norm)
        agent.actor_optimizer.step()
        
        # Critic损失
        values = agent.critic(global_states).squeeze()
        critic_loss = F.mse_loss(values, returns)
        
        # 更新Critic
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), self.config.max_grad_norm)
        agent.critic_optimizer.step()
        
        return actor_loss.item(), critic_loss.item(), entropy_loss.item()
    
    def save_models(self, filepath: str):
        """保存所有智能体模型"""
        import os
        os.makedirs(filepath, exist_ok=True)
        
        for agent_id, agent in self.agents.items():
            torch.save({
                'actor_state_dict': agent.actor.state_dict(),
                'critic_state_dict': agent.critic.state_dict(),
                'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
                'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
            }, f"{filepath}/mappo_{agent_id}.pth")
        
        print(f"✓ MAPPO模型已保存到: {filepath}")
    
    def load_models(self, filepath: str):
        """加载所有智能体模型"""
        for agent_id, agent in self.agents.items():
            checkpoint = torch.load(f"{filepath}/mappo_{agent_id}.pth", 
                                  map_location=agent.device)
            
            agent.actor.load_state_dict(checkpoint['actor_state_dict'])
            agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        print(f"✓ MAPPO模型已加载: {filepath}")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        stats = {}
        for agent_id, agent in self.agents.items():
            stats[agent_id] = {
                'actor_loss_avg': np.mean(agent.actor_losses[-10:]) if agent.actor_losses else 0.0,
                'critic_loss_avg': np.mean(agent.critic_losses[-10:]) if agent.critic_losses else 0.0,
                'entropy_loss_avg': np.mean(agent.entropy_losses[-10:]) if agent.entropy_losses else 0.0
            }
        
        return stats
    
    def reset_hidden_states(self):
        """重置隐藏状态 (为MAPPO不需要，但保持接口兼容性)"""
        pass