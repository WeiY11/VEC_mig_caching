"""
CAMTD3深度强化学习算法 - 对应算法创新优化
Context-Aware Multi-agent TD3 (基于注意力机制的改进版TD3)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from copy import deepcopy

from config import config
from algorithms.matd3 import ReplayBuffer, MATD3Agent

class FeatureAttention(nn.Module):
    """
    特征注意力模块 (Feature Attention Module)
    用于动态加权状态特征的重要性
    """
    def __init__(self, input_dim: int, reduction_ratio: int = 4):
        super(FeatureAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim // reduction_ratio)
        self.fc2 = nn.Linear(input_dim // reduction_ratio, input_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Squeeze-and-Excitation style attention
        # x: [batch_size, input_dim]
        attention = F.relu(self.fc1(x))
        attention = self.fc2(attention)
        attention = self.sigmoid(attention)
        return x * attention

class CAMActor(nn.Module):
    """
    基于注意力机制的Actor网络
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CAMActor, self).__init__()
        
        # 特征注意力层
        self.attention = FeatureAttention(state_dim)
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
        # 初始化权重
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)
        self.fc3.bias.data.uniform_(-3e-3, 3e-3)
    
    def forward(self, state):
        # 应用注意力机制
        x = self.attention(state)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # 输出范围[-1, 1]
        return x

class CAMCritic(nn.Module):
    """
    基于注意力机制的Critic网络
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CAMCritic, self).__init__()
        
        # 特征注意力层 (仅对状态应用)
        self.state_attention = FeatureAttention(state_dim)
        
        # Q1网络
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2网络 (Twin网络)
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, state, action):
        # 对状态应用注意力
        state_weighted = self.state_attention(state)
        x = torch.cat([state_weighted, action], dim=-1)
        
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
        state_weighted = self.state_attention(state)
        x = torch.cat([state_weighted, action], dim=-1)
        q1 = F.relu(self.fc1_q1(x))
        q1 = F.relu(self.fc2_q1(q1))
        q1 = self.fc3_q1(q1)
        return q1

class CAMTD3Agent(MATD3Agent):
    """
    CAMTD3智能体 - 继承自MATD3Agent，替换网络结构
    """
    def __init__(self, agent_id: str, state_dim: int, action_dim: int):
        super().__init__(agent_id, state_dim, action_dim)
        
        # 重新初始化网络为CAM版本
        self.actor = CAMActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.actor_target = CAMActor(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic = CAMCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        self.critic_target = CAMCritic(state_dim, action_dim, self.hidden_dim).to(self.device)
        
        # 复制目标网络权重
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 重新初始化优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
        # 重新初始化调度器
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=5000, gamma=0.95)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=5000, gamma=0.95)
        
        print(f"CAMTD3Agent {agent_id} initialized with Feature Attention")
