"""
单智能体算法模块初始化文件
包含多种经典深度强化学习算法实现
"""

# DDPG算法
from .ddpg import DDPGEnvironment, DDPGAgent, DDPGActor, DDPGCritic, DDPGReplayBuffer

# TD3算法  
from .td3 import TD3Environment, TD3Agent, TD3Actor, TD3Critic, TD3ReplayBuffer
from .td3_latency_energy import TD3LatencyEnergyEnvironment
from .td3_hierarchical import TD3HierarchicalEnvironment, HierarchicalTD3Config
from .td3_hybrid_fusion import CAMTD3Environment

# DQN算法
from .dqn import DQNEnvironment, DQNAgent, DQNNetwork, DQNReplayBuffer

# PPO算法
from .ppo import PPOEnvironment, PPOAgent, PPOActor, PPOCritic, PPOBuffer

# SAC算法
from .sac import SACEnvironment, SACAgent, SACActor, SACCritic, SACReplayBuffer

# Enhanced TD3 - 5项高级优化
from .enhanced_td3_config import (
    EnhancedTD3Config,
    create_baseline_config,
    create_full_enhanced_config,
    create_queue_focused_config,
    create_exploration_focused_config,
    create_dynamic_topology_config,
)
from .enhanced_td3_agent import EnhancedTD3Agent
from .enhanced_td3_wrapper import EnhancedTD3Wrapper, EnhancedTD3Environment, EnhancedCAMTD3Environment
from .quantile_critic import DistributionalCritic, QuantileNetwork, QuantileHuberLoss
from .queue_aware_replay import QueueAwareReplayBuffer
from .queue_dynamics_model import QueueDynamicsModel, ModelBasedRollout, ModelTrainer
from .gat_router import GATRouterActor, GATLayer, VehicleRSUAttention, RSURSUCollaborativeAttention

__all__ = [
    # DDPG
    'DDPGEnvironment', 'DDPGAgent', 'DDPGActor', 'DDPGCritic', 'DDPGReplayBuffer',
    
    # TD3
    'TD3Environment', 'TD3Agent', 'TD3Actor', 'TD3Critic', 'TD3ReplayBuffer', 'TD3LatencyEnergyEnvironment',
    'TD3HierarchicalEnvironment', 'HierarchicalTD3Config', 'CAMTD3Environment',
    
    # Enhanced TD3
    'EnhancedTD3Agent', 'EnhancedTD3Config',
    'EnhancedTD3Wrapper', 'EnhancedTD3Environment', 'EnhancedCAMTD3Environment',
    'create_baseline_config', 'create_full_enhanced_config', 'create_queue_focused_config',
    'create_exploration_focused_config', 'create_dynamic_topology_config',
    'DistributionalCritic', 'QuantileNetwork', 'QuantileHuberLoss',
    'QueueAwareReplayBuffer', 'QueueDynamicsModel', 'ModelBasedRollout', 'ModelTrainer',
    'GATRouterActor', 'GATLayer', 'VehicleRSUAttention', 'RSURSUCollaborativeAttention',
    
    # DQN
    'DQNEnvironment', 'DQNAgent', 'DQNNetwork', 'DQNReplayBuffer',
    
    # PPO
    'PPOEnvironment', 'PPOAgent', 'PPOActor', 'PPOCritic', 'PPOBuffer',
    
    # SAC
    'SACEnvironment', 'SACAgent', 'SACActor', 'SACCritic', 'SACReplayBuffer'
]
