#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置适配器
将项目的config/system_config.py适配为xuance和启发式算法所需的配置格式

【功能】
1. 读取现有系统配置
2. 转换为xuance算法配置字典
3. 支持命令行参数覆盖
4. 确保所有算法使用一致的系统配置
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from config import config


def create_base_config(num_vehicles: int = 12, seed: int = 42) -> Dict[str, Any]:
    """
    创建基础配置字典（从system_config读取）
    
    【参数】
    - num_vehicles: 车辆数量（允许覆盖）
    - seed: 随机种子
    
    【返回】
    - 基础配置字典
    """
    base_config = {
        # 网络拓扑（固定RSU和UAV）
        'num_vehicles': num_vehicles,
        'num_rsus': 4,  # 固定
        'num_uavs': 2,  # 固定
        
        # 仿真参数
        'simulation_time': getattr(config, 'simulation_time', 1000),
        'time_slot': getattr(config.network, 'time_slot_duration', 0.2),
        'task_arrival_rate': getattr(config.task, 'arrival_rate', 2.5),
        
        # 资源配置
        'computation_capacity': 800,
        'bandwidth': 15,
        'cache_capacity': 80,
        'transmission_power': 0.15,
        'computation_power': 1.2,
        
        # 高负载模式
        'high_load_mode': True,
        'task_complexity_multiplier': 1.5,
        'rsu_load_divisor': 4.0,
        'uav_load_divisor': 2.0,
        'enhanced_task_generation': True,
        
        # 随机种子
        'random_seed': seed,
    }
    
    return base_config


def create_xuance_config(
    algorithm: str,
    num_episodes: int = 200,
    seed: int = 42,
    num_vehicles: int = 12,
    max_steps: int = 100
) -> Dict[str, Any]:
    """
    创建xuance算法配置
    
    【参数】
    - algorithm: 算法名称（TD3, DDPG, SAC, PPO, DQN）
    - num_episodes: 训练轮次
    - seed: 随机种子
    - num_vehicles: 车辆数量
    - max_steps: 每轮最大步数
    
    【返回】
    - xuance配置字典
    """
    # 基础环境配置
    base_config = create_base_config(num_vehicles, seed)
    
    # 状态和动作维度（固定拓扑）
    # 状态维度: num_vehicles*5 + num_rsus*5 + num_uavs*5 + 8
    state_dim = num_vehicles * 5 + 4 * 5 + 2 * 5 + 8
    
    # 动作维度: 3(分配) + 4(RSU) + 2(UAV) + 7(控制) = 16
    action_dim = 16
    
    # xuance算法配置
    xuance_config = {
        # 环境配置
        'env_config': base_config,
        'state_dim': state_dim,
        'action_dim': action_dim,
        
        # 训练配置
        'num_episodes': num_episodes,
        'max_steps_per_episode': max_steps,
        'seed': seed,
        
        # 算法通用参数（从config.rl读取）
        'hidden_dim': getattr(config.rl, 'hidden_dim', 256),
        'lr': getattr(config.rl, 'lr', 0.0003),
        'actor_lr': getattr(config.rl, 'actor_lr', 0.0003),
        'critic_lr': getattr(config.rl, 'critic_lr', 0.0003),
        'gamma': getattr(config.rl, 'gamma', 0.99),
        'tau': getattr(config.rl, 'tau', 0.005),
        'batch_size': getattr(config.rl, 'batch_size', 128),
        'buffer_size': getattr(config.rl, 'buffer_size', 100000),
        
        # 算法特定参数
        'algorithm': algorithm,
    }
    
    # 算法特定配置
    if algorithm.upper() in ['TD3', 'DDPG']:
        xuance_config.update({
            'policy_noise': getattr(config.rl, 'policy_noise', 0.1),
            'noise_clip': getattr(config.rl, 'noise_clip', 0.3),
            'policy_delay': getattr(config.rl, 'policy_delay', 2),
            'exploration_noise': getattr(config.rl, 'exploration_noise', 0.05),
        })
    
    elif algorithm.upper() == 'SAC':
        xuance_config.update({
            'alpha': 0.2,  # 熵温度系数
            'auto_alpha': True,  # 自动调整alpha
            'target_entropy': -action_dim,
        })
    
    elif algorithm.upper() == 'PPO':
        xuance_config.update({
            'clip_epsilon': 0.2,
            'value_coef': 0.5,
            'entropy_coef': 0.01,
            'gae_lambda': 0.95,
            'n_epochs': 10,
        })
    
    elif algorithm.upper() == 'DQN':
        xuance_config.update({
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 10,
        })
    
    return xuance_config


def apply_config_overrides(overrides: Dict[str, Any]):
    """
    应用配置覆盖（通过环境变量）
    
    【参数】
    - overrides: 要覆盖的配置字典
    """
    if overrides:
        os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)


def get_algorithm_save_dir(algorithm: str, base_dir: str = None) -> Path:
    """
    获取算法结果保存目录
    
    【参数】
    - algorithm: 算法名称
    - base_dir: 基础目录（默认为baseline_comparison/results）
    
    【返回】
    - 保存目录路径
    """
    if base_dir is None:
        base_dir = Path(__file__).parent.parent.parent / "results"
    else:
        base_dir = Path(base_dir)
    
    # 创建算法特定目录
    save_dir = base_dir / algorithm.lower()
    save_dir.mkdir(parents=True, exist_ok=True)
    
    return save_dir


if __name__ == "__main__":
    # 测试配置生成
    print("="*80)
    print("配置适配器测试")
    print("="*80)
    
    algorithms = ['TD3', 'DDPG', 'SAC', 'PPO', 'DQN']
    
    for algo in algorithms:
        print(f"\n{algo} 配置:")
        cfg = create_xuance_config(algo, num_episodes=200, seed=42, num_vehicles=12)
        print(f"  状态维度: {cfg['state_dim']}")
        print(f"  动作维度: {cfg['action_dim']}")
        print(f"  训练轮次: {cfg['num_episodes']}")
        print(f"  隐藏层维度: {cfg['hidden_dim']}")
        print(f"  学习率: {cfg['lr']}")
        print(f"  保存目录: {get_algorithm_save_dir(algo)}")
    
    print("\n" + "="*80)
    print("配置适配器测试完成！")
    print("="*80)








