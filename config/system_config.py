#!/usr/bin/env python3
"""
系统配置
"""

import os
from typing import Dict, Any

class ExperimentConfig:
    """实验配置类"""
    
    def __init__(self):
        self.num_episodes = 1000
        self.num_runs = 3
        self.save_interval = 100
        self.eval_interval = 50
        self.log_interval = 10
        self.max_steps_per_episode = 200
        self.warmup_episodes = 10
        self.use_timestamp = True
        self.timestamp_format = "%Y%m%d_%H%M%S"

class RLConfig:
    """强化学习配置类"""
    
    def __init__(self):
        self.num_agents = 3
        self.state_dim = 20
        self.action_dim = 10
        self.hidden_dim = 256
        self.lr = 0.0003
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.memory_size = 100000
        self.noise_std = 0.1
        self.policy_delay = 2
        self.noise_clip = 0.5
        self.exploration_noise = 0.1
        self.policy_noise = 0.2
        self.target_noise = 0.2
        self.update_freq = 1
        self.buffer_size = 100000
        self.warmup_steps = 1000
        
        # 奖励权重
        self.reward_weight_delay = 0.3
        self.reward_weight_energy = 0.2
        self.reward_weight_completion = 0.3
        self.reward_weight_cache = 0.2
        self.reward_weight_loss = 0.1

class QueueConfig:
    """队列配置类"""
    
    def __init__(self):
        self.max_lifetime = 5
        self.max_queue_size = 100
        self.priority_levels = 4
        self.aging_factor = 0.1

class TaskConfig:
    """任务配置类"""
    
    def __init__(self):
        self.num_priority_levels = 4
        self.task_compute_density = 1000  # cycles/bit
        self.arrival_rate = 2.0  # tasks/second
        self.data_size_range = (1e5, 1e7)  # bytes
        self.compute_cycles_range = (1e8, 1e10)  # cycles
        self.deadline_range = (1.0, 10.0)  # seconds

class ComputeConfig:
    """计算配置类"""
    
    def __init__(self):
        self.parallel_efficiency = 0.8
        self.vehicle_kappa1 = 1e-28
        self.vehicle_kappa2 = 1e-26
        self.vehicle_static_power = 0.5  # W
        self.rsu_kappa = 1e-27
        self.rsu_kappa2 = 1e-26
        self.uav_kappa = 1e-27
        
        # CPU频率范围
        self.vehicle_cpu_freq_range = (1e9, 3e9)  # 1-3 GHz
        self.rsu_cpu_freq_range = (2e9, 5e9)  # 2-5 GHz
        self.uav_cpu_freq_range = (1e9, 2e9)  # 1-2 GHz
        
        # 默认CPU频率
        self.vehicle_default_freq = 2e9  # 2 GHz
        self.rsu_default_freq = 3e9  # 3 GHz
        self.uav_default_freq = 1.5e9  # 1.5 GHz
        
        # 节点CPU频率（用于初始化）
        self.vehicle_cpu_freq = self.vehicle_default_freq
        self.rsu_cpu_freq = self.rsu_default_freq
        self.uav_cpu_freq = self.uav_default_freq
        
        # 内存配置
        self.vehicle_memory_size = 8e9  # 8 GB
        self.rsu_memory_size = 32e9  # 32 GB
        self.uav_memory_size = 4e9  # 4 GB
        
        # UAV特殊配置
        self.uav_hover_power = 50.0  # W

class NetworkConfig:
    """网络配置类"""
    
    def __init__(self):
        self.time_slot_duration = 0.1  # seconds
        self.bandwidth = 20e6  # Hz
        self.carrier_frequency = 2.4e9  # Hz
        self.noise_power = -174  # dBm/Hz
        self.path_loss_exponent = 2.0
        self.coverage_radius = 1000  # meters
        self.interference_threshold = 0.1
        self.handover_threshold = 0.2
        
        # 节点数量配置
        self.num_vehicles = 50
        self.num_rsus = 10
        self.num_uavs = 5
        
        # 网络拓扑参数
        self.area_width = 5000  # meters
        self.area_height = 5000  # meters
        self.min_distance = 50  # meters
        
        # 连接参数
        self.max_connections_per_node = 10
        self.connection_timeout = 30  # seconds

class CommunicationConfig:
    """通信配置类"""
    
    def __init__(self):
        self.vehicle_tx_power = 23  # dBm
        self.rsu_tx_power = 30  # dBm
        self.uav_tx_power = 20  # dBm
        self.circuit_power = 0.1  # W
        self.noise_figure = 9  # dB
        
        # 带宽配置
        self.total_bandwidth = 20e6  # 20 MHz
        self.channel_bandwidth = 1e6  # 1 MHz per channel
        self.uplink_bandwidth = 10e6  # 10 MHz
        self.downlink_bandwidth = 10e6  # 10 MHz
        
        # 传播参数
        self.carrier_frequency = 2.4e9  # 2.4 GHz
        self.speed_of_light = 3e8  # m/s
        self.antenna_gain = 1.0  # dBi
        
        # 调制参数
        self.modulation_order = 4  # QPSK
        self.coding_rate = 0.5

class MigrationConfig:
    """迁移配置类"""
    
    def __init__(self):
        self.migration_bandwidth = 100e6  # bps
        self.migration_threshold = 0.8
        self.migration_cost_factor = 0.1
        
        # UAV迁移参数
        self.uav_min_battery = 0.2  # 20%
        self.migration_delay_threshold = 1.0  # seconds
        self.max_migration_distance = 1000  # meters
        
        # 迁移成本参数
        self.migration_energy_cost = 0.1  # J per bit
        self.migration_time_penalty = 0.05  # seconds

class CacheConfig:
    """缓存配置类"""
    
    def __init__(self):
        # 缓存容量配置
        self.vehicle_cache_capacity = 1e9  # 1 GB
        self.rsu_cache_capacity = 10e9  # 10 GB
        self.uav_cache_capacity = 2e9  # 2 GB
        
        # 缓存策略配置
        self.cache_replacement_policy = 'LRU'  # LRU, LFU, RANDOM
        self.cache_hit_threshold = 0.8
        self.cache_update_interval = 1.0  # seconds
        
        # 缓存预测参数
        self.prediction_window = 10  # time slots
        self.popularity_decay_factor = 0.9
        self.request_history_size = 100

class SystemConfig:
    """系统配置类"""
    
    def __init__(self):
        # 基本系统配置
        self.device = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
        self.num_threads = int(os.environ.get('NUM_THREADS', '4'))
        self.random_seed = int(os.environ.get('RANDOM_SEED', '42'))
        
        # 网络配置
        self.num_vehicles = 12
        self.num_rsus = 6
        self.num_uavs = 2
        
        # 仿真配置
        self.simulation_time = 1000
        self.time_slot = 0.1
        
        # 性能配置
        self.enable_performance_optimization = True
        self.batch_size_optimization = True
        self.parallel_environments = 6
        
        # 子配置模块
        self.queue = QueueConfig()
        self.task = TaskConfig()
        self.compute = ComputeConfig()
        self.network = NetworkConfig()
        self.communication = CommunicationConfig()
        self.migration = MigrationConfig()
        self.cache = CacheConfig()
        
        # 实验配置
        self.experiment = ExperimentConfig()
        
        # 强化学习配置
        self.rl = RLConfig()
        
    def get_config_dict(self) -> Dict[str, Any]:
        """获取配置字典"""
        return {
            'device': self.device,
            'num_threads': self.num_threads,
            'random_seed': self.random_seed,
            'num_vehicles': self.num_vehicles,
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs,
            'simulation_time': self.simulation_time,
            'time_slot': self.time_slot,
            'enable_performance_optimization': self.enable_performance_optimization,
            'batch_size_optimization': self.batch_size_optimization,
            'parallel_environments': self.parallel_environments
        }
    
    def update_config(self, **kwargs):
        """更新配置"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 全局配置实例
config = SystemConfig()