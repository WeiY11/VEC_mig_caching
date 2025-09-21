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
        self.task_compute_density = 500  # cycles/bit - 符合内存规范
        self.arrival_rate = 1.35  # tasks/second - 精细调整为理想负载
        
        # 统一的任务数据大小配置 (bytes)
        self.data_size_range = (5e6, 25e6)  # 5MB - 25MB - 符合内存规范
        self.task_data_size_range = self.data_size_range  # 兼容性别名
        
        # 计算周期配置 (自动计算，确保一致性)
        self.compute_cycles_range = (1e8, 1e10)  # cycles
        
        # 截止时间配置
        self.deadline_range = (1.0, 10.0)  # seconds
        
        # 输出比例配置
        self.task_output_ratio = 0.05  # 输出大小是输入大小的5%
        
        # 任务类型阈值 (时隙数)
        self.delay_thresholds = {
            'extremely_sensitive': 2,    # τ₁ = 2 时隙
            'sensitive': 5,              # τ₂ = 5 时隙 
            'moderately_tolerant': 10,   # τ₃ = 10 时隙
        }
    
    def get_task_type(self, max_delay_slots: int) -> int:
        """
        根据最大延迟时隙数确定任务类型
        对应论文第3.1节任务分类框架
        
        Args:
            max_delay_slots: 任务最大可容忍延迟时隙数
            
        Returns:
            任务类型值 (1-4)
        """
        if max_delay_slots <= self.delay_thresholds['extremely_sensitive']:
            return 1  # EXTREMELY_DELAY_SENSITIVE
        elif max_delay_slots <= self.delay_thresholds['sensitive']:
            return 2  # DELAY_SENSITIVE
        elif max_delay_slots <= self.delay_thresholds['moderately_tolerant']:
            return 3  # MODERATELY_DELAY_TOLERANT
        else:
            return 4  # DELAY_TOLERANT

class ComputeConfig:
    """计算配置类"""
    
    def __init__(self):
        self.parallel_efficiency = 0.8
        
        # 车辆能耗参数 - 对应论文式(5)-(9)
        self.vehicle_kappa1 = 1e-28
        self.vehicle_kappa2 = 1e-26
        self.vehicle_static_power = 0.5  # W
        self.vehicle_idle_power = 0.1   # W (空闲功耗)
        
        # RSU能耗参数 - 对应论文式(20)-(21)
        self.rsu_kappa = 1e-27
        self.rsu_kappa2 = 1e-26
        self.rsu_static_power = 2.0  # W
        
        # UAV能耗参数 - 对应论文式(25)-(30)
        self.uav_kappa = 1e-27
        self.uav_kappa3 = 1e-27  # 修复缺失的参数
        self.uav_static_power = 1.0  # W
        self.uav_hover_power = 50.0  # W (悬停功耗)
        
        # CPU频率范围 - 符合内存规范
        self.vehicle_cpu_freq_range = (8e9, 25e9)  # 8-25 GHz
        self.rsu_cpu_freq_range = (45e9, 55e9)  # 50 GHz左右
        self.uav_cpu_freq_range = (7e9, 9e9)  # 8 GHz左右
        
        # 默认CPU频率 - 符合内存规范
        self.vehicle_default_freq = 16e9  # 16 GHz
        self.rsu_default_freq = 50e9  # 50 GHz
        self.uav_default_freq = 8e9  # 8 GHz
        
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
        self.time_slot_duration = 0.2  # seconds - 优化为更合理的时隙长度
        self.bandwidth = 20e6  # Hz
        self.carrier_frequency = 2.4e9  # Hz
        self.noise_power = -174  # dBm/Hz
        self.path_loss_exponent = 2.0
        self.coverage_radius = 1000  # meters
        self.interference_threshold = 0.1
        self.handover_threshold = 0.2
        
        # 节点数量配置
        self.num_vehicles = 12  # 恢复到原始设置
        self.num_rsus = 6       # 恢复到原始设置
        self.num_uavs = 2       # 恢复到原始设置，符合论文要求
        
        # 网络拓扑参数
        self.area_width = 2500  # meters - 缩小仿真区域
        self.area_height = 2500  # meters
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
        
        # 带宽配置 - 符合内存规范
        self.total_bandwidth = 50e6  # 50 MHz
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
        
        # 迁移触发阈值
        self.rsu_overload_threshold = 0.8
        self.uav_overload_threshold = 0.7
        self.rsu_underload_threshold = 0.3
        
        # UAV迁移参数
        self.uav_min_battery = 0.2  # 20%
        self.migration_delay_threshold = 1.0  # seconds
        self.max_migration_distance = 1000  # meters
        
        # 迁移成本参数
        self.migration_alpha_comp = 0.4  # 计算成本权重
        self.migration_alpha_tx = 0.3    # 传输成本权重
        self.migration_alpha_lat = 0.3   # 延迟成本权重
        
        self.migration_energy_cost = 0.1  # J per bit
        self.migration_time_penalty = 0.05  # seconds
        
        # 冷却期参数
        self.cooldown_period = 10.0  # seconds

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
        self.num_vehicles = 12  # 恢复到原始设置
        self.num_rsus = 6       # 恢复到原始设置
        self.num_uavs = 2       # 恢复到原始设置，符合论文要求
        
        # 仿真配置
        self.simulation_time = 1000
        self.time_slot = 0.2
        
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