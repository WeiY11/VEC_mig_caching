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
        
        # 奖励权重 - 对应论文目标函数权重
        self.reward_weight_delay = 0.4     # ω_T: 时延权重
        self.reward_weight_energy = 0.3    # ω_E: 能耗权重
        self.reward_weight_loss = 0.3      # ω_D: 数据丢失权重
        self.reward_weight_completion = 0.2
        self.reward_weight_cache = 0.1

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
        self.task_compute_density = 400  # 🔧 降低计算密度，适应现实算力
        self.arrival_rate = 2.5   # tasks/second - 🚀 12车辆极高负载优化
        
        # 🔧 重新设计：任务参数 - 分层设计不同复杂度任务
        self.data_size_range = (0.5e6/8, 15e6/8)  # 0.5-15 Mbits = 0.0625-1.875 MB
        self.task_data_size_range = self.data_size_range  # 兼容性别名
        
        # 任务类型特化参数
        self.task_type_specs = {
            1: {'data_range': (0.5e6/8, 3e6/8),   'compute_density': 300},  # 极敏感：小数据,低密度
            2: {'data_range': (2e6/8, 8e6/8),     'compute_density': 400},  # 敏感：中数据,中密度  
            3: {'data_range': (5e6/8, 12e6/8),    'compute_density': 500},  # 中容忍：大数据,中高密度
            4: {'data_range': (8e6/8, 15e6/8),    'compute_density': 600}   # 容忍：最大数据,高密度
        }
        
        # 计算周期配置 (自动计算，确保一致性)
        self.compute_cycles_range = (1e8, 1e10)  # cycles
        
        # 截止时间配置
        self.deadline_range = (1.0, 10.0)  # seconds
        
        # 输出比例配置
        self.task_output_ratio = 0.05  # 输出大小是输入大小的5%
        
        # 🔧 重新设计：任务类型阈值 - 基于12GHz RSU实际处理能力
        self.delay_thresholds = {
            'extremely_sensitive': 4,    # τ₁ = 4 时隙 = 0.8s (RSU快速处理)
            'sensitive': 10,             # τ₂ = 10 时隙 = 2.0s (Vehicle处理)
            'moderately_tolerant': 25,   # τ₃ = 25 时隙 = 5.0s (UAV/复杂任务)
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
        
        # 🔧 修复：车辆能耗参数 - 基于实际硬件校准
        self.vehicle_kappa1 = 5.12e-31  # 基于Intel NUC i7实际校准
        self.vehicle_kappa2 = 2.40e-20  # 频率平方项系数
        self.vehicle_static_power = 8.0  # W (现实车载芯片静态功耗)
        self.vehicle_idle_power = 3.5   # W (空闲功耗)
        
        # 🔧 修复：RSU能耗参数 - 基于12GHz边缘服务器校准
        self.rsu_kappa = 2.8e-31  # 12GHz高性能CPU的功耗系数
        self.rsu_kappa2 = 2.8e-31
        self.rsu_static_power = 25.0  # W (12GHz边缘服务器静态功耗)
        
        # 🔧 修复：UAV能耗参数 - 基于实际UAV硬件校准
        self.uav_kappa = 8.89e-31  # 功耗受限的UAV芯片
        self.uav_kappa3 = 8.89e-31  # 修复后参数
        self.uav_static_power = 2.5  # W (轻量化设计)
        self.uav_hover_power = 25.0  # W (更合理的悬停功耗)
        
        # CPU频率范围 - 符合内存规范
        self.vehicle_cpu_freq_range = (8e9, 25e9)  # 8-25 GHz
        self.rsu_cpu_freq_range = (45e9, 55e9)  # 50 GHz左右
        self.uav_cpu_freq_range = (7e9, 9e9)  # 8 GHz左右
        
        # 🔧 修复：调整为现实硬件频率
        self.vehicle_default_freq = 2.5e9  # 2.5 GHz (Tesla FSD等车载芯片)
        self.rsu_default_freq = 12e9  # 12 GHz (边缘服务器高性能CPU)
        self.uav_default_freq = 1.8e9  # 1.8 GHz (功耗限制下的UAV)
        
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
    """3GPP标准通信配置类"""
    
    def __init__(self):
        # 3GPP标准发射功率
        self.vehicle_tx_power = 23.0  # dBm (200mW) - 3GPP标准
        self.rsu_tx_power = 46.0      # dBm (40W) - 3GPP标准
        self.uav_tx_power = 30.0      # dBm (1W) - 3GPP标准
        self.circuit_power = 0.1      # W
        self.noise_figure = 9.0       # dB - 3GPP标准
        
        # 3GPP标准带宽配置
        self.total_bandwidth = 20e6   # 20 MHz - 3GPP标准
        self.channel_bandwidth = 1e6  # 1 MHz per channel
        self.uplink_bandwidth = 10e6  # 10 MHz
        self.downlink_bandwidth = 10e6  # 10 MHz
        
        # 3GPP标准传播参数
        self.carrier_frequency = 2.0e9  # 2 GHz - 3GPP标准频率
        self.speed_of_light = 3e8       # m/s
        self.thermal_noise_density = -174.0  # dBm/Hz - 3GPP标准
        
        # 3GPP标准天线增益
        self.antenna_gain_rsu = 15.0     # dBi
        self.antenna_gain_uav = 5.0      # dBi
        self.antenna_gain_vehicle = 3.0  # dBi
        
        # 3GPP标准路径损耗参数
        self.los_threshold = 50.0        # m - 3GPP TS 38.901
        self.los_decay_factor = 100.0    # m
        self.shadowing_std_los = 4.0     # dB
        self.shadowing_std_nlos = 8.0    # dB
        
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
        # 队列/切换阈值（用于车辆跟随与过载切换）
        self.follow_handover_distance = 30.0  # meters，车辆跟随触发的最小距离改善
        self.queue_switch_diff = 3            # 个，目标RSU较当前RSU队列至少少N个才切换
        self.rsu_queue_overload_len = 8       # 个，认为RSU队列过载的长度阈值
        self.service_jitter_ratio = 0.2       # 服务速率±20%抖动
        
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
        
        # 🚀 12车辆高负载场景网络配置
        self.num_vehicles = 12  # 保持12车辆，通过其他方式创造高负载
        self.num_rsus = 6       # 保持RSU数量
        self.num_uavs = 2       # 保持UAV数量
        
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