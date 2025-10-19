#!/usr/bin/env python3
"""
系统配置
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


@dataclass(frozen=True)
class TaskProfileSpec:
    """描述单类任务的数据范围与计算密度"""
    task_type: int
    data_range: Tuple[float, float]
    compute_density: float


@dataclass(frozen=True)
class TaskScenarioSpec:
    """应用场景及其对应的任务类型与额外参数"""
    name: str
    min_deadline: float
    max_deadline: float
    task_type: int
    relax_factor: float
    weight: float

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
        self.noise_std = 0.05          # 降低噪声标准差
        self.policy_delay = 2
        self.noise_clip = 0.3           # 降低噪声裁剪
        self.exploration_noise = 0.05   # 降低探索噪声
        self.policy_noise = 0.1         # 降低策略噪声
        self.target_noise = 0.1         # 降低目标噪声
        self.update_freq = 1
        self.buffer_size = 100000
        self.warmup_steps = 1000
        
        # 🎯 核心奖励权重（统一奖励函数）
        # Objective = ω_T × 时延 + ω_E × 能耗
        self.reward_weight_delay = 2.0     # ω_T: 时延权重，目标<0.25s
        self.reward_weight_energy = 1.2    # ω_E: 能耗权重
        self.reward_penalty_dropped = 0.02 # 轻微惩罚（保证完成率约束）
        
        # ❌ 已弃用参数（保留以兼容旧代码）
        self.reward_weight_loss = 0.0      # 已移除：data_loss是时延的衍生指标
        self.reward_weight_completion = 0.0  # 已集成到dropped_penalty
        self.reward_weight_cache = 0.3       # 缓存命中率 / 淘汰成本权重
        self.reward_weight_migration = 0.2   # 迁移收益 / 成本权重

        # 🎯 延时-能耗优化目标阈值（供算法动态调整）
        self.latency_target = 0.20          # 目标平均延时（秒）
        self.latency_upper_tolerance = 0.30 # 超过此值触发强化惩罚
        self.energy_target = 2200.0         # 目标能耗（焦耳）
        self.energy_upper_tolerance = 3200.0# 超过此值触发强化惩罚

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

        # Deadline 放松参数
        self.deadline_relax_default = 1.2
        self.deadline_relax_fallback = 1.3

        # 任务类型特化参数（Dataclass形式）
        self.task_profiles: Dict[int, TaskProfileSpec] = {
            1: TaskProfileSpec(1, (0.5e6/8, 3e6/8), 300),
            2: TaskProfileSpec(2, (2e6/8, 8e6/8), 400),
            3: TaskProfileSpec(3, (5e6/8, 12e6/8), 500),
            4: TaskProfileSpec(4, (8e6/8, 15e6/8), 600),
        }
        # 兼容旧字段格式
        self.task_type_specs = {
            k: {'data_range': v.data_range, 'compute_density': v.compute_density}
            for k, v in self.task_profiles.items()
        }

        # 场景定义
        self.scenarios: List[TaskScenarioSpec] = [
            TaskScenarioSpec('emergency_brake', 0.2, 0.6, 1, 1.6, 0.08),
            TaskScenarioSpec('collision_avoid', 0.3, 0.6, 1, 1.6, 0.07),
            TaskScenarioSpec('navigation', 0.9, 1.9, 2, 1.35, 0.25),
            TaskScenarioSpec('traffic_signal', 1.1, 2.0, 2, 1.35, 0.15),
            TaskScenarioSpec('video_process', 2.2, 4.8, 3, 1.25, 0.20),
            TaskScenarioSpec('image_recognition', 2.5, 4.9, 3, 1.25, 0.15),
            TaskScenarioSpec('data_analysis', 5.5, 12.0, 4, 1.15, 0.08),
            TaskScenarioSpec('ml_training', 8.0, 18.0, 4, 1.15, 0.02),
        ]
        self._scenario_weights = [scenario.weight for scenario in self.scenarios]
        self._scenario_lookup = {scenario.name: scenario for scenario in self.scenarios}
        self.type_priority_weights = self._compute_type_priority_weights()
    
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

    def sample_scenario(self) -> TaskScenarioSpec:
        """按预设权重随机选择一个任务场景。"""
        return random.choices(self.scenarios, weights=self._scenario_weights, k=1)[0]

    def get_profile(self, task_type: int) -> TaskProfileSpec:
        """获取任务类型对应的数据范围与计算密度配置。"""
        return self.task_profiles.get(
            task_type,
            TaskProfileSpec(task_type, self.data_size_range, self.task_compute_density)
        )

    def get_relax_factor(self, task_type: int) -> float:
        """根据任务类型返回默认的deadline放松系数。"""
        for scenario in self.scenarios:
            if scenario.task_type == task_type:
                return scenario.relax_factor
        return self.deadline_relax_default

    def _compute_type_priority_weights(self) -> Dict[int, float]:
        """根据场景权重汇总任务类型重要性，用于协同优化权重。"""
        totals = defaultdict(float)
        for scenario in self.scenarios:
            totals[scenario.task_type] += scenario.weight

        # 确保每个任务类型至少具备基线权重
        for task_type in self.task_profiles.keys():
            totals.setdefault(task_type, 1.0)

        values = list(totals.values())
        mean_val = sum(values) / len(values) if values else 1.0
        if mean_val <= 0:
            mean_val = 1.0

        priority_weights = {
            task_type: float(max(0.1, totals[task_type] / mean_val))
            for task_type in self.task_profiles.keys()
        }
        return priority_weights

    def get_priority_weight(self, task_type: int) -> float:
        """返回指定任务类型的优先级权重。"""
        return float(self.type_priority_weights.get(task_type, 1.0))


class ServiceConfig:
    """服务能力配置：控制节点每个时隙可处理的任务数量与工作量"""

    def __init__(self):
        # RSU 服务能力
        self.rsu_base_service = 4
        self.rsu_max_service = 9
        self.rsu_work_capacity = 2.5  # 相当于每个时隙的工作单位
        self.rsu_queue_boost_divisor = 5.0

        # UAV 服务能力
        self.uav_base_service = 3
        self.uav_max_service = 6
        self.uav_work_capacity = 1.7
        self.uav_queue_boost_divisor = 4.0


class StatsConfig:
    """统计与监控配置"""

    def __init__(self):
        self.drop_log_interval = 200
        self.task_report_interval = 100

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
        self.uav_cpu_freq_range = (1.5e9, 9e9)  # 1.5-9 GHz，包含优化后的1.8GHz
        
        # 🔧 修复：优化UAV计算能力以平衡系统负载
        self.vehicle_default_freq = 2.5e9  # 2.5 GHz (保持车载芯片)
        self.rsu_default_freq = 12e9  # 恢复12GHz - 高性能边缘计算
        self.uav_default_freq = 1.8e9  # 🔧 优化至1.8GHz - 平衡负载与能耗
        
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
        self.num_rsus = 4       # 更新为4个RSU（单一路段双路口场景）
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
        
        # 🔧 调整：合理的迁移触发阈值
        self.rsu_overload_threshold = 0.8   # 恢复到80%，更合理的触发点
        self.uav_overload_threshold = 0.75  # UAV 75%负载触发，略早于RSU
        self.rsu_underload_threshold = 0.3
        # 队列/切换阈值（用于车辆跟随与过载切换）
        self.follow_handover_distance = 30.0  # meters，车辆跟随触发的最小距离改善
        # 🔧 最终优化：统一队列管理标准
        self.queue_switch_diff = 5            # 个，目标RSU较当前RSU队列至少少5个才切换  
        self.rsu_queue_overload_len = 15      # 个，基于实际观察提高到15个任务过载阈值
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
        
        # 🔧 用户要求：每秒触发一次迁移决策
        self.cooldown_period = 1.0  # 1秒冷却期，实现每秒最多一次迁移

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
        self.num_rsus = 4       # 更新为4个RSU
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
        self.service = ServiceConfig()
        self.stats = StatsConfig()
        
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
