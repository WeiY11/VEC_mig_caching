#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理模块 - 参考Xuance框架风格设计

功能：
1. YAML配置文件加载（defaults.yaml）
2. argparse命令行参数覆盖
3. 环境变量覆盖（最高优先级）
4. 配置校验与冲突检测
5. 配置导出与变更追踪

优先级（从高到低）：
    环境变量 > 命令行参数 > YAML配置 > Python默认值

使用示例：
    from config.unified_config import get_config, parse_args
    
    # 方式1：使用默认配置
    cfg = get_config()
    
    # 方式2：命令行参数覆盖
    args = parse_args()
    cfg = get_config(args)
    
    # 方式3：指定YAML文件
    cfg = get_config(yaml_file="experiments/my_config.yaml")
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None


# =============================================================================
# 配置数据类定义（Xuance风格：分层dataclass）
# =============================================================================

@dataclass
class NetworkTopologyConfig:
    """网络拓扑配置"""
    num_vehicles: int = 12
    num_rsus: int = 4
    num_uavs: int = 2
    area_width: float = 1030.0
    area_height: float = 2060.0
    coverage_radius: float = 300.0
    uav_coverage_radius: float = 500.0
    uav_altitude: float = 100.0


@dataclass
class CommunicationConfig:
    """通信参数配置（3GPP标准）"""
    bandwidth: float = 40e6              # 40 MHz
    carrier_frequency: float = 3.5e9     # 3.5 GHz
    noise_power: float = -100.0          # dBm
    path_loss_exponent: float = 3.0
    
    # 发射功率 (dBm)
    vehicle_tx_power: float = 30.0       # 1W
    rsu_tx_power: float = 40.0           # 10W
    uav_tx_power: float = 23.0           # 0.2W
    
    # 天线增益 (dBi)
    antenna_gain_rsu: float = 15.0
    antenna_gain_uav: float = 5.0
    antenna_gain_vehicle: float = 3.0
    
    # 遮挡模型
    enable_blockage: bool = True
    building_density: float = 0.3
    
    # 编码效率
    coding_efficiency: float = 0.9


@dataclass
class ComputeConfig:
    """计算资源配置"""
    # 总计算资源池
    total_vehicle_compute: float = 18e9   # 18 GHz
    total_rsu_compute: float = 50e9       # 50 GHz
    total_uav_compute: float = 14e9       # 14 GHz
    
    # CPU频率范围 (Hz)
    vehicle_cpu_freq_min: float = 1.0e9
    vehicle_cpu_freq_max: float = 2.0e9
    rsu_cpu_freq: float = 12.5e9
    uav_cpu_freq_min: float = 6.0e9
    uav_cpu_freq_max: float = 8.0e9
    
    # 能耗参数
    vehicle_kappa: float = 1.5e-28
    rsu_kappa: float = 5.0e-32
    uav_kappa: float = 8.89e-31
    vehicle_static_power: float = 5.0     # W
    rsu_static_power: float = 25.0        # W
    uav_static_power: float = 2.5         # W
    uav_hover_power: float = 15.0         # W
    
    parallel_efficiency: float = 0.8


@dataclass
class TaskConfig:
    """任务生成配置"""
    arrival_rate: float = 3.5             # tasks/s
    data_size_min: float = 5e6            # 5 MB
    data_size_max: float = 10e6           # 10 MB
    compute_density: float = 2.5          # cycles/bit
    compute_cycles_min: float = 1e8
    compute_cycles_max: float = 5e9
    deadline_min: float = 1.0             # s
    deadline_max: float = 6.5             # s
    output_ratio: float = 0.05


@dataclass
class QueueConfig:
    """队列管理配置"""
    max_lifetime: int = 10
    max_queue_size: int = 100
    priority_levels: int = 4
    aging_factor: float = 0.25
    max_load_factor: float = 1.5
    
    # 队列容量
    rsu_nominal_capacity: float = 50.0
    uav_nominal_capacity: float = 30.0
    vehicle_nominal_capacity: float = 20.0


@dataclass
class MigrationConfig:
    """任务迁移配置"""
    migration_bandwidth: float = 100e6    # bps
    migration_threshold: float = 0.8
    rsu_overload_threshold: float = 0.70
    uav_overload_threshold: float = 0.70
    cooldown_period: float = 0.5          # s
    max_migration_distance: float = 1000  # m


@dataclass
class CacheConfig:
    """缓存配置"""
    vehicle_cache_capacity: float = 100e6   # 100 MB
    rsu_cache_capacity: float = 200e6       # 200 MB
    uav_cache_capacity: float = 150e6       # 150 MB
    cache_replacement_policy: str = "HYBRID"
    cache_hit_threshold: float = 0.8
    enable_predictive_caching: bool = True


@dataclass
class ServiceConfig:
    """RSU/UAV服务能力配置"""
    # RSU服务能力
    rsu_base_service: int = 10
    rsu_max_service: int = 25
    rsu_work_capacity: float = 6.0
    rsu_queue_boost_divisor: float = 4.0
    
    # UAV服务能力
    uav_base_service: int = 8
    uav_max_service: int = 16
    uav_work_capacity: float = 4.5
    uav_queue_boost_divisor: float = 2.0


@dataclass
class NormalizationConfig:
    """归一化配置"""
    # 数值稳定
    metric_epsilon: float = 1e-6
    
    # 位置/速度尺度
    vehicle_position_range: float = 2060.0
    rsu_position_range: float = 2060.0
    uav_position_range: float = 2060.0
    uav_altitude_range: float = 200.0
    vehicle_speed_range: float = 50.0
    
    # 队列容量
    vehicle_queue_capacity: float = 20.0
    rsu_queue_capacity: float = 20.0
    uav_queue_capacity: float = 20.0
    
    # 能耗参考
    vehicle_energy_reference: float = 1000.0
    rsu_energy_reference: float = 1000.0
    uav_energy_reference: float = 1000.0
    
    # 全局性能参考
    delay_reference: float = 4.0
    delay_upper_reference: float = 6.5
    energy_reference: float = 500.0
    energy_upper_reference: float = 800.0


@dataclass
class TD3Config:
    """TD3算法配置（统一版本）"""
    # 网络结构
    hidden_dim: int = 256
    graph_embed_dim: int = 128
    
    # 学习率
    actor_lr: float = 9e-5
    critic_lr: float = 9e-5
    
    # 训练参数
    batch_size: int = 384
    buffer_size: int = 100000
    warmup_steps: int = 1000
    
    # TD3特有
    gamma: float = 0.99
    tau: float = 0.005
    policy_delay: int = 2
    
    # 探索参数
    exploration_noise: float = 0.18
    noise_decay: float = 0.9996
    min_noise: float = 0.05
    target_noise: float = 0.05
    noise_clip: float = 0.2
    
    # 正则化
    gradient_clip_norm: float = 0.5
    use_gradient_clip: bool = True
    cql_alpha: float = 0.12
    
    # 注意力机制
    use_actor_attention: bool = True
    use_critic_attention: bool = True
    attention_min_gate: float = 0.6


@dataclass
class RewardConfig:
    """奖励函数配置"""
    # 权重
    weight_delay: float = 0.5
    weight_energy: float = 0.5
    penalty_dropped: float = 1.0
    weight_completion_gap: float = 1.0
    weight_loss_ratio: float = 1.0
    
    # 🆕 卸载效率奖励权重（鼓励边缘卸载而非本地处理）
    weight_offload_efficiency: float = 1.5  # 边缘卸载奖励权重，默认1.5
    
    # 归一化范围
    latency_min: float = 0.05
    latency_target: float = 0.3
    latency_max: float = 2.0
    energy_min: float = 1000.0
    energy_target: float = 10000.0
    energy_max: float = 25000.0
    
    # 归一化选项
    use_dynamic_normalization: bool = False
    reward_scale: float = 1.0


@dataclass
class ExperimentConfig:
    """实验配置"""
    num_episodes: int = 1000
    max_steps_per_episode: int = 200
    eval_interval: int = 50
    save_interval: int = 100
    log_interval: int = 20
    warmup_episodes: int = 10
    num_runs: int = 3
    random_seed: int = 42


@dataclass
class SystemConfig:
    """系统配置"""
    device: str = "auto"                  # auto, cpu, cuda
    num_threads: int = 4
    time_slot_duration: float = 1.0       # s
    simulation_time: int = 1000           # s
    enable_performance_optimization: bool = True


@dataclass
class UnifiedConfig:
    """
    统一配置容器 - Xuance风格
    
    整合所有子配置，提供单一访问入口
    """
    # 子配置模块
    network: NetworkTopologyConfig = field(default_factory=NetworkTopologyConfig)
    communication: CommunicationConfig = field(default_factory=CommunicationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    task: TaskConfig = field(default_factory=TaskConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    migration: MigrationConfig = field(default_factory=MigrationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)  # 🆕 服务能力配置
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)  # 🆕 归一化配置
    td3: TD3Config = field(default_factory=TD3Config)
    reward: RewardConfig = field(default_factory=RewardConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    
    # 元信息
    config_source: str = "default"
    config_version: str = "2.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    def to_yaml(self, file_path: str):
        """导出为YAML文件"""
        if not HAS_YAML:
            raise ImportError("PyYAML is required. Install with: pip install pyyaml")
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, allow_unicode=True)
        print(f"✅ 配置已导出到 {file_path}")
    
    def to_json(self, file_path: str):
        """导出为JSON文件"""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"✅ 配置已导出到 {file_path}")


# =============================================================================
# 配置加载与合并
# =============================================================================

def _deep_update(base: Dict, update: Dict) -> Dict:
    """递归合并字典"""
    result = copy.deepcopy(base)
    for key, value in update.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(file_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not HAS_YAML:
        warnings.warn("PyYAML未安装，跳过YAML配置加载")
        return {}
    
    path = Path(file_path)
    if not path.exists():
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def _apply_env_overrides(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    应用环境变量覆盖
    
    环境变量命名规则：VEC_{SECTION}_{PARAM}
    例如：VEC_TD3_ACTOR_LR=1e-4
    """
    env_mapping = {
        # TD3参数
        'VEC_TD3_HIDDEN_DIM': ('td3', 'hidden_dim', int),
        'VEC_TD3_ACTOR_LR': ('td3', 'actor_lr', float),
        'VEC_TD3_CRITIC_LR': ('td3', 'critic_lr', float),
        'VEC_TD3_BATCH_SIZE': ('td3', 'batch_size', int),
        'VEC_TD3_BUFFER_SIZE': ('td3', 'buffer_size', int),
        'VEC_TD3_GAMMA': ('td3', 'gamma', float),
        'VEC_TD3_TAU': ('td3', 'tau', float),
        'VEC_TD3_EXPLORATION_NOISE': ('td3', 'exploration_noise', float),
        'VEC_TD3_NOISE_DECAY': ('td3', 'noise_decay', float),
        'VEC_TD3_MIN_NOISE': ('td3', 'min_noise', float),
        'VEC_TD3_GRADIENT_CLIP': ('td3', 'gradient_clip_norm', float),
        'VEC_TD3_CQL_ALPHA': ('td3', 'cql_alpha', float),
        'VEC_TD3_POLICY_DELAY': ('td3', 'policy_delay', int),
        
        # 奖励参数
        'VEC_REWARD_WEIGHT_DELAY': ('reward', 'weight_delay', float),
        'VEC_REWARD_WEIGHT_ENERGY': ('reward', 'weight_energy', float),
        'VEC_REWARD_PENALTY_DROPPED': ('reward', 'penalty_dropped', float),
        'VEC_REWARD_LATENCY_TARGET': ('reward', 'latency_target', float),
        'VEC_REWARD_ENERGY_TARGET': ('reward', 'energy_target', float),
        
        # 网络拓扑
        'VEC_NUM_VEHICLES': ('network', 'num_vehicles', int),
        'VEC_NUM_RSUS': ('network', 'num_rsus', int),
        'VEC_NUM_UAVS': ('network', 'num_uavs', int),
        
        # 任务参数
        'VEC_TASK_ARRIVAL_RATE': ('task', 'arrival_rate', float),
        
        # 通信参数
        'VEC_COMM_BANDWIDTH': ('communication', 'bandwidth', float),
        
        # 实验参数
        'VEC_NUM_EPISODES': ('experiment', 'num_episodes', int),
        'VEC_MAX_STEPS': ('experiment', 'max_steps_per_episode', int),
        'VEC_RANDOM_SEED': ('experiment', 'random_seed', int),
        
        # 系统参数
        'VEC_DEVICE': ('system', 'device', str),
        'VEC_TIME_SLOT': ('system', 'time_slot_duration', float),
    }
    
    # 兼容旧的环境变量命名（TD3_*）
    legacy_mapping = {
        'TD3_HIDDEN_DIM': ('td3', 'hidden_dim', int),
        'TD3_ACTOR_LR': ('td3', 'actor_lr', float),
        'TD3_CRITIC_LR': ('td3', 'critic_lr', float),
        'TD3_BATCH_SIZE': ('td3', 'batch_size', int),
        'TD3_TAU': ('td3', 'tau', float),
        'TD3_EXPLORATION_NOISE': ('td3', 'exploration_noise', float),
        'TD3_NOISE_DECAY': ('td3', 'noise_decay', float),
        'TD3_MIN_NOISE': ('td3', 'min_noise', float),
        'TD3_GRADIENT_CLIP': ('td3', 'gradient_clip_norm', float),
        'TD3_CQL_ALPHA': ('td3', 'cql_alpha', float),
        'TD3_POLICY_DELAY': ('td3', 'policy_delay', int),
    }
    
    all_mappings = {**env_mapping, **legacy_mapping}
    
    for env_var, (section, param, dtype) in all_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            try:
                if section not in config_dict:
                    config_dict[section] = {}
                config_dict[section][param] = dtype(value)
            except (ValueError, TypeError):
                warnings.warn(f"无法解析环境变量 {env_var}={value}")
    
    return config_dict


def _dict_to_config(config_dict: Dict[str, Any]) -> UnifiedConfig:
    """将字典转换为UnifiedConfig对象"""
    
    def _create_dataclass(cls, data: Dict):
        """安全创建dataclass实例"""
        if data is None:
            return cls()
        # 只保留cls中定义的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    return UnifiedConfig(
        network=_create_dataclass(NetworkTopologyConfig, config_dict.get('network')),
        communication=_create_dataclass(CommunicationConfig, config_dict.get('communication')),
        compute=_create_dataclass(ComputeConfig, config_dict.get('compute')),
        task=_create_dataclass(TaskConfig, config_dict.get('task')),
        queue=_create_dataclass(QueueConfig, config_dict.get('queue')),
        migration=_create_dataclass(MigrationConfig, config_dict.get('migration')),
        cache=_create_dataclass(CacheConfig, config_dict.get('cache')),
        service=_create_dataclass(ServiceConfig, config_dict.get('service')),  # 🆕
        normalization=_create_dataclass(NormalizationConfig, config_dict.get('normalization')),  # 🆕
        td3=_create_dataclass(TD3Config, config_dict.get('td3')),
        reward=_create_dataclass(RewardConfig, config_dict.get('reward')),
        experiment=_create_dataclass(ExperimentConfig, config_dict.get('experiment')),
        system=_create_dataclass(SystemConfig, config_dict.get('system')),
        config_source=config_dict.get('config_source', 'merged'),
        config_version=config_dict.get('config_version', '2.0'),
    )


# =============================================================================
# argparse 命令行参数定义（Xuance风格）
# =============================================================================

def create_parser() -> argparse.ArgumentParser:
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="VEC边缘计算系统 - 统一配置",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 配置文件
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='YAML配置文件路径')
    parser.add_argument('--export-config', type=str, default=None,
                        help='导出当前配置到指定文件')
    
    # 算法选择
    parser.add_argument('--algorithm', '-a', type=str, default='OPTIMIZED_TD3',
                        choices=['TD3', 'OPTIMIZED_TD3', 'ENHANCED_TD3', 'DDPG', 'PPO', 'SAC'],
                        help='训练算法')
    
    # 网络拓扑
    parser.add_argument('--num-vehicles', type=int, default=None,
                        help='车辆数量')
    parser.add_argument('--num-rsus', type=int, default=None,
                        help='RSU数量')
    parser.add_argument('--num-uavs', type=int, default=None,
                        help='UAV数量')
    
    # TD3超参数
    parser.add_argument('--hidden-dim', type=int, default=None,
                        help='隐藏层维度')
    parser.add_argument('--actor-lr', type=float, default=None,
                        help='Actor学习率')
    parser.add_argument('--critic-lr', type=float, default=None,
                        help='Critic学习率')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='批次大小')
    parser.add_argument('--buffer-size', type=int, default=None,
                        help='经验回放缓冲区大小')
    parser.add_argument('--gamma', type=float, default=None,
                        help='折扣因子')
    parser.add_argument('--tau', type=float, default=None,
                        help='软更新系数')
    parser.add_argument('--exploration-noise', type=float, default=None,
                        help='初始探索噪声')
    parser.add_argument('--noise-decay', type=float, default=None,
                        help='噪声衰减率')
    parser.add_argument('--min-noise', type=float, default=None,
                        help='最小噪声')
    
    # 奖励配置
    parser.add_argument('--reward-weight-delay', type=float, default=None,
                        help='时延权重')
    parser.add_argument('--reward-weight-energy', type=float, default=None,
                        help='能耗权重')
    
    # 任务配置
    parser.add_argument('--arrival-rate', type=float, default=None,
                        help='任务到达率')
    
    # 实验配置
    parser.add_argument('--episodes', type=int, default=None,
                        help='训练轮次')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='每轮最大步数')
    parser.add_argument('--seed', type=int, default=None,
                        help='随机种子')
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='评估间隔')
    
    # 系统配置
    parser.add_argument('--device', type=str, default=None,
                        choices=['auto', 'cpu', 'cuda'],
                        help='计算设备')
    parser.add_argument('--time-slot', type=float, default=None,
                        help='时隙长度(秒)')
    
    # 调试选项
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='详细输出')
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印配置不执行')
    
    return parser


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """解析命令行参数"""
    parser = create_parser()
    return parser.parse_args(args)


def _apply_args_overrides(config_dict: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """应用命令行参数覆盖"""
    
    # 参数映射：argparse参数名 -> (section, param)
    args_mapping = {
        'num_vehicles': ('network', 'num_vehicles'),
        'num_rsus': ('network', 'num_rsus'),
        'num_uavs': ('network', 'num_uavs'),
        'hidden_dim': ('td3', 'hidden_dim'),
        'actor_lr': ('td3', 'actor_lr'),
        'critic_lr': ('td3', 'critic_lr'),
        'batch_size': ('td3', 'batch_size'),
        'buffer_size': ('td3', 'buffer_size'),
        'gamma': ('td3', 'gamma'),
        'tau': ('td3', 'tau'),
        'exploration_noise': ('td3', 'exploration_noise'),
        'noise_decay': ('td3', 'noise_decay'),
        'min_noise': ('td3', 'min_noise'),
        'reward_weight_delay': ('reward', 'weight_delay'),
        'reward_weight_energy': ('reward', 'weight_energy'),
        'arrival_rate': ('task', 'arrival_rate'),
        'episodes': ('experiment', 'num_episodes'),
        'max_steps': ('experiment', 'max_steps_per_episode'),
        'seed': ('experiment', 'random_seed'),
        'eval_interval': ('experiment', 'eval_interval'),
        'device': ('system', 'device'),
        'time_slot': ('system', 'time_slot_duration'),
    }
    
    for arg_name, (section, param) in args_mapping.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            if section not in config_dict:
                config_dict[section] = {}
            config_dict[section][param] = value
    
    return config_dict


# =============================================================================
# 配置校验
# =============================================================================

def validate_config(cfg: UnifiedConfig) -> List[str]:
    """
    校验配置参数合理性
    
    返回：警告信息列表（空列表表示无问题）
    """
    warnings_list = []
    
    # TD3参数校验
    if cfg.td3.actor_lr > 1e-2:
        warnings_list.append(f"⚠️ actor_lr={cfg.td3.actor_lr} 过大，建议<1e-3")
    if cfg.td3.exploration_noise > 0.5:
        warnings_list.append(f"⚠️ exploration_noise={cfg.td3.exploration_noise} 过大，建议<0.3")
    if cfg.td3.batch_size < 32:
        warnings_list.append(f"⚠️ batch_size={cfg.td3.batch_size} 过小，建议>=64")
    
    # 奖励权重校验
    total_weight = cfg.reward.weight_delay + cfg.reward.weight_energy
    if abs(total_weight - 1.0) > 0.01:
        warnings_list.append(f"⚠️ 奖励权重之和={total_weight}，建议归一化为1.0")
    
    # 网络拓扑校验
    if cfg.network.num_vehicles < 1:
        warnings_list.append("❌ 车辆数量必须>=1")
    if cfg.network.num_rsus < 1:
        warnings_list.append("❌ RSU数量必须>=1")
    
    # 任务参数校验
    if cfg.task.arrival_rate > 10:
        warnings_list.append(f"⚠️ arrival_rate={cfg.task.arrival_rate} 过高，可能导致系统过载")
    
    return warnings_list


# =============================================================================
# 主入口函数
# =============================================================================

def get_config(
    args: Optional[argparse.Namespace] = None,
    yaml_file: Optional[str] = None,
    apply_env: bool = True,
    validate: bool = True,
) -> UnifiedConfig:
    """
    获取统一配置（Xuance风格）
    
    加载优先级：环境变量 > 命令行参数 > YAML配置 > 默认值
    
    参数：
        args: 命令行参数（可选）
        yaml_file: YAML配置文件路径（可选）
        apply_env: 是否应用环境变量覆盖
        validate: 是否进行配置校验
    
    返回：
        UnifiedConfig 配置对象
    """
    # Step 1: 从默认值开始
    config_dict = UnifiedConfig().to_dict()
    
    # Step 2: 加载YAML配置
    yaml_path = yaml_file
    if yaml_path is None and args is not None:
        yaml_path = getattr(args, 'config', None)
    
    if yaml_path is None:
        # 尝试加载默认配置文件
        default_yaml = Path(__file__).parent / 'defaults.yaml'
        if default_yaml.exists():
            yaml_path = str(default_yaml)
    
    if yaml_path:
        yaml_config = _load_yaml(yaml_path)
        if yaml_config:
            config_dict = _deep_update(config_dict, yaml_config)
            config_dict['config_source'] = yaml_path
    
    # Step 3: 应用命令行参数覆盖
    if args is not None:
        config_dict = _apply_args_overrides(config_dict, args)
        config_dict['config_source'] = 'args+' + config_dict.get('config_source', 'default')
    
    # Step 4: 应用环境变量覆盖（最高优先级）
    if apply_env:
        config_dict = _apply_env_overrides(config_dict)
    
    # Step 5: 转换为配置对象
    cfg = _dict_to_config(config_dict)
    
    # Step 6: 自动检测设备
    if cfg.system.device == 'auto':
        try:
            import torch
            cfg.system.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            cfg.system.device = 'cpu'
    
    # Step 7: 校验配置
    if validate:
        warnings_list = validate_config(cfg)
        for w in warnings_list:
            print(w)
    
    return cfg


def print_config(cfg: UnifiedConfig, sections: Optional[List[str]] = None):
    """打印配置摘要"""
    print("\n" + "="*60)
    print("📋 VEC系统配置摘要")
    print("="*60)
    
    all_sections = {
        'network': ('🌐 网络拓扑', cfg.network),
        'td3': ('🤖 TD3算法', cfg.td3),
        'reward': ('🎯 奖励函数', cfg.reward),
        'task': ('📦 任务生成', cfg.task),
        'experiment': ('🧪 实验设置', cfg.experiment),
        'system': ('⚙️ 系统配置', cfg.system),
    }
    
    target_sections = sections or list(all_sections.keys())
    
    for sec_key in target_sections:
        if sec_key in all_sections:
            title, sec_obj = all_sections[sec_key]
            print(f"\n{title}:")
            for k, v in asdict(sec_obj).items():
                print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print(f"配置来源: {cfg.config_source}")
    print("="*60 + "\n")


# =============================================================================
# 兼容性接口：与现有system_config.py的桥接
# =============================================================================

def create_legacy_compatible_config(cfg: UnifiedConfig):
    """
    创建与旧版system_config.py兼容的配置对象
    
    用于平滑迁移，让旧代码能继续工作
    """
    from types import SimpleNamespace
    
    # 创建兼容的config对象
    legacy = SimpleNamespace()
    
    # 顶层属性
    legacy.num_vehicles = cfg.network.num_vehicles
    legacy.num_rsus = cfg.network.num_rsus
    legacy.num_uavs = cfg.network.num_uavs
    legacy.device = cfg.system.device
    legacy.time_slot = cfg.system.time_slot_duration
    legacy.simulation_time = cfg.system.simulation_time
    legacy.random_seed = cfg.experiment.random_seed
    
    # network子配置
    legacy.network = SimpleNamespace(**asdict(cfg.network))
    legacy.network.time_slot_duration = cfg.system.time_slot_duration
    legacy.network.bandwidth = cfg.communication.bandwidth
    legacy.network.carrier_frequency = cfg.communication.carrier_frequency
    
    # communication子配置
    legacy.communication = SimpleNamespace(**asdict(cfg.communication))
    
    # compute子配置
    legacy.compute = SimpleNamespace(**asdict(cfg.compute))
    
    # task子配置
    legacy.task = SimpleNamespace()
    legacy.task.arrival_rate = cfg.task.arrival_rate
    legacy.task.data_size_range = (cfg.task.data_size_min, cfg.task.data_size_max)
    legacy.task.task_data_size_range = legacy.task.data_size_range
    legacy.task.task_compute_density = cfg.task.compute_density
    legacy.task.compute_cycles_range = (cfg.task.compute_cycles_min, cfg.task.compute_cycles_max)
    legacy.task.deadline_range = (cfg.task.deadline_min, cfg.task.deadline_max)
    legacy.task.task_output_ratio = cfg.task.output_ratio
    
    # queue子配置
    legacy.queue = SimpleNamespace(**asdict(cfg.queue))
    
    # migration子配置
    legacy.migration = SimpleNamespace(**asdict(cfg.migration))
    
    # cache子配置
    legacy.cache = SimpleNamespace(**asdict(cfg.cache))
    
    # 🆕 service子配置（RSU/UAV服务能力）
    legacy.service = SimpleNamespace(**asdict(cfg.service))
    
    # 🆕 normalization子配置
    legacy.normalization = SimpleNamespace(**asdict(cfg.normalization))
    
    # rl子配置（兼容RLConfig）
    legacy.rl = SimpleNamespace()
    legacy.rl.hidden_dim = cfg.td3.hidden_dim
    legacy.rl.actor_lr = cfg.td3.actor_lr
    legacy.rl.critic_lr = cfg.td3.critic_lr
    legacy.rl.lr = cfg.td3.actor_lr
    legacy.rl.batch_size = cfg.td3.batch_size
    legacy.rl.memory_size = cfg.td3.buffer_size
    legacy.rl.buffer_size = cfg.td3.buffer_size
    legacy.rl.gamma = cfg.td3.gamma
    legacy.rl.tau = cfg.td3.tau
    legacy.rl.policy_delay = cfg.td3.policy_delay
    legacy.rl.exploration_noise = cfg.td3.exploration_noise
    legacy.rl.noise_decay = cfg.td3.noise_decay
    legacy.rl.min_noise = cfg.td3.min_noise
    legacy.rl.target_noise = cfg.td3.target_noise
    legacy.rl.noise_clip = cfg.td3.noise_clip
    legacy.rl.reward_weight_delay = cfg.reward.weight_delay
    legacy.rl.reward_weight_energy = cfg.reward.weight_energy
    legacy.rl.reward_penalty_dropped = cfg.reward.penalty_dropped
    legacy.rl.latency_target = cfg.reward.latency_target
    legacy.rl.latency_min = cfg.reward.latency_min
    legacy.rl.latency_upper_tolerance = cfg.reward.latency_max
    legacy.rl.energy_target = cfg.reward.energy_target
    legacy.rl.energy_min = cfg.reward.energy_min
    legacy.rl.energy_upper_tolerance = cfg.reward.energy_max
    legacy.rl.reward_scale = cfg.reward.reward_scale
    
    # experiment子配置
    legacy.experiment = SimpleNamespace()
    legacy.experiment.num_episodes = cfg.experiment.num_episodes
    legacy.experiment.max_steps_per_episode = cfg.experiment.max_steps_per_episode
    legacy.experiment.eval_interval = cfg.experiment.eval_interval
    legacy.experiment.save_interval = cfg.experiment.save_interval
    legacy.experiment.log_interval = cfg.experiment.log_interval
    legacy.experiment.warmup_episodes = cfg.experiment.warmup_episodes
    legacy.experiment.num_runs = cfg.experiment.num_runs
    
    return legacy


# 全局统一配置实例（延迟初始化）
_unified_config: Optional[UnifiedConfig] = None

def get_unified_config() -> UnifiedConfig:
    """获取全局统一配置实例"""
    global _unified_config
    if _unified_config is None:
        _unified_config = get_config(validate=False)
    return _unified_config


if __name__ == '__main__':
    # 测试配置系统
    args = parse_args()
    cfg = get_config(args)
    print_config(cfg)
    
    # 导出配置
    if args.export_config:
        if args.export_config.endswith('.yaml'):
            cfg.to_yaml(args.export_config)
        else:
            cfg.to_json(args.export_config)
