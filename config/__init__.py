"""
配置模块 - VEC边缘计算系统

提供两种配置访问方式：
1. 传统方式（兼容）：
   from config import config
   print(config.num_vehicles)

2. 统一配置方式（推荐 - Xuance风格）：
   from config.unified_config import get_config, parse_args
   cfg = get_config()
   print(cfg.network.num_vehicles)

配置优先级：环境变量 > 命令行参数 > YAML配置 > Python默认值
"""

# 传统配置接口（保持向后兼容）
from .system_config import SystemConfig, NormalizationConfig, config
from .algorithm_config import AlgorithmConfig
from .network_config import NetworkConfig

# 统一配置接口（Xuance风格 - 推荐使用）
from .unified_config import (
    # 配置数据类
    UnifiedConfig,
    TD3Config as UnifiedTD3Config,
    RewardConfig,
    NetworkTopologyConfig,
    CommunicationConfig,
    ComputeConfig,
    TaskConfig,
    QueueConfig,
    MigrationConfig,
    CacheConfig,
    ExperimentConfig,
    # 核心函数
    get_config,
    parse_args,
    print_config,
    validate_config,
    get_unified_config,
    create_legacy_compatible_config,
)

__all__ = [
    # 传统接口
    'SystemConfig', 
    'NormalizationConfig', 
    'AlgorithmConfig', 
    'NetworkConfig', 
    'config',
    # 统一配置接口
    'UnifiedConfig',
    'UnifiedTD3Config',
    'RewardConfig',
    'NetworkTopologyConfig',
    'CommunicationConfig',
    'ComputeConfig',
    'TaskConfig',
    'QueueConfig',
    'MigrationConfig',
    'CacheConfig',
    'ExperimentConfig',
    'get_config',
    'parse_args',
    'print_config',
    'validate_config',
    'get_unified_config',
    'create_legacy_compatible_config',
]
