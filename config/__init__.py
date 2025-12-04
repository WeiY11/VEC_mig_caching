"""
配置模块 - VEC边缘计算系统

🔧 2024-12-05 重构：统一使用 defaults.yaml 作为配置源

使用方式：
1. 简单用法（推荐）：
   from config import config
   print(config.num_vehicles)
   print(config.network.num_rsus)

2. 带参数覆盖：
   from config.unified_config import get_config, parse_args
   args = parse_args(['--num-vehicles', '20'])
   cfg = get_config(args)

3. 使用实验配置：
   from config.unified_config import get_config
   cfg = get_config(yaml_file='experiments/high_load.yaml')

配置优先级：环境变量 > 命令行参数 > YAML配置 > Python默认值
"""

import warnings

# =============================================================================
# 核心：统一配置接口（Xuance风格）
# =============================================================================
from .unified_config import (
    # 配置数据类
    UnifiedConfig,
    TD3Config as UnifiedTD3Config,
    RewardConfig,
    NetworkTopologyConfig,
    CommunicationConfig as UnifiedCommunicationConfig,
    ComputeConfig,
    TaskConfig as UnifiedTaskConfig,
    QueueConfig as UnifiedQueueConfig,
    MigrationConfig as UnifiedMigrationConfig,
    CacheConfig as UnifiedCacheConfig,
    ServiceConfig,
    NormalizationConfig as UnifiedNormalizationConfig,
    ExperimentConfig,
    SystemConfig as UnifiedSystemConfig,
    # 核心函数
    get_config,
    parse_args,
    print_config,
    validate_config,
    get_unified_config,
    create_legacy_compatible_config,
)

# =============================================================================
# 全局配置实例 - 基于 defaults.yaml
# =============================================================================
# 🔧 重点：这是唯一的配置源，所有 `from config import config` 都使用此对象
_unified_cfg = get_unified_config()
config = create_legacy_compatible_config(_unified_cfg)

# 添加便捷属性，使 config 可以直接访问统一配置
config._unified = _unified_cfg


# =============================================================================
# 兼容性导入（废弃警告）
# =============================================================================
def _get_deprecated_system_config():
    """延迟导入旧的 SystemConfig（带废弃警告）"""
    warnings.warn(
        "直接导入 SystemConfig 已废弃，请使用 'from config import config' 或 "
        "'from config.unified_config import get_config'",
        DeprecationWarning,
        stacklevel=3
    )
    from .system_config import SystemConfig as _LegacySystemConfig
    return _LegacySystemConfig

def _get_deprecated_normalization_config():
    """延迟导入旧的 NormalizationConfig（带废弃警告）"""
    warnings.warn(
        "直接导入 NormalizationConfig 已废弃，请使用 'config.normalization'",
        DeprecationWarning,
        stacklevel=3
    )
    from .system_config import NormalizationConfig as _LegacyNormConfig
    return _LegacyNormConfig

# 保持向后兼容的导入
try:
    from .algorithm_config import AlgorithmConfig
except ImportError:
    AlgorithmConfig = None

try:
    from .network_config import NetworkConfig
except ImportError:
    NetworkConfig = None


# =============================================================================
# 导出符号
# =============================================================================
__all__ = [
    # 🌟 推荐使用
    'config',                # 全局配置实例
    'get_config',           # 获取配置（支持覆盖）
    'parse_args',           # 解析命令行参数
    'print_config',         # 打印配置
    'validate_config',      # 验证配置
    'UnifiedConfig',        # 统一配置类
    
    # 子配置类
    'UnifiedTD3Config',
    'RewardConfig',
    'NetworkTopologyConfig',
    'UnifiedCommunicationConfig',
    'ComputeConfig',
    'UnifiedTaskConfig',
    'UnifiedQueueConfig',
    'UnifiedMigrationConfig',
    'UnifiedCacheConfig',
    'ServiceConfig',
    'UnifiedNormalizationConfig',
    'ExperimentConfig',
    'UnifiedSystemConfig',
    
    # 兼容性接口
    'AlgorithmConfig',
    'NetworkConfig',
    'get_unified_config',
    'create_legacy_compatible_config',
]
