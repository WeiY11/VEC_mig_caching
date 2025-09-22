"""
分层强化学习配置文件
包含战略层、战术层、执行层的超参数设置
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class StrategicLayerConfig:
    """战略层配置 - SAC算法"""
    # 网络结构
    state_dim: int = 50
    action_dim: int = 10
    hidden_dim: int = 256
    num_hidden_layers: int = 3
    
    # 学习参数
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_alpha: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    target_entropy: float = -10.0  # -action_dim
    
    # 训练参数
    batch_size: int = 256
    buffer_size: int = 1000000
    warmup_steps: int = 1000
    update_frequency: int = 1
    target_update_frequency: int = 1
    
    # 探索参数
    exploration_noise: float = 0.1
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    
    # 网络初始化
    weight_init: str = "xavier_uniform"
    bias_init: str = "zeros"
    activation: str = "relu"
    output_activation: str = "tanh"
    
    # 正则化
    weight_decay: float = 1e-4
    dropout_rate: float = 0.0
    gradient_clip: float = 1.0
    
    # 决策频率（每多少步做一次战略决策）
    decision_frequency: int = 100
    
    # 状态特征权重
    state_feature_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.state_feature_weights is None:
            self.state_feature_weights = {
                'system_load': 0.3,
                'network_quality': 0.25,
                'energy_efficiency': 0.2,
                'vehicle_density': 0.15,
                'weather_condition': 0.1
            }


@dataclass
class TacticalLayerConfig:
    """战术层配置 - MATD3算法"""
    # 智能体配置
    num_agents: int = 8  # RSU + UAV数量
    state_dim: int = 30
    action_dim: int = 8
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    
    # 学习参数
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    gamma: float = 0.95
    tau: float = 0.01
    
    # 训练参数
    batch_size: int = 128
    buffer_size: int = 500000
    warmup_steps: int = 500
    update_frequency: int = 2
    target_update_frequency: int = 2
    
    # TD3特有参数
    policy_delay: int = 2
    policy_noise: float = 0.2
    noise_clip: float = 0.5
    exploration_noise: float = 0.1
    
    # 网络配置
    weight_init: str = "xavier_uniform"
    bias_init: str = "zeros"
    activation: str = "relu"
    output_activation: str = "tanh"
    
    # 正则化
    weight_decay: float = 1e-4
    dropout_rate: float = 0.0
    gradient_clip: float = 0.5
    
    # 多智能体特有参数
    centralized_training: bool = True
    decentralized_execution: bool = True
    shared_experience: bool = True
    communication_enabled: bool = True
    communication_range: float = 1000.0  # 通信范围(m)
    
    # 协调机制
    coordination_weight: float = 0.3
    individual_weight: float = 0.7
    consensus_threshold: float = 0.8
    
    # 决策频率（每多少步做一次战术决策）
    decision_frequency: int = 20
    
    # 状态特征权重
    state_feature_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.state_feature_weights is None:
            self.state_feature_weights = {
                'resource_allocation': 0.35,
                'load_balancing': 0.25,
                'coordination_efficiency': 0.2,
                'service_quality': 0.15,
                'energy_consumption': 0.05
            }


@dataclass
class OperationalLayerConfig:
    """执行层配置 - TD3算法"""
    # 智能体配置
    num_agents: int = 8  # RSU + UAV数量
    state_dim: int = 40
    action_dim: int = 6
    hidden_dim: int = 128
    num_hidden_layers: int = 2
    
    # 学习参数
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    gamma: float = 0.9
    tau: float = 0.005
    
    # 训练参数
    batch_size: int = 64
    buffer_size: int = 200000
    warmup_steps: int = 200
    update_frequency: int = 1
    target_update_frequency: int = 1
    
    # TD3特有参数
    policy_delay: int = 2
    policy_noise: float = 0.1
    noise_clip: float = 0.3
    exploration_noise: float = 0.05
    
    # 网络配置
    weight_init: str = "xavier_uniform"
    bias_init: str = "zeros"
    activation: str = "relu"
    output_activation: str = "tanh"
    
    # 正则化
    weight_decay: float = 1e-5
    dropout_rate: float = 0.0
    gradient_clip: float = 0.3
    
    # 控制参数
    control_frequency: int = 1  # 每步都执行控制
    action_smoothing: float = 0.1
    safety_constraints: bool = True
    
    # 动作空间约束
    action_bounds: Dict[str, Tuple[float, float]] = None
    
    # 状态特征权重
    state_feature_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.action_bounds is None:
            self.action_bounds = {
                'cpu_frequency': (0.1, 1.0),
                'memory_allocation': (0.1, 1.0),
                'transmission_power': (0.1, 1.0),
                'compute_allocation': (0.1, 1.0),
                'bandwidth_allocation': (0.1, 1.0),
                'cache_allocation': (0.1, 1.0)
            }
        
        if self.state_feature_weights is None:
            self.state_feature_weights = {
                'real_time_metrics': 0.4,
                'resource_utilization': 0.3,
                'control_precision': 0.2,
                'safety_constraints': 0.1
            }


@dataclass
class HierarchicalConfig:
    """分层架构总体配置"""
    # 层级配置
    strategic_config: StrategicLayerConfig = None
    tactical_config: TacticalLayerConfig = None
    operational_config: OperationalLayerConfig = None
    
    # 环境配置
    num_rsus: int = 5
    num_uavs: int = 3
    num_vehicles: int = 50
    area_width: float = 2000.0  # m
    area_height: float = 2000.0  # m
    
    # 训练配置
    max_episode_steps: int = 1000
    num_episodes: int = 200
    eval_interval: int = 20
    save_interval: int = 50
    
    # 分层协调参数
    information_sharing: bool = True
    hierarchical_reward_shaping: bool = True
    layer_synchronization: bool = True
    
    # 奖励权重
    strategic_reward_weight: float = 1.0
    tactical_reward_weight: float = 0.8
    operational_reward_weight: float = 0.6
    
    # 性能指标权重
    latency_weight: float = 0.3
    energy_weight: float = 0.25
    success_rate_weight: float = 0.25
    cost_efficiency_weight: float = 0.2
    
    # 3GPP标准参数
    gpp_params: Dict[str, float] = None
    
    def __post_init__(self):
        # 初始化层级配置
        if self.strategic_config is None:
            self.strategic_config = StrategicLayerConfig()
        
        if self.tactical_config is None:
            self.tactical_config = TacticalLayerConfig()
            self.tactical_config.num_agents = self.num_rsus + self.num_uavs
        
        if self.operational_config is None:
            self.operational_config = OperationalLayerConfig()
            self.operational_config.num_agents = self.num_rsus + self.num_uavs
        
        # 3GPP标准参数
        if self.gpp_params is None:
            self.gpp_params = {
                # 频率参数
                'carrier_frequency': 2.0e9,  # 2 GHz
                'bandwidth': 20e6,  # 20 MHz
                'subcarrier_spacing': 15e3,  # 15 kHz
                
                # 功率参数
                'max_tx_power_rsu': 46.0,  # 46 dBm (40W)
                'max_tx_power_uav': 30.0,  # 30 dBm (1W)
                'max_tx_power_vehicle': 23.0,  # 23 dBm (200mW)
                
                # 天线参数
                'antenna_gain_rsu': 15.0,  # 15 dBi
                'antenna_gain_uav': 5.0,   # 5 dBi
                'antenna_gain_vehicle': 3.0,  # 3 dBi
                
                # 路径损耗参数
                'path_loss_exponent': 3.5,
                'shadowing_std': 8.0,  # dB
                'noise_figure': 9.0,   # dB
                'thermal_noise': -174.0,  # dBm/Hz
                
                # 移动性参数
                'vehicle_speed_min': 10.0,  # m/s
                'vehicle_speed_max': 30.0,  # m/s
                'uav_height': 100.0,  # m
                
                # QoS参数
                'latency_threshold': 100.0,  # ms
                'reliability_threshold': 0.99,
                'data_rate_threshold': 1.0,  # Mbps
                
                # 资源参数
                'rsu_compute_capacity': 1000.0,  # GFLOPS
                'uav_compute_capacity': 500.0,   # GFLOPS
                'rsu_storage_capacity': 1000.0,  # GB
                'uav_storage_capacity': 100.0,   # GB
            }


# 预定义配置
def get_default_hierarchical_config() -> HierarchicalConfig:
    """获取默认分层配置"""
    return HierarchicalConfig()


def get_lightweight_hierarchical_config() -> HierarchicalConfig:
    """获取轻量级分层配置（用于快速测试）"""
    config = HierarchicalConfig()
    
    # 减少网络规模
    config.strategic_config.hidden_dim = 128
    config.strategic_config.num_hidden_layers = 2
    config.tactical_config.hidden_dim = 64
    config.operational_config.hidden_dim = 64
    
    # 减少缓冲区大小
    config.strategic_config.buffer_size = 100000
    config.tactical_config.buffer_size = 50000
    config.operational_config.buffer_size = 20000
    
    # 减少批次大小
    config.strategic_config.batch_size = 64
    config.tactical_config.batch_size = 32
    config.operational_config.batch_size = 16
    
    # 减少环境规模
    config.num_vehicles = 20
    config.max_episode_steps = 500
    
    return config


def get_performance_hierarchical_config() -> HierarchicalConfig:
    """获取高性能分层配置（用于最终训练）"""
    config = HierarchicalConfig()
    
    # 增加网络规模
    config.strategic_config.hidden_dim = 512
    config.strategic_config.num_hidden_layers = 4
    config.tactical_config.hidden_dim = 256
    config.tactical_config.num_hidden_layers = 3
    config.operational_config.hidden_dim = 256
    config.operational_config.num_hidden_layers = 3
    
    # 增加缓冲区大小
    config.strategic_config.buffer_size = 2000000
    config.tactical_config.buffer_size = 1000000
    config.operational_config.buffer_size = 500000
    
    # 增加批次大小
    config.strategic_config.batch_size = 512
    config.tactical_config.batch_size = 256
    config.operational_config.batch_size = 128
    
    # 更精细的学习率
    config.strategic_config.lr_actor = 1e-4
    config.strategic_config.lr_critic = 1e-4
    config.tactical_config.lr_actor = 5e-5
    config.tactical_config.lr_critic = 5e-5
    config.operational_config.lr_actor = 5e-5
    config.operational_config.lr_critic = 5e-5
    
    # 增加训练回合数
    config.num_episodes = 500
    config.max_episode_steps = 2000
    
    return config


def get_research_hierarchical_config() -> HierarchicalConfig:
    """获取研究用分层配置（符合论文要求）"""
    config = HierarchicalConfig()
    
    # 论文中的网络结构
    config.strategic_config.hidden_dim = 256
    config.strategic_config.num_hidden_layers = 3
    config.tactical_config.hidden_dim = 128
    config.tactical_config.num_hidden_layers = 2
    config.operational_config.hidden_dim = 128
    config.operational_config.num_hidden_layers = 2
    
    # 论文中的学习参数
    config.strategic_config.lr_actor = 3e-4
    config.strategic_config.lr_critic = 3e-4
    config.strategic_config.gamma = 0.99
    config.tactical_config.lr_actor = 1e-4
    config.tactical_config.lr_critic = 1e-4
    config.tactical_config.gamma = 0.95
    config.operational_config.lr_actor = 1e-4
    config.operational_config.lr_critic = 1e-4
    config.operational_config.gamma = 0.9
    
    # 论文中的环境设置
    config.num_rsus = 5
    config.num_uavs = 3
    config.num_vehicles = 50
    config.area_width = 2000.0
    config.area_height = 2000.0
    
    # 论文中的训练设置
    config.num_episodes = 300
    config.max_episode_steps = 1000
    config.eval_interval = 20
    config.save_interval = 50
    
    return config


# 配置验证函数
def validate_hierarchical_config(config: HierarchicalConfig) -> bool:
    """验证分层配置的有效性"""
    try:
        # 检查基本参数
        assert config.num_rsus > 0, "RSU数量必须大于0"
        assert config.num_uavs > 0, "UAV数量必须大于0"
        assert config.num_vehicles > 0, "车辆数量必须大于0"
        assert config.area_width > 0, "区域宽度必须大于0"
        assert config.area_height > 0, "区域高度必须大于0"
        
        # 检查训练参数
        assert config.num_episodes > 0, "训练回合数必须大于0"
        assert config.max_episode_steps > 0, "最大步数必须大于0"
        assert config.eval_interval > 0, "评估间隔必须大于0"
        assert config.save_interval > 0, "保存间隔必须大于0"
        
        # 检查层级配置
        assert config.strategic_config.state_dim > 0, "战略层状态维度必须大于0"
        assert config.strategic_config.action_dim > 0, "战略层动作维度必须大于0"
        assert config.tactical_config.num_agents == config.num_rsus + config.num_uavs, \
               "战术层智能体数量必须等于RSU+UAV数量"
        assert config.operational_config.num_agents == config.num_rsus + config.num_uavs, \
               "执行层智能体数量必须等于RSU+UAV数量"
        
        # 检查学习率
        assert 0 < config.strategic_config.lr_actor < 1, "战略层Actor学习率必须在(0,1)范围内"
        assert 0 < config.strategic_config.lr_critic < 1, "战略层Critic学习率必须在(0,1)范围内"
        assert 0 < config.tactical_config.lr_actor < 1, "战术层Actor学习率必须在(0,1)范围内"
        assert 0 < config.tactical_config.lr_critic < 1, "战术层Critic学习率必须在(0,1)范围内"
        assert 0 < config.operational_config.lr_actor < 1, "执行层Actor学习率必须在(0,1)范围内"
        assert 0 < config.operational_config.lr_critic < 1, "执行层Critic学习率必须在(0,1)范围内"
        
        # 检查折扣因子
        assert 0 < config.strategic_config.gamma < 1, "战略层折扣因子必须在(0,1)范围内"
        assert 0 < config.tactical_config.gamma < 1, "战术层折扣因子必须在(0,1)范围内"
        assert 0 < config.operational_config.gamma < 1, "执行层折扣因子必须在(0,1)范围内"
        
        print("✅ 分层配置验证通过")
        return True
        
    except AssertionError as e:
        print(f"❌ 分层配置验证失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 分层配置验证出错: {e}")
        return False


# 配置工厂函数
def create_hierarchical_config(config_type: str = "default") -> HierarchicalConfig:
    """创建分层配置"""
    config_type = config_type.lower()
    
    if config_type == "default":
        config = get_default_hierarchical_config()
    elif config_type == "lightweight":
        config = get_lightweight_hierarchical_config()
    elif config_type == "performance":
        config = get_performance_hierarchical_config()
    elif config_type == "research":
        config = get_research_hierarchical_config()
    else:
        raise ValueError(f"不支持的配置类型: {config_type}")
    
    # 验证配置
    if not validate_hierarchical_config(config):
        raise ValueError("配置验证失败")
    
    return config


if __name__ == "__main__":
    # 测试配置
    print("🧪 测试分层配置...")
    
    # 测试默认配置
    default_config = create_hierarchical_config("default")
    print(f"默认配置 - 战略层状态维度: {default_config.strategic_config.state_dim}")
    
    # 测试轻量级配置
    lightweight_config = create_hierarchical_config("lightweight")
    print(f"轻量级配置 - 战略层隐藏层维度: {lightweight_config.strategic_config.hidden_dim}")
    
    # 测试高性能配置
    performance_config = create_hierarchical_config("performance")
    print(f"高性能配置 - 训练回合数: {performance_config.num_episodes}")
    
    # 测试研究配置
    research_config = create_hierarchical_config("research")
    print(f"研究配置 - 车辆数量: {research_config.num_vehicles}")
    
    print("✅ 所有配置测试通过!")