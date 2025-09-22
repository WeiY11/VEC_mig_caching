"""
分层强化学习测试配置
定义各种测试场景和参数
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class TestConfig:
    """基础测试配置"""
    name: str
    description: str
    num_episodes: int
    max_steps_per_episode: int
    evaluation_frequency: int
    save_results: bool
    generate_plots: bool


@dataclass
class LayerTestConfig(TestConfig):
    """单层测试配置"""
    layer_type: str  # 'strategic', 'tactical', 'operational'
    test_metrics: List[str]
    performance_thresholds: Dict[str, float]


@dataclass
class IntegrationTestConfig(TestConfig):
    """集成测试配置"""
    test_coordination: bool
    test_information_flow: bool
    test_decision_consistency: bool
    coordination_threshold: float
    consistency_threshold: float


@dataclass
class BenchmarkTestConfig(TestConfig):
    """基准测试配置"""
    baseline_algorithms: List[str]
    comparison_metrics: List[str]
    statistical_significance: bool
    confidence_level: float


# 预定义测试配置

# 快速测试配置（用于开发调试）
QUICK_TEST_CONFIG = {
    'strategic': LayerTestConfig(
        name="quick_strategic",
        description="战略层快速测试",
        num_episodes=10,
        max_steps_per_episode=50,
        evaluation_frequency=5,
        save_results=False,
        generate_plots=False,
        layer_type="strategic",
        test_metrics=["episode_rewards", "convergence_speed"],
        performance_thresholds={"avg_reward": 10.0, "convergence_speed": 20}
    ),
    'tactical': LayerTestConfig(
        name="quick_tactical",
        description="战术层快速测试",
        num_episodes=10,
        max_steps_per_episode=50,
        evaluation_frequency=5,
        save_results=False,
        generate_plots=False,
        layer_type="tactical",
        test_metrics=["episode_rewards", "coordination_efficiency"],
        performance_thresholds={"avg_reward": 15.0, "coordination_efficiency": 0.6}
    ),
    'operational': LayerTestConfig(
        name="quick_operational",
        description="执行层快速测试",
        num_episodes=10,
        max_steps_per_episode=50,
        evaluation_frequency=5,
        save_results=False,
        generate_plots=False,
        layer_type="operational",
        test_metrics=["episode_rewards", "control_precision"],
        performance_thresholds={"avg_reward": 20.0, "control_precision": 0.7}
    )
}

# 标准测试配置
STANDARD_TEST_CONFIG = {
    'strategic': LayerTestConfig(
        name="standard_strategic",
        description="战略层标准测试",
        num_episodes=50,
        max_steps_per_episode=100,
        evaluation_frequency=10,
        save_results=True,
        generate_plots=True,
        layer_type="strategic",
        test_metrics=["episode_rewards", "episode_losses", "decision_quality", 
                     "convergence_speed", "stability_score", "exploration_efficiency"],
        performance_thresholds={
            "avg_reward": 30.0,
            "convergence_speed": 40,
            "stability_score": 0.7,
            "exploration_efficiency": 0.5
        }
    ),
    'tactical': LayerTestConfig(
        name="standard_tactical",
        description="战术层标准测试",
        num_episodes=50,
        max_steps_per_episode=100,
        evaluation_frequency=10,
        save_results=True,
        generate_plots=True,
        layer_type="tactical",
        test_metrics=["episode_rewards", "episode_losses", "coordination_efficiency",
                     "load_balance_score", "communication_overhead", "convergence_speed", "multi_agent_sync"],
        performance_thresholds={
            "avg_reward": 40.0,
            "coordination_efficiency": 0.7,
            "load_balance_score": 0.6,
            "multi_agent_sync": 0.8
        }
    ),
    'operational': LayerTestConfig(
        name="standard_operational",
        description="执行层标准测试",
        num_episodes=50,
        max_steps_per_episode=100,
        evaluation_frequency=10,
        save_results=True,
        generate_plots=True,
        layer_type="operational",
        test_metrics=["episode_rewards", "episode_losses", "control_precision",
                     "response_time", "safety_violations", "energy_efficiency", "control_stability"],
        performance_thresholds={
            "avg_reward": 50.0,
            "control_precision": 0.8,
            "response_time": 0.01,  # 10ms
            "safety_violations": 0.05,
            "energy_efficiency": 0.7
        }
    )
}

# 集成测试配置
INTEGRATION_TEST_CONFIG = IntegrationTestConfig(
    name="hierarchical_integration",
    description="分层系统集成测试",
    num_episodes=30,
    max_steps_per_episode=100,
    evaluation_frequency=10,
    save_results=True,
    generate_plots=True,
    test_coordination=True,
    test_information_flow=True,
    test_decision_consistency=True,
    coordination_threshold=0.7,
    consistency_threshold=0.8
)

# 基准测试配置
BENCHMARK_TEST_CONFIG = BenchmarkTestConfig(
    name="performance_benchmark",
    description="性能基准测试",
    num_episodes=20,
    max_steps_per_episode=100,
    evaluation_frequency=5,
    save_results=True,
    generate_plots=True,
    baseline_algorithms=["random", "greedy", "single_agent"],
    comparison_metrics=["avg_reward", "avg_latency", "success_rate", "energy_efficiency"],
    statistical_significance=True,
    confidence_level=0.95
)

# 压力测试配置
STRESS_TEST_CONFIG = {
    'high_load': TestConfig(
        name="high_load_stress",
        description="高负载压力测试",
        num_episodes=100,
        max_steps_per_episode=200,
        evaluation_frequency=20,
        save_results=True,
        generate_plots=True
    ),
    'long_duration': TestConfig(
        name="long_duration_stress",
        description="长时间运行测试",
        num_episodes=500,
        max_steps_per_episode=100,
        evaluation_frequency=50,
        save_results=True,
        generate_plots=True
    ),
    'resource_limited': TestConfig(
        name="resource_limited_stress",
        description="资源受限测试",
        num_episodes=50,
        max_steps_per_episode=100,
        evaluation_frequency=10,
        save_results=True,
        generate_plots=True
    )
}

# 研究测试配置（用于论文实验）
RESEARCH_TEST_CONFIG = {
    'ablation_study': TestConfig(
        name="ablation_study",
        description="消融研究测试",
        num_episodes=100,
        max_steps_per_episode=150,
        evaluation_frequency=25,
        save_results=True,
        generate_plots=True
    ),
    'parameter_sensitivity': TestConfig(
        name="parameter_sensitivity",
        description="参数敏感性分析",
        num_episodes=200,
        max_steps_per_episode=100,
        evaluation_frequency=20,
        save_results=True,
        generate_plots=True
    ),
    'scalability_test': TestConfig(
        name="scalability_test",
        description="可扩展性测试",
        num_episodes=150,
        max_steps_per_episode=120,
        evaluation_frequency=30,
        save_results=True,
        generate_plots=True
    )
}


def get_test_config(config_type: str, test_name: str) -> Optional[TestConfig]:
    """
    获取指定的测试配置
    
    Args:
        config_type: 配置类型 - 'quick', 'standard', 'integration', 'benchmark', 'stress', 'research'
        test_name: 测试名称
    
    Returns:
        对应的测试配置，如果不存在则返回None
    """
    config_map = {
        'quick': QUICK_TEST_CONFIG,
        'standard': STANDARD_TEST_CONFIG,
        'integration': {'integration': INTEGRATION_TEST_CONFIG},
        'benchmark': {'benchmark': BENCHMARK_TEST_CONFIG},
        'stress': STRESS_TEST_CONFIG,
        'research': RESEARCH_TEST_CONFIG
    }
    
    if config_type in config_map:
        configs = config_map[config_type]
        return configs.get(test_name)
    
    return None


def list_available_configs() -> Dict[str, List[str]]:
    """
    列出所有可用的测试配置
    
    Returns:
        配置类型到测试名称列表的映射
    """
    return {
        'quick': list(QUICK_TEST_CONFIG.keys()),
        'standard': list(STANDARD_TEST_CONFIG.keys()),
        'integration': ['integration'],
        'benchmark': ['benchmark'],
        'stress': list(STRESS_TEST_CONFIG.keys()),
        'research': list(RESEARCH_TEST_CONFIG.keys())
    }


def validate_test_config(config: TestConfig) -> List[str]:
    """
    验证测试配置的有效性
    
    Args:
        config: 要验证的测试配置
    
    Returns:
        验证错误列表，如果为空则配置有效
    """
    errors = []
    
    # 基本验证
    if config.num_episodes <= 0:
        errors.append("num_episodes must be positive")
    
    if config.max_steps_per_episode <= 0:
        errors.append("max_steps_per_episode must be positive")
    
    if config.evaluation_frequency <= 0:
        errors.append("evaluation_frequency must be positive")
    
    if config.evaluation_frequency > config.num_episodes:
        errors.append("evaluation_frequency cannot be greater than num_episodes")
    
    # 层测试特定验证
    if isinstance(config, LayerTestConfig):
        valid_layers = ['strategic', 'tactical', 'operational']
        if config.layer_type not in valid_layers:
            errors.append(f"layer_type must be one of {valid_layers}")
        
        if not config.test_metrics:
            errors.append("test_metrics cannot be empty")
        
        if not config.performance_thresholds:
            errors.append("performance_thresholds cannot be empty")
    
    # 集成测试特定验证
    if isinstance(config, IntegrationTestConfig):
        if not (0 <= config.coordination_threshold <= 1):
            errors.append("coordination_threshold must be between 0 and 1")
        
        if not (0 <= config.consistency_threshold <= 1):
            errors.append("consistency_threshold must be between 0 and 1")
    
    # 基准测试特定验证
    if isinstance(config, BenchmarkTestConfig):
        if not config.baseline_algorithms:
            errors.append("baseline_algorithms cannot be empty")
        
        if not config.comparison_metrics:
            errors.append("comparison_metrics cannot be empty")
        
        if config.statistical_significance and not (0 < config.confidence_level < 1):
            errors.append("confidence_level must be between 0 and 1")
    
    return errors


def create_custom_test_config(
    name: str,
    description: str,
    num_episodes: int,
    max_steps_per_episode: int = 100,
    evaluation_frequency: int = 10,
    save_results: bool = True,
    generate_plots: bool = True,
    **kwargs
) -> TestConfig:
    """
    创建自定义测试配置
    
    Args:
        name: 测试名称
        description: 测试描述
        num_episodes: 测试回合数
        max_steps_per_episode: 每回合最大步数
        evaluation_frequency: 评估频率
        save_results: 是否保存结果
        generate_plots: 是否生成图表
        **kwargs: 其他配置参数
    
    Returns:
        自定义测试配置
    """
    config = TestConfig(
        name=name,
        description=description,
        num_episodes=num_episodes,
        max_steps_per_episode=max_steps_per_episode,
        evaluation_frequency=evaluation_frequency,
        save_results=save_results,
        generate_plots=generate_plots
    )
    
    # 验证配置
    errors = validate_test_config(config)
    if errors:
        raise ValueError(f"Invalid test configuration: {', '.join(errors)}")
    
    return config


# 测试场景定义
TEST_SCENARIOS = {
    'development': {
        'description': '开发阶段测试场景',
        'configs': ['quick'],
        'recommended_order': ['strategic', 'tactical', 'operational']
    },
    'validation': {
        'description': '验证阶段测试场景',
        'configs': ['standard', 'integration'],
        'recommended_order': ['strategic', 'tactical', 'operational', 'integration']
    },
    'performance': {
        'description': '性能评估场景',
        'configs': ['benchmark', 'stress'],
        'recommended_order': ['benchmark', 'high_load', 'long_duration']
    },
    'research': {
        'description': '研究实验场景',
        'configs': ['research'],
        'recommended_order': ['ablation_study', 'parameter_sensitivity', 'scalability_test']
    }
}


def get_test_scenario(scenario_name: str) -> Optional[Dict]:
    """
    获取测试场景配置
    
    Args:
        scenario_name: 场景名称
    
    Returns:
        场景配置字典，如果不存在则返回None
    """
    return TEST_SCENARIOS.get(scenario_name)


def print_test_config_summary(config: TestConfig):
    """
    打印测试配置摘要
    
    Args:
        config: 测试配置
    """
    print(f"📋 测试配置: {config.name}")
    print(f"   描述: {config.description}")
    print(f"   回合数: {config.num_episodes}")
    print(f"   每回合步数: {config.max_steps_per_episode}")
    print(f"   评估频率: {config.evaluation_frequency}")
    print(f"   保存结果: {'是' if config.save_results else '否'}")
    print(f"   生成图表: {'是' if config.generate_plots else '否'}")
    
    if isinstance(config, LayerTestConfig):
        print(f"   测试层: {config.layer_type}")
        print(f"   测试指标: {', '.join(config.test_metrics)}")
        print(f"   性能阈值: {config.performance_thresholds}")
    
    elif isinstance(config, IntegrationTestConfig):
        print(f"   测试协调性: {'是' if config.test_coordination else '否'}")
        print(f"   测试信息流: {'是' if config.test_information_flow else '否'}")
        print(f"   测试决策一致性: {'是' if config.test_decision_consistency else '否'}")
    
    elif isinstance(config, BenchmarkTestConfig):
        print(f"   基准算法: {', '.join(config.baseline_algorithms)}")
        print(f"   比较指标: {', '.join(config.comparison_metrics)}")
        print(f"   统计显著性: {'是' if config.statistical_significance else '否'}")