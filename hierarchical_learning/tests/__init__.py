"""
分层强化学习测试模块
提供各层独立验证和集成测试功能
"""

from .hierarchical_tester import HierarchicalTester
from .test_config import (
    TestConfig,
    LayerTestConfig,
    IntegrationTestConfig,
    BenchmarkTestConfig,
    QUICK_TEST_CONFIG,
    STANDARD_TEST_CONFIG,
    INTEGRATION_TEST_CONFIG,
    BENCHMARK_TEST_CONFIG,
    STRESS_TEST_CONFIG,
    RESEARCH_TEST_CONFIG,
    get_test_config,
    list_available_configs,
    validate_test_config,
    create_custom_test_config,
    get_test_scenario,
    print_test_config_summary,
    TEST_SCENARIOS
)

__all__ = [
    'HierarchicalTester',
    'TestConfig',
    'LayerTestConfig', 
    'IntegrationTestConfig',
    'BenchmarkTestConfig',
    'QUICK_TEST_CONFIG',
    'STANDARD_TEST_CONFIG',
    'INTEGRATION_TEST_CONFIG',
    'BENCHMARK_TEST_CONFIG',
    'STRESS_TEST_CONFIG',
    'RESEARCH_TEST_CONFIG',
    'get_test_config',
    'list_available_configs',
    'validate_test_config',
    'create_custom_test_config',
    'get_test_scenario',
    'print_test_config_summary',
    'TEST_SCENARIOS'
]

__version__ = '1.0.0'
__author__ = 'VEC Migration Caching Team'
__description__ = '分层强化学习测试框架'