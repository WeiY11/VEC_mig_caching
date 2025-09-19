"""
实验模块初始化文件
包含性能评估和对比分析功能
"""
from .evaluation import (
    ExperimentResult, BaselineAlgorithms, PerformanceMetrics, ExperimentRunner
)

__all__ = [
    'ExperimentResult', 'BaselineAlgorithms', 'PerformanceMetrics', 'ExperimentRunner'
]