"""
实验模块初始化文件
包含性能评估和对比分析功能
"""
from .evaluation import (
    ExperimentResult, BaselineAlgorithms, PerformanceMetrics, ExperimentRunner
)
__all__ = [
    'ExperimentResult',
    'BaselineAlgorithms',
    'PerformanceMetrics',
    'ExperimentRunner',
]

try:
    from .hierarchical_td3_workflow import (
        get_default_multitask_scenarios,
        run_hierarchical_multitask_training,
        distill_hierarchical_policy,
        DistillationConfig,
    )

    __all__.extend(
        [
            'get_default_multitask_scenarios',
            'run_hierarchical_multitask_training',
            'distill_hierarchical_policy',
            'DistillationConfig',
        ]
    )
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    # Hierarchical workflow is optional and not required for comparison scripts.
    pass
