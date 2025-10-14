#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卸载策略对比实验框架

【模块说明】
本模块提供车联网边缘计算卸载策略对比实验的完整实现。

【包含内容】
- offloading_strategies: 4种卸载策略实现
- run_offloading_comparison: 参数扫描实验框架
- visualize_offloading_comparison: 结果可视化工具
- test_offloading_strategies: 框架测试

【快速开始】
>>> from offloading_strategies import create_offloading_strategy
>>> strategy = create_offloading_strategy("LocalOnly")
>>> action = strategy.select_action(state)

【版本】v1.0
【日期】2025-10-13
"""

__version__ = "1.0.0"
__author__ = "VEC Project Team"
__all__ = [
    'create_offloading_strategy',
    'OffloadingComparisonExperiment',
    'OffloadingComparisonVisualizer'
]

# 可选：导入核心类（避免在导入时执行主程序）
try:
    from .offloading_strategies import (
        create_offloading_strategy,
        LocalOnlyStrategy,
        RSUOnlyStrategy,
        UAVOnlyStrategy,
        HybridDRLStrategy
    )
except ImportError:
    # 如果作为脚本运行，跳过导入
    pass

