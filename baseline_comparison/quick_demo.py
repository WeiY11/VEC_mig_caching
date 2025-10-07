#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
超快速演示 - 测试2个算法各5轮
验证实验流程是否正常
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from baseline_comparison.run_baseline_comparison import BaselineComparisonExperiment


def quick_demo():
    """快速演示"""
    print("="*60)
    print("超快速演示：测试TD3和Random各5轮")
    print("="*60)
    
    experiment = BaselineComparisonExperiment()
    
    # 测试TD3（5轮训练）
    print("\n测试1: TD3算法（5轮）")
    print("-"*60)
    result_td3 = experiment.run_drl_algorithm('TD3', num_episodes=5)
    experiment.results['TD3'] = result_td3
    
    # 测试Random（5轮运行）
    print("\n测试2: Random算法（5轮）")
    print("-"*60)
    result_random = experiment.run_baseline_algorithm('Random', num_episodes=5)
    experiment.results['Random'] = result_random
    
    # 对比分析
    print("\n" + "="*60)
    print("对比结果:")
    print("="*60)
    print(f"TD3:")
    print(f"  时延: {result_td3['avg_delay']:.3f}s")
    print(f"  能耗: {result_td3['avg_energy']:.1f}J")
    print(f"  完成率: {result_td3['avg_completion_rate']:.1%}")
    
    print(f"\nRandom:")
    print(f"  时延: {result_random['avg_delay']:.3f}s")
    print(f"  能耗: {result_random['avg_energy']:.1f}J")
    print(f"  完成率: {result_random['avg_completion_rate']:.1%}")
    
    # 计算提升
    delay_improve = (result_random['avg_delay'] - result_td3['avg_delay']) / result_random['avg_delay'] * 100
    energy_improve = (result_random['avg_energy'] - result_td3['avg_energy']) / result_random['avg_energy'] * 100
    completion_improve = (result_td3['avg_completion_rate'] - result_random['avg_completion_rate']) * 100
    
    print(f"\nTD3 相比 Random 的性能提升:")
    print(f"  时延降低: {delay_improve:.1f}%")
    print(f"  能耗降低: {energy_improve:.1f}%")
    print(f"  完成率提升: {completion_improve:.1f}%")
    
    print("\n" + "="*60)
    print("演示完成！实验流程正常！")
    print("="*60)
    print("\n下一步可以运行完整实验:")
    print("  python run_baseline_comparison.py --episodes 200")
    print("="*60)


if __name__ == "__main__":
    quick_demo()

