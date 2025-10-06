#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版环境测试脚本（无emoji）
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def test_all():
    """运行所有测试"""
    print("="*60)
    print("消融实验环境测试")
    print("="*60)
    
    success_count = 0
    total_tests = 5
    
    # 测试1: 配置加载
    print("\n[测试1] 配置加载...")
    try:
        from ablation_experiments.ablation_configs import get_all_ablation_configs
        configs = get_all_ablation_configs()
        print(f"  成功: 加载了 {len(configs)} 个配置")
        success_count += 1
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试2: 依赖导入
    print("\n[测试2] 依赖导入...")
    try:
        from config import config
        from single_agent.td3 import TD3Environment
        from evaluation.system_simulator import CompleteSystemSimulator
        import numpy as np
        print("  成功: 所有依赖导入正常")
        success_count += 1
    except Exception as e:
        print(f"  失败: {e}")
    
    # 测试3: TD3环境
    print("\n[测试3] TD3环境创建...")
    try:
        from single_agent.td3 import TD3Environment
        td3_env = TD3Environment()
        print(f"  成功: 状态维度={td3_env.state_dim}, 动作维度={td3_env.action_dim}")
        success_count += 1
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试4: 系统仿真器
    print("\n[测试4] 系统仿真器...")
    try:
        from evaluation.system_simulator import CompleteSystemSimulator
        simulator = CompleteSystemSimulator()
        simulator.reset()
        states = simulator.get_all_node_states()
        print(f"  成功: 节点数={len(states)}")
        success_count += 1
    except Exception as e:
        print(f"  失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试5: 配置应用
    print("\n[测试5] 消融配置应用...")
    try:
        from ablation_experiments.ablation_configs import get_config_by_name
        from config import config
        full_config = get_config_by_name('Full-System')
        full_config.apply_to_system()
        print(f"  成功: 配置应用正常")
        success_count += 1
    except Exception as e:
        print(f"  失败: {e}")
    
    # 汇总结果
    print("\n" + "="*60)
    print(f"测试结果: {success_count}/{total_tests} 通过")
    print("="*60)
    
    if success_count == total_tests:
        print("\n所有测试通过! 可以开始运行消融实验")
        print("\n建议命令:")
        print("  cd ablation_experiments")
        print("  python run_ablation_td3.py --episodes 30 --quick")
    else:
        print(f"\n警告: {total_tests - success_count} 个测试失败")
        print("请检查环境配置")
    
    return success_count == total_tests


if __name__ == "__main__":
    test_all()

