#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
验证消融实验环境是否正常工作

【用途】
在运行完整实验前，快速验证环境配置
"""

import sys
from pathlib import Path

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from ablation_experiments.ablation_configs import get_all_ablation_configs


def test_configs():
    """测试配置加载"""
    print("="*60)
    print("🧪 测试1: 配置加载")
    print("="*60)
    
    try:
        configs = get_all_ablation_configs()
        print(f"✓ 成功加载 {len(configs)} 个配置")
        
        for i, config in enumerate(configs, 1):
            print(f"  {i}. {config.name}")
        
        return True
    except Exception as e:
        print(f"✗ 配置加载失败: {e}")
        return False


def test_imports():
    """测试依赖导入"""
    print("\n" + "="*60)
    print("🧪 测试2: 依赖导入")
    print("="*60)
    
    imports = {
        'config': 'from config import config',
        'TD3Environment': 'from single_agent.td3 import TD3Environment',
        'VECSystemSimulator': 'from evaluation.system_simulator import VECSystemSimulator',
        'numpy': 'import numpy as np',
        'matplotlib': 'import matplotlib.pyplot as plt'
    }
    
    success_count = 0
    for name, import_str in imports.items():
        try:
            exec(import_str)
            print(f"  ✓ {name}")
            success_count += 1
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
    
    print(f"\n导入成功率: {success_count}/{len(imports)}")
    return success_count == len(imports)


def test_td3_creation():
    """测试TD3环境创建"""
    print("\n" + "="*60)
    print("🧪 测试3: TD3环境创建")
    print("="*60)
    
    try:
        from single_agent.td3 import TD3Environment
        td3_env = TD3Environment()
        print(f"  ✓ TD3环境创建成功")
        print(f"  状态维度: {td3_env.state_dim}")
        print(f"  动作维度: {td3_env.action_dim}")
        return True
    except Exception as e:
        print(f"  ✗ TD3环境创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulator_creation():
    """测试仿真器创建"""
    print("\n" + "="*60)
    print("🧪 测试4: 系统仿真器创建")
    print("="*60)
    
    try:
        from evaluation.system_simulator import VECSystemSimulator
        simulator = VECSystemSimulator()
        print(f"  ✓ 仿真器创建成功")
        
        # 测试重置
        simulator.reset()
        print(f"  ✓ 仿真器重置成功")
        
        # 测试状态获取
        states = simulator.get_all_node_states()
        metrics = simulator.get_system_metrics()
        print(f"  ✓ 状态获取成功")
        print(f"  节点数: {len(states)}")
        
        return True
    except Exception as e:
        print(f"  ✗ 仿真器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_ablation_config_apply():
    """测试消融配置应用"""
    print("\n" + "="*60)
    print("🧪 测试5: 消融配置应用")
    print("="*60)
    
    try:
        from ablation_experiments.ablation_configs import get_config_by_name
        from config import config
        
        # 测试Full-System配置
        full_config = get_config_by_name('Full-System')
        full_config.apply_to_system()
        
        # 检查配置是否应用
        if hasattr(config, 'ablation'):
            print(f"  ✓ 消融配置应用成功")
            print(f"  Cache: {config.ablation.enable_cache}")
            print(f"  Migration: {config.ablation.enable_migration}")
            return True
        else:
            print(f"  ✗ 消融配置未正确应用")
            return False
    except Exception as e:
        print(f"  ✗ 配置应用失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*60)
    print("🚀 开始快速测试")
    print("="*60)
    
    tests = [
        ("配置加载", test_configs),
        ("依赖导入", test_imports),
        ("TD3环境", test_td3_creation),
        ("系统仿真器", test_simulator_creation),
        ("配置应用", test_ablation_config_apply)
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # 汇总结果
    print("\n" + "="*60)
    print("📊 测试结果汇总")
    print("="*60)
    
    success_count = sum(1 for _, result in results if result)
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {name}: {status}")
    
    print(f"\n总体: {success_count}/{len(results)} 测试通过")
    
    if success_count == len(results):
        print("\n🎉 所有测试通过! 可以开始运行消融实验")
        print("\n建议命令:")
        print("  快速测试: python run_ablation_td3.py --episodes 30 --quick")
        print("  标准实验: python run_ablation_td3.py --episodes 200")
    else:
        print("\n⚠️ 部分测试失败，请检查环境配置")
    
    print("="*60)


if __name__ == "__main__":
    run_all_tests()

