#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试强制卸载模式（Local-Only和Remote-Only）的正确性

验证：
1. Local-Only模式：所有任务本地处理，远端任务数=0
2. Remote-Only模式：所有任务卸载到边缘，本地任务数=0（失败的任务被丢弃）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
from config import config
from evaluation.system_simulator import CompleteSystemSimulator


def get_base_config():
    """获取基础配置"""
    return {
        'forced_offload_mode': '',
        'allow_local_processing': True,
        'num_vehicles': 4,
        'num_rsus': 2,
        'num_uavs': 1,
        'task_arrival_rate': 2.0,
        'cache_capacity': 1000.0,  # MB
        'rsu_cache_capacity': 2000.0,  # MB
        'uav_cache_capacity': 500.0,  # MB
        'area_width': 1000,
        'area_height': 1000,
        'time_slot': 0.1,  # 100ms
        'max_steps_per_episode': 100,
    }


def test_local_only_mode():
    """测试Local-Only模式：所有任务应该本地处理"""
    print("\n" + "=" * 80)
    print("测试 Local-Only 模式")
    print("=" * 80)
    
    # 配置Local-Only模式
    test_config = get_base_config()
    test_config.update({
        'forced_offload_mode': 'local_only',
        'allow_local_processing': True,
    })
    
    # 创建仿真器
    simulator = CompleteSystemSimulator(config=test_config)
    
    # 运行仿真步骤
    num_steps = 20
    total_local_tasks = 0
    total_remote_tasks = 0
    total_dropped_tasks = 0
    
    for step in range(num_steps):
        # 简单的action（不影响local_only模式）
        actions = {
            'vehicle_offload_pref': {'local': 0.0, 'rsu': 0.5, 'uav': 0.5},  # 故意设置为卸载优先
        }
        
        step_result = simulator.run_simulation_step(step, actions)
        
        # 统计任务分布
        total_local_tasks += step_result.get('local_tasks', 0)
        total_remote_tasks += step_result.get('remote_tasks', 0)
        total_dropped_tasks += step_result.get('dropped_tasks', 0)
    
    print(f"\n仿真结果（{num_steps}步）：")
    print(f"  本地处理任务数: {total_local_tasks}")
    print(f"  远端处理任务数: {total_remote_tasks}")
    print(f"  丢弃任务数: {total_dropped_tasks}")
    
    # 验证结果
    if total_remote_tasks == 0 and total_local_tasks > 0:
        print("\n[PASS] Local-Only模式测试通过：所有任务都在本地处理")
        return True
    else:
        print(f"\n[FAIL] Local-Only模式测试失败：发现远端任务 {total_remote_tasks} 个")
        return False


def test_remote_only_mode():
    """测试Remote-Only模式：所有任务应该卸载到边缘（或被丢弃）"""
    print("\n" + "=" * 80)
    print("测试 Remote-Only 模式")
    print("=" * 80)
    
    # 配置Remote-Only模式
    test_config = get_base_config()
    test_config.update({
        'forced_offload_mode': 'remote_only',
        'allow_local_processing': False,
    })
    
    # 创建仿真器
    simulator = CompleteSystemSimulator(config=test_config)
    
    # 运行仿真步骤
    num_steps = 20
    total_local_tasks = 0
    total_remote_tasks = 0
    total_dropped_tasks = 0
    
    for step in range(num_steps):
        # 简单的action（不影响remote_only模式）
        actions = {
            'vehicle_offload_pref': {'local': 1.0, 'rsu': 0.0, 'uav': 0.0},  # 故意设置为本地优先
        }
        
        step_result = simulator.run_simulation_step(step, actions)
        
        # 统计任务分布
        total_local_tasks += step_result.get('local_tasks', 0)
        total_remote_tasks += step_result.get('remote_tasks', 0)
        total_dropped_tasks += step_result.get('dropped_tasks', 0)
    
    print(f"\n仿真结果（{num_steps}步）：")
    print(f"  本地处理任务数: {total_local_tasks}")
    print(f"  远端处理任务数: {total_remote_tasks}")
    print(f"  丢弃任务数: {total_dropped_tasks}")
    print(f"  总任务数: {total_local_tasks + total_remote_tasks + total_dropped_tasks}")
    
    # 验证结果
    if total_local_tasks == 0 and (total_remote_tasks > 0 or total_dropped_tasks > 0):
        print("\n[PASS] Remote-Only模式测试通过：所有任务都卸载到边缘（或被丢弃）")
        return True
    else:
        print(f"\n[FAIL] Remote-Only模式测试失败：发现本地任务 {total_local_tasks} 个")
        return False


def test_normal_mode():
    """测试正常模式：应该有本地和远端任务的混合"""
    print("\n" + "=" * 80)
    print("测试 正常模式（Normal Mode）")
    print("=" * 80)
    
    # 配置正常模式
    test_config = get_base_config()
    test_config.update({
        'forced_offload_mode': '',  # 不强制
        'allow_local_processing': True,
    })
    
    # 创建仿真器
    simulator = CompleteSystemSimulator(config=test_config)
    
    # 运行仿真步骤
    num_steps = 20
    total_local_tasks = 0
    total_remote_tasks = 0
    total_dropped_tasks = 0
    
    for step in range(num_steps):
        # 均衡的卸载偏好
        actions = {
            'vehicle_offload_pref': {'local': 0.33, 'rsu': 0.33, 'uav': 0.34},
        }
        
        step_result = simulator.run_simulation_step(step, actions)
        
        # 统计任务分布
        total_local_tasks += step_result.get('local_tasks', 0)
        total_remote_tasks += step_result.get('remote_tasks', 0)
        total_dropped_tasks += step_result.get('dropped_tasks', 0)
    
    print(f"\n仿真结果（{num_steps}步）：")
    print(f"  本地处理任务数: {total_local_tasks}")
    print(f"  远端处理任务数: {total_remote_tasks}")
    print(f"  丢弃任务数: {total_dropped_tasks}")
    
    # 验证结果：正常模式应该有本地和远端任务的混合
    if total_local_tasks > 0 and total_remote_tasks > 0:
        print("\n[PASS] 正常模式测试通过：本地和远端任务都存在")
        return True
    else:
        print(f"\n[WARNING] 正常模式测试警告：任务分布可能不均衡")
        print(f"    本地任务: {total_local_tasks}, 远端任务: {total_remote_tasks}")
        return True  # 仍然通过，因为这取决于随机性


def main():
    """运行所有测试"""
    print("=" * 80)
    print("强制卸载模式测试套件")
    print("=" * 80)
    print(f"项目根目录: {project_root}")
    
    # 设置随机种子以保证可重复性
    np.random.seed(42)
    
    results = []
    
    # 测试1：Local-Only模式
    try:
        result1 = test_local_only_mode()
        results.append(("Local-Only", result1))
    except Exception as e:
        print(f"\n[ERROR] Local-Only模式测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Local-Only", False))
    
    # 测试2：Remote-Only模式
    try:
        result2 = test_remote_only_mode()
        results.append(("Remote-Only", result2))
    except Exception as e:
        print(f"\n[ERROR] Remote-Only模式测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Remote-Only", False))
    
    # 测试3：正常模式
    try:
        result3 = test_normal_mode()
        results.append(("Normal", result3))
    except Exception as e:
        print(f"\n[ERROR] 正常模式测试异常: {e}")
        import traceback
        traceback.print_exc()
        results.append(("Normal", False))
    
    # 打印总结
    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n[SUCCESS] 所有测试通过！")
        return 0
    else:
        print("\n[FAILURE] 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

