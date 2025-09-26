#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务分类测试 - 验证四级分类是否按论文逻辑工作
"""

import sys
import os
import numpy as np

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from config import config as sys_config
    from evaluation.system_simulator import CompleteSystemSimulator
except Exception as e:
    print(f"Import failed: {e}")
    exit(1)


def test_task_classification():
    """测试任务分类逻辑"""
    print("=== Task Classification Test ===")
    
    # 创建仿真器
    config = {
        "num_vehicles": 3, "num_rsus": 4, "num_uavs": 2,
        "cache_capacity": 100, "computation_capacity": 1000, 
        "bandwidth": 20, "transmission_power": 0.1, "computation_power": 1.0,
        "time_slot": 0.2, "simulation_time": 20, "task_arrival_rate": 1.0,
        "high_load_mode": False
    }
    
    sim = CompleteSystemSimulator(config)
    sim.current_time = 0.0  # 初始化时间
    
    # 生成200个任务并统计分类
    classification_stats = {1: [], 2: [], 3: [], 4: []}
    deadline_by_type = {1: [], 2: [], 3: [], 4: []}
    app_scenarios = []
    
    print("生成任务样本...")
    
    for i in range(200):
        sim.current_time = i * 0.1  # 模拟时间推进
        task = sim.generate_task('V_0')
        task_type = task['task_type']
        deadline_duration = task['deadline'] - task['arrival_time']
        
        classification_stats[task_type].append(task)
        deadline_by_type[task_type].append(deadline_duration)
        
        # 记录应用场景（如果有的话）
        if hasattr(sim, '_last_app_name'):
            app_scenarios.append(sim._last_app_name)
    
    # 分析分类结果
    print(f"\n=== 分类统计结果 ===")
    total_tasks = sum(len(tasks) for tasks in classification_stats.values())
    
    for task_type in [1, 2, 3, 4]:
        count = len(classification_stats[task_type])
        percentage = count / total_tasks * 100
        deadlines = deadline_by_type[task_type]
        
        if deadlines:
            min_deadline = min(deadlines)
            max_deadline = max(deadlines)
            avg_deadline = np.mean(deadlines)
            print(f"类型{task_type}: {count}个任务 ({percentage:.1f}%)")
            print(f"  deadline范围: {min_deadline:.2f}-{max_deadline:.2f}s (平均{avg_deadline:.2f}s)")
            
            # 检查是否符合论文阈值
            time_slot = 0.2
            if task_type == 1:
                expected_max = 4 * time_slot  # τ₁ = 4时隙 = 0.8s
                if max_deadline > expected_max + 0.1:
                    print(f"  ❌ 超出类型1阈值: 最大{max_deadline:.2f}s > {expected_max:.2f}s")
                else:
                    print(f"  ✅ 符合类型1阈值: ≤{expected_max:.2f}s")
            elif task_type == 2:
                expected_min = 4 * time_slot
                expected_max = 10 * time_slot  # τ₂ = 10时隙 = 2.0s
                if min_deadline <= expected_min or max_deadline > expected_max + 0.1:
                    print(f"  ❌ 不符合类型2阈值: {min_deadline:.2f}-{max_deadline:.2f}s 应在({expected_min:.2f}, {expected_max:.2f}]")
                else:
                    print(f"  ✅ 符合类型2阈值: ({expected_min:.2f}, {expected_max:.2f}]s")
            elif task_type == 3:
                expected_min = 10 * time_slot
                expected_max = 25 * time_slot  # τ₃ = 25时隙 = 5.0s
                if min_deadline <= expected_min or max_deadline > expected_max + 0.1:
                    print(f"  ❌ 不符合类型3阈值: {min_deadline:.2f}-{max_deadline:.2f}s 应在({expected_min:.2f}, {expected_max:.2f}]")
                else:
                    print(f"  ✅ 符合类型3阈值: ({expected_min:.2f}, {expected_max:.2f}]s")
            else:  # task_type == 4
                expected_min = 25 * time_slot
                if min_deadline <= expected_min:
                    print(f"  ❌ 不符合类型4阈值: 最小{min_deadline:.2f}s 应>{expected_min:.2f}s")
                else:
                    print(f"  ✅ 符合类型4阈值: >{expected_min:.2f}s")
    
    # 验证卸载策略
    print(f"\n=== 卸载策略验证 ===")
    test_tasks = [
        {'task_type': 1, 'priority': 0.9, 'data_size': 0.3},  # 类型1高优先级
        {'task_type': 2, 'priority': 0.5, 'data_size': 0.8},  # 类型2中优先级
        {'task_type': 3, 'priority': 0.3, 'data_size': 1.2},  # 类型3低优先级
        {'task_type': 4, 'priority': 0.2, 'data_size': 2.0}   # 类型4低优先级
    ]
    
    for test_task in test_tasks:
        print(f"\n测试任务类型{test_task['task_type']}:")
        
        # 模拟候选集确定
        if test_task['task_type'] == 1:
            expected_candidates = "仅本地"
        elif test_task['task_type'] == 2:
            expected_candidates = "本地+近距离RSU" 
        elif test_task['task_type'] == 3:
            expected_candidates = "本地+RSU+近距离UAV"
        else:
            expected_candidates = "所有节点"
            
        print(f"  预期候选集: {expected_candidates}")
        print(f"  任务优先级: {test_task['priority']}")
        print(f"  数据大小: {test_task['data_size']}MB")


if __name__ == "__main__":
    test_task_classification()
