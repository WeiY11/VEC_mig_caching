#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速能耗测试 - 避免Unicode问题
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


def test_energy_calculation():
    """测试能耗计算"""
    print("=== Energy Calculation Test ===")
    
    # 创建简化配置
    config = {
        "num_vehicles": 5, "num_rsus": 4, "num_uavs": 2,
        "cache_capacity": 100, "computation_capacity": 1000, 
        "bandwidth": 20, "transmission_power": 0.1, "computation_power": 1.0,
        "time_slot": 0.2, "simulation_time": 20, "task_arrival_rate": 0.5,
        "high_load_mode": False  # 关闭高负载模式
    }
    
    try:
        sim = CompleteSystemSimulator(config)
        print(f"Created: {len(sim.vehicles)} vehicles, {len(sim.rsus)} RSUs, {len(sim.uavs)} UAVs")
        
        # 运行10步
        total_energy_series = []
        for i in range(10):
            stats = sim.run_simulation_step(i)
            total_energy = stats.get('total_energy', 0.0)
            total_energy_series.append(total_energy)
            
            if i % 3 == 0:  # 每3步输出一次
                print(f"Step {i}: generated={stats.get('generated_tasks', 0)}, energy={total_energy:.2f}J")
        
        # 最终统计
        final_stats = sim.calculate_final_statistics()
        breakdown = final_stats.get('energy_breakdown', {})
        
        print(f"\nFinal Results:")
        print(f"Total tasks: {final_stats['total_tasks']}")
        print(f"Completed: {final_stats['completed_tasks']}")
        print(f"Total energy: {final_stats['total_energy']:.2f}J")
        print(f"Breakdown:")
        for component, energy in breakdown.items():
            print(f"  {component}: {energy:.2f}J")
            
        # 能耗合理性检查
        total_expected = sum(breakdown.values())
        print(f"Sum of components: {total_expected:.2f}J")
        print(f"Difference: {abs(final_stats['total_energy'] - total_expected):.2f}J")
        
        return final_stats
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    test_energy_calculation()
