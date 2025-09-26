#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时参数检查器 - 验证代码运行时实际使用的参数
"""

import sys
import os

# 添加项目根目录到路径
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

try:
    from config import config as sys_config
    from evaluation.system_simulator import CompleteSystemSimulator
except Exception as e:
    print(f"导入失败: {e}")
    exit(1)


def check_actual_params():
    """检查代码运行时实际使用的参数"""
    print("=== 实际运行参数检查 ===")
    
    # 创建仿真器实例，提供完整配置
    config = {
        "num_vehicles": 2, "num_rsus": 4, "num_uavs": 2,
        "cache_capacity": 100, "computation_capacity": 1000, 
        "bandwidth": 20, "transmission_power": 0.1, "computation_power": 1.0,
        "time_slot": 0.2, "simulation_time": 100, "task_arrival_rate": 1.0
    }
    sim = CompleteSystemSimulator(config)
    
    # 创建测试任务
    test_task = {
        'id': 'test_task',
        'vehicle_id': 'V_0',
        'data_size': 1.0,  # MB
        'data_size_bytes': 1e6,
        'compute_density': 400,
        'computation_requirement': 100
    }
    
    # 测试RSU处理
    rsu_node = sim.rsus[0]  # 使用第一个RSU
    
    print(f"RSU节点: {rsu_node}")
    print(f"sys_config可用: {sys_config is not None}")
    
    if sys_config is not None:
        # 打印配置参数
        print(f"配置RSU频率: {sys_config.compute.rsu_default_freq/1e9:.1f}GHz")
        print(f"配置RSU kappa: {sys_config.compute.rsu_kappa:.2e}")
        print(f"配置RSU静态功耗: {sys_config.compute.rsu_static_power}W")
        
        # 实际获取参数的代码路径
        cpu_freq = float(getattr(sys_config.compute, 'rsu_default_freq', 6e9))
        kappa = float(getattr(sys_config.compute, 'rsu_kappa', 2.8e-31))
        static_power = float(getattr(sys_config.compute, 'rsu_static_power', 25.0))
        
        print(f"实际获取RSU频率: {cpu_freq/1e9:.1f}GHz")
        print(f"实际获取RSU kappa: {kappa:.2e}")
        print(f"实际获取RSU静态功耗: {static_power}W")
        
        # 计算示例能耗
        task_compute_density = 100  # 修复后的密度
        data_bytes = min(1e6, 5e6)
        bits = data_bytes * 8.0
        total_cycles = bits * task_compute_density
        exec_time = total_cycles / cpu_freq
        dynamic_power = kappa * (cpu_freq ** 3)
        total_power = dynamic_power + static_power
        computation_energy = total_power * exec_time
        
        print(f"\n--- 实际计算过程 ---")
        print(f"数据大小: {data_bytes/1e6:.1f}MB = {bits:.0f} bits")
        print(f"计算周期: {total_cycles:.0f}")
        print(f"执行时间: {exec_time:.3f}s")
        print(f"动态功率: {dynamic_power:.1f}W")
        print(f"总功率: {total_power:.1f}W")
        print(f"计算能耗: {computation_energy:.3f}J")
        
        # 检查问题
        if total_power > 100:
            print(f"❌ 功率异常高: {total_power:.1f}W")
        if exec_time > 1.0:
            print(f"❌ 执行时间过长: {exec_time:.3f}s")
        if computation_energy > 100:
            print(f"❌ 计算能耗过高: {computation_energy:.3f}J")
            
        # UAV悬停能耗检查
        hover_power = float(getattr(sys_config.compute, 'uav_hover_power', 25.0))
        time_slot = 0.2
        num_uavs = len(sim.uavs)
        expected_hover_per_step = hover_power * time_slot * num_uavs
        expected_hover_100_steps = expected_hover_per_step * 100
        
        print(f"\n--- UAV悬停能耗检查 ---")
        print(f"悬停功率: {hover_power}W")
        print(f"时隙长度: {time_slot}s")
        print(f"UAV数量: {num_uavs}")
        print(f"每步悬停能耗: {expected_hover_per_step}J")
        print(f"100步预期悬停能耗: {expected_hover_100_steps}J")


if __name__ == "__main__":
    check_actual_params()
