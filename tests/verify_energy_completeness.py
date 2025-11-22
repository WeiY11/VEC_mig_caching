#!/usr/bin/env python3
"""
能耗计算完整性验证脚本
Verify that all energy components are properly tracked
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.system_simulator import CompleteSystemSimulator
from config import config
import numpy as np

def test_energy_completeness():
    """测试能耗计算的完整性"""
    print("=" * 80)
    print("能耗计算完整性测试 (Energy Calculation Completeness Test)")
    print("=" * 80)
    
    # 创建仿真器
    scenario_config = {
        'num_vehicles': 4,
        'num_rsus': 2,
        'num_uavs': 1,
        'task_arrival_rate': 2.0,  # 2 tasks/s
    }
    
    simulator = CompleteSystemSimulator(scenario_config)
    
    # 运行几个仿真步
    num_steps = 10
    print(f"\n运行 {num_steps} 个仿真步...")
    
    for step in range(num_steps):
        # Simple action - encourage remote offloading
        actions = {
            'offload_preference': {
                'local': 0.2,
                'rsu': 0.5,
                'uav': 0.3
            }
        }
        
        step_stats = simulator.run_simulation_step(step, actions)
    
    # 分析能耗统计
    print("\n" + "=" * 80)
    print("能耗分解 (Energy Breakdown)")
    print("=" * 80)
    
    stats = simulator.stats
    
    # 提取能耗组件
    total_energy = stats.get('total_energy', 0.0)
    energy_compute = stats.get('energy_compute', 0.0)
    energy_uplink = stats.get('energy_transmit_uplink', 0.0)
    energy_downlink = stats.get('energy_transmit_downlink', 0.0)
    energy_cache = stats.get('energy_cache', 0.0)
    
    print(f"\n总能耗 (Total Energy):        {total_energy:>12.2f} J")
    print(f"  - 计算能耗 (Compute):         {energy_compute:>12.2f} J  ({energy_compute/max(total_energy, 1)*100:>5.1f}%)")
    print(f"  - 上行传输 (Uplink TX):       {energy_uplink:>12.2f} J  ({energy_uplink/max(total_energy, 1)*100:>5.1f}%)")
    print(f"  - 下行传输 (Downlink TX):     {energy_downlink:>12.2f} J  ({energy_downlink/max(total_energy, 1)*100:>5.1f}%)")
    print(f"  - 缓存访问 (Cache):          {energy_cache:>12.2f} J  ({energy_cache/max(total_energy, 1)*100:>5.1f}%)")
    
    # 验证能耗守恒
    components_sum = energy_compute + energy_uplink + energy_downlink + energy_cache
    discrepancy = abs(total_energy - components_sum)
    
    print(f"\n组件总和 (Components Sum):    {components_sum:>12.2f} J")
    print(f"误差 (Discrepancy):          {discrepancy:>12.2f} J  ({discrepancy/max(total_energy, 1)*100:>5.1f}%)")
    
    # 检查是否通过
    if discrepancy < 0.01 * total_energy or discrepancy < 1.0:  # 1% tolerance or 1J
        print("\n✅ 测试通过 (PASS): 能耗组件完整，守恒性良好")
        status = "PASS"
    else:
        print("\n❌ 测试失败 (FAIL): 能耗组件不完整或存在计算错误")
        status = "FAIL"
    
    # 延迟分解
    print("\n" + "=" * 80)
    print("延迟分解 (Delay Breakdown)")
    print("=" * 80)
    
    total_delay = stats.get('total_delay', 0.0)
    delay_processing = stats.get('delay_processing', 0.0)
    delay_waiting = stats.get('delay_waiting', 0.0)
    delay_uplink = stats.get('delay_uplink', 0.0)
    delay_downlink = stats.get('delay_downlink', 0.0)
    delay_cache = stats.get('delay_cache', 0.0)
    
    print(f"\n总延迟 (Total Delay):        {total_delay:>12.4f} s")
    print(f"  - 处理延迟 (Processing):     {delay_processing:>12.4f} s  ({delay_processing/max(total_delay, 1)*100:>5.1f}%)")
    print(f"  - 等待延迟 (Waiting):        {delay_waiting:>12.4f} s  ({delay_waiting/max(total_delay, 1)*100:>5.1f}%)")
    print(f"  - 上行延迟 (Uplink):         {delay_uplink:>12.4f} s  ({delay_uplink/max(total_delay, 1)*100:>5.1f}%)")
    print(f"  - 下行延迟 (Downlink):       {delay_downlink:>12.4f} s  ({delay_downlink/max(total_delay, 1)*100:>5.1f}%)")
    print(f"  - 缓存延迟 (Cache):          {delay_cache:>12.4f} s  ({delay_cache/max(total_delay, 1)*100:>5.1f}%)")
    
    # 任务统计
    print("\n" + "=" * 80)
    print("任务统计 (Task Statistics)")
    print("=" * 80)
    
    print(f"\n总任务 (Total Tasks):        {stats.get('total_tasks', 0)}")
    print(f"完成任务 (Completed):         {stats.get('completed_tasks', 0)}")
    print(f"丢弃任务 (Dropped):          {stats.get('dropped_tasks', 0)}")
    print(f"完成率 (Completion Rate):    {stats.get('completed_tasks', 0) / max(stats.get('total_tasks', 1), 1) * 100:.1f}%")
    
    # 卸载统计
    print("\n缓存命中率 (Cache Hit Rate):  {:.1%}".format(
        stats.get('cache_hits', 0) / max(stats.get('cache_requests', 1), 1)
    ))
    
    print("\n" + "=" * 80)
    print(f"测试结果: {status}")
    print("=" * 80)
    
    return status == "PASS"

if __name__ == "__main__":
    try:
        success = test_energy_completeness()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试异常 (Exception): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
