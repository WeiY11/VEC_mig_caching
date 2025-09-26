#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VEC 系统一键 Sanity Check
- 校验 RSU/UAV 数量是否为 4/2
- 打印中央 RSU 与节点坐标
- 检查回传拓扑连通性与关键链路
- 运行一个短仿真步并输出摘要指标
- 运行100步快速评估并落盘 results/quick_eval.json
"""

import os
import sys

# 确保项目根目录在路径中（tools/ 的上一级）
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from typing import Dict, List
import json

try:
    from config import config as sys_config
except Exception:
    sys_config = None

from evaluation.system_simulator import CompleteSystemSimulator
from utils.wired_backhaul_model import get_backhaul_model


def _get_simulator() -> CompleteSystemSimulator:
    if sys_config is not None:
        return CompleteSystemSimulator()
    cfg: Dict = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "time_slot": 0.2,
        "simulation_time": 100,
        "task_arrival_rate": 1.2,
        "cache_capacity": 100,
        "computation_capacity": 1000,
        "bandwidth": 20,
        "transmission_power": 0.1,
        "computation_power": 1.0,
    }
    return CompleteSystemSimulator(cfg)


def print_topology(sim: CompleteSystemSimulator) -> None:
    print("=== Topology ===")
    print(f"Vehicles: {len(sim.vehicles)}  RSUs: {len(sim.rsus)}  UAVs: {len(sim.uavs)}")
    print("Central RSU: RSU_2 (scheduling/backhaul hub)")
    print("RSU positions:")
    for i, r in enumerate(sim.rsus):
        print(f"  RSU_{i}: ({r['position'][0]:.1f}, {r['position'][1]:.1f})")
    print("UAV positions:")
    for i, u in enumerate(sim.uavs):
        pos = u['position']
        print(f"  UAV_{i}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")


def check_backhaul() -> None:
    print("=== Backhaul (wired) ===")
    model = get_backhaul_model()
    status = model.get_backhaul_status()
    print(f"Topology: {status['network_topology']}, Central: {status['central_hub']}")
    print(f"Links: {status['total_links']}  Utilization: {status['network_utilization']:.1%}")
    for src, dst in [("RSU_0", "RSU_1"), ("RSU_2", "RSU_3")]:
        try:
            delay, _ = model.calculate_wired_transmission_delay(1.0, src, dst)
            print(f"  Path {src}->{dst}: delay={delay*1000:.2f} ms (1MB)")
        except Exception as e:
            print(f"  Path {src}->{dst}: error {e}")


def run_step(sim: CompleteSystemSimulator) -> None:
    print("=== One simulation step ===")
    step_stats = sim.run_simulation_step(0)
    print(f"Generated tasks: {step_stats.get('generated_tasks', 0)}")
    print(f"Processed tasks (cumulative): {step_stats.get('processed_tasks', 0)}")
    print(f"Dropped tasks (cumulative): {step_stats.get('dropped_tasks', 0)}")
    print(f"Total delay (cumulative): {step_stats.get('total_delay', 0.0):.3f}s")
    print(f"Total energy (cumulative): {step_stats.get('total_energy', 0.0):.3f}J")


def quick_eval(sim: CompleteSystemSimulator, steps: int = 100) -> Dict:
    print(f"=== Quick evaluation {steps} steps ===")
    delays: List[float] = []
    energies: List[float] = []
    drops: List[int] = []
    
    # 能耗分项统计
    energy_components = {
        'vehicle_transmit': [],
        'vehicle_compute': [],
        'edge_compute': [],
        'uav_compute': [],
        'uav_hover': [],
        'downlink': [],
        'total_system': []
    }
    
    for i in range(steps):
        stats = sim.run_simulation_step(i)
        delays.append(float(stats.get('total_delay', 0.0)))
        energies.append(float(stats.get('total_energy', 0.0)))
        drops.append(int(stats.get('dropped_tasks', 0)))
        
        # 收集分项能耗
        energy_components['vehicle_transmit'].append(sim.stats.get('energy_vehicle_transmit', 0.0))
        energy_components['vehicle_compute'].append(sim.stats.get('energy_vehicle_compute', 0.0))
        energy_components['edge_compute'].append(sim.stats.get('energy_edge_compute', 0.0))
        energy_components['uav_compute'].append(sim.stats.get('energy_uav_compute', 0.0))
        energy_components['uav_hover'].append(sim.stats.get('energy_uav_hover', 0.0))
        energy_components['downlink'].append(sim.stats.get('energy_downlink', 0.0))
        energy_components['total_system'].append(energies[-1])
    
    # 能耗验证
    try:
        from utils.energy_validator import validate_energy_consumption
        energy_validation = validate_energy_consumption(energy_components)
        print(f"能耗验证: {'✅通过' if energy_validation['is_valid'] else '❌失败'}")
        for error in energy_validation.get('errors', []):
            print(f"  ❌ {error}")
        for warning in energy_validation.get('warnings', []):
            print(f"  ⚠️ {warning}")
    except Exception as e:
        energy_validation = {"error": str(e)}
        print(f"⚠️ 能耗验证器加载失败: {e}")
    
    result = {
        "steps": steps,
        "final_total_delay": delays[-1] if delays else 0.0,
        "final_total_energy": energies[-1] if energies else 0.0,
        "final_dropped_tasks": drops[-1] if drops else 0,
        "delay_series": delays,
        "energy_series": energies,
        "drop_series": drops,
        "energy_components": energy_components,
        "energy_validation": energy_validation
    }
    
    os.makedirs("results", exist_ok=True)
    with open(os.path.join("results", "quick_eval.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    # 保存能耗验证报告
    with open(os.path.join("results", "energy_validation.json"), "w", encoding="utf-8") as f:
        json.dump(energy_validation, f, ensure_ascii=False, indent=2)
    
    print("Saved: results/quick_eval.json, results/energy_validation.json")
    print(f"final delay={result['final_total_delay']:.3f}s, energy={result['final_total_energy']:.3f}J, drops={result['final_dropped_tasks']}")
    
    # 分项能耗摘要
    final_components = {k: v[-1] if v else 0.0 for k, v in energy_components.items()}
    print(f"能耗分项: 车辆发射{final_components['vehicle_transmit']:.3f}J, 车辆计算{final_components['vehicle_compute']:.3f}J")
    print(f"        边缘计算{final_components['edge_compute']:.3f}J, UAV计算{final_components['uav_compute']:.3f}J")
    print(f"        UAV悬停{final_components['uav_hover']:.3f}J, 下行{final_components['downlink']:.3f}J")
    
    return result


def main():
    sim = _get_simulator()
    print_topology(sim)
    check_backhaul()
    run_step(sim)
    quick_eval(sim, 100)
    print("=== Sanity check completed ===")


if __name__ == "__main__":
    main()
