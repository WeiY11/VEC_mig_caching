#!/usr/bin/env python3
"""简单的能耗测试"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.system_simulator import CompleteSystemSimulator
import traceback

try:
    print("Creating simulator...")
    scenario_config = {
        'num_vehicles': 2,
        'num_rsus': 1,
        'num_uavs': 0,
        'task_arrival_rate': 1.0,
    }
    
    simulator = CompleteSystemSimulator(scenario_config)
    print("Simulator created successfully")
    
    print("\nRunning 5 simulation steps...")
    for step in range(5):
        actions = {
            'offload_preference': {
                'local': 0.3,
                'rsu': 0.7,
                'uav': 0.0
            }
        }
        
        try:
            step_stats = simulator.run_simulation_step(step, actions)
            print(f"  Step {step}: OK")
        except Exception as e:
            print(f"  Step {step}: FAILED - {e}")
            traceback.print_exc()
            break
    
    print("\nEnergy Stats:")
    stats = simulator.stats
    print(f"  Total Energy:      {stats.get('total_energy', 0):.2f} J")
    print(f"  Compute:           {stats.get('energy_compute', 0):.2f} J")
    print(f"  Uplink TX:         {stats.get('energy_transmit_uplink', 0):.2f} J")
    print(f"  Downlink TX:       {stats.get('energy_transmit_downlink', 0):.2f} J")
    print(f"  Cache:             {stats.get('energy_cache', 0):.2f} J")
    
    total = stats.get('total_energy', 0)
    sum_components = (stats.get('energy_compute', 0) + 
                      stats.get('energy_transmit_uplink', 0) + 
                      stats.get('energy_transmit_downlink', 0) + 
                      stats.get('energy_cache', 0))
    
    print(f"\n  Sum of components: {sum_components:.2f} J")
    print(f"  Difference:        {abs(total - sum_components):.2f} J")
    
    if abs(total - sum_components) < 1.0:
        print("\n✅ Energy conservation OK")
    else:
        print("\n⚠️ Energy mismatch detected")
    
except Exception as e:
    print(f"\n❌ Test failed: {e}")
    traceback.print_exc()
    sys.exit(1)
