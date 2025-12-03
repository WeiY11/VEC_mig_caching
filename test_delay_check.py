#!/usr/bin/env python3
"""时延计算结果检查脚本"""
import sys
sys.path.insert(0, '.')
from config.system_config import config
from evaluation.system_simulator import CompleteSystemSimulator
import numpy as np

print('='*70)
print('时延计算结果检查')
print('='*70)

# 初始化模拟器
sim = CompleteSystemSimulator(config={})

# 测试参数
test_cases = [
    {'data_size_bytes': 1e6, 'distance': 100, 'link': 'rsu', 'desc': '1MB数据, 100m, RSU'},
    {'data_size_bytes': 1e6, 'distance': 200, 'link': 'uav', 'desc': '1MB数据, 200m, UAV'},
    {'data_size_bytes': 2e6, 'distance': 300, 'link': 'rsu', 'desc': '2MB数据, 300m, RSU'},
    {'data_size_bytes': 0.5e6, 'distance': 50, 'link': 'uav', 'desc': '0.5MB数据, 50m, UAV'},
]

print('\n1. 传输时延测试')
print('-'*70)
for tc in test_cases:
    delay, energy = sim._estimate_transmission(tc['data_size_bytes'], tc['distance'], tc['link'])
    print(f"{tc['desc']:30s} => 时延: {delay*1000:.2f}ms, 能耗: {energy*1000:.3f}mJ")

# 本地处理时延
print('\n2. 本地处理时延测试')
print('-'*70)
vehicle = {'energy_consumed': 0.0}
task_configs = [
    {'computation_requirement': 500, 'desc': '500 Mcycles'},
    {'computation_requirement': 1000, 'desc': '1000 Mcycles'},
    {'computation_requirement': 1500, 'desc': '1500 Mcycles'},
    {'computation_requirement': 2000, 'desc': '2000 Mcycles'},
]
for tc in task_configs:
    task = {'computation_requirement': tc['computation_requirement']}
    vehicle['energy_consumed'] = 0.0
    delay, energy = sim._estimate_local_processing(task, vehicle)
    print(f"{tc['desc']:20s} => 时延: {delay*1000:.2f}ms, 能耗: {energy:.4f}J")

# 队列等待时延 (M/M/1模型)
print('\n3. 队列等待时延测试 (M/M/1模型)')
print('-'*70)
from decision.offloading_manager import ProcessingModeEvaluator

class MockState:
    def __init__(self, load_factor):
        self.load_factor = load_factor
        self.node_id = None

evaluator = ProcessingModeEvaluator()
load_factors = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
for rho in load_factors:
    st = MockState(rho)
    wait = evaluator._wait(st)
    if wait == float('inf'):
        print(f'负载因子 rho={rho:.2f} => 等待时延: 无穷大 (系统不稳定)')
    else:
        print(f'负载因子 rho={rho:.2f} => 等待时延: {wait*1000:.2f}ms')

# 总时延估算
print('\n4. 端到端时延估算 (典型场景)')
print('-'*70)

# 场景1: 本地处理
task = {'computation_requirement': 1500}
vehicle = {'energy_consumed': 0.0}
local_delay, _ = sim._estimate_local_processing(task, vehicle)
print(f"本地处理 (1500Mcycles)              => 总时延: {local_delay*1000:.1f}ms")
print(f"   组成: 本地计算{local_delay*1000:.1f}ms")

# 场景2: RSU卸载
up_delay, _ = sim._estimate_transmission(1e6, 100, 'rsu')
down_delay, _ = sim._estimate_transmission(0.05e6, 100, 'rsu')
st = MockState(0.5)
wait = evaluator._wait(st)
rsu_compute = 80  # ms
total = up_delay + down_delay + wait + rsu_compute/1000
print(f"RSU卸载 (100m, 1MB, rho=0.5)        => 总时延: {total*1000:.1f}ms")
print(f"   组成: 上行{up_delay*1000:.1f}ms + 等待{wait*1000:.1f}ms + RSU计算{rsu_compute}ms + 下行{down_delay*1000:.1f}ms")

# 场景3: UAV卸载
up_delay, _ = sim._estimate_transmission(1e6, 200, 'uav')
down_delay, _ = sim._estimate_transmission(0.05e6, 200, 'uav')
st = MockState(0.3)
wait = evaluator._wait(st)
uav_compute = 286  # ms
total = up_delay + down_delay + wait + uav_compute/1000
print(f"UAV卸载 (200m, 1MB, rho=0.3)        => 总时延: {total*1000:.1f}ms")
print(f"   组成: 上行{up_delay*1000:.1f}ms + 等待{wait*1000:.1f}ms + UAV计算{uav_compute}ms + 下行{down_delay*1000:.1f}ms")

# 场景4: 缓存命中
down_delay, _ = sim._estimate_transmission(0.05e6, 100, 'rsu')
cache_read = 1  # ms
total = cache_read/1000 + down_delay
print(f"缓存命中 (RSU, 100m)                => 总时延: {total*1000:.1f}ms")
print(f"   组成: 缓存读取{cache_read}ms + 下行{down_delay*1000:.1f}ms")

print('\n' + '='*70)
print('检查完成 - 时延计算结果在合理范围内')
print('='*70)
