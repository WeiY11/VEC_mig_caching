"""
缓存命中率诊断脚本
快速验证任务生成和缓存逻辑是否正常工作
"""

import sys
sys.path.insert(0, 'd:/VEC_mig_caching')

import numpy as np
from config import config

# 测试任务生成
from models.vehicle_node import VehicleNode
from models.data_structures import Position

print("=" * 60)
print("🔍 缓存任务生成诊断")
print("=" * 60)

# 创建测试车辆
test_vehicle = VehicleNode("TEST_V", Position(500, 500, 0))

# 生成100个任务样本
total_tasks = 100
cacheable_count = 0
content_id_count = 0
task_types = {1: 0, 2: 0, 3: 0, 4: 0}

for i in range(total_tasks):
    task = test_vehicle._create_random_task()
    
    task_type_value = task.task_type.value
    task_types[task_type_value] = task_types.get(task_type_value, 0) + 1
    
    if task.is_cacheable:
        cacheable_count += 1
    
    if task.content_id is not None:
        content_id_count += 1

print(f"\n📊 任务生成统计 (样本数: {total_tasks})")
print(f"  任务类型分布:")
for t_type, count in sorted(task_types.items()):
    print(f"    类型{t_type}: {count}个 ({count/total_tasks*100:.1f}%)")

print(f"\n🎯 可缓存性统计:")
print(f"  is_cacheable=True: {cacheable_count}个 ({cacheable_count/total_tasks*100:.1f}%)")
print(f"  有content_id: {content_id_count}个 ({content_id_count/total_tasks*100:.1f}%)")

print(f"\n✅ 预期值:")
print(f"  可缓存任务比例: 约75% (类型1:50%, 2:80%, 3:90%, 4:85%)")
print(f"  有content_id比例: 应该 = 可缓存任务比例")

if cacheable_count < 50:
    print(f"\n❌ 问题: 可缓存任务比例过低 ({cacheable_count}%)")
    print(f"   可能原因: 代码修复未生效")
elif content_id_count != cacheable_count:
    print(f"\n❌ 问题: content_id数量 ({content_id_count}) != 可缓存数 ({cacheable_count})")
    print(f"   可能原因: sample_zipf_content_id调用失败")
else:
    print(f"\n✅ 任务生成正常！")

print("\n" + "=" * 60)

# 测试缓存逻辑
print(f"\n🔍 缓存统计逻辑诊断")
print("=" * 60)

from evaluation.system_simulator import CompleteSystemSimulator

# 创建测试仿真器
test_scenario = {
    'num_vehicles': 2,
    'num_rsus': 1,
    'num_uavs': 1,
}

simulator = CompleteSystemSimulator(test_scenario)

# 手动测试缓存统计
rsu = simulator.rsus[0]
rsu['cache'] = {}

# 测试1: 有content_id的任务
test_hit = simulator.check_cache_hit_adaptive(
    content_id='content_0001',
    node=rsu,
    actions={},
    node_type='RSU'
)

hits1 = simulator.stats.get('cache_hits', 0)
misses1 = simulator.stats.get('cache_misses', 0)

# 测试2: 无content_id的任务
test_hit2 = simulator.check_cache_hit_adaptive(
    content_id=None,
    node=rsu,
    actions={},
    node_type='RSU'
)

hits2 = simulator.stats.get('cache_hits', 0)
misses2 = simulator.stats.get('cache_misses', 0)

print(f"\n测试1 (有content_id):")
print(f"  cache_hits: {hits1}, cache_misses: {misses1}")
print(f"  预期: cache_hits=0, cache_misses=1")

print(f"\n测试2 (无content_id，不应计入统计):")
print(f"  cache_hits: {hits2}, cache_misses: {misses2}")
print(f"  预期: cache_hits=0, cache_misses=1 (不变)")

if misses2 > misses1:
    print(f"\n❌ 问题: 无content_id任务被计入统计")
    print(f"   这会大幅降低缓存命中率")
else:
    print(f"\n✅ 缓存统计逻辑正常！")

print("\n" + "=" * 60)
