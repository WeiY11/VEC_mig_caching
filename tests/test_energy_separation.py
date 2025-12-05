"""
快速验证静态/动态能耗分离功能
"""
from evaluation.system_simulator import CompleteSystemSimulator

# 创建一个最小配置
config = {
    'num_vehicles': 2,
    'num_rsus': 1, 
    'num_uavs': 1,
    'time_slot': 0.1
}

sim = CompleteSystemSimulator(config)
sim.reset()

# 运行10步模拟
for i in range(10):
    stats = sim.run_simulation_step(i)

# 检查能耗统计
print('=== 能耗统计验证 ===')
print(f'total_energy: {stats.get("total_energy", 0):.4f} J')
print(f'energy_compute: {stats.get("energy_compute", 0):.4f} J')
print(f'energy_static: {stats.get("energy_static", 0):.4f} J')
print(f'energy_dynamic: {stats.get("energy_dynamic", 0):.4f} J')

# 验证：static + dynamic 应该接近 energy_compute
static = stats.get('energy_static', 0)
dynamic = stats.get('energy_dynamic', 0)
compute = stats.get('energy_compute', 0)
print(f'\n验证 static + dynamic = {static + dynamic:.4f} J')
print(f'energy_compute = {compute:.4f} J')
print(f'差值: {abs(compute - (static + dynamic)):.6f} J')

if abs(compute - (static + dynamic)) < 0.001:
    print('✅ 能耗分解正确')
else:
    print('❌ 能耗分解有误差')
