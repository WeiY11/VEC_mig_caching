#!/usr/bin/env python3
"""
快速测试参数扫描功能
运行3个车辆数配置，2个算法，各20轮
"""
import sys
import io
from pathlib import Path

# 设置UTF-8输出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 添加父目录
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from run_parameter_sweep import ParameterSweepExperiment

print("="*80)
print("参数对比折线图 - 快速测试")
print("="*80)
print("配置:")
print("  车辆数: 8, 12, 16")
print("  算法: TD3, Greedy")
print("  轮次: 20")
print("  固定拓扑: 4 RSU + 2 UAV")
print("预计时间: 约5分钟")
print("="*80)

# 创建扫描实验
sweep_exp = ParameterSweepExperiment()

# 运行车辆数扫描
vehicle_counts = [8, 12, 16]
algorithms = ['TD3', 'Greedy']
episodes = 20

sweep_exp.run_vehicle_sweep(
    vehicle_counts=vehicle_counts,
    algorithms=algorithms,
    episodes_per_config=episodes,
    seed=42
)

print("\n" + "="*80)
print("快速测试完成！")
print("="*80)
print("查看生成的参数对比折线图:")
print(f"  {sweep_exp.analysis_dir}/parameter_comparison_lines.png")
print("")
print("图表包含:")
print("  - 4个子图（时延、能耗、完成率、目标函数）")
print("  - X轴: 车辆数（8, 12, 16）")
print("  - 多条线: 不同算法（TD3 vs Greedy）")
print("  - 数据点标记: 清晰可见")
print("="*80)


