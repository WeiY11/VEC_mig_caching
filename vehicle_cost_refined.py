#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆数量成本分析图 - 极简精致版

逻辑:
- 固定带宽为 50 MHz
- 变化车辆数为 [6, 8, 10, 12, 14]
- 归一化基准与 bandwidth_cost_refined.py 保持一致 (基于 30MHz, 12Veh)
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# ==========================================
# 1. 系统参数 (保持完全一致)
# ==========================================
NOISE_POWER_DBM = -174.0
NOISE_FIGURE = 9.0
TX_POWER = 0.15
DISTANCE = 150.0
FREQUENCY = 3.5e9
TASK_ARRIVAL_RATE = 3.5
AVG_DATA_SIZE_MB = 7.5
EPISODE_DURATION = 20.0
RSU_CPU_FREQ = 17.5e9
TASK_COMPUTE_DENSITY = 40.0  # 保持 40.0 cycles/bit

def dbm_to_watt(dbm): return 10 ** ((dbm - 30) / 10)

def calculate_metrics(bandwidth_hz, num_vehicles):
    """计算指定带宽和车辆数下的指标"""
    # 通信模型
    c = 3e8
    pl_db = 20 * np.log10(DISTANCE) + 20 * np.log10(FREQUENCY) + 20 * np.log10(4 * np.pi / c)
    rx_power_dbm = 10 * np.log10(TX_POWER * 1000) - pl_db
    rx_power_w = dbm_to_watt(rx_power_dbm)
    noise_w = dbm_to_watt(NOISE_POWER_DBM) * bandwidth_hz * (10 ** (NOISE_FIGURE / 10))
    snr = rx_power_w / noise_w
    capacity = bandwidth_hz * np.log2(1 + snr)
    
    # 时延计算
    data_bits = AVG_DATA_SIZE_MB * 1024 * 1024 * 8
    trans_delay = data_bits / capacity
    trans_energy = TX_POWER * trans_delay
    
    # 计算时延
    proc_delay = (data_bits * TASK_COMPUTE_DENSITY) / RSU_CPU_FREQ
    proc_energy = (5.0e-32 * (RSU_CPU_FREQ**3) + 25.0) * proc_delay
    
    # 总指标
    # 注意：在简单理论模型中，单任务时延不随车辆数变化(除非引入排队论)，但总能耗随车辆数线性增加
    total_delay = 2 * trans_delay + proc_delay
    total_energy = (num_vehicles * TASK_ARRIVAL_RATE * EPISODE_DURATION) * (2 * trans_energy + proc_energy)
    
    return total_delay, total_energy

def main():
    # 1. 获取归一化基准 (必须与带宽图保持一致)
    # 带宽图中，最大值出现在 30MHz (12辆车)
    ref_delay, ref_energy = calculate_metrics(30 * 1e6, 12)
    d_ref = ref_delay * 1.5
    e_ref = ref_energy * 1.5
    
    # 2. 准备实验数据
    target_bandwidth = 50 * 1e6
    vehicle_counts = [6, 8, 10, 12, 14]
    
    costs = []
    for v in vehicle_counts:
        d, e = calculate_metrics(target_bandwidth, v)
        # 计算归一化成本
        cost = 0.5 * (d / d_ref) + 0.5 * (e / e_ref)
        costs.append(cost)
        
    # ==========================================
    # 3. 绘图 - 极简精致风格
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    line_color = '#003366'
    fill_color = '#E6F0FF'
    marker_fill = 'white'
    
    # 绘制阴影和曲线
    ax.fill_between(vehicle_counts, costs, 0, color=fill_color, alpha=0.5)
    ax.plot(vehicle_counts, costs, color=line_color, linewidth=2.5, linestyle='-', zorder=10)
    ax.plot(vehicle_counts, costs, marker='o', markersize=9, linestyle='None', 
            markerfacecolor=marker_fill, markeredgecolor=line_color, markeredgewidth=2, zorder=11)
    
    # 坐标轴美化
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # 网格
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#999999', zorder=0)
    ax.grid(axis='x', visible=False)
    
    # 标签
    ax.set_xlabel('Number of Vehicles', fontsize=12, fontweight='bold', labelpad=10, color='#333333')
    ax.set_ylabel('Average Cost (Normalized)', fontsize=12, fontweight='bold', labelpad=10, color='#333333')
    
    # 刻度
    ax.set_xticks(vehicle_counts)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5, color='#333333')
    
    # 自动调整Y轴范围
    y_min = min(costs) * 0.9
    y_max = max(costs) * 1.1
    ax.set_ylim(y_min, y_max)
    
    # 数值标签
    for x, y in zip(vehicle_counts, costs):
        ax.annotate(f'{y:.3f}', xy=(x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=line_color)

    plt.tight_layout()
    
    output_file = f"vehicle_cost_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 车辆成本分析图已保存: {output_file}")
    
    # 验证一致性
    print(f"验证: 12辆车时的成本 = {costs[3]:.3f}")
    plt.close()

if __name__ == "__main__":
    main()
