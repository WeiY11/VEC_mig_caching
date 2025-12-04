#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带宽成本对比图 - 仅显示综合成本

基于Shannon信道容量公式计算不同带宽下的综合成本
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# 系统参数配置
NOISE_POWER_DBM = -174.0
NOISE_FIGURE = 9.0
TX_POWER = 0.15  # W
DISTANCE = 150.0  # m
FREQUENCY = 3.5e9  # Hz
TASK_ARRIVAL_RATE = 3.5
NUM_VEHICLES = 12
AVG_DATA_SIZE_MB = 7.5
EPISODE_DURATION = 20.0
RSU_CPU_FREQ = 17.5e9
TASK_COMPUTE_DENSITY = 2.5

def dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)

def calculate_path_loss(distance, frequency):
    c = 3e8
    pl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
    return pl_db

def calculate_snr(bandwidth_hz):
    path_loss_db = calculate_path_loss(DISTANCE, FREQUENCY)
    rx_power_dbm = 10 * np.log10(TX_POWER * 1000) - path_loss_db
    rx_power_w = dbm_to_watt(rx_power_dbm)
    noise_density_w_hz = dbm_to_watt(NOISE_POWER_DBM)
    noise_power_w = noise_density_w_hz * bandwidth_hz * (10 ** (NOISE_FIGURE / 10))
    snr = rx_power_w / noise_power_w
    return snr

def calculate_shannon_capacity(bandwidth_hz):
    snr = calculate_snr(bandwidth_hz)
    capacity_bps = bandwidth_hz * np.log2(1 + snr)
    return capacity_bps

def calculate_transmission_delay(data_size_bytes, bandwidth_hz):
    capacity_bps = calculate_shannon_capacity(bandwidth_hz)
    data_size_bits = data_size_bytes * 8
    delay_s = data_size_bits / capacity_bps
    return delay_s

def calculate_transmission_energy(data_size_bytes, bandwidth_hz):
    delay_s = calculate_transmission_delay(data_size_bytes, bandwidth_hz)
    energy_j = TX_POWER * delay_s
    return energy_j

def calculate_processing_delay(data_size_bytes):
    data_size_bits = data_size_bytes * 8
    compute_cycles = data_size_bits * TASK_COMPUTE_DENSITY
    delay_s = compute_cycles / RSU_CPU_FREQ
    return delay_s

def calculate_processing_energy(data_size_bytes):
    kappa = 5.0e-32
    static_power = 25.0
    proc_time = calculate_processing_delay(data_size_bytes)
    dynamic_power = kappa * (RSU_CPU_FREQ ** 3)
    total_power = dynamic_power + static_power
    energy_j = total_power * proc_time
    return energy_j

def calculate_total_episode_metrics(bandwidth_hz):
    total_tasks = NUM_VEHICLES * TASK_ARRIVAL_RATE * EPISODE_DURATION
    avg_data_bytes = AVG_DATA_SIZE_MB * 1024 * 1024
    trans_delay = calculate_transmission_delay(avg_data_bytes, bandwidth_hz)
    trans_energy = calculate_transmission_energy(avg_data_bytes, bandwidth_hz)
    proc_delay = calculate_processing_delay(avg_data_bytes)
    proc_energy = calculate_processing_energy(avg_data_bytes)
    total_delay_per_task = 2 * trans_delay + proc_delay
    total_energy_per_task = 2 * trans_energy + proc_energy
    avg_delay = total_delay_per_task
    total_energy = total_tasks * total_energy_per_task
    
    return {
        'bandwidth_mhz': bandwidth_hz / 1e6,
        'avg_delay': avg_delay,
        'total_energy': total_energy,
    }

def main():
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 带宽范围 30-70 MHz
    bandwidths_mhz = [30, 40, 50, 60, 70]
    
    results = []
    for bw_mhz in bandwidths_mhz:
        bw_hz = bw_mhz * 1e6
        metrics = calculate_total_episode_metrics(bw_hz)
        results.append(metrics)
    
    # 提取数据
    bandwidths = [r['bandwidth_mhz'] for r in results]
    delays = [r['avg_delay'] for r in results]
    energies = [r['total_energy'] for r in results]
    
    # 计算归一化成本
    norm_delays = np.array(delays) / max(delays)
    norm_energies = np.array(energies) / max(energies)
    costs = 0.5 * norm_delays + 0.5 * norm_energies
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # 绘制成本曲线
    ax.plot(bandwidths, costs, 'd-', linewidth=3, markersize=14, 
            color='#9b59b6', label='归一化综合成本', markeredgecolor='white', markeredgewidth=2)
    
    # 设置标题和标签
    ax.set_xlabel('带宽 (MHz)', fontsize=16, fontweight='bold')
    ax.set_ylabel('归一化成本', fontsize=16, fontweight='bold')
    ax.set_title('带宽综合成本分析 (30-70 MHz)\n成本 = 0.5 × 归一化时延 + 0.5 × 归一化能耗', 
                 fontsize=18, fontweight='bold', pad=15)
    
    # 优化网格
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=1.2)
    ax.set_axisbelow(True)
    
    # 添加数值标注
    for i, (x, y) in enumerate(zip(bandwidths, costs)):
        ax.annotate(f'{y:.3f}', (x, y), 
                   textcoords="offset points", 
                   xytext=(0, 12), 
                   ha='center', 
                   fontsize=12, 
                   fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
    
    # 添加图例
    ax.legend(fontsize=13, loc='upper right', framealpha=0.9)
    
    # 设置y轴范围
    y_min = min(costs) * 0.95
    y_max = max(costs) * 1.05
    ax.set_ylim(y_min, y_max)
    
    # 添加底部说明
    fig.text(0.5, 0.02, 
             f'系统参数: {NUM_VEHICLES}车辆 × {TASK_ARRIVAL_RATE} tasks/s | 平均任务{AVG_DATA_SIZE_MB}MB | RSU {RSU_CPU_FREQ/1e9:.1f}GHz',
             ha='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    
    # 保存图表
    output_file = f"bandwidth_cost_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 成本对比图已保存: {output_file}")
    
    # 打印数值
    print("\n" + "=" * 60)
    print("📊 带宽综合成本对比 (30-70 MHz)")
    print("=" * 60)
    print(f"{'带宽(MHz)':<12} {'归一化成本':<15} {'成本降低':<12}")
    print("-" * 60)
    for i, (bw, cost) in enumerate(zip(bandwidths, costs)):
        if i == 0:
            improvement = "基准"
        else:
            improvement = f"-{(costs[0] - cost) / costs[0] * 100:.1f}%"
        print(f"{bw:<12.0f} {cost:<15.3f} {improvement:<12}")
    print("=" * 60)
    
    plt.show()

if __name__ == "__main__":
    main()
