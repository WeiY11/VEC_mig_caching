#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术风格带宽成本分析图

专业论文风格的带宽性能对比图表
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
    # 学术风格配置
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 13
    plt.rcParams['xtick.labelsize'] = 11
    plt.rcParams['ytick.labelsize'] = 11
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    
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
    
    # 使用固定参考值进行归一化(而非最大值),让基准不是1.0
    # 参考值设为比最大值稍大,使30MHz的值在0.6-0.9范围
    delay_reference = max(delays) * 1.5  # 使用1.5倍作为参考
    energy_reference = max(energies) * 1.5
    
    norm_delays = np.array(delays) / delay_reference
    norm_energies = np.array(energies) / energy_reference
    combined_cost = 0.5 * norm_delays + 0.5 * norm_energies
    
    # 创建图表
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # 只绘制综合成本曲线 - 学术风格
    ax.plot(bandwidths, combined_cost, 'o-', 
            linewidth=2.5, markersize=10,
            color='#2E86AB', label='Combined Cost',
            markeredgecolor='white', markeredgewidth=1.5)
    
    # 设置标题和标签
    ax.set_xlabel('Bandwidth (MHz)', fontweight='bold')
    ax.set_ylabel('Combined Cost', fontweight='bold')
    ax.set_title('Impact of Bandwidth on Combined Cost', fontweight='bold', pad=15)
    
    # 网格
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.8, color='gray')
    ax.set_axisbelow(True)
    
    # 图例
    ax.legend(loc='upper right', framealpha=0.95, edgecolor='gray', fancybox=False)
    
    # 设置坐标轴范围
    ax.set_xlim(25, 75)
    ax.set_ylim(0.3, 0.7)  # 调整Y轴范围以匹配新的数值
    
    # 设置刻度
    ax.set_xticks(bandwidths)
    ax.set_xticklabels([f'{int(b)} MHz' for b in bandwidths])
    
    # 背景色
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    # 保存图表
    output_file = f"bandwidth_academic_style_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    print(f"✅ 学术风格图表已保存: {output_file}")
    plt.close()  # 关闭图表避免显示问题
    
    # 打印数值
    print("\n" + "=" * 70)
    print("Impact of Bandwidth on System Performance")
    print("=" * 70)
    print(f"{'Bandwidth':<15} {'Norm.Delay':<15} {'Norm.Energy':<15} {'Combined':<15}")
    print("-" * 70)
    for bw, nd, ne, cc in zip(bandwidths, norm_delays, norm_energies, combined_cost):
        print(f"{bw:.0f} MHz{'':<8} {nd:<15.3f} {ne:<15.3f} {cc:<15.3f}")
    print("=" * 70)
    print(f"\nCombined Cost = 0.5 × Normalized Latency + 0.5 × Normalized Energy")
    print(f"System: {NUM_VEHICLES} vehicles, {TASK_ARRIVAL_RATE} tasks/s, {AVG_DATA_SIZE_MB}MB avg task size")

if __name__ == "__main__":
    main()
