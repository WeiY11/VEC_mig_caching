#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带宽理论成本分析脚本

基于Shannon信道容量公式计算不同带宽下的理论性能指标:
- 传输时延
- 通信能耗
- 系统吞吐量
- 综合成本

公式基础:
1. Shannon容量: C = B * log2(1 + SNR)  (bps)
2. 传输时延: T_trans = Data_size / C  (s)
3. 通信能耗: E_comm = P_tx * T_trans  (J)
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

# ================================================================================
# 系统参数配置 (基于system_config.py)
# ================================================================================

# 通信参数
NOISE_POWER_DBM = -174.0  # dBm/Hz 热噪声密度
NOISE_FIGURE = 9.0  # dB 噪声系数
TX_POWER = 0.15  # W (150 mW) 车辆发射功率
DISTANCE = 150.0  # m 平均通信距离
PATH_LOSS_EXPONENT = 2.7  # 路径损耗指数
FREQUENCY = 3.5e9  # Hz (3.5 GHz) 载波频率

# 任务参数 (基于实际训练场景)
TASK_ARRIVAL_RATE = 3.5  # tasks/s 任务到达率
NUM_VEHICLES = 12  # 车辆数量
AVG_DATA_SIZE_MB = 7.5  # MB 平均任务数据大小
EPISODE_DURATION = 20.0  # s (200 steps * 0.1s) 每轮仿真时长

# 计算参数
RSU_CPU_FREQ = 17.5e9  # Hz RSU CPU频率
TASK_COMPUTE_DENSITY = 2.5  # cycles/bit 计算密度

# ================================================================================
# 理论模型计算函数
# ================================================================================

def dbm_to_watt(dbm):
    """将dBm转换为瓦特"""
    return 10 ** ((dbm - 30) / 10)

def calculate_path_loss(distance, frequency):
    """
    计算自由空间路径损耗
    PL = 20*log10(d) + 20*log10(f) + 20*log10(4π/c) - 147.55
    """
    c = 3e8  # 光速
    wavelength = c / frequency
    pl_db = 20 * np.log10(distance) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
    return pl_db

def calculate_snr(bandwidth_hz):
    """
    计算信噪比 (SNR)
    
    参数:
        bandwidth_hz: 带宽 (Hz)
    
    返回:
        snr: 线性信噪比
    """
    # 接收功率 (考虑路径损耗)
    path_loss_db = calculate_path_loss(DISTANCE, FREQUENCY)
    rx_power_dbm = 10 * np.log10(TX_POWER * 1000) - path_loss_db
    rx_power_w = dbm_to_watt(rx_power_dbm)
    
    # 噪声功率
    noise_density_w_hz = dbm_to_watt(NOISE_POWER_DBM)
    noise_power_w = noise_density_w_hz * bandwidth_hz * (10 ** (NOISE_FIGURE / 10))
    
    # SNR
    snr = rx_power_w / noise_power_w
    return snr

def calculate_shannon_capacity(bandwidth_hz):
    """
    计算Shannon信道容量
    
    C = B * log2(1 + SNR)  (bps)
    """
    snr = calculate_snr(bandwidth_hz)
    capacity_bps = bandwidth_hz * np.log2(1 + snr)
    return capacity_bps

def calculate_transmission_delay(data_size_bytes, bandwidth_hz):
    """
    计算传输时延
    
    T_trans = Data_size / Capacity  (s)
    """
    capacity_bps = calculate_shannon_capacity(bandwidth_hz)
    data_size_bits = data_size_bytes * 8
    delay_s = data_size_bits / capacity_bps
    return delay_s

def calculate_transmission_energy(data_size_bytes, bandwidth_hz):
    """
    计算传输能耗
    
    E_trans = P_tx * T_trans  (J)
    """
    delay_s = calculate_transmission_delay(data_size_bytes, bandwidth_hz)
    energy_j = TX_POWER * delay_s
    return energy_j

def calculate_processing_delay(data_size_bytes):
    """
    计算处理时延 (RSU计算)
    
    T_proc = Compute_cycles / CPU_freq  (s)
    """
    data_size_bits = data_size_bytes * 8
    compute_cycles = data_size_bits * TASK_COMPUTE_DENSITY
    delay_s = compute_cycles / RSU_CPU_FREQ
    return delay_s

def calculate_processing_energy(data_size_bytes):
    """
    计算处理能耗 (RSU)
    
    E_proc = kappa * f^3 * t + P_static * t  (J)
    """
    # 基于system_config.py的参数
    kappa = 5.0e-32  # W/(Hz)^3
    static_power = 25.0  # W
    
    proc_time = calculate_processing_delay(data_size_bytes)
    dynamic_power = kappa * (RSU_CPU_FREQ ** 3)
    total_power = dynamic_power + static_power
    energy_j = total_power * proc_time
    return energy_j

def calculate_total_episode_metrics(bandwidth_hz):
    """
    计算一个episode的总体性能指标
    
    返回:
        dict: 包含时延、能耗、完成率、数据丢失率等指标
    """
    # 计算episode内总任务数
    total_tasks = NUM_VEHICLES * TASK_ARRIVAL_RATE * EPISODE_DURATION
    
    # 平均数据大小
    avg_data_bytes = AVG_DATA_SIZE_MB * 1024 * 1024
    
    # 单任务指标
    trans_delay = calculate_transmission_delay(avg_data_bytes, bandwidth_hz)
    trans_energy = calculate_transmission_energy(avg_data_bytes, bandwidth_hz)
    proc_delay = calculate_processing_delay(avg_data_bytes)
    proc_energy = calculate_processing_energy(avg_data_bytes)
    
    # 总时延和能耗 (上行+下行)
    total_delay_per_task = 2 * trans_delay + proc_delay  # 上行+处理+下行
    total_energy_per_task = 2 * trans_energy + proc_energy
    
    # Episode总指标
    avg_delay = total_delay_per_task
    total_energy = total_tasks * total_energy_per_task
    
    # 估算完成率 (假设足够的计算资源)
    # 简化模型:如果平均时延 < 最大容忍时延6.5s,则完成
    max_tolerable_delay = 6.5  # s
    completion_rate = min(1.0, max_tolerable_delay / avg_delay) if avg_delay > 0 else 1.0
    
    # 估算数据丢失率 (基于队列溢出概率)
    # 简化模型:假设与任务到达速率和处理速率的比值相关
    service_rate = 1.0 / total_delay_per_task if total_delay_per_task > 0 else float('inf')
    utilization = (TASK_ARRIVAL_RATE * NUM_VEHICLES) / (service_rate * 4)  # 4个RSU
    data_loss_rate = max(0.0, min(1.0, (utilization - 0.7) / 0.5)) if utilization > 0.7 else 0.0
    
    return {
        'bandwidth_mhz': bandwidth_hz / 1e6,
        'avg_delay': avg_delay,
        'total_energy': total_energy,
        'completion_rate': completion_rate,
        'data_loss_rate': data_loss_rate,
        'single_task_trans_delay': trans_delay,
        'single_task_trans_energy': trans_energy,
        'single_task_proc_delay': proc_delay,
        'shannon_capacity_mbps': calculate_shannon_capacity(bandwidth_hz) / 1e6,
        'snr_db': 10 * np.log10(calculate_snr(bandwidth_hz)),
    }

# ================================================================================
# 主程序
# ================================================================================

def main():
    print("=" * 80)
    print("📊 带宽理论成本分析")
    print("=" * 80)
    print()
    
    # 带宽范围 (MHz) - 30-70MHz
    bandwidths_mhz = [30, 40, 50, 60, 70]
    
    results = []
    
    print(f"{'带宽(MHz)':<12} {'时延(s)':<12} {'能耗(J)':<12} {'完成率':<12} {'数据丢失率':<12} {'信道容量(Mbps)':<15}")
    print("-" * 80)
    
    for bw_mhz in bandwidths_mhz:
        bw_hz = bw_mhz * 1e6
        metrics = calculate_total_episode_metrics(bw_hz)
        results.append(metrics)
        
        print(f"{metrics['bandwidth_mhz']:<12.1f} "
              f"{metrics['avg_delay']:<12.3f} "
              f"{metrics['total_energy']:<12.1f} "
              f"{metrics['completion_rate']:<12.3f} "
              f"{metrics['data_loss_rate']:<12.4f} "
              f"{metrics['shannon_capacity_mbps']:<15.1f}")
    
    print("=" * 80)
    print()
    
    # 保存结果
    output_file = f"bandwidth_theoretical_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'system_parameters': {
                'num_vehicles': NUM_VEHICLES,
                'task_arrival_rate': TASK_ARRIVAL_RATE,
                'avg_data_size_mb': AVG_DATA_SIZE_MB,
                'tx_power_w': TX_POWER,
                'rsu_cpu_freq_ghz': RSU_CPU_FREQ / 1e9,
                'episode_duration_s': EPISODE_DURATION,
            },
            'results': results,
        }, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 结果已保存到: {output_file}")
    print()
    
    # 生成可视化图表
    generate_plots(results)
    
    # 理论分析总结
    print_theoretical_analysis(results)

def generate_plots(results):
    """生成理论分析图表"""
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    bandwidths = [r['bandwidth_mhz'] for r in results]
    delays = [r['avg_delay'] for r in results]
    energies = [r['total_energy'] for r in results]
    capacities = [r['shannon_capacity_mbps'] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('带宽理论成本分析 (30-70 MHz)', fontsize=16, fontweight='bold', y=0.995)
    
    # 1. 时延 vs 带宽
    axes[0, 0].plot(bandwidths, delays, 'o-', linewidth=2.5, markersize=10, color='#e74c3c', label='理论时延')
    axes[0, 0].set_xlabel('带宽 (MHz)', fontsize=13, fontweight='bold')
    axes[0, 0].set_ylabel('平均时延 (s)', fontsize=13, fontweight='bold')
    axes[0, 0].set_title('平均时延 vs 带宽', fontsize=14, fontweight='bold', pad=10)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    axes[0, 0].legend(fontsize=11)
    # 添加数值标注
    for i, (x, y) in enumerate(zip(bandwidths, delays)):
        axes[0, 0].annotate(f'{y:.4f}s', (x, y), textcoords="offset points", 
                           xytext=(0,8), ha='center', fontsize=9)
    
    # 2. 能耗 vs 带宽
    axes[0, 1].plot(bandwidths, energies, 's-', linewidth=2.5, markersize=10, color='#3498db', label='理论能耗')
    axes[0, 1].set_xlabel('带宽 (MHz)', fontsize=13, fontweight='bold')
    axes[0, 1].set_ylabel('总能耗 (J)', fontsize=13, fontweight='bold')
    axes[0, 1].set_title('Episode总能耗 vs 带宽', fontsize=14, fontweight='bold', pad=10)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    axes[0, 1].legend(fontsize=11)
    # 添加数值标注
    for i, (x, y) in enumerate(zip(bandwidths, energies)):
        axes[0, 1].annotate(f'{y:.0f}J', (x, y), textcoords="offset points", 
                           xytext=(0,8), ha='center', fontsize=9)
    
    # 3. Shannon容量 vs 带宽
    axes[1, 0].plot(bandwidths, capacities, '^-', linewidth=2.5, markersize=10, color='#2ecc71', label='Shannon容量')
    axes[1, 0].set_xlabel('带宽 (MHz)', fontsize=13, fontweight='bold')
    axes[1, 0].set_ylabel('信道容量 (Mbps)', fontsize=13, fontweight='bold')
    axes[1, 0].set_title('Shannon信道容量 vs 带宽', fontsize=14, fontweight='bold', pad=10)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    axes[1, 0].legend(fontsize=11)
    # 添加数值标注
    for i, (x, y) in enumerate(zip(bandwidths, capacities)):
        axes[1, 0].annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                           xytext=(0,8), ha='center', fontsize=9)
    
    # 4. 归一化成本对比
    norm_delays = np.array(delays) / max(delays)
    norm_energies = np.array(energies) / max(energies)
    costs = 0.5 * norm_delays + 0.5 * norm_energies
    
    axes[1, 1].plot(bandwidths, costs, 'd-', linewidth=2.5, markersize=10, color='#9b59b6', label='综合成本')
    axes[1, 1].set_xlabel('带宽 (MHz)', fontsize=13, fontweight='bold')
    axes[1, 1].set_ylabel('归一化成本', fontsize=13, fontweight='bold')
    axes[1, 1].set_title('归一化综合成本 vs 带宽', fontsize=14, fontweight='bold', pad=10)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    axes[1, 1].legend(fontsize=11)
    # 添加数值标注
    for i, (x, y) in enumerate(zip(bandwidths, costs)):
        axes[1, 1].annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                           xytext=(0,8), ha='center', fontsize=9)
    
    plt.tight_layout()
    output_plot = f"bandwidth_theoretical_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_plot, dpi=300, bbox_inches='tight')
    print(f"✅ 图表已保存到: {output_plot}")
    plt.close()

def print_theoretical_analysis(results):
    """打印理论分析总结"""
    print()
    print("=" * 80)
    print("📝 理论分析总结")
    print("=" * 80)
    print()
    
    print("【关键发现】")
    print()
    
    # 1. 带宽影响分析
    bw_30 = results[0]
    bw_100 = results[-1]
    
    delay_improvement = (bw_30['avg_delay'] - bw_100['avg_delay']) / bw_30['avg_delay'] * 100
    energy_change = (bw_100['total_energy'] - bw_30['total_energy']) / bw_30['total_energy'] * 100
    
    print(f"1. 带宽从30MHz增加到100MHz:")
    print(f"   • 时延改善: {delay_improvement:.1f}%")
    print(f"   • 能耗变化: {energy_change:+.1f}%")
    print()
    
    # 2. Shannon容量分析
    print("2. Shannon信道容量:")
    for r in results[::2]:  # 每隔一个显示
        print(f"   • {r['bandwidth_mhz']:.0f} MHz → {r['shannon_capacity_mbps']:.1f} Mbps "
              f"(SNR: {r['snr_db']:.1f} dB)")
    print()
    
    # 3. 最优带宽推荐
    costs = []
    for r in results:
        norm_delay = r['avg_delay'] / results[0]['avg_delay']
        norm_energy = r['total_energy'] / results[0]['total_energy']
        cost = 0.5 * norm_delay + 0.5 * norm_energy
        costs.append(cost)
    
    optimal_idx = np.argmin(costs)
    optimal_bw = results[optimal_idx]['bandwidth_mhz']
    
    print(f"3. 最优带宽配置 (理论):")
    print(f"   • 推荐带宽: {optimal_bw:.0f} MHz")
    print(f"   • 预期时延: {results[optimal_idx]['avg_delay']:.3f} s")
    print(f"   • 预期能耗: {results[optimal_idx]['total_energy']:.1f} J")
    print(f"   • 信道容量: {results[optimal_idx]['shannon_capacity_mbps']:.1f} Mbps")
    print()
    
    print("【理论模型假设】")
    print("• 通信模型: Shannon信道容量公式")
    print("• 路径损耗: 自由空间传播模型")
    print(f"• 平均通信距离: {DISTANCE} m")
    print(f"• 发射功率: {TX_POWER * 1000} mW")
    print(f"• 任务负载: {NUM_VEHICLES} 车辆 × {TASK_ARRIVAL_RATE} tasks/s")
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
