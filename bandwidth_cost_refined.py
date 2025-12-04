#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学术风格带宽成本分析图 - 极简精致版 (数值两位小数)

特点:
- 顶级期刊配色 (深邃蓝)
- 精致的Marker (白底深边)
- 极简网格 (仅水平虚线)
- 阴影填充 (增加层次感)
- 数值保留两位小数
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.ticker as ticker

# ==========================================
# 1. 数据准备 (保持不变)
# ==========================================
# 系统参数
NOISE_POWER_DBM = -174.0
NOISE_FIGURE = 9.0
TX_POWER = 0.15
DISTANCE = 150.0
FREQUENCY = 3.5e9
TASK_ARRIVAL_RATE = 3.5
NUM_VEHICLES = 12
AVG_DATA_SIZE_MB = 7.5
EPISODE_DURATION = 20.0
RSU_CPU_FREQ = 17.5e9
TASK_COMPUTE_DENSITY = 40.0 # 保持 40.0 cycles/bit

def dbm_to_watt(dbm): return 10 ** ((dbm - 30) / 10)

def calculate_metrics(bandwidth_hz):
    # 简化的计算逻辑，直接复用之前的公式
    c = 3e8
    pl_db = 20 * np.log10(DISTANCE) + 20 * np.log10(FREQUENCY) + 20 * np.log10(4 * np.pi / c)
    rx_power_dbm = 10 * np.log10(TX_POWER * 1000) - pl_db
    rx_power_w = dbm_to_watt(rx_power_dbm)
    noise_w = dbm_to_watt(NOISE_POWER_DBM) * bandwidth_hz * (10 ** (NOISE_FIGURE / 10))
    snr = rx_power_w / noise_w
    capacity = bandwidth_hz * np.log2(1 + snr)
    
    data_bits = AVG_DATA_SIZE_MB * 1024 * 1024 * 8
    trans_delay = data_bits / capacity
    trans_energy = TX_POWER * trans_delay
    
    proc_delay = (data_bits * TASK_COMPUTE_DENSITY) / RSU_CPU_FREQ
    # kappa * f^3 * t + static * t
    proc_energy = (5.0e-32 * (RSU_CPU_FREQ**3) + 25.0) * proc_delay
    
    total_delay = 2 * trans_delay + proc_delay
    total_energy = (NUM_VEHICLES * TASK_ARRIVAL_RATE * EPISODE_DURATION) * (2 * trans_energy + proc_energy)
    
    return total_delay, total_energy

def main():
    # 准备数据
    bandwidths = [30, 40, 50, 60, 70]
    delays = []
    energies = []
    
    for bw in bandwidths:
        d, e = calculate_metrics(bw * 1e6)
        delays.append(d)
        energies.append(e)
        
    # 归一化 (基准值设为30MHz时的1.5倍，保持之前的逻辑)
    d_ref = max(delays) * 1.5
    e_ref = max(energies) * 1.5
    
    costs = 0.5 * (np.array(delays) / d_ref) + 0.5 * (np.array(energies) / e_ref)

    # ==========================================
    # 2. 绘图 - 极简精致风格
    # ==========================================
    
    # 设置全局字体
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    # 创建画布
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 核心配色
    line_color = '#003366'  # 深邃蓝
    fill_color = '#E6F0FF'  # 极淡的蓝色背景
    marker_fill = 'white'   # Marker填充色
    
    # 1. 绘制阴影区域 (增加层次感)
    ax.fill_between(bandwidths, costs, 0, color=fill_color, alpha=0.5)
    
    # 2. 绘制主曲线
    # zorder=10 确保线在网格之上
    ax.plot(bandwidths, costs, 
            color=line_color, 
            linewidth=2.5, 
            linestyle='-',
            zorder=10)
            
    # 3. 绘制标记点 (精心设计: 白底+深边)
    ax.plot(bandwidths, costs, 
            marker='o', 
            markersize=9, 
            linestyle='None', # 不画线，只画点(线在上面已经画了)
            markerfacecolor=marker_fill, 
            markeredgecolor=line_color, 
            markeredgewidth=2,
            zorder=11)
    
    # 4. 坐标轴美化
    # 移除上方和右侧边框
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # 加粗左侧和下方坐标轴
    ax.spines['left'].set_linewidth(1.2)
    ax.spines['bottom'].set_linewidth(1.2)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # 5. 网格线 (仅保留水平虚线)
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#999999', zorder=0)
    ax.grid(axis='x', visible=False) # 隐藏垂直网格
    
    # 6. 标签与标题
    ax.set_xlabel('Bandwidth (MHz)', fontsize=12, fontweight='bold', labelpad=10, color='#333333')
    ax.set_ylabel('Average Cost (Normalized)', fontsize=12, fontweight='bold', labelpad=10, color='#333333')
    
    # 7. 刻度设置
    ax.set_xticks(bandwidths)
    ax.tick_params(axis='both', which='major', labelsize=11, width=1.2, length=5, color='#333333')
    
    # 设置Y轴范围 (留出头部空间)
    y_min = min(costs) * 0.9
    y_max = max(costs) * 1.1
    ax.set_ylim(y_min, y_max)
    
    # 8. 添加数值标签 (修改为两位小数)
    for x, y in zip(bandwidths, costs):
        ax.annotate(f'{y:.2f}',  # <--- 这里修改为两位小数
                    xy=(x, y), 
                    xytext=(0, 10), 
                    textcoords='offset points',
                    ha='center', 
                    va='bottom',
                    fontsize=10,
                    fontweight='bold',
                    color=line_color)

    plt.tight_layout()
    
    # 保存
    output_file = f"bandwidth_cost_refined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 精致版图表(两位小数)已保存: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
