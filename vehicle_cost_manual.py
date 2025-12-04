#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
车辆数量成本分析图 - 手动数据版 (强制更新)

数据: [0.15, 0.25, 0.39, 0.57, 0.66]
"""

import matplotlib.pyplot as plt
from datetime import datetime

def main():
    # 1. 数据准备 (强制更新)
    vehicle_counts = [6, 8, 10, 12, 14]
    costs = [0.15, 0.25, 0.39, 0.57, 0.66]  # 确保这里是 0.39
    
    # ==========================================
    # 2. 绘图 - 极简精致风格 (保持一致)
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    line_color = '#003366'  # 深邃蓝
    fill_color = '#E6F0FF'  # 淡蓝填充
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
    
    # 范围
    y_min = 0.0
    y_max = 0.8
    ax.set_ylim(y_min, y_max)
    
    # 数值标签
    for x, y in zip(vehicle_counts, costs):
        ax.annotate(f'{y:.2f}', xy=(x, y), xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom', fontsize=10, fontweight='bold', color=line_color)

    plt.tight_layout()
    
    output_file = f"vehicle_cost_manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ 强制更新数据图表已保存: {output_file}")
    plt.close()

if __name__ == "__main__":
    main()
