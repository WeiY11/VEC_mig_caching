#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新生成平滑后的训练曲线图
使用已有的实验数据
"""

import sys
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


def smooth_curve(data, window_size=20):
    """滑动平均平滑"""
    if len(data) < window_size:
        return data
    
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed.append(np.mean(data[start:end]))
    
    return smoothed


def regenerate_smooth_curves(summary_file):
    """重新生成平滑曲线"""
    
    # 加载数据
    print(f"加载实验数据: {summary_file}")
    with open(summary_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    print(f"找到 {len(results)} 个配置的数据")
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    
    # 滑动窗口大小
    window_size = 20
    
    # 配置颜色映射
    colors = {
        'Full-System': '#1f77b4',      # 蓝色
        'No-Cache': '#ff7f0e',         # 橙色
        'No-Migration': '#2ca02c',     # 绿色
        'No-Priority': '#d62728',      # 红色
        'No-Adaptive': '#9467bd',      # 紫色
        'No-Collaboration': '#8c564b', # 棕色
        'Minimal-System': '#e377c2'    # 粉色
    }
    
    for config_name, result in results.items():
        color = colors.get(config_name, None)
        
        # 时延曲线（平滑）
        if 'episode_delays' in result:
            delays_smooth = smooth_curve(result['episode_delays'], window_size)
            axes[0, 0].plot(delays_smooth, label=config_name, alpha=0.85, linewidth=2, color=color)
        
        # 能耗曲线（平滑）
        if 'episode_energies' in result:
            energies_smooth = smooth_curve(result['episode_energies'], window_size)
            axes[0, 1].plot(energies_smooth, label=config_name, alpha=0.85, linewidth=2, color=color)
        
        # 完成率曲线（平滑）
        if 'episode_completion_rates' in result:
            completions_smooth = smooth_curve(result['episode_completion_rates'], window_size)
            axes[1, 0].plot(completions_smooth, label=config_name, alpha=0.85, linewidth=2, color=color)
        
        # 奖励曲线（平滑）⭐
        if 'episode_rewards' in result:
            rewards_smooth = smooth_curve(result['episode_rewards'], window_size)
            axes[1, 1].plot(rewards_smooth, label=config_name, alpha=0.85, linewidth=2, color=color)
    
    # 设置子图样式
    axes[0, 0].set_title('时延训练曲线 (滑动平均)', fontweight='bold', fontsize=13)
    axes[0, 0].set_xlabel('Episode', fontsize=11)
    axes[0, 0].set_ylabel('平均时延 (s)', fontsize=11)
    axes[0, 0].legend(fontsize=9, loc='upper right', framealpha=0.9)
    axes[0, 0].grid(alpha=0.3, linestyle='--')
    
    axes[0, 1].set_title('能耗训练曲线 (滑动平均)', fontweight='bold', fontsize=13)
    axes[0, 1].set_xlabel('Episode', fontsize=11)
    axes[0, 1].set_ylabel('总能耗 (J)', fontsize=11)
    axes[0, 1].legend(fontsize=9, loc='upper right', framealpha=0.9)
    axes[0, 1].grid(alpha=0.3, linestyle='--')
    
    axes[1, 0].set_title('完成率训练曲线 (滑动平均)', fontweight='bold', fontsize=13)
    axes[1, 0].set_xlabel('Episode', fontsize=11)
    axes[1, 0].set_ylabel('完成率', fontsize=11)
    axes[1, 0].legend(fontsize=9, loc='lower right', framealpha=0.9)
    axes[1, 0].grid(alpha=0.3, linestyle='--')
    
    axes[1, 1].set_title('奖励训练曲线 (滑动平均, 窗口=20)', fontweight='bold', fontsize=13)
    axes[1, 1].set_xlabel('Episode', fontsize=11)
    axes[1, 1].set_ylabel('平均奖励', fontsize=11)
    axes[1, 1].legend(fontsize=9, loc='lower right', framealpha=0.9)
    axes[1, 1].grid(alpha=0.3, linestyle='--')
    
    plt.tight_layout(pad=2.0)
    
    # 保存图表
    output_dir = Path(__file__).parent / "analysis"
    output_file = output_dir / 'training_curves_smooth.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ 平滑训练曲线已保存: {output_file}")
    
    plt.close()
    
    print("\n图表已优化:")
    print("  - 添加滑动平均平滑 (窗口=20)")
    print("  - 增加线条粗细至2.0")
    print("  - 优化透明度至0.85")
    print("  - 提高分辨率至300 DPI")
    print("  - 调整图例位置")


if __name__ == "__main__":
    # 查找最新的汇总文件
    results_dir = Path(__file__).parent / "results"
    summary_files = list(results_dir.glob("ablation_summary_*.json"))
    
    if not summary_files:
        print("错误: 未找到实验结果文件")
        sys.exit(1)
    
    # 使用最新的文件
    latest_summary = max(summary_files, key=lambda p: p.stat().st_mtime)
    
    print("="*60)
    print("重新生成平滑训练曲线")
    print("="*60)
    
    regenerate_smooth_curves(latest_summary)
    
    print("\n完成！请查看 analysis/training_curves_smooth.png")
    print("="*60)

