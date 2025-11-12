#!/usr/bin/env python3
"""
从训练结果JSON文件生成奖励曲线图
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import argparse
import os
from pathlib import Path
from datetime import datetime

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_training_results(json_path: str) -> dict:
    """加载训练结果JSON文件"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 找不到文件 {json_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"[ERROR] 文件 {json_path} 不是有效的JSON格式: {e}")
        return None

def generate_reward_curve(json_path: str, output_path: str = None, 
                         window_size: int = 20, dpi: int = 300,
                         show_raw: bool = True, show_smooth: bool = True):
    """
    生成奖励曲线图
    
    Args:
        json_path: JSON文件路径
        output_path: 输出图片路径（如果为None，自动生成）
        window_size: 移动平均窗口大小
        dpi: 图片分辨率
        show_raw: 是否显示原始数据
        show_smooth: 是否显示平滑曲线
    """
    print(f"[INFO] 读取训练结果: {json_path}")
    results = load_training_results(json_path)
    
    if results is None:
        return False
    
    # 提取奖励数据
    episode_rewards = results.get('episode_rewards', [])
    if not episode_rewards:
        print("[ERROR] JSON文件中没有找到 episode_rewards 数据")
        return False
    
    episode_rewards = np.array(episode_rewards)
    episodes = np.arange(1, len(episode_rewards) + 1)
    
    # 获取算法信息
    algorithm = results.get('algorithm', 'Unknown')
    num_episodes = len(episode_rewards)
    
    print(f"   算法: {algorithm}")
    print(f"   训练轮次: {num_episodes}")
    print(f"   奖励范围: [{episode_rewards.min():.2f}, {episode_rewards.max():.2f}]")
    print(f"   平均奖励: {episode_rewards.mean():.2f}")
    
    # 计算移动平均
    if len(episode_rewards) >= window_size:
        moving_avg = np.convolve(episode_rewards, np.ones(window_size)/window_size, mode='valid')
        episodes_ma = episodes[window_size-1:]
        
        # 计算标准差（用于置信区间）
        moving_std = []
        for i in range(len(moving_avg)):
            window_data = episode_rewards[i:i+window_size]
            moving_std.append(np.std(window_data))
        moving_std = np.array(moving_std)
    else:
        moving_avg = episode_rewards
        episodes_ma = episodes
        moving_std = np.zeros_like(episode_rewards)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    
    # 绘制原始数据（半透明）
    if show_raw:
        ax.plot(episodes, episode_rewards, alpha=0.3, color='blue', 
               linewidth=0.5, label='原始奖励')
    
    # 绘制平滑曲线
    if show_smooth and len(episode_rewards) >= window_size:
        ax.plot(episodes_ma, moving_avg, color='red', linewidth=2.5, 
               label=f'{window_size}轮移动平均')
        
        # 绘制置信区间
        ax.fill_between(episodes_ma, 
                        moving_avg - moving_std, 
                        moving_avg + moving_std,
                        alpha=0.2, color='red', 
                        label='±1标准差')
    
    # 设置标签和标题
    ax.set_xlabel('训练轮次 (Episode)', fontsize=12)
    ax.set_ylabel('奖励值 (Reward)', fontsize=12)
    ax.set_title(f'{algorithm} 训练奖励曲线\n(共{num_episodes}轮, 最终平均: {episode_rewards[-100:].mean():.2f})', 
                 fontsize=14, fontweight='bold')
    
    # 添加网格和图例
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='best', frameon=True, fancybox=True, shadow=True)
    
    # 添加统计信息文本框
    stats_text = f'统计信息:\n'
    stats_text += f'初始(前10轮): {episode_rewards[:10].mean():.2f}\n'
    stats_text += f'最终(后10轮): {episode_rewards[-10:].mean():.2f}\n'
    stats_text += f'提升: {episode_rewards[-10:].mean() - episode_rewards[:10].mean():.2f}\n'
    stats_text += f'最佳: {episode_rewards.max():.2f}'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # 确定输出路径
    if output_path is None:
        json_dir = os.path.dirname(json_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(json_dir, f'reward_curve_{timestamp}.png')
    
    # 保存图片
    print(f"[INFO] 保存图片: {output_path}")
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"[SUCCESS] 奖励曲线已生成!")
    return True

def main():
    parser = argparse.ArgumentParser(
        description='从训练结果JSON文件生成奖励曲线图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 基本用法（自动生成输出路径）
  python generate_reward_curve.py results/single_agent/td3/training_results_20251109_184958.json
  
  # 指定输出路径
  python generate_reward_curve.py input.json -o reward_curve.png
  
  # 自定义移动平均窗口
  python generate_reward_curve.py input.json --window 50
  
  # 只显示平滑曲线（不显示原始数据）
  python generate_reward_curve.py input.json --no-raw
  
  # 高分辨率输出（适合论文）
  python generate_reward_curve.py input.json --dpi 600
        """
    )
    
    parser.add_argument('json_file', help='训练结果JSON文件路径')
    parser.add_argument('-o', '--output', help='输出图片路径（默认：同目录下自动命名）')
    parser.add_argument('--window', type=int, default=20, 
                       help='移动平均窗口大小（默认：20）')
    parser.add_argument('--dpi', type=int, default=300,
                       help='图片分辨率DPI（默认：300）')
    parser.add_argument('--no-raw', action='store_true',
                       help='不显示原始数据点（只显示平滑曲线）')
    parser.add_argument('--no-smooth', action='store_true',
                       help='不显示平滑曲线（只显示原始数据）')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.json_file):
        print(f"[ERROR] 文件不存在: {args.json_file}")
        return
    
    # 生成奖励曲线
    success = generate_reward_curve(
        json_path=args.json_file,
        output_path=args.output,
        window_size=args.window,
        dpi=args.dpi,
        show_raw=not args.no_raw,
        show_smooth=not args.no_smooth
    )
    
    if not success:
        print("[ERROR] 生成失败")
        return

if __name__ == "__main__":
    main()

