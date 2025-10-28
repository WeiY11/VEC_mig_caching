"""
Kaggle专用训练脚本
针对Kaggle环境优化的快速训练脚本

使用方法:
    python kaggle_train.py --quick       # 快速测试（50轮）
    python kaggle_train.py --standard    # 标准训练（200轮）
    python kaggle_train.py --full        # 完整训练（500轮）
    python kaggle_train.py --algorithm SAC --episodes 100  # 自定义
"""

import os
import sys
import argparse
import torch
import time
from datetime import datetime

def setup_kaggle_environment():
    """配置Kaggle环境"""
    print("=" * 60)
    print("🚀 VEC边缘计算迁移与缓存系统 - Kaggle训练")
    print("=" * 60)
    
    # 检查GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        print(f"✅ CUDA版本: {torch.version.cuda}")
    else:
        print("⚠️  未检测到GPU，将使用CPU训练（速度较慢）")
    
    print(f"✅ PyTorch版本: {torch.__version__}")
    print("=" * 60)
    print()

def main():
    parser = argparse.ArgumentParser(description='Kaggle专用训练脚本')
    
    # 预设模式
    parser.add_argument('--quick', action='store_true', 
                        help='快速测试模式（50轮，约10分钟）')
    parser.add_argument('--standard', action='store_true',
                        help='标准训练模式（200轮，约40分钟）')
    parser.add_argument('--full', action='store_true',
                        help='完整训练模式（500轮，约2小时）')
    
    # 自定义参数
    parser.add_argument('--algorithm', type=str, default='TD3',
                        choices=['TD3', 'DDPG', 'SAC', 'PPO', 'DQN'],
                        help='选择训练算法')
    parser.add_argument('--episodes', type=int, default=None,
                        help='训练轮次（覆盖预设模式）')
    parser.add_argument('--num-vehicles', type=int, default=12,
                        help='车辆数量')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--save-interval', type=int, default=50,
                        help='模型保存间隔')
    
    args = parser.parse_args()
    
    # 设置环境
    setup_kaggle_environment()
    
    # 确定训练轮次
    if args.episodes is None:
        if args.quick:
            episodes = 50
            mode_name = "快速测试"
        elif args.standard:
            episodes = 200
            mode_name = "标准训练"
        elif args.full:
            episodes = 500
            mode_name = "完整训练"
        else:
            episodes = 100  # 默认
            mode_name = "默认"
    else:
        episodes = args.episodes
        mode_name = "自定义"
    
    # 构建训练命令
    cmd = f"python train_single_agent.py --algorithm {args.algorithm} --episodes {episodes} --seed {args.seed} --num-vehicles {args.num_vehicles}"
    
    # 显示配置
    print(f"📋 训练配置")
    print(f"   模式: {mode_name}")
    print(f"   算法: {args.algorithm}")
    print(f"   轮次: {episodes}")
    print(f"   车辆数: {args.num_vehicles}")
    print(f"   随机种子: {args.seed}")
    print(f"   预计时间: {episodes * 0.2:.0f}分钟 (估算)")
    print()
    
    # 开始训练
    print(f"🎯 开始训练...")
    print(f"执行命令: {cmd}")
    print("=" * 60)
    print()
    
    start_time = time.time()
    ret = os.system(cmd)
    elapsed = time.time() - start_time
    
    print()
    print("=" * 60)
    if ret == 0:
        print(f"✅ 训练完成！")
        print(f"⏱️  用时: {elapsed/60:.1f}分钟")
        print(f"📁 结果保存在: results/single_agent/{args.algorithm.lower()}/")
    else:
        print(f"❌ 训练失败（返回码: {ret}）")
        sys.exit(1)
    print("=" * 60)

if __name__ == "__main__":
    main()

