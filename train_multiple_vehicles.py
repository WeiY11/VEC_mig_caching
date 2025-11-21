#!/usr/bin/env python3
"""
批量训练不同车辆数的TD3模型

使用方法:
python train_multiple_vehicles.py --algorithm TD3 --episodes 1200 --vehicles 8,10,12,14,16
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='批量训练不同车辆数的模型')
    parser.add_argument('--algorithm', type=str, default='TD3', 
                       help='算法类型')
    parser.add_argument('--episodes', type=int, default=1200,
                       help='训练轮次')
    parser.add_argument('--vehicles', type=str, default='8,10,12,14,16',
                       help='车辆数列表，逗号分隔')
    
    args = parser.parse_args()
    
    # 解析车辆数列表
    vehicle_counts = [int(v.strip()) for v in args.vehicles.split(',')]
    
    print("=" * 80)
    print("批量训练任务启动")
    print("=" * 80)
    print(f"算法: {args.algorithm}")
    print(f"训练轮次: {args.episodes}")
    print(f"车辆数列表: {vehicle_counts}")
    print("=" * 80)
    print()
    
    # 依次训练每个车辆数配置
    for num_vehicles in vehicle_counts:
        print(f"\n{'='*80}")
        print(f"开始训练: {num_vehicles}辆车")
        print(f"{'='*80}\n")
        
        # 构建训练命令
        cmd = [
            sys.executable,
            'train_single_agent.py',
            '--algorithm', args.algorithm,
            '--episodes', str(args.episodes),
            '--num-vehicles', str(num_vehicles)
        ]
        
        # 执行训练
        try:
            start_time = datetime.now()
            print(f"执行命令: {' '.join(cmd)}")
            print(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            result = subprocess.run(cmd, check=True)
            
            end_time = datetime.now()
            duration = end_time - start_time
            
            print(f"\n✅ {num_vehicles}辆车训练完成")
            print(f"耗时: {duration}")
            print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ {num_vehicles}辆车训练失败")
            print(f"错误代码: {e.returncode}")
            print("继续下一个配置...\n")
            continue
        except KeyboardInterrupt:
            print("\n\n⚠️  用户中断训练")
            sys.exit(1)
    
    print("\n" + "=" * 80)
    print("✅ 所有训练任务完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()
