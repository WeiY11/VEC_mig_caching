#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数扫描实验 - 生成参数对比折线图
在不同参数配置下运行算法，生成性能对比折线图

【典型用途】
1. 车辆数扫描：8, 12, 16, 20, 24辆车
2. 负载强度扫描：不同任务到达率
3. 网络条件扫描：不同带宽配置

【输出】
- 参数对比折线图（X轴=参数值，Y轴=性能，多条线=不同算法）
- 包含数据点标记，清晰展示每个配置的性能
"""

import os
import sys
import json
import argparse
import numpy as np
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# 修复Windows编码
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from run_baseline_comparison import BaselineComparisonExperiment


class ParameterSweepExperiment:
    """参数扫描实验类"""
    
    def __init__(self, save_dir: str = None):
        self.save_dir = Path(save_dir) if save_dir else Path(__file__).parent / "parameter_sweep_results"
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        self.analysis_dir = self.save_dir / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # 存储所有参数配置的结果
        # {parameter_value: {algorithm_name: result_dict}}
        self.sweep_results = {}
    
    def run_vehicle_sweep(self, vehicle_counts: list, algorithms: list, 
                         episodes_per_config: int = 200, seed: int = 42):
        """
        车辆数扫描实验
        
        Args:
            vehicle_counts: 车辆数列表，如[8, 12, 16, 20, 24]
            algorithms: 算法列表，如['TD3', 'DDPG', 'Greedy']
            episodes_per_config: 每个配置的训练轮次
            seed: 随机种子
        """
        print("="*80)
        print("车辆数扫描实验")
        print("="*80)
        print(f"  车辆数配置: {vehicle_counts}")
        print(f"  对比算法: {algorithms}")
        print(f"  每配置轮次: {episodes_per_config}")
        print(f"  固定拓扑: 4 RSU + 2 UAV")
        print(f"  预计时间: ~{len(vehicle_counts) * len(algorithms) * episodes_per_config * 1.5 / 60:.0f}分钟")
        print("="*80)
        
        for num_vehicles in vehicle_counts:
            print(f"\n{'='*80}")
            print(f"运行配置: {num_vehicles}辆车")
            print(f"{'='*80}")
            
            # 创建该配置的实验环境
            experiment = BaselineComparisonExperiment(num_vehicles=num_vehicles)
            
            # 运行所有算法
            config_results = {}
            for algo_name in algorithms:
                print(f"\n运行算法: {algo_name} ({num_vehicles}辆车)")
                
                # 判断算法类型
                drl_algos = ['TD3', 'DDPG', 'SAC', 'PPO', 'DQN']
                if algo_name in drl_algos:
                    result = experiment.run_drl_algorithm(algo_name, episodes_per_config, seed)
                else:
                    result = experiment.run_baseline_algorithm(algo_name, episodes_per_config, seed)
                
                config_results[algo_name] = result
                
                # 保存单个结果
                result_file = self.save_dir / f"{algo_name}_{num_vehicles}v.json"
                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2, ensure_ascii=False)
            
            self.sweep_results[num_vehicles] = config_results
        
        print("\n" + "="*80)
        print("参数扫描完成！")
        print("="*80)
        
        # 生成对比折线图
        self.generate_parameter_comparison_plots()
        
        # 保存汇总结果
        self.save_sweep_summary()
    
    def generate_parameter_comparison_plots(self):
        """生成参数对比折线图"""
        if not self.sweep_results:
            print("警告: 没有扫描结果")
            return
        
        print("\n生成参数对比折线图...")
        
        # 提取数据
        param_values = sorted(self.sweep_results.keys())  # 车辆数
        algorithms = list(next(iter(self.sweep_results.values())).keys())  # 算法列表
        
        # 为每个算法准备数据
        algo_data = {}
        for algo_name in algorithms:
            delays = []
            energies = []
            completions = []
            objectives = []
            
            for param_val in param_values:
                if param_val in self.sweep_results and algo_name in self.sweep_results[param_val]:
                    result = self.sweep_results[param_val][algo_name]
                    delays.append(result['avg_delay'])
                    energies.append(result['avg_energy'])
                    completions.append(result['avg_completion_rate'] * 100)
                    
                    # 计算目标函数值
                    obj_value = 2.0 * result['avg_delay'] + 1.2 * (result['avg_energy'] / 600.0)
                    objectives.append(obj_value)
                else:
                    delays.append(np.nan)
                    energies.append(np.nan)
                    completions.append(np.nan)
                    objectives.append(np.nan)
            
            algo_data[algo_name] = {
                'delays': delays,
                'energies': energies,
                'completions': completions,
                'objectives': objectives
            }
        
        # 创建4个子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 11))
        fig.suptitle('Performance vs Number of Vehicles (Fixed Topology: 4 RSU + 2 UAV)', 
                    fontsize=15, fontweight='bold')
        
        # 定义样式
        markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6C757D', 
                 '#17BEBB', '#9B59B6', '#E67E22', '#3498DB', '#E74C3C']
        
        # 绘制每个算法
        for idx, algo_name in enumerate(algorithms):
            marker = markers[idx % len(markers)]
            color = colors[idx % len(colors)]
            data = algo_data[algo_name]
            
            # 子图1: 时延 vs 车辆数
            axes[0, 0].plot(param_values, data['delays'], 
                          marker=marker, markersize=8, label=algo_name,
                          color=color, linewidth=2.5, alpha=0.85,
                          markerfacecolor='white', markeredgewidth=2.5, 
                          markeredgecolor=color)
            
            # 子图2: 能耗 vs 车辆数
            axes[0, 1].plot(param_values, data['energies'],
                          marker=marker, markersize=8, label=algo_name,
                          color=color, linewidth=2.5, alpha=0.85,
                          markerfacecolor='white', markeredgewidth=2.5,
                          markeredgecolor=color)
            
            # 子图3: 完成率 vs 车辆数
            axes[1, 0].plot(param_values, data['completions'],
                          marker=marker, markersize=8, label=algo_name,
                          color=color, linewidth=2.5, alpha=0.85,
                          markerfacecolor='white', markeredgewidth=2.5,
                          markeredgecolor=color)
            
            # 子图4: 目标函数 vs 车辆数
            axes[1, 1].plot(param_values, data['objectives'],
                          marker=marker, markersize=8, label=algo_name,
                          color=color, linewidth=2.5, alpha=0.85,
                          markerfacecolor='white', markeredgewidth=2.5,
                          markeredgecolor=color)
        
        # 配置子图1: 时延
        axes[0, 0].set_title('Average Task Delay vs Vehicle Count', 
                            fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Number of Vehicles', fontsize=11)
        axes[0, 0].set_ylabel('Delay (seconds)', fontsize=11)
        axes[0, 0].legend(fontsize=9, framealpha=0.95, loc='best')
        axes[0, 0].grid(alpha=0.3, linestyle='--')
        axes[0, 0].set_xticks(param_values)
        
        # 配置子图2: 能耗
        axes[0, 1].set_title('Total Energy Consumption vs Vehicle Count',
                            fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Number of Vehicles', fontsize=11)
        axes[0, 1].set_ylabel('Energy (Joules)', fontsize=11)
        axes[0, 1].legend(fontsize=9, framealpha=0.95, loc='best')
        axes[0, 1].grid(alpha=0.3, linestyle='--')
        axes[0, 1].set_xticks(param_values)
        
        # 配置子图3: 完成率
        axes[1, 0].set_title('Task Completion Rate vs Vehicle Count',
                            fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Number of Vehicles', fontsize=11)
        axes[1, 0].set_ylabel('Completion Rate (%)', fontsize=11)
        axes[1, 0].legend(fontsize=9, framealpha=0.95, loc='best')
        axes[1, 0].grid(alpha=0.3, linestyle='--')
        axes[1, 0].set_xticks(param_values)
        axes[1, 0].set_ylim([85, 100])  # 完成率通常在这个范围
        
        # 配置子图4: 目标函数
        axes[1, 1].set_title('Objective Function (J=2.0×Delay+1.2×Energy) vs Vehicle Count',
                            fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Number of Vehicles', fontsize=11)
        axes[1, 1].set_ylabel('Objective Value (Lower is Better)', fontsize=11)
        axes[1, 1].legend(fontsize=9, framealpha=0.95, loc='best')
        axes[1, 1].grid(alpha=0.3, linestyle='--')
        axes[1, 1].set_xticks(param_values)
        
        plt.tight_layout()
        plot_path = self.analysis_dir / 'parameter_comparison_lines.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  [OK] 参数对比折线图: {plot_path}")
        plt.close()
    
    def save_sweep_summary(self):
        """保存扫描结果汇总"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = self.save_dir / f"sweep_summary_{timestamp}.json"
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.sweep_results, f, indent=2, ensure_ascii=False)
        
        print(f"  [OK] 扫描结果汇总: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description='参数扫描实验 - 生成参数对比折线图',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 车辆数扫描（TD3算法）
  python run_parameter_sweep.py --param vehicles --values 8 12 16 20 24 --algorithms TD3 --episodes 200
  
  # 多算法对比
  python run_parameter_sweep.py --param vehicles --values 8 12 16 --algorithms TD3 DDPG Greedy --episodes 200
  
  # 快速测试
  python run_parameter_sweep.py --param vehicles --values 8 12 16 --algorithms TD3 --episodes 50
        """
    )
    
    parser.add_argument('--param', type=str, default='vehicles', 
                       choices=['vehicles', 'load', 'bandwidth'],
                       help='扫描参数类型（当前只支持vehicles）')
    parser.add_argument('--values', type=int, nargs='+', required=True,
                       help='参数值列表，如: --values 8 12 16 20 24')
    parser.add_argument('--algorithms', type=str, nargs='+', required=True,
                       help='对比算法列表，如: --algorithms TD3 DDPG Greedy')
    parser.add_argument('--episodes', type=int, default=200,
                       help='每个配置的训练轮次（默认:200）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子（默认:42）')
    parser.add_argument('--quick', action='store_true',
                       help='快速模式（50轮）')
    
    args = parser.parse_args()
    
    # 确定轮次
    episodes = 50 if args.quick else args.episodes
    
    print("\n" + "="*80)
    print("参数扫描实验配置")
    print("="*80)
    print(f"  扫描参数: {args.param}")
    print(f"  参数值: {args.values}")
    print(f"  对比算法: {args.algorithms}")
    print(f"  每配置轮次: {episodes}")
    print(f"  固定拓扑: 4 RSU + 2 UAV")
    print(f"  总实验数: {len(args.values)} × {len(args.algorithms)} = {len(args.values) * len(args.algorithms)}")
    print(f"  预计时间: ~{len(args.values) * len(args.algorithms) * episodes * 1.5 / 60:.0f}分钟")
    print("="*80)
    
    # 创建扫描实验
    sweep_exp = ParameterSweepExperiment()
    
    if args.param == 'vehicles':
        sweep_exp.run_vehicle_sweep(args.values, args.algorithms, episodes, args.seed)
    else:
        print(f"[WARN] 参数类型 '{args.param}' 暂未实现")
    
    print("\n" + "="*80)
    print("参数扫描实验完成！")
    print("="*80)
    print(f"  结果目录: {sweep_exp.save_dir}")
    print(f"  图表目录: {sweep_exp.analysis_dir}")
    print(f"  查看图表: {sweep_exp.analysis_dir}/parameter_comparison_lines.png")
    print("="*80)


if __name__ == "__main__":
    main()


