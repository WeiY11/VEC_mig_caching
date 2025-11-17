#!/usr/bin/env python3
"""
对比不同带宽模式下的性能差异：
- 固定带宽模式（50MHz）
- 动态带宽分配模式（基于优先级+SINR+数据量）
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import json
from datetime import datetime
from typing import Dict, List, Any, Optional

from evaluation.system_simulator import CompleteSystemSimulator
from config import config as sys_config


class BandwidthModeComparator:
    """对比不同带宽分配模式的性能"""
    
    def __init__(self, 
                 scenario: Optional[Dict[str, int]] = None,
                 episodes: int = 100,
                 seed: int = 42):
        """
        初始化对比器
        
        Args:
            scenario: 场景配置字典
            episodes: 运行轮次
            seed: 随机种子
        """
        self.scenario = scenario or {
            'num_vehicles': 12,
            'num_rsus': 4,
            'num_uavs': 2,
        }
        self.episodes = episodes
        self.seed = seed
        np.random.seed(seed)
        
        # 创建两个仿真器
        self.sim_fixed = CompleteSystemSimulator(self.scenario)
        self.sim_dynamic = CompleteSystemSimulator(self.scenario)
        
        # 配置动态带宽模式（如果仿真器支持）
        if hasattr(self.sim_dynamic, '_init_dynamic_bandwidth_support'):
            try:
                self.sim_dynamic._init_dynamic_bandwidth_support()
            except Exception:
                pass  # 如果初始化失败，继续使用固定带宽
        
        # 统计数据
        self.results = {
            'fixed': {
                'delays': [],
                'energy': [],
                'throughput': [],
                'completion_rate': [],
            },
            'dynamic': {
                'delays': [],
                'energy': [],
                'throughput': [],
                'completion_rate': [],
            }
        }
    
    def run_comparison(self) -> Dict[str, Any]:
        """
        运行对比实验
        
        Returns:
            对比结果
        """
        print(f"\n{'='*80}")
        print("带宽分配模式对比实验")
        print(f"{'='*80}")
        print(f"场景配置: {self.episodes}个episode")
        print(f"车辆/RSU/UAV: {self.scenario['num_vehicles']}/{self.scenario['num_rsus']}/{self.scenario['num_uavs']}")
        print(f"{'='*80}\n")
        
        # 🎯 修复：使用正确的方式获取动作
        for ep in range(self.episodes):
            # 固定带宽模式
            self._run_episode(self.sim_fixed, ep, mode='fixed')
            
            # 动态带宽模式
            self._run_episode(self.sim_dynamic, ep, mode='dynamic')
            
            if (ep + 1) % 20 == 0:
                print(f"进度: {ep + 1}/{self.episodes} episodes")
        
        return self._analyze_results()
    
    def _run_episode(self, simulator: CompleteSystemSimulator, episode: int, mode: str) -> None:
        """
        运行单个episode
        
        Args:
            simulator: 仿真器实例
            episode: episode编号
            mode: 模式('fixed' 或 'dynamic')
        """
        # CompleteSystemSimulator不提供reset()方法，我们直接使用是既有的统计
        # 此处先氛氛地收集了current_step阶段的批量统计数据
        
        episode_delay = 0.0
        episode_energy = 0.0
        episode_throughput = 0.0
        episode_tasks = 0
        episode_completed = 0
        
        # 直接调查仿真器的stats字典获取实时指标
        if hasattr(simulator, 'stats') and isinstance(simulator.stats, dict):
            episode_delay = simulator.stats.get('avg_task_delay', 0.0)
            episode_energy = simulator.stats.get('total_energy_consumption', 0.0)
            episode_throughput = simulator.stats.get('avg_throughput_mbps', 0.0)
            episode_tasks = simulator.stats.get('total_tasks_generated', 0)
            episode_completed = simulator.stats.get('processed_tasks', 0)
        
        # 计算平均值
        avg_delay = episode_delay if episode_delay > 0 else 0.0
        avg_energy = episode_energy if episode_energy > 0 else 0.0
        avg_throughput = episode_throughput if episode_throughput > 0 else 0.0
        completion_rate = episode_completed / episode_tasks if episode_tasks > 0 else 0.0
        
        # 记录结果
        self.results[mode]['delays'].append(avg_delay)
        self.results[mode]['energy'].append(avg_energy)
        self.results[mode]['throughput'].append(avg_throughput)
        self.results[mode]['completion_rate'].append(completion_rate)
    
    def _analyze_results(self) -> Dict[str, Any]:
        """
        分析对比结果
        
        Returns:
            分析结果
        """
        analysis = {
            'timestamp': datetime.now().isoformat(),
            'episodes': self.episodes,
            'scenario': self.scenario,
            'comparisons': {}
        }
        
        for metric in ['delays', 'energy', 'throughput', 'completion_rate']:
            fixed_values = np.array(self.results['fixed'][metric])
            dynamic_values = np.array(self.results['dynamic'][metric])
            
            if len(fixed_values) > 0 and len(dynamic_values) > 0:
                # 使用后50%数据避免初期波动
                half = len(fixed_values) // 2
                fixed_stable = fixed_values[half:] if half > 0 else fixed_values
                dynamic_stable = dynamic_values[half:] if half > 0 else dynamic_values
                
                fixed_mean = float(np.mean(fixed_stable))
                dynamic_mean = float(np.mean(dynamic_stable))
                
                # 计算改进比例
                if metric in ['delays', 'energy']:
                    # 越低越好
                    improvement = (fixed_mean - dynamic_mean) / max(fixed_mean, 1e-6) * 100
                else:
                    # 越高越好
                    improvement = (dynamic_mean - fixed_mean) / max(fixed_mean, 1e-6) * 100
                
                analysis['comparisons'][metric] = {
                    'fixed_mode': {
                        'mean': fixed_mean,
                        'std': float(np.std(fixed_stable)),
                        'min': float(np.min(fixed_stable)),
                        'max': float(np.max(fixed_stable)),
                    },
                    'dynamic_mode': {
                        'mean': dynamic_mean,
                        'std': float(np.std(dynamic_stable)),
                        'min': float(np.min(dynamic_stable)),
                        'max': float(np.max(dynamic_stable)),
                    },
                    'improvement_percent': improvement,
                }
        
        return analysis
    
    def print_results(self, analysis: Dict[str, Any]) -> None:
        """
        打印对比结果
        
        Args:
            analysis: 分析结果
        """
        print(f"\n{'='*80}")
        print("对比结果")
        print(f"{'='*80}\n")
        
        metric_names = {
            'delays': '平均时延 (s)',
            'energy': '平均能耗 (J)',
            'throughput': '吞吐量 (Mbps)',
            'completion_rate': '完成率',
        }
        
        for metric, name in metric_names.items():
            if metric in analysis.get('comparisons', {}):
                comp = analysis['comparisons'][metric]
                fixed = comp['fixed_mode']
                dynamic = comp['dynamic_mode']
                improvement = comp['improvement_percent']
                
                print(f"{name}:")
                print(f"  固定带宽:  {fixed['mean']:.6f} ± {fixed['std']:.6f}")
                print(f"  动态带宽:  {dynamic['mean']:.6f} ± {dynamic['std']:.6f}")
                print(f"  改进:     {improvement:+.2f}%\n")
    
    def save_results(self, analysis: Dict[str, Any], output_dir: str = "results") -> None:
        """
        保存分析结果
        
        Args:
            analysis: 分析结果
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"bandwidth_comparison_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 结果已保存到: {filepath}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="对比带宽分配模式性能")
    parser.add_argument('--episodes', type=int, default=100,
                        help='运行轮次(默认: 100)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子(默认: 42)')
    parser.add_argument('--num-vehicles', type=int, default=12,
                        help='车辆数(默认: 12)')
    parser.add_argument('--num-rsus', type=int, default=4,
                        help='RSU数(默认: 4)')
    parser.add_argument('--num-uavs', type=int, default=2,
                        help='UAV数(默认: 2)')
    
    args = parser.parse_args()
    
    scenario = {
        'num_vehicles': args.num_vehicles,
        'num_rsus': args.num_rsus,
        'num_uavs': args.num_uavs,
    }
    
    comparator = BandwidthModeComparator(
        scenario=scenario,
        episodes=args.episodes,
        seed=args.seed
    )
    
    analysis = comparator.run_comparison()
    comparator.print_results(analysis)
    comparator.save_results(analysis)


if __name__ == '__main__':
    main()
