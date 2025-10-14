#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结果管理器
统一管理所有算法的结果保存，确保格式一致且互不干扰

【功能】
1. 独立保存每个算法的结果
2. 统一的JSON格式（兼容现有分析脚本）
3. 时间戳命名避免覆盖
4. 结果汇总和统计
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional


class ResultsManager:
    """结果管理器"""
    
    def __init__(self, base_dir: str = None):
        """
        初始化结果管理器
        
        【参数】
        - base_dir: 基础保存目录（默认为baseline_comparison/results）
        """
        if base_dir is None:
            self.base_dir = Path(__file__).parent.parent.parent / "results"
        else:
            self.base_dir = Path(base_dir)
        
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_results(
        self,
        algorithm: str,
        results: Dict[str, Any],
        algorithm_type: str = 'DRL',
        save_dir: Optional[str] = None
    ) -> Path:
        """
        保存算法结果
        
        【参数】
        - algorithm: 算法名称
        - results: 结果字典
        - algorithm_type: 算法类型（DRL或Heuristic）
        - save_dir: 自定义保存目录（覆盖默认）
        
        【返回】
        - 保存的文件路径
        """
        # 确定保存目录
        if save_dir is None:
            save_dir = self.base_dir / algorithm.lower()
        else:
            save_dir = Path(save_dir)
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成文件名（带时间戳）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{algorithm.lower()}_{timestamp}.json"
        filepath = save_dir / filename
        
        # 补充元信息
        results['algorithm'] = algorithm
        results['algorithm_type'] = algorithm_type
        results['timestamp'] = timestamp
        results['save_path'] = str(filepath)
        
        # 保存JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 结果已保存: {filepath}")
        
        # 同时保存一份最新结果（方便快速访问）
        latest_file = save_dir / f"{algorithm.lower()}_latest.json"
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        加载结果文件
        
        【参数】
        - filepath: 结果文件路径
        
        【返回】
        - 结果字典
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"结果文件不存在: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        return results
    
    def get_latest_results(self, algorithm: str) -> Optional[Dict[str, Any]]:
        """
        获取算法的最新结果
        
        【参数】
        - algorithm: 算法名称
        
        【返回】
        - 最新结果字典，如果不存在则返回None
        """
        latest_file = self.base_dir / algorithm.lower() / f"{algorithm.lower()}_latest.json"
        
        if not latest_file.exists():
            return None
        
        return self.load_results(str(latest_file))
    
    def list_algorithm_results(self, algorithm: str) -> List[Path]:
        """
        列出算法的所有结果文件
        
        【参数】
        - algorithm: 算法名称
        
        【返回】
        - 结果文件路径列表（按时间排序）
        """
        algo_dir = self.base_dir / algorithm.lower()
        
        if not algo_dir.exists():
            return []
        
        # 查找所有JSON文件（排除latest）
        result_files = [
            f for f in algo_dir.glob(f"{algorithm.lower()}_*.json")
            if not f.name.endswith('_latest.json')
        ]
        
        # 按修改时间排序
        result_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        return result_files
    
    def summarize_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        汇总结果统计信息
        
        【参数】
        - results: 原始结果字典
        
        【返回】
        - 汇总统计字典
        """
        summary = {
            'algorithm': results.get('algorithm', 'Unknown'),
            'num_episodes': results.get('num_episodes', 0),
            'seed': results.get('seed', results.get('random_seed', 0)),
        }
        
        # 提取关键指标
        if 'avg_delay' in results:
            summary['avg_delay'] = results['avg_delay']
            summary['std_delay'] = results.get('std_delay', 0)
        
        if 'avg_energy' in results:
            summary['avg_energy'] = results['avg_energy']
            summary['std_energy'] = results.get('std_energy', 0)
        
        if 'avg_completion_rate' in results:
            summary['avg_completion_rate'] = results['avg_completion_rate']
        
        if 'execution_time' in results:
            summary['execution_time'] = results['execution_time']
        
        # 从episode数据计算（如果可用）
        if 'episode_rewards' in results:
            rewards = results['episode_rewards']
            if len(rewards) > 0:
                summary['initial_reward'] = float(np.mean(rewards[:10])) if len(rewards) >= 10 else float(np.mean(rewards))
                summary['final_reward'] = float(np.mean(rewards[-10:])) if len(rewards) >= 10 else float(np.mean(rewards))
                summary['reward_improvement'] = summary['final_reward'] - summary['initial_reward']
        
        if 'episode_delays' in results:
            delays = results['episode_delays']
            if len(delays) > 0:
                stable_start = len(delays) // 2
                summary['stable_avg_delay'] = float(np.mean(delays[stable_start:]))
                summary['stable_std_delay'] = float(np.std(delays[stable_start:]))
        
        if 'episode_energies' in results:
            energies = results['episode_energies']
            if len(energies) > 0:
                stable_start = len(energies) // 2
                summary['stable_avg_energy'] = float(np.mean(energies[stable_start:]))
                summary['stable_std_energy'] = float(np.std(energies[stable_start:]))
        
        return summary
    
    def print_summary(self, results: Dict[str, Any]):
        """
        打印结果摘要
        
        【参数】
        - results: 结果字典
        """
        summary = self.summarize_results(results)
        
        print("\n" + "="*80)
        print(f"算法结果摘要: {summary['algorithm']}")
        print("="*80)
        
        print(f"训练轮次: {summary.get('num_episodes', 'N/A')}")
        print(f"随机种子: {summary.get('seed', 'N/A')}")
        
        if 'avg_delay' in summary:
            print(f"平均时延: {summary['avg_delay']:.3f}±{summary.get('std_delay', 0):.3f}s")
        
        if 'avg_energy' in summary:
            print(f"平均能耗: {summary['avg_energy']:.1f}±{summary.get('std_energy', 0):.1f}J")
        
        if 'avg_completion_rate' in summary:
            print(f"任务完成率: {summary['avg_completion_rate']:.2%}")
        
        if 'initial_reward' in summary and 'final_reward' in summary:
            print(f"初始奖励: {summary['initial_reward']:.3f}")
            print(f"最终奖励: {summary['final_reward']:.3f}")
            print(f"奖励提升: {summary.get('reward_improvement', 0):.3f}")
        
        if 'execution_time' in summary:
            print(f"运行耗时: {summary['execution_time']:.1f}秒")
        
        print("="*80 + "\n")
    
    def compare_algorithms(self, algorithms: List[str]) -> Dict[str, Any]:
        """
        对比多个算法的最新结果
        
        【参数】
        - algorithms: 算法名称列表
        
        【返回】
        - 对比结果字典
        """
        comparison = {
            'algorithms': algorithms,
            'summaries': {},
            'best_delay': None,
            'best_energy': None,
            'best_completion': None,
        }
        
        min_delay = float('inf')
        min_energy = float('inf')
        max_completion = 0.0
        
        for algo in algorithms:
            results = self.get_latest_results(algo)
            
            if results is None:
                print(f"⚠️  {algo}: 无结果数据")
                continue
            
            summary = self.summarize_results(results)
            comparison['summaries'][algo] = summary
            
            # 更新最佳值
            if 'avg_delay' in summary and summary['avg_delay'] < min_delay:
                min_delay = summary['avg_delay']
                comparison['best_delay'] = algo
            
            if 'avg_energy' in summary and summary['avg_energy'] < min_energy:
                min_energy = summary['avg_energy']
                comparison['best_energy'] = algo
            
            if 'avg_completion_rate' in summary and summary['avg_completion_rate'] > max_completion:
                max_completion = summary['avg_completion_rate']
                comparison['best_completion'] = algo
        
        return comparison


if __name__ == "__main__":
    # 测试结果管理器
    print("="*80)
    print("结果管理器测试")
    print("="*80)
    
    manager = ResultsManager()
    
    # 测试保存
    test_results = {
        'num_episodes': 200,
        'seed': 42,
        'avg_delay': 0.245,
        'std_delay': 0.032,
        'avg_energy': 156.3,
        'std_energy': 12.4,
        'avg_completion_rate': 0.952,
        'episode_rewards': list(np.random.randn(200) * 10 - 50),
        'episode_delays': list(np.random.rand(200) * 0.1 + 0.2),
        'episode_energies': list(np.random.rand(200) * 20 + 150),
    }
    
    # 保存测试结果
    filepath = manager.save_results('TD3', test_results, 'DRL')
    print(f"测试结果已保存: {filepath}")
    
    # 打印摘要
    manager.print_summary(test_results)
    
    # 列出结果文件
    files = manager.list_algorithm_results('TD3')
    print(f"找到 {len(files)} 个TD3结果文件")
    
    print("\n" + "="*80)
    print("结果管理器测试完成！")
    print("="*80)








