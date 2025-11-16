#!/usr/bin/env python3
"""
性能评估器
用于评估算法性能和系统指标
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

class PerformanceEvaluator:
    """性能评估器"""
    
    def __init__(self):
        self.metrics = {}
        self.baseline_results = {}
    
    def evaluate_algorithm(self, algorithm_name: str, results: Dict) -> Dict:
        """评估单个算法性能"""
        stats = results.get('statistics', {})
        
        # 基本性能指标
        performance = {
            'algorithm': algorithm_name,
            'total_tasks': stats.get('total_tasks', 0),
            'completed_tasks': stats.get('completed_tasks', 0),
            'completion_rate': stats.get('completion_rate', 0.0),
            'avg_delay': stats.get('avg_delay', 0.0),
            'total_energy': stats.get('total_energy', 0.0),
            'cache_hit_rate': stats.get('cache_hit_rate', 0.0),
            'drop_rate': stats.get('drop_rate', 0.0)
        }
        
        # 3GPP标准相关指标
        performance.update({
            'avg_sinr': stats.get('avg_sinr', 0.0),
            'avg_throughput': stats.get('avg_throughput', 0.0),
            'handover_success_rate': stats.get('handover_success_rate', 1.0),
            'path_loss_avg': stats.get('path_loss_avg', 0.0),
            'interference_level': stats.get('interference_level', 0.0),
            'channel_quality': stats.get('channel_quality', 0.0)
        })
        
        # 分层学习相关指标
        performance.update({
            'strategic_reward': stats.get('strategic_reward', 0.0),
            'tactical_reward': stats.get('tactical_reward', 0.0),
            'operational_reward': stats.get('operational_reward', 0.0),
            'coordination_efficiency': stats.get('coordination_efficiency', 0.0),
            'decision_consistency': stats.get('decision_consistency', 0.0)
        })
        
        # 计算衍生指标
        if performance['completed_tasks'] > 0:
            performance['energy_efficiency'] = performance['completed_tasks'] / max(performance['total_energy'], 1)
            performance['delay_efficiency'] = 1.0 / max(performance['avg_delay'], 0.001)
            performance['spectral_efficiency'] = performance['avg_throughput'] / max(stats.get('bandwidth_used', 1), 1)
        else:
            performance['energy_efficiency'] = 0.0
            performance['delay_efficiency'] = 0.0
            performance['spectral_efficiency'] = 0.0
        
        # 综合性能分数
        performance['composite_score'] = self.calculate_composite_score(performance)
        
        self.metrics[algorithm_name] = performance
        return performance
    
    def calculate_composite_score(self, performance: Dict) -> float:
        """计算综合性能分数"""
        # 权重设置
        weights = {
            'completion_rate': 0.25,
            'delay_efficiency': 0.2,
            'energy_efficiency': 0.2,
            'cache_hit_rate': 0.15,
            'sinr_quality': 0.1,
            'handover_success_rate': 0.1
        }
        
        # 归一化处理
        normalized_scores = {
            'completion_rate': performance['completion_rate'],
            'delay_efficiency': min(performance['delay_efficiency'], 10.0) / 10.0,
            'energy_efficiency': min(performance['energy_efficiency'] / 1000, 1.0),
            'cache_hit_rate': performance['cache_hit_rate'],
            'sinr_quality': performance.get('avg_sinr', 0.0) / 30.0,  # 假设最大SINR为30dB
            'handover_success_rate': performance.get('handover_success_rate', 1.0)
        }
        
        # 加权求和
        composite_score = sum(
            weights[metric] * min(max(normalized_scores[metric], 0.0), 1.0)
            for metric in weights.keys()
        )
        
        return composite_score
    
    def compare_algorithms(self, results_dict: Dict[str, Dict]) -> Dict:
        """比较多个算法"""
        comparison = {}
        
        # 评估每个算法
        for algorithm, results in results_dict.items():
            comparison[algorithm] = self.evaluate_algorithm(algorithm, results)
        
        # 找出最佳算法
        best_algorithm = max(comparison.keys(), 
                           key=lambda x: comparison[x]['composite_score'])
        
        # 计算改进幅度
        improvements = {}
        best_performance = comparison[best_algorithm]
        
        for algorithm, performance in comparison.items():
            if algorithm != best_algorithm:
                improvements[algorithm] = self.calculate_improvements(
                    best_performance, performance
                )
        
        return {
            'individual_performance': comparison,
            'best_algorithm': best_algorithm,
            'improvements': improvements,
            'ranking': self.rank_algorithms(comparison)
        }
    
    def calculate_improvements(self, best: Dict, current: Dict) -> Dict:
        """计算改进幅度"""
        improvements = {}
        
        metrics_to_compare = [
            'completion_rate', 'avg_delay', 'total_energy', 
            'cache_hit_rate', 'composite_score'
        ]
        
        for metric in metrics_to_compare:
            best_val = best.get(metric, 0)
            current_val = current.get(metric, 0)
            
            if metric == 'avg_delay' or metric == 'total_energy':
                # 对于时延和能耗，越小越好
                if current_val > 0:
                    improvement = (current_val - best_val) / current_val * 100
                else:
                    improvement = 0
            else:
                # 对于其他指标，越大越好
                if current_val > 0:
                    improvement = (best_val - current_val) / current_val * 100
                else:
                    improvement = 0
            
            improvements[f'{metric}_improvement'] = improvement
        
        return improvements
    
    def rank_algorithms(self, comparison: Dict) -> List[Tuple[str, float]]:
        """算法排名"""
        ranking = [(alg, perf['composite_score']) 
                  for alg, perf in comparison.items()]
        ranking.sort(key=lambda x: x[1], reverse=True)
        return ranking
    
    def generate_performance_report(self, comparison_results: Dict) -> str:
        """生成性能报告"""
        report = "# 算法性能评估报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # 算法排名
        report += "## 算法排名\n\n"
        ranking = comparison_results['ranking']
        for i, (algorithm, score) in enumerate(ranking, 1):
            report += f"{i}. **{algorithm}** - 综合分数: {score:.3f}\n"
        
        report += "\n## 详细性能指标\n\n"
        report += "| 算法 | 完成率 | 平均时延(s) | 总能耗(J) | 缓存命中率 | 综合分数 |\n"
        report += "|------|--------|-------------|-----------|------------|----------|\n"
        
        for algorithm, performance in comparison_results['individual_performance'].items():
            report += f"| {algorithm} | {performance['completion_rate']:.2%} | "
            report += f"{performance['avg_delay']:.3f} | {performance['total_energy']:.1f} | "
            report += f"{performance['cache_hit_rate']:.2%} | {performance['composite_score']:.3f} |\n"
        
        # 最佳算法分析
        best_alg = comparison_results['best_algorithm']
        report += f"\n## 最佳算法: {best_alg}\n\n"
        
        best_perf = comparison_results['individual_performance'][best_alg]
        report += f"- **完成率**: {best_perf['completion_rate']:.2%}\n"
        report += f"- **平均时延**: {best_perf['avg_delay']:.3f}s\n"
        report += f"- **总能耗**: {best_perf['total_energy']:.1f}J\n"
        report += f"- **缓存命中率**: {best_perf['cache_hit_rate']:.2%}\n"
        report += f"- **综合分数**: {best_perf['composite_score']:.3f}\n"
        
        # 改进分析
        if comparison_results['improvements']:
            report += "\n## 改进分析\n\n"
            for algorithm, improvements in comparison_results['improvements'].items():
                report += f"### {best_alg} vs {algorithm}\n\n"
                for metric, improvement in improvements.items():
                    if 'improvement' in metric:
                        metric_name = metric.replace('_improvement', '')
                        report += f"- **{metric_name}**: {improvement:+.1f}%\n"
                report += "\n"
        
        return report
    
    def plot_performance_comparison(self, comparison_results: Dict, save_path: Optional[str] = None):
        """绘制性能对比图"""
        algorithms = list(comparison_results['individual_performance'].keys())
        
        # 提取指标数据
        completion_rates = [comparison_results['individual_performance'][alg]['completion_rate'] 
                          for alg in algorithms]
        avg_delays = [comparison_results['individual_performance'][alg]['avg_delay'] 
                     for alg in algorithms]
        cache_hit_rates = [comparison_results['individual_performance'][alg]['cache_hit_rate'] 
                          for alg in algorithms]
        composite_scores = [comparison_results['individual_performance'][alg]['composite_score'] 
                           for alg in algorithms]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('算法性能对比', fontsize=16, fontweight='bold')
        
        # 完成率对比
        axes[0, 0].bar(algorithms, completion_rates, color='skyblue')
        axes[0, 0].set_title('任务完成率')
        axes[0, 0].set_ylabel('完成率')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 平均时延对比
        axes[0, 1].bar(algorithms, avg_delays, color='lightcoral')
        axes[0, 1].set_title('平均时延')
        axes[0, 1].set_ylabel('时延 (秒)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 缓存命中率对比
        axes[1, 0].bar(algorithms, cache_hit_rates, color='lightgreen')
        axes[1, 0].set_title('缓存命中率')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 综合分数对比
        axes[1, 1].bar(algorithms, composite_scores, color='gold')
        axes[1, 1].set_title('综合性能分数')
        axes[1, 1].set_ylabel('分数')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"性能对比图已保存: {save_path}")
        
        plt.show()
    
    def save_results(self, comparison_results: Dict, filepath: str):
        """保存评估结果"""
        # 准备JSON序列化的数据
        json_data = {
            'timestamp': datetime.now().isoformat(),
            'evaluation_results': comparison_results,
            'summary': {
                'total_algorithms': len(comparison_results['individual_performance']),
                'best_algorithm': comparison_results['best_algorithm'],
                'best_score': comparison_results['individual_performance'][
                    comparison_results['best_algorithm']
                ]['composite_score']
            }
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存: {filepath}")

def test_evaluator():
    """测试评估器"""
    print("🧪 测试性能评估器...")
    
    # 模拟算法结果
    mock_results = {
        'MATD3': {
            'statistics': {
                'total_tasks': 1000,
                'completed_tasks': 850,
                'completion_rate': 0.85,
                'avg_delay': 0.12,
                'total_energy': 5000,
                'cache_hit_rate': 0.75,
                'drop_rate': 0.15
            }
        },
        'MADDPG': {
            'statistics': {
                'total_tasks': 1000,
                'completed_tasks': 800,
                'completion_rate': 0.80,
                'avg_delay': 0.15,
                'total_energy': 5500,
                'cache_hit_rate': 0.65,
                'drop_rate': 0.20
            }
        },
        'Random': {
            'statistics': {
                'total_tasks': 1000,
                'completed_tasks': 600,
                'completion_rate': 0.60,
                'avg_delay': 0.25,
                'total_energy': 7000,
                'cache_hit_rate': 0.30,
                'drop_rate': 0.40
            }
        }
    }
    
    # 创建评估器
    evaluator = PerformanceEvaluator()
    
    # 进行比较
    comparison = evaluator.compare_algorithms(mock_results)
    
    # 生成报告
    report = evaluator.generate_performance_report(comparison)
    print("\n" + "="*50)
    print(report)
    
    # 绘制对比图
    evaluator.plot_performance_comparison(comparison)
    
    print("✅ 评估器测试完成")

if __name__ == "__main__":
    test_evaluator()