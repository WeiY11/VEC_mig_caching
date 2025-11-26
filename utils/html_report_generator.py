"""
HTML训练报告生成器 (Refactored with Jinja2)
生成全面详细的训练结果HTML报告，包含可视化图表、性能指标和系统统计
"""
import os
import json
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import jinja2

class HTMLReportGenerator:
    """HTML训练报告生成器"""
    
    def __init__(self):
        # Setup Jinja2 environment
        template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
    def generate_full_report(self, 
                           algorithm: str,
                           training_env: Any,
                           training_time: float,
                           results: Dict,
                           simulator_stats: Optional[Dict] = None) -> str:
        """
        生成完整的HTML报告
        """
        template = self.env.get_template('report_template.html')
        
        # Prepare data context for the template
        context = self._prepare_context(algorithm, training_env, training_time, results, simulator_stats)
        
        return template.render(**context)

    def _prepare_context(self, algorithm, training_env, training_time, results, simulator_stats):
        """Prepare the dictionary context for Jinja2 template"""
        
        final_perf = results.get('final_performance', {})
        num_episodes = len(training_env.episode_rewards)
        
        # Calculate improvement
        initial_reward = training_env.episode_rewards[0] if training_env.episode_rewards else 0
        final_reward = training_env.episode_rewards[-1] if training_env.episode_rewards else 0
        reward_improvement = ((final_reward - initial_reward) / abs(initial_reward) * 100) if initial_reward != 0 else 0

        # Prepare chart data
        chart_data = {
            'rewards': training_env.episode_rewards,
            'delay': training_env.episode_metrics.get('avg_delay', []),
            'energy': training_env.episode_metrics.get('total_energy', []),
            # New Metrics
            'offload_local': training_env.episode_metrics.get('local_offload_ratio', []),
            'offload_rsu': training_env.episode_metrics.get('rsu_offload_ratio', []),
            'offload_uav': training_env.episode_metrics.get('uav_offload_ratio', []),
            'migration_success': training_env.episode_metrics.get('migration_success_rate', []),
            'migration_cost': training_env.episode_metrics.get('migration_avg_cost', []),
            'cache_hit': training_env.episode_metrics.get('cache_hit_rate', []),
            'cache_util': training_env.episode_metrics.get('cache_utilization', []),
            'rsu_hotspot': training_env.episode_metrics.get('rsu_hotspot_peak', []),
            'queue_overload': training_env.episode_metrics.get('queue_overload_events', [])
        }
        
        # Smart Insights
        smart_insights = self._generate_smart_insights_data(training_env, results)
        recommendations = self._generate_recommendations_data(algorithm, training_env, results)
        
        # Detailed Metrics
        detailed_metrics = self._prepare_detailed_metrics(training_env)
        
        # System Stats
        cache_stats = {
            'hit_rate': np.mean(training_env.episode_metrics.get('cache_hit_rate', [0]))
        }
        offload_stats = {
            'ratio': np.mean(training_env.episode_metrics.get('offload_ratio', [0]))
        }
        rsu_stats = {
            'utilization': np.mean(training_env.episode_metrics.get('rsu_utilization', [0]))
        }

        return {
            'algorithm': algorithm,
            'generated_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'num_episodes': num_episodes,
            'training_time_hours': round(training_time / 3600, 2),
            'final_perf': final_perf,
            'reward_improvement': reward_improvement,
            'chart_data': chart_data,
            'smart_insights': smart_insights,
            'recommendations': recommendations,
            'detailed_metrics': detailed_metrics,
            'cache_stats': cache_stats,
            'offload_stats': offload_stats,
            'rsu_stats': rsu_stats,
            'training_config': results.get('training_config', {}),
            'simulator_stats': simulator_stats or {}
        }

    def _generate_smart_insights_data(self, training_env, results) -> List[Dict]:
        """Generate structured data for smart insights"""
        insights = []
        rewards = training_env.episode_rewards
        
        if not rewards:
            return []

        # Convergence Analysis
        last_20_percent = rewards[-len(rewards)//5:] if len(rewards) >= 5 else rewards
        variance = np.var(last_20_percent)
        mean_reward = np.mean(last_20_percent)
        cv = np.sqrt(variance) / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        if cv < 0.1:
            insights.append({
                'title': '收敛性优秀',
                'level': 'success',
                'description': f'算法收敛良好，后期稳定性高（变异系数: {cv:.3f}）。'
            })
        elif cv < 0.3:
            insights.append({
                'title': '收敛性良好',
                'level': 'warning',
                'description': f'算法基本收敛，但仍有波动（变异系数: {cv:.3f}）。'
            })
        else:
            insights.append({
                'title': '未完全收敛',
                'level': 'danger',
                'description': f'算法波动较大（变异系数: {cv:.3f}），建议增加训练轮次。'
            })

        # Performance Analysis
        final_perf = results.get('final_performance', {})
        completion = final_perf.get('avg_completion', 0)
        
        if completion > 0.95:
            insights.append({
                'title': '任务完成率极高',
                'level': 'success',
                'description': f'系统能够处理绝大多数任务（{completion*100:.1f}%）。'
            })
        elif completion < 0.8:
            insights.append({
                'title': '任务丢失严重',
                'level': 'danger',
                'description': f'当前任务完成率仅为 {completion*100:.1f}%，需检查资源瓶颈。'
            })

        return insights

    def _generate_recommendations_data(self, algorithm, training_env, results) -> List[str]:
        """Generate list of recommendation strings"""
        recs = []
        rewards = training_env.episode_rewards
        
        if len(rewards) < 200:
            recs.append('增加训练轮次：当前训练轮次较少，建议至少训练200轮以保证收敛。')
            
        final_perf = results.get('final_performance', {})
        if final_perf.get('avg_delay', 0) > 1.5:
            recs.append('优化时延：平均时延较高，考虑增加对时延的惩罚权重。')
            
        if algorithm == 'TD3':
            recs.append('TD3参数微调：如果收敛不稳定，尝试调整策略噪声或延迟更新频率。')
            
        return recs

    def _prepare_detailed_metrics(self, training_env) -> List[Dict]:
        """Prepare detailed metrics list"""
        metrics_list = []
        
        def get_stats(name, key, unit='', reverse=False):
            data = training_env.episode_metrics.get(key, [])
            if not data:
                return None
            mean_val = np.mean(data)
            return {
                'name': name,
                'mean': f"{mean_val:.3f}{unit}",
                'std': f"{np.std(data):.3f}",
                'min': f"{np.min(data):.3f}",
                'max': f"{np.max(data):.3f}",
                'rating': '优秀' if (mean_val < 1.0 if reverse else mean_val > 0.9) else '一般',
                'rating_color': 'success' if (mean_val < 1.0 if reverse else mean_val > 0.9) else 'warning'
            }

        m1 = get_stats('平均时延', 'avg_delay', 's', reverse=True)
        if m1: metrics_list.append(m1)
        
        m2 = get_stats('总能耗', 'total_energy', 'J', reverse=True)
        if m2: metrics_list.append(m2)
        
        m3 = get_stats('任务完成率', 'task_completion_rate', '', reverse=False)
        if m3: metrics_list.append(m3)
        
        m4 = get_stats('缓存命中率', 'cache_hit_rate', '', reverse=False)
        if m4: metrics_list.append(m4)

        return metrics_list

    def save_report(self, html_content: str, output_path: str) -> bool:
        """
        保存HTML报告到指定路径
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Error saving report: {e}")
            return False
