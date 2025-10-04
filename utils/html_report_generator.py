"""
HTML训练报告生成器
生成全面详细的训练结果HTML报告，包含可视化图表、性能指标和系统统计
"""
import os
import json
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


class HTMLReportGenerator:
    """HTML训练报告生成器"""
    
    def __init__(self):
        self.report_sections = []
        
    def generate_full_report(self, 
                           algorithm: str,
                           training_env: Any,
                           training_time: float,
                           results: Dict,
                           simulator_stats: Optional[Dict] = None) -> str:
        """
        生成完整的HTML报告
        
        Args:
            algorithm: 算法名称
            training_env: 训练环境对象
            training_time: 训练总时间（秒）
            results: 训练结果字典
            simulator_stats: 仿真器统计信息
            
        Returns:
            HTML报告内容字符串
        """
        html_parts = []
        
        # 添加HTML头部和样式
        html_parts.append(self._generate_html_header(algorithm))
        
        # 1. 执行摘要
        html_parts.append(self._generate_executive_summary(algorithm, training_env, training_time, results))
        
        # 2. 训练配置
        html_parts.append(self._generate_training_config(results))
        
        # 3. 性能指标总览
        html_parts.append(self._generate_performance_overview(training_env, results))
        
        # 4. 训练曲线可视化
        html_parts.append(self._generate_training_charts(algorithm, training_env))
        
        # 5. 详细指标分析
        html_parts.append(self._generate_detailed_metrics(training_env))
        
        # 6. 算法超参数和网络架构
        html_parts.append(self._generate_algorithm_details(algorithm, training_env))
        
        # 7. 训练过程深度分析
        html_parts.append(self._generate_training_analysis(training_env, results))
        
        # 8. 每轮详细数据表格
        html_parts.append(self._generate_episode_data_table(training_env, results))
        
        # 9. 系统统计信息
        if simulator_stats:
            html_parts.append(self._generate_system_statistics(simulator_stats))
        
        # 10. 自适应控制器统计
        html_parts.append(self._generate_adaptive_controller_stats(training_env))
        
        # 11. 收敛性分析
        html_parts.append(self._generate_convergence_analysis(training_env))
        
        # 12. 指标相关性分析（新增）
        html_parts.append(self._generate_correlation_analysis(training_env))
        
        # 13. 逐指标趋势分析（新增）
        html_parts.append(self._generate_per_metric_analysis(training_env))
        
        # 14. 性能雷达图和对比（新增）
        html_parts.append(self._generate_radar_chart_analysis(training_env, results))
        
        # 15. 完整数据导出表格（新增）
        html_parts.append(self._generate_complete_data_table(training_env))
        
        # 16. 峰值和异常分析（新增）
        html_parts.append(self._generate_peak_anomaly_analysis(training_env))
        
        # 17. 学习曲线平滑度分析（新增）
        html_parts.append(self._generate_smoothness_analysis(training_env))
        
        # 18. 建议和结论
        html_parts.append(self._generate_recommendations(training_env, results))
        
        # 添加HTML尾部
        html_parts.append(self._generate_html_footer())
        
        return '\n'.join(html_parts)
    
    def _generate_html_header(self, algorithm: str) -> str:
        """生成HTML头部和CSS样式"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{algorithm} 训练报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: #f8f9fa;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        
        .section-title {{
            font-size: 1.8em;
            color: #667eea;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }}
        
        .section-subtitle {{
            font-size: 1.3em;
            color: #764ba2;
            margin: 25px 0 15px 0;
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: #666;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        
        .metric-unit {{
            font-size: 0.5em;
            color: #999;
        }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        
        .chart-title {{
            font-size: 1.2em;
            margin-bottom: 15px;
            color: #333;
            font-weight: 600;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        th {{
            background: #667eea;
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #eee;
        }}
        
        tr:hover {{
            background: #f8f9fa;
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            transition: width 0.3s ease;
        }}
        
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .recommendation-title {{
            font-weight: 600;
            color: #856404;
            margin-bottom: 5px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .data-table {{
            overflow-x: auto;
        }}
        
        .highlight {{
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .metric-card {{
                break-inside: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 {algorithm} 训练报告</h1>
            <div class="subtitle">生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        </div>
        <div class="content">
"""
    
    def _generate_executive_summary(self, algorithm: str, training_env: Any, 
                                    training_time: float, results: Dict) -> str:
        """生成执行摘要"""
        final_perf = results.get('final_performance', {})
        avg_reward = final_perf.get('avg_reward', 0)
        avg_delay = final_perf.get('avg_delay', 0)
        avg_completion = final_perf.get('avg_completion', 0)
        
        num_episodes = len(training_env.episode_rewards)
        training_hours = training_time / 3600
        
        # 计算改进幅度（基于Episode总奖励）
        initial_reward = training_env.episode_rewards[0] if training_env.episode_rewards else 0
        final_reward = training_env.episode_rewards[-1] if training_env.episode_rewards else 0
        # 注意：负值奖励，越大越好（-100改进到-50是100%改进）
        improvement = ((final_reward - initial_reward) / abs(initial_reward) * 100) if initial_reward != 0 else 0
        
        return f"""
        <div class="section">
            <h2 class="section-title">📊 执行摘要</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">算法类型</div>
                    <div class="metric-value">{algorithm}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">训练轮次</div>
                    <div class="metric-value">{num_episodes}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">训练时长</div>
                    <div class="metric-value">{training_hours:.2f} <span class="metric-unit">小时</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">平均每轮时间</div>
                    <div class="metric-value">{training_time/num_episodes:.2f} <span class="metric-unit">秒</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">🎯 最终性能指标</h3>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Episode总奖励</div>
                    <div class="metric-value">{final_perf.get('avg_episode_reward', avg_reward * 100):.2f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">训练优化目标</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">每步平均奖励</div>
                    <div class="metric-value">{avg_reward:.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">便于对比评估</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">平均时延</div>
                    <div class="metric-value">{avg_delay:.3f} <span class="metric-unit">秒</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">任务完成率</div>
                    <div class="metric-value">{avg_completion*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">性能改进</div>
                    <div class="metric-value" style="color: {'#28a745' if improvement > 0 else '#dc3545'}">
                        {improvement:+.1f} <span class="metric-unit">%</span>
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">基于Episode总奖励</div>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <div class="metric-label">训练完成度</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%;">100%</div>
                </div>
            </div>
        </div>
"""
    
    def _generate_training_config(self, results: Dict) -> str:
        """生成训练配置信息"""
        config = results.get('training_config', {})
        
        return f"""
        <div class="section">
            <h2 class="section-title">⚙️ 训练配置</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>配置项</th>
                        <th>值</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>训练轮次</td>
                        <td><span class="highlight">{config.get('num_episodes', 'N/A')}</span></td>
                    </tr>
                    <tr>
                        <td>每轮最大步数</td>
                        <td>{config.get('max_steps_per_episode', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>训练总时长</td>
                        <td>{config.get('training_time_hours', 0):.3f} 小时</td>
                    </tr>
                    <tr>
                        <td>开始时间</td>
                        <td>{results.get('training_start_time', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>智能体类型</td>
                        <td><span class="status-badge status-info">{results.get('agent_type', 'single_agent')}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
"""
    
    def _generate_performance_overview(self, training_env: Any, results: Dict) -> str:
        """生成性能总览"""
        metrics = training_env.episode_metrics
        
        # 计算统计信息
        def calc_stats(data_list):
            if not data_list:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            arr = np.array(data_list)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        delay_stats = calc_stats(metrics.get('avg_delay', []))
        energy_stats = calc_stats(metrics.get('total_energy', []))
        completion_stats = calc_stats(metrics.get('task_completion_rate', []))
        cache_hit_stats = calc_stats(metrics.get('cache_hit_rate', []))
        
        return f"""
        <div class="section">
            <h2 class="section-title">📈 性能总览</h2>
            
            <h3 class="section-subtitle">平均任务时延</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{delay_stats['mean']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{delay_stats['std']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{delay_stats['min']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{delay_stats['max']:.3f} <span class="metric-unit">秒</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">总能耗</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{energy_stats['mean']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{energy_stats['std']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{energy_stats['min']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{energy_stats['max']:.1f} <span class="metric-unit">J</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">任务完成率</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{completion_stats['mean']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{completion_stats['std']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{completion_stats['min']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{completion_stats['max']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">缓存命中率</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{cache_hit_stats['mean']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{cache_hit_stats['std']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{cache_hit_stats['min']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{cache_hit_stats['max']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_training_charts(self, algorithm: str, training_env: Any) -> str:
        """生成训练曲线图表"""
        charts_html = []
        
        charts_html.append(f"""
        <div class="section">
            <h2 class="section-title">📊 训练曲线可视化</h2>
""")
        
        # 1. 奖励曲线
        if training_env.episode_rewards:
            reward_chart = self._create_reward_chart(training_env.episode_rewards)
            charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">奖励演化曲线</div>
                <img src="data:image/png;base64,{reward_chart}" alt="奖励曲线">
            </div>
""")
        
        # 2. 多指标对比图
        multi_metric_chart = self._create_multi_metric_chart(training_env.episode_metrics)
        charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">关键性能指标演化</div>
                <img src="data:image/png;base64,{multi_metric_chart}" alt="多指标对比">
            </div>
""")
        
        # 3. 能耗和时延对比
        energy_delay_chart = self._create_energy_delay_chart(training_env.episode_metrics)
        charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">能耗与时延权衡分析</div>
                <img src="data:image/png;base64,{energy_delay_chart}" alt="能耗时延">
            </div>
        </div>
""")
        
        return '\n'.join(charts_html)
    
    def _create_reward_chart(self, rewards: List[float]) -> str:
        """创建奖励曲线图并返回base64编码"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, label='Episode Reward', color='#667eea', linewidth=2)
        
        # 添加移动平均
        window = min(20, len(rewards) // 5)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), moving_avg, 
                   label=f'Moving Average ({window})', color='#764ba2', linewidth=2.5)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward Over Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_multi_metric_chart(self, metrics: Dict) -> str:
        """创建多指标对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Key Performance Metrics Evolution', fontsize=16, fontweight='bold')
        
        # 1. 任务完成率
        if 'task_completion_rate' in metrics and metrics['task_completion_rate']:
            ax = axes[0, 0]
            data = [x * 100 for x in metrics['task_completion_rate']]
            ax.plot(data, color='#28a745', linewidth=2)
            ax.set_title('Task Completion Rate (%)')
            ax.set_ylabel('Completion Rate (%)')
            ax.grid(True, alpha=0.3)
        
        # 2. 平均时延
        if 'avg_delay' in metrics and metrics['avg_delay']:
            ax = axes[0, 1]
            ax.plot(metrics['avg_delay'], color='#dc3545', linewidth=2)
            ax.set_title('Average Task Delay (s)')
            ax.set_ylabel('Delay (s)')
            ax.grid(True, alpha=0.3)
        
        # 3. 缓存命中率
        if 'cache_hit_rate' in metrics and metrics['cache_hit_rate']:
            ax = axes[1, 0]
            data = [x * 100 for x in metrics['cache_hit_rate']]
            ax.plot(data, color='#17a2b8', linewidth=2)
            ax.set_title('Cache Hit Rate (%)')
            ax.set_ylabel('Hit Rate (%)')
            ax.set_xlabel('Episode')
            ax.grid(True, alpha=0.3)
        
        # 4. 数据丢失率
        if 'data_loss_ratio_bytes' in metrics and metrics['data_loss_ratio_bytes']:
            ax = axes[1, 1]
            data = [x * 100 for x in metrics['data_loss_ratio_bytes']]
            ax.plot(data, color='#ffc107', linewidth=2)
            ax.set_title('Data Loss Ratio (%)')
            ax.set_ylabel('Loss Ratio (%)')
            ax.set_xlabel('Episode')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_energy_delay_chart(self, metrics: Dict) -> str:
        """创建能耗和时延对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 能耗曲线
        if 'total_energy' in metrics and metrics['total_energy']:
            ax1.plot(metrics['total_energy'], color='#ff6b6b', linewidth=2, label='Total Energy')
            ax1.set_ylabel('Energy (J)', fontsize=12)
            ax1.set_title('Energy Consumption Over Episodes', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 时延曲线
        if 'avg_delay' in metrics and metrics['avg_delay']:
            ax2.plot(metrics['avg_delay'], color='#4ecdc4', linewidth=2, label='Avg Delay')
            ax2.set_ylabel('Delay (s)', fontsize=12)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_title('Average Task Delay Over Episodes', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64编码字符串"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _generate_detailed_metrics(self, training_env: Any) -> str:
        """生成详细指标分析"""
        metrics = training_env.episode_metrics
        
        # 计算最近表现 vs 初始表现
        def compare_performance(data_list, window=20):
            if not data_list or len(data_list) < window:
                return "N/A", "N/A", "N/A"
            initial = np.mean(data_list[:window])
            final = np.mean(data_list[-window:])
            improvement = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
            return f"{initial:.4f}", f"{final:.4f}", f"{improvement:+.2f}%"
        
        # 分析各项指标
        delay_initial, delay_final, delay_improve = compare_performance(metrics.get('avg_delay', []))
        energy_initial, energy_final, energy_improve = compare_performance(metrics.get('total_energy', []))
        completion_initial, completion_final, completion_improve = compare_performance(
            metrics.get('task_completion_rate', [])
        )
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔍 详细指标分析</h2>
            
            <h3 class="section-subtitle">性能改进对比（首20轮 vs 末20轮）</h3>
            
            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>初始表现</th>
                        <th>最终表现</th>
                        <th>改进幅度</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>平均时延</td>
                        <td>{delay_initial} 秒</td>
                        <td>{delay_final} 秒</td>
                        <td><span class="status-badge status-success">{delay_improve}</span></td>
                    </tr>
                    <tr>
                        <td>总能耗</td>
                        <td>{energy_initial} J</td>
                        <td>{energy_final} J</td>
                        <td><span class="status-badge status-success">{energy_improve}</span></td>
                    </tr>
                    <tr>
                        <td>任务完成率</td>
                        <td>{completion_initial}</td>
                        <td>{completion_final}</td>
                        <td><span class="status-badge status-success">{completion_improve}</span></td>
                    </tr>
                </tbody>
            </table>
            
            <h3 class="section-subtitle">训练稳定性分析</h3>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">奖励方差</div>
                    <div class="metric-value">{np.var(training_env.episode_rewards):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">奖励范围</div>
                    <div class="metric-value">
                        {np.max(training_env.episode_rewards) - np.min(training_env.episode_rewards):.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最佳奖励</div>
                    <div class="metric-value">{np.max(training_env.episode_rewards):.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最差奖励</div>
                    <div class="metric-value">{np.min(training_env.episode_rewards):.3f}</div>
                </div>
            </div>
        </div>
"""
    
    def _generate_system_statistics(self, simulator_stats: Dict) -> str:
        """生成系统统计信息"""
        return f"""
        <div class="section">
            <h2 class="section-title">🖥️ 系统统计信息</h2>
            
            <h3 class="section-subtitle">中央RSU调度器</h3>
            <table>
                <thead>
                    <tr>
                        <th>统计项</th>
                        <th>值</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>调度调用次数</td>
                        <td>{simulator_stats.get('scheduling_calls', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>负载均衡指数</td>
                        <td>{simulator_stats.get('load_balance_index', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>系统健康状态</td>
                        <td><span class="status-badge status-success">{simulator_stats.get('system_health', 'N/A')}</span></td>
                    </tr>
                </tbody>
            </table>
            
            <h3 class="section-subtitle">迁移统计</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">RSU迁移数据量</div>
                    <div class="metric-value">{simulator_stats.get('rsu_migration_data', 0):.1f} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">车辆跟随迁移</div>
                    <div class="metric-value">{simulator_stats.get('handover_migrations', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">UAV迁移次数</div>
                    <div class="metric-value">{simulator_stats.get('uav_migration_count', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">回传网络能耗</div>
                    <div class="metric-value">{simulator_stats.get('backhaul_total_energy', 0):.2f} <span class="metric-unit">J</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_algorithm_details(self, algorithm: str, training_env: Any) -> str:
        """生成算法超参数和网络架构详情"""
        # 获取算法特定配置
        algo_params = {}
        if hasattr(training_env.agent_env, 'actor') and hasattr(training_env.agent_env.actor, 'fc1'):
            # 网络结构信息
            actor = training_env.agent_env.actor
            if hasattr(actor, 'fc1'):
                algo_params['actor_layer1'] = actor.fc1.out_features if hasattr(actor.fc1, 'out_features') else 'N/A'
            if hasattr(actor, 'fc2'):
                algo_params['actor_layer2'] = actor.fc2.out_features if hasattr(actor.fc2, 'out_features') else 'N/A'
        
        # 获取学习率等超参数
        if hasattr(training_env.agent_env, 'actor_optimizer'):
            algo_params['actor_lr'] = training_env.agent_env.actor_optimizer.param_groups[0]['lr']
        if hasattr(training_env.agent_env, 'critic_optimizer'):
            algo_params['critic_lr'] = training_env.agent_env.critic_optimizer.param_groups[0]['lr']
        if hasattr(training_env.agent_env, 'gamma'):
            algo_params['gamma'] = training_env.agent_env.gamma
        if hasattr(training_env.agent_env, 'tau'):
            algo_params['tau'] = training_env.agent_env.tau
        if hasattr(training_env.agent_env, 'policy_noise'):
            algo_params['policy_noise'] = training_env.agent_env.policy_noise
        if hasattr(training_env.agent_env, 'noise_clip'):
            algo_params['noise_clip'] = training_env.agent_env.noise_clip
        if hasattr(training_env.agent_env, 'policy_delay'):
            algo_params['policy_delay'] = training_env.agent_env.policy_delay
        
        params_html = ""
        if algo_params:
            for key, value in algo_params.items():
                params_html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td><code>{value}</code></td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔧 算法配置详情</h2>
            
            <h3 class="section-subtitle">算法类型</h3>
            <p style="font-size: 1.1em; margin: 15px 0;">
                <span class="highlight">{algorithm}</span> - 
                {'Twin Delayed Deep Deterministic Policy Gradient' if algorithm == 'TD3' else
                 'Deep Deterministic Policy Gradient' if algorithm == 'DDPG' else
                 'Soft Actor-Critic' if algorithm == 'SAC' else
                 'Proximal Policy Optimization' if algorithm == 'PPO' else
                 'Deep Q-Network' if algorithm == 'DQN' else algorithm}
            </p>
            
            <h3 class="section-subtitle">网络架构</h3>
            <table>
                <thead>
                    <tr>
                        <th>组件</th>
                        <th>配置</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>状态维度</td>
                        <td><code>{getattr(training_env.agent_env, 'state_dim', 'N/A')}</code></td>
                    </tr>
                    <tr>
                        <td>动作维度</td>
                        <td><code>{getattr(training_env.agent_env, 'action_dim', 'N/A')}</code></td>
                    </tr>
                    <tr>
                        <td>动作空间类型</td>
                        <td><code>{'连续' if algorithm in ['DDPG', 'TD3', 'SAC', 'PPO'] else '离散'}</code></td>
                    </tr>
                    {params_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">训练技巧</h3>
            <div style="margin: 20px 0; line-height: 2;">
                {'• <strong>目标网络</strong>: 使用软更新 (τ=' + str(algo_params.get('tau', 'N/A')) + ')<br>' if 'tau' in algo_params else ''}
                {'• <strong>延迟策略更新</strong>: 每' + str(algo_params.get('policy_delay', 'N/A')) + '步更新一次Actor<br>' if algorithm == 'TD3' else ''}
                {'• <strong>目标策略平滑</strong>: 噪声=' + str(algo_params.get('policy_noise', 'N/A')) + ', 裁剪=' + str(algo_params.get('noise_clip', 'N/A')) + '<br>' if algorithm == 'TD3' else ''}
                • <strong>经验回放</strong>: 使用Replay Buffer存储经验<br>
                • <strong>批量训练</strong>: 从Replay Buffer采样进行训练
            </div>
        </div>
"""
    
    def _generate_training_analysis(self, training_env: Any, results: Dict) -> str:
        """生成训练过程深度分析"""
        rewards = training_env.episode_rewards
        
        # 分段分析：前25%，中间50%，后25%
        n = len(rewards)
        if n < 4:
            return ""
        
        quarter = n // 4
        early_rewards = rewards[:quarter]
        mid_rewards = rewards[quarter:3*quarter]
        late_rewards = rewards[3*quarter:]
        
        early_avg = np.mean(early_rewards)
        mid_avg = np.mean(mid_rewards)
        late_avg = np.mean(late_rewards)
        
        early_std = np.std(early_rewards)
        mid_std = np.std(mid_rewards)
        late_std = np.std(late_rewards)
        
        # 计算趋势
        improvement_early_to_mid = ((mid_avg - early_avg) / abs(early_avg) * 100) if early_avg != 0 else 0
        improvement_mid_to_late = ((late_avg - mid_avg) / abs(mid_avg) * 100) if mid_avg != 0 else 0
        
        # 检测异常值
        all_rewards = np.array(rewards)
        q1 = np.percentile(all_rewards, 25)
        q3 = np.percentile(all_rewards, 75)
        iqr = q3 - q1
        outliers = np.sum((all_rewards < q1 - 1.5*iqr) | (all_rewards > q3 + 1.5*iqr))
        
        # 生成更多可视化
        phase_chart = self._create_phase_analysis_chart(early_rewards, mid_rewards, late_rewards)
        distribution_chart = self._create_distribution_chart(rewards)
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔬 训练过程深度分析</h2>
            
            <h3 class="section-subtitle">分阶段性能分析</h3>
            <table>
                <thead>
                    <tr>
                        <th>阶段</th>
                        <th>轮次范围</th>
                        <th>平均奖励</th>
                        <th>标准差</th>
                        <th>相对改进</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="status-badge status-info">探索期 (前25%)</span></td>
                        <td>1 - {quarter}</td>
                        <td>{early_avg:.3f}</td>
                        <td>{early_std:.3f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td><span class="status-badge status-warning">学习期 (中50%)</span></td>
                        <td>{quarter+1} - {3*quarter}</td>
                        <td>{mid_avg:.3f}</td>
                        <td>{mid_std:.3f}</td>
                        <td style="color: {'#28a745' if improvement_early_to_mid > 0 else '#dc3545'}">
                            {improvement_early_to_mid:+.1f}%
                        </td>
                    </tr>
                    <tr>
                        <td><span class="status-badge status-success">收敛期 (后25%)</span></td>
                        <td>{3*quarter+1} - {n}</td>
                        <td>{late_avg:.3f}</td>
                        <td>{late_std:.3f}</td>
                        <td style="color: {'#28a745' if improvement_mid_to_late > 0 else '#dc3545'}">
                            {improvement_mid_to_late:+.1f}%
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <div class="chart-container">
                <div class="chart-title">三阶段奖励分布对比</div>
                <img src="data:image/png;base64,{phase_chart}" alt="阶段分析">
            </div>
            
            <div class="chart-container">
                <div class="chart-title">奖励分布直方图与核密度估计</div>
                <img src="data:image/png;base64,{distribution_chart}" alt="分布图">
            </div>
            
            <h3 class="section-subtitle">统计特征</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">偏度 (Skewness)</div>
                    <div class="metric-value">{self._calculate_skewness(rewards):.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        {self._interpret_skewness(self._calculate_skewness(rewards))}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">峰度 (Kurtosis)</div>
                    <div class="metric-value">{self._calculate_kurtosis(rewards):.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        {self._interpret_kurtosis(self._calculate_kurtosis(rewards))}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">变异系数 (CV)</div>
                    <div class="metric-value">{(np.std(rewards) / abs(np.mean(rewards)) * 100):.2f}<span class="metric-unit">%</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        相对变异程度
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">异常值数量</div>
                    <div class="metric-value">{outliers}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        基于IQR方法检测
                    </div>
                </div>
            </div>
        </div>
"""
    
    def _generate_episode_data_table(self, training_env: Any, results: Dict) -> str:
        """生成每轮详细数据表格"""
        rewards = training_env.episode_rewards
        metrics = training_env.episode_metrics
        
        # 只显示前10轮、中间10轮、最后10轮
        n = len(rewards)
        if n <= 30:
            display_indices = list(range(n))
        else:
            display_indices = (list(range(10)) + 
                             ['...'] + 
                             list(range(n//2 - 5, n//2 + 5)) + 
                             ['...'] + 
                             list(range(n-10, n)))
        
        rows_html = ""
        for idx in display_indices:
            if idx == '...':
                rows_html += """
                <tr style="text-align: center; background: #f0f0f0;">
                    <td colspan="8">...</td>
                </tr>
"""
            else:
                episode_num = idx + 1
                reward = rewards[idx]
                delay = metrics.get('avg_delay', [0])[idx] if idx < len(metrics.get('avg_delay', [])) else 0
                energy = metrics.get('total_energy', [0])[idx] if idx < len(metrics.get('total_energy', [])) else 0
                completion = metrics.get('task_completion_rate', [0])[idx] if idx < len(metrics.get('task_completion_rate', [])) else 0
                cache_hit = metrics.get('cache_hit_rate', [0])[idx] if idx < len(metrics.get('cache_hit_rate', [])) else 0
                data_loss = metrics.get('data_loss_ratio_bytes', [0])[idx] if idx < len(metrics.get('data_loss_ratio_bytes', [])) else 0
                migration_success = metrics.get('migration_success_rate', [0])[idx] if idx < len(metrics.get('migration_success_rate', [])) else 0
                
                # 根据阶段着色
                if idx < n // 4:
                    phase_color = '#e3f2fd'  # 蓝色 - 探索期
                elif idx < 3 * n // 4:
                    phase_color = '#fff3e0'  # 橙色 - 学习期
                else:
                    phase_color = '#e8f5e9'  # 绿色 - 收敛期
                
                rows_html += f"""
                <tr style="background: {phase_color}">
                    <td><strong>{episode_num}</strong></td>
                    <td>{reward:.3f}</td>
                    <td>{delay:.4f}s</td>
                    <td>{energy:.1f}J</td>
                    <td>{completion*100:.2f}%</td>
                    <td>{cache_hit*100:.1f}%</td>
                    <td>{data_loss*100:.2f}%</td>
                    <td>{migration_success*100:.1f}%</td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">📋 每轮详细数据表</h2>
            
            <p style="margin-bottom: 20px; color: #666;">
                完整训练数据记录（显示前10轮、中间10轮、最后10轮）
            </p>
            
            <div class="data-table" style="max-height: 600px; overflow-y: auto;">
                <table>
                    <thead style="position: sticky; top: 0; z-index: 10;">
                        <tr>
                            <th>轮次</th>
                            <th>奖励</th>
                            <th>平均时延</th>
                            <th>总能耗</th>
                            <th>完成率</th>
                            <th>缓存命中率</th>
                            <th>数据丢失率</th>
                            <th>迁移成功率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top: 20px; display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #e3f2fd; border: 1px solid #90caf9;"></div>
                    <span>探索期 (前25%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #fff3e0; border: 1px solid #ffb74d;"></div>
                    <span>学习期 (中50%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #e8f5e9; border: 1px solid #81c784;"></div>
                    <span>收敛期 (后25%)</span>
                </div>
            </div>
        </div>
"""
    
    def _generate_convergence_analysis(self, training_env: Any) -> str:
        """生成收敛性分析"""
        rewards = training_env.episode_rewards
        
        if len(rewards) < 20:
            return ""
        
        # 计算移动平均的方差（衡量收敛性）
        window = 20
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        recent_variance = np.var(moving_avg[-window:]) if len(moving_avg) >= window else np.var(moving_avg)
        
        # 判断是否收敛
        convergence_threshold = np.var(rewards) * 0.1  # 10%的总方差
        is_converged = recent_variance < convergence_threshold
        
        # 估算收敛轮次
        convergence_episode = 0
        if is_converged:
            for i in range(len(moving_avg) - window, -1, -1):
                if np.var(moving_avg[i:i+window]) > convergence_threshold:
                    convergence_episode = i + window
                    break
        
        # 创建收敛图表
        convergence_chart = self._create_convergence_chart(rewards, moving_avg, convergence_episode)
        
        return f"""
        <div class="section">
            <h2 class="section-title">📉 收敛性分析</h2>
            
            <div class="metrics-grid">
                <div class="metric-card" style="border-left-color: {'#28a745' if is_converged else '#ffc107'}">
                    <div class="metric-label">收敛状态</div>
                    <div class="metric-value" style="color: {'#28a745' if is_converged else '#ffc107'}">
                        {'已收敛 ✓' if is_converged else '收敛中 ○'}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">预估收敛轮次</div>
                    <div class="metric-value">{convergence_episode if convergence_episode > 0 else 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">近期方差</div>
                    <div class="metric-value">{recent_variance:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">收敛阈值</div>
                    <div class="metric-value">{convergence_threshold:.3f}</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">收敛过程可视化</div>
                <img src="data:image/png;base64,{convergence_chart}" alt="收敛分析">
            </div>
            
            <h3 class="section-subtitle">收敛性评价</h3>
            <p style="line-height: 2; font-size: 1.05em; padding: 15px; background: white; border-radius: 8px;">
                {self._generate_convergence_comment(is_converged, convergence_episode, len(rewards))}
            </p>
        </div>
"""
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        if len(data) == 0:
            return 0
        from scipy import stats
        return float(stats.skew(data))
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        if len(data) == 0:
            return 0
        from scipy import stats
        return float(stats.kurtosis(data))
    
    def _interpret_skewness(self, skewness):
        """解释偏度"""
        if abs(skewness) < 0.5:
            return "接近对称分布"
        elif skewness > 0:
            return "右偏分布（正偏）"
        else:
            return "左偏分布（负偏）"
    
    def _interpret_kurtosis(self, kurtosis):
        """解释峰度"""
        if abs(kurtosis) < 0.5:
            return "接近正态分布"
        elif kurtosis > 0:
            return "尖峰分布"
        else:
            return "平峰分布"
    
    def _generate_convergence_comment(self, is_converged, convergence_episode, total_episodes):
        """生成收敛性评论"""
        if not is_converged:
            return "⚠️ 训练尚未完全收敛，建议增加训练轮次或调整超参数（如学习率）以达到更稳定的性能。"
        elif convergence_episode < total_episodes * 0.5:
            return f"✅ 训练在第{convergence_episode}轮左右达到收敛，收敛速度较快，表明算法和超参数配置良好。"
        elif convergence_episode < total_episodes * 0.75:
            return f"✓ 训练在第{convergence_episode}轮左右达到收敛，收敛速度适中，性能稳定。"
        else:
            return f"○ 训练在第{convergence_episode}轮才达到收敛，收敛较慢，可能需要调整学习率或网络架构。"
    
    def _create_phase_analysis_chart(self, early, mid, late):
        """创建三阶段分析箱线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [early, mid, late]
        positions = [1, 2, 3]
        labels = ['Exploration\n(First 25%)', 'Learning\n(Middle 50%)', 'Convergence\n(Last 25%)']
        
        bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        
        colors = ['#90caf9', '#ffb74d', '#81c784']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Phase Analysis (Boxplot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._fig_to_base64(fig)
    
    def _create_distribution_chart(self, rewards):
        """创建奖励分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1.hist(rewards, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax1.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax1.set_xlabel('Reward', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Reward Distribution Histogram', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(rewards, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_convergence_chart(self, rewards, moving_avg, convergence_point):
        """创建收敛图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励和移动平均
        ax1.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
        ax1.plot(range(len(moving_avg)), moving_avg, color='#667eea', linewidth=2, label='Moving Average')
        if convergence_point > 0:
            ax1.axvline(convergence_point, color='red', linestyle='--', linewidth=2, label=f'Convergence Point: {convergence_point}')
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward Convergence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 移动方差
        window = 20
        rolling_var = []
        for i in range(len(rewards) - window + 1):
            rolling_var.append(np.var(rewards[i:i+window]))
        
        ax2.plot(rolling_var, color='#ff6b6b', linewidth=2)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Rolling Variance', fontsize=12)
        ax2.set_title('Rolling Variance (Stability Indicator)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_adaptive_controller_stats(self, training_env: Any) -> str:
        """生成自适应控制器统计"""
        cache_metrics = training_env.adaptive_cache_controller.get_cache_metrics()
        migration_metrics = training_env.adaptive_migration_controller.get_migration_metrics()
        
        return f"""
        <div class="section">
            <h2 class="section-title">🤖 自适应控制器统计</h2>
            
            <h3 class="section-subtitle">缓存控制器</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">有效性</div>
                    <div class="metric-value">{cache_metrics.get('effectiveness', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">缓存利用率</div>
                    <div class="metric-value">{cache_metrics.get('utilization', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">迁移控制器</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">有效性</div>
                    <div class="metric-value">{migration_metrics.get('effectiveness', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">决策准确度</div>
                    <div class="metric-value">{migration_metrics.get('decision_quality', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_correlation_analysis(self, training_env: Any) -> str:
        """生成指标相关性分析"""
        metrics = training_env.episode_metrics
        
        # 提取关键指标
        metric_names = ['avg_delay', 'total_energy', 'task_completion_rate', 
                       'cache_hit_rate', 'data_loss_ratio_bytes', 'migration_success_rate']
        
        available_metrics = {}
        for name in metric_names:
            if name in metrics and metrics[name]:
                available_metrics[name] = metrics[name]
        
        if len(available_metrics) < 2:
            return ""
        
        # 计算相关性矩阵
        correlation_chart = self._create_correlation_heatmap(available_metrics)
        
        # 计算强相关对
        strong_correlations = self._find_strong_correlations(available_metrics)
        
        corr_text = ""
        for corr in strong_correlations[:5]:  # 显示前5个
            corr_text += f"""
            <div class="recommendation" style="border-left-color: {'#28a745' if corr['value'] > 0 else '#dc3545'}">
                <div class="recommendation-title">
                    {corr['metric1']} ↔️ {corr['metric2']}: 
                    <strong>{'正相关' if corr['value'] > 0 else '负相关'}</strong> 
                    (r={corr['value']:.3f})
                </div>
                <div>{corr['interpretation']}</div>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔗 指标相关性分析</h2>
            
            <p style="margin-bottom: 20px; line-height: 1.8;">
                通过分析不同性能指标之间的相关关系，可以发现系统行为的内在联系和优化方向。
            </p>
            
            <div class="chart-container">
                <div class="chart-title">指标相关性热力图</div>
                <img src="data:image/png;base64,{correlation_chart}" alt="相关性热力图">
            </div>
            
            <h3 class="section-subtitle">强相关关系解读</h3>
            {corr_text if corr_text else '<p>未发现显著的强相关关系</p>'}
            
            <h3 class="section-subtitle">相关性解释</h3>
            <div style="padding: 15px; background: white; border-radius: 8px; line-height: 1.8;">
                • <strong>正相关 (r > 0.5)</strong>: 两个指标趋向于同时增大或减小<br>
                • <strong>负相关 (r < -0.5)</strong>: 一个指标增大时另一个趋向于减小<br>
                • <strong>弱相关 (|r| < 0.5)</strong>: 两个指标之间关系不明显<br>
                • <strong>相关系数范围</strong>: -1 (完全负相关) 到 +1 (完全正相关)
            </div>
        </div>
"""
    
    def _generate_per_metric_analysis(self, training_env: Any) -> str:
        """生成逐指标详细趋势分析"""
        metrics = training_env.episode_metrics
        
        # 为每个关键指标生成独立的详细图表
        metrics_charts = self._create_all_metrics_charts(metrics)
        
        charts_html = ""
        for metric_info in metrics_charts:
            charts_html += f"""
            <div class="chart-container">
                <div class="chart-title">{metric_info['title']}</div>
                <img src="data:image/png;base64,{metric_info['chart']}" alt="{metric_info['name']}">
                <p style="margin-top: 10px; color: #666; font-size: 0.95em;">
                    {metric_info['description']}
                </p>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">📊 逐指标详细趋势分析</h2>
            
            <p style="margin-bottom: 20px; line-height: 1.8;">
                每个性能指标的完整演化过程，包含原始数据、移动平均、趋势线和置信区间。
            </p>
            
            {charts_html}
        </div>
"""
    
    def _generate_radar_chart_analysis(self, training_env: Any, results: Dict) -> str:
        """生成性能雷达图分析"""
        metrics = training_env.episode_metrics
        
        # 计算不同阶段的归一化性能
        radar_chart = self._create_radar_chart(training_env, metrics)
        
        # 计算综合评分
        n = len(training_env.episode_rewards)
        if n < 4:
            return ""
        
        quarter = n // 4
        
        # 归一化各项指标并计算综合分数
        def normalize_metric(values, inverse=False):
            """归一化到0-100分"""
            if not values:
                return 0
            arr = np.array(values)
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val == min_val:
                return 50
            normalized = (arr - min_val) / (max_val - min_val)
            if inverse:  # 对于越小越好的指标
                normalized = 1 - normalized
            return float(np.mean(normalized[-quarter:]) * 100)
        
        scores = {
            '任务完成率': normalize_metric(metrics.get('task_completion_rate', []), inverse=False),
            '缓存命中率': normalize_metric(metrics.get('cache_hit_rate', []), inverse=False),
            '时延性能': normalize_metric(metrics.get('avg_delay', []), inverse=True),
            '能耗效率': normalize_metric(metrics.get('total_energy', []), inverse=True),
            '数据可靠性': normalize_metric(metrics.get('data_loss_ratio_bytes', []), inverse=True),
            '迁移成功率': normalize_metric(metrics.get('migration_success_rate', []), inverse=False)
        }
        
        overall_score = np.mean(list(scores.values()))
        
        scores_html = ""
        for metric_name, score in scores.items():
            color = '#28a745' if score >= 70 else '#ffc107' if score >= 50 else '#dc3545'
            scores_html += f"""
            <div class="metric-card">
                <div class="metric-label">{metric_name}</div>
                <div class="metric-value" style="color: {color}">{score:.1f}</div>
                <div class="progress-bar" style="height: 10px; margin-top: 10px;">
                    <div class="progress-fill" style="width: {score}%; font-size: 0;"></div>
                </div>
            </div>
"""
        
        grade = 'A+' if overall_score >= 90 else 'A' if overall_score >= 80 else 'B' if overall_score >= 70 else 'C' if overall_score >= 60 else 'D'
        grade_color = '#28a745' if overall_score >= 70 else '#ffc107' if overall_score >= 60 else '#dc3545'
        
        return f"""
        <div class="section">
            <h2 class="section-title">🎯 综合性能雷达图</h2>
            
            <div class="chart-container">
                <div class="chart-title">多维性能雷达图（三阶段对比）</div>
                <img src="data:image/png;base64,{radar_chart}" alt="性能雷达图">
            </div>
            
            <h3 class="section-subtitle">各维度性能评分（后25%轮次）</h3>
            <div class="metrics-grid">
                {scores_html}
            </div>
            
            <h3 class="section-subtitle">综合性能评级</h3>
            <div style="text-align: center; padding: 30px; background: white; border-radius: 10px;">
                <div style="font-size: 4em; font-weight: bold; color: {grade_color}; margin-bottom: 10px;">
                    {grade}
                </div>
                <div style="font-size: 1.5em; color: #666;">
                    综合得分: {overall_score:.1f}/100
                </div>
                <div style="margin-top: 15px; color: #999;">
                    {'优秀' if overall_score >= 80 else '良好' if overall_score >= 70 else '及格' if overall_score >= 60 else '需改进'}
                </div>
            </div>
        </div>
"""
    
    def _generate_complete_data_table(self, training_env: Any) -> str:
        """生成完整的可导出数据表格"""
        rewards = training_env.episode_rewards
        metrics = training_env.episode_metrics
        
        # 生成CSV格式的数据
        csv_data = "Episode,Reward,Avg_Delay,Total_Energy,Completion_Rate,Cache_Hit_Rate,Data_Loss_Ratio,Migration_Success_Rate\n"
        
        for i in range(len(rewards)):
            csv_data += f"{i+1},{rewards[i]:.6f}"
            for metric_name in ['avg_delay', 'total_energy', 'task_completion_rate', 
                               'cache_hit_rate', 'data_loss_ratio_bytes', 'migration_success_rate']:
                if metric_name in metrics and i < len(metrics[metric_name]):
                    csv_data += f",{metrics[metric_name][i]:.6f}"
                else:
                    csv_data += ",0"
            csv_data += "\n"
        
        # Base64编码CSV数据供下载
        csv_b64 = base64.b64encode(csv_data.encode()).decode()
        
        # 生成统计摘要
        summary_rows = ""
        for metric_name, display_name in [
            ('avg_delay', '平均时延'),
            ('total_energy', '总能耗'),
            ('task_completion_rate', '任务完成率'),
            ('cache_hit_rate', '缓存命中率'),
            ('data_loss_ratio_bytes', '数据丢失率'),
            ('migration_success_rate', '迁移成功率')
        ]:
            if metric_name in metrics and metrics[metric_name]:
                data = np.array(metrics[metric_name])
                summary_rows += f"""
                <tr>
                    <td><strong>{display_name}</strong></td>
                    <td>{np.mean(data):.6f}</td>
                    <td>{np.std(data):.6f}</td>
                    <td>{np.min(data):.6f}</td>
                    <td>{np.percentile(data, 25):.6f}</td>
                    <td>{np.median(data):.6f}</td>
                    <td>{np.percentile(data, 75):.6f}</td>
                    <td>{np.max(data):.6f}</td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">💾 完整数据导出</h2>
            
            <p style="margin-bottom: 20px;">
                以下是所有训练轮次的完整数据统计，可以下载CSV文件进行进一步分析。
            </p>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="data:text/csv;base64,{csv_b64}" 
                   download="training_data.csv" 
                   style="display: inline-block; padding: 15px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; text-decoration: none; border-radius: 8px; font-size: 1.1em; 
                          box-shadow: 0 4px 12px rgba(0,0,0,0.2); transition: transform 0.2s;"
                   onmouseover="this.style.transform='translateY(-3px)'"
                   onmouseout="this.style.transform='translateY(0)'">
                    📥 下载完整CSV数据
                </a>
                <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    包含 {len(rewards)} 轮训练数据，8列指标
                </div>
            </div>
            
            <h3 class="section-subtitle">数据统计摘要（所有指标）</h3>
            <div class="data-table" style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>均值</th>
                            <th>标准差</th>
                            <th>最小值</th>
                            <th>Q1 (25%)</th>
                            <th>中位数</th>
                            <th>Q3 (75%)</th>
                            <th>最大值</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>奖励</strong></td>
                            <td>{np.mean(rewards):.6f}</td>
                            <td>{np.std(rewards):.6f}</td>
                            <td>{np.min(rewards):.6f}</td>
                            <td>{np.percentile(rewards, 25):.6f}</td>
                            <td>{np.median(rewards):.6f}</td>
                            <td>{np.percentile(rewards, 75):.6f}</td>
                            <td>{np.max(rewards):.6f}</td>
                        </tr>
                        {summary_rows}
                    </tbody>
                </table>
            </div>
        </div>
"""
    
    def _generate_peak_anomaly_analysis(self, training_env: Any) -> str:
        """生成峰值和异常分析"""
        rewards = training_env.episode_rewards
        
        # 找出最佳和最差的episodes
        rewards_arr = np.array(rewards)
        top_5_idx = np.argsort(rewards_arr)[-5:][::-1]
        bottom_5_idx = np.argsort(rewards_arr)[:5]
        
        # 找出异常波动
        if len(rewards) > 1:
            reward_changes = np.diff(rewards)
            large_jumps_idx = np.where(np.abs(reward_changes) > np.std(reward_changes) * 2)[0]
        else:
            large_jumps_idx = []
        
        top_html = ""
        for rank, idx in enumerate(top_5_idx, 1):
            top_html += f"""
            <tr style="background: #e8f5e9;">
                <td>{rank}</td>
                <td><strong>Episode {idx + 1}</strong></td>
                <td style="color: #28a745; font-weight: bold;">{rewards[idx]:.3f}</td>
                <td>{self._get_episode_description(training_env, idx)}</td>
            </tr>
"""
        
        bottom_html = ""
        for rank, idx in enumerate(bottom_5_idx, 1):
            bottom_html += f"""
            <tr style="background: #ffebee;">
                <td>{rank}</td>
                <td><strong>Episode {idx + 1}</strong></td>
                <td style="color: #dc3545; font-weight: bold;">{rewards[idx]:.3f}</td>
                <td>{self._get_episode_description(training_env, idx)}</td>
            </tr>
"""
        
        jumps_html = ""
        for idx in large_jumps_idx[:10]:  # 只显示前10个
            change = reward_changes[idx]
            jumps_html += f"""
            <tr>
                <td>Episode {idx + 1} → {idx + 2}</td>
                <td style="color: {'#28a745' if change > 0 else '#dc3545'}; font-weight: bold;">
                    {change:+.3f}
                </td>
                <td style="color: {'#28a745' if change > 0 else '#dc3545'}">
                    {abs(change / rewards[idx]) * 100:.1f}%
                </td>
                <td>{'显著提升 ↗' if change > 0 else '显著下降 ↘'}</td>
            </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">⚡ 峰值与异常分析</h2>
            
            <h3 class="section-subtitle">🏆 最佳表现 Top 5</h3>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>轮次</th>
                        <th>奖励</th>
                        <th>特征</th>
                    </tr>
                </thead>
                <tbody>
                    {top_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">📉 最差表现 Bottom 5</h3>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>轮次</th>
                        <th>奖励</th>
                        <th>特征</th>
                    </tr>
                </thead>
                <tbody>
                    {bottom_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">🔄 显著波动事件</h3>
            {'<table><thead><tr><th>轮次变化</th><th>奖励变化</th><th>变化率</th><th>趋势</th></tr></thead><tbody>' + jumps_html + '</tbody></table>' if jumps_html else '<p style="color: #666;">未检测到显著的奖励波动，训练过程相对平稳。</p>'}
            
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px;">
                <strong>💡 提示：</strong> 显著波动通常由探索策略、学习率或环境随机性引起。
                如果波动过大，考虑调整探索噪声或学习率。
            </div>
        </div>
"""
    
    def _generate_smoothness_analysis(self, training_env: Any) -> str:
        """生成学习曲线平滑度分析"""
        rewards = training_env.episode_rewards
        
        if len(rewards) < 10:
            return ""
        
        # 计算平滑度指标
        # 1. 一阶差分的标准差（波动性）
        first_diff = np.diff(rewards)
        volatility = np.std(first_diff)
        
        # 2. 自相关性
        autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1] if len(rewards) > 1 else 0
        
        # 3. 趋势强度
        x = np.arange(len(rewards))
        trend_coef = np.polyfit(x, rewards, 1)[0]
        
        # 4. 平滑指数（基于移动平均的偏离）
        window = min(20, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        deviations = []
        for i in range(len(moving_avg)):
            deviations.append(abs(rewards[i + window//2] - moving_avg[i]))
        smoothness_score = 100 - min(100, np.mean(deviations) * 10)
        
        # 创建平滑度分析图表
        smoothness_chart = self._create_smoothness_chart(rewards, first_diff)
        
        # 评价
        smoothness_grade = '优秀' if smoothness_score >= 70 else '良好' if smoothness_score >= 50 else '一般' if smoothness_score >= 30 else '较差'
        volatility_grade = '低' if volatility < np.std(rewards) * 0.3 else '中' if volatility < np.std(rewards) * 0.6 else '高'
        
        return f"""
        <div class="section">
            <h2 class="section-title">📈 学习曲线平滑度分析</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平滑度评分</div>
                    <div class="metric-value" style="color: {'#28a745' if smoothness_score >= 70 else '#ffc107' if smoothness_score >= 50 else '#dc3545'}">
                        {smoothness_score:.1f}
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {smoothness_grade}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">波动性</div>
                    <div class="metric-value">{volatility:.3f}</div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {volatility_grade}波动
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">自相关系数</div>
                    <div class="metric-value">{autocorr:.3f}</div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {'强' if abs(autocorr) > 0.7 else '中' if abs(autocorr) > 0.4 else '弱'}相关
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">总体趋势</div>
                    <div class="metric-value" style="color: {'#28a745' if trend_coef > 0 else '#dc3545'}">
                        {trend_coef:+.3f}
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {'上升' if trend_coef > 0 else '下降'}趋势
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">奖励变化率分析</div>
                <img src="data:image/png;base64,{smoothness_chart}" alt="平滑度分析">
            </div>
            
            <h3 class="section-subtitle">平滑度解读</h3>
            <div style="padding: 15px; background: white; border-radius: 8px; line-height: 2;">
                • <strong>平滑度评分</strong>: {smoothness_score:.1f}/100 - 表示学习曲线的稳定程度<br>
                • <strong>波动性</strong>: {volatility:.3f} - 相邻轮次奖励变化的标准差<br>
                • <strong>自相关</strong>: {autocorr:.3f} - 反映连续轮次之间的相似性<br>
                • <strong>趋势系数</strong>: {trend_coef:+.3f} - {'正值表示整体进步，数值越大进步越快' if trend_coef > 0 else '负值表示性能下降，需要关注'}<br>
                <br>
                <strong>💡 建议</strong>: 
                {'学习曲线平滑，训练稳定，可以考虑加快学习速度。' if smoothness_score >= 70 else 
                 '学习曲线波动适中，训练正常进行。' if smoothness_score >= 50 else
                 '学习曲线波动较大，建议降低学习率或增加批量大小以提高稳定性。'}
            </div>
        </div>
"""
    
    # 辅助方法
    def _create_correlation_heatmap(self, metrics_dict):
        """创建相关性热力图"""
        import seaborn as sns
        
        # 准备数据
        data = []
        labels = []
        for name, values in metrics_dict.items():
            data.append(values)
            labels.append(name.replace('_', ' ').title())
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(data)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # 添加数值
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Metrics Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _find_strong_correlations(self, metrics_dict):
        """找出强相关关系"""
        names = list(metrics_dict.keys())
        correlations = []
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                corr = np.corrcoef(metrics_dict[names[i]], metrics_dict[names[j]])[0, 1]
                if abs(corr) > 0.5:  # 强相关阈值
                    interpretation = ""
                    if abs(corr) > 0.8:
                        interpretation = "非常强的相关性，这两个指标几乎同步变化"
                    elif abs(corr) > 0.6:
                        interpretation = "强相关性，这两个指标有明显的关联"
                    else:
                        interpretation = "中等相关性，这两个指标有一定关联"
                    
                    correlations.append({
                        'metric1': names[i].replace('_', ' ').title(),
                        'metric2': names[j].replace('_', ' ').title(),
                        'value': corr,
                        'interpretation': interpretation
                    })
        
        # 按绝对值排序
        correlations.sort(key=lambda x: abs(x['value']), reverse=True)
        return correlations
    
    def _create_all_metrics_charts(self, metrics):
        """为所有指标创建详细图表"""
        metric_configs = [
            {'name': 'avg_delay', 'title': 'Average Task Delay Evolution', 'unit': 'seconds', 
             'description': 'Reflects the average delay in processing tasks, including transmission, queuing, and computation delays'},
            {'name': 'total_energy', 'title': 'Total Energy Consumption Evolution', 'unit': 'Joules',
             'description': 'Total system energy consumption, including computation, transmission, and migration energy'},
            {'name': 'task_completion_rate', 'title': 'Task Completion Rate Evolution', 'unit': '%',
             'description': 'Ratio of successfully completed tasks to total tasks, measuring system reliability'},
            {'name': 'cache_hit_rate', 'title': 'Cache Hit Rate Evolution', 'unit': '%',
             'description': 'Ratio of requests directly served from cache, reflecting cache policy effectiveness'},
            {'name': 'data_loss_ratio_bytes', 'title': 'Data Loss Ratio Evolution', 'unit': '%',
             'description': 'Ratio of data lost due to timeout or insufficient resources'},
            {'name': 'migration_success_rate', 'title': 'Migration Success Rate Evolution', 'unit': '%',
             'description': 'Ratio of successfully executed migrations to total migration operations'}
        ]
        
        charts = []
        for config in metric_configs:
            if config['name'] in metrics and metrics[config['name']]:
                chart = self._create_detailed_metric_chart(
                    metrics[config['name']], 
                    config['title'],
                    config['unit']
                )
                charts.append({
                    'name': config['name'],
                    'title': config['title'],
                    'chart': chart,
                    'description': config['description']
                })
        
        return charts
    
    def _create_detailed_metric_chart(self, data, title, unit):
        """创建单个指标的详细图表"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(1, len(data) + 1)
        
        # 原始数据
        ax.plot(episodes, data, alpha=0.3, color='gray', label='Raw Data', linewidth=1)
        
        # 移动平均
        window = min(20, len(data) // 5)
        if window > 1:
            moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(data) + 1), moving_avg, 
                   color='#667eea', linewidth=2.5, label=f'Moving Average ({window})')
            
            # 趋势线
            x = np.arange(len(moving_avg))
            z = np.polyfit(x, moving_avg, 2)
            p = np.poly1d(z)
            ax.plot(range(window, len(data) + 1), p(x), 
                   '--', color='#dc3545', linewidth=2, label='Trend (Polynomial)')
        
        ax.set_xlabel('Episode', fontsize=12)
        # 移除中文标题中的"演化"部分，直接使用英文标题
        ax.set_ylabel(f'Value ({unit})', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_radar_chart(self, training_env, metrics):
        """创建性能雷达图"""
        n = len(training_env.episode_rewards)
        if n < 4:
            return self._fig_to_base64(plt.figure())
        
        quarter = n // 4
        
        # 提取三个阶段的数据
        categories = ['Task\nCompletion', 'Cache Hit\nRate', 'Delay\nPerformance', 'Energy\nEfficiency', 'Data\nReliability', 'Migration\nSuccess']
        
        def get_stage_score(metric_name, stage_slice, inverse=False):
            if metric_name not in metrics or not metrics[metric_name]:
                return 0
            values = metrics[metric_name][stage_slice]
            if not values:
                return 0
            score = np.mean(values)
            # 归一化到0-1
            all_values = metrics[metric_name]
            min_val, max_val = np.min(all_values), np.max(all_values)
            if max_val == min_val:
                return 0.5
            normalized = (score - min_val) / (max_val - min_val)
            return 1 - normalized if inverse else normalized
        
        early_scores = [
            get_stage_score('task_completion_rate', slice(0, quarter)),
            get_stage_score('cache_hit_rate', slice(0, quarter)),
            get_stage_score('avg_delay', slice(0, quarter), inverse=True),
            get_stage_score('total_energy', slice(0, quarter), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(0, quarter), inverse=True),
            get_stage_score('migration_success_rate', slice(0, quarter))
        ]
        
        mid_scores = [
            get_stage_score('task_completion_rate', slice(quarter, 3*quarter)),
            get_stage_score('cache_hit_rate', slice(quarter, 3*quarter)),
            get_stage_score('avg_delay', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('total_energy', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('migration_success_rate', slice(quarter, 3*quarter))
        ]
        
        late_scores = [
            get_stage_score('task_completion_rate', slice(3*quarter, n)),
            get_stage_score('cache_hit_rate', slice(3*quarter, n)),
            get_stage_score('avg_delay', slice(3*quarter, n), inverse=True),
            get_stage_score('total_energy', slice(3*quarter, n), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(3*quarter, n), inverse=True),
            get_stage_score('migration_success_rate', slice(3*quarter, n))
        ]
        
        # 闭合雷达图
        early_scores += early_scores[:1]
        mid_scores += mid_scores[:1]
        late_scores += late_scores[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, early_scores, 'o-', linewidth=2, label='Exploration Phase', color='#90caf9')
        ax.fill(angles, early_scores, alpha=0.15, color='#90caf9')
        
        ax.plot(angles, mid_scores, 'o-', linewidth=2, label='Learning Phase', color='#ffb74d')
        ax.fill(angles, mid_scores, alpha=0.15, color='#ffb74d')
        
        ax.plot(angles, late_scores, 'o-', linewidth=2, label='Convergence Phase', color='#81c784')
        ax.fill(angles, late_scores, alpha=0.15, color='#81c784')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Performance Radar Chart by Training Phase', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _get_episode_description(self, training_env, idx):
        """获取episode的简要描述"""
        metrics = training_env.episode_metrics
        
        # 提取该episode的关键特征
        features = []
        
        if 'task_completion_rate' in metrics and idx < len(metrics['task_completion_rate']):
            rate = metrics['task_completion_rate'][idx]
            if rate >= 0.95:
                features.append("极高完成率")
            elif rate <= 0.85:
                features.append("较低完成率")
        
        if 'cache_hit_rate' in metrics and idx < len(metrics['cache_hit_rate']):
            rate = metrics['cache_hit_rate'][idx]
            if rate >= 0.8:
                features.append("高缓存命中")
            elif rate <= 0.4:
                features.append("低缓存命中")
        
        if 'avg_delay' in metrics and idx < len(metrics['avg_delay']):
            delay = metrics['avg_delay'][idx]
            avg_delay = np.mean(metrics['avg_delay'])
            if delay < avg_delay * 0.8:
                features.append("低延迟")
            elif delay > avg_delay * 1.2:
                features.append("高延迟")
        
        return ", ".join(features) if features else "正常表现"
    
    def _create_smoothness_chart(self, rewards, first_diff):
        """创建平滑度分析图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励及其移动平均
        window = min(20, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        ax1.plot(rewards, alpha=0.5, label='Raw Reward', color='gray')
        ax1.plot(range(window//2, window//2 + len(moving_avg)), moving_avg, 
                linewidth=2.5, label=f'MA({window})', color='#667eea')
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward Smoothness', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 变化率（一阶差分）
        ax2.bar(range(len(first_diff)), first_diff, color=['#28a745' if x > 0 else '#dc3545' for x in first_diff], alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.std(first_diff), color='red', linestyle='--', linewidth=1, label=f'±1 Std ({np.std(first_diff):.2f})')
        ax2.axhline(y=-np.std(first_diff), color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Episode Transition', fontsize=12)
        ax2.set_ylabel('Reward Change', fontsize=12)
        ax2.set_title('Episode-to-Episode Change Rate', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_recommendations(self, training_env: Any, results: Dict) -> str:
        """生成建议和结论"""
        recommendations = []
        
        # 基于性能指标生成建议
        final_perf = results.get('final_performance', {})
        avg_completion = final_perf.get('avg_completion', 0)
        
        if avg_completion < 0.9:
            recommendations.append({
                'title': '任务完成率偏低',
                'content': '建议增加计算资源或优化任务调度策略，以提高任务完成率至90%以上。'
            })
        
        if len(training_env.episode_rewards) > 20:
            recent_var = np.var(training_env.episode_rewards[-20:])
            if recent_var > 1000:
                recommendations.append({
                    'title': '训练不够稳定',
                    'content': '最近轮次的奖励方差较大，建议调整学习率或增加训练轮次以提高稳定性。'
                })
        
        # 能耗建议
        avg_energy = np.mean(training_env.episode_metrics.get('total_energy', [0]))
        if avg_energy > 500:
            recommendations.append({
                'title': '能耗较高',
                'content': f'平均能耗为{avg_energy:.1f}J，建议优化计算卸载策略，增加本地处理比例。'
            })
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"""
            <div class="recommendation">
                <div class="recommendation-title">💡 {rec['title']}</div>
                <div>{rec['content']}</div>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">💡 建议与结论</h2>
            
            {recommendations_html if recommendations_html else '<p>训练表现良好，各项指标均在正常范围内。</p>'}
            
            <h3 class="section-subtitle">总体评价</h3>
            <p style="line-height: 1.8; font-size: 1.1em;">
                本次训练成功完成 <span class="highlight">{len(training_env.episode_rewards)}</span> 个轮次，
                Episode总奖励为 <span class="highlight">{final_perf.get('avg_episode_reward', final_perf.get('avg_reward', 0) * 100):.2f}</span>
                （每步平均 <span class="highlight">{final_perf.get('avg_reward', 0):.3f}</span>），
                任务完成率达到 <span class="highlight">{avg_completion*100:.1f}%</span>。
                {'训练过程稳定，模型收敛良好。' if np.var(training_env.episode_rewards[-20:]) < 1000 else '建议继续优化以提高稳定性。'}
            </p>
        </div>
"""
    
    def _generate_html_footer(self) -> str:
        """生成HTML尾部"""
        return f"""
        </div>
        <div class="footer">
            <p>VEC Migration Caching System - Training Report</p>
            <p>Generated by HTML Report Generator v1.0</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                © 2025 All Rights Reserved | <a href="#">Documentation</a> | <a href="#">GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    def save_report(self, html_content: str, filepath: str) -> bool:
        """
        保存HTML报告到文件
        
        Args:
            html_content: HTML内容
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"保存报告失败: {e}")
            return False
