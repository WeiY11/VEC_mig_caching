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

# 配置中文字体
import platform
import matplotlib.font_manager as fm

def configure_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    
    # 尝试设置中文字体
    try:
        if system == 'Windows':
            # Windows系统使用微软雅黑或SimHei
            plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong']
        elif system == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Songti SC', 'STHeiti', 'Arial Unicode MS']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'DejaVu Sans']
        
        # 解决负号显示问题
        plt.rcParams['axes.unicode_minus'] = False
        
    except Exception as e:
        # 如果设置失败，使用英文标签
        print(f"⚠️ 中文字体配置失败: {e}，将使用英文标签")
        pass

# 初始化字体配置
configure_chinese_font()


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
        
        # 12. 建议和结论
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
        
        # 计算改进幅度
        initial_reward = training_env.episode_rewards[0] if training_env.episode_rewards else 0
        final_reward = training_env.episode_rewards[-1] if training_env.episode_rewards else 0
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
                    <div class="metric-label">平均奖励</div>
                    <div class="metric-value">{avg_reward:.3f}</div>
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
        labels = ['探索期\n(前25%)', '学习期\n(中50%)', '收敛期\n(后25%)']
        
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
                最终平均奖励为 <span class="highlight">{final_perf.get('avg_reward', 0):.3f}</span>，
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
