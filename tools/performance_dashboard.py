#!/usr/bin/env python3
"""
性能指标仪表板
提供实时性能监控和综合指标展示
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
from pathlib import Path
import time
from datetime import datetime
import json

# 设置中文字体和符号
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = ['sans-serif']

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

class PerformanceDashboard:
    """性能指标仪表板类"""
    
    def __init__(self):
        self.metrics_history = []
        self.algorithm_results = {}
        self.real_time_data = {
            'timestamps': [],
            'rewards': [],
            'delays': [],
            'energy': [],
            'cache_hits': [],
            'completion_rates': []
        }
        
    def update_real_time_metrics(self, metrics: Dict):
        """更新实时指标数据"""
        current_time = datetime.now()
        self.real_time_data['timestamps'].append(current_time)
        self.real_time_data['rewards'].append(metrics.get('avg_reward', 0))
        self.real_time_data['delays'].append(metrics.get('avg_task_delay', 0))
        self.real_time_data['energy'].append(metrics.get('total_energy_consumption', 0))
        self.real_time_data['cache_hits'].append(metrics.get('cache_hit_rate', 0))
        self.real_time_data['completion_rates'].append(metrics.get('task_completion_rate', 0))
        
        # 保持最近100个数据点
        max_points = 100
        for key in self.real_time_data:
            if len(self.real_time_data[key]) > max_points:
                self.real_time_data[key] = self.real_time_data[key][-max_points:]
    
    def create_comprehensive_dashboard(self, save_path: str = None):
        """创建综合性能仪表板"""
        # 设置图形布局
        fig = plt.figure(figsize=(20, 16))
        gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 主标题
        fig.suptitle('VEC边缘计算系统性能监控仪表板', fontsize=20, fontweight='bold', y=0.95)
        
        # 1. 实时性能监控 (左上角，2x2)
        ax_realtime = fig.add_subplot(gs[0:2, 0:2])
        self._plot_realtime_metrics(ax_realtime)
        
        # 2. 算法性能对比 (右上角，2x2)
        ax_comparison = fig.add_subplot(gs[0:2, 2:4])
        self._plot_algorithm_comparison(ax_comparison)
        
        # 3. 系统资源利用率 (左下第一行)
        ax_resources = fig.add_subplot(gs[2, 0:2])
        self._plot_resource_utilization(ax_resources)
        
        # 4. 缓存性能分析 (右下第一行)
        ax_cache = fig.add_subplot(gs[2, 2:4])
        self._plot_cache_performance(ax_cache)
        
        # 5. 能耗效率分析 (左下第二行)
        ax_energy = fig.add_subplot(gs[3, 0:2])
        self._plot_energy_efficiency(ax_energy)
        
        # 6. 系统健康度指标 (右下第二行)
        ax_health = fig.add_subplot(gs[3, 2:4])
        self._plot_system_health(ax_health)
        
        # 添加时间戳
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        fig.text(0.99, 0.01, f'更新时间: {timestamp}', ha='right', va='bottom', fontsize=10)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 性能仪表板已保存到: {save_path}")
        
        plt.show()
        return fig
    
    def _plot_realtime_metrics(self, ax):
        """绘制实时性能指标"""
        if not self.real_time_data['timestamps']:
            # 生成模拟数据用于演示
            times = [datetime.now() for _ in range(50)]
            rewards = [-50 + i * 0.8 + np.random.normal(0, 5) for i in range(50)]
            delays = [0.5 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) for i in range(50)]
            
            ax.plot(range(50), rewards, 'b-', label='奖励', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(range(50), delays, 'r-', label='时延', linewidth=2)
            
            ax.set_xlabel('时间步')
            ax.set_ylabel('奖励值', color='b')
            ax2.set_ylabel('时延 (秒)', color='r')
            ax.set_title('实时性能监控', fontweight='bold')
            
            # 合并图例
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        else:
            # 使用真实数据
            times = self.real_time_data['timestamps']
            ax.plot(times, self.real_time_data['rewards'], 'b-', label='奖励', linewidth=2)
            ax2 = ax.twinx()
            ax2.plot(times, self.real_time_data['delays'], 'r-', label='时延', linewidth=2)
            
            ax.set_xlabel('时间')
            ax.set_ylabel('奖励值', color='b')
            ax2.set_ylabel('时延 (秒)', color='r')
            ax.set_title('实时性能监控', fontweight='bold')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_algorithm_comparison(self, ax):
        """绘制算法性能对比"""
        # 模拟算法对比数据
        algorithms = ['MADDPG', 'MATD3', 'MAPPO', 'TD3', 'DDPG']
        metrics = {
            '平均奖励': [0.75, 0.82, 0.68, 0.71, 0.65],
            '完成率': [0.85, 0.88, 0.80, 0.83, 0.78],
            '能耗效率': [0.72, 0.79, 0.70, 0.74, 0.69]
        }
        
        x = np.arange(len(algorithms))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, (metric, values) in enumerate(metrics.items()):
            ax.bar(x + i * width, values, width, label=metric, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('算法')
        ax.set_ylabel('性能指标')
        ax.set_title('算法性能对比', fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(algorithms, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    def _plot_resource_utilization(self, ax):
        """绘制系统资源利用率"""
        resources = ['CPU', 'Memory', 'Network', 'Storage']
        utilization = [0.65, 0.78, 0.45, 0.52]  # 模拟数据
        colors = ['#FF6B6B' if u > 0.8 else '#FFA726' if u > 0.6 else '#66BB6A' for u in utilization]
        
        bars = ax.barh(resources, utilization, color=colors, alpha=0.8)
        ax.set_xlabel('利用率')
        ax.set_title('系统资源利用率', fontweight='bold')
        ax.set_xlim(0, 1)
        
        # 添加数值标签
        for bar, value in zip(bars, utilization):
            ax.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.1%}', va='center', fontweight='bold')
        
        ax.grid(True, alpha=0.3, axis='x')
    
    def _plot_cache_performance(self, ax):
        """绘制缓存性能分析"""
        # 模拟缓存数据
        cache_types = ['热点内容', '用户偏好', '计算结果', '临时数据']
        hit_rates = [0.85, 0.72, 0.68, 0.45]
        
        # 创建饼图
        colors = plt.cm.Set3(np.linspace(0, 1, len(cache_types)))
        wedges, texts, autotexts = ax.pie(hit_rates, labels=cache_types, autopct='%1.1f%%',
                                         colors=colors, startangle=90)
        
        ax.set_title('缓存命中率分布', fontweight='bold')
        
        # 美化文本
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    def _plot_energy_efficiency(self, ax):
        """绘制能耗效率分析"""
        # 模拟能耗数据
        time_periods = ['00:00', '04:00', '08:00', '12:00', '16:00', '20:00']
        energy_consumption = [120, 95, 180, 220, 200, 150]  # 瓦特
        task_completion = [45, 35, 85, 110, 95, 70]  # 任务数
        
        # 计算能耗效率 (任务数/能耗)
        efficiency = [t/e for t, e in zip(task_completion, energy_consumption)]
        
        ax.plot(time_periods, efficiency, 'go-', linewidth=3, markersize=8, label='能耗效率')
        ax.fill_between(time_periods, efficiency, alpha=0.3, color='green')
        
        ax.set_xlabel('时间段')
        ax.set_ylabel('效率 (任务数/瓦特)')
        ax.set_title('24小时能耗效率趋势', fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_system_health(self, ax):
        """绘制系统健康度指标"""
        # 系统健康度指标
        health_metrics = {
            '网络延迟': 0.85,
            '服务可用性': 0.95,
            '错误率': 0.02,
            '响应时间': 0.78,
            '吞吐量': 0.88
        }
        
        # 转换为雷达图数据
        categories = list(health_metrics.keys())
        values = list(health_metrics.values())
        
        # 错误率需要反转 (越低越好)
        values[2] = 1 - values[2]
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        values += values[:1]  # 闭合图形
        angles += angles[:1]
        
        # 清除当前轴并创建极坐标图
        ax.clear()
        ax = plt.subplot(4, 4, 16, projection='polar')  # 重新创建极坐标轴
        
        ax.plot(angles, values, 'o-', linewidth=2, color='#FF6B6B')
        ax.fill(angles, values, alpha=0.25, color='#FF6B6B')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        ax.set_title('系统健康度评估', fontweight='bold', pad=20)
        
        # 添加网格线
        ax.grid(True)
    
    def create_real_time_monitor(self, save_path: str = None):
        """创建实时监控界面"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('实时性能监控', fontsize=16, fontweight='bold')
        
        # 生成模拟实时数据
        time_points = list(range(50))
        
        # 1. 奖励趋势
        rewards = [-50 + i * 0.8 + np.random.normal(0, 5) for i in time_points]
        axes[0, 0].plot(time_points, rewards, 'b-', linewidth=2)
        axes[0, 0].set_title('奖励趋势')
        axes[0, 0].set_ylabel('奖励值')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 时延监控
        delays = [0.5 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) for i in time_points]
        axes[0, 1].plot(time_points, delays, 'r-', linewidth=2)
        axes[0, 1].set_title('平均时延')
        axes[0, 1].set_ylabel('时延 (秒)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 能耗监控
        energy = [100 + 20 * np.sin(i/15) + np.random.normal(0, 5) for i in time_points]
        axes[0, 2].plot(time_points, energy, 'g-', linewidth=2)
        axes[0, 2].set_title('能耗监控')
        axes[0, 2].set_ylabel('能耗 (焦耳)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 缓存命中率
        cache_hits = [0.7 + 0.2 * np.sin(i/8) + np.random.normal(0, 0.05) for i in time_points]
        cache_hits = [max(0, min(1, h)) for h in cache_hits]  # 限制在0-1之间
        axes[1, 0].plot(time_points, cache_hits, 'm-', linewidth=2)
        axes[1, 0].set_title('缓存命中率')
        axes[1, 0].set_ylabel('命中率')
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 任务完成率
        completion = [0.6 + 0.3 * np.sin(i/12) + np.random.normal(0, 0.05) for i in time_points]
        completion = [max(0, min(1, c)) for c in completion]
        axes[1, 1].plot(time_points, completion, 'c-', linewidth=2)
        axes[1, 1].set_title('任务完成率')
        axes[1, 1].set_ylabel('完成率')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 系统负载
        load = [0.4 + 0.4 * np.sin(i/20) + np.random.normal(0, 0.1) for i in time_points]
        load = [max(0, min(1, l)) for l in load]
        axes[1, 2].plot(time_points, load, 'orange', linewidth=2)
        axes[1, 2].set_title('系统负载')
        axes[1, 2].set_ylabel('负载率')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        
        # 设置x轴标签
        for ax in axes.flat:
            ax.set_xlabel('时间步')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 实时监控界面已保存到: {save_path}")
        
        plt.show()
        return fig

def create_performance_dashboard(training_env=None, save_path: str = None):
    """创建性能仪表板的便捷函数"""
    dashboard = PerformanceDashboard()
    
    # 如果有训练环境，更新数据
    if training_env:
        # 模拟从训练环境获取数据
        for i in range(10):  # 模拟10个时间点的数据
            metrics = {
                'avg_reward': getattr(training_env, 'episode_rewards', [0])[-1] if hasattr(training_env, 'episode_rewards') and training_env.episode_rewards else 0,
                'avg_task_delay': 0.5 + np.random.normal(0, 0.1),
                'total_energy_consumption': 100 + np.random.normal(0, 10),
                'cache_hit_rate': 0.8 + np.random.normal(0, 0.1),
                'task_completion_rate': 0.7 + np.random.normal(0, 0.1)
            }
            dashboard.update_real_time_metrics(metrics)
    
    return dashboard.create_comprehensive_dashboard(save_path)

def create_real_time_monitor(save_path: str = None):
    """创建实时监控界面的便捷函数"""
    dashboard = PerformanceDashboard()
    return dashboard.create_real_time_monitor(save_path)

if __name__ == "__main__":
    # 测试仪表板功能
    print("🧪 测试性能仪表板...")
    
    # 创建综合仪表板
    create_performance_dashboard(save_path="test_dashboard.png")
    
    # 创建实时监控
    create_real_time_monitor(save_path="test_realtime_monitor.png")
    
    print("✅ 性能仪表板测试完成")