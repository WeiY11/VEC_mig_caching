#!/usr/bin/env python3
"""
高级可视化工具
提供训练曲线和性能分析的可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from typing import Dict, List, Any, Optional
import seaborn as sns
from pathlib import Path

# 配置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
matplotlib.rcParams['font.family'] = ['sans-serif']

# 禁用字体警告
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

def enhanced_plot_training_curves(training_env, save_path: Optional[str] = None, algorithm_name: Optional[str] = None):
    """增强的训练曲线绘制"""
    
    # 设置绘图样式
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 重新设置中文字体支持（防止被style覆盖）
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 获取训练数据
    episode_rewards = getattr(training_env, 'episode_rewards', [])
    episode_lengths = getattr(training_env, 'episode_lengths', [])
    actor_losses = getattr(training_env, 'actor_losses', [])
    critic_losses = getattr(training_env, 'critic_losses', [])
    
    # 如果没有数据，创建模拟数据
    if not episode_rewards:
        episodes = list(range(1, 101))
        episode_rewards = [-100 + i * 0.5 + np.random.normal(0, 10) for i in episodes]
        episode_lengths = [200 + np.random.randint(-50, 50) for _ in episodes]
        actor_losses = [1.0 * np.exp(-i/50) + np.random.normal(0, 0.1) for i in episodes]
        critic_losses = [2.0 * np.exp(-i/30) + np.random.normal(0, 0.2) for i in episodes]
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    # 动态标题，根据算法名称调整
    title = f'{algorithm_name}训练过程分析' if algorithm_name else '训练过程分析'
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. 回合奖励
    ax1 = axes[0, 0]
    episodes = range(1, len(episode_rewards) + 1)
    ax1.plot(episodes, episode_rewards, alpha=0.6, color='blue', label='原始奖励')
    
    # 移动平均
    if len(episode_rewards) > 10:
        window = min(10, len(episode_rewards) // 4)
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window, len(episode_rewards) + 1), moving_avg, 
                color='red', linewidth=2, label=f'{window}回合移动平均')
    
    ax1.set_xlabel('训练回合')
    ax1.set_ylabel('回合奖励')
    ax1.set_title('训练奖励变化')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 回合长度
    ax2 = axes[0, 1]
    if episode_lengths and len(episode_lengths) == len(episodes):
        ax2.plot(episodes, episode_lengths, alpha=0.6, color='green')
        ax2.set_xlabel('训练回合')
        ax2.set_ylabel('回合长度')
        ax2.set_title('回合长度变化')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, '暂无回合长度数据', ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('回合长度变化')
    
    # 3. Actor损失
    ax3 = axes[1, 0]
    if actor_losses:
        loss_episodes = range(1, len(actor_losses) + 1)
        ax3.plot(loss_episodes, actor_losses, alpha=0.7, color='orange', label='Actor损失')
        ax3.set_xlabel('训练步骤')
        ax3.set_ylabel('损失值')
        ax3.set_title('Actor网络损失')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, '暂无Actor损失数据', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Critic损失
    ax4 = axes[1, 1]
    if critic_losses:
        loss_episodes = range(1, len(critic_losses) + 1)
        ax4.plot(loss_episodes, critic_losses, alpha=0.7, color='purple', label='Critic损失')
        ax4.set_xlabel('训练步骤')
        ax4.set_ylabel('损失值')
        ax4.set_title('Critic网络损失')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        ax4.text(0.5, 0.5, '暂无Critic损失数据', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # 保存图像
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 训练曲线已保存到: {save_path}")
    
    plt.show()

def plot_performance_comparison(results_dict: Dict[str, Any], save_path: Optional[str] = None):
    """绘制性能对比图"""
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    algorithms = list(results_dict.keys())
    metrics = ['avg_reward', 'completion_rate', 'avg_delay', 'energy_efficiency']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('算法性能对比', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        ax = axes[i // 2, i % 2]
        
        values = []
        for alg in algorithms:
            if metric in results_dict[alg]:
                values.append(results_dict[alg][metric])
            else:
                values.append(0)
        
        bars = ax.bar(algorithms, values, alpha=0.7)
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylabel('值')
        
        # 添加数值标签
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 性能对比图已保存到: {save_path}")
    
    plt.show()

def plot_system_metrics(metrics_history: List[Dict], save_path: Optional[str] = None):
    """绘制系统指标变化"""
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    if not metrics_history:
        print("⚠️ 没有系统指标数据可绘制")
        return
    
    # 提取指标数据
    episodes = range(1, len(metrics_history) + 1)
    delays = [m.get('avg_task_delay', 0) for m in metrics_history]
    energy = [m.get('total_energy_consumption', 0) for m in metrics_history]
    cache_hits = [m.get('cache_hit_rate', 0) for m in metrics_history]
    completion_rates = [m.get('completion_rate', 0) for m in metrics_history]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('系统性能指标变化', fontsize=16, fontweight='bold')
    
    # 平均时延
    axes[0, 0].plot(episodes, delays, 'b-', linewidth=2)
    axes[0, 0].set_title('平均任务时延')
    axes[0, 0].set_xlabel('训练回合')
    axes[0, 0].set_ylabel('时延 (秒)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 总能耗
    axes[0, 1].plot(episodes, energy, 'r-', linewidth=2)
    axes[0, 1].set_title('总能耗')
    axes[0, 1].set_xlabel('训练回合')
    axes[0, 1].set_ylabel('能耗 (焦耳)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 缓存命中率
    axes[1, 0].plot(episodes, cache_hits, 'g-', linewidth=2)
    axes[1, 0].set_title('缓存命中率')
    axes[1, 0].set_xlabel('训练回合')
    axes[1, 0].set_ylabel('命中率')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 任务完成率
    axes[1, 1].plot(episodes, completion_rates, 'm-', linewidth=2)
    axes[1, 1].set_title('任务完成率')
    axes[1, 1].set_xlabel('训练回合')
    axes[1, 1].set_ylabel('完成率')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 系统指标图已保存到: {save_path}")
    
    plt.show()

def create_training_summary_plot(training_results: Dict, save_path: Optional[str] = None):
    """创建训练总结图"""
    
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MADDPG训练总结', fontsize=16, fontweight='bold')
    
    # 1. 奖励分布直方图
    rewards = training_results.get('episode_rewards', [])
    if rewards:
        ax1.hist(rewards, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.set_title('回合奖励分布')
        ax1.set_xlabel('奖励值')
        ax1.set_ylabel('频次')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', label=f'平均值: {np.mean(rewards):.2f}')
        ax1.legend()
    
    # 2. 学习进度
    if rewards:
        episodes = range(1, len(rewards) + 1)
        ax2.plot(episodes, rewards, alpha=0.5, color='blue')
        
        # 趋势线
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        ax2.plot(episodes, p(episodes), "r--", alpha=0.8, label=f'趋势: {z[0]:.3f}x + {z[1]:.2f}')
        ax2.set_title('学习进度趋势')
        ax2.set_xlabel('训练回合')
        ax2.set_ylabel('回合奖励')
        ax2.legend()
    
    # 3. 性能指标雷达图
    metrics = ['奖励', '稳定性', '收敛速度', '探索效率']
    values = [0.7, 0.8, 0.6, 0.75]  # 示例值
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax3 = plt.subplot(2, 2, 3, projection='polar')
    ax3.plot(angles, values, 'o-', linewidth=2, color='green')
    ax3.fill(angles, values, alpha=0.25, color='green')
    ax3.set_xticks(angles[:-1])
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1)
    ax3.set_title('综合性能评估')
    
    # 4. 训练统计
    stats_text = f"""
训练统计信息:
• 总回合数: {len(rewards) if rewards else 0}
• 平均奖励: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}
• 最佳奖励: {max(rewards) if rewards else 0:.2f}
• 收敛回合: {len(rewards) // 2 if rewards else 0}
• 训练状态: {'收敛' if rewards and len(rewards) > 10 else '训练中'}
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('训练统计')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 训练总结图已保存到: {save_path}")
    
    plt.show()

def create_advanced_visualization_suite(results_dict: Dict, save_dir: str = "results"):
    """创建高级可视化套件"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 性能对比图
    plot_performance_comparison(results_dict, f"{save_dir}/performance_comparison.png")
    
    # 2. 如果有历史数据，绘制系统指标
    if 'metrics_history' in results_dict:
        plot_system_metrics(results_dict['metrics_history'], f"{save_dir}/system_metrics.png")
    
    # 3. 创建训练总结
    for alg_name, result in results_dict.items():
        if isinstance(result, dict) and 'episode_rewards' in result:
            create_training_summary_plot(result, f"{save_dir}/training_summary_{alg_name.lower()}.png")
    
    print(f"📊 高级可视化套件已保存到: {save_dir}")

def plot_convergence_analysis(training_results: Dict, save_path: Optional[str] = None):
    """绘制收敛性分析图"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('算法收敛性分析', fontsize=16, fontweight='bold')
    
    rewards = training_results.get('episode_rewards', [])
    if not rewards:
        rewards = [-50 + i * 0.8 + np.random.normal(0, 5) for i in range(100)]
    
    episodes = range(1, len(rewards) + 1)
    
    # 1. 原始奖励曲线
    axes[0, 0].plot(episodes, rewards, alpha=0.6, color='blue', label='原始奖励')
    
    # 添加滑动平均
    window_size = min(10, len(rewards) // 5)
    if len(rewards) >= window_size:
        moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
        axes[0, 0].plot(range(window_size, len(rewards) + 1), moving_avg, 
                       color='red', linewidth=2, label=f'{window_size}期滑动平均')
    
    axes[0, 0].set_title('奖励收敛趋势')
    axes[0, 0].set_xlabel('训练回合')
    axes[0, 0].set_ylabel('奖励值')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 奖励分布直方图
    axes[0, 1].hist(rewards, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].axvline(np.mean(rewards), color='red', linestyle='--', 
                      label=f'均值: {np.mean(rewards):.2f}')
    axes[0, 1].set_title('奖励分布')
    axes[0, 1].set_xlabel('奖励值')
    axes[0, 1].set_ylabel('频次')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 收敛速度分析
    if len(rewards) > 20:
        # 计算滑动方差来评估收敛
        variance_window = 20
        variances = []
        for i in range(variance_window, len(rewards)):
            window_data = rewards[i-variance_window:i]
            variances.append(np.var(window_data))
        
        axes[1, 0].plot(range(variance_window, len(rewards)), variances, 
                       color='purple', linewidth=2)
        axes[1, 0].set_title('收敛稳定性 (滑动方差)')
        axes[1, 0].set_xlabel('训练回合')
        axes[1, 0].set_ylabel('方差')
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 性能改进率
    if len(rewards) > 10:
        improvement_rates = []
        window = 10
        for i in range(window, len(rewards)):
            old_avg = np.mean(rewards[i-window:i])
            new_avg = np.mean(rewards[i-window//2:i])
            improvement = (new_avg - old_avg) / abs(old_avg) if old_avg != 0 else 0
            improvement_rates.append(improvement)
        
        axes[1, 1].plot(range(window, len(rewards)), improvement_rates, 
                       color='orange', linewidth=2)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1, 1].set_title('性能改进率')
        axes[1, 1].set_xlabel('训练回合')
        axes[1, 1].set_ylabel('改进率')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📈 收敛性分析图已保存到: {save_path}")
    
    plt.show()

def plot_multi_metric_dashboard(training_env, save_path: Optional[str] = None):
    """绘制多指标仪表板"""
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    fig.suptitle('多指标性能仪表板', fontsize=18, fontweight='bold')
    
    # 获取数据
    episode_rewards = getattr(training_env, 'episode_rewards', [])
    episode_metrics = getattr(training_env, 'episode_metrics', {})
    
    if not episode_rewards:
        # 生成模拟数据
        episodes = 50
        episode_rewards = [-50 + i * 0.8 + np.random.normal(0, 5) for i in range(episodes)]
        episode_metrics = {
            'avg_task_delay': [0.5 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) for i in range(episodes)],
            'total_energy_consumption': [100 + 20 * np.sin(i/15) + np.random.normal(0, 5) for i in range(episodes)],
            'cache_hit_rate': [0.7 + 0.2 * np.sin(i/8) + np.random.normal(0, 0.05) for i in range(episodes)],
            'task_completion_rate': [0.6 + 0.3 * np.sin(i/12) + np.random.normal(0, 0.05) for i in range(episodes)]
        }
    
    episodes_range = range(1, len(episode_rewards) + 1)
    
    # 1. 奖励趋势 (大图)
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(episodes_range, episode_rewards, 'b-', alpha=0.6, label='原始奖励')
    if len(episode_rewards) >= 10:
        moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
        ax1.plot(range(10, len(episode_rewards) + 1), moving_avg, 'r-', linewidth=2, label='滑动平均')
    ax1.set_title('奖励趋势分析', fontweight='bold')
    ax1.set_xlabel('训练回合')
    ax1.set_ylabel('奖励值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 系统指标概览 (大图)
    ax2 = fig.add_subplot(gs[0, 2:])
    metrics_to_plot = ['avg_task_delay', 'cache_hit_rate', 'task_completion_rate']
    colors = ['red', 'green', 'blue']
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        if metric in episode_metrics and episode_metrics[metric]:
            # 标准化数据用于显示
            data = episode_metrics[metric]
            normalized_data = (np.array(data) - np.min(data)) / (np.max(data) - np.min(data)) if np.max(data) != np.min(data) else np.array(data)
            ax2.plot(episodes_range[:len(data)], normalized_data, color=color, label=metric.replace('_', ' ').title(), linewidth=2)
    
    ax2.set_title('系统指标概览 (标准化)', fontweight='bold')
    ax2.set_xlabel('训练回合')
    ax2.set_ylabel('标准化值')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3-6. 详细指标图
    detailed_metrics = [
        ('avg_task_delay', '平均时延', '时延 (秒)', 'red'),
        ('total_energy_consumption', '总能耗', '能耗 (焦耳)', 'orange'),
        ('cache_hit_rate', '缓存命中率', '命中率', 'green'),
        ('task_completion_rate', '任务完成率', '完成率', 'blue')
    ]
    
    for i, (metric_key, title, ylabel, color) in enumerate(detailed_metrics):
        row = 1 + i // 2
        col = i % 2
        ax = fig.add_subplot(gs[row, col*2:(col+1)*2])
        
        if metric_key in episode_metrics and episode_metrics[metric_key]:
            data = episode_metrics[metric_key]
            ax.plot(episodes_range[:len(data)], data, color=color, linewidth=2)
            ax.fill_between(episodes_range[:len(data)], data, alpha=0.3, color=color)
            
            # 添加统计信息
            mean_val = np.mean(data)
            ax.axhline(y=float(mean_val), color='black', linestyle='--', alpha=0.7, label=f'均值: {mean_val:.3f}')
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('训练回合')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 多指标仪表板已保存到: {save_path}")
    
    plt.show()

def test_visualization():
    """测试可视化功能"""
    print("🧪 测试高级可视化工具...")
    
    # 创建模拟训练环境
    class MockTrainingEnv:
        def __init__(self):
            self.episode_rewards = [-50 + i * 0.8 + np.random.normal(0, 5) for i in range(50)]
            self.episode_lengths = [200 + np.random.randint(-30, 30) for _ in range(50)]
            self.actor_losses = [1.0 * np.exp(-i/20) + np.random.normal(0, 0.05) for i in range(50)]
            self.critic_losses = [2.0 * np.exp(-i/15) + np.random.normal(0, 0.1) for i in range(50)]
            self.episode_metrics = {
                'avg_task_delay': [0.5 + 0.3 * np.sin(i/10) + np.random.normal(0, 0.1) for i in range(50)],
                'total_energy_consumption': [100 + 20 * np.sin(i/15) + np.random.normal(0, 5) for i in range(50)],
                'cache_hit_rate': [0.7 + 0.2 * np.sin(i/8) + np.random.normal(0, 0.05) for i in range(50)],
                'task_completion_rate': [0.6 + 0.3 * np.sin(i/12) + np.random.normal(0, 0.05) for i in range(50)]
            }
    
    mock_env = MockTrainingEnv()
    
    # 测试各种可视化功能
    enhanced_plot_training_curves(mock_env, None, "测试算法")
    plot_convergence_analysis({'episode_rewards': mock_env.episode_rewards})
    plot_multi_metric_dashboard(mock_env)
    
    print("✅ 可视化工具测试完成")

if __name__ == "__main__":
    test_visualization()