#!/usr/bin/env python3
"""
简洁优美的可视化系统
只生成最核心、最有用的图表，避免冗余
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional
from datetime import datetime
import warnings

# 禁用matplotlib警告
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')

# 设置全局样式 - 修复中文字体问题
plt.style.use('default')
plt.rcParams.update({
    'font.size': 11,
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

# 禁用字体警告并使用英文标签
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib.font_manager')

# 定义现代化配色方案
COLORS = {
    'primary': '#2E86AB',      # 蓝色 - 主要数据
    'secondary': '#A23B72',    # 紫红 - 次要数据 
    'success': '#F18F01',      # 橙色 - 成功/改善
    'warning': '#C73E1D',      # 红色 - 警告/问题
    'neutral': '#6C757D',      # 灰色 - 中性
    'light': '#E9ECEF'         # 浅灰 - 背景
}

class ModernVisualizer:
    """现代化可视化器 - 简洁优美"""
    
    def __init__(self):
        self.style_applied = False
    
    def _apply_modern_style(self, ax, title: str = ""):
        """应用现代化样式"""
        if title:
            ax.set_title(title, fontsize=13, fontweight='600', pad=15)
        
        # 美化坐标轴
        ax.tick_params(colors='#555555', which='both')
        ax.spines['bottom'].set_color('#CCCCCC')
        ax.spines['left'].set_color('#CCCCCC')
        
        # 添加微妙的网格
        ax.grid(True, linestyle='--', alpha=0.4, color='#DDDDDD')
        ax.set_axisbelow(True)
    
    def plot_training_overview(self, training_env, algorithm: str, save_path: str):
        """绘制训练总览 - 核心指标一图展现"""
        
        # 创建2x3布局，包含核心目标指标
        fig, ((ax1, ax2, ax5), (ax3, ax4, ax6)) = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{algorithm} Training Overview', fontsize=16, fontweight='bold', y=0.95)
        
        # 1. 奖励收敛曲线（左上）- 🔧 修改为平均每步奖励
        episodes = range(1, len(training_env.episode_rewards) + 1)
        
        # 获取每episode的步数（默认200步）
        max_steps = getattr(training_env, 'max_steps_per_episode', 200)
        if hasattr(training_env, 'episode_steps'):
            # 如果有记录实际步数，使用实际步数
            step_counts = training_env.episode_steps
        else:
            # 使用配置的最大步数
            try:
                from config import config
                max_steps = config.experiment.max_steps_per_episode
            except:
                max_steps = 200
            step_counts = [max_steps] * len(training_env.episode_rewards)
        
        # 🔧 计算平均每步奖励
        avg_step_rewards = []
        for i, episode_reward in enumerate(training_env.episode_rewards):
            steps = step_counts[i] if i < len(step_counts) else max_steps
            avg_step_reward = episode_reward / max(1, steps)
            avg_step_rewards.append(avg_step_reward)
        
        # 原始平均每步奖励（淡色）
        ax1.plot(episodes, avg_step_rewards, 
                color=COLORS['neutral'], alpha=0.4, linewidth=1, label='Raw Avg Step Reward')
        
        # 移动平均（突出显示）
        if len(avg_step_rewards) > 10:
            window = max(5, len(avg_step_rewards) // 20)
            moving_avg = np.convolve(avg_step_rewards, 
                                   np.ones(window)/window, mode='valid')
            ax1.plot(range(window, len(episodes) + 1), moving_avg,
                    color=COLORS['primary'], linewidth=3, label=f'{window}-Episode Avg')
        
        self._apply_modern_style(ax1, 'Reward Convergence (Per Step)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Avg Reward per Step')
        ax1.legend(frameon=False)
        
        # 2. 系统性能指标（右上）
        metrics = ['task_completion_rate', 'avg_delay']
        metric_names = ['Completion Rate (%)', 'Avg Delay (s)']
        colors = [COLORS['success'], COLORS['warning']]
        
        for i, (metric, name, color) in enumerate(zip(metrics, metric_names, colors)):
            if training_env.episode_metrics.get(metric):
                ax2_twin = ax2 if i == 0 else ax2.twinx()
                
                data = training_env.episode_metrics[metric]
                if metric == 'task_completion_rate':
                    data = [x * 100 for x in data]  # 转为百分比
                
                ax2_twin.plot(episodes[:len(data)], data, 
                            color=color, linewidth=2.5, label=name)
                ax2_twin.set_ylabel(name, color=color)
                ax2_twin.tick_params(axis='y', labelcolor=color)
        
        self._apply_modern_style(ax2, 'System Performance')
        ax2.set_xlabel('Episode')
        
        # 3. 能耗与缓存效率（左下）
        if training_env.episode_metrics.get('total_energy'):
            # 归一化能耗到[0,1]用于显示
            energy_data = training_env.episode_metrics['total_energy']
            normalized_energy = [(x - min(energy_data)) / (max(energy_data) - min(energy_data)) 
                                for x in energy_data] if len(set(energy_data)) > 1 else [0.5] * len(energy_data)
            
            ax3.fill_between(episodes[:len(normalized_energy)], normalized_energy, 
                           alpha=0.6, color=COLORS['secondary'], label='Energy Trend')
        
        if training_env.episode_metrics.get('cache_hit_rate'):
            cache_data = training_env.episode_metrics['cache_hit_rate']
            ax3_twin = ax3.twinx()
            ax3_twin.plot(episodes[:len(cache_data)], 
                         [x * 100 for x in cache_data],
                         color=COLORS['primary'], linewidth=2.5, label='Cache Hit Rate (%)')
            ax3_twin.set_ylabel('Cache Hit Rate (%)', color=COLORS['primary'])
            ax3_twin.tick_params(axis='y', labelcolor=COLORS['primary'])
        
        self._apply_modern_style(ax3, 'Resource Efficiency')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Normalized Energy', color=COLORS['secondary'])
        ax3.tick_params(axis='y', labelcolor=COLORS['secondary'])
        
        # 4. 收敛性分析（右下）- 🔧 基于平均每步奖励方差
        if len(avg_step_rewards) > 20:
            # 滚动方差分析
            window_size = max(5, len(avg_step_rewards) // 4)
            rolling_var = []
            for i in range(window_size, len(avg_step_rewards)):
                window_data = avg_step_rewards[i-window_size:i]
                rolling_var.append(np.var(window_data))
            
            ax4.plot(range(window_size, len(avg_step_rewards)), rolling_var,
                    color=COLORS['warning'], linewidth=2)
            
            # 添加收敛阈值线（调整为适合平均每步奖励的范围）
            if rolling_var:
                convergence_threshold = np.mean(rolling_var)
                ax4.axhline(y=convergence_threshold, color=COLORS['neutral'], 
                           linestyle='--', alpha=0.7, label=f'Convergence Threshold: {convergence_threshold:.4f}')
        
        self._apply_modern_style(ax4, 'Convergence Stability')
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Reward Variance')
        if ax4.get_legend_handles_labels()[0]:
            ax4.legend(frameon=False)
        
        # 5. 核心目标指标：时延与能耗（右上）
        if training_env.episode_metrics.get('avg_delay') or training_env.episode_metrics.get('total_energy'):
            # 时延曲线（左轴）
            if training_env.episode_metrics.get('avg_delay'):
                delay_data = training_env.episode_metrics['avg_delay']
                ax5.plot(episodes[:len(delay_data)], delay_data, 
                        color=COLORS['warning'], linewidth=2.5, label='Avg Delay (s)')
                ax5.set_ylabel('Avg Delay (s)', color=COLORS['warning'])
                ax5.tick_params(axis='y', labelcolor=COLORS['warning'])
            
            # 能耗曲线（右轴）
            if training_env.episode_metrics.get('total_energy'):
                energy_data = training_env.episode_metrics['total_energy']
                ax5_twin = ax5.twinx()
                # 归一化能耗到合理显示范围
                max_energy = max(energy_data) if energy_data else 1000
                normalized_energy = [e / max_energy * 10 for e in energy_data]  # 缩放到0-10范围
                ax5_twin.plot(episodes[:len(normalized_energy)], normalized_energy,
                             color=COLORS['secondary'], linewidth=2.5, label='Energy (norm)')
                ax5_twin.set_ylabel('Normalized Energy', color=COLORS['secondary'])
                ax5_twin.tick_params(axis='y', labelcolor=COLORS['secondary'])
            
            self._apply_modern_style(ax5, 'Delay & Energy Trends')
            ax5.set_xlabel('Episode')
        else:
            ax5.text(0.5, 0.5, 'No Delay/Energy Data', ha='center', va='center', transform=ax5.transAxes)
            self._apply_modern_style(ax5, 'Delay & Energy Trends')
        
        # 6. 数据丢失率与迁移成功率（右下）
        if (training_env.episode_metrics.get('task_completion_rate') or 
            training_env.episode_metrics.get('migration_success_rate')):
            
            # 数据丢失率（从完成率推导）
            if training_env.episode_metrics.get('task_completion_rate'):
                completion_data = training_env.episode_metrics['task_completion_rate']
                loss_data = [(1.0 - c) * 100 for c in completion_data]  # 转为丢失率百分比
                ax6.plot(episodes[:len(loss_data)], loss_data,
                        color=COLORS['warning'], linewidth=2.5, label='Data Loss Rate (%)')
                ax6.set_ylabel('Data Loss Rate (%)', color=COLORS['warning'])
                ax6.tick_params(axis='y', labelcolor=COLORS['warning'])
            
            # 迁移成功率（右轴）
            if training_env.episode_metrics.get('migration_success_rate'):
                migration_data = training_env.episode_metrics['migration_success_rate']
                ax6_twin = ax6.twinx()
                migration_percent = [m * 100 for m in migration_data]
                ax6_twin.plot(episodes[:len(migration_percent)], migration_percent,
                             color=COLORS['success'], linewidth=2.5, label='Migration Success (%)')
                ax6_twin.set_ylabel('Migration Success (%)', color=COLORS['success'])
                ax6_twin.tick_params(axis='y', labelcolor=COLORS['success'])
            
            self._apply_modern_style(ax6, 'Loss Rate & Migration')
            ax6.set_xlabel('Episode')
        else:
            ax6.text(0.5, 0.5, 'No Loss/Migration Data', ha='center', va='center', transform=ax6.transAxes)
            self._apply_modern_style(ax6, 'Loss Rate & Migration')
        
        # 全局美化
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        
        # 保存高质量图片
        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 {algorithm} Training Overview with Core Metrics Saved: {save_path}")
    
    def plot_performance_summary(self, results_dict: Dict, save_path: str):
        """绘制性能对比总结 - 算法间对比"""
        
        if len(results_dict) < 2:
            print("⚠️ 算法数量不足，跳过对比图生成")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Algorithm Performance Comparison', fontsize=16, fontweight='bold', y=0.95)
        
        # 1. 最终性能对比（左图）- 🔧 使用平均每步奖励
        algorithms = list(results_dict.keys())
        final_rewards = []
        final_completions = []
        
        for alg in algorithms:
            perf = results_dict[alg].get('final_performance', {})
            episode_rewards = results_dict[alg].get('episode_rewards', [])
            
            # 🔧 计算平均每步奖励
            if episode_rewards:
                try:
                    from config import config
                    max_steps = config.experiment.max_steps_per_episode
                except:
                    max_steps = 200
                final_step_reward = episode_rewards[-1] / max_steps
            else:
                final_step_reward = perf.get('avg_reward', 0) / 200  # 回退方案
                
            final_rewards.append(final_step_reward)
            final_completions.append(perf.get('avg_completion', 0) * 100)
        
        x_pos = np.arange(len(algorithms))
        
        # 柱状图
        bars1 = ax1.bar(x_pos - 0.2, final_rewards, 0.4, 
                       color=COLORS['primary'], alpha=0.8, label='Avg Reward')
        
        ax1_twin = ax1.twinx()
        bars2 = ax1_twin.bar(x_pos + 0.2, final_completions, 0.4,
                            color=COLORS['success'], alpha=0.8, label='Completion Rate (%)')
        
        # 美化
        ax1.set_xlabel('Algorithm')
        ax1.set_ylabel('Avg Step Reward', color=COLORS['primary'])
        ax1_twin.set_ylabel('Completion Rate (%)', color=COLORS['success'])
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(algorithms)
        
        # 添加数值标签 - 🔧 调整格式适应平均每步奖励
        for bar, val in zip(bars1, final_rewards):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (abs(val) * 0.05),
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)
        
        for bar, val in zip(bars2, final_completions):
            ax1_twin.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        
        self._apply_modern_style(ax1, 'Final Performance Comparison')
        
        # 2. 收敛速度对比（右图）- 🔧 使用平均每步奖励
        for alg, result in results_dict.items():
            rewards = result.get('episode_rewards', [])
            if rewards:
                # 🔧 转换为平均每步奖励
                try:
                    from config import config
                    max_steps = config.experiment.max_steps_per_episode
                except:
                    max_steps = 200
                
                avg_step_rewards = [r / max_steps for r in rewards]
                
                # 归一化到[0,1]以便对比
                min_r, max_r = min(avg_step_rewards), max(avg_step_rewards)
                if max_r > min_r:
                    normalized = [(r - min_r) / (max_r - min_r) for r in avg_step_rewards]
                else:
                    normalized = [0.5] * len(avg_step_rewards)
                
                ax2.plot(range(1, len(normalized) + 1), normalized,
                        linewidth=2.5, label=alg, alpha=0.8)
        
        self._apply_modern_style(ax2, 'Convergence Speed Comparison')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Normalized Reward (0-1)')
        ax2.legend(frameon=False, loc='lower right')
        
        # 3. 时延对比（左下）
        delay_found = False
        for alg, result in results_dict.items():
            delay_data = result.get('episode_metrics', {}).get('avg_delay', [])
            if delay_data:
                ax3.plot(range(1, len(delay_data) + 1), delay_data,
                        linewidth=2.5, label=alg, alpha=0.8)
                delay_found = True
        
        if delay_found:
            self._apply_modern_style(ax3, 'Average Delay Comparison')
            ax3.set_xlabel('Episode')
            ax3.set_ylabel('Avg Delay (s)')
            ax3.legend(frameon=False)
        else:
            ax3.text(0.5, 0.5, 'No Delay Data', ha='center', va='center', transform=ax3.transAxes)
            self._apply_modern_style(ax3, 'Average Delay Comparison')
        
        # 4. 能耗与数据丢失率对比（右下）
        energy_found = False
        loss_found = False
        
        # 能耗对比（左轴）
        for alg, result in results_dict.items():
            energy_data = result.get('episode_metrics', {}).get('total_energy', [])
            if energy_data:
                # 归一化能耗便于显示
                max_energy = max(energy_data) if energy_data else 1000
                normalized_energy = [e / max_energy for e in energy_data]
                ax4.plot(range(1, len(normalized_energy) + 1), normalized_energy,
                        linewidth=2.5, label=f'{alg} Energy', alpha=0.8)
                energy_found = True
        
        # 数据丢失率对比（右轴）
        if energy_found:
            ax4_twin = ax4.twinx()
        else:
            ax4_twin = ax4
            
        for alg, result in results_dict.items():
            completion_data = result.get('episode_metrics', {}).get('task_completion_rate', [])
            if completion_data:
                loss_data = [(1.0 - c) * 100 for c in completion_data]
                ax4_twin.plot(range(1, len(loss_data) + 1), loss_data,
                            linewidth=2.5, label=f'{alg} Loss Rate (%)', 
                            linestyle='--', alpha=0.8)
                loss_found = True
        
        if energy_found or loss_found:
            self._apply_modern_style(ax4, 'Energy & Loss Rate')
            ax4.set_xlabel('Episode')
            if energy_found:
                ax4.set_ylabel('Normalized Energy', color=COLORS['secondary'])
                ax4.tick_params(axis='y', labelcolor=COLORS['secondary'])
            if loss_found:
                ax4_twin.set_ylabel('Data Loss Rate (%)', color=COLORS['warning'])
                ax4_twin.tick_params(axis='y', labelcolor=COLORS['warning'])
            
            # 合并图例
            lines1, labels1 = ax4.get_legend_handles_labels()
            lines2, labels2 = ax4_twin.get_legend_handles_labels()
            ax4.legend(lines1 + lines2, labels1 + labels2, frameon=False, loc='upper right')
        else:
            ax4.text(0.5, 0.5, 'No Energy/Loss Data', ha='center', va='center', transform=ax4.transAxes)
            self._apply_modern_style(ax4, 'Energy & Loss Rate')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        plt.savefig(save_path, dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"📊 Performance Comparison with Core Metrics Saved: {save_path}")


# 全局可视化器实例
_visualizer = ModernVisualizer()

def create_training_chart(training_env, algorithm: str, save_path: str):
    """创建训练图表 - 统一入口"""
    _visualizer.plot_training_overview(training_env, algorithm, save_path)

def create_comparison_chart(results_dict: Dict, save_path: str):
    """创建对比图表 - 统一入口"""
    _visualizer.plot_performance_summary(results_dict, save_path)

def plot_objective_function_breakdown(training_env, algorithm: str, save_path: str):
    """绘制目标函数分解图 - 显示 ω_T×delay + ω_E×energy + ω_D×loss 的变化"""
    
    # 计算目标函数各组成部分
    episodes = range(1, len(training_env.episode_rewards) + 1)
    
    # 获取权重
    try:
        from config import config
        w_delay = config.rl.reward_weight_delay      # ω_T
        w_energy = config.rl.reward_weight_energy    # ω_E  
        w_loss = config.rl.reward_weight_loss        # ω_D
    except:
        w_delay, w_energy, w_loss = 0.4, 0.3, 0.3
    
    # 计算各组成部分
    delay_components = []
    energy_components = []
    loss_components = []
    total_objectives = []
    
    for i in episodes:
        idx = i - 1
        if idx < len(training_env.episode_metrics.get('avg_delay', [])):
            delay = training_env.episode_metrics['avg_delay'][idx]
            delay_norm = delay / 1.0  # 归一化
            delay_component = w_delay * delay_norm
            delay_components.append(delay_component)
        
        if idx < len(training_env.episode_metrics.get('total_energy', [])):
            energy = training_env.episode_metrics['total_energy'][idx]
            energy_norm = energy / 1000.0  # 归一化
            energy_component = w_energy * energy_norm
            energy_components.append(energy_component)
        
        if idx < len(training_env.episode_metrics.get('task_completion_rate', [])):
            completion = training_env.episode_metrics['task_completion_rate'][idx]
            loss_rate = 1.0 - completion
            loss_component = w_loss * loss_rate
            loss_components.append(loss_component)
        
        # 计算总目标函数值
        if delay_components and energy_components and loss_components:
            total_obj = delay_components[-1] + energy_components[-1] + loss_components[-1]
            total_objectives.append(total_obj)
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(f'{algorithm} Objective Function Analysis', fontsize=16, fontweight='bold')
    
    # 左图：组成部分分解
    if delay_components:
        ax1.plot(episodes[:len(delay_components)], delay_components,
                color=COLORS['warning'], linewidth=2.5, label=f'ω_T × Delay ({w_delay})')
    if energy_components:
        ax1.plot(episodes[:len(energy_components)], energy_components,
                color=COLORS['secondary'], linewidth=2.5, label=f'ω_E × Energy ({w_energy})')
    if loss_components:
        ax1.plot(episodes[:len(loss_components)], loss_components,
                color=COLORS['primary'], linewidth=2.5, label=f'ω_D × Loss ({w_loss})')
    
    ax1.set_title('Objective Function Components')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Component Value')
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)
    
    # 右图：总目标函数与奖励对比
    if total_objectives:
        ax2.plot(episodes[:len(total_objectives)], total_objectives,
                color=COLORS['warning'], linewidth=3, label='Objective (to minimize)')
        
        # 对应的奖励（负目标函数）
        rewards_from_obj = [-obj for obj in total_objectives]
        ax2_twin = ax2.twinx()
        ax2_twin.plot(episodes[:len(rewards_from_obj)], rewards_from_obj,
                     color=COLORS['success'], linewidth=3, label='Reward (-Objective)')
        
        ax2.set_ylabel('Objective Value', color=COLORS['warning'])
        ax2_twin.set_ylabel('Reward Value', color=COLORS['success'])
        ax2.tick_params(axis='y', labelcolor=COLORS['warning'])
        ax2_twin.tick_params(axis='y', labelcolor=COLORS['success'])
        
        # 合并图例
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, frameon=False)
    
    ax2.set_title('Objective vs Reward')
    ax2.set_xlabel('Episode')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"📊 {algorithm} Objective Function Analysis Saved: {save_path}")


def cleanup_old_charts(algorithm_dir: str):
    """清理旧的冗余图表"""
    import os
    import glob
    
    # 要删除的冗余图表
    patterns_to_remove = [
        f"{algorithm_dir}/enhanced_training_curves.png",
        f"{algorithm_dir}/convergence_analysis.png", 
        f"{algorithm_dir}/multi_metric_dashboard.png",
        f"{algorithm_dir}/performance_dashboard.png",
        f"{algorithm_dir}/realtime_monitor.png"
    ]
    
    removed_count = 0
    for pattern in patterns_to_remove:
        for file_path in glob.glob(pattern):
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError:
                pass
    
    if removed_count > 0:
        print(f"🧹 Cleaned up {removed_count} redundant charts")

def get_summary_text(training_env, algorithm: str) -> str:
    """生成训练总结文本 - 使用平均每步奖励"""
    if not training_env.episode_rewards:
        return "No training data available"
    
    # 🔧 计算平均每步奖励
    try:
        from config import config
        max_steps = config.experiment.max_steps_per_episode
    except:
        max_steps = 200
    
    # 转换为平均每步奖励
    avg_step_rewards = [reward / max_steps for reward in training_env.episode_rewards]
    
    final_step_reward = avg_step_rewards[-1]
    max_step_reward = max(avg_step_rewards)
    step_improvement = final_step_reward - avg_step_rewards[0] if len(avg_step_rewards) > 1 else 0
    
    # 计算收敛状态（基于平均每步奖励方差）
    if len(avg_step_rewards) > 20:
        recent_var = np.var(avg_step_rewards[-20:])
        convergence_status = "Converged" if recent_var < 2 else "Converging" if recent_var < 10 else "Exploring"
    else:
        convergence_status = "Insufficient Data"
    
    summary = f"""
{algorithm} Training Summary (Per-Step Rewards)
═══════════════════════════════════════
Episodes: {len(training_env.episode_rewards)}
Final Avg Step Reward: {final_step_reward:.3f}
Best Avg Step Reward: {max_step_reward:.3f}
Step Reward Improvement: {step_improvement:+.3f}
Status: {convergence_status}
Steps per Episode: {max_steps}
    """
    
    return summary.strip()
