#!/usr/bin/env python3
"""
对比算法结果可视化工具

可视化所有算法（启发式、元启发式、DRL）的性能对比
生成多种图表用于论文展示
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple, Optional
import pandas as pd
from datetime import datetime

# 设置中文支持
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 设置样式
sns.set_style("whitegrid")
sns.set_palette("husl")


class AlgorithmVisualizer:
    """算法结果可视化器"""
    
    def __init__(self, results_dir: str = None):
        """
        初始化可视化器
        
        Args:
            results_dir: 结果目录路径
        """
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
            
        self.algorithms = {
            # 启发式算法
            'Random': {'type': 'Heuristic', 'color': '#FF6B6B', 'marker': 'o'},
            'Greedy': {'type': 'Heuristic', 'color': '#4ECDC4', 'marker': 's'},
            'RoundRobin': {'type': 'Heuristic', 'color': '#45B7D1', 'marker': '^'},
            'LocalFirst': {'type': 'Heuristic', 'color': '#96CEB4', 'marker': 'v'},
            'NearestNode': {'type': 'Heuristic', 'color': '#FECA57', 'marker': '<'},
            'MinDelay': {'type': 'Heuristic', 'color': '#DDA0DD', 'marker': '>'},
            'MinEnergy': {'type': 'Heuristic', 'color': '#98D8C8', 'marker': 'p'},
            'LoadBalance': {'type': 'Heuristic', 'color': '#F7DC6F', 'marker': '*'},
            'HybridGreedy': {'type': 'Heuristic', 'color': '#BB8FCE', 'marker': 'h'},
            
            # 元启发式算法
            'GA': {'type': 'MetaHeuristic', 'color': '#FF7F50', 'marker': 'D'},
            'PSO': {'type': 'MetaHeuristic', 'color': '#32CD32', 'marker': 'X'},
            
            # DRL算法
            'DQN': {'type': 'DRL', 'color': '#FF1493', 'marker': 'o'},
            'DDPG': {'type': 'DRL', 'color': '#00CED1', 'marker': 's'},
            'TD3': {'type': 'DRL', 'color': '#FFD700', 'marker': '^'},
            'SAC': {'type': 'DRL', 'color': '#9370DB', 'marker': 'v'},
            'PPO': {'type': 'DRL', 'color': '#20B2AA', 'marker': 'p'}
        }
        
    def load_latest_results(self) -> Dict[str, Dict]:
        """加载所有算法的最新结果"""
        results = {}
        
        for algo_name in self.algorithms.keys():
            algo_dir = self.results_dir / algo_name.lower()
            latest_file = algo_dir / f"{algo_name.lower()}_latest.json"
            
            if latest_file.exists():
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        results[algo_name] = data
                        print(f"✓ 加载 {algo_name} 结果")
                except Exception as e:
                    print(f"✗ 加载 {algo_name} 失败: {e}")
            else:
                print(f"⚠ 未找到 {algo_name} 结果文件")
                
        return results
    
    def plot_performance_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """绘制性能对比图（柱状图）"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        algorithms = []
        delays = []
        energies = []
        rewards = []
        completion_rates = []
        colors = []
        
        # 提取数据
        for algo, data in results.items():
            algorithms.append(algo)
            delays.append(data.get('avg_delay', 0))
            energies.append(data.get('avg_energy', 0))
            rewards.append(data.get('final_reward', 0))
            completion_rates.append(data.get('avg_completion_rate', 0) * 100)
            colors.append(self.algorithms[algo]['color'])
        
        # 1. 平均时延
        bars1 = ax1.bar(algorithms, delays, color=colors, alpha=0.8)
        ax1.set_title('平均任务时延对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('时延 (秒)', fontsize=12)
        ax1.set_ylim(0, max(delays) * 1.2)
        self._add_value_labels(ax1, bars1)
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. 平均能耗
        bars2 = ax2.bar(algorithms, energies, color=colors, alpha=0.8)
        ax2.set_title('平均能耗对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('能耗 (焦耳)', fontsize=12)
        ax2.set_ylim(0, max(energies) * 1.2)
        self._add_value_labels(ax2, bars2, fmt='{:.0f}')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. 最终奖励
        bars3 = ax3.bar(algorithms, rewards, color=colors, alpha=0.8)
        ax3.set_title('最终奖励对比', fontsize=14, fontweight='bold')
        ax3.set_ylabel('奖励值', fontsize=12)
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        self._add_value_labels(ax3, bars3)
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. 任务完成率
        bars4 = ax4.bar(algorithms, completion_rates, color=colors, alpha=0.8)
        ax4.set_title('任务完成率对比', fontsize=14, fontweight='bold')
        ax4.set_ylabel('完成率 (%)', fontsize=12)
        ax4.set_ylim(0, 105)
        self._add_value_labels(ax4, bars4, fmt='{:.1f}%')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.suptitle('VEC系统算法性能对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 性能对比图已保存: {save_path}")
        plt.show()
        
    def plot_learning_curves(self, results: Dict[str, Dict], save_path: str = None):
        """绘制学习曲线（只适用于有episode数据的算法）"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        for algo, data in results.items():
            if 'episode_rewards' in data:
                episodes = range(1, len(data['episode_rewards']) + 1)
                algo_info = self.algorithms[algo]
                
                # 奖励曲线
                ax1.plot(episodes, data['episode_rewards'], 
                        label=algo, 
                        color=algo_info['color'],
                        marker=algo_info['marker'],
                        markevery=max(1, len(episodes)//20),
                        markersize=6,
                        alpha=0.8,
                        linewidth=2)
                
                # 时延曲线
                if 'episode_delays' in data:
                    ax2.plot(episodes, data['episode_delays'],
                            label=algo,
                            color=algo_info['color'],
                            marker=algo_info['marker'],
                            markevery=max(1, len(episodes)//20),
                            markersize=6,
                            alpha=0.8,
                            linewidth=2)
        
        ax1.set_xlabel('训练轮次', fontsize=12)
        ax1.set_ylabel('平均奖励', fontsize=12)
        ax1.set_title('奖励学习曲线', fontsize=14, fontweight='bold')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        ax2.set_xlabel('训练轮次', fontsize=12)
        ax2.set_ylabel('平均时延 (秒)', fontsize=12)
        ax2.set_title('时延变化曲线', fontsize=14, fontweight='bold')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('算法学习曲线对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 学习曲线图已保存: {save_path}")
        plt.show()
        
    def plot_algorithm_type_comparison(self, results: Dict[str, Dict], save_path: str = None):
        """按算法类型分组对比"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 按类型分组
        type_data = {'Heuristic': [], 'MetaHeuristic': [], 'DRL': []}
        
        for algo, data in results.items():
            algo_type = self.algorithms[algo]['type']
            type_data[algo_type].append({
                'name': algo,
                'delay': data.get('avg_delay', 0),
                'energy': data.get('avg_energy', 0),
                'reward': data.get('final_reward', 0),
                'completion': data.get('avg_completion_rate', 0) * 100
            })
        
        # 箱线图数据准备
        delay_data = []
        energy_data = []
        reward_data = []
        completion_data = []
        types = []
        
        for algo_type, algos in type_data.items():
            if algos:
                delays = [a['delay'] for a in algos]
                energies = [a['energy'] for a in algos]
                rewards = [a['reward'] for a in algos]
                completions = [a['completion'] for a in algos]
                
                delay_data.extend(delays)
                energy_data.extend(energies)
                reward_data.extend(rewards)
                completion_data.extend(completions)
                types.extend([algo_type] * len(algos))
        
        # 创建DataFrame
        df = pd.DataFrame({
            'Type': types,
            'Delay': delay_data,
            'Energy': energy_data,
            'Reward': reward_data,
            'Completion': completion_data
        })
        
        # 1. 时延箱线图
        sns.boxplot(x='Type', y='Delay', data=df, ax=ax1)
        ax1.set_title('不同类型算法时延分布', fontsize=14, fontweight='bold')
        ax1.set_ylabel('时延 (秒)', fontsize=12)
        
        # 2. 能耗箱线图
        sns.boxplot(x='Type', y='Energy', data=df, ax=ax2)
        ax2.set_title('不同类型算法能耗分布', fontsize=14, fontweight='bold')
        ax2.set_ylabel('能耗 (焦耳)', fontsize=12)
        
        # 3. 奖励箱线图
        sns.boxplot(x='Type', y='Reward', data=df, ax=ax3)
        ax3.set_title('不同类型算法奖励分布', fontsize=14, fontweight='bold')
        ax3.set_ylabel('奖励值', fontsize=12)
        
        # 4. 完成率箱线图
        sns.boxplot(x='Type', y='Completion', data=df, ax=ax4)
        ax4.set_title('不同类型算法完成率分布', fontsize=14, fontweight='bold')
        ax4.set_ylabel('完成率 (%)', fontsize=12)
        
        plt.suptitle('算法类型性能对比', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 算法类型对比图已保存: {save_path}")
        plt.show()
        
    def plot_radar_chart(self, results: Dict[str, Dict], algorithms: List[str] = None, 
                        save_path: str = None):
        """绘制雷达图（多维度性能对比）"""
        if algorithms is None:
            algorithms = list(results.keys())[:5]  # 默认显示前5个算法
            
        # 指标
        metrics = ['时延', '能耗', '完成率', '奖励', '稳定性']
        num_metrics = len(metrics)
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for algo in algorithms:
            if algo in results:
                data = results[algo]
                
                # 归一化数据（0-1范围）
                values = []
                values.append(1 - data.get('avg_delay', 1) / 2)  # 时延越小越好
                values.append(1 - data.get('avg_energy', 5000) / 10000)  # 能耗越小越好
                values.append(data.get('avg_completion_rate', 0))  # 完成率越高越好
                values.append((data.get('final_reward', -20) + 20) / 20)  # 奖励归一化
                values.append(1 - data.get('std_delay', 0.5) / 1)  # 稳定性（std越小越好）
                
                values = [max(0, min(1, v)) for v in values]  # 确保在0-1范围
                values += values[:1]
                
                # 绘制
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=algo, color=self.algorithms[algo]['color'])
                ax.fill(angles, values, alpha=0.1, color=self.algorithms[algo]['color'])
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics, fontsize=12)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=10)
        ax.grid(True)
        
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
        plt.title('算法多维度性能对比雷达图', fontsize=16, fontweight='bold', pad=20)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ 雷达图已保存: {save_path}")
        plt.show()
        
    def generate_latex_table(self, results: Dict[str, Dict], save_path: str = None):
        """生成LaTeX格式的表格（用于论文）"""
        latex_lines = []
        latex_lines.append("\\begin{table}[htbp]")
        latex_lines.append("\\centering")
        latex_lines.append("\\caption{VEC系统中不同算法的性能对比}")
        latex_lines.append("\\label{tab:algorithm_comparison}")
        latex_lines.append("\\begin{tabular}{lcccccc}")
        latex_lines.append("\\hline")
        latex_lines.append("算法 & 类型 & 平均时延(s) & 平均能耗(J) & 完成率(\\%) & 最终奖励 \\\\")
        latex_lines.append("\\hline")
        
        # 按类型排序
        sorted_algos = sorted(results.keys(), 
                            key=lambda x: (self.algorithms[x]['type'], x))
        
        current_type = None
        for algo in sorted_algos:
            data = results[algo]
            algo_type = self.algorithms[algo]['type']
            
            # 添加分隔线
            if current_type and current_type != algo_type:
                latex_lines.append("\\hline")
            current_type = algo_type
            
            # 提取数据
            delay = f"{data.get('avg_delay', 0):.3f}"
            energy = f"{data.get('avg_energy', 0):.1f}"
            completion = f"{data.get('avg_completion_rate', 0) * 100:.1f}"
            reward = f"{data.get('final_reward', 0):.3f}"
            
            latex_lines.append(f"{algo} & {algo_type} & {delay} & {energy} & {completion} & {reward} \\\\")
        
        latex_lines.append("\\hline")
        latex_lines.append("\\end{tabular}")
        latex_lines.append("\\end{table}")
        
        latex_table = '\n'.join(latex_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(latex_table)
            print(f"✓ LaTeX表格已保存: {save_path}")
        
        print("\nLaTeX表格代码:")
        print(latex_table)
        
        return latex_table
    
    def _add_value_labels(self, ax, bars, fmt='{:.3f}'):
        """在柱状图上添加数值标签"""
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)


def main():
    parser = argparse.ArgumentParser(description='可视化算法对比结果')
    parser.add_argument('--results-dir', type=str, default=None,
                       help='结果目录路径')
    parser.add_argument('--save-dir', type=str, default='./figures',
                       help='图表保存目录')
    parser.add_argument('--algorithms', nargs='+', default=None,
                       help='要显示的算法列表')
    parser.add_argument('--format', choices=['png', 'pdf', 'svg'], default='png',
                       help='图片保存格式')
    args = parser.parse_args()
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化器
    visualizer = AlgorithmVisualizer(args.results_dir)
    
    # 加载结果
    print("加载算法结果...")
    results = visualizer.load_latest_results()
    
    if not results:
        print("⚠️ 没有找到任何算法结果！")
        return
    
    print(f"\n找到 {len(results)} 个算法的结果")
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 性能对比柱状图
    print("\n生成性能对比图...")
    visualizer.plot_performance_comparison(
        results, 
        save_path=save_dir / f"performance_comparison_{timestamp}.{args.format}"
    )
    
    # 2. 学习曲线
    print("\n生成学习曲线图...")
    visualizer.plot_learning_curves(
        results,
        save_path=save_dir / f"learning_curves_{timestamp}.{args.format}"
    )
    
    # 3. 算法类型对比
    print("\n生成算法类型对比图...")
    visualizer.plot_algorithm_type_comparison(
        results,
        save_path=save_dir / f"type_comparison_{timestamp}.{args.format}"
    )
    
    # 4. 雷达图
    print("\n生成雷达图...")
    # 选择代表性算法
    selected_algos = ['TD3', 'GA', 'PSO', 'Greedy', 'HybridGreedy']
    available_algos = [a for a in selected_algos if a in results]
    visualizer.plot_radar_chart(
        results,
        algorithms=available_algos,
        save_path=save_dir / f"radar_chart_{timestamp}.{args.format}"
    )
    
    # 5. LaTeX表格
    print("\n生成LaTeX表格...")
    visualizer.generate_latex_table(
        results,
        save_path=save_dir / f"results_table_{timestamp}.tex"
    )
    
    print(f"\n✅ 所有图表已保存至: {save_dir}")


if __name__ == "__main__":
    main()










