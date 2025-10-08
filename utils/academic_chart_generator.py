"""
学术论文图表生成器
生成符合IEEE/ACM标准的高质量学术图表
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from typing import Dict, List, Optional, Tuple
import seaborn as sns
from scipy import stats
import pandas as pd

# 设置学术风格
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 13
plt.rcParams['axes.unicode_minus'] = False

# IEEE标准配色（色盲友好）
ACADEMIC_COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#D55E00',
    'purple': '#CC78BC',
    'brown': '#CA9161',
    'pink': '#FBAFE4',
    'gray': '#949494',
    'yellow': '#ECE133',
    'cyan': '#56B4E9'
}

class AcademicChartGenerator:
    """学术图表生成器"""
    
    def __init__(self, dpi: int = 300, style: str = 'ieee'):
        """
        初始化学术图表生成器
        
        Args:
            dpi: 图表分辨率（默认300 DPI，适合论文）
            style: 图表风格（'ieee', 'acm', 'springer'）
        """
        self.dpi = dpi
        self.style = style
        self.colors = list(ACADEMIC_COLORS.values())
    
    def generate_convergence_comparison(self, 
                                       algorithms_data: Dict[str, List[float]], 
                                       save_path: str,
                                       title: str = "Algorithm Convergence Comparison",
                                       xlabel: str = "Episode",
                                       ylabel: str = "Average Reward",
                                       confidence_interval: bool = True,
                                       window_size: int = 20) -> str:
        """
        生成算法收敛性对比图
        
        Args:
            algorithms_data: {算法名: [奖励列表]}
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            confidence_interval: 是否显示置信区间
            window_size: 移动平均窗口大小
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        
        for i, (algo_name, rewards) in enumerate(algorithms_data.items()):
            episodes = np.arange(1, len(rewards) + 1)
            color = self.colors[i % len(self.colors)]
            
            # 移动平均
            if len(rewards) >= window_size:
                moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
                episodes_ma = episodes[window_size-1:]
                
                # 主线
                ax.plot(episodes_ma, moving_avg, label=algo_name, 
                       color=color, linewidth=2)
                
                # 置信区间
                if confidence_interval and len(rewards) >= window_size * 2:
                    moving_std = []
                    for j in range(len(moving_avg)):
                        window_data = rewards[j:j+window_size]
                        moving_std.append(np.std(window_data))
                    moving_std = np.array(moving_std)
                    
                    ax.fill_between(episodes_ma, 
                                   moving_avg - moving_std, 
                                   moving_avg + moving_std,
                                   alpha=0.2, color=color)
            else:
                ax.plot(episodes, rewards, label=algo_name, 
                       color=color, linewidth=2, alpha=0.7)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_boxplot_comparison(self,
                                    algorithms_data: Dict[str, List[float]],
                                    save_path: str,
                                    title: str = "Algorithm Performance Distribution",
                                    ylabel: str = "Average Reward",
                                    show_means: bool = True) -> str:
        """
        生成箱线图对比（展示分布和统计特性）
        
        Args:
            algorithms_data: {算法名: [奖励列表]}
            save_path: 保存路径
            title: 图表标题
            ylabel: Y轴标签
            show_means: 是否显示均值
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        data = [rewards for rewards in algorithms_data.values()]
        labels = list(algorithms_data.keys())
        
        # 创建箱线图
        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                       showmeans=show_means, meanline=True,
                       medianprops={'color': 'red', 'linewidth': 2},
                       meanprops={'color': 'blue', 'linewidth': 2, 'linestyle': '--'})
        
        # 上色
        for patch, color in zip(bp['boxes'], self.colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 添加统计信息
        for i, (algo_name, rewards) in enumerate(algorithms_data.items()):
            mean_val = np.mean(rewards)
            median_val = np.median(rewards)
            ax.text(i+1, mean_val, f'μ={mean_val:.2f}', 
                   ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_cdf_plot(self,
                         algorithms_data: Dict[str, List[float]],
                         save_path: str,
                         title: str = "Cumulative Distribution Function",
                         xlabel: str = "Average Reward",
                         ylabel: str = "CDF") -> str:
        """
        生成累积分布函数（CDF）图
        
        Args:
            algorithms_data: {算法名: [奖励列表]}
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 5), dpi=self.dpi)
        
        for i, (algo_name, rewards) in enumerate(algorithms_data.items()):
            sorted_data = np.sort(rewards)
            cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            
            ax.plot(sorted_data, cdf, label=algo_name,
                   color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_violin_plot(self,
                            algorithms_data: Dict[str, List[float]],
                            save_path: str,
                            title: str = "Algorithm Performance Distribution (Violin Plot)",
                            ylabel: str = "Average Reward") -> str:
        """
        生成小提琴图（展示分布密度）
        
        Args:
            algorithms_data: {算法名: [奖励列表]}
            save_path: 保存路径
            title: 图表标题
            ylabel: Y轴标签
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # 准备数据
        data_list = []
        labels = []
        for algo_name, rewards in algorithms_data.items():
            data_list.extend(rewards)
            labels.extend([algo_name] * len(rewards))
        
        df = pd.DataFrame({'Algorithm': labels, 'Reward': data_list})
        
        # 创建小提琴图
        parts = ax.violinplot([algorithms_data[algo] for algo in algorithms_data.keys()],
                              positions=range(len(algorithms_data)),
                              showmeans=True, showmedians=True)
        
        # 上色
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(self.colors[i % len(self.colors)])
            pc.set_alpha(0.7)
        
        ax.set_xticks(range(len(algorithms_data)))
        ax.set_xticklabels(algorithms_data.keys())
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_heatmap(self,
                        correlation_matrix: np.ndarray,
                        labels: List[str],
                        save_path: str,
                        title: str = "Metric Correlation Heatmap",
                        cmap: str = 'coolwarm') -> str:
        """
        生成相关性热力图
        
        Args:
            correlation_matrix: 相关性矩阵
            labels: 指标标签
            save_path: 保存路径
            title: 图表标题
            cmap: 颜色映射
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 8), dpi=self.dpi)
        
        im = ax.imshow(correlation_matrix, cmap=cmap, vmin=-1, vmax=1)
        
        # 设置刻度
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        
        # 旋转X轴标签
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # 添加数值标注
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)
        
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_bar_comparison(self,
                               algorithms_data: Dict[str, float],
                               save_path: str,
                               title: str = "Algorithm Performance Comparison",
                               ylabel: str = "Average Reward",
                               error_bars: Optional[Dict[str, float]] = None) -> str:
        """
        生成柱状图对比（含误差棒）
        
        Args:
            algorithms_data: {算法名: 平均值}
            save_path: 保存路径
            title: 图表标题
            ylabel: Y轴标签
            error_bars: {算法名: 标准差}（可选）
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        algorithms = list(algorithms_data.keys())
        values = list(algorithms_data.values())
        x_pos = np.arange(len(algorithms))
        
        # 创建柱状图
        bars = ax.bar(x_pos, values, color=self.colors[:len(algorithms)], 
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # 添加误差棒
        if error_bars:
            errors = [error_bars.get(algo, 0) for algo in algorithms]
            ax.errorbar(x_pos, values, yerr=errors, fmt='none', 
                       ecolor='black', capsize=5, capthick=2)
        
        # 添加数值标注
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(algorithms)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_radar_chart(self,
                            algorithms_data: Dict[str, Dict[str, float]],
                            save_path: str,
                            title: str = "Multi-dimensional Performance Comparison") -> str:
        """
        生成雷达图（多维度性能对比）
        
        Args:
            algorithms_data: {算法名: {指标名: 值}}
            save_path: 保存路径
            title: 图表标题
            
        Returns:
            保存路径
        """
        # 获取所有指标
        metrics = list(next(iter(algorithms_data.values())).keys())
        num_metrics = len(metrics)
        
        # 计算角度
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'), dpi=self.dpi)
        
        for i, (algo_name, metric_values) in enumerate(algorithms_data.items()):
            values = [metric_values[m] for m in metrics]
            values += values[:1]  # 闭合
            
            color = self.colors[i % len(self.colors)]
            ax.plot(angles, values, 'o-', linewidth=2, label=algo_name, color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_ylim(0, 1)
        ax.set_title(title, y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_scatter_with_regression(self,
                                        x_data: np.ndarray,
                                        y_data: np.ndarray,
                                        save_path: str,
                                        title: str = "Correlation Analysis",
                                        xlabel: str = "Metric X",
                                        ylabel: str = "Metric Y",
                                        show_regression: bool = True) -> str:
        """
        生成散点图+回归线
        
        Args:
            x_data: X轴数据
            y_data: Y轴数据
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            show_regression: 是否显示回归线
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        # 散点图
        ax.scatter(x_data, y_data, alpha=0.6, s=50, 
                  color=ACADEMIC_COLORS['blue'], edgecolors='black', linewidth=0.5)
        
        # 回归线
        if show_regression:
            z = np.polyfit(x_data, y_data, 1)
            p = np.poly1d(z)
            ax.plot(x_data, p(x_data), "--", color=ACADEMIC_COLORS['red'], 
                   linewidth=2, label=f'y={z[0]:.3f}x+{z[1]:.3f}')
            
            # 计算R²
            r_squared = np.corrcoef(x_data, y_data)[0, 1] ** 2
            ax.text(0.05, 0.95, f'$R^2$ = {r_squared:.3f}', 
                   transform=ax.transAxes, fontsize=12, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_learning_curve_with_variance(self,
                                             episodes: np.ndarray,
                                             mean_rewards: np.ndarray,
                                             std_rewards: np.ndarray,
                                             save_path: str,
                                             title: str = "Learning Curve with Variance",
                                             xlabel: str = "Episode",
                                             ylabel: str = "Average Reward") -> str:
        """
        生成学习曲线（带方差阴影）
        
        Args:
            episodes: Episode数组
            mean_rewards: 平均奖励数组
            std_rewards: 标准差数组
            save_path: 保存路径
            title: 图表标题
            xlabel: X轴标签
            ylabel: Y轴标签
            
        Returns:
            保存路径
        """
        fig, ax = plt.subplots(figsize=(10, 6), dpi=self.dpi)
        
        # 主线
        ax.plot(episodes, mean_rewards, color=ACADEMIC_COLORS['blue'], 
               linewidth=2.5, label='Mean Reward')
        
        # 置信区间 (±1σ, 约68%)
        ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                       alpha=0.3, color=ACADEMIC_COLORS['blue'], label='±1σ')
        
        # 置信区间 (±2σ, 约95%)
        ax.fill_between(episodes, mean_rewards - 2*std_rewards, mean_rewards + 2*std_rewards,
                       alpha=0.15, color=ACADEMIC_COLORS['blue'], label='±2σ')
        
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return save_path


def example_usage():
    """使用示例"""
    generator = AcademicChartGenerator(dpi=300)
    
    # 示例数据
    algorithms_data = {
        'TD3': np.random.randn(200) * 10 - 50,
        'DDPG': np.random.randn(200) * 12 - 55,
        'SAC': np.random.randn(200) * 8 - 45,
        'PPO': np.random.randn(200) * 15 - 60,
    }
    
    # 生成各种图表
    generator.generate_convergence_comparison(algorithms_data, 'convergence.png')
    generator.generate_boxplot_comparison(algorithms_data, 'boxplot.png')
    generator.generate_cdf_plot(algorithms_data, 'cdf.png')
    generator.generate_violin_plot(algorithms_data, 'violin.png')
    
    print("Academic charts generated successfully!")


if __name__ == "__main__":
    example_usage()
