"""
🎨 高端训练实时可视化系统
Advanced Real-time Training Visualization System

功能特性：
- 📊 多指标动态曲线图（奖励、损失、命中率等）
- 🔥 热力图展示（状态分布、动作分布）
- 📈 梯度流可视化
- ⚡ 系统资源监控（CPU、GPU、内存）
- 🎯 性能指标面板
- 🌈 精美配色方案

使用方法：
from utils.advanced_training_visualizer import AdvancedTrainingVisualizer
visualizer = AdvancedTrainingVisualizer()
visualizer.start()  # 启动可视化服务器
visualizer.update(episode, metrics)  # 更新数据
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from collections import deque
from typing import Dict, List, Optional, Any, Tuple
import threading
import time
from datetime import datetime
import psutil
import warnings
warnings.filterwarnings('ignore')

# 尝试导入GPU监控
try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

# 设置高端样式
plt.style.use('dark_background')
sns.set_palette("husl")


class AdvancedTrainingVisualizer:
    """
    🎨 高端训练可视化器
    
    特点：
    - 实时动态更新
    - 多面板布局
    - 精美配色
    - 性能优化
    """
    
    def __init__(self, max_history: int = 500, update_interval: int = 1):
        """
        初始化可视化器
        
        Args:
            max_history: 最大历史记录数
            update_interval: 更新间隔（秒）
        """
        self.max_history = max_history
        self.update_interval = update_interval
        
        # 数据存储
        self.episodes = deque(maxlen=max_history)
        self.rewards = deque(maxlen=max_history)
        self.losses = deque(maxlen=max_history)
        self.hit_rates = deque(maxlen=max_history)
        self.delays = deque(maxlen=max_history)
        self.energies = deque(maxlen=max_history)
        self.success_rates = deque(maxlen=max_history)
        
        # 系统监控
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
        self.gpu_usage = deque(maxlen=100) if GPU_AVAILABLE else None
        
        # 额外统计
        self.action_distribution = np.zeros(10)  # 假设最多10维动作
        self.state_distribution = []
        self.gradient_norms = deque(maxlen=max_history)
        
        # 控制变量
        self.running = False
        self.paused = False
        self.current_episode = 0
        
        # 配色方案（高端渐变色）
        self.colors = {
            'reward': '#00D9FF',      # 青色
            'loss': '#FF6B6B',        # 红色
            'hit_rate': '#4ECDC4',    # 蓝绿色
            'delay': '#FFE66D',       # 黄色
            'energy': '#A8E6CF',      # 薄荷绿
            'success': '#FF8B94',     # 粉红色
            'cpu': '#95E1D3',         # 浅绿
            'memory': '#F38181',      # 珊瑚红
            'gpu': '#AA96DA',         # 紫色
            'grid': '#2D3748',        # 深灰
            'text': '#E2E8F0'         # 浅灰
        }
        
        # 初始化图形
        self.fig = None
        self.axes = {}
        self.lines = {}
        self.texts = {}
        
    def setup_figure(self):
        """设置高端图形布局"""
        # 创建大型图形窗口
        self.fig = plt.figure(figsize=(20, 12), facecolor='#1A202C')
        self.fig.suptitle('🎯 Deep Reinforcement Learning Training Dashboard', 
                         fontsize=20, fontweight='bold', color=self.colors['text'], y=0.98)
        
        # 创建复杂网格布局
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.35, wspace=0.3,
                     left=0.05, right=0.98, top=0.94, bottom=0.05)
        
        # === 第一行：主要指标曲线 ===
        self.axes['reward'] = self.fig.add_subplot(gs[0, :2])
        self.axes['loss'] = self.fig.add_subplot(gs[0, 2:])
        
        # === 第二行：性能指标 ===
        self.axes['hit_rate'] = self.fig.add_subplot(gs[1, 0])
        self.axes['delay'] = self.fig.add_subplot(gs[1, 1])
        self.axes['energy'] = self.fig.add_subplot(gs[1, 2])
        self.axes['success'] = self.fig.add_subplot(gs[1, 3])
        
        # === 第三行：热力图和分布 ===
        self.axes['action_dist'] = self.fig.add_subplot(gs[2, :2])
        self.axes['gradient'] = self.fig.add_subplot(gs[2, 2:])
        
        # === 第四行：系统资源监控 ===
        self.axes['system'] = self.fig.add_subplot(gs[3, :3])
        self.axes['stats'] = self.fig.add_subplot(gs[3, 3])
        
        # 设置每个子图
        self._setup_reward_plot()
        self._setup_loss_plot()
        self._setup_performance_plots()
        self._setup_distribution_plots()
        self._setup_system_plot()
        self._setup_stats_panel()
        
    def _setup_reward_plot(self):
        """设置奖励曲线图"""
        ax = self.axes['reward']
        ax.set_facecolor('#2D3748')
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.set_title('📈 Episode Reward (Moving Average)', 
                    fontsize=14, fontweight='bold', color=self.colors['text'], pad=10)
        ax.set_xlabel('Episode', color=self.colors['text'], fontsize=11)
        ax.set_ylabel('Reward', color=self.colors['text'], fontsize=11)
        ax.tick_params(colors=self.colors['text'])
        
        # 创建多条线（原始+平滑）
        self.lines['reward_raw'], = ax.plot([], [], alpha=0.3, 
                                            color=self.colors['reward'], linewidth=1)
        self.lines['reward_smooth'], = ax.plot([], [], 
                                               color=self.colors['reward'], linewidth=2.5)
        ax.legend(['Raw', 'Smooth (MA50)'], loc='upper left', 
                 facecolor='#2D3748', edgecolor=self.colors['text'], 
                 labelcolor=self.colors['text'])
        
    def _setup_loss_plot(self):
        """设置损失曲线图"""
        ax = self.axes['loss']
        ax.set_facecolor('#2D3748')
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.set_title('📉 Training Loss (Log Scale)', 
                    fontsize=14, fontweight='bold', color=self.colors['text'], pad=10)
        ax.set_xlabel('Episode', color=self.colors['text'], fontsize=11)
        ax.set_ylabel('Loss', color=self.colors['text'], fontsize=11)
        ax.set_yscale('log')
        ax.tick_params(colors=self.colors['text'])
        
        self.lines['loss'], = ax.plot([], [], color=self.colors['loss'], linewidth=2.5)
        
    def _setup_performance_plots(self):
        """设置性能指标小图"""
        metrics = [
            ('hit_rate', '🎯 Cache Hit Rate', '%'),
            ('delay', '⏱️ Average Delay', 'ms'),
            ('energy', '⚡ Energy Consumption', 'J'),
            ('success', '✅ Success Rate', '%')
        ]
        
        for key, title, unit in metrics:
            ax = self.axes[key]
            ax.set_facecolor('#2D3748')
            ax.grid(True, alpha=0.2, color=self.colors['grid'])
            ax.set_title(title, fontsize=11, fontweight='bold', 
                        color=self.colors['text'], pad=8)
            ax.tick_params(colors=self.colors['text'], labelsize=9)
            
            self.lines[key], = ax.plot([], [], color=self.colors[key], linewidth=2)
            
            # 添加当前值文本
            self.texts[key] = ax.text(0.95, 0.95, '', transform=ax.transAxes,
                                     fontsize=16, fontweight='bold',
                                     color=self.colors[key],
                                     ha='right', va='top')
    
    def _setup_distribution_plots(self):
        """设置分布图"""
        # 动作分布柱状图
        ax = self.axes['action_dist']
        ax.set_facecolor('#2D3748')
        ax.set_title('🎲 Action Distribution (Recent 100)', 
                    fontsize=12, fontweight='bold', color=self.colors['text'], pad=10)
        ax.set_xlabel('Action Dimension', color=self.colors['text'])
        ax.set_ylabel('Frequency', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        
        # 梯度范数图
        ax = self.axes['gradient']
        ax.set_facecolor('#2D3748')
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.set_title('📊 Gradient Norm', 
                    fontsize=12, fontweight='bold', color=self.colors['text'], pad=10)
        ax.set_xlabel('Episode', color=self.colors['text'])
        ax.set_ylabel('Norm', color=self.colors['text'])
        ax.tick_params(colors=self.colors['text'])
        ax.set_yscale('log')
        
        self.lines['gradient'], = ax.plot([], [], color='#FFD93D', linewidth=2)
        
    def _setup_system_plot(self):
        """设置系统资源监控"""
        ax = self.axes['system']
        ax.set_facecolor('#2D3748')
        ax.grid(True, alpha=0.2, color=self.colors['grid'])
        ax.set_title('💻 System Resource Monitor', 
                    fontsize=12, fontweight='bold', color=self.colors['text'], pad=10)
        ax.set_xlabel('Time (s)', color=self.colors['text'])
        ax.set_ylabel('Usage (%)', color=self.colors['text'])
        ax.set_ylim(0, 100)
        ax.tick_params(colors=self.colors['text'])
        
        self.lines['cpu'], = ax.plot([], [], color=self.colors['cpu'], 
                                     linewidth=2, label='CPU')
        self.lines['memory'], = ax.plot([], [], color=self.colors['memory'], 
                                        linewidth=2, label='Memory')
        if GPU_AVAILABLE:
            self.lines['gpu'], = ax.plot([], [], color=self.colors['gpu'], 
                                         linewidth=2, label='GPU')
        
        ax.legend(loc='upper left', facecolor='#2D3748', 
                 edgecolor=self.colors['text'], labelcolor=self.colors['text'])
        
    def _setup_stats_panel(self):
        """设置统计信息面板"""
        ax = self.axes['stats']
        ax.set_facecolor('#2D3748')
        ax.axis('off')
        
        # 添加标题
        ax.text(0.5, 0.95, '📊 Statistics', transform=ax.transAxes,
               fontsize=14, fontweight='bold', color=self.colors['text'],
               ha='center', va='top')
        
        # 初始化统计文本
        self.texts['stats_text'] = ax.text(0.1, 0.80, '', transform=ax.transAxes,
                                           fontsize=10, color=self.colors['text'],
                                           va='top', family='monospace')
        
    def update(self, episode: int, metrics: Dict[str, Any]):
        """
        更新可视化数据
        
        Args:
            episode: 当前episode
            metrics: 指标字典
                - reward: 奖励值
                - loss: 损失值（可选）
                - hit_rate: 缓存命中率
                - delay: 延迟
                - energy: 能耗
                - success_rate: 成功率
                - action: 动作向量（可选）
                - gradient_norm: 梯度范数（可选）
        """
        self.current_episode = episode
        
        # 更新数据
        self.episodes.append(episode)
        self.rewards.append(metrics.get('reward', 0))
        self.losses.append(metrics.get('loss', 0))
        self.hit_rates.append(metrics.get('hit_rate', 0) * 100)
        self.delays.append(metrics.get('delay', 0))
        self.energies.append(metrics.get('energy', 0))
        self.success_rates.append(metrics.get('success_rate', 0) * 100)
        
        # 更新动作分布
        if 'action' in metrics and metrics['action'] is not None:
            action = np.array(metrics['action'])
            if len(action) <= len(self.action_distribution):
                for i, a in enumerate(action[:len(self.action_distribution)]):
                    self.action_distribution[i] += abs(a)
        
        # 更新梯度
        if 'gradient_norm' in metrics:
            self.gradient_norms.append(metrics['gradient_norm'])
        
        # 更新系统资源
        self._update_system_stats()
        
    def _update_system_stats(self):
        """更新系统资源统计"""
        self.cpu_usage.append(psutil.cpu_percent())
        self.memory_usage.append(psutil.virtual_memory().percent)
        
        if GPU_AVAILABLE and self.gpu_usage is not None:
            try:
                import GPUtil as gpu_util
                gpus = gpu_util.getGPUs()
                if gpus:
                    self.gpu_usage.append(gpus[0].load * 100)
            except Exception:
                pass
    
    def _update_plots(self, frame):
        """动画更新函数"""
        if self.paused or len(self.episodes) == 0:
            return []
        
        updated_artists = []
        
        # 更新奖励曲线
        episodes_arr = np.array(self.episodes)
        rewards_arr = np.array(self.rewards)
        
        self.lines['reward_raw'].set_data(episodes_arr, rewards_arr)
        
        # 计算移动平均
        if len(rewards_arr) > 50:
            smooth_rewards = np.convolve(rewards_arr, np.ones(50)/50, mode='valid')
            smooth_episodes = episodes_arr[49:]
            self.lines['reward_smooth'].set_data(smooth_episodes, smooth_rewards)
        
        self.axes['reward'].relim()
        self.axes['reward'].autoscale_view()
        
        # 更新损失曲线
        if len(self.losses) > 0 and max(self.losses) > 0:
            self.lines['loss'].set_data(episodes_arr, np.array(self.losses))
            self.axes['loss'].relim()
            self.axes['loss'].autoscale_view()
        
        # 更新性能指标
        metrics_data = {
            'hit_rate': (self.hit_rates, '%.1f%%'),
            'delay': (self.delays, '%.2f ms'),
            'energy': (self.energies, '%.2f J'),
            'success': (self.success_rates, '%.1f%%')
        }
        
        for key, (data, fmt) in metrics_data.items():
            if len(data) > 0:
                self.lines[key].set_data(episodes_arr, np.array(data))
                self.axes[key].relim()
                self.axes[key].autoscale_view()
                
                # 更新当前值文本
                current_val = data[-1]
                self.texts[key].set_text(fmt % current_val)
        
        # 更新动作分布
        ax = self.axes['action_dist']
        ax.clear()
        ax.set_facecolor('#2D3748')
        ax.set_title('🎲 Action Distribution (Recent 100)', 
                    fontsize=12, fontweight='bold', color=self.colors['text'], pad=10)
        ax.tick_params(colors=self.colors['text'])
        
        if np.sum(self.action_distribution) > 0:
            # 使用 colormap
            from matplotlib import cm
            colors_gradient = cm.get_cmap('viridis')(np.linspace(0.2, 0.9, len(self.action_distribution)))
            ax.bar(range(len(self.action_distribution)), 
                  self.action_distribution / max(np.sum(self.action_distribution), 1),
                  color=colors_gradient, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # 更新梯度图
        if len(self.gradient_norms) > 0:
            grad_arr = np.array(self.gradient_norms)
            grad_episodes = episodes_arr[-len(grad_arr):]
            self.lines['gradient'].set_data(grad_episodes, grad_arr)
            self.axes['gradient'].relim()
            self.axes['gradient'].autoscale_view()
        
        # 更新系统资源
        time_axis = np.arange(len(self.cpu_usage))
        self.lines['cpu'].set_data(time_axis, np.array(self.cpu_usage))
        self.lines['memory'].set_data(time_axis, np.array(self.memory_usage))
        if GPU_AVAILABLE and self.gpu_usage:
            self.lines['gpu'].set_data(time_axis, np.array(self.gpu_usage))
        
        self.axes['system'].relim()
        self.axes['system'].autoscale_view()
        
        # 更新统计面板
        self._update_stats_panel()
        
        return updated_artists
    
    def _update_stats_panel(self):
        """更新统计信息面板"""
        if len(self.episodes) == 0:
            return
        
        # 计算统计数据
        stats_text = f"""
Episode: {self.current_episode}
─────────────────────
Avg Reward: {np.mean(self.rewards):.2f}
Max Reward: {np.max(self.rewards):.2f}
Min Reward: {np.min(self.rewards):.2f}
─────────────────────
Hit Rate: {np.mean(self.hit_rates):.1f}%
Delay: {np.mean(self.delays):.2f} ms
Energy: {np.mean(self.energies):.2f} J
─────────────────────
CPU: {self.cpu_usage[-1] if self.cpu_usage else 0:.1f}%
Memory: {self.memory_usage[-1] if self.memory_usage else 0:.1f}%
"""
        if GPU_AVAILABLE and self.gpu_usage:
            stats_text += f"GPU: {self.gpu_usage[-1]:.1f}%\n"
        
        stats_text += f"""─────────────────────
Time: {datetime.now().strftime('%H:%M:%S')}
"""
        
        self.texts['stats_text'].set_text(stats_text)
    
    def start(self, interval: int = 1000):
        """
        启动可视化
        
        Args:
            interval: 刷新间隔（毫秒）
        """
        if self.fig is None:
            self.setup_figure()
        
        if self.fig is None:
            raise RuntimeError("无法初始化图形")
        
        self.running = True
        
        # 创建动画
        self.ani = animation.FuncAnimation(
            self.fig, self._update_plots, 
            interval=interval, blit=False, cache_frame_data=False
        )
        
        # 添加键盘控制
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        
        print("🎨 高端可视化已启动！")
        print("   - 按 'p' 暂停/继续")
        print("   - 按 's' 保存当前图像")
        print("   - 按 'q' 退出")
        
        plt.show(block=False)
        plt.pause(0.1)
    
    def _on_key_press(self, event):
        """键盘事件处理"""
        if self.fig is None:
            return
            
        if event.key == 'p':
            self.paused = not self.paused
            print(f"{'⏸️  已暂停' if self.paused else '▶️  继续'}")
        elif event.key == 's':
            filename = f"training_viz_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.fig.savefig(filename, dpi=300, facecolor='#1A202C')
            print(f"💾 已保存图像: {filename}")
        elif event.key == 'q':
            self.stop()
    
    def stop(self):
        """停止可视化"""
        self.running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
        plt.close(self.fig)
        print("🛑 可视化已停止")
    
    def save(self, filename: Optional[str] = None):
        """保存当前图形"""
        if self.fig is None:
            print("⚠️  图形未初始化，无法保存")
            return
            
        if filename is None:
            filename = f"training_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        self.fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='#1A202C')
        print(f"💾 Dashboard saved: {filename}")


# 简化接口函数
def create_visualizer(max_history: int = 500) -> AdvancedTrainingVisualizer:
    """
    创建高端训练可视化器（简化接口）
    
    Args:
        max_history: 最大历史记录数
        
    Returns:
        可视化器实例
    """
    return AdvancedTrainingVisualizer(max_history=max_history)


if __name__ == "__main__":
    """测试可视化器"""
    print("🎨 测试高端训练可视化器...")
    
    visualizer = create_visualizer()
    visualizer.start()
    
    # 模拟训练数据
    for episode in range(1000):
        # 模拟指标
        metrics = {
            'reward': np.random.randn() * 10 + 50 + episode * 0.1,
            'loss': 1.0 / (1 + episode * 0.01) + np.random.randn() * 0.1,
            'hit_rate': min(0.9, 0.5 + episode * 0.001) + np.random.randn() * 0.05,
            'delay': 100 - episode * 0.05 + np.random.randn() * 5,
            'energy': 50 - episode * 0.02 + np.random.randn() * 2,
            'success_rate': min(0.95, 0.7 + episode * 0.0005),
            'action': np.random.randn(10),
            'gradient_norm': 1.0 / (1 + episode * 0.01)
        }
        
        visualizer.update(episode, metrics)
        time.sleep(0.05)
    
    print("✅ 测试完成！按 'q' 退出")
    plt.show()
