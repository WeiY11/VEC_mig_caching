"""
训练过程中的任务处理方式分布集成模块

集成任务卸载分析功能到训练循环中，自动记录和分析任务处理方式的演化趋势。

使用示例：
    # 在train_single_agent.py的训练函数中
    from utils.training_analytics_integration import TaskAnalyticsTracker
    
    analytics_tracker = TaskAnalyticsTracker(enable_logging=True, log_interval=10)
    
    for episode in range(num_episodes):
        analytics_tracker.start_episode(episode)
        
        for step in range(max_steps):
            step_result = simulator.run_simulation_step(step, actions)
            analytics_tracker.record_step(step, step_result)
        
        analytics_tracker.end_episode()
    
    # 获取统计结果
    analytics_tracker.print_summary(top_n=20)
    analytics_tracker.export_csv('task_distribution_analysis.csv')
"""

from typing import Dict, Optional, List
from utils.task_offloading_analytics import (
    TaskOffloadingAnalytics,
    TaskDistribution,
    EpisodeTaskStatistics
)


class TaskAnalyticsTracker:
    """
    任务处理方式分布统计跟踪器
    
    提供一个便捷的接口来在训练过程中记录和分析任务处理方式的分布情况。
    """
    
    def __init__(self, enable_logging: bool = True, log_interval: int = 10):
        """
        初始化跟踪器
        
        Args:
            enable_logging: 是否启用控制台日志输出
            log_interval: 每隔多少步输出一次日志
        """
        self.analytics = TaskOffloadingAnalytics()
        self.analytics.enable_logging = enable_logging
        self.analytics.log_interval = log_interval
        self.training_history = {
            'episodes': [],
            'episode_lengths': [],
            'local_ratios': [],
            'rsu_ratios': [],
            'uav_ratios': [],
            'success_rates': [],
            'cache_hit_rates': []
        }
    
    def start_episode(self, episode: int) -> None:
        """开始记录新的episode"""
        self.analytics.start_episode(episode)
    
    def record_step(self, step: int, step_result: Dict) -> None:
        """记录单个仿真步的任务分布信息"""
        self.analytics.record_step(step, step_result)
    
    def end_episode(self) -> Optional[EpisodeTaskStatistics]:
        """
        结束episode记录，返回该episode的统计信息
        
        Returns:
            EpisodeTaskStatistics: 该episode的统计摘要
        """
        stats = self.analytics.finalize_episode()
        if stats is not None:
            self._update_training_history(stats)
        return stats
    
    def _update_training_history(self, stats: EpisodeTaskStatistics) -> None:
        """更新训练历史记录"""
        self.training_history['episodes'].append(stats.episode)
        self.training_history['episode_lengths'].append(stats.num_steps)
        self.training_history['local_ratios'].append(stats.local_ratio)
        self.training_history['rsu_ratios'].append(stats.rsu_ratio)
        self.training_history['uav_ratios'].append(stats.uav_ratio)
        self.training_history['success_rates'].append(stats.success_ratio)
        self.training_history['cache_hit_rates'].append(stats.rsu_cache_hit_rate)
    
    def get_latest_episode_stats(self) -> Optional[Dict]:
        """获取最新episode的统计信息"""
        return self.analytics.get_episode_summary()
    
    def get_evolution_trend(self) -> Dict[str, List[float]]:
        """获取任务处理方式的演化趋势"""
        return self.analytics.get_evolution_trend()
    
    def print_summary(self, top_n: int = 20) -> None:
        """打印训练统计摘要（最近top_n个episodes）"""
        self.analytics.print_summary(top_n=top_n)
    
    def export_csv(self, filepath: str) -> None:
        """导出统计数据为CSV格式"""
        self.analytics.export_csv(filepath)
        print(f"✓ 数据已导出到: {filepath}")
    
    def get_training_summary(self) -> Dict:
        """获取完整的训练统计汇总"""
        total_episodes = len(self.training_history['episodes'])
        
        if total_episodes == 0:
            return {'error': 'No episodes recorded'}
        
        import numpy as np
        
        return {
            'total_episodes': total_episodes,
            'total_steps': sum(self.training_history['episode_lengths']),
            'local_ratio_avg': np.mean(self.training_history['local_ratios']),
            'local_ratio_std': np.std(self.training_history['local_ratios']),
            'local_ratio_trend': 'increasing' if self.training_history['local_ratios'][-1] > self.training_history['local_ratios'][0] else 'decreasing',
            'rsu_ratio_avg': np.mean(self.training_history['rsu_ratios']),
            'rsu_ratio_std': np.std(self.training_history['rsu_ratios']),
            'uav_ratio_avg': np.mean(self.training_history['uav_ratios']),
            'uav_ratio_std': np.std(self.training_history['uav_ratios']),
            'success_rate_avg': np.mean(self.training_history['success_rates']),
            'success_rate_min': np.min(self.training_history['success_rates']),
            'success_rate_max': np.max(self.training_history['success_rates']),
            'cache_hit_rate_avg': np.mean(self.training_history['cache_hit_rates']),
        }
    
    def print_training_summary(self) -> None:
        """打印完整的训练统计汇总"""
        summary = self.get_training_summary()
        
        if 'error' in summary:
            print(f"⚠️  {summary['error']}")
            return
        
        print(f"\n{'='*80}")
        print(f"{'📊 训练统计汇总':<40}")
        print(f"{'='*80}")
        print(f"总Episode数: {summary['total_episodes']}")
        print(f"总步数: {summary['total_steps']}")
        print()
        print("任务处理方式分布:")
        print(f"  本地处理占比:  {summary['local_ratio_avg']:.1%} ± {summary['local_ratio_std']:.1%} ({summary['local_ratio_trend']})")
        print(f"  RSU处理占比:   {summary['rsu_ratio_avg']:.1%} ± {summary['rsu_ratio_std']:.1%}")
        print(f"  UAV处理占比:   {summary['uav_ratio_avg']:.1%} ± {summary['uav_ratio_std']:.1%}")
        print()
        print("性能指标:")
        print(f"  任务成功率: {summary['success_rate_avg']:.1%} (范围: {summary['success_rate_min']:.1%} - {summary['success_rate_max']:.1%})")
        print(f"  缓存命中率: {summary['cache_hit_rate_avg']:.1%}")
        print(f"{'='*80}\n")


def create_analytics_callback(tracker: TaskAnalyticsTracker, num_episodes: int = 100):
    """
    创建一个回调函数，可以集成到训练循环中
    
    Returns:
        callable: 回调函数，签名为(episode, step, step_result) -> None
    """
    def callback(episode: int, step: int, step_result: Dict) -> None:
        if step == 0:
            tracker.start_episode(episode)
        
        tracker.record_step(step, step_result)
    
    return callback
