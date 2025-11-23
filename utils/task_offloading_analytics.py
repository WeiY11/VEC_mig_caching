"""
任务处理方式分布统计模块

用于统计和分析单个episode中任务的处理方式分布情况：
- 本地处理的任务数量及占比
- RSU基站处理的任务数量及占比
- UAV无人机处理的任务数量及占比

支持按episode统计、按agent step统计，以及跨episode的统计分析。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np


@dataclass
class TaskDistribution:
    """单个时间步或episode的任务处理方式分布统计"""
    
    # 时间戳和标识
    timestamp: datetime = field(default_factory=datetime.now)
    episode: int = -1
    step: int = -1
    
    # 任务数量统计
    total_generated: int = 0        # 生成的总任务数
    local_processed: int = 0         # 本地处理的任务数
    rsu_processed: int = 0           # RSU处理的任务数
    uav_processed: int = 0           # UAV处理的任务数
    dropped_tasks: int = 0           # 被丢弃的任务数
    
    # 分项统计（可选）
    rsu_cache_hits: int = 0         # RSU缓存命中的任务
    rsu_cache_misses: int = 0        # RSU缓存未命中的任务
    
    @property
    def successfully_processed(self) -> int:
        """成功处理的任务总数"""
        return self.local_processed + self.rsu_processed + self.uav_processed
    
    @property
    def local_ratio(self) -> float:
        """本地处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.local_processed / self.total_generated
    
    @property
    def rsu_ratio(self) -> float:
        """RSU处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.rsu_processed / self.total_generated
    
    @property
    def uav_ratio(self) -> float:
        """UAV处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.uav_processed / self.total_generated
    
    @property
    def drop_ratio(self) -> float:
        """任务丢弃率"""
        if self.total_generated == 0:
            return 0.0
        return self.dropped_tasks / self.total_generated
    
    @property
    def success_ratio(self) -> float:
        """任务成功处理率"""
        if self.total_generated == 0:
            return 0.0
        return self.successfully_processed / self.total_generated
    
    def to_dict(self) -> Dict:
        """转换为字典格式，便于日志输出和数据保存"""
        return {
            'episode': self.episode,
            'step': self.step,
            'timestamp': self.timestamp.isoformat(),
            'generated': self.total_generated,
            'local': self.local_processed,
            'rsu': self.rsu_processed,
            'uav': self.uav_processed,
            'dropped': self.dropped_tasks,
            'rsu_hits': self.rsu_cache_hits,
            'rsu_misses': self.rsu_cache_misses,
            'local_ratio': f"{self.local_ratio:.1%}",
            'rsu_ratio': f"{self.rsu_ratio:.1%}",
            'uav_ratio': f"{self.uav_ratio:.1%}",
            'drop_ratio': f"{self.drop_ratio:.1%}",
            'success_ratio': f"{self.success_ratio:.1%}",
        }


@dataclass
class EpisodeTaskStatistics:
    """单个episode的任务处理方式总体统计"""
    
    episode: int = -1
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    # 累计统计
    total_generated: int = 0         # 总生成任务数
    total_local: int = 0             # 累计本地处理
    total_rsu: int = 0               # 累计RSU处理
    total_uav: int = 0               # 累计UAV处理
    total_dropped: int = 0           # 累计丢弃任务
    
    # RSU分项统计
    total_rsu_cache_hits: int = 0   # 总缓存命中数
    total_rsu_cache_misses: int = 0 # 总缓存未命中数
    
    # 历史记录（用于趋势分析）
    step_distributions: List[TaskDistribution] = field(default_factory=list)
    
    @property
    def num_steps(self) -> int:
        """该episode的步数"""
        return len(self.step_distributions)
    
    @property
    def local_ratio(self) -> float:
        """本地处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.total_local / self.total_generated
    
    @property
    def rsu_ratio(self) -> float:
        """RSU处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.total_rsu / self.total_generated
    
    @property
    def uav_ratio(self) -> float:
        """UAV处理占比"""
        if self.total_generated == 0:
            return 0.0
        return self.total_uav / self.total_generated
    
    @property
    def drop_ratio(self) -> float:
        """任务丢弃率"""
        if self.total_generated == 0:
            return 0.0
        return self.total_dropped / self.total_generated
    
    @property
    def success_ratio(self) -> float:
        """任务成功处理率"""
        if self.total_generated == 0:
            return 0.0
        return (self.total_generated - self.total_dropped) / self.total_generated
    
    @property
    def rsu_cache_hit_rate(self) -> float:
        """RSU缓存命中率"""
        total_rsu_tasks = self.total_rsu_cache_hits + self.total_rsu_cache_misses
        if total_rsu_tasks == 0:
            return 0.0
        return self.total_rsu_cache_hits / total_rsu_tasks
    
    @property
    def avg_local_ratio(self) -> float:
        """平均每步的本地处理占比"""
        if not self.step_distributions:
            return 0.0
        return float(np.mean([d.local_ratio for d in self.step_distributions]))
    
    @property
    def avg_rsu_ratio(self) -> float:
        """平均每步的RSU处理占比"""
        if not self.step_distributions:
            return 0.0
        return float(np.mean([d.rsu_ratio for d in self.step_distributions]))
    
    @property
    def avg_uav_ratio(self) -> float:
        """平均每步的UAV处理占比"""
        if not self.step_distributions:
            return 0.0
        return float(np.mean([d.uav_ratio for d in self.step_distributions]))
    
    def update_from_step(self, step_dist: TaskDistribution) -> None:
        """使用单个step的统计更新episode累计数据"""
        self.total_generated += step_dist.total_generated
        self.total_local += step_dist.local_processed
        self.total_rsu += step_dist.rsu_processed
        self.total_uav += step_dist.uav_processed
        self.total_dropped += step_dist.dropped_tasks
        self.total_rsu_cache_hits += step_dist.rsu_cache_hits
        self.total_rsu_cache_misses += step_dist.rsu_cache_misses
        self.step_distributions.append(step_dist)
    
    def finalize(self) -> None:
        """标记episode结束时间"""
        self.end_time = datetime.now()
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        duration = (self.end_time - self.start_time).total_seconds() if self.end_time else None
        return {
            'episode': self.episode,
            'steps': self.num_steps,
            'duration': duration,
            'total_generated': self.total_generated,
            'total_local': self.total_local,
            'total_rsu': self.total_rsu,
            'total_uav': self.total_uav,
            'total_dropped': self.total_dropped,
            'local_ratio': f"{self.local_ratio:.1%}",
            'rsu_ratio': f"{self.rsu_ratio:.1%}",
            'uav_ratio': f"{self.uav_ratio:.1%}",
            'drop_ratio': f"{self.drop_ratio:.1%}",
            'success_ratio': f"{self.success_ratio:.1%}",
            'rsu_cache_hit_rate': f"{self.rsu_cache_hit_rate:.1%}",
            'avg_local_ratio': f"{self.avg_local_ratio:.1%}",
            'avg_rsu_ratio': f"{self.avg_rsu_ratio:.1%}",
            'avg_uav_ratio': f"{self.avg_uav_ratio:.1%}",
        }


class TaskOffloadingAnalytics:
    """任务处理方式分布分析器"""
    
    def __init__(self):
        """初始化分析器"""
        self.current_episode: Optional[EpisodeTaskStatistics] = None
        self.episode_history: List[EpisodeTaskStatistics] = []
        self.enable_logging = True
        self.log_interval = 10  # 每10个step输出一次日志
        # 🔧 新增：用于追踪累积统计的基线值（每个episode开始时重置）
        self._cumulative_baseline: Dict[str, int] = {}
    
    def start_episode(self, episode: int) -> None:
        """开始新的episode统计"""
        if self.current_episode is not None:
            self.finalize_episode()
        
        self.current_episode = EpisodeTaskStatistics(episode=episode)
        # 🔧 重置累积基线（用于计算单步增量）
        self._cumulative_baseline = {}
    
    def record_step(self, step: int, step_result: Dict) -> None:
        """从simulator的step_result记录单步统计
        
        Args:
            step: 步数
            step_result: 来自simulator.run_simulation_step()的返回值
                        注意：step_result中的某些统计是累积值（如dropped_tasks）
        """
        if self.current_episode is None:
            raise RuntimeError("Must call start_episode() before record_step()")
        
        # 🔧 关键修复：step_result中的dropped_tasks是累积值，需要计算增量
        # 获取当前累积统计
        current_dropped_cumulative = int(step_result.get('dropped_tasks', 0))
        current_cache_hits = int(step_result.get('cache_hits', 0))
        current_cache_misses = int(step_result.get('cache_misses', 0))
        
        # 计算单步增量
        previous_dropped = self._cumulative_baseline.get('dropped_tasks', 0)
        step_dropped_increment = max(0, current_dropped_cumulative - previous_dropped)
        prev_hits = self._cumulative_baseline.get('cache_hits', 0)
        prev_misses = self._cumulative_baseline.get('cache_misses', 0)
        step_cache_hits = max(0, current_cache_hits - prev_hits)
        step_cache_misses = max(0, current_cache_misses - prev_misses)
        
        # 更新基线
        self._cumulative_baseline['dropped_tasks'] = current_dropped_cumulative
        self._cumulative_baseline['cache_hits'] = current_cache_hits
        self._cumulative_baseline['cache_misses'] = current_cache_misses
        
        # 提取step_result中的任务分布信息（其他字段是单步值）
        dist = TaskDistribution(
            episode=self.current_episode.episode,
            step=step,
            total_generated=int(step_result.get('generated_tasks', 0)),
            local_processed=int(step_result.get('local_tasks', 0)),
            rsu_processed=int(step_result.get('remote_tasks', 0)),  # 包括RSU和UAV
            # 注意：当前system_simulator未分离RSU和UAV，需要增强
            uav_processed=0,  # 待改进：需要从simulator分离出UAV任务数
            dropped_tasks=step_dropped_increment,  # 🔧 使用增量值而非累积值
            rsu_cache_hits=step_cache_hits,
        )
        
        # 尝试从step_result中获取更详细的统计（如果有）
        if 'rsu_tasks' in step_result:
            dist.rsu_processed = int(step_result['rsu_tasks'])
        if 'uav_tasks' in step_result:
            dist.uav_processed = int(step_result['uav_tasks'])
        if 'rsu_cache_hits' in step_result:
            dist.rsu_cache_hits = int(step_result['rsu_cache_hits'])
        if 'rsu_cache_misses' in step_result:
            dist.rsu_cache_misses = int(step_result['rsu_cache_misses'])
        if dist.rsu_cache_misses == 0 and step_cache_misses > 0:
            dist.rsu_cache_misses = step_cache_misses
        
        # 更新episode统计
        self.current_episode.update_from_step(dist)
        
        # 定期输出日志
        if self.enable_logging and (step + 1) % self.log_interval == 0:
            self._log_step_distribution(dist)
    
    def finalize_episode(self) -> Optional[EpisodeTaskStatistics]:
        """结束当前episode统计并保存"""
        if self.current_episode is None:
            return None
        
        self.current_episode.finalize()
        self.episode_history.append(self.current_episode)
        
        if self.enable_logging:
            self._log_episode_summary(self.current_episode)
        
        result = self.current_episode
        self.current_episode = None
        return result
    
    def get_episode_summary(self, episode: Optional[int] = None) -> Optional[Dict]:
        """获取指定episode的统计摘要"""
        if episode is None:
            # 返回最新episode
            if self.episode_history:
                return self.episode_history[-1].to_dict()
            elif self.current_episode is not None:
                return self.current_episode.to_dict()
            return None
        
        # 查找指定episode
        for ep_stat in self.episode_history:
            if ep_stat.episode == episode:
                return ep_stat.to_dict()
        return None
    
    def get_evolution_trend(self) -> Dict[str, List[float]]:
        """获取任务处理方式的演化趋势（跨episode）"""
        if not self.episode_history:
            return {}
        
        trends = {
            'episodes': [],
            'local_ratio': [],
            'rsu_ratio': [],
            'uav_ratio': [],
            'drop_ratio': [],
            'success_ratio': []
        }
        
        for ep_stat in self.episode_history:
            trends['episodes'].append(ep_stat.episode)
            trends['local_ratio'].append(ep_stat.local_ratio)
            trends['rsu_ratio'].append(ep_stat.rsu_ratio)
            trends['uav_ratio'].append(ep_stat.uav_ratio)
            trends['drop_ratio'].append(ep_stat.drop_ratio)
            trends['success_ratio'].append(ep_stat.success_ratio)
        
        return trends
    
    def _log_step_distribution(self, dist: TaskDistribution) -> None:
        """输出单步统计到日志"""
        print(
            f"[Episode {dist.episode}, Step {dist.step}] "
            f"Tasks: generated={dist.total_generated}, "
            f"local={dist.local_processed}({dist.local_ratio:.1%}), "
            f"rsu={dist.rsu_processed}({dist.rsu_ratio:.1%}), "
            f"uav={dist.uav_processed}({dist.uav_ratio:.1%}), "
            f"dropped={dist.dropped_tasks}({dist.drop_ratio:.1%})"
        )
    
    def _log_episode_summary(self, stats: EpisodeTaskStatistics) -> None:
        """输出episode总结到日志"""
        print(
            f"\n{'='*80}\n"
            f"📊 Episode {stats.episode} 任务处理方式分布统计\n"
            f"{'='*80}"
        )
        print(f"总步数: {stats.num_steps}")
        print(f"总生成任务数: {stats.total_generated}")
        print()
        print("任务分布占比:")
        print(f"  ✓ 本地处理: {stats.total_local:>6} 任务 ({stats.local_ratio:>6.1%})")
        print(f"  ✓ RSU处理:  {stats.total_rsu:>6} 任务 ({stats.rsu_ratio:>6.1%})")
        print(f"  ✓ UAV处理:  {stats.total_uav:>6} 任务 ({stats.uav_ratio:>6.1%})")
        print(f"  ✗ 被丢弃:   {stats.total_dropped:>6} 任务 ({stats.drop_ratio:>6.1%})")
        print()
        print("补充指标:")
        print(f"  任务成功率: {stats.success_ratio:.1%}")
        if stats.total_rsu > 0:
            print(f"  RSU缓存命中率: {stats.rsu_cache_hit_rate:.1%}")
        print(f"  平均本地占比: {stats.avg_local_ratio:.1%}")
        print(f"  平均RSU占比: {stats.avg_rsu_ratio:.1%}")
        print(f"  平均UAV占比: {stats.avg_uav_ratio:.1%}")
        print(f"{'='*80}\n")
    
    def export_csv(self, filepath: str) -> None:
        """导出统计数据为CSV格式"""
        import csv
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    'episode', 'steps', 'total_generated', 'total_local', 
                    'total_rsu', 'total_uav', 'total_dropped',
                    'local_ratio', 'rsu_ratio', 'uav_ratio', 'drop_ratio',
                    'success_ratio', 'rsu_cache_hit_rate'
                ]
            )
            writer.writeheader()
            for ep_stat in self.episode_history:
                data = ep_stat.to_dict()
                writer.writerow({
                    'episode': data['episode'],
                    'steps': data['steps'],
                    'total_generated': data['total_generated'],
                    'total_local': data['total_local'],
                    'total_rsu': data['total_rsu'],
                    'total_uav': data['total_uav'],
                    'total_dropped': data['total_dropped'],
                    'local_ratio': data['local_ratio'],
                    'rsu_ratio': data['rsu_ratio'],
                    'uav_ratio': data['uav_ratio'],
                    'drop_ratio': data['drop_ratio'],
                    'success_ratio': data['success_ratio'],
                    'rsu_cache_hit_rate': data['rsu_cache_hit_rate'],
                })
    
    def print_summary(self, top_n: int = 10) -> None:
        """打印汇总统计（最近top_n个episode）"""
        if not self.episode_history:
            print("No episode history available")
            return
        
        episodes_to_show = self.episode_history[-top_n:]
        
        print(f"\n{'='*100}")
        print(f"{'📈 任务处理方式演化趋势':<50}")
        print(f"{'='*100}")
        print(
            f"{'Episode':<10} {'Local':<12} {'RSU':<12} {'UAV':<12} "
            f"{'Dropped':<12} {'Success':<12}"
        )
        print("-" * 100)
        
        for stats in episodes_to_show:
            print(
                f"{stats.episode:<10} "
                f"{stats.local_ratio:<12.1%} "
                f"{stats.rsu_ratio:<12.1%} "
                f"{stats.uav_ratio:<12.1%} "
                f"{stats.drop_ratio:<12.1%} "
                f"{stats.success_ratio:<12.1%}"
            )
        
        print(f"{'='*100}\n")


# 全局分析器实例（方便在训练脚本中使用）
_global_analytics = TaskOffloadingAnalytics()


def get_global_analytics() -> TaskOffloadingAnalytics:
    """获取全局分析器实例"""
    return _global_analytics
