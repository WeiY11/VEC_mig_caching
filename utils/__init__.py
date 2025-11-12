"""
工具模块
通用工具函数和辅助类
"""

from .logger import Logger
from .metrics import Metrics, MovingAverage, PerformanceTracker
from .data_processor import DataProcessor
from .data_validator import SystemMetricsValidator
from .energy_validator import EnergyValidator, validate_energy_consumption
from .unified_reward_calculator import (
    calculate_unified_reward,
    calculate_simple_reward,  # 向后兼容
    calculate_enhanced_reward,  # 向后兼容
    calculate_sac_reward  # 向后兼容
)
from .common import (
    dbm_to_watts,
    watts_to_dbm,
    linear_to_db,
    calculate_distance
)

# 添加缺失的工具函数
import numpy as np

def generate_poisson_arrivals(rate: float, duration: float) -> list:
    """生成泊松到达时间序列"""
    arrivals = []
    current_time = 0.0
    while current_time < duration:
        inter_arrival = np.random.exponential(1.0 / rate)
        current_time += inter_arrival
        if current_time < duration:
            arrivals.append(current_time)
    return arrivals

def db_to_linear(db_value: float) -> float:
    """dB值转换为线性值"""
    return 10 ** (db_value / 10.0)

def sigmoid(x: float) -> float:
    """Sigmoid激活函数"""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

def calculate_3d_distance(pos1, pos2):
    """计算3D空间中两点间的距离"""
    if isinstance(pos1, tuple) and isinstance(pos2, tuple):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2)
    else:
        # 处理具有x,y,z属性的对象
        try:
            # type: ignore - 这里假设对象有x,y,z属性
            return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)  # type: ignore
        except AttributeError:
            # 如果没有x,y,z属性，尝试作为数组处理
            pos1_arr = np.array(pos1)
            pos2_arr = np.array(pos2)
            return np.sqrt(np.sum((pos1_arr - pos2_arr)**2))

def calculate_zipf_probability(rank: int, total_items: int, exponent: float = 0.8) -> float:
    """
    计算Zipf分布概率
    
    【功能】模拟内容流行度的Zipf分布（少数热门，大量冷门）
    【论文对应】Zipf law - 内容访问符合幂律分布
    
    【参数】
    - rank: 内容排名（1=最热门）
    - total_items: 总内容数量
    - exponent: Zipf指数（典型值0.6-1.2，越大越集中）
    
    【返回值】该排名的概率（0-1）
    
    【公式】P(k) = (1/k^s) / Σ(1/i^s), i=1..N
    """
    if rank <= 0 or total_items <= 0:
        return 0.0
    
    # 归一化常数（Zipf分布的分母）
    normalization = sum(1.0 / (i ** exponent) for i in range(1, total_items + 1))
    
    # 计算该排名的概率
    probability = (1.0 / (rank ** exponent)) / normalization
    
    return probability

def sample_zipf_content_id(num_contents: int = 1000, exponent: float = 0.8) -> str:
    """
    按Zipf分布采样内容ID
    
    【功能】生成符合Zipf热度分布的内容ID
    【论文对应】Section 2.7 "Collaborative Caching"
    
    【参数】
    - num_contents: 内容库大小（默认1000）
    - exponent: Zipf指数（默认0.8）
    
    【返回值】内容ID字符串
    
    【说明】
    - 热门内容（rank 1-10）：约50%概率被访问
    - 中等内容（rank 11-100）：约35%概率
    - 冷门内容（rank 101-1000）：约15%概率
    """
    # 计算每个排名的概率
    probabilities = [
        calculate_zipf_probability(rank, num_contents, exponent)
        for rank in range(1, num_contents + 1)
    ]
    
    # 按概率采样排名
    rank = np.random.choice(range(1, num_contents + 1), p=probabilities)
    
    # 返回内容ID
    return f"content_{rank:04d}"

def sample_pareto(scale: float, shape: float = 1.5) -> float:
    """
    采样帕累托分布（重尾分布）
    
    【功能】生成符合重尾特征的数值（大量小值+少量大值）
    【论文对应】真实任务大小分布的建模
    
    【参数】
    - scale: 最小值（分布下界）
    - shape: 形状参数（α，典型值1.16-1.5）
        * α=1.16: 80-20法则（20%内容占80%流量）
        * α=1.5: 更温和的重尾
        * α=2.0: 接近指数分布
    
    【返回值】采样值
    
    【分布特征】
    - P(X > x) = (scale/x)^shape
    - 均值 = scale·shape/(shape-1)  (shape > 1)
    - 方差很大，存在极端值
    
    【使用示例】
    >>> # 任务大小：最小0.5MB，80-20分布
    >>> task_size = sample_pareto(0.5e6, shape=1.16)
    """
    return (np.random.pareto(shape) + 1) * scale

def sample_heavy_tailed_task_size(min_size: float, max_size: float, shape: float = 1.5) -> float:
    """
    采样重尾任务大小
    
    【功能】生成符合真实分布的任务数据大小
    【特征】大量小任务 + 少量大任务
    
    【参数】
    - min_size: 最小任务大小（bytes）
    - max_size: 最大任务大小（bytes）
    - shape: 帕累托形状参数
    
    【返回值】任务大小（bytes）
    """
    while True:
        size = sample_pareto(min_size, shape)
        if size <= max_size:
            return size
        # 如果超出上界，重新采样（rejection sampling）

class ExponentialMovingAverage:
    """指数移动平均"""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float) -> float:
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def get_value(self) -> float:
        """获取当前指数移动平均值"""
        return self.value if self.value is not None else 0.0

__all__ = ['Logger', 'Metrics', 'MovingAverage', 'PerformanceTracker', 'DataProcessor', 
           'SystemMetricsValidator', 'EnergyValidator', 'validate_energy_consumption', 
           'calculate_unified_reward', 'calculate_simple_reward', 'calculate_enhanced_reward', 'calculate_sac_reward',
           'generate_poisson_arrivals', 'db_to_linear', 'sigmoid', 'calculate_3d_distance', 'ExponentialMovingAverage',
           'calculate_zipf_probability', 'sample_zipf_content_id', 'sample_pareto', 'sample_heavy_tailed_task_size',
           'dbm_to_watts', 'watts_to_dbm', 'linear_to_db', 'calculate_distance']