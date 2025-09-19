"""
工具模块
通用工具函数和辅助类
"""

from .logger import Logger
from .metrics import Metrics, MovingAverage, PerformanceTracker
from .data_processor import DataProcessor

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
        return np.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2 + (pos1.z - pos2.z)**2)

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

__all__ = ['Logger', 'Metrics', 'MovingAverage', 'PerformanceTracker', 'DataProcessor', 
           'generate_poisson_arrivals', 'db_to_linear', 'sigmoid', 'calculate_3d_distance', 'ExponentialMovingAverage']