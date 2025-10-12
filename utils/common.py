"""
工具函数模块
提供系统中常用的数学计算和转换函数
"""
import numpy as np
import math
from typing import Tuple, List


def db_to_linear(db_value: float) -> float:
    """将dB值转换为线性值"""
    return 10 ** (db_value / 10)


def linear_to_db(linear_value: float) -> float:
    """将线性值转换为dB值"""
    return 10 * math.log10(linear_value)


def dbm_to_watts(dbm_value: float) -> float:
    """将dBm值转换为瓦特"""
    return 10 ** ((dbm_value - 30) / 10)


def watts_to_dbm(watts_value: float) -> float:
    """将瓦特值转换为dBm"""
    return 30 + 10 * math.log10(watts_value)


def calculate_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """计算两点间欧几里得距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def calculate_3d_distance(pos1: Tuple[float, float, float], 
                         pos2: Tuple[float, float, float]) -> float:
    """计算三维空间中两点间距离"""
    return math.sqrt((pos1[0] - pos2[0])**2 + 
                    (pos1[1] - pos2[1])**2 + 
                    (pos1[2] - pos2[2])**2)


def sigmoid(x: float) -> float:
    """Sigmoid激活函数"""
    return 1 / (1 + math.exp(-x))


def generate_poisson_arrivals(rate: float, time_duration: float) -> int:
    """生成泊松到达过程的任务数量"""
    return np.random.poisson(rate * time_duration)


def generate_exponential_service_time(rate: float) -> float:
    """生成指数分布的服务时间"""
    return np.random.exponential(1 / rate)


# Zipf归一化常数缓存（优化性能）
_zipf_normalization_cache = {}

def calculate_zipf_probability(rank: int, num_items: int, exponent: float = 1.0) -> float:
    """
    计算Zipf分布的概率（优化版：缓存归一化常数）
    
    性能优化：归一化常数只计算一次，后续O(1)查表
    """
    cache_key = (num_items, exponent)
    
    # 检查缓存
    if cache_key not in _zipf_normalization_cache:
        _zipf_normalization_cache[cache_key] = sum(1 / (i ** exponent) for i in range(1, num_items + 1))
    
    normalization = _zipf_normalization_cache[cache_key]
    return (1 / (rank ** exponent)) / normalization


class MovingAverage:
    """移动平均计算器"""
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.values = []
    
    def update(self, value: float) -> float:
        """更新值并返回移动平均"""
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)
        return sum(self.values) / len(self.values)
    
    def get_average(self) -> float:
        """获取当前移动平均值"""
        if not self.values:
            return 0.0
        return sum(self.values) / len(self.values)


class ExponentialMovingAverage:
    """指数移动平均计算器"""
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha
        self.value = None
    
    def update(self, new_value: float) -> float:
        """更新值并返回指数移动平均"""
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value
    
    def get_value(self) -> float:
        """获取当前指数移动平均值"""
        return self.value if self.value is not None else 0.0