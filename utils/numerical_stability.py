#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数值稳定性工具模块
提供各种数值计算的稳定性保障
"""

import numpy as np
import math
from typing import Union, Optional

# 数值稳定性常量
EPSILON = 1e-12  # 极小值阈值
MAX_SAFE_VALUE = 1e10  # 最大安全值
MIN_SAFE_VALUE = 1e-10  # 最小安全值

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 分母为零时的默认返回值
        
    Returns:
        安全的除法结果
    """
    if abs(denominator) < EPSILON:
        return default
    
    result = numerator / denominator
    
    # 检查结果是否超出安全范围
    if abs(result) > MAX_SAFE_VALUE:
        return MAX_SAFE_VALUE if result > 0 else -MAX_SAFE_VALUE
    
    return result

def safe_sqrt(value: float) -> float:
    """
    安全平方根计算
    
    Args:
        value: 输入值
        
    Returns:
        安全的平方根结果
    """
    if value < 0:
        return 0.0
    
    return math.sqrt(max(value, 0.0))

def safe_log(value: float, base: Optional[float] = None) -> float:
    """
    安全对数计算
    
    Args:
        value: 输入值
        base: 对数底数，None表示自然对数
        
    Returns:
        安全的对数结果
    """
    if value <= 0:
        return -MAX_SAFE_VALUE
    
    if base is None:
        return math.log(max(value, MIN_SAFE_VALUE))
    else:
        if base <= 0 or base == 1:
            return 0.0
        return math.log(max(value, MIN_SAFE_VALUE)) / math.log(base)

def safe_exp(value: float) -> float:
    """
    安全指数计算
    
    Args:
        value: 输入值
        
    Returns:
        安全的指数结果
    """
    # 限制输入范围防止溢出
    if value > 700:  # exp(700) 接近 float64 上限
        return MAX_SAFE_VALUE
    elif value < -700:
        return MIN_SAFE_VALUE
    
    return math.exp(value)

def safe_power(base: float, exponent: float) -> float:
    """
    安全幂运算
    
    Args:
        base: 底数
        exponent: 指数
        
    Returns:
        安全的幂运算结果
    """
    if base == 0:
        return 0.0 if exponent > 0 else 1.0
    
    if base < 0 and not isinstance(exponent, int):
        # 负数的非整数次幂，返回绝对值的幂
        base = abs(base)
    
    try:
        # 使用对数计算避免直接幂运算溢出
        if base > 0:
            log_result = exponent * math.log(base)
            if abs(log_result) > 700:
                return MAX_SAFE_VALUE if log_result > 0 else MIN_SAFE_VALUE
            return math.exp(log_result)
        else:
            return pow(base, exponent)
    except (OverflowError, ValueError):
        return MAX_SAFE_VALUE if base > 1 and exponent > 0 else MIN_SAFE_VALUE

def clamp(value: float, min_val: float = -MAX_SAFE_VALUE, max_val: float = MAX_SAFE_VALUE) -> float:
    """
    将值限制在指定范围内
    
    Args:
        value: 输入值
        min_val: 最小值
        max_val: 最大值
        
    Returns:
        限制后的值
    """
    return max(min_val, min(value, max_val))

def validate_sinr(sinr: float) -> float:
    """
    验证和修正SINR值
    
    Args:
        sinr: 信噪比值
        
    Returns:
        修正后的SINR值
    """
    if np.isnan(sinr) or np.isinf(sinr):
        return MIN_SAFE_VALUE
    
    return clamp(sinr, MIN_SAFE_VALUE, MAX_SAFE_VALUE)

def validate_energy(energy: float) -> float:
    """
    验证能耗值
    
    Args:
        energy: 能耗值
        
    Returns:
        修正后的能耗值
    """
    if np.isnan(energy) or np.isinf(energy) or energy < 0:
        return 0.0
    
    return clamp(energy, 0.0, MAX_SAFE_VALUE)

def validate_delay(delay: float) -> float:
    """
    验证延迟值
    
    Args:
        delay: 延迟值
        
    Returns:
        修正后的延迟值
    """
    if np.isnan(delay) or np.isinf(delay) or delay < 0:
        return 0.0
    
    return clamp(delay, 0.0, MAX_SAFE_VALUE)

def validate_probability(prob: float) -> float:
    """
    验证概率值
    
    Args:
        prob: 概率值
        
    Returns:
        修正后的概率值 (0-1)
    """
    if np.isnan(prob) or np.isinf(prob):
        return 0.5  # 默认概率
    
    return clamp(prob, 0.0, 1.0)

def validate_cpu_frequency(freq: float) -> float:
    """
    验证CPU频率值
    
    Args:
        freq: CPU频率 (Hz)
        
    Returns:
        修正后的CPU频率
    """
    if np.isnan(freq) or np.isinf(freq) or freq <= 0:
        return 1e9  # 默认1GHz
    
    # CPU频率合理范围: 100MHz - 100GHz (扩展范围以支持内存规范)
    return clamp(freq, 1e8, 1e12)

def validate_data_size(size: float) -> float:
    """
    验证数据大小
    
    Args:
        size: 数据大小 (bytes)
        
    Returns:
        修正后的数据大小
    """
    if np.isnan(size) or np.isinf(size) or size < 0:
        return 1e6  # 默认1MB
    
    # 数据大小合理范围: 1KB - 1GB
    return clamp(size, 1e3, 1e9)

def check_numerical_health(value: float, name: str = "value") -> bool:
    """
    检查数值健康状态
    
    Args:
        value: 要检查的值
        name: 值的名称（用于错误报告）
        
    Returns:
        是否健康
    """
    if np.isnan(value):
        print(f"⚠️ 检测到NaN值: {name}")
        return False
    
    if np.isinf(value):
        print(f"⚠️ 检测到无穷值: {name}")
        return False
    
    if abs(value) > MAX_SAFE_VALUE:
        print(f"⚠️ 值过大: {name} = {value}")
        return False
    
    return True

class NumericalStabilityMonitor:
    """数值稳定性监控器"""
    
    def __init__(self):
        self.warning_count = 0
        self.error_count = 0
        self.max_warnings = 100
    
    def check_and_fix(self, value: float, name: str, validator_func=None) -> float:
        """
        检查并修复数值
        
        Args:
            value: 原始值
            name: 值名称
            validator_func: 验证函数
            
        Returns:
            修复后的值
        """
        if not check_numerical_health(value, name):
            self.warning_count += 1
            if self.warning_count <= self.max_warnings:
                print(f"🔧 自动修复数值问题: {name}")
        
        if validator_func:
            return validator_func(value)
        else:
            return clamp(value)
    
    def get_statistics(self) -> dict:
        """获取监控统计"""
        return {
            'warning_count': self.warning_count,
            'error_count': self.error_count
        }

# 全局监控器实例
numerical_monitor = NumericalStabilityMonitor()
