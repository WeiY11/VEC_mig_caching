#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化工具模块
为不同算法提供优化的批次大小和内存管理配置
"""

import gc
import time
import psutil
import numpy as np
from typing import Dict, Any, Optional, Tuple
from functools import wraps
import logging


# 优化的批次大小配置 - 基于内存中的性能配置
OPTIMIZED_BATCH_SIZES = {
    # 多智能体算法
    'MATD3': 384,
    'MADDPG': 384, 
    'MAPPO': 384,
    'QMIX': 48,
    'SAC-MA': 384,
    
    # 单智能体算法
    'DQN': 48,
    'DDPG': 192,
    'TD3': 192,
    'SAC': 384,
    'PPO': 96
}

# 内存管理配置
MEMORY_CONFIG = {
    'gc_frequency': 100,      # 每100步执行一次垃圾回收
    'max_memory_usage': 0.8,  # 最大内存使用率80%
    'buffer_size_limit': 100000,  # 经验回放缓冲区大小限制
    'state_cache_size': 1000,  # 状态缓存大小
}

# 计算优化配置
COMPUTE_CONFIG = {
    'use_vectorization': True,   # 使用向量化计算
    'batch_computation': True,   # 批量计算
    'memory_pinning': False,     # 内存固定 (GPU)
    'gradient_accumulation': 2,  # 梯度累积步数
}


class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.computation_times = []
        self.step_count = 0
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.step_count = 0
        
    def record_step(self, computation_time: float = None):
        """记录步骤性能"""
        self.step_count += 1
        
        # 记录内存使用
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        # 记录计算时间
        if computation_time is not None:
            self.computation_times.append(computation_time)
        
        # 定期清理历史记录
        if len(self.memory_usage) > 1000:
            self.memory_usage = self.memory_usage[-500:]
        if len(self.computation_times) > 1000:
            self.computation_times = self.computation_times[-500:]
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计"""
        total_time = time.time() - self.start_time if self.start_time else 0
        
        return {
            'total_steps': self.step_count,
            'total_time': total_time,
            'steps_per_second': self.step_count / max(total_time, 1e-6),
            'avg_memory_usage': np.mean(self.memory_usage) if self.memory_usage else 0,
            'max_memory_usage': max(self.memory_usage) if self.memory_usage else 0,
            'avg_computation_time': np.mean(self.computation_times) if self.computation_times else 0,
            'total_computation_time': sum(self.computation_times) if self.computation_times else 0
        }


class MemoryManager:
    """内存管理器"""
    
    def __init__(self):
        self.gc_counter = 0
        self.memory_threshold = MEMORY_CONFIG['max_memory_usage']
        
    def check_memory_usage(self) -> bool:
        """检查内存使用情况"""
        memory_percent = psutil.virtual_memory().percent / 100.0
        return memory_percent < self.memory_threshold
    
    def cleanup_if_needed(self, force: bool = False):
        """根据需要清理内存"""
        self.gc_counter += 1
        
        should_cleanup = (
            force or 
            self.gc_counter % MEMORY_CONFIG['gc_frequency'] == 0 or
            not self.check_memory_usage()
        )
        
        if should_cleanup:
            # 执行垃圾回收
            collected = gc.collect()
            
            # 记录清理结果
            memory_after = psutil.virtual_memory().percent
            logging.debug(f"内存清理: 回收 {collected} 个对象, 内存使用: {memory_after:.1f}%")
            
            return True
        
        return False
    
    def optimize_buffer_size(self, algorithm: str, current_size: int) -> int:
        """优化缓冲区大小"""
        # 根据内存使用情况动态调整
        memory_percent = psutil.virtual_memory().percent / 100.0
        
        if memory_percent > 0.9:
            # 内存紧张，减小缓冲区
            return min(current_size, MEMORY_CONFIG['buffer_size_limit'] // 2)
        elif memory_percent < 0.5:
            # 内存充足，可以增大缓冲区
            return min(current_size * 2, MEMORY_CONFIG['buffer_size_limit'])
        else:
            # 正常范围
            return min(current_size, MEMORY_CONFIG['buffer_size_limit'])


def performance_timer(func):
    """性能计时装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        computation_time = end_time - start_time
        
        # 如果实例有performance_monitor，记录时间
        if hasattr(args[0], 'performance_monitor'):
            args[0].performance_monitor.record_step(computation_time)
        
        return result
    return wrapper


def memory_efficient_batch_processing(data: np.ndarray, batch_size: int, 
                                     process_func: callable) -> list:
    """
    内存高效的批处理
    
    Args:
        data: 输入数据
        batch_size: 批次大小
        process_func: 处理函数
        
    Returns:
        处理结果列表
    """
    results = []
    memory_manager = MemoryManager()
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # 处理批次
        batch_result = process_func(batch)
        results.append(batch_result)
        
        # 内存管理
        if i % (batch_size * 10) == 0:  # 每10个批次检查一次
            memory_manager.cleanup_if_needed()
    
    return results


def get_optimal_batch_size(algorithm: str, available_memory_gb: float = None) -> int:
    """
    获取算法的最优批次大小
    
    Args:
        algorithm: 算法名称
        available_memory_gb: 可用内存(GB)
        
    Returns:
        最优批次大小
    """
    base_batch_size = OPTIMIZED_BATCH_SIZES.get(algorithm, 64)
    
    if available_memory_gb is None:
        # 自动检测可用内存
        memory_info = psutil.virtual_memory()
        available_memory_gb = memory_info.available / (1024**3)
    
    # 根据可用内存调整批次大小
    if available_memory_gb < 4:
        # 内存不足4GB，减半
        return base_batch_size // 2
    elif available_memory_gb < 8:
        # 内存4-8GB，保持原值
        return base_batch_size
    else:
        # 内存充足，可以适当增大
        return min(base_batch_size * 2, 512)  # 最大不超过512


def optimize_numpy_arrays(arrays: list) -> list:
    """
    优化numpy数组的内存使用
    
    Args:
        arrays: numpy数组列表
        
    Returns:
        优化后的数组列表
    """
    optimized = []
    
    for arr in arrays:
        if isinstance(arr, np.ndarray):
            # 转换为更紧凑的数据类型
            if arr.dtype == np.float64:
                # float64 -> float32 (如果精度允许)
                optimized.append(arr.astype(np.float32))
            elif arr.dtype == np.int64:
                # 尝试更小的整数类型
                if arr.max() < 32767 and arr.min() > -32768:
                    optimized.append(arr.astype(np.int16))
                elif arr.max() < 2147483647 and arr.min() > -2147483648:
                    optimized.append(arr.astype(np.int32))
                else:
                    optimized.append(arr)
            else:
                optimized.append(arr)
        else:
            optimized.append(arr)
    
    return optimized


def create_performance_optimized_config(algorithm: str) -> Dict[str, Any]:
    """
    创建性能优化配置
    
    Args:
        algorithm: 算法名称
        
    Returns:
        优化配置字典
    """
    return {
        'batch_size': get_optimal_batch_size(algorithm),
        'buffer_size': min(MEMORY_CONFIG['buffer_size_limit'], 
                          get_optimal_batch_size(algorithm) * 200),
        'memory_config': MEMORY_CONFIG.copy(),
        'compute_config': COMPUTE_CONFIG.copy(),
        'use_performance_monitor': True,
        'enable_memory_management': True,
    }


class OptimizedTrainingLoop:
    """优化的训练循环"""
    
    def __init__(self, algorithm: str):
        self.algorithm = algorithm
        self.config = create_performance_optimized_config(algorithm)
        self.performance_monitor = PerformanceMonitor()
        self.memory_manager = MemoryManager()
        
    @performance_timer
    def training_step(self, *args, **kwargs):
        """优化的训练步骤"""
        # 内存检查
        if not self.memory_manager.check_memory_usage():
            logging.warning(f"内存使用率过高，强制清理")
            self.memory_manager.cleanup_if_needed(force=True)
        
        # 实际训练逻辑由子类实现
        return self._execute_training_step(*args, **kwargs)
    
    def _execute_training_step(self, *args, **kwargs):
        """子类需要实现的训练步骤"""
        raise NotImplementedError
    
    def get_performance_report(self) -> str:
        """获取性能报告"""
        stats = self.performance_monitor.get_performance_stats()
        
        return f"""
🚀 性能报告 - {self.algorithm}
{'=' * 40}
总步数: {stats['total_steps']}
总时间: {stats['total_time']:.2f}s
步数/秒: {stats['steps_per_second']:.2f}
平均内存使用: {stats['avg_memory_usage']:.1f}%
最大内存使用: {stats['max_memory_usage']:.1f}%
平均计算时间: {stats['avg_computation_time']:.4f}s
总计算时间: {stats['total_computation_time']:.2f}s
"""


# 全局性能监控器
global_performance_monitor = PerformanceMonitor()


def get_system_performance_info() -> Dict:
    """获取系统性能信息"""
    memory = psutil.virtual_memory()
    cpu = psutil.cpu_percent(interval=1)
    
    return {
        'cpu_percent': cpu,
        'memory_total_gb': memory.total / (1024**3),
        'memory_available_gb': memory.available / (1024**3),
        'memory_percent': memory.percent,
        'recommended_batch_sizes': OPTIMIZED_BATCH_SIZES.copy()
    }