#!/usr/bin/env python3
"""
统一时间管理系统
解决仿真时间和系统时间不一致的问题
"""

import time
from typing import Optional

class UnifiedTimeManager:
    """统一的时间管理器"""
    
    def __init__(self):
        self.simulation_start_time = time.time()
        self.simulation_time_elapsed = 0.0
        self.time_scale = 1.0  # 时间加速比例
        self.current_step = 0
        self.time_slot_duration = 0.1  # 每个时隙0.1秒
        
    def reset(self):
        """重置时间管理器"""
        self.simulation_start_time = time.time()
        self.simulation_time_elapsed = 0.0
        self.current_step = 0
    
    def advance_step(self):
        """推进一个仿真步骤"""
        self.current_step += 1
        self.simulation_time_elapsed = self.current_step * self.time_slot_duration
    
    def get_simulation_time(self) -> float:
        """获取仿真时间(秒)"""
        return self.simulation_time_elapsed
    
    def get_real_time_elapsed(self) -> float:
        """获取真实经过时间(秒)"""
        return time.time() - self.simulation_start_time
    
    def get_simulation_hour(self) -> int:
        """获取仿真时间对应的小时(0-23)"""
        # 将仿真时间映射到一天24小时
        hours_elapsed = self.simulation_time_elapsed / 3600
        return int(hours_elapsed % 24)
    
    def get_time_slot_in_hour(self) -> int:
        """获取当前小时内的时隙编号"""
        time_slots_per_hour = int(3600 / self.time_slot_duration)  # 每小时18000个时隙
        return self.current_step % time_slots_per_hour
    
    def time_since_simulation_start(self) -> float:
        """获取自仿真开始以来的时间(仿真时间)"""
        return self.simulation_time_elapsed
    
    def is_within_time_window(self, start_time: float, window_duration: float) -> bool:
        """检查当前是否在指定时间窗口内"""
        current_time = self.get_simulation_time()
        return start_time <= current_time <= (start_time + window_duration)

# 全局时间管理器实例
_global_time_manager = UnifiedTimeManager()

def get_simulation_time() -> float:
    """获取当前仿真时间"""
    return _global_time_manager.get_simulation_time()

def advance_simulation_time():
    """推进仿真时间"""
    _global_time_manager.advance_step()

def reset_simulation_time():
    """重置仿真时间"""
    _global_time_manager.reset()

def get_time_manager() -> UnifiedTimeManager:
    """获取时间管理器实例"""
    return _global_time_manager
