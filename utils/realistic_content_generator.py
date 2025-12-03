#!/usr/bin/env python3
"""
Realistic VEC内容生成器
为仿真生成符合真实VEC场景的内容请求
"""

import numpy as np
import random
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum

class VECContentType(Enum):
    """VEC内容类型"""
    TRAFFIC_INFO = "traffic_info"
    NAVIGATION = "navigation"  
    SAFETY_ALERT = "safety_alert"
    PARKING_INFO = "parking_info"
    WEATHER_INFO = "weather_info"
    MAP_DATA = "map_data"
    ENTERTAINMENT = "entertainment"
    SENSOR_DATA = "sensor_data"

@dataclass
class VECContentSpec:
    """VEC内容规格"""
    content_type: VECContentType
    size_mb: float
    priority: int  # 1-5
    freshness_duration: float  # 秒
    access_pattern: str  # 'frequent', 'burst', 'periodic', 'rare'

class RealisticContentGenerator:
    """Realistic内容生成器"""
    
    def __init__(self):
        # VEC内容类型规格
        # 🔧 P0修复：对齐数据大小与TaskConfig.task_profiles定义
        # task_profiles数据范围：
        #   类型1: 50KB-200KB
        #   类型2: 600KB-1.5MB
        #   类型3: 2MB-4MB
        #   类型4: 4.5MB-8MB
        self.content_specs = {
            # 类型1: 极度敏感 (50KB-200KB)
            VECContentType.SAFETY_ALERT: VECContentSpec(
                content_type=VECContentType.SAFETY_ALERT,
                size_mb=0.15,          # 150KB - 安全警报 (类型1: 50-200KB)
                priority=5,             # 最高优先级
                freshness_duration=30.0,  # 30秒有效期
                access_pattern='burst'
            ),
            VECContentType.SENSOR_DATA: VECContentSpec(
                content_type=VECContentType.SENSOR_DATA,
                size_mb=0.25,          # 250KB - 传感器数据 (类型1: 200-300KB)
                priority=4,             # 高优先级
                freshness_duration=10.0,  # 10秒有效期
                access_pattern='frequent'
            ),
            
            # 类型2: 敏感 (600KB-1.5MB)
            VECContentType.NAVIGATION: VECContentSpec(
                content_type=VECContentType.NAVIGATION,
                size_mb=1.2,           # 1.2MB - 路径数据 (类型2: 1-2MB)
                priority=4,             # 高优先级
                freshness_duration=300.0,  # 5分钟有效期
                access_pattern='burst'
            ),
            VECContentType.WEATHER_INFO: VECContentSpec(
                content_type=VECContentType.WEATHER_INFO,
                size_mb=1.5,           # 1.5MB - 天气数据 (类型2: 1-2MB)
                priority=2,             # 低优先级
                freshness_duration=1800.0, # 30分钟有效期
                access_pattern='periodic'
            ),
            
            # 类型3: 中度容忍 (2MB-4MB)
            VECContentType.MAP_DATA: VECContentSpec(
                content_type=VECContentType.MAP_DATA,
                size_mb=3.5,           # 3.5MB - 地图瓦片 (类型3: 3-5MB)
                priority=3,             # 中等优先级
                freshness_duration=3600.0, # 1小时有效期
                access_pattern='periodic'
            ),
            VECContentType.PARKING_INFO: VECContentSpec(
                content_type=VECContentType.PARKING_INFO,
                size_mb=4.0,           # 4.0MB - 停车信息 (类型3: 3-5MB)
                priority=3,             # 中等优先级
                freshness_duration=600.0, # 10分钟有效期
                access_pattern='periodic'
            ),
            
            # 类型4: 容忍 (4.5MB-8MB)
            VECContentType.TRAFFIC_INFO: VECContentSpec(
                content_type=VECContentType.TRAFFIC_INFO,
                size_mb=6.0,           # 6.0MB - 实时交通数据 (类型4: 5-8MB)
                priority=3,             # 中等优先级
                freshness_duration=60.0,  # 1分钟有效期
                access_pattern='frequent'
            ),
            VECContentType.ENTERTAINMENT: VECContentSpec(
                content_type=VECContentType.ENTERTAINMENT,
                size_mb=8.0,           # 8.0MB - 视频/音乐 (类型4: 5-10MB)
                priority=1,             # 最低优先级
                freshness_duration=7200.0, # 2小时有效期
                access_pattern='rare'
            ),
        }
        
        # 内容生成权重（基于现实使用频率，但调整以匹配任务类型分布）
        # 🔧 P1修复：调整为符合VEC实际场景的分布
        # 目标：类型1(极度敏感)=35%, 类型2(敏感)=25%, 类型3(中度)=25%, 类型4(容忍)=15%
        # 合理性：紧急任务应占更高比例，低优先级任务应较少
        self.content_weights = {
            # 类型1: 极度敏感 - 35%
            VECContentType.SAFETY_ALERT: 0.20,    # 20% -> 类型1 (安全警报)
            VECContentType.SENSOR_DATA: 0.15,     # 15% -> 类型1 (传感器数据)
            
            # 类型2: 敏感 - 25%
            VECContentType.NAVIGATION: 0.15,      # 15% -> 类型2 (导航)
            VECContentType.WEATHER_INFO: 0.10,    # 10% -> 类型2 (天气信息)
            
            # 类型3: 中度容忍 - 25%
            VECContentType.MAP_DATA: 0.15,        # 15% -> 类型3 (地图数据)
            VECContentType.PARKING_INFO: 0.10,    # 10% -> 类型3 (停车信息)
            
            # 类型4: 容忍 - 15%
            VECContentType.TRAFFIC_INFO: 0.10,    # 10% -> 类型4 (交通信息)
            VECContentType.ENTERTAINMENT: 0.05,   # 5%  -> 类型4 (娱乐)
        }
        
        # 内容ID计数器
        self.content_counters = {ct: 0 for ct in VECContentType}
        # 控制内容目录规模，促进内容复用以产生缓存命中
        try:
            import os
            self.catalog_size = int(os.environ.get("CONTENT_CATALOG_SIZE", "200"))
            self.catalog_size = max(50, self.catalog_size)
        except Exception:
            self.catalog_size = 200
    
    def generate_content_request(self, vehicle_id: str, step: int) -> Tuple[str, VECContentSpec]:
        """
        生成realistic的内容请求
        
        Returns:
            (content_id, content_spec)
        """
        # 根据权重选择内容类型
        content_types = list(self.content_weights.keys())
        weights = list(self.content_weights.values())
        
        selected_type = np.random.choice(content_types, p=weights)
        
        # 生成内容ID（循环有限目录，移除车辆维度以提高跨车辆复用）
        self.content_counters[selected_type] += 1
        counter_mod = self.content_counters[selected_type] % self.catalog_size
        content_id = f"{selected_type.value}_{counter_mod:04d}"
        
        # 获取内容规格
        base_spec = self.content_specs[selected_type]
        
        # 添加一些随机变化
        size_variation = np.random.uniform(0.8, 1.2)
        actual_size = base_spec.size_mb * size_variation
        
        # 创建实际规格
        actual_spec = VECContentSpec(
            content_type=selected_type,
            size_mb=actual_size,
            priority=base_spec.priority,
            freshness_duration=base_spec.freshness_duration,
            access_pattern=base_spec.access_pattern
        )
        
        return content_id, actual_spec
    
    def get_content_size(self, content_id: str) -> float:
        """根据内容ID获取大小"""
        # 从ID推断内容类型
        for content_type, spec in self.content_specs.items():
            if content_type.value in content_id.lower():
                # 添加一些随机变化
                return spec.size_mb * np.random.uniform(0.9, 1.1)
        
        return 1.0  # 默认大小
    
    def get_realistic_cache_statistics(self, cache: Dict) -> Dict:
        """获取realistic的缓存统计"""
        if not cache:
            return {
                'total_items': 0,
                'total_size_mb': 0.0,
                'content_distribution': {},
                'avg_item_size': 0.0
            }
        
        total_size_mb = 0.0
        content_distribution = {}
        
        for content_id, item in cache.items():
            if isinstance(item, dict):
                size_mb = item.get('size', 1.0)
                content_type = item.get('content_type', 'general')
            else:
                size_mb = self.get_content_size(content_id)
                content_type = self._infer_content_type(content_id)
            
            total_size_mb += size_mb
            content_distribution[content_type] = content_distribution.get(content_type, 0) + 1
        
        return {
            'total_items': len(cache),
            'total_size_mb': total_size_mb,
            'content_distribution': content_distribution,
            'avg_item_size': total_size_mb / len(cache) if cache else 0.0
        }
    
    def _infer_content_type(self, content_id: str) -> str:
        """推断内容类型"""
        content_id_lower = content_id.lower()
        
        for content_type in VECContentType:
            if content_type.value in content_id_lower:
                return content_type.value
        
        return 'general'

# 全局内容生成器
_global_content_generator = RealisticContentGenerator()

def generate_realistic_content(vehicle_id: str, step: int) -> Tuple[str, float, int]:
    """
    生成realistic内容请求
    
    Returns:
        (content_id, size_mb, priority)
    """
    content_id, spec = _global_content_generator.generate_content_request(vehicle_id, step)
    return content_id, spec.size_mb, spec.priority

def get_realistic_content_size(content_id: str) -> float:
    """获取realistic内容大小"""
    return _global_content_generator.get_content_size(content_id)
