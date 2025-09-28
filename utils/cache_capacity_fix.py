#!/usr/bin/env python3
"""
缓存容量计算修复工具
解决单位不匹配的根本性错误
"""

from typing import Dict, Any

def calculate_correct_cache_utilization(cache: Dict, cache_capacity_mb: float) -> float:
    """
    正确计算缓存利用率
    
    Args:
        cache: 缓存字典 {content_id: {'size': float, ...}}
        cache_capacity_mb: 缓存容量(MB)
    
    Returns:
        缓存利用率 [0.0, 1.0]
    """
    if not cache or cache_capacity_mb <= 0:
        return 0.0
    
    # 计算已使用的缓存大小(MB)
    total_used_mb = 0.0
    for item in cache.values():
        if isinstance(item, dict) and 'size' in item:
            total_used_mb += float(item.get('size', 0.0))
        else:
            # 兼容简单格式，默认1MB
            total_used_mb += 1.0
    
    utilization = total_used_mb / cache_capacity_mb
    return min(1.0, max(0.0, utilization))  # 限制在[0,1]范围内

def calculate_available_cache_capacity(cache: Dict, cache_capacity_mb: float) -> float:
    """
    正确计算可用缓存容量
    
    Returns:
        可用容量(MB)
    """
    if not cache or cache_capacity_mb <= 0:
        return cache_capacity_mb
    
    total_used_mb = 0.0
    for item in cache.values():
        if isinstance(item, dict) and 'size' in item:
            total_used_mb += float(item.get('size', 0.0))
        else:
            total_used_mb += 1.0
    
    available_mb = cache_capacity_mb - total_used_mb
    return max(0.0, available_mb)

def get_realistic_content_size(content_type: str) -> float:
    """
    根据内容类型返回realistic大小(MB)
    """
    content_size_map = {
        'traffic_info': 0.1,      # 交通信息：100KB
        'navigation': 0.5,        # 导航数据：500KB
        'safety_alert': 0.05,     # 安全警报：50KB
        'parking_info': 0.2,      # 停车信息：200KB
        'weather_info': 0.3,      # 天气信息：300KB
        'map_data': 10.0,         # 地图数据：10MB
        'entertainment': 50.0,    # 娱乐内容：50MB
        'sensor_data': 0.1,       # 传感器数据：100KB
        'default': 1.0            # 默认：1MB
    }
    
    return content_size_map.get(content_type, content_size_map['default'])

def fix_cache_in_simulator_data(simulator_data: Dict) -> Dict:
    """
    修复仿真器中的缓存数据结构
    """
    fixed_data = simulator_data.copy()
    
    # 修复RSU缓存
    if 'rsus' in fixed_data:
        for rsu in fixed_data['rsus']:
            if 'cache' in rsu:
                fixed_cache = {}
                for content_id, content_info in rsu['cache'].items():
                    if isinstance(content_info, dict):
                        # 已经是正确格式
                        fixed_cache[content_id] = content_info
                    else:
                        # 转换为正确格式
                        fixed_cache[content_id] = {
                            'size': get_realistic_content_size('default'),
                            'timestamp': 0.0,
                            'reason': 'legacy_data'
                        }
                rsu['cache'] = fixed_cache
            
            # 确保有缓存容量配置
            if 'cache_capacity' not in rsu:
                rsu['cache_capacity'] = 1000.0  # 1GB for RSU
    
    # 修复UAV缓存
    if 'uavs' in fixed_data:
        for uav in fixed_data['uavs']:
            if 'cache' in uav:
                fixed_cache = {}
                for content_id, content_info in uav['cache'].items():
                    if isinstance(content_info, dict):
                        fixed_cache[content_id] = content_info
                    else:
                        fixed_cache[content_id] = {
                            'size': get_realistic_content_size('default'),
                            'timestamp': 0.0,
                            'reason': 'legacy_data'
                        }
                uav['cache'] = fixed_cache
            
            if 'cache_capacity' not in uav:
                uav['cache_capacity'] = 200.0  # 200MB for UAV
    
    return fixed_data
