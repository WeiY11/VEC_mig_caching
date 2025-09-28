#!/usr/bin/env python3
"""
现实VEC缓存仿真系统
基于真实车联网场景的缓存内容和用户行为建模
"""

import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
import math

class ContentType(Enum):
    """VEC内容类型枚举"""
    TRAFFIC_INFO = "traffic_info"          # 交通信息
    MAP_DATA = "map_data"                  # 地图数据
    SAFETY_ALERT = "safety_alert"          # 安全警报
    ENTERTAINMENT = "entertainment"         # 娱乐内容
    NAVIGATION = "navigation"               # 导航数据
    SENSOR_DATA = "sensor_data"            # 传感器数据
    WEATHER_INFO = "weather_info"          # 天气信息
    PARKING_INFO = "parking_info"          # 停车信息

@dataclass
class VECContent:
    """VEC内容项"""
    content_id: str
    content_type: ContentType
    size_mb: float
    freshness_lifetime: float  # 内容有效期（秒）
    location_relevance: Tuple[float, float]  # 地理相关性(lat, lon)
    relevance_radius: float  # 相关半径（米）
    priority_level: int  # 优先级 1-5
    creation_time: float
    is_real_time: bool = False  # 是否实时内容
    popularity_score: float = 0.0  # 流行度分数
    
    def is_fresh(self, current_time: float) -> bool:
        """检查内容是否仍然新鲜"""
        return (current_time - self.creation_time) < self.freshness_lifetime
    
    def location_distance(self, lat: float, lon: float) -> float:
        """计算到指定位置的距离"""
        # 简化的距离计算（使用欧几里得距离）
        dlat = self.location_relevance[0] - lat
        dlon = self.location_relevance[1] - lon
        return math.sqrt(dlat**2 + dlon**2) * 111000  # 转换为米

@dataclass 
class Vehicle:
    """车辆状态"""
    vehicle_id: str
    position: Tuple[float, float]  # (lat, lon)
    speed: float  # km/h
    direction: float  # 度数
    route: List[Tuple[float, float]]  # 路线点
    preferences: Dict[ContentType, float]  # 内容偏好权重


class RealisticVECCacheSimulator:
    """
    现实VEC缓存仿真器
    基于真实车联网场景建模
    """
    
    def __init__(self, node_id: str, node_type: str, position: Tuple[float, float]):
        self.node_id = node_id
        self.node_type = node_type  # 'vehicle', 'rsu', 'uav'
        self.position = position
        
        # 容量配置
        capacity_map = {
            'vehicle': 200.0,    # 200MB
            'rsu': 2000.0,      # 2GB  
            'uav': 500.0        # 500MB
        }
        self.capacity = capacity_map.get(node_type, 200.0)
        self.current_usage = 0.0
        
        # 缓存存储
        self.cached_contents: Dict[str, VECContent] = {}
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        
        # 地理相关性配置
        self.relevance_radius = {
            'vehicle': 2000.0,   # 2km
            'rsu': 5000.0,      # 5km
            'uav': 10000.0      # 10km
        }.get(node_type, 2000.0)
        
        # 内容类型权重（基于节点类型）
        self.content_type_weights = self._get_content_weights()
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'location_hits': 0,     # 地理相关命中
            'fresh_hits': 0,        # 新鲜内容命中
            'stale_misses': 0,      # 过期内容未命中
            'distance_misses': 0,   # 距离过远未命中
            'capacity_evictions': 0  # 容量不足驱逐
        }
        
        # 实时内容生成器
        self.real_time_contents = {}
        self.last_traffic_update = 0
        self.last_weather_update = 0
        
    def _get_content_weights(self) -> Dict[ContentType, float]:
        """根据节点类型获取内容权重"""
        if self.node_type == 'vehicle':
            return {
                ContentType.TRAFFIC_INFO: 0.8,
                ContentType.NAVIGATION: 0.9,
                ContentType.SAFETY_ALERT: 1.0,
                ContentType.ENTERTAINMENT: 0.6,
                ContentType.MAP_DATA: 0.7,
                ContentType.PARKING_INFO: 0.8,
                ContentType.WEATHER_INFO: 0.5,
                ContentType.SENSOR_DATA: 0.3
            }
        elif self.node_type == 'rsu':
            return {
                ContentType.TRAFFIC_INFO: 1.0,
                ContentType.MAP_DATA: 0.9,
                ContentType.SAFETY_ALERT: 1.0,
                ContentType.SENSOR_DATA: 0.8,
                ContentType.WEATHER_INFO: 0.7,
                ContentType.PARKING_INFO: 0.9,
                ContentType.NAVIGATION: 0.6,
                ContentType.ENTERTAINMENT: 0.3
            }
        else:  # UAV
            return {
                ContentType.TRAFFIC_INFO: 0.9,
                ContentType.WEATHER_INFO: 1.0,
                ContentType.SENSOR_DATA: 0.9,
                ContentType.SAFETY_ALERT: 0.8,
                ContentType.MAP_DATA: 0.7,
                ContentType.NAVIGATION: 0.5,
                ContentType.PARKING_INFO: 0.4,
                ContentType.ENTERTAINMENT: 0.2
            }
    
    def generate_realistic_content_request(self, requester_pos: Tuple[float, float], 
                                         vehicle_profile: Optional[Vehicle] = None) -> VECContent:
        """
        生成现实的内容请求
        基于地理位置、车辆特征和时间模式
        """
        current_time = time.time()
        
        # 根据时间和位置确定内容类型分布
        content_probabilities = self._calculate_content_probabilities(
            requester_pos, current_time, vehicle_profile
        )
        
        # 选择内容类型
        content_types = list(content_probabilities.keys())
        probabilities = list(content_probabilities.values())
        selected_type = np.random.choice(content_types, p=probabilities)
        
        # 生成具体内容
        content = self._generate_content_by_type(selected_type, requester_pos, current_time)
        
        return content
    
    def _calculate_content_probabilities(self, position: Tuple[float, float], 
                                       current_time: float,
                                       vehicle_profile: Optional[Vehicle] = None) -> Dict[ContentType, float]:
        """计算内容类型概率分布"""
        base_probs = {
            ContentType.TRAFFIC_INFO: 0.25,
            ContentType.MAP_DATA: 0.15,
            ContentType.SAFETY_ALERT: 0.1,
            ContentType.NAVIGATION: 0.2,
            ContentType.ENTERTAINMENT: 0.1,
            ContentType.PARKING_INFO: 0.08,
            ContentType.WEATHER_INFO: 0.07,
            ContentType.SENSOR_DATA: 0.05
        }
        
        # 时间因素调整
        hour = int((current_time % 86400) / 3600)
        if 7 <= hour <= 9 or 17 <= hour <= 19:  # 高峰时段
            base_probs[ContentType.TRAFFIC_INFO] *= 2.0
            base_probs[ContentType.PARKING_INFO] *= 1.5
        elif 22 <= hour or hour <= 6:  # 夜间
            base_probs[ContentType.ENTERTAINMENT] *= 1.5
            base_probs[ContentType.TRAFFIC_INFO] *= 0.5
        
        # 车辆偏好调整
        if vehicle_profile and vehicle_profile.preferences:
            for content_type, weight in vehicle_profile.preferences.items():
                if content_type in base_probs:
                    base_probs[content_type] *= weight
        
        # 地理位置调整（示例：市中心更多停车信息）
        lat, lon = position
        if self._is_urban_area(lat, lon):
            base_probs[ContentType.PARKING_INFO] *= 1.8
            base_probs[ContentType.TRAFFIC_INFO] *= 1.3
        
        # 归一化
        total = sum(base_probs.values())
        return {k: v/total for k, v in base_probs.items()}
    
    def _generate_content_by_type(self, content_type: ContentType, 
                                position: Tuple[float, float], 
                                current_time: float) -> VECContent:
        """根据类型生成具体内容"""
        
        # 内容配置映射
        content_configs = {
            ContentType.TRAFFIC_INFO: {
                'size_range': (0.1, 0.5),
                'lifetime_range': (30, 180),   # 30秒-3分钟
                'radius_range': (500, 2000),
                'priority': 4,
                'is_real_time': True
            },
            ContentType.MAP_DATA: {
                'size_range': (5.0, 20.0),
                'lifetime_range': (3600, 86400),  # 1小时-1天
                'radius_range': (1000, 5000),
                'priority': 3,
                'is_real_time': False
            },
            ContentType.SAFETY_ALERT: {
                'size_range': (0.05, 0.2),
                'lifetime_range': (10, 60),     # 10秒-1分钟
                'radius_range': (200, 1000),
                'priority': 5,
                'is_real_time': True
            },
            ContentType.ENTERTAINMENT: {
                'size_range': (10.0, 100.0),
                'lifetime_range': (7200, 259200),  # 2小时-3天
                'radius_range': (0, 0),  # 不依赖地理位置
                'priority': 1,
                'is_real_time': False
            },
            ContentType.NAVIGATION: {
                'size_range': (1.0, 5.0),
                'lifetime_range': (300, 1800),   # 5-30分钟
                'radius_range': (1000, 3000),
                'priority': 4,
                'is_real_time': True
            },
            ContentType.PARKING_INFO: {
                'size_range': (0.2, 1.0),
                'lifetime_range': (60, 600),     # 1-10分钟
                'radius_range': (200, 500),
                'priority': 3,
                'is_real_time': True
            },
            ContentType.WEATHER_INFO: {
                'size_range': (0.5, 2.0),
                'lifetime_range': (1800, 3600),  # 30分钟-1小时
                'radius_range': (5000, 20000),
                'priority': 2,
                'is_real_time': True
            },
            ContentType.SENSOR_DATA: {
                'size_range': (0.1, 1.0),
                'lifetime_range': (5, 30),       # 5-30秒
                'radius_range': (100, 500),
                'priority': 3,
                'is_real_time': True
            }
        }
        
        config = content_configs[content_type]
        
        # 生成内容属性
        size_mb = np.random.uniform(*config['size_range'])
        lifetime = np.random.uniform(*config['lifetime_range'])
        
        if config['radius_range'][0] > 0:
            radius = np.random.uniform(*config['radius_range'])
            # 添加一些随机偏移
            lat_offset = np.random.uniform(-0.01, 0.01)
            lon_offset = np.random.uniform(-0.01, 0.01)
            content_position = (position[0] + lat_offset, position[1] + lon_offset)
        else:
            radius = 0
            content_position = (0, 0)  # 全局内容
        
        # 生成唯一ID
        content_id = f"{content_type.value}_{int(current_time)}_{random.randint(1000,9999)}"
        
        # 计算流行度分数
        popularity = self._calculate_popularity_score(content_type, position, current_time)
        
        return VECContent(
            content_id=content_id,
            content_type=content_type,
            size_mb=size_mb,
            freshness_lifetime=lifetime,
            location_relevance=content_position,
            relevance_radius=radius,
            priority_level=config['priority'],
            creation_time=current_time,
            is_real_time=config['is_real_time'],
            popularity_score=popularity
        )
    
    def _calculate_popularity_score(self, content_type: ContentType, 
                                  position: Tuple[float, float], 
                                  current_time: float) -> float:
        """计算内容流行度分数"""
        base_popularity = {
            ContentType.TRAFFIC_INFO: 0.8,
            ContentType.NAVIGATION: 0.7,
            ContentType.SAFETY_ALERT: 0.9,
            ContentType.MAP_DATA: 0.6,
            ContentType.ENTERTAINMENT: 0.4,
            ContentType.PARKING_INFO: 0.5,
            ContentType.WEATHER_INFO: 0.3,
            ContentType.SENSOR_DATA: 0.2
        }
        
        popularity = base_popularity.get(content_type, 0.5)
        
        # 时间调整
        hour = int((current_time % 86400) / 3600)
        if content_type == ContentType.TRAFFIC_INFO and (7 <= hour <= 9 or 17 <= hour <= 19):
            popularity *= 1.5
        elif content_type == ContentType.ENTERTAINMENT and (19 <= hour <= 23):
            popularity *= 1.3
        
        # 添加随机因素
        popularity *= np.random.uniform(0.8, 1.2)
        
        return min(1.0, popularity)
    
    def _is_urban_area(self, lat: float, lon: float) -> bool:
        """简单的城市区域判断"""
        # 这里可以集成真实的城市边界数据
        # 现在使用简单的随机判断
        return np.random.random() < 0.6
    
    def request_content(self, content: VECContent, requester_pos: Tuple[float, float]) -> Tuple[bool, str, Dict]:
        """
        处理内容请求
        Returns: (cache_hit, action_description, metrics)
        """
        current_time = time.time()
        self.stats['total_requests'] += 1
        
        # 检查缓存中是否存在
        if content.content_id in self.cached_contents:
            cached_content = self.cached_contents[content.content_id]
            
            # 检查内容是否仍然新鲜
            if cached_content.is_fresh(current_time):
                # 检查地理相关性
                distance = cached_content.location_distance(*requester_pos)
                if cached_content.relevance_radius == 0 or distance <= cached_content.relevance_radius:
                    # 成功命中
                    self.stats['cache_hits'] += 1
                    self.stats['fresh_hits'] += 1
                    if distance <= cached_content.relevance_radius:
                        self.stats['location_hits'] += 1
                    
                    # 更新访问模式
                    self.access_patterns[content.content_id].append(current_time)
                    
                    metrics = {
                        'hit_type': 'cache_hit',
                        'freshness': 'fresh',
                        'distance': distance,
                        'content_age': current_time - cached_content.creation_time
                    }
                    
                    return True, f"Cache Hit - Fresh {content.content_type.value}", metrics
                else:
                    # 地理位置不相关
                    self.stats['distance_misses'] += 1
                    metrics = {'hit_type': 'distance_miss', 'distance': distance}
                    return False, f"Cache Miss - Distance too far ({distance:.0f}m)", metrics
            else:
                # 内容过期
                self.stats['stale_misses'] += 1
                self._evict_content(content.content_id)
                metrics = {
                    'hit_type': 'stale_miss', 
                    'content_age': current_time - cached_content.creation_time
                }
                return False, f"Cache Miss - Stale content", metrics
        
        # 缓存未命中，决定是否缓存
        should_cache, cache_reason = self._should_cache_content(content, requester_pos)
        
        if should_cache:
            success = self._add_content_to_cache(content)
            if success:
                metrics = {'hit_type': 'cache_miss_cached', 'cache_reason': cache_reason}
                return False, f"Cache Miss - Cached: {cache_reason}", metrics
            else:
                metrics = {'hit_type': 'cache_miss_no_space', 'cache_reason': 'No space'}
                return False, f"Cache Miss - No space available", metrics
        else:
            metrics = {'hit_type': 'cache_miss_not_cached', 'cache_reason': cache_reason}
            return False, f"Cache Miss - Not cached: {cache_reason}", metrics
    
    def _should_cache_content(self, content: VECContent, requester_pos: Tuple[float, float]) -> Tuple[bool, str]:
        """决定是否缓存内容"""
        
        # 检查容量
        if self.current_usage + content.size_mb > self.capacity:
            if not self._can_make_space(content.size_mb):
                return False, "Insufficient capacity"
        
        # 内容类型权重
        type_weight = self.content_type_weights.get(content.content_type, 0.5)
        
        # 地理相关性评分
        if content.relevance_radius > 0:
            distance = content.location_distance(*requester_pos)
            geo_score = max(0, 1.0 - distance / (2 * content.relevance_radius))
        else:
            geo_score = 1.0  # 全局内容
        
        # 优先级评分
        priority_score = content.priority_level / 5.0
        
        # 流行度评分
        popularity_score = content.popularity_score
        
        # 实时性奖励
        realtime_bonus = 0.2 if content.is_real_time else 0.0
        
        # 综合评分
        cache_score = (0.3 * type_weight + 
                      0.25 * geo_score + 
                      0.2 * priority_score + 
                      0.15 * popularity_score + 
                      0.1 + realtime_bonus)
        
        # 缓存决策阈值
        cache_threshold = 0.6
        
        if cache_score >= cache_threshold:
            return True, f"High score ({cache_score:.2f})"
        else:
            return False, f"Low score ({cache_score:.2f})"
    
    def _add_content_to_cache(self, content: VECContent) -> bool:
        """添加内容到缓存"""
        # 确保有足够空间
        if self.current_usage + content.size_mb > self.capacity:
            if not self._make_space(content.size_mb):
                return False
        
        # 添加到缓存
        self.cached_contents[content.content_id] = content
        self.current_usage += content.size_mb
        
        return True
    
    def _can_make_space(self, required_space: float) -> bool:
        """检查是否能腾出足够空间"""
        available_for_eviction = sum(content.size_mb for content in self.cached_contents.values())
        return available_for_eviction >= required_space
    
    def _make_space(self, required_space: float) -> bool:
        """腾出缓存空间"""
        if not self.cached_contents:
            return False
        
        current_time = time.time()
        
        # 计算每个内容的驱逐分数（越高越容易被驱逐）
        eviction_candidates = []
        
        for content_id, content in self.cached_contents.items():
            # 时间因子
            age = current_time - content.creation_time
            time_factor = age / content.freshness_lifetime
            
            # 访问频率因子
            access_count = len(self.access_patterns[content_id])
            frequency_factor = 1.0 / max(1, access_count)
            
            # 优先级因子（低优先级容易被驱逐）
            priority_factor = (6 - content.priority_level) / 5.0
            
            # 大小因子
            size_factor = content.size_mb / 50.0  # 归一化
            
            # 综合驱逐分数
            eviction_score = (0.4 * time_factor + 
                            0.3 * frequency_factor + 
                            0.2 * priority_factor + 
                            0.1 * size_factor)
            
            eviction_candidates.append((eviction_score, content_id, content))
        
        # 按驱逐分数排序
        eviction_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 执行驱逐
        freed_space = 0.0
        for score, content_id, content in eviction_candidates:
            if freed_space >= required_space:
                break
            
            self._evict_content(content_id)
            freed_space += content.size_mb
            self.stats['capacity_evictions'] += 1
        
        return freed_space >= required_space
    
    def _evict_content(self, content_id: str):
        """驱逐指定内容"""
        if content_id in self.cached_contents:
            content = self.cached_contents.pop(content_id)
            self.current_usage -= content.size_mb
            if content_id in self.access_patterns:
                del self.access_patterns[content_id]
    
    def periodic_cleanup(self):
        """定期清理过期内容"""
        current_time = time.time()
        expired_contents = []
        
        for content_id, content in self.cached_contents.items():
            if not content.is_fresh(current_time):
                expired_contents.append(content_id)
        
        for content_id in expired_contents:
            self._evict_content(content_id)
    
    def get_comprehensive_statistics(self) -> Dict:
        """获取综合统计信息"""
        total_requests = self.stats['total_requests']
        
        return {
            'node_info': {
                'node_id': self.node_id,
                'node_type': self.node_type,
                'position': self.position,
                'capacity_mb': self.capacity
            },
            'cache_performance': {
                'total_requests': total_requests,
                'cache_hits': self.stats['cache_hits'],
                'hit_rate': self.stats['cache_hits'] / max(1, total_requests),
                'location_hit_rate': self.stats['location_hits'] / max(1, total_requests),
                'fresh_hit_rate': self.stats['fresh_hits'] / max(1, total_requests)
            },
            'miss_analysis': {
                'stale_misses': self.stats['stale_misses'],
                'distance_misses': self.stats['distance_misses'],
                'capacity_evictions': self.stats['capacity_evictions']
            },
            'resource_utilization': {
                'current_usage_mb': self.current_usage,
                'usage_ratio': self.current_usage / self.capacity,
                'cached_items': len(self.cached_contents),
                'avg_item_size_mb': self.current_usage / max(1, len(self.cached_contents))
            },
            'content_distribution': self._get_content_type_distribution()
        }
    
    def _get_content_type_distribution(self) -> Dict[str, int]:
        """获取缓存内容类型分布"""
        distribution = defaultdict(int)
        for content in self.cached_contents.values():
            distribution[content.content_type.value] += 1
        return dict(distribution)


# 测试函数
def test_realistic_vec_cache():
    """测试现实VEC缓存系统"""
    print("🧪 测试现实VEC缓存系统...")
    
    # 创建RSU缓存
    rsu_cache = RealisticVECCacheSimulator("rsu_001", "rsu", (39.9042, 116.4074))
    
    # 创建车辆profile
    vehicle = Vehicle(
        vehicle_id="vehicle_001",
        position=(39.9050, 116.4080),
        speed=60.0,
        direction=45.0,
        route=[(39.9050, 116.4080), (39.9060, 116.4090)],
        preferences={
            ContentType.TRAFFIC_INFO: 1.0,
            ContentType.NAVIGATION: 0.8,
            ContentType.ENTERTAINMENT: 0.3
        }
    )
    
    # 仿真内容请求
    for i in range(50):
        # 生成内容请求
        content = rsu_cache.generate_realistic_content_request(
            vehicle.position, vehicle
        )
        
        # 处理请求
        hit, action, metrics = rsu_cache.request_content(content, vehicle.position)
        
        print(f"请求 {i+1}: {content.content_type.value} - {'命中' if hit else '未命中'}")
        print(f"  动作: {action}")
        print(f"  指标: {metrics}")
        
        # 模拟车辆移动
        vehicle.position = (
            vehicle.position[0] + np.random.uniform(-0.001, 0.001),
            vehicle.position[1] + np.random.uniform(-0.001, 0.001)
        )
        
        time.sleep(0.1)
        
        # 定期清理
        if i % 10 == 0:
            rsu_cache.periodic_cleanup()
    
    # 输出统计
    stats = rsu_cache.get_comprehensive_statistics()
    print(f"\n📊 最终统计:")
    print(f"缓存命中率: {stats['cache_performance']['hit_rate']:.2%}")
    print(f"地理相关命中率: {stats['cache_performance']['location_hit_rate']:.2%}")
    print(f"新鲜内容命中率: {stats['cache_performance']['fresh_hit_rate']:.2%}")
    print(f"容量利用率: {stats['resource_utilization']['usage_ratio']:.2%}")
    print(f"内容类型分布: {stats['content_distribution']}")
    
    print("✅ 现实VEC缓存测试完成")


if __name__ == "__main__":
    test_realistic_vec_cache()
