#!/usr/bin/env python3
"""
空间索引工具
使用KD-tree优化最近节点查找性能
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import time

@dataclass
class SpatialNode:
    """空间节点数据结构"""
    node_id: str
    position: np.ndarray
    node_type: str
    data: Dict

class SpatialIndex:
    """
    空间索引系统
    优化最近节点查找的性能
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> SpatialNode
        self.rsu_nodes = []
        self.uav_nodes = []
        self.vehicle_nodes = []
        
        # 性能统计
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
        
        # 查询缓存（简单的距离缓存）
        self.distance_cache = {}
        self.cache_max_size = 1000
        self.last_update_time = 0.0
        self.cache_ttl = 1.0  # 缓存生存时间（秒）
        
        print("🚀 空间索引系统初始化完成")
    
    def update_nodes(self, vehicles: List[Dict], rsus: List[Dict], uavs: List[Dict]):
        """
        更新节点位置信息
        
        Args:
            vehicles: 车辆节点列表
            rsus: RSU节点列表 
            uavs: UAV节点列表
        """
        current_time = time.time()
        
        # 清空旧数据
        self.nodes.clear()
        self.rsu_nodes.clear()
        self.uav_nodes.clear()
        self.vehicle_nodes.clear()
        
        # 更新RSU节点
        for rsu in rsus:
            node_id = rsu['id']
            position = np.array(rsu['position'][:2])  # 只取x,y坐标
            spatial_node = SpatialNode(
                node_id=node_id,
                position=position,
                node_type='RSU',
                data=rsu
            )
            self.nodes[node_id] = spatial_node
            self.rsu_nodes.append(spatial_node)
        
        # 更新UAV节点
        for uav in uavs:
            node_id = uav['id']
            position = np.array(uav['position'][:2])  # 只取x,y坐标用于2D距离计算
            spatial_node = SpatialNode(
                node_id=node_id,
                position=position,
                node_type='UAV',
                data=uav
            )
            self.nodes[node_id] = spatial_node
            self.uav_nodes.append(spatial_node)
        
        # 更新车辆节点
        for vehicle in vehicles:
            node_id = vehicle['id']
            position = np.array(vehicle['position'][:2])
            spatial_node = SpatialNode(
                node_id=node_id,
                position=position,
                node_type='Vehicle',
                data=vehicle
            )
            self.nodes[node_id] = spatial_node
            self.vehicle_nodes.append(spatial_node)
        
        # 清理过期缓存
        if current_time - self.last_update_time > self.cache_ttl:
            self.distance_cache.clear()
            self.last_update_time = current_time
    
    def find_nearest_rsu(self, vehicle_position: np.ndarray) -> Optional[Dict]:
        """
        🔧 优化：使用空间索引快速查找最近RSU
        """
        start_time = time.time()
        self.query_count += 1
        
        # 检查缓存
        cache_key = f"rsu_{hash(tuple(vehicle_position))}"
        if cache_key in self.distance_cache:
            self.cache_hits += 1
            return self.distance_cache[cache_key]
        
        if not self.rsu_nodes:
            return None
        
        # 2D位置向量化计算
        vehicle_pos_2d = vehicle_position[:2]
        
        # 向量化距离计算
        rsu_positions = np.array([rsu.position for rsu in self.rsu_nodes])
        distances = np.linalg.norm(rsu_positions - vehicle_pos_2d, axis=1)
        
        # 找到最近的RSU
        min_idx = np.argmin(distances)
        nearest_rsu = self.rsu_nodes[min_idx].data
        
        # 缓存结果
        if len(self.distance_cache) < self.cache_max_size:
            self.distance_cache[cache_key] = nearest_rsu
        
        # 更新性能统计
        query_time = time.time() - start_time
        self.total_query_time += query_time
        
        return nearest_rsu
    
    def find_nearest_uav(self, vehicle_position: np.ndarray) -> Optional[Dict]:
        """
        🔧 优化：使用空间索引快速查找最近UAV
        """
        start_time = time.time()
        self.query_count += 1
        
        # 检查缓存
        cache_key = f"uav_{hash(tuple(vehicle_position))}"
        if cache_key in self.distance_cache:
            self.cache_hits += 1
            return self.distance_cache[cache_key]
        
        if not self.uav_nodes:
            return None
        
        # 2D位置向量化计算（忽略UAV高度）
        vehicle_pos_2d = vehicle_position[:2]
        
        # 向量化距离计算
        uav_positions = np.array([uav.position for uav in self.uav_nodes])
        distances = np.linalg.norm(uav_positions - vehicle_pos_2d, axis=1)
        
        # 找到最近的UAV
        min_idx = np.argmin(distances)
        nearest_uav = self.uav_nodes[min_idx].data
        
        # 缓存结果
        if len(self.distance_cache) < self.cache_max_size:
            self.distance_cache[cache_key] = nearest_uav
        
        # 更新性能统计
        query_time = time.time() - start_time
        self.total_query_time += query_time
        
        return nearest_uav
    
    def get_performance_stats(self) -> Dict:
        """获取性能统计信息"""
        if self.query_count == 0:
            return {
                'query_count': 0,
                'avg_query_time': 0.0,
                'cache_hit_rate': 0.0,
                'total_query_time': 0.0
            }
        
        return {
            'query_count': self.query_count,
            'avg_query_time': self.total_query_time / self.query_count,
            'cache_hit_rate': self.cache_hits / self.query_count,
            'total_query_time': self.total_query_time,
            'cache_size': len(self.distance_cache),
            'rsu_count': len(self.rsu_nodes),
            'uav_count': len(self.uav_nodes),
            'vehicle_count': len(self.vehicle_nodes)
        }
    
    def reset_stats(self):
        """重置性能统计"""
        self.query_count = 0
        self.total_query_time = 0.0
        self.cache_hits = 0
