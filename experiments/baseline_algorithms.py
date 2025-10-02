#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Baseline算法集合
用于与DRL算法进行性能对比

包含以下Baseline:
1. Random: 随机选择
2. Greedy: 贪心选择（最小负载）
3. RoundRobin: 轮询分配
4. LoadBalanced: 负载均衡
5. NearestNode: 最近节点优先
6. LocalFirst: 本地优先策略
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class BaselineDecision:
    """Baseline决策结果"""
    selected_node: str
    node_type: str  # 'vehicle', 'rsu', 'uav'
    decision_time: float
    estimated_delay: float


class BaselineAlgorithm:
    """Baseline算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.decision_count = 0
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """做出卸载决策"""
        raise NotImplementedError
    
    def reset(self):
        """重置算法状态"""
        self.decision_count = 0


class RandomAlgorithm(BaselineAlgorithm):
    """随机算法 - 随机选择处理节点"""
    
    def __init__(self):
        super().__init__("Random")
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """随机选择节点"""
        import time
        
        # 收集所有候选节点
        candidates = []
        
        # 添加本地节点
        candidates.append(('vehicle', current_vehicle_id, vehicles[current_vehicle_id]))
        
        # 添加RSU节点
        for i, rsu in enumerate(rsus):
            if self._is_in_range(vehicles[current_vehicle_id], rsu, range_limit=500):
                candidates.append(('rsu', i, rsu))
        
        # 添加UAV节点
        for i, uav in enumerate(uavs):
            if self._is_in_range(vehicles[current_vehicle_id], uav, range_limit=600):
                candidates.append(('uav', i, uav))
        
        # 随机选择
        if candidates:
            node_type, node_id, node = candidates[np.random.randint(0, len(candidates))]
            selected_node = f"{node_type}_{node_id}"
        else:
            # 默认本地处理
            node_type = 'vehicle'
            selected_node = f"vehicle_{current_vehicle_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=node_type,
            decision_time=time.time(),
            estimated_delay=0.5  # 随机算法无精确估计
        )
    
    def _is_in_range(self, vehicle, node, range_limit):
        """判断节点是否在通信范围内"""
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        # 简化的距离计算
        if len(v_pos) == 2 and len(n_pos) == 2:
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            # UAV有高度
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        else:
            distance = 0
        
        return distance <= range_limit


class GreedyAlgorithm(BaselineAlgorithm):
    """贪心算法 - 选择负载最小的节点"""
    
    def __init__(self):
        super().__init__("Greedy")
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """选择负载最小的节点"""
        import time
        
        min_load = float('inf')
        best_node = None
        best_type = None
        best_id = None
        
        # 检查本地节点
        local_load = len(vehicles[current_vehicle_id].get('computation_queue', []))
        if local_load < min_load:
            min_load = local_load
            best_node = vehicles[current_vehicle_id]
            best_type = 'vehicle'
            best_id = current_vehicle_id
        
        # 检查RSU节点
        for i, rsu in enumerate(rsus):
            if self._is_in_range(vehicles[current_vehicle_id], rsu, 500):
                rsu_load = len(rsu.get('computation_queue', []))
                if rsu_load < min_load:
                    min_load = rsu_load
                    best_node = rsu
                    best_type = 'rsu'
                    best_id = i
        
        # 检查UAV节点
        for i, uav in enumerate(uavs):
            if self._is_in_range(vehicles[current_vehicle_id], uav, 600):
                uav_load = len(uav.get('computation_queue', []))
                if uav_load < min_load:
                    min_load = uav_load
                    best_node = uav
                    best_type = 'uav'
                    best_id = i
        
        selected_node = f"{best_type}_{best_id}" if best_node else f"vehicle_{current_vehicle_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=best_type or 'vehicle',
            decision_time=time.time(),
            estimated_delay=min_load * 0.05  # 简化的延迟估计
        )
    
    def _is_in_range(self, vehicle, node, range_limit):
        """判断节点是否在通信范围内"""
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        if len(v_pos) == 2 and len(n_pos) == 2:
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        else:
            distance = 0
        
        return distance <= range_limit


class RoundRobinAlgorithm(BaselineAlgorithm):
    """轮询算法 - 按顺序轮流分配"""
    
    def __init__(self):
        super().__init__("RoundRobin")
        self.current_index = 0
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """轮询选择节点"""
        import time
        
        # 构建候选列表
        candidates = []
        candidates.append(('vehicle', current_vehicle_id))
        
        for i in range(len(rsus)):
            if self._is_in_range(vehicles[current_vehicle_id], rsus[i], 500):
                candidates.append(('rsu', i))
        
        for i in range(len(uavs)):
            if self._is_in_range(vehicles[current_vehicle_id], uavs[i], 600):
                candidates.append(('uav', i))
        
        # 轮询选择
        if candidates:
            node_type, node_id = candidates[self.current_index % len(candidates)]
            self.current_index += 1
        else:
            node_type, node_id = 'vehicle', current_vehicle_id
        
        selected_node = f"{node_type}_{node_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=node_type,
            decision_time=time.time(),
            estimated_delay=0.3
        )
    
    def _is_in_range(self, vehicle, node, range_limit):
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        if len(v_pos) == 2 and len(n_pos) == 2:
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            distance = np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        else:
            distance = 0
        
        return distance <= range_limit
    
    def reset(self):
        super().reset()
        self.current_index = 0


class LoadBalancedAlgorithm(BaselineAlgorithm):
    """负载均衡算法 - 综合考虑负载和距离"""
    
    def __init__(self):
        super().__init__("LoadBalanced")
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """综合负载和距离选择节点"""
        import time
        
        best_score = float('inf')
        best_node_type = 'vehicle'
        best_node_id = current_vehicle_id
        
        vehicle = vehicles[current_vehicle_id]
        
        # 评估本地节点
        local_load = len(vehicle.get('computation_queue', []))
        local_score = local_load * 0.8  # 本地无传输延迟，权重低
        
        if local_score < best_score:
            best_score = local_score
            best_node_type = 'vehicle'
            best_node_id = current_vehicle_id
        
        # 评估RSU节点
        for i, rsu in enumerate(rsus):
            distance = self._calculate_distance(vehicle, rsu)
            if distance <= 500:
                rsu_load = len(rsu.get('computation_queue', []))
                # 综合评分：负载 + 距离/100
                score = rsu_load + distance / 100
                if score < best_score:
                    best_score = score
                    best_node_type = 'rsu'
                    best_node_id = i
        
        # 评估UAV节点
        for i, uav in enumerate(uavs):
            distance = self._calculate_distance(vehicle, uav)
            if distance <= 600:
                uav_load = len(uav.get('computation_queue', []))
                # UAV能力较弱，负载权重高
                score = uav_load * 1.2 + distance / 100
                if score < best_score:
                    best_score = score
                    best_node_type = 'uav'
                    best_node_id = i
        
        selected_node = f"{best_node_type}_{best_node_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=best_node_type,
            decision_time=time.time(),
            estimated_delay=best_score * 0.05
        )
    
    def _calculate_distance(self, vehicle, node):
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        if len(v_pos) == 2 and len(n_pos) == 2:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        return 0


class NearestNodeAlgorithm(BaselineAlgorithm):
    """最近节点优先算法"""
    
    def __init__(self):
        super().__init__("NearestNode")
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """选择最近的节点"""
        import time
        
        vehicle = vehicles[current_vehicle_id]
        min_distance = float('inf')
        best_node_type = 'vehicle'
        best_node_id = current_vehicle_id
        
        # 检查RSU
        for i, rsu in enumerate(rsus):
            distance = self._calculate_distance(vehicle, rsu)
            if distance < min_distance and distance <= 500:
                min_distance = distance
                best_node_type = 'rsu'
                best_node_id = i
        
        # 检查UAV
        for i, uav in enumerate(uavs):
            distance = self._calculate_distance(vehicle, uav)
            if distance < min_distance and distance <= 600:
                min_distance = distance
                best_node_type = 'uav'
                best_node_id = i
        
        selected_node = f"{best_node_type}_{best_node_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=best_node_type,
            decision_time=time.time(),
            estimated_delay=min_distance / 1000  # 简化估计
        )
    
    def _calculate_distance(self, vehicle, node):
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        if len(v_pos) == 2 and len(n_pos) == 2:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        return 0


class LocalFirstAlgorithm(BaselineAlgorithm):
    """本地优先算法 - 优先本地处理，负载过高时卸载"""
    
    def __init__(self, local_threshold=5):
        super().__init__("LocalFirst")
        self.local_threshold = local_threshold
        
    def make_decision(self, task, vehicles, rsus, uavs, current_vehicle_id) -> BaselineDecision:
        """本地优先策略"""
        import time
        
        vehicle = vehicles[current_vehicle_id]
        local_load = len(vehicle.get('computation_queue', []))
        
        # 如果本地负载不高，优先本地处理
        if local_load < self.local_threshold:
            self.decision_count += 1
            return BaselineDecision(
                selected_node=f"vehicle_{current_vehicle_id}",
                node_type='vehicle',
                decision_time=time.time(),
                estimated_delay=local_load * 0.05
            )
        
        # 否则寻找最佳卸载节点
        best_score = float('inf')
        best_node_type = 'rsu'
        best_node_id = 0
        
        # 优先选择RSU
        for i, rsu in enumerate(rsus):
            distance = self._calculate_distance(vehicle, rsu)
            if distance <= 500:
                rsu_load = len(rsu.get('computation_queue', []))
                score = rsu_load + distance / 200
                if score < best_score:
                    best_score = score
                    best_node_type = 'rsu'
                    best_node_id = i
        
        # 如果RSU都不理想，考虑UAV
        if best_score > 10:
            for i, uav in enumerate(uavs):
                distance = self._calculate_distance(vehicle, uav)
                if distance <= 600:
                    uav_load = len(uav.get('computation_queue', []))
                    score = uav_load + distance / 200
                    if score < best_score:
                        best_score = score
                        best_node_type = 'uav'
                        best_node_id = i
        
        selected_node = f"{best_node_type}_{best_node_id}"
        
        self.decision_count += 1
        
        return BaselineDecision(
            selected_node=selected_node,
            node_type=best_node_type,
            decision_time=time.time(),
            estimated_delay=best_score * 0.05
        )
    
    def _calculate_distance(self, vehicle, node):
        v_pos = vehicle.get('position', (0, 0))
        n_pos = node.get('position', (0, 0))
        
        if len(v_pos) == 2 and len(n_pos) == 2:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2)
        elif len(v_pos) == 2 and len(n_pos) == 3:
            return np.sqrt((v_pos[0] - n_pos[0])**2 + (v_pos[1] - n_pos[1])**2 + n_pos[2]**2)
        return 0


# ==================== Baseline算法工厂 ====================

class BaselineFactory:
    """Baseline算法工厂类"""
    
    @staticmethod
    def get_all_baselines() -> Dict[str, BaselineAlgorithm]:
        """获取所有Baseline算法"""
        return {
            'Random': RandomAlgorithm(),
            'Greedy': GreedyAlgorithm(),
            'RoundRobin': RoundRobinAlgorithm(),
            'LoadBalanced': LoadBalancedAlgorithm(),
            'NearestNode': NearestNodeAlgorithm(),
            'LocalFirst': LocalFirstAlgorithm()
        }
    
    @staticmethod
    def get_baseline(name: str) -> BaselineAlgorithm:
        """获取指定Baseline算法"""
        baselines = BaselineFactory.get_all_baselines()
        if name not in baselines:
            raise ValueError(f"Unknown baseline algorithm: {name}")
        return baselines[name]

