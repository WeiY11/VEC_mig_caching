#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的Baseline算法实现
适配固定拓扑（4 RSU + 2 UAV），使用正确的动作维度（16维）

【包含的Baseline】
1. Random - 随机选择策略
2. Greedy - 贪心最小负载
3. RoundRobin - 轮询分配
4. LocalFirst - 本地优先
5. NearestNode - 最近节点优先（使用距离信息）
"""

import numpy as np
from typing import Dict, Optional, Any


class BaselineAlgorithm:
    """Baseline算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.step_count = 0
        self.rr_index = 0  # 用于RoundRobin
        self.env = None  # 训练环境引用
    
    def update_environment(self, env):
        """注入训练环境引用"""
        self.env = env
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态向量（维度=num_vehicles*5 + num_rsus*5 + num_uavs*5 + 8）
            
        Returns:
            16维动作向量：
            - [0:3] 任务分配（local, RSU, UAV）
            - [3:7] RSU选择（4个RSU）
            - [7:9] UAV选择（2个UAV）
            - [9:16] 控制参数（7维）
        """
        raise NotImplementedError
    
    def reset(self):
        """重置算法状态"""
        self.step_count = 0
        self.rr_index = 0
    
    def _extract_loads_from_state(self, state: np.ndarray, num_vehicles: int = 12) -> tuple:
        """
        从状态向量提取负载信息
        
        状态结构（固定拓扑 4 RSU + 2 UAV）:
        - [0 : num_vehicles*5] 车辆状态
        - [num_vehicles*5 : num_vehicles*5+20] RSU状态（4个RSU×5维）
        - [num_vehicles*5+20 : num_vehicles*5+30] UAV状态（2个UAV×5维）
        - [num_vehicles*5+30 : num_vehicles*5+38] 全局状态（8维）
        
        每个节点的5维状态：
        - 0: x坐标
        - 1: y坐标
        - 2: 速度/高度
        - 3: CPU负载（0-1）
        - 4: 队列长度（归一化）
        """
        vehicle_offset = num_vehicles * 5
        rsu_offset = vehicle_offset
        uav_offset = vehicle_offset + 4 * 5
        
        # 提取车辆平均负载
        vehicle_loads = []
        for i in range(min(num_vehicles, len(state) // 5)):
            cpu_idx = i * 5 + 3
            if cpu_idx < len(state):
                vehicle_loads.append(state[cpu_idx])
        local_load = float(np.mean(vehicle_loads)) if vehicle_loads else 0.5
        
        # 提取RSU负载（4个）
        rsu_loads = []
        for i in range(4):
            cpu_idx = rsu_offset + i * 5 + 3
            if cpu_idx < len(state):
                rsu_loads.append(state[cpu_idx])
            else:
                rsu_loads.append(1.0)
        
        # 提取UAV负载（2个）
        uav_loads = []
        for i in range(2):
            cpu_idx = uav_offset + i * 5 + 3
            if cpu_idx < len(state):
                uav_loads.append(state[cpu_idx])
            else:
                uav_loads.append(1.0)
        
        return local_load, rsu_loads, uav_loads


class RandomBaseline(BaselineAlgorithm):
    """随机策略 - 完全随机选择"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.step_count += 1
        # 16维随机动作
        return np.random.uniform(-1, 1, 16)


class GreedyBaseline(BaselineAlgorithm):
    """贪心策略 - 选择负载最小的节点"""
    
    def __init__(self):
        super().__init__("Greedy")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 提取负载信息
        local_load, rsu_loads, uav_loads = self._extract_loads_from_state(state, num_vehicles)
        
        # 选择负载最小的执行位置
        loads = [local_load, min(rsu_loads), min(uav_loads)]
        best = int(np.argmin(loads))
        
        # 任务分配（3维）
        action[0] = 0.9 if best == 0 else -0.9  # Local
        action[1] = 0.9 if best == 1 else -0.9  # RSU
        action[2] = 0.9 if best == 2 else -0.9  # UAV
        
        # RSU选择（4维）- 选择负载最小的RSU
        if best == 1:
            best_rsu_idx = int(np.argmin(rsu_loads))
            for i in range(4):
                action[3 + i] = 0.9 if i == best_rsu_idx else -0.9
        else:
            action[3:7] = -0.9  # 不选RSU时全部设为负
        
        # UAV选择（2维）- 选择负载最小的UAV
        if best == 2:
            best_uav_idx = int(np.argmin(uav_loads))
            for i in range(2):
                action[7 + i] = 0.9 if i == best_uav_idx else -0.9
        else:
            action[7:9] = -0.9  # 不选UAV时全部设为负
        
        # 控制参数（7维）- 保守策略
        action[9:16] = 0.0  # 默认中性控制
        
        self.step_count += 1
        return action


class RoundRobinBaseline(BaselineAlgorithm):
    """轮询策略 - 依次分配到不同节点"""
    
    def __init__(self):
        super().__init__("RoundRobin")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 轮询选择执行位置（local -> RSU -> UAV）
        phase = self.rr_index % 3
        
        if phase == 0:  # Local
            action[0:3] = [0.9, -0.9, -0.9]
            action[3:7] = -0.9
            action[7:9] = -0.9
        elif phase == 1:  # RSU
            action[0:3] = [-0.9, 0.9, -0.9]
            # 轮询选择RSU
            rsu_idx = (self.rr_index // 3) % 4
            for i in range(4):
                action[3 + i] = 0.9 if i == rsu_idx else -0.9
            action[7:9] = -0.9
        else:  # UAV
            action[0:3] = [-0.9, -0.9, 0.9]
            action[3:7] = -0.9
            # 轮询选择UAV
            uav_idx = (self.rr_index // 3) % 2
            for i in range(2):
                action[7 + i] = 0.9 if i == uav_idx else -0.9
        
        # 控制参数
        action[9:16] = 0.0
        
        self.rr_index += 1
        self.step_count += 1
        return action


class LocalFirstBaseline(BaselineAlgorithm):
    """本地优先策略 - 尽量本地处理"""
    
    def __init__(self):
        super().__init__("LocalFirst")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 提取负载
        local_load, rsu_loads, uav_loads = self._extract_loads_from_state(state, num_vehicles)
        
        # 本地优先：只有本地过载时才卸载
        if local_load < 0.8:  # 本地未过载
            action[0:3] = [0.9, -0.9, -0.9]  # Local
            action[3:7] = -0.9
            action[7:9] = -0.9
        else:  # 本地过载，选择RSU或UAV
            if min(rsu_loads) < min(uav_loads):
                action[0:3] = [-0.9, 0.9, -0.9]  # RSU
                best_rsu = int(np.argmin(rsu_loads))
                for i in range(4):
                    action[3 + i] = 0.9 if i == best_rsu else -0.9
                action[7:9] = -0.9
            else:
                action[0:3] = [-0.9, -0.9, 0.9]  # UAV
                action[3:7] = -0.9
                best_uav = int(np.argmin(uav_loads))
                for i in range(2):
                    action[7 + i] = 0.9 if i == best_uav else -0.9
        
        # 控制参数
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class NearestNodeBaseline(BaselineAlgorithm):
    """最近节点优先 - 基于距离选择"""
    
    def __init__(self):
        super().__init__("NearestNode")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 提取负载
        local_load, rsu_loads, uav_loads = self._extract_loads_from_state(state, num_vehicles)
        
        # 如果本地负载过高，选择最近的非过载节点
        if local_load > 0.7:
            # 找到负载较低的RSU和UAV
            valid_rsus = [i for i, load in enumerate(rsu_loads) if load < 0.8]
            valid_uavs = [i for i, load in enumerate(uav_loads) if load < 0.8]
            
            if valid_rsus or valid_uavs:
                # 简化：随机选择RSU或UAV（在真实系统中会使用距离）
                if valid_rsus and (not valid_uavs or np.random.rand() < 0.7):
                    # 选择RSU
                    action[0:3] = [-0.9, 0.9, -0.9]
                    best_rsu = np.random.choice(valid_rsus)
                    for i in range(4):
                        action[3 + i] = 0.9 if i == best_rsu else -0.9
                    action[7:9] = -0.9
                else:
                    # 选择UAV
                    action[0:3] = [-0.9, -0.9, 0.9]
                    action[3:7] = -0.9
                    best_uav = np.random.choice(valid_uavs) if valid_uavs else 0
                    for i in range(2):
                        action[7 + i] = 0.9 if i == best_uav else -0.9
            else:
                # 所有边缘节点都过载，本地处理
                action[0:3] = [0.9, -0.9, -0.9]
                action[3:7] = -0.9
                action[7:9] = -0.9
        else:
            # 本地负载正常，本地处理
            action[0:3] = [0.9, -0.9, -0.9]
            action[3:7] = -0.9
            action[7:9] = -0.9
        
        # 控制参数
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


def create_baseline_algorithm(algorithm_name: str) -> BaselineAlgorithm:
    """
    创建Baseline算法实例
    
    Args:
        algorithm_name: 算法名称
        
    Returns:
        Baseline算法实例
    """
    algorithm_name_upper = algorithm_name.upper()
    
    if algorithm_name_upper == 'RANDOM':
        return RandomBaseline()
    elif algorithm_name_upper == 'GREEDY':
        return GreedyBaseline()
    elif algorithm_name_upper == 'ROUNDROBIN':
        return RoundRobinBaseline()
    elif algorithm_name_upper == 'LOCALFIRST':
        return LocalFirstBaseline()
    elif algorithm_name_upper == 'NEARESTNODE':
        return NearestNodeBaseline()
    else:
        raise ValueError(f"不支持的Baseline算法: {algorithm_name}")


