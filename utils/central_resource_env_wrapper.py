#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中央资源分配环境包装器 - 分层架构扩展

【功能】
为TD3智能体提供扩展的状态和动作空间，支持：
- Phase 1: 中央智能体决策（资源分配）
- Phase 2: 本地执行层（优先级调度）

【状态空间扩展】（约80维）
- 车辆状态：队列长度、优先级分布、位置、信道质量 (12×5=60维)
- RSU状态：负载、队列、可用资源 (4×3=12维)
- UAV状态：负载、电量、位置 (2×4=8维)

【动作空间扩展】（约30维）
- 带宽分配：12车辆连续动作 [0,1] (12维)
- 车辆计算分配：12车辆连续动作 [0,1] (12维)
- RSU计算分配：4个RSU连续动作 [0,1] (4维)
- UAV计算分配：2个UAV连续动作 [0,1] (2维)
"""

import numpy as np
from typing import Dict, Tuple, Any
import torch


class CentralResourceEnvWrapper:
    """
    中央资源分配环境包装器
    
    包装原有环境，扩展状态和动作空间以支持分层资源分配架构
    """
    
    def __init__(self, base_env):
        """
        初始化包装器
        
        Args:
            base_env: 基础环境（VECEnv等）
        """
        self.base_env = base_env
        self.simulator = base_env.simulator if hasattr(base_env, 'simulator') else None
        
        # 从simulator获取节点数量
        if self.simulator:
            self.num_vehicles = len(self.simulator.vehicles)
            self.num_rsus = len(self.simulator.rsus)
            self.num_uavs = len(self.simulator.uavs)
        else:
            self.num_vehicles = 12
            self.num_rsus = 4
            self.num_uavs = 2
        
        # 🎯 扩展状态空间维度
        # 车辆状态：[队列长度, 类型1任务数, 类型2任务数, 类型3任务数, 类型4任务数] × 12 = 60
        # RSU状态：[负载率, 队列长度, 可用资源] × 4 = 12
        # UAV状态：[负载率, 电量, 队列长度, 可用资源] × 2 = 8
        # 总计：60 + 12 + 8 = 80维
        self.extended_state_dim = self.num_vehicles * 5 + self.num_rsus * 3 + self.num_uavs * 4
        
        # 🎯 扩展动作空间维度
        # 带宽分配(12) + 车辆计算分配(12) + RSU计算分配(4) + UAV计算分配(2) = 30维
        self.extended_action_dim = self.num_vehicles * 2 + self.num_rsus + self.num_uavs
        
        print(f"🎯 中央资源分配架构已启用")
        print(f"   状态空间维度: {self.extended_state_dim}")
        print(f"   动作空间维度: {self.extended_action_dim}")
        print(f"   节点配置: {self.num_vehicles}车辆 + {self.num_rsus}RSU + {self.num_uavs}UAV")
    
    def get_extended_state(self) -> np.ndarray:
        """
        获取扩展状态（供中央智能体观测）
        
        Returns:
            扩展状态向量 (extended_state_dim,)
        """
        if not self.simulator:
            return np.zeros(self.extended_state_dim)
        
        state_components = []
        
        # ========== 车辆状态 (60维) ==========
        for vehicle in self.simulator.vehicles:
            # 总队列长度
            total_queue = sum(len(vehicle['task_queue_by_priority'][p]) for p in [1, 2, 3, 4])
            
            # 各优先级任务数
            type1_tasks = len(vehicle['task_queue_by_priority'][1])
            type2_tasks = len(vehicle['task_queue_by_priority'][2])
            type3_tasks = len(vehicle['task_queue_by_priority'][3])
            type4_tasks = len(vehicle['task_queue_by_priority'][4])
            
            state_components.extend([
                total_queue / 20.0,  # 归一化队列长度
                type1_tasks / 5.0,   # 归一化各类型任务数
                type2_tasks / 5.0,
                type3_tasks / 5.0,
                type4_tasks / 5.0,
            ])
        
        # ========== RSU状态 (12维) ==========
        for rsu in self.simulator.rsus:
            load_rate = rsu.get('compute_usage', 0.0)
            queue_len = len(rsu.get('computation_queue', []))
            available_resource = rsu.get('allocated_compute', 15e9) / 15e9  # 归一化
            
            state_components.extend([
                load_rate,
                queue_len / 20.0,  # 归一化队列长度
                available_resource,
            ])
        
        # ========== UAV状态 (8维) ==========
        for uav in self.simulator.uavs:
            load_rate = uav.get('compute_usage', 0.0)
            battery_level = uav.get('battery_level', 1.0)
            queue_len = len(uav.get('computation_queue', []))
            available_resource = uav.get('allocated_compute', 4e9) / 4e9  # 归一化
            
            state_components.extend([
                load_rate,
                battery_level,
                queue_len / 10.0,  # 归一化队列长度
                available_resource,
            ])
        
        return np.array(state_components, dtype=np.float32)
    
    def parse_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """
        解析中央智能体的动作向量
        
        Args:
            action: 动作向量 (extended_action_dim,)
                前12维: 带宽分配比例
                中间12维: 车辆计算分配比例
                后4维: RSU计算分配比例
                最后2维: UAV计算分配比例
        
        Returns:
            资源分配字典
        """
        action = np.clip(action, 0, 1)  # 确保在[0,1]范围内
        
        # 解析各部分
        bandwidth_alloc = action[:self.num_vehicles]
        vehicle_compute_alloc = action[self.num_vehicles:self.num_vehicles*2]
        rsu_compute_alloc = action[self.num_vehicles*2:self.num_vehicles*2+self.num_rsus]
        uav_compute_alloc = action[self.num_vehicles*2+self.num_rsus:]
        
        return {
            'bandwidth': bandwidth_alloc,
            'vehicle_compute': vehicle_compute_alloc,
            'rsu_compute': rsu_compute_alloc,
            'uav_compute': uav_compute_alloc,
        }
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步仿真（Phase 1 + Phase 2）
        
        Args:
            action: 中央智能体的动作向量
        
        Returns:
            (next_state, reward, done, info)
        """
        # ========== Phase 1: 解析并应用资源分配 ==========
        allocation_dict = self.parse_action(action)
        
        if self.simulator:
            self.simulator.apply_resource_allocation(allocation_dict)
        
        # ========== Phase 2: 执行本地调度 ==========
        if self.simulator:
            self.simulator.execute_phase2_scheduling()
        
        # ========== 执行基础环境的step ==========
        _, base_reward, done, info = self.base_env.step(action[:self.base_env.action_space.shape[0]])
        
        # ========== 获取扩展状态 ==========
        next_state = self.get_extended_state()
        
        # ========== 计算增强奖励 ==========
        enhanced_reward = self._calculate_enhanced_reward(base_reward, allocation_dict, info)
        
        # ========== 更新info ==========
        if self.simulator and hasattr(self.simulator, 'resource_pool'):
            info['resource_state'] = self.simulator.resource_pool.get_resource_state()
            info['vehicle_utilization'] = np.mean([v['compute_usage'] for v in self.simulator.vehicles])
            info['rsu_utilization'] = np.mean([r['compute_usage'] for r in self.simulator.rsus])
            info['uav_utilization'] = np.mean([u['compute_usage'] for u in self.simulator.uavs])
        
        return next_state, enhanced_reward, done, info
    
    def _calculate_enhanced_reward(self, base_reward: float, allocation_dict: Dict, info: Dict) -> float:
        """
        计算增强奖励函数
        
        Args:
            base_reward: 基础奖励（时延+能耗）
            allocation_dict: 资源分配字典
            info: 额外信息
        
        Returns:
            增强奖励
        """
        # 基础奖励（时延+能耗）
        reward = base_reward
        
        if not self.simulator or not hasattr(self.simulator, 'resource_pool'):
            return reward
        
        # 🎯 资源利用率奖励（鼓励充分利用资源）
        resource_state = self.simulator.resource_pool.get_resource_state()
        vehicle_util = resource_state['vehicle_utilization']
        rsu_util = resource_state['rsu_utilization']
        uav_util = resource_state['uav_utilization']
        
        # 目标利用率：70-90%（过高导致拥塞，过低浪费资源）
        def utilization_reward(util):
            if 0.7 <= util <= 0.9:
                return 0.1  # 良好利用率奖励
            elif util > 0.95:
                return -0.2  # 过载惩罚
            elif util < 0.3:
                return -0.1  # 资源浪费惩罚
            else:
                return 0.0
        
        util_reward = (utilization_reward(vehicle_util) + 
                      utilization_reward(rsu_util) + 
                      utilization_reward(uav_util)) / 3.0
        
        # 🎯 分配公平性奖励（避免资源分配过于集中）
        def fairness_metric(allocation: np.ndarray) -> float:
            """Jain's fairness index"""
            if len(allocation) == 0:
                return 1.0
            sum_x = np.sum(allocation)
            sum_x2 = np.sum(allocation ** 2)
            n = len(allocation)
            if sum_x2 == 0:
                return 1.0
            return (sum_x ** 2) / (n * sum_x2)
        
        bandwidth_fairness = fairness_metric(allocation_dict['bandwidth'])
        compute_fairness = fairness_metric(allocation_dict['vehicle_compute'])
        
        fairness_reward = 0.05 * (bandwidth_fairness + compute_fairness - 1.5)  # 鼓励接近1.0
        
        # 总奖励
        enhanced_reward = reward + util_reward + fairness_reward
        
        return enhanced_reward
    
    def reset(self) -> np.ndarray:
        """重置环境并返回扩展初始状态"""
        self.base_env.reset()
        return self.get_extended_state()
    
    def __getattr__(self, name):
        """代理其他属性到基础环境"""
        return getattr(self.base_env, name)


def create_central_resource_env(base_env):
    """
    工厂函数：创建中央资源分配环境
    
    Args:
        base_env: 基础环境对象
    
    Returns:
        包装后的环境
    """
    return CentralResourceEnvWrapper(base_env)


