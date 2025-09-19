#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强车辆移动性建模
针对固定UAV和RSU环境下的车辆智能移动策略
"""

import numpy as np
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum

from .data_structures import Position
from config import config

class MobilityStrategy(Enum):
    """车辆移动策略枚举"""
    RANDOM_WALK = "random_walk"  # 随机游走
    COVERAGE_OPTIMIZATION = "coverage_optimization"  # 覆盖优化
    CONNECTIVITY_SEEKING = "connectivity_seeking"  # 连接性寻求
    ENERGY_EFFICIENT = "energy_efficient"  # 能效优化
    TASK_ORIENTED = "task_oriented"  # 任务导向

@dataclass
class TrajectoryPoint:
    """轨迹点数据结构"""
    position: Position
    timestamp: float
    velocity: np.ndarray
    acceleration: np.ndarray
    connectivity_score: float  # 与固定节点的连接性评分
    
class VehicleMobilityModel:
    """
    增强车辆移动性模型
    专门针对固定UAV和RSU环境优化的车辆移动策略
    """
    
    def __init__(self, vehicle_id: str, initial_position: Position):
        self.vehicle_id = vehicle_id
        self.current_position = initial_position
        self.velocity = np.array([0.0, 0.0])  # m/s
        self.acceleration = np.array([0.0, 0.0])  # m/s²
        
        # 移动性参数
        self.max_speed = 30.0  # m/s (约108 km/h)
        self.max_acceleration = 3.0  # m/s²
        self.min_speed = 2.0  # m/s (最小移动速度)
        
        # 轨迹历史和预测
        self.trajectory_history: List[TrajectoryPoint] = []
        self.predicted_trajectory: List[TrajectoryPoint] = []
        self.trajectory_window = 50  # 保留轨迹点数量
        
        # 移动策略
        self.current_strategy = MobilityStrategy.COVERAGE_OPTIMIZATION
        self.strategy_weights = {
            'connectivity': 0.4,  # 连接性权重
            'coverage': 0.3,      # 覆盖范围权重
            'energy': 0.2,        # 能效权重
            'task_load': 0.1      # 任务负载权重
        }
        
        # 固定节点信息（由外部更新）
        self.rsu_positions: List[Position] = []
        self.uav_positions: List[Position] = []
        self.rsu_coverage_radius = 300.0  # RSU覆盖半径 (m)
        self.uav_coverage_radius = 500.0  # UAV覆盖半径 (m)
        
        # 路径规划
        self.target_position: Optional[Position] = None
        self.waypoints: List[Position] = []
        
    def update_fixed_nodes_info(self, rsu_positions: List[Position], uav_positions: List[Position]):
        """更新固定节点位置信息"""
        self.rsu_positions = rsu_positions
        self.uav_positions = uav_positions
        
    def calculate_connectivity_score(self, position: Position) -> float:
        """
        计算给定位置的连接性评分
        考虑与RSU和UAV的连接质量
        """
        connectivity_score = 0.0
        
        # RSU连接性评分
        for rsu_pos in self.rsu_positions:
            distance = self._calculate_distance(position, rsu_pos)
            if distance <= self.rsu_coverage_radius:
                # 距离越近，连接质量越好
                rsu_score = 1.0 - (distance / self.rsu_coverage_radius)
                connectivity_score += rsu_score * 0.6  # RSU权重0.6
                
        # UAV连接性评分
        for uav_pos in self.uav_positions:
            distance = self._calculate_distance(position, uav_pos)
            if distance <= self.uav_coverage_radius:
                # UAV连接性评分，考虑电池状态
                uav_score = 1.0 - (distance / self.uav_coverage_radius)
                connectivity_score += uav_score * 0.4  # UAV权重0.4
                
        return min(connectivity_score, 1.0)
    
    def predict_trajectory(self, time_horizon: float = 3.0, time_step: float = 0.3) -> List[TrajectoryPoint]:
        """
        预测未来轨迹
        基于当前速度、加速度和移动策略
        """
        predicted_points = []
        current_pos = Position(self.current_position.x, self.current_position.y, 0)
        current_vel = self.velocity.copy()
        current_acc = self.acceleration.copy()
        
        steps = int(time_horizon / time_step)
        
        for i in range(steps):
            # 根据移动策略调整加速度
            strategy_acc = self._calculate_strategy_acceleration(current_pos, current_vel)
            current_acc = self._smooth_acceleration(current_acc, strategy_acc, 0.3)
            
            # 更新速度和位置
            current_vel += current_acc * time_step
            current_vel = self._limit_velocity(current_vel)
            
            new_x = current_pos.x + current_vel[0] * time_step
            new_y = current_pos.y + current_vel[1] * time_step
            
            # 边界处理
            new_x, new_y, current_vel = self._handle_boundaries(new_x, new_y, current_vel)
            
            current_pos = Position(new_x, new_y, 0)
            connectivity_score = self.calculate_connectivity_score(current_pos)
            
            predicted_point = TrajectoryPoint(
                position=current_pos,
                timestamp=i * time_step,
                velocity=current_vel.copy(),
                acceleration=current_acc.copy(),
                connectivity_score=connectivity_score
            )
            predicted_points.append(predicted_point)
            
        self.predicted_trajectory = predicted_points
        return predicted_points
    
    def _calculate_strategy_acceleration(self, position: Position, velocity: np.ndarray) -> np.ndarray:
        """
        根据移动策略计算目标加速度
        """
        if self.current_strategy == MobilityStrategy.CONNECTIVITY_SEEKING:
            return self._connectivity_seeking_acceleration(position)
        elif self.current_strategy == MobilityStrategy.COVERAGE_OPTIMIZATION:
            return self._coverage_optimization_acceleration(position)
        elif self.current_strategy == MobilityStrategy.ENERGY_EFFICIENT:
            return self._energy_efficient_acceleration(velocity)
        else:
            return self._random_walk_acceleration()
    
    def _connectivity_seeking_acceleration(self, position: Position) -> np.ndarray:
        """
        连接性寻求策略：向连接性更好的区域移动
        """
        best_direction = np.array([0.0, 0.0])
        best_score = self.calculate_connectivity_score(position)
        
        # 采样周围8个方向
        for angle in np.linspace(0, 2*math.pi, 8, endpoint=False):
            sample_distance = 50.0  # 采样距离
            sample_x = position.x + sample_distance * math.cos(angle)
            sample_y = position.y + sample_distance * math.sin(angle)
            sample_pos = Position(sample_x, sample_y, 0)
            
            score = self.calculate_connectivity_score(sample_pos)
            if score > best_score:
                best_score = score
                best_direction = np.array([math.cos(angle), math.sin(angle)])
        
        return best_direction * self.max_acceleration * 0.5
    
    def _coverage_optimization_acceleration(self, position: Position) -> np.ndarray:
        """
        覆盖优化策略：在覆盖区域边缘移动以最大化服务范围
        """
        # 寻找覆盖区域边缘
        edge_direction = np.array([0.0, 0.0])
        
        for rsu_pos in self.rsu_positions:
            distance = self._calculate_distance(position, rsu_pos)
            if distance < self.rsu_coverage_radius:
                # 在RSU覆盖范围内，向边缘移动
                direction = np.array([position.x - rsu_pos.x, position.y - rsu_pos.y])
                direction = direction / (np.linalg.norm(direction) + 1e-6)
                edge_direction += direction * 0.3
        
        # 添加随机扰动以避免局部最优
        random_direction = np.random.normal(0, 0.1, 2)
        edge_direction += random_direction
        
        # 归一化并应用加速度
        if np.linalg.norm(edge_direction) > 0:
            edge_direction = edge_direction / np.linalg.norm(edge_direction)
            return edge_direction * self.max_acceleration * 0.4
        
        return self._random_walk_acceleration()
    
    def _energy_efficient_acceleration(self, velocity: np.ndarray) -> np.ndarray:
        """
        能效优化策略：保持稳定速度，减少急加速
        """
        target_speed = 15.0  # 目标巡航速度
        current_speed = np.linalg.norm(velocity)
        
        if current_speed < target_speed:
            # 加速到目标速度
            if current_speed > 0:
                direction = velocity / current_speed
            else:
                direction = np.random.normal(0, 1, 2)
                direction = direction / np.linalg.norm(direction)
            return direction * self.max_acceleration * 0.3
        else:
            # 减速或保持
            return -velocity * 0.1
    
    def _random_walk_acceleration(self) -> np.ndarray:
        """
        随机游走策略
        """
        angle = np.random.uniform(0, 2 * math.pi)
        magnitude = np.random.uniform(0, self.max_acceleration * 0.5)
        return np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)])
    
    def _smooth_acceleration(self, current_acc: np.ndarray, target_acc: np.ndarray, smoothing: float) -> np.ndarray:
        """
        平滑加速度变化
        """
        return current_acc * (1 - smoothing) + target_acc * smoothing
    
    def _limit_velocity(self, velocity: np.ndarray) -> np.ndarray:
        """
        限制速度在合理范围内
        """
        speed = np.linalg.norm(velocity)
        if speed > self.max_speed:
            return velocity * (self.max_speed / speed)
        elif speed < self.min_speed and speed > 0:
            return velocity * (self.min_speed / speed)
        return velocity
    
    def _handle_boundaries(self, x: float, y: float, velocity: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        处理边界碰撞
        """
        area_width = config.network.area_width
        area_height = config.network.area_height
        
        new_velocity = velocity.copy()
        
        if x <= 0 or x >= area_width:
            new_velocity[0] = -new_velocity[0]
            x = max(0, min(area_width, x))
        
        if y <= 0 or y >= area_height:
            new_velocity[1] = -new_velocity[1]
            y = max(0, min(area_height, y))
        
        return x, y, new_velocity
    
    def _calculate_distance(self, pos1: Position, pos2: Position) -> float:
        """
        计算两点间距离
        """
        return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)
    
    def update_mobility_strategy(self, strategy_params: Dict[str, float]):
        """
        根据智能体动作更新移动策略
        """
        # 从智能体动作中提取移动策略参数
        mobility_strategy = strategy_params.get('mobility_strategy', 0.0)
        speed_adjustment = strategy_params.get('speed_adjustment', 0.0)
        route_optimization = strategy_params.get('route_optimization', 0.0)
        
        # 根据mobility_strategy选择策略
        if mobility_strategy < -0.5:
            self.current_strategy = MobilityStrategy.ENERGY_EFFICIENT
        elif mobility_strategy < 0:
            self.current_strategy = MobilityStrategy.RANDOM_WALK
        elif mobility_strategy < 0.5:
            self.current_strategy = MobilityStrategy.CONNECTIVITY_SEEKING
        else:
            self.current_strategy = MobilityStrategy.COVERAGE_OPTIMIZATION
        
        # 调整速度参数
        speed_factor = (speed_adjustment + 1) / 2  # 归一化到[0,1]
        self.max_speed = 15.0 + speed_factor * 15.0  # 15-30 m/s
        
        # 调整策略权重
        if route_optimization > 0:
            self.strategy_weights['connectivity'] = 0.5
            self.strategy_weights['coverage'] = 0.3
        else:
            self.strategy_weights['connectivity'] = 0.3
            self.strategy_weights['coverage'] = 0.4
    
    def step(self, time_step: float) -> Position:
        """
        执行一步移动更新
        """
        # 计算策略加速度
        strategy_acc = self._calculate_strategy_acceleration(self.current_position, self.velocity)
        
        # 平滑加速度变化
        self.acceleration = self._smooth_acceleration(self.acceleration, strategy_acc, 0.3)
        
        # 更新速度
        self.velocity += self.acceleration * time_step
        self.velocity = self._limit_velocity(self.velocity)
        
        # 更新位置
        new_x = self.current_position.x + self.velocity[0] * time_step
        new_y = self.current_position.y + self.velocity[1] * time_step
        
        # 边界处理
        new_x, new_y, self.velocity = self._handle_boundaries(new_x, new_y, self.velocity)
        
        # 更新当前位置
        self.current_position = Position(new_x, new_y, 0)
        
        # 记录轨迹
        connectivity_score = self.calculate_connectivity_score(self.current_position)
        trajectory_point = TrajectoryPoint(
            position=self.current_position,
            timestamp=time_step,
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            connectivity_score=connectivity_score
        )
        
        self.trajectory_history.append(trajectory_point)
        
        # 限制历史长度
        if len(self.trajectory_history) > self.trajectory_window:
            self.trajectory_history.pop(0)
        
        return self.current_position
    
    def get_mobility_metrics(self) -> Dict[str, float]:
        """
        获取移动性相关指标
        """
        if len(self.trajectory_history) < 2:
            return {
                'average_speed': 0.0,
                'connectivity_score': 0.0,
                'trajectory_efficiency': 0.0,
                'energy_efficiency': 1.0
            }
        
        # 平均速度
        speeds = [np.linalg.norm(point.velocity) for point in self.trajectory_history]
        average_speed = np.mean(speeds)
        
        # 平均连接性评分
        connectivity_scores = [point.connectivity_score for point in self.trajectory_history]
        average_connectivity = np.mean(connectivity_scores)
        
        # 轨迹效率（直线距离/实际距离）
        if len(self.trajectory_history) >= 10:
            start_pos = self.trajectory_history[-10].position
            end_pos = self.trajectory_history[-1].position
            straight_distance = self._calculate_distance(start_pos, end_pos)
            
            actual_distance = 0.0
            for i in range(len(self.trajectory_history)-9, len(self.trajectory_history)):
                if i > 0:
                    actual_distance += self._calculate_distance(
                        self.trajectory_history[i-1].position,
                        self.trajectory_history[i].position
                    )
            
            trajectory_efficiency = straight_distance / (actual_distance + 1e-6)
        else:
            trajectory_efficiency = 1.0
        
        # 能效评估（基于速度变化）
        if len(speeds) >= 2:
            speed_variance = np.var(speeds)
            energy_efficiency = 1.0 / (1.0 + speed_variance / 100.0)  # 速度变化越小，能效越高
        else:
            energy_efficiency = 1.0
        
        return {
            'average_speed': average_speed,
            'connectivity_score': average_connectivity,
            'trajectory_efficiency': trajectory_efficiency,
            'energy_efficiency': energy_efficiency
        }