#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
固定UAV专门动作空间设计
针对悬停UAV的服务决策优化
"""

import numpy as np
from typing import Dict, List, Tuple
from enum import Enum

class UAVActionType(Enum):
    """固定UAV动作类型枚举"""
    POWER_MANAGEMENT = "power_management"  # 电池功率管理
    SERVICE_PRIORITY = "service_priority"  # 服务优先级调整
    COVERAGE_OPTIMIZATION = "coverage_optimization"  # 覆盖区域优化
    EMERGENCY_RESPONSE = "emergency_response"  # 应急响应模式
    COORDINATION_MODE = "coordination_mode"  # 与其他节点协调
    HOVER_EFFICIENCY = "hover_efficiency"  # 悬停效率调节
    BANDWIDTH_ALLOCATION = "bandwidth_allocation"  # 带宽分配策略
    CACHE_STRATEGY = "cache_strategy"  # 缓存策略选择

class FixedUAVActionSpace:
    """固定UAV专门动作空间"""
    
    def __init__(self):
        # 固定UAV的8维专业化动作空间
        self.action_dim = 8
        self.action_bounds = (-1.0, 1.0)  # 连续动作范围
        
        # 动作维度定义
        self.action_definitions = {
            0: "电池功率管理级别 [-1,1] -> [节能模式, 高性能模式]",
            1: "服务优先级权重 [-1,1] -> [低优先级, 高优先级]", 
            2: "覆盖区域调整 [-1,1] -> [缩小覆盖, 扩大覆盖]",
            3: "应急响应敏感度 [-1,1] -> [标准模式, 应急模式]",
            4: "协调积极性 [-1,1] -> [独立工作, 积极协调]",
            5: "悬停效率优化 [-1,1] -> [稳定悬停, 动态调整]",
            6: "带宽分配策略 [-1,1] -> [保守分配, 激进分配]",
            7: "缓存策略 [-1,1] -> [被动缓存, 主动预缓存]"
        }
        
    def interpret_action(self, action: np.ndarray) -> Dict[str, float]:
        """解释UAV动作向量为具体决策参数"""
        if len(action) != self.action_dim:
            raise ValueError(f"动作维度不匹配: 期望{self.action_dim}, 实际{len(action)}")
            
        # 将[-1,1]范围的动作映射到具体参数
        interpreted = {
            'power_level': self._map_to_range(action[0], 0.3, 1.0),  # 功率级别 30%-100%
            'service_priority': self._map_to_range(action[1], 0.1, 1.0),  # 服务优先级权重
            'coverage_radius': self._map_to_range(action[2], 0.5, 1.5),  # 覆盖半径倍数
            'emergency_threshold': self._map_to_range(action[3], 0.2, 0.8),  # 应急响应阈值
            'coordination_weight': self._map_to_range(action[4], 0.0, 1.0),  # 协调权重
            'hover_stability': self._map_to_range(action[5], 0.6, 1.0),  # 悬停稳定性
            'bandwidth_ratio': self._map_to_range(action[6], 0.3, 1.0),  # 带宽分配比例
            'cache_aggressiveness': self._map_to_range(action[7], 0.1, 0.9)  # 缓存积极性
        }
        
        return interpreted
    
    def _map_to_range(self, value: float, min_val: float, max_val: float) -> float:
        """将[-1,1]范围映射到指定范围"""
        # 将[-1,1]映射到[0,1]
        normalized = (value + 1.0) / 2.0
        # 映射到目标范围
        return min_val + normalized * (max_val - min_val)
    
    def get_action_description(self) -> str:
        """获取动作空间描述"""
        desc = "固定UAV专业化动作空间 (8维连续动作):\n"
        for i, definition in self.action_definitions.items():
            desc += f"  [{i}] {definition}\n"
        return desc
    
    def validate_action(self, action: np.ndarray) -> bool:
        """验证动作是否有效"""
        if len(action) != self.action_dim:
            return False
        return np.all(action >= -1.0) and np.all(action <= 1.0)
    
    def get_default_action(self) -> np.ndarray:
        """获取默认动作（中性设置）"""
        return np.zeros(self.action_dim)
    
    def get_emergency_action(self) -> np.ndarray:
        """获取应急模式动作"""
        return np.array([
            0.5,   # 中等功率
            1.0,   # 最高优先级
            0.8,   # 扩大覆盖
            1.0,   # 最高应急敏感度
            0.9,   # 积极协调
            0.9,   # 高稳定性
            0.8,   # 较多带宽分配
            0.6    # 中等缓存积极性
        ])
    
    def get_energy_saving_action(self) -> np.ndarray:
        """获取节能模式动作"""
        return np.array([
            -0.8,  # 低功率
            -0.2,  # 较低优先级
            -0.3,  # 缩小覆盖
            -0.5,  # 标准应急模式
            0.2,   # 适度协调
            0.8,   # 高稳定性（节能）
            -0.4,  # 保守带宽分配
            -0.3   # 被动缓存
        ])

class UAVActionDecomposer:
    """UAV动作分解器 - 将动作转换为具体执行参数"""
    
    def __init__(self):
        self.action_space = FixedUAVActionSpace()
    
    def decompose_uav_action(self, action: np.ndarray) -> Dict[str, any]:
        """将UAV动作分解为具体执行参数"""
        if not self.action_space.validate_action(action):
            raise ValueError("无效的UAV动作")
        
        interpreted = self.action_space.interpret_action(action)
        
        # 转换为具体执行参数
        execution_params = {
            # 电池管理参数
            'battery_management': {
                'power_level': interpreted['power_level'],
                'energy_saving_mode': interpreted['power_level'] < 0.5
            },
            
            # 服务策略参数
            'service_strategy': {
                'priority_weight': interpreted['service_priority'],
                'coverage_radius_multiplier': interpreted['coverage_radius'],
                'emergency_response_enabled': interpreted['emergency_threshold'] > 0.5
            },
            
            # 协调参数
            'coordination': {
                'cooperation_weight': interpreted['coordination_weight'],
                'bandwidth_allocation_ratio': interpreted['bandwidth_ratio']
            },
            
            # 悬停控制参数
            'hover_control': {
                'stability_factor': interpreted['hover_stability'],
                'position_adjustment_enabled': interpreted['hover_stability'] < 0.8
            },
            
            # 缓存策略参数
            'cache_strategy': {
                'aggressiveness': interpreted['cache_aggressiveness'],
                'proactive_caching': interpreted['cache_aggressiveness'] > 0.5
            }
        }
        
        return execution_params
    
    def get_action_impact_metrics(self, action: np.ndarray) -> Dict[str, float]:
        """计算动作对各项指标的预期影响"""
        interpreted = self.action_space.interpret_action(action)
        
        # 预期影响评估
        impacts = {
            'battery_consumption_rate': 0.2 + 0.6 * interpreted['power_level'],  # 功率越高消耗越大
            'service_quality': 0.3 + 0.7 * interpreted['service_priority'],  # 优先级越高质量越好
            'coverage_efficiency': 0.4 + 0.6 * interpreted['coverage_radius'],  # 覆盖越大效率可能降低
            'emergency_readiness': interpreted['emergency_threshold'],  # 应急准备度
            'coordination_benefit': 0.2 + 0.8 * interpreted['coordination_weight'],  # 协调收益
            'hover_stability': interpreted['hover_stability'],  # 悬停稳定性
            'bandwidth_efficiency': 0.5 + 0.5 * interpreted['bandwidth_ratio'],  # 带宽效率
            'cache_hit_rate': 0.3 + 0.4 * interpreted['cache_aggressiveness']  # 缓存命中率
        }
        
        return impacts

# 使用示例和测试
if __name__ == "__main__":
    # 创建UAV动作空间
    uav_action_space = FixedUAVActionSpace()
    decomposer = UAVActionDecomposer()
    
    print("=== 固定UAV动作空间设计 ===")
    print(uav_action_space.get_action_description())
    
    # 测试不同动作模式
    print("\n=== 动作模式测试 ===")
    
    # 默认动作
    default_action = uav_action_space.get_default_action()
    print(f"\n默认动作: {default_action}")
    default_params = decomposer.decompose_uav_action(default_action)
    print(f"执行参数: {default_params['service_strategy']}")
    
    # 应急动作
    emergency_action = uav_action_space.get_emergency_action()
    print(f"\n应急动作: {emergency_action}")
    emergency_impacts = decomposer.get_action_impact_metrics(emergency_action)
    print(f"预期影响: 服务质量={emergency_impacts['service_quality']:.2f}, 电池消耗={emergency_impacts['battery_consumption_rate']:.2f}")
    
    # 节能动作
    energy_saving_action = uav_action_space.get_energy_saving_action()
    print(f"\n节能动作: {energy_saving_action}")
    energy_impacts = decomposer.get_action_impact_metrics(energy_saving_action)
    print(f"预期影响: 电池消耗={energy_impacts['battery_consumption_rate']:.2f}, 悬停稳定性={energy_impacts['hover_stability']:.2f}")