"""
模型模块初始化文件
包含VEC系统核心数据结构和节点实现
"""
from .data_structures import (
    Task, TaskType, QueueSlot, Position, NodeState, NodeType,
    CommunicationLink, SystemMetrics
)
from .base_node import BaseNode
from .vehicle_node import VehicleNode
from .rsu_node import RSUNode
from .uav_node import UAVNode

__all__ = [
    # 数据结构
    'Task', 'TaskType', 'QueueSlot', 'Position', 'NodeState', 'NodeType',
    'CommunicationLink', 'SystemMetrics',
    
    # 节点类
    'BaseNode', 'VehicleNode', 'RSUNode', 'UAVNode'
]