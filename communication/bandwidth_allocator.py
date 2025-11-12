#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态带宽分配调度器

【功能】
实现智能带宽调度算法，替代硬编码的"总带宽/4"固定分配

【算法】
基于优先级+负载+信道质量的混合调度策略：
- 考虑任务优先级（高优先级获得更多带宽）
- 考虑任务数据量（大任务适当倾斜）
- 考虑信道质量SINR（好信道多分配，提升效率）
- 保证最小带宽（避免饿死）
- 总带宽约束（不超预算）

【论文价值】
体现资源优化能力，符合VEC系统智能调度理念

【使用示例】
```python
from communication.bandwidth_allocator import BandwidthAllocator

allocator = BandwidthAllocator(total_bandwidth=100e6)  # 100 MHz

# 活跃链路
active_links = [
    {'task_id': 't1', 'priority': 1, 'sinr': 20.0, 'data_size': 5e6},
    {'task_id': 't2', 'priority': 3, 'sinr': 15.0, 'data_size': 2e6},
]

# 分配带宽
allocations = allocator.allocate_bandwidth(active_links)
# 返回: {'t1': 70000000.0, 't2': 30000000.0}
```
"""

import numpy as np
from typing import Dict, List, Optional


class BandwidthAllocator:
    """
    带宽分配调度器
    
    【策略】
    基于优先级+负载+信道质量的加权比例公平调度
    
    【参数】
    - total_bandwidth: 总可用带宽 (Hz)
    - min_bandwidth: 最小保证带宽 (Hz)
    - priority_weight: 优先级权重系数
    - quality_weight: 信道质量权重系数
    - size_weight: 数据量权重系数
    """
    
    def __init__(
        self,
        total_bandwidth: float = 100e6,
        min_bandwidth: float = 1e6,
        priority_weight: float = 0.4,
        quality_weight: float = 0.3,
        size_weight: float = 0.3
    ):
        """
        初始化带宽分配器
        
        Args:
            total_bandwidth: 总带宽 (Hz)，默认100 MHz
            min_bandwidth: 最小保证带宽 (Hz)，默认1 MHz
            priority_weight: 优先级权重（0-1）
            quality_weight: 信道质量权重（0-1）
            size_weight: 数据量权重（0-1）
        """
        self.total_bandwidth = total_bandwidth
        self.min_bandwidth = min_bandwidth
        
        # 归一化权重
        total_weight = priority_weight + quality_weight + size_weight
        self.priority_weight = priority_weight / total_weight
        self.quality_weight = quality_weight / total_weight
        self.size_weight = size_weight / total_weight
        
        # 统计信息
        self.allocation_history = []
    
    def allocate_bandwidth(
        self,
        active_links: List[Dict],
        allocation_mode: str = 'hybrid'
    ) -> Dict[str, float]:
        """
        分配带宽给活跃链路
        
        Args:
            active_links: 活跃链路列表，每项包含：
                - task_id: 任务ID (str)
                - priority: 优先级 (int, 1-4, 1最高)
                - sinr: 信道SINR (linear或dB)
                - data_size: 数据量 (bits)
                - node_type: 可选，节点类型
            allocation_mode: 分配模式
                - 'hybrid': 混合策略（默认）
                - 'priority_only': 仅按优先级
                - 'quality_only': 仅按信道质量
                - 'equal': 平均分配
        
        Returns:
            带宽分配字典 {task_id: allocated_bandwidth (Hz)}
        """
        if not active_links:
            return {}
        
        n_links = len(active_links)
        
        # 特殊情况：单链路
        if n_links == 1:
            task_id = active_links[0]['task_id']
            return {task_id: self.total_bandwidth}
        
        # 根据模式选择分配策略
        if allocation_mode == 'equal':
            return self._allocate_equal(active_links)
        elif allocation_mode == 'priority_only':
            return self._allocate_priority_only(active_links)
        elif allocation_mode == 'quality_only':
            return self._allocate_quality_only(active_links)
        else:  # hybrid
            return self._allocate_hybrid(active_links)
    
    def _allocate_hybrid(self, active_links: List[Dict]) -> Dict[str, float]:
        """
        混合策略：优先级 + 信道质量 + 数据量
        
        权重公式：
        W_i = α × (5-priority_i)/4 + β × sqrt(SINR_i)/max(sqrt(SINR)) 
              + γ × min(data_size_i/1MB, 10)/10
        """
        weights = {}
        
        # 计算归一化因子
        max_sinr_sqrt = 0.0
        for link in active_links:
            sinr = link.get('sinr', 10.0)
            if sinr > 1000:  # 判断是否为dB值（通常>100dB不合理）
                sinr = 10 ** (sinr / 10)  # dB转线性
            sinr = max(sinr, 0.1)  # 避免除零
            max_sinr_sqrt = max(max_sinr_sqrt, np.sqrt(sinr))
        
        max_sinr_sqrt = max(max_sinr_sqrt, 1.0)
        
        # 计算每个链路的权重
        for link in active_links:
            task_id = link['task_id']
            
            # 1. 优先级权重（1最高，4最低，归一化到0-1）
            priority = link.get('priority', 3)
            priority_norm = (5 - priority) / 4.0  # 1->1.0, 4->0.25
            
            # 2. 信道质量权重（SINR越高，权重越大）
            sinr = link.get('sinr', 10.0)
            if sinr > 1000:
                sinr = 10 ** (sinr / 10)
            sinr = max(sinr, 0.1)
            quality_norm = np.sqrt(sinr) / max_sinr_sqrt
            
            # 3. 数据量权重（限制最大影响为10）
            data_size = link.get('data_size', 1e6)
            size_norm = min(data_size / 1e6, 10.0) / 10.0
            
            # 综合权重
            weight = (
                self.priority_weight * priority_norm +
                self.quality_weight * quality_norm +
                self.size_weight * size_norm
            )
            
            weights[task_id] = max(weight, 0.01)  # 避免权重为0
        
        # 按权重比例分配
        return self._proportional_allocation(weights)
    
    def _allocate_priority_only(self, active_links: List[Dict]) -> Dict[str, float]:
        """仅按优先级分配"""
        weights = {}
        for link in active_links:
            task_id = link['task_id']
            priority = link.get('priority', 3)
            # 优先级1权重4，优先级4权重1
            weights[task_id] = 5 - priority
        
        return self._proportional_allocation(weights)
    
    def _allocate_quality_only(self, active_links: List[Dict]) -> Dict[str, float]:
        """仅按信道质量分配"""
        weights = {}
        for link in active_links:
            task_id = link['task_id']
            sinr = link.get('sinr', 10.0)
            if sinr > 1000:
                sinr = 10 ** (sinr / 10)
            weights[task_id] = max(sinr, 0.1)
        
        return self._proportional_allocation(weights)
    
    def _allocate_equal(self, active_links: List[Dict]) -> Dict[str, float]:
        """平均分配"""
        n_links = len(active_links)
        bw_per_link = self.total_bandwidth / n_links
        
        allocations = {}
        for link in active_links:
            allocations[link['task_id']] = bw_per_link
        
        return allocations
    
    def _proportional_allocation(self, weights: Dict[str, float]) -> Dict[str, float]:
        """
        按权重比例分配，保证最小带宽
        
        Args:
            weights: {task_id: weight}
        
        Returns:
            {task_id: allocated_bandwidth}
        """
        n_tasks = len(weights)
        total_weight = sum(weights.values())
        
        if total_weight == 0:
            # 权重全为0，平均分配
            avg_bw = self.total_bandwidth / n_tasks
            return {task_id: avg_bw for task_id in weights.keys()}
        
        allocations = {}
        remaining_bw = self.total_bandwidth
        guaranteed_tasks = []  # 需要最小保证的任务
        
        # 第一轮：按比例分配
        for task_id, weight in weights.items():
            allocated = (weight / total_weight) * self.total_bandwidth
            
            if allocated < self.min_bandwidth:
                # 分配的带宽低于最小保证，先记录
                allocations[task_id] = self.min_bandwidth
                guaranteed_tasks.append(task_id)
                remaining_bw -= self.min_bandwidth
            else:
                allocations[task_id] = allocated
                remaining_bw -= allocated
        
        # 第二轮：重新分配（如果有最小保证的任务）
        if guaranteed_tasks and remaining_bw < 0:
            # 带宽不足，需要调整
            # 简化处理：所有任务等分
            avg_bw = self.total_bandwidth / n_tasks
            for task_id in allocations:
                allocations[task_id] = max(avg_bw, self.min_bandwidth * 0.5)
        
        # 确保不超过总带宽
        current_total = sum(allocations.values())
        if current_total > self.total_bandwidth:
            scale_factor = self.total_bandwidth / current_total
            for task_id in allocations:
                allocations[task_id] *= scale_factor
        
        return allocations
    
    def get_allocation_stats(self, allocations: Dict[str, float]) -> Dict:
        """
        获取分配统计信息
        
        Args:
            allocations: 带宽分配结果
        
        Returns:
            统计信息字典
        """
        if not allocations:
            return {
                'total_allocated': 0,
                'num_links': 0,
                'avg_bandwidth': 0,
                'min_bandwidth': 0,
                'max_bandwidth': 0,
                'utilization': 0
            }
        
        bw_values = list(allocations.values())
        total_allocated = sum(bw_values)
        
        return {
            'total_allocated': total_allocated,
            'num_links': len(allocations),
            'avg_bandwidth': np.mean(bw_values),
            'min_bandwidth': np.min(bw_values),
            'max_bandwidth': np.max(bw_values),
            'std_bandwidth': np.std(bw_values),
            'utilization': total_allocated / self.total_bandwidth
        }
    
    def update_total_bandwidth(self, new_bandwidth: float):
        """更新总带宽（适应网络条件变化）"""
        self.total_bandwidth = new_bandwidth


# 便捷函数
def allocate_bandwidth_simple(
    active_tasks: List[Dict],
    total_bandwidth: float = 100e6
) -> Dict[str, float]:
    """
    简化的带宽分配接口
    
    Args:
        active_tasks: 活跃任务列表
        total_bandwidth: 总带宽
    
    Returns:
        带宽分配结果
    """
    allocator = BandwidthAllocator(total_bandwidth=total_bandwidth)
    return allocator.allocate_bandwidth(active_tasks)




