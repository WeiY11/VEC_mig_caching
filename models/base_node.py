"""
VEC系统节点实现 - 对应论文第2节系统模型
包含车辆、RSU、UAV三种节点类型的具体实现
"""
import numpy as np
import math
import time
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .data_structures import (
    Task, QueueSlot, Position, NodeState, NodeType, 
    CommunicationLink, TaskType
)
from config import config
from utils import generate_poisson_arrivals, ExponentialMovingAverage


class BaseNode(ABC):
    """
    抽象基础节点类
    定义所有计算节点的通用接口和属性
    """
    def __init__(self, node_id: str, node_type: NodeType, position: Position):
        self.node_id = node_id
        self.node_type = node_type
        self.state = NodeState(node_id=node_id, node_type=node_type, position=position)
        
        # 多优先级生命周期队列 - 对应论文第2.3节
        self.queues: Dict[Tuple[int, int], QueueSlot] = {}
        self._initialize_queues()
        
        # 性能统计
        self.processed_tasks: List[Task] = []
        self.dropped_tasks: List[Task] = []
        self.energy_consumption_history: List[float] = []
        
        # 平均任务复杂度统计 - 修复单位：字节转比特
        self._avg_task_complexity: float = config.task.task_compute_density * float(np.mean(config.task.data_size_range)) * 8
        
        # 移动平均计算器
        self.avg_arrival_rate = ExponentialMovingAverage(alpha=0.1)
        self.avg_service_rate = ExponentialMovingAverage(alpha=0.1)
        self.avg_waiting_time = ExponentialMovingAverage(alpha=0.1)
    
    def _initialize_queues(self):
        """初始化多优先级生命周期队列结构"""
        max_lifetime = config.queue.max_lifetime
        num_priorities = config.task.num_priority_levels
        
        # 根据节点类型确定队列维度
        if self.node_type == NodeType.VEHICLE:
            # 车辆维护完整的L×P队列矩阵
            lifetime_range = range(1, max_lifetime + 1)
        else:
            # RSU和UAV维护(L-1)×P队列矩阵
            lifetime_range = range(1, max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, num_priorities + 1):
                self.queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
    
    @abstractmethod
    def get_processing_capacity(self) -> float:
        """获取处理能力 (bits/时隙)"""
        pass
    
    @abstractmethod
    def calculate_processing_delay(self, task: Task) -> float:
        """计算任务处理时延"""
        pass
    
    @abstractmethod  
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """计算能耗"""
        pass
    
    def add_task_to_queue(self, task: Task) -> bool:
        """
        将任务添加到相应的队列槽位
        根据任务的剩余生命周期和优先级确定槽位
        """
        lifetime = task.remaining_lifetime_slots
        priority = task.priority
        
        # 检查队列容量
        if self._check_queue_capacity():
            queue_key = (lifetime, priority)
            if queue_key in self.queues:
                return self.queues[queue_key].add_task(task)
            else:
                # 如果生命周期超出范围，任务被丢弃
                self.dropped_tasks.append(task)
                task.is_dropped = True
                return False
        else:
            # 队列已满，任务被丢弃
            self.dropped_tasks.append(task)
            task.is_dropped = True
            return False
    
    def _check_queue_capacity(self) -> bool:
        """检查队列容量是否允许新任务"""
        total_data = sum(queue.data_volume for queue in self.queues.values())
        
        if self.node_type == NodeType.VEHICLE:
            return total_data < config.queue.vehicle_queue_capacity
        elif self.node_type == NodeType.RSU:
            return total_data < config.queue.rsu_queue_capacity
        else:  # UAV
            return total_data < config.queue.uav_queue_capacity
    
    def get_next_task_to_process(self) -> Optional[Task]:
        """
        获取下一个待处理任务
        按照非抢占式优先级调度策略: 高优先级优先，同优先级FIFO
        """
        # 按优先级从高到低遍历 (priority=1是最高优先级)
        for priority in range(1, config.task.num_priority_levels + 1):
            # 在同一优先级内，按生命周期紧迫程度遍历
            for lifetime in sorted(self.queues.keys()):
                if lifetime[1] == priority:  # 匹配优先级
                    queue = self.queues[lifetime]
                    if not queue.is_empty():
                        return queue.get_next_task()
        return None
    
    def process_task(self, task: Task) -> bool:
        """
        处理任务
        返回是否成功处理
        """
        # 计算处理时延
        processing_delay = self.calculate_processing_delay(task)
        
        # 检查是否超出截止时间
        if task.is_deadline_violated():
            self.dropped_tasks.append(task)
            task.is_dropped = True
            return False
        
        # 执行任务处理
        task.start_time = time.time()
        task.processing_delay = processing_delay
        task.assigned_node_id = self.node_id
        
        # 计算能耗
        energy_cost = self.calculate_energy_consumption(processing_delay)
        self.state.total_energy += energy_cost
        self.energy_consumption_history.append(energy_cost)
        # 限制历史记录长度，防止内存泄漏
        if len(self.energy_consumption_history) > 100:
            self.energy_consumption_history.pop(0)
        
        # 模拟处理完成
        task.completion_time = task.start_time + processing_delay
        task.is_completed = True
        
        # 从队列中移除任务
        self._remove_task_from_queue(task)
        
        # 添加到已处理任务列表
        self.processed_tasks.append(task)
        # 限制已处理任务列表长度，防止内存泄漏
        if len(self.processed_tasks) > 50:
            self.processed_tasks.pop(0)
        
        # 更新统计信息
        self._update_statistics()
        
        return True
    
    def _remove_task_from_queue(self, task: Task):
        """从队列中移除指定任务"""
        for queue in self.queues.values():
            removed_task = queue.remove_task(task.task_id)
            if removed_task:
                break
    
    def predict_waiting_time(self, task: Task) -> float:
        """
        预测任务等待时间 - 对应论文式(2)和式(3)
        使用M/M/1非抢占式优先级队列模型
        """
        priority = task.priority
        
        # 计算到达率和服务率
        arrival_rates = self._calculate_arrival_rates_by_priority()
        service_rate = self._calculate_service_rate()
        
        if service_rate <= 0:
            return float('inf')
        
        # 计算负载因子
        rho_values = {p: arrival_rates.get(p, 0) / service_rate 
                     for p in range(1, config.task.num_priority_levels + 1)}
        
        # 检查稳定性条件
        total_rho = sum(rho_values.values())
        if total_rho >= 1.0:
            return float('inf')
        
        # 计算优先级为priority的任务平均等待时间 - 论文式(2)
        numerator = sum(rho_values[p] for p in range(1, priority + 1))
        
        denominator1 = 1 - sum(rho_values[p] for p in range(1, priority))
        denominator2 = 1 - sum(rho_values[p] for p in range(1, priority + 1))
        
        if denominator1 <= 0 or denominator2 <= 0:
            return float('inf')
        
        waiting_time = (1 / service_rate) * (numerator / (denominator1 * denominator2))
        
        return waiting_time
    
    def _calculate_arrival_rates_by_priority(self) -> Dict[int, float]:
        """计算各优先级任务的到达率"""
        arrival_rates = {}
        for priority in range(1, config.task.num_priority_levels + 1):
            # 统计各优先级队列中的任务数量
            total_tasks = sum(len(queue.task_list) 
                            for (l, p), queue in self.queues.items() 
                            if p == priority)
            # 转换为到达率 (简化估算)
            arrival_rates[priority] = total_tasks / config.network.time_slot_duration
        return arrival_rates
    
    def _calculate_service_rate(self) -> float:
        """计算平均服务率 (tasks/秒)"""
        if hasattr(self, '_avg_task_complexity') and self._avg_task_complexity > 0:
            return self.state.cpu_frequency / self._avg_task_complexity
        else:
            # 使用默认计算复杂度估算 - 修复单位：字节转比特
            avg_complexity = config.task.task_compute_density * float(np.mean(config.task.task_data_size_range)) * 8
            return self.state.cpu_frequency / avg_complexity
    
    def _update_statistics(self):
        """更新节点统计信息"""
        # 更新负载因子
        if len(self.processed_tasks) > 0:
            recent_tasks = self.processed_tasks[-10:]  # 最近10个任务
            avg_arrival = len(recent_tasks) / (10 * config.network.time_slot_duration)
            self.avg_arrival_rate.update(avg_arrival)
            
            avg_service = self._calculate_service_rate()
            self.avg_service_rate.update(avg_service)
            
            self.state.update_load_factor(
                self.avg_arrival_rate.get_value(),
                self.avg_service_rate.get_value()
            )
        
        # 更新队列长度
        self.state.queue_length = sum(len(queue.task_list) for queue in self.queues.values())
        
        # 更新平均等待时间
        if len(self.processed_tasks) > 0:
            recent_waiting_times = [task.waiting_delay for task in self.processed_tasks[-10:]]
            if recent_waiting_times:
                avg_waiting_time = float(np.mean(recent_waiting_times))
                self.avg_waiting_time.update(avg_waiting_time)
                self.state.avg_waiting_time = self.avg_waiting_time.get_value()
    
    def update_queue_lifetimes(self):
        """
        更新队列中任务的生命周期
        每个时隙开始时调用
        """
        # 创建新的队列结构
        new_queues = {}
        
        for (lifetime, priority), queue in self.queues.items():
            if queue.is_empty():
                # 保持空队列结构
                new_queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
            else:
                # 将任务移动到新的生命周期队列
                new_lifetime = max(1, lifetime - 1)
                new_key = (new_lifetime, priority)
                
                if new_key not in new_queues:
                    new_queues[new_key] = QueueSlot(new_lifetime, priority)
                
                # 移动任务
                for task in queue.task_list:
                    if new_lifetime > 0:
                        new_queues[new_key].add_task(task)
                    else:
                        # 生命周期用尽，任务被丢弃
                        task.is_dropped = True
                        self.dropped_tasks.append(task)
        
        self.queues = new_queues
    
    def get_state_vector(self) -> np.ndarray:
        """获取节点状态向量，用于强化学习"""
        state_features = [
            self.state.cpu_utilization,
            self.state.load_factor,
            self.state.queue_length / 100.0,  # 归一化
            self.state.avg_waiting_time / 10.0,  # 归一化
            len(self.processed_tasks) / 1000.0,  # 归一化
            len(self.dropped_tasks) / 1000.0,   # 归一化
        ]
        
        # 添加各优先级队列状态
        for priority in range(1, config.task.num_priority_levels + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            state_features.append(priority_tasks / 50.0)  # 归一化
        
        return np.array(state_features, dtype=np.float32)