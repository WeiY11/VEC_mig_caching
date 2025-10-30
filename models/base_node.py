"""
VEC系统节点实现 - 对应论文第2节系统模型
包含车辆、RSU、UAV三种节点类型的具体实现
"""
import logging
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


logger = logging.getLogger(__name__)


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
        self._queue_data_usage: float = 0.0
        self._queue_instability_alerted = False
        
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
        ���������ӵ���Ӧ�Ķ��в�λ
        ���������ʣ���������ں����ȼ�ȷ����λ
        """
        lifetime = task.remaining_lifetime_slots
        priority = task.priority

        if lifetime <= 0:
            self._register_drop(task)
            return False

        if not self._has_capacity_for(task):
            logger.debug("Queue capacity reached for %s; dropping task %s", self.node_id, task.task_id)
            self._register_drop(task)
            return False

        queue_key = (lifetime, priority)
        if queue_key not in self.queues:
            self._register_drop(task)
            return False

        if self.queues[queue_key].add_task(task):
            self._queue_data_usage += task.data_size
            task.queue_arrival_time = time.time()
            task.waiting_delay = 0.0
            return True

        return False

    def _has_capacity_for(self, task: Task) -> bool:
        """�����������Ƿ������������"""
        capacity_limit = self._get_queue_capacity_limit()
        if capacity_limit <= 0:
            return True
        return (self._queue_data_usage + task.data_size) <= capacity_limit

    def _get_queue_capacity_limit(self) -> float:
        """��ȡ��ǰ�ڵ��Ӧ�Ķ�������Ϣ"""
        if self.node_type == NodeType.VEHICLE:
            return getattr(config.queue, 'vehicle_queue_capacity', -1.0)
        if self.node_type == NodeType.RSU:
            return getattr(config.queue, 'rsu_queue_capacity', -1.0)
        return getattr(config.queue, 'uav_queue_capacity', -1.0)

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
        ��������
        ���򷵻ؽ��Ƿ�ɹ��������
        """
        # ���㴦��ʱ��
        processing_delay = self.calculate_processing_delay(task)

        # ����Ƿ�Խ����ֹʱ��
        if task.is_deadline_violated():
            self._remove_task_from_queue(task)
            self._register_drop(task)
            return False

        # ִ����������
        task_start_time = time.time()
        task.start_time = task_start_time
        if task.queue_arrival_time is not None:
            task.waiting_delay = max(0.0, task_start_time - task.queue_arrival_time)
        else:
            task.waiting_delay = max(0.0, task.waiting_delay)
        task.processing_delay = processing_delay
        task.assigned_node_id = self.node_id

        # ���㹦�ˣ?
        energy_cost = self.calculate_energy_consumption(processing_delay)
        self.state.total_energy += energy_cost
        self.energy_consumption_history.append(energy_cost)
        # ������ʷ��¼������ֹ�ڴ����?
        if len(self.energy_consumption_history) > 100:
            self.energy_consumption_history.pop(0)

        # ģ�⴦��������
        task.completion_time = task_start_time + processing_delay
        task.is_completed = True

        # �Ӷ������Ƴ�����
        self._remove_task_from_queue(task)

        # ������Ѵ��������б?
        self.processed_tasks.append(task)
        # ������Ѵ��������б�������ֹ�ڴ洢��
        if len(self.processed_tasks) > 50:
            self.processed_tasks.pop(0)

        # ����ͳ����Ϣ
        self._update_statistics()

        return True

    def _remove_task_from_queue(self, task: Task) -> bool:
        """从队列中移除指定任务"""
        for queue in self.queues.values():
            removed_task = queue.remove_task(task.task_id)
            if removed_task:
                self._queue_data_usage = max(0.0, self._queue_data_usage - removed_task.data_size)
                removed_task.queue_arrival_time = None
                return True
        return False

    def _register_drop(self, task: Task):
        """记录任务被丢弃"""
        if not task.is_dropped:
            task.is_dropped = True
            self.dropped_tasks.append(task)
    
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
        warning_ratio = getattr(config.queue, 'stability_warning_ratio', 0.9)
        instability_threshold = getattr(config.queue, 'global_rho_threshold', 1.0)

        if total_rho >= warning_ratio:
            self.state.stability_warning = True
            if total_rho >= instability_threshold and not self._queue_instability_alerted:
                logger.warning("Queue at node %s is unstable (total rho=%.3f)", self.node_id, total_rho)
                self._queue_instability_alerted = True
        else:
            self.state.stability_warning = False
            self._queue_instability_alerted = False

        if total_rho >= instability_threshold:
            return float('inf')

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
        �����и����е����ָʻ��ڣ�ÿ��ʱ�غ���
        """
        if self.node_type == NodeType.VEHICLE:
            lifetime_range = range(1, config.queue.max_lifetime + 1)
        else:
            lifetime_range = range(1, config.queue.max_lifetime)

        new_queues = {}
        for lifetime in lifetime_range:
            for priority in range(1, config.task.num_priority_levels + 1):
                new_queues[(lifetime, priority)] = QueueSlot(lifetime, priority)

        for (lifetime, priority), queue in self.queues.items():
            for task in queue.task_list:
                if task.is_deadline_violated():
                    self._register_drop(task)
                    continue

                new_lifetime = lifetime - 1
                if new_lifetime >= 1 and (new_lifetime, priority) in new_queues:
                    new_queues[(new_lifetime, priority)].add_task(task)
                else:
                    self._register_drop(task)

        self.queues = new_queues
        self._queue_data_usage = sum(slot.data_volume for slot in self.queues.values())

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
