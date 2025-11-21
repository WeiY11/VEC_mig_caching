"""
多优先级生命周期队列管理器 - 对应论文第4.3节
实现VEC系统中的分层队列系统和M/M/1非抢占式优先级队列模型
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from models.data_structures import Task, QueueSlot, NodeType
from config import config
from utils import ExponentialMovingAverage


@dataclass
class QueueStatistics:
    """队列统计信息"""
    total_arrivals: int = 0
    total_departures: int = 0
    total_drops: int = 0
    avg_waiting_time: float = 0.0
    avg_queue_length: float = 0.0
    avg_service_time: float = 0.0
    
    def __post_init__(self):
        # 初始化优先级统计字典
        self.priority_arrivals: Dict[int, int] = defaultdict(int)
        self.priority_waiting_times: Dict[int, float] = defaultdict(float)


class PriorityQueueManager:
    """
    多优先级生命周期队列管理器
    
    实现功能:
    1. 多维队列管理 (生命周期 × 优先级)
    2. M/M/1非抢占式优先级队列预测
    3. 队列统计与性能分析
    4. 负载均衡与容量管理
    """
    
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        
        # 队列维度参数
        self.max_lifetime = config.queue.max_lifetime  # L
        self.num_priorities = config.task.num_priority_levels  # P
        
        # 队列结构 - {(lifetime, priority): QueueSlot}
        self.queues: Dict[Tuple[int, int], QueueSlot] = {}
        self._initialize_queues()
        
        # 容量限制
        self.max_capacity = self._get_queue_capacity()
        self.current_usage = 0.0
        
        # 统计信息
        self.statistics = QueueStatistics()
        
        # M/M/1模型参数
        self.arrival_rates: Dict[int, float] = defaultdict(float)  # λ_i (按优先级)
        self.service_rate: float = 0.0  # μ
        self.load_factors: Dict[int, float] = defaultdict(float)  # ρ_i = λ_i/μ
        
        # 移动平均计算器
        self.avg_calculators = {
            'arrival_rate': ExponentialMovingAverage(alpha=0.1),
            'service_rate': ExponentialMovingAverage(alpha=0.1),
            'waiting_time': ExponentialMovingAverage(alpha=0.1)
        }
        
        # 时间窗口统计
        self.time_window_size = 6  # 统计窗口大小 (时隙)
        self.recent_arrivals: List[Dict[int, int]] = []  # 最近到达统计
        self.recent_services: List[int] = []  # 最近服务统计
    
    def _initialize_queues(self):
        """初始化队列结构"""
        if self.node_type == NodeType.VEHICLE:
            # 车辆维护完整的L×P队列矩阵
            lifetime_range = range(1, self.max_lifetime + 1)
        else:
            # RSU和UAV维护(L-1)×P队列矩阵  
            lifetime_range = range(1, self.max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, self.num_priorities + 1):
                self.queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
    
    def _get_queue_capacity(self) -> float:
        """获取队列容量限制"""
        if self.node_type == NodeType.VEHICLE:
            return config.queue.vehicle_queue_capacity
        elif self.node_type == NodeType.RSU:
            return config.queue.rsu_queue_capacity
        else:  # UAV
            return config.queue.uav_queue_capacity
    
    def add_task(self, task: Task) -> bool:
        """
        添加任务到队列
        
        Args:
            task: 待添加的任务
            
        Returns:
            是否成功添加
        """
        # 检查容量限制
        if self.current_usage + task.data_size > self.max_capacity:
            self._handle_queue_overflow(task)
            return False
        
        # 确定队列位置
        lifetime = task.remaining_lifetime_slots
        priority = task.priority
        queue_key = (lifetime, priority)
        
        # 检查队列是否存在
        if queue_key not in self.queues:
            # 生命周期超出范围，任务被丢弃
            self.statistics.total_drops += 1
            return False
        
        # 添加到相应队列
        success = self.queues[queue_key].add_task(task)
        
        if success:
            self.current_usage += task.data_size
            self.statistics.total_arrivals += 1
            self.statistics.priority_arrivals[priority] += 1
            
            # 更新到达率统计
            self._update_arrival_statistics(priority)
        
        return success
    
    def get_next_task(self) -> Optional[Task]:
        """
        获取下一个待处理任务
        按照非抢占式优先级调度策略
        """
        # 按优先级从高到低遍历 (priority=1是最高优先级)
        for priority in range(1, self.num_priorities + 1):
            # 在同一优先级内，按生命周期紧迫程度遍历
            for lifetime in range(1, self.max_lifetime + 1):
                queue_key = (lifetime, priority)
                if queue_key in self.queues and not self.queues[queue_key].is_empty():
                    # 找到下一个任务
                    queue = self.queues[queue_key]
                    task = queue.get_next_task()
                    if task:
                        return task
        
        return None
    
    def remove_task(self, task: Task) -> bool:
        """
        从队列中移除任务
        
        Args:
            task: 待移除的任务
            
        Returns:
            是否成功移除
        """
        # 遍历所有队列找任务
        for queue in self.queues.values():
            removed_task = queue.remove_task(task.task_id)
            if removed_task:
                self.current_usage -= removed_task.data_size
                self.statistics.total_departures += 1
                
                # 更新服务统计
                self._update_service_statistics()
                
                return True
        
        return False
    
    def predict_waiting_time_mm1(self, task: Task) -> float:
        """
        🚀 创新优化:M/M/1非抢占式优先级队列模型 + 短期负载预测
        
        创新点:
        1. 融合短期负载趋势预测(提前上特征工程)
        2. 动态调整稳定性保障系数
        3. 考虑队列瞬时波动修正
        
        Args:
            task: 待预测的任务
            
        Returns:
            预测等待时间 (秒)
        """
        priority = task.priority
        
        # 检查优先级的有效性
        if priority < 1 or priority > len(self.load_factors):
            return float('inf')
        
        # 检查服务率的有效性
        if self.service_rate <= 1e-10:  # 防止除以零
            return float('inf')
        
        # 🆕 创新:动态稳定性阈值(根据当前负载调整)
        # 高负载时放宽阈值,低负载时提高阈值
        total_rho = sum(self.load_factors.values())
        if total_rho > 0.85:
            stability_threshold = 0.98  # 高负载放宽
        elif total_rho > 0.70:
            stability_threshold = 0.96
        else:
            stability_threshold = 0.95  # 低负载严格
        
        # 检查稳定性条件
        if total_rho >= stability_threshold:
            return float('inf')  # 系统不稳定
        
        # 🆕 创新:短期负载趋势预测(提前上特征工程)
        # 基于最近的到达率和服务率计算趋势
        load_trend_multiplier = 1.0
        if len(self.recent_arrivals) >= 3:
            # 计算最近到达率增长
            recent_rho_values = []
            for arrivals_dict in self.recent_arrivals[-3:]:
                slot_arrivals = arrivals_dict.get(priority, 0)
                slot_rho = slot_arrivals / max(1e-9, self.service_rate * config.network.time_slot_duration)
                recent_rho_values.append(slot_rho)
            
            if len(recent_rho_values) >= 2:
                # 趋势上升时,增加预测等待时间
                trend = recent_rho_values[-1] - recent_rho_values[0]
                if trend > 0.05:  # 明显上升趋势
                    load_trend_multiplier = 1.2
                elif trend < -0.05:  # 明显下降趋势
                    load_trend_multiplier = 0.9
        
        # 计算优先级为priority的任务平均等待时间 - 论文式(2)
        numerator = sum(self.load_factors.get(p, 0) for p in range(1, priority + 1))
        
        # 分母计算 - 添加数值稳定性检查
        denominator1 = 1 - sum(self.load_factors.get(p, 0) for p in range(1, priority))
        denominator2 = 1 - sum(self.load_factors.get(p, 0) for p in range(1, priority + 1))
        
        # 防止分母过小
        min_denominator = 1e-6
        if denominator1 <= min_denominator or denominator2 <= min_denominator:
            return float('inf')
        
        # 论文式(2): T_wait = (1/μ) * [Σρ_i] / [(1-Σρ_{i<p})(1-Σρ_{i≤p})]
        base_waiting_time = (1 / self.service_rate) * (numerator / (denominator1 * denominator2))
        
        # 🆕 创新:应用负载趋势修正
        waiting_time = base_waiting_time * load_trend_multiplier
        
        # 🆕 创新:队列瞬时波动修正(考虑当前实际队列长度)
        # 如果当前队列明显过载,增加预测时间
        current_queue_length = sum(len(queue.task_list) for (l, p), queue in self.queues.items() if p == priority)
        expected_queue_length = self.load_factors.get(priority, 0) / (1 - total_rho + 1e-9)
        
        if current_queue_length > expected_queue_length * 1.3:  # 超出预期30%
            congestion_factor = min(1.5, current_queue_length / max(1.0, expected_queue_length))
            waiting_time *= congestion_factor
        
        # 限制等待时间在合理范围内
        max_waiting_time = 100.0  # 最多100秒
        waiting_time = min(waiting_time, max_waiting_time)
        
        return max(0.0, waiting_time)  # 确保非负
    
    def predict_waiting_time_instantaneous(self, task: Task) -> float:
        """
        基于瞬时队列状态预测等待时间
        对应论文式(4)的瞬时积压预测
        
        Args:
            task: 待预测的任务
            
        Returns:
            预测等待时间 (秒)
        """
        priority = task.priority
        
        # 计算当前正在服务的任务剩余时间(简化)
        current_service_remaining = 0.0  # 实际中需要跟踪当前服务任务
        
        # 计算优先级更高的任务总处理时间
        higher_priority_workload = 0.0
        for p in range(1, priority):
            for (lifetime, prio), queue in self.queues.items():
                if prio == p and not queue.is_empty():
                    # 计算该队列的工作负载
                    queue_workload = queue.data_volume * config.task.task_compute_density
                    higher_priority_workload += queue_workload
        
        # 假设的平均CPU频率 (简化)
        avg_cpu_freq = self._get_average_cpu_frequency()
        
        # 瞬时等待时间预测 - 对应论文式(4)
        if avg_cpu_freq > 0:
            waiting_time = (current_service_remaining + higher_priority_workload) / avg_cpu_freq
        else:
            waiting_time = float('inf')
        
        return waiting_time
    
    def _get_average_cpu_frequency(self) -> float:
        """获取平均CPU频率 (简化实现)"""
        if self.node_type == NodeType.VEHICLE:
            freq_range = config.compute.vehicle_cpu_freq_range
            return float(np.mean(freq_range))
        elif self.node_type == NodeType.RSU:
            return config.compute.rsu_cpu_freq
        else:  # UAV
            return config.compute.uav_cpu_freq
    
    def update_lifetime(self):
        """
        更新所有任务的生命周期
        每个时隙开始时调用
        """
        new_queues = {}
        dropped_tasks = []
        
        for (lifetime, priority), queue in self.queues.items():
            if queue.is_empty():
                # 保持空队列结构
                new_queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
            else:
                # 计算新的生命周期
                new_lifetime = max(0, lifetime - 1)
                
                if new_lifetime > 0:
                    # 任务移动到新的生命周期队列
                    new_key = (new_lifetime, priority)
                    if new_key not in new_queues:
                        new_queues[new_key] = QueueSlot(new_lifetime, priority)
                    
                    # 移动所有任务
                    for task in queue.task_list:
                        new_queues[new_key].add_task(task)
                else:
                    # 生命周期用尽，任务被丢弃
                    for task in queue.task_list:
                        task.is_dropped = True
                        dropped_tasks.append(task)
                        self.current_usage -= task.data_size
                        self.statistics.total_drops += 1
        
        # 确保所有队列位置都有对应的队列对象
        self._ensure_all_queues_exist(new_queues)
        
        self.queues = new_queues
        
        return dropped_tasks
    
    def _ensure_all_queues_exist(self, queue_dict: Dict):
        """确保所有队列位置都存在"""
        if self.node_type == NodeType.VEHICLE:
            lifetime_range = range(1, self.max_lifetime + 1)
        else:
            lifetime_range = range(1, self.max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, self.num_priorities + 1):
                key = (lifetime, priority)
                if key not in queue_dict:
                    queue_dict[key] = QueueSlot(lifetime, priority)
    
    def _handle_queue_overflow(self, task: Task):
        """处理队列溢出"""
        # 尝试通过丢弃低优先级任务来腾出空间
        freed_space = self._drop_low_priority_tasks(task.data_size)
        
        if freed_space >= task.data_size:
            # 成功腾出空间，重新尝试添加
            self.add_task(task)
        else:
            # 无法腾出足够空间，丢弃当前任务
            self.statistics.total_drops += 1
    
    def _drop_low_priority_tasks(self, required_space: float) -> float:
        """丢弃低优先级任务以腾出空间"""
        freed_space = 0.0
        
        # 从最低优先级开始丢弃
        for priority in range(self.num_priorities, 0, -1):
            if freed_space >= required_space:
                break
            
            for lifetime in range(self.max_lifetime, 0, -1):
                queue_key = (lifetime, priority)
                if queue_key in self.queues and not self.queues[queue_key].is_empty():
                    queue = self.queues[queue_key]
                    
                    # 丢弃队列中的任务
                    while not queue.is_empty() and freed_space < required_space:
                        task = queue.task_list.pop()
                        freed_space += task.data_size
                        self.current_usage -= task.data_size
                        task.is_dropped = True
                        self.statistics.total_drops += 1
                        
                        # 更新队列数据量
                        queue.data_volume -= task.data_size
        
        return freed_space
    
    def _update_arrival_statistics(self, priority: int):
        """更新到达率统计"""
        # 记录当前时隙的到达
        current_slot_arrivals = defaultdict(int)
        current_slot_arrivals[priority] += 1
        
        self.recent_arrivals.append(dict(current_slot_arrivals))
        
        # 限制历史长度
        if len(self.recent_arrivals) > self.time_window_size:
            self.recent_arrivals.pop(0)
        
        # 计算到达率
        self._calculate_arrival_rates()
    
    def _update_service_statistics(self):
        """更新服务率统计"""
        # 记录当前时隙的服务
        self.recent_services.append(1)
        
        # 限制历史长度
        if len(self.recent_services) > self.time_window_size:
            self.recent_services.pop(0)
        
        # 计算服务率
        self._calculate_service_rate()
    
    def _calculate_arrival_rates(self):
        """计算各优先级的到达率"""
        if not self.recent_arrivals:
            return
        
        window_duration = len(self.recent_arrivals) * config.network.time_slot_duration
        
        for priority in range(1, self.num_priorities + 1):
            total_arrivals = sum(arrivals.get(priority, 0) for arrivals in self.recent_arrivals)
            self.arrival_rates[priority] = total_arrivals / window_duration
    
    def _calculate_service_rate(self):
        """计算服务率"""
        if not self.recent_services:
            return
        
        window_duration = len(self.recent_services) * config.network.time_slot_duration
        total_services = sum(self.recent_services)
        self.service_rate = total_services / window_duration
        
        # 更新负载因子
        for priority in range(1, self.num_priorities + 1):
            if self.service_rate > 0:
                self.load_factors[priority] = self.arrival_rates[priority] / self.service_rate
            else:
                self.load_factors[priority] = 0.0
    
    def get_queue_state_vector(self) -> np.ndarray:
        """获取队列状态向量"""
        state_features = []
        
        # 基本队列信息
        state_features.extend([
            self.current_usage / self.max_capacity,  # 容量利用率
            len([q for q in self.queues.values() if not q.is_empty()]) / len(self.queues),  # 活跃队列比例
            sum(self.load_factors.values()),  # 总负载因子
            self.service_rate / 100.0,  # 归一化服务率
        ])
        
        # 各优先级队列状态
        for priority in range(1, self.num_priorities + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            priority_data = sum(queue.data_volume 
                              for (l, p), queue in self.queues.items() 
                              if p == priority)
            
            state_features.extend([
                priority_tasks / 50.0,  # 归一化任务数
                priority_data / self.max_capacity,  # 归一化数据量
                self.arrival_rates.get(priority, 0.0) / 10.0,  # 归一化到达率
                self.load_factors.get(priority, 0.0),  # 负载因子
            ])
        
        return np.array(state_features, dtype=np.float32)
    
    def get_queue_statistics(self) -> Dict:
        """获取队列统计信息"""
        total_requests = self.statistics.total_arrivals + self.statistics.total_drops
        
        return {
            'total_arrivals': self.statistics.total_arrivals,
            'total_departures': self.statistics.total_departures,
            'total_drops': self.statistics.total_drops,
            'drop_rate': self.statistics.total_drops / max(1, total_requests),
            'capacity_utilization': self.current_usage / self.max_capacity,
            'total_load_factor': sum(self.load_factors.values()),
            'service_rate': self.service_rate,
            'arrival_rates_by_priority': dict(self.arrival_rates),
            'load_factors_by_priority': dict(self.load_factors),
            'active_queues': len([q for q in self.queues.values() if not q.is_empty()]),
            'avg_queue_length': np.mean([len(q.task_list) for q in self.queues.values()]),
        }
    
    def is_stable(self) -> bool:
        """检查队列系统是否稳定"""
        total_load = sum(self.load_factors.values())
        return total_load < config.queue.max_load_factor
    
    def get_utilization(self) -> float:
        """获取队列利用率"""
        return self.current_usage / self.max_capacity
    
    def get_priority_distribution(self) -> Dict[int, float]:
        """获取各优先级任务分布"""
        total_tasks = sum(len(queue.task_list) for queue in self.queues.values())
        
        if total_tasks == 0:
            return {p: 0.0 for p in range(1, self.num_priorities + 1)}
        
        distribution = {}
        for priority in range(1, self.num_priorities + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            distribution[priority] = priority_tasks / total_tasks
        
        return distribution
