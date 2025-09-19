"""
车辆节点实现 - 对应论文第2.1节和第5.1节
实现车辆的本地计算、任务生成和移动模型
"""
import numpy as np
import time
import math
from typing import List, Optional, Tuple

from .base_node import BaseNode
from .data_structures import Task, Position, NodeType, TaskType
from config import config
from utils import generate_poisson_arrivals


class VehicleNode(BaseNode):
    """
    车辆节点类 - 对应论文车辆模型 v ∈ V
    
    主要功能:
    1. 任务生成 (按泊松过程)
    2. 本地计算处理
    3. 车辆移动模拟
    4. 能耗模型 (对应论文式5-9)
    """
    
    def __init__(self, vehicle_id: str, initial_position: Position):
        super().__init__(vehicle_id, NodeType.VEHICLE, initial_position)
        
        # 车辆特有属性
        self.velocity = np.array([0.0, 0.0])  # 速度向量 (m/s)
        self.max_speed = 30.0  # 最大速度 (m/s)
        self.trajectory: List[Position] = [initial_position]
        
        # 计算资源配置 - 对应论文第5.1节
        self._setup_compute_resources()
        
        # 任务生成参数
        self.task_generation_rate = config.task.arrival_rate
        self.generated_tasks: List[Task] = []
        
        # 能耗模型参数 - 论文式(5)
        self.kappa1 = config.compute.vehicle_kappa1
        self.kappa2 = config.compute.vehicle_kappa2  
        self.static_power = config.compute.vehicle_static_power
        self.idle_power = config.compute.vehicle_static_power
        
        # 传输功率
        self.state.tx_power = config.communication.vehicle_tx_power
    
    def _setup_compute_resources(self):
        """设置车辆计算资源"""
        # 从配置范围中随机选择CPU频率
        freq_range = config.compute.vehicle_cpu_freq_range
        self.state.cpu_frequency = np.random.uniform(freq_range[0], freq_range[1])
        
        # 设置可用带宽
        self.state.available_bandwidth = config.communication.total_bandwidth / config.network.num_vehicles
    
    def get_processing_capacity(self) -> float:
        """
        获取车辆本地处理能力 - 对应论文式(5)
        D^local_n = (f_n * Δt) / c
        """
        delta_t = config.network.time_slot_duration
        compute_density = config.task.task_compute_density
        parallel_efficiency = config.compute.parallel_efficiency
        
        return (self.state.cpu_frequency * delta_t * parallel_efficiency) / compute_density
    
    def calculate_processing_delay(self, task: Task) -> float:
        """
        计算本地处理时延 - 对应论文式(6)
        T_comp,j,n = C_j / (f_n * η_parallel)
        """
        parallel_efficiency = config.compute.parallel_efficiency
        return task.compute_cycles / (self.state.cpu_frequency * parallel_efficiency)
    
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """
        计算处理能耗 - 对应论文式(7)-(9)
        P^comp_n(f_n, U_n) = κ₁f_n³ + κ₂f_n²U_n + P_static
        """
        # 计算当前CPU利用率
        utilization = min(1.0, processing_time / config.network.time_slot_duration)
        
        # 动态功率模型 - 论文式(7)
        dynamic_power = (self.kappa1 * (self.state.cpu_frequency ** 3) +
                        self.kappa2 * (self.state.cpu_frequency ** 2) * utilization +
                        self.static_power)
        
        # 计算活动时间和空闲时间的能耗 - 论文式(8)
        active_time = processing_time
        idle_time = max(0, config.network.time_slot_duration - active_time)
        
        total_energy = dynamic_power * active_time + self.idle_power * idle_time
        
        # 更新功耗状态
        self.state.current_power = dynamic_power
        
        return total_energy
    
    def generate_tasks(self, current_time_slot: int) -> List[Task]:
        """
        生成新任务 - 按泊松过程到达
        对应论文第2.1节任务模型
        """
        # 按泊松过程生成任务数量
        num_tasks = generate_poisson_arrivals(
            self.task_generation_rate, 
            config.network.time_slot_duration
        )
        
        new_tasks = []
        for _ in range(num_tasks):
            task = self._create_random_task()
            new_tasks.append(task)
            self.generated_tasks.append(task)
        
        return new_tasks
    
    def _create_random_task(self) -> Task:
        """创建随机任务"""
        # 随机生成任务属性
        data_size_range = config.task.task_data_size_range
        data_size = np.random.uniform(data_size_range[0], data_size_range[1])
        
        # 修复：数据大小是字节，需要转换为比特再乘以计算密度
        compute_cycles = data_size * 8 * config.task.task_compute_density  # 字节转比特
        result_size = data_size * config.task.task_output_ratio
        
        # 随机选择最大延迟容忍度 (指数分布，偏向较小值)
        max_delay_slots = max(1, int(np.random.exponential(5.0)))
        
        # 根据延迟确定任务类型和优先级
        task_type_value = config.get_task_type(max_delay_slots)
        task_type = TaskType(task_type_value)
        
        # 优先级与任务类型相关 (类型值小的优先级高)
        priority = task_type_value + np.random.randint(0, 2)  # 增加随机性
        priority = min(priority, config.task.num_priority_levels)
        
        task = Task(
            data_size=data_size,
            compute_cycles=compute_cycles,
            result_size=result_size,
            max_delay_slots=max_delay_slots,
            task_type=task_type,
            priority=priority,
            source_vehicle_id=self.node_id
        )
        
        return task
    
    def update_position(self, time_step: float):
        """
        更新车辆位置
        简单的移动模型：匀速直线运动，到边界时转向
        """
        # 更新位置
        new_x = self.state.position.x + self.velocity[0] * time_step
        new_y = self.state.position.y + self.velocity[1] * time_step
        
        # 边界检查和转向
        area_width = config.network.area_width
        area_height = config.network.area_height
        
        if new_x <= 0 or new_x >= area_width:
            self.velocity[0] = -self.velocity[0]
            new_x = max(0, min(area_width, new_x))
        
        if new_y <= 0 or new_y >= area_height:
            self.velocity[1] = -self.velocity[1]
            new_y = max(0, min(area_height, new_y))
        
        # 更新位置
        self.state.position.x = new_x
        self.state.position.y = new_y
        
        # 记录轨迹
        self.trajectory.append(Position(new_x, new_y, 0))
        
        # 限制轨迹历史长度
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)
    
    def set_random_velocity(self):
        """设置随机速度"""
        # 随机方向
        angle = np.random.uniform(0, 2 * math.pi)
        # 随机速度大小
        speed = np.random.uniform(5.0, self.max_speed)
        
        self.velocity[0] = speed * math.cos(angle)
        self.velocity[1] = speed * math.sin(angle)
    
    def step(self, time_step: float) -> Tuple[List[Task], List[Task]]:
        """
        车辆节点单步更新
        
        Returns:
            (新生成的任务列表, 本地处理完成的任务列表)
        """
        # 1. 更新位置
        self.update_position(time_step)
        
        # 2. 更新队列生命周期
        self.update_queue_lifetimes()
        
        # 3. 生成新任务
        current_slot = int(time.time() / config.network.time_slot_duration)
        new_tasks = self.generate_tasks(current_slot)
        
        # 4. 处理本地队列中的任务
        processed_tasks = []
        
        # 获取本时隙可处理的数据量
        processing_capacity = self.get_processing_capacity()
        remaining_capacity = processing_capacity
        
        # 注意：新生成的任务不在这里直接处理
        # 它们将被送到卸载决策器进行决策
        
        # 处理队列中的任务
        while remaining_capacity > 0:
            next_task = self.get_next_task_to_process()
            if next_task is None:
                break
            
            if next_task.data_size <= remaining_capacity:
                if self.process_task(next_task):
                    processed_tasks.append(next_task)
                    remaining_capacity -= next_task.data_size
                else:
                    break
            else:
                # 任务太大，无法在本时隙完成
                break
        
        # 5. 更新统计信息
        self._update_statistics()
        
        return new_tasks, processed_tasks
    
    def can_process_immediately(self, task: Task) -> bool:
        """
        检查是否能立即本地处理任务
        
        Args:
            task: 要检查的任务
            
        Returns:
            bool: 是否能立即处理
        """
        # 检查CPU资源是否足够
        processing_capacity = self.get_processing_capacity()
        
        # 检查当前负载
        current_load = self.state.load_factor
        
        # 如果当前负载过高，无法立即处理
        if current_load > 0.9:
            return False
        
        # 检查任务大小是否在处理能力范围内
        if task.data_size > processing_capacity:
            return False
        
        return True
    
    def process_task_immediately(self, task: Task) -> Tuple[bool, float]:
        """
        立即处理任务
        
        Args:
            task: 要处理的任务
            
        Returns:
            Tuple[bool, float]: (是否成功, 处理延迟)
        """
        if not self.can_process_immediately(task):
            return False, 0.0
        
        # 计算处理延迟
        processing_delay = self.calculate_processing_delay(task)
        
        # 检查是否满足延迟要求
        max_allowed_delay = task.max_delay_slots * config.network.time_slot_duration
        if processing_delay > max_allowed_delay:
            return False, 0.0
        
        # 更新负载状态
        time_slot_duration = config.network.time_slot_duration
        load_increase = processing_delay / time_slot_duration
        self.state.load_factor = min(1.0, self.state.load_factor + load_increase)
        
        # 更新统计信息 - 使用基类的统计方法
        if hasattr(self, 'statistics'):
            self.statistics['processed_tasks'] = self.statistics.get('processed_tasks', 0) + 1
        
        return True, processing_delay