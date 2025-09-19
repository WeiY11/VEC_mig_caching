"""
UAV节点实现 - 对应论文第2.2节和第5.5节
实现UAV的空中边缘计算、能耗管理和移动性支持
"""
import numpy as np
import time
import math
from typing import List, Optional, Tuple, Dict

from .base_node import BaseNode
from .data_structures import Task, Position, NodeType
from config import config


class UAVNode(BaseNode):
    """
    UAV节点类 - 对应论文无人机 u ∈ U
    
    主要功能:
    1. 空中边缘计算处理
    2. 悬停能耗管理 (对应论文式5.5节)
    3. 电池电量管理
    4. 3D位置管理
    """
    
    def __init__(self, uav_id: str, position: Position):
        super().__init__(uav_id, NodeType.UAV, position)
        
        # UAV计算资源配置 - 对应论文第5.5节
        self.state.cpu_frequency = config.compute.uav_default_freq
        self.state.tx_power = config.communication.uav_tx_power
        self.state.available_bandwidth = config.communication.total_bandwidth / config.network.num_uavs
        
        # 能耗模型参数 - 论文式(25)-(28)
        self.kappa3 = config.compute.uav_kappa
        self.state.hover_power = config.compute.uav_hover_power
        
        # 电池管理
        self.state.battery_level = 1.0  # 初始电池满电
        self.battery_capacity = 50000.0  # 电池容量 (Wh)
        self.min_battery_threshold = config.migration.uav_min_battery
        
        # 飞行参数 (简化为固定悬停)
        self.is_hovering = True
        self.hover_efficiency = 0.8  # 悬停效率
        
        # 通信覆盖参数
        self.coverage_radius = 800.0  # UAV覆盖半径更大
        self.altitude = position.z if position.z > 0 else config.network.uav_height
        
        # 服务质量参数
        self.max_concurrent_tasks = 10  # 最大并发任务数
        self.service_area_center = Position(position.x, position.y, self.altitude)
        
        # 迁移相关
        self.migration_cooldown = 0
        self.is_migration_source = False
        
    def get_processing_capacity(self) -> float:
        """
        获取UAV处理能力 - 对应论文式(26)
        D^proc_u = (f_u * Δt) / c
        """
        delta_t = config.network.time_slot_duration
        compute_density = config.task.task_compute_density
        
        # 考虑电池电量对性能的影响
        battery_factor = max(0.5, self.state.battery_level)  # 低电量时性能下降
        
        return (self.state.cpu_frequency * delta_t * battery_factor) / compute_density
    
    def calculate_processing_delay(self, task: Task) -> float:
        """
        计算UAV处理时延 - 对应论文式(27)
        T_exec,j,u = C_j / f_u
        """
        # 考虑电池电量影响
        battery_factor = max(0.5, self.state.battery_level)
        effective_frequency = self.state.cpu_frequency * battery_factor
        
        return task.compute_cycles / effective_frequency
    
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """
        计算UAV总能耗 - 对应论文式(28)-(30)
        包括计算能耗、通信能耗和悬停能耗
        """
        # 1. 计算能耗 - 论文式(28)
        compute_energy = self.kappa3 * (self.state.cpu_frequency ** 2) * processing_time
        
        # 2. 悬停能耗 - 论文式(29)-(30)
        # 简化的悬停功率模型 (论文中UAV位置固定)
        hover_energy = self.state.hover_power * config.network.time_slot_duration
        
        # 3. 总能耗
        total_energy = compute_energy + hover_energy
        
        # 更新电池电量
        self._update_battery_level(total_energy)
        
        # 更新功耗状态
        self.state.current_power = total_energy / config.network.time_slot_duration
        
        return total_energy
    
    def calculate_communication_energy(self, data_size: float, communication_time: float) -> float:
        """
        计算通信能耗 - 对应论文第5.5.1节
        包括接收和发送能耗
        """
        # 接收功率 (通常比发射功率小)
        rx_power = 0.1  # W
        
        # 通信能耗 = 发射能耗 + 接收能耗
        tx_energy = self.state.tx_power * communication_time
        rx_energy = rx_power * communication_time
        
        total_comm_energy = tx_energy + rx_energy
        
        # 更新电池电量
        self._update_battery_level(total_comm_energy)
        
        return total_comm_energy
    
    def _update_battery_level(self, energy_consumed: float):
        """更新电池电量"""
        # 将焦耳转换为瓦时
        energy_wh = energy_consumed / 3600.0
        
        # 更新电池电量
        battery_depletion = energy_wh / self.battery_capacity
        self.state.battery_level = max(0.0, self.state.battery_level - battery_depletion)
    
    def is_battery_low(self) -> bool:
        """检查电池是否低电量"""
        return self.state.battery_level <= self.min_battery_threshold
    
    def can_accept_task(self, task: Task) -> bool:
        """
        检查UAV是否能接受新任务
        考虑电池电量、当前负载等因素
        """
        # 检查电池电量
        if self.is_battery_low():
            return False
        
        # 检查当前负载
        if self.is_overloaded():
            return False
        
        # 检查并发任务数
        current_tasks = sum(len(queue.task_list) for queue in self.queues.values())
        if current_tasks >= self.max_concurrent_tasks:
            return False
        
        # 估算任务能耗是否可承受
        estimated_energy = self._estimate_task_energy(task)
        energy_wh = estimated_energy / 3600.0
        required_battery = energy_wh / self.battery_capacity
        
        return (self.state.battery_level - required_battery) > self.min_battery_threshold
    
    def _estimate_task_energy(self, task: Task) -> float:
        """估算任务总能耗"""
        # 估算处理时间
        processing_time = self.calculate_processing_delay(task)
        
        # 估算通信时间 (简化)
        communication_time = (task.data_size + task.result_size) / 10e6  # 假设10Mbps
        
        # 计算总能耗
        compute_energy = self.kappa3 * (self.state.cpu_frequency ** 2) * processing_time
        comm_energy = self.state.tx_power * communication_time
        hover_energy = self.state.hover_power * (processing_time + communication_time)
        
        return compute_energy + comm_energy + hover_energy
    
    def process_offloaded_task(self, task: Task) -> Tuple[bool, float]:
        """
        处理卸载任务
        
        Returns:
            (是否成功, 总处理时延)
        """
        # 1. 检查是否能接受任务
        if not self.can_accept_task(task):
            return False, 0.0
        
        # 2. 计算处理时延
        processing_delay = self.calculate_processing_delay(task)
        waiting_delay = self.predict_waiting_time(task)
        
        total_delay = waiting_delay + processing_delay
        
        # 3. 检查截止时间
        if task.is_deadline_violated():
            return False, total_delay
        
        # 4. 添加到处理队列
        if not self.add_task_to_queue(task):
            return False, total_delay
        
        # 5. 执行处理 (在实际调度中处理)
        # 注意：不调用process_task，避免重复设置completion_time
        # 在CompleteSystemSimulator中会正确设置包含传输时延的completion_time
        
        # 检查是否能成功处理
        if task.is_deadline_violated():
            success = False
        else:
            # 模拟处理成功
            success = True
            task.processing_delay = processing_delay
            task.assigned_node_id = self.node_id
            
            # 添加到已处理任务列表
            self.processed_tasks.append(task)
        
        return success, total_delay
    
    def is_overloaded(self) -> bool:
        """检查UAV是否过载"""
        overload_threshold = config.migration.uav_overload_threshold
        return self.state.load_factor > overload_threshold
    
    def calculate_migration_urgency(self) -> float:
        """
        计算迁移紧急度
        考虑电池电量、负载状况等因素
        """
        urgency = 0.0
        
        # 电池电量因素
        if self.state.battery_level < self.min_battery_threshold * 2:
            battery_urgency = (self.min_battery_threshold * 2 - self.state.battery_level) / self.min_battery_threshold
            urgency += battery_urgency * 0.5
        
        # 负载因素
        if self.is_overloaded():
            load_urgency = (self.state.load_factor - config.migration.uav_overload_threshold) / (1.0 - config.migration.uav_overload_threshold)
            urgency += load_urgency * 0.3
        
        # 队列长度因素
        queue_length = sum(len(queue.task_list) for queue in self.queues.values())
        if queue_length > self.max_concurrent_tasks * 0.8:
            queue_urgency = (queue_length - self.max_concurrent_tasks * 0.8) / (self.max_concurrent_tasks * 0.2)
            urgency += queue_urgency * 0.2
        
        return min(1.0, urgency)
    
    def get_coverage_vehicles(self, vehicle_positions: Dict) -> List[str]:
        """获取覆盖范围内的车辆列表"""
        covered_vehicles = []
        
        for vehicle_id, vehicle_pos in vehicle_positions.items():
            # 计算3D距离 (考虑UAV高度)
            distance = self.state.position.distance_to(vehicle_pos)
            if distance <= self.coverage_radius:
                covered_vehicles.append(vehicle_id)
        
        return covered_vehicles
    
    def optimize_position(self, vehicle_positions: Dict):
        """
        优化UAV位置以最大化覆盖效果
        简化实现：移动到覆盖车辆的重心位置
        """
        if not vehicle_positions:
            return
        
        covered_vehicles = self.get_coverage_vehicles(vehicle_positions)
        
        if covered_vehicles:
            # 计算覆盖车辆的重心
            center_x = float(np.mean([vehicle_positions[vid].x for vid in covered_vehicles]))
            center_y = float(np.mean([vehicle_positions[vid].y for vid in covered_vehicles]))
            
            # 更新位置 (保持高度)
            self.state.position.x = center_x
            self.state.position.y = center_y
            self.service_area_center = Position(center_x, center_y, self.altitude)
    
    def recharge_battery(self, recharge_rate: float = 0.1):
        """
        电池充电 (在基站或特殊充电点)
        """
        self.state.battery_level = min(1.0, self.state.battery_level + recharge_rate)
    
    def step(self, time_step: float, vehicle_positions: Optional[Dict] = None) -> List[Task]:
        """
        UAV节点单步更新
        
        Args:
            time_step: 时间步长
            vehicle_positions: 车辆位置字典
            
        Returns:
            已处理完成的任务列表
        """
        # 1. 更新队列生命周期
        self.update_queue_lifetimes()
        
        # 2. 更新迁移冷却
        if self.migration_cooldown > 0:
            self.migration_cooldown -= 1
        
        # 3. 优化位置 (如果需要)
        if vehicle_positions:
            self.optimize_position(vehicle_positions)
        
        # 4. 处理队列中的任务
        processed_tasks = []
        
        # 获取本时隙可处理的数据量
        processing_capacity = self.get_processing_capacity()
        remaining_capacity = processing_capacity
        
        while remaining_capacity > 0 and not self.is_battery_low():
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
                break
        
        # 5. 悬停能耗 (即使不处理任务也需要悬停)
        hover_energy = self.state.hover_power * time_step
        self._update_battery_level(hover_energy)
        
        # 6. 更新统计信息
        self._update_statistics()
        
        # 7. 检查是否需要紧急迁移
        if self.is_battery_low() and not processed_tasks:
            # 如果电池低且没有处理任务，可能需要迁移
            self.is_migration_source = True
        
        return processed_tasks
    
    def get_state_vector(self) -> np.ndarray:
        """获取UAV状态向量，用于强化学习"""
        base_state = super().get_state_vector()
        
        # 添加UAV特有状态
        uav_specific_state = [
            self.state.battery_level,  # 电池电量
            float(self.is_battery_low()),  # 是否低电量
            float(self.is_overloaded()),  # 是否过载
            self.calculate_migration_urgency(),  # 迁移紧急度
            self.migration_cooldown / 10.0,  # 归一化冷却时间
            len([queue for queue in self.queues.values() if not queue.is_empty()]) / 10.0,  # 活跃队列比例
            self.state.position.z / 1000.0,  # 归一化高度
        ]
        
        return np.concatenate([base_state, uav_specific_state])