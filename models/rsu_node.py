"""
RSU节点实现 - 对应论文第2.2节和第5.3节
实现RSU的边缘计算、缓存管理和任务迁移功能
"""
import numpy as np
import time
import math
import random
from typing import Dict, List, Optional, Tuple, Set

from .base_node import BaseNode
from .data_structures import Task, Position, NodeType, CommunicationLink
from config import config
from utils import sigmoid


class RSUNode(BaseNode):
    """
    RSU节点类 - 对应论文路边基础设施单元 r ∈ R
    
    主要功能:
    1. 边缘计算处理
    2. 智能缓存管理 (对应论文第7节)
    3. 任务迁移支持
    4. 缓存命中预测 (论文式1)
    """
    
    def __init__(self, rsu_id: str, position: Position):
        super().__init__(rsu_id, NodeType.RSU, position)
        
        # RSU计算资源配置 - 对应论文第5.3节
        self.state.cpu_frequency = config.compute.rsu_cpu_freq
        self.state.tx_power = config.communication.rsu_tx_power
        self.state.available_bandwidth = config.communication.total_bandwidth / config.network.num_rsus
        
        # 缓存系统 - 对应论文第2.2节
        self.cache_capacity = config.cache.rsu_cache_capacity  # S_cache,r
        self.cached_results: Dict[str, Task] = {}  # 缓存的任务结果
        self.cache_decisions: Dict[str, bool] = {}  # z_j,r 缓存决策变量
        
        # 缓存统计与预测
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.request_history: Dict[str, List[float]] = {}  # 请求历史
        self.cache_heat_map: Dict[str, float] = {}  # 内容热度
        
        # 迁移相关
        self.migration_bandwidth = config.migration.migration_bandwidth
        self.is_migration_target = False
        self.migration_cooldown = 0
        
        # 能耗模型参数 - 论文式(20)-(21)
        self.kappa2 = config.compute.rsu_kappa2
        
        # 通信覆盖范围
        self.coverage_radius = 500.0  # 覆盖半径 (米)
        
        # 邻居RSU列表 (用于协作缓存)
        self.neighbor_rsus: Set[str] = set()
    
    def get_processing_capacity(self) -> float:
        """
        获取RSU处理能力 - 对应论文式(20)
        D^proc_k = (f_k * Δt) / c
        """
        delta_t = config.network.time_slot_duration
        compute_density = config.task.task_compute_density
        
        return (self.state.cpu_frequency * delta_t) / compute_density
    
    def calculate_processing_delay(self, task: Task) -> float:
        """
        计算RSU处理时延 - 对应论文式(21)
        T_comp,j,k = C_j / f_k
        """
        return task.compute_cycles / self.state.cpu_frequency
    
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """
        计算RSU处理能耗 - 对应论文式(22)
        P^comp_k = κ₂(f_k)³
        """
        # RSU处理功率
        processing_power = self.kappa2 * (self.state.cpu_frequency ** 3)
        
        # 计算实际活动时间内的能耗
        active_time = min(processing_time, config.network.time_slot_duration)
        energy_consumption = processing_power * active_time
        
        # 更新功耗状态
        self.state.current_power = processing_power
        
        return energy_consumption
    
    def check_cache_hit(self, task: Task) -> bool:
        """
        检查缓存命中 - 对应论文式(1)缓存决策z_j,r
        """
        task_type_key = f"{task.task_type.value}_{task.data_size}_{task.compute_cycles}"
        
        # 检查是否已缓存该类型任务的结果
        if task_type_key in self.cached_results:
            self.cache_hit_count += 1
            return True
        else:
            self.cache_miss_count += 1
            return False
    
    def predict_cache_request_probability(self, task: Task) -> float:
        """
        预测缓存请求概率 - 对应论文式(1)
        P_req,j,r(t) = σ(α₀ + α₁H_j + α₂λ_vj,req + α₃F_t + α₄R_area)
        """
        # 获取逻辑回归参数
        alpha0 = config.cache.logistic_alpha0
        alpha1 = config.cache.logistic_alpha1
        alpha2 = config.cache.logistic_alpha2
        alpha3 = config.cache.logistic_alpha3
        alpha4 = config.cache.logistic_alpha4
        
        # 计算特征值
        task_key = f"{task.task_type.value}_{task.data_size}"
        
        # H_j: 历史请求频率
        historical_frequency = self.request_history.get(task_key, [])
        H_j = len(historical_frequency) / 100.0  # 归一化
        
        # λ_vj,req: 源车辆的平均请求率 (简化)
        lambda_req = 0.1  # 默认值，实际中需要统计
        
        # F_t: 时间因素 (简化为当前小时)
        current_hour = time.localtime().tm_hour
        F_t = math.sin(2 * math.pi * current_hour / 24)  # 周期性时间特征
        
        # R_area: 区域特征 (简化为距离最近车辆的距离倒数)
        R_area = 0.5  # 默认值
        
        # 计算逻辑回归值
        linear_combination = (alpha0 + alpha1 * H_j + alpha2 * lambda_req + 
                            alpha3 * F_t + alpha4 * R_area)
        
        probability = sigmoid(linear_combination)
        
        return probability
    
    def calculate_cache_hit_rate(self) -> float:
        """
        计算缓存命中率 - 对应论文第2.2节
        """
        total_requests = self.cache_hit_count + self.cache_miss_count
        if total_requests == 0:
            return 0.0
        
        return self.cache_hit_count / total_requests
    
    def make_cache_decision(self, task: Task) -> bool:
        """
        制定缓存决策 - 基于预测概率和容量约束
        """
        if not config.cache.cache_hit_prediction_enabled:
            return False
        
        # 检查缓存容量
        current_cache_usage = sum(cached_task.result_size 
                                for cached_task in self.cached_results.values())
        
        if current_cache_usage + task.result_size > self.cache_capacity:
            # 缓存已满，需要替换策略
            if not self._cache_replacement(task.result_size):
                return False
        
        # 预测请求概率
        request_probability = self.predict_cache_request_probability(task)
        
        # 基于概率的缓存决策
        cache_threshold = 0.5  # 可配置阈值
        should_cache = request_probability > cache_threshold
        
        if should_cache:
            task_key = f"{task.task_type.value}_{task.data_size}_{task.compute_cycles}"
            self.cached_results[task_key] = task
            self.cache_decisions[task_key] = True
        
        return should_cache
    
    def _cache_replacement(self, required_size: float) -> bool:
        """
        缓存替换策略 - LRU (Least Recently Used)
        """
        if config.cache.cache_replacement_policy == "LRU":
            return self._lru_replacement(required_size)
        elif config.cache.cache_replacement_policy == "LFU":
            return self._lfu_replacement(required_size)
        else:
            return self._random_replacement(required_size)
    
    def _lru_replacement(self, required_size: float) -> bool:
        """LRU替换策略"""
        # 按最后访问时间排序
        sorted_items = sorted(self.cached_results.items(), 
                            key=lambda x: getattr(x[1], 'last_access_time', 0))
        
        freed_size = 0.0
        removed_keys = []
        
        for key, cached_task in sorted_items:
            if freed_size >= required_size:
                break
            
            freed_size += cached_task.result_size
            removed_keys.append(key)
        
        # 移除选中的缓存项
        for key in removed_keys:
            del self.cached_results[key]
            if key in self.cache_decisions:
                del self.cache_decisions[key]
        
        return freed_size >= required_size
    
    def _lfu_replacement(self, required_size: float) -> bool:
        """LFU替换策略"""
        # 基于访问频率的替换 (简化实现)
        sorted_items = sorted(self.cached_results.items(),
                            key=lambda x: self.request_history.get(x[0], []), 
                            reverse=False)
        
        freed_size = 0.0
        removed_keys = []
        
        for key, cached_task in sorted_items:
            if freed_size >= required_size:
                break
            
            freed_size += cached_task.result_size
            removed_keys.append(key)
        
        for key in removed_keys:
            del self.cached_results[key]
            if key in self.cache_decisions:
                del self.cache_decisions[key]
        
        return freed_size >= required_size
    
    def _random_replacement(self, required_size: float) -> bool:
        """随机替换策略"""
        cache_items = list(self.cached_results.items())
        random.shuffle(cache_items)
        
        freed_size = 0.0
        removed_keys = []
        
        for key, cached_task in cache_items:
            if freed_size >= required_size:
                break
            
            freed_size += cached_task.result_size
            removed_keys.append(key)
        
        for key in removed_keys:
            del self.cached_results[key]
            if key in self.cache_decisions:
                del self.cache_decisions[key]
        
        return freed_size >= required_size
    
    def process_offloaded_task(self, task: Task) -> Tuple[bool, float]:
        """
        处理卸载任务
        
        Returns:
            (是否成功, 总处理时延)
        """
        # 1. 检查缓存命中
        if self.check_cache_hit(task):
            # 缓存命中，直接返回结果
            cache_response_delay = 0.001  # 极短的响应时延
            return True, cache_response_delay
        
        # 2. 缓存未命中，需要计算处理
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
            
            # 6. 制定缓存决策
            self.make_cache_decision(task)
            
            # 更新请求历史
            task_key = f"{task.task_type.value}_{task.data_size}"
            if task_key not in self.request_history:
                self.request_history[task_key] = []
            self.request_history[task_key].append(time.time())
            
            # 限制历史记录长度
            if len(self.request_history[task_key]) > 100:
                self.request_history[task_key].pop(0)
        
        return success, total_delay
    
    def calculate_migration_cost(self, task: Task, target_rsu_id: str) -> float:
        """
        计算任务迁移成本 - 对应论文第6节
        """
        # 迁移传输时延
        migration_delay = task.data_size / self.migration_bandwidth
        
        # 迁移计算成本 (基于复杂度)
        computation_cost = task.compute_cycles / 1e9  # 归一化
        
        # 距离相关成本 (需要获取目标RSU位置)
        distance_cost = 1.0  # 简化实现
        
        # 总迁移成本 - 对应论文式(31)
        alpha_comp = config.migration.migration_alpha_comp
        alpha_tx = config.migration.migration_alpha_tx
        alpha_lat = config.migration.migration_alpha_lat
        
        total_cost = (alpha_comp * computation_cost + 
                     alpha_tx * migration_delay + 
                     alpha_lat * distance_cost)
        
        return total_cost
    
    def is_overloaded(self) -> bool:
        """检查RSU是否过载"""
        overload_threshold = config.migration.rsu_overload_threshold
        return self.state.load_factor > overload_threshold
    
    def is_underloaded(self) -> bool:
        """检查RSU是否低负载"""
        underload_threshold = config.migration.rsu_underload_threshold
        return self.state.load_factor < underload_threshold
    
    def get_coverage_vehicles(self, vehicle_positions: Dict[str, Position]) -> List[str]:
        """获取覆盖范围内的车辆列表"""
        covered_vehicles = []
        
        for vehicle_id, vehicle_pos in vehicle_positions.items():
            distance = self.state.position.distance_to(vehicle_pos)
            if distance <= self.coverage_radius:
                covered_vehicles.append(vehicle_id)
        
        return covered_vehicles
    
    def step(self, time_step: float) -> List[Task]:
        """
        RSU节点单步更新
        
        Returns:
            已处理完成的任务列表
        """
        # 1. 更新队列生命周期
        self.update_queue_lifetimes()
        
        # 2. 更新迁移冷却
        if self.migration_cooldown > 0:
            self.migration_cooldown -= 1
        
        # 3. 处理队列中的任务
        processed_tasks = []
        
        # 获取本时隙可处理的数据量
        processing_capacity = self.get_processing_capacity()
        remaining_capacity = processing_capacity
        
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
                break
        
        # 4. 更新统计信息
        self._update_statistics()
        
        return processed_tasks
    
    def get_state_vector(self) -> np.ndarray:
        """获取RSU状态向量，用于强化学习"""
        base_state = super().get_state_vector()
        
        # 添加RSU特有状态
        rsu_specific_state = [
            self.calculate_cache_hit_rate(),
            len(self.cached_results) / 100.0,  # 归一化缓存项数量
            (self.cache_capacity - sum(task.result_size for task in self.cached_results.values())) / self.cache_capacity,  # 剩余缓存容量比例
            float(self.is_overloaded()),
            float(self.is_underloaded()),
            self.migration_cooldown / 10.0,  # 归一化冷却时间
        ]
        
        return np.concatenate([base_state, rsu_specific_state])