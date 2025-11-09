#!/usr/bin/env python3
"""
完整系统仿真器

用于测试完整的车联网边缘缓存系统，提供高保真的车辆、RSU、UAV交互仿真。
支持任务生成、卸载决策、缓存管理、迁移策略等功能。

Complete system simulator for testing the full vehicular edge caching system.
Provides high-fidelity simulation of vehicle, RSU, and UAV interactions.
"""

import numpy as np
import torch
import random
import os
import logging
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
from collections import deque, defaultdict

# 🔑 修复：导入统一时间管理器
# Unified time manager for consistent simulation timing
from utils.unified_time_manager import get_simulation_time, advance_simulation_time, reset_simulation_time

# 🔑 修复：导入realistic内容生成器
# Realistic content generator for simulating various content types
from utils.realistic_content_generator import generate_realistic_content, get_realistic_content_size
from utils.spatial_index import SpatialIndex
from decision.two_stage_planner import TwoStagePlanner, PlanEntry


class CentralResourcePool:
    """
    中央资源池管理器
    
    【功能】
    Phase 1的核心组件：集中管理所有可分配资源（带宽、计算资源）
    供中央智能体决策使用，实现全局资源优化
    
    【管理的资源】
    1. 总带宽：50 MHz（上行+下行）
    2. 总RSU计算：60 GHz（4个RSU共享）
    3. 总UAV计算：8 GHz（2个UAV共享）
    4. 总本地计算：2 GHz（12车辆共享）
    
    【Phase 1决策】
    中央智能体生成资源分配向量：
    - bandwidth_allocation[12]: 每个车辆的带宽分配比例
    - rsu_compute_allocation[4]: 每个RSU的计算资源分配比例
    - uav_compute_allocation[2]: 每个UAV的计算资源分配比例
    - vehicle_compute_allocation[12]: 每个车辆的本地计算分配比例
    
    【Phase 2执行】
    根据分配结果，各节点执行本地调度
    """
    
    def __init__(self, config):
        """
        初始化中央资源池
        
        Args:
            config: 系统配置对象
        """
        # 🎯 总资源池（从config读取）
        self.total_bandwidth = getattr(config.network, 'bandwidth', 50e6)  # 50 MHz
        self.total_vehicle_compute = getattr(config.compute, 'total_vehicle_compute', 2e9)  # 2 GHz
        self.total_rsu_compute = getattr(config.compute, 'total_rsu_compute', 60e9)  # 60 GHz
        self.total_uav_compute = getattr(config.compute, 'total_uav_compute', 8e9)  # 8 GHz
        
        # 节点数量
        self.num_vehicles = getattr(config.network, 'num_vehicles', 12)
        self.num_rsus = getattr(config.network, 'num_rsus', 4)
        self.num_uavs = getattr(config.network, 'num_uavs', 2)
        
        # 🔄 当前分配状态（初始化为均匀分配）
        self.bandwidth_allocation = np.ones(self.num_vehicles) / self.num_vehicles  # 均匀分配
        self.vehicle_compute_allocation = np.ones(self.num_vehicles) / self.num_vehicles
        self.rsu_compute_allocation = np.ones(self.num_rsus) / self.num_rsus
        self.uav_compute_allocation = np.ones(self.num_uavs) / self.num_uavs
        
        # 📊 资源使用统计
        self.bandwidth_usage = 0.0  # 当前带宽使用率
        self.vehicle_compute_usage = np.zeros(self.num_vehicles)
        self.rsu_compute_usage = np.zeros(self.num_rsus)
        self.uav_compute_usage = np.zeros(self.num_uavs)
        
    def update_allocation(self, allocation_dict: Dict[str, np.ndarray]):
        """
        更新资源分配（Phase 1决策）
        
        Args:
            allocation_dict: 包含各资源分配向量的字典
                - 'bandwidth': [num_vehicles]
                - 'vehicle_compute': [num_vehicles]
                - 'rsu_compute': [num_rsus]
                - 'uav_compute': [num_uavs]
        """
        if 'bandwidth' in allocation_dict:
            self.bandwidth_allocation = self._normalize(allocation_dict['bandwidth'])
        if 'vehicle_compute' in allocation_dict:
            self.vehicle_compute_allocation = self._normalize(allocation_dict['vehicle_compute'])
        if 'rsu_compute' in allocation_dict:
            self.rsu_compute_allocation = self._normalize(allocation_dict['rsu_compute'])
        if 'uav_compute' in allocation_dict:
            self.uav_compute_allocation = self._normalize(allocation_dict['uav_compute'])
    
    def get_vehicle_bandwidth(self, vehicle_idx: int) -> float:
        """获取指定车辆的分配带宽（Hz）"""
        return self.bandwidth_allocation[vehicle_idx] * self.total_bandwidth
    
    def get_vehicle_compute(self, vehicle_idx: int) -> float:
        """获取指定车辆的分配计算资源（Hz）"""
        return self.vehicle_compute_allocation[vehicle_idx] * self.total_vehicle_compute
    
    def get_rsu_compute(self, rsu_idx: int) -> float:
        """获取指定RSU的分配计算资源（Hz）"""
        return self.rsu_compute_allocation[rsu_idx] * self.total_rsu_compute
    
    def get_uav_compute(self, uav_idx: int) -> float:
        """获取指定UAV的分配计算资源（Hz）"""
        return self.uav_compute_allocation[uav_idx] * self.total_uav_compute
    
    def update_usage_stats(self, vehicle_usage=None, rsu_usage=None, uav_usage=None):
        """更新资源使用统计"""
        if vehicle_usage is not None:
            self.vehicle_compute_usage = vehicle_usage
        if rsu_usage is not None:
            self.rsu_compute_usage = rsu_usage
        if uav_usage is not None:
            self.uav_compute_usage = uav_usage
    
    def get_resource_state(self) -> Dict[str, Any]:
        """
        获取资源池状态（供智能体观测）
        
        Returns:
            包含资源分配和使用情况的字典
        """
        return {
            'total_bandwidth': self.total_bandwidth,
            'total_vehicle_compute': self.total_vehicle_compute,
            'total_rsu_compute': self.total_rsu_compute,
            'total_uav_compute': self.total_uav_compute,
            'bandwidth_allocation': self.bandwidth_allocation.copy(),
            'vehicle_compute_allocation': self.vehicle_compute_allocation.copy(),
            'rsu_compute_allocation': self.rsu_compute_allocation.copy(),
            'uav_compute_allocation': self.uav_compute_allocation.copy(),
            'vehicle_compute_usage': self.vehicle_compute_usage.copy(),
            'rsu_compute_usage': self.rsu_compute_usage.copy(),
            'uav_compute_usage': self.uav_compute_usage.copy(),
            # 📊 资源利用率
            'vehicle_utilization': np.mean(self.vehicle_compute_usage),
            'rsu_utilization': np.mean(self.rsu_compute_usage),
            'uav_utilization': np.mean(self.uav_compute_usage),
        }
    
    @staticmethod
    def _normalize(arr: np.ndarray) -> np.ndarray:
        """归一化分配向量，确保总和为1"""
        arr = np.clip(arr, 0, 1)  # 确保非负且<=1
        total = np.sum(arr)
        if total > 1e-6:
            return arr / total
        else:
            # 如果全为0，返回均匀分配
            return np.ones_like(arr) / len(arr)


class CompleteSystemSimulator:
    """
    完整系统仿真器
    
    该类实现了车联网边缘计算系统的完整仿真，包括：
    - 车辆移动模型（沿主干道双路口场景）
    - RSU和UAV部署与管理
    - 任务生成与分配
    - 缓存管理与协同
    - 智能迁移策略
    - 性能统计与监控
    
    Complete system simulator for vehicular edge computing.
    """
    
    def __init__(self, config: Dict = None):
        """
        初始化仿真器
        
        Args:
            config: 配置字典，包含网络拓扑、仿真参数等
                   如果为None，则使用默认配置
        """
        self.config = config or self.get_default_config()
        self.allow_local_processing = bool(self.config.get('allow_local_processing', True))
        forced_mode = str(self.config.get('forced_offload_mode', '')).strip().lower()
        self.forced_offload_mode = forced_mode if forced_mode in {'local_only', 'remote_only'} else ''
        self.override_topology = self.config.get('override_topology', False)
        
        # 统一系统配置入口（若可用）
        # Try to load system-wide configuration if available
        try:
            from config import config as sys_config
            self.sys_config = sys_config
        except Exception:
            self.sys_config = None
        
        # 网络拓扑参数：车辆、RSU、UAV数量
        # Network topology parameters: number of vehicles, RSUs, and UAVs
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.num_vehicles = getattr(self.sys_config.network, 'num_vehicles', 12)
            self.num_rsus = getattr(self.sys_config.network, 'num_rsus', 6)
            self.num_uavs = getattr(self.sys_config.network, 'num_uavs', 2)
        else:
            self.num_vehicles = self.config.get('num_vehicles', 12)
            self.num_rsus = self.config.get('num_rsus', 4)  # 🔑 修复：使用正确的默认值
            self.num_uavs = self.config.get('num_uavs', 2)
        if self.sys_config is not None and not self.override_topology:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300)
        else:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300) if self.sys_config is not None else 300
        self.coverage_radius = self.config.get('coverage_radius', default_radius)

        # 仿真参数：时间、时隙、任务到达率
        # Simulation parameters: time, time slot, task arrival rate
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.simulation_time = getattr(self.sys_config, 'simulation_time', 1000)
            self.time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.1)  # 🚀 适应高负载时隙
            self.task_arrival_rate = getattr(self.sys_config.task, 'arrival_rate', 2.5)  # 🚀 高负载到达率
        else:
            self.simulation_time = self.config.get('simulation_time', 1000)
            self.time_slot = self.config.get('time_slot', 0.1)  # 🚀 高负载默认时隙
            self.task_arrival_rate = self.config.get('task_arrival_rate', 2.5)  # 🚀 高负载默认到达率
        
        # 子配置对象引用
        # Sub-configuration object references
        self.task_config = getattr(self.sys_config, 'task', None) if self.sys_config is not None else None
        self.service_config = getattr(self.sys_config, 'service', None) if self.sys_config is not None else None
        self.stats_config = getattr(self.sys_config, 'stats', None) if self.sys_config is not None else None
        
        # 性能统计与运行状态
        # Performance statistics and runtime state
        self.stats = self._fresh_stats_dict()
        self.queue_config = getattr(self.sys_config, 'queue', None)
        queue_cfg = self.queue_config
        self.queue_stability_threshold = float(getattr(queue_cfg, 'global_rho_threshold', 1.0)) if queue_cfg is not None else 1.0
        self.queue_warning_ratio = float(getattr(queue_cfg, 'stability_warning_ratio', 0.9)) if queue_cfg is not None else 0.9
        self.node_max_load_factor = float(getattr(queue_cfg, 'max_load_factor', 1.0)) if queue_cfg is not None else 1.0
        self.rsu_nominal_capacity = float(getattr(queue_cfg, 'rsu_nominal_capacity', 20.0)) if queue_cfg is not None else 20.0
        self.uav_nominal_capacity = float(getattr(queue_cfg, 'uav_nominal_capacity', 10.0)) if queue_cfg is not None else 10.0
        self.queue_overflow_margin = float(getattr(queue_cfg, 'overflow_margin', 1.2)) if queue_cfg is not None else 1.2
        self.cache_config = getattr(self.sys_config, 'cache', None)
        self.cache_pressure_guard = float(getattr(self.cache_config, 'pressure_guard_ratio', 0.05)) if self.cache_config is not None else 0.05
        delay_clip_from_cfg = getattr(self.stats_config, 'delay_clip_upper', None) if self.stats_config is not None else None
        self.delay_clip_upper = float(delay_clip_from_cfg if delay_clip_from_cfg is not None else self.config.get('delay_clip_upper', 0.0) or 0.0)
        self.migration_delay_weight = float(self.config.get('migration_delay_weight', 600.0))
        self.migration_energy_weight = float(self.config.get('migration_energy_weight', 1.0))
        self._queue_overload_warning_active = False
        self._queue_warning_triggered = False
        self.active_tasks: List[Dict] = []  # 每项: {id, vehicle_id, arrival_time, deadline, work_remaining, node_type, node_idx}
        self.task_counter = 0
        self.current_step = 0
        self.current_time = 0.0
        # Two-stage planning toggle (env-controlled)
        self._two_stage_enabled = (os.environ.get('TWO_STAGE_MODE', '').strip() in {'1', 'true', 'True'})
        self._two_stage_planner: TwoStagePlanner | None = None
        self.spatial_index: Optional[SpatialIndex] = SpatialIndex()
        self._central_resource_enabled = os.environ.get('CENTRAL_RESOURCE', '').strip() in {'1', 'true', 'True'}
        
        # 🎯 中央资源池初始化（Phase 1核心组件）
        # Central resource pool initialization (Phase 1 core component)
        if self.sys_config is not None:
            self.resource_pool = CentralResourcePool(self.sys_config)
        else:
            # 如果没有sys_config，使用默认配置创建一个临时config对象
            from types import SimpleNamespace
            temp_config = SimpleNamespace(
                network=SimpleNamespace(bandwidth=50e6, num_vehicles=12, num_rsus=4, num_uavs=2),
                compute=SimpleNamespace(total_vehicle_compute=2e9, total_rsu_compute=60e9, total_uav_compute=8e9)
            )
            self.resource_pool = CentralResourcePool(temp_config)
        
        # 🔧 读取资源配置参数（CPU频率、带宽等）
        # Read resource configuration parameters (CPU frequency, bandwidth, etc.)
        # ⚠️ 注意：资源现在从中央资源池分配，这里保留兼容性
        if self.sys_config is not None and not self.override_topology:
            self.rsu_cpu_freq = getattr(self.sys_config.compute, 'rsu_default_freq', 15e9)
            self.uav_cpu_freq = getattr(self.sys_config.compute, 'uav_default_freq', 4e9)
            self.vehicle_cpu_freq = getattr(self.sys_config.compute, 'vehicle_default_freq', 0.167e9)
            self.bandwidth = getattr(self.sys_config.network, 'bandwidth', 50e6)
        else:
            self.rsu_cpu_freq = self.config.get('rsu_cpu_freq', 15e9)  # Hz
            self.uav_cpu_freq = self.config.get('uav_cpu_freq', 4e9)  # Hz
            self.vehicle_cpu_freq = self.config.get('vehicle_cpu_freq', 0.167e9)  # Hz
            self.bandwidth = self.config.get('bandwidth', 50e6)  # Hz
        
        # 初始化组件（车辆、RSU、UAV等）
        # Initialize components (vehicles, RSUs, UAVs, etc.)
        self.initialize_components()
        self._reset_runtime_states()
    
    def get_default_config(self) -> Dict:
        """
        获取默认配置参数
        
        提供系统仿真的默认配置，包括网络拓扑、计算能力、
        带宽、功率等关键参数。
        
        Returns:
            包含所有默认配置参数的字典
        """
        return {
            'num_vehicles': 12,
            'num_rsus': 6,
            'num_uavs': 2,
            'simulation_time': 1000,
            'time_slot': 0.1,
            'task_arrival_rate': 0.8,
            'cache_capacity': 100,
            'computation_capacity': 1000,  # MIPS
            'bandwidth': 20,  # MHz
            'transmission_power': 0.1,  # W
            'computation_power': 1.0,  # W
            'rsu_base_service': 4,
            'rsu_max_service': 9,
            'rsu_work_capacity': 2.5,
            'uav_base_service': 3,
            'uav_max_service': 6,
            'uav_work_capacity': 1.7,
            'drop_log_interval': 400,
            'task_report_interval': 100,
            'task_compute_density': 400,
        }
    
    def initialize_components(self):
        """
        初始化系统组件
        
        创建并配置系统的所有组件，包括：
        - 车辆初始化（位置、速度、方向等）
        - RSU节点部署（位置、覆盖范围、缓存容量等）
        - UAV节点部署（位置、覆盖范围、计算能力等）
        - 中央RSU调度器初始化
        - 迁移管理器初始化
        
        Initialize system components including vehicles, RSUs, and UAVs.
        """
        # 🛣️ 主干道-双路口初始化
        # Main road with two intersections initialization
        # 坐标系统 0..1000，主干道沿 x 轴中线 y=500，从左向右；两个路口位于 x=300 和 x=700
        self.road_y = 500.0
        self.intersections = {  # 信号灯相位 周期 T，绿灯比例 g
            'L': {'x': 300.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 0.0},
            'R': {'x': 700.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 15.0},
        }

        # 车辆初始化：落在道路上，方向为东(0)或西(pi)，车道内微扰
        # Vehicle initialization: positioned on road, heading east (0) or west (pi), with lane perturbation
        self.vehicles = []
        for i in range(self.num_vehicles):
            go_east = np.random.rand() < 0.6  # 60% 向东行驶
            base_dir = 0.0 if go_east else np.pi
            x0 = np.random.uniform(100.0, 900.0)
            y0 = self.road_y + np.random.uniform(-6.0, 6.0)  # 简单两车道路幅
            v0 = np.random.uniform(12.0, 22.0)  # 初始速度 12-22 m/s
            vehicle = {
                'id': f'V_{i}',
                'position': np.array([x0, y0], dtype=float),
                'velocity': v0,
                'direction': base_dir,
                'lane_bias': y0 - self.road_y,  # 车道偏差
                'tasks': [],
                'energy_consumed': 0.0,
                'device_cache': {},  # 车载缓存
                'device_cache_capacity': 32.0,  # 车载缓存容量(MB)
                # 🎯 Phase 2本地调度参数
                'cpu_freq': self.vehicle_cpu_freq,  # 分配的CPU频率（Hz）
                'allocated_bandwidth': 0.0,  # 分配的带宽（Hz）
                'task_queue_by_priority': {1: [], 2: [], 3: [], 4: []},  # 按优先级分类的任务队列
                'compute_usage': 0.0,  # 当前计算使用率
            }
            self.vehicles.append(vehicle)
        print("车辆初始化完成：主干道双路口场景")
        
        # RSU节点初始化
        # RSU node initialization
        self.rsus = []
        # 🔑 动态RSU部署：根据num_rsus均匀分布在道路上
        if self.num_rsus <= 4:
            # 原始固定4个RSU的部署位置
            rsu_positions = [
                np.array([300.0, 500.0]),
                np.array([500.0, 500.0]),
                np.array([700.0, 500.0]),
                np.array([900.0, 500.0]),
            ]
        else:
            # 动态生成RSU位置，均匀分布在200-900之间
            rsu_positions = []
            spacing = 700.0 / (self.num_rsus - 1)  # 均匀间隔
            for i in range(self.num_rsus):
                x_pos = 200.0 + i * spacing
                rsu_positions.append(np.array([x_pos, 500.0]))
        
        # 创建RSU节点
        # Create RSU nodes with configuration
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': rsu_positions[i],
                'coverage_radius': self.coverage_radius,  # 覆盖半径(m)
                'cache': {},  # 缓存字典
                'cache_capacity': self.config['cache_capacity'],  # 缓存容量(MB)
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'rsu_cache_capacity', 10e9) if self.sys_config is not None else 10e9),
                'cpu_freq': self.rsu_cpu_freq,  # 🆕 CPU频率(Hz)
                'computation_queue': [],  # 计算任务队列
                'energy_consumed': 0.0,  # 累计能耗(J)
                # 🎯 Phase 2资源调度参数
                'allocated_compute': self.rsu_cpu_freq,  # 分配的计算资源（Hz）
                'compute_usage': 0.0,  # 当前计算使用率
                'connected_vehicles': [],  # 接入的车辆列表
            }
            self.rsus.append(rsu)
        
        # UAV节点初始化
        # UAV node initialization
        self.uavs = []
        # 🔑 动态UAV部署：根据num_uavs均匀分布
        if self.num_uavs <= 2:
            # 原始2枚UAV的部署位置
            uav_positions = [
                np.array([300.0, 500.0, 120.0]),  # x, y, z(高度)
                np.array([700.0, 500.0, 120.0]),
            ]
        else:
            # 动态生成UAV位置，均匀分布在道路上方
            uav_positions = []
            spacing = 600.0 / (self.num_uavs - 1)  # 均匀间隔
            for i in range(self.num_uavs):
                x_pos = 200.0 + i * spacing
                uav_positions.append(np.array([x_pos, 500.0, 120.0]))
        
        # 创建UAV节点
        # Create UAV nodes with configuration
        for i in range(self.num_uavs):
            uav = {
                'id': f'UAV_{i}',
                'position': uav_positions[i],  # 固定悬停位置
                'velocity': 0.0,  # 当前速度(m/s)
                'coverage_radius': 350.0,  # 覆盖半径(m)
                'cache': {},  # 缓存字典
                'cache_capacity': self.config['cache_capacity'],  # 缓存容量(MB)
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'uav_cache_capacity', 2e9) if self.sys_config is not None else 2e9),
                'cpu_freq': self.uav_cpu_freq,  # 🆕 CPU频率(Hz)
                'computation_queue': [],  # 计算任务队列
                'energy_consumed': 0.0,  # 累计能耗(J)
                # 🎯 Phase 2资源调度参数
                'allocated_compute': self.uav_cpu_freq,  # 分配的计算资源（Hz）
                'compute_usage': 0.0,  # 当前计算使用率
                'battery_level': 1.0,  # 电量水平
                'connected_vehicles': [],  # 服务的车辆列表
            }
            self.uavs.append(uav)
        
        print(f"创建了 {self.num_vehicles} 车辆, {self.num_rsus} RSU, {self.num_uavs} UAV")
        
        # 🏢 初始化中央RSU调度器(选择RSU_2作为中央调度中心)
        # Initialize central RSU scheduler for coordinated task management
        try:
            from utils.central_rsu_scheduler import create_central_scheduler
            central_rsu_id = f"RSU_{2 if self.num_rsus > 2 else 0}"
            self.central_scheduler = create_central_scheduler(central_rsu_id)
            print(f"中央RSU调度器已启用: {central_rsu_id}")
        except Exception as e:
            print(f"中央调度器加载失败: {e}")
            self.central_scheduler = None
        
        # 懒加载迁移管理器
        # Lazy load migration manager for task migration strategies
        try:
            from migration.migration_manager import TaskMigrationManager
            if not hasattr(self, 'migration_manager') or self.migration_manager is None:
                self.migration_manager = TaskMigrationManager()
        except Exception:
            self.migration_manager = None
        
        # 一致性自检（不强制终止，仅提示）
        # Consistency check for topology configuration
        try:
            expected_rsus, expected_uavs = 4, 2
            if self.num_rsus != expected_rsus or self.num_uavs != expected_uavs:
                print(
                    f"[Topology] num_rsus={self.num_rsus}, num_uavs={self.num_uavs}, "
                    f"recommended {expected_rsus}/{expected_uavs} to match the paper setup."
                )
            print("[Topology] Central RSU configured as RSU_2 for coordination.")
        except Exception:
            pass

        self._init_mm1_predictor()
        self._refresh_spatial_index(update_static=True, update_vehicle=True)
    
    # ========== Phase 2本地调度逻辑 ==========
    
    def apply_resource_allocation(self, allocation_dict: Dict[str, np.ndarray]):
        """
        应用中央智能体的资源分配决策（Phase 1 → Phase 2）
        
        Args:
            allocation_dict: 中央智能体生成的资源分配字典
                - 'bandwidth': [num_vehicles]  带宽分配比例
                - 'vehicle_compute': [num_vehicles]  车辆计算分配比例
                - 'rsu_compute': [num_rsus]  RSU计算分配比例
                - 'uav_compute': [num_uavs]  UAV计算分配比例
        """
        # 更新资源池
        self.resource_pool.update_allocation(allocation_dict)
        
        # 应用到各节点
        for i, vehicle in enumerate(self.vehicles):
            vehicle['allocated_bandwidth'] = self.resource_pool.get_vehicle_bandwidth(i)
            vehicle['cpu_freq'] = self.resource_pool.get_vehicle_compute(i)
        
        for i, rsu in enumerate(self.rsus):
            rsu['allocated_compute'] = self.resource_pool.get_rsu_compute(i)
        
        for i, uav in enumerate(self.uavs):
            uav['allocated_compute'] = self.resource_pool.get_uav_compute(i)
    
    def vehicle_priority_scheduling(self, vehicle: Dict):
        """
        车辆端优先级队列调度（Phase 2执行层）
        
        【策略】
        1. 按任务优先级（类型1>2>3>4）排序
        2. 优先分配计算资源给高优先级任务
        3. 如果本地资源不足，标记为待卸载
        
        Args:
            vehicle: 车辆对象字典
        """
        # 获取分配的计算资源
        allocated_cpu = vehicle['cpu_freq']
        time_slot = self.time_slot
        
        # 合并所有优先级队列到一个列表，按优先级排序
        all_tasks = []
        for priority in [1, 2, 3, 4]:  # 从高到低
            all_tasks.extend(vehicle['task_queue_by_priority'][priority])
        
        if not all_tasks:
            vehicle['compute_usage'] = 0.0
            return
        
        # 计算本时隙可用的总计算周期
        available_cycles = allocated_cpu * time_slot
        used_cycles = 0.0
        
        for task in all_tasks:
            if 'compute_cycles' in task:
                task_cycles = task['compute_cycles']
                if used_cycles + task_cycles <= available_cycles:
                    # 本地可以处理
                    task['processing_node'] = 'local'
                    task['can_process_local'] = True
                    used_cycles += task_cycles
                else:
                    # 本地资源不足，需要卸载
                    task['processing_node'] = 'offload'
                    task['can_process_local'] = False
        
        # 更新计算使用率
        vehicle['compute_usage'] = used_cycles / max(available_cycles, 1e-9)
    
    def rsu_dynamic_resource_allocation(self, rsu: Dict, rsu_idx: int):
        """
        RSU端动态资源分配（Phase 2执行层）
        
        【策略】
        1. 为接入的车辆动态分配带宽
        2. 根据任务优先级分配计算时间片
        3. 优先服务高优先级任务
        
        Args:
            rsu: RSU对象字典
            rsu_idx: RSU索引
        """
        # 获取分配的计算资源
        allocated_compute = rsu['allocated_compute']
        time_slot = self.time_slot
        
        # 计算本时隙可用的总计算周期
        available_cycles = allocated_compute * time_slot
        
        # 获取所有待处理任务（从computation_queue）
        tasks = rsu['computation_queue']
        if not tasks:
            rsu['compute_usage'] = 0.0
            return
        
        # 按优先级排序（假设任务有task_type字段）
        sorted_tasks = sorted(tasks, key=lambda t: t.get('task_type', 4))
        
        # 分配计算资源
        used_cycles = 0.0
        for task in sorted_tasks:
            if 'compute_cycles' in task:
                task_cycles = task['compute_cycles']
                if used_cycles + task_cycles <= available_cycles:
                    task['can_process'] = True
                    used_cycles += task_cycles
                else:
                    task['can_process'] = False  # 资源不足，需等待下一时隙
        
        # 更新计算使用率
        rsu['compute_usage'] = used_cycles / max(available_cycles, 1e-9)
    
    def uav_dynamic_resource_allocation(self, uav: Dict, uav_idx: int):
        """
        UAV端动态资源分配（Phase 2执行层）
        
        【策略】
        1. 考虑电量水平调整服务能力
        2. 优先服务信道质量好的车辆
        3. 低电量时降低服务范围
        
        Args:
            uav: UAV对象字典
            uav_idx: UAV索引
        """
        # 获取分配的计算资源（考虑电量因子）
        allocated_compute = uav['allocated_compute']
        battery_factor = max(0.5, uav['battery_level'])  # 低电量时性能下降
        effective_compute = allocated_compute * battery_factor
        
        time_slot = self.time_slot
        available_cycles = effective_compute * time_slot
        
        # 获取所有待处理任务
        tasks = uav['computation_queue']
        if not tasks:
            uav['compute_usage'] = 0.0
            return
        
        # 按优先级排序
        sorted_tasks = sorted(tasks, key=lambda t: t.get('task_type', 4))
        
        # 分配计算资源
        used_cycles = 0.0
        for task in sorted_tasks:
            if 'compute_cycles' in task:
                task_cycles = task['compute_cycles']
                if used_cycles + task_cycles <= available_cycles:
                    task['can_process'] = True
                    used_cycles += task_cycles
                else:
                    task['can_process'] = False
        
        # 更新计算使用率
        uav['compute_usage'] = used_cycles / max(available_cycles, 1e-9)
    
    def execute_phase2_scheduling(self):
        """
        执行Phase 2的所有本地调度逻辑
        
        【流程】
        1. 车辆端：优先级调度
        2. RSU端：动态资源分配
        3. UAV端：动态资源分配
        4. 更新资源使用统计
        """
        # 车辆端调度
        for vehicle in self.vehicles:
            self.vehicle_priority_scheduling(vehicle)
        
        # RSU端调度
        for i, rsu in enumerate(self.rsus):
            self.rsu_dynamic_resource_allocation(rsu, i)
        
        # UAV端调度
        for i, uav in enumerate(self.uavs):
            self.uav_dynamic_resource_allocation(uav, i)
        
        # 更新资源池统计
        vehicle_usage = np.array([v['compute_usage'] for v in self.vehicles])
        rsu_usage = np.array([r['compute_usage'] for r in self.rsus])
        uav_usage = np.array([u['compute_usage'] for u in self.uavs])
        self.resource_pool.update_usage_stats(vehicle_usage, rsu_usage, uav_usage)
    
    # ========== Phase 2结束 ==========
    
    def _setup_scenario(self):
        """
        设置仿真场景
        
        重新初始化所有组件并重置运行时状态，用于开始新的仿真回合。
        
        Setup simulation scenario for a new episode.
        """
        # 重新初始化组件（如果需要）
        self.initialize_components()
        self._reset_runtime_states()
        print("初始化了 6 个缓存管理器")

    def _fresh_stats_dict(self) -> Dict[str, float]:
        """
        创建新的统计字典，保证关键指标齐全
        
        Returns:
            包含所有性能指标的字典，包括任务统计、延迟、能耗、缓存命中率等
        """
        return {
            'total_tasks': 0,  # 总任务数
            'processed_tasks': 0,  # 已处理任务数
            'completed_tasks': 0,  # 已完成任务数
            'dropped_tasks': 0,  # 丢弃任务数
            'generated_data_bytes': 0.0,  # 生成的数据总量(字节)
            'dropped_data_bytes': 0.0,  # 丢弃的数据总量(字节)
            'total_delay': 0.0,  # 总延迟(秒)
            'total_energy': 0.0,  # 总能耗(焦耳)
            'energy_uplink': 0.0,  # 上行能耗(焦耳)
            'energy_downlink': 0.0,  # 下行能耗(焦耳)
            'energy_transmit_uplink': 0.0,  # 上行传输能耗
            'energy_transmit_downlink': 0.0,  # 下行传输能耗
            'energy_compute': 0.0,  # 计算能耗(焦耳)
            'energy_cache': 0.0,  # 缓存命中能耗
            'delay_processing': 0.0,  # 计算阶段延迟
            'delay_waiting': 0.0,  # 排队等待延迟
            'delay_uplink': 0.0,  # 上传延迟
            'delay_downlink': 0.0,  # 下载延迟
            'delay_cache': 0.0,  # 缓存命中提供的延迟
            'local_cache_hits': 0,  # 本地缓存命中次数
            'cache_hits': 0,  # 缓存命中次数
            'cache_misses': 0,  # 缓存未命中次数
            'cache_requests': 0,  # 缓存请求次数
            'cache_hit_rate': 0.0,  # 缓存命中率
            'migrations_executed': 0,  # 执行的迁移次数
            'migrations_successful': 0,  # 成功的迁移次数
            'rsu_migration_delay': 0.0,  # RSU迁移延迟(秒)
            'rsu_migration_energy': 0.0,  # RSU迁移能耗(焦耳)
            'rsu_migration_data': 0.0,  # RSU迁移数据量(MB)
            'uav_migration_distance': 0.0,  # UAV迁移距离(米)
            'uav_migration_count': 0,  # UAV迁移次数
            'task_generation': {'total': 0, 'by_type': {}, 'by_scenario': {}},  # 任务生成统计
            'drop_stats': {  # 任务丢弃详细统计
                'total': 0,
                'wait_time_sum': 0.0,
                'queue_sum': 0,
                'by_type': {},
                'by_scenario': {},
                'by_reason': {}
            },
            'remote_rejections': {
                'total': 0,
                'by_type': {'RSU': 0, 'UAV': 0},
                'by_reason': {}
            },
            'queue_rho_sum': 0.0,
            'queue_rho_max': 0.0,
            'queue_overload_flag': False,
            'queue_overload_events': 0,
            'queue_rho_by_node': {},
            'queue_overflow_drops': 0,
            'central_scheduler_calls': 0,
            'central_scheduler_last_decisions': 0,
            'central_scheduler_migrations': 0,
        }

    def _update_central_scheduler(self, step_summary: Dict[str, Any]) -> None:
        scheduler = getattr(self, 'central_scheduler', None)
        if scheduler is None:
            return
        try:
            rsu_snapshots: List[Dict[str, Any]] = []
            for idx, rsu in enumerate(self.rsus):
                rsu_snapshots.append({
                    'id': rsu.get('id', f'RSU_{idx}'),
                    'position': np.array(rsu.get('position', [0.0, 0.0])),
                    'computation_queue': rsu.get('computation_queue', []),
                    'cpu_usage': float(rsu.get('compute_usage', 0.0)),
                    'cpu_frequency': float(rsu.get('allocated_compute', rsu.get('cpu_freq', 0.0))),
                    'cache_usage': float(rsu.get('cache_utilization', 0.0)),
                    'cache_hit_rate': float(self.stats.get('cache_hit_rate', 0.0)),
                    'cached_content': rsu.get('cache', {}),
                    'served_vehicles': len(rsu.get('connected_vehicles', [])),
                    'coverage_vehicles': len(rsu.get('coverage_list', [])),
                    'bandwidth_usage': float(step_summary.get('remote_tasks', 0.0)) / max(1, len(self.vehicles)),
                    'avg_response_time': float(self.stats.get('avg_task_delay', 0.0)),
                    'task_completion_rate': float(self.stats.get('task_completion_rate', 0.0)),
                    'energy_consumption': float(rsu.get('energy_consumed', 0.0)),
                })
            scheduler.collect_all_rsu_loads(rsu_snapshots)
            incoming_tasks = max(1, int(step_summary.get('generated_tasks', 0)))
            decisions = scheduler.global_load_balance_scheduling(incoming_task_count=incoming_tasks)
            migrations = scheduler.intelligent_migration_coordination()
            self.stats['central_scheduler_calls'] = self.stats.get('central_scheduler_calls', 0) + 1
            self.stats['central_scheduler_last_decisions'] = len(decisions)
            self.stats['central_scheduler_migrations'] = self.stats.get('central_scheduler_migrations', 0) + len(migrations)
        except Exception as exc:
            logging.debug("Central scheduler update failed: %s", exc)

    def _accumulate_delay(self, bucket: str, value: float) -> None:
        """Ensure分项延迟与总延迟同步。"""
        try:
            amount = max(0.0, float(value))
        except (TypeError, ValueError):
            return
        if amount <= 0.0:
            return
        self.stats[bucket] = self.stats.get(bucket, 0.0) + amount
        self.stats['total_delay'] = self.stats.get('total_delay', 0.0) + amount

    def _accumulate_energy(self, bucket: str, value: float) -> None:
        """Ensure分项能耗与总能耗同步。"""
        try:
            amount = max(0.0, float(value))
        except (TypeError, ValueError):
            return
        if amount <= 0.0:
            return
        self.stats[bucket] = self.stats.get(bucket, 0.0) + amount
        self.stats['total_energy'] = self.stats.get('total_energy', 0.0) + amount

    def _register_cache_request(self, hit: bool) -> None:
        """更新缓存命中统计与命中率。"""
        self.stats['cache_requests'] = self.stats.get('cache_requests', 0) + 1
        if hit:
            self.stats['cache_hits'] = self.stats.get('cache_hits', 0) + 1
        else:
            self.stats['cache_misses'] = self.stats.get('cache_misses', 0) + 1
        total = self.stats['cache_hits'] + self.stats['cache_misses']
        self.stats['cache_hit_rate'] = self.stats['cache_hits'] / max(1, total)

    def _prepare_step_usage_counters(self) -> None:
        """在单步开始前清零本地使用计数。"""
        for vehicle in self.vehicles:
            vehicle['local_cycle_used'] = 0.0
            vehicle['compute_usage'] = 0.0

    def _record_queue_drop(self, task: Dict, node_type: str) -> None:
        """记录因队列溢出导致的任务丢弃。"""
        self.stats['dropped_tasks'] = self.stats.get('dropped_tasks', 0) + 1
        self.stats['queue_overflow_drops'] = self.stats.get('queue_overflow_drops', 0) + 1
        data_bytes = float(task.get('data_size_bytes', task.get('data_size', 0.0) * 1e6))
        self.stats['dropped_data_bytes'] = self.stats.get('dropped_data_bytes', 0.0) + data_bytes
        task['dropped'] = True
        task['drop_reason'] = 'queue_overflow'
        drop_stats = self.stats.setdefault('drop_stats', {
            'total': 0,
            'wait_time_sum': 0.0,
            'queue_sum': 0,
            'by_type': {},
            'by_scenario': {},
            'by_reason': {}
        })
        drop_stats['total'] = drop_stats.get('total', 0) + 1
        task_type = task.get('task_type', 'unknown')
        scenario = task.get('app_scenario', 'unknown')
        reason = 'queue_overflow'
        by_type = drop_stats.setdefault('by_type', {})
        by_scenario = drop_stats.setdefault('by_scenario', {})
        by_reason = drop_stats.setdefault('by_reason', {})
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_scenario[scenario] = by_scenario.get(scenario, 0) + 1
        by_reason[reason] = by_reason.get(reason, 0) + 1

    def _enforce_queue_capacity(self, node: Dict, node_type: str, step_summary: Dict[str, Any]) -> None:
        """在入队后执行，确保队列受控。"""
        queue = node.get('computation_queue', [])
        if not isinstance(queue, list):
            return
        nominal_capacity = self.rsu_nominal_capacity if node_type == 'RSU' else self.uav_nominal_capacity
        max_queue = int(max(1, round(nominal_capacity * self.node_max_load_factor * self.queue_overflow_margin)))
        overflow = len(queue) - max_queue
        if overflow <= 0:
            return
        dropped = 0
        while overflow > 0 and queue:
            dropped_task = queue.pop()  # 丢弃最新的任务，保护早到任务
            self._record_queue_drop(dropped_task, node_type)
            dropped += 1
            overflow -= 1
        if dropped:
            step_summary['dropped_tasks'] = step_summary.get('dropped_tasks', 0) + dropped
            step_summary['queue_overflow_drops'] = step_summary.get('queue_overflow_drops', 0) + dropped

    def _try_serve_from_vehicle_cache(self, vehicle: Dict, task: Dict, step_summary: Dict[str, Any],
                                      cache_controller: Optional[Any]) -> bool:
        """尝试直接使用车载缓存提供内容。"""
        content_id = task.get('content_id')
        if not content_id:
            return False
        cache = vehicle.get('device_cache') or {}
        cached_entry = cache.get(content_id)
        if cached_entry is None:
            return False
        hit_delay = max(0.002, min(0.05, 0.2 * self.time_slot))
        hit_energy = float(self.config.get('local_cache_energy', 0.15))
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + hit_energy
        self.stats['local_cache_hits'] = self.stats.get('local_cache_hits', 0) + 1
        self._register_cache_request(True)
        self._accumulate_delay('delay_cache', hit_delay)
        self._accumulate_energy('energy_cache', hit_energy)
        self.stats['processed_tasks'] = self.stats.get('processed_tasks', 0) + 1
        self.stats['completed_tasks'] = self.stats.get('completed_tasks', 0) + 1
        step_summary['local_cache_hits'] = step_summary.get('local_cache_hits', 0) + 1
        cached_entry['timestamp'] = self.current_time
        if cache_controller is not None:
            try:
                cache_controller.record_cache_result(content_id, was_hit=True)
            except Exception:
                pass
        return True

    def _reset_runtime_states(self):
        """
        重置运行时状态（用于episode重启）
        
        清空所有运行时数据，包括仿真时间、任务计数、统计数据、
        车辆和节点状态等。
        
        Reset runtime states for starting a new episode.
        """
        reset_simulation_time()
        self.current_step = 0
        self.current_time = 0.0
        self._queue_overload_warning_active = False
        self._queue_warning_triggered = False
        self.task_counter = 0
        self.stats = self._fresh_stats_dict()
        self.active_tasks = []
        self._scheduling_params = {
            'priority_bias': 0.5,
            'deadline_bias': 0.5,
            'reorder_window': 3,
        }
        self._last_app_name = 'unknown'

        # 閲嶇疆杞﹁締/鑺傜偣鐘舵€?
        for vehicle in self.vehicles:
            vehicle.setdefault('tasks', [])
            vehicle['tasks'].clear()
            vehicle['energy_consumed'] = 0.0
            vehicle['device_cache'] = {}
            vehicle['device_cache_capacity'] = vehicle.get('device_cache_capacity', 32.0)

        for idx, rsu in enumerate(self.rsus):
            rsu.setdefault('cache', {})
            rsu['computation_queue'] = []
            rsu['energy_consumed'] = 0.0

        for idx, uav in enumerate(self.uavs):
            uav.setdefault('cache', {})
            uav['computation_queue'] = []
            uav['energy_consumed'] = 0.0

        if hasattr(self, 'mm1_prediction_window'):
            self._build_mm1_trackers()
            self._reset_mm1_step_buffers()
            self._mm1_last_prediction_step = -self.mm1_prediction_interval
        self._prepare_step_usage_counters()

    def _update_scheduling_params(self, params: Optional[Dict[str, float]]) -> None:
        """??????????????????????"""
        if not isinstance(params, dict):
            return
        bias = params.get('priority_bias')
        if bias is not None:
            try:
                bias_val = float(bias)
            except (TypeError, ValueError):
                bias_val = None
            else:
                self._scheduling_params['priority_bias'] = float(np.clip(bias_val, 0.0, 1.0))
        deadline_bias = params.get('deadline_bias')
        if deadline_bias is not None:
            try:
                d_val = float(deadline_bias)
            except (TypeError, ValueError):
                d_val = None
            else:
                self._scheduling_params['deadline_bias'] = float(np.clip(d_val, 0.0, 1.0))
        window = params.get('reorder_window')
        if window is not None:
            try:
                window_val = int(round(float(window)))
            except (TypeError, ValueError):
                window_val = None
            else:
                self._scheduling_params['reorder_window'] = max(1, min(32, window_val))

    def _init_mm1_predictor(self):
        """Initialize M/M/1 queue performance predictor settings and buffers."""
        if getattr(self, 'queue_config', None) is not None:
            window_cfg = getattr(self.queue_config, 'prediction_window', None)
            interval_cfg = getattr(self.queue_config, 'prediction_interval', None)
        else:
            window_cfg = None
            interval_cfg = None

        window = self.config.get('mm1_prediction_window', window_cfg if window_cfg is not None else 12)
        interval = self.config.get('mm1_prediction_interval', interval_cfg if interval_cfg is not None else 5)

        try:
            window = int(window)
        except (TypeError, ValueError):
            window = 12
        window = max(3, window)

        try:
            interval = int(interval)
        except (TypeError, ValueError):
            interval = 5
        interval = max(1, interval)

        self.mm1_prediction_window = window
        self.mm1_prediction_interval = interval
        self._mm1_last_prediction_step = -self.mm1_prediction_interval
        self._build_mm1_trackers()
        self._reset_mm1_step_buffers()

    def _mm1_node_key(self, node_type: str, node_idx: int) -> str:
        return f"{node_type}_{int(node_idx)}"

    def _build_mm1_trackers(self):
        """Create rolling buffers for each node participating in remote processing."""
        self._mm1_trackers: Dict[str, Dict[str, deque]] = {}
        node_keys = [self._mm1_node_key('RSU', idx) for idx, _ in enumerate(self.rsus)]
        node_keys.extend(self._mm1_node_key('UAV', idx) for idx, _ in enumerate(self.uavs))

        for key in node_keys:
            self._mm1_trackers[key] = {
                'arrivals': deque(maxlen=self.mm1_prediction_window),
                'services': deque(maxlen=self.mm1_prediction_window),
                'queue_lengths': deque(maxlen=self.mm1_prediction_window),
                'delays': deque(maxlen=self.mm1_prediction_window),
            }

    def _reset_mm1_step_buffers(self):
        """Reset per-step accumulation buffers for MM1 metrics."""
        if not hasattr(self, '_mm1_trackers'):
            return
        self._mm1_step_arrivals: defaultdict[str, int] = defaultdict(int)
        self._mm1_step_services: defaultdict[str, int] = defaultdict(int)
        self._mm1_step_delays: defaultdict[str, List[float]] = defaultdict(list)
        self._mm1_step_queue_lengths: Dict[str, int] = {}

    def _record_mm1_arrival(self, node_type: str, node_idx: int):
        if not hasattr(self, '_mm1_trackers'):
            return
        key = self._mm1_node_key(node_type, node_idx)
        self._mm1_step_arrivals[key] += 1

    def _record_mm1_service(self, node_type: str, node_idx: int, delay: float):
        if not hasattr(self, '_mm1_trackers'):
            return
        key = self._mm1_node_key(node_type, node_idx)
        self._mm1_step_services[key] += 1
        if delay is not None and delay >= 0.0:
            self._mm1_step_delays[key].append(float(delay))

    def _record_mm1_queue_length(self, node_type: str, node_idx: int, queue_len: int):
        if not hasattr(self, '_mm1_trackers'):
            return
        key = self._mm1_node_key(node_type, node_idx)
        self._mm1_step_queue_lengths[key] = int(queue_len)

    def _finalize_mm1_step(self, step: int) -> Dict[str, Any]:
        """Update rolling statistics and return predictions when scheduled."""
        if not hasattr(self, '_mm1_trackers'):
            return {}

        for key, tracker in self._mm1_trackers.items():
            tracker['arrivals'].append(self._mm1_step_arrivals.get(key, 0))
            tracker['services'].append(self._mm1_step_services.get(key, 0))
            tracker['queue_lengths'].append(self._mm1_step_queue_lengths.get(key, 0))
            delays = self._mm1_step_delays.get(key)
            avg_delay = float(np.mean(delays)) if delays else 0.0
            tracker['delays'].append(avg_delay)

        predictions: Dict[str, Any] = {}
        if step - self._mm1_last_prediction_step < self.mm1_prediction_interval:
            return predictions

        for key, tracker in self._mm1_trackers.items():
            window_steps = max(1, len(tracker['arrivals']))
            time_horizon = max(window_steps * float(self.time_slot), 1e-6)
            total_arrivals = sum(tracker['arrivals'])
            total_services = sum(tracker['services'])

            arrival_rate = total_arrivals / time_horizon
            service_rate = total_services / time_horizon
            if service_rate > 1e-6:
                rho = arrival_rate / service_rate
            else:
                rho = float('inf') if arrival_rate > 0.0 else 0.0
            stable = service_rate > arrival_rate and service_rate > 1e-6

            theoretical_queue = None
            theoretical_delay = None
            if stable:
                denom = max(1e-6, 1.0 - rho)
                theoretical_queue = (rho * rho) / denom
                theoretical_delay = 1.0 / max(1e-6, service_rate - arrival_rate)

            queue_samples = list(tracker['queue_lengths'])
            actual_queue = float(sum(queue_samples) / len(queue_samples)) if queue_samples else 0.0
            delay_samples = [d for d in tracker['delays'] if d > 0.0]
            actual_delay = float(sum(delay_samples) / len(delay_samples)) if delay_samples else 0.0

            predictions[key] = {
                'arrival_rate': arrival_rate,
                'service_rate': service_rate,
                'rho': rho,
                'stable': stable,
                'theoretical_queue': theoretical_queue,
                'actual_queue': actual_queue,
                'theoretical_delay': theoretical_delay,
                'actual_delay': actual_delay,
            }

        self._mm1_last_prediction_step = step
        return predictions
    
    def _get_realistic_content_size(self, content_id: str) -> float:
        """
        🔑 修复：使用realistic内容生成器获取大小
        
        根据内容ID获取真实的内容大小（MB），考虑不同类型内容的实际大小分布。
        
        Args:
            content_id: 内容ID
            
        Returns:
            内容大小（MB）
            
        Get realistic content size using content generator.
        """
        return get_realistic_content_size(content_id)
    
    def _calculate_available_cache_capacity(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        🔑 修复：正确计算可用缓存容量(MB)
        
        遍历缓存中的所有项目，累计已使用的空间，计算剩余可用容量。
        
        Args:
            cache: 缓存字典
            cache_capacity_mb: 缓存总容量（MB）
            
        Returns:
            可用缓存容量（MB）
            
        Calculate available cache capacity correctly.
        """
        if not cache or cache_capacity_mb <= 0:
            return cache_capacity_mb
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                # 兼容旧格式
                # Compatible with old format
                total_used_mb += 1.0
        
        available_mb = cache_capacity_mb - total_used_mb
        return max(0.0, available_mb)
    
    def _infer_content_type(self, content_id: str) -> str:
        """
        🔑 修复：根据内容ID推断内容类型
        
        根据内容ID中的关键字推断内容类型，用于缓存策略决策。
        
        Args:
            content_id: 内容ID
            
        Returns:
            内容类型字符串（如'traffic_info'、'navigation'等）
            
        Infer content type from content ID.
        """
        content_id_lower = content_id.lower()
        
        if 'traffic' in content_id_lower:
            return 'traffic_info'  # 交通信息
        elif 'nav' in content_id_lower or 'route' in content_id_lower:
            return 'navigation'  # 导航信息
        elif 'safety' in content_id_lower or 'alert' in content_id_lower:
            return 'safety_alert'  # 安全警报
        elif 'park' in content_id_lower:
            return 'parking_info'  # 停车信息
        elif 'weather' in content_id_lower:
            return 'weather_info'  # 天气信息
        elif 'map' in content_id_lower:
            return 'map_data'
        elif 'video' in content_id_lower or 'entertainment' in content_id_lower:
            return 'entertainment'
        elif 'sensor' in content_id_lower:
            return 'sensor_data'
        else:
            return 'general'
    
    def generate_task(self, vehicle_id: str) -> Dict:
        """
        生成计算任务 - 使用配置驱动的任务场景定义
        
        根据配置的任务场景（如导航、视频、安全警报等）生成具有
        不同特征的计算任务，包括数据大小、计算需求、截止时间等。
        
        Args:
            vehicle_id: 生成任务的车辆ID
            
        Returns:
            任务字典，包含任务的所有属性和要求
            
        Generate computational tasks with scenario-driven configuration.
        """
        self.task_counter += 1

        task_cfg = getattr(self.sys_config, 'task', None) if self.sys_config is not None else None
        time_slot = getattr(self.sys_config.network, 'time_slot_duration', self.time_slot) if self.sys_config is not None else self.time_slot

        # 默认场景参数
        scenario_name = 'fallback'
        relax_factor_applied = self.config.get('deadline_relax_fallback', 1.3)
        initial_type = 3

        if task_cfg is not None:
            scenario = task_cfg.sample_scenario()
            scenario_name = scenario.name
            relax_factor_applied = scenario.relax_factor or task_cfg.deadline_relax_default

            deadline_duration = np.random.uniform(scenario.min_deadline, scenario.max_deadline)
            deadline_duration *= relax_factor_applied
            max_delay_slots = max(1, int(deadline_duration / max(time_slot, 1e-6)))
            initial_type = scenario.task_type or task_cfg.get_task_type(
                max_delay_slots, time_slot=time_slot
            )

            profile = task_cfg.get_profile(initial_type)
            data_min, data_max = profile.data_range
            data_size_bytes = float(np.random.uniform(data_min, data_max))
            compute_density = profile.compute_density
        else:
            deadline_duration = np.random.uniform(0.5, 3.0) * relax_factor_applied
            initial_type = int(np.random.randint(1, 5))
            data_size_mb = np.random.exponential(0.5)
            data_size_bytes = data_size_mb * 1e6
            compute_density = self.config.get('task_compute_density', 400)
            max_delay_slots = max(
                1,
                int(deadline_duration / max(self.config.get('time_slot', self.time_slot), 0.1)),
            )

        # 任务复杂度控制
        data_size_mb = data_size_bytes / 1e6
        effective_density = compute_density
        complexity_multiplier = 1.0

        if self.config.get('high_load_mode', False):
            complexity_multiplier = self.config.get('task_complexity_multiplier', 1.5)
            data_size_mb = min(data_size_mb * 1.1, 2.0)
            data_size_bytes = data_size_mb * 1e6
            effective_density = min(effective_density * 1.05, 200)

        total_bits = data_size_bytes * 8
        base_cycles = total_bits * effective_density
        adjusted_cycles = base_cycles * complexity_multiplier
        computation_mips = adjusted_cycles / 1e6

        cacheable_hint = scenario_name in {'video_process', 'image_recognition', 'data_analysis', 'ml_training'}
        if task_cfg is not None:
            refined_type = task_cfg.get_task_type(
                max_delay_slots,
                data_size=data_size_bytes,
                compute_cycles=adjusted_cycles,
                compute_density=effective_density,
                time_slot=time_slot,
                system_load=self.config.get('system_load_hint'),
                is_cacheable=cacheable_hint,
            )
            task_type = max(initial_type, refined_type)
        else:
            task_type = initial_type

        task = {
            'id': f'task_{self.task_counter}',
            'vehicle_id': vehicle_id,
            'arrival_time': self.current_time,
            'data_size': data_size_mb,
            'data_size_bytes': data_size_bytes,
            'computation_requirement': computation_mips,
            'compute_cycles': adjusted_cycles,
            'deadline': self.current_time + deadline_duration,
            'content_id': f'content_{np.random.randint(0, 100)}',
            'priority': np.random.uniform(0.1, 1.0),
            'task_type': task_type,
            'app_scenario': scenario_name,
            'app_name': scenario_name,
            'compute_density': effective_density,
            'complexity_multiplier': complexity_multiplier,
            'max_delay_slots': max_delay_slots,
            'deadline_relax_factor': relax_factor_applied
        }

        self._last_app_name = scenario_name

        # 馃搳 浠诲姟缁熻鏀堕泦
        gen_stats = self.stats.setdefault('task_generation', {'total': 0, 'by_type': {}, 'by_scenario': {}})
        gen_stats['total'] += 1
        by_type = gen_stats.setdefault('by_type', {})
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_scenario = gen_stats.setdefault('by_scenario', {})
        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1

        report_interval = self.stats_config.task_report_interval if getattr(self, 'stats_config', None) else self.config.get('task_report_interval', 100)
        report_interval = max(1, int(report_interval))
        if gen_stats['total'] % report_interval == 0:
            total_classified = sum(by_type.values()) or 1
            type1_pct = by_type.get(1, 0) / total_classified * 100
            type2_pct = by_type.get(2, 0) / total_classified * 100
            type3_pct = by_type.get(3, 0) / total_classified * 100
            type4_pct = by_type.get(4, 0) / total_classified * 100
            print(
                f"任务分类统计({gen_stats['total']}): "
                f"类型1={type1_pct:.1f}%, 类型2={type2_pct:.1f}%, 类型3={type3_pct:.1f}%, 类型4={type4_pct:.1f}%"
            )
            print(
                f"   当前任务: {scenario_name}, {deadline_duration:.2f}s → "
                f"类型{task_type}, 数据{data_size_mb:.2f}MB"
            )

        return task
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        计算两点之间的欧几里得距离（支持2D和3D坐标自动转换）
        Calculate Euclidean distance between two points (supports automatic 2D/3D conversion)
        
        该方法能够智能处理2D和3D坐标的混合情况：
        - 如果其中一个点是2D，另一个是3D，自动将2D点扩展为3D（z=0）
        - 然后使用NumPy的线性代数模块计算欧几里得距离
        
        This method intelligently handles mixed 2D/3D coordinates:
        - If one point is 2D and the other is 3D, automatically extends 2D to 3D (z=0)
        - Then uses NumPy's linear algebra module to calculate Euclidean distance
        
        参数 Args:
            pos1: 第一个点的坐标数组 (可以是2D或3D) | Coordinate array of first point (can be 2D or 3D)
            pos2: 第二个点的坐标数组 (可以是2D或3D) | Coordinate array of second point (can be 2D or 3D)
            
        返回 Returns:
            float: 两点之间的距离（米） | Distance between two points (meters)
        """
        # 处理维度不匹配的情况：将2D坐标扩展为3D
        # Handle dimension mismatch: extend 2D coordinates to 3D
        if len(pos1) == 3 and len(pos2) == 2:
            pos2 = np.append(pos2, 0)  # 2D转3D，z坐标设为0 | 2D to 3D, set z=0
        elif len(pos1) == 2 and len(pos2) == 3:
            pos1 = np.append(pos1, 0)
        
        # 使用NumPy计算L2范数（欧几里得距离）
        # Use NumPy to calculate L2 norm (Euclidean distance)
        return np.linalg.norm(pos1 - pos2)
    
    
    def _refresh_spatial_index(self, update_static: bool = True, update_vehicle: bool = True) -> None:
        """
        保持空间索引与实体位置同步。
        update_static=False 时仅刷新车辆索引，避免重复构建静态KD-tree。
        """
        if not getattr(self, 'spatial_index', None):
            return
        try:
            if update_static:
                self.spatial_index.update_static_nodes(self.rsus, self.uavs)
            if update_vehicle:
                self.spatial_index.update_vehicle_nodes(self.vehicles)
        except Exception:
            # 索引刷新失败时回退至朴素遍历逻辑
            pass
    
    
    def _find_least_loaded_node(self, node_type: str, exclude_node: Dict = None) -> Dict:
        """
        寻找负载最轻的节点（用于任务分配和迁移决策）
        Find the least loaded node (for task assignment and migration decisions)
        
        该方法根据队列长度来衡量节点负载，选择最空闲的节点：
        - 支持RSU和UAV两种节点类型
        - 可以排除特定节点（如当前已过载节点）
        - 通过比较computation_queue长度找到最佳候选
        - 用于负载均衡和智能任务调度
        
        This method measures node load by queue length and selects the most idle node:
        - Supports both RSU and UAV node types
        - Can exclude specific nodes (e.g., currently overloaded node)
        - Finds best candidate by comparing computation_queue length
        - Used for load balancing and intelligent task scheduling
        
        参数 Args:
            node_type: 节点类型 'RSU' 或 'UAV' | Node type 'RSU' or 'UAV'
            exclude_node: 需要排除的节点（可选） | Node to exclude (optional)
            
        返回 Returns:
            Dict: 负载最轻的节点字典，如果没有候选返回None | Least loaded node dict, or None if no candidates
        """
        # 根据节点类型筛选候选节点，排除指定节点
        # Filter candidates by node type, excluding specified node
        if node_type == 'RSU':
            candidates = [rsu for rsu in self.rsus if rsu != exclude_node]
        elif node_type == 'UAV':
            candidates = [uav for uav in self.uavs if uav != exclude_node]
        else:
            return None
        
        if not candidates:
            return None
        
        # 找到队列长度最短的节点（负载最轻）
        # Find the node with the shortest queue (least loaded)
        # 使用min函数配合lambda表达式，按computation_queue长度排序
        # Use min function with lambda to sort by computation_queue length
        best_node = min(candidates, key=lambda n: len(n.get('computation_queue', [])))
        return best_node
    
    def _process_node_queues(self):
        """
        🔑 关键修复：处理RSU和UAV队列中的任务，防止任务堆积
        
        遍历所有RSU和UAV节点，处理它们计算队列中的任务。
        这是任务执行的核心逻辑。
        
        Process tasks in RSU and UAV queues to prevent task accumulation.
        """
        # 处理所有RSU队列
        for idx, rsu in enumerate(self.rsus):
            self._process_single_node_queue(rsu, 'RSU', idx)
        
        # 处理所有UAV队列
        for idx, uav in enumerate(self.uavs):
            self._process_single_node_queue(uav, 'UAV', idx)
    
    def _get_node_capacity_scale(self, node: Dict, node_type: str) -> float:
        """根据中央资源分配结果计算节点处理能力缩放因子。"""
        if node_type == 'RSU':
            baseline = float(getattr(self, 'rsu_cpu_freq', 15e9))
        else:
            baseline = float(getattr(self, 'uav_cpu_freq', 4e9))
        allocated = float(node.get('allocated_compute', baseline))
        if baseline <= 0:
            return 1.0
        scale = allocated / baseline
        return float(np.clip(scale, 0.2, 3.0))

    def _is_node_admissible(self, node: Dict, node_type: str) -> bool:
        """检查节点是否允许新的卸载任务进入。"""
        queue_len = len(node.get('computation_queue', []))
        capacity = self.rsu_nominal_capacity if node_type == 'RSU' else self.uav_nominal_capacity
        ratio = queue_len / max(1.0, capacity)
        usage = float(node.get('compute_usage', 0.0))
        threshold = max(0.5, self.node_max_load_factor)
        return (ratio < threshold) and (usage < threshold or usage == 0.0)

    def _record_offload_rejection(self, node_type: str, reason: str = 'unknown') -> None:
        """记录由于拥塞/策略导致的远端卸载拒绝。"""
        stats = self.stats.setdefault('remote_rejections', {
            'total': 0,
            'by_type': {'RSU': 0, 'UAV': 0},
            'by_reason': {}
        })
        stats['total'] = stats.get('total', 0) + 1
        by_type = stats.setdefault('by_type', {})
        by_type[node_type] = by_type.get(node_type, 0) + 1
        by_reason = stats.setdefault('by_reason', {})
        by_reason[reason] = by_reason.get(reason, 0) + 1

    def _process_single_node_queue(self, node: Dict, node_type: str, node_idx: int) -> None:
        """
        处理单个节点的计算队列
        
        实现动态任务调度，根据队列长度自适应调整处理能力：
        - 基础处理能力：每个时隙处理固定数量的任务
        - 动态提升：队列过长时增加处理能力
        - 工作量计算：基于任务的计算需求
        
        Args:
            node: 节点字典（RSU或UAV）
            node_type: 节点类型（'RSU'或'UAV'）
            
        Process single node's computation queue with adaptive scheduling.
        """
        queue = node.get('computation_queue', [])
        queue_len = len(queue)
        if queue_len == 0:
            self._record_mm1_queue_length(node_type, node_idx, 0)
            return

        # 根据节点类型获取处理能力配置
        # Get processing capacity configuration based on node type
        if node_type == 'RSU':
            if self.service_config:
                base_capacity = int(self.service_config.rsu_base_service)  # 基础处理能力
                max_service = int(self.service_config.rsu_max_service)  # 最大处理能力
                boost_divisor = float(self.service_config.rsu_queue_boost_divisor)  # 动态提升除数
                work_capacity_cfg = float(self.service_config.rsu_work_capacity)  # 工作容量
            else:
                base_capacity = int(self.config.get('rsu_base_service', 4))
                max_service = int(self.config.get('rsu_max_service', 9))
                boost_divisor = 5.0
                work_capacity_cfg = float(self.config.get('rsu_work_capacity', 2.5))
        elif node_type == 'UAV':
            if self.service_config:
                base_capacity = int(self.service_config.uav_base_service)
                max_service = int(self.service_config.uav_max_service)
                boost_divisor = float(self.service_config.uav_queue_boost_divisor)
                work_capacity_cfg = float(self.service_config.uav_work_capacity)
            else:
                base_capacity = int(self.config.get('uav_base_service', 3))
                max_service = int(self.config.get('uav_max_service', 6))
                boost_divisor = 4.0
                work_capacity_cfg = float(self.config.get('uav_work_capacity', 1.7))
        else:
            base_capacity = 2
            max_service = 4
            boost_divisor = 5.0
            work_capacity_cfg = 1.2

        capacity_scale = self._get_node_capacity_scale(node, node_type)
        base_capacity = max(1, int(round(base_capacity * capacity_scale)))
        max_service = max(base_capacity, int(round(max_service * capacity_scale)))
        work_capacity_cfg *= capacity_scale

        if queue_len > base_capacity:
            dynamic_boost = int(np.ceil((queue_len - base_capacity) / boost_divisor))
        else:
            dynamic_boost = 0

        tasks_to_process = min(queue_len, base_capacity + dynamic_boost)
        tasks_to_process = min(tasks_to_process, max_service)
        tasks_to_process = max(tasks_to_process, min(queue_len, base_capacity))

        new_queue: List[Dict] = []
        current_time = getattr(self, 'current_time', 0.0)
        
        # 🔧 修复v2：移除频率缩放，使用固定的work_capacity
        # work_capacity_cfg已经是基于实际硬件校准的经验值
        work_capacity = self.time_slot * work_capacity_cfg

        for idx, task in enumerate(queue):
            if current_time - task.get('queued_at', -1e9) < self.time_slot:
                new_queue.append(task)
                continue

            if idx >= tasks_to_process:
                new_queue.append(task)
                continue

            previous_work = float(task.get('work_remaining', 0.5))
            remaining_work = max(0.0, previous_work - work_capacity)
            task['work_remaining'] = remaining_work
            consumed_ratio = (previous_work - remaining_work) / max(work_capacity, 1e-9)
            consumed_ratio = float(np.clip(consumed_ratio, 0.0, 1.0))
            incremental_service = consumed_ratio * self.time_slot
            task['service_time'] = task.get('service_time', 0.0) + incremental_service

            if task['work_remaining'] > 0.0:
                new_queue.append(task)
                continue

            self.stats['completed_tasks'] += 1
            self.stats['processed_tasks'] = self.stats.get('processed_tasks', 0) + 1

            actual_delay = current_time - task.get('arrival_time', current_time)
            clip_upper = getattr(self, 'delay_clip_upper', 0.0)
            if clip_upper > 0.0:
                actual_delay = min(actual_delay, clip_upper)
            actual_delay = max(0.0, actual_delay)
            service_time = min(actual_delay, task.get('service_time', actual_delay))
            wait_delay = max(0.0, actual_delay - service_time)
            self._accumulate_delay('delay_processing', service_time)
            if wait_delay > 0.0:
                self._accumulate_delay('delay_waiting', wait_delay)
            self._record_mm1_service(node_type, node_idx, actual_delay)

            vehicle_id = task.get('vehicle_id', 'V_0')
            vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
            if vehicle is not None:
                node_pos = node.get('position', np.zeros(3))
                if len(node_pos) == 2:
                    node_pos = np.append(node_pos, 0.0)
                vehicle_pos = vehicle.get('position', np.zeros(3))
                if len(vehicle_pos) == 2:
                    vehicle_pos = np.append(vehicle_pos, 0.0)
                distance = np.linalg.norm(node_pos - vehicle_pos)
                result_size = task.get('data_size_bytes', task.get('data_size', 1.0) * 1e6) * 0.1
                down_delay, down_energy = self._estimate_transmission(result_size, distance, node_type.lower())
                self.stats['energy_downlink'] = self.stats.get('energy_downlink', 0.0) + down_energy
                self._accumulate_delay('delay_downlink', down_delay)
                self._accumulate_energy('energy_transmit_downlink', down_energy)

            if node_type == 'RSU':
                processing_power = 50.0
            elif node_type == 'UAV':
                processing_power = 20.0
            else:
                processing_power = 10.0

            task_energy = processing_power * work_capacity
            self._accumulate_energy('energy_compute', task_energy)
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + task_energy

            task['completed'] = True

        node['computation_queue'] = new_queue
        self._record_mm1_queue_length(node_type, node_idx, len(new_queue))


    def find_nearest_rsu(self, vehicle_pos: np.ndarray) -> Dict:
        """
        ??????????????????RSU?
        Fallback to brute-force iteration when the index is unavailable.
        """
        if not self.rsus:
            return None

        vehicle_vec = np.asarray(vehicle_pos, dtype=float)
        best_node = None
        best_distance = float('inf')

        if getattr(self, 'spatial_index', None):
            nearest = self.spatial_index.find_nearest_rsu(vehicle_vec, return_distance=True)
            if nearest:
                _, node, dist = nearest
                coverage = float(node.get('coverage_radius', self.coverage_radius))
                if dist <= coverage:
                    return node
                best_node = node
                best_distance = dist

            max_radius = self.spatial_index.rsu_max_radius or max(
                (float(rsu.get('coverage_radius', self.coverage_radius)) for rsu in self.rsus),
                default=self.coverage_radius,
            )
            neighbors = self.spatial_index.query_rsus_within_radius(vehicle_vec, max_radius)
            for _, node, dist in neighbors:
                coverage = float(node.get('coverage_radius', self.coverage_radius))
                if dist <= coverage and dist < best_distance:
                    best_node = node
                    best_distance = dist

            if best_node and best_distance <= best_node.get('coverage_radius', self.coverage_radius):
                return best_node

        for rsu in self.rsus:
            distance = self.calculate_distance(vehicle_vec, rsu['position'])
            coverage = float(rsu.get('coverage_radius', self.coverage_radius))
            if distance <= coverage and distance < best_distance:
                best_node = rsu
                best_distance = distance

        return best_node

    def find_nearest_uav(self, vehicle_pos: np.ndarray) -> Dict:
        """
        ???????????UAV???
        """
        if not self.uavs:
            return None

        vehicle_vec = np.asarray(vehicle_pos, dtype=float)
        if getattr(self, 'spatial_index', None):
            nearest = self.spatial_index.find_nearest_uav(vehicle_vec, return_distance=True)
            if nearest:
                return nearest[1]

        min_distance = float('inf')
        nearest_uav = None
        for uav in self.uavs:
            distance = self.calculate_distance(vehicle_vec, uav['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_uav = uav

        return nearest_uav

    def check_cache_hit(self, content_id: str, node: Dict) -> bool:
        """
        检查缓存命中
        
        Args:
            content_id: 内容ID
            node: 节点字典
            
        Returns:
            True表示命中，False表示未命中
            
        Check if content is cached in the node.
        """
        if content_id in node.get('cache', {}):
            self.stats['cache_hits'] += 1
            return True
        else:
            self.stats['cache_misses'] += 1
            return False
    
    def check_cache_hit_adaptive(
        self,
        content_id: str,
        node: Dict,
        agents_actions: Dict = None,
        node_type: str = 'RSU'
    ) -> bool:
        """
        🌟 智能体控制的自适应缓存检查
        
        结合智能缓存控制器，实现自适应的缓存策略：
        - 基础缓存命中检查
        - 缓存未命中时的智能决策（是否缓存、如何淘汰）
        - 协同缓存传播（RSU到车辆、RSU到RSU）
        - 内容热度追踪
        
        Args:
            content_id: 内容ID
            node: 节点字典
            agents_actions: 智能体动作字典（包含cache_controller）
            node_type: 节点类型（'RSU'或'UAV'）
            
        Returns:
            True表示命中，False表示未命中
            
        Adaptive cache checking with intelligent caching controller.
        """
        # 基础缓存检查
        # Basic cache check
        cache = node.get('cache', {})
        cache_hit = bool(content_id and cache and content_id in cache)
        self._register_cache_request(cache_hit)
        
        # 更新统计
        # Update statistics
        if cache_hit:
            self.stats['cache_hits'] += 1
            if node_type == 'RSU':
                self._propagate_cache_after_hit(content_id, node, agents_actions)
        else:
            self.stats['cache_misses'] += 1
            
            # 🌟 如果有智能体控制器，执行自适应缓存策略
            # Execute adaptive caching strategy with intelligent controller
            if agents_actions and 'cache_controller' in agents_actions:
                cache_controller = agents_actions['cache_controller']
                rl_guidance = agents_actions.get('rl_guidance') if isinstance(agents_actions, dict) else None
                cache_preference = 0.5
                if isinstance(rl_guidance, dict):
                    tradeoff_weights = rl_guidance.get('tradeoff_weights')
                    if isinstance(tradeoff_weights, (list, tuple)) and len(tradeoff_weights) >= 2:
                        cache_preference = float(np.clip(tradeoff_weights[1], 0.0, 1.0))
                    else:
                        cache_bias = rl_guidance.get('cache_bias')
                        if isinstance(cache_bias, (list, tuple)) and len(cache_bias) > 0:
                            cache_preference = float(np.clip(np.mean(cache_bias), 0.0, 1.0))
                    energy_pressure_vec = rl_guidance.get('energy_pressure')
                    if isinstance(energy_pressure_vec, (list, tuple, np.ndarray)):
                        energy_pressure = float(np.clip(np.asarray(energy_pressure_vec, dtype=float).reshape(-1)[0], 0.35, 1.8))
                        cache_preference = float(np.clip(cache_preference * energy_pressure, 0.0, 1.0))

                
                # 更新内容热度
                # Update content heat
                cache_controller.update_content_heat(content_id)
                cache_controller.record_cache_result(content_id, was_hit=False)
                
                # 🔑 修复：使用realistic内容大小和正确容量计算
                # Fix: Use realistic content size and correct capacity calculation
                data_size = self._get_realistic_content_size(content_id)
                capacity_limit = node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
                available_capacity = self._calculate_available_cache_capacity(
                    node.get('cache', {}), capacity_limit
                )
                
                guard_ratio = getattr(self, 'cache_pressure_guard', 0.05)
                pressure_ratio = available_capacity / max(1.0, capacity_limit)
                severe_pressure = pressure_ratio < guard_ratio

                # 调用智能控制器判断是否缓存（在极端压力下直接跳过写入）
                if severe_pressure:
                    should_cache = False
                    reason = 'pressure_guard'
                    evictions = []
                else:
                    should_cache, reason, evictions = cache_controller.should_cache_content(
                        content_id,
                        data_size,
                        available_capacity,
                        node.get('cache', {}),
                        capacity_limit
                    )
                
                # 如果决定缓存，执行淘汰和写入操作
                if should_cache and cache_preference < 0.35:
                    should_cache = False
                elif not should_cache and cache_preference > 0.7 and available_capacity >= data_size:
                    should_cache = True
                    reason = reason or 'RL-guided cache'
                    evictions = []
                elif should_cache and available_capacity < data_size and not evictions:
                    should_cache = False

                # If decided to cache, perform eviction and write operations
                if should_cache:
                    if 'cache' not in node:
                        node['cache'] = {}
                    cache_dict = node['cache']
                    reclaimed = 0.0
                    # 执行淘汰操作，回收空间
                    # Perform eviction to reclaim space
                    for evict_id in evictions:
                        removed = cache_dict.pop(evict_id, None)
                        if removed:
                            reclaimed += float(removed.get('size', 0.0) or 0.0)
                            cache_controller.cache_stats['evicted_items'] += 1
                    if reclaimed > 0.0:
                        available_capacity += reclaimed
                    if available_capacity < data_size:
                        return cache_hit
                    # 写入新内容到缓存
                    # Write new content to cache
                    cache_dict[content_id] = {
                        'size': data_size,
                        'timestamp': self.current_time,
                        'reason': reason,
                        'content_type': self._infer_content_type(content_id)
                    }
                    # 统计协同缓存写入
                    # Count collaborative cache writes
                    if 'Collaborative cache' in reason:
                        cache_controller.cache_stats['collaborative_writes'] += 1
        
        # 记录缓存控制器统计（缓存命中情况）
        # Record cache controller statistics (cache hit case)
        if agents_actions and 'cache_controller' in agents_actions and cache_hit:
            cache_controller = agents_actions['cache_controller'] 
            cache_controller.record_cache_result(content_id, was_hit=True)
            cache_controller.update_content_heat(content_id)
            
        return cache_hit
    
    def _calculate_node_rho(self, node: Dict, node_type: str) -> float:
        """Estimate queue utilization (?) based on nominal capacities."""
        if node_type == 'RSU':
            capacity = max(1.0, float(self.rsu_nominal_capacity))
        elif node_type == 'UAV':
            capacity = max(1.0, float(self.uav_nominal_capacity))
        else:
            capacity = 1.0
        queue_length = len(node.get('computation_queue', []))
        return float(queue_length / capacity)

    def _calculate_enhanced_load_factor(self, node: Dict, node_type: str) -> float:
        """
        馃敡 淇锛氱粺涓€鍜宺ealistic鐨勮礋杞藉洜瀛愯绠?
        鍩轰簬瀹為檯闃熷垪璐熻浇锛屼笉浣跨敤铏氬亣鐨勯檺鍒?
        """
        queue_length = len(node.get('computation_queue', []))
        
        # 馃敡 鍩轰簬瀹為檯瑙傚療璋冩暣瀹归噺鍩哄噯
        if node_type == 'RSU':
            # 鍩轰簬瀹為檯娴嬭瘯锛孯SU澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            queue_factor = self._calculate_node_rho(node, 'RSU')
        else:  # UAV
            # UAV澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            queue_factor = self._calculate_node_rho(node, 'UAV')
        
        # 馃敡 淇锛氫娇鐢ㄦ纭殑缂撳瓨璁＄畻
        cache_utilization = self._calculate_correct_cache_utilization(
            node.get('cache', {}), 
            node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
        )
        
        # 馃敡 绠€鍖栦絾鍑嗙‘鐨勮礋杞借绠?
        load_factor = (
            0.8 * queue_factor +           # 闃熷垪鏄富瑕佽礋杞芥寚鏍?0%
            0.2 * cache_utilization       # 缂撳瓨鍒╃敤鐜?0%
        )
        
        # 馃敡 涓嶉檺鍒跺湪1.0锛屽厑璁告樉绀虹湡瀹炶繃杞界▼搴?
        return max(0.0, load_factor)
    
    def _monitor_queue_stability(self) -> Dict[str, Any]:
        """Monitor aggregate queue load and report stability metrics."""
        node_rhos: Dict[str, float] = {}
        overloaded_nodes: Dict[str, float] = {}
        approaching_nodes: Dict[str, float] = {}
        total_rho = 0.0
        max_rho = 0.0
        warning_threshold = self.queue_warning_ratio * self.node_max_load_factor if self.node_max_load_factor > 0 else self.queue_warning_ratio

        for idx, rsu in enumerate(self.rsus):
            rho = self._calculate_node_rho(rsu, 'RSU')
            node_id = f'RSU_{idx}'
            node_rhos[node_id] = rho
            total_rho += rho
            max_rho = max(max_rho, rho)
            if rho >= self.node_max_load_factor:
                overloaded_nodes[node_id] = rho
            elif rho >= warning_threshold:
                approaching_nodes[node_id] = rho

        for idx, uav in enumerate(self.uavs):
            rho = self._calculate_node_rho(uav, 'UAV')
            node_id = f'UAV_{idx}'
            node_rhos[node_id] = rho
            total_rho += rho
            max_rho = max(max_rho, rho)
            if rho >= self.node_max_load_factor:
                overloaded_nodes[node_id] = rho
            elif rho >= warning_threshold:
                approaching_nodes[node_id] = rho

        overloaded = total_rho >= self.queue_stability_threshold
        self.stats['queue_rho_sum'] = total_rho
        self.stats['queue_rho_max'] = max_rho
        self.stats['queue_overload_flag'] = overloaded
        self.stats['queue_rho_by_node'] = dict(node_rhos)
        if overloaded:
            self.stats['queue_overload_events'] = self.stats.get('queue_overload_events', 0) + 1

        if overloaded and not self._queue_overload_warning_active:
            detail = ', '.join(f"{node}:{rho:.2f}" for node, rho in overloaded_nodes.items()) or 'none'
            print(f"[Stability] Σρ={total_rho:.2f} exceeds threshold {self.queue_stability_threshold:.2f}. Overloaded nodes: {detail}")
        elif not overloaded and self._queue_overload_warning_active:
            print('[Stability] Queue load returned below stability threshold.')

        if not overloaded:
            if approaching_nodes and not self._queue_warning_triggered:
                detail = ', '.join(f"{node}:{rho:.2f}" for node, rho in approaching_nodes.items())
                print(f"[Stability] Queue load approaching limit: {detail}")
                self._queue_warning_triggered = True
            elif not approaching_nodes:
                self._queue_warning_triggered = False
        else:
            self._queue_warning_triggered = True

        self._queue_overload_warning_active = overloaded

        return {
            'queue_rho_sum': total_rho,
            'queue_rho_max': max_rho,
            'queue_overload_flag': overloaded,
            'queue_rho_by_node': node_rhos,
            'queue_overloaded_nodes': overloaded_nodes,
            'queue_warning_nodes': approaching_nodes
        }


    def _summarize_task_types(self) -> Dict[str, Any]:
        """Aggregate per-task-type queues, active counts, and deadline slack."""
        num_types = 4
        queue_counts = np.zeros(num_types, dtype=float)
        active_counts = np.zeros(num_types, dtype=float)
        deadline_sums = np.zeros(num_types, dtype=float)
        deadline_counts = np.zeros(num_types, dtype=float)

        current_time = getattr(self, "current_time", 0.0)

        def _record(entry: Dict[str, Any]) -> Optional[int]:
            task_type = int(entry.get("task_type", 0) or 0) - 1
            if 0 <= task_type < num_types:
                remaining = max(0.0, entry.get("deadline", current_time) - current_time)
                deadline_sums[task_type] += remaining
                deadline_counts[task_type] += 1.0
                return task_type
            return None

        for node in list(self.rsus) + list(self.uavs):
            for task in node.get("computation_queue", []):
                idx = _record(task)
                if idx is not None:
                    queue_counts[idx] += 1.0

        for task in self.active_tasks:
            idx = _record(task)
            if idx is not None:
                active_counts[idx] += 1.0

        if self.task_config is not None and hasattr(self.task_config, "deadline_range"):
            deadline_upper = float(getattr(self.task_config, "deadline_range", (1.0, 10.0))[1])
        else:
            fallback_range = self.config.get("deadline_range", (1.0, 10.0))
            if isinstance(fallback_range, (list, tuple)) and len(fallback_range) >= 2:
                deadline_upper = float(fallback_range[1])
            else:
                deadline_upper = float(self.config.get("deadline_range_max", 10.0))
        deadline_upper = max(deadline_upper, 1.0)

        queue_total = float(queue_counts.sum())
        active_total = float(active_counts.sum())

        def _normalize(counts: np.ndarray, total: float) -> List[float]:
            if total <= 0.0:
                return [0.0] * num_types
            return [float(np.clip(val / total, 0.0, 1.0)) for val in counts]

        deadline_features = []
        for idx in range(num_types):
            if deadline_counts[idx] > 0.0:
                avg_remaining = deadline_sums[idx] / deadline_counts[idx]
                deadline_features.append(float(np.clip(avg_remaining / deadline_upper, 0.0, 1.0)))
            else:
                deadline_features.append(0.0)

        return {
            "task_type_queue_distribution": _normalize(queue_counts, queue_total),
            "task_type_active_distribution": _normalize(active_counts, active_total),
            "task_type_deadline_remaining": deadline_features,
            "task_type_queue_counts": [float(c) for c in queue_counts],
            "task_type_active_counts": [float(c) for c in active_counts],
        }
    
    def _calculate_correct_cache_utilization(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        馃敡 璁＄畻姝ｇ‘鐨勭紦瀛樺埄鐢ㄧ巼
        """
        if not cache or cache_capacity_mb <= 0:
            return 0.0
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                total_used_mb += 1.0  # 兼容旧格式
        
        utilization = total_used_mb / cache_capacity_mb
        return min(1.0, max(0.0, utilization))

    # ==================== 新增：一步仿真涉及的核心辅助函数 ====================
    # Core helper functions for single-step simulation

    def _update_vehicle_positions(self):
        """
        简单更新车辆位置，模拟车辆沿主干道移动
        
        实现了逼真的车辆移动模型，包括：
        - 速度的加减速变化
        - 路口减速行为
        - 车道切换和横向漂移
        - 周期性边界条件（环形道路）
        
        Simple vehicle position update with realistic movement simulation.
        """
        for vehicle in self.vehicles:
            position = vehicle.get('position')
            if position is None or len(position) < 2:
                continue

            # === 1) 更新速度（缓慢加减速 + 交叉口减速） ===
            # Update velocity with gradual acceleration and intersection slowdown
            base_speed = float(vehicle.get('velocity', 15.0))
            accel_state = vehicle.setdefault('speed_accel', 0.0)
            accel_state = 0.7 * accel_state + np.random.uniform(-0.4, 0.4)

            # 在接近路口时降低速度，避免高速冲过交叉口
            # Slow down near intersections
            for intersection in self.intersections.values():
                dist_to_signal = abs(position[0] - intersection['x'])
                if dist_to_signal < 40.0:
                    accel_state = min(accel_state, -0.8)
                    break

            new_speed = np.clip(base_speed + accel_state, 5.0, 32.0)
            vehicle['speed_accel'] = accel_state
            vehicle['velocity'] = new_speed

            # === 2) 方向保持，同时允许轻微扰动 ===
            direction = vehicle.get('direction', 0.0)
            heading_jitter = vehicle.setdefault('heading_jitter', 0.0)
            heading_jitter = 0.6 * heading_jitter + np.random.uniform(-0.01, 0.01)
            direction = (direction + heading_jitter) % (2 * np.pi)
            vehicle['direction'] = direction
            vehicle['heading_jitter'] = heading_jitter

            dx = np.cos(direction) * new_speed * self.time_slot
            dy = np.sin(direction) * new_speed * self.time_slot

            # === 3) 渚у悜婕傜Щ锛堟ā鎷熻交寰崲閬擄級 ===
            lane_bias = vehicle.get('lane_bias', position[1] - self.road_y)
            lane_switch_timer = vehicle.setdefault('lane_switch_timer', np.random.randint(80, 160))
            lane_switch_timer -= 1
            if lane_switch_timer <= 0 and np.random.rand() < 0.1:
                lane_bias = np.clip(lane_bias + np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.5),
                                    -6.0, 6.0)
                lane_switch_timer = np.random.randint(120, 220)
            vehicle['lane_switch_timer'] = lane_switch_timer
            vehicle['lane_bias'] = lane_bias

            lateral_state = vehicle.setdefault('lateral_state', 0.0)
            lateral_state = 0.5 * lateral_state + np.random.uniform(-0.25, 0.25)
            vehicle['lateral_state'] = np.clip(lateral_state, -2.0, 2.0)

            # === 4) 应用位置更新（x 环路，y 叠加 lane_bias 与漂移影响） ===
            new_x = (position[0] + dx) % 1000.0
            baseline_lane_y = float(self.road_y + lane_bias)
            new_y = baseline_lane_y + vehicle['lateral_state']
            new_y = np.clip(new_y, self.road_y - 6.5, self.road_y + 6.5)

            vehicle['position'][0] = new_x
            vehicle['position'][1] = new_y

        self._refresh_spatial_index(update_static=False, update_vehicle=True)

    def _sample_arrivals(self) -> int:
        """鎸夋硦鏉捐繃绋嬮噰鏍锋瘡杞︽瘡鏃堕殭鐨勪换鍔″埌杈炬暟"""
        lam = max(1e-6, float(self.task_arrival_rate) * float(self.time_slot))
        return int(np.random.poisson(lam))

    def _choose_offload_target(self, actions: Dict, rsu_available: bool, uav_available: bool) -> str:
        """鏍规嵁鏅鸿兘浣撴彁渚涚殑鍋忓ソ閫夋嫨鍗歌浇鐩爣"""
        prefs = actions.get('vehicle_offload_pref') or {}
        probs = np.array([
            max(0.0, float(prefs.get('local', 0.0))),
            max(0.0, float(prefs.get('rsu', 0.0))) if rsu_available else 0.0,
            max(0.0, float(prefs.get('uav', 0.0))) if uav_available else 0.0,
        ], dtype=float)

        guidance = actions.get('rl_guidance') or {}
        if isinstance(guidance, dict):
            guide_prior = np.array(guidance.get('offload_prior', []), dtype=float)
            if guide_prior.size >= 3:
                probs *= np.clip(guide_prior[:3], 1e-4, None)
            distance_focus = np.array(guidance.get('distance_focus', []), dtype=float)
            if distance_focus.size >= 3:
                probs *= np.clip(distance_focus[:3], 0.2, None)
            cache_focus = np.array(guidance.get('cache_focus', []), dtype=float)
            if cache_focus.size >= 3:
                probs *= np.clip(cache_focus[:3], 0.2, None)
            energy_pressure_vec = guidance.get('energy_pressure')
            if isinstance(energy_pressure_vec, (list, tuple, np.ndarray)):
                pressure = float(np.clip(np.asarray(energy_pressure_vec, dtype=float).reshape(-1)[0], 0.35, 1.8))
                energy_weights = np.array([1.0 / pressure, pressure, pressure], dtype=float)
                probs *= energy_weights

        if probs.sum() <= 0:
            probs = np.array([
                0.34,
                0.33 if rsu_available else 0.0,
                0.33 if uav_available else 0.0
            ], dtype=float)

        if probs.sum() <= 0:
            return 'local'

        probs = probs / probs.sum()
        target_labels = np.array(['local', 'rsu', 'uav'])
        return str(np.random.choice(target_labels, p=probs))

    def _estimate_remote_work_units(self, task: Dict, node_type: str) -> float:
        """
        估计远程节点的工作量单位（供队列调度使用）
        
        🔧 修复v2：不再使用频率缩放，直接使用固定的base_divisor
        原因：base_divisor是经验校准值，已经包含了硬件差异
        """
        requirement = float(task.get('computation_requirement', 1500.0))
        
        # 使用固定的base_divisor（这些值是基于实际硬件校准的）
        # RSU: 高性能边缘服务器，base_divisor较大
        # UAV: 低功耗无人机芯片，base_divisor较小（执行更慢）
        if node_type == 'RSU':
            base_divisor = 1200.0  # RSU固定值
        else:  # UAV
            base_divisor = 1600.0  # UAV固定值
        
        work_units = requirement / base_divisor
        return float(np.clip(work_units, 0.5, 12.0))

    def _estimate_local_processing(self, task: Dict, vehicle: Dict) -> Tuple[float, float]:
        """浼拌鏈湴澶勭悊鐨勫欢杩熶笌鑳借€?"""
        cpu_freq = 2.5e9
        power = 6.5
        if self.sys_config is not None:
            cpu_freq = getattr(self.sys_config.compute, 'vehicle_cpu_freq', cpu_freq)
            power = getattr(self.sys_config.compute, 'vehicle_static_power', power)
        else:
            cpu_freq = float(self.config.get('vehicle_cpu_freq', cpu_freq))
            power = float(self.config.get('vehicle_static_power', power))

        requirement = float(task.get('computation_requirement', 1500.0)) * 1e6  # cycles
        processing_time = requirement / max(cpu_freq, 1e6)
        processing_time = float(np.clip(processing_time, 0.03, 0.8))
        energy = float(power) * processing_time
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + energy
        return processing_time, energy

    def _estimate_transmission(self, data_size_bytes: float, distance: float, link: str) -> Tuple[float, float]:
        """
        估计上传耗时与能耗
        
        🔧 修复v2：使用固定的base_rate（基于实际硬件测量）
        """
        # 基础速率（bit/s）- 这些值是基于实际网络环境校准的
        if link == 'uav':
            base_rate = 45e6  # 45 Mbps - UAV链路（受限于移动性和功率）
            power_w = 0.12
        else:  # RSU
            base_rate = 80e6  # 80 Mbps - RSU链路（更稳定的固定链路）
            power_w = 0.18

        # 考虑距离衰减
        attenuation = 1.0 + max(0.0, distance) / 800.0
        rate = base_rate / attenuation
        delay = (float(data_size_bytes) * 8.0) / max(rate, 1e6)
        delay = float(np.clip(delay, 0.01, 1.2))
        energy = power_w * delay
        return delay, energy

    def _append_active_task(self, task_entry: Dict):
        """灏嗕换鍔¤褰曞姞鍏ユ椿璺冨垪琛?"""
        self.active_tasks.append(task_entry)

    def _cleanup_active_tasks(self):
        """绉婚櫎宸茬粡瀹屾垚鎴栦涪寮冪殑浠诲姟"""
        self.active_tasks = [
            task for task in self.active_tasks
            if not task.get('completed') and not task.get('dropped')
        ]

    def _handle_deadlines(self):
        """妫€鏌ラ槦鍒椾换鍔℃槸鍚﹁秴鏈熷苟涓㈠純"""
        for node_list, node_type in ((self.rsus, 'RSU'), (self.uavs, 'UAV')):
            for idx, node in enumerate(node_list):
                queue = node.get('computation_queue', [])
                if not queue:
                    continue

                remaining = []
                drop_stats = self.stats.setdefault('drop_stats', {
                    'total': 0,
                    'wait_time_sum': 0.0,
                    'queue_sum': 0,
                    'by_type': {},
                    'by_scenario': {},
                    'by_reason': {}
                })
                by_type = drop_stats.setdefault('by_type', {})
                by_scenario = drop_stats.setdefault('by_scenario', {})
                log_interval = self.stats_config.drop_log_interval if getattr(self, 'stats_config', None) else self.config.get('drop_log_interval', 400)
                log_interval = max(1, int(log_interval))
                for task in queue:
                    if self.current_time > task.get('deadline', float('inf')):
                        task['dropped'] = True
                        self.stats['dropped_tasks'] += 1
                        self.stats['dropped_data_bytes'] += float(task.get('data_size_bytes', 0.0))

                        drop_stats['total'] += 1
                        wait_time = max(0.0, self.current_time - task.get('queued_at', task.get('arrival_time', self.current_time)))
                        drop_stats['wait_time_sum'] += wait_time
                        drop_stats['queue_sum'] += len(queue)
                        task_type = task.get('task_type', 'unknown')
                        by_type[task_type] = by_type.get(task_type, 0) + 1
                        scenario_name = task.get('app_scenario', 'unknown')
                        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1

                        if drop_stats['total'] % log_interval == 0:
                            avg_wait = drop_stats['wait_time_sum'] / max(1, drop_stats['total'])
                            avg_queue = drop_stats['queue_sum'] / max(1, drop_stats['total'])
                            print(
                                f"鈿狅笍 Dropped tasks: {drop_stats['total']} "
                                f"(avg wait {avg_wait:.2f}s, avg queue {avg_queue:.1f}) "
                                f"latest type {task_type}, scenario {scenario_name}"
                            )
                        continue
                    remaining.append(task)
                node['computation_queue'] = remaining

    def _store_in_vehicle_cache(self, vehicle: Dict, content_id: str, size_mb: float,
                                cache_controller: Optional[Any] = None):
        """灏嗗唴瀹规帹閫佸埌杞﹁浇缂撳瓨锛屼娇鐢ㄧ畝鍗昄RU娣樻卑"""
        if size_mb <= 0.0:
            return
        capacity = float(vehicle.get('device_cache_capacity', 32.0))
        if size_mb > capacity:
            return
        cache = vehicle.setdefault('device_cache', {})
        total_used = sum(float(meta.get('size', 0.0) or 0.0) for meta in cache.values())
        if total_used + size_mb > capacity:
            # LRU娣樻卑
            ordered = sorted(cache.items(), key=lambda item: item[1].get('timestamp', 0.0))
            for cid, meta in ordered:
                removed_size = float(meta.get('size', 0.0) or 0.0)
                cache.pop(cid, None)
                total_used -= removed_size
                if cache_controller:
                    cache_controller.cache_stats['evicted_items'] += 1
                if total_used + size_mb <= capacity:
                    break
        if total_used + size_mb > capacity:
            return
        cache[content_id] = {
            'size': size_mb,
            'timestamp': self.current_time,
            'source': 'rsu_push'
        }
        if cache_controller:
            cache_controller.cache_stats['collaborative_writes'] += 1

    def _store_in_neighbor_rsu_cache(self, neighbor: Dict, content_id: str, size_mb: float,
                                     content_meta: Dict, cache_controller: Optional[Any]):
        """灏濊瘯灏嗗唴瀹规帹閫佸埌閭昏繎RSU"""
        if size_mb <= 0.0:
            return
        cache = neighbor.setdefault('cache', {})
        if content_id in cache:
            return
        capacity = neighbor.get('cache_capacity', 1000.0)
        available = self._calculate_available_cache_capacity(cache, capacity)
        cache_snapshot = dict(cache)
        should_store = available >= size_mb
        evictions: List[str] = []
        reason = 'RSU_push_neighbor'
        if cache_controller is not None:
            should_store, reason, evictions = cache_controller.should_cache_content(
                content_id, size_mb, available, cache_snapshot, capacity
            )
        if not should_store:
            return
        for cid in evictions:
            removed = cache.pop(cid, None)
            if removed:
                available += float(removed.get('size', 0.0) or 0.0)
                if cache_controller:
                    cache_controller.cache_stats['evicted_items'] += 1
        if available < size_mb:
            return
        cache[content_id] = {
            'size': size_mb,
            'timestamp': self.current_time,
            'reason': reason,
            'source': content_meta.get('source', 'rsu_hit')
        }
        if cache_controller:
            cache_controller.cache_stats['collaborative_writes'] += 1

    def _propagate_cache_after_hit(self, content_id: str, rsu_node: Dict, agents_actions: Optional[Dict]):
        """RSU鍛戒腑鍚庡悜杞﹁締鍜岄偦杩慠SU鎺ㄩ€佸唴瀹?"""
        cache_meta = rsu_node.get('cache', {}).get(content_id)
        if not cache_meta:
            return
        size_mb = float(cache_meta.get('size', 0.0) or self._get_realistic_content_size(content_id))
        cache_controller = None
        if agents_actions:
            cache_controller = agents_actions.get('cache_controller')

        # 仅在RSU之间传播缓存
        coverage = rsu_node.get('coverage_radius', 300.0)
        if getattr(self, 'spatial_index', None):
            neighbor_candidates = self.spatial_index.query_rsus_within_radius(rsu_node['position'], coverage * 1.2)
            for _, neighbor, _ in neighbor_candidates:
                if neighbor is rsu_node:
                    continue
                self._store_in_neighbor_rsu_cache(neighbor, content_id, size_mb, cache_meta, cache_controller)
        else:
            for neighbor in self.rsus:
                if neighbor is rsu_node:
                    continue
                distance = self.calculate_distance(neighbor['position'], rsu_node['position'])
                if distance <= coverage * 1.2:
                    self._store_in_neighbor_rsu_cache(neighbor, content_id, size_mb, cache_meta, cache_controller)

    def _dispatch_task(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict):
        """鏍规嵁鍔ㄤ綔鍒嗛厤浠诲姟"""
        cache_controller = None
        if isinstance(actions, dict):
            cache_controller = actions.get('cache_controller')
        if cache_controller is None:
            cache_controller = getattr(self, 'adaptive_cache_controller', None)

        content_id = task.get('content_id')
        # 车辆端不再维护本地缓存，直接根据策略决定卸载或本地计算
        forced_mode = getattr(self, 'forced_offload_mode', '')
        if forced_mode != 'remote_only':
            if self._try_serve_from_vehicle_cache(vehicle, task, step_summary, cache_controller):
                return
        if forced_mode == 'local_only':
            self._handle_local_processing(vehicle, task, step_summary)
            return

        # 🔧 修复：remote_only模式的正确处理
        if forced_mode == 'remote_only':
            rsu_available = len(self.rsus) > 0
            uav_available = len(self.uavs) > 0
            
            assigned = False
            if rsu_available or uav_available:
                target = self._choose_offload_target(actions, rsu_available, uav_available)
                if target == 'rsu' and rsu_available:
                    assigned = self._assign_to_rsu(vehicle, task, actions, step_summary)
                elif target == 'uav' and uav_available:
                    assigned = self._assign_to_uav(vehicle, task, actions, step_summary)
            
            if not assigned:
                # remote_only模式下卸载失败，丢弃任务（不fallback到本地处理）
                self._record_forced_drop(vehicle, task, step_summary, reason='remote_only_offload_failed')
            return

        # 正常模式：尝试卸载，失败则本地处理
        rsu_available = len(self.rsus) > 0
        uav_available = len(self.uavs) > 0
        target = self._choose_offload_target(actions, rsu_available, uav_available)

        assigned = False
        if target == 'rsu' and rsu_available:
            assigned = self._assign_to_rsu(vehicle, task, actions, step_summary)
        elif target == 'uav' and uav_available:
            assigned = self._assign_to_uav(vehicle, task, actions, step_summary)

        if not assigned:
            self._handle_local_processing(vehicle, task, step_summary)

    def _assign_to_rsu(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict) -> bool:
        """???RSU??????????????????"""
        if not self.rsus:
            return False

        vehicle_pos = np.asarray(vehicle.get('position', [0.0, 0.0]), dtype=float)
        candidates = []
        if getattr(self, 'spatial_index', None):
            max_radius = self.spatial_index.rsu_max_radius or max(
                (float(rsu.get('coverage_radius', self.coverage_radius)) for rsu in self.rsus),
                default=self.coverage_radius,
            )
            candidates = self.spatial_index.query_rsus_within_radius(vehicle_pos, max_radius)
            if not candidates:
                nearest = self.spatial_index.find_nearest_rsu(vehicle_pos, return_distance=True)
                if nearest:
                    candidates = [nearest]

        if not candidates:
            candidates = [
                (idx, rsu, self.calculate_distance(vehicle_pos, rsu['position']))
                for idx, rsu in enumerate(self.rsus)
            ]

        filtered = [
            (idx, node, dist)
            for idx, node, dist in candidates
            if dist <= float(node.get('coverage_radius', self.coverage_radius))
        ]
        if not filtered:
            return False

        candidate_indices = np.array([idx for idx, _, _ in filtered], dtype=int)
        distances = np.array([dist for _, _, dist in filtered], dtype=float)

        probs = np.ones_like(distances)
        rsu_pref = actions.get('rsu_selection_probs')
        if isinstance(rsu_pref, (list, tuple, np.ndarray)) and len(rsu_pref) == len(self.rsus):
            probs *= np.array([max(0.0, float(rsu_pref[idx])) for idx in candidate_indices], dtype=float)

        guidance = actions.get('rl_guidance') or {}
        if isinstance(guidance, dict):
            rsu_prior = np.array(guidance.get('rsu_prior', []), dtype=float)
            if rsu_prior.size >= len(self.rsus):
                probs *= np.clip(rsu_prior[candidate_indices], 1e-4, None)
            cache_focus = guidance.get('cache_focus')
            if isinstance(cache_focus, (list, tuple)) and len(cache_focus) >= 2:
                cache_weight = float(np.clip(cache_focus[1], 0.0, 1.0))
                probs = np.power(probs, 0.8 + 0.4 * cache_weight)
            distance_focus = guidance.get('distance_focus')
            if isinstance(distance_focus, (list, tuple)) and len(distance_focus) >= 2:
                distance_weight = float(np.clip(distance_focus[1], 0.0, 1.0))
                probs = np.power(probs, 0.8 + 0.4 * distance_weight)

        weights = probs
        if weights.sum() <= 0:
            weights = np.ones_like(weights)

        weights = weights / weights.sum()
        ordered_choices = list(np.random.choice(
            np.arange(len(candidate_indices)),
            size=len(candidate_indices),
            replace=False,
            p=weights
        ))
        attempted = False
        for choice in ordered_choices:
            rsu_idx = int(candidate_indices[choice])
            distance = float(distances[choice])
            node = self.rsus[rsu_idx]
            if not self._is_node_admissible(node, 'RSU'):
                continue
            attempted = True
            success = self._handle_remote_assignment(vehicle, task, node, 'RSU', rsu_idx, distance, actions, step_summary)
            if success:
                step_summary['remote_tasks'] += 1
                return True
        reason = 'rsu_overloaded' if not attempted else 'assignment_failed'
        self._record_offload_rejection('RSU', reason)
        step_summary['remote_refusals'] = step_summary.get('remote_refusals', 0) + 1
        return False


    def _assign_to_uav(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict) -> bool:
        """???UAV????????????????"""
        if not self.uavs:
            return False

        vehicle_pos = np.asarray(vehicle.get('position', [0.0, 0.0]), dtype=float)
        candidates = []
        if getattr(self, 'spatial_index', None):
            max_radius = self.spatial_index.uav_max_radius or max(
                (float(uav.get('coverage_radius', 350.0)) for uav in self.uavs),
                default=350.0,
            )
            candidates = self.spatial_index.query_uavs_within_radius(vehicle_pos, max_radius)
            if not candidates:
                nearest = self.spatial_index.find_nearest_uav(vehicle_pos, return_distance=True)
                if nearest:
                    candidates = [nearest]

        if not candidates:
            candidates = [
                (idx, uav, self.calculate_distance(vehicle_pos, uav['position']))
                for idx, uav in enumerate(self.uavs)
            ]

        filtered = [
            (idx, node, dist)
            for idx, node, dist in candidates
            if dist <= float(node.get('coverage_radius', 350.0))
        ]
        if not filtered:
            return False

        candidate_indices = np.array([idx for idx, _, _ in filtered], dtype=int)
        distances = np.array([dist for _, _, dist in filtered], dtype=float)

        probs = np.ones_like(distances)
        uav_pref = actions.get('uav_selection_probs')
        if isinstance(uav_pref, (list, tuple, np.ndarray)) and len(uav_pref) == len(self.uavs):
            probs *= np.array([max(0.0, float(uav_pref[idx])) for idx in candidate_indices], dtype=float)

        guidance = actions.get('rl_guidance') or {}
        if isinstance(guidance, dict):
            uav_prior = np.array(guidance.get('uav_prior', []), dtype=float)
            if uav_prior.size >= len(self.uavs):
                probs *= np.clip(uav_prior[candidate_indices], 1e-4, None)
            distance_focus = guidance.get('distance_focus')
            if isinstance(distance_focus, (list, tuple)) and len(distance_focus) >= 3:
                distance_weight = float(np.clip(distance_focus[2], 0.0, 1.0))
                probs = np.power(probs, 0.8 + 0.4 * distance_weight)

        weights = probs
        if weights.sum() <= 0:
            weights = np.ones_like(weights)

        weights = weights / weights.sum()
        ordered_choices = list(np.random.choice(
            np.arange(len(candidate_indices)),
            size=len(candidate_indices),
            replace=False,
            p=weights
        ))
        attempted = False
        for choice in ordered_choices:
            uav_idx = int(candidate_indices[choice])
            distance = float(distances[choice])
            node = self.uavs[uav_idx]
            if not self._is_node_admissible(node, 'UAV'):
                continue
            attempted = True
            success = self._handle_remote_assignment(vehicle, task, node, 'UAV', uav_idx, distance, actions, step_summary)
            if success:
                step_summary['remote_tasks'] += 1
                return True
        reason = 'uav_overloaded' if not attempted else 'assignment_failed'
        self._record_offload_rejection('UAV', reason)
        step_summary['remote_refusals'] = step_summary.get('remote_refusals', 0) + 1
        return False


    def _handle_remote_assignment(
        self,
        vehicle: Dict,
        task: Dict,
        node: Dict,
        node_type: str,
        node_idx: int,
        distance: float,
        actions: Dict,
        step_summary: Dict
    ) -> bool:
        """
        执行远程卸载：缓存判定、建立队列并记录统计
        
        处理任务到远程节点（RSU或UAV）的卸载过程：
        1. 检查缓存命中
        2. 计算上传延迟和能耗
        3. 估算任务工作量
        4. 将任务加入节点队列
        
        Args:
            vehicle: 车辆字典
            task: 任务字典
            node: 目标节点字典
            node_type: 节点类型（'RSU'或'UAV'）
            node_idx: 节点索引
            distance: 车辆到节点的距离
            actions: 智能体动作字典
            step_summary: 步骤统计摘要
            
        Returns:
            True表示成功卸载，False表示失败
            
        Execute remote offloading with cache checking and queue management.
        """
        actions = actions or {}
        self._reset_mm1_step_buffers()
        cache_hit = False

        # 检查缓存命中
        if node_type == 'RSU':
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='RSU')
        else:
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='UAV')

        if cache_hit:
            # 缓存命中：快速完成
            # Cache hit: quick completion
            delay = max(0.02, 0.2 * self.time_slot)
            power = 18.0 if node_type == 'RSU' else 12.0
            energy = power * delay * 0.1
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self._accumulate_delay('delay_cache', delay)
            self._accumulate_energy('energy_cache', energy)
            self.stats['energy_downlink'] = self.stats.get('energy_downlink', 0.0) + energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + energy
            return True

        # 缓存未命中：计算上传开销
        # Cache miss: calculate upload overhead
        upload_delay, upload_energy = self._estimate_transmission(task.get('data_size_bytes', 1e6), distance, node_type.lower())
        self._accumulate_delay('delay_uplink', upload_delay)
        self.stats['energy_uplink'] += upload_energy
        self._accumulate_energy('energy_transmit_uplink', upload_energy)
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + upload_energy

        # 估算远程工作量并创建任务条目
        # Estimate remote workload and create task entry
        work_units = self._estimate_remote_work_units(task, node_type)
        task_entry = {
            'id': task['id'],
            'vehicle_id': task['vehicle_id'],
            'arrival_time': self.current_time + upload_delay,
            'deadline': task['deadline'],
            'data_size': task.get('data_size', 1.0),
            'data_size_bytes': task.get('data_size_bytes', 1e6),
            'content_id': task.get('content_id'),
            'computation_requirement': task.get('computation_requirement', 1500.0),
            'work_remaining': work_units,
            'queued_at': self.current_time,
            'node_type': node_type,
            'node_idx': node_idx,
            'upload_delay': upload_delay,
            'priority': task.get('priority', 0.5),
            'task_type': task.get('task_type'),
            'app_scenario': task.get('app_scenario'),
            'deadline_relax_factor': task.get('deadline_relax_factor', 1.0)
        }

        queue = node.setdefault('computation_queue', [])
        queue.append(task_entry)
        self._enforce_queue_capacity(node, node_type, step_summary)
        self._apply_queue_scheduling(node, node_type)
        self._append_active_task(task_entry)
        self._record_mm1_arrival(node_type, node_idx)
        return True

    def _apply_queue_scheduling(self, node: Dict, node_type: str) -> None:
        """??????????????????"""
        if node_type not in ('RSU', 'UAV'):
            return
        queue = node.get('computation_queue')
        if not isinstance(queue, list) or len(queue) <= 1:
            return
        params = getattr(self, '_scheduling_params', None)
        if not params:
            return
        priority_bias = float(np.clip(params.get('priority_bias', 0.5), 0.0, 1.0))
        deadline_bias = float(np.clip(params.get('deadline_bias', 0.5), 0.0, 1.0))
        window = int(max(1, params.get('reorder_window', 1)))
        window = min(window, len(queue))
        if window <= 1:
            return
        current_time = getattr(self, 'current_time', 0.0)
        scored: List[Tuple[float, float, int]] = []
        for idx, task in enumerate(queue):
            try:
                priority_raw = float(task.get('priority', 4.0))
            except (TypeError, ValueError):
                priority_raw = 4.0
            priority_score = 1.0 - float(np.clip((priority_raw - 1.0) / 3.0, 0.0, 1.0))
            deadline_value = float(task.get('deadline', current_time))
            slack = deadline_value - current_time
            slack_norm = float(np.clip(slack / max(self.time_slot * 8.0, 1e-6), 0.0, 1.0))
            deadline_score = 1.0 - slack_norm
            wait = current_time - float(task.get('queued_at', current_time))
            wait_norm = float(np.clip(wait / max(self.time_slot * 8.0, 1e-6), 0.0, 1.0))
            weight_delay = priority_bias
            weight_deadline = deadline_bias
            weight_wait = max(0.0, 1.0 - (weight_delay + weight_deadline))
            total = weight_delay + weight_deadline + weight_wait
            if total <= 0.0:
                weight_delay, weight_deadline, weight_wait = 0.4, 0.4, 0.2
                total = 1.0
            weight_delay /= total
            weight_deadline /= total
            weight_wait /= total
            score = (weight_delay * priority_score) + (weight_deadline * deadline_score) + (weight_wait * wait_norm)
            scored.append((score, -wait, idx))
        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        selected_indices = [entry[2] for entry in scored[:window]]
        selected_set = set(selected_indices)
        reordered = [queue[idx] for idx in selected_indices]
        remainder = [queue[i] for i in range(len(queue)) if i not in selected_set]
        queue[:] = reordered + remainder

    def _handle_local_processing(self, vehicle: Dict, task: Dict, step_summary: Dict):
        """
        本地处理任务
        
        在车辆本地设备上处理任务，计算延迟和能耗。
        
        Args:
            vehicle: 车辆字典
            task: 任务字典
            step_summary: 步骤统计摘要
            
        Handle task processing on local vehicle device.
        """
        if not getattr(self, 'allow_local_processing', True):
            self._record_forced_drop(vehicle, task, step_summary, reason='local_processing_disabled')
            return

        processing_delay, energy = self._estimate_local_processing(task, vehicle)
        self.stats['processed_tasks'] += 1
        self.stats['completed_tasks'] += 1
        self._accumulate_delay('delay_processing', processing_delay)
        self._accumulate_energy('energy_compute', energy)
        cpu_freq = float(vehicle.get('cpu_freq', self.vehicle_cpu_freq))
        cycles_consumed = processing_delay * cpu_freq
        vehicle['local_cycle_used'] = vehicle.get('local_cycle_used', 0.0) + cycles_consumed
        available_cycles = max(1e-6, cpu_freq * self.time_slot)
        vehicle['compute_usage'] = float(np.clip(vehicle['local_cycle_used'] / available_cycles, 0.0, 1.0))
        step_summary['local_tasks'] += 1

    def _record_forced_drop(self, vehicle: Dict, task: Dict, step_summary: Dict, reason: str = 'forced_drop') -> None:
        """记录因策略约束导致的任务丢弃事件"""
        self.stats['dropped_tasks'] = self.stats.get('dropped_tasks', 0) + 1
        self.stats['dropped_data_bytes'] = self.stats.get('dropped_data_bytes', 0.0) + float(task.get('data_size_bytes', 0.0))

        drop_stats = self.stats.setdefault('drop_stats', {
            'total': 0,
            'wait_time_sum': 0.0,
            'queue_sum': 0,
            'by_type': {},
            'by_scenario': {},
            'by_reason': {}
        })
        drop_stats['total'] = drop_stats.get('total', 0) + 1
        task_type = task.get('task_type', 'unknown')
        scenario_name = task.get('app_scenario', 'unknown')
        by_type = drop_stats.setdefault('by_type', {})
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_scenario = drop_stats.setdefault('by_scenario', {})
        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1
        by_reason = drop_stats.setdefault('by_reason', {})
        by_reason[reason] = by_reason.get(reason, 0) + 1

        step_summary['dropped_tasks'] = step_summary.get('dropped_tasks', 0) + 1
        forced_key = 'forced_drops'
        step_summary[forced_key] = step_summary.get(forced_key, 0) + 1
        step_summary['last_forced_drop_reason'] = reason

    
    def check_adaptive_migration(self, agents_actions: Dict = None):
        """馃幆 澶氱淮搴︽櫤鑳借縼绉绘鏌?(闃堝€艰Е鍙?璐熻浇宸Е鍙?璺熼殢杩佺Щ)"""
        if not agents_actions or 'migration_controller' not in agents_actions:
            return
        
        migration_controller = agents_actions['migration_controller']
        coordinator = getattr(self, 'strategy_coordinator', None)
        joint_params = agents_actions.get('joint_strategy_params', {}) if isinstance(agents_actions, dict) else {}
        
        hotspot_map: Dict[str, float] = {}
        collaborative_system = getattr(self, 'collaborative_cache', None)
        if collaborative_system is not None and hasattr(collaborative_system, 'get_hotspot_intensity'):
            try:
                hotspot_map = collaborative_system.get_hotspot_intensity()
            except Exception:
                hotspot_map = {}
        
        # 馃攳 鏀堕泦鎵€鏈夎妭鐐圭姸鎬佺敤浜庨偦灞呮瘮杈?
        all_node_states = {}
        
        # RSU鐘舵€佹敹闆?
        for i, rsu in enumerate(self.rsus):
            queue = rsu.get('computation_queue', [])
            queue_len = len(queue)
            cache_capacity = rsu.get('cache_capacity', 1000.0)
            available_cache = self._calculate_available_cache_capacity(rsu.get('cache', {}), cache_capacity)
            storage_load = 0.0 if cache_capacity <= 0 else 1.0 - (available_cache / max(1.0, cache_capacity))
            total_data = sum(task.get('data_size', 1.0) for task in queue)
            bandwidth_capacity = rsu.get('bandwidth_capacity', 50.0)
            bandwidth_load = float(np.clip(total_data / max(1.0, bandwidth_capacity), 0.0, 0.99))
            cpu_load = float(np.clip(queue_len / 10.0, 0.0, 0.99))

            all_node_states[f'rsu_{i}'] = {
                'cpu_load': cpu_load,
                'bandwidth_load': bandwidth_load,
                'storage_load': float(np.clip(storage_load, 0.0, 0.99)),
                'load_factor': self._calculate_enhanced_load_factor(rsu, 'RSU'),
                'battery_level': 1.0,
                'node_type': 'RSU',
                'queue_length': queue_len,
                'cache_capacity': cache_capacity,
                'cache_available': available_cache,
                'hotspot_intensity': float(np.clip(hotspot_map.get(f'RSU_{i}', 0.0), 0.0, 1.0))
            }

        # UAV鐘舵€佹敹闆?
        for i, uav in enumerate(self.uavs):
            queue = uav.get('computation_queue', [])
            queue_len = len(queue)
            cache_capacity = uav.get('cache_capacity', 200.0)
            available_cache = self._calculate_available_cache_capacity(uav.get('cache', {}), cache_capacity)
            storage_load = 0.0 if cache_capacity <= 0 else 1.0 - (available_cache / max(1.0, cache_capacity))
            total_data = sum(task.get('data_size', 1.0) for task in queue)
            bandwidth_capacity = uav.get('bandwidth_capacity', 15.0)
            bandwidth_load = float(np.clip(total_data / max(1.0, bandwidth_capacity), 0.0, 0.99))
            cpu_load = float(np.clip(queue_len / 12.0, 0.0, 0.99))

            all_node_states[f'uav_{i}'] = {
                'cpu_load': cpu_load,
                'bandwidth_load': bandwidth_load,
                'storage_load': float(np.clip(storage_load, 0.0, 0.99)),
                'load_factor': self._calculate_enhanced_load_factor(uav, 'UAV'),
                'battery_level': uav.get('battery_level', 1.0),
                'node_type': 'UAV',
                'queue_length': queue_len,
                'cache_capacity': cache_capacity,
                'cache_available': available_cache,
                'hotspot_intensity': 0.0
            }
        
        # 馃彚 RSU杩佺Щ妫€鏌?(闃堝€?璐熻浇宸Е鍙?
        for i, rsu in enumerate(self.rsus):
            node_id = f'rsu_{i}'
            current_state = all_node_states[node_id]
            
            # 更新负载历史
            migration_controller.update_node_load(node_id, current_state['load_factor'])
            
            # 🔄 多维度迁移触发检查
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"馃幆 {node_id} 瑙﹀彂杩佺Щ: {reason} (绱ф€ュ害:{urgency:.3f})")
                if coordinator is not None:
                    try:
                        coordinator.notify_migration_triggered(node_id, reason, urgency, current_state)
                    except Exception as exc:
                        print(f"⚠️ 联合策略协调器记录RSU迁移异常: {exc}")
                
                # 鎵цRSU闂磋縼绉?
                result = self.execute_rsu_migration(i, urgency, coordinator=coordinator, joint_params=joint_params)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
                if coordinator is not None:
                    try:
                        coordinator.notify_migration_result(
                            node_id,
                            bool(result.get('success')),
                            {'type': 'rsu', 'metadata': result}
                        )
                    except Exception as exc:
                        print(f"⚠️ 联合策略协调器记录RSU迁移结果异常: {exc}")
        
        # 馃殎 UAV杩佺Щ妫€鏌?
        for i, uav in enumerate(self.uavs):
            node_id = f'uav_{i}'
            current_state = all_node_states[node_id]
            
            # 更新负载历史
            migration_controller.update_node_load(node_id, current_state['load_factor'], current_state['battery_level'])
            
            # 🔄 多维度迁移触发检查
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"馃幆 {node_id} 瑙﹀彂杩佺Щ: {reason} (绱ф€ュ害:{urgency:.3f})")
                if coordinator is not None:
                    try:
                        coordinator.notify_migration_triggered(node_id, reason, urgency, current_state)
                    except Exception as exc:
                        print(f"⚠️ 联合策略协调器记录UAV迁移异常: {exc}")
                
                # UAV杩佺Щ鍒癛SU
                result = self.execute_uav_migration(i, urgency, coordinator=coordinator, joint_params=joint_params)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
                if coordinator is not None:
                    try:
                        coordinator.notify_migration_result(
                            node_id,
                            bool(result.get('success')),
                            {'type': 'uav', 'metadata': result}
                        )
                    except Exception as exc:
                        print(f"⚠️ 联合策略协调器记录UAV迁移结果异常: {exc}")
        
        # 馃殫 杞﹁締璺熼殢杩佺Щ妫€鏌?
        self._check_vehicle_handover_migration(migration_controller)
    
    def _check_vehicle_handover_migration(self, migration_controller):
        """车辆跟随迁移：当车辆远离当前边缘节点覆盖时触发迁移。"""
        handover_count = 0

        for task in list(self.active_tasks):
            if task.get('node_type') not in ('RSU', 'UAV'):
                continue

            try:
                vehicle = next(v for v in self.vehicles if v['id'] == task['vehicle_id'])
            except StopIteration:
                continue

            origin_node_type = task['node_type']
            origin_node_idx = task.get('node_idx')
            if origin_node_type == 'RSU' and origin_node_idx is not None and 0 <= origin_node_idx < len(self.rsus):
                current_node = self.rsus[origin_node_idx]
            elif origin_node_type == 'UAV' and origin_node_idx is not None and 0 <= origin_node_idx < len(self.uavs):
                current_node = self.uavs[origin_node_idx]
            else:
                continue

            current_pos = np.array(vehicle.get('position', [0.0, 0.0, 0.0]))
            distance_to_current = self.calculate_distance(current_pos, current_node['position'])
            coverage_radius = current_node.get('coverage_radius', 500.0)

            vehicle_speed = float(np.linalg.norm(vehicle.get('velocity', [0.0, 0.0, 0.0])))
            speed_factor = max(0.70, 1.0 - vehicle_speed / 200.0)
            trigger_threshold = coverage_radius * speed_factor

            if distance_to_current <= trigger_threshold:
                continue

            current_queue_before = len(current_node.get('computation_queue', []))
            current_load = float(current_node.get('cpu_usage', 0.5))
            current_score = distance_to_current + current_queue_before * 30 + current_load * 200

            best_new_node = None
            best_node_idx = None
            best_node_type = None
            best_metric = float('inf')

            for idx, rsu in enumerate(self.rsus):
                dist = self.calculate_distance(current_pos, rsu['position'])
                if dist > rsu.get('coverage_radius', 500.0):
                    continue
                queue_len = len(rsu.get('computation_queue', []))
                cpu_load = float(rsu.get('cpu_usage', 0.5))
                score = dist + queue_len * 30 + cpu_load * 200
                if score < best_metric:
                    best_metric = score
                    best_new_node = rsu
                    best_node_idx = idx
                    best_node_type = 'RSU'

            if best_new_node is None or best_metric > current_score * 0.7:
                for idx, uav in enumerate(self.uavs):
                    dist = self.calculate_distance(current_pos, uav['position'])
                    if dist > uav.get('coverage_radius', 350.0):
                        continue
                    queue_len = len(uav.get('computation_queue', []))
                    cpu_load = float(uav.get('cpu_usage', 0.5))
                    score = dist + queue_len * 40 + cpu_load * 220
                    if score < best_metric:
                        best_metric = score
                        best_new_node = uav
                        best_node_idx = idx
                        best_node_type = 'UAV'

            if not best_new_node:
                continue

            should_switch = (best_node_type != task['node_type'] or best_node_idx != origin_node_idx) and best_metric < current_score * 0.7
            if not should_switch:
                continue

            origin_queue_after = current_queue_before
            if origin_node_idx is not None:
                if task['node_type'] == 'RSU':
                    origin_queue = self.rsus[origin_node_idx].get('computation_queue', [])
                    filtered = [t for t in origin_queue if t.get('id') != task['id']]
                    self.rsus[origin_node_idx]['computation_queue'] = filtered
                    origin_queue_after = len(filtered)
                elif task['node_type'] == 'UAV':
                    origin_queue = self.uavs[origin_node_idx].get('computation_queue', [])
                    filtered = [t for t in origin_queue if t.get('id') != task['id']]
                    self.uavs[origin_node_idx]['computation_queue'] = filtered
                    origin_queue_after = len(filtered)

            best_new_node.setdefault('computation_queue', [])
            target_queue_before = len(best_new_node['computation_queue'])
            migrated_task = {
                'id': task['id'],
                'vehicle_id': task['vehicle_id'],
                'arrival_time': task['arrival_time'],
                'deadline': task['deadline'],
                'data_size': task.get('data_size', 2.0),
                'computation_requirement': task.get('computation_requirement', 1000),
                'content_id': task.get('content_id'),
                'compute_time_needed': task.get('compute_time_needed', 1.0),
                'work_remaining': task.get('work_remaining', 0.5),
                'cache_hit': task.get('cache_hit', False),
                'queued_at': self.current_time,
                'migrated_from': f"{task['node_type']}_{task.get('node_idx')}",
                'task_type': task.get('task_type'),
                'app_scenario': task.get('app_scenario'),
                'deadline_relax_factor': task.get('deadline_relax_factor', 1.0)
            }
            best_new_node['computation_queue'].append(migrated_task)
            self._apply_queue_scheduling(best_new_node, best_node_type)
            target_queue_after = len(best_new_node['computation_queue'])

            handover_count += 1

            print(
                f"[VehicleMigration] handover #{handover_count}: vehicle {task['vehicle_id']} task {task['id']} "
                f"{origin_node_type}_{origin_node_idx} -> {best_node_type}_{best_node_idx}"
            )
            print(
                f"   Trigger: distance {distance_to_current:.1f}m > threshold {trigger_threshold:.1f}m "
                f"(speed {vehicle_speed:.1f} m/s)"
            )
            improvement = 0.0
            if current_score > 1e-6:
                improvement = (1 - best_metric / current_score) * 100.0
            print(
                f"   Score: {current_score:.1f} -> {best_metric:.1f} (improvement {improvement:.1f}%)"
            )
            print(
                f"   Queue trend: {origin_node_type}_{origin_node_idx}: {current_queue_before} -> {origin_queue_after}, "
                f"{best_node_type}_{best_node_idx}: {target_queue_before} -> {target_queue_after}"
            )

            task['node_type'] = best_node_type
            task['node_idx'] = best_node_idx

            self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
            self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
            self.stats['handover_migrations'] = self.stats.get('handover_migrations', 0) + 1
            migration_controller.record_migration_result(True, cost=5.0, delay_saved=0.3)

        if handover_count > 0:
            print(f"[Migration] Executed {handover_count} vehicle-following migrations.")

    def run_simulation_step(self, step: int, actions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        执行单个仿真步，返回截至当前的累计统计数据
        
        这是仿真的核心方法，执行一个时间步的所有操作：
        1. 更新车辆位置
        2. 生成并分配新任务
        3. 执行智能迁移策略
        4. 处理节点队列中的任务
        5. 检查超时并清理
        
        Args:
            step: 当前仿真步数
            actions: 智能体的动作字典（可选），包含缓存控制器、迁移控制器等
            
        Returns:
            包含累计统计数据的字典
            
        Execute a single simulation step and return cumulative statistics.
        """
        actions = actions or {}
        self._update_scheduling_params(actions.get('scheduling_params'))
        self._prepare_step_usage_counters()
        if self._central_resource_enabled and hasattr(self, 'resource_pool'):
            try:
                self.execute_phase2_scheduling()
            except Exception as exc:
                logging.debug("Phase-2 scheduling execution failed: %s", exc)

        # 推进仿真时间
        advance_simulation_time()
        self.current_step += 1
        self.current_time = get_simulation_time()

        # 当前步骤的统计摘要
        step_summary = {
            'generated_tasks': 0,  # 本步生成的任务数
            'local_tasks': 0,  # 本地处理的任务数
            'remote_tasks': 0,  # 远程卸载的任务数
            'local_cache_hits': 0,  # 本地缓存命中次数
            'queue_overflow_drops': 0  # 本步因队列溢出的丢弃
        }

        # 1. 更新车辆位置
        # Update vehicle positions based on movement model
        self._update_vehicle_positions()

        # 2. 生成任务并（可选）两阶段规划后分配
        # Generate new tasks for each vehicle first (batch), then optionally plan
        tasks_batch: List[Tuple[int, Dict, Dict]] = []
        for vidx, vehicle in enumerate(self.vehicles):
            arrivals = self._sample_arrivals()
            if arrivals <= 0:
                continue
            vehicle_id = vehicle['id']
            for _ in range(arrivals):
                task = self.generate_task(vehicle_id)
                step_summary['generated_tasks'] += 1
                self.stats['total_tasks'] += 1
                self.stats['generated_data_bytes'] += float(task.get('data_size_bytes', 0.0))
                tasks_batch.append((vidx, vehicle, task))

        # Stage-1 planning (coarse assignment + resource estimation)
        # If STAGE1_ALG is present (Dual-stage controller mode), we skip heuristic
        # planning here because Stage-1 decisions are embedded in the action vector.
        plan_map: Dict[str, PlanEntry] = {}
        if self._two_stage_enabled and tasks_batch and (os.environ.get('STAGE1_ALG', '').strip() == ''):
            if self._two_stage_planner is None:
                self._two_stage_planner = TwoStagePlanner()
            plan_map = self._two_stage_planner.build_plan(self, tasks_batch)

        # Dispatch tasks (use plan if available)
        for vidx, vehicle, task in tasks_batch:
            plan_entry = plan_map.get(task.get('id') or task.get('task_id', '')) if plan_map else None
            if plan_entry is not None:
                self._dispatch_task_with_plan(vehicle, task, plan_entry, actions, step_summary)
            else:
                self._dispatch_task(vehicle, task, actions, step_summary)

        # 3. 智能迁移策略
        # Execute intelligent migration strategy
        if actions:
            self.check_adaptive_migration(actions)

        # 4. 处理队列中的任务
        # Process tasks in node queues
        self._process_node_queues()

        # 5. 妫€鏌ヨ秴鏃跺苟娓呯悊
        self._handle_deadlines()
        self._cleanup_active_tasks()

        # 姹囨€讳俊鎭?
        step_summary.update({
            'current_time': self.current_time,
            'rsu_queue_lengths': [len(rsu.get('computation_queue', [])) for rsu in self.rsus],
            'uav_queue_lengths': [len(uav.get('computation_queue', [])) for uav in self.uavs],
            'active_tasks': len(self.active_tasks)
        })

        stability_metrics = self._monitor_queue_stability()
        step_summary.update(stability_metrics)
        step_summary.update(self._summarize_task_types())
        mm1_predictions = self._finalize_mm1_step(self.current_step)
        step_summary['mm1_predictions'] = mm1_predictions

        if self._central_resource_enabled:
            self._update_central_scheduler(step_summary)

        cumulative_stats = dict(self.stats)
        cumulative_stats.update(step_summary)
        return cumulative_stats

    def _dispatch_task_with_plan(self, vehicle: Dict, task: Dict, plan: PlanEntry,
                                 actions: Dict, step_summary: Dict):
        """Dispatch a task following the Stage-1 plan entry.

        Falls back to legacy dispatch if the target is not feasible.
        """
        try:
            # Local processing
            if plan.target_type == 'local' or plan.target_idx is None:
                return self._handle_local_processing(vehicle, task, step_summary)

            # Remote: RSU/UAV explicit target
            if plan.target_type == 'rsu':
                idx = int(plan.target_idx)
                if 0 <= idx < len(self.rsus):
                    node = self.rsus[idx]
                    distance = self.calculate_distance(vehicle.get('position', np.zeros(2)), node['position'])
                    ok = self._handle_remote_assignment(vehicle, task, node, 'RSU', idx, distance, actions or {}, step_summary)
                    if ok:
                        step_summary['remote_tasks'] += 1
                        return True
            elif plan.target_type == 'uav':
                idx = int(plan.target_idx)
                if 0 <= idx < len(self.uavs):
                    node = self.uavs[idx]
                    distance = self.calculate_distance(vehicle.get('position', np.zeros(2)), node['position'])
                    ok = self._handle_remote_assignment(vehicle, task, node, 'UAV', idx, distance, actions or {}, step_summary)
                    if ok:
                        step_summary['remote_tasks'] += 1
                        return True
        except Exception:
            # On any failure, fall back to legacy path
            pass

        # Fallback: legacy selection
        return self._dispatch_task(vehicle, task, actions, step_summary)
    
    def execute_rsu_migration(self, source_rsu_idx: int, urgency: float,
                              coordinator: Optional['StrategyCoordinator'] = None,
                              joint_params: Optional[Dict] = None) -> Dict[str, float]:
        """
        执行RSU到RSU的迁移并返回成本/延迟指标
        
        实现RSU间的任务迁移，通过有线回程网络传输任务：
        1. 选择负载最轻的目标RSU
        2. 检查迁移容忍度（避免不必要的迁移）
        3. 根据紧急度确定迁移任务数量
        4. 通过有线网络传输任务
        5. 记录迁移成本和延迟节省
        
        Args:
            source_rsu_idx: 源RSU的索引
            urgency: 迁移紧急度（0.0-1.0）
            
        Returns:
            包含迁移结果的字典：
            - success: 是否成功
            - cost: 迁移成本（能耗+延迟）
            - delay_saved: 节省的延迟
            
        Execute RSU-to-RSU migration via wired backhaul network.
        """
        source_rsu = self.rsus[source_rsu_idx]
        source_queue = source_rsu.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        candidates = []
        for i, rsu in enumerate(self.rsus):
            if i == source_rsu_idx:
                continue
            queue_len = len(rsu.get('computation_queue', []))
            cpu_load = min(0.99, queue_len / 10.0)
            score = queue_len + cpu_load * 10.0
            candidates.append((i, queue_len, cpu_load, score))

        if not candidates:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_idx, target_queue_len, target_cpu_load, _ = min(candidates, key=lambda x: x[3])
        source_queue_len = len(source_queue)
        queue_diff = target_queue_len - source_queue_len

        all_queue_lens = [len(rsu.get('computation_queue', [])) for rsu in self.rsus]
        system_queue_variance = np.var(all_queue_lens)
        if system_queue_variance > 50:
            migration_tolerance = 8
        elif system_queue_variance > 20:
            migration_tolerance = 5
        else:
            migration_tolerance = 3
        if queue_diff > migration_tolerance:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        backoff = 0.0
        if joint_params:
            try:
                backoff = float(joint_params.get('migration_backoff', 0.0) or 0.0)
            except (TypeError, ValueError):
                backoff = 0.0
        backoff = float(np.clip(backoff, 0.0, 1.0))

        migration_ratio = max(0.1, min(0.5, urgency * (1.0 - 0.4 * backoff) + 0.05))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        source_rsu_id = source_rsu['id']
        target_rsu_id = target_rsu['id']
        migrated_tasks = source_queue[:tasks_to_migrate]
        total_data_size = sum(task.get('data_size', 1.0) for task in migrated_tasks)
        if total_data_size <= 0.0:
            total_data_size = tasks_to_migrate * 1.0
        if coordinator is not None and migrated_tasks:
            try:
                coordinator.prepare_prefetch(source_rsu, target_rsu, migrated_tasks, urgency)
            except Exception as exc:
                print(f"⚠️ 迁移前预取协调失败({source_rsu_id}->{target_rsu_id}): {exc}")

        source_rsu['computation_queue'] = source_queue[tasks_to_migrate:]
        target_rsu['computation_queue'].extend(migrated_tasks)

        queue_relief = max(0.0, source_queue_len - len(source_rsu['computation_queue']))
        delay_saved = max(0.0, queue_relief * self.time_slot)
        migration_cost = 0.0
        try:
            from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
            wired_delay = calculate_rsu_to_rsu_delay(total_data_size, source_rsu_id, target_rsu_id)
            wired_energy = calculate_rsu_to_rsu_energy(total_data_size, source_rsu_id, target_rsu_id, wired_delay)
            self.stats['rsu_migration_delay'] = self.stats.get('rsu_migration_delay', 0.0) + wired_delay
            self.stats['rsu_migration_energy'] = self.stats.get('rsu_migration_energy', 0.0) + wired_energy
            self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + total_data_size
            migration_cost = (self.migration_energy_weight * wired_energy) + (self.migration_delay_weight * wired_delay)
        except Exception:
            migration_cost = total_data_size * 0.2

        return {
            'success': True,
            'cost': migration_cost,
            'delay_saved': delay_saved,
            'target_node': target_rsu_id,
            'tasks_migrated': tasks_to_migrate
        }
    
    def execute_uav_migration(self, source_uav_idx: int, urgency: float,
                              coordinator: Optional['StrategyCoordinator'] = None,
                              joint_params: Optional[Dict] = None) -> Dict[str, float]:
        """
        执行UAV到RSU的迁移并返回成本/延迟指标
        执行UAV到RSU的迁移并返回成本/延迟指标
        
        实现UAV到RSU的任务迁移，通过无线链路传输任务：
        1. 根据距离和负载选择目标RSU
        2. 考虑无线传输的可靠性（基于距离和负载）
        3. 动态调整迁移比例（UAV更激进）
        4. 模拟无线传输延迟和能耗
        5. 记录迁移统计信息
        
        Args:
            source_uav_idx: 源UAV的索引
            urgency: 迁移紧急度（0.0-1.0）
            
        Returns:
            包含迁移结果的字典：
            - success: 是否成功（考虑无线链路可靠性）
            - cost: 迁移成本（能耗+延迟）
            - delay_saved: 节省的延迟
            
        Execute UAV-to-RSU migration via wireless link.
        """
        source_uav = self.uavs[source_uav_idx]
        source_queue = source_uav.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        # 寻找候选目标RSU，考虑距离和负载
        # Find candidate target RSUs considering distance and load
        uav_position = source_uav['position']
        candidates = []
        for i, rsu in enumerate(self.rsus):
            queue_len = len(rsu.get('computation_queue', []))
            distance = self.calculate_distance(uav_position, rsu['position'])
            cpu_load = min(0.99, queue_len / 10.0)
            score = distance * 0.01 + queue_len + cpu_load * 10.0
            candidates.append((i, queue_len, cpu_load, distance, score))

        if not candidates:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        # 选择综合得分最佳的目标RSU
        # Select the best target RSU based on composite score
        target_idx, target_queue_len, target_cpu_load, distance, _ = min(candidates, key=lambda x: x[4])
        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        # UAV迁移更激进（比例更高），并结合迁移退避参数
        source_queue_len = len(source_queue)
        backoff = 0.0
        if joint_params:
            try:
                backoff = float(joint_params.get('migration_backoff', 0.0) or 0.0)
            except (TypeError, ValueError):
                backoff = 0.0
        backoff = float(np.clip(backoff, 0.0, 1.0))
        migration_ratio = max(0.2, min(0.6, (urgency + 0.1) * (1.0 - 0.3 * backoff)))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        # 无线链路可靠性模型：考虑距离、负载和紧急度
        # Wireless link reliability model: consider distance, load, and urgency
        base_success_rate = 0.75
        distance_penalty = min(0.35, distance / 1200.0)  # 距离越远成功率越低
        load_penalty = min(0.25, target_queue_len / 16.0)  # 目标负载越高成功率越低
        urgency_bonus = min(0.2, urgency)  # 紧急度提供额外成功率
        actual_success_rate = np.clip(base_success_rate - distance_penalty - load_penalty + urgency_bonus, 0.35, 0.95)
        if np.random.random() > actual_success_rate:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        # 执行迁移
        # Execute migration
        migrated_tasks = source_queue[:tasks_to_migrate]
        source_uav['computation_queue'] = source_queue[tasks_to_migrate:]
        target_rsu['computation_queue'].extend(migrated_tasks)
        if coordinator is not None and migrated_tasks:
            try:
                coordinator.prepare_prefetch(source_uav, target_rsu, migrated_tasks, urgency)
            except Exception as exc:
                print(f"⚠️ UAV迁移前预取协调失败(UAV_{source_uav_idx}->{target_rsu.get('id')}): {exc}")

        total_data_size = sum(task.get('data_size', 1.0) for task in migrated_tasks)
        if total_data_size <= 0.0:
            total_data_size = tasks_to_migrate * 1.0
        # Estimate wireless transfer characteristics
        wireless_rate = 12.0  # MB/s
        wireless_delay = (total_data_size / wireless_rate)
        wireless_energy = total_data_size * 0.15 + distance * 0.01
        queue_relief = max(0.0, source_queue_len - len(source_uav['computation_queue']))
        delay_saved = max(0.0, queue_relief * self.time_slot)

        self.stats['uav_migration_distance'] = self.stats.get('uav_migration_distance', 0.0) + distance
        self.stats['uav_migration_count'] = self.stats.get('uav_migration_count', 0) + 1

        migration_cost = (self.migration_energy_weight * wireless_energy) + (self.migration_delay_weight * wireless_delay)
        return {
            'success': True,
            'cost': migration_cost,
            'delay_saved': delay_saved,
            'target_node': target_rsu.get('id'),
            'tasks_migrated': tasks_to_migrate
        }

    def get_central_scheduling_report(self) -> Dict[str, Any]:
        scheduler = getattr(self, 'central_scheduler', None)
        if scheduler is None:
            return {'status': 'not_available', 'message': '中央调度器未启用'}
        try:
            status = scheduler.get_global_scheduling_status()
            rsu_details: Dict[str, Dict[str, float]] = {}
            for rsu_id, load_info in scheduler.rsu_loads.items():
                rsu_details[rsu_id] = {
                    'cpu_usage': float(getattr(load_info, 'cpu_usage', 0.0)),
                    'queue_length': int(getattr(load_info, 'queue_length', 0)),
                    'cache_usage': float(getattr(load_info, 'cache_usage', 0.0)),
                    'served_vehicles': int(getattr(load_info, 'served_vehicles', 0)),
                    'bandwidth_usage': float(getattr(load_info, 'network_bandwidth_usage', 0.0)),
                }
            return {
                'status': 'ok',
                'message': '中央调度器运行中',
                'scheduling_calls': status.get('global_metrics', {}).get('scheduling_decisions_count', 0),
                'central_scheduler_status': status,
                'rsu_details': rsu_details,
                'migrations_triggered': self.stats.get('central_scheduler_migrations', 0),
            }
        except Exception as exc:
            logging.debug("Central scheduling report failed: %s", exc)
            return {'status': 'error', 'message': str(exc)}
