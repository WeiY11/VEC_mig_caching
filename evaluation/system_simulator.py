#!/usr/bin/env python3
"""
完整系统仿真器

用于测试完整的车联网边缘缓存系统，提供高保真的车辆、RSU、UAV交互仿真。
支持任务生成、卸载决策、缓存管理、迁移策略等功能。

Complete system simulator for testing the full vehicular edge caching system.
Provides high-fidelity simulation of vehicle, RSU, and UAV interactions.
"""

import math
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
from decision.strategy_coordinator import StrategyCoordinator

try:
    from communication.bandwidth_allocator import BandwidthAllocator
except ImportError:  # pragma: no cover - optional module
    BandwidthAllocator = None

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
    
    def __init__(self, config: Optional[Dict] = None):
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
        except (ImportError, AttributeError, ModuleNotFoundError) as e:
            logging.debug(f"System config not available: {e}")
            self.sys_config = None
        
        # 网络拓扑参数：车辆、RSU、UAV数量
        # Network topology parameters: number of vehicles, RSUs, and UAVs
        if self.sys_config is not None and not self.override_topology:
            self.num_vehicles = getattr(self.sys_config.network, 'num_vehicles', 12)
            self.num_rsus = getattr(self.sys_config.network, 'num_rsus', 6)
            self.num_uavs = getattr(self.sys_config.network, 'num_uavs', 2)
        else:
            self.num_vehicles = self.config.get('num_vehicles', 12)
            self.num_rsus = self.config.get('num_rsus', 4)  # 🔑 修复：使用正确的默认值
            self.num_uavs = self.config.get('num_uavs', 2)
        if self.sys_config is not None and not self.override_topology:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300)
            default_uav_radius = getattr(self.sys_config.network, 'uav_coverage_radius', 350)
            default_uav_altitude = getattr(self.sys_config.network, 'uav_altitude', 120.0)
        else:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300) if self.sys_config is not None else 300
            default_uav_radius = getattr(self.sys_config.network, 'uav_coverage_radius', 350) if self.sys_config is not None else 350
            default_uav_altitude = getattr(self.sys_config.network, 'uav_altitude', 120.0) if self.sys_config is not None else 120.0
        self.coverage_radius = self.config.get('coverage_radius', default_radius)
        self.uav_coverage_radius = self.config.get('uav_coverage_radius', default_uav_radius)
        self.uav_altitude = self.config.get('uav_altitude', default_uav_altitude)

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
        self.vehicle_nominal_capacity = float(getattr(queue_cfg, 'vehicle_nominal_capacity', 20.0)) if queue_cfg is not None else 20.0
        self.queue_overflow_margin = float(getattr(queue_cfg, 'overflow_margin', 1.2)) if queue_cfg is not None else 1.2
        self.cache_config = getattr(self.sys_config, 'cache', None)
        self.communication_config = getattr(self.sys_config, 'communication', None)
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
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.rsu_cpu_freq = getattr(self.sys_config.compute, 'rsu_default_freq', 15e9)
            self.uav_cpu_freq = getattr(self.sys_config.compute, 'uav_default_freq', 4e9)
            self.vehicle_cpu_freq = getattr(self.sys_config.compute, 'vehicle_default_freq', 0.167e9)
            self.bandwidth = getattr(self.sys_config.network, 'bandwidth', 50e6)
        else:
            self.rsu_cpu_freq = self.config.get('rsu_cpu_freq', 15e9)  # Hz
            self.uav_cpu_freq = self.config.get('uav_cpu_freq', 4e9)  # Hz
            self.vehicle_cpu_freq = self.config.get('vehicle_cpu_freq', 0.167e9)  # Hz
            self.bandwidth = self.config.get('bandwidth', 50e6)  # Hz

        # 基准频率用于计算capacity scale，保持统一参照（默认 15/4GHz）
        self.rsu_reference_freq = float(self.config.get('rsu_reference_freq', 15e9))
        self.uav_reference_freq = float(self.config.get('uav_reference_freq', 4e9))
        
        # 初始化组件（车辆、RSU、UAV等）
        # Initialize components (vehicles, RSUs, UAVs, etc.)
        self.initialize_components()
        self._reset_runtime_states()
        self._init_dynamic_bandwidth_support()
    
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
        # 🔧 修复：根据用户指定的坐标系统重新调整（UAV_0为原点，向右X+，向下Y-）
        # 用户坐标：UAV_0(0,0), UAV_1(0,-1030), 但系统内部需要正坐标
        # 解决方案：将整体场景向Y轴正方向偏移1545m，确保所有坐标都为正值
        # 🎯 场景范围：X: [-515, 515] → [0, 1030], Y: [-1545, 515] → [0, 2060]
        self.offset_y = 1545.0  # Y轴偏移量，使最小坐标为0
        self.offset_x = 515.0   # X轴偏移量，使最小坐标为0
        
        # 转换后的场景范围
        self.scenario_width = 1030.0   # X轴范围: 0 ~ 1030m
        self.scenario_height = 2060.0  # Y轴范围: 0 ~ 2060m
        
        # 主干道和路口位置（转换后的坐标）
        self.road_center_x = 515.0  # 主干道X坐标（0+515）
        self.road_width = 30.0      # 道路宽度
        self.road_y = self.offset_y  # 为了兼容旧代码，设为上路口Y坐标
        
        # 两个十字路口位置（转换后）
        intersection_0_y = 1545.0  # 上路口：原(0,0) → (515, 1545)
        intersection_1_y = 515.0   # 下路口：原(0,-1030) → (515, 515)
        
        self.intersections = {  # 信号灯相位 周期 T，绿灯比例 g
            'upper': {'x': self.road_center_x, 'y': intersection_0_y, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 0.0},
            'lower': {'x': self.road_center_x, 'y': intersection_1_y, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 15.0},
        }

        # 车辆初始化：落在道路上，方向为东(0)或西(pi)，车道内微扰
        # Vehicle initialization: positioned on road, heading east (0) or west (pi), with lane perturbation
        # 🔧 修复：根据新场景范围调整车辆初始化区域
        self.vehicles = []
        for i in range(self.num_vehicles):
            # 随机分布在主干道和两个路口的横向道路上
            road_choice = np.random.rand()
            if road_choice < 0.5:  # 50%在主干道（纵向）
                go_north = np.random.rand() < 0.5
                x0 = self.road_center_x + np.random.uniform(-self.road_width/2, self.road_width/2)
                y0 = np.random.uniform(515.0, 1545.0)  # 在两个路口之间
                base_dir = -np.pi/2 if go_north else np.pi/2  # 北或南
            else:  # 50%在横向道路
                intersection_y = intersection_0_y if np.random.rand() < 0.5 else intersection_1_y
                go_east = np.random.rand() < 0.6
                x0 = np.random.uniform(50.0, 980.0)  # 横向道路范围
                y0 = intersection_y + np.random.uniform(-self.road_width/2, self.road_width/2)
                base_dir = 0.0 if go_east else np.pi  # 东或西
                    
            v0 = np.random.uniform(8.0, 15.0)  # 初始速度 8-15 m/s (~29-54 km/h，降低移动速度)
            vehicle = {
                'id': f'V_{i}',
                'position': np.array([x0, y0], dtype=float),
                'velocity': v0,
                'direction': base_dir,
                'lane_bias': 0.0,  # 车道偏差
                'tasks': [],
                'energy_consumed': 0.0,
                'device_cache': {},  # 车载缓存
                'device_cache_capacity': 100.0,  # 车载缓存容量(MB) - 100MB
                # 🎯 Phase 2本地调度参数
                'cpu_freq': self.vehicle_cpu_freq,  # 分配的CPU频率（Hz）
                'cpu_frequency': self.vehicle_cpu_freq,  # 🔧 新增：与状态编码字段名一致
                'allocated_bandwidth': 0.0,  # 分配的带宽（Hz）
                'task_queue_by_priority': {1: [], 2: [], 3: [], 4: []},  # 按优先级分类的任务队列
                'compute_usage': 0.0,  # 当前计算使用率
                'queue_length': 0,  # 🔧 新增：当前队列长度（用于状态编码）
            }
            self.vehicles.append(vehicle)
        print(f"车辆初始化完成：主幹道双路口场景，场景范围X:[0,{self.scenario_width:.0f}] Y:[0,{self.scenario_height:.0f}]")
        
        # RSU节点初始化
        # RSU node initialization
        self.rsus = []
        # 🔧 修复：将用户坐标转换为系统内部正坐标
        # 用户坐标（横向X，纵向Y） → 系统坐标（X+515, Y+1545）
        # 道路布局：两个十字路口中心(515,1545)和(515,515)，每个路口向四方延伸515m，道路宽30m
        if self.num_rsus <= 4:
            # 🎯 用户指定坐标（标准笛卡尔坐标系）→ 转换后的系统坐标：
            # RSU_0: (100, 65) → (615, 1610)
            # RSU_1: (-65, -150) → (450, 1395)
            # RSU_2: (100, -750) → (615, 795)
            # RSU_3: (-65, -1150) → (450, 395)
            rsu_positions = [
                np.array([100.0 + self.offset_x, 65.0 + self.offset_y]),       # RSU_0: (615, 1610)
                np.array([-65.0 + self.offset_x, -150.0 + self.offset_y]),     # RSU_1: (450, 1395)
                np.array([100.0 + self.offset_x, -750.0 + self.offset_y]),     # RSU_2: (615, 795)
                np.array([-65.0 + self.offset_x, -1150.0 + self.offset_y]),    # RSU_3: (450, 395)
            ]
        else:
            # 动态生成RSU位置，均匀分布在道路交叉口周围
            rsu_positions = []
            spacing = 1500.0 / (self.num_rsus - 1)  # 均匀间隔
            for i in range(self.num_rsus):
                y_pos = 300.0 + i * spacing
                x_pos = 350.0 if i % 2 == 0 else 650.0  # 交错左右（道路外）
                rsu_positions.append(np.array([x_pos, y_pos]))
        
        # 创建RSU节点
        # Create RSU nodes with configuration
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': rsu_positions[i],
                'coverage_radius': self.coverage_radius,  # 覆盖半径(m)
                'cache': {},  # 缓存字典
                'cache_capacity': 200.0,  # 缓存容量(MB) - 200MB边缘服务器缓存
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'rsu_cache_capacity', 200e6) if self.sys_config is not None else 200e6),
                'cpu_freq': self.rsu_cpu_freq,  # 🆕 CPU频率(Hz)
                'cpu_frequency': self.rsu_cpu_freq,  # 🔧 新增：与状态编码字段名一致
                'computation_queue': [],  # 计算任务队列
                'energy_consumed': 0.0,  # 累计能耗(J)
                # 🎯 Phase 2资源调度参数
                'allocated_compute': self.rsu_cpu_freq,  # 分配的计算资源（Hz）
                'compute_usage': 0.0,  # 当前计算使用率
                'connected_vehicles': [],  # 接入的车辆列表
                'recent_cache_hit_rate': 0.5,  # 🔧 新增：近期缓存命中率（用于状态编码）
                'cache_hits_window': 0,  # 🔧 统计窗口内的缓存命中次数
                'cache_requests_window': 0,  # 🔧 统计窗口内的缓存请求次数
            }
            self.rsus.append(rsu)
        
        # UAV节点初始化
        # UAV node initialization
        self.uavs = []
        # 🔧 修复：将用户坐标转换为系统内部正坐标
        # 用户坐标（横向X，纵向Y） → 系统坐标（X+515, Y+1545）
        # 两个UAV分别在十字路口中心上空，间距1030m
        if self.num_uavs <= 2:
            # 🎯 用户指定坐标（标准笛卡尔坐标系）→ 转换后的系统坐标：
            # UAV_0: (0, 0) → (515, 1545) - 上路口中心上空
            # UAV_1: (0, -1030) → (515, 515) - 下路口中心上空
            uav_positions = [
                np.array([0.0 + self.offset_x, 0.0 + self.offset_y, self.uav_altitude]),        # UAV_0: (515, 1545, alt)
                np.array([0.0 + self.offset_x, -1030.0 + self.offset_y, self.uav_altitude]),    # UAV_1: (515, 515, alt)
            ]
        else:
            # 动态生成UAV位置，均匀分布在道路上方，避免与RSU重叠
            uav_positions = []
            spacing = 1500.0 / (self.num_uavs - 1)  # 均匀间隔
            for i in range(self.num_uavs):
                x_pos = 500.0  # 保持在主干道中央
                y_pos = 300.0 + i * spacing
                uav_positions.append(np.array([x_pos, y_pos, self.uav_altitude]))
        
        # 创建UAV节点
        # Create UAV nodes with configuration
        for i in range(self.num_uavs):
            uav = {
                'id': f'UAV_{i}',
                'position': uav_positions[i],  # 固定悬停位置
                'velocity': 0.0,  # 当前速度(m/s)
                'coverage_radius': self.uav_coverage_radius,  # 🔧 修复: 从配置读取覆盖半径
                'cache': {},  # 缓存字典
                'cache_capacity': 150.0,  # 缓存容量(MB) - 150MB轻量级UAV缓存
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'uav_cache_capacity', 150e6) if self.sys_config is not None else 150e6),
                'cpu_freq': self.uav_cpu_freq,  # 🆕 CPU频率(Hz)
                'cpu_frequency': self.uav_cpu_freq,  # 🔧 新增：与状态编码字段名一致
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
        except (ImportError, AttributeError, RuntimeError) as e:
            logging.warning(f"中央调度器加载失败: {e}")
            self.central_scheduler = None
        
        # 懒加载迁移管理器
        # Lazy load migration manager for task migration strategies
        try:
            from migration.migration_manager import TaskMigrationManager
            if not hasattr(self, 'migration_manager') or self.migration_manager is None:
                self.migration_manager = TaskMigrationManager()
        except (ImportError, AttributeError) as e:
            logging.debug(f"Migration manager not available: {e}")
            self.migration_manager = None

        # 初始化自适应缓存控制器
        try:
            from utils.adaptive_control import AdaptiveCacheController
            self.adaptive_cache_controller = AdaptiveCacheController(
                cache_capacity=1000.0  # Default RSU capacity
            )
            print("自适应缓存控制器已启用")
        except (ImportError, AttributeError, RuntimeError) as e:
            logging.warning(f"自适应缓存控制器加载失败: {e}")
            self.adaptive_cache_controller = None
        
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
        except (ValueError, TypeError) as e:
            logging.warning(f"Topology consistency check failed: {e}")

        self._init_mm1_predictor()
        self._refresh_spatial_index(update_static=True, update_vehicle=True)
    
    # ========== Phase 2本地调度逻辑 ==========
    
    def apply_resource_allocation(self, allocation_dict: Dict[str, np.ndarray]):
        """
        应用中央智能体的资源分配决策（Phase 1 -> Phase 2）
        
        Args:
            allocation_dict: 中央智能体生成的资源分配字典
                - 'bandwidth': [num_vehicles]  带宽分配比例
                - 'vehicle_compute': [num_vehicles]  车辆计算分配比例
                - 'rsu_compute': [num_rsus]  RSU计算分配比例
                - 'uav_compute': [num_uavs]  UAV计算分配比例
        """
        alloc_dict = dict(allocation_dict)
        base_bandwidth = self._prepare_bandwidth_vector(alloc_dict.get('bandwidth'))
        alloc_dict['bandwidth'] = base_bandwidth
        if self.dynamic_bandwidth_enabled:
            adjusted_bandwidth, stats = self._apply_dynamic_bandwidth(base_bandwidth)
            alloc_dict['bandwidth'] = adjusted_bandwidth
            self._last_dynamic_bandwidth = adjusted_bandwidth.copy()
            if stats:
                self.stats['bandwidth_allocator_utilization'] = stats.get('utilization', 0.0)
                self.stats['bandwidth_allocator_avg_bw'] = stats.get('avg_bandwidth', 0.0)
                self.stats['bandwidth_allocator_num_links'] = stats.get('num_links', 0)
                self.stats['bandwidth_allocator_updates'] = self.stats.get('bandwidth_allocator_updates', 0) + 1
        
        self.resource_pool.update_allocation(alloc_dict)
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle['allocated_bandwidth'] = self.resource_pool.get_vehicle_bandwidth(i)
            # 🔧 P2修复：统一命名 cpu_freq → allocated_compute
            vehicle['allocated_compute'] = self.resource_pool.get_vehicle_compute(i)
            vehicle['cpu_freq'] = vehicle['allocated_compute']  # 保持向后兼容
        
        for i, rsu in enumerate(self.rsus):
            rsu['allocated_compute'] = self.resource_pool.get_rsu_compute(i)
        
        for i, uav in enumerate(self.uavs):
            uav['allocated_compute'] = self.resource_pool.get_uav_compute(i)

    def _init_dynamic_bandwidth_support(self) -> None:
        """配置并初始化动态带宽分配功能。"""
        self.dynamic_bandwidth_enabled = False
        self.bandwidth_allocator = None
        self._bandwidth_allocator_mode = 'hybrid'
        self._bandwidth_allocation_blend = 0.6
        self._bandwidth_demand_floor_bits = 0.5e6 * 8.0
        self._bandwidth_idle_demand_bits = 0.1e6 * 8.0
        self._last_dynamic_bandwidth = np.ones(max(1, self.num_vehicles), dtype=float) / max(1, self.num_vehicles)

        env_blend = os.environ.get('BANDWIDTH_ALLOCATOR_BLEND')
        if env_blend:
            try:
                self._bandwidth_allocation_blend = float(env_blend)
            except ValueError:
                pass
        self._bandwidth_allocation_blend = float(np.clip(self._bandwidth_allocation_blend, 0.0, 1.0))

        comm_cfg_flag = bool(getattr(self.communication_config, 'use_bandwidth_allocator', False)) if self.communication_config is not None else False
        dict_flag = bool(self.config.get('use_bandwidth_allocator', False))
        env_flag = os.environ.get('USE_BANDWIDTH_ALLOCATOR')
        env_flag_active = bool(env_flag and env_flag.lower() in {'1', 'true', 'yes', 'on'})
        config_flag = False
        if self.sys_config is not None:
            try:
                from config import config as global_config  # type: ignore
                config_flag = bool(getattr(global_config.communication, 'use_bandwidth_allocator', False))
            except Exception:
                config_flag = False

        should_enable = comm_cfg_flag or dict_flag or env_flag_active or config_flag
        if should_enable and BandwidthAllocator is None:
            logging.warning("BandwidthAllocator module unavailable, dynamic bandwidth disabled.")
            should_enable = False
        if not should_enable:
            self.stats['dynamic_bandwidth_enabled'] = False
            return

        total_bw = float(getattr(self.resource_pool, 'total_bandwidth', max(1e6, self.bandwidth)))
        if self.communication_config is not None:
            min_channel = float(getattr(self.communication_config, 'channel_bandwidth', total_bw / max(1, self.num_vehicles)))
        else:
            min_channel = total_bw / max(1, self.num_vehicles)
        min_channel = max(0.25 * min_channel, 0.5e6)
        if BandwidthAllocator is None:
            logging.warning("BandwidthAllocator is not available")
            self.bandwidth_allocator = None
            self.stats['dynamic_bandwidth_enabled'] = False
            return
        try:
            self.bandwidth_allocator = BandwidthAllocator(total_bandwidth=total_bw, min_bandwidth=min_channel)
        except (TypeError, ValueError, AttributeError) as exc:
            logging.warning(f"Failed to initialize BandwidthAllocator: {exc}")
            self.bandwidth_allocator = None
            self.stats['dynamic_bandwidth_enabled'] = False
            return

        self.dynamic_bandwidth_enabled = True
        self.stats['dynamic_bandwidth_enabled'] = True
        print("✅ 动态带宽分配器已启用：结合RL动作与实时队列/SINR需求自动调整带宽")

    def _prepare_bandwidth_vector(self, raw_vector: Optional[np.ndarray]) -> np.ndarray:
        """归一化中央智能体输出的带宽向量，保证维度一致。"""
        if raw_vector is None:
            base = np.array(self.resource_pool.bandwidth_allocation, copy=True)
            return self._normalize_vector(base)
        arr = np.asarray(raw_vector, dtype=float).flatten()
        if arr.size == self.num_vehicles:
            base = arr
        else:
            base = np.ones(self.num_vehicles, dtype=float)
            limit = min(arr.size, self.num_vehicles)
            if limit > 0:
                base[:limit] = arr[:limit]
            if limit < self.num_vehicles and limit > 0:
                base[limit:] = np.mean(arr[:limit])
        return self._normalize_vector(base)

    def _apply_dynamic_bandwidth(self, base_vector: np.ndarray) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """执行动态带宽分配并与RL提议混合。"""
        if not self.dynamic_bandwidth_enabled or self.bandwidth_allocator is None:
            return base_vector, None
        requests = self._collect_bandwidth_requests(base_vector)
        if not requests:
            return base_vector, None
        allocations = self.bandwidth_allocator.allocate_bandwidth(
            requests, allocation_mode=self._bandwidth_allocator_mode
        )
        if not allocations:
            return base_vector, None

        dyn_vector = np.zeros_like(base_vector)
        total_bw = max(1e-9, float(self.bandwidth_allocator.total_bandwidth))
        for idx, vehicle in enumerate(self.vehicles):
            dyn_vector[idx] = float(allocations.get(vehicle['id'], 0.0)) / total_bw
        dyn_vector = self._normalize_vector(dyn_vector)
        blended = np.clip(
            self._bandwidth_allocation_blend * dyn_vector + (1.0 - self._bandwidth_allocation_blend) * base_vector,
            0.0,
            1.0,
        )
        blended = self._normalize_vector(blended)
        stats = self.bandwidth_allocator.get_allocation_stats(allocations)
        return blended, stats

    def _collect_bandwidth_requests(self, base_vector: np.ndarray) -> List[Dict[str, float]]:
        """构建带宽分配器需要的活跃链路描述。"""
        if self.num_vehicles <= 0:
            return []
        requests: List[Dict[str, float]] = []
        for idx, vehicle in enumerate(self.vehicles):
            queue = vehicle.get('task_queue_by_priority', {})
            total_bits = 0.0
            highest_priority = 4
            for priority in range(1, 5):
                tasks = queue.get(priority, [])
                if tasks and highest_priority == 4:
                    highest_priority = priority
                for task in tasks:
                    data_bytes = task.get('data_size_bytes')
                    if data_bytes is None:
                        data_bytes = task.get('data_size', 1.0) * 1e6
                    total_bits += max(0.0, float(data_bytes)) * 8.0
            if total_bits <= 0.0:
                total_bits = self._bandwidth_idle_demand_bits
            else:
                total_bits = max(total_bits, self._bandwidth_demand_floor_bits)
            rl_bias = float(base_vector[idx]) if idx < base_vector.size else 0.0
            total_bits *= max(0.2, 0.7 + 0.6 * rl_bias)
            request = {
                'task_id': vehicle['id'],
                'priority': min(max(highest_priority, 1), 4),
                'sinr': self._estimate_vehicle_sinr(vehicle),
                'data_size': total_bits,
                'node_type': 'vehicle',
            }
            requests.append(request)
        return requests

    def _estimate_vehicle_sinr(self, vehicle: Dict[str, Any]) -> float:
        """基于最近的RSU/UAV距离估算车辆链路SINR。"""
        if not (self.rsus or self.uavs):
            return 10.0
        vehicle_pos = vehicle.get('position')
        if vehicle_pos is None:
            return 10.0
        position = np.asarray(vehicle_pos, dtype=float)
        freq_hz = self._get_comm_value('carrier_frequency', 3.5e9)
        freq_ghz = max(freq_hz / 1e9, 0.5)
        noise_density_dbm = self._get_comm_value('thermal_noise_density', -174.0)
        per_vehicle_bw = max(
            1e6, float(getattr(self.resource_pool, 'total_bandwidth', self.bandwidth)) / max(1, self.num_vehicles)
        )
        noise_dbm = noise_density_dbm + 10.0 * math.log10(per_vehicle_bw)
        best_linear = 0.5
        for node in list(self.rsus) + list(self.uavs):
            node_pos = node.get('position')
            if node_pos is None:
                continue
            dist = float(self.calculate_distance(position, np.asarray(node_pos, dtype=float)))
            d_km = max(dist / 1000.0, 0.001)
            path_loss_db = 32.4 + 21.0 * math.log10(d_km) + 20.0 * math.log10(freq_ghz)
            if node in self.rsus:
                tx_power_dbm = self._get_comm_value('rsu_tx_power', 46.0)
                tx_gain = self._get_comm_value('antenna_gain_rsu', 15.0)
            else:
                tx_power_dbm = self._get_comm_value('uav_tx_power', 30.0)
                tx_gain = self._get_comm_value('antenna_gain_uav', 5.0)
            rx_gain = self._get_comm_value('antenna_gain_vehicle', 3.0)
            rx_power_dbm = tx_power_dbm + tx_gain + rx_gain - path_loss_db
            sinr_db = rx_power_dbm - noise_dbm
            best_linear = max(best_linear, 10.0 ** (sinr_db / 10.0))
        return float(max(0.1, best_linear))

    def _get_comm_value(self, attr: str, default: float) -> float:
        """从通信配置中安全获取参数。"""
        cfg = self.communication_config
        if cfg is not None and hasattr(cfg, attr):
            try:
                return float(getattr(cfg, attr))
            except Exception:
                return float(default)
        return float(default)

    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """通用归一化工具，确保向量求和为1。"""
        if vector.size == 0:
            return vector
        vec = np.clip(vector.astype(float), 0.0, None)
        total = vec.sum()
        if total <= 1e-9:
            return np.ones_like(vec) / len(vec)
        return vec / total
    def vehicle_priority_scheduling(self, vehicle: Dict):
        """
        车辆端优先级队列调度（Phase 2执行层）
        
        🚀 融合Luo论文队列模型：
        - 车辆侧维护L个生命周期队列（队列l表示还有l个时隙到截止时间）
        - 队列l输入：(1)本车队列l+1未处理的 (2)新生成时延约束=l的 (3)V2V迁移来的l+1数据
        - 队列l输出：(1)Offload→RSU队列l-1 (2)Migrate→其他车队列l-1 (3)Local处理 (4)Remain→本车队列l-1
        - 每个时隙结束时，未处理任务生命周期-1（降级到下一队列）
        
        【策略】
        1. 按任务优先级（类型1>2>3>4）排序
        2. 优先分配计算资源给高优先级任务
        3. 如果本地资源不足，标记为待卸载
        
        Args:
            vehicle: 车辆对象字典
        """
        # 获取分配的计算资源
        # 🔧 修复：统一使用allocated_compute字段
        allocated_cpu = vehicle.get('allocated_compute', vehicle.get('cpu_freq', self.vehicle_cpu_freq))
        time_slot = self.time_slot
        
        # 🆕 论文模型：初始化生命周期队列结构（如果不存在）
        if 'lifetime_queues' not in vehicle:
            vehicle['lifetime_queues'] = self._init_lifetime_queues_vehicle()
        
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
        
        🚀 融合Luo论文队列模型：
        - RSU侧维护L-1个生命周期队列（最短1个时隙从车传到RSU）
        - 队列l输入：(1)自己队列l+1上时隙未处理的 (2)车辆V2I卸载来的剩余寿命l+1数据
        - 队列l输出：(1)ECN计算处理 (2)未处理部分→队列l-1（l=1时过期删除）
        - 每个时隙结束时，未处理任务降级到l-1队列
        
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
        
        # 🆕 论文模型：初始化生命周期队列结构（如果不存在）
        if 'lifetime_queues' not in rsu:
            rsu['lifetime_queues'] = self._init_lifetime_queues_rsu()
        
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
        
        🚀 融合Luo论文队列模型：
        - UAV侧类似RSU，维护L-1个生命周期队列
        - 队列流转逻辑同RSU（论文将UAV视为移动基站）
        
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
        # 🔧 修复：基于分配的计算资源计算可用周期，而非有效计算资源
        # 这样compute_usage始终基于allocated_compute，不会超过100%
        available_cycles = allocated_compute * time_slot
        
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
        # 🔧 修复：考虑电量因子的影响，但使用率仍基于allocated_compute
        # 如果电量低，实际能处理的cycles会减少，但reported usage基于总分配
        actual_processed = min(used_cycles, effective_compute * time_slot)
        uav['compute_usage'] = actual_processed / max(available_cycles, 1e-9)
    
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
        self._init_dynamic_bandwidth_support()
        print("初始化了 6 个缓存管理器")

    def _fresh_stats_dict(self) -> Dict[str, Any]:
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
            # 按任务类别统计时延性能
            'task_type_delay_stats': {
                1: {'total_delay': 0.0, 'count': 0, 'max_delay': 0.0, 'deadline_violations': 0, 'deadline': 0.2},
                2: {'total_delay': 0.0, 'count': 0, 'max_delay': 0.0, 'deadline_violations': 0, 'deadline': 0.3},
                3: {'total_delay': 0.0, 'count': 0, 'max_delay': 0.0, 'deadline_violations': 0, 'deadline': 0.4},
                4: {'total_delay': 0.0, 'count': 0, 'max_delay': 0.0, 'deadline_violations': 0, 'deadline': 0.6}
            },
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
            
            # 🔧 修复：处理迁移指令并记录能耗与延迟
            for cmd in migrations:
                if 'wired_transmission' in cmd:
                    wired_stats = cmd['wired_transmission']
                    # 记录迁移能耗 (J)
                    energy = wired_stats.get('energy_j', 0.0)
                    self._accumulate_energy('rsu_migration_energy', energy)
                    self.stats['energy_consumed'] = self.stats.get('energy_consumed', 0.0) + energy # 确保计入总能耗
                    
                    # 记录迁移延迟 (s) - 注意：这是后台传输延迟，不直接阻塞任务，但计入系统开销
                    delay_ms = wired_stats.get('delay_ms', 0.0)
                    delay_s = delay_ms / 1000.0
                    self._accumulate_delay('rsu_migration_delay', delay_s)
                    
                    # 记录迁移数据量
                    data_mb = wired_stats.get('data_size_mb', 0.0)
                    self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + data_mb
            
            self.stats['central_scheduler_calls'] = self.stats.get('central_scheduler_calls', 0) + 1
            self.stats['central_scheduler_last_decisions'] = len(decisions)
            self.stats['central_scheduler_migrations'] = self.stats.get('central_scheduler_migrations', 0) + len(migrations)
            
            # 🚀 创新: 轨迹感知预迁移 (Trajectory-Aware Pre-Migration)
            mobility_migrations = self._check_mobility_migration()
            self.stats['mobility_migrations'] = self.stats.get('mobility_migrations', 0) + mobility_migrations
            
        except Exception as exc:
            logging.debug("Central scheduler update failed: %s", exc)

    def _check_mobility_migration(self) -> int:
        """
        🚀 创新: 轨迹感知预迁移机制
        检测车辆是否即将离开当前RSU覆盖范围，如果是，则提前将任务迁移到下一个RSU。
        """
        migration_count = 0
        if not self.rsus:
            return 0
            
        for vehicle in self.vehicles:
            # 1. 确定当前连接的RSU
            v_pos = vehicle.get('position')
            if v_pos is None:
                continue
            
            current_rsu = None
            min_dist = float('inf')
            
            # 找到最近的RSU
            for rsu in self.rsus:
                dist = self.calculate_distance(v_pos, rsu['position'])
                if dist < min_dist:
                    min_dist = dist
                    current_rsu = rsu
            
            if not current_rsu or min_dist > current_rsu['coverage_radius']:
                continue
                
            # 2. 检查是否在边缘区域 (覆盖半径的90%)
            if min_dist > current_rsu['coverage_radius'] * 0.9:
                # 车辆即将离开，触发预迁移
                
                # 3. 预测下一个RSU (基于移动方向)
                direction = vehicle.get('direction', 0.0)
                next_rsu = None
                best_forward_dist = float('inf')
                
                for rsu in self.rsus:
                    if rsu['id'] == current_rsu['id']:
                        continue
                        
                    # 检查是否在前方
                    dx = rsu['position'][0] - v_pos[0]
                    # 如果向东(direction ~ 0)，dx应为正；向西(direction ~ pi)，dx应为负
                    is_forward = (abs(direction) < 1.0 and dx > 0) or (abs(direction) > 2.0 and dx < 0)
                    
                    if is_forward:
                        dist = self.calculate_distance(v_pos, rsu['position'])
                        if dist < best_forward_dist:
                            best_forward_dist = dist
                            next_rsu = rsu
                
                if next_rsu:
                    # 4. 执行迁移：将该车辆在当前RSU队列中的任务移动到下一个RSU
                    queue = current_rsu.get('computation_queue', [])
                    tasks_to_move = []
                    
                    remaining_queue = []
                    for task in queue:
                        # 检查任务归属
                        tid = task.get('vehicle_id') or task.get('source_vehicle_id')
                        if tid == vehicle['id']:
                            tasks_to_move.append(task)
                        else:
                            remaining_queue.append(task)
                    
                    if tasks_to_move:
                        # 更新队列
                        current_rsu['computation_queue'] = remaining_queue
                        next_rsu.setdefault('computation_queue', []).extend(tasks_to_move)
                        
                        migration_count += len(tasks_to_move)
                        # 记录迁移开销 (简化)
                        # 假设每任务迁移消耗 0.05J (无线信令开销)
                        migration_energy = 0.05 * len(tasks_to_move)
                        self._accumulate_delay('migration_delay', 0.02 * len(tasks_to_move)) # 20ms per task
                        self._accumulate_energy('uav_migration_energy', migration_energy) # 借用uav_migration_energy字段或新建字段
                        self.stats['energy_consumed'] = self.stats.get('energy_consumed', 0.0) + migration_energy
                else:
                    pass
                        
        return migration_count

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

    def _record_task_type_delay(self, task: Dict, actual_delay: float) -> None:
        """
        按任务类别记录时延统计
        
        Args:
            task: 任务字典，必须包含 task_type 和 deadline 字段
            actual_delay: 实际时延(秒)
        """
        task_type = task.get('task_type')
        if task_type is None or task_type not in [1, 2, 3, 4]:
            return
        
        # 获取该任务类别的统计数据
        type_stats = self.stats['task_type_delay_stats'].get(task_type)
        if type_stats is None:
            # 如果不存在，创建默认统计
            default_deadlines = {1: 0.2, 2: 0.3, 3: 0.4, 4: 0.6}
            type_stats = {
                'total_delay': 0.0,
                'count': 0,
                'max_delay': 0.0,
                'deadline_violations': 0,
                'deadline': default_deadlines.get(task_type, 0.5)
            }
            self.stats['task_type_delay_stats'][task_type] = type_stats
        
        # 更新统计数据
        type_stats['total_delay'] += actual_delay
        type_stats['count'] += 1
        type_stats['max_delay'] = max(type_stats['max_delay'], actual_delay)
        
        # 检查是否超过deadline
        task_deadline = task.get('deadline')  # 任务的实际deadline(绝对时间)
        arrival_time = task.get('arrival_time', 0.0)
        if task_deadline is not None:
            # deadline是绝对时间，需要转换为相对时间限制
            deadline_limit = task_deadline - arrival_time
            if actual_delay > deadline_limit:
                type_stats['deadline_violations'] += 1
        else:
            # 如果deadline不存在，使用类别默认deadline
            if actual_delay > type_stats['deadline']:
                type_stats['deadline_violations'] += 1

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
        """记录因队列溢出导致的任务丢弃。
        
        🔧 关键修复：防止重复统计已丢弃的任务
        """
        # 🔧 如果任务已经被标记为丢弃，直接返回，避免重复计数
        if task.get('dropped', False):
            return
        
        self.stats['dropped_tasks'] = self.stats.get('dropped_tasks', 0) + 1
        self.stats['queue_overflow_drops'] = self.stats.get('queue_overflow_drops', 0) + 1
        data_bytes = float(task.get('data_size_bytes', task.get('data_size', 0.0) * 1e6))
        self.stats['dropped_data_bytes'] = self.stats.get('dropped_data_bytes', 0.0) + data_bytes
        task['dropped'] = True
        task['drop_reason'] = 'queue_overflow'
        drop_stats_default: Dict[str, Any] = {
            'total': 0,
            'wait_time_sum': 0.0,
            'queue_sum': 0,
            'by_type': {},
            'by_scenario': {},
            'by_reason': {}
        }
        drop_stats = self.stats.setdefault('drop_stats', drop_stats_default)
        if not isinstance(drop_stats, dict):
            drop_stats = drop_stats_default
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
        """在入队后执行，确保队列受控
        
        🔧 紧急修复：大幅提高队列溢出边界，减少丢弃
        """
        # 🔧 修复：Vehicle使用task_queue_by_priority结构
        if node_type == 'VEHICLE':
            queue_dict = node.get('task_queue_by_priority', {})
            if not isinstance(queue_dict, dict):
                return
            
            # 计算总队列长度
            total_queue_length = sum(len(tasks) for tasks in queue_dict.values())
            
            # 🔧 优化：从配置读取Vehicle队列容量，与配置系统统一
            vehicle_nominal_capacity = getattr(self, 'vehicle_nominal_capacity', 20.0)
            overflow_margin = 2.0  # 允许队列达到名义容量的2倍
            # 最大容量 = 20 × 1.5(node_max_load_factor) × 2.0(overflow_margin) = 60个任务
            max_queue = int(max(1, round(vehicle_nominal_capacity * self.node_max_load_factor * overflow_margin)))
            
            overflow = total_queue_length - max_queue
            if overflow <= 0:
                return
            
            # 从低优先级开始丢弃任务
            dropped = 0
            for priority in [4, 3, 2, 1]:  # 从低到高
                if overflow <= 0:
                    break
                queue = queue_dict.get(priority, [])
                while overflow > 0 and queue:
                    dropped_task = queue.pop()  # 丢弃最新的任务
                    # 🆕 Luo论文队列模型：丢弃任务时从lifetime_queues同步移除
                    self._remove_task_from_lifetime_queues(node, dropped_task)
                    self._record_queue_drop(dropped_task, node_type)
                    dropped += 1
                    overflow -= 1
            
            if dropped:
                step_summary['dropped_tasks'] = step_summary.get('dropped_tasks', 0) + dropped
                step_summary['queue_overflow_drops'] = step_summary.get('queue_overflow_drops', 0) + dropped
            return
        
        # RSU/UAV使用computation_queue结构
        queue = node.get('computation_queue', [])
        if not isinstance(queue, list):
            return
        nominal_capacity = self.rsu_nominal_capacity if node_type == 'RSU' else self.uav_nominal_capacity
        # 🔧 修复：调整溢出边界到合理水平 (3.0 → 2.0)
        # 2倍边界在保证缓冲的同时，避免队列积压过长影响实时性
        # RSU: 50 × 2.0 = 100个任务, UAV: 30 × 2.0 = 60个任务
        overflow_margin = 2.0  # 允许队列长度达到名义容量的2倍
        max_queue = int(max(1, round(nominal_capacity * self.node_max_load_factor * overflow_margin)))
        overflow = len(queue) - max_queue
        if overflow <= 0:
            return
        dropped = 0
        while overflow > 0 and queue:
            dropped_task = queue.pop()  # 丢弃最新的任务，保护早到任务
            # 🆕 Luo论文队列模型：队列溢出丢弃时从lifetime_queues同步移除
            self._remove_task_from_lifetime_queues(node, dropped_task)
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
        
        # 🔧 优化5: 不可缓存任务直接跳过缓存检查
        if not content_id or not task.get('is_cacheable', False):
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
        
        # 按任务类别记录时延统计
        self._record_task_type_delay(task, hit_delay)
        
        cached_entry['timestamp'] = self.current_time
        if cache_controller is not None:
            try:
                cache_controller.record_cache_result(content_id, was_hit=True)
            except (AttributeError, TypeError, ValueError) as e:
                logging.debug(f"Cache controller update failed: {e}")
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
    
    def _init_lifetime_queues_vehicle(self) -> Dict[int, List]:
        """
        🆕 Luo论文队列模型：初始化车辆侧生命周期队列
        
        车辆维护L个队列（队列l = 还有1到L个时隙到截止时间）
        对应论文图2(a)：车辆侧多队列结构
        
        Returns:
            Dict[lifetime, List[Task]]: 键为剩余生命周期，值为任务列表
        """
        max_lifetime = getattr(self.queue_config, 'max_lifetime', 10) if hasattr(self, 'queue_config') else 10
        return {l: [] for l in range(1, max_lifetime + 1)}
    
    def _init_lifetime_queues_rsu(self) -> Dict[int, List]:
        """
        🆕 Luo论文队列模型：RSU侧生命周期队列
        
        RSU维护L-1个队列（队列l = 还有1到L-1个时隙，因为RSU不产生数据）
        对应论文图2(b)：RSU侧多队列结构
        
        Returns:
            Dict[lifetime, List[Task]]: 键为剩余生命周期，值为任务列表
        """
        max_lifetime = getattr(self.queue_config, 'max_lifetime', 10) if hasattr(self, 'queue_config') else 10
        # RSU最大队列号为L-1（最短1个时隙从车传到RSU）
        return {l: [] for l in range(1, max_lifetime)}
    
    def _update_lifetime_queues(self, node: Dict, node_type: str, step_summary: Dict[str, Any]) -> None:
        """
        🆕 Luo论文队列模型：每个时隙更新生命周期队列
        
        核心逻辑：
        1. 队列l中未处理的任务 → 降级到队列l-1
        2. l=1时未处理的任务 → 过期删除，计入惩罚
        
        对应论文第3.2节：“每过一个时隙，所有没被处理/转移的数据队列索引减1”
        
        Args:
            node: 车辆/RSU/UAV节点对象
            node_type: 节点类型
            step_summary: 当前时隙的统计数据
        """
        if 'lifetime_queues' not in node:
            return
        
        lifetime_queues = node['lifetime_queues']
        new_queues = {}
        dropped_count = 0
        urgency_promoted_count = 0  # 🚀 创新：统计紧急提升的任务数
        
        # 🚀 创新1：自适应降级速度 - 根据节点负载调整
        # 高负载时加速降级（腾出队列空间），低负载时正常降级
        node_load = self._calculate_node_rho(node, node_type)
        if node_load > 0.8:  # 高负载
            degradation_step = 2  # 生命周期减2（加速过期）
        elif node_load > 0.6:  # 中等负载
            degradation_step = 1  # 正常降级
        else:  # 低负载
            degradation_step = 1  # 正常降级
        
        # 从高到低遍历每个生命周期队列
        for lifetime in sorted(lifetime_queues.keys(), reverse=True):
            tasks = lifetime_queues[lifetime]
            if not tasks:
                # 空队列，保持结构
                new_queues[lifetime] = []
                continue
            
            # 🚀 创新2：跨队列优先级提升机制
            # 即将过期的任务（lifetime <= 2）自动提升优先级
            for task in tasks:
                if lifetime <= 2 and 'task_type' in task:
                    original_priority = task.get('task_type', 4)
                    # 紧急提升：降低task_type数值（数值越小优先级越高）
                    if original_priority > 1 and not task.get('urgency_promoted', False):
                        task['task_type'] = max(1, original_priority - 1)
                        task['urgency_promoted'] = True  # 标记为紧急提升
                        urgency_promoted_count += 1
            
            # 生命周期降级（自适应步长）
            new_lifetime = max(0, lifetime - degradation_step)
            
            if new_lifetime > 0:
                # 还有剩余时间，任务降级到下一队列
                if new_lifetime not in new_queues:
                    new_queues[new_lifetime] = []
                # 更新任务的剩余生命周期字段
                for task in tasks:
                    if 'remaining_lifetime_slots' in task:
                        task['remaining_lifetime_slots'] = new_lifetime
                new_queues[new_lifetime].extend(tasks)
            else:
                # 生命周期用尽，任务过期删除
                for task in tasks:
                    task['is_dropped'] = True
                    task['drop_reason'] = 'lifetime_expired'
                    self._record_queue_drop(task, node_type)
                    dropped_count += 1
        
        # 确保所有队列位置都存在
        max_lifetime = getattr(self.queue_config, 'max_lifetime', 10) if hasattr(self, 'queue_config') else 10
        if node_type == 'VEHICLE':
            for l in range(1, max_lifetime + 1):
                if l not in new_queues:
                    new_queues[l] = []
        else:  # RSU/UAV
            for l in range(1, max_lifetime):
                if l not in new_queues:
                    new_queues[l] = []
        
        # 更新节点的队列
        node['lifetime_queues'] = new_queues
        
        # 🚀 创新3：智能预测与主动迁移触发
        # 检查队列2和队列1中的任务数量，如果过多则触发迁移预警
        if node_type in ('RSU', 'UAV'):
            critical_tasks = len(new_queues.get(1, [])) + len(new_queues.get(2, []))
            total_tasks = sum(len(q) for q in new_queues.values())
            if critical_tasks > 0 and total_tasks > 0:
                urgency_ratio = critical_tasks / total_tasks
                if urgency_ratio > 0.3:  # 超过30%的任务即将过期
                    node['migration_urgency'] = min(1.0, urgency_ratio * 2)  # 触发迁移紧急度
                    step_summary['migration_triggers'] = step_summary.get('migration_triggers', 0) + 1
        
        # 统计过期任务和优化指标
        if dropped_count > 0:
            step_summary['lifetime_expired_tasks'] = step_summary.get('lifetime_expired_tasks', 0) + dropped_count
            step_summary['dropped_tasks'] = step_summary.get('dropped_tasks', 0) + dropped_count
        
        if urgency_promoted_count > 0:
            step_summary['urgency_promoted_tasks'] = step_summary.get('urgency_promoted_tasks', 0) + urgency_promoted_count
    
    def _remove_task_from_lifetime_queues(self, node: Dict, task: Dict) -> bool:
        """
        🆕 Luo论文队列模型：从clifetime_queues中移除已完成/迁移的任务
        
        防止已完成的任务继续在lifetime_queues中降级，避免内存泄漏和数据不一致
        
        Args:
            node: 节点对象
            task: 要移除的任务
            
        Returns:
            是否成功移除
        """
        if 'lifetime_queues' not in node:
            return False
        
        lifetime_queues = node['lifetime_queues']
        task_id = task.get('id')
        
        # 遍历所有生命周期队列查找并移除任务
        for lifetime, tasks in lifetime_queues.items():
            for i, t in enumerate(tasks):
                if t.get('id') == task_id:
                    tasks.pop(i)
                    return True
        
        return False

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

        # 🔧 修复: 使用 RealisticContentGenerator 统一生成内容，确保高比例可缓存任务
        from utils.realistic_content_generator import generate_realistic_content
        
        # 生成真实的 VEC 内容（包括 content_id, size, priority）
        content_id, content_size_mb, content_priority = generate_realistic_content(vehicle_id, self.current_step)
        
        # 从 content_id 推断 VEC 场景类型（如 traffic_info, navigation 等）
        # content_id 格式为 "{content_type}_{counter:04d}" (例如 "traffic_info_0012" 或 "entertainment_0001")
        # 🔧 修复：使用 rsplit 从右侧分割一次，正确提取包含下划线的类型名
        if '_' in content_id:
            vec_content_type = content_id.rsplit('_', 1)[0]
        else:
            vec_content_type = 'general'
        
        # 映射 VEC 内容类型到仿真器场景名称（用于统计和日志）
        # 这些都是可缓存的真实 VEC 场景
        scenario_name = vec_content_type
        
        # 🔧 P0修复：对齐vec_type_configs与TaskConfig.task_profiles定义
        # 根据 VEC 内容类型设置计算和时延特性
        # task_profiles定义：
        #   类型1: 50-200KB, 60 cycles/bit, ≤0.2s (2 slots)
        #   类型2: 600KB-1.5MB, 90 cycles/bit, ≤0.4s (4 slots)
        #   类型3: 2-4MB, 120 cycles/bit, ≤0.5s (5 slots)
        #   类型4: 4.5-8MB, 150 cycles/bit, ≤0.8s (8 slots)
        vec_type_configs = {
            # 类型1: 极度敏感 - 紧急制动、碰撞避免
            'safety_alert': {'compute_density': 60, 'deadline_range': (0.18, 0.22), 'task_type': 1, 'cache_priority': 1.0},
            'sensor_data': {'compute_density': 60, 'deadline_range': (0.18, 0.22), 'task_type': 1, 'cache_priority': 0.95},
            
            # 类型2: 敏感 - 导航、交通信号
            'navigation': {'compute_density': 90, 'deadline_range': (0.38, 0.42), 'task_type': 2, 'cache_priority': 0.85},
            'weather_info': {'compute_density': 90, 'deadline_range': (0.38, 0.42), 'task_type': 2, 'cache_priority': 0.7},
            
            # 类型3: 中度容忍 - 视频处理、图像识别
            'map_data': {'compute_density': 120, 'deadline_range': (0.48, 0.52), 'task_type': 3, 'cache_priority': 0.8},
            'parking_info': {'compute_density': 120, 'deadline_range': (0.48, 0.52), 'task_type': 3, 'cache_priority': 0.75},
            
            # 类型4: 容忍 - 数据分析、娱乐
            'traffic_info': {'compute_density': 150, 'deadline_range': (0.78, 0.84), 'task_type': 4, 'cache_priority': 0.9},
            'entertainment': {'compute_density': 150, 'deadline_range': (0.78, 0.84), 'task_type': 4, 'cache_priority': 0.5},
        }
        
        # 获取该 VEC 类型的配置（如果未知类型则使用默认值）
        vec_config = vec_type_configs.get(vec_content_type, {
            'compute_density': 400,
            'deadline_range': (0.5, 3.0),
            'task_type': 3,
            'cache_priority': 0.5
        })
        
        # 设置任务参数
        compute_density = vec_config['compute_density']
        deadline_duration = np.random.uniform(*vec_config['deadline_range'])
        initial_type = vec_config['task_type']
        cache_priority = vec_config['cache_priority']
        
        # 使用从 RealisticContentGenerator 获得的真实数据大小
        data_size_mb = content_size_mb
        data_size_bytes = data_size_mb * 1e6
        
        # 时间槽配置
        relax_factor_applied = self.config.get('deadline_relax_fallback', 1.3)
        deadline_duration *= relax_factor_applied
        max_delay_slots = max(
            1,
            int(deadline_duration / max(self.config.get('time_slot', self.time_slot), 0.1)),
        )

        # 任务复杂度控制
        effective_density = compute_density
        complexity_multiplier = 1.0

        if self.config.get('high_load_mode', False):
            complexity_multiplier = self.config.get('task_complexity_multiplier', 1.5)
            data_size_mb = min(data_size_mb * 1.1, 12.0)
            data_size_bytes = data_size_mb * 1e6
            effective_density = min(effective_density * 1.05, 200)

        total_bits = data_size_bytes * 8
        base_cycles = total_bits * effective_density
        adjusted_cycles = base_cycles * complexity_multiplier
        computation_mips = adjusted_cycles / 1e6

        # 所有 VEC 内容都是可缓存的（这是 VEC 缓存的核心）
        cacheable_hint = True
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
            'content_id': content_id,  # 🔧 优化: 仅可缓存任务有content_id
            'is_cacheable': cacheable_hint,  # 🔧 优化3: 添加明确的缓存标记
            'cache_priority': cache_priority,  # 🔧 优化4: 添加缓存优先级
            'priority': np.random.uniform(0.1, 1.0),
            'task_type': task_type,
            'app_scenario': scenario_name,
            'app_name': scenario_name,
            'compute_density': effective_density,
            'complexity_multiplier': complexity_multiplier,
            'max_delay_slots': max_delay_slots,
            'deadline_relax_factor': relax_factor_applied,
            # 🆕 Luo论文队列模型：添加剩余生命周期字段
            'remaining_lifetime_slots': max_delay_slots,  # 初始生命周期 = 最大延迟时隙数
        }

        self._last_app_name = scenario_name

        # 馃搳 浠诲姟缁熻鏀堕泦
        gen_stats_default: Dict[str, Any] = {'total': 0, 'by_type': {}, 'by_scenario': {}}
        gen_stats = self.stats.setdefault('task_generation', gen_stats_default)
        if not isinstance(gen_stats, dict):
            gen_stats = gen_stats_default
        gen_stats['total'] = (gen_stats.get('total', 0) or 0) + 1
        by_type = gen_stats.setdefault('by_type', {})
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_scenario = gen_stats.setdefault('by_scenario', {})
        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1

        stats_cfg = getattr(self, 'stats_config', None)
        report_interval = stats_cfg.task_report_interval if stats_cfg is not None else self.config.get('task_report_interval', 100)
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
            
            # 🔧 优化7: 添加缓存统计实时监控
            cache_hits = self.stats.get('cache_hits', 0)
            cache_misses = self.stats.get('cache_misses', 0)
            total_cache_requests = cache_hits + cache_misses
            if total_cache_requests > 0:
                cache_hit_rate = cache_hits / total_cache_requests
                local_hits = self.stats.get('local_cache_hits', 0)
                print(
                    f"   💾 缓存统计: 命中率={cache_hit_rate:.2%} "
                    f"(总命中:{cache_hits}, 本地:{local_hits}, 未命中:{cache_misses})"
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
        distance = np.linalg.norm(pos1 - pos2)
        return float(distance)
    
    
    def _refresh_spatial_index(self, update_static: bool = True, update_vehicle: bool = True) -> None:
        """
        保持空间索引与实体位置同步。
        update_static=False 时仅刷新车辆索引，避免重复构建静态KD-tree。
        """
        if not getattr(self, 'spatial_index', None):
            return
        try:
            if update_static and self.spatial_index is not None:
                self.spatial_index.update_static_nodes(self.rsus, self.uavs)
            if update_vehicle and self.spatial_index is not None:
                self.spatial_index.update_vehicle_nodes(self.vehicles)
        except (AttributeError, TypeError, ValueError) as e:
            # 索引刷新失败时回退至朴素遍历逻辑
            logging.debug(f"Spatial index update failed, falling back to brute force: {e}")
    
    
    def _find_least_loaded_node(self, node_type: str, exclude_node: Optional[Dict] = None) -> Optional[Dict]:
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
        best_node: Optional[Dict] = min(candidates, key=lambda n: len(n.get('computation_queue', [])))
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
            reference = float(getattr(self, 'rsu_reference_freq', 15e9))
            baseline = float(getattr(self, 'rsu_cpu_freq', reference))
        else:
            reference = float(getattr(self, 'uav_reference_freq', 4e9))
            baseline = float(getattr(self, 'uav_cpu_freq', reference))
        allocated = float(node.get('allocated_compute', baseline))
        denominator = max(reference, 1e-9)
        scale = allocated / denominator
        return float(np.clip(scale, 0.2, 3.0))

    def _is_node_admissible(self, node: Dict, node_type: str) -> bool:
        """检查节点是否允许新的卸载任务进入
        
        🔧 紧急修复：大幅放宽准入阈值，让UAV也能接受任务
        """
        queue_len = len(node.get('computation_queue', []))
        capacity = self.rsu_nominal_capacity if node_type == 'RSU' else self.uav_nominal_capacity
        ratio = queue_len / max(1.0, capacity)
        usage = float(node.get('compute_usage', 0.0))
        
        # 🔧 紧急修复：大幅放宽阈值，让节点能够接受更多任务
        # 原阈值过于严格，导致大量任务被拒绝
        if node_type == 'UAV':
            queue_threshold = 5.0  # UAV队列允许500%容量（极度宽松）
            usage_threshold = 5.0  # UAV使用率允许500%
        else:  # RSU
            queue_threshold = 3.0  # RSU队列允许300%容量（宽松）
            usage_threshold = 3.0  # RSU使用率允许300%
        
        # 队列检查：队列长度 < 阈值
        queue_ok = ratio < queue_threshold
        # 使用率检查：使用率 < 阈值 或者 使用率为0（初始状态）
        usage_ok = usage < usage_threshold or usage == 0.0
        
        return queue_ok and usage_ok

    def _record_offload_rejection(self, node_type: str, reason: str = 'unknown') -> None:
        """记录由于拥塞/策略导致的远端卸载拒绝。"""
        stats_default: Dict[str, Any] = {
            'total': 0,
            'by_type': {'RSU': 0, 'UAV': 0},
            'by_reason': {}
        }
        stats = self.stats.setdefault('remote_rejections', stats_default)
        if not isinstance(stats, dict):
            stats = stats_default
        stats['total'] = stats.get('total', 0) + 1
        by_type = stats.setdefault('by_type', {})
        if isinstance(by_type, dict):
            by_type[node_type] = by_type.get(node_type, 0) + 1
        by_reason = stats.setdefault('by_reason', {})
        if isinstance(by_reason, dict):
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
            # 🔧 修复：即使队列为空，RSU/UAV也消耗静态功耗
            if node_type in ['RSU', 'UAV']:
                # 获取静态功耗配置
                if node_type == 'RSU':
                    static_power = getattr(self.sys_config.compute, 'rsu_static_power', 25.0) if self.sys_config else 25.0
                else:
                    static_power = getattr(self.sys_config.compute, 'uav_static_power', 2.5) if self.sys_config else 2.5
                
                # 计算静态能耗
                static_energy = static_power * self.time_slot
                
                # 累加能耗
                self._accumulate_energy('energy_compute', static_energy)
                node['energy_consumed'] = node.get('energy_consumed', 0.0) + static_energy

            self._record_mm1_queue_length(node_type, node_idx, 0)
            return

        # 根据节点类型获取处理能力配置
        # Get processing capacity configuration based on node type
        # 🔧 修复: 增强配置一致性检查
        if node_type == 'RSU':
            if self.service_config and hasattr(self.service_config, 'rsu_base_service'):
                base_capacity = int(self.service_config.rsu_base_service)  # 基础处理能力
                max_service = int(getattr(self.service_config, 'rsu_max_service', 9))  # 最大处理能力
                boost_divisor = float(getattr(self.service_config, 'rsu_queue_boost_divisor', 5.0))  # 动态提升除数
                work_capacity_cfg = float(getattr(self.service_config, 'rsu_work_capacity', 2.5))  # 工作容量
            else:
                base_capacity = int(self.config.get('rsu_base_service', 4))
                max_service = int(self.config.get('rsu_max_service', 9))
                boost_divisor = 5.0
                work_capacity_cfg = float(self.config.get('rsu_work_capacity', 2.5))
        elif node_type == 'UAV':
            if self.service_config and hasattr(self.service_config, 'uav_base_service'):
                base_capacity = int(self.service_config.uav_base_service)
                max_service = int(getattr(self.service_config, 'uav_max_service', 6))
                boost_divisor = float(getattr(self.service_config, 'uav_queue_boost_divisor', 4.0))
                work_capacity_cfg = float(getattr(self.service_config, 'uav_work_capacity', 1.7))
            else:
                base_capacity = int(self.config.get('uav_base_service', 3))
                max_service = int(self.config.get('uav_max_service', 6))
                boost_divisor = 4.0
                work_capacity_cfg = float(self.config.get('uav_work_capacity', 1.7))
        else:
            # 未知节点类型使用默认值
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
        
        # 🔧 修复v3：基于实际计算周期和CPU频率计算处理进度
        # 问题原因：原work_remaining=0.5是抽象值，导致任务总是4-5个时隙完成
        # 解决方案：使用实际的compute_cycles和cpu_freq计算
        
        # 获取节点CPU频率
        if node_type == 'RSU':
            cpu_freq = getattr(self.sys_config.compute, 'rsu_cpu_freq', 12.5e9) if self.sys_config else 12.5e9
        elif node_type == 'UAV':
            cpu_freq = getattr(self.sys_config.compute, 'uav_cpu_freq', 5.0e9) if self.sys_config else 5.0e9
        else:
            cpu_freq = 2.5e9  # Vehicle默认
        
        # 每个时隙可处理的计算周期数
        cycles_per_slot = cpu_freq * self.time_slot
        
        # 本时隙已使用的周期（用于容量限制）
        total_cycles_used = 0.0

        for idx, task in enumerate(queue):
            if current_time - task.get('queued_at', -1e9) < self.time_slot:
                new_queue.append(task)
                continue

            if idx >= tasks_to_process:
                new_queue.append(task)
                continue
            
            # 🔧 修复v3：使用实际剩余计算周期
            # 首次处理时，从compute_cycles初始化
            if 'remaining_cycles' not in task:
                task['remaining_cycles'] = float(task.get('compute_cycles', 1e9))
            
            previous_cycles = task['remaining_cycles']
            
            # 计算本时隙可分配给此任务的周期数
            # 容量限制：节点每时隙只能处理 cycles_per_slot 周期
            available_cycles = max(0.0, cycles_per_slot - total_cycles_used)
            cycles_to_process = min(previous_cycles, available_cycles)
            
            remaining_cycles = max(0.0, previous_cycles - cycles_to_process)
            task['remaining_cycles'] = remaining_cycles
            total_cycles_used += cycles_to_process
            
            # 计算实际处理时间和服务时间
            actual_processing_time = cycles_to_process / cpu_freq if cpu_freq > 0 else 0.0
            task['service_time'] = task.get('service_time', 0.0) + actual_processing_time
            
            # 兼容性：保留work_remaining用于其他模块
            original_cycles = float(task.get('compute_cycles', 1e9))
            if original_cycles > 0:
                task['work_remaining'] = remaining_cycles / original_cycles
            else:
                task['work_remaining'] = 0.0
            
            consumed_ratio = cycles_to_process / max(previous_cycles, 1e-9)
            consumed_ratio = float(np.clip(consumed_ratio, 0.0, 1.0))
            incremental_service = actual_processing_time

            # 🔧 修复：计算RSU/UAV处理能耗
            # Fix: Calculate energy consumption for RSU/UAV processing
            if node_type in ['RSU', 'UAV']:
                # 获取节点配置
                if node_type == 'RSU':
                    cpu_freq = getattr(self.sys_config.compute, 'rsu_cpu_freq', 12.5e9) if self.sys_config else 12.5e9
                    static_power = getattr(self.sys_config.compute, 'rsu_static_power', 25.0) if self.sys_config else 25.0
                else:
                    cpu_freq = getattr(self.sys_config.compute, 'uav_cpu_freq', 5.0e9) if self.sys_config else 5.0e9
                    static_power = getattr(self.sys_config.compute, 'uav_static_power', 2.5) if self.sys_config else 2.5
                
                # 动态功耗系数
                kappa = 1e-28
                dynamic_power = kappa * (cpu_freq ** 3)
                
                # 计算本时隙消耗的能耗
                step_energy = (dynamic_power + static_power) * incremental_service
                
                # 累加能耗
                self._accumulate_energy('energy_compute', step_energy)
                node['energy_consumed'] = node.get('energy_consumed', 0.0) + step_energy

            if task.get('remaining_cycles', 0.0) > 0.0:
                new_queue.append(task)
                continue

            # 🆕 Luo论文队列模型：任务完成时同步从clifetime_queues中移除
            self._remove_task_from_lifetime_queues(node, task)
            
            # DEBUG LOGGING - ENTRY
            # print(f"[DEBUG] Processing task {task.get('id')} at {node_type}, Content: {task.get('content_id')}")

            # 🔧 修复：任务完成后尝试缓存内容
            if node_type in ['RSU', 'UAV']:
                cache_ctrl = getattr(self, 'adaptive_cache_controller', None)
                content_id = task.get('content_id')
                if cache_ctrl and content_id:
                    try:
                        # 获取内容大小和缓存状态
                        data_size = self._get_realistic_content_size(content_id)
                        cache_snapshot = node.get('cache', {})
                        capacity = float(node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0))
                        used = sum(float(item.get('size', 0.0)) for item in cache_snapshot.values())
                        available = max(0.0, capacity - used)
                        
                        # 决策是否缓存
                        should_cache, reason, evictions = cache_ctrl.should_cache_content(
                            content_id, data_size, available, cache_snapshot, capacity,
                            cache_priority=task.get('priority', 0.5)
                        )
                        
                        # DEBUG LOGGING
                        print(f"[DEBUG] Content: {content_id}, Should: {should_cache}, Reason: {reason}")
                        
                        if should_cache:
                            if 'cache' not in node:
                                node['cache'] = {}
                            cache_dict = node['cache']
                            
                            # 执行淘汰
                            reclaimed = 0.0
                            for evict_id in evictions:
                                removed = cache_dict.pop(evict_id, None)
                                if removed:
                                    reclaimed += float(removed.get('size', 0.0) or 0.0)
                                    cache_ctrl.cache_stats['evicted_items'] += 1
                            
                            if reclaimed > 0.0:
                                available += reclaimed
                                
                            # 写入缓存
                            if available >= data_size:
                                cache_dict[content_id] = {
                                    'size': data_size,
                                    'timestamp': self.current_time,
                                    'reason': reason or 'post_process_cache',
                                    'content_type': self._infer_content_type(content_id)
                                }
                                # 更新热度
                                cache_ctrl.update_content_heat(content_id)
                                print(f"[DEBUG] Cached {content_id} at {node_type}")
                    except Exception as e:
                        print(f"[DEBUG] Cache error: {e}")
                        pass

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
            
            # 按任务类别记录时延统计
            self._record_task_type_delay(task, actual_delay)

            vehicle_id = task.get('vehicle_id', 'V_0')
            vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)

            # 🔥 深度修复：正确的CMOS能耗模型
            # E_total = (P_dynamic + P_static) × t_processing
            # P_dynamic = κ × f³，但 t_processing = C / f
            # 因此能耗应随频率增加而优化，而非暴涨
            
            if node_type == 'RSU':
                # RSU能耗参数
                cpu_freq = node.get('cpu_freq', 12.5e9)  # 12.5 GHz
                kappa = 5.0e-32  # W/(Hz)³
                static_power = 25.0  # W
                
                # 🔧 修复: 增强配置一致性检查
                if self.sys_config is not None and hasattr(self.sys_config, 'compute'):
                    cpu_freq = getattr(self.sys_config.compute, 'rsu_cpu_freq', cpu_freq)
                    kappa = getattr(self.sys_config.compute, 'rsu_kappa', kappa)
                    static_power = getattr(self.sys_config.compute, 'rsu_static_power', static_power)
                
                # 🔧 修复v3：使用任务实际的compute_cycles计算处理时间和能耗
                task_compute_cycles = float(task.get('compute_cycles', 1e9))
                # 实际处理时间 = 计算周期 / CPU频率
                task_processing_time = task_compute_cycles / cpu_freq
                
                # 动态功耗 = κ × f³
                dynamic_power = kappa * (cpu_freq ** 3)
                # 总能耗 = (动态功耗 + 静态功耗) × 实际处理时间
                task_energy = (dynamic_power + static_power) * task_processing_time
                
            elif node_type == 'UAV':
                # 🔧 优化: 统一从配置读取UAV能耗参数
                # UAV能耗参数（包含悬停功耗）
                
                # 默认值：基于NVIDIA Jetson Xavier NX
                default_cpu_freq = 3.5e9   # 3.5 GHz（匹配配置）
                default_kappa3 = 8.89e-31  # W/(Hz)³
                default_static = 2.5       # W
                default_hover = 15.0       # W - 轻量级四旋翼（匹配配置）
                
                # 优先从配置读取
                if self.sys_config is not None and hasattr(self.sys_config, 'compute'):
                    cpu_freq = getattr(self.sys_config.compute, 'uav_cpu_freq', default_cpu_freq)
                    kappa3 = getattr(self.sys_config.compute, 'uav_kappa3', default_kappa3)
                    static_power = getattr(self.sys_config.compute, 'uav_static_power', default_static)
                    hover_power = getattr(self.sys_config.compute, 'uav_hover_power', default_hover)
                else:
                    cpu_freq = node.get('cpu_freq', default_cpu_freq)
                    kappa3 = default_kappa3
                    static_power = default_static
                    hover_power = default_hover
                
                # 🔧 修复v3：使用任务实际的compute_cycles
                task_compute_cycles = float(task.get('compute_cycles', 1e9))
                task_processing_time = task_compute_cycles / cpu_freq
                
                # 动态功耗 = κ × f³
                dynamic_power = kappa3 * (cpu_freq ** 3)
                # UAV总能耗 = (动态 + 静态 + 悬停) × 实际处理时间
                task_energy = (dynamic_power + static_power + hover_power) * task_processing_time
                
            else:
                # 其他节点类型使用简化模型
                task_compute_cycles = float(task.get('compute_cycles', 1e9))
                task_energy = 1e-9 * task_compute_cycles  # 简化：每cycle约1nJ
            self._accumulate_energy('energy_compute', task_energy)
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + task_energy

            # 🔧 修复：添加下行传输能耗（将处理结果传回车辆）
            # Fix: Add downlink transmission energy (return result to vehicle)
            result_size = task.get('data_size_bytes', 1e6) * 0.05  # Result is typically 5% of input
            if result_size > 0:
                # Find the vehicle to calculate distance
                vehicle_id = task.get('vehicle_id', 'V_0')
                vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
                
                if vehicle:
                    v_pos = np.array(vehicle.get('position', [0.0, 0.0, 0.0]))
                    n_pos = np.array(node.get('position', [0.0, 0.0, 0.0]))
                    distance = self.calculate_distance(v_pos, n_pos)
                    
                    down_delay, down_energy = self._estimate_transmission(
                        result_size, distance, node_type.lower()
                    )
                    
                    # Accumulate downlink delay and energy
                    self._accumulate_delay('delay_downlink', down_delay)
                    self._accumulate_energy('energy_transmit_downlink', down_energy)
                    self.stats['energy_downlink'] = self.stats.get('energy_downlink', 0.0) + down_energy
                    node['energy_consumed'] = node.get('energy_consumed', 0.0) + down_energy

            task['completed'] = True

        node['computation_queue'] = new_queue
        self._record_mm1_queue_length(node_type, node_idx, len(new_queue))


    def find_nearest_rsu(self, vehicle_pos: np.ndarray) -> Optional[Dict]:
        """
        ??????????????????RSU?
        Fallback to brute-force iteration when the index is unavailable.
        """
        if not self.rsus:
            return None

        vehicle_vec = np.asarray(vehicle_pos, dtype=float)
        best_node: Optional[Dict] = None
        best_distance = float('inf')

        spatial_index = getattr(self, 'spatial_index', None)
        if spatial_index is not None:
            nearest = spatial_index.find_nearest_rsu(vehicle_vec, return_distance=True)
            if nearest:
                _, node, dist = nearest
                coverage = float(node.get('coverage_radius', self.coverage_radius))
                if dist <= coverage:
                    return node
                best_node = node
                best_distance = dist

            max_radius = spatial_index.rsu_max_radius or max(
                (float(rsu.get('coverage_radius', self.coverage_radius)) for rsu in self.rsus),
                default=self.coverage_radius,
            )
            neighbors = spatial_index.query_rsus_within_radius(vehicle_vec, max_radius)
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

    def find_nearest_uav(self, vehicle_pos: np.ndarray) -> Optional[Dict]:
        """
        ???????????UAV???
        """
        if not self.uavs:
            return None

        vehicle_vec = np.asarray(vehicle_pos, dtype=float)
        spatial_index = getattr(self, 'spatial_index', None)
        if spatial_index is not None:
            nearest = spatial_index.find_nearest_uav(vehicle_vec, return_distance=True)
            if nearest:
                return nearest[1]

        min_distance = float('inf')
        nearest_uav: Optional[Dict] = None
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
        agents_actions: Optional[Dict] = None,
        node_type: str = 'RSU',
        task: Optional[Dict] = None  # 🔧 优化9: 添加task参数以获取cache_priority
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
        # 🔧 优化6: 不可缓存内容直接返回未命中，不参与统计
        if not content_id:
            return False
        
        # 基础缓存检查
        # Basic cache check
        cache = node.get('cache', {})
        cache_hit = bool(content_id and cache and content_id in cache)
        
        # 🔧 修复：只统计有content_id的任务，避免统计扭曲
        # 不可缓存的任务不应该影响缓存命中率统计
        self._register_cache_request(cache_hit)
        
        # 更新统计
        # Update statistics
        if cache_hit:
            self.stats['cache_hits'] += 1
            # 🔧 新增：更新RSU缓存命中率统计（用于状态编码）
            if node_type == 'RSU':
                node['cache_hits_window'] = node.get('cache_hits_window', 0) + 1
                node['cache_requests_window'] = node.get('cache_requests_window', 0) + 1
                # 每100次请求更新一次命中率（滚动窗口）
                if node['cache_requests_window'] >= 100:
                    node['recent_cache_hit_rate'] = node['cache_hits_window'] / node['cache_requests_window']
                    # 重置窗口
                    node['cache_hits_window'] = 0
                    node['cache_requests_window'] = 0
                elif node['cache_requests_window'] > 0:
                    # 实时更新（但不重置）
                    node['recent_cache_hit_rate'] = node['cache_hits_window'] / node['cache_requests_window']
                self._propagate_cache_after_hit(content_id, node, agents_actions)
        else:
            self.stats['cache_misses'] += 1
            # 🔧 新增：更新RSU缓存统计（未命中）
            if node_type == 'RSU':
                node['cache_requests_window'] = node.get('cache_requests_window', 0) + 1
                if node['cache_requests_window'] >= 100:
                    node['recent_cache_hit_rate'] = node.get('cache_hits_window', 0) / node['cache_requests_window']
                    node['cache_hits_window'] = 0
                    node['cache_requests_window'] = 0
                elif node['cache_requests_window'] > 0:
                    node['recent_cache_hit_rate'] = node.get('cache_hits_window', 0) / node['cache_requests_window']
            
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
                    # 🔧 优化10: 传入cache_priority加强缓存决策
                    cache_priority = task.get('cache_priority', 0.0) if task else 0.0
                    should_cache, reason, evictions = cache_controller.should_cache_content(
                        content_id,
                        data_size,
                        available_capacity,
                        node.get('cache', {}),
                        capacity_limit,
                        cache_priority  # 传入优先级
                    )
                
                # 缓存写入温启动：前warmup次请求尽量缓存，避免冷启动长期0命中
                total_requests_so_far = cache_controller.cache_stats.get('total_requests', 0)
                warmup_threshold = 100
                if total_requests_so_far < warmup_threshold and available_capacity >= data_size:
                    should_cache = True
                    reason = reason or 'warmup_cache'
                    evictions = []

                # RL引导：概率缩放而不是硬性拦截
                if not should_cache and cache_preference > 0.7 and available_capacity >= data_size:
                    should_cache = True
                    reason = reason or 'RL-guided cache'
                    evictions = []
                elif should_cache and cache_preference < 0.2:
                    # 在极低偏好时可放弃
                    should_cache = False
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

    def _update_node_connections(self):
        """
        🔧 修复: 更新RSU和UAV的即时连接计数
        
        根据当前车辆位置计算哪些车辆在各节点的覆盖范围内，
        并更新 served_vehicles 和 coverage_vehicles 计数器。
        
        优先级：RSU > UAV（避免重复计数）
        
        Update immediate connection counts for RSUs and UAVs based on coverage.
        Priority: RSU > UAV (avoid double counting).
        """
        # 清空连接列表（已经在run_simulation_step开头重置了计数器）
        for rsu in self.rsus:
            rsu['connected_vehicles'] = []
        for uav in self.uavs:
            uav['connected_vehicles'] = []
        
        # 遍历所有车辆，检查覆盖
        for vehicle in self.vehicles:
            v_pos = vehicle.get('position')
            if v_pos is None or len(v_pos) < 2:
                continue
            
            vehicle_id = vehicle.get('id', '')
            connected_to_rsu = False
            
            # 1. 检查RSU覆盖（优先级最高）
            for rsu in self.rsus:
                distance = self.calculate_distance(v_pos, rsu['position'])
                rsu_radius = rsu.get('coverage_radius', self.coverage_radius)
                if distance <= rsu_radius:
                    rsu['served_vehicles'] += 1
                    rsu['coverage_vehicles'] += 1
                    rsu['connected_vehicles'].append(vehicle_id)
                    connected_to_rsu = True
                    break  # 只连接到最近的RSU
            
            # 2. 如果没有RSU覆盖，检查UAV覆盖
            if not connected_to_rsu:
                for uav in self.uavs:
                    uav_pos = uav['position']
                    # 3D距离计算
                    if len(uav_pos) >= 3 and len(v_pos) == 2:
                        distance_2d = np.sqrt((v_pos[0] - uav_pos[0])**2 + (v_pos[1] - uav_pos[1])**2)
                        distance_3d = np.sqrt(distance_2d**2 + uav_pos[2]**2)
                    else:
                        distance_3d = self.calculate_distance(v_pos, uav_pos[:2] if len(uav_pos) >= 2 else uav_pos)
                    
                    uav_radius = uav.get('coverage_radius', self.uav_coverage_radius)
                    if distance_3d <= uav_radius:
                        uav['served_vehicles'] += 1
                        uav['connected_vehicles'].append(vehicle_id)
                        break  # 只连接到最近的UAV

    def _update_vehicle_positions(self):
        """
        简单更新车辆位置，模拟车辆沿主干道移动
        
        实现了逼真的车辆移动模型，包括：
        - 速度的加减速变化
        - 路口减速行为（根据车辆行驶方向智能判断）
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

            # 🔧 修复：在接近路口时降低速度，根据车辆行驶方向智能判断距离
            # Slow down near intersections based on vehicle heading direction
            direction = vehicle.get('direction', 0.0)
            for intersection in self.intersections.values():
                # 判断车辆主要行驶方向：东西向(0或π) vs 南北向(π/2或-π/2)
                is_horizontal = abs(np.cos(direction)) > abs(np.sin(direction))  # 东西向
                
                if is_horizontal:
                    # 横向行驶的车辆检查Y坐标距离
                    dist_to_signal = abs(position[1] - intersection['y'])
                else:
                    # 纵向行驶的车辆检查X坐标距离
                    dist_to_signal = abs(position[0] - intersection['x'])
                
                if dist_to_signal < 40.0:
                    accel_state = min(accel_state, -0.8)
                    break

            new_speed = np.clip(base_speed + accel_state, 5.0, 20.0)  # 降低最大速度到20m/s (~72km/h)
            vehicle['speed_accel'] = accel_state
            vehicle['velocity'] = new_speed

            # === 2) 方向保持，同时允许轻微扰动 ===
            heading_jitter = vehicle.setdefault('heading_jitter', 0.0)
            heading_jitter = 0.6 * heading_jitter + np.random.uniform(-0.01, 0.01)
            direction = (direction + heading_jitter) % (2 * np.pi)
            vehicle['direction'] = direction
            vehicle['heading_jitter'] = heading_jitter

            dx = np.cos(direction) * new_speed * self.time_slot
            dy = np.sin(direction) * new_speed * self.time_slot

            # === 3) 横向漂移（模拟轻微换道） ===
            # 根据车辆行驶方向决定车道偏移的应用方式
            is_horizontal = abs(np.cos(direction)) > abs(np.sin(direction))
            lane_bias = vehicle.get('lane_bias', 0.0)
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

            # === 4) 应用位置更新 ===
            # 🔧 修复：使用正确的场景尺寸边界 (1030 x 2060)
            new_x = position[0] + dx
            new_y = position[1] + dy
            
            # 🔧 修复：应用车道偏移（垂直于车辆前进方向）
            # 车道偏移应该垂直于前进方向，模拟车道内的左右微调
            if is_horizontal:
                # 横向行驶（东西向）：车道偏移应用到Y方向（垂直于前进方向）
                new_y += lane_bias + lateral_state
            else:
                # 纵向行驶（南北向）：车道偏移应用到X方向（垂直于前进方向）
                new_x += lane_bias + lateral_state
            
            # 🔧 修复：周期性边界条件（匹配场景实际尺寸）
            new_x = new_x % self.scenario_width   # 0 ~ 1030m
            new_y = new_y % self.scenario_height  # 0 ~ 2060m

            vehicle['position'][0] = new_x
            vehicle['position'][1] = new_y

        self._refresh_spatial_index(update_static=False, update_vehicle=True)
        
        # 🔧 修复2: 更新RSU/UAV的即时连接计数
        # Update immediate connection counts after vehicle movement
        self._update_node_connections()

    def _sample_arrivals(self) -> int:
        """鎸夋硦鏉捐繃绋嬮噰鏍锋瘡杞︽瘡鏃堕殭鐨勪换鍔″埌杈炬暟"""
        lam = max(1e-6, float(self.task_arrival_rate) * float(self.time_slot))
        return int(np.random.poisson(lam))

    def _choose_offload_target(self, actions: Dict, rsu_available: bool, uav_available: bool) -> str:
        """
        根据智能体偏好选择卸载目标
        
        🔧 优化：添加队列感知的决策逻辑
        - 考虑各类节点的队列负载状态
        - 动态调整卸载概率避免过载节点
        - 智能体偏好仍然是主要决策因素
        """
        import os
        
        prefs = actions.get('vehicle_offload_pref') or {}
        base_probs = np.array([
            max(0.0, float(prefs.get('local', 0.0))),
            max(0.0, float(prefs.get('rsu', 0.0))) if rsu_available else 0.0,
            max(0.0, float(prefs.get('uav', 0.0))) if uav_available else 0.0,
        ], dtype=float)
        
        # 🔧 修复NaN问题：清理初始概率中的NaN值
        base_probs = np.nan_to_num(base_probs, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 🔧 优化：队列感知的决策调整
        # 计算各类节点的平均队列负载
        queue_factors = np.ones(3, dtype=float)
        
        # 1. 计算RSU平均队列负载（如果可用）
        if rsu_available and self.rsus:
            rsu_queue_loads = []
            for rsu in self.rsus:
                queue_len = len(rsu.get('computation_queue', []))
                capacity = self.rsu_nominal_capacity
                load = queue_len / max(1.0, capacity)
                rsu_queue_loads.append(load)
            avg_rsu_load = np.mean(rsu_queue_loads) if rsu_queue_loads else 0.0
            # 负载越高，选择概率越低（但不完全拒绝）
            # 使用sigmoid-like衰减：当负载>1时开始显著降低
            queue_factors[1] = 1.0 / (1.0 + max(0.0, avg_rsu_load - 0.5))
        
        # 2. 计算UAV平均队列负载（如果可用）
        if uav_available and self.uavs:
            uav_queue_loads = []
            for uav in self.uavs:
                queue_len = len(uav.get('computation_queue', []))
                capacity = self.uav_nominal_capacity
                load = queue_len / max(1.0, capacity)
                uav_queue_loads.append(load)
            avg_uav_load = np.mean(uav_queue_loads) if uav_queue_loads else 0.0
            queue_factors[2] = 1.0 / (1.0 + max(0.0, avg_uav_load - 0.5))
        
        # 3. 本地处理（车辆）的负载因子保持为1.0
        # 本地处理通常作为fallback，不需要额外调整
        
        # 🔧 控制队列感知的影响程度（可通过环境变量调整）
        queue_weight = float(os.environ.get('QUEUE_AWARE_WEIGHT', '0.3'))
        adjusted_factors = 1.0 - queue_weight + queue_weight * queue_factors
        
        # 应用队列感知调整
        probs = base_probs * adjusted_factors
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # 🔧 禁用guidance干扰：对比实验时不应用guidance修正，保持智能体原始决策
        apply_guidance = os.environ.get('APPLY_RL_GUIDANCE', '0') == '1'
        
        if apply_guidance:
            guidance = actions.get('rl_guidance') or {}
            if isinstance(guidance, dict):
                guide_prior = np.array(guidance.get('offload_prior', []), dtype=float)
                if guide_prior.size >= 3:
                    guide_prior = np.nan_to_num(guide_prior[:3], nan=1.0, posinf=1.0, neginf=1.0)
                    probs *= np.clip(guide_prior, 1e-4, None)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                
                distance_focus = np.array(guidance.get('distance_focus', []), dtype=float)
                if distance_focus.size >= 3:
                    distance_focus = np.nan_to_num(distance_focus[:3], nan=1.0, posinf=1.0, neginf=1.0)
                    probs *= np.clip(distance_focus, 0.2, None)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                
                cache_focus = np.array(guidance.get('cache_focus', []), dtype=float)
                if cache_focus.size >= 3:
                    cache_focus = np.nan_to_num(cache_focus[:3], nan=1.0, posinf=1.0, neginf=1.0)
                    probs *= np.clip(cache_focus, 0.2, None)
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
                
                energy_pressure_vec = guidance.get('energy_pressure')
                if isinstance(energy_pressure_vec, (list, tuple, np.ndarray)):
                    pressure_arr = np.asarray(energy_pressure_vec, dtype=float).reshape(-1)
                    pressure_arr = np.nan_to_num(pressure_arr, nan=1.0, posinf=1.0, neginf=1.0)
                    pressure = float(np.clip(pressure_arr[0], 0.35, 1.8))
                    energy_weights = np.array([1.0 / pressure, pressure, pressure], dtype=float)
                    energy_weights = np.nan_to_num(energy_weights, nan=1.0, posinf=1.0, neginf=1.0)
                    probs *= energy_weights
                    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)

        # 最终检查：如果概率总和仍然为0或无效，使用默认概率
        if not np.isfinite(probs).all() or probs.sum() <= 0:
            probs = np.array([
                0.34,
                0.33 if rsu_available else 0.0,
                0.33 if uav_available else 0.0
            ], dtype=float)

        if probs.sum() <= 0:
            return 'local'

        # 归一化前再次清理NaN
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs / probs.sum()
        
        # 最后一次安全检查
        if not np.isfinite(probs).all():
            return 'local'
        
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
        # 🔧 修复: 增强配置一致性检查
        if self.sys_config is not None and hasattr(self.sys_config, 'compute'):
            cpu_freq = getattr(self.sys_config.compute, 'vehicle_cpu_freq', cpu_freq)
            power = getattr(self.sys_config.compute, 'vehicle_static_power', power)
        else:
            cpu_freq = float(self.config.get('vehicle_cpu_freq', cpu_freq))
            power = float(self.config.get('vehicle_static_power', power))

        requirement = float(task.get('computation_requirement', 1500.0)) * 1e6  # cycles
        # 🔧 修复问题2：应用并行效率参数（与能耗模型保持一致）
        parallel_eff = 0.8
        if self.sys_config is not None and hasattr(self.sys_config, 'compute'):
            parallel_eff = getattr(self.sys_config.compute, 'parallel_efficiency', 0.8)
        else:
            parallel_eff = float(self.config.get('parallel_efficiency', 0.8))
        processing_time = requirement / max(cpu_freq * parallel_eff, 1e6)
        # Allow genuine compute latency to surface by avoiding artificial clipping
        processing_time = max(float(processing_time), 1e-6)
        
        # 🔥 关键修复：使用完整的动态+静态功耗模型
        # E_total = P_dynamic × t_active + P_static × t_active
        # P_dynamic = κ₁ × f³
        kappa1 = 1.5e-28  # W/(Hz)³ - 动态功耗系数
        if self.sys_config is not None and hasattr(self.sys_config, 'compute'):
            kappa1 = getattr(self.sys_config.compute, 'vehicle_kappa1', kappa1)
        else:
            kappa1 = float(self.config.get('vehicle_kappa1', kappa1))
        
        dynamic_power = kappa1 * (cpu_freq ** 3)  # 动态功耗：P = κ₁ × f³
        energy = (dynamic_power + power) * processing_time  # 总能耗 = (动态+静态) × 时间
        
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + energy
        return processing_time, energy

    def _estimate_transmission(self, data_size_bytes: float, distance: float, link: str, 
                              vehicle: Optional[Dict] = None) -> Tuple[float, float]:
        """
        估计上传耗时与能耗
        
        🔧 P0修复：支持动态带宽分配，使用vehicle['allocated_bandwidth']
        """
        # 🔧 P0修复：优先使用车辆的动态分配带宽
        if vehicle is not None and 'allocated_bandwidth' in vehicle:
            # 使用动态分配的带宽
            allocated_bandwidth = float(vehicle['allocated_bandwidth'])
            total_bandwidth = float(getattr(self.resource_pool, 'total_bandwidth', self.bandwidth))
            base_rate = allocated_bandwidth * total_bandwidth
            # print(f"✅ 使用动态分配带宽: {base_rate/1e6:.2f} MHz (ratio={allocated_bandwidth:.3f})")
        else:
            # 回退到默认带宽（从配置读取）
            if link == 'uav':
                # UAV下行带宽：优先从配置读取，默认50 MHz
                if self.sys_config is not None and hasattr(self.sys_config, 'communication'):
                    base_rate = getattr(self.sys_config.communication, 'uav_downlink_bandwidth', 50e6)
                else:
                    base_rate = float(self.config.get('uav_downlink_bandwidth', 50e6))
            else:  # RSU
                # RSU下行带宽：优先从配置读取，默认1000 MHz (1 GHz)
                if self.sys_config is not None and hasattr(self.sys_config, 'communication'):
                    base_rate = getattr(self.sys_config.communication, 'rsu_downlink_bandwidth', 1000e6)
                else:
                    base_rate = float(self.config.get('rsu_downlink_bandwidth', 1000e6))
        
        # 设置发射功率
        if link == 'uav':
            power_w = 0.12
        else:  # RSU
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
                stats_cfg = getattr(self, 'stats_config', None)
                # 🔧 修复: 增强配置一致性检查
                log_interval = 400  # 默认值
                if stats_cfg is not None and hasattr(stats_cfg, 'drop_log_interval'):
                    log_interval = stats_cfg.drop_log_interval
                else:
                    log_interval = self.config.get('drop_log_interval', 400)
                log_interval = max(1, int(log_interval))
                for task in queue:
                    # 🔧 修复:检查任务是否已经被丢弃,避免重复计数
                    if task.get('dropped', False):
                        continue
                    
                    if self.current_time > task.get('deadline', float('inf')):
                        task['dropped'] = True
                        task['drop_reason'] = 'deadline_exceeded'
                        
                        # 🆕 Luo论文队列模型：过期任务从clifetime_queues中移除
                        self._remove_task_from_lifetime_queues(node, task)
                        
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
        spatial_index = getattr(self, 'spatial_index', None)
        if spatial_index is not None:
            neighbor_candidates = spatial_index.query_rsus_within_radius(rsu_node['position'], coverage * 1.2)
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
        """根据动作分配任务"""
        cache_controller = None
        if isinstance(actions, dict):
            cache_controller = actions.get('cache_controller')
        if cache_controller is None:
            cache_controller = getattr(self, 'adaptive_cache_controller', None)

        content_id = task.get('content_id')
        
        # 🔧 修复：更新内容热度，确保缓存控制器能感知到内容访问
        if cache_controller is not None and content_id:
            try:
                cache_controller.update_content_heat(content_id)
            except Exception:
                pass

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
        """
        将任务分配到RSU处理
        
        🔧 优化：增强队列感知的节点选择逻辑
        - 结合智能体偏好和队列负载状态
        - 距离因素作为辅助参考
        - 避免向过载节点卸载任务
        """
        if not self.rsus:
            return False

        vehicle_pos = np.asarray(vehicle.get('position', [0.0, 0.0]), dtype=float)
        candidates = []
        spatial_index = getattr(self, 'spatial_index', None)
        if spatial_index is not None:
            max_radius = spatial_index.rsu_max_radius or max(
                (float(rsu.get('coverage_radius', self.coverage_radius)) for rsu in self.rsus),
                default=self.coverage_radius,
            )
            candidates = spatial_index.query_rsus_within_radius(vehicle_pos, max_radius)
            if not candidates:
                nearest = spatial_index.find_nearest_rsu(vehicle_pos, return_distance=True)
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

        # 🔧 优化：初始化权重数组
        probs = np.ones_like(distances)
        
        # 1. 应用智能体的RSU选择偏好
        rsu_pref = actions.get('rsu_selection_probs')
        if isinstance(rsu_pref, (list, tuple, np.ndarray)) and len(rsu_pref) == len(self.rsus):
            pref_values = np.array([max(0.0, float(rsu_pref[idx])) for idx in candidate_indices], dtype=float)
            pref_values = np.nan_to_num(pref_values, nan=1.0, posinf=1.0, neginf=1.0)
            pref_values = np.maximum(pref_values, 1e-10)
            # 🔧 优化：增强智能体偏好的影响力（使用幂次放大）
            pref_values = np.power(pref_values, 1.5)  # 放大偏好差异
            probs *= pref_values
            probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 2. 🔧 优化：添加队列负载因子
        queue_factors = np.ones_like(distances)
        for i, idx in enumerate(candidate_indices):
            rsu = self.rsus[idx]
            queue_len = len(rsu.get('computation_queue', []))
            capacity = self.rsu_nominal_capacity
            load = queue_len / max(1.0, capacity)
            # 负载越高，选择概率越低
            # 使用软衰减：queue_factor = exp(-load * decay_rate)
            queue_factors[i] = np.exp(-load * 0.5)  # decay_rate=0.5
        
        probs *= queue_factors
        probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 3. 🔧 优化：添加距离因子（距离越近越好）
        max_dist = max(distances) if len(distances) > 0 else 1.0
        distance_factors = 1.0 - 0.3 * (distances / max(max_dist, 1e-6))  # 最远节点衰减30%
        distance_factors = np.clip(distance_factors, 0.5, 1.0)
        probs *= distance_factors
        probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)

        # 4. 应用rl_guidance（如果启用）
        guidance = actions.get('rl_guidance') or {}
        if isinstance(guidance, dict):
            rsu_prior = np.array(guidance.get('rsu_prior', []), dtype=float)
            if rsu_prior.size >= len(self.rsus):
                rsu_prior = np.nan_to_num(rsu_prior, nan=1.0, posinf=1.0, neginf=1.0)
                prior_vals = np.clip(rsu_prior[candidate_indices], 1e-4, None)
                probs *= prior_vals
                probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
                probs = np.maximum(probs, 1e-10)
            
            cache_focus = guidance.get('cache_focus')
            if isinstance(cache_focus, (list, tuple)) and len(cache_focus) >= 2:
                cache_weight = float(np.clip(cache_focus[1], 0.0, 1.0))
                cache_weight = np.nan_to_num(cache_weight, nan=0.0)
                power_val = 0.8 + 0.4 * cache_weight
                probs = np.maximum(probs, 1e-10)
                probs = np.power(probs, power_val)
                probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
                probs = np.maximum(probs, 1e-10)
            
            distance_focus = guidance.get('distance_focus')
            if isinstance(distance_focus, (list, tuple)) and len(distance_focus) >= 2:
                distance_weight = float(np.clip(distance_focus[1], 0.0, 1.0))
                distance_weight = np.nan_to_num(distance_weight, nan=0.0)
                power_val = 0.8 + 0.4 * distance_weight
                probs = np.maximum(probs, 1e-10)
                probs = np.power(probs, power_val)
                probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
                probs = np.maximum(probs, 1e-10)

        weights = probs
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = np.maximum(weights, 1e-10)
        
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.ones_like(weights)
            weight_sum = weights.sum()

        weights = weights / weight_sum
        weights = np.nan_to_num(weights, nan=1.0/len(weights), posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
        
        final_sum = weights.sum()
        if final_sum > 0 and np.isfinite(final_sum):
            weights = weights / final_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        if not np.isfinite(weights).all():
            weights = np.ones_like(weights) / len(weights)
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
        """
        将任务分配到UAV处理
        
        🔧 优化：增强队列感知的节点选择逻辑
        - 结合智能体偏好和队列负载状态
        - 距离因素作为辅助参考
        - 避免向过载节点卸载任务
        """
        if not self.uavs:
            return False

        vehicle_pos = np.asarray(vehicle.get('position', [0.0, 0.0]), dtype=float)
        candidates = []
        spatial_index = getattr(self, 'spatial_index', None)
        if spatial_index is not None:
            max_radius = spatial_index.uav_max_radius or max(
                (float(uav.get('coverage_radius', 350.0)) for uav in self.uavs),
                default=350.0,
            )
            candidates = spatial_index.query_uavs_within_radius(vehicle_pos, max_radius)
            if not candidates:
                nearest = spatial_index.find_nearest_uav(vehicle_pos, return_distance=True)
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

        # 🔧 优化：初始化权重数组
        probs = np.ones_like(distances)
        
        # 1. 应用智能体的UAV选择偏好
        uav_pref = actions.get('uav_selection_probs')
        if isinstance(uav_pref, (list, tuple, np.ndarray)) and len(uav_pref) == len(self.uavs):
            pref_values = np.array([max(0.0, float(uav_pref[idx])) for idx in candidate_indices], dtype=float)
            pref_values = np.nan_to_num(pref_values, nan=1.0, posinf=1.0, neginf=1.0)
            pref_values = np.maximum(pref_values, 1e-10)
            # 🔧 优化：增强智能体偏好的影响力（使用幂次放大）
            pref_values = np.power(pref_values, 1.5)  # 放大偏好差异
            probs *= pref_values
            probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 2. 🔧 优化：添加队列负载因子
        queue_factors = np.ones_like(distances)
        for i, idx in enumerate(candidate_indices):
            uav = self.uavs[idx]
            queue_len = len(uav.get('computation_queue', []))
            capacity = self.uav_nominal_capacity
            load = queue_len / max(1.0, capacity)
            # 负载越高，选择概率越低
            # UAV容量较小，使用更强的衰减
            queue_factors[i] = np.exp(-load * 0.7)  # decay_rate=0.7 (比RSU更强)
        
        probs *= queue_factors
        probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
        
        # 3. 🔧 优化：添加距离因子（距离越近越好）
        max_dist = max(distances) if len(distances) > 0 else 1.0
        distance_factors = 1.0 - 0.4 * (distances / max(max_dist, 1e-6))  # UAV距离影响更大
        distance_factors = np.clip(distance_factors, 0.4, 1.0)
        probs *= distance_factors
        probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)

        # 4. 应用rl_guidance（如果启用）
        guidance = actions.get('rl_guidance') or {}
        if isinstance(guidance, dict):
            uav_prior = np.array(guidance.get('uav_prior', []), dtype=float)
            if uav_prior.size >= len(self.uavs):
                uav_prior = np.nan_to_num(uav_prior, nan=1.0, posinf=1.0, neginf=1.0)
                prior_vals = np.clip(uav_prior[candidate_indices], 1e-4, None)
                probs *= prior_vals
                probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
                probs = np.maximum(probs, 1e-10)
            
            distance_focus = guidance.get('distance_focus')
            if isinstance(distance_focus, (list, tuple)) and len(distance_focus) >= 3:
                distance_weight = float(np.clip(distance_focus[2], 0.0, 1.0))
                distance_weight = np.nan_to_num(distance_weight, nan=0.0)
                power_val = 0.8 + 0.4 * distance_weight
                probs = np.maximum(probs, 1e-10)
                probs = np.power(probs, power_val)
                probs = np.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=1.0)
                probs = np.maximum(probs, 1e-10)

        weights = probs
        weights = np.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
        weights = np.maximum(weights, 1e-10)
        
        weight_sum = weights.sum()
        if weight_sum <= 0 or not np.isfinite(weight_sum):
            weights = np.ones_like(weights)
            weight_sum = weights.sum()

        weights = weights / weight_sum
        weights = np.nan_to_num(weights, nan=1.0/len(weights), posinf=0.0, neginf=0.0)
        weights = np.clip(weights, 0.0, 1.0)
        
        final_sum = weights.sum()
        if final_sum > 0 and np.isfinite(final_sum):
            weights = weights / final_sum
        else:
            weights = np.ones_like(weights) / len(weights)
        
        if not np.isfinite(weights).all():
            weights = np.ones_like(weights) / len(weights)
        
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
        # 🔧 优化11: 传入task参数以使用cache_priority
        if node_type == 'RSU':
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='RSU', task=task)
        else:
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='UAV', task=task)

        if cache_hit:
            # ✅ 修复：缓存命中几乎无能耗，只有极短的内存访问延迟
            # Cache hit: minimal delay (memory access ~1ms), negligible energy
            delay = 0.001  # 1ms - 内存访问延迟
            
            # ✅ 缓存读取能耗可忽略不计（存储器访问功耗 << 0.01J）
            # Cache read energy is negligible (memory access power << 0.01J)
            energy = 0.0  # 缓存命中无显著能耗
            
            # ✅ 如果需要返回结果，计算下行传输能耗（很小，结果只有输入的5%）
            # If result needs to be returned, calculate downlink transmission energy
            result_size = task.get('data_size_bytes', 1e6) * 0.05  # 结果是输入的5%
            if result_size > 0:
                down_delay, down_energy = self._estimate_transmission(result_size, float(distance), node_type.lower())
                delay += down_delay  # 加上下行延迟
                energy = down_energy  # 只有下行传输有能耗
            
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self._accumulate_delay('delay_cache', delay)
            self._accumulate_energy('energy_cache', energy)
            
            # 🔧 增强状态转移透明度：记录缓存命中任务详情
            target_key = 'rsu' if node_type == 'RSU' else 'uav'
            execution_detail = {
                'task_id': task.get('id', 'unknown'),
                'vehicle_id': vehicle.get('id', 'unknown'),
                'target_type': target_key,
                'target_id': node_idx,
                'result': 'completed',
                'delay': delay,
                'energy': energy,
                'data_size_mb': task.get('data_size', 0.0),
                'task_type': task.get('task_type', 0),
                'cache_hit': True,
            }
            step_summary['task_execution_details'].append(execution_detail)
            
            # 更新执行摘要
            exec_summary = step_summary['execution_summary']
            exec_summary['completed'] += 1
            exec_summary['cache_hits'] += 1
            exec_summary['offload_distribution'][target_key] += 1
            
            # 计算平均延迟和能耗（加权平均）
            target_count = exec_summary['offload_distribution'][target_key]
            prev_avg_delay = exec_summary['avg_delay_by_target'][target_key]
            prev_avg_energy = exec_summary['avg_energy_by_target'][target_key]
            exec_summary['avg_delay_by_target'][target_key] = ((target_count - 1) * prev_avg_delay + delay) / target_count
            exec_summary['avg_energy_by_target'][target_key] = ((target_count - 1) * prev_avg_energy + energy) / target_count
        # 🔧 记录可视化事件 (缓存命中)
        if 'step_events' in step_summary:
            try:
                v_id = int(vehicle['id'].split('_')[1])
                step_summary['step_events'].append({
                    'type': node_type.lower(),
                    'vehicle_id': v_id,
                    'target_id': node_idx
                })
            except (IndexError, ValueError):
                pass
            return True

        # 缓存未命中：计算上传开销
        # Cache miss: calculate upload overhead
        # 🔧 P0修复：传递vehicle参数以使用动态分配带宽
        upload_delay, upload_energy = self._estimate_transmission(
            task.get('data_size_bytes', 1e6), distance, node_type.lower(), vehicle=vehicle
        )
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
            # 保留原始计算周期以便资源利用率统计（RSU/UAV compute_usage）
            'compute_cycles': float(task.get('compute_cycles', 0.0) or task.get('computation_requirement', 1500.0) * 1e6),
            'work_remaining': work_units,
            'queued_at': self.current_time,
            'node_type': node_type,
            'node_idx': node_idx,
            'upload_delay': upload_delay,
            'priority': task.get('priority', 0.5),
            'task_type': task.get('task_type'),
            'app_scenario': task.get('app_scenario'),
            'deadline_relax_factor': task.get('deadline_relax_factor', 1.0),
            # 🆕 Luo论文队列模型：保留剩余生命周期字段
            'remaining_lifetime_slots': task.get('remaining_lifetime_slots', task.get('max_delay_slots', 5)),
        }

        # 原有队列系统：添加到 computation_queue
        queue = node.setdefault('computation_queue', [])
        queue.append(task_entry)
        
        # 🆕 Luo论文队列模型：同步添加到 lifetime_queues
        # 根据任务的剩余生命周期，加入相应队列
        if 'lifetime_queues' in node:
            lifetime = task_entry.get('remaining_lifetime_slots', 5)
            # 确保 lifetime 在合理范围内
            max_lifetime = getattr(self.queue_config, 'max_lifetime', 10) if hasattr(self, 'queue_config') else 10
            if node_type in ('RSU', 'UAV'):
                # RSU/UAV 最大 L-1
                lifetime = max(1, min(lifetime, max_lifetime - 1))
            else:
                # Vehicle 最大 L
                lifetime = max(1, min(lifetime, max_lifetime))
            
            # 添加任务到对应的生命周期队列
            if lifetime in node['lifetime_queues']:
                node['lifetime_queues'][lifetime].append(task_entry)
        
        self._enforce_queue_capacity(node, node_type, step_summary)
        self._apply_queue_scheduling(node, node_type)
        self._append_active_task(task_entry)
        self._record_mm1_arrival(node_type, node_idx)
        # 🔥 记录RSU/UAV任务统计
        if node_type == 'RSU':
            step_summary['rsu_tasks'] = step_summary.get('rsu_tasks', 0) + 1
        elif node_type == 'UAV':
            step_summary['uav_tasks'] = step_summary.get('uav_tasks', 0) + 1
        
        # 🔧 增强状态转移透明度：记录远程卸载任务详情（排队中）
        target_key = 'rsu' if node_type == 'RSU' else 'uav'
        execution_detail = {
            'task_id': task.get('id', 'unknown'),
            'vehicle_id': vehicle.get('id', 'unknown'),
            'target_type': target_key,
            'target_id': node_idx,
            'result': 'queued',  # 任务被排队，稍后处理
            'delay': upload_delay,  # 已知的上传延迟
            'energy': upload_energy,  # 已知的上传能耗
            'data_size_mb': task.get('data_size', 0.0),
            'task_type': task.get('task_type', 0),
            'cache_hit': False,
            'queue_position': len(queue),  # 队列位置
        }
        step_summary['task_execution_details'].append(execution_detail)
        
        # 更新执行摘要
        exec_summary = step_summary['execution_summary']
        exec_summary['offload_distribution'][target_key] += 1
        
        # 🔧 记录可视化事件 (远程卸载)
        if 'step_events' in step_summary:
            try:
                v_id = int(vehicle['id'].split('_')[1])
                step_summary['step_events'].append({
                    'type': node_type.lower(),
                    'vehicle_id': v_id,
                    'target_id': node_idx
                })
            except (IndexError, ValueError):
                pass
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
        🔧 2024-12-02 修复：检查任务是否在deadline内完成
        
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
        
        # 🔧 修复：检查本地处理是否在deadline内完成
        # 任务完成时间 = 当前时间 + 处理延迟
        completion_time = self.current_time + processing_delay
        task_deadline = task.get('deadline', float('inf'))
        
        if completion_time > task_deadline:
            # 任务无法在deadline内完成，标记为丢弃
            self.stats['dropped_tasks'] = self.stats.get('dropped_tasks', 0) + 1
            self.stats['dropped_data_bytes'] = self.stats.get('dropped_data_bytes', 0.0) + float(task.get('data_size_bytes', 0.0))
            task['dropped'] = True
            task['drop_reason'] = 'local_deadline_exceeded'
            step_summary['dropped_tasks'] = step_summary.get('dropped_tasks', 0) + 1
            step_summary['local_drops'] = step_summary.get('local_drops', 0) + 1
            return
        
        self.stats['processed_tasks'] += 1
        self.stats['completed_tasks'] += 1
        self._accumulate_delay('delay_processing', processing_delay)
        self._accumulate_energy('energy_compute', energy)
        
        # 按任务类别记录时延统计
        self._record_task_type_delay(task, processing_delay)
        
        cpu_freq = float(vehicle.get('cpu_freq', self.vehicle_cpu_freq))
        cycles_consumed = processing_delay * cpu_freq
        vehicle['local_cycle_used'] = vehicle.get('local_cycle_used', 0.0) + cycles_consumed
        available_cycles = max(1e-6, cpu_freq * self.time_slot)
        vehicle['compute_usage'] = float(np.clip(vehicle['local_cycle_used'] / available_cycles, 0.0, 1.0))
        
        # 🔧 新增：更新车辆队列长度（用于状态编码）
        # 统计所有优先级队列的总长度
        queue_length = sum(len(queue) for queue in vehicle.get('task_queue_by_priority', {}).values())
        vehicle['queue_length'] = queue_length
        
        # 🔧 修复：本地处理完成后尝试缓存内容
        cache_ctrl = getattr(self, 'adaptive_cache_controller', None)
        content_id = task.get('content_id')
        if cache_ctrl and content_id:
            try:
                # 获取内容大小和缓存状态
                data_size = self._get_realistic_content_size(content_id)
                if 'device_cache' not in vehicle:
                    vehicle['device_cache'] = {}
                cache_snapshot = vehicle['device_cache']
                
                # 车辆缓存容量 (默认 500MB)
                capacity = float(vehicle.get('cache_capacity', 500.0)) 
                used = sum(float(item.get('size', 0.0)) for item in cache_snapshot.values())
                available = max(0.0, capacity - used)
                
                # 决策是否缓存
                should_cache, reason, evictions = cache_ctrl.should_cache_content(
                    content_id, data_size, available, cache_snapshot, capacity,
                    cache_priority=task.get('priority', 0.5)
                )
                
                if should_cache:
                    # 执行淘汰
                    reclaimed = 0.0
                    for evict_id in evictions:
                        removed = cache_snapshot.pop(evict_id, None)
                        if removed:
                            reclaimed += float(removed.get('size', 0.0) or 0.0)
                            cache_ctrl.cache_stats['evicted_items'] += 1
                    
                    if reclaimed > 0.0:
                        available += reclaimed
                        
                    # 写入缓存
                    if available >= data_size:
                        cache_snapshot[content_id] = {
                            'size': data_size,
                            'timestamp': getattr(self, 'current_time', 0.0),
                            'reason': reason or 'local_process_cache',
                            'content_type': self._infer_content_type(content_id)
                        }
                        # 更新热度
                        cache_ctrl.update_content_heat(content_id)
            except Exception:
                pass
        
        step_summary['local_tasks'] += 1
        
        # 🔧 增强状态转移透明度：记录本地处理任务详情
        execution_detail = {
            'task_id': task.get('id', 'unknown'),
            'vehicle_id': vehicle.get('id', 'unknown'),
            'target_type': 'local',
            'target_id': None,
            'result': 'completed',
            'delay': processing_delay,
            'energy': energy,
            'data_size_mb': task.get('data_size', 0.0),
            'task_type': task.get('task_type', 0),
            'cache_hit': False,
        }
        step_summary['task_execution_details'].append(execution_detail)
        
        # 更新执行摘要
        exec_summary = step_summary['execution_summary']
        exec_summary['completed'] += 1
        exec_summary['offload_distribution']['local'] += 1
        
        # 计算平均延迟和能耗（加权平均）
        local_count = exec_summary['offload_distribution']['local']
        prev_avg_delay = exec_summary['avg_delay_by_target']['local']
        prev_avg_energy = exec_summary['avg_energy_by_target']['local']
        exec_summary['avg_delay_by_target']['local'] = ((local_count - 1) * prev_avg_delay + processing_delay) / local_count
        exec_summary['avg_energy_by_target']['local'] = ((local_count - 1) * prev_avg_energy + energy) / local_count
        
        # 🔧 记录可视化事件
        if 'step_events' in step_summary:
            try:
                v_id = int(vehicle['id'].split('_')[1])
                step_summary['step_events'].append({
                    'type': 'local',
                    'vehicle_id': v_id,
                    'target_id': 0
                })
            except (IndexError, ValueError):
                pass

    def _record_forced_drop(self, vehicle: Dict, task: Dict, step_summary: Dict, reason: str = 'forced_drop') -> None:
        """记录因策略约束导致的任务丢弃事件
        
        🔧 关键修复：防止重复统计已丢弃的任务
        """
        # 🔧 如果任务已经被标记为丢弃，直接返回，避免重复计数
        if task.get('dropped', False):
            return
        
        task['dropped'] = True  # 立即标记，防止后续重复处理
        task['drop_reason'] = reason
        
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
        
        # 🔧 增强状态转移透明度：记录丢弃任务详情
        execution_detail = {
            'task_id': task.get('id', 'unknown'),
            'vehicle_id': vehicle.get('id', 'unknown'),
            'target_type': 'dropped',
            'target_id': None,
            'result': 'dropped',
            'delay': 0.0,
            'energy': 0.0,
            'data_size_mb': task.get('data_size', 0.0),
            'task_type': task.get('task_type', 0),
            'cache_hit': False,
            'drop_reason': reason,
        }
        step_summary['task_execution_details'].append(execution_detail)
        
        # 更新执行摘要
        exec_summary = step_summary['execution_summary']
        exec_summary['dropped'] += 1
        drop_reasons = exec_summary['drop_reasons']
        drop_reasons[reason] = drop_reasons.get(reason, 0) + 1

    
    def check_adaptive_migration(self, agents_actions: Optional[Dict] = None):
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
            except (AttributeError, TypeError, RuntimeError) as e:
                logging.debug(f"Failed to get hotspot intensity: {e}")
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
                'hotspot_intensity': float(np.clip(hotspot_map.get(f'RSU_{i}', 0.0), 0.0, 1.0)),
                # 🔧 修复：添加cpu_frequency字段供智能体使用
                'cpu_frequency': rsu.get('cpu_freq', 12.5e9),  # RSU计算频率
                'cpu_utilization': cpu_load,  # CPU利用率（与td3_optimized.py保持一致）
                'queue_utilization': cpu_load,  # 队列利用率
                'cache_utilization': storage_load,  # 缓存利用率
                'energy_consumption': rsu.get('energy_consumed', 0.0),  # 能耗
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
                'hotspot_intensity': 0.0,
                # 🔧 修复：添加cpu_frequency字段供智能体使用
                'cpu_frequency': uav.get('cpu_freq', 5.0e9),  # UAV计算频率
                'cpu_utilization': cpu_load,  # CPU利用率
                'queue_utilization': cpu_load,  # 队列利用率
                'energy_consumption': uav.get('energy_consumed', 0.0),  # 能耗
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
                    except (AttributeError, RuntimeError) as exc:
                        logging.warning(f"⚠️ 联合策略协调器记录RSU迁移异常: {exc}")
                
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
                    except (AttributeError, RuntimeError) as exc:
                        logging.warning(f"⚠️ 联合策略协调器记录RSU迁移结果异常: {exc}")
        
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
                    except (AttributeError, RuntimeError) as exc:
                        logging.warning(f"⚠️ 联合策略协调器记录UAV迁移异常: {exc}")
                
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
                    except (AttributeError, RuntimeError) as exc:
                        logging.warning(f"⚠️ 联合策略协调器记录UAV迁移结果异常: {exc}")
        
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
                    origin_node = self.rsus[origin_node_idx]
                    origin_queue = origin_node.get('computation_queue', [])
                    filtered = [t for t in origin_queue if t.get('id') != task['id']]
                    origin_node['computation_queue'] = filtered
                    # 🆕 Luo论文队列模型：迁移任务从源节点lifetime_queues移除
                    self._remove_task_from_lifetime_queues(origin_node, task)
                    origin_queue_after = len(filtered)
                elif task['node_type'] == 'UAV':
                    origin_node = self.uavs[origin_node_idx]
                    origin_queue = origin_node.get('computation_queue', [])
                    filtered = [t for t in origin_queue if t.get('id') != task['id']]
                    origin_node['computation_queue'] = filtered
                    # 🆕 Luo论文队列模型：迁移任务从源节点lifetime_queues移除
                    self._remove_task_from_lifetime_queues(origin_node, task)
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
                'deadline_relax_factor': task.get('deadline_relax_factor', 1.0),
                # 🆕 Luo论文队列模型：迁移任务保留remaining_lifetime_slots
                'remaining_lifetime_slots': task.get('remaining_lifetime_slots', task.get('max_delay_slots', 5)),
            }
            best_new_node['computation_queue'].append(migrated_task)
            
            # 🆕 Luo论文队列模型：迁移任务也需添加到目标节点的lifetime_queues
            if 'lifetime_queues' in best_new_node:
                lifetime = migrated_task.get('remaining_lifetime_slots', 5)
                max_lifetime = getattr(self.queue_config, 'max_lifetime', 10) if hasattr(self, 'queue_config') else 10
                if best_node_type in ('RSU', 'UAV'):
                    lifetime = max(1, min(lifetime, max_lifetime - 1))
                else:
                    lifetime = max(1, min(lifetime, max_lifetime))
                if lifetime in best_new_node['lifetime_queues']:
                    best_new_node['lifetime_queues'][lifetime].append(migrated_task)
            
            best_node_type = best_node_type or 'RSU'
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

            # 🔧 修复：计算并记录迁移的数据量、延迟和能耗
            migration_data_mb = task.get('data_size', 2.0)  # MB
            migration_delay_s = migration_data_mb * 8.0 / 50.0  # 无线传输，50 Mbps带宽
            migration_energy_j = 0.2 * migration_delay_s  # 传输功率0.2W
            
            # 累加到统计数据
            self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + migration_data_mb
            self._accumulate_delay('rsu_migration_delay', migration_delay_s)
            self._accumulate_energy('rsu_migration_energy', migration_energy_j)

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
        
        # 🔧 修复1: 重置RSU/UAV即时连接计数器
        # Reset immediate connection counters for RSUs and UAVs at the start of each step
        for rsu in self.rsus:
            rsu['served_vehicles'] = 0
            rsu['coverage_vehicles'] = 0
        
        for uav in self.uavs:
            uav['served_vehicles'] = 0
        
        if self._central_resource_enabled and hasattr(self, 'resource_pool'):
            try:
                self.execute_phase2_scheduling()
            except (AttributeError, RuntimeError) as exc:
                logging.debug(f"Phase-2 scheduling execution failed: {exc}")

        # 推进仿真时间
        advance_simulation_time()
        self.current_step += 1
        self.current_time = get_simulation_time()

        # 当前步骤的统计摘要
        step_summary: Dict[str, Any] = {
            'generated_tasks': 0,  # 本步生成的任务数
            'local_tasks': 0,  # 本地处理的任务数
            'remote_tasks': 0,  # 远程卸载的任务数
            'rsu_tasks': 0,  # RSU处理的任务数
            'uav_tasks': 0,  # UAV处理的任务数
            'local_cache_hits': 0,  # 本地缓存命中次数
            'queue_overflow_drops': 0,  # 本步因队列溢出的丢弃
            'step_events': [],  # 🔧 新增：用于实时可视化的事件列表
            'vehicle_positions': [],  # 🔧 新增：用于实时可视化的车辆位置
            # 🔧 增强状态转移透明度：详细任务执行反馈
            'task_execution_details': [],  # 每个任务的详细执行信息
            'execution_summary': {  # 本步执行摘要
                'completed': 0,  # 成功完成的任务数
                'dropped': 0,  # 丢弃的任务数
                'cache_hits': 0,  # 缓存命中数
                'offload_distribution': {'local': 0, 'rsu': 0, 'uav': 0},
                'avg_delay_by_target': {'local': 0.0, 'rsu': 0.0, 'uav': 0.0},
                'avg_energy_by_target': {'local': 0.0, 'rsu': 0.0, 'uav': 0.0},
                'drop_reasons': {},  # 丢弃原因统计
            }
        }

        # 1. 更新车辆位置
        # Update vehicle positions based on movement model
        self._update_vehicle_positions()
        
        # 🔧 记录车辆位置供可视化使用
        for v in self.vehicles:
            try:
                v_id = int(v['id'].split('_')[1])
                step_summary['vehicle_positions'].append({
                    'id': v_id,
                    'x': float(v['position'][0]),
                    'y': float(v['position'][1]),
                    'dir': float(v.get('direction', 0.0))
                })
            except (IndexError, ValueError):
                pass

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

        # 🆕 Luo论文队列模型：每个时隙开始前，更新所有节点的生命周期队列
        # 核心机制：队列l中未处理的任务降级到队列l-1，l=1时过期任务被删除
        for vehicle in self.vehicles:
            self._update_lifetime_queues(vehicle, 'VEHICLE', step_summary)
        
        for idx, rsu in enumerate(self.rsus):
            self._update_lifetime_queues(rsu, 'RSU', step_summary)
        
        for idx, uav in enumerate(self.uavs):
            self._update_lifetime_queues(uav, 'UAV', step_summary)

        # 4. 处理队列中的任务
        # Process tasks in node queues
        self._process_node_queues()

        # 5. 妫€鏌ヨ秴鏃跺苟娓呯悊
        self._handle_deadlines()
        self._cleanup_active_tasks()

        # 汇总信息
        step_summary['current_time'] = self.current_time
        step_summary['rsu_queue_lengths'] = [len(rsu.get('computation_queue', [])) for rsu in self.rsus]
        step_summary['uav_queue_lengths'] = [len(uav.get('computation_queue', [])) for uav in self.uavs]
        step_summary['active_tasks'] = len(self.active_tasks)
        
        # 🔧 新增：计算卸载比例指标（用于奖励函数）
        total_tasks = step_summary['local_tasks'] + step_summary['rsu_tasks'] + step_summary['uav_tasks']
        if total_tasks > 0:
            step_summary['local_offload_ratio'] = step_summary['local_tasks'] / total_tasks
            step_summary['rsu_offload_ratio'] = step_summary['rsu_tasks'] / total_tasks
            step_summary['uav_offload_ratio'] = step_summary['uav_tasks'] / total_tasks
        else:
            # 默认值（没有任务时）
            step_summary['local_offload_ratio'] = 0.33
            step_summary['rsu_offload_ratio'] = 0.33
            step_summary['uav_offload_ratio'] = 0.34

        stability_metrics = self._monitor_queue_stability()
        for key, value in stability_metrics.items():
            step_summary[key] = value
        task_type_summary = self._summarize_task_types()
        for key, value in task_type_summary.items():
            step_summary[key] = value
        mm1_predictions = self._finalize_mm1_step(self.current_step)
        if isinstance(mm1_predictions, dict):
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
        except (AttributeError, TypeError, ValueError):
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
            except (AttributeError, RuntimeError) as exc:
                logging.warning(f"⚠️ 迁移前预取协调失败({source_rsu_id}->{target_rsu_id}): {exc}")

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
        except (ImportError, AttributeError, ValueError) as e:
            logging.debug(f"Wired backhaul model not available, using fallback: {e}")
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
            except (AttributeError, RuntimeError) as exc:
                logging.warning(f"⚠️ UAV迁移前预取协调失败(UAV_{source_uav_idx}->{target_rsu.get('id')}): {exc}")

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

    def get_task_type_delay_report(self) -> str:
        """
        生成按任务类别的时延性能报告
        
        Returns:
            格式化的报告字符串
        """
        stats = self.stats.get('task_type_delay_stats', {})
        if not stats:
            return "⚠️ 未收集到按任务类别的时延统计数据"
        
        report_lines = []
        report_lines.append("\n" + "="*80)
        report_lines.append("📊 按任务类别的时延性能统计")
        report_lines.append("="*80)
        report_lines.append(f"{'Type':<10} {'Count':<10} {'Avg Delay(s)':<15} {'Max Delay(s)':<15} {'Violations':<12} {'Vio Rate':<10} {'Deadline(s)'}")
        report_lines.append("-"*80)
        
        task_type_names = {
            1: "极度敏感",
            2: "敏感",
            3: "中度容忍",
            4: "容忍"
        }
        
        total_tasks = 0
        total_violations = 0
        
        for task_type in sorted(stats.keys()):
            type_stats = stats[task_type]
            count = type_stats.get('count', 0)
            total_delay = type_stats.get('total_delay', 0.0)
            max_delay = type_stats.get('max_delay', 0.0)
            violations = type_stats.get('deadline_violations', 0)
            deadline = type_stats.get('deadline', 0.0)
            
            if count > 0:
                avg_delay = total_delay / count
                vio_rate = violations / count
            else:
                avg_delay = 0.0
                vio_rate = 0.0
            
            total_tasks += count
            total_violations += violations
            
            type_name = task_type_names.get(task_type, f"Type-{task_type}")
            report_lines.append(
                f"{type_name:<10} {count:<10} {avg_delay:<15.4f} {max_delay:<15.4f} {violations:<12} "
                f"{vio_rate:<10.1%} {deadline:<.2f}"
            )
        
        report_lines.append("-"*80)
        overall_vio_rate = total_violations / total_tasks if total_tasks > 0 else 0.0
        report_lines.append(f"总计: {total_tasks} 个任务, {total_violations} 个超deadline ({overall_vio_rate:.1%})")
        report_lines.append("="*80)
        
        return "\n".join(report_lines)

    def visualize_task_type_delay_stats(self, output_dir: str = 'test_results'):
        """
        生成任务类别时延统计的可视化图表
        
        Args:
            output_dir: 输出目录
        """
        try:
            from tools.visualize_task_type_delay import visualize_task_type_delay_stats
            visualize_task_type_delay_stats(self.stats, output_dir)
        except ImportError as e:
            print(f"⚠️ 无法导入可视化模块: {e}")
            print("请确保 tools/visualize_task_type_delay.py 文件存在")
        except Exception as e:
            print(f"❌ 生成可视化图表时出错: {e}")
