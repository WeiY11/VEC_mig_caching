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
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime

# 🔑 修复：导入统一时间管理器
# Unified time manager for consistent simulation timing
from utils.unified_time_manager import get_simulation_time, advance_simulation_time, reset_simulation_time

# 🔑 修复：导入realistic内容生成器
# Realistic content generator for simulating various content types
from utils.realistic_content_generator import generate_realistic_content, get_realistic_content_size
from utils.spatial_index import SpatialIndex
from decision.two_stage_planner import TwoStagePlanner, PlanEntry

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
            self.time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)  # 🚀 适应高负载时隙
            self.task_arrival_rate = getattr(self.sys_config.task, 'arrival_rate', 2.5)  # 🚀 高负载到达率
        else:
            self.simulation_time = self.config.get('simulation_time', 1000)
            self.time_slot = self.config.get('time_slot', 0.2)  # 🚀 高负载默认时隙
            self.task_arrival_rate = self.config.get('task_arrival_rate', 2.5)  # 🚀 高负载默认到达率
        
        # 子配置对象引用
        # Sub-configuration object references
        self.task_config = getattr(self.sys_config, 'task', None) if self.sys_config is not None else None
        self.service_config = getattr(self.sys_config, 'service', None) if self.sys_config is not None else None
        self.stats_config = getattr(self.sys_config, 'stats', None) if self.sys_config is not None else None
        
        # 性能统计与运行状态
        # Performance statistics and runtime state
        self.stats = self._fresh_stats_dict()
        self.active_tasks: List[Dict] = []  # 每项: {id, vehicle_id, arrival_time, deadline, work_remaining, node_type, node_idx}
        self.task_counter = 0
        self.current_step = 0
        self.current_time = 0.0
        # Two-stage planning toggle (env-controlled)
        self._two_stage_enabled = (os.environ.get('TWO_STAGE_MODE', '').strip() in {'1', 'true', 'True'})
        self._two_stage_planner: TwoStagePlanner | None = None
        self.spatial_index: Optional[SpatialIndex] = SpatialIndex()
        
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
            'drop_log_interval': 200,
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
                'device_cache_capacity': 32.0  # 车载缓存容量(MB)
            }
            self.vehicles.append(vehicle)
        print("🛣️ 车辆初始化完成：主干道双路口场景")
        
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
                'computation_queue': [],  # 计算任务队列
                'energy_consumed': 0.0  # 累计能耗(J)
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
                'computation_queue': [],  # 计算任务队列
                'energy_consumed': 0.0  # 累计能耗(J)
            }
            self.uavs.append(uav)
        
        print(f"✅ 创建了 {self.num_vehicles} 车辆, {self.num_rsus} RSU, {self.num_uavs} UAV")
        
        # 🏢 初始化中央RSU调度器(选择RSU_2作为中央调度中心)
        # Initialize central RSU scheduler for coordinated task management
        try:
            from utils.central_rsu_scheduler import create_central_scheduler
            central_rsu_id = f"RSU_{2 if self.num_rsus > 2 else 0}"
            self.central_scheduler = create_central_scheduler(central_rsu_id)
            print(f"🏢 中央RSU调度器已启用: {central_rsu_id}")
        except Exception as e:
            print(f"⚠️ 中央调度器加载失败: {e}")
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

        self._refresh_spatial_index(update_static=True, update_vehicle=True)
    
    def _setup_scenario(self):
        """
        设置仿真场景
        
        重新初始化所有组件并重置运行时状态，用于开始新的仿真回合。
        
        Setup simulation scenario for a new episode.
        """
        # 重新初始化组件（如果需要）
        self.initialize_components()
        self._reset_runtime_states()
        print("✅ 初始化了 6 个缓存管理器")

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
            'local_cache_hits': 0,  # 本地缓存命中次数
            'cache_hits': 0,  # 缓存命中次数
            'cache_misses': 0,  # 缓存未命中次数
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
                'by_scenario': {}
            }
        }

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
        self.task_counter = 0
        self.stats = self._fresh_stats_dict()
        self.active_tasks = []
        self._last_app_name = 'unknown'

        # 閲嶇疆杞﹁締/鑺傜偣鐘舵€?
        for vehicle in self.vehicles:
            vehicle.setdefault('tasks', [])
            vehicle['tasks'].clear()
            vehicle['energy_consumed'] = 0.0
            vehicle['device_cache'] = {}
            vehicle['device_cache_capacity'] = vehicle.get('device_cache_capacity', 32.0)

        for rsu in self.rsus:
            rsu.setdefault('cache', {})
            rsu['computation_queue'] = []
            rsu['energy_consumed'] = 0.0

        for uav in self.uavs:
            uav.setdefault('cache', {})
            uav['computation_queue'] = []
            uav['energy_consumed'] = 0.0
    
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
        task_type = 3

        if task_cfg is not None:
            scenario = task_cfg.sample_scenario()
            scenario_name = scenario.name
            relax_factor_applied = scenario.relax_factor or task_cfg.deadline_relax_default

            deadline_duration = np.random.uniform(scenario.min_deadline, scenario.max_deadline)
            deadline_duration *= relax_factor_applied
            max_delay_slots = max(1, int(deadline_duration / max(time_slot, 1e-6)))
            task_type = scenario.task_type or task_cfg.get_task_type(max_delay_slots)

            profile = task_cfg.get_profile(task_type)
            data_min, data_max = profile.data_range
            data_size_bytes = float(np.random.uniform(data_min, data_max))
            compute_density = profile.compute_density
        else:
            deadline_duration = np.random.uniform(0.5, 3.0) * relax_factor_applied
            task_type = int(np.random.randint(1, 5))
            data_size_mb = np.random.exponential(0.5)
            data_size_bytes = data_size_mb * 1e6
            compute_density = self.config.get('task_compute_density', 400)
            max_delay_slots = max(1, int(deadline_duration / max(self.config.get('time_slot', self.time_slot), 0.1)))

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
        computation_cycles = total_bits * effective_density
        computation_mips = (computation_cycles / 1e6) * complexity_multiplier

        task = {
            'id': f'task_{self.task_counter}',
            'vehicle_id': vehicle_id,
            'arrival_time': self.current_time,
            'data_size': data_size_mb,
            'data_size_bytes': data_size_bytes,
            'computation_requirement': computation_mips,
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
                f"📊 任务分类统计({gen_stats['total']}): "
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
        for rsu in self.rsus:
            self._process_single_node_queue(rsu, 'RSU')
        
        # 处理所有UAV队列
        for uav in self.uavs:
            self._process_single_node_queue(uav, 'UAV')
    

    def _process_single_node_queue(self, node: Dict, node_type: str) -> None:
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

        if queue_len > base_capacity:
            dynamic_boost = int(np.ceil((queue_len - base_capacity) / boost_divisor))
        else:
            dynamic_boost = 0

        tasks_to_process = min(queue_len, base_capacity + dynamic_boost)
        tasks_to_process = min(tasks_to_process, max_service)
        tasks_to_process = max(tasks_to_process, min(queue_len, base_capacity))

        new_queue: List[Dict] = []
        current_time = getattr(self, 'current_time', 0.0)
        work_capacity = self.time_slot * work_capacity_cfg

        for idx, task in enumerate(queue):
            if current_time - task.get('queued_at', -1e9) < self.time_slot:
                new_queue.append(task)
                continue

            if idx >= tasks_to_process:
                new_queue.append(task)
                continue

            remaining_work = float(task.get('work_remaining', 0.5)) - work_capacity
            task['work_remaining'] = max(0.0, remaining_work)

            if task['work_remaining'] > 0.0:
                new_queue.append(task)
                continue

            self.stats['completed_tasks'] += 1
            self.stats['processed_tasks'] = self.stats.get('processed_tasks', 0) + 1

            actual_delay = current_time - task.get('arrival_time', current_time)
            actual_delay = max(0.001, min(actual_delay, 20.0))
            self.stats['total_delay'] += actual_delay

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
                self.stats['total_delay'] += down_delay
                self.stats['total_energy'] += down_energy

            if node_type == 'RSU':
                processing_power = 50.0
            elif node_type == 'UAV':
                processing_power = 20.0
            else:
                processing_power = 10.0

            task_energy = processing_power * work_capacity
            self.stats['total_energy'] += task_energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + task_energy

            task['completed'] = True

        node['computation_queue'] = new_queue


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
        cache_hit = content_id in node.get('cache', {})
        
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
                
                # 调用智能控制器判断是否缓存
                # Call intelligent controller to decide whether to cache
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
    
    def _calculate_enhanced_load_factor(self, node: Dict, node_type: str) -> float:
        """
        馃敡 淇锛氱粺涓€鍜宺ealistic鐨勮礋杞藉洜瀛愯绠?
        鍩轰簬瀹為檯闃熷垪璐熻浇锛屼笉浣跨敤铏氬亣鐨勯檺鍒?
        """
        queue_length = len(node.get('computation_queue', []))
        
        # 馃敡 鍩轰簬瀹為檯瑙傚療璋冩暣瀹归噺鍩哄噯
        if node_type == 'RSU':
            # 鍩轰簬瀹為檯娴嬭瘯锛孯SU澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            base_capacity = 20.0  
            queue_factor = queue_length / base_capacity
        else:  # UAV
            # UAV澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            base_capacity = 10.0
            queue_factor = queue_length / base_capacity
        
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
        """估计远程节点的工作量单位（供队列调度使用）"""
        requirement = float(task.get('computation_requirement', 1500.0))
        base_divisor = 1200.0 if node_type == 'RSU' else 1600.0
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
        """浼拌涓婁紶鑰楁椂涓庤兘鑰?"""
        # 鏈夋晥鍚炲悙閲?(bit/s)
        if link == 'uav':
            base_rate = 45e6
            power_w = 0.12
        else:
            base_rate = 80e6
            power_w = 0.18

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
                    'by_scenario': {}
                })
                by_type = drop_stats.setdefault('by_type', {})
                by_scenario = drop_stats.setdefault('by_scenario', {})
                log_interval = self.stats_config.drop_log_interval if getattr(self, 'stats_config', None) else self.config.get('drop_log_interval', 200)
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

        # 鎺ㄩ€佸埌瑕嗙洊鑼冨洿鍐呯殑杞﹁締
        coverage = rsu_node.get('coverage_radius', 300.0)
        if getattr(self, 'spatial_index', None):
            vehicles_in_range = self.spatial_index.query_vehicles_within_radius(rsu_node['position'], coverage * 0.8)
            for _, vehicle_node, _ in vehicles_in_range:
                self._store_in_vehicle_cache(vehicle_node, content_id, size_mb, cache_controller)
        else:
            for vehicle in self.vehicles:
                distance = self.calculate_distance(vehicle.get('position', np.zeros(2)), rsu_node['position'])
                if distance <= coverage * 0.8:
                    self._store_in_vehicle_cache(vehicle, content_id, size_mb, cache_controller)

        # ?????RSU
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
        vehicle_cache = vehicle.setdefault('device_cache', {})
        if content_id and content_id in vehicle_cache:
            vehicle_cache[content_id]['timestamp'] = self.current_time
            local_delay = 0.02
            local_energy = 0.0
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self.stats['total_delay'] += local_delay
            self.stats['total_energy'] += local_energy
            self.stats['cache_hits'] += 1
            self.stats['local_cache_hits'] = self.stats.get('local_cache_hits', 0) + 1
            vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + local_energy
            step_summary['local_cache_hits'] = step_summary.get('local_cache_hits', 0) + 1
            if cache_controller is not None:
                cache_controller.record_cache_result(content_id, True)
                cache_controller.update_content_heat(content_id)
            return

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
        choice = int(np.random.choice(np.arange(len(candidate_indices)), p=weights))
        rsu_idx = int(candidate_indices[choice])
        distance = float(distances[choice])
        node = self.rsus[rsu_idx]
        success = self._handle_remote_assignment(vehicle, task, node, 'RSU', rsu_idx, distance, actions, step_summary)
        if success:
            step_summary['remote_tasks'] += 1
        return success


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
        choice = int(np.random.choice(np.arange(len(candidate_indices)), p=weights))
        uav_idx = int(candidate_indices[choice])
        distance = float(distances[choice])
        node = self.uavs[uav_idx]
        success = self._handle_remote_assignment(vehicle, task, node, 'UAV', uav_idx, distance, actions, step_summary)
        if success:
            step_summary['remote_tasks'] += 1
        return success


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
            self.stats['total_delay'] += delay
            self.stats['total_energy'] += energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + energy
            return True

        # 缓存未命中：计算上传开销
        # Cache miss: calculate upload overhead
        upload_delay, upload_energy = self._estimate_transmission(task.get('data_size_bytes', 1e6), distance, node_type.lower())
        self.stats['total_delay'] += upload_delay
        self.stats['energy_uplink'] += upload_energy
        self.stats['total_energy'] += upload_energy
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
        self._append_active_task(task_entry)
        return True

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
        processing_delay, energy = self._estimate_local_processing(task, vehicle)
        self.stats['processed_tasks'] += 1
        self.stats['completed_tasks'] += 1
        self.stats['total_delay'] += processing_delay
        self.stats['total_energy'] += energy
        step_summary['local_tasks'] += 1

    
    def check_adaptive_migration(self, agents_actions: Dict = None):
        """馃幆 澶氱淮搴︽櫤鑳借縼绉绘鏌?(闃堝€艰Е鍙?璐熻浇宸Е鍙?璺熼殢杩佺Щ)"""
        if not agents_actions or 'migration_controller' not in agents_actions:
            return
        
        migration_controller = agents_actions['migration_controller']
        
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
                
                # 鎵цRSU闂磋縼绉?
                result = self.execute_rsu_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
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
                
                # UAV杩佺Щ鍒癛SU
                result = self.execute_uav_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
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

        # 推进仿真时间
        advance_simulation_time()
        self.current_step += 1
        self.current_time = get_simulation_time()

        # 当前步骤的统计摘要
        step_summary = {
            'generated_tasks': 0,  # 本步生成的任务数
            'local_tasks': 0,  # 本地处理的任务数
            'remote_tasks': 0,  # 远程卸载的任务数
            'local_cache_hits': 0  # 本地缓存命中次数
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

        step_summary.update(self._summarize_task_types())

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
    
    def execute_rsu_migration(self, source_rsu_idx: int, urgency: float) -> Dict[str, float]:
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

        # 寻找候选目标RSU
        # Find candidate target RSUs
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

        # 选择负载最轻的目标RSU
        # Select the least loaded target RSU
        target_idx, target_queue_len, target_cpu_load, _ = min(candidates, key=lambda x: x[3])
        source_queue_len = len(source_queue)
        queue_diff = target_queue_len - source_queue_len

        # 动态迁移容忍度：根据系统队列方差调整
        # Dynamic migration tolerance based on system queue variance
        all_queue_lens = [len(rsu.get('computation_queue', [])) for rsu in self.rsus]
        system_queue_variance = np.var(all_queue_lens)
        if system_queue_variance > 50:
            migration_tolerance = 8  # 高方差：允许更大的队列差异
        elif system_queue_variance > 20:
            migration_tolerance = 5
        else:
            migration_tolerance = 3  # 低方差：严格控制迁移
        if queue_diff > migration_tolerance:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        # 根据紧急度确定迁移比例
        # Determine migration ratio based on urgency
        migration_ratio = max(0.1, min(0.5, urgency))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        source_rsu_id = source_rsu['id']
        target_rsu_id = target_rsu['id']
        avg_task_size = 2.0
        total_data_size = tasks_to_migrate * avg_task_size

        migrated_tasks = source_queue[:tasks_to_migrate]
        source_rsu['computation_queue'] = source_queue[tasks_to_migrate:]
        target_rsu['computation_queue'].extend(migrated_tasks)

        delay_saved = max(0.0, (source_queue_len - target_queue_len) * self.time_slot)
        migration_cost = 0.0
        try:
            from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
            wired_delay = calculate_rsu_to_rsu_delay(total_data_size, source_rsu_id, target_rsu_id)
            wired_energy = calculate_rsu_to_rsu_energy(total_data_size, source_rsu_id, target_rsu_id, wired_delay)
            self.stats['rsu_migration_delay'] = self.stats.get('rsu_migration_delay', 0.0) + wired_delay
            self.stats['rsu_migration_energy'] = self.stats.get('rsu_migration_energy', 0.0) + wired_energy
            self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + total_data_size
            migration_cost = wired_energy + wired_delay * 1000.0
        except Exception:
            migration_cost = total_data_size * 0.2

        return {'success': True, 'cost': migration_cost, 'delay_saved': delay_saved}
    
    def execute_uav_migration(self, source_uav_idx: int, urgency: float) -> Dict[str, float]:
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

        # UAV迁移更激进（比例更高）
        # UAV migration is more aggressive (higher ratio)
        source_queue_len = len(source_queue)
        migration_ratio = max(0.2, min(0.6, urgency + 0.1))
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

        total_data_size = sum(task.get('data_size', 1.0) for task in migrated_tasks) or (tasks_to_migrate * 1.0)
        # Estimate wireless transfer characteristics
        wireless_rate = 12.0  # MB/s
        wireless_delay = (total_data_size / wireless_rate)
        wireless_energy = total_data_size * 0.15 + distance * 0.01
        delay_saved = max(0.0, (source_queue_len - target_queue_len) * self.time_slot)

        self.stats['uav_migration_distance'] = self.stats.get('uav_migration_distance', 0.0) + distance
        self.stats['uav_migration_count'] = self.stats.get('uav_migration_count', 0) + 1

        migration_cost = wireless_energy + wireless_delay * 800.0
        return {'success': True, 'cost': migration_cost, 'delay_saved': delay_saved}


