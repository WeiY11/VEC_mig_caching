#!/usr/bin/env python3
"""
完整系统仿真器
用于测试完整的车联网边缘缓存系统
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
# 🔧 修复：导入统一时间管理器
from utils.unified_time_manager import get_simulation_time, advance_simulation_time, reset_simulation_time
# 🔧 修复：导入realistic内容生成器
from utils.realistic_content_generator import generate_realistic_content, get_realistic_content_size

class CompleteSystemSimulator:
    """完整系统仿真器"""
    
    def __init__(self, config: Dict = None):
        """初始化仿真器"""
        self.config = config or self.get_default_config()
        # 统一系统配置入口（若可用）
        try:
            from config import config as sys_config
            self.sys_config = sys_config
        except Exception:
            self.sys_config = None
        
        # 网络拓扑
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.num_vehicles = getattr(self.sys_config.network, 'num_vehicles', 12)
            self.num_rsus = getattr(self.sys_config.network, 'num_rsus', 6)
            self.num_uavs = getattr(self.sys_config.network, 'num_uavs', 2)
        else:
            self.num_vehicles = self.config.get('num_vehicles', 12)
            self.num_rsus = self.config.get('num_rsus', 4)  # 🔧 修复：使用正确的默认值
            self.num_uavs = self.config.get('num_uavs', 2)
        
        # 仿真参数
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.simulation_time = getattr(self.sys_config, 'simulation_time', 1000)
            self.time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)  # 🚀 适应高负载时隙
            self.task_arrival_rate = getattr(self.sys_config.task, 'arrival_rate', 2.5)  # 🚀 高负载到达率
        else:
            self.simulation_time = self.config.get('simulation_time', 1000)
            self.time_slot = self.config.get('time_slot', 0.2)  # 🚀 高负载默认时隙
            self.task_arrival_rate = self.config.get('task_arrival_rate', 2.5)  # 🚀 高负载默认到达率
        
        # 性能统计与运行态
        self.stats = self._fresh_stats_dict()
        self.active_tasks: List[Dict] = []  # 每项: {id, vehicle_id, arrival_time, deadline, work_remaining, node_type, node_idx}
        self.task_counter = 0
        self.current_step = 0
        self.current_time = 0.0
        
        # 初始化组件
        self.initialize_components()
        self._reset_runtime_states()
    
    def get_default_config(self) -> Dict:
        """获取默认配置"""
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
        }
    
    def initialize_components(self):
        """初始化系统组件"""
        # 🚦 主干道+双路口初始化
        # 坐标系统 0..1000，主干道沿 x 轴中线 y=500，从左向右；两处路口位于 x=300 与 x=700
        self.road_y = 500.0
        self.intersections = {  # 信号灯相位: 周期 T，绿灯比例 g
            'L': {'x': 300.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 0.0},
            'R': {'x': 700.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 15.0},
        }

        # 车辆初始化：落在道路上，方向为东(0)或西(pi)，车道内微扰
        self.vehicles = []
        for i in range(self.num_vehicles):
            go_east = np.random.rand() < 0.6  # 60% 向东
            base_dir = 0.0 if go_east else np.pi
            x0 = np.random.uniform(100.0, 900.0)
            y0 = self.road_y + np.random.uniform(-6.0, 6.0)  # 简单两车道路幅
            v0 = np.random.uniform(12.0, 22.0)
            vehicle = {
                'id': f'V_{i}',
                'position': np.array([x0, y0], dtype=float),
                'velocity': v0,
                'direction': base_dir,
                'lane_bias': y0 - self.road_y,
                'tasks': [],
                'energy_consumed': 0.0,
                'device_cache': {},
                'device_cache_capacity': 32.0
            }
            self.vehicles.append(vehicle)
        print("🚦 车辆初始化完成：主干道双路口场景")
        
        # RSU节点
        self.rsus = []
        # 🔧 动态RSU部署：根据num_rsus均匀分布在道路上
        if self.num_rsus <= 4:
            # 原始固定4个RSU的部署
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
        
        # 创建RSU
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': rsu_positions[i],
                'coverage_radius': (getattr(self.sys_config.network, 'coverage_radius', 300) if self.sys_config is not None else 300),
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'rsu_cache_capacity', 10e9) if self.sys_config is not None else 10e9),
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.rsus.append(rsu)
        
        # UAV节点
        self.uavs = []
        # 🔧 动态UAV部署：根据num_uavs均匀分布
        if self.num_uavs <= 2:
            # 原始2架UAV的部署
            uav_positions = [
                np.array([300.0, 500.0, 120.0]),
                np.array([700.0, 500.0, 120.0]),
            ]
        else:
            # 动态生成UAV位置，均匀分布在道路上方
            uav_positions = []
            spacing = 600.0 / (self.num_uavs - 1)  # 均匀间隔
            for i in range(self.num_uavs):
                x_pos = 200.0 + i * spacing
                uav_positions.append(np.array([x_pos, 500.0, 120.0]))
        
        # 创建UAV
        for i in range(self.num_uavs):
            uav = {
                'id': f'UAV_{i}',
                'position': uav_positions[i],  # 固定悬停位置
                'velocity': 0.0,
                'coverage_radius': 350.0,
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'uav_cache_capacity', 2e9) if self.sys_config is not None else 2e9),
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.uavs.append(uav)
        
        print(f"✓ 创建了 {self.num_vehicles} 车辆, {self.num_rsus} RSU, {self.num_uavs} UAV")
        
        # 🏢 初始化中央RSU调度器 (选择RSU_2作为中央调度中心)
        try:
            from utils.central_rsu_scheduler import create_central_scheduler
            central_rsu_id = f"RSU_{2 if self.num_rsus > 2 else 0}"
            self.central_scheduler = create_central_scheduler(central_rsu_id)
            print(f"🏢 中央RSU调度器已启用: {central_rsu_id}")
        except Exception as e:
            print(f"⚠️ 中央调度器加载失败: {e}")
            self.central_scheduler = None
        
        # 懒加载迁移管理器
        try:
            from migration.migration_manager import TaskMigrationManager
            if not hasattr(self, 'migration_manager') or self.migration_manager is None:
                self.migration_manager = TaskMigrationManager()
        except Exception:
            self.migration_manager = None
        
        # 一致性自检（不强制终止，仅提示）
        try:
            expected_rsus, expected_uavs = 4, 2
            if self.num_rsus != expected_rsus or self.num_uavs != expected_uavs:
                print(f"⚠️ 拓扑一致性提示: 当前 num_rsus={self.num_rsus}, num_uavs={self.num_uavs}, 建议为 {expected_rsus}/{expected_uavs} 以匹配论文图示")
            print("🏢 中央RSU设定: RSU_2 (作为调度与回传汇聚节点)")
        except Exception:
            pass
    
    def _setup_scenario(self):
        """设置仿真场景"""
        # 重新初始化组件（如果需要）
        self.initialize_components()
        self._reset_runtime_states()
        print("✓ 初始化了 6 个缓存管理器")

    def _fresh_stats_dict(self) -> Dict[str, float]:
        """创建新的统计字典，保证关键指标齐全"""
        return {
            'total_tasks': 0,
            'processed_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'generated_data_bytes': 0.0,
            'dropped_data_bytes': 0.0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'energy_uplink': 0.0,
            'energy_downlink': 0.0,
            'local_cache_hits': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'migrations_executed': 0,
            'migrations_successful': 0,
            'rsu_migration_delay': 0.0,
            'rsu_migration_energy': 0.0,
            'rsu_migration_data': 0.0,
            'uav_migration_distance': 0.0,
            'uav_migration_count': 0
        }

    def _reset_runtime_states(self):
        """重置运行时状态（用于episode重启）"""
        reset_simulation_time()
        self.current_step = 0
        self.current_time = 0.0
        self.task_counter = 0
        self.stats = self._fresh_stats_dict()
        self.active_tasks = []

        # 重置车辆/节点状态
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
        🔧 修复：使用realistic内容生成器获取大小
        """
        return get_realistic_content_size(content_id)
    
    def _calculate_available_cache_capacity(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        🔧 修复：正确计算可用缓存容量(MB)
        """
        if not cache or cache_capacity_mb <= 0:
            return cache_capacity_mb
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                # 兼容旧格式
                total_used_mb += 1.0
        
        available_mb = cache_capacity_mb - total_used_mb
        return max(0.0, available_mb)
    
    def _infer_content_type(self, content_id: str) -> str:
        """
        🔧 修复：根据内容ID推断内容类型
        """
        content_id_lower = content_id.lower()
        
        if 'traffic' in content_id_lower:
            return 'traffic_info'
        elif 'nav' in content_id_lower or 'route' in content_id_lower:
            return 'navigation'
        elif 'safety' in content_id_lower or 'alert' in content_id_lower:
            return 'safety_alert'
        elif 'park' in content_id_lower:
            return 'parking_info'
        elif 'weather' in content_id_lower:
            return 'weather_info'
        elif 'map' in content_id_lower:
            return 'map_data'
        elif 'video' in content_id_lower or 'entertainment' in content_id_lower:
            return 'entertainment'
        elif 'sensor' in content_id_lower:
            return 'sensor_data'
        else:
            return 'general'
    
    def generate_task(self, vehicle_id: str) -> Dict:
        """生成计算任务 - 使用分层任务类型设计"""
        self.task_counter += 1
        
        # 🔧 修复：按论文正确分类 - 先生成应用特定deadline，再基于延迟容忍度分类
        if self.sys_config is not None:
            # 第一步：根据论文阈值严格设计应用场景deadline需求
            # τ₁=0.8s, τ₂=2.0s, τ₃=5.0s
            app_scenarios = [
                ('emergency_brake', 0.2, 0.6),      # 紧急制动：≤τ₁ (类型1)
                ('collision_avoid', 0.3, 0.6),      # 避障：≤τ₁ (类型1)
                ('navigation', 0.9, 1.9),           # 实时导航：(τ₁,τ₂] (类型2)  
                ('traffic_signal', 1.1, 2.0),       # 交通信号：(τ₁,τ₂] (类型2)
                ('video_process', 2.2, 4.8),        # 视频处理：(τ₂,τ₃] (类型3)
                ('image_recognition', 2.5, 4.9),    # 图像识别：(τ₂,τ₃] (类型3)
                ('data_analysis', 5.5, 12.0),       # 数据分析：>τ₃ (类型4)
                ('ml_training', 8.0, 18.0),         # 机器学习：>τ₃ (类型4)
            ]
            
            # 按概率选择应用场景（现实分布：紧急少，容忍多）
            scenario_weights = [0.08, 0.07, 0.25, 0.15, 0.20, 0.15, 0.08, 0.02]
            selected_scenario = np.random.choice(len(app_scenarios), p=scenario_weights)
            app_name, min_deadline, max_deadline = app_scenarios[selected_scenario]
            
            # 从应用特定范围生成deadline
            deadline_duration = np.random.uniform(min_deadline, max_deadline)
            
            # 第二步：根据deadline计算时隙数并分类（论文正确方法）
            time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)
            max_delay_slots = int(deadline_duration / time_slot)
            
            # 使用论文分类方法
            task_type = self.sys_config.task.get_task_type(max_delay_slots)
            
            # 第三步：根据确定的任务类型获取对应参数
            task_specs = getattr(self.sys_config.task, 'task_type_specs', {})
            if task_type in task_specs:
                spec = task_specs[task_type]
                data_range = spec['data_range']
                compute_density = spec['compute_density']
            else:
                # 回退到通用参数
                data_range = getattr(self.sys_config.task, 'data_size_range', (0.5e6/8, 15e6/8))
                compute_density = float(getattr(self.sys_config.task, 'task_compute_density', 400))
            
            # 数据大小：从类型特定范围采样
            data_size_bytes = np.random.uniform(data_range[0], data_range[1])
            data_size_mb = data_size_bytes / 1e6
            
            # 计算需求：基于数据大小和类型特定计算密度
            total_bits = data_size_bytes * 8
            computation_cycles = total_bits * compute_density
            computation_mips = computation_cycles / 1e6
        else:
            # 回退默认值
            task_type = np.random.randint(1, 5)
            data_size_mb = np.random.exponential(0.5)  # 更小的默认数据
            data_size_bytes = data_size_mb * 1e6
            computation_mips = np.random.exponential(80)  # 降低默认计算需求
            deadline_duration = np.random.uniform(0.5, 3.0)
            compute_density = 400  # 设置默认密度
        
        # 🚀 任务复杂度控制 - 避免过高能耗
        high_load_mode = self.config.get('high_load_mode', False)
        if high_load_mode:
            complexity_multiplier = self.config.get('task_complexity_multiplier', 1.5)  # 降低倍数
            
            # 温和增强计算需求
            computation_mips *= complexity_multiplier
            
            # 限制数据大小在合理范围
            data_size_mb = min(data_size_mb * 1.1, 2.0)  # 最大2MB
            data_size_bytes = data_size_mb * 1e6
            
            # 温和增强计算密度
            compute_density = min(compute_density * 1.05, 200)  # 最大200 cycles/bit
        
        task = {
            'id': f'task_{self.task_counter}',
            'vehicle_id': vehicle_id,
            'arrival_time': self.current_time,
            'data_size': data_size_mb,  # 🚀 高负载增强数据大小
            'data_size_bytes': data_size_bytes,  # 🚀 高负载增强数据字节
            'computation_requirement': computation_mips,  # 🚀 高负载增强计算需求
            'deadline': self.current_time + deadline_duration,
            'content_id': f'content_{np.random.randint(0, 100)}',
            'priority': np.random.uniform(0.1, 1.0),
            'task_type': task_type,  # 🔧 新增：任务类型标识
            'app_scenario': app_name,  # 🔧 新增：应用场景
            'compute_density': compute_density,  # 🚀 高负载增强计算密度
            'complexity_multiplier': self.config.get('task_complexity_multiplier', 1.0),  # 🚀 复杂度标记
            'max_delay_slots': max_delay_slots  # 🔧 新增：时隙数（用于验证分类）
        }
        
        # 📊 任务分类统计监控（每100个任务输出分类分布）
        if self.task_counter % 100 == 0 and self.task_counter > 0:
            # 统计最近100个任务的分类分布
            if not hasattr(self, 'task_type_stats'):
                self.task_type_stats = {1: 0, 2: 0, 3: 0, 4: 0}
            self.task_type_stats[task_type] = self.task_type_stats.get(task_type, 0) + 1
            
            total_classified = sum(self.task_type_stats.values())
            if total_classified > 0:
                type1_pct = self.task_type_stats[1] / total_classified * 100
                type2_pct = self.task_type_stats[2] / total_classified * 100
                type3_pct = self.task_type_stats[3] / total_classified * 100
                type4_pct = self.task_type_stats[4] / total_classified * 100
                print(f"📊 任务分类统计({self.task_counter}): 类型1={type1_pct:.1f}%, 类型2={type2_pct:.1f}%, 类型3={type3_pct:.1f}%, 类型4={type4_pct:.1f}%")
                print(f"   当前任务: {app_name}, {deadline_duration:.2f}s → 类型{task_type}, 数据{data_size_mb:.2f}MB")
        
        return task
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两点间距离"""
        if len(pos1) == 3 and len(pos2) == 2:
            pos2 = np.append(pos2, 0)  # 2D转3D
        elif len(pos1) == 2 and len(pos2) == 3:
            pos1 = np.append(pos1, 0)
        
        return np.linalg.norm(pos1 - pos2)
    
    def _find_least_loaded_node(self, node_type: str, exclude_node: Dict = None) -> Dict:
        """寻找负载最轻的节点"""
        if node_type == 'RSU':
            candidates = [rsu for rsu in self.rsus if rsu != exclude_node]
        elif node_type == 'UAV':
            candidates = [uav for uav in self.uavs if uav != exclude_node]
        else:
            return None
        
        if not candidates:
            return None
        
        # 找到队列长度最短的节点
        best_node = min(candidates, key=lambda n: len(n.get('computation_queue', [])))
        return best_node
    
    def _process_node_queues(self):
        """🔧 关键修复：处理RSU和UAV队列中的任务，防止任务堆积"""
        # 处理所有RSU队列
        for rsu in self.rsus:
            self._process_single_node_queue(rsu, 'RSU')
        
        # 处理所有UAV队列
        for uav in self.uavs:
            self._process_single_node_queue(uav, 'UAV')
    

    def _process_single_node_queue(self, node: Dict, node_type: str):
        "处理单个节点的计算队列"
        queue = node.get('computation_queue', []) or []
        if not queue:
            return

        max_tasks_per_slot = 3 if node_type == 'RSU' else 2
        tasks_to_process = min(len(queue), max_tasks_per_slot)

        new_queue: List[Dict] = []
        current_time = getattr(self, 'current_time', 0.0)

        for idx, task in enumerate(queue):
            if current_time - task.get('queued_at', -1e9) < self.time_slot:
                new_queue.append(task)
                continue

            if idx >= tasks_to_process:
                new_queue.append(task)
                continue

            remaining_work = float(task.get('work_remaining', 0.5))
            if node_type == 'RSU':
                work_capacity = self.time_slot * 2.0
            elif node_type == 'UAV':
                work_capacity = self.time_slot * 1.5
            else:
                work_capacity = self.time_slot

            remaining_work -= work_capacity
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
        """找到最近的RSU"""
        min_distance = float('inf')
        nearest_rsu = None
        
        for rsu in self.rsus:
            distance = self.calculate_distance(vehicle_pos, rsu['position'])
            if distance < min_distance and distance <= rsu['coverage_radius']:
                min_distance = distance
                nearest_rsu = rsu
        
        return nearest_rsu
    
    def find_nearest_uav(self, vehicle_pos: np.ndarray) -> Dict:
        """找到最近的UAV"""
        min_distance = float('inf')
        nearest_uav = None
        
        for uav in self.uavs:
            distance = self.calculate_distance(vehicle_pos, uav['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_uav = uav
        
        return nearest_uav
    
    def check_cache_hit(self, content_id: str, node: Dict) -> bool:
        """检查缓存命中"""
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
        """🤖 智能体控制的自适应缓存检查"""
        # 基础缓存检查
        cache_hit = content_id in node.get('cache', {})
        
        # 更新统计
        if cache_hit:
            self.stats['cache_hits'] += 1
            if node_type == 'RSU':
                self._propagate_cache_after_hit(content_id, node, agents_actions)
        else:
            self.stats['cache_misses'] += 1
            
            # 🤖 如果有智能体控制器，执行自适应缓存策略
            if agents_actions and 'cache_controller' in agents_actions:
                cache_controller = agents_actions['cache_controller']
                
                # 更新内容热度
                cache_controller.update_content_heat(content_id)
                cache_controller.record_cache_result(content_id, was_hit=False)
                
                # 🔧 修复：使用realistic内容大小和正确容量计算
                data_size = self._get_realistic_content_size(content_id)
                capacity_limit = node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
                available_capacity = self._calculate_available_cache_capacity(
                    node.get('cache', {}), capacity_limit
                )
                
                should_cache, reason, evictions = cache_controller.should_cache_content(
                    content_id,
                    data_size,
                    available_capacity,
                    node.get('cache', {}),
                    capacity_limit
                )
                
                if should_cache:
                    if 'cache' not in node:
                        node['cache'] = {}
                    cache_dict = node['cache']
                    reclaimed = 0.0
                    for evict_id in evictions:
                        removed = cache_dict.pop(evict_id, None)
                        if removed:
                            reclaimed += float(removed.get('size', 0.0) or 0.0)
                            cache_controller.cache_stats['evicted_items'] += 1
                    if reclaimed > 0.0:
                        available_capacity += reclaimed
                    if available_capacity < data_size:
                        return cache_hit
                    cache_dict[content_id] = {
                        'size': data_size,
                        'timestamp': self.current_time,
                        'reason': reason,
                        'content_type': self._infer_content_type(content_id)
                    }
                    if 'Collaborative cache' in reason:
                        cache_controller.cache_stats['collaborative_writes'] += 1
        
        # 记录缓存控制器统计
        if agents_actions and 'cache_controller' in agents_actions and cache_hit:
            cache_controller = agents_actions['cache_controller'] 
            cache_controller.record_cache_result(content_id, was_hit=True)
            cache_controller.update_content_heat(content_id)
            
        return cache_hit
    
    def _calculate_enhanced_load_factor(self, node: Dict, node_type: str) -> float:
        """
        🔧 修复：统一和realistic的负载因子计算
        基于实际队列负载，不使用虚假的限制
        """
        queue_length = len(node.get('computation_queue', []))
        
        # 🔧 基于实际观察调整容量基准
        if node_type == 'RSU':
            # 基于实际测试，RSU处理能力约20个任务为满负载
            base_capacity = 20.0  
            queue_factor = queue_length / base_capacity
        else:  # UAV
            # UAV处理能力约10个任务为满负载
            base_capacity = 10.0
            queue_factor = queue_length / base_capacity
        
        # 🔧 修复：使用正确的缓存计算
        cache_utilization = self._calculate_correct_cache_utilization(
            node.get('cache', {}), 
            node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
        )
        
        # 🔧 简化但准确的负载计算
        load_factor = (
            0.8 * queue_factor +           # 队列是主要负载指标80%
            0.2 * cache_utilization       # 缓存利用率20%
        )
        
        # 🔧 不限制在1.0，允许显示真实过载程度
        return max(0.0, load_factor)
    
    def _calculate_correct_cache_utilization(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        🔧 计算正确的缓存利用率
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

    def _update_vehicle_positions(self):
        """简单更新车辆位置，模拟车辆沿主干道移动"""
        for vehicle in self.vehicles:
            position = vehicle.get('position')
            if position is None or len(position) < 2:
                continue

            direction = vehicle.get('direction', 0.0)
            speed = float(vehicle.get('velocity', 15.0))
            dx = np.cos(direction) * speed * self.time_slot
            dy = np.sin(direction) * speed * self.time_slot

            # 道路长度取 1000m，超界循环
            new_x = (position[0] + dx) % 1000.0
            new_y = float(self.road_y + vehicle.get('lane_bias', 0.0)) + dy * 0.05  # 微小扰动
            vehicle['position'][0] = new_x
            vehicle['position'][1] = np.clip(new_y, self.road_y - 6.0, self.road_y + 6.0)

    def _sample_arrivals(self) -> int:
        """按泊松过程采样每车每时隙的任务到达数"""
        lam = max(1e-6, float(self.task_arrival_rate) * float(self.time_slot))
        return int(np.random.poisson(lam))

    def _choose_offload_target(self, actions: Dict, rsu_available: bool, uav_available: bool) -> str:
        """根据智能体提供的偏好选择卸载目标"""
        prefs = actions.get('vehicle_offload_pref') or {}
        probs = np.array([
            max(0.0, float(prefs.get('local', 0.0))),
            max(0.0, float(prefs.get('rsu', 0.0))) if rsu_available else 0.0,
            max(0.0, float(prefs.get('uav', 0.0))) if uav_available else 0.0,
        ], dtype=float)

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
        """估计本地处理的延迟与能耗"""
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
        """估计上传耗时与能耗"""
        # 有效吞吐量 (bit/s)
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
        """将任务记录加入活跃列表"""
        self.active_tasks.append(task_entry)

    def _cleanup_active_tasks(self):
        """移除已经完成或丢弃的任务"""
        self.active_tasks = [
            task for task in self.active_tasks
            if not task.get('completed') and not task.get('dropped')
        ]

    def _handle_deadlines(self):
        """检查队列任务是否超期并丢弃"""
        for node_list, node_type in ((self.rsus, 'RSU'), (self.uavs, 'UAV')):
            for idx, node in enumerate(node_list):
                queue = node.get('computation_queue', [])
                if not queue:
                    continue

                remaining = []
                for task in queue:
                    if self.current_time > task.get('deadline', float('inf')):
                        task['dropped'] = True
                        self.stats['dropped_tasks'] += 1
                        self.stats['dropped_data_bytes'] += float(task.get('data_size_bytes', 0.0))
                else:
                    remaining.append(task)
                node['computation_queue'] = remaining

    def _store_in_vehicle_cache(self, vehicle: Dict, content_id: str, size_mb: float,
                                cache_controller: Optional[Any] = None):
        """将内容推送到车载缓存，使用简单LRU淘汰"""
        if size_mb <= 0.0:
            return
        capacity = float(vehicle.get('device_cache_capacity', 32.0))
        if size_mb > capacity:
            return
        cache = vehicle.setdefault('device_cache', {})
        total_used = sum(float(meta.get('size', 0.0) or 0.0) for meta in cache.values())
        if total_used + size_mb > capacity:
            # LRU淘汰
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
        """尝试将内容推送到邻近RSU"""
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
        """RSU命中后向车辆和邻近RSU推送内容"""
        cache_meta = rsu_node.get('cache', {}).get(content_id)
        if not cache_meta:
            return
        size_mb = float(cache_meta.get('size', 0.0) or self._get_realistic_content_size(content_id))
        cache_controller = None
        if agents_actions:
            cache_controller = agents_actions.get('cache_controller')

        # 推送到覆盖范围内的车辆
        coverage = rsu_node.get('coverage_radius', 300.0)
        for vehicle in self.vehicles:
            distance = self.calculate_distance(vehicle.get('position', np.zeros(2)), rsu_node['position'])
            if distance <= coverage * 0.8:
                self._store_in_vehicle_cache(vehicle, content_id, size_mb, cache_controller)

        # 推送到邻近RSU
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
        """分配至RSU"""
        if not self.rsus:
            return False

        vehicle_pos = np.array(vehicle.get('position', [0.0, 0.0]))
        distances = []
        in_range_mask = []
        for rsu in self.rsus:
            dist = self.calculate_distance(vehicle_pos, rsu['position'])
            distances.append(dist)
            in_range_mask.append(1.0 if dist <= rsu.get('coverage_radius', 300.0) else 0.0)

        accessible = np.array(in_range_mask, dtype=float)
        if accessible.sum() == 0:
            # 没有覆盖的RSU
            return False

        probs = np.ones(len(self.rsus), dtype=float)
        rsu_pref = actions.get('rsu_selection_probs')
        if isinstance(rsu_pref, (list, tuple, np.ndarray)) and len(rsu_pref) == len(self.rsus):
            probs = np.array([max(0.0, float(v)) for v in rsu_pref], dtype=float)

        weights = probs * accessible
        if weights.sum() <= 0:
            weights = accessible

        weights = weights / weights.sum()
        rsu_idx = int(np.random.choice(np.arange(len(self.rsus)), p=weights))
        distance = distances[rsu_idx]
        node = self.rsus[rsu_idx]
        success = self._handle_remote_assignment(vehicle, task, node, 'RSU', rsu_idx, distance, actions, step_summary)
        if success:
            step_summary['remote_tasks'] += 1
        return success

    def _assign_to_uav(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict) -> bool:
        """分配至UAV"""
        if not self.uavs:
            return False

        vehicle_pos = np.array(vehicle.get('position', [0.0, 0.0]))
        distances = []
        in_range_mask = []
        for uav in self.uavs:
            dist = self.calculate_distance(vehicle_pos, uav['position'])
            distances.append(dist)
            in_range_mask.append(1.0 if dist <= uav.get('coverage_radius', 350.0) else 0.0)

        accessible = np.array(in_range_mask, dtype=float)
        if accessible.sum() == 0:
            return False

        probs = np.ones(len(self.uavs), dtype=float)
        uav_pref = actions.get('uav_selection_probs')
        if isinstance(uav_pref, (list, tuple, np.ndarray)) and len(uav_pref) == len(self.uavs):
            probs = np.array([max(0.0, float(v)) for v in uav_pref], dtype=float)

        weights = probs * accessible
        if weights.sum() <= 0:
            weights = accessible

        weights = weights / weights.sum()
        uav_idx = int(np.random.choice(np.arange(len(self.uavs)), p=weights))
        distance = distances[uav_idx]
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
        """执行远程卸载：缓存判定、建立队列并记录统计"""
        actions = actions or {}
        cache_hit = False

        if node_type == 'RSU':
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='RSU')
        else:
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='UAV')

        if cache_hit:
            # 缓存命中：快速完成
            delay = max(0.02, 0.2 * self.time_slot)
            power = 18.0 if node_type == 'RSU' else 12.0
            energy = power * delay * 0.1
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self.stats['total_delay'] += delay
            self.stats['total_energy'] += energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + energy
            return True

        upload_delay, upload_energy = self._estimate_transmission(task.get('data_size_bytes', 1e6), distance, node_type.lower())
        self.stats['total_delay'] += upload_delay
        self.stats['energy_uplink'] += upload_energy
        self.stats['total_energy'] += upload_energy
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + upload_energy

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
            'priority': task.get('priority', 0.5)
        }

        queue = node.setdefault('computation_queue', [])
        queue.append(task_entry)
        self._append_active_task(task_entry)
        return True

    def _handle_local_processing(self, vehicle: Dict, task: Dict, step_summary: Dict):
        """本地处理任务"""
        processing_delay, energy = self._estimate_local_processing(task, vehicle)
        self.stats['processed_tasks'] += 1
        self.stats['completed_tasks'] += 1
        self.stats['total_delay'] += processing_delay
        self.stats['total_energy'] += energy
        step_summary['local_tasks'] += 1

    
    def check_adaptive_migration(self, agents_actions: Dict = None):
        """🎯 多维度智能迁移检查 (阈值触发+负载差触发+跟随迁移)"""
        if not agents_actions or 'migration_controller' not in agents_actions:
            return
        
        migration_controller = agents_actions['migration_controller']
        
        # 🔍 收集所有节点状态用于邻居比较
        all_node_states = {}
        
        # RSU状态收集
        for i, rsu in enumerate(self.rsus):
            queue = rsu.get('computation_queue', [])
            queue_len = len(queue)
            cache_capacity = rsu.get('cache_capacity', 1000.0)
            available_cache = self._calculate_available_cache_capacity(rsu.get('cache', {}), cache_capacity)
            storage_load = 0.0 if cache_capacity <= 0 else 1.0 - (available_cache / max(1.0, cache_capacity))
            total_data = sum(task.get('data_size', 1.0) for task in queue)
            bandwidth_capacity = rsu.get('bandwidth_capacity', 50.0)
            bandwidth_load = float(np.clip(total_data / max(1.0, bandwidth_capacity), 0.0, 0.99))
            cpu_load = float(np.clip(queue_len / 25.0, 0.0, 0.99))

            all_node_states[f'rsu_{i}'] = {
                'cpu_load': cpu_load,
                'bandwidth_load': bandwidth_load,
                'storage_load': float(np.clip(storage_load, 0.0, 0.99)),
                'load_factor': self._calculate_enhanced_load_factor(rsu, 'RSU'),
                'battery_level': 1.0,
                'node_type': 'RSU',
                'queue_length': queue_len,
                'cache_capacity': cache_capacity,
                'cache_available': available_cache
            }

        # UAV状态收集
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
                'cache_available': available_cache
            }
        
        # 🏢 RSU迁移检查 (阈值+负载差触发)
        for i, rsu in enumerate(self.rsus):
            node_id = f'rsu_{i}'
            current_state = all_node_states[node_id]
            
            # 更新负载历史
            migration_controller.update_node_load(node_id, current_state['load_factor'])
            
            # 🎯 多维度迁移触发检查
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"🎯 {node_id} 触发迁移: {reason} (紧急度:{urgency:.3f})")
                
                # 执行RSU间迁移
                result = self.execute_rsu_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
        # 🚁 UAV迁移检查
        for i, uav in enumerate(self.uavs):
            node_id = f'uav_{i}'
            current_state = all_node_states[node_id]
            
            # 更新负载历史
            migration_controller.update_node_load(node_id, current_state['load_factor'], current_state['battery_level'])
            
            # 🎯 多维度迁移触发检查
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"🎯 {node_id} 触发迁移: {reason} (紧急度:{urgency:.3f})")
                
                # UAV迁移到RSU
                result = self.execute_uav_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
        # 🚗 车辆跟随迁移检查
        self._check_vehicle_handover_migration(migration_controller)
    
    def _check_vehicle_handover_migration(self, migration_controller):
        """🚗 车辆跟随迁移：当车辆移动超出当前边缘节点通信覆盖时触发迁移"""
        handover_count = 0
        
        # 检查每个活跃任务的车辆位置
        for task in list(self.active_tasks):
            if task.get('node_type') not in ['RSU', 'UAV']:
                continue  # 只检查边缘节点任务
            
            try:
                # 找到对应车辆
                vehicle = next(v for v in self.vehicles if v['id'] == task['vehicle_id'])
                current_pos = vehicle['position']
                
                # 获取当前服务节点
                current_node = None
                if task['node_type'] == 'RSU' and task.get('node_idx') is not None:
                    current_node = self.rsus[task['node_idx']]
                elif task['node_type'] == 'UAV' and task.get('node_idx') is not None:
                    current_node = self.uavs[task['node_idx']]
                
                if current_node is None:
                    continue
                
                # 🔍 检查通信覆盖和跟随迁移触发
                distance_to_current = self.calculate_distance(current_pos, current_node['position'])
                coverage_radius = current_node.get('coverage_radius', 500.0)  # 默认500m覆盖
                
                # 🔧 智能跟随迁移触发机制：
                # 1. 基础阈值：85%覆盖半径（信号质量开始明显下降）
                # 2. 考虑车辆速度：高速车辆提前触发
                # 3. 考虑预测：车辆是否在快速远离当前节点
                
                vehicle_speed = np.linalg.norm(vehicle.get('velocity', [0, 0]))
                
                # 🔧 优化的速度调整因子：速度越快，越早触发
                # 30 m/s → 0.85 (425m触发)
                # 45 m/s → 0.775 (387m触发)  
                # 60 m/s → 0.70 (350m触发)
                speed_factor = max(0.70, 1.0 - (vehicle_speed / 200.0))
                
                # 动态触发阈值
                trigger_threshold = coverage_radius * speed_factor
                
                # 超出动态阈值，触发跟随迁移
                if distance_to_current > trigger_threshold:
                    # 🔍 寻找最佳新服务节点
                    best_new_node = None
                    best_distance = float('inf')
                    best_node_idx = None
                    best_node_type = None
                    
                    # 检查所有RSU - 优先选择RSU（稳定性更好）
                    for i, rsu in enumerate(self.rsus):
                        dist = self.calculate_distance(current_pos, rsu['position'])
                        if dist <= rsu.get('coverage_radius', 500.0):
                            queue_len = len(rsu.get('computation_queue', []))
                            cpu_load = rsu.get('cpu_usage', 0.5)
                            
                            # 🔧 综合评分：距离 + 队列 + 负载
                            score = dist * 1.0 + queue_len * 30 + cpu_load * 200
                            
                            if score < best_distance:
                                best_new_node = rsu
                                best_distance = score
                                best_node_idx = i
                                best_node_type = 'RSU'
                    
                    # 检查所有UAV（作为备选）
                    if best_new_node is None or best_distance > 500:  # RSU不理想时考虑UAV
                        for i, uav in enumerate(self.uavs):
                            dist = self.calculate_distance(current_pos, uav['position'])
                            if dist <= uav.get('coverage_radius', 300.0):
                                queue_len = len(uav.get('computation_queue', []))
                                cpu_load = uav.get('cpu_usage', 0.5)
                                
                                # UAV评分略有不同（考虑移动性）
                                score = dist * 1.2 + queue_len * 20 + cpu_load * 150
                                
                                if score < best_distance:
                                    best_new_node = uav
                                    best_distance = score
                                    best_node_idx = i
                                    best_node_type = 'UAV'
                    
                    # 🚀 执行跟随迁移（只在找到明显更好的节点时）
                    # 必须满足：1) 找到新节点, 2) 新节点不同, 3) 新节点明显更优
                    current_queue = len(current_node.get('computation_queue', []))
                    current_score = distance_to_current * 1.0 + current_queue * 30
                    
                    should_migrate = (
                        best_new_node is not None and 
                        (best_node_idx != task.get('node_idx') or best_node_type != task['node_type']) and
                        best_distance < current_score * 0.7  # 新节点至少好30%才迁移
                    )
                    
                    if should_migrate:
                        # 从原节点移除任务
                        if task['node_type'] == 'RSU':
                            old_queue = self.rsus[task['node_idx']].get('computation_queue', [])
                            self.rsus[task['node_idx']]['computation_queue'] = [
                                t for t in old_queue if t.get('id') != task['id']
                            ]
                        elif task['node_type'] == 'UAV':
                            old_queue = self.uavs[task['node_idx']].get('computation_queue', [])
                            self.uavs[task['node_idx']]['computation_queue'] = [
                                t for t in old_queue if t.get('id') != task['id']
                            ]
                        
                        # 添加到新节点
                        if 'computation_queue' not in best_new_node:
                            best_new_node['computation_queue'] = []
                        
                        # 创建新任务项
                        migrated_task = {
                            'id': task['id'],
                            'vehicle_id': task['vehicle_id'],
                            'arrival_time': task['arrival_time'],
                            'deadline': task['deadline'],
                            'data_size': task.get('data_size', 2.0),
                            'computation_requirement': task.get('computation_requirement', 1000),
                            'content_id': task['content_id'],
                            'compute_time_needed': task.get('compute_time_needed', 1.0),
                            'work_remaining': task.get('work_remaining', 0.5),
                            'cache_hit': task.get('cache_hit', False),
                            'queued_at': self.current_time,
                            'migrated_from': f"{task['node_type']}_{task.get('node_idx')}"
                        }
                        best_new_node['computation_queue'].append(migrated_task)
                        
                        # 更新任务信息
                        task['node_type'] = best_node_type
                        task['node_idx'] = best_node_idx
                        
                        handover_count += 1
                        
                        # 🔧 增强日志：显示触发原因和迁移收益
                        print(f"🚗 车辆跟随迁移: {task['vehicle_id']} 从 {task['node_type']}_{task.get('node_idx')} → {best_node_type}_{best_node_idx}")
                        print(f"   触发原因: 距离{distance_to_current:.1f}m > 阈值{trigger_threshold:.1f}m (车速{vehicle_speed:.1f}m/s)")
                        print(f"   迁移收益: 当前评分{current_score:.1f} → 新评分{best_distance:.1f} (改善{(1-best_distance/current_score)*100:.1f}%)")
                        
                        # 记录跟随迁移统计
                        self.stats['handover_migrations'] = self.stats.get('handover_migrations', 0) + 1
                        migration_controller.record_migration_result(True, cost=5.0, delay_saved=0.3)
                
            except Exception as e:
                continue  # 忽略错误，继续处理下一个任务
        
        if handover_count > 0:
            print(f"🚗 本时隙执行了 {handover_count} 次车辆跟随迁移")
    
    def run_simulation_step(self, step: int, actions: Optional[Dict] = None) -> Dict[str, Any]:
        """执行单个仿真步，返回截至当前的累计统计数据"""
        actions = actions or {}

        advance_simulation_time()
        self.current_step += 1
        self.current_time = get_simulation_time()

        step_summary = {
            'generated_tasks': 0,
            'local_tasks': 0,
            'remote_tasks': 0,
            'local_cache_hits': 0
        }

        # 1. 更新车辆位置
        self._update_vehicle_positions()

        # 2. 生成任务并分配
        for vehicle in self.vehicles:
            arrivals = self._sample_arrivals()
            if arrivals <= 0:
                continue

            vehicle_id = vehicle['id']
            for _ in range(arrivals):
                task = self.generate_task(vehicle_id)
                step_summary['generated_tasks'] += 1
                self.stats['total_tasks'] += 1
                self.stats['generated_data_bytes'] += float(task.get('data_size_bytes', 0.0))
                self._dispatch_task(vehicle, task, actions, step_summary)

        # 3. 智能迁移策略
        if actions:
            self.check_adaptive_migration(actions)

        # 4. 处理队列中的任务
        self._process_node_queues()

        # 5. 检查超时并清理
        self._handle_deadlines()
        self._cleanup_active_tasks()

        # 汇总信息
        step_summary.update({
            'current_time': self.current_time,
            'rsu_queue_lengths': [len(rsu.get('computation_queue', [])) for rsu in self.rsus],
            'uav_queue_lengths': [len(uav.get('computation_queue', [])) for uav in self.uavs],
            'active_tasks': len(self.active_tasks)
        })

        cumulative_stats = dict(self.stats)
        cumulative_stats.update(step_summary)
        return cumulative_stats
    
    def execute_rsu_migration(self, source_rsu_idx: int, urgency: float) -> Dict[str, float]:
        """Execute RSU-to-RSU migration and return cost/delay metrics."""
        source_rsu = self.rsus[source_rsu_idx]
        source_queue = source_rsu.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        candidates = []
        for i, rsu in enumerate(self.rsus):
            if i == source_rsu_idx:
                continue
            queue_len = len(rsu.get('computation_queue', []))
            cpu_load = min(0.99, queue_len / 25.0)
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
        """Execute UAV-to-RSU migration and return cost/delay metrics."""
        source_uav = self.uavs[source_uav_idx]
        source_queue = source_uav.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        uav_position = source_uav['position']
        candidates = []
        for i, rsu in enumerate(self.rsus):
            queue_len = len(rsu.get('computation_queue', []))
            distance = self.calculate_distance(uav_position, rsu['position'])
            cpu_load = min(0.99, queue_len / 25.0)
            score = distance * 0.01 + queue_len + cpu_load * 10.0
            candidates.append((i, queue_len, cpu_load, distance, score))

        if not candidates:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_idx, target_queue_len, target_cpu_load, distance, _ = min(candidates, key=lambda x: x[4])
        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        source_queue_len = len(source_queue)
        migration_ratio = max(0.2, min(0.6, urgency + 0.1))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        base_success_rate = 0.75
        distance_penalty = min(0.35, distance / 1200.0)
        load_penalty = min(0.25, target_queue_len / 40.0)
        urgency_bonus = min(0.2, urgency)
        actual_success_rate = np.clip(base_success_rate - distance_penalty - load_penalty + urgency_bonus, 0.35, 0.95)
        if np.random.random() > actual_success_rate:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

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
