#!/usr/bin/env python3
"""
完整系统仿真器
用于测试完整的车联网边缘缓存系统
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

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
        if self.sys_config is not None:
            self.num_vehicles = getattr(self.sys_config.network, 'num_vehicles', 12)
            self.num_rsus = getattr(self.sys_config.network, 'num_rsus', 6)
            self.num_uavs = getattr(self.sys_config.network, 'num_uavs', 2)
        else:
            self.num_vehicles = self.config.get('num_vehicles', 12)
            self.num_rsus = self.config.get('num_rsus', 6)
            self.num_uavs = self.config.get('num_uavs', 2)
        
        # 仿真参数
        if self.sys_config is not None:
            self.simulation_time = getattr(self.sys_config, 'simulation_time', 1000)
            self.time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)  # 🚀 适应高负载时隙
            self.task_arrival_rate = getattr(self.sys_config.task, 'arrival_rate', 2.5)  # 🚀 高负载到达率
        else:
            self.simulation_time = self.config.get('simulation_time', 1000)
            self.time_slot = self.config.get('time_slot', 0.2)  # 🚀 高负载默认时隙
            self.task_arrival_rate = self.config.get('task_arrival_rate', 2.5)  # 🚀 高负载默认到达率
        
        # 性能统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        # 跨时隙在制任务管理
        self.active_tasks: List[Dict] = []  # 每项: {id, vehicle_id, arrival_time, deadline, work_remaining, node_type, node_idx}
        self.task_counter = 0
        
        # 初始化组件
        self.initialize_components()
    
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
        # 🚦 十字路口车辆初始化 - 右下角十字路口出发
        self.vehicles = []
        intersection_center = [750, 750]  # 右下角十字路口中心位置
        
        for i in range(self.num_vehicles):
            # 🚗 从十字路口的四个方向随机选择一个出发方向
            directions = [0, np.pi/2, np.pi, 3*np.pi/2]  # 东、北、西、南
            base_direction = np.random.choice(directions)
            
            # 在十字路口附近的随机位置生成车辆（模拟真实交通流）
            spawn_offset = np.random.uniform(-30, 30, 2)  # 十字路口±30m范围内
            initial_position = np.array(intersection_center) + spawn_offset
            
            # 确保车辆在有效范围内
            initial_position = np.clip(initial_position, [50, 50], [950, 950])
            
            # 添加随机偏移到基础方向，模拟真实驾驶行为
            direction_noise = np.random.uniform(-np.pi/8, np.pi/8)  # ±22.5度随机偏移
            final_direction = base_direction + direction_noise
            
            vehicle = {
                'id': f'V_{i}',
                'position': initial_position,
                'velocity': np.random.uniform(15, 25),  # 15-25 m/s (城市交通速度)
                'direction': final_direction,
                'tasks': [],
                'energy_consumed': 0.0,
                'spawn_time': 0.0,  # 记录生成时间
                'from_intersection': True,  # 标记从十字路口出发
                'target_direction': base_direction  # 记录目标方向用于导航
            }
            self.vehicles.append(vehicle)
            
        print(f"🚦 所有车辆从右下角十字路口({intersection_center})出发完成")
        
        # RSU节点
        self.rsus = []
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': np.random.uniform(0, 1000, 2),
                'coverage_radius': (getattr(self.sys_config.network, 'coverage_radius', 200) if self.sys_config is not None else 200),
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'rsu_cache_capacity', 10e9) if self.sys_config is not None else 10e9),
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.rsus.append(rsu)
        
        # UAV节点
        self.uavs = []
        for i in range(self.num_uavs):
            uav = {
                'id': f'UAV_{i}',
                'position': np.random.uniform(0, 1000, 3),  # x, y, z坐标
                'velocity': np.random.uniform(20, 50),
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
    
    def _setup_scenario(self):
        """设置仿真场景"""
        # 重新初始化组件（如果需要）
        self.initialize_components()
        print("✓ 初始化了 6 个缓存管理器")
    
    def generate_task(self, vehicle_id: str) -> Dict:
        """生成计算任务 - 使用分层任务类型设计"""
        self.task_counter += 1
        
        # 🔧 新设计：先确定任务类型，再分配对应参数
        if self.sys_config is not None:
            # 随机选择任务类型（1-4）
            task_type = np.random.randint(1, 5)
            
            # 获取任务类型特化参数
            task_specs = getattr(self.sys_config.task, 'task_type_specs', {})
            if task_type in task_specs:
                spec = task_specs[task_type]
                data_range = spec['data_range']
                compute_density = spec['compute_density']
            else:
                # 回退到通用参数
                data_range = getattr(self.sys_config.task, 'data_size_range', (0.5e6/8, 15e6/8))
                compute_density = float(getattr(self.sys_config.task, 'task_compute_density', 400))
            
            # 根据任务类型分配deadline
            delay_thresholds = getattr(self.sys_config.task, 'delay_thresholds', {})
            time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)
            
            if task_type == 1:  # 极敏感
                max_slots = delay_thresholds.get('extremely_sensitive', 4)
                deadline_duration = np.random.uniform(0.5, max_slots * time_slot)
            elif task_type == 2:  # 敏感
                max_slots = delay_thresholds.get('sensitive', 10)
                deadline_duration = np.random.uniform(1.0, max_slots * time_slot)
            elif task_type == 3:  # 中度容忍
                max_slots = delay_thresholds.get('moderately_tolerant', 25)
                deadline_duration = np.random.uniform(2.0, max_slots * time_slot)
            else:  # 延迟容忍
                deadline_duration = np.random.uniform(5.0, 15.0)
            
            # 数据大小：从类型特定范围采样
            data_size_bytes = np.random.uniform(data_range[0], data_range[1])
            data_size_mb = data_size_bytes / 1e6  # 转MB用于兼容
            
            # 计算需求：基于数据大小和类型特定计算密度
            total_bits = data_size_bytes * 8
            computation_cycles = total_bits * compute_density
            computation_mips = computation_cycles / 1e6  # 转为MIPS单位以兼容旧接口
        else:
            # 回退默认值
            task_type = np.random.randint(1, 5)
            data_size_mb = np.random.exponential(0.5)  # 更小的默认数据
            data_size_bytes = data_size_mb * 1e6
            computation_mips = np.random.exponential(80)  # 降低默认计算需求
            deadline_duration = np.random.uniform(0.5, 3.0)
            compute_density = 400  # 设置默认密度
        
        # 🚀 12车辆高负载场景：任务复杂度增强
        high_load_mode = self.config.get('high_load_mode', False)
        if high_load_mode:
            complexity_multiplier = self.config.get('task_complexity_multiplier', 2.0)
            
            # 增强计算需求
            computation_mips *= complexity_multiplier
            
            # 适度增加数据大小（限制最大值避免过度）
            data_size_mb = min(data_size_mb * 1.2, 3.0)
            data_size_bytes = data_size_mb * 1e6
            
            # 增强计算密度
            compute_density *= 1.1
        
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
            'compute_density': compute_density,  # 🚀 高负载增强计算密度
            'complexity_multiplier': self.config.get('task_complexity_multiplier', 1.0)  # 🚀 复杂度标记
        }
        
        self.stats['total_tasks'] += 1
        return task
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """计算两点间距离"""
        if len(pos1) == 3 and len(pos2) == 2:
            pos2 = np.append(pos2, 0)  # 2D转3D
        elif len(pos1) == 2 and len(pos2) == 3:
            pos1 = np.append(pos1, 0)
        
        return np.linalg.norm(pos1 - pos2)
    
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
    
    def check_cache_hit_adaptive(self, content_id: str, node: Dict, agents_actions: Dict = None) -> bool:
        """🤖 智能体控制的自适应缓存检查"""
        # 基础缓存检查
        cache_hit = content_id in node.get('cache', {})
        
        # 更新统计
        if cache_hit:
            self.stats['cache_hits'] += 1
        else:
            self.stats['cache_misses'] += 1
            
            # 🤖 如果有智能体控制器，执行自适应缓存策略
            if agents_actions and 'cache_controller' in agents_actions:
                cache_controller = agents_actions['cache_controller']
                
                # 更新内容热度
                cache_controller.update_content_heat(content_id)
                cache_controller.record_cache_result(content_id, was_hit=False)
                
                # 检查是否应该缓存此内容
                data_size = 1.0  # 默认大小MB
                available_capacity = node.get('cache_capacity', 100) - len(node.get('cache', {}))
                
                should_cache, reason = cache_controller.should_cache_content(
                    content_id, data_size, available_capacity
                )
                
                if should_cache:
                    # 执行缓存操作
                    if 'cache' not in node:
                        node['cache'] = {}
                    node['cache'][content_id] = {
                        'size': data_size,
                        'timestamp': self.current_time,
                        'reason': reason
                    }
        
        # 记录缓存控制器统计
        if agents_actions and 'cache_controller' in agents_actions and cache_hit:
            cache_controller = agents_actions['cache_controller'] 
            cache_controller.record_cache_result(content_id, was_hit=True)
            cache_controller.update_content_heat(content_id)
            
        return cache_hit
    
    def _calculate_enhanced_load_factor(self, node: Dict, node_type: str) -> float:
        """🚀 增强的负载因子计算 - 12车辆高负载场景优化"""
        queue_length = len(node.get('computation_queue', []))
        
        # 根据节点类型设置容量参数
        if node_type == 'RSU':
            base_capacity = 6.0  # 12车辆高负载优化
            queue_factor = queue_length / base_capacity
        else:  # UAV
            base_capacity = 3.5  # 12车辆高负载优化
            queue_factor = queue_length / base_capacity
        
        # 多维度负载评估
        cpu_utilization = min(0.9, queue_length * 0.2)  # CPU利用率
        
        # 缓存负载评估
        cache_size = len(node.get('cache', {}))
        cache_capacity = node.get('cache_capacity', 100)
        memory_utilization = cache_size / max(cache_capacity, 1)
        
        # 任务复杂度影响
        complexity_factor = 2.0  # 12车辆高负载场景复杂度
        
        # 加权综合负载
        load_factor = (
            0.7 * queue_factor * complexity_factor +  # 队列负载70%
            0.25 * cpu_utilization +                  # CPU利用率25%  
            0.05 * memory_utilization                 # 内存利用率5%
        )
        
        return min(1.0, load_factor)  # 限制在[0,1]范围
    
    def check_adaptive_migration(self, agents_actions: Dict = None):
        """🎯 多维度智能迁移检查 (阈值触发+负载差触发+跟随迁移)"""
        if not agents_actions or 'migration_controller' not in agents_actions:
            return
        
        migration_controller = agents_actions['migration_controller']
        
        # 🔍 收集所有节点状态用于邻居比较
        all_node_states = {}
        
        # RSU状态收集
        for i, rsu in enumerate(self.rsus):
            queue_len = len(rsu.get('computation_queue', []))
            all_node_states[f'rsu_{i}'] = {
                'cpu_load': min(0.95, queue_len * 0.15),  # 基于队列长度估算CPU负载
                'bandwidth_load': np.random.uniform(0.3, 0.9),  # 模拟带宽使用率
                'storage_load': np.random.uniform(0.2, 0.8),    # 模拟存储使用率
                'load_factor': self._calculate_enhanced_load_factor(rsu, 'RSU'),
                'battery_level': 1.0,
                'node_type': 'RSU',
                'queue_length': queue_len
            }
        
        # UAV状态收集
        for i, uav in enumerate(self.uavs):
            queue_len = len(uav.get('computation_queue', []))
            all_node_states[f'uav_{i}'] = {
                'cpu_load': min(0.95, queue_len * 0.2),  # UAV负载计算稍高
                'bandwidth_load': np.random.uniform(0.4, 0.9),  # UAV带宽压力更大
                'storage_load': np.random.uniform(0.1, 0.5),    # UAV存储较少
                'load_factor': self._calculate_enhanced_load_factor(uav, 'UAV'),
                'battery_level': uav.get('battery_level', 1.0),
                'node_type': 'UAV',
                'queue_length': queue_len
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
                success = self.execute_rsu_migration(i, urgency)
                if success:
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=10.0, delay_saved=0.5)
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
                success = self.execute_uav_migration(i, urgency)
                if success:
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=20.0, delay_saved=1.0)
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
                
                # 🔍 检查通信覆盖
                distance_to_current = self.calculate_distance(current_pos, current_node['position'])
                coverage_radius = current_node.get('coverage_radius', 500.0)  # 默认500m覆盖
                
                # 超出覆盖范围，触发跟随迁移
                if distance_to_current > coverage_radius * 1.2:  # 120%覆盖半径外触发
                    # 🔍 寻找最佳新服务节点
                    best_new_node = None
                    best_distance = float('inf')
                    best_node_idx = None
                    best_node_type = None
                    
                    # 检查所有RSU
                    for i, rsu in enumerate(self.rsus):
                        dist = self.calculate_distance(current_pos, rsu['position'])
                        if dist <= rsu.get('coverage_radius', 500.0) and dist < best_distance:
                            queue_len = len(rsu.get('computation_queue', []))
                            # 考虑距离和负载的综合评分
                            score = dist + queue_len * 50  # 队列长度权重
                            if score < best_distance:
                                best_new_node = rsu
                                best_distance = score
                                best_node_idx = i
                                best_node_type = 'RSU'
                    
                    # 检查所有UAV (如果没有合适的RSU)
                    if best_new_node is None:
                        for i, uav in enumerate(self.uavs):
                            dist = self.calculate_distance(current_pos, uav['position'])
                            if dist <= uav.get('coverage_radius', 300.0) and dist < best_distance:
                                queue_len = len(uav.get('computation_queue', []))
                                score = dist + queue_len * 30
                                if score < best_distance:
                                    best_new_node = uav
                                    best_distance = score
                                    best_node_idx = i
                                    best_node_type = 'UAV'
                    
                    # 🚀 执行跟随迁移
                    if best_new_node is not None and (best_node_idx != task.get('node_idx') or best_node_type != task['node_type']):
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
                        
                        print(f"🚗 车辆跟随迁移: {task['vehicle_id']} 从 {task.get('migrated_from', 'unknown')} → {best_node_type}_{best_node_idx} (距离:{distance_to_current:.1f}m > {coverage_radius:.1f}m)")
                        
                        # 记录跟随迁移统计
                        self.stats['handover_migrations'] = self.stats.get('handover_migrations', 0) + 1
                        migration_controller.record_migration_result(True, cost=5.0, delay_saved=0.3)
                
            except Exception as e:
                continue  # 忽略错误，继续处理下一个任务
        
        if handover_count > 0:
            print(f"🚗 本时隙执行了 {handover_count} 次车辆跟随迁移")
    
    def execute_rsu_migration(self, source_rsu_idx: int, urgency: float) -> bool:
        """🔌 RSU间任务迁移 - 基于有线回传网络"""
        source_rsu = self.rsus[source_rsu_idx]
        source_queue = source_rsu.get('computation_queue', [])
        
        if not source_queue:
            print(f"⚠️ RSU_{source_rsu_idx} 队列为空，无法迁移")
            return False
        
        # 🎯 智能目标RSU选择：排除源RSU，综合考虑队列长度和负载
        candidates = []
        for i in range(len(self.rsus)):
            if i != source_rsu_idx:  # 排除源RSU
                rsu = self.rsus[i]
                queue_len = len(rsu.get('computation_queue', []))
                cpu_load = min(0.95, queue_len * 0.15)  # 估算CPU负载
                
                # 综合评分：队列长度 + 负载权重
                score = queue_len + cpu_load * 10  # 负载权重更高
                candidates.append((i, queue_len, cpu_load, score))
        
        if not candidates:
            print(f"⚠️ RSU_{source_rsu_idx} 找不到合适的迁移目标")
            return False
        
        # 选择评分最低的RSU作为目标
        target_idx, target_queue_len, target_cpu_load, _ = min(candidates, key=lambda x: x[3])
        source_queue_len = len(source_queue)
        
        # 🎯 负载差检查：只要目标不比源更忙即可迁移
        if target_queue_len > source_queue_len:
            print(f"⚠️ RSU_{source_rsu_idx}→RSU_{target_idx} 目标更忙，放弃迁移 (源:{source_queue_len} vs 目标:{target_queue_len})")
            return False
        
        # 🔥 确保至少迁移1个任务
        migration_ratio = max(0.1, min(0.5, urgency))  # 最少10%，最多50%
        tasks_to_migrate = max(1, int(len(source_queue) * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, len(source_queue))
        
        if tasks_to_migrate > 0:
            target_rsu = self.rsus[target_idx]
            if 'computation_queue' not in target_rsu:
                target_rsu['computation_queue'] = []
            
            # 🔌 计算有线传输成本
            source_rsu_id = source_rsu['id']
            target_rsu_id = target_rsu['id']
            
            # 估算迁移数据大小 (任务元数据 + 中间结果)
            avg_task_size = 2.0  # MB per task (metadata + partial results)
            total_data_size = tasks_to_migrate * avg_task_size
            
            try:
                from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
                
                # 计算有线传输延迟和能耗
                wired_delay = calculate_rsu_to_rsu_delay(total_data_size, source_rsu_id, target_rsu_id)
                wired_energy = calculate_rsu_to_rsu_energy(total_data_size, source_rsu_id, target_rsu_id, wired_delay)
                
                # 执行迁移
                migrated_tasks = source_queue[:tasks_to_migrate]
                source_rsu['computation_queue'] = source_queue[tasks_to_migrate:]
                target_rsu['computation_queue'].extend(migrated_tasks)
                
                # 记录有线传输成本
                self.stats['rsu_migration_delay'] = self.stats.get('rsu_migration_delay', 0.0) + wired_delay
                self.stats['rsu_migration_energy'] = self.stats.get('rsu_migration_energy', 0.0) + wired_energy
                self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + total_data_size
                
                print(f"🔌 RSU迁移 {source_rsu_id}→{target_rsu_id}: {tasks_to_migrate}个任务, 有线传输{total_data_size:.1f}MB, 延迟{wired_delay*1000:.2f}ms")
                
                return True
                
            except Exception as e:
                print(f"⚠️ 有线传输计算失败，使用简化迁移: {e}")
                # 回退到简单迁移
                migrated_tasks = source_queue[:tasks_to_migrate]
                source_rsu['computation_queue'] = source_queue[tasks_to_migrate:]
                target_rsu['computation_queue'].extend(migrated_tasks)
                return True
        
        return False
    
    def execute_uav_migration(self, source_uav_idx: int, urgency: float) -> bool:
        """🚁 UAV到RSU的任务迁移 - 无线到有线网络"""
        source_uav = self.uavs[source_uav_idx]
        source_queue = source_uav.get('computation_queue', [])
        
        if not source_queue:
            print(f"⚠️ UAV_{source_uav_idx} 队列为空，无法迁移")
            return False
        
        # 🎯 智能目标RSU选择：综合考虑队列、负载和距离
        candidates = []
        uav_position = source_uav['position']
        
        for i, rsu in enumerate(self.rsus):
            queue_len = len(rsu.get('computation_queue', []))
            cpu_load = min(0.95, queue_len * 0.15)
            
            # 计算UAV到RSU的距离
            distance = self.calculate_distance(uav_position, rsu['position'])
            
            # 综合评分：队列 + 负载 + 距离权重
            score = queue_len + cpu_load * 10 + distance * 0.01
            candidates.append((i, queue_len, cpu_load, distance, score))
        
        if not candidates:
            return False
        
        # 选择综合评分最低的RSU
        target_idx, target_queue_len, target_cpu_load, distance, _ = min(candidates, key=lambda x: x[4])
        source_queue_len = len(source_queue)
        
        # 🔥 UAV迁移条件更宽松（因为无线链路比有线更不稳定）
        max_acceptable_queue = source_queue_len + 10  # RSU可以接受更多任务
        if target_queue_len > max_acceptable_queue:
            print(f"⚠️ UAV_{source_uav_idx}→RSU_{target_idx} 目标RSU太忙，放弃迁移 (目标:{target_queue_len} > 限制:{max_acceptable_queue})")
            return False
        
        # 🚀 执行迁移
        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []
        
        # 计算无线传输成本
        tasks_to_migrate = len(source_queue)
        migration_data_size = tasks_to_migrate * 1.5  # UAV任务通常较小
        
        # 📡 记录无线到有线的混合传输
        wireless_delay = distance * 0.001  # 简化的无线传输延迟
        
        target_rsu['computation_queue'].extend(source_queue)
        source_uav['computation_queue'] = []
        
        # 记录UAV迁移统计
        self.stats['uav_migration_count'] = self.stats.get('uav_migration_count', 0) + 1
        self.stats['uav_migration_distance'] = self.stats.get('uav_migration_distance', 0.0) + distance
        
        print(f"🚁 UAV迁移 UAV_{source_uav_idx}→RSU_{target_idx}: {tasks_to_migrate}个任务, 距离{distance:.1f}m, 无线延迟{wireless_delay*1000:.2f}ms")
        
        return True
    
    def _execute_central_rsu_scheduling(self):
        """🏢 执行中央RSU全局调度 - 基于有线回传网络"""
        try:
            # 🔌 模拟有线网络信息收集延迟
            info_collection_start = self.current_time
            
            # 为RSU添加必要的状态信息
            central_rsu_id = f"RSU_{2 if self.num_rsus > 2 else 0}"
            print(f"🔍 {central_rsu_id}通过有线网络收集RSU负载信息...")
            
            total_collection_delay = 0.0
            total_collection_energy = 0.0
            
            for i, rsu in enumerate(self.rsus):
                rsu_id = rsu['id']
                
                # 跳过中央RSU自己
                if rsu_id == central_rsu_id:
                    continue
                
                # 计算信息收集的有线传输成本
                info_size_mb = 0.1  # 100KB的状态信息
                try:
                    from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
                    
                    collection_delay = calculate_rsu_to_rsu_delay(info_size_mb, rsu_id, central_rsu_id)
                    collection_energy = calculate_rsu_to_rsu_energy(info_size_mb, rsu_id, central_rsu_id, collection_delay)
                    
                    total_collection_delay += collection_delay
                    total_collection_energy += collection_energy
                    
                except Exception:
                    # 回退到简化模型
                    collection_delay = 0.005  # 5ms固定延迟
                    collection_energy = 0.1   # 0.1J固定能耗
                    total_collection_delay += collection_delay
                    total_collection_energy += collection_energy
                
                # 更新RSU状态信息
                if 'cpu_usage' not in rsu:
                    queue_len = len(rsu.get('computation_queue', []))
                    rsu['cpu_usage'] = min(0.9, queue_len * 0.15)
                if 'cache_hit_rate' not in rsu:
                    rsu['cache_hit_rate'] = np.random.uniform(0.3, 0.8)
                if 'avg_response_time' not in rsu:
                    rsu['avg_response_time'] = rsu['cpu_usage'] * 100 + 50
                if 'task_completion_rate' not in rsu:
                    rsu['task_completion_rate'] = max(0.1, 1.0 - rsu['cpu_usage'])
            
            # 收集负载信息
            rsu_loads = self.central_scheduler.collect_all_rsu_loads(self.rsus)
            
            # 📈 生成全局调度决策
            estimated_tasks = max(1, int(self.task_arrival_rate * self.time_slot * 3))
            scheduling_decisions = self.central_scheduler.global_load_balance_scheduling(estimated_tasks)
            
            # 🚀 执行智能迁移协调
            migration_commands = self.central_scheduler.intelligent_migration_coordination(0.7)
            
            # 🔌 计算调度指令分发的有线传输成本
            if len(scheduling_decisions) > 0:
                command_size_mb = 0.05  # 50KB的调度指令
                total_command_delay = 0.0
                total_command_energy = 0.0
                
                for rsu_id in scheduling_decisions.keys():
                    if rsu_id != central_rsu_id:
                        try:
                            from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
                            cmd_delay = calculate_rsu_to_rsu_delay(command_size_mb, central_rsu_id, rsu_id)
                            cmd_energy = calculate_rsu_to_rsu_energy(command_size_mb, central_rsu_id, rsu_id, cmd_delay)
                            total_command_delay += cmd_delay
                            total_command_energy += cmd_energy
                        except Exception:
                            total_command_delay += 0.002  # 2ms回退延迟
                            total_command_energy += 0.05   # 0.05J回退能耗
            
            # 📊 显示调度状态
            if len(rsu_loads) > 0:
                max_load_rsu = max(rsu_loads.items(), key=lambda x: x[1].cpu_usage)
                min_load_rsu = min(rsu_loads.items(), key=lambda x: x[1].cpu_usage)
                
                print(f"🏢 中央调度报告: 管理{len(rsu_loads)}个RSU")
                print(f"   📊 最高负载: {max_load_rsu[0]} (负载:{max_load_rsu[1].cpu_usage:.1%}, 队列:{max_load_rsu[1].queue_length})")
                print(f"   📊 最低负载: {min_load_rsu[0]} (负载:{min_load_rsu[1].cpu_usage:.1%}, 队列:{min_load_rsu[1].queue_length})")
                print(f"   🎯 调度决策: {len(scheduling_decisions)}个, 迁移指令: {len(migration_commands)}个")
                print(f"   🔌 有线网络: 信息收集{total_collection_delay*1000:.1f}ms, 指令分发{total_command_delay*1000:.1f}ms")
                
                # 更新统计
                if not hasattr(self.stats, 'central_scheduling_calls'):
                    self.stats['central_scheduling_calls'] = 0
                self.stats['central_scheduling_calls'] += 1
                
                # 记录有线网络开销
                self.stats['backhaul_collection_delay'] = self.stats.get('backhaul_collection_delay', 0.0) + total_collection_delay
                self.stats['backhaul_command_delay'] = self.stats.get('backhaul_command_delay', 0.0) + total_command_delay
                self.stats['backhaul_total_energy'] = self.stats.get('backhaul_total_energy', 0.0) + total_collection_energy + total_command_energy
                
        except Exception as e:
            print(f"⚠️ 中央调度执行异常: {e}")
    
    def get_central_scheduling_report(self) -> Dict:
        """📋 获取中央调度完整报告"""
        if not hasattr(self, 'central_scheduler') or not self.central_scheduler:
            return {'status': 'not_available', 'message': '中央调度器未启用'}
        
        try:
            # 获取全局状态
            status = self.central_scheduler.get_global_scheduling_status()
            
            # 添加RSU详细信息
            rsu_details = {}
            for rsu in self.rsus:
                rsu_id = rsu['id']
                rsu_details[rsu_id] = {
                    'position': rsu['position'].tolist(),
                    'queue_length': len(rsu.get('computation_queue', [])),
                    'cpu_usage': rsu.get('cpu_usage', 0.0),
                    'cache_usage': len(rsu.get('cache', {})) / rsu.get('cache_capacity', 100),
                    'energy_consumed': rsu.get('energy_consumed', 0.0)
                }
            
            report = {
                'central_scheduler_status': status,
                'rsu_details': rsu_details,
                'scheduling_calls': self.stats.get('central_scheduling_calls', 0),
                'timestamp': getattr(self, 'current_time', 0.0)
            }
            
            return report
            
        except Exception as e:
            return {'status': 'error', 'message': f'报告生成失败: {e}'}
    
    def calculate_transmission_delay(self, data_size: float, distance: float, tx_node_type: str = 'vehicle') -> float:
        """计算传输时延 - 基于SINR的完整3GPP模型"""
        # 获取3GPP参数
        if self.sys_config is not None:
            # 发射功率 (dBm)
            if tx_node_type == 'rsu':
                tx_power_dbm = getattr(self.sys_config.communication, 'rsu_tx_power', 46.0)
            elif tx_node_type == 'uav':
                tx_power_dbm = getattr(self.sys_config.communication, 'uav_tx_power', 30.0)
            else:  # vehicle
                tx_power_dbm = getattr(self.sys_config.communication, 'vehicle_tx_power', 23.0)
            
            # 系统参数
            bandwidth_hz = getattr(self.sys_config.communication, 'total_bandwidth', 20e6)
            noise_figure_db = getattr(self.sys_config.communication, 'noise_figure', 9.0)
            thermal_noise_dbm_hz = getattr(self.sys_config.communication, 'thermal_noise_density', -174.0)
        else:
            # 回退默认值
            tx_power_dbm = 30.0
            bandwidth_hz = 20e6
            noise_figure_db = 9.0
            thermal_noise_dbm_hz = -174.0
        
        # 路径损耗计算 (Free Space + 简化衰减)
        d_m = max(float(distance), 1.0)
        carrier_freq_hz = getattr(self.sys_config.communication, 'carrier_frequency', 2.4e9) if self.sys_config else 2.4e9
        path_loss_db = 32.45 + 20 * np.log10(d_m/1000) + 20 * np.log10(carrier_freq_hz/1e9)
        
        # 接收信号功率 (dBm)
        rx_signal_dbm = tx_power_dbm - path_loss_db
        
        # 热噪声功率 (dBm)
        noise_power_dbm = thermal_noise_dbm_hz + 10 * np.log10(bandwidth_hz) + noise_figure_db
        
        # 干扰功率计算 (简化：假设附近有其他发射源)
        interference_power_dbm = self._calculate_interference_power(distance, tx_node_type)
        
        # 总噪声+干扰功率 (线性域相加，转回dB)
        noise_linear = 10**(noise_power_dbm/10)
        interference_linear = 10**(interference_power_dbm/10)
        total_noise_interference_dbm = 10 * np.log10(noise_linear + interference_linear)
        
        # SINR计算 (dB)
        sinr_db = rx_signal_dbm - total_noise_interference_dbm
        
        # Shannon容量计算
        if sinr_db > -10:  # SINR > -10dB才能通信
            sinr_linear = 10**(sinr_db/10)
            capacity_bps = bandwidth_hz * np.log2(1 + sinr_linear)
            delay = (data_size * 8) / capacity_bps if capacity_bps > 0 else float('inf')  # 转为bits
        else:
            delay = float('inf')  # SINR太低，无法传输
        
        return max(delay, 0.001)  # 最小1ms
    
    def _calculate_interference_power(self, distance: float, tx_node_type: str) -> float:
        """计算干扰功率 - 简化3GPP干扰模型"""
        # 干扰源：假设附近有2-3个同类型发射源
        num_interferers = 2 if tx_node_type == 'vehicle' else 1  # 车辆密度高，干扰源多
        
        # 干扰源平均距离（比期望信号源远）- 数值稳定
        base_distance = max(distance, 10.0)  # 最小10米
        avg_interferer_distance = base_distance * np.random.uniform(1.5, 3.0)
        
        # 干扰源发射功率（与期望源相同类型）
        if self.sys_config is not None:
            if tx_node_type == 'rsu':
                interferer_tx_power_dbm = getattr(self.sys_config.communication, 'rsu_tx_power', 46.0)
            elif tx_node_type == 'uav':
                interferer_tx_power_dbm = getattr(self.sys_config.communication, 'uav_tx_power', 30.0)
            else:
                interferer_tx_power_dbm = getattr(self.sys_config.communication, 'vehicle_tx_power', 23.0)
        else:
            interferer_tx_power_dbm = 30.0
        
        # 干扰源路径损耗 - 数值稳定
        carrier_freq_hz = getattr(self.sys_config.communication, 'carrier_frequency', 2.4e9) if self.sys_config else 2.4e9
        interferer_path_loss = 32.45 + 20 * np.log10(max(avg_interferer_distance/1000, 0.001)) + 20 * np.log10(carrier_freq_hz/1e9)
        
        # 单个干扰源接收功率
        single_interferer_rx_dbm = interferer_tx_power_dbm - interferer_path_loss
        
        # 多个干扰源功率叠加 (线性域)
        if single_interferer_rx_dbm > -120:  # 干扰源不能太弱
            single_interferer_linear = 10**(single_interferer_rx_dbm/10)
            total_interference_linear = num_interferers * single_interferer_linear
            total_interference_dbm = 10 * np.log10(total_interference_linear)
        else:
            total_interference_dbm = -120.0  # 最小干扰功率
        
        return total_interference_dbm
    
    def calculate_computation_delay(self, computation_req: float, node: Dict, data_size_bytes: float = None, compute_density_cycles_per_bit: float = None, cpu_freq: float = None) -> float:
        """计算计算时延（统一为 cycles / CPU_freq + 排队等待）"""
        # 计算需求统一：cycles = data_size_bits * density；若未给出，退回computation_req/MIPS
        if self.sys_config is not None and data_size_bytes is not None:
            bits = float(data_size_bytes) * 8.0
            density = compute_density_cycles_per_bit if compute_density_cycles_per_bit is not None else float(getattr(self.sys_config.task, 'task_compute_density', 500))
            total_cycles = bits * density
            # CPU频率
            if cpu_freq is None:
                cpu_freq = float(getattr(self.sys_config.compute, 'rsu_default_freq', 50e9))
            exec_time = total_cycles / max(cpu_freq, 1.0)
        else:
            # 兼容旧路径：computation_req 单位 MIPS，capacity 1000 MIPS
            computation_capacity = self.config['computation_capacity']
            exec_time = computation_req / computation_capacity
        # 排队等待
        queue_length = len(node.get('computation_queue', []))
        queue_delay = queue_length * 0.01
        return queue_delay + float(exec_time)
    
    def calculate_energy_consumption(self, task: Dict, processing_node: Dict, 
                                   transmission_distance: float, node_type: str = 'Vehicle') -> float:
        """计算能耗 - 统一使用system_config功率参数与dBm→W转换"""
        
        def dbm_to_watts(dbm_value):
            """dBm转换为瓦特"""
            return 10**((dbm_value - 30) / 10)
        
        # 传输能耗 - 使用system_config功率
        if self.sys_config is not None:
            if node_type == 'RSU':
                tx_power_dbm = getattr(self.sys_config.communication, 'rsu_tx_power', 46.0)
            elif node_type == 'UAV':
                tx_power_dbm = getattr(self.sys_config.communication, 'uav_tx_power', 30.0)
            else:
                tx_power_dbm = getattr(self.sys_config.communication, 'vehicle_tx_power', 23.0)
            transmission_power_w = dbm_to_watts(tx_power_dbm)
        else:
            transmission_power_w = self.config['transmission_power']  # 回退
        
        # 传输时延（用于能耗计算）
        tx_type = 'vehicle' if node_type == 'Vehicle' else node_type.lower()
        transmission_time = self.calculate_transmission_delay(task['data_size'], transmission_distance, tx_type)
        transmission_energy = transmission_power_w * transmission_time
        
        # 计算能耗 - 使用CPU频率与kappa参数
        if self.sys_config is not None:
            # 根据节点类型选择CPU频率和功率模型
            if processing_node in self.rsus:
                cpu_freq = float(getattr(self.sys_config.compute, 'rsu_default_freq', 50e9))
                kappa = float(getattr(self.sys_config.compute, 'rsu_kappa', 1e-27))
                static_power = float(getattr(self.sys_config.compute, 'rsu_static_power', 2.0))
            elif processing_node in self.uavs:
                cpu_freq = float(getattr(self.sys_config.compute, 'uav_default_freq', 8e9))
                kappa = float(getattr(self.sys_config.compute, 'uav_kappa3', 1e-27))
                static_power = float(getattr(self.sys_config.compute, 'uav_static_power', 1.0))
            else:  # vehicle
                cpu_freq = float(getattr(self.sys_config.compute, 'vehicle_default_freq', 16e9))
                kappa = float(getattr(self.sys_config.compute, 'vehicle_kappa1', 1e-28))
                static_power = float(getattr(self.sys_config.compute, 'vehicle_static_power', 0.5))
            
            # 🔧 修复：计算时间 - 使用任务特定计算密度
            task_compute_density = task.get('compute_density', 400)  # 获取任务特定密度
            computation_time = self.calculate_computation_delay(
                task['computation_requirement'], processing_node,
                data_size_bytes=task.get('data_size_bytes', task['data_size']*1e6),
                compute_density_cycles_per_bit=task_compute_density,
                cpu_freq=cpu_freq
            )
            computation_time = float(np.clip(computation_time, 0.0, 5.0))
            
            # 动态功率模型：P = kappa * f^3 + P_static
            dynamic_power = kappa * (cpu_freq ** 3) + static_power
            computation_energy = dynamic_power * computation_time
        else:
            # 回退旧模型
            computation_power = self.config['computation_power']
            computation_time = self.calculate_computation_delay(task['computation_requirement'], processing_node)
            computation_time = float(np.clip(computation_time, 0.0, 5.0))
            computation_energy = computation_power * computation_time
        
        total_energy = transmission_energy + computation_energy
        
        # 数值修正：仅处理无限值和NaN
        if not np.isfinite(total_energy):
            total_energy = 100.0  # 仅修正无效值，不限制合理的高值
        
        return total_energy
    
    def process_task(self, task: Dict, agents_actions: Dict = None) -> Dict:
        """处理单个任务（单时隙下可直接完成，否则转入在制任务池）"""
        vehicle = next(v for v in self.vehicles if v['id'] == task['vehicle_id'])
        
        # 默认决策：就近卸载
        if agents_actions is None:
            # 寻找最近的处理节点
            nearest_rsu = self.find_nearest_rsu(vehicle['position'])
            nearest_uav = self.find_nearest_uav(vehicle['position'])
            
            # 选择最近的节点
            if nearest_rsu is not None:
                processing_node = nearest_rsu
                node_type = 'RSU'
            elif nearest_uav is not None:
                processing_node = nearest_uav
                node_type = 'UAV'
            else:
                # 本地处理
                processing_node = vehicle
                node_type = 'Vehicle'
        else:
            # 使用智能体的卸载偏好选择节点（本地/RSU/UAV），并在同类中进一步按概率选择具体节点
            aa = (agents_actions or {})
            pref = aa.get('vehicle_offload_pref', {})
            
            # 🔧 修复：预先计算最近节点，避免UnboundLocalError
            nearest_rsu = self.find_nearest_rsu(vehicle['position'])
            nearest_uav = self.find_nearest_uav(vehicle['position'])
            p_local = float(pref.get('local', 0.34))
            p_rsu = float(pref.get('rsu', 0.33))
            p_uav = float(pref.get('uav', 0.33))
            
            # 🔧 修复：确保概率归一化
            prob_sum = p_local + p_rsu + p_uav
            if prob_sum <= 0:
                p_local, p_rsu, p_uav = 0.34, 0.33, 0.33
                prob_sum = 1.0
            p_local /= prob_sum
            p_rsu /= prob_sum
            p_uav /= prob_sum
            
            # 大类选择
            choice = np.random.choice(['Vehicle', 'RSU', 'UAV'], p=[p_local, p_rsu, p_uav])
            if choice == 'RSU' and self.rsus:
                # 若给出rsu_selection_probs则按其分布选择，否则选择最近RSU
                rsu_probs = aa.get('rsu_selection_probs')
                if isinstance(rsu_probs, list) and len(rsu_probs) == len(self.rsus):
                    # 🔧 修复：RSU概率归一化
                    rsu_probs = np.array(rsu_probs)
                    rsu_prob_sum = np.sum(rsu_probs)
                    if rsu_prob_sum > 0:
                        rsu_probs = rsu_probs / rsu_prob_sum
                    else:
                        rsu_probs = np.ones(len(self.rsus)) / len(self.rsus)
                    idx = np.random.choice(range(len(self.rsus)), p=rsu_probs)
                    processing_node = self.rsus[idx]
                else:
                    processing_node = nearest_rsu or vehicle
                node_type = 'RSU' if processing_node in self.rsus else 'Vehicle'
            elif choice == 'UAV' and self.uavs:
                uav_probs = aa.get('uav_selection_probs')
                if isinstance(uav_probs, list) and len(uav_probs) == len(self.uavs):
                    # 🔧 修复：UAV概率归一化
                    uav_probs = np.array(uav_probs)
                    uav_prob_sum = np.sum(uav_probs)
                    if uav_prob_sum > 0:
                        uav_probs = uav_probs / uav_prob_sum
                    else:
                        uav_probs = np.ones(len(self.uavs)) / len(self.uavs)
                    idx = np.random.choice(range(len(self.uavs)), p=uav_probs)
                    processing_node = self.uavs[idx]
                else:
                    processing_node = nearest_uav or vehicle
                node_type = 'UAV' if processing_node in self.uavs else 'Vehicle'
            else:
                processing_node = vehicle
                node_type = 'Vehicle'
        
        # 🤖 检查缓存命中（支持智能体控制）
        cache_hit = self.check_cache_hit_adaptive(task['content_id'], processing_node, agents_actions)
        
        # 计算距离
        if node_type == 'Vehicle':
            distance = 0  # 本地处理
        else:
            distance = self.calculate_distance(vehicle['position'], processing_node['position'])
        
        # 计算时延（传入发射节点类型以正确计算SINR）
        tx_type = 'vehicle' if node_type == 'Vehicle' else node_type.lower()
        if cache_hit:
            total_delay = self.calculate_transmission_delay(task['data_size'], distance, tx_type)
            compute_time_needed = 0.0
        else:
            transmission_delay = self.calculate_transmission_delay(task['data_size'], distance, tx_type)
            # 统一：cycles/CPU_freq 路径；根据节点类型取频率
            cpu_freq = None
            if self.sys_config is not None:
                if processing_node in self.rsus:
                    cpu_freq = float(getattr(self.sys_config.compute, 'rsu_default_freq', 50e9))
                elif processing_node in self.uavs:
                    cpu_freq = float(getattr(self.sys_config.compute, 'uav_default_freq', 8e9))
                else:
                    cpu_freq = float(getattr(self.sys_config.compute, 'vehicle_default_freq', 16e9))
            # 🔧 修复：使用任务特定的计算密度
            task_compute_density = task.get('compute_density', 
                float(getattr(self.sys_config.task, 'task_compute_density', 400)) if self.sys_config is not None else 400)
            
            computation_delay = self.calculate_computation_delay(
                task['computation_requirement'], processing_node,
                data_size_bytes=task['data_size']*1e6 if task.get('data_size', 1.0) < 100 else task['data_size'],
                compute_density_cycles_per_bit=task_compute_density,
                cpu_freq=cpu_freq
            )
            total_delay = transmission_delay + computation_delay
            compute_time_needed = computation_delay
        
        # 🔧 修复：放宽时延阈值，避免过度截断
        if not np.isfinite(total_delay):
            total_delay = 1.0  # 仅修正无效值
        elif total_delay > 15.0:  # 放宽阈值从10s到15s
            total_delay = min(total_delay, 15.0)  # 软截断，而非硬设为1.0s
        
        # 计算能耗（传入节点类型）
        energy_consumption = self.calculate_energy_consumption(task, processing_node, distance, node_type)
        
        # 检查是否满足截止时间
        completion_time = task['arrival_time'] + total_delay
        
        if completion_time <= task['deadline']:
            # 🔧 修复：只要能在deadline内完成就算成功，不强制要求单时隙完成
            if total_delay <= self.time_slot:
                # 单时隙内完成
                self.stats['completed_tasks'] += 1
                self.stats['total_delay'] += total_delay
                self.stats['total_energy'] += energy_consumption
                
                # 更新节点能耗
                processing_node['energy_consumed'] += energy_consumption
                
                # 更新缓存（简化）
                if not cache_hit and 'cache' in processing_node:
                    if len(processing_node['cache']) < processing_node.get('cache_capacity', 100):
                        processing_node['cache'][task['content_id']] = True
                
                result = {
                    'task_id': task['id'],
                    'status': 'completed',
                    'delay': total_delay,
                    'energy': energy_consumption,
                    'processing_node': processing_node['id'],
                    'cache_hit': cache_hit
                }
            else:
                # 跨时隙完成：进入节点队列进行处理
                node_idx = None
                if node_type == 'RSU':
                    node_idx = self.rsus.index(processing_node) if processing_node in self.rsus else None
                elif node_type == 'UAV':
                    node_idx = self.uavs.index(processing_node) if processing_node in self.uavs else None
                
                work_remaining = max(0.0, compute_time_needed - self.time_slot) if not cache_hit else 0.0
                
                # 🔧 修复：RSU/UAV任务进入节点队列，而非全局active_tasks
                if node_type in ['RSU', 'UAV']:
                    if 'computation_queue' not in processing_node:
                        processing_node['computation_queue'] = []
                    
                    queue_task = {
                        'id': task['id'],
                        'vehicle_id': task['vehicle_id'],
                        'arrival_time': task['arrival_time'],
                        'deadline': task['deadline'],
                        'data_size': task['data_size'],
                        'computation_requirement': task['computation_requirement'],
                        'content_id': task['content_id'],
                        'compute_time_needed': compute_time_needed,
                        'work_remaining': work_remaining,
                        'cache_hit': cache_hit,
                        'queued_at': self.current_time,
                        'expected_completion_time': completion_time
                    }
                    processing_node['computation_queue'].append(queue_task)
                    print(f"📋 任务 {task['id']} 进入 {processing_node['id']} 队列，当前队列长度: {len(processing_node['computation_queue'])}")
                else:
                    # Vehicle本地任务仍使用active_tasks
                    self.active_tasks.append({
                        'id': task['id'],
                        'vehicle_id': task['vehicle_id'],
                        'arrival_time': task['arrival_time'],
                        'deadline': task['deadline'],
                        'work_remaining': work_remaining,
                        'node_type': node_type,
                        'node_idx': node_idx,
                        'content_id': task['content_id'],
                        'expected_completion_time': completion_time
                    })
                
                result = {
                    'task_id': task['id'],
                    'status': 'in_progress',
                    'delay': 0.0,  # 跨时隙任务delay在完成时计算
                    'energy': energy_consumption,
                    'processing_node': processing_node['id'] if node_idx is not None else None,
                    'cache_hit': cache_hit
                }
        else:
            # 即使全力处理也无法在deadline内完成，直接丢弃
            self.stats['dropped_tasks'] += 1
            result = {
                'task_id': task['id'],
                'status': 'dropped',
                'delay': float('inf'),
                'energy': 0,
                'processing_node': None,
                'cache_hit': False
            }
        
        return result
    
    def update_mobility(self):
        """更新移动性"""
        # 更新车辆位置
        for vehicle in self.vehicles:
            # 简单的直线移动模型
            dx = vehicle['velocity'] * np.cos(vehicle['direction']) * self.time_slot
            dy = vehicle['velocity'] * np.sin(vehicle['direction']) * self.time_slot
            
            vehicle['position'][0] += dx
            vehicle['position'][1] += dy
            
            # 边界处理
            if vehicle['position'][0] < 0 or vehicle['position'][0] > 1000:
                vehicle['direction'] = np.pi - vehicle['direction']
            if vehicle['position'][1] < 0 or vehicle['position'][1] > 1000:
                vehicle['direction'] = -vehicle['direction']
            
            # 保持在边界内
            vehicle['position'] = np.clip(vehicle['position'], 0, 1000)
        
        # 更新UAV位置（简化的巡航模式）
        for uav in self.uavs:
            # UAV在固定高度巡航
            angle = self.current_time * 0.01  # 慢速旋转
            radius = 300
            center = [500, 500]  # 区域中心
            
            uav['position'][0] = center[0] + radius * np.cos(angle)
            uav['position'][1] = center[1] + radius * np.sin(angle)
            uav['position'][2] = 100  # 固定高度100m
    
    def simulate_time_slot(self, agents_actions: Dict = None) -> List[Dict]:
        """仿真一个时隙"""
        results = []
        
        # 更新移动性
        self.update_mobility()

        # 🏢 中央RSU全局负载收集与调度 (每10步执行一次)
        if hasattr(self, 'central_scheduler') and self.central_scheduler:
            if not hasattr(self, '_central_schedule_counter'):
                self._central_schedule_counter = 0
            self._central_schedule_counter += 1
            
            if self._central_schedule_counter % 10 == 0:  # 每10步收集一次负载信息
                self._execute_central_rsu_scheduling()

        # 🤖 检查智能体控制的自适应迁移
        self.check_adaptive_migration(agents_actions)

        # 先推进在制任务（车辆跟随 + 过载到空闲），并按概率使用智能体偏好
        advanced_tasks = []
        for t in list(self.active_tasks):
            # 找到车辆位置与最近RSU/UAV
            vehicle = next(v for v in self.vehicles if v['id'] == t['vehicle_id'])
            nearest_rsu = self.find_nearest_rsu(vehicle['position'])
            nearest_uav = self.find_nearest_uav(vehicle['position'])
            # 车辆跟随迁移：若绑定RSU且与最近RSU不同，按一定概率切换，避免频繁抖动
            if t['node_type'] == 'RSU' and nearest_rsu is not None:
                current_node = self.rsus[t['node_idx']] if t['node_idx'] is not None else None
                if current_node is None or current_node is not nearest_rsu:
                    # 使用温和门限：仅当距离差显著或队列差显著时切换
                    from config import config
                    should_switch = True
                    if current_node is not None:
                        d_cur = self.calculate_distance(vehicle['position'], current_node['position'])
                        d_new = self.calculate_distance(vehicle['position'], nearest_rsu['position'])
                        q_cur = len(current_node.get('computation_queue', []))
                        q_new = len(nearest_rsu.get('computation_queue', []))
                        should_switch = ((d_cur - d_new) > config.migration.follow_handover_distance) or ((q_cur - q_new) > config.migration.queue_switch_diff)
                    if should_switch:
                        t['node_idx'] = self.rsus.index(nearest_rsu)
            # 过载到空闲：若当前绑定为RSU且队列过长，则切到队列更短的RSU
            if t['node_type'] == 'RSU' and t['node_idx'] is not None:
                q_len = len(self.rsus[t['node_idx']].get('computation_queue', []))
                from config import config
                if q_len > config.migration.rsu_queue_overload_len:
                    # 找最短队列RSU
                    best_idx = min(range(len(self.rsus)), key=lambda i: len(self.rsus[i].get('computation_queue', [])))
                    t['node_idx'] = best_idx
            # 执行一时隙的工作推进（小幅随机性，模拟服务速率波动）
            from config import config
            j = config.migration.service_jitter_ratio
            service = np.random.uniform(1.0 - j, 1.0 + j) * self.time_slot
            t['work_remaining'] = max(0.0, t['work_remaining'] - service)
            # 完成/超时判断
            current_time = getattr(self, 'current_time', 0.0)
            if t['work_remaining'] <= 0.0:
                # 🔧 修复：跨时隙任务完成，正确计算统计
                self.stats['completed_tasks'] += 1
                
                # 计算实际总时延（修复：使用预期完成时间或当前时间差）
                if 'expected_completion_time' in t:
                    actual_delay = t['expected_completion_time'] - t['arrival_time']
                else:
                    actual_delay = current_time - t['arrival_time']
                
                # 修复时延范围，避免异常值
                actual_delay = max(0.001, min(actual_delay, 30.0))
                self.stats['total_delay'] += actual_delay
                
                # 累计能耗（改进估算：基于实际处理时间与节点类型）
                if t.get('node_type') == 'RSU':
                    processing_power = 50.0  # W，RSU功率较高
                elif t.get('node_type') == 'UAV':
                    processing_power = 20.0  # W，UAV功率中等
                else:
                    processing_power = 5.0   # W，车辆功率较低
                
                processing_energy = processing_power * actual_delay
                self.stats['total_energy'] += processing_energy
                
                print(f"✅ 跨时隙任务 {t['id']} 完成: 时延{actual_delay:.3f}s, 节点{t.get('node_type', 'Unknown')}")
            elif current_time >= t['deadline']:
                # 超时丢弃
                self.stats['dropped_tasks'] += 1
                print(f"❌ 任务 {t['id']} 超时丢弃: 超时{current_time - t['deadline']:.3f}s")
            else:
                # 继续处理
                advanced_tasks.append(t)
        self.active_tasks = advanced_tasks
        
        # 为每个车辆生成任务 - 优化任务生成逻辑（读取system_config到达率）
        for vehicle in self.vehicles:
            # 使用更稳定的任务生成策略
            # 基础概率 + 随机扰动，确保大部分时间步都有任务
            base_rate = (getattr(self, 'task_arrival_rate', self.config['task_arrival_rate'])) * self.time_slot
            # 增加最小任务生成概率，避免连续多个时间步无任务
            adjusted_rate = max(base_rate, 0.1)  # 至少10%的概率生成任务
            
            if np.random.random() < adjusted_rate:
                task = self.generate_task(vehicle['id'])
                result = self.process_task(task, agents_actions)
                results.append(result)
        
        # 如果所有车辆都没有生成任务，强制为一个随机车辆生成任务
        # 这确保训练过程中始终有数据流
        if not results and len(self.vehicles) > 0:
            random_vehicle = np.random.choice(self.vehicles)
            task = self.generate_task(random_vehicle['id'])
            result = self.process_task(task, agents_actions)
            results.append(result)
        
        # 集成迁移：在处理完任务后调用迁移一步
        if self.migration_manager is not None:
            # 简化节点状态与位置适配
            class _Pos:
                def __init__(self, x, y, z=0.0):
                    self.x, self.y, self.z = x, y, z
                def distance_to(self, other):
                    oz = getattr(other, 'z', 0.0)
                    return float(np.linalg.norm(np.array([self.x, self.y, self.z]) - np.array([other.x, other.y, oz])))
            class _State:
                def __init__(self, load_factor=0.0, cpu_frequency=1.0, battery_level=1.0):
                    self.load_factor = load_factor
                    self.cpu_frequency = cpu_frequency
                    self.battery_level = battery_level
            node_states, node_positions = {}, {}
            # RSU
            for i, rsu in enumerate(self.rsus):
                q_len = len(rsu.get('computation_queue', []))
                node_states[f"rsu_{i}"] = _State(load_factor=min(0.99, q_len/10.0), cpu_frequency=self.config['computation_capacity'])
                node_positions[f"rsu_{i}"] = _Pos(rsu['position'][0], rsu['position'][1], 0.0)
            # UAV
            for i, uav in enumerate(self.uavs):
                q_len = len(uav.get('computation_queue', []))
                node_states[f"uav_{i}"] = _State(load_factor=min(0.99, q_len/10.0), cpu_frequency=self.config['computation_capacity'], battery_level=1.0)
                node_positions[f"uav_{i}"] = _Pos(uav['position'][0], uav['position'][1], uav['position'][2])
            self._last_migration_step_stats = self.migration_manager.step(node_states, node_positions)
        else:
            self._last_migration_step_stats = {'migrations_planned': 0, 'migrations_executed': 0, 'migrations_successful': 0}
        return results
    
    def run_simulation(self, num_time_slots: int = 1000, agents_actions: Dict = None) -> Dict:
        """运行完整仿真"""
        print(f"🚀 开始仿真 {num_time_slots} 个时隙...")
        
        # 重置统计
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        all_results = []
        
        for time_slot in range(num_time_slots):
            self.current_time = time_slot * self.time_slot
            
            # 仿真当前时隙
            slot_results = self.simulate_time_slot(agents_actions)
            all_results.extend(slot_results)
            
            # 进度显示
            if (time_slot + 1) % 100 == 0:
                progress = (time_slot + 1) / num_time_slots * 100
                print(f"仿真进度: {progress:.1f}%")
        
        # 计算最终统计
        final_stats = self.calculate_final_statistics()
        
        print("✅ 仿真完成")
        return {
            'statistics': final_stats,
            'detailed_results': all_results,
            'system_state': {
                'vehicles': self.vehicles,
                'rsus': self.rsus,
                'uavs': self.uavs
            }
        }
    
    def calculate_final_statistics(self) -> Dict:
        """计算最终统计结果"""
        total_tasks = self.stats['total_tasks']
        completed_tasks = self.stats['completed_tasks']
        
        if total_tasks == 0:
            return {
                'total_tasks': 0,
                'completed_tasks': 0,
                'dropped_tasks': 0,
                'completion_rate': 0.0,
                'drop_rate': 0.0,
                'avg_delay': 0.0,
                'total_energy': 0.0,
                'cache_hit_rate': 0.0
            }
        
        completion_rate = completed_tasks / total_tasks
        drop_rate = self.stats['dropped_tasks'] / total_tasks
        avg_delay = self.stats['total_delay'] / max(completed_tasks, 1)
        
        total_cache_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        cache_hit_rate = self.stats['cache_hits'] / max(total_cache_requests, 1)
        
        return {
            'total_tasks': total_tasks,
            'completed_tasks': completed_tasks,
            'dropped_tasks': self.stats['dropped_tasks'],
            'completion_rate': completion_rate,
            'drop_rate': drop_rate,
            'avg_delay': avg_delay,
            'total_energy': self.stats['total_energy'],
            'cache_hit_rate': cache_hit_rate
        }
    
    def get_system_state(self) -> Dict:
        """获取系统状态"""
        return {
            'vehicles': len(self.vehicles),
            'rsus': len(self.rsus),
            'uavs': len(self.uavs),
            'current_time': getattr(self, 'current_time', 0),
            'statistics': self.stats
        }
    
    def reset(self):
        """重置仿真器"""
        self.initialize_components()
        self.stats = {
            'total_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.current_time = 0
    
    def run_simulation_step(self, step: int, agents_actions: Dict = None) -> Dict:
        """运行单个仿真步骤"""
        # 更新当前时间
        self.current_time = step * self.time_slot
        
        # 运行一个时隙的仿真
        results = self.simulate_time_slot(agents_actions)
        
        # 🔧 修复：计算步骤统计，包含跨时隙任务完成情况
        completed_results = [r for r in results if r['status'] == 'completed']
        dropped_results = [r for r in results if r['status'] == 'dropped']
        in_progress_results = [r for r in results if r['status'] == 'in_progress']
        
        # 获取本步总任务数和完成数（包含跨时隙完成）
        total_tasks_this_step = self.stats['total_tasks']  # 累计总任务数
        completed_tasks_this_step = self.stats['completed_tasks']  # 累计完成数（含跨时隙）
        dropped_tasks_this_step = self.stats['dropped_tasks']  # 累计丢弃数
        
        # 计算本步新生成的任务数量
        new_tasks_generated = len(results)
        
        step_stats = {
            # 🔧 修复：使用累计统计而非单步结果
            'generated_tasks': new_tasks_generated,  # 本步生成的任务数
            'processed_tasks': completed_tasks_this_step,  # 累计完成任务数（含跨时隙）
            'dropped_tasks': dropped_tasks_this_step,  # 累计丢弃任务数
            'total_delay': self.stats.get('total_delay', 0.0),  # 累计总时延
            'total_energy': self.stats.get('total_energy', 0.0),  # 累计总能耗
            'cache_hits': sum(1 for r in results if r.get('cache_hit', False)),  # 本步缓存命中
            'cache_misses': sum(1 for r in results if not r.get('cache_hit', False)),  # 本步缓存未命中
            # 迁移统计
            'migrations_planned': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_planned', 0),
            'migrations_executed': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_executed', 0),
            'migrations_successful': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_successful', 0),
            
            # 保持原有字段以兼容其他代码
            'tasks_generated': new_tasks_generated,
            'tasks_completed': completed_tasks_this_step,  # 累计完成数
            'tasks_dropped': dropped_tasks_this_step,
            'avg_delay': (self.stats['total_delay'] / max(1, completed_tasks_this_step)) if completed_tasks_this_step > 0 else 0.0,
            
            # 调试信息
            'active_tasks_count': len(self.active_tasks),
            'single_slot_completed': len(completed_results),
            'cross_slot_in_progress': len(in_progress_results)
        }
        
        return step_stats

def test_simulator():
    """测试仿真器"""
    print("🧪 测试完整系统仿真器...")
    
    # 创建仿真器
    simulator = CompleteSystemSimulator()
    
    # 运行短期仿真
    results = simulator.run_simulation(num_time_slots=100)
    
    # 显示结果
    stats = results['statistics']
    print("\n📊 仿真结果:")
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  完成任务数: {stats['completed_tasks']}")
    print(f"  完成率: {stats['completion_rate']:.2%}")
    print(f"  平均时延: {stats['avg_delay']:.3f}s")
    print(f"  总能耗: {stats['total_energy']:.1f}J")
    print(f"  缓存命中率: {stats['cache_hit_rate']:.2%}")
    
    print("✅ 仿真器测试完成")

if __name__ == "__main__":
    test_simulator()