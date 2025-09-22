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
        
        # 网络拓扑
        self.num_vehicles = self.config.get('num_vehicles', 12)
        self.num_rsus = self.config.get('num_rsus', 6)
        self.num_uavs = self.config.get('num_uavs', 2)
        
        # 仿真参数
        self.simulation_time = self.config.get('simulation_time', 1000)
        self.time_slot = self.config.get('time_slot', 0.1)
        
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
        # 车辆节点
        self.vehicles = []
        for i in range(self.num_vehicles):
            vehicle = {
                'id': f'V_{i}',
                'position': np.random.uniform(0, 1000, 2),  # x, y坐标
                'velocity': np.random.uniform(10, 30),  # m/s
                'direction': np.random.uniform(0, 2*np.pi),  # 弧度
                'tasks': [],
                'energy_consumed': 0.0
            }
            self.vehicles.append(vehicle)
        
        # RSU节点
        self.rsus = []
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': np.random.uniform(0, 1000, 2),
                'coverage_radius': 200,  # 覆盖半径
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
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
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.uavs.append(uav)
        
        print(f"✓ 创建了 {self.num_vehicles} 车辆, {self.num_rsus} RSU, {self.num_uavs} UAV")
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
        """生成计算任务"""
        self.task_counter += 1
        task = {
            'id': f'task_{self.task_counter}',
            'vehicle_id': vehicle_id,
            'arrival_time': self.current_time,
            'data_size': np.random.exponential(1.0),  # MB
            'computation_requirement': np.random.exponential(120),  # MIPS（略增以提高跨时隙概率）
            'deadline': self.current_time + np.random.uniform(0.5, 3.0),  # 0.5~3s窗口，允许跨时隙
            'content_id': f'content_{np.random.randint(0, 100)}',
            'priority': np.random.uniform(0.1, 1.0)
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
    
    def calculate_transmission_delay(self, data_size: float, distance: float) -> float:
        """计算传输时延"""
        # 简化的传输时延模型
        bandwidth_mhz = self.config['bandwidth']
        # 考虑距离对信号衰减的影响
        d_m = max(float(distance), 1.0)  # 数值稳定：最小1米，避免log10(0)
        path_loss = 32.45 + 20 * np.log10(d_m/1000) + 20 * np.log10(2.4)  # 2.4GHz
        snr = 30 - path_loss  # 假设发射功率30dBm
        
        # Shannon公式计算容量
        if snr > 0:
            capacity_mbps = bandwidth_mhz * np.log2(1 + 10**(snr/10))
            delay = data_size / capacity_mbps  # 秒
        else:
            delay = float('inf')  # 信号太弱，无法传输
        
        return max(delay, 0.001)  # 最小1ms
    
    def calculate_computation_delay(self, computation_req: float, node: Dict) -> float:
        """计算计算时延"""
        # 简化的计算时延模型
        computation_capacity = self.config['computation_capacity']  # MIPS
        
        # 考虑队列等待时间
        queue_length = len(node.get('computation_queue', []))
        queue_delay = queue_length * 0.01  # 每个任务平均10ms
        
        # 计算执行时间
        execution_delay = computation_req / computation_capacity
        
        return queue_delay + execution_delay
    
    def calculate_energy_consumption(self, task: Dict, processing_node: Dict, 
                                   transmission_distance: float) -> float:
        """计算能耗"""
        # 传输能耗
        transmission_power = self.config['transmission_power']  # W
        transmission_time = self.calculate_transmission_delay(
            task['data_size'], transmission_distance
        )
        transmission_energy = transmission_power * transmission_time
        
        # 计算能耗
        computation_power = self.config['computation_power']  # W
        computation_time = self.calculate_computation_delay(
            task['computation_requirement'], processing_node
        )
        # 数值稳定与上限约束，避免异常能耗冲击学习
        computation_time = float(np.clip(computation_time, 0.0, 5.0))
        computation_energy = computation_power * computation_time
        
        total_energy = transmission_energy + computation_energy
        
        # 数值修正：避免异常值
        if not np.isfinite(total_energy) or total_energy > 10000:
            total_energy = 2000.0  # 修正为合理值
        
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
            p_local = float(pref.get('local', 0.34))
            p_rsu = float(pref.get('rsu', 0.33))
            p_uav = float(pref.get('uav', 0.33))
            # 大类选择
            choice = np.random.choice(['Vehicle', 'RSU', 'UAV'], p=[p_local, p_rsu, p_uav])
            if choice == 'RSU' and self.rsus:
                # 若给出rsu_selection_probs则按其分布选择，否则选择最近RSU
                rsu_probs = aa.get('rsu_selection_probs')
                if isinstance(rsu_probs, list) and len(rsu_probs) == len(self.rsus):
                    idx = np.random.choice(range(len(self.rsus)), p=np.array(rsu_probs))
                    processing_node = self.rsus[idx]
                else:
                    processing_node = nearest_rsu or vehicle
                node_type = 'RSU' if processing_node in self.rsus else 'Vehicle'
            elif choice == 'UAV' and self.uavs:
                uav_probs = aa.get('uav_selection_probs')
                if isinstance(uav_probs, list) and len(uav_probs) == len(self.uavs):
                    idx = np.random.choice(range(len(self.uavs)), p=np.array(uav_probs))
                    processing_node = self.uavs[idx]
                else:
                    processing_node = nearest_uav or vehicle
                node_type = 'UAV' if processing_node in self.uavs else 'Vehicle'
            else:
                processing_node = vehicle
                node_type = 'Vehicle'
        
        # 检查缓存命中
        cache_hit = self.check_cache_hit(task['content_id'], processing_node)
        
        # 计算距离
        if node_type == 'Vehicle':
            distance = 0  # 本地处理
        else:
            distance = self.calculate_distance(vehicle['position'], processing_node['position'])
        
        # 计算时延
        if cache_hit:
            total_delay = self.calculate_transmission_delay(task['data_size'], distance)
            compute_time_needed = 0.0
        else:
            transmission_delay = self.calculate_transmission_delay(task['data_size'], distance)
            computation_delay = self.calculate_computation_delay(task['computation_requirement'], processing_node)
            total_delay = transmission_delay + computation_delay
            compute_time_needed = computation_delay
        
        # 数值修正
        if not np.isfinite(total_delay) or total_delay > 10:
            total_delay = 1.0  # 修正为1秒
        
        # 计算能耗
        energy_consumption = self.calculate_energy_consumption(task, processing_node, distance)
        
        # 检查是否满足截止时间
        completion_time = task['arrival_time'] + total_delay
        if completion_time <= task['deadline'] and total_delay <= self.time_slot:
            # 任务成功完成
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
            # 未在本时隙完成：进入在制任务池，记录剩余工作量与当前绑定节点
            node_type = node_type
            node_idx = None
            if node_type == 'RSU':
                node_idx = self.rsus.index(processing_node) if processing_node in self.rsus else None
            elif node_type == 'UAV':
                node_idx = self.uavs.index(processing_node) if processing_node in self.uavs else None
            work_remaining = max(0.0, compute_time_needed - self.time_slot) if not cache_hit else 0.0
            self.active_tasks.append({
                'id': task['id'],
                'vehicle_id': task['vehicle_id'],
                'arrival_time': task['arrival_time'],
                'deadline': task['deadline'],
                'work_remaining': work_remaining,
                'node_type': node_type,
                'node_idx': node_idx,
                'content_id': task['content_id'],
            })
            result = {
                'task_id': task['id'],
                'status': 'in_progress',
                'delay': 0.0,
                'energy': energy_consumption,
                'processing_node': processing_node['id'] if node_idx is not None else None,
                'cache_hit': cache_hit
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
            self.current_time = getattr(self, 'current_time', 0.0)
            if t['work_remaining'] <= 0.0:
                self.stats['completed_tasks'] += 1
                # 估计一次能耗（简化：按时间槽功耗）
                self.stats['total_energy'] += 0.1
            elif self.current_time >= t['deadline']:
                self.stats['dropped_tasks'] += 1
            else:
                advanced_tasks.append(t)
        self.active_tasks = advanced_tasks
        
        # 为每个车辆生成任务 - 优化任务生成逻辑
        for vehicle in self.vehicles:
            # 使用更稳定的任务生成策略
            # 基础概率 + 随机扰动，确保大部分时间步都有任务
            base_rate = self.config['task_arrival_rate'] * self.time_slot
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
        
        # 计算步骤统计 - 修正字段名以匹配train_multi_agent.py的期望
        completed_results = [r for r in results if r['status'] == 'completed']
        dropped_results = [r for r in results if r['status'] == 'dropped']
        
        step_stats = {
            # 修正字段名映射
            'generated_tasks': len(results),  # 生成的任务数
            'processed_tasks': len(completed_results),  # 成功处理的任务数
            'dropped_tasks': len(dropped_results),  # 丢弃的任务数
            'total_delay': sum(r['delay'] for r in completed_results) if completed_results else 0.0,  # 总时延
            'total_energy': sum(r['energy'] for r in results),  # 总能耗
            'cache_hits': sum(1 for r in results if r.get('cache_hit', False)),  # 缓存命中数
            'cache_misses': sum(1 for r in results if not r.get('cache_hit', False)),  # 缓存未命中数
            # 迁移统计
            'migrations_planned': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_planned', 0),
            'migrations_executed': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_executed', 0),
            'migrations_successful': (getattr(self, '_last_migration_step_stats', {}) or {}).get('migrations_successful', 0),
            
            # 保持原有字段以兼容其他代码
            'tasks_generated': len(results),
            'tasks_completed': len(completed_results),
            'tasks_dropped': len(dropped_results),
            'avg_delay': np.mean([r['delay'] for r in completed_results]) if completed_results else 0.0,
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