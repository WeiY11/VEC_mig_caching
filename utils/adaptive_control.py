#!/usr/bin/env python3
"""
自适应缓存和迁移控制组件
允许智能体学习和控制缓存迁移的关键参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
# 🔧 修复：导入统一时间管理器
from .unified_time_manager import get_simulation_time

class AdaptiveCacheController:
    """
    自适应缓存控制器
    智能体可以控制缓存策略的关键参数
    """
    
    def __init__(self, cache_capacity: float = 100.0):
        self.cache_capacity = cache_capacity
        
        # 🔧 优化：调整智能体可控制的缓存参数为更合理的初始值
        self.agent_params = {
            'heat_threshold_high': 0.7,      # 高热度阈值：70% [0.5-0.9]
            'heat_threshold_medium': 0.35,   # 中热度阈值：35% [0.2-0.6]
            'prefetch_ratio': 0.05,          # 预取比例：5% [0.02-0.15]
            'collaboration_weight': 0.3      # 协作权重：30% [0.0-0.8]
        }
        
        # 🔧 优化：调整参数有效范围，更适合实际缓存场景
        self.param_bounds = {
            'heat_threshold_high': (0.5, 0.9),      # 高热度阈值范围缩小
            'heat_threshold_medium': (0.2, 0.6),    # 中热度阈值范围调整
            'prefetch_ratio': (0.02, 0.15),         # 预取比例范围缩小，避免过度预取
            'collaboration_weight': (0.0, 0.8)      # 协作权重上限降低
        }
        
        # 缓存统计
        self.cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'current_utilization': 0.0,
            'hit_rate_history': []
        }
        
        # 热度计算
        self.content_heat = defaultdict(float)
        self.access_history = defaultdict(list)
        
        print(f"🤖 自适应缓存控制器初始化完成")
    
    def update_agent_params(self, agent_actions: Dict[str, float]):
        """
        根据智能体动作更新缓存参数
        
        Args:
            agent_actions: 格式 {'cache_param_0': 0.7, 'cache_param_1': -0.4, ...}
        """
        if not isinstance(agent_actions, dict):
            return
            
        param_names = list(self.param_bounds.keys())
        
        # 🔧 修复：直接使用语义化参数名映射
        param_mapping = {
            'heat_threshold_high': 'heat_threshold_high',
            'heat_threshold_medium': 'heat_threshold_medium', 
            'prefetch_ratio': 'prefetch_ratio',
            'collaboration_weight': 'collaboration_weight'
        }
        
        for param_name, action_key in param_mapping.items():
            if action_key in agent_actions:
                # 将智能体动作 [-1,1] 映射到参数范围
                action_value = np.clip(agent_actions[action_key], -1.0, 1.0)
                param_min, param_max = self.param_bounds[param_name]
                
                # 线性映射: [-1,1] → [param_min, param_max]
                normalized_value = (action_value + 1.0) / 2.0
                param_value = param_min + normalized_value * (param_max - param_min)
                
                self.agent_params[param_name] = param_value
        
        # 确保参数逻辑一致性：中阈值 < 高阈值
        if self.agent_params['heat_threshold_medium'] >= self.agent_params['heat_threshold_high']:
            self.agent_params['heat_threshold_medium'] = self.agent_params['heat_threshold_high'] - 0.1
    
    def update_content_heat(self, content_id: str, access_weight: float = 1.0):
        """更新内容热度"""
        # 🔧 修复：使用统一仿真时间
        current_time = get_simulation_time()
        
        # 更新访问历史
        self.access_history[content_id].append(current_time)
        
        # 保持历史长度
        if len(self.access_history[content_id]) > 50:
            self.access_history[content_id].pop(0)
        
        # 🔧 优化：改进热度计算，更适合仿真环境
        # 计算最近访问窗口（从1小时改为10分钟，适应仿真）
        recent_accesses = [t for t in self.access_history[content_id] 
                          if current_time - t < 600]  # 10分钟内的访问，适应仿真时间
        
        # 频率热度：使用平方根避免极端值dominance
        frequency_heat = min(1.0, np.sqrt(len(recent_accesses) / 8.0))  # 8次访问达到满热度
        
        # 最近性热度：指数衰减更平滑
        if self.access_history[content_id]:
            last_access = self.access_history[content_id][-1]
            time_since_last = current_time - last_access
            recency_heat = np.exp(-time_since_last / 120.0)  # 2分钟半衰期
        else:
            recency_heat = 0.0
        
        # 🔧 优化：综合热度计算，平衡频率和最近性
        self.content_heat[content_id] = min(1.0, 0.6 * frequency_heat + 0.4 * recency_heat)
    
    def should_cache_content(self, content_id: str, data_size: float, available_capacity: float) -> Tuple[bool, str]:
        """
        🤖 基于智能体学习参数的缓存决策
        
        Returns:
            (should_cache, reason)
        """
        # 获取内容热度
        heat = self.content_heat.get(content_id, 0.0)
        
        # 使用智能体学习的阈值
        high_threshold = self.agent_params['heat_threshold_high']
        medium_threshold = self.agent_params['heat_threshold_medium']
        prefetch_ratio = self.agent_params['prefetch_ratio']
        
        # 计算容量阈值
        capacity_threshold = self.cache_capacity * prefetch_ratio
        
        # 🤖 智能体参数驱动的决策逻辑
        if heat > high_threshold and available_capacity > data_size:
            return True, f"高热度缓存 (热度:{heat:.2f} > {high_threshold:.2f})"
        
        elif heat > medium_threshold and available_capacity > capacity_threshold:
            return True, f"中热度预取 (热度:{heat:.2f} > {medium_threshold:.2f})"
        
        elif available_capacity > data_size and heat > 0.1:  # 基础缓存条件
            collaboration_weight = self.agent_params['collaboration_weight']
            # 基于协作权重的概率性缓存
            cache_probability = heat * collaboration_weight
            if np.random.random() < cache_probability:
                return True, f"协作缓存 (概率:{cache_probability:.2f})"
        
        return False, f"不缓存 (热度:{heat:.2f}, 可用容量:{available_capacity:.1f}MB)"
    
    def record_cache_result(self, content_id: str, was_hit: bool):
        """记录缓存结果"""
        self.cache_stats['total_requests'] += 1
        
        if was_hit:
            self.cache_stats['cache_hits'] += 1
        else:
            self.cache_stats['cache_misses'] += 1
        
        # 更新命中率历史
        if self.cache_stats['total_requests'] > 0:
            hit_rate = self.cache_stats['cache_hits'] / self.cache_stats['total_requests']
            self.cache_stats['hit_rate_history'].append(hit_rate)
            
            # 保持历史长度
            if len(self.cache_stats['hit_rate_history']) > 100:
                self.cache_stats['hit_rate_history'].pop(0)
    
    def get_cache_metrics(self) -> Dict:
        """获取缓存效果指标"""
        total_requests = self.cache_stats['total_requests']
        if total_requests == 0:
            return {
                'hit_rate': 0.0,
                'effectiveness': 0.0,
                'utilization': 0.0,
                'total_requests': 0,
                'agent_params': dict(self.agent_params)
            }
        
        hit_rate = self.cache_stats['cache_hits'] / total_requests
        utilization = self.cache_stats['current_utilization']
        
        # 效果指标：命中率 × 利用率
        effectiveness = hit_rate * min(1.0, utilization)
        
        return {
            'hit_rate': hit_rate,
            'effectiveness': effectiveness,
            'utilization': utilization,
            'total_requests': total_requests,
            'agent_params': dict(self.agent_params)
        }


class AdaptiveMigrationController:
    """
    自适应迁移控制器
    智能体可以控制迁移策略的关键参数
    """
    
    def __init__(self):
        # 🤖 平衡迁移机制：适中的阈值设置
        self.agent_params = {
            'rsu_overload_threshold': 0.2,     # 恢复到适中水平，避免过度迁移
            'uav_battery_threshold': 0.25,     # 恢复到适中水平
            'migration_cost_weight': 0.3,      # 恢复迁移成本权重
            'urgency_threshold_rsu': 0.1,      # 恢复RSU紧急阈值
            'urgency_threshold_uav': 0.15      # 恢复UAV紧急阈值
        }
        
        # 🎯 扩大参数范围，允许更灵活的迁移策略
        self.param_bounds = {
            'rsu_overload_threshold': (0.05, 0.4),  # 🔧 从(0.3,0.8)扩展到(0.05,0.4)
            'uav_battery_threshold': (0.10, 0.3),   # 🔧 从(0.15,0.4)调整到(0.10,0.3)
            'migration_cost_weight': (0.1, 0.6),    # 🔧 从(0.2,0.7)调整到(0.1,0.6)
            'urgency_threshold_rsu': (0.05, 0.25),  # 🔧 新增：RSU紧急阈值范围
            'urgency_threshold_uav': (0.10, 0.30)   # 🔧 新增：UAV紧急阈值范围
        }
        
        # 迁移统计
        self.migration_stats = {
            'total_triggers': 0,
            'successful_migrations': 0,
            'total_cost': 0.0,
            'avg_delay_saved': 0.0,
            'success_rate_history': []
        }
        
        # 节点状态历史
        self.node_load_history = defaultdict(list)
        self.last_migration_time = defaultdict(float)
        
        print(f"🤖 自适应迁移控制器初始化完成")
    
    def update_agent_params(self, agent_actions: Dict[str, float]):
        """
        根据智能体动作更新迁移参数
        
        Args:
            agent_actions: 格式 {'migration_param_0': 0.3, 'migration_param_1': -0.6, ...}
        """
        if not isinstance(agent_actions, dict):
            return
            
        param_names = list(self.param_bounds.keys())
        
        # 🔧 修复：语义化迁移参数映射
        param_mapping = {
            'rsu_overload_threshold': 'rsu_overload_threshold',
            'uav_battery_threshold': 'uav_battery_threshold',
            'migration_cost_weight': 'migration_cost_weight'
        }
        
        for param_name, action_key in param_mapping.items():
            if action_key in agent_actions:
                action_value = np.clip(agent_actions[action_key], -1.0, 1.0)
                param_min, param_max = self.param_bounds[param_name]
                
                normalized_value = (action_value + 1.0) / 2.0
                param_value = param_min + normalized_value * (param_max - param_min)
                
                self.agent_params[param_name] = param_value
    
    def update_node_load(self, node_id: str, load_factor: float, battery_level: float = 1.0):
        """更新节点负载历史"""
        # 🔧 修复：使用统一仿真时间
        current_time = get_simulation_time()
        
        self.node_load_history[node_id].append({
            'time': current_time,
            'load': load_factor,
            'battery': battery_level
        })
        
        # 保持历史长度
        if len(self.node_load_history[node_id]) > 50:
            self.node_load_history[node_id].pop(0)
    
    def should_trigger_migration(self, node_id: str, current_state: Dict, neighbor_states: Dict = None) -> Tuple[bool, str, float]:
        """
        🎯 智能多维度迁移触发机制
        
        触发条件：
        1. 资源阈值触发：CPU/带宽/存储任一资源>85%
        2. 负载差触发：与邻近节点负载差>20%
        3. 跟随迁移：车辆移动超出通信覆盖
        
        Returns:
            (should_migrate, reason, urgency_score)
        """
        # 🔧 修复：使用统一仿真时间
        current_time = get_simulation_time()
        
        # 🔧 用户要求：缩短冷却期到1秒，实现每秒触发迁移决策
        if (node_id in self.last_migration_time and 
            current_time - self.last_migration_time[node_id] < 1.0):  # 1秒冷却期，每秒可触发
            return False, "冷却期内", 0.0
        
        # 获取节点状态
        cpu_load = current_state.get('cpu_load', current_state.get('load_factor', 0.0))
        bandwidth_load = current_state.get('bandwidth_load', 0.0)
        storage_load = current_state.get('storage_load', 0.0)
        battery_level = current_state.get('battery_level', 1.0)
        
        urgency_score = 0.0
        migration_reason = ""
        
        # 🎯 多维度触发条件检查
        if node_id.startswith("rsu_"):
            # 1️⃣ 资源阈值触发 (降低到60%阈值，更容易触发)
            resource_overload = False
            overload_resources = []
            
            # 🔧 用户要求：降低过载阈值到85%，更早触发迁移
            if cpu_load > 0.85:  # 85%CPU阈值
                resource_overload = True
                overload_resources.append(f"CPU:{cpu_load:.1%}")
                
            if bandwidth_load > 0.85:  # 85%带宽阈值
                resource_overload = True
                overload_resources.append(f"带宽:{bandwidth_load:.1%}")
                
            if storage_load > 0.85:  # 85%存储阈值
                resource_overload = True
                overload_resources.append(f"存储:{storage_load:.1%}")
            
            # 2️⃣ 负载差触发 (与邻近节点差>20%)
            load_diff_trigger = False
            max_load_diff = 0.0
            
            if neighbor_states:
                current_avg_load = (cpu_load + bandwidth_load + storage_load) / 3
                for neighbor_id, neighbor_state in neighbor_states.items():
                    if neighbor_id != node_id:
                        neighbor_cpu = neighbor_state.get('cpu_load', neighbor_state.get('load_factor', 0.0))
                        neighbor_bw = neighbor_state.get('bandwidth_load', 0.0)
                        neighbor_storage = neighbor_state.get('storage_load', 0.0)
                        neighbor_avg_load = (neighbor_cpu + neighbor_bw + neighbor_storage) / 3
                        
                        load_diff = current_avg_load - neighbor_avg_load
                        max_load_diff = max(max_load_diff, load_diff)
                        
                        # 🔧 用户要求：降低负载差阈值到20%
                        if load_diff > 0.2:  # 负载差>20%
                            load_diff_trigger = True
            
            # 🔥 计算迁移紧急度
            if resource_overload:
                resource_urgency = max(cpu_load, bandwidth_load, storage_load) - 0.85
                urgency_score += resource_urgency * 2.0  # 资源过载权重高
                migration_reason = f"资源过载({','.join(overload_resources)})"
            
            if load_diff_trigger:
                diff_urgency = max_load_diff - 0.2
                urgency_score += diff_urgency * 1.5  # 负载差权重中等
                if migration_reason:
                    migration_reason += f" + 负载差({max_load_diff:.1%})"
                else:
                    migration_reason = f"负载差过大({max_load_diff:.1%})"
            
            # 🔧 优化：更积极的迁移策略，敢于尝试有风险的迁移
            if urgency_score > 0.05:  # 降低触发阈值，更积极地尝试迁移
                self.migration_stats['total_triggers'] += 1
                self.last_migration_time[node_id] = current_time
                return True, migration_reason, urgency_score
        
        elif node_id.startswith("uav_"):
            # 🚁 UAV多维度触发条件
            uav_battery_threshold = self.agent_params['uav_battery_threshold']
            
            # 1️⃣ 电池电量触发
            battery_urgency = 0.0
            if battery_level < uav_battery_threshold:
                battery_urgency = (uav_battery_threshold - battery_level) / uav_battery_threshold
                urgency_score += battery_urgency * 3.0  # 电池紧急权重最高
                migration_reason = f"UAV电池低({battery_level:.1%})"
            
            # 2️⃣ 负载过载触发
            if cpu_load > 0.8:  # UAV CPU负载阈值80%
                load_urgency = (cpu_load - 0.8) / 0.2
                urgency_score += load_urgency * 2.0
                if migration_reason:
                    migration_reason += f" + CPU过载({cpu_load:.1%})"
                else:
                    migration_reason = f"UAV CPU过载({cpu_load:.1%})"
            
            # 3️⃣ 与邻近RSU负载差
            if neighbor_states:
                max_load_diff = 0.0
                for neighbor_id, neighbor_state in neighbor_states.items():
                    if neighbor_id.startswith("rsu_"):  # 只与RSU比较
                        neighbor_load = neighbor_state.get('cpu_load', neighbor_state.get('load_factor', 0.0))
                        load_diff = cpu_load - neighbor_load
                        max_load_diff = max(max_load_diff, load_diff)
                
                # 🔧 用户要求：保持20%负载差阈值
                if max_load_diff > 0.2:  # UAV比RSU高20%以上  
                    diff_urgency = max_load_diff - 0.2
                    urgency_score += diff_urgency * 1.5
                    if migration_reason:
                        migration_reason += f" + 负载差({max_load_diff:.1%})"
                    else:
                        migration_reason = f"与RSU负载差过大({max_load_diff:.1%})"
            
                # 🔧 优化：UAV也采用更积极的迁移策略
                if urgency_score > 0.08:  # 降低UAV触发阈值，更积极地平衡负载
                    self.migration_stats['total_triggers'] += 1
                    self.last_migration_time[node_id] = current_time
                    return True, migration_reason, urgency_score
        
        return False, f"无需迁移 (CPU:{cpu_load:.1%}, 电池:{battery_level:.1%})", urgency_score
    
    def _calculate_load_trend(self, node_id: str) -> float:
        """计算负载趋势"""
        history = self.node_load_history.get(node_id, [])
        if len(history) < 3:
            return 0.0
        
        # 计算最近的负载变化趋势
        recent_loads = [entry['load'] for entry in history[-5:]]
        if len(recent_loads) < 2:
            return 0.0
        
        # 简单线性趋势
        trend = (recent_loads[-1] - recent_loads[0]) / len(recent_loads)
        return np.clip(trend * 5, -1.0, 1.0)  # 归一化趋势
    
    def record_migration_result(self, success: bool, cost: float = 0.0, delay_saved: float = 0.0):
        """记录迁移结果"""
        if success:
            self.migration_stats['successful_migrations'] += 1
            self.migration_stats['total_cost'] += cost
            self.migration_stats['avg_delay_saved'] = (
                (self.migration_stats['avg_delay_saved'] * (self.migration_stats['successful_migrations'] - 1) + delay_saved) /
                self.migration_stats['successful_migrations']
            )
        
        # 更新成功率历史
        if self.migration_stats['total_triggers'] > 0:
            success_rate = self.migration_stats['successful_migrations'] / self.migration_stats['total_triggers']
            self.migration_stats['success_rate_history'].append(success_rate)
            
            if len(self.migration_stats['success_rate_history']) > 100:
                self.migration_stats['success_rate_history'].pop(0)
    
    def get_migration_metrics(self) -> Dict:
        """获取迁移效果指标"""
        total_triggers = self.migration_stats['total_triggers']
        if total_triggers == 0:
            return {
                'success_rate': 0.0,
                'effectiveness': 0.0,
                'avg_cost': 0.0,
                'total_triggers': 0,
                'agent_params': dict(self.agent_params)
            }
        
        success_rate = self.migration_stats['successful_migrations'] / total_triggers
        avg_cost = self.migration_stats['total_cost'] / max(1, self.migration_stats['successful_migrations'])
        
        # 效果指标：成功率 × (1 - 归一化成本)
        cost_factor = min(1.0, avg_cost / 100.0)  # 假设100为最大成本
        effectiveness = success_rate * (1.0 - cost_factor * self.agent_params['migration_cost_weight'])
        
        return {
            'success_rate': success_rate,
            'effectiveness': effectiveness,
            'avg_cost': avg_cost,
            'total_triggers': total_triggers,
            'agent_params': dict(self.agent_params)
        }


def map_agent_actions_to_params(agent_actions: np.ndarray) -> Tuple[Dict, Dict]:
    """
    🔧 修复：将智能体动作映射为语义化的缓存和迁移参数
    
    Args:
        agent_actions: 长度为7的数组，来自18维动作的后7维
                      [heat_high, heat_med, prefetch, collab, rsu_thresh, uav_thresh, mig_cost]
    
    Returns:
        (cache_params, migration_params) - 使用语义化命名
    """
    if len(agent_actions) < 7:
        agent_actions = np.pad(agent_actions, (0, 7 - len(agent_actions)), mode='constant', constant_values=0.0)
    
    # 🔧 修复：语义化参数映射，便于理解和调试
    cache_params = {
        'heat_threshold_high': np.clip(agent_actions[0], -1.0, 1.0),      # 高热度阈值
        'heat_threshold_medium': np.clip(agent_actions[1], -1.0, 1.0),    # 中热度阈值  
        'prefetch_ratio': np.clip(agent_actions[2], -1.0, 1.0),           # 预取比例
        'collaboration_weight': np.clip(agent_actions[3], -1.0, 1.0),     # 协作权重
    }
    
    migration_params = {
        'rsu_overload_threshold': np.clip(agent_actions[4], -1.0, 1.0),   # RSU过载阈值
        'uav_battery_threshold': np.clip(agent_actions[5], -1.0, 1.0),    # UAV电池阈值
        'migration_cost_weight': np.clip(agent_actions[6], -1.0, 1.0),    # 迁移成本权重
    }
    
    return cache_params, migration_params
