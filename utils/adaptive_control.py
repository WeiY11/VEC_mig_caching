#!/usr/bin/env python3
"""
自适应缓存和迁移控制组件
允许智能体学习和控制缓存迁移的关键参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict

class AdaptiveCacheController:
    """
    自适应缓存控制器
    智能体可以控制缓存策略的关键参数
    """
    
    def __init__(self, cache_capacity: float = 100.0):
        self.cache_capacity = cache_capacity
        
        # 🤖 智能体可控制的缓存参数
        self.agent_params = {
            'heat_threshold_high': 0.8,      # 高热度阈值 [0.5-0.95]
            'heat_threshold_medium': 0.4,    # 中热度阈值 [0.2-0.7]
            'prefetch_ratio': 0.1,           # 预取比例 [0.05-0.3]
            'collaboration_weight': 0.5      # 协作权重 [0.0-1.0]
        }
        
        # 参数有效范围
        self.param_bounds = {
            'heat_threshold_high': (0.5, 0.95),
            'heat_threshold_medium': (0.2, 0.7),
            'prefetch_ratio': (0.05, 0.3),
            'collaboration_weight': (0.0, 1.0)
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
        
        for i, param_name in enumerate(param_names):
            action_key = f'cache_param_{i}'
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
        current_time = time.time()
        
        # 更新访问历史
        self.access_history[content_id].append(current_time)
        
        # 保持历史长度
        if len(self.access_history[content_id]) > 50:
            self.access_history[content_id].pop(0)
        
        # 计算热度：基于访问频率和时效性
        recent_accesses = [t for t in self.access_history[content_id] 
                          if current_time - t < 3600]  # 1小时内的访问
        
        frequency_heat = len(recent_accesses) / 10.0  # 频率热度
        recency_heat = max(0, 1.0 - (current_time - self.access_history[content_id][-1]) / 3600) if self.access_history[content_id] else 0
        
        # 综合热度计算
        self.content_heat[content_id] = min(1.0, 0.7 * frequency_heat + 0.3 * recency_heat)
    
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
        # 🤖 高负载场景优化的智能体参数
        self.agent_params = {
            'rsu_overload_threshold': 0.45,    # 🚀 高负载场景优化阈值 [0.3-0.7]
            'uav_battery_threshold': 0.25,     # UAV电池阈值 [0.15-0.4] 
            'migration_cost_weight': 0.4       # 适中成本权重 [0.2-0.7]
        }
        
        # 🎯 高负载场景优化的参数范围
        self.param_bounds = {
            'rsu_overload_threshold': (0.3, 0.8),   # 适合高负载的范围
            'uav_battery_threshold': (0.15, 0.4),   # UAV电池范围
            'migration_cost_weight': (0.2, 0.7)     # 成本权重范围
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
        
        for i, param_name in enumerate(param_names):
            action_key = f'migration_param_{i}'
            if action_key in agent_actions:
                action_value = np.clip(agent_actions[action_key], -1.0, 1.0)
                param_min, param_max = self.param_bounds[param_name]
                
                normalized_value = (action_value + 1.0) / 2.0
                param_value = param_min + normalized_value * (param_max - param_min)
                
                self.agent_params[param_name] = param_value
    
    def update_node_load(self, node_id: str, load_factor: float, battery_level: float = 1.0):
        """更新节点负载历史"""
        current_time = time.time()
        
        self.node_load_history[node_id].append({
            'time': current_time,
            'load': load_factor,
            'battery': battery_level
        })
        
        # 保持历史长度
        if len(self.node_load_history[node_id]) > 50:
            self.node_load_history[node_id].pop(0)
    
    def should_trigger_migration(self, node_id: str, current_state: Dict) -> Tuple[bool, str, float]:
        """
        🤖 基于智能体学习参数的迁移决策
        
        Returns:
            (should_migrate, reason, urgency_score)
        """
        current_time = time.time()
        
        # 检查冷却期 (防止频繁迁移)
        if (node_id in self.last_migration_time and 
            current_time - self.last_migration_time[node_id] < 60.0):  # 60秒冷却期
            return False, "冷却期内", 0.0
        
        # 获取智能体学习的阈值
        rsu_threshold = self.agent_params['rsu_overload_threshold']
        uav_battery_threshold = self.agent_params['uav_battery_threshold']
        cost_weight = self.agent_params['migration_cost_weight']
        
        load_factor = current_state.get('load_factor', 0.0)
        battery_level = current_state.get('battery_level', 1.0)
        
        urgency_score = 0.0
        
        # 🤖 智能体参数驱动的迁移决策
        if node_id.startswith("rsu_"):
            if load_factor > rsu_threshold:
                # 计算迁移紧急性
                load_urgency = (load_factor - rsu_threshold) / (1.0 - rsu_threshold)
                
                # 基于负载趋势调整
                trend_factor = self._calculate_load_trend(node_id)
                urgency_score = load_urgency * (1.0 + trend_factor) * (1.0 - cost_weight)
                
                if urgency_score > 0.3:  # 紧急性阈值
                    self.migration_stats['total_triggers'] += 1
                    self.last_migration_time[node_id] = current_time
                    return True, f"RSU过载 (负载:{load_factor:.2f} > {rsu_threshold:.2f})", urgency_score
        
        elif node_id.startswith("uav_"):
            # UAV电池和负载双重检查
            battery_urgency = 0.0
            load_urgency = 0.0
            
            if battery_level < uav_battery_threshold:
                battery_urgency = (uav_battery_threshold - battery_level) / uav_battery_threshold
            
            if load_factor > 0.8:  # UAV负载阈值相对固定
                load_urgency = (load_factor - 0.8) / 0.2
            
            urgency_score = max(battery_urgency, load_urgency) * (1.0 - cost_weight * 0.5)  # UAV迁移成本权重降低
            
            if urgency_score > 0.4:  # UAV紧急性阈值稍高
                reason = f"UAV电池低:{battery_level:.1%}" if battery_urgency > load_urgency else f"UAV过载:{load_factor:.2f}"
                self.migration_stats['total_triggers'] += 1
                self.last_migration_time[node_id] = current_time
                return True, reason, urgency_score
        
        return False, f"无需迁移 (RSU负载:{load_factor:.2f}, UAV电池:{battery_level:.1%})", urgency_score
    
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
    将智能体动作数组映射为缓存和迁移参数
    
    Args:
        agent_actions: 长度为7的数组，来自18维动作的后7维
                      [cache_0, cache_1, cache_2, cache_3, migration_0, migration_1, migration_2]
    
    Returns:
        (cache_params, migration_params)
    """
    if len(agent_actions) < 7:
        # 如果动作不足，使用默认值
        agent_actions = np.pad(agent_actions, (0, 7 - len(agent_actions)), mode='constant', constant_values=0.0)
    
    # 构造参数字典
    cache_params = {
        'cache_param_0': agent_actions[0],  # heat_threshold_high
        'cache_param_1': agent_actions[1],  # heat_threshold_medium
        'cache_param_2': agent_actions[2],  # prefetch_ratio
        'cache_param_3': agent_actions[3],  # collaboration_weight
    }
    
    migration_params = {
        'migration_param_0': agent_actions[4],  # rsu_overload_threshold
        'migration_param_1': agent_actions[5],  # uav_battery_threshold
        'migration_param_2': agent_actions[6],  # migration_cost_weight
    }
    
    return cache_params, migration_params
