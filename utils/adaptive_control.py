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
            'hit_rate_history': [],
            'evicted_items': 0,
            'collaborative_writes': 0
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

    def should_cache_content(
        self,
        content_id: str,
        data_size: float,
        available_capacity: float,
        cache_snapshot: Dict,
        total_capacity_mb: float
    ) -> Tuple[bool, str, List[str]]:
        """Decide whether to cache a content item. Returns eviction candidates when needed."""
        heat = self.content_heat.get(content_id, 0.0)

        high_threshold = self.agent_params['heat_threshold_high']
        medium_threshold = self.agent_params['heat_threshold_medium']
        prefetch_ratio = self.agent_params['prefetch_ratio']

        capacity_reference = total_capacity_mb if total_capacity_mb > 0 else self.cache_capacity
        capacity_threshold = capacity_reference * prefetch_ratio
        utilization = 1.0 - (available_capacity / max(1.0, capacity_reference))
        eviction_candidates: List[str] = []

        current_time = get_simulation_time()

        def _select_evictions(required_space: float) -> List[str]:
            if not cache_snapshot or required_space <= 0:
                return []
            scored_items: List[Tuple[float, str]] = []
            max_capacity = max(1.0, capacity_reference)
            for cid, meta in cache_snapshot.items():
                size = float(meta.get('size', 0.0) or 0.0)
                if size <= 0.0:
                    size = 0.1
                history = self.access_history.get(cid, [])
                freq = float(len(history))
                last_access = history[-1] if history else float(meta.get('timestamp', 0.0) or 0.0)
                age = max(0.0, current_time - last_access)
                heat_score = float(self.content_heat.get(cid, 0.0))

                size_score = min(1.0, size / max_capacity)
                freq_score = 1.0 - np.tanh(freq / 5.0)  # 高频越小得分越低
                age_score = np.tanh(age / 600.0)  # 超过10分钟逐步接近1
                inverse_heat = 1.0 - heat_score

                value = 0.4 * inverse_heat + 0.3 * age_score + 0.2 * size_score + 0.1 * freq_score
                scored_items.append((value, cid))

            scored_items.sort(key=lambda x: x[0], reverse=True)
            removed: List[str] = []
            reclaimed = 0.0
            for value, cid in scored_items:
                size = float(cache_snapshot.get(cid, {}).get('size', 0.0) or 0.0)
                removed.append(cid)
                reclaimed += size
                if reclaimed >= required_space:
                    break
            return removed

        if heat > high_threshold:
            if available_capacity > data_size:
                return True, f"High-heat cache (heat:{heat:.2f}>{high_threshold:.2f})", eviction_candidates
            eviction_candidates = _select_evictions(data_size - available_capacity)
            if eviction_candidates:
                return True, f"High-heat cache with eviction x{len(eviction_candidates)}", eviction_candidates

        if heat > medium_threshold and available_capacity > max(data_size, capacity_threshold):
            return True, f"Medium-heat prefetch (heat:{heat:.2f}>{medium_threshold:.2f})", eviction_candidates

        if heat > 0.1:
            collaboration_weight = self.agent_params['collaboration_weight']
            cache_probability = heat * collaboration_weight * max(0.0, 1.2 - utilization)
            if np.random.random() < cache_probability:
                if available_capacity > data_size:
                    return True, f"Collaborative cache (p={cache_probability:.2f})", eviction_candidates
                eviction_candidates = _select_evictions(data_size - available_capacity)
                if eviction_candidates:
                    return True, f"Collaborative cache with eviction x{len(eviction_candidates)}", eviction_candidates

        return False, f"Skip cache (heat:{heat:.2f}, free:{available_capacity:.1f}MB)", eviction_candidates

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
        """返回缓存效果指标与关键参数。"""
        total_requests = self.cache_stats['total_requests']
        if total_requests == 0:
            return {
                'hit_rate': 0.0,
                'effectiveness': 0.0,
                'utilization': 0.0,
                'total_requests': 0,
                'evicted_items': 0,
                'collaborative_writes': 0,
                'agent_params': dict(self.agent_params)
            }

        hit_rate = self.cache_stats['cache_hits'] / total_requests
        utilization = self.cache_stats['current_utilization']

        effectiveness = hit_rate * min(1.0, utilization)

        return {
            'hit_rate': hit_rate,
            'effectiveness': effectiveness,
            'utilization': utilization,
            'total_requests': total_requests,
            'evicted_items': self.cache_stats['evicted_items'],
            'collaborative_writes': self.cache_stats['collaborative_writes'],
            'agent_params': dict(self.agent_params)
        }
class AdaptiveMigrationController:
    """
    自适应迁移控制器
    智能体可以控制迁移策略的关键参数
    """

    def __init__(self):
        # 🤖 DRL可学习的迁移参数（初始值为合理默认值）
        self.agent_params = {
            'cpu_overload_threshold': 0.85,    # CPU过载阈值（DRL可调整70-95%）
            'bandwidth_overload_threshold': 0.85,  # 带宽过载阈值
            'load_diff_threshold': 0.20,       # 负载差触发阈值（DRL可调整10-40%）
            'uav_battery_threshold': 0.25,     # UAV电池阈值
            'migration_cost_weight': 0.3,      # 迁移成本权重
            'urgency_threshold_rsu': 0.1,      # RSU紧急阈值
            'urgency_threshold_uav': 0.15      # UAV紧急阈值
        }

        # 🎯 DRL可调整的参数范围
        self.param_bounds = {
            'cpu_overload_threshold': (0.70, 0.95),      # CPU阈值70-95%
            'bandwidth_overload_threshold': (0.70, 0.95), # 带宽阈值70-95%
            'load_diff_threshold': (0.10, 0.40),         # 负载差阈值10-40%
            'uav_battery_threshold': (0.15, 0.40),       # UAV电池15-40%
            'migration_cost_weight': (0.1, 0.6),         # 成本权重0.1-0.6
            'urgency_threshold_rsu': (0.05, 0.25),       # RSU紧急度5-25%
            'urgency_threshold_uav': (0.10, 0.30)        # UAV紧急度10-30%
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
        🔧 根据智能体动作更新迁移参数（激活DRL控制）

        Args:
            agent_actions: DRL输出的迁移参数字典
                {
                    'cpu_overload_threshold': -1~1,
                    'bandwidth_overload_threshold': -1~1,
                    'load_diff_threshold': -1~1,
                    'uav_battery_threshold': -1~1
                }
        """
        if not isinstance(agent_actions, dict):
            return

        # 🔧 激活：将DRL动作映射到实际参数范围
        for param_name, action_value in agent_actions.items():
            if param_name in self.param_bounds:
                # 动作值从[-1, 1]映射到参数范围
                action_value = np.clip(action_value, -1.0, 1.0)
                param_min, param_max = self.param_bounds[param_name]

                # 归一化到[0, 1]再映射到实际范围
                normalized_value = (action_value + 1.0) / 2.0
                param_value = param_min + normalized_value * (param_max - param_min)

                # 更新参数
                self.agent_params[param_name] = param_value

    def get_current_params(self) -> Dict[str, float]:
        """🔧 获取当前DRL控制的迁移参数（用于监控和调试）"""
        return {
            'cpu_threshold': self.agent_params.get('cpu_overload_threshold', 0.85),
            'bandwidth_threshold': self.agent_params.get('bandwidth_overload_threshold', 0.85),
            'load_diff_threshold': self.agent_params.get('load_diff_threshold', 0.20),
            'uav_battery_threshold': self.agent_params.get('uav_battery_threshold', 0.25),
        }

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
            # 1️⃣ 资源阈值触发（🔧 使用DRL可调整的阈值）
            resource_overload = False
            overload_resources = []

            # 🔧 激活DRL控制：使用agent_params中的动态阈值
            cpu_threshold = self.agent_params.get('cpu_overload_threshold', 0.85)
            bw_threshold = self.agent_params.get('bandwidth_overload_threshold', 0.85)

            if cpu_load > cpu_threshold:  # DRL可调整的CPU阈值（70-95%）
                resource_overload = True
                overload_resources.append(f"CPU:{cpu_load:.1%}")
                urgency_score += (cpu_load - cpu_threshold) / (1.0 - cpu_threshold)

            if bandwidth_load > bw_threshold:  # DRL可调整的带宽阈值（70-95%）
                resource_overload = True
                overload_resources.append(f"带宽:{bandwidth_load:.1%}")
                urgency_score += (bandwidth_load - bw_threshold) / (1.0 - bw_threshold)

            if storage_load > 0.85:  # 存储阈值保持固定（较少成为瓶颈）
                resource_overload = True
                overload_resources.append(f"存储:{storage_load:.1%}")

            # 2️⃣ 负载差触发（🔧 使用DRL可调整的负载差阈值）
            load_diff_trigger = False
            max_load_diff = 0.0

            # 🔧 激活DRL控制：使用动态负载差阈值
            load_diff_threshold = self.agent_params.get('load_diff_threshold', 0.20)

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

                        # 🔧 激活DRL控制：负载差阈值由DRL动态调整（10-40%）
                        if load_diff > load_diff_threshold:
                            load_diff_trigger = True

            # 🔥 计算迁移紧急度（🔧 使用DRL参数计算）
            if resource_overload:
                # 使用实际触发阈值计算紧急度
                resource_urgency = max(cpu_load - cpu_threshold, bandwidth_load - bw_threshold, 0.0)
                urgency_score += resource_urgency * 2.0  # 资源过载权重高
                migration_reason = f"资源过载({','.join(overload_resources)})"

            if load_diff_trigger:
                diff_urgency = max_load_diff - load_diff_threshold
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
            # 🚁 UAV多维度触发条件（🔧 激活DRL控制）

            # 🔧 获取DRL可调整的阈值
            uav_battery_threshold = self.agent_params.get('uav_battery_threshold', 0.25)
            cpu_threshold = self.agent_params.get('cpu_overload_threshold', 0.85)
            load_diff_threshold = self.agent_params.get('load_diff_threshold', 0.20)

            # 1️⃣ 电池电量触发（DRL可调整阈值15-40%）
            battery_urgency = 0.0
            if battery_level < uav_battery_threshold:
                battery_urgency = (uav_battery_threshold - battery_level) / max(0.01, uav_battery_threshold)
                urgency_score += battery_urgency * 3.0  # 电池紧急权重最高
                migration_reason = f"UAV电池低({battery_level:.1%})"

            # 2️⃣ 负载过载触发（🔧 使用DRL可调整的CPU阈值）
            # UAV使用稍低的阈值（-5%），因为UAV资源更有限
            uav_cpu_threshold = max(0.70, cpu_threshold - 0.05)
            if cpu_load > uav_cpu_threshold:
                load_urgency = (cpu_load - uav_cpu_threshold) / (1.0 - uav_cpu_threshold)
                urgency_score += load_urgency * 2.0
                if migration_reason:
                    migration_reason += f" + CPU过载({cpu_load:.1%})"
                else:
                    migration_reason = f"UAV CPU过载({cpu_load:.1%})"

            # 3️⃣ 与邻近RSU负载差（🔧 使用DRL可调整的负载差阈值）
            if neighbor_states:
                max_load_diff = 0.0
                for neighbor_id, neighbor_state in neighbor_states.items():
                    if neighbor_id.startswith("rsu_"):  # 只与RSU比较
                        neighbor_load = neighbor_state.get('cpu_load', neighbor_state.get('load_factor', 0.0))
                        load_diff = cpu_load - neighbor_load
                        max_load_diff = max(max_load_diff, load_diff)

                # 🔧 激活DRL控制：负载差阈值动态调整（10-40%）
                if max_load_diff > load_diff_threshold:
                    diff_urgency = max_load_diff - load_diff_threshold
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
            'avg_delay_saved': self.migration_stats['avg_delay_saved'],
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
    """Map continuous actions to semantic cache/migration parameters."""
    if len(agent_actions) < 8:
        agent_actions = np.pad(agent_actions, (0, 8 - len(agent_actions)), mode='constant', constant_values=0.0)

    cache_params = {
        'heat_threshold_high': np.clip(agent_actions[0], -1.0, 1.0),
        'heat_threshold_medium': np.clip(agent_actions[1], -1.0, 1.0),
        'prefetch_ratio': np.clip(agent_actions[2], -1.0, 1.0),
        'collaboration_weight': np.clip(agent_actions[3], -1.0, 1.0),
    }

    migration_params = {
        'cpu_overload_threshold': np.clip(agent_actions[4], -1.0, 1.0),
        'bandwidth_overload_threshold': np.clip(agent_actions[5], -1.0, 1.0),
        'uav_battery_threshold': np.clip(agent_actions[6], -1.0, 1.0),
        'load_diff_threshold': np.clip(agent_actions[7], -1.0, 1.0),
    }

    return cache_params, migration_params
