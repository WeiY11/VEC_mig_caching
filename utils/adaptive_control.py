#!/usr/bin/env python3
"""
自适应缓存和迁移控制组件
允许智能体学习和控制缓存迁移的关键参数
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import time
from collections import defaultdict
import os
# 🔧 修复：导入统一时间管理器
from .unified_time_manager import get_simulation_time

class AdaptiveCacheController:
    """
    自适应缓存控制器
    智能体可以控制缓存策略的关键参数
    """

    def __init__(self, cache_capacity: float = 100.0):
        self.cache_capacity = cache_capacity

        # 🔧 修复：降低缓存阈值，提高命中率
        self.agent_params = {
            'heat_threshold_high': 0.5,      # 降低高热度阈值：50% [从0.7降到0.5]
            'heat_threshold_medium': 0.25,   # 降低中热度阈值：25% [从0.35降到0.25]
            'prefetch_ratio': 0.08,          # 降低预取比例：8% [从0.05降到0.08，更积极缓存]
            'collaboration_weight': 0.5      # 增加协作权重：50% [从0.3增到0.5]
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
            'collaborative_writes': 0,
            'prefetch_events': 0
        }

        # 热度计算
        self.content_heat = defaultdict(float)
        self.access_history = defaultdict(list)

        # 联动控制参数（由策略协调器/智能体动态调整）
        self.joint_params = {
            'prefetch_lead_time': 0.4,  # 秒，更保守的预取窗口
            'migration_backoff': 0.2,   # 初始退避系数，逐步放开
        }

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

        # 频率热度：使用平方根避免极端值
        frequency_heat = min(1.0, np.sqrt(len(recent_accesses) / 5.0))  # 🔧 从8次降到5次，更容易达到满热度

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
        total_capacity_mb: float,
        cache_priority: float = 0.0  # 🔧 优化8: 添加缓存优先级参数
    ) -> Tuple[bool, str, List[str]]:
        """
        Decide whether to cache a content item. Returns eviction candidates when needed.
        
        🔧 优化: 结合任务的cache_priority进行更智能的决策
        - 高优先级任务更容易被缓存
        - 低优先级任务需要更高的热度
        """
        heat = self.content_heat.get(content_id, 0.0)
        
        # 🔧 优化: 结合cache_priority调整热度
        # 高优先级任务（如video_process）即使热度较低也可能被缓存
        adjusted_heat = heat + cache_priority * 0.3  # cache_priority提供最多30%加成

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

        # 🔧 优化: 使用adjusted_heat进行决策
        if adjusted_heat > high_threshold:
            if available_capacity > data_size:
                reason = f"High-heat cache (heat:{heat:.2f}"
                if cache_priority > 0:
                    reason += f"+priority:{cache_priority:.2f}"
                reason += f">{high_threshold:.2f})"
                return True, reason, eviction_candidates
            eviction_candidates = _select_evictions(data_size - available_capacity)
            if eviction_candidates:
                return True, f"High-heat cache with eviction x{len(eviction_candidates)}", eviction_candidates

        if adjusted_heat > medium_threshold and available_capacity > max(data_size, capacity_threshold):
            reason = f"Medium-heat prefetch (heat:{heat:.2f}"
            if cache_priority > 0:
                reason += f"+priority:{cache_priority:.2f}"
            reason += f">{medium_threshold:.2f})"
            return True, reason, eviction_candidates

        # 🔧 修复：更积极的缓存策略，降低阈值
        # 对于热度>0.05的内容，就可能被缓存
        if adjusted_heat > 0.05:
            collaboration_weight = self.agent_params['collaboration_weight']
            cache_probability = adjusted_heat * collaboration_weight * max(0.0, 1.2 - utilization)
            if np.random.random() < cache_probability:
                if available_capacity > data_size:
                    return True, f"Collaborative cache (p={cache_probability:.2f})", eviction_candidates
                eviction_candidates = _select_evictions(data_size - available_capacity)
                if eviction_candidates:
                    return True, f"Collaborative cache with eviction x{len(eviction_candidates)}", eviction_candidates

        return False, f"Skip cache (heat:{heat:.2f}, priority:{cache_priority:.2f}, free:{available_capacity:.1f}MB)", eviction_candidates

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
                'miss_rate': 0.0,  # 🔧 修复：添加miss_rate字段
                'effectiveness': 0.0,
                'utilization': 0.0,
                'total_requests': 0,
                'evicted_items': 0,
                'collaborative_writes': 0,
                'prefetch_events': self.cache_stats['prefetch_events'],
                'agent_params': dict(self.agent_params),
                'joint_params': dict(self.joint_params)
            }

        hit_rate = self.cache_stats['cache_hits'] / total_requests
        miss_rate = 1.0 - hit_rate  # 🔧 修复：计算miss_rate
        utilization = self.cache_stats['current_utilization']

        effectiveness = hit_rate * min(1.0, utilization)

        return {
            'hit_rate': hit_rate,
            'miss_rate': miss_rate,  # 🔧 修复：添加miss_rate到返回值
            'effectiveness': effectiveness,
            'utilization': utilization,
            'total_requests': total_requests,
            'evicted_items': self.cache_stats['evicted_items'],
            'collaborative_writes': self.cache_stats['collaborative_writes'],
            'prefetch_events': self.cache_stats['prefetch_events'],
            'agent_params': dict(self.agent_params),
            'joint_params': dict(self.joint_params)
        }

    def apply_joint_params(self, joint_params: Dict[str, float]) -> None:
        """应用策略协调器下发的联动参数（如预取窗口）。"""
        if not isinstance(joint_params, dict):
            return
        lead_time = joint_params.get('prefetch_lead_time')
        if lead_time is not None:
            lead_time = float(np.clip(lead_time, 0.0, 10.0))
            self.joint_params['prefetch_lead_time'] = lead_time

        backoff = joint_params.get('migration_backoff')
        if backoff is not None:
            backoff = float(np.clip(backoff, 0.0, 1.0))
            self.joint_params['migration_backoff'] = backoff

    def register_prefetch_event(self, count: int = 1) -> None:
        """记录一次预取事件，供监控和奖励计算使用。"""
        try:
            count = int(max(0, count))
        except Exception:
            count = 0
        self.cache_stats['prefetch_events'] += count

    def get_joint_params_snapshot(self) -> Dict[str, float]:
        """返回当前缓存侧的联动参数配置。"""
        return dict(self.joint_params)
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

        # 联动控制状态
        self.joint_backoff_factor = 0.2
        self.dynamic_threshold_scale = 1.0
        self.cache_feedback = {'hit_rate': 0.0, 'miss_rate': 0.0}

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

    def apply_joint_params(self, joint_params: Dict[str, float]) -> None:
        """应用联合策略参数（如迁移退避系数）。"""
        if not isinstance(joint_params, dict):
            return
        backoff = joint_params.get('migration_backoff')
        if backoff is not None:
            backoff = float(np.clip(backoff, 0.0, 1.0))
            self.joint_backoff_factor = backoff

    def ingest_cache_feedback(self, hit_rate: float, miss_rate: float) -> None:
        """根据缓存命中率动态调整迁移阈值."""
        hit_rate = float(np.clip(hit_rate, 0.0, 1.0))
        miss_rate = float(np.clip(miss_rate, 0.0, 1.0))
        self.cache_feedback = {'hit_rate': hit_rate, 'miss_rate': miss_rate}

        # 命中率越高，迁移阈值越宽松；命中率下降则收紧
        # 线性缩放到 [0.85, 1.15] 区间
        scale = 1.0 + 0.3 * (0.5 - miss_rate)
        self.dynamic_threshold_scale = float(np.clip(scale, 0.7, 1.2))

    def get_current_params(self) -> Dict[str, float]:
        """🔧 获取当前DRL控制的迁移参数（用于监控和调试）"""
        return {
            'cpu_threshold': self.agent_params.get('cpu_overload_threshold', 0.85),
            'bandwidth_threshold': self.agent_params.get('bandwidth_overload_threshold', 0.85),
            'load_diff_threshold': self.agent_params.get('load_diff_threshold', 0.20),
            'uav_battery_threshold': self.agent_params.get('uav_battery_threshold', 0.25),
        }

    def get_joint_params_snapshot(self) -> Dict[str, float]:
        """返回迁移侧联动参数。"""
        return {
            'migration_backoff': self.joint_backoff_factor,
            'threshold_scale': self.dynamic_threshold_scale,
            'cache_feedback': dict(self.cache_feedback)
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
        cooldown = 1.0 + 4.0 * float(np.clip(self.joint_backoff_factor, 0.0, 1.0))
        if (node_id in self.last_migration_time and 
            current_time - self.last_migration_time[node_id] < cooldown):  # 联动冷却期
            return False, "冷却期内", 0.0

        # 获取节点状态
        cpu_load = current_state.get('cpu_load', current_state.get('load_factor', 0.0))
        bandwidth_load = current_state.get('bandwidth_load', 0.0)
        storage_load = current_state.get('storage_load', 0.0)
        battery_level = current_state.get('battery_level', 1.0)
        hotspot_intensity = float(np.clip(current_state.get('hotspot_intensity', 0.0), 0.0, 1.0))
        
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
            load_diff_threshold = self.agent_params.get('load_diff_threshold', 0.20)

            scale = float(np.clip(self.dynamic_threshold_scale, 0.7, 1.2))
            cpu_bounds = self.param_bounds['cpu_overload_threshold']
            bw_bounds = self.param_bounds['bandwidth_overload_threshold']
            diff_bounds = self.param_bounds['load_diff_threshold']
            cpu_threshold = float(np.clip(cpu_threshold * scale, cpu_bounds[0], cpu_bounds[1]))
            bw_threshold = float(np.clip(bw_threshold * scale, bw_bounds[0], bw_bounds[1]))
            load_diff_threshold = float(np.clip(load_diff_threshold * scale, diff_bounds[0], diff_bounds[1]))

            if hotspot_intensity > 0.0:
                cpu_threshold = max(
                    self.param_bounds['cpu_overload_threshold'][0],
                    cpu_threshold - 0.1 * hotspot_intensity
                )
                bw_threshold = max(
                    self.param_bounds['bandwidth_overload_threshold'][0],
                    bw_threshold - 0.05 * hotspot_intensity
                )
                load_diff_threshold = max(
                    self.param_bounds['load_diff_threshold'][0],
                    load_diff_threshold - 0.05 * hotspot_intensity
                )
                if not migration_reason:
                    migration_reason = "热门内容预热"

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
                overload_reason = f"资源过载({','.join(overload_resources)})"
                migration_reason = overload_reason if not migration_reason else f"{migration_reason} + {overload_reason}"

            if load_diff_trigger:
                diff_urgency = max_load_diff - load_diff_threshold
                urgency_score += diff_urgency * 1.5  # 负载差权重中等
                if migration_reason:
                    migration_reason += f" + 负载差({max_load_diff:.1%})"
                else:
                    migration_reason = f"负载差过大({max_load_diff:.1%})"

            # 🔧 优化：更积极的迁移策略，敢于尝试有风险的迁移
            if urgency_score > 0.1:  # 阈值更保守，避免早期过度迁移
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
            scale = float(np.clip(self.dynamic_threshold_scale, 0.7, 1.2))
            uav_cpu_threshold_base = self.agent_params.get('cpu_overload_threshold', 0.85)
            uav_cpu_threshold_base = float(np.clip(uav_cpu_threshold_base * scale, self.param_bounds['cpu_overload_threshold'][0], self.param_bounds['cpu_overload_threshold'][1]))
            uav_cpu_threshold = max(0.70, uav_cpu_threshold_base - 0.05)
            load_diff_threshold = float(np.clip(load_diff_threshold * scale, self.param_bounds['load_diff_threshold'][0], self.param_bounds['load_diff_threshold'][1]))
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
                if urgency_score > 0.12:  # 阈值更保守，避免早期过度迁移
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
                'agent_params': dict(self.agent_params),
                'joint_params': self.get_joint_params_snapshot()
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
            'avg_delay_saved': self.migration_stats['avg_delay_saved'],
            'agent_params': dict(self.agent_params),
            'joint_params': self.get_joint_params_snapshot()
        }


def map_agent_actions_to_params(agent_actions: np.ndarray) -> Tuple[Dict, Dict, Dict]:
    """Map continuous actions to semantic cache/migration/joint parameters."""
    if len(agent_actions) < 10:
        agent_actions = np.pad(agent_actions, (0, 10 - len(agent_actions)), mode='constant', constant_values=0.0)

    clipped_actions = np.clip(agent_actions, -1.0, 1.0)
    try:
        action_scale = float(os.environ.get("CACHE_ACTION_SCALE", 1.0))
    except Exception:
        action_scale = 1.0
    scaled_actions = clipped_actions * action_scale  # 放大可控幅度，提升缓存/迁移可调性

    def _map_to_range(value: float, low: float, high: float) -> float:
        value = np.clip(value, -1.0, 1.0)
        return low + ((value + 1.0) * 0.5) * (high - low)

    cache_params = {
        'heat_threshold_high': scaled_actions[0],
        'heat_threshold_medium': scaled_actions[1],
        'prefetch_ratio': scaled_actions[2],
        'collaboration_weight': scaled_actions[3],
    }

    migration_params = {
        'cpu_overload_threshold': scaled_actions[4],
        'bandwidth_overload_threshold': scaled_actions[5],
        'uav_battery_threshold': scaled_actions[6],
        'load_diff_threshold': scaled_actions[7],
    }

    joint_params = {
        'prefetch_lead_time': _map_to_range(scaled_actions[8], 0.0, 5.0),
        'migration_backoff': _map_to_range(scaled_actions[9], 0.0, 1.0)
    }

    return cache_params, migration_params, joint_params
