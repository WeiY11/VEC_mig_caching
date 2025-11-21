"""
边缘缓存管理系统 - 对应论文第7节
实现智能缓存策略、协作缓存和背包优化算法
"""
import numpy as np
import time
import math
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
from collections import defaultdict, OrderedDict
# 🔧 修复：导入统一时间管理器
from utils.unified_time_manager import get_simulation_time

from models.data_structures import Task, TaskType
from config import config
from utils.common import calculate_zipf_probability, ExponentialMovingAverage


class CacheReplacementPolicy(Enum):
    """缓存替换策略枚举"""
    LRU = "lru"      # Least Recently Used
    LFU = "lfu"      # Least Frequently Used
    FIFO = "fifo"    # First In First Out
    HYBRID = "hybrid" # 混合策略


@dataclass
class CachedItem:
    """缓存项数据结构"""
    content_id: str
    data_size: float
    access_count: int = 0
    last_access_time: float = 0.0
    cache_time: float = 0.0
    
    # 热度相关
    historical_heat: float = 0.0
    slot_heat: float = 0.0
    zipf_popularity: float = 0.0
    
    # 预测相关
    predicted_requests: float = 0.0
    cache_value: float = 0.0


class HeatBasedCacheStrategy:
    """
    基于热度的缓存策略 - 对应论文第7节
    结合历史热度、时间槽热度和Zipf流行度分布
    """
    
    def __init__(self, slot_duration: Optional[float] = None, total_slots: Optional[int] = None):
        """
        初始化热度策略
        
        Args:
            slot_duration: 时间槽时长（秒），None则自适应
            total_slots: 总时间槽数，None则自适应
        """
        # 🔧 创新优化:自适应热度参数动态调整
        # 🎯 核心创新:根据系统负载和时隙模式动态调整衰减速度
        self.decay_factor = 0.88          # 🚀 创新:进一步加快冷却(0.92→0.88),快速响应内容流行度变化
        self.heat_mix_factor = 0.6        # 🚀 创新:更重视实时热度(0.7→0.6),捕捉动态热点
        self.zipf_exponent = 0.8          # Zipf分布参数
                
        # 🆕 创新:自适应参数(根据系统负载动态调整)
        self.adaptive_decay_enabled = True  # 启用自适应衰减
        self.min_decay_factor = 0.80      # 高负载时最小衰减(更激进淘汰)
        self.max_decay_factor = 0.92      # 低负载时最大衰减(更保守缓存)
        self.system_load_threshold = 0.7  # 负载阈值
        
        # 🚀 自适应时间槽配置
        self.slot_duration = slot_duration if slot_duration is not None else 10.0  # 默认10秒
        self.total_slots = total_slots if total_slots is not None else 200  # 默认200槽
        self.adaptive_slot = (slot_duration is None)  # 是否启用自适应
        
        # 热度统计
        self.historical_heat: Dict[str, float] = defaultdict(float)
        self.slot_heat: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.current_slot = 0
        self.simulation_start_time = get_simulation_time()  # 记录仿真开始时间
        
        # 自适应调整相关
        self.access_count_per_slot = defaultdict(int)  # 每个槽的访问计数
        self.last_slot_adjustment = 0
        
        # 访问统计
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.content_popularity_rank: Dict[str, int] = {}
        
        # 移动平均计算器
        self.avg_heat = ExponentialMovingAverage(alpha=0.1)
        
        # 性能优化：记录上次排名更新（用于惰性更新）
        self._last_rank_update = 0
    
    def update_heat(self, content_id: str, access_weight: float = 1.0, system_load: float = 0.5):
        """
        🚀 创新优化:自适应热度更新 - 根据系统负载动态调整衰减
            
        创新点:
        1. 高负载时加快衰减,快速腾出空间
        2. 低负载时减慢衰减,保留更多历史信息
        3. 引入访问间隔加权,频繁访问的内容获得更高权重
        """
        # 🆕 创新:自适应衰减因子(根据系统负载)
        if self.adaptive_decay_enabled:
            if system_load > self.system_load_threshold:
                # 高负载:激进淘汰,快速响应
                current_decay = self.min_decay_factor
            else:
                # 低负载:保守缓存,利用历史
                current_decay = self.max_decay_factor
        else:
            current_decay = self.decay_factor
            
        # 🆕 创新:访问间隔加权(频繁访问获得boost)
        access_boost = 1.0
        if content_id in self.access_history and len(self.access_history[content_id]) >= 2:
            # 计算最近两次访问间隔
            last_interval = get_simulation_time() - self.access_history[content_id][-1]
            if last_interval < 30.0:  # 30秒内再次访问,视为高频
                access_boost = 1.5  # 提升50%权重
            
        # 更新历史热度 - 式(35) + 创新自适应机制
        self.historical_heat[content_id] = (current_decay * self.historical_heat[content_id] + 
                                           access_weight * access_boost)
        
        # 🚀 自适应时间槽计算
        simulation_time = get_simulation_time()
        current_slot = int(simulation_time / self.slot_duration) % self.total_slots
        self.slot_heat[content_id][current_slot] += access_weight
        
        # 记录当前槽的访问计数（用于自适应调整）
        self.access_count_per_slot[current_slot] += 1
        
        # 🚀 自适应调整时间槽粒度（每1000次访问检查一次）
        if self.adaptive_slot and len(self.historical_heat) % 1000 == 0:
            self._adjust_slot_granularity()
        
        # 🔧 修复：记录仿真时间（优化：只保留最近20次，减少80%内存）
        self.access_history[content_id].append(get_simulation_time())
        
        # 限制历史长度（优化：从100减少到20）
        if len(self.access_history[content_id]) > 20:
            self.access_history[content_id].pop(0)
    
    def _adjust_slot_granularity(self):
        """
        自适应调整时间槽粒度
        根据访问密度动态调整slot_duration
        """
        if len(self.access_count_per_slot) < 10:
            return  # 数据不足，不调整
        
        # 计算平均每槽访问数
        avg_accesses_per_slot = np.mean(list(self.access_count_per_slot.values()))
        
        # 目标：每槽20-50次访问为最佳（既能捕捉模式，又不过于细碎）
        if avg_accesses_per_slot > 100:
            # 访问太密集，增加槽时长
            self.slot_duration = min(30.0, self.slot_duration * 1.5)
        elif avg_accesses_per_slot < 10:
            # 访问太稀疏，减小槽时长
            self.slot_duration = max(5.0, self.slot_duration * 0.8)
        
        # 记录调整
        self.last_slot_adjustment = get_simulation_time()
    
    def calculate_combined_heat(self, content_id: str) -> float:
        """
        计算综合热度 - 对应论文式(37)
        Heat(c) = η * H_hist(c) + (1-η) * H_slot(c,t)
        """
        hist_heat = self.historical_heat.get(content_id, 0.0)
        
        # 🚀 使用自适应时间槽
        simulation_time = get_simulation_time()
        current_slot = int(simulation_time / self.slot_duration) % self.total_slots
        slot_heat = self.slot_heat[content_id].get(current_slot, 0.0)
        
        combined_heat = (self.heat_mix_factor * hist_heat + 
                        (1 - self.heat_mix_factor) * slot_heat)
        
        return combined_heat
    
    def calculate_zipf_popularity(self, content_id: str, total_contents: int) -> float:
        """
        计算Zipf流行度（优化版：惰性更新排名）
        
        性能优化：仅在访问历史变化超过阈值时重新排名，减少99%计算
        """
        # 计算当前总访问数
        current_total_accesses = sum(len(h) for h in self.access_history.values())
        
        # 仅在访问历史变化超过100次时重新排名
        if not hasattr(self, '_last_rank_update') or \
           current_total_accesses - self._last_rank_update > 100:
            
            # 根据访问次数排名
            access_counts = {cid: len(history) for cid, history in self.access_history.items()}
            sorted_contents = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
            
            self.content_popularity_rank.clear()
            for rank, (cid, _) in enumerate(sorted_contents, 1):
                self.content_popularity_rank[cid] = rank
            
            self._last_rank_update = current_total_accesses
        
        rank = self.content_popularity_rank.get(content_id, total_contents)
        return calculate_zipf_probability(rank, total_contents, self.zipf_exponent)
    
    def cleanup_stale_data(self, current_time: float, staleness_threshold: float = 7200):
        """
        清理过期数据（优化：防止内存泄漏）
        
        Args:
            current_time: 当前仿真时间
            staleness_threshold: 过期阈值（秒，默认2小时）
        """
        stale_contents = []
        
        # 找出过期内容
        for content_id in list(self.historical_heat.keys()):
            if content_id in self.access_history and self.access_history[content_id]:
                last_access = self.access_history[content_id][-1]
                if current_time - last_access > staleness_threshold:
                    stale_contents.append(content_id)
        
        # 清理或降低热度
        for content_id in stale_contents:
            # 降低热度但不完全删除（允许重新变热）
            self.historical_heat[content_id] *= 0.3
            
            # 如果热度太低，完全删除
            if self.historical_heat[content_id] < 0.01:
                del self.historical_heat[content_id]
                if content_id in self.access_history:
                    del self.access_history[content_id]
                if content_id in self.slot_heat:
                    del self.slot_heat[content_id]
                if content_id in self.content_popularity_rank:
                    del self.content_popularity_rank[content_id]
    
    def get_cache_priority(self, content_id: str, data_size: float, 
                          total_contents: int) -> float:
        """
        计算缓存优先级
        综合热度、流行度、大小等因素
        """
        # 基础热度
        heat = self.calculate_combined_heat(content_id)
        
        # Zipf流行度
        zipf_pop = self.calculate_zipf_popularity(content_id, total_contents)
        
        # 大小惩罚 (小文件优先)
        size_penalty = math.log(1 + data_size / 1e6)  # MB级别
        
        # 最近性奖励
        recency_bonus = 0.0
        if content_id in self.access_history and self.access_history[content_id]:
            last_access = self.access_history[content_id][-1]
            # 🔧 修复：使用仿真时间计算间隔
            time_since_access = get_simulation_time() - last_access
            recency_bonus = max(0, 1.0 - time_since_access / 600)  # 10分钟内的奖励(适应仿真)
        
        # 综合优先级（优化权重：更重视实际访问热度）
        priority = (
            0.5 * heat +           # 增加热度权重（从0.4→0.5），更重视实际访问
            0.2 * zipf_pop +       # 降低Zipf权重（从0.3→0.2），减少理论假设依赖
            0.25 * recency_bonus - # 增加新鲜度权重（从0.2→0.25），快速响应变化
            0.05 * size_penalty    # 降低大小惩罚（从0.1→0.05），允许缓存更多内容
        )
        
        return max(0.0, priority)


class CollaborativeCacheManager:
    """
    协作缓存管理器
    实现邻居协作和背包优化算法
    """
    
    def __init__(self, node_id: str, node_type: Optional[str] = None):
        self.node_id = node_id
        self.node_type = node_type if node_type else "RSU"  # 默认RSU
        
        # 🎯 P0-1优化：根据节点类型设置容量和策略
        if self.node_type == "Vehicle":
            self.cache_capacity = config.cache.vehicle_cache_capacity
            policy_name = config.cache.vehicle_cache_policy.lower()
        elif self.node_type == "UAV":
            self.cache_capacity = config.cache.uav_cache_capacity
            policy_name = config.cache.uav_cache_policy.lower()
        else:  # RSU
            self.cache_capacity = config.cache.rsu_cache_capacity
            policy_name = config.cache.rsu_cache_policy.lower()
        
        # 缓存存储
        self.cached_items: Dict[str, CachedItem] = {}
        self.current_usage = 0.0
        
        # 替换策略
        # 🎯 P0-1优化：使用针对性策略配置
        if policy_name == "lru":
            self.replacement_policy = CacheReplacementPolicy.LRU
        elif policy_name == "lfu":
            self.replacement_policy = CacheReplacementPolicy.LFU
        elif policy_name == "fifo":
            self.replacement_policy = CacheReplacementPolicy.FIFO
        else:
            self.replacement_policy = CacheReplacementPolicy.HYBRID
        self.heat_strategy = HeatBasedCacheStrategy()
        
        # 邻居协作
        self.neighbor_nodes: Set[str] = set()
        self.neighbor_cache_states: Dict[str, Set[str]] = {}
        self.collaboration_sync_interval = 300  # 5分钟同步一次
        self.last_sync_time = 0.0
        
        # 🔧 修复：降低预取激进程度
        self.prefetch_window_ratio = 0.03  # 预取窗口降至3%，减少资源占用
        self.prefetch_threshold = 0.8      # 提高预取阈值，更加谨慎
        
        # 统计信息
        self.cache_stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'prefetch_hits': 0,
            'collaboration_saves': 0
        }
        
        # 背包优化相关
        self.knapsack_enabled = True
        self.value_weights = {
            'hit_value': 3.0,
            'cost_penalty': 1.0,
            'over_budget_penalty': 5.0,
            'energy_penalty': 0.2
        }
    
    def request_content(self, content_id: str, data_size: float) -> Tuple[bool, str]:
        """
        请求内容 - 对应论文第7节的四类动作
        
        Returns:
            (是否命中, 动作类型)
        """
        self.cache_stats['total_requests'] += 1
        
        # 🎯 P1-1优化：每100次请求执行一次预测缓存
        if self.cache_stats['total_requests'] % 100 == 0:
            predicted_contents = self.predictive_caching()
            # 记录预测结果（可用于后续预加载）
            if not hasattr(self, '_predicted_contents'):
                self._predicted_contents = set()
            self._predicted_contents.update(predicted_contents)
        
        # 更新热度
        self.heat_strategy.update_heat(content_id)
        
        # 检查本地缓存命中
        if content_id in self.cached_items:
            self._handle_cache_hit(content_id)
            return True, "cache_hit"  # 动作0
        
        # 检查邻居协作
        if self._check_neighbor_collaboration(content_id):
            self.cache_stats['collaboration_saves'] += 1
            return True, "neighbor_hit"
        
        # 缓存未命中，决定缓存动作
        action = self._decide_cache_action(content_id, data_size)
        
        if action == 1:
            # 高热度内容，直接缓存
            success = self._add_to_cache(content_id, data_size)
            self.cache_stats['cache_misses'] += 1
            return False, "cache_and_store" if success else "cache_full"
        
        elif action == 2:
            # 中等热度内容，预取
            success = self._prefetch_content(content_id, data_size)
            self.cache_stats['cache_misses'] += 1
            return False, "prefetch" if success else "prefetch_failed"
        
        elif action == 3:
            # 背包替换
            success = self._knapsack_replacement(content_id, data_size)
            self.cache_stats['cache_misses'] += 1
            return False, "knapsack_replace" if success else "replace_failed"
        
        else:
            # 不缓存
            self.cache_stats['cache_misses'] += 1
            return False, "no_cache"
    
    def _handle_cache_hit(self, content_id: str):
        """处理缓存命中"""
        if content_id in self.cached_items:
            item = self.cached_items[content_id]
            item.access_count += 1
            # 🔧 修复：使用仿真时间
            item.last_access_time = get_simulation_time()
            
            self.cache_stats['cache_hits'] += 1
            
            # 更新LRU顺序 (如果使用LRU策略)
            if self.replacement_policy == CacheReplacementPolicy.LRU:
                # 重新插入以更新顺序
                self.cached_items[content_id] = self.cached_items.pop(content_id)
    
    def _check_neighbor_collaboration(self, content_id: str) -> bool:
        """
        🎯 P1-2优化：检查邻居协作缓存（含成本评估）
        """
        for neighbor_id, cached_contents in self.neighbor_cache_states.items():
            if content_id in cached_contents:
                # 🔥 调用成本评估，只在值得时才协作
                should_collaborate, cost = self._evaluate_collaboration_cost(content_id, neighbor_id)
                if should_collaborate:
                    return True
        return False
    
    def _decide_cache_action(self, content_id: str, data_size: float) -> int:
        """
        决定缓存动作 - 对应论文决策逻辑
        
        Returns:
            0: 已缓存, 1: 高热度缓存, 2: 预取, 3: 背包替换
        """
        # 计算内容热度
        heat = self.heat_strategy.calculate_combined_heat(content_id)
        
        # 获取可用容量
        available_capacity = self.cache_capacity - self.current_usage
        
        # 🔧 优化：基于实际热度范围[0,1]设置合理阈值
        high_heat_threshold = 0.7   # 70%热度触发高优先级缓存
        medium_heat_threshold = 0.4  # 40%热度触发中等优先级缓存
        capacity_threshold = self.cache_capacity * 0.05  # 5%容量保留阈值
        
        # 决策逻辑
        if heat > high_heat_threshold and available_capacity > capacity_threshold:
            return 1  # 高热度且有足够容量，直接缓存
        
        elif medium_heat_threshold < heat <= high_heat_threshold:
            return 2  # 中等热度，预取
        
        elif available_capacity <= 0 and self.knapsack_enabled:
            return 3  # 容量不足，背包替换
        
        else:
            return 0  # 不缓存
    
    def _add_to_cache(self, content_id: str, data_size: float) -> bool:
        """添加内容到缓存"""
        if self.current_usage + data_size > self.cache_capacity:
            # 容量不足，尝试替换
            if not self._make_space(data_size):
                return False
        
        # 创建缓存项
        item = CachedItem(
            content_id=content_id,
            data_size=data_size,
            # 🔧 修复：使用仿真时间
            cache_time=get_simulation_time(),
            last_access_time=get_simulation_time()
        )
        
        # 计算热度和优先级
        item.historical_heat = self.heat_strategy.historical_heat.get(content_id, 0.0)
        item.cache_value = self.heat_strategy.get_cache_priority(content_id, data_size, len(self.cached_items) + 1)
        
        self.cached_items[content_id] = item
        self.current_usage += data_size
        
        return True
    
    def _prefetch_content(self, content_id: str, data_size: float) -> bool:
        """预取内容"""
        # 检查预取窗口容量
        prefetch_capacity = self.cache_capacity * self.prefetch_window_ratio
        
        if data_size <= prefetch_capacity:
            # 在预取窗口内尝试缓存
            return self._add_to_cache(content_id, data_size)
        
        return False
    
    def _knapsack_replacement(self, content_id: str, data_size: float) -> bool:
        """
        背包优化替换 - 对应论文背包算法
        最大化缓存价值，约束总容量
        """
        if not self.cached_items:
            return self._add_to_cache(content_id, data_size)
        
        # 计算新内容的价值
        new_value = self.heat_strategy.get_cache_priority(content_id, data_size, len(self.cached_items) + 1)
        
        # 候选替换项列表 (价值, 大小, content_id)
        candidates = []
        for cid, item in self.cached_items.items():
            value = item.cache_value
            candidates.append((value, item.data_size, cid))
        
        # 贪心背包算法：按价值密度排序
        candidates.sort(key=lambda x: x[0] / x[1], reverse=False)  # 价值密度从低到高
        
        # 寻找可以释放的空间
        freed_space = 0.0
        items_to_remove = []
        
        for value, size, cid in candidates:
            if freed_space >= data_size:
                break
            
            # 如果新内容价值更高，则替换
            if new_value > value:
                freed_space += size
                items_to_remove.append(cid)
        
        # 执行替换
        if freed_space >= data_size:
            for cid in items_to_remove:
                self._evict_item(cid)
            
            return self._add_to_cache(content_id, data_size)
        
        return False
    
    def _make_space(self, required_space: float) -> bool:
        """
        根据替换策略腾出空间
        🎯 P3-3优化：批量淘汰优化
        """
        # 🔥 批量淘汰优化：预留额外空间减少频繁淘汰
        # 一次淘汰释放120%的所需空间
        buffer_ratio = 1.2
        target_space = required_space * buffer_ratio
        
        if self.replacement_policy == CacheReplacementPolicy.LRU:
            return self._lru_eviction(target_space)
        elif self.replacement_policy == CacheReplacementPolicy.LFU:
            return self._lfu_eviction(target_space)
        elif self.replacement_policy == CacheReplacementPolicy.FIFO:
            return self._fifo_eviction(target_space)
        else:  # HYBRID
            return self._hybrid_eviction(target_space)
    
    def _lru_eviction(self, required_space: float) -> bool:
        """LRU替换策略"""
        sorted_items = sorted(self.cached_items.items(), 
                            key=lambda x: x[1].last_access_time)
        
        freed_space = 0.0
        for content_id, item in sorted_items:
            if freed_space >= required_space:
                break
            
            freed_space += item.data_size
            self._evict_item(content_id)
        
        return freed_space >= required_space
    
    def _lfu_eviction(self, required_space: float) -> bool:
        """LFU替换策略"""
        sorted_items = sorted(self.cached_items.items(), 
                            key=lambda x: x[1].access_count)
        
        freed_space = 0.0
        for content_id, item in sorted_items:
            if freed_space >= required_space:
                break
            
            freed_space += item.data_size
            self._evict_item(content_id)
        
        return freed_space >= required_space
    
    def _fifo_eviction(self, required_space: float) -> bool:
        """FIFO替换策略"""
        sorted_items = sorted(self.cached_items.items(), 
                            key=lambda x: x[1].cache_time)
        
        freed_space = 0.0
        for content_id, item in sorted_items:
            if freed_space >= required_space:
                break
            
            freed_space += item.data_size
            self._evict_item(content_id)
        
        return freed_space >= required_space
    
    def _hybrid_eviction(self, required_space: float) -> bool:
        """
        🎯 P3-1优化：混合替换策略（自适应权重）
        综合考虑：时间性、频率、价值（权重自适应）
        """
        # 🔥 自适应权重计算
        weights = self._adaptive_hybrid_weights()
        
        # 综合考虑访问频率、最近性和缓存价值
        scored_items = []
        
        for content_id, item in self.cached_items.items():
            # 🔧 修复：计算综合分数 (分数越低越容易被替换)
            recency_score = (get_simulation_time() - item.last_access_time) / 600  # 改为10分钟适应仿真
            frequency_score = 1.0 / max(1, item.access_count)
            value_score = 1.0 / max(0.1, item.cache_value)
            
            # 🎯 使用自适应权重
            total_score = (weights['recency'] * recency_score + 
                          weights['frequency'] * frequency_score + 
                          weights['value'] * value_score)
            scored_items.append((total_score, content_id, item))
        
        # 按分数排序，分数高的优先替换
        scored_items.sort(key=lambda x: x[0], reverse=True)
        
        freed_space = 0.0
        for score, content_id, item in scored_items:
            if freed_space >= required_space:
                break
            
            freed_space += item.data_size
            self._evict_item(content_id)
        
        return freed_space >= required_space
    
    # 🎯 P3-1优化：自适应权重计算
    def _adaptive_hybrid_weights(self) -> Dict[str, float]:
        """
        根据当前缓存状态动态调整混合策略权重
        
        Returns:
            {'recency': float, 'frequency': float, 'value': float}
        """
        # 默认权重
        weights = {'recency': 0.4, 'frequency': 0.3, 'value': 0.3}
        
        if not self.cached_items:
            return weights
        
        # 计算缓存使用率
        usage_ratio = self.current_usage / self.cache_capacity
        
        # 计算命中率
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0.5
        
        # 🔥 自适应规则：
        # 1. 高使用率(>80%) → 提高frequency权重，保留高频内容
        if usage_ratio > 0.8:
            weights['frequency'] = 0.4
            weights['recency'] = 0.3
            weights['value'] = 0.3
        
        # 2. 低命中率(<60%) → 提高value权重，优化热度选择
        if hit_rate < 0.6:
            weights['value'] = 0.4
            weights['recency'] = 0.35
            weights['frequency'] = 0.25
        
        # 3. 高命中率(>85%) → 提高recency权重，加快更新
        if hit_rate > 0.85:
            weights['recency'] = 0.5
            weights['frequency'] = 0.25
            weights['value'] = 0.25
        
        return weights
    
    def _evict_item(self, content_id: str):
        """从缓存中移除项目"""
        if content_id in self.cached_items:
            item = self.cached_items.pop(content_id)
            self.current_usage -= item.data_size
            self.cache_stats['evictions'] += 1
    
    # 🎯 P1-1优化：预测式缓存预加载
    def predictive_caching(self, prediction_horizon: Optional[int] = None) -> List[str]:
        """
        基于热度趋势预测未来高需求内容
        
        Args:
            prediction_horizon: 预测数量，默认使用配置值
            
        Returns:
            应该预加载的内容ID列表
        """
        if not config.cache.enable_predictive_caching:
            return []
        
        if prediction_horizon is None:
            prediction_horizon = config.cache.prediction_horizon
        
        predictions = []
        current_time = get_simulation_time()
        prediction_threshold = config.cache.prediction_threshold
        
        # 遍历所有有访问历史的内容
        for content_id in self.heat_strategy.access_history.keys():
            access_times = self.heat_strategy.access_history[content_id]
            
            # 至少需3次访问才能预测趋势
            if len(access_times) < 3:
                continue
            
            # 计算访问增长率
            recent_accesses = len([t for t in access_times if current_time - t < 60])  # 最近1分钟
            older_accesses = len([t for t in access_times if 60 <= current_time - t < 120])  # 1-2分钟前
            
            if older_accesses > 0:
                growth_rate = recent_accesses / older_accesses
                if growth_rate > prediction_threshold:  # 增长超过50%
                    # 预测未来需求
                    predicted_requests = recent_accesses * growth_rate
                    
                    # 更新缓存项的预测值
                    if content_id in self.cached_items:
                        self.cached_items[content_id].predicted_requests = predicted_requests
                    
                    predictions.append((content_id, predicted_requests))
        
        # 返回预测需求最高的前N个
        predictions.sort(key=lambda x: x[1], reverse=True)
        return [cid for cid, _ in predictions[:prediction_horizon]]
    
    # 🎯 P1-2优化：协作缓存成本评估
    def _evaluate_collaboration_cost(self, content_id: str, neighbor_id: str) -> Tuple[bool, float]:
        """
        评估协作缓存的成本效益
        
        Args:
            content_id: 内容ID
            neighbor_id: 邻居节点ID
            
        Returns:
            (是否协作, 协作成本)
        """
        # 计算从邻居获取的延迟成本
        # 假设邻居距离存储在 neighbor_distances 中
        if not hasattr(self, 'neighbor_distances'):
            self.neighbor_distances = {}  # 初始化邻居距离字典
        
        distance = self.neighbor_distances.get(neighbor_id, 500)  # 默认500m
        transmission_delay = distance / 3e8 * 1000  # 光速传播延迟(ms)
        bandwidth_cost = 10  # 带宽占用成本(简化)
        
        collaboration_cost = transmission_delay + bandwidth_cost
        
        # 与本地缓存比较
        local_cache_cost = 50  # 本地缓存的固定成本
        
        # 协作成本小于本地成本的1.2倍才值得协作
        return collaboration_cost < local_cache_cost * 1.2, collaboration_cost
    
    # 🎯 P2-2优化：动态容量调整
    def adaptive_capacity_allocation(self, current_load: float, hit_rate: float) -> float:
        """
        根据负载和命中率动态调整缓存容量
        
        策略：
        - 高负载低命中率 → 增加容量
        - 低负载高命中率 → 减少容量（节能）
        
        Args:
            current_load: 当前负载 (0.0-1.0+)
            hit_rate: 缓存命中率 (0.0-1.0)
            
        Returns:
            新的缓存容量
        """
        if not config.cache.enable_dynamic_capacity:
            return self.cache_capacity
        
        base_capacity = self.cache_capacity
        
        # 负载因子：0.0-1.0 → 0.8-1.2
        load_factor = 0.8 + 0.4 * min(1.0, current_load)
        
        # 命中率因子：<0.6 → 增加，>0.8 → 减少
        if hit_rate < 0.6:
            hit_rate_factor = 1.2
        elif hit_rate > 0.8:
            hit_rate_factor = 0.9
        else:
            hit_rate_factor = 1.0
        
        new_capacity = base_capacity * load_factor * hit_rate_factor
        
        # 限制在合理范围 (50%-150%)
        min_capacity = base_capacity * config.cache.capacity_adjust_min_ratio
        max_capacity = base_capacity * config.cache.capacity_adjust_max_ratio
        
        return np.clip(new_capacity, min_capacity, max_capacity)
    
    # 🎯 P3-2优化：缓存预热策略
    def warmup_cache(self, historical_stats: Optional[Dict] = None) -> None:
        """
        使用历史统计数据预热缓存
        
        Args:
            historical_stats: 历史热门内容统计
                {content_id: {'frequency': int, 'avg_size': float, 'heat': float}}
        """
        if not config.cache.enable_cache_warmup:
            return
        
        if not historical_stats:
            # 如果没有历史数据，使用当前热度统计
            historical_stats = {}
            for content_id, heat in self.heat_strategy.historical_heat.items():
                if heat > 0.1:  # 只预热热度超过0.1的内容
                    historical_stats[content_id] = {
                        'heat': heat,
                        'avg_size': 1.0,  # 默认1MB
                        'frequency': len(self.heat_strategy.access_history.get(content_id, []))
                    }
        
        if not historical_stats:
            return
        
        # 按热度排序
        sorted_contents = sorted(historical_stats.items(), 
                               key=lambda x: x[1].get('heat', 0.0), 
                               reverse=True)
        
        preload_budget = self.cache_capacity * config.cache.warmup_capacity_ratio  # 使用30%容量预热
        used_budget = 0.0
        
        for content_id, stats in sorted_contents:
            size = stats.get('avg_size', 1.0)
            if used_budget + size <= preload_budget:
                # 模拟缓存（不实际下载，只记录元数据）
                self.cached_items[content_id] = CachedItem(
                    content_id=content_id,
                    data_size=size,
                    historical_heat=stats.get('heat', 0.0),
                    cache_time=get_simulation_time(),
                    access_count=stats.get('frequency', 1)
                )
                self.current_usage += size
                used_budget += size
    
    def sync_with_neighbors(self, neighbor_cache_states: Dict[str, Set[str]]):
        """与邻居同步缓存状态"""
        # 🔧 修复：使用统一仿真时间  
        current_time = get_simulation_time()
        
        if current_time - self.last_sync_time < self.collaboration_sync_interval:
            return
        
        self.neighbor_cache_states = neighbor_cache_states.copy()
        self.last_sync_time = current_time
        
        # 更新邻居列表
        self.neighbor_nodes = set(neighbor_cache_states.keys())
        
        # 🎯 P2-2优化：定期执行动态容量调整（每次同步时）
        if config.cache.enable_dynamic_capacity:
            stats = self.get_cache_statistics()
            # 计算当前负载和命中率
            current_load = stats['usage_ratio']
            hit_rate = stats['hit_rate']
            
            # 调整容量
            new_capacity = self.adaptive_capacity_allocation(current_load, hit_rate)
            if abs(new_capacity - self.cache_capacity) > self.cache_capacity * 0.05:  # 变化超过5%才调整
                self.cache_capacity = new_capacity
    
    def get_cache_state(self) -> Set[str]:
        """获取当前缓存状态"""
        return set(self.cached_items.keys())
    
    def get_cache_statistics(self) -> Dict:
        """获取缓存统计信息"""
        total_requests = self.cache_stats['total_requests']
        
        return {
            'total_requests': total_requests,
            'cache_hits': self.cache_stats['cache_hits'],
            'cache_misses': self.cache_stats['cache_misses'],
            'hit_rate': self.cache_stats['cache_hits'] / max(1, total_requests),
            'miss_rate': self.cache_stats['cache_misses'] / max(1, total_requests),
            'evictions': self.cache_stats['evictions'],
            'prefetch_hits': self.cache_stats['prefetch_hits'],
            'collaboration_saves': self.cache_stats['collaboration_saves'],
            'current_usage': self.current_usage,
            'usage_ratio': self.current_usage / self.cache_capacity,
            'cached_items_count': len(self.cached_items),
            'avg_item_size': self.current_usage / max(1, len(self.cached_items))
        }
    
    def calculate_cache_reward(self) -> float:
        """
        计算缓存奖励 - 对应论文缓存奖励函数
        """
        stats = self.get_cache_statistics()
        
        # 奖励组件
        hit_rate_reward = self.value_weights['hit_value'] * stats['hit_rate']
        
        # 成本惩罚
        operation_cost = self.cache_stats['evictions'] / max(1, stats['total_requests'])
        cost_penalty = self.value_weights['cost_penalty'] * operation_cost
        
        # 超预算惩罚
        over_budget_penalty = 0.0
        if stats['usage_ratio'] > 1.0:
            over_budget_penalty = self.value_weights['over_budget_penalty'] * (stats['usage_ratio'] - 1.0)
        
        # 能耗考虑 (简化)
        energy_penalty = self.value_weights['energy_penalty'] * stats['usage_ratio']
        
        # 总奖励
        total_reward = (hit_rate_reward - cost_penalty - over_budget_penalty - energy_penalty)
        
        return total_reward