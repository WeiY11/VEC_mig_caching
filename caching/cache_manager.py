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
    
    def __init__(self):
        # 热度参数
        self.decay_factor = 0.9           # ρ 衰减因子
        self.heat_mix_factor = 0.7        # η 热度混合系数
        self.zipf_exponent = 0.8          # Zipf分布参数
        
        # 热度统计
        self.historical_heat: Dict[str, float] = defaultdict(float)
        self.slot_heat: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.current_slot = 0
        self.total_slots = 24  # 一天24个时间槽
        
        # 访问统计
        self.access_history: Dict[str, List[float]] = defaultdict(list)
        self.content_popularity_rank: Dict[str, int] = {}
        
        # 移动平均计算器
        self.avg_heat = ExponentialMovingAverage(alpha=0.1)
    
    def update_heat(self, content_id: str, access_weight: float = 1.0):
        """
        更新内容热度 - 对应论文式(35)-(36)
        """
        # 更新历史热度 - 式(35)
        self.historical_heat[content_id] = (self.decay_factor * self.historical_heat[content_id] + 
                                           access_weight)
        
        # 更新时间槽热度 - 式(36)
        current_slot = int(time.time() / 3600) % self.total_slots  # 小时级时间槽
        self.slot_heat[content_id][current_slot] += access_weight
        
        # 记录访问历史
        self.access_history[content_id].append(time.time())
        
        # 限制历史长度
        if len(self.access_history[content_id]) > 100:
            self.access_history[content_id].pop(0)
    
    def calculate_combined_heat(self, content_id: str) -> float:
        """
        计算综合热度 - 对应论文式(37)
        Heat(c) = η * H_hist(c) + (1-η) * H_slot(c,t)
        """
        hist_heat = self.historical_heat.get(content_id, 0.0)
        
        current_slot = int(time.time() / 3600) % self.total_slots
        slot_heat = self.slot_heat[content_id].get(current_slot, 0.0)
        
        combined_heat = (self.heat_mix_factor * hist_heat + 
                        (1 - self.heat_mix_factor) * slot_heat)
        
        return combined_heat
    
    def calculate_zipf_popularity(self, content_id: str, total_contents: int) -> float:
        """计算Zipf流行度"""
        if content_id not in self.content_popularity_rank:
            # 根据访问次数排名
            access_counts = {cid: len(history) for cid, history in self.access_history.items()}
            sorted_contents = sorted(access_counts.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (cid, _) in enumerate(sorted_contents, 1):
                self.content_popularity_rank[cid] = rank
        
        rank = self.content_popularity_rank.get(content_id, total_contents)
        return calculate_zipf_probability(rank, total_contents, self.zipf_exponent)
    
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
            time_since_access = time.time() - last_access
            recency_bonus = max(0, 1.0 - time_since_access / 3600)  # 1小时内的奖励
        
        # 综合优先级
        priority = (0.4 * heat + 0.3 * zipf_pop + 0.2 * recency_bonus - 0.1 * size_penalty)
        
        return max(0.0, priority)


class CollaborativeCacheManager:
    """
    协作缓存管理器
    实现邻居协作和背包优化算法
    """
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.cache_capacity = config.cache.rsu_cache_capacity
        
        # 缓存存储
        self.cached_items: Dict[str, CachedItem] = {}
        self.current_usage = 0.0
        
        # 替换策略
        policy_name = config.cache.cache_replacement_policy.lower()
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
        
        # 预取参数
        self.prefetch_window_ratio = 0.1  # 预取窗口占总容量10%
        self.prefetch_threshold = 0.6     # 预取阈值
        
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
            item.last_access_time = time.time()
            
            self.cache_stats['cache_hits'] += 1
            
            # 更新LRU顺序 (如果使用LRU策略)
            if self.replacement_policy == CacheReplacementPolicy.LRU:
                # 重新插入以更新顺序
                self.cached_items[content_id] = self.cached_items.pop(content_id)
    
    def _check_neighbor_collaboration(self, content_id: str) -> bool:
        """检查邻居协作缓存"""
        for neighbor_id, cached_contents in self.neighbor_cache_states.items():
            if content_id in cached_contents:
                # 邻居有此内容，可以协作获取
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
        
        # 定义阈值
        high_heat_threshold = 0.8
        medium_heat_threshold = 0.4
        capacity_threshold = self.cache_capacity * 0.1  # 10%容量阈值
        
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
            cache_time=time.time(),
            last_access_time=time.time()
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
        """根据替换策略腾出空间"""
        if self.replacement_policy == CacheReplacementPolicy.LRU:
            return self._lru_eviction(required_space)
        elif self.replacement_policy == CacheReplacementPolicy.LFU:
            return self._lfu_eviction(required_space)
        elif self.replacement_policy == CacheReplacementPolicy.FIFO:
            return self._fifo_eviction(required_space)
        else:  # HYBRID
            return self._hybrid_eviction(required_space)
    
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
        """混合替换策略"""
        # 综合考虑访问频率、最近性和缓存价值
        scored_items = []
        
        for content_id, item in self.cached_items.items():
            # 计算综合分数 (分数越低越容易被替换)
            recency_score = (time.time() - item.last_access_time) / 3600  # 小时
            frequency_score = 1.0 / max(1, item.access_count)
            value_score = 1.0 / max(0.1, item.cache_value)
            
            total_score = 0.4 * recency_score + 0.3 * frequency_score + 0.3 * value_score
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
    
    def _evict_item(self, content_id: str):
        """从缓存中移除项目"""
        if content_id in self.cached_items:
            item = self.cached_items.pop(content_id)
            self.current_usage -= item.data_size
            self.cache_stats['evictions'] += 1
    
    def sync_with_neighbors(self, neighbor_cache_states: Dict[str, Set[str]]):
        """与邻居同步缓存状态"""
        current_time = time.time()
        
        if current_time - self.last_sync_time < self.collaboration_sync_interval:
            return
        
        self.neighbor_cache_states = neighbor_cache_states.copy()
        self.last_sync_time = current_time
        
        # 更新邻居列表
        self.neighbor_nodes = set(neighbor_cache_states.keys())
    
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