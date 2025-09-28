#!/usr/bin/env python3
"""
增强的缓存仿真实现
提供更真实和高效的缓存仿真机制
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass
import random

@dataclass
class SimulationCacheItem:
    """仿真缓存项"""
    content_id: str
    size: float
    access_count: int = 0
    last_access_time: float = 0.0
    cache_time: float = 0.0
    heat_score: float = 0.0
    content_type: str = "data"  # data, video, app, etc.


class EnhancedCacheSimulator:
    """
    增强的缓存仿真器
    专门为VEC仿真优化的缓存实现
    """
    
    def __init__(self, node_id: str, capacity: float = 1000.0):
        self.node_id = node_id
        self.capacity = capacity  # MB
        self.current_usage = 0.0
        
        # 缓存存储
        self.cached_items: Dict[str, SimulationCacheItem] = {}
        self.access_order = deque()  # LRU tracking
        
        # 🎯 仿真优化的热度计算
        self.content_heat = defaultdict(float)
        self.access_history = defaultdict(lambda: deque(maxlen=20))  # 限制历史长度
        self.content_types = defaultdict(str)
        
        # 缓存策略参数（可由智能体调整）
        self.strategy_params = {
            'heat_threshold': 0.5,      # 热度阈值
            'size_penalty_factor': 0.1, # 大小惩罚因子
            'type_preference': {         # 内容类型偏好
                'critical': 1.0,         # 关键数据
                'video': 0.8,            # 视频内容
                'app': 0.6,              # 应用数据
                'general': 0.4           # 一般数据
            },
            'freshness_decay': 0.95      # 新鲜度衰减因子
        }
        
        # 统计信息
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'bytes_served': 0,
            'hit_rate_history': deque(maxlen=100)
        }
        
        # 仿真时间管理
        self.simulation_start = time.time()
        self.last_cleanup_time = 0
        
    def update_strategy_params(self, params: Dict[str, float]):
        """更新策略参数（智能体控制）"""
        for key, value in params.items():
            if key in self.strategy_params:
                self.strategy_params[key] = np.clip(value, 0.0, 1.0)
    
    def get_simulation_time(self) -> float:
        """获取仿真时间"""
        return time.time() - self.simulation_start
    
    def calculate_content_heat(self, content_id: str) -> float:
        """
        🔥 优化的热度计算 - 适合短期仿真
        """
        current_time = self.get_simulation_time()
        
        # 获取访问历史
        history = self.access_history[content_id]
        if not history:
            return 0.0
        
        # 计算频率热度（最近5分钟内）
        recent_window = 300  # 5分钟窗口
        recent_accesses = sum(1 for t in history if current_time - t < recent_window)
        frequency_heat = min(1.0, recent_accesses / 5.0)  # 5次访问为满分
        
        # 计算时效性热度（指数衰减）
        last_access = history[-1]
        time_since_last = current_time - last_access
        recency_heat = np.exp(-time_since_last / 60.0)  # 1分钟半衰期
        
        # 计算访问密度热度
        if len(history) >= 2:
            time_span = history[-1] - history[0] + 1  # 避免除零
            density_heat = min(1.0, len(history) / time_span * 60)  # 每分钟访问次数
        else:
            density_heat = 0.0
        
        # 综合热度
        combined_heat = (0.4 * frequency_heat + 
                        0.4 * recency_heat + 
                        0.2 * density_heat)
        
        # 考虑内容类型权重
        content_type = self.content_types.get(content_id, 'general')
        type_weight = self.strategy_params['type_preference'].get(content_type, 0.4)
        
        final_heat = combined_heat * type_weight
        self.content_heat[content_id] = final_heat
        
        return final_heat
    
    def should_cache_content(self, content_id: str, content_size: float, 
                           content_type: str = 'general') -> Tuple[bool, str, float]:
        """
        智能缓存决策
        Returns: (should_cache, reason, cache_priority)
        """
        # 记录内容类型
        self.content_types[content_id] = content_type
        
        # 计算热度
        heat = self.calculate_content_heat(content_id)
        
        # 大小惩罚
        size_penalty = self.strategy_params['size_penalty_factor'] * np.log(1 + content_size / 10.0)
        
        # 计算缓存优先级
        cache_priority = heat - size_penalty
        
        # 缓存决策
        heat_threshold = self.strategy_params['heat_threshold']
        available_space = self.capacity - self.current_usage
        
        if cache_priority > heat_threshold and available_space >= content_size:
            return True, f"高优先级缓存 (优先级:{cache_priority:.3f})", cache_priority
        elif cache_priority > heat_threshold * 0.7 and available_space >= content_size * 2:
            return True, f"条件缓存 (优先级:{cache_priority:.3f})", cache_priority
        else:
            return False, f"不缓存 (优先级:{cache_priority:.3f} < {heat_threshold:.3f})", cache_priority
    
    def request_content(self, content_id: str, content_size: float = 1.0, 
                       content_type: str = 'general') -> Tuple[bool, str]:
        """
        处理内容请求
        Returns: (cache_hit, action_taken)
        """
        current_time = self.get_simulation_time()
        self.stats['total_requests'] += 1
        
        # 更新访问历史
        self.access_history[content_id].append(current_time)
        
        # 检查缓存命中
        if content_id in self.cached_items:
            # 缓存命中
            item = self.cached_items[content_id]
            item.access_count += 1
            item.last_access_time = current_time
            
            # 更新LRU顺序
            if content_id in self.access_order:
                self.access_order.remove(content_id)
            self.access_order.append(content_id)
            
            self.stats['cache_hits'] += 1
            self.stats['bytes_served'] += item.size
            
            return True, f"缓存命中 ({item.access_count}次访问)"
        
        # 缓存未命中
        self.stats['cache_misses'] += 1
        
        # 决定是否缓存
        should_cache, reason, priority = self.should_cache_content(content_id, content_size, content_type)
        
        if should_cache:
            # 执行缓存
            success = self._add_to_cache(content_id, content_size, content_type, priority)
            if success:
                return False, f"缓存未命中，已缓存 - {reason}"
            else:
                return False, f"缓存未命中，缓存失败 - 容量不足"
        else:
            return False, f"缓存未命中，不缓存 - {reason}"
    
    def _add_to_cache(self, content_id: str, size: float, content_type: str, priority: float) -> bool:
        """添加内容到缓存"""
        # 检查容量
        if self.current_usage + size > self.capacity:
            # 需要腾出空间
            if not self._make_space(size):
                return False
        
        # 创建缓存项
        current_time = self.get_simulation_time()
        item = SimulationCacheItem(
            content_id=content_id,
            size=size,
            cache_time=current_time,
            last_access_time=current_time,
            heat_score=priority,
            content_type=content_type
        )
        
        self.cached_items[content_id] = item
        self.current_usage += size
        self.access_order.append(content_id)
        
        return True
    
    def _make_space(self, required_space: float) -> bool:
        """腾出缓存空间"""
        if not self.cached_items:
            return False
        
        # 计算所有项目的替换优先级（越低越容易被替换）
        replacement_candidates = []
        current_time = self.get_simulation_time()
        
        for content_id, item in self.cached_items.items():
            # 计算替换分数（越低越容易被替换）
            age_factor = current_time - item.last_access_time  # 时间因子
            heat_factor = 1.0 / max(0.1, self.calculate_content_heat(content_id))  # 热度因子
            size_factor = item.size / 10.0  # 大小因子
            
            replacement_score = age_factor * heat_factor + size_factor
            replacement_candidates.append((replacement_score, content_id, item))
        
        # 按替换分数排序
        replacement_candidates.sort(key=lambda x: x[0], reverse=True)
        
        # 执行替换
        freed_space = 0.0
        for score, content_id, item in replacement_candidates:
            if freed_space >= required_space:
                break
            
            self._evict_item(content_id)
            freed_space += item.size
        
        return freed_space >= required_space
    
    def _evict_item(self, content_id: str):
        """从缓存中驱逐项目"""
        if content_id in self.cached_items:
            item = self.cached_items.pop(content_id)
            self.current_usage -= item.size
            self.stats['evictions'] += 1
            
            if content_id in self.access_order:
                self.access_order.remove(content_id)
    
    def periodic_cleanup(self):
        """定期清理和统计更新"""
        current_time = self.get_simulation_time()
        
        # 每30秒执行一次清理
        if current_time - self.last_cleanup_time < 30:
            return
        
        self.last_cleanup_time = current_time
        
        # 更新统计
        if self.stats['total_requests'] > 0:
            hit_rate = self.stats['cache_hits'] / self.stats['total_requests']
            self.stats['hit_rate_history'].append(hit_rate)
        
        # 清理过期的访问历史
        cutoff_time = current_time - 600  # 保留10分钟内的历史
        for content_id in list(self.access_history.keys()):
            history = self.access_history[content_id]
            # 移除过期访问记录
            while history and history[0] < cutoff_time:
                history.popleft()
            
            # 如果访问历史为空，移除该内容的记录
            if not history:
                del self.access_history[content_id]
                if content_id in self.content_heat:
                    del self.content_heat[content_id]
    
    def get_cache_statistics(self) -> Dict:
        """获取缓存统计信息"""
        total_requests = self.stats['total_requests']
        
        return {
            'node_id': self.node_id,
            'total_requests': total_requests,
            'cache_hits': self.stats['cache_hits'],
            'cache_misses': self.stats['cache_misses'],
            'hit_rate': self.stats['cache_hits'] / max(1, total_requests),
            'miss_rate': self.stats['cache_misses'] / max(1, total_requests),
            'evictions': self.stats['evictions'],
            'bytes_served': self.stats['bytes_served'],
            'current_usage': self.current_usage,
            'usage_ratio': self.current_usage / self.capacity,
            'cached_items_count': len(self.cached_items),
            'avg_item_size': self.current_usage / max(1, len(self.cached_items)),
            'heat_scores': dict(self.content_heat),
            'strategy_params': dict(self.strategy_params)
        }
    
    def reset_stats(self):
        """重置统计信息（用于新episode）"""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'bytes_served': 0,
            'hit_rate_history': deque(maxlen=100)
        }
        self.simulation_start = time.time()
        self.last_cleanup_time = 0


def create_enhanced_cache_simulator(node_type: str, node_id: str) -> EnhancedCacheSimulator:
    """
    根据节点类型创建缓存仿真器
    """
    capacity_map = {
        'vehicle': 100.0,   # 100MB
        'rsu': 1000.0,     # 1GB  
        'uav': 200.0       # 200MB
    }
    
    capacity = capacity_map.get(node_type, 100.0)
    return EnhancedCacheSimulator(node_id, capacity)


# 测试函数
def test_enhanced_cache_simulator():
    """测试增强缓存仿真器"""
    print("🧪 测试增强缓存仿真器...")
    
    cache = EnhancedCacheSimulator("test_rsu", 100.0)
    
    # 模拟内容请求
    contents = ['video1', 'data1', 'video2', 'app1', 'data2']
    types = ['video', 'critical', 'video', 'app', 'general']
    
    for i in range(20):
        content_id = random.choice(contents)
        content_type = types[contents.index(content_id)]
        content_size = random.uniform(1.0, 10.0)
        
        hit, action = cache.request_content(content_id, content_size, content_type)
        print(f"请求 {content_id}: {'命中' if hit else '未命中'} - {action}")
        
        time.sleep(0.1)  # 模拟时间流逝
    
    # 输出统计信息
    stats = cache.get_cache_statistics()
    print(f"\n📊 缓存统计:")
    print(f"命中率: {stats['hit_rate']:.2%}")
    print(f"使用率: {stats['usage_ratio']:.2%}")
    print(f"缓存项目数: {stats['cached_items_count']}")
    
    print("✅ 测试完成")


if __name__ == "__main__":
    test_enhanced_cache_simulator()
