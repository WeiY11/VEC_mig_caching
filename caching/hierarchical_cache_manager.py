"""
分层缓存管理器 - L1热点缓存 + L2常规缓存
实现高效的两级缓存架构，提升缓存命中率
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from collections import OrderedDict, defaultdict
import time
import hashlib

class HierarchicalCacheManager:
    """
    分层缓存管理器
    - L1: 热点缓存（小容量，高速访问）
    - L2: 常规缓存（大容量，普通访问）
    """
    
    def __init__(self, node_id: str, l1_capacity: float = 2e9, l2_capacity: float = 8e9):
        """
        初始化分层缓存
        
        Args:
            node_id: 节点标识（如 RSU_0）
            l1_capacity: L1缓存容量（默认2GB）
            l2_capacity: L2缓存容量（默认8GB）
        """
        self.node_id = node_id
        
        # L1热点缓存 - 使用OrderedDict实现LRU
        self.l1_cache = OrderedDict()
        self.l1_capacity = l1_capacity
        self.l1_size = 0
        
        # L2常规缓存 - 使用OrderedDict实现LFU
        self.l2_cache = OrderedDict()
        self.l2_capacity = l2_capacity
        self.l2_size = 0
        
        # 访问频率统计（用于LFU）
        self.access_frequency = defaultdict(int)
        self.access_history = []
        
        # 性能统计
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'evictions': 0,
            'promotions': 0,  # L2提升到L1
            'demotions': 0,   # L1降级到L2
        }
        
        # 热度阈值（访问次数超过此值提升到L1）
        self.heat_threshold = 3
        self.time_window = 100  # 时间窗口内的访问统计
        
    def get(self, content_id: str) -> Optional[Dict[str, Any]]:
        """
        获取缓存内容（先查L1，再查L2）
        
        Args:
            content_id: 内容标识
            
        Returns:
            缓存的内容数据，如果未命中返回None
        """
        # 先查L1缓存
        if content_id in self.l1_cache:
            # LRU更新：移到末尾
            self.l1_cache.move_to_end(content_id)
            self.stats['l1_hits'] += 1
            self.access_frequency[content_id] += 1
            self._record_access(content_id, 'L1')
            return self.l1_cache[content_id]
        
        # 再查L2缓存
        if content_id in self.l2_cache:
            self.stats['l2_hits'] += 1
            self.access_frequency[content_id] += 1
            self._record_access(content_id, 'L2')
            
            # 检查是否需要提升到L1
            if self.access_frequency[content_id] >= self.heat_threshold:
                self._promote_to_l1(content_id)
            
            return self.l2_cache[content_id]
        
        # 缓存未命中
        self.stats['misses'] += 1
        return None
    
    def put(self, content_id: str, content: Dict[str, Any], size: float, 
            is_hot: bool = False) -> bool:
        """
        添加内容到缓存
        
        Args:
            content_id: 内容标识
            content: 内容数据
            size: 内容大小（字节）
            is_hot: 是否标记为热点内容（直接放入L1）
            
        Returns:
            是否成功缓存
        """
        # 如果内容太大，无法缓存
        if size > self.l2_capacity:
            return False
        
        # 如果已存在，更新
        if content_id in self.l1_cache or content_id in self.l2_cache:
            return self._update_content(content_id, content, size)
        
        # 决定放入哪一层
        if is_hot or (size <= self.l1_capacity * 0.1):  # 热点或小文件放L1
            return self._put_to_l1(content_id, content, size)
        else:
            return self._put_to_l2(content_id, content, size)
    
    def _put_to_l1(self, content_id: str, content: Dict[str, Any], size: float) -> bool:
        """添加到L1缓存"""
        # 如果需要，先腾出空间
        while self.l1_size + size > self.l1_capacity and self.l1_cache:
            self._evict_from_l1()
        
        # 如果还是放不下，尝试放L2
        if self.l1_size + size > self.l1_capacity:
            return self._put_to_l2(content_id, content, size)
        
        # 添加到L1
        self.l1_cache[content_id] = {
            'content': content,
            'size': size,
            'timestamp': time.time(),
            'layer': 'L1'
        }
        self.l1_size += size
        self.access_frequency[content_id] = 1
        return True
    
    def _put_to_l2(self, content_id: str, content: Dict[str, Any], size: float) -> bool:
        """添加到L2缓存"""
        # 如果需要，先腾出空间
        while self.l2_size + size > self.l2_capacity and self.l2_cache:
            self._evict_from_l2()
        
        # 如果还是放不下，放弃
        if self.l2_size + size > self.l2_capacity:
            return False
        
        # 添加到L2
        self.l2_cache[content_id] = {
            'content': content,
            'size': size,
            'timestamp': time.time(),
            'layer': 'L2'
        }
        self.l2_size += size
        self.access_frequency[content_id] = 1
        return True
    
    def _promote_to_l1(self, content_id: str):
        """
        🎯 P2-1优化：智能提升策略
        将内容从 L2 提升到 L1，基于多维度评估
        """
        if content_id not in self.l2_cache:
            return
        
        item = self.l2_cache[content_id]
        size = item['size']
        
        # 🔥 智能提升决策：综合评估是否值得提升
        if not self._should_promote(content_id, size):
            return
        
        # 如果L1空间不足，先降级一些内容到L2
        while self.l1_size + size > self.l1_capacity and self.l1_cache:
            self._demote_from_l1()
        
        # 移动到L1
        self.l1_cache[content_id] = item
        self.l1_cache[content_id]['layer'] = 'L1'
        self.l1_size += size
        
        del self.l2_cache[content_id]
        self.l2_size -= size
        
        self.stats['promotions'] += 1
    
    # 🎯 P2-1优化：智能提升决策
    def _should_promote(self, content_id: str, size: float) -> bool:
        """
        评估内容是否应该从 L2 提升到 L1
        
        考虑因素：
        1. 访问频率：高频访问优先提升
        2. 内容大小：小文件更适合 L1
        3. 访问趋势：近期访问增长
        4. L1 使用率：避免过度拥堵
        
        Returns:
            是否应该提升
        """
        # 计算 L1 使用率
        l1_usage_ratio = self.l1_size / self.l1_capacity
        
        # 1. 频率分数
        freq = self.access_frequency.get(content_id, 0)
        freq_score = min(1.0, freq / (self.heat_threshold * 2))
        
        # 2. 大小分数（越小越好）
        size_ratio = size / (self.l1_capacity * 0.2)  # L1单个内容不超过20%
        size_score = 1.0 - min(1.0, size_ratio)
        
        # 3. 趋势分数：近期访问增长
        recent_accesses = len([h for h in self.access_history[-20:] if h['content_id'] == content_id])
        trend_score = min(1.0, recent_accesses / 5)
        
        # 综合评分：频率40% + 大小30% + 趋势30%
        total_score = 0.4 * freq_score + 0.3 * size_score + 0.3 * trend_score
        
        # 决策阈值：根据 L1 使用率动态调整
        if l1_usage_ratio < 0.5:
            threshold = 0.5  # L1 空闲，降低阈值
        elif l1_usage_ratio < 0.8:
            threshold = 0.65  # 正常使用
        else:
            threshold = 0.8   # L1 拥堵，提高阈值
        
        return total_score >= threshold
    
    def _demote_from_l1(self):
        """将L1中最冷的内容降级到L2"""
        if not self.l1_cache:
            return
        
        # 找出访问频率最低的内容
        min_freq = float('inf')
        coldest_id = None
        for cid in self.l1_cache:
            if self.access_frequency[cid] < min_freq:
                min_freq = self.access_frequency[cid]
                coldest_id = cid
        
        if coldest_id:
            item = self.l1_cache[coldest_id]
            size = item['size']
            
            # 确保L2有空间
            while self.l2_size + size > self.l2_capacity and self.l2_cache:
                self._evict_from_l2()
            
            # 移动到L2
            self.l2_cache[coldest_id] = item
            self.l2_cache[coldest_id]['layer'] = 'L2'
            self.l2_size += size
            
            del self.l1_cache[coldest_id]
            self.l1_size -= size
            
            self.stats['demotions'] += 1
    
    def _evict_from_l1(self):
        """从L1驱逐（使用LRU策略）"""
        if not self.l1_cache:
            return
        
        # 驱逐最早的（最少最近使用）
        content_id, item = self.l1_cache.popitem(last=False)
        self.l1_size -= item['size']
        
        # 尝试降级到L2而不是完全驱逐
        if self._put_to_l2(content_id, item['content'], item['size']):
            self.stats['demotions'] += 1
        else:
            self.stats['evictions'] += 1
            del self.access_frequency[content_id]
    
    def _evict_from_l2(self):
        """从L2驱逐（使用LFU策略）"""
        if not self.l2_cache:
            return
        
        # 找出访问频率最低的内容
        min_freq = float('inf')
        evict_id = None
        for cid in self.l2_cache:
            if self.access_frequency[cid] < min_freq:
                min_freq = self.access_frequency[cid]
                evict_id = cid
        
        if evict_id:
            item = self.l2_cache[evict_id]
            del self.l2_cache[evict_id]
            self.l2_size -= item['size']
            del self.access_frequency[evict_id]
            self.stats['evictions'] += 1
    
    def _update_content(self, content_id: str, content: Dict[str, Any], size: float) -> bool:
        """更新已存在的内容"""
        if content_id in self.l1_cache:
            old_size = self.l1_cache[content_id]['size']
            self.l1_size = self.l1_size - old_size + size
            self.l1_cache[content_id] = {
                'content': content,
                'size': size,
                'timestamp': time.time(),
                'layer': 'L1'
            }
            return True
        elif content_id in self.l2_cache:
            old_size = self.l2_cache[content_id]['size']
            self.l2_size = self.l2_size - old_size + size
            self.l2_cache[content_id] = {
                'content': content,
                'size': size,
                'timestamp': time.time(),
                'layer': 'L2'
            }
            return True
        return False
    
    def _record_access(self, content_id: str, layer: str):
        """记录访问历史"""
        self.access_history.append({
            'content_id': content_id,
            'layer': layer,
            'timestamp': time.time()
        })
        
        # 维护窗口大小
        if len(self.access_history) > self.time_window:
            self.access_history.pop(0)
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        total_hits = self.stats['l1_hits'] + self.stats['l2_hits']
        total_requests = total_hits + self.stats['misses']
        
        return {
            'node_id': self.node_id,
            'l1_utilization': self.l1_size / self.l1_capacity if self.l1_capacity > 0 else 0,
            'l2_utilization': self.l2_size / self.l2_capacity if self.l2_capacity > 0 else 0,
            'l1_hit_rate': self.stats['l1_hits'] / total_requests if total_requests > 0 else 0,
            'l2_hit_rate': self.stats['l2_hits'] / total_requests if total_requests > 0 else 0,
            'total_hit_rate': total_hits / total_requests if total_requests > 0 else 0,
            'promotions': self.stats['promotions'],
            'demotions': self.stats['demotions'],
            'evictions': self.stats['evictions'],
            'l1_items': len(self.l1_cache),
            'l2_items': len(self.l2_cache),
        }
    
    def clear_stats(self):
        """清空统计信息"""
        self.stats = {
            'l1_hits': 0,
            'l2_hits': 0,
            'misses': 0,
            'evictions': 0,
            'promotions': 0,
            'demotions': 0,
        }
        self.access_history = []
    
    def get_hot_contents(self, top_n: int = 10) -> List[Tuple[str, int]]:
        """获取最热门的内容"""
        sorted_contents = sorted(
            self.access_frequency.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        return sorted_contents[:top_n]
