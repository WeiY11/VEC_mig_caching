"""
缓存策略基类
定义缓存策略的通用接口
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, List

class CachePolicy(ABC):
    """缓存策略抽象基类"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.access_count = {}
        self.access_time = {}
    
    @abstractmethod
    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        pass
    
    @abstractmethod
    def put(self, key: str, value: Any) -> None:
        """存储缓存项"""
        pass
    
    @abstractmethod
    def evict(self) -> Optional[str]:
        """驱逐缓存项"""
        pass
    
    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)
    
    def is_full(self) -> bool:
        """检查缓存是否已满"""
        return self.size() >= self.capacity
    
    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_count.clear()
        self.access_time.clear()
    
    def keys(self) -> List[str]:
        """获取所有缓存键"""
        return list(self.cache.keys())
    
    def hit_rate(self, total_requests: int) -> float:
        """计算缓存命中率"""
        if total_requests == 0:
            return 0.0
        hits = sum(self.access_count.values())
        return hits / total_requests