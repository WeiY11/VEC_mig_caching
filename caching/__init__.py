"""
缓存模块
实现各种缓存策略和管理机制
"""

from .cache_policy import CachePolicy
from .lru_cache import LRUCache
from .popularity_cache import PopularityBasedCache

__all__ = ['CachePolicy', 'LRUCache', 'PopularityBasedCache']