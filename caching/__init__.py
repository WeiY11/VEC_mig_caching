"""
缓存模块
实现各种缓存策略和管理机制
"""

from .cache_policy import CachePolicy
from .cache_manager import CollaborativeCacheManager, HeatBasedCacheStrategy

# 导出核心组件
__all__ = ['CachePolicy', 'CollaborativeCacheManager', 'HeatBasedCacheStrategy']