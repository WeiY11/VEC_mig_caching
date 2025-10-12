"""
增强版系统仿真器 - 集成高级缓存功能
包含分层缓存、LSTM预测和RSU协作
"""
import numpy as np
import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import torch

# 导入原始系统仿真器
from .system_simulator import CompleteSystemSimulator

# 导入高级缓存组件
from caching.collaborative_cache_system import CollaborativeCacheSystem
from caching.cache_manager import HeatBasedCacheStrategy

# 导入配置
from config import config


class EnhancedSystemSimulator(CompleteSystemSimulator):
    """
    增强版系统仿真器
    在原有基础上集成高级缓存功能
    """
    
    def __init__(self, scenario: Dict[str, int] = None):
        """
        初始化增强版仿真器
        
        Args:
            scenario: 场景配置
        """
        # 调用父类初始化
        super().__init__(scenario)
        
        # 替换缓存管理器为协作缓存系统
        self.use_advanced_cache = True
        
        if self.use_advanced_cache:
            print("[EnhancedSimulator] Initializing advanced caching system...")
            
            # 创建协作缓存系统
            self.collaborative_cache = CollaborativeCacheSystem(
                num_rsus=self.num_rsus,
                enable_prediction=True  # 启用LSTM预测
            )
            
            # 创建热度策略（使用优化的HeatBasedCacheStrategy，支持自适应时间槽）
            self.heat_strategy = HeatBasedCacheStrategy(
                slot_duration=None,   # 自适应时间槽（根据访问密度调整5-30秒）
                total_slots=200       # 总共200个时间槽
            )
            print("[EnhancedSimulator] Using adaptive HeatBasedCacheStrategy")
            
            # 预热缓存系统
            self._warmup_cache_system()
            
            print("[EnhancedSimulator] Advanced caching initialized")
        
        # 缓存性能统计
        self.cache_performance = {
            'hits': 0,
            'misses': 0,
            'predictions_made': 0,
            'prediction_hits': 0,
            'collaborative_hits': 0,
            'lstm_training_count': 0
        }
        
        # 内容访问历史（用于训练LSTM）
        self.content_access_history = []
        self.content_metadata = {}
        
    def _warmup_cache_system(self):
        """
        预热缓存系统
        生成一些初始数据让LSTM有训练样本
        """
        # 生成100个预热请求（让统计预测器建立初始模式）
        for i in range(100):
            content_id = f"warmup_content_{i % 20}"
            
            # 模拟Zipf分布的访问模式
            if np.random.random() < 0.7:  # 70%概率访问热门内容
                content_id = f"warmup_content_{np.random.choice([0, 1, 2, 3, 4])}"
            
            # 记录到热度策略
            self.heat_strategy.update_heat(content_id, access_weight=1.0)
            
            # 添加到协作缓存
            rsu_id = f"RSU_{i % self.num_rsus}"
            self.collaborative_cache.put_content(
                rsu_id,
                content_id,
                {'content': f"data_{content_id}", 'size': np.random.uniform(1e6, 10e6)},
                np.random.uniform(1e6, 10e6)
            )
    
    def process_task_with_cache(self, task: Any, node_id: str) -> Dict[str, float]:
        """
        处理任务时考虑高级缓存
        
        Args:
            task: 任务对象
            node_id: 处理节点ID
            
        Returns:
            处理结果指标
        """
        # 生成内容ID（基于任务特征，增加到5000种以减少哈希冲突）
        content_id = f"task_content_{hash(str(task.task_id)) % 5000}"
        
        # 记录访问到热度策略
        self.heat_strategy.update_heat(content_id, access_weight=1.0)
        
        # 尝试从缓存获取
        cache_hit = False
        cache_source = 'miss'
        
        if self.use_advanced_cache:
            # 使用协作缓存系统
            cached_content = self.collaborative_cache.get_content(node_id, content_id)
            
            if cached_content is not None:
                cache_hit = True
                self.cache_performance['hits'] += 1
                
                # 判断命中来源
                if node_id in self.collaborative_cache.cache_managers:
                    cache_manager = self.collaborative_cache.cache_managers[node_id]
                    if content_id in cache_manager.l1_cache:
                        cache_source = 'L1'
                    elif content_id in cache_manager.l2_cache:
                        cache_source = 'L2'
                    else:
                        cache_source = 'remote'
                        self.cache_performance['collaborative_hits'] += 1
            else:
                # 缓存未命中，需要处理并缓存
                self.cache_performance['misses'] += 1
                
                # 处理任务后缓存结果
                content = {
                    'content': f"processed_task_{task.task_id}",
                    'size': task.data_size,
                    'task_id': task.task_id
                }
                
                # 使用LSTM预测决定是否缓存
                should_cache = self._should_cache_content(content_id, task.data_size)
                
                if should_cache:
                    self.collaborative_cache.put_content(
                        node_id,
                        content_id,
                        content,
                        task.data_size,
                        {'timestamp': current_time, 'predicted_hot': should_cache}
                    )
        
        # 计算处理时延（缓存命中时延更低）
        if cache_hit:
            # 缓存命中，时延大幅降低
            if cache_source == 'L1':
                delay_reduction = 0.8  # L1命中，减少80%时延
            elif cache_source == 'L2':
                delay_reduction = 0.6  # L2命中，减少60%时延
            else:
                delay_reduction = 0.4  # 远程命中，减少40%时延
        else:
            delay_reduction = 0.0  # 未命中，无减少
        
        # 每100个任务触发一次维护
        if hasattr(self, 'processed_tasks'):
            self.processed_tasks += 1
            if self.processed_tasks % 100 == 0:
                self._periodic_cache_maintenance()
        else:
            self.processed_tasks = 1
        
        return {
            'cache_hit': cache_hit,
            'cache_source': cache_source,
            'delay_reduction': delay_reduction,
            'content_id': content_id
        }
    
    def _should_cache_content(self, content_id: str, size: float) -> bool:
        """
        使用热度策略决定是否缓存内容
        
        Args:
            content_id: 内容ID
            size: 内容大小
            
        Returns:
            是否应该缓存
        """
        # 如果内容太大，不缓存
        if size > 10e9:  # 10GB
            return False
        
        # 使用HeatBasedCacheStrategy计算优先级
        total_contents = len(self.heat_strategy.historical_heat) + 1
        cache_priority = self.heat_strategy.get_cache_priority(content_id, size, total_contents)
        
        self.cache_performance['predictions_made'] += 1
        
        # 基于优先级决定（阈值0.3，经验值）
        threshold = 0.3
        should_cache = cache_priority > threshold
        
        if should_cache:
            self.cache_performance['prediction_hits'] += 1
        
        return should_cache
    
    def _periodic_cache_maintenance(self):
        """
        定期缓存维护
        """
        # 触发协作缓存的维护
        if self.use_advanced_cache:
            self.collaborative_cache.periodic_maintenance()
        
        # 定期清理过期数据（每1000个任务）
        if self.processed_tasks % 1000 == 0:
            from utils.unified_time_manager import get_simulation_time
            current_time = get_simulation_time()
            self.heat_strategy.cleanup_stale_data(current_time, staleness_threshold=7200)
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计数据
        """
        stats = {
            'basic_performance': self.cache_performance,
        }
        
        if self.use_advanced_cache:
            # 获取协作缓存统计
            collab_stats = self.collaborative_cache.get_system_stats()
            stats['collaborative'] = collab_stats
            
            # 计算整体指标
            total_requests = self.cache_performance['hits'] + self.cache_performance['misses']
            if total_requests > 0:
                stats['overall_hit_rate'] = self.cache_performance['hits'] / total_requests
                stats['collaborative_contribution'] = self.cache_performance['collaborative_hits'] / total_requests
                stats['prediction_accuracy'] = (
                    self.cache_performance['prediction_hits'] / self.cache_performance['predictions_made']
                    if self.cache_performance['predictions_made'] > 0 else 0
                )
            
            # 热度策略状态
            stats['heat_strategy_status'] = {
                'slot_duration': self.heat_strategy.slot_duration,
                'total_slots': self.heat_strategy.total_slots,
                'unique_contents': len(self.heat_strategy.historical_heat),
                'total_accesses': sum(len(h) for h in self.heat_strategy.access_history.values()),
                'adaptive_enabled': self.heat_strategy.adaptive_slot
            }
        
        return stats
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        重写step方法，集成缓存处理
        
        Args:
            action: 动作向量
            
        Returns:
            下一状态、奖励、是否结束、信息字典
        """
        # 调用父类step
        next_state, reward, done, info = super().step(action)
        
        # 处理缓存相关逻辑
        if hasattr(self, 'current_tasks') and self.current_tasks:
            for task in self.current_tasks[:5]:  # 处理前5个任务
                # 决定处理节点
                node_id = f"RSU_{np.random.randint(0, self.num_rsus)}"
                
                # 处理任务并更新缓存
                cache_result = self.process_task_with_cache(task, node_id)
                
                # 根据缓存结果调整奖励
                if cache_result['cache_hit']:
                    # 缓存命中，额外奖励
                    cache_bonus = 0.1 * cache_result['delay_reduction']
                    reward += cache_bonus
                    
                    # 更新info
                    if 'cache_bonus' not in info:
                        info['cache_bonus'] = 0
                    info['cache_bonus'] += cache_bonus
        
        # 添加缓存统计到info
        info['cache_stats'] = self.get_cache_statistics()
        
        return next_state, reward, done, info
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            初始状态
        """
        # 调用父类reset
        initial_state = super().reset()
        
        # 重置缓存统计
        self.cache_performance = {
            'hits': 0,
            'misses': 0,
            'predictions_made': 0,
            'prediction_hits': 0,
            'collaborative_hits': 0,
            'lstm_training_count': self.cache_performance.get('lstm_training_count', 0)
        }
        
        # 不重置LSTM模型（保留学习成果）
        # 但清理过旧的访问历史
        if len(self.content_access_history) > 10000:
            self.content_access_history = self.content_access_history[-5000:]
        
        return initial_state
    
    def close(self):
        """
        关闭仿真器
        """
        # 关闭协作缓存系统
        if self.use_advanced_cache:
            self.collaborative_cache.shutdown()
            
            # 打印热度策略统计
            print(f"[EnhancedSimulator] Heat strategy: {len(self.heat_strategy.historical_heat)} unique contents")
            print(f"                     Adaptive slot duration: {self.heat_strategy.slot_duration:.1f}s")
        
        # 调用父类close
        if hasattr(super(), 'close'):
            super().close()


def create_enhanced_environment(scenario: Dict[str, int] = None) -> EnhancedSystemSimulator:
    """
    创建增强版环境
    
    Args:
        scenario: 场景配置
        
    Returns:
        增强版仿真器实例
    """
    return EnhancedSystemSimulator(scenario)
