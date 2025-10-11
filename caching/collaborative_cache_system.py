"""
协作缓存系统 - RSU间共享缓存信息
实现RSU节点间的缓存协作，提高整体缓存命中率
"""
import numpy as np
from typing import Dict, List, Optional, Tuple, Set, Any
from collections import defaultdict, deque
import time
import threading
import queue
from dataclasses import dataclass
from enum import Enum

# 导入分层缓存管理器
from .hierarchical_cache_manager import HierarchicalCacheManager
from .lstm_popularity_predictor import LSTMPopularityPredictor


class MessageType(Enum):
    """消息类型枚举"""
    CACHE_QUERY = "cache_query"          # 缓存查询
    CACHE_RESPONSE = "cache_response"    # 缓存响应
    CACHE_UPDATE = "cache_update"        # 缓存更新通知
    POPULARITY_SHARE = "popularity_share" # 流行度信息共享
    PREFETCH_REQUEST = "prefetch_request" # 预取请求
    LOAD_BALANCE = "load_balance"        # 负载均衡请求


@dataclass
class CacheMessage:
    """缓存消息数据类"""
    msg_type: MessageType
    sender_id: str
    receiver_id: str
    content_id: str
    data: Dict[str, Any]
    timestamp: float


class CollaborativeCacheSystem:
    """
    协作缓存系统
    管理多个RSU节点的缓存协作
    """
    
    def __init__(self, num_rsus: int = 4, enable_prediction: bool = True):
        """
        初始化协作缓存系统
        
        Args:
            num_rsus: RSU节点数量
            enable_prediction: 是否启用流行度预测
        """
        self.num_rsus = num_rsus
        self.enable_prediction = enable_prediction
        
        # 为每个RSU创建分层缓存管理器
        self.cache_managers = {}
        for i in range(num_rsus):
            node_id = f"RSU_{i}"
            # RSU的L1和L2缓存容量
            self.cache_managers[node_id] = HierarchicalCacheManager(
                node_id=node_id,
                l1_capacity=3e9,  # 3GB L1
                l2_capacity=7e9   # 7GB L2
            )
        
        # 流行度预测器（全局共享）
        self.popularity_predictor = None
        if enable_prediction:
            self.popularity_predictor = LSTMPopularityPredictor()
        
        # 缓存目录（记录哪个RSU有哪些内容）
        self.cache_directory = defaultdict(set)  # content_id -> set of RSU IDs
        
        # 节点负载信息
        self.node_loads = defaultdict(float)
        
        # 消息队列（模拟RSU间通信）
        self.message_queues = defaultdict(queue.Queue)
        
        # 协作统计
        self.collab_stats = {
            'remote_hits': 0,           # 远程缓存命中
            'local_hits': 0,            # 本地缓存命中
            'prefetch_success': 0,      # 预取成功
            'load_balance_triggers': 0, # 负载均衡触发次数
            'shared_messages': 0,       # 共享消息数
            'collaboration_benefit': 0  # 协作收益
        }
        
        # 启动消息处理线程
        self.running = True
        self.message_processors = []
        for node_id in self.cache_managers:
            processor = threading.Thread(
                target=self._process_messages,
                args=(node_id,),
                daemon=True
            )
            processor.start()
            self.message_processors.append(processor)
    
    def get_content(self, requesting_rsu: str, content_id: str) -> Optional[Dict[str, Any]]:
        """
        获取内容（支持协作查找）
        
        Args:
            requesting_rsu: 请求的RSU节点ID
            content_id: 内容ID
            
        Returns:
            内容数据，如果未找到返回None
        """
        # 1. 首先检查本地缓存
        local_cache = self.cache_managers[requesting_rsu]
        content = local_cache.get(content_id)
        
        if content is not None:
            self.collab_stats['local_hits'] += 1
            self._update_cache_directory(content_id, requesting_rsu)
            
            # 记录访问用于预测
            if self.popularity_predictor:
                self.popularity_predictor.record_access(
                    content_id, 
                    time.time(),
                    {'rsu': requesting_rsu}
                )
            
            return content
        
        # 2. 查询其他RSU节点
        remote_content = self._query_remote_caches(requesting_rsu, content_id)
        
        if remote_content is not None:
            self.collab_stats['remote_hits'] += 1
            
            # 可选：将热门内容复制到本地
            if self._should_replicate(content_id):
                self._replicate_content(requesting_rsu, content_id, remote_content)
            
            return remote_content
        
        # 3. 缓存未命中，可能需要从源获取
        return None
    
    def put_content(self, rsu_id: str, content_id: str, content: Dict[str, Any], 
                   size: float, metadata: Dict = None) -> bool:
        """
        存储内容到缓存
        
        Args:
            rsu_id: RSU节点ID
            content_id: 内容ID
            content: 内容数据
            size: 内容大小
            metadata: 元数据
            
        Returns:
            是否成功缓存
        """
        # 判断是否为热点内容
        is_hot = False
        if self.popularity_predictor:
            # 预测流行度
            predictions = self.popularity_predictor.predict(
                [content_id], 
                time.time()
            )
            is_hot = predictions.get(content_id, 0) > 0.7
        
        # 存储到本地缓存
        cache_manager = self.cache_managers[rsu_id]
        success = cache_manager.put(content_id, content, size, is_hot)
        
        if success:
            # 更新缓存目录
            self._update_cache_directory(content_id, rsu_id)
            
            # 通知其他节点（用于协作）
            self._broadcast_cache_update(rsu_id, content_id, metadata)
            
            # 如果是热点内容，考虑预复制到其他节点
            if is_hot:
                self._proactive_replication(content_id, content, size, rsu_id)
        
        return success
    
    def _query_remote_caches(self, requesting_rsu: str, content_id: str) -> Optional[Dict[str, Any]]:
        """
        查询远程RSU缓存
        
        Args:
            requesting_rsu: 请求节点
            content_id: 内容ID
            
        Returns:
            找到的内容，或None
        """
        # 检查缓存目录
        if content_id in self.cache_directory:
            candidate_rsus = self.cache_directory[content_id] - {requesting_rsu}
            
            # 选择负载最低的RSU
            if candidate_rsus:
                best_rsu = min(candidate_rsus, key=lambda x: self.node_loads[x])
                
                # 直接从最佳RSU获取（模拟）
                remote_cache = self.cache_managers[best_rsu]
                content = remote_cache.get(content_id)
                
                if content:
                    # 更新负载信息
                    self.node_loads[best_rsu] += 0.1
                    return content
        
        # 如果目录中没有，广播查询（更耗时）
        for rsu_id, cache_manager in self.cache_managers.items():
            if rsu_id != requesting_rsu:
                content = cache_manager.get(content_id)
                if content:
                    # 更新目录
                    self._update_cache_directory(content_id, rsu_id)
                    return content
        
        return None
    
    def _should_replicate(self, content_id: str) -> bool:
        """
        决定是否应该复制内容
        
        Args:
            content_id: 内容ID
            
        Returns:
            是否应该复制
        """
        # 基于访问频率决定
        if self.popularity_predictor:
            predictions = self.popularity_predictor.predict([content_id], time.time())
            return predictions.get(content_id, 0) > 0.6
        
        # 简单策略：如果被访问超过3次就复制
        access_count = sum(
            1 for cache in self.cache_managers.values() 
            if content_id in cache.access_frequency
        )
        return access_count > 3
    
    def _replicate_content(self, target_rsu: str, content_id: str, content: Dict[str, Any]):
        """
        复制内容到目标RSU
        
        Args:
            target_rsu: 目标RSU
            content_id: 内容ID
            content: 内容数据
        """
        if 'size' in content:
            size = content['size']
        else:
            size = len(str(content))
        
        cache_manager = self.cache_managers[target_rsu]
        success = cache_manager.put(content_id, content['content'], size)
        
        if success:
            self._update_cache_directory(content_id, target_rsu)
            self.collab_stats['collaboration_benefit'] += 1
    
    def _proactive_replication(self, content_id: str, content: Dict[str, Any], 
                              size: float, source_rsu: str):
        """
        主动复制热点内容到其他节点
        
        Args:
            content_id: 内容ID
            content: 内容数据
            size: 内容大小
            source_rsu: 源RSU
        """
        # 选择负载较低的节点进行复制
        target_rsus = []
        for rsu_id in self.cache_managers:
            if rsu_id != source_rsu and self.node_loads[rsu_id] < 0.7:
                target_rsus.append(rsu_id)
        
        # 复制到最多2个节点
        for target_rsu in target_rsus[:2]:
            cache_manager = self.cache_managers[target_rsu]
            if cache_manager.put(content_id, content, size, is_hot=True):
                self._update_cache_directory(content_id, target_rsu)
                self.collab_stats['prefetch_success'] += 1
    
    def _update_cache_directory(self, content_id: str, rsu_id: str):
        """更新缓存目录"""
        self.cache_directory[content_id].add(rsu_id)
    
    def _broadcast_cache_update(self, sender_rsu: str, content_id: str, metadata: Dict = None):
        """
        广播缓存更新消息
        
        Args:
            sender_rsu: 发送节点
            content_id: 内容ID
            metadata: 元数据
        """
        for rsu_id in self.cache_managers:
            if rsu_id != sender_rsu:
                msg = CacheMessage(
                    msg_type=MessageType.CACHE_UPDATE,
                    sender_id=sender_rsu,
                    receiver_id=rsu_id,
                    content_id=content_id,
                    data=metadata or {},
                    timestamp=time.time()
                )
                self.message_queues[rsu_id].put(msg)
                self.collab_stats['shared_messages'] += 1
    
    def _process_messages(self, node_id: str):
        """
        处理节点消息（在独立线程中运行）
        
        Args:
            node_id: 节点ID
        """
        while self.running:
            try:
                # 获取消息（超时1秒）
                msg = self.message_queues[node_id].get(timeout=1)
                
                if msg.msg_type == MessageType.CACHE_UPDATE:
                    # 处理缓存更新通知
                    self._handle_cache_update(node_id, msg)
                    
                elif msg.msg_type == MessageType.POPULARITY_SHARE:
                    # 处理流行度共享
                    self._handle_popularity_share(node_id, msg)
                    
                elif msg.msg_type == MessageType.LOAD_BALANCE:
                    # 处理负载均衡请求
                    self._handle_load_balance(node_id, msg)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing message for {node_id}: {e}")
    
    def _handle_cache_update(self, node_id: str, msg: CacheMessage):
        """处理缓存更新消息"""
        # 更新本地的缓存目录信息
        self._update_cache_directory(msg.content_id, msg.sender_id)
        
        # 如果启用预测，记录访问模式
        if self.popularity_predictor:
            self.popularity_predictor.record_access(
                msg.content_id,
                msg.timestamp,
                msg.data
            )
    
    def _handle_popularity_share(self, node_id: str, msg: CacheMessage):
        """处理流行度共享消息"""
        if self.popularity_predictor and 'popularity_scores' in msg.data:
            # 合并流行度信息
            for content_id, score in msg.data['popularity_scores'].items():
                self.popularity_predictor.popularity_scores[content_id].append(
                    (msg.timestamp, score)
                )
    
    def _handle_load_balance(self, node_id: str, msg: CacheMessage):
        """处理负载均衡请求"""
        # 如果本节点负载较低，接收内容
        if self.node_loads[node_id] < 0.5:
            content = msg.data.get('content')
            size = msg.data.get('size', 0)
            if content and size:
                cache_manager = self.cache_managers[node_id]
                if cache_manager.put(msg.content_id, content, size):
                    self._update_cache_directory(msg.content_id, node_id)
                    self.collab_stats['load_balance_triggers'] += 1
    
    def trigger_load_balancing(self):
        """
        触发负载均衡
        将高负载节点的内容迁移到低负载节点
        """
        # 找出高负载和低负载节点
        high_load_rsus = [
            rsu for rsu, load in self.node_loads.items() 
            if load > 0.8
        ]
        low_load_rsus = [
            rsu for rsu, load in self.node_loads.items() 
            if load < 0.3
        ]
        
        if high_load_rsus and low_load_rsus:
            for high_rsu in high_load_rsus:
                cache_manager = self.cache_managers[high_rsu]
                
                # 获取最冷的内容
                cold_contents = []
                for content_id in cache_manager.l2_cache:
                    freq = cache_manager.access_frequency[content_id]
                    if freq < 2:  # 访问次数少于2次
                        cold_contents.append(content_id)
                
                # 迁移到低负载节点
                for content_id in cold_contents[:5]:  # 最多迁移5个
                    target_rsu = low_load_rsus[0]
                    
                    # 发送负载均衡消息
                    content_data = cache_manager.get(content_id)
                    if content_data:
                        msg = CacheMessage(
                            msg_type=MessageType.LOAD_BALANCE,
                            sender_id=high_rsu,
                            receiver_id=target_rsu,
                            content_id=content_id,
                            data={
                                'content': content_data,
                                'size': content_data.get('size', 0)
                            },
                            timestamp=time.time()
                        )
                        self.message_queues[target_rsu].put(msg)
    
    def share_popularity_predictions(self):
        """
        在RSU间共享流行度预测信息
        """
        if not self.popularity_predictor:
            return
        
        # 获取预测的热门内容
        top_predictions = self.popularity_predictor.get_top_predictions(20)
        
        # 广播给所有节点
        for rsu_id in self.cache_managers:
            msg = CacheMessage(
                msg_type=MessageType.POPULARITY_SHARE,
                sender_id="GLOBAL",
                receiver_id=rsu_id,
                content_id="",
                data={'popularity_scores': dict(top_predictions)},
                timestamp=time.time()
            )
            self.message_queues[rsu_id].put(msg)
            self.collab_stats['shared_messages'] += 1
    
    def get_system_stats(self) -> Dict[str, Any]:
        """
        获取系统统计信息
        
        Returns:
            系统统计数据
        """
        # 收集各节点统计
        node_stats = {}
        total_l1_hits = 0
        total_l2_hits = 0
        total_misses = 0
        
        for rsu_id, cache_manager in self.cache_managers.items():
            stats = cache_manager.get_stats()
            node_stats[rsu_id] = stats
            total_l1_hits += cache_manager.stats['l1_hits']
            total_l2_hits += cache_manager.stats['l2_hits']
            total_misses += cache_manager.stats['misses']
        
        # 计算整体命中率
        total_local_hits = total_l1_hits + total_l2_hits
        total_hits = total_local_hits + self.collab_stats['remote_hits']
        total_requests = total_hits + total_misses
        
        overall_hit_rate = total_hits / total_requests if total_requests > 0 else 0
        collaboration_rate = self.collab_stats['remote_hits'] / total_requests if total_requests > 0 else 0
        
        return {
            'node_stats': node_stats,
            'overall_hit_rate': overall_hit_rate,
            'collaboration_rate': collaboration_rate,
            'local_hit_rate': total_local_hits / total_requests if total_requests > 0 else 0,
            'remote_hit_rate': self.collab_stats['remote_hits'] / total_requests if total_requests > 0 else 0,
            'prefetch_success': self.collab_stats['prefetch_success'],
            'load_balance_triggers': self.collab_stats['load_balance_triggers'],
            'shared_messages': self.collab_stats['shared_messages'],
            'collaboration_benefit': self.collab_stats['collaboration_benefit'],
            'cache_directory_size': len(self.cache_directory),
            'unique_contents': len(set().union(*self.cache_directory.values())) if self.cache_directory else 0
        }
    
    def periodic_maintenance(self):
        """
        定期维护任务
        应该定期调用（如每分钟）
        """
        # 1. 触发负载均衡
        self.trigger_load_balancing()
        
        # 2. 共享流行度预测
        self.share_popularity_predictions()
        
        # 3. 训练预测模型
        if self.popularity_predictor and not self.popularity_predictor.is_trained:
            self.popularity_predictor.train(epochs=5)
        
        # 4. 清理过期的缓存目录条目
        self._cleanup_cache_directory()
        
        # 5. 更新节点负载
        self._update_node_loads()
    
    def _cleanup_cache_directory(self):
        """清理缓存目录中的过期条目"""
        to_remove = []
        
        for content_id, rsu_set in self.cache_directory.items():
            # 验证每个RSU是否真的有这个内容
            valid_rsus = set()
            for rsu_id in rsu_set:
                cache_manager = self.cache_managers[rsu_id]
                if content_id in cache_manager.l1_cache or content_id in cache_manager.l2_cache:
                    valid_rsus.add(rsu_id)
            
            if valid_rsus:
                self.cache_directory[content_id] = valid_rsus
            else:
                to_remove.append(content_id)
        
        # 删除无效条目
        for content_id in to_remove:
            del self.cache_directory[content_id]
    
    def _update_node_loads(self):
        """更新节点负载信息"""
        for rsu_id, cache_manager in self.cache_managers.items():
            stats = cache_manager.get_stats()
            # 基于缓存利用率和请求量计算负载
            utilization = (stats['l1_utilization'] + stats['l2_utilization']) / 2
            self.node_loads[rsu_id] = utilization * 0.7 + self.node_loads[rsu_id] * 0.3
    
    def shutdown(self):
        """关闭系统"""
        self.running = False
        for processor in self.message_processors:
            processor.join(timeout=2)
        
        # 保存预测模型
        if self.popularity_predictor and self.popularity_predictor.is_trained:
            self.popularity_predictor.save_model("cache_popularity_model.pth")
