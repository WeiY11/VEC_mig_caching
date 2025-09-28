#!/usr/bin/env python3
"""
优先级队列管理器
为VEC系统提供基于任务类型和紧急程度的优先级调度
"""

import heapq
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class TaskPriority(Enum):
    """任务优先级枚举"""
    EMERGENCY = 1      # 紧急任务(碰撞避免、紧急制动)
    CRITICAL = 2       # 关键任务(安全警报、交通信号)  
    HIGH = 3           # 高优先级(实时导航、传感器数据)
    NORMAL = 4         # 普通任务(地图更新、停车信息)
    LOW = 5            # 低优先级(娱乐内容、数据分析)

@dataclass
class PriorityTask:
    """优先级任务"""
    priority: int          # 优先级数字(1最高，5最低)
    urgency_score: float   # 紧急度分数
    deadline: float        # 截止时间
    task_id: str          # 任务ID
    task_data: Dict       # 任务数据
    enqueue_time: float   # 入队时间
    
    def __lt__(self, other):
        """定义比较逻辑，用于堆排序"""
        # 优先级数字越小越优先，紧急度分数越高越优先
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.urgency_score > other.urgency_score

class IntelligentPriorityQueue:
    """智能优先级队列"""
    
    def __init__(self, max_capacity: int = 30):
        self.max_capacity = max_capacity
        self.queue = []  # 优先级堆
        self.task_map = {}  # 任务ID映射
        self.dropped_tasks = []  # 丢弃任务记录
        
        # 统计信息
        self.stats = {
            'total_enqueued': 0,
            'total_dequeued': 0,
            'total_dropped': 0,
            'priority_distribution': {p.value: 0 for p in TaskPriority},
            'avg_wait_time': 0.0
        }
    
    def calculate_task_priority(self, task: Dict) -> Tuple[TaskPriority, float]:
        """
        根据任务特征计算优先级和紧急度
        """
        task_type = task.get('task_type', 3)
        deadline = task.get('deadline', 10.0)
        current_time = time.time()
        remaining_time = deadline - current_time
        
        # 基于任务类型的基础优先级
        if task_type == 1:  # 极度延迟敏感
            base_priority = TaskPriority.EMERGENCY
        elif task_type == 2:  # 延迟敏感
            base_priority = TaskPriority.CRITICAL
        elif task_type == 3:  # 中度容忍
            base_priority = TaskPriority.HIGH
        else:  # 延迟容忍
            base_priority = TaskPriority.NORMAL
        
        # 基于截止时间的紧急度调整
        if remaining_time < 0.5:  # 0.5秒内截止
            urgency_score = 1.0
            if base_priority.value > 1:
                base_priority = TaskPriority.CRITICAL  # 提升优先级
        elif remaining_time < 1.0:  # 1秒内截止
            urgency_score = 0.8
        elif remaining_time < 2.0:  # 2秒内截止
            urgency_score = 0.6
        else:
            urgency_score = max(0.1, 1.0 - remaining_time / 10.0)  # 基于剩余时间
        
        # 基于应用类型的额外调整
        app_name = task.get('app_name', '').lower()
        if 'emergency' in app_name or 'brake' in app_name:
            base_priority = TaskPriority.EMERGENCY
            urgency_score = min(1.0, urgency_score + 0.5)
        elif 'safety' in app_name or 'collision' in app_name:
            base_priority = TaskPriority.CRITICAL
            urgency_score = min(1.0, urgency_score + 0.3)
        elif 'entertainment' in app_name or 'video' in app_name:
            base_priority = TaskPriority.LOW
            urgency_score = max(0.1, urgency_score - 0.2)
        
        return base_priority, urgency_score
    
    def enqueue(self, task: Dict) -> bool:
        """
        将任务加入优先级队列
        
        Returns:
            True if successfully enqueued, False if dropped
        """
        current_time = time.time()
        task_id = task.get('id', f"task_{current_time}")
        
        # 计算优先级
        priority, urgency = self.calculate_task_priority(task)
        
        # 创建优先级任务
        priority_task = PriorityTask(
            priority=priority.value,
            urgency_score=urgency,
            deadline=task.get('deadline', current_time + 10.0),
            task_id=task_id,
            task_data=task,
            enqueue_time=current_time
        )
        
        # 检查容量
        if len(self.queue) >= self.max_capacity:
            # 容量满，尝试丢弃低优先级任务
            if not self._make_space_for_task(priority_task):
                self.dropped_tasks.append(task)
                self.stats['total_dropped'] += 1
                return False
        
        # 加入队列
        heapq.heappush(self.queue, priority_task)
        self.task_map[task_id] = priority_task
        self.stats['total_enqueued'] += 1
        self.stats['priority_distribution'][priority.value] += 1
        
        return True
    
    def dequeue(self) -> Optional[Dict]:
        """
        取出最高优先级任务
        """
        if not self.queue:
            return None
        
        priority_task = heapq.heappop(self.queue)
        
        # 检查是否已过期
        current_time = time.time()
        if current_time > priority_task.deadline:
            # 任务已过期，丢弃
            self.dropped_tasks.append(priority_task.task_data)
            self.stats['total_dropped'] += 1
            
            # 递归尝试下一个任务
            return self.dequeue()
        
        # 更新统计
        self.stats['total_dequeued'] += 1
        wait_time = current_time - priority_task.enqueue_time
        self._update_avg_wait_time(wait_time)
        
        # 从映射中移除
        if priority_task.task_id in self.task_map:
            del self.task_map[priority_task.task_id]
        
        return priority_task.task_data
    
    def _make_space_for_task(self, new_task: PriorityTask) -> bool:
        """
        为新任务腾出空间，丢弃低优先级任务
        """
        if not self.queue:
            return False
        
        # 找到最低优先级任务
        lowest_priority_task = max(self.queue, key=lambda t: (t.priority, -t.urgency_score))
        
        # 只有新任务优先级更高才丢弃旧任务
        if (new_task.priority < lowest_priority_task.priority or 
            (new_task.priority == lowest_priority_task.priority and 
             new_task.urgency_score > lowest_priority_task.urgency_score)):
            
            # 移除低优先级任务
            self.queue.remove(lowest_priority_task)
            heapq.heapify(self.queue)  # 重新堆化
            
            self.dropped_tasks.append(lowest_priority_task.task_data)
            self.stats['total_dropped'] += 1
            
            if lowest_priority_task.task_id in self.task_map:
                del self.task_map[lowest_priority_task.task_id]
            
            return True
        
        return False
    
    def _update_avg_wait_time(self, wait_time: float):
        """更新平均等待时间"""
        current_avg = self.stats['avg_wait_time']
        total_dequeued = self.stats['total_dequeued']
        
        if total_dequeued == 1:
            self.stats['avg_wait_time'] = wait_time
        else:
            # 移动平均
            alpha = 0.1
            self.stats['avg_wait_time'] = alpha * wait_time + (1 - alpha) * current_avg
    
    def get_queue_statistics(self) -> Dict:
        """获取队列统计信息"""
        current_time = time.time()
        
        # 计算当前队列中各优先级分布
        current_priority_dist = {p.value: 0 for p in TaskPriority}
        pending_deadlines = []
        
        for task in self.queue:
            current_priority_dist[task.priority] += 1
            time_to_deadline = task.deadline - current_time
            pending_deadlines.append(time_to_deadline)
        
        return {
            'queue_length': len(self.queue),
            'capacity_utilization': len(self.queue) / self.max_capacity,
            'current_priority_distribution': current_priority_dist,
            'total_stats': dict(self.stats),
            'avg_time_to_deadline': np.mean(pending_deadlines) if pending_deadlines else 0.0,
            'urgent_tasks_count': sum(1 for t in self.queue if t.urgency_score > 0.8)
        }
    
    def clear_expired_tasks(self) -> int:
        """清理过期任务"""
        current_time = time.time()
        expired_count = 0
        
        # 从队列中移除过期任务
        valid_tasks = []
        for task in self.queue:
            if current_time <= task.deadline:
                valid_tasks.append(task)
            else:
                self.dropped_tasks.append(task.task_data)
                self.stats['total_dropped'] += 1
                expired_count += 1
                
                if task.task_id in self.task_map:
                    del self.task_map[task.task_id]
        
        # 重建堆
        self.queue = valid_tasks
        heapq.heapify(self.queue)
        
        return expired_count

def create_priority_queue_for_node(node_type: str) -> IntelligentPriorityQueue:
    """
    为不同类型节点创建优先级队列
    """
    capacity_map = {
        'rsu': 25,      # RSU更高容量
        'uav': 15,      # UAV中等容量
        'vehicle': 10   # 车辆较低容量
    }
    
    capacity = capacity_map.get(node_type, 20)
    return IntelligentPriorityQueue(capacity)

import numpy as np
