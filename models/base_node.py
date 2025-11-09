"""
VEC绯荤粺鑺傜偣瀹炵幇 - 瀵瑰簲璁烘枃绗?鑺傜郴缁熸ā鍨?
鍖呭惈杞﹁締銆丷SU銆乁AV涓夌鑺傜偣绫诲瀷鐨勫叿浣撳疄鐜?
"""
import logging
import numpy as np
import math
import time
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod

from .data_structures import (
    Task, QueueSlot, Position, NodeState, NodeType, 
    CommunicationLink, TaskType
)
from config import config
from utils import generate_poisson_arrivals, ExponentialMovingAverage


logger = logging.getLogger(__name__)


class BaseNode(ABC):
    """
    鎶借薄鍩虹鑺傜偣绫?
    瀹氫箟鎵€鏈夎绠楄妭鐐圭殑閫氱敤鎺ュ彛鍜屽睘鎬?
    """
    def __init__(self, node_id: str, node_type: NodeType, position: Position):
        self.node_id = node_id
        self.node_type = node_type
        self.state = NodeState(node_id=node_id, node_type=node_type, position=position)
        
        # 澶氫紭鍏堢骇鐢熷懡鍛ㄦ湡闃熷垪 - 瀵瑰簲璁烘枃绗?.3鑺?
        self.queues: Dict[Tuple[int, int], QueueSlot] = {}
        self._initialize_queues()
        self._queue_data_usage: float = 0.0
        self._queue_instability_alerted = False
        
        # 鎬ц兘缁熻
        self.processed_tasks: List[Task] = []
        self.dropped_tasks: List[Task] = []
        self.energy_consumption_history: List[float] = []
        
        # 骞冲潎浠诲姟澶嶆潅搴︾粺璁?- 淇鍗曚綅锛氬瓧鑺傝浆姣旂壒
        self._avg_task_complexity: float = config.task.task_compute_density * float(np.mean(config.task.data_size_range)) * 8
        
        # 绉诲姩骞冲潎璁＄畻鍣?
        self.avg_arrival_rate = ExponentialMovingAverage(alpha=0.1)
        self.avg_service_rate = ExponentialMovingAverage(alpha=0.1)
        self.avg_waiting_time = ExponentialMovingAverage(alpha=0.1)
    
    def _record_energy_usage(self, energy_cost: float) -> None:
        """璁板綍鑳借€楀苟缁存姢鍘嗗彶绐楀彛銆?""
        if energy_cost <= 0:
            return
        self.state.total_energy += energy_cost
        self.energy_consumption_history.append(energy_cost)
        if len(self.energy_consumption_history) > 100:
            self.energy_consumption_history.pop(0)
    
    def _initialize_queues(self):
        """鍒濆鍖栧浼樺厛绾х敓鍛藉懆鏈熼槦鍒楃粨鏋?""
        max_lifetime = config.queue.max_lifetime
        num_priorities = config.task.num_priority_levels
        
        # 鏍规嵁鑺傜偣绫诲瀷纭畾闃熷垪缁村害
        if self.node_type == NodeType.VEHICLE:
            # 杞﹁締缁存姢瀹屾暣鐨凩脳P闃熷垪鐭╅樀
            lifetime_range = range(1, max_lifetime + 1)
        else:
            # RSU鍜孶AV缁存姢(L-1)脳P闃熷垪鐭╅樀
            lifetime_range = range(1, max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, num_priorities + 1):
                self.queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
    
    @abstractmethod
    def get_processing_capacity(self) -> float:
        """鑾峰彇澶勭悊鑳藉姏 (bits/鏃堕殭)"""
        pass
    
    @abstractmethod
    def calculate_processing_delay(self, task: Task) -> float:
        """璁＄畻浠诲姟澶勭悊鏃跺欢"""
        pass
    
    @abstractmethod  
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """璁＄畻鑳借€?""
        pass
    
    def add_task_to_queue(self, task: Task) -> bool:
        """
        锟斤拷锟斤拷锟斤拷锟斤拷锟接碉拷锟斤拷应锟侥讹拷锟叫诧拷位
        锟斤拷锟斤拷锟斤拷锟斤拷锟绞ｏ拷锟斤拷锟斤拷锟斤拷锟斤拷诤锟斤拷锟斤拷燃锟饺凤拷锟斤拷锟轿?
        """
        lifetime = task.remaining_lifetime_slots
        priority = task.priority

        if lifetime <= 0:
            self._register_drop(task)
            return False

        if not self._has_capacity_for(task):
            logger.debug("Queue capacity reached for %s; dropping task %s", self.node_id, task.task_id)
            self._register_drop(task)
            return False

        queue_key = (lifetime, priority)
        if queue_key not in self.queues:
            self._register_drop(task)
            return False

        if self.queues[queue_key].add_task(task):
            self._queue_data_usage += task.data_size
            task.queue_arrival_time = time.time()
            task.waiting_delay = 0.0
            return True

        return False

    def _has_capacity_for(self, task: Task) -> bool:
        """锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟角凤拷锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷锟?""
        capacity_limit = self._get_queue_capacity_limit()
        if capacity_limit <= 0:
            return True
        return (self._queue_data_usage + task.data_size) <= capacity_limit

    def _get_queue_capacity_limit(self) -> float:
        """锟斤拷取锟斤拷前锟节碉拷锟接︼拷亩锟斤拷锟斤拷锟斤拷锟较?""
        if self.node_type == NodeType.VEHICLE:
            return getattr(config.queue, 'vehicle_queue_capacity', -1.0)
        if self.node_type == NodeType.RSU:
            return getattr(config.queue, 'rsu_queue_capacity', -1.0)
        return getattr(config.queue, 'uav_queue_capacity', -1.0)

    def get_next_task_to_process(self) -> Optional[Task]:
        """
        鑾峰彇涓嬩竴涓緟澶勭悊浠诲姟
        鎸夌収闈炴姠鍗犲紡浼樺厛绾ц皟搴︾瓥鐣? 楂樹紭鍏堢骇浼樺厛锛屽悓浼樺厛绾IFO
        """
        # 鎸変紭鍏堢骇浠庨珮鍒颁綆閬嶅巻 (priority=1鏄渶楂樹紭鍏堢骇)
        for priority in range(1, config.task.num_priority_levels + 1):
            # 鍦ㄥ悓涓€浼樺厛绾у唴锛屾寜鐢熷懡鍛ㄦ湡绱ц揩绋嬪害閬嶅巻
            for lifetime in sorted(self.queues.keys()):
                if lifetime[1] == priority:  # 鍖归厤浼樺厛绾?
                    queue = self.queues[lifetime]
                    if not queue.is_empty():
                        return queue.get_next_task()
        return None
    
    def process_task(self, task: Task) -> bool:
        """
        锟斤拷锟斤拷锟斤拷锟斤拷
        锟斤拷锟津返回斤拷锟角凤拷晒锟斤拷锟斤拷锟斤拷锟斤拷
        """
        # 锟斤拷锟姐处锟斤拷时锟斤拷
        processing_delay = self.calculate_processing_delay(task)

        # 锟斤拷锟斤拷欠锟皆斤拷锟斤拷锟街故憋拷锟?
        if task.is_deadline_violated():
            self._remove_task_from_queue(task)
            self._register_drop(task)
            return False

        # 执锟斤拷锟斤拷锟斤拷锟斤拷锟斤拷
        task_start_time = time.time()
        task.start_time = task_start_time
        if task.queue_arrival_time is not None:
            task.waiting_delay = max(0.0, task_start_time - task.queue_arrival_time)
        else:
            task.waiting_delay = max(0.0, task.waiting_delay)
        task.processing_delay = processing_delay
        task.assigned_node_id = self.node_id

        # 锟斤拷锟姐功锟剿?
        energy_cost = self.calculate_energy_consumption(processing_delay)
        self._record_energy_usage(energy_cost)

        # 模锟解处锟斤拷锟斤拷锟斤拷锟斤拷
        task.completion_time = task_start_time + processing_delay
        task.is_completed = True

        # 锟接讹拷锟斤拷锟斤拷锟狡筹拷锟斤拷锟斤拷
        self._remove_task_from_queue(task)

        # 锟斤拷锟斤拷锟斤拷汛锟斤拷锟斤拷锟斤拷锟斤拷斜?
        self.processed_tasks.append(task)
        # 锟斤拷锟斤拷锟斤拷汛锟斤拷锟斤拷锟斤拷锟斤拷斜锟斤拷锟斤拷锟斤拷锟街癸拷诖娲拷锟?
        if len(self.processed_tasks) > 50:
            self.processed_tasks.pop(0)

        # 锟斤拷锟斤拷统锟斤拷锟斤拷息
        self._update_statistics()

        return True

    def _remove_task_from_queue(self, task: Task) -> bool:
        """浠庨槦鍒椾腑绉婚櫎鎸囧畾浠诲姟"""
        for queue in self.queues.values():
            removed_task = queue.remove_task(task.task_id)
            if removed_task:
                self._queue_data_usage = max(0.0, self._queue_data_usage - removed_task.data_size)
                removed_task.queue_arrival_time = None
                return True
        return False

    def _register_drop(self, task: Task):
        """璁板綍浠诲姟琚涪寮?""
        if not task.is_dropped:
            task.is_dropped = True
            self.dropped_tasks.append(task)
    
    def predict_waiting_time(self, task: Task) -> float:
        """
        棰勬祴浠诲姟绛夊緟鏃堕棿 - 瀵瑰簲璁烘枃寮?2)鍜屽紡(3)
        浣跨敤M/M/1闈炴姠鍗犲紡浼樺厛绾ч槦鍒楁ā鍨?
        """
        priority = task.priority
        
        # 璁＄畻鍒拌揪鐜囧拰鏈嶅姟鐜?
        arrival_rates = self._calculate_arrival_rates_by_priority()
        service_rate = self._calculate_service_rate()
        
        if service_rate <= 0:
            return float('inf')
        
        # 璁＄畻璐熻浇鍥犲瓙
        rho_values = {p: arrival_rates.get(p, 0) / service_rate 
                     for p in range(1, config.task.num_priority_levels + 1)}
        
        # 妫€鏌ョǔ瀹氭€ф潯浠?
        total_rho = sum(rho_values.values())
        warning_ratio = getattr(config.queue, 'stability_warning_ratio', 0.9)
        instability_threshold = getattr(config.queue, 'global_rho_threshold', 1.0)

        if total_rho >= warning_ratio:
            self.state.stability_warning = True
            if total_rho >= instability_threshold and not self._queue_instability_alerted:
                logger.warning("Queue at node %s is unstable (total rho=%.3f)", self.node_id, total_rho)
                self._queue_instability_alerted = True
        else:
            self.state.stability_warning = False
            self._queue_instability_alerted = False

        if total_rho >= instability_threshold:
            return float('inf')

        numerator = sum(rho_values[p] for p in range(1, priority + 1))
        
        denominator1 = 1 - sum(rho_values[p] for p in range(1, priority))
        denominator2 = 1 - sum(rho_values[p] for p in range(1, priority + 1))
        
        if denominator1 <= 0 or denominator2 <= 0:
            return float('inf')
        
        waiting_time = (1 / service_rate) * (numerator / (denominator1 * denominator2))
        
        return waiting_time
    
    def _calculate_arrival_rates_by_priority(self) -> Dict[int, float]:
        """璁＄畻鍚勪紭鍏堢骇浠诲姟鐨勫埌杈剧巼"""
        arrival_rates = {}
        for priority in range(1, config.task.num_priority_levels + 1):
            # 缁熻鍚勪紭鍏堢骇闃熷垪涓殑浠诲姟鏁伴噺
            total_tasks = sum(len(queue.task_list) 
                            for (l, p), queue in self.queues.items() 
                            if p == priority)
            # 杞崲涓哄埌杈剧巼 (绠€鍖栦及绠?
            arrival_rates[priority] = total_tasks / config.network.time_slot_duration
        return arrival_rates
    
    def _calculate_service_rate(self) -> float:
        """璁＄畻骞冲潎鏈嶅姟鐜?(tasks/绉?"""
        if hasattr(self, '_avg_task_complexity') and self._avg_task_complexity > 0:
            return self.state.cpu_frequency / self._avg_task_complexity
        else:
            # 浣跨敤榛樿璁＄畻澶嶆潅搴︿及绠?- 淇鍗曚綅锛氬瓧鑺傝浆姣旂壒
            avg_complexity = config.task.task_compute_density * float(np.mean(config.task.task_data_size_range)) * 8
            return self.state.cpu_frequency / avg_complexity
    
    def _update_statistics(self):
        """鏇存柊鑺傜偣缁熻淇℃伅"""
        # 鏇存柊璐熻浇鍥犲瓙
        if len(self.processed_tasks) > 0:
            recent_tasks = self.processed_tasks[-10:]  # 鏈€杩?0涓换鍔?
            avg_arrival = len(recent_tasks) / (10 * config.network.time_slot_duration)
            self.avg_arrival_rate.update(avg_arrival)
            
            avg_service = self._calculate_service_rate()
            self.avg_service_rate.update(avg_service)
            
            self.state.update_load_factor(
                self.avg_arrival_rate.get_value(),
                self.avg_service_rate.get_value()
            )
        
        # 鏇存柊闃熷垪闀垮害
        self.state.queue_length = sum(len(queue.task_list) for queue in self.queues.values())
        
        # 鏇存柊骞冲潎绛夊緟鏃堕棿
        if len(self.processed_tasks) > 0:
            recent_waiting_times = [task.waiting_delay for task in self.processed_tasks[-10:]]
            if recent_waiting_times:
                avg_waiting_time = float(np.mean(recent_waiting_times))
                self.avg_waiting_time.update(avg_waiting_time)
                self.state.avg_waiting_time = self.avg_waiting_time.get_value()
    
    def update_queue_lifetimes(self):
        """
        锟斤拷锟斤拷锟叫革拷锟斤拷锟叫碉拷锟斤拷锟街甘伙拷锟节ｏ拷每锟斤拷时锟截猴拷锟斤拷
        """
        if self.node_type == NodeType.VEHICLE:
            lifetime_range = range(1, config.queue.max_lifetime + 1)
        else:
            lifetime_range = range(1, config.queue.max_lifetime)

        new_queues = {}
        for lifetime in lifetime_range:
            for priority in range(1, config.task.num_priority_levels + 1):
                new_queues[(lifetime, priority)] = QueueSlot(lifetime, priority)

        for (lifetime, priority), queue in self.queues.items():
            for task in queue.task_list:
                if task.is_deadline_violated():
                    self._register_drop(task)
                    continue

                new_lifetime = lifetime - 1
                if new_lifetime >= 1 and (new_lifetime, priority) in new_queues:
                    new_queues[(new_lifetime, priority)].add_task(task)
                else:
                    self._register_drop(task)

        self.queues = new_queues
        self._queue_data_usage = sum(slot.data_volume for slot in self.queues.values())

    def get_state_vector(self) -> np.ndarray:
        """鑾峰彇鑺傜偣鐘舵€佸悜閲忥紝鐢ㄤ簬寮哄寲瀛︿範"""
        state_features = [
            self.state.cpu_utilization,
            self.state.load_factor,
            self.state.queue_length / 100.0,  # 褰掍竴鍖?
            self.state.avg_waiting_time / 10.0,  # 褰掍竴鍖?
            len(self.processed_tasks) / 1000.0,  # 褰掍竴鍖?
            len(self.dropped_tasks) / 1000.0,   # 褰掍竴鍖?
        ]
        
        # 娣诲姞鍚勪紭鍏堢骇闃熷垪鐘舵€?
        for priority in range(1, config.task.num_priority_levels + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            state_features.append(priority_tasks / 50.0)  # 褰掍竴鍖?
        
        return np.array(state_features, dtype=np.float32)
