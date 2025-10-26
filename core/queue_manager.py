"""
澶氫紭鍏堢骇鐢熷懡鍛ㄦ湡闃熷垪绠＄悊鍣?- 瀵瑰簲璁烘枃绗?.3鑺?
瀹炵幇VEC绯荤粺涓殑鍒嗗眰闃熷垪绯荤粺鍜孧/M/1闈炴姠鍗犲紡浼樺厛绾ч槦鍒楁ā鍨?
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from models.data_structures import Task, QueueSlot, NodeType
from config import config
from utils import ExponentialMovingAverage


@dataclass
class QueueStatistics:
    """闃熷垪缁熻淇℃伅"""
    total_arrivals: int = 0
    total_departures: int = 0
    total_drops: int = 0
    avg_waiting_time: float = 0.0
    avg_queue_length: float = 0.0
    avg_service_time: float = 0.0
    
    def __post_init__(self):
        # 鍒濆鍖栦紭鍏堢骇缁熻瀛楀吀
        self.priority_arrivals: Dict[int, int] = defaultdict(int)
        self.priority_waiting_times: Dict[int, float] = defaultdict(float)


class PriorityQueueManager:
    """
    澶氫紭鍏堢骇鐢熷懡鍛ㄦ湡闃熷垪绠＄悊鍣?
    
    瀹炵幇鍔熻兘:
    1. 澶氱淮闃熷垪绠＄悊 (鐢熷懡鍛ㄦ湡 脳 浼樺厛绾?
    2. M/M/1闈炴姠鍗犲紡浼樺厛绾ч槦鍒楅娴?
    3. 闃熷垪缁熻涓庢€ц兘鍒嗘瀽
    4. 璐熻浇鍧囪　涓庡閲忕鐞?
    """
    
    def __init__(self, node_id: str, node_type: NodeType):
        self.node_id = node_id
        self.node_type = node_type
        
        # 闃熷垪缁村害鍙傛暟
        self.max_lifetime = config.queue.max_lifetime  # L
        self.num_priorities = config.task.num_priority_levels  # P
        
        # 闃熷垪缁撴瀯 - {(lifetime, priority): QueueSlot}
        self.queues: Dict[Tuple[int, int], QueueSlot] = {}
        self._initialize_queues()
        
        # 瀹归噺闄愬埗
        self.max_capacity = self._get_queue_capacity()
        self.current_usage = 0.0
        
        # 缁熻淇℃伅
        self.statistics = QueueStatistics()
        
        # M/M/1妯″瀷鍙傛暟
        self.arrival_rates: Dict[int, float] = defaultdict(float)  # 位_i (鎸変紭鍏堢骇)
        self.service_rate: float = 0.0  # 渭
        self.load_factors: Dict[int, float] = defaultdict(float)  # 蟻_i = 位_i/渭
        
        # 绉诲姩骞冲潎璁＄畻鍣?
        self.avg_calculators = {
            'arrival_rate': ExponentialMovingAverage(alpha=0.1),
            'service_rate': ExponentialMovingAverage(alpha=0.1),
            'waiting_time': ExponentialMovingAverage(alpha=0.1)
        }
        
        # 鏃堕棿绐楀彛缁熻
        self.time_window_size = 6  # 缁熻绐楀彛澶у皬 (鏃堕殭)
        self.recent_arrivals: List[Dict[int, int]] = []  # 鏈€杩戝埌杈剧粺璁?
        self.recent_services: List[int] = []  # 鏈€杩戞湇鍔＄粺璁?
    
    def _initialize_queues(self):
        """鍒濆鍖栭槦鍒楃粨鏋?""
        if self.node_type == NodeType.VEHICLE:
            # 杞﹁締缁存姢瀹屾暣鐨凩脳P闃熷垪鐭╅樀
            lifetime_range = range(1, self.max_lifetime + 1)
        else:
            # RSU鍜孶AV缁存姢(L-1)脳P闃熷垪鐭╅樀  
            lifetime_range = range(1, self.max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, self.num_priorities + 1):
                self.queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
    
    def _get_queue_capacity(self) -> float:
        """鑾峰彇闃熷垪瀹归噺闄愬埗"""
        if self.node_type == NodeType.VEHICLE:
            return config.queue.vehicle_queue_capacity
        elif self.node_type == NodeType.RSU:
            return config.queue.rsu_queue_capacity
        else:  # UAV
            return config.queue.uav_queue_capacity
    
    def add_task(self, task: Task) -> bool:
        """
        娣诲姞浠诲姟鍒伴槦鍒?
        
        Args:
            task: 寰呮坊鍔犵殑浠诲姟
            
        Returns:
            鏄惁鎴愬姛娣诲姞
        """
        # 妫€鏌ュ閲忛檺鍒?
        if self.current_usage + task.data_size > self.max_capacity:
            self._handle_queue_overflow(task)
            return False
        
        # 纭畾闃熷垪浣嶇疆
        lifetime = task.remaining_lifetime_slots
        priority = task.priority
        queue_key = (lifetime, priority)
        
        # 妫€鏌ラ槦鍒楁槸鍚﹀瓨鍦?
        if queue_key not in self.queues:
            # 鐢熷懡鍛ㄦ湡瓒呭嚭鑼冨洿锛屼换鍔¤涓㈠純
            self.statistics.total_drops += 1
            return False
        
        # 娣诲姞鍒扮浉搴旈槦鍒?
        success = self.queues[queue_key].add_task(task)
        
        if success:
            self.current_usage += task.data_size
            self.statistics.total_arrivals += 1
            self.statistics.priority_arrivals[priority] += 1
            
            # 鏇存柊鍒拌揪鐜囩粺璁?
            self._update_arrival_statistics(priority)
        
        return success
    
    def get_next_task(self) -> Optional[Task]:
        """
        鑾峰彇涓嬩竴涓緟澶勭悊浠诲姟
        鎸夌収闈炴姠鍗犲紡浼樺厛绾ц皟搴︾瓥鐣?
        """
        # 鎸変紭鍏堢骇浠庨珮鍒颁綆閬嶅巻 (priority=1鏄渶楂樹紭鍏堢骇)
        for priority in range(1, self.num_priorities + 1):
            # 鍦ㄥ悓涓€浼樺厛绾у唴锛屾寜鐢熷懡鍛ㄦ湡绱ц揩绋嬪害閬嶅巻
            for lifetime in range(1, self.max_lifetime + 1):
                queue_key = (lifetime, priority)
                if queue_key in self.queues and not self.queues[queue_key].is_empty():
                    # 鎵惧埌涓嬩竴涓换鍔?
                    queue = self.queues[queue_key]
                    task = queue.get_next_task()
                    if task:
                        return task
        
        return None
    
    def remove_task(self, task: Task) -> bool:
        """
        浠庨槦鍒椾腑绉婚櫎浠诲姟
        
        Args:
            task: 寰呯Щ闄ょ殑浠诲姟
            
        Returns:
            鏄惁鎴愬姛绉婚櫎
        """
        # 閬嶅巻鎵€鏈夐槦鍒楀鎵句换鍔?
        for queue in self.queues.values():
            removed_task = queue.remove_task(task.task_id)
            if removed_task:
                self.current_usage -= removed_task.data_size
                self.statistics.total_departures += 1
                
                # 鏇存柊鏈嶅姟缁熻
                self._update_service_statistics()
                
                return True
        
        return False
    
    def predict_waiting_time_mm1(self, task: Task) -> float:
        """
        浣跨敤M/M/1闈炴姠鍗犲紡浼樺厛绾ч槦鍒楁ā鍨嬮娴嬬瓑寰呮椂闂?
        瀵瑰簲璁烘枃寮?2)鍜屽紡(3)
        娣诲姞鏁板€肩ǔ瀹氭€т繚闅?
        
        Args:
            task: 寰呴娴嬬殑浠诲姟
            
        Returns:
            棰勬祴绛夊緟鏃堕棿 (绉?
        """
        priority = task.priority
        
        # 妫€鏌ヤ紭鍏堢骇鐨勬湁鏁堟€?
        if priority < 1 or priority > len(self.load_factors):
            return float('inf')
        
        # 妫€鏌ユ湇鍔＄巼鐨勬湁鏁堟€?
        if self.service_rate <= 1e-10:  # 闃叉闄や互闆?
            return float('inf')
        
        # 妫€鏌ョǔ瀹氭€ф潯浠?
        total_rho = sum(self.load_factors.values())
        if total_rho >= 0.99:  # 鐣欐湁涓€瀹氱殑绋冲畾鎬т綑閲?
            return float('inf')  # 绯荤粺涓嶇ǔ瀹?
        
        # 璁＄畻浼樺厛绾т负priority鐨勪换鍔″钩鍧囩瓑寰呮椂闂?- 璁烘枃寮?2)
        numerator = sum(self.load_factors.get(p, 0) for p in range(1, priority + 1))
        
        # 鍒嗘瘝璁＄畻 - 娣诲姞鏁板€肩ǔ瀹氭€ф鏌?
        denominator1 = 1 - sum(self.load_factors.get(p, 0) for p in range(1, priority))
        denominator2 = 1 - sum(self.load_factors.get(p, 0) for p in range(1, priority + 1))
        
        # 闃叉鍒嗘瘝杩囧皬
        min_denominator = 1e-6
        if denominator1 <= min_denominator or denominator2 <= min_denominator:
            return float('inf')
        
        # 璁烘枃寮?2): T_wait = (1/渭) * [危蟻_i] / [(1-危蟻_{i<p})(1-危蟻_{i鈮})]
        waiting_time = (1 / self.service_rate) * (numerator / (denominator1 * denominator2))
        
        # 闄愬埗绛夊緟鏃堕棿鍦ㄥ悎鐞嗚寖鍥村唴
        max_waiting_time = 100.0  # 鏈€澶?00绉?
        waiting_time = min(waiting_time, max_waiting_time)
        
        return max(0.0, waiting_time)  # 纭繚闈炶礋
    
    def predict_waiting_time_instantaneous(self, task: Task) -> float:
        """
        鍩轰簬鐬椂闃熷垪鐘舵€侀娴嬬瓑寰呮椂闂?
        瀵瑰簲璁烘枃寮?4)鐨勭灛鏃剁Н鍘嬮娴?
        
        Args:
            task: 寰呴娴嬬殑浠诲姟
            
        Returns:
            棰勬祴绛夊緟鏃堕棿 (绉?
        """
        priority = task.priority
        
        # 璁＄畻褰撳墠姝ｅ湪鏈嶅姟鐨勪换鍔″墿浣欐椂闂?(绠€鍖?
        current_service_remaining = 0.0  # 瀹為檯涓渶瑕佽窡韪綋鍓嶆湇鍔′换鍔?
        
        # 璁＄畻浼樺厛绾ф洿楂樼殑浠诲姟鎬诲鐞嗘椂闂?
        higher_priority_workload = 0.0
        for p in range(1, priority):
            for (lifetime, prio), queue in self.queues.items():
                if prio == p and not queue.is_empty():
                    # 璁＄畻璇ラ槦鍒楃殑宸ヤ綔璐熻浇
                    queue_workload = queue.data_volume * config.task.task_compute_density
                    higher_priority_workload += queue_workload
        
        # 鍋囪鐨勫钩鍧嘋PU棰戠巼 (绠€鍖?
        avg_cpu_freq = self._get_average_cpu_frequency()
        
        # 鐬椂绛夊緟鏃堕棿棰勬祴 - 瀵瑰簲璁烘枃寮?4)
        if avg_cpu_freq > 0:
            waiting_time = (current_service_remaining + higher_priority_workload) / avg_cpu_freq
        else:
            waiting_time = float('inf')
        
        return waiting_time
    
    def _get_average_cpu_frequency(self) -> float:
        """鑾峰彇骞冲潎CPU棰戠巼 (绠€鍖栧疄鐜?"""
        if self.node_type == NodeType.VEHICLE:
            freq_range = config.compute.vehicle_cpu_freq_range
            return float(np.mean(freq_range))
        elif self.node_type == NodeType.RSU:
            return config.compute.rsu_cpu_freq
        else:  # UAV
            return config.compute.uav_cpu_freq
    
    def update_lifetime(self):
        """
        鏇存柊鎵€鏈変换鍔＄殑鐢熷懡鍛ㄦ湡
        姣忎釜鏃堕殭寮€濮嬫椂璋冪敤
        """
        new_queues = {}
        dropped_tasks = []
        
        for (lifetime, priority), queue in self.queues.items():
            if queue.is_empty():
                # 淇濇寔绌洪槦鍒楃粨鏋?
                new_queues[(lifetime, priority)] = QueueSlot(lifetime, priority)
            else:
                # 璁＄畻鏂扮殑鐢熷懡鍛ㄦ湡
                new_lifetime = max(0, lifetime - 1)
                
                if new_lifetime > 0:
                    # 浠诲姟绉诲姩鍒版柊鐨勭敓鍛藉懆鏈熼槦鍒?
                    new_key = (new_lifetime, priority)
                    if new_key not in new_queues:
                        new_queues[new_key] = QueueSlot(new_lifetime, priority)
                    
                    # 绉诲姩鎵€鏈変换鍔?
                    for task in queue.task_list:
                        new_queues[new_key].add_task(task)
                else:
                    # 鐢熷懡鍛ㄦ湡鐢ㄥ敖锛屼换鍔¤涓㈠純
                    for task in queue.task_list:
                        task.is_dropped = True
                        dropped_tasks.append(task)
                        self.current_usage -= task.data_size
                        self.statistics.total_drops += 1
        
        # 纭繚鎵€鏈夐槦鍒椾綅缃兘鏈夊搴旂殑闃熷垪瀵硅薄
        self._ensure_all_queues_exist(new_queues)
        
        self.queues = new_queues
        
        return dropped_tasks
    
    def _ensure_all_queues_exist(self, queue_dict: Dict):
        """纭繚鎵€鏈夐槦鍒椾綅缃兘瀛樺湪"""
        if self.node_type == NodeType.VEHICLE:
            lifetime_range = range(1, self.max_lifetime + 1)
        else:
            lifetime_range = range(1, self.max_lifetime)
        
        for lifetime in lifetime_range:
            for priority in range(1, self.num_priorities + 1):
                key = (lifetime, priority)
                if key not in queue_dict:
                    queue_dict[key] = QueueSlot(lifetime, priority)
    
    def _handle_queue_overflow(self, task: Task):
        """澶勭悊闃熷垪婧㈠嚭"""
        # 灏濊瘯閫氳繃涓㈠純浣庝紭鍏堢骇浠诲姟鏉ヨ吘鍑虹┖闂?
        freed_space = self._drop_low_priority_tasks(task.data_size)
        
        if freed_space >= task.data_size:
            # 鎴愬姛鑵惧嚭绌洪棿锛岄噸鏂板皾璇曟坊鍔?
            self.add_task(task)
        else:
            # 鏃犳硶鑵惧嚭瓒冲绌洪棿锛屼涪寮冨綋鍓嶄换鍔?
            self.statistics.total_drops += 1
    
    def _drop_low_priority_tasks(self, required_space: float) -> float:
        """涓㈠純浣庝紭鍏堢骇浠诲姟浠ヨ吘鍑虹┖闂?""
        freed_space = 0.0
        
        # 浠庢渶浣庝紭鍏堢骇寮€濮嬩涪寮?
        for priority in range(self.num_priorities, 0, -1):
            if freed_space >= required_space:
                break
            
            for lifetime in range(self.max_lifetime, 0, -1):
                queue_key = (lifetime, priority)
                if queue_key in self.queues and not self.queues[queue_key].is_empty():
                    queue = self.queues[queue_key]
                    
                    # 涓㈠純闃熷垪涓殑浠诲姟
                    while not queue.is_empty() and freed_space < required_space:
                        task = queue.task_list.pop()
                        freed_space += task.data_size
                        self.current_usage -= task.data_size
                        task.is_dropped = True
                        self.statistics.total_drops += 1
                        
                        # 鏇存柊闃熷垪鏁版嵁閲?
                        queue.data_volume -= task.data_size
        
        return freed_space
    
    def _update_arrival_statistics(self, priority: int):
        """鏇存柊鍒拌揪鐜囩粺璁?""
        # 璁板綍褰撳墠鏃堕殭鐨勫埌杈?
        current_slot_arrivals = defaultdict(int)
        current_slot_arrivals[priority] += 1
        
        self.recent_arrivals.append(dict(current_slot_arrivals))
        
        # 闄愬埗鍘嗗彶闀垮害
        if len(self.recent_arrivals) > self.time_window_size:
            self.recent_arrivals.pop(0)
        
        # 璁＄畻鍒拌揪鐜?
        self._calculate_arrival_rates()
    
    def _update_service_statistics(self):
        """鏇存柊鏈嶅姟鐜囩粺璁?""
        # 璁板綍褰撳墠鏃堕殭鐨勬湇鍔?
        self.recent_services.append(1)
        
        # 闄愬埗鍘嗗彶闀垮害
        if len(self.recent_services) > self.time_window_size:
            self.recent_services.pop(0)
        
        # 璁＄畻鏈嶅姟鐜?
        self._calculate_service_rate()
    
    def _calculate_arrival_rates(self):
        """璁＄畻鍚勪紭鍏堢骇鐨勫埌杈剧巼"""
        if not self.recent_arrivals:
            return
        
        window_duration = len(self.recent_arrivals) * config.network.time_slot_duration
        
        for priority in range(1, self.num_priorities + 1):
            total_arrivals = sum(arrivals.get(priority, 0) for arrivals in self.recent_arrivals)
            self.arrival_rates[priority] = total_arrivals / window_duration
    
    def _calculate_service_rate(self):
        """璁＄畻鏈嶅姟鐜?""
        if not self.recent_services:
            return
        
        window_duration = len(self.recent_services) * config.network.time_slot_duration
        total_services = sum(self.recent_services)
        self.service_rate = total_services / window_duration
        
        # 鏇存柊璐熻浇鍥犲瓙
        for priority in range(1, self.num_priorities + 1):
            if self.service_rate > 0:
                self.load_factors[priority] = self.arrival_rates[priority] / self.service_rate
            else:
                self.load_factors[priority] = 0.0
    
    def get_queue_state_vector(self) -> np.ndarray:
        """鑾峰彇闃熷垪鐘舵€佸悜閲?""
        state_features = []
        
        # 鍩烘湰闃熷垪淇℃伅
        state_features.extend([
            self.current_usage / self.max_capacity,  # 瀹归噺鍒╃敤鐜?
            len([q for q in self.queues.values() if not q.is_empty()]) / len(self.queues),  # 娲昏穬闃熷垪姣斾緥
            sum(self.load_factors.values()),  # 鎬昏礋杞藉洜瀛?
            self.service_rate / 100.0,  # 褰掍竴鍖栨湇鍔＄巼
        ])
        
        # 鍚勪紭鍏堢骇闃熷垪鐘舵€?
        for priority in range(1, self.num_priorities + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            priority_data = sum(queue.data_volume 
                              for (l, p), queue in self.queues.items() 
                              if p == priority)
            
            state_features.extend([
                priority_tasks / 50.0,  # 褰掍竴鍖栦换鍔℃暟
                priority_data / self.max_capacity,  # 褰掍竴鍖栨暟鎹噺
                self.arrival_rates.get(priority, 0.0) / 10.0,  # 褰掍竴鍖栧埌杈剧巼
                self.load_factors.get(priority, 0.0),  # 璐熻浇鍥犲瓙
            ])
        
        return np.array(state_features, dtype=np.float32)
    
    def get_queue_statistics(self) -> Dict:
        """鑾峰彇闃熷垪缁熻淇℃伅"""
        total_requests = self.statistics.total_arrivals + self.statistics.total_drops
        
        return {
            'total_arrivals': self.statistics.total_arrivals,
            'total_departures': self.statistics.total_departures,
            'total_drops': self.statistics.total_drops,
            'drop_rate': self.statistics.total_drops / max(1, total_requests),
            'capacity_utilization': self.current_usage / self.max_capacity,
            'total_load_factor': sum(self.load_factors.values()),
            'service_rate': self.service_rate,
            'arrival_rates_by_priority': dict(self.arrival_rates),
            'load_factors_by_priority': dict(self.load_factors),
            'active_queues': len([q for q in self.queues.values() if not q.is_empty()]),
            'avg_queue_length': np.mean([len(q.task_list) for q in self.queues.values()]),
        }
    
    def is_stable(self) -> bool:
        """妫€鏌ラ槦鍒楃郴缁熸槸鍚︾ǔ瀹?""
        total_load = sum(self.load_factors.values())
        return total_load < config.queue.max_load_factor
    
    def get_utilization(self) -> float:
        """鑾峰彇闃熷垪鍒╃敤鐜?""
        return self.current_usage / self.max_capacity
    
    def get_priority_distribution(self) -> Dict[int, float]:
        """鑾峰彇鍚勪紭鍏堢骇浠诲姟鍒嗗竷"""
        total_tasks = sum(len(queue.task_list) for queue in self.queues.values())
        
        if total_tasks == 0:
            return {p: 0.0 for p in range(1, self.num_priorities + 1)}
        
        distribution = {}
        for priority in range(1, self.num_priorities + 1):
            priority_tasks = sum(len(queue.task_list) 
                               for (l, p), queue in self.queues.items() 
                               if p == priority)
            distribution[priority] = priority_tasks / total_tasks
        
        return distribution
