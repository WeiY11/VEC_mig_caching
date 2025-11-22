"""
车辆节点实现 - 对应论文第2.1节和第5.1节
实现车辆的本地计算、任务生成和移动模型
"""
import numpy as np
import time
import math
from typing import List, Optional, Tuple

from .base_node import BaseNode
from .data_structures import Task, Position, NodeType, TaskType
from config import config
from utils import (
    generate_poisson_arrivals, 
    sample_zipf_content_id, 
    sample_heavy_tailed_task_size
)


class VehicleNode(BaseNode):
    """
    车辆节点类 - 对应论文车辆模型 v ∈ V
    
    主要功能:
    1. 任务生成 (按泊松过程)
    2. 本地计算处理
    3. 车辆移动模拟
    4. 能耗模型 (对应论文式5-9)
    """
    
    def __init__(self, vehicle_id: str, initial_position: Position):
        super().__init__(vehicle_id, NodeType.VEHICLE, initial_position)
        
        # 车辆特有属性
        self.velocity = np.array([0.0, 0.0])  # 速度向量 (m/s)
        self.max_speed = 60.0  # 最大速度 (m/s) - 高速公路场景 (216 km/h)
        self.trajectory: List[Position] = [initial_position]
        
        # 计算资源配置 - 对应论文第5.1节
        self._setup_compute_resources()
        
        # 任务生成参数
        self.task_generation_rate = config.task.arrival_rate
        self.generated_tasks: List[Task] = []
        
        # 能耗模型参数 - 论文式(5)
        self.kappa1 = config.compute.vehicle_kappa1
        self.kappa2 = config.compute.vehicle_kappa2  
        self.static_power = config.compute.vehicle_static_power
        self.idle_power = config.compute.vehicle_static_power
        
        # 传输功率
        self.state.tx_power = config.communication.vehicle_tx_power
    
    def _setup_compute_resources(self):
        """设置车辆计算资源"""
        # 从配置范围中随机选择CPU频率
        freq_range = config.compute.vehicle_cpu_freq_range
        self.state.cpu_frequency = np.random.uniform(freq_range[0], freq_range[1])
        
        # 设置可用带宽
        self.state.available_bandwidth = config.communication.total_bandwidth / config.network.num_vehicles
    
    def get_processing_capacity(self) -> float:
        """
        鑾峰彇杞﹁締鏈湴澶勭悊鑳藉姏 (bytes/鏃堕殭) - 瀵瑰簲璁烘枃寮?5)
        D^local_n = (f_n * 螖t) / c
        """
        delta_t = config.network.time_slot_duration
        compute_density = config.task.task_compute_density
        parallel_efficiency = config.compute.parallel_efficiency
        bits_capacity = (self.state.cpu_frequency * delta_t * parallel_efficiency) / compute_density
        return bits_capacity / 8.0

    def calculate_processing_delay(self, task: Task) -> float:
        """
        计算本地处理时延 - 对应论文式(6)
        T_comp,j,n = C_j / (f_n * η_parallel)
        """
        parallel_efficiency = config.compute.parallel_efficiency
        return task.compute_cycles / (self.state.cpu_frequency * parallel_efficiency)
    
    def calculate_energy_consumption(self, processing_time: float) -> float:
        """
        璁＄畻澶勭悊鑳借€?- 瀵瑰簲璁烘枃寮?7)-(9)
        P^comp_n(f_n, U_n) = 魏鈧乫_n鲁 + 魏鈧俧_n虏U_n + P_static
        """
        slot_duration = max(config.network.time_slot_duration, 1e-9)
        utilization = min(1.0, processing_time / slot_duration)
        
        # 鍔ㄦ€佸姛鐜囨ā鍨?- 瀵瑰簲璁烘枃寮?7)
        dynamic_power = (self.kappa1 * (self.state.cpu_frequency ** 3) +
                        self.kappa2 * (self.state.cpu_frequency ** 2) * utilization +
                        self.static_power)
        
        total_energy = dynamic_power * processing_time
        
        # 鏇存柊鍔熻€楃姸鎬?
        self.state.current_power = dynamic_power
        
        return total_energy

    def generate_tasks(self, current_time_slot: int) -> List[Task]:
        """
        生成新任务 - 按泊松过程到达
        对应论文第2.1节任务模型
        """
        # 按泊松过程生成任务数量
        num_tasks = generate_poisson_arrivals(
            self.task_generation_rate, 
            config.network.time_slot_duration
        )
        
        new_tasks = []
        for _ in range(num_tasks):
            task = self._create_random_task()
            new_tasks.append(task)
            self.generated_tasks.append(task)
        
        return new_tasks
    
    def _create_random_task(self) -> Task:
        """
        创建随机任务 - 优化版
        
        【功能】基于场景生成真实任务，支持：
        1. 场景化任务生成（8种应用场景）
        2. 重尾数据大小分布（帕累托分布）
        3. Zipf内容热度分布（协作缓存）
        4. 合理的属性组合（避免不可能完成的任务）
        
        【改进说明】
        - ✅ 基于场景抽样：按权重选择应用场景
        - ✅ 重尾分布：数据大小符合帕累托分布（大量小任务+少量大任务）
        - ✅ Zipf热度：内容访问符合Zipf分布（少数热门+大量冷门）
        - ✅ 属性合理：紧急任务小数据+短期限，容忍任务大数据+长期限
        
        【论文对应】
        - 场景定义：Section 2.1 "Task Model"
        - Zipf分布：Section 2.7 "Collaborative Caching"
        - 重尾分布：真实任务流量特征
        """
        # ========== 步骤1：基于权重选择应用场景 ==========
        scenario = config.task.sample_scenario()
        scenario_name = scenario.name
        task_type_value = scenario.task_type
        
        # ========== 步骤2：获取场景规格和参数 ==========
        profile = config.task.get_profile(task_type_value)
        data_range = profile.data_range
        compute_density = profile.compute_density
        
        # ========== 步骤3：生成数据大小（重尾分布） ==========
        # 使用帕累托分布模拟真实任务大小分布
        # shape=1.5: 温和的重尾（约70%小任务，30%中大任务）
        data_size = sample_heavy_tailed_task_size(
            min_size=data_range[0],
            max_size=data_range[1],
            shape=1.5  # 帕累托形状参数
        )
        
        # ========== 步骤4：计算任务计算量 ==========
        # 计算密度随场景类型变化
        # 类型1（紧急）：低计算密度（60 cycles/bit）
        # 类型4（分析）：高计算密度（150 cycles/bit）
        compute_cycles = data_size * 8 * compute_density  # 字节转比特
        result_size = data_size * config.task.task_output_ratio
        
        # ========== 步骤5：确定截止时间 ==========
        # 基于场景规格的截止时间范围
        deadline_seconds = np.random.uniform(
            scenario.min_deadline,
            scenario.max_deadline
        )
        
        # 转换为时隙数
        time_slot_duration = config.network.time_slot_duration
        max_delay_slots = max(1, int(deadline_seconds / time_slot_duration))
        
        # ========== 步骤6：动态调整任务分类 ==========
        # 考虑数据大小、计算量、系统负载等多维特征
        system_load = self.state.load_factor if hasattr(self, 'state') else None
        
        # 🔧 修复：扩大可缓存任务范范围以提升缓存命中率
        # 判断任务是否可缓存 - 基于VEC应用实际情况
        # 类型1（紧急）: 50%可缓存（安全警报、紧急通知等区域相关信息）
        # 类型2（导航）: 80%可缓存（地图、POI、路线信息）
        # 类型3（视频）: 90%可缓存（流媒体、娱乐内容）
        # 类型4（分析）: 85%可缓存（数据分析结果、计算结果）
        cacheable_probs = {
            1: 0.50,  # 紧急任务：部分可缓存
            2: 0.80,  # 导航任务：高度可缓存
            3: 0.90,  # 视频任务：最可缓存
            4: 0.85   # 分析任务：高度可缓存
        }
        is_cacheable = np.random.random() < cacheable_probs.get(task_type_value, 0.5)
        
        task_type_adjusted = config.task.get_task_type(
            max_delay_slots=max_delay_slots,
            data_size=data_size,
            compute_cycles=compute_cycles,
            compute_density=compute_density,
            time_slot=time_slot_duration,
            system_load=system_load,
            is_cacheable=is_cacheable
        )
        task_type = TaskType(task_type_adjusted)
        
        # ========== 步骤7：分配优先级 ==========
        # 优先级与任务类型相关 (类型值小的优先级高)
        # 类型1：优先级1-2（最高）
        # 类型2：优先级2-3
        # 类型3：优先级3-4
        # 类型4：优先级4（最低）
        priority = task_type_adjusted + np.random.randint(0, 2)
        priority = min(priority, config.task.num_priority_levels)
        
        # ========== 步骤8：生成内容ID（Zipf分布） ==========
        content_id = None
        if is_cacheable:
            # 可缓存任务：按Zipf分布选择内容
            # 内容库大小1000，Zipf指数0.8
            content_id = sample_zipf_content_id(num_contents=1000, exponent=0.8)
        
        # ========== 步骤9：创建任务对象 ==========
        task = Task(
            data_size=data_size,
            compute_cycles=compute_cycles,
            result_size=result_size,
            max_delay_slots=max_delay_slots,
            task_type=task_type,
            priority=priority,
            source_vehicle_id=self.node_id,
            content_id=content_id,
            is_cacheable=is_cacheable,
            scenario_name=scenario_name
        )
        
        return task
    
    def update_position(self, time_step: float):
        """
        更新车辆位置
        简单的移动模型：匀速直线运动，到边界时转向
        """
        # 更新位置
        new_x = self.state.position.x + self.velocity[0] * time_step
        new_y = self.state.position.y + self.velocity[1] * time_step
        
        # 边界检查和转向
        area_width = config.network.area_width
        area_height = config.network.area_height
        
        if new_x <= 0 or new_x >= area_width:
            self.velocity[0] = -self.velocity[0]
            new_x = max(0, min(area_width, new_x))
        
        if new_y <= 0 or new_y >= area_height:
            self.velocity[1] = -self.velocity[1]
            new_y = max(0, min(area_height, new_y))
        
        # 更新位置
        self.state.position.x = new_x
        self.state.position.y = new_y
        
        # 记录轨迹
        self.trajectory.append(Position(new_x, new_y, 0))
        
        # 限制轨迹历史长度
        if len(self.trajectory) > 100:
            self.trajectory.pop(0)
    
    def set_random_velocity(self):
        """设置随机速度"""
        # 随机方向
        angle = np.random.uniform(0, 2 * math.pi)
        # 随机速度大小 (30-60 m/s，高速公路场景)
        speed = np.random.uniform(30.0, self.max_speed)
        
        self.velocity[0] = speed * math.cos(angle)
        self.velocity[1] = speed * math.sin(angle)
    
    def step(self, time_step: float) -> Tuple[List[Task], List[Task]]:
        """
        车辆节点单步更新
        
        Returns:
            (新生成的任务列表, 本地处理完成的任务列表)
        """
        # 1. 更新位置
        self.update_position(time_step)
        
        # 2. 更新队列生命周期
        self.update_queue_lifetimes()
        
        # 3. 生成新任务
        current_slot = int(time.time() / config.network.time_slot_duration)
        new_tasks = self.generate_tasks(current_slot)
        
        # 4. 处理本地队列中的任务
        processed_tasks = []
        
        # 获取本时隙可处理的数据量
        processing_capacity = self.get_processing_capacity()
        remaining_capacity = processing_capacity
        
        # 注意：新生成的任务不在这里直接处理
        # 它们将被送到卸载决策器进行决策
        
        # 处理队列中的任务
        while remaining_capacity > 0:
            next_task = self.get_next_task_to_process()
            if next_task is None:
                break
            
            if next_task.data_size <= remaining_capacity:
                if self.process_task(next_task):
                    processed_tasks.append(next_task)
                    remaining_capacity -= next_task.data_size
                else:
                    break
            else:
                # 任务太大，无法在本时隙完成
                break
        
        # 5. 鏇存柊缁熻淇℃伅
        self._update_statistics()
        
        # 6. 鎺掗櫎绌洪棿鏃堕棿鍐呮墍闇€鐨勫€奸噺
        self._apply_idle_energy(time_step, processed_tasks)
        
        return new_tasks, processed_tasks
    
    def _apply_idle_energy(self, time_step: float, processed_tasks: List[Task]) -> None:
        """缁熻姣忛殭鍚庤鍙戣捣鐨勫€奸噺锛屼负闈欓噷闂撮潤鎸?"""
        total_processing_time = sum(getattr(task, 'processing_delay', 0.0) for task in processed_tasks)
        idle_time = max(0.0, time_step - total_processing_time)
        if idle_time <= 0 or self.idle_power <= 0:
            return
        idle_energy = self.idle_power * idle_time
        self._record_energy_usage(idle_energy)

    def can_process_immediately(self, task: Task) -> bool:
        """
        检查是否能立即本地处理任务
        
        Args:
            task: 要检查的任务
            
        Returns:
            bool: 是否能立即处理
        """
        # 检查CPU资源是否足够
        processing_capacity = self.get_processing_capacity()
        
        # 检查当前负载
        current_load = self.state.load_factor
        
        # 如果当前负载过高，无法立即处理
        if current_load > 0.9:
            return False
        
        # 检查任务大小是否在处理能力范围内
        if task.data_size > processing_capacity:
            return False
        
        return True
    
    def process_task_immediately(self, task: Task) -> Tuple[bool, float]:
        """
        立即处理任务
        
        Args:
            task: 要处理的任务
            
        Returns:
            Tuple[bool, float]: (是否成功, 处理延迟)
        """
        if not self.can_process_immediately(task):
            return False, 0.0
        
        # 计算处理延迟
        processing_delay = self.calculate_processing_delay(task)
        
        # 检查是否满足延迟要求
        max_allowed_delay = task.max_delay_slots * config.network.time_slot_duration
        if processing_delay > max_allowed_delay:
            return False, 0.0
        
        # 更新负载状态
        time_slot_duration = config.network.time_slot_duration
        load_increase = processing_delay / time_slot_duration
        self.state.load_factor = min(1.0, self.state.load_factor + load_increase)
        
        # 更新统计信息 - 使用基类的统计方法
        if hasattr(self, 'statistics'):
            self.statistics['processed_tasks'] = self.statistics.get('processed_tasks', 0) + 1

        

        return True, processing_delay

