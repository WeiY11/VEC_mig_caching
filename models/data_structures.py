"""
VEC系统核心数据结构
对应论文第2节系统模型中的基本组件定义
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from enum import Enum
import uuid
import time


class TaskType(Enum):
    """任务类型枚举 - 对应论文第3.1节任务分类框架"""
    EXTREMELY_DELAY_SENSITIVE = 1    # 极度延迟敏感型: T_max,j ≤ τ₁
    DELAY_SENSITIVE = 2              # 延迟敏感型: τ₁ < T_max,j ≤ τ₂  
    MODERATELY_DELAY_TOLERANT = 3    # 中度延迟容忍型: τ₂ < T_max,j ≤ τ₃
    DELAY_TOLERANT = 4               # 延迟容忍型: T_max,j > τ₃


class NodeType(Enum):
    """节点类型枚举"""
    VEHICLE = "vehicle"
    RSU = "rsu"
    UAV = "uav"


@dataclass
class Task:
    """
    计算任务类 - 对应论文第2.1节任务模型
    每个任务具有论文中定义的属性: D_j, C_j, c, S_j, T_max,j, λ'_j, v_j
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # 基本属性 - 论文第2.1节
    data_size: float = 0.0              # D_j: 任务输入数据大小 (bits)
    compute_cycles: float = 0.0         # C_j: 任务处理所需计算量 (CPU cycles)
    result_size: float = 0.0            # S_j: 任务输出结果大小 (bits)
    max_delay_slots: int = 0            # T_max,j: 任务最大可容忍延迟 (时隙数)
    
    # 任务分类与优先级
    task_type: TaskType = TaskType.DELAY_TOLERANT
    priority: int = 1                   # 优先级 (1=最高, P=最低)
    
    # 生成与调度信息
    source_vehicle_id: str = ""         # v_j: 生成任务的源车辆
    generation_time: float = 0.0       # 任务生成时间戳
    deadline: float = 0.0               # 任务截止时间
    
    # 🆕 内容相关属性（协作缓存支持）
    content_id: Optional[str] = None    # 内容ID（可缓存任务需要）
    is_cacheable: bool = False          # 是否可缓存
    scenario_name: Optional[str] = None # 场景名称（如'emergency_brake'）
    
    # 执行状态
    assigned_node_id: Optional[str] = None      # 分配的执行节点
    start_time: Optional[float] = None          # 开始执行时间
    completion_time: Optional[float] = None     # 完成时间
    is_completed: bool = False
    is_dropped: bool = False
    
    # 时延记录 - 用于性能分析
    transmission_delays: Dict[str, float] = field(default_factory=dict)
    waiting_delay: float = 0.0
    processing_delay: float = 0.0
    queue_arrival_time: Optional[float] = None   # 进入队列时间戳
    cache_last_access_time: Optional[float] = None
    cache_access_count: int = 0
    
    @property
    def compute_density(self) -> float:
        """计算密度 c = C_j / D_j (cycles/bit)"""
        return self.compute_cycles / self.data_size if self.data_size > 0 else 0.0
    
    @property
    def total_delay(self) -> float:
        """总端到端时延"""
        if self.completion_time is not None and self.generation_time > 0:
            return self.completion_time - self.generation_time
        return 0.0
    
    @property
    def remaining_lifetime_slots(self) -> int:
        """剩余生命周期 (时隙数)"""
        if self.deadline > 0:
            current_time = time.time()
            remaining_time = self.deadline - current_time
            from config import config
            return max(0, int(remaining_time / config.network.time_slot_duration))
        return 0
    
    def is_deadline_violated(self) -> bool:
        """检查是否违反截止时间"""
        return time.time() > self.deadline
    
    def __post_init__(self):
        """初始化后处理"""
        if self.generation_time == 0.0:
            self.generation_time = time.time()
        
        if self.deadline == 0.0:
            from config import config
            self.deadline = (self.generation_time + 
                           self.max_delay_slots * config.network.time_slot_duration)


@dataclass
class QueueSlot:
    """
    队列槽位类 - 对应论文第2.3节多优先级生命周期队列模型
    每个槽位对应特定的生命周期l和优先级p
    """
    lifetime: int                   # l: 剩余生命周期
    priority: int                   # p: 优先级等级
    data_volume: float = 0.0        # 队列中的数据量 (bits)
    task_list: List[Task] = field(default_factory=list)
    
    def add_task(self, task: Task) -> bool:
        """向队列槽位添加任务"""
        self.task_list.append(task)
        self.data_volume += task.data_size
        return True
    
    def remove_task(self, task_id: str) -> Optional[Task]:
        """从队列槽位移除任务"""
        for i, task in enumerate(self.task_list):
            if task.task_id == task_id:
                removed_task = self.task_list.pop(i)
                self.data_volume -= removed_task.data_size
                return removed_task
        return None
    
    def get_next_task(self) -> Optional[Task]:
        """获取下一个待处理任务 (FIFO)"""
        if self.task_list:
            return self.task_list[0]
        return None
    
    def is_empty(self) -> bool:
        """检查队列槽位是否为空"""
        return len(self.task_list) == 0


@dataclass  
class Position:
    """位置信息类"""
    x: float = 0.0
    y: float = 0.0  
    z: float = 0.0  # UAV使用，车辆和RSU通常为0
    
    def distance_to(self, other: 'Position') -> float:
        """计算到另一个位置的距离"""
        from utils import calculate_3d_distance
        return calculate_3d_distance((self.x, self.y, self.z), 
                                   (other.x, other.y, other.z))
    
    def distance_2d_to(self, other: 'Position') -> float:
        """计算2D距离 (忽略高度)"""
        from utils import calculate_distance
        return calculate_distance((self.x, self.y), (other.x, other.y))


@dataclass
class NodeState:
    """节点状态信息类"""
    # 基本信息
    node_id: str
    node_type: NodeType
    position: Position = field(default_factory=Position)
    
    # 计算资源
    cpu_frequency: float = 0.0          # f_n: 计算能力 (cycles/秒)
    cpu_utilization: float = 0.0        # CPU利用率 (0-1)
    is_active: bool = True              # 节点是否激活
    
    # 通信资源
    tx_power: float = 0.0               # P_tx: 发射功率 (W)
    available_bandwidth: float = 0.0    # 可用带宽 (Hz)
    
    # 能耗信息
    current_power: float = 0.0          # 当前功耗 (W)
    total_energy: float = 0.0           # 累计能耗 (J)
    
    # UAV特有属性
    battery_level: float = 1.0          # 电池电量 (0-1)
    hover_power: float = 0.0            # 悬停功耗 (W)
    
    # 负载统计
    load_factor: float = 0.0            # 负载因子 ρ
    queue_length: int = 0               # 队列长度
    avg_waiting_time: float = 0.0       # 平均等待时间
    stability_warning: bool = False     # 队列稳定性告警标志

    def update_utilization(self, active_time: float, total_time: float):
        """更新CPU利用率"""
        if total_time > 0:
            self.cpu_utilization = min(1.0, active_time / total_time)
    
    def update_load_factor(self, arrival_rate: float, service_rate: float):
        """更新负载因子 ρ = λ/μ"""
        if service_rate > 0:
            self.load_factor = arrival_rate / service_rate
    
    def is_overloaded(self, threshold: float = 0.8) -> bool:
        """检查节点是否过载"""
        return self.load_factor > threshold
    
    def get_remaining_capacity(self) -> float:
        """获取剩余处理能力"""
        return max(0.0, 1.0 - self.cpu_utilization)


@dataclass
class CommunicationLink:
    """通信链路类 - 对应论文第5.2节无线通信模型"""
    source_id: str
    destination_id: str
    
    # 信道参数
    distance: float = 0.0               # 节点间距离 (m)
    channel_gain: float = 0.0           # h: 信道增益 (线性值)
    path_loss_db: float = 0.0           # 路径损耗 (dB)
    los_probability: float = 0.0        # 视距概率
    
    # 通信质量
    sinr_db: float = 0.0                # 信噪干扰比 (dB)  
    data_rate: float = 0.0              # R: 传输速率 (bps)
    allocated_bandwidth: float = 0.0    # 分配的带宽 (Hz)
    
    # 时延参数
    propagation_delay: float = 0.0      # 传播时延 (s)
    transmission_delay: float = 0.0     # 传输时延 (s)
    processing_delay: float = 0.0       # 处理时延 (s)
    
    def update_channel_state(self, new_distance: float, 
                           interference_power: float,
                           noise_power: float):
        """更新信道状态"""
        self.distance = new_distance
        # 这里会调用论文第5.2节的信道模型计算
        # 具体实现在communication模块中
        pass
    
    @property
    def total_delay(self) -> float:
        """总通信时延"""
        return self.propagation_delay + self.transmission_delay + self.processing_delay
    
    @property
    def sinr_linear(self) -> float:
        """线性SINR值"""
        from utils import db_to_linear
        return db_to_linear(self.sinr_db)


@dataclass
class SystemMetrics:
    """系统性能指标类"""
    # 时延指标
    avg_task_delay: float = 0.0         # 平均任务时延
    max_task_delay: float = 0.0         # 最大任务时延
    delay_violation_rate: float = 0.0   # 时延违约率
    
    # 能耗指标  
    total_energy_consumption: float = 0.0    # 总能耗
    avg_energy_per_task: float = 0.0         # 每任务平均能耗
    
    # 数据丢失指标
    total_data_loss: float = 0.0        # 总数据丢失量 (bits)
    data_loss_rate: float = 0.0         # 数据丢失率
    
    # 系统效率指标
    task_completion_rate: float = 0.0   # 任务完成率
    cache_hit_rate: float = 0.0         # 缓存命中率
    migration_success_rate: float = 0.0 # 迁移成功率
    avg_queue_utilization: float = 0.0  # 平均队列利用率
    
    # 资源利用率
    avg_cpu_utilization: float = 0.0    # 平均CPU利用率
    avg_bandwidth_utilization: float = 0.0  # 平均带宽利用率
    
    def update_delay_metrics(self, completed_tasks: List[Task]):
        """更新时延相关指标"""
        if not completed_tasks:
            return
        
        delays = [task.total_delay for task in completed_tasks]
        # 过滤掉无效值（inf, nan）
        valid_delays = [d for d in delays if np.isfinite(d) and d >= 0]
        
        if valid_delays:
            self.avg_task_delay = float(np.mean(valid_delays))
            self.max_task_delay = float(np.max(valid_delays))
        else:
            # 如果没有有效的延迟值，设置为默认值
            self.avg_task_delay = 0.0
            self.max_task_delay = 0.0
        
        # 计算时延违约率
        violated_tasks = [task for task in completed_tasks 
                         if task.is_deadline_violated()]
        self.delay_violation_rate = len(violated_tasks) / len(completed_tasks)
    
    def update_energy_metrics(self, total_energy: float, num_tasks: int):
        """更新能耗相关指标"""
        self.total_energy_consumption = total_energy
        if num_tasks > 0:
            self.avg_energy_per_task = total_energy / num_tasks
    
    def update_data_loss_metrics(self, dropped_tasks: List[Task], total_tasks: int):
        """更新数据丢失相关指标"""
        if dropped_tasks:
            self.total_data_loss = sum(task.data_size for task in dropped_tasks)
        
        if total_tasks > 0:
            self.data_loss_rate = len(dropped_tasks) / total_tasks
    
    def get_weighted_cost(self, weight_delay: float = 0.4, 
                         weight_energy: float = 0.3,
                         weight_loss: float = 0.3) -> float:
        """
        计算加权总成本 - 对应论文式(24)目标函数
        """
        normalized_delay = self.avg_task_delay / 1.0  # 归一化时延
        normalized_energy = self.total_energy_consumption / 1000.0  # 归一化能耗
        normalized_loss = self.data_loss_rate  # 数据丢失率已是[0,1]范围
        
        return (weight_delay * normalized_delay + 
                weight_energy * normalized_energy + 
                weight_loss * normalized_loss)
