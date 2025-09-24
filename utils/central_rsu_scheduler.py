#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
中央RSU骨干调度系统
基于现有中央RSU实现全局负载收集与任务调度
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
import time
import logging


@dataclass
class RSULoadInfo:
    """RSU负载信息数据结构"""
    rsu_id: str
    position: np.ndarray
    
    # 队列状态
    queue_length: int = 0
    queue_tasks: List[Dict] = field(default_factory=list)
    
    # 计算资源
    cpu_usage: float = 0.0
    cpu_frequency: float = 0.0
    available_compute: float = 0.0
    
    # 缓存状态
    cache_usage: float = 0.0
    cache_hit_rate: float = 0.0
    cached_content_count: int = 0
    
    # 网络状态
    served_vehicles: int = 0
    coverage_vehicles: int = 0
    network_bandwidth_usage: float = 0.0
    
    # 性能指标
    avg_response_time: float = 0.0
    task_completion_rate: float = 0.0
    energy_consumption: float = 0.0
    
    # 时间戳
    last_updated: float = field(default_factory=time.time)


@dataclass
class GlobalSchedulingDecision:
    """全局调度决策"""
    target_rsu_id: str
    task_allocation_ratio: float  # 分配给该RSU的任务比例
    priority_level: int          # 优先级 (1-5, 5最高)
    expected_response_time: float
    reason: str                  # 调度原因


class CentralRSUScheduler:
    """🏢 中央RSU骨干调度系统"""
    
    def __init__(self, central_rsu_id: str = "RSU_2", history_window: int = 20):
        """
        初始化中央RSU调度器
        
        Args:
            central_rsu_id: 中央RSU的ID (通常是RSU_2，位于中央位置)
            history_window: 负载历史记录窗口大小
        """
        self.central_rsu_id = central_rsu_id
        self.history_window = history_window
        
        # 📊 RSU负载信息收集
        self.rsu_loads: Dict[str, RSULoadInfo] = {}
        self.load_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        
        # 🎯 全局调度决策
        self.scheduling_decisions: Dict[str, GlobalSchedulingDecision] = {}
        self.task_allocation_matrix: np.ndarray = None  # RSU间任务分配矩阵
        
        # 📈 性能监控
        self.global_metrics = {
            'total_system_load': 0.0,
            'load_balance_index': 0.0,  # 负载均衡指数 (0-1, 1为完全均衡)
            'global_response_time': 0.0,
            'system_throughput': 0.0,
            'scheduling_decisions_count': 0,
            'successful_migrations': 0,
            'last_scheduling_time': 0.0
        }
        
        # ⚙️ 调度策略配置
        self.config = {
            'load_balance_threshold': 0.7,      # 负载均衡阈值
            'response_time_threshold': 100.0,   # 响应时间阈值(ms)
            'min_allocation_ratio': 0.1,        # 最小分配比例
            'max_allocation_ratio': 0.4,        # 最大分配比例
            'scheduling_interval': 1.0,         # 调度间隔(秒)
            'load_prediction_weight': 0.3,      # 负载预测权重
            'fairness_weight': 0.4,             # 公平性权重
            'efficiency_weight': 0.3            # 效率性权重
        }
        
        # 🧠 智能调度算法
        self.load_predictor = LoadPredictor()
        self.allocation_optimizer = AllocationOptimizer()
        
        logging.info(f"🏢 中央RSU调度器初始化完成，调度中心: {central_rsu_id}")
    
    def collect_rsu_load_info(self, rsu_data: Dict) -> RSULoadInfo:
        """
        🔍 收集单个RSU的负载信息
        
        Args:
            rsu_data: RSU状态数据字典
            
        Returns:
            RSULoadInfo: 结构化的负载信息
        """
        rsu_id = rsu_data.get('id', 'unknown')
        
        # 提取负载信息
        load_info = RSULoadInfo(
            rsu_id=rsu_id,
            position=np.array(rsu_data.get('position', [0, 0])),
            
            # 队列状态
            queue_length=len(rsu_data.get('computation_queue', [])),
            queue_tasks=rsu_data.get('computation_queue', []),
            
            # 计算资源
            cpu_usage=rsu_data.get('cpu_usage', 0.0),
            cpu_frequency=rsu_data.get('cpu_frequency', 0.0),
            available_compute=max(0, rsu_data.get('cpu_frequency', 0.0) * (1 - rsu_data.get('cpu_usage', 0.0))),
            
            # 缓存状态
            cache_usage=rsu_data.get('cache_usage', 0.0),
            cache_hit_rate=rsu_data.get('cache_hit_rate', 0.0),
            cached_content_count=len(rsu_data.get('cached_content', {})),
            
            # 网络状态
            served_vehicles=rsu_data.get('served_vehicles', 0),
            coverage_vehicles=rsu_data.get('coverage_vehicles', 0),
            network_bandwidth_usage=rsu_data.get('bandwidth_usage', 0.0),
            
            # 性能指标
            avg_response_time=rsu_data.get('avg_response_time', 0.0),
            task_completion_rate=rsu_data.get('task_completion_rate', 0.0),
            energy_consumption=rsu_data.get('energy_consumption', 0.0)
        )
        
        # 更新历史记录
        self.rsu_loads[rsu_id] = load_info
        self.load_history[rsu_id].append(load_info.cpu_usage)
        
        return load_info
    
    def collect_all_rsu_loads(self, rsu_list: List[Dict]) -> Dict[str, RSULoadInfo]:
        """
        📊 收集所有接入RSU的负载信息
        
        Args:
            rsu_list: 所有RSU的状态数据列表
            
        Returns:
            Dict[str, RSULoadInfo]: RSU负载信息字典
        """
        collected_loads = {}
        
        for rsu_data in rsu_list:
            rsu_id = rsu_data.get('id', 'unknown')
            
            # 跳过中央RSU自己（调度中心）
            if rsu_id == self.central_rsu_id:
                continue
                
            load_info = self.collect_rsu_load_info(rsu_data)
            collected_loads[rsu_id] = load_info
            
        # 计算全局指标
        self._update_global_metrics()
        
        logging.debug(f"📊 收集了 {len(collected_loads)} 个RSU的负载信息")
        return collected_loads
    
    def global_load_balance_scheduling(self, incoming_task_count: int = 1) -> Dict[str, GlobalSchedulingDecision]:
        """
        🎯 全局负载均衡调度算法
        
        Args:
            incoming_task_count: 即将到达的任务数量
            
        Returns:
            Dict[str, GlobalSchedulingDecision]: 全局调度决策
        """
        if not self.rsu_loads:
            logging.warning("⚠️ 无RSU负载信息，跳过调度")
            return {}
        
        # 1️⃣ 计算负载均衡指数
        load_balance_index = self._calculate_load_balance_index()
        
        # 2️⃣ 负载预测
        predicted_loads = self.load_predictor.predict_future_loads(self.load_history, steps=3)
        
        # 3️⃣ 优化任务分配
        allocation_matrix = self.allocation_optimizer.optimize_allocation(
            current_loads=self.rsu_loads,
            predicted_loads=predicted_loads,
            incoming_tasks=incoming_task_count,
            fairness_weight=self.config['fairness_weight'],
            efficiency_weight=self.config['efficiency_weight']
        )
        
        # 4️⃣ 生成调度决策
        scheduling_decisions = {}
        
        for rsu_id, load_info in self.rsu_loads.items():
            # 计算分配比例
            allocation_ratio = allocation_matrix.get(rsu_id, 0.0)
            
            # 限制分配范围
            allocation_ratio = np.clip(
                allocation_ratio,
                self.config['min_allocation_ratio'],
                self.config['max_allocation_ratio']
            )
            
            # 计算优先级
            priority = self._calculate_priority(load_info, predicted_loads.get(rsu_id, 0.0))
            
            # 预测响应时间
            expected_response_time = self._estimate_response_time(
                load_info, allocation_ratio * incoming_task_count
            )
            
            # 生成调度原因
            reason = self._generate_scheduling_reason(load_info, allocation_ratio, priority)
            
            # 创建调度决策
            decision = GlobalSchedulingDecision(
                target_rsu_id=rsu_id,
                task_allocation_ratio=allocation_ratio,
                priority_level=priority,
                expected_response_time=expected_response_time,
                reason=reason
            )
            
            scheduling_decisions[rsu_id] = decision
        
        # 5️⃣ 更新调度统计
        self.scheduling_decisions = scheduling_decisions
        self.global_metrics['scheduling_decisions_count'] += 1
        self.global_metrics['last_scheduling_time'] = time.time()
        
        logging.info(f"🎯 生成全局调度决策，目标RSU数量: {len(scheduling_decisions)}")
        return scheduling_decisions
    
    def intelligent_migration_coordination(self, overload_threshold: float = 0.8) -> List[Dict]:
        """
        🚀 智能迁移协调 - 基于全局视角的任务迁移
        
        Args:
            overload_threshold: 过载阈值
            
        Returns:
            List[Dict]: 迁移指令列表
        """
        migration_commands = []
        
        if not self.rsu_loads:
            return migration_commands
        
        # 🔍 识别过载和空闲节点
        overloaded_rsus = []
        underloaded_rsus = []
        
        for rsu_id, load_info in self.rsu_loads.items():
            load_factor = self._calculate_normalized_load(load_info)
            
            if load_factor > overload_threshold:
                overloaded_rsus.append((rsu_id, load_info, load_factor))
            elif load_factor < 0.3:  # 空闲阈值
                underloaded_rsus.append((rsu_id, load_info, load_factor))
        
        # ⚖️ 执行负载均衡迁移
        for source_rsu_id, source_load, source_factor in overloaded_rsus:
            # 选择最佳目标RSU
            if underloaded_rsus:
                # 按负载和距离选择目标
                target_candidates = []
                source_pos = source_load.position
                
                for target_rsu_id, target_load, target_factor in underloaded_rsus:
                    target_pos = target_load.position
                    distance = np.linalg.norm(source_pos - target_pos)
                    
                    # 综合评分：负载低 + 距离近
                    score = (1 - target_factor) * 0.7 + (1 / (distance + 1)) * 0.3
                    target_candidates.append((target_rsu_id, target_load, score))
                
                # 选择最佳目标
                if target_candidates:
                    best_target = max(target_candidates, key=lambda x: x[2])
                    target_rsu_id, target_load, _ = best_target
                    
                    # 计算迁移任务数量
                    migrate_count = max(1, int(source_load.queue_length * 0.3))
                    
                    # 🔌 计算有线传输成本
                    migration_data_size = migrate_count * 2.0  # MB per task
                    try:
                        from utils.wired_backhaul_model import get_backhaul_model
                        backhaul = get_backhaul_model()
                        cost_info = backhaul.estimate_migration_cost(
                            migration_data_size, source_rsu_id, target_rsu_id
                        )
                        wired_delay = cost_info['transmission_delay']
                        wired_energy = cost_info['energy_consumption']
                        total_cost = cost_info['total_cost']
                    except Exception:
                        # 回退到简化成本模型
                        wired_delay = 0.01   # 10ms
                        wired_energy = 0.5   # 0.5J
                        total_cost = 1.0
                    
                    # 🎯 评估迁移收益 (考虑有线传输成本)
                    load_benefit = source_factor - target_load.cpu_usage  # 负载均衡收益
                    transmission_cost = total_cost * 0.05  # 🔧 降低传输成本权重从0.1到0.05
                    net_benefit = load_benefit - transmission_cost
                    
                    # 🔧 降低收益阈值，更容易触发迁移
                    if net_benefit > 0.05:  # 从0.1降低到0.05
                        # 生成迁移指令
                        migration_cmd = {
                            'type': 'task_migration',
                            'source_rsu': source_rsu_id,
                            'target_rsu': target_rsu_id,
                            'task_count': migrate_count,
                            'urgency': 'high' if source_factor > 0.9 else 'medium',
                            'expected_benefit': net_benefit,
                            'wired_transmission': {
                                'data_size_mb': migration_data_size,
                                'delay_ms': wired_delay * 1000,
                                'energy_j': wired_energy,
                                'total_cost': total_cost
                            },
                            'coordination_time': time.time()
                        }
                        
                        migration_commands.append(migration_cmd)
                        
                        # 更新目标负载（预估）
                        target_load.queue_length += migrate_count
                        target_load.cpu_usage = min(1.0, target_load.cpu_usage + 0.1)
        
        logging.info(f"🚀 生成智能迁移协调指令: {len(migration_commands)} 条")
        return migration_commands
    
    def get_global_scheduling_status(self) -> Dict[str, Any]:
        """
        📈 获取全局调度状态报告
        
        Returns:
            Dict: 全局调度状态信息
        """
        status = {
            'central_rsu_id': self.central_rsu_id,
            'managed_rsu_count': len(self.rsu_loads),
            'global_metrics': self.global_metrics.copy(),
            'current_decisions': len(self.scheduling_decisions),
            'system_health': self._assess_system_health(),
            'load_distribution': self._get_load_distribution(),
            'scheduling_efficiency': self._calculate_scheduling_efficiency(),
            'timestamp': time.time()
        }
        
        return status
    
    # ==================== 私有方法 ====================
    
    def _calculate_load_balance_index(self) -> float:
        """计算负载均衡指数"""
        if not self.rsu_loads:
            return 0.0
        
        loads = [info.cpu_usage for info in self.rsu_loads.values()]
        if not loads:
            return 0.0
        
        mean_load = np.mean(loads)
        std_load = np.std(loads)
        
        # 负载均衡指数 = 1 - (标准差 / 均值)，值越大越均衡
        balance_index = max(0.0, 1.0 - (std_load / (mean_load + 1e-6)))
        return balance_index
    
    def _calculate_normalized_load(self, load_info: RSULoadInfo) -> float:
        """计算标准化负载因子"""
        # 综合CPU、队列和网络负载
        cpu_factor = load_info.cpu_usage
        queue_factor = min(1.0, load_info.queue_length / 10.0)  # 队列长度标准化
        network_factor = load_info.network_bandwidth_usage
        
        # 加权综合负载
        normalized_load = (cpu_factor * 0.5 + queue_factor * 0.3 + network_factor * 0.2)
        return min(1.0, normalized_load)
    
    def _calculate_priority(self, load_info: RSULoadInfo, predicted_load: float) -> int:
        """计算调度优先级 (1-5)"""
        current_load = self._calculate_normalized_load(load_info)
        
        # 综合当前负载和预测负载
        combined_load = current_load * 0.7 + predicted_load * 0.3
        
        if combined_load < 0.2:
            return 5  # 最高优先级，空闲节点
        elif combined_load < 0.4:
            return 4
        elif combined_load < 0.6:
            return 3
        elif combined_load < 0.8:
            return 2
        else:
            return 1  # 最低优先级，过载节点
    
    def _estimate_response_time(self, load_info: RSULoadInfo, additional_tasks: float) -> float:
        """预测响应时间"""
        # 基础响应时间
        base_time = 50.0  # ms
        
        # 队列延迟
        queue_delay = (load_info.queue_length + additional_tasks) * 10.0
        
        # CPU负载延迟
        cpu_delay = load_info.cpu_usage * 30.0
        
        # 网络延迟
        network_delay = load_info.network_bandwidth_usage * 20.0
        
        total_response_time = base_time + queue_delay + cpu_delay + network_delay
        return total_response_time
    
    def _generate_scheduling_reason(self, load_info: RSULoadInfo, allocation_ratio: float, priority: int) -> str:
        """生成调度原因说明"""
        reasons = []
        
        if priority >= 4:
            reasons.append("节点空闲，高优先级分配")
        elif priority <= 2:
            reasons.append("节点过载，限制分配")
        
        if load_info.cache_hit_rate > 0.7:
            reasons.append("缓存命中率高")
        
        if load_info.queue_length == 0:
            reasons.append("无队列积压")
        
        if allocation_ratio > 0.3:
            reasons.append("大比例任务分配")
        
        return "; ".join(reasons) if reasons else "标准负载均衡分配"
    
    def _update_global_metrics(self):
        """更新全局性能指标"""
        if not self.rsu_loads:
            return
        
        # 总系统负载
        total_load = sum(self._calculate_normalized_load(info) for info in self.rsu_loads.values())
        self.global_metrics['total_system_load'] = total_load / len(self.rsu_loads)
        
        # 负载均衡指数
        self.global_metrics['load_balance_index'] = self._calculate_load_balance_index()
        
        # 全局响应时间
        response_times = [info.avg_response_time for info in self.rsu_loads.values() if info.avg_response_time > 0]
        self.global_metrics['global_response_time'] = np.mean(response_times) if response_times else 0.0
        
        # 系统吞吐量
        completion_rates = [info.task_completion_rate for info in self.rsu_loads.values()]
        self.global_metrics['system_throughput'] = np.sum(completion_rates)
    
    def _assess_system_health(self) -> str:
        """评估系统健康状态"""
        balance_index = self.global_metrics['load_balance_index']
        avg_response_time = self.global_metrics['global_response_time']
        
        if balance_index > 0.8 and avg_response_time < 100:
            return "excellent"
        elif balance_index > 0.6 and avg_response_time < 200:
            return "good"
        elif balance_index > 0.4 and avg_response_time < 300:
            return "fair"
        else:
            return "poor"
    
    def _get_load_distribution(self) -> Dict[str, float]:
        """获取负载分布情况"""
        distribution = {}
        for rsu_id, load_info in self.rsu_loads.items():
            distribution[rsu_id] = self._calculate_normalized_load(load_info)
        return distribution
    
    def _calculate_scheduling_efficiency(self) -> float:
        """计算调度效率"""
        if self.global_metrics['scheduling_decisions_count'] == 0:
            return 0.0
        
        # 基于负载均衡指数和系统吞吐量
        balance_score = self.global_metrics['load_balance_index']
        throughput_score = min(1.0, self.global_metrics['system_throughput'] / 100.0)
        
        efficiency = (balance_score * 0.6 + throughput_score * 0.4)
        return efficiency


class LoadPredictor:
    """📈 负载预测器"""
    
    def __init__(self):
        self.prediction_model = 'exponential_smoothing'  # 指数平滑
        self.alpha = 0.3  # 平滑参数
    
    def predict_future_loads(self, load_history: Dict[str, deque], steps: int = 3) -> Dict[str, float]:
        """
        预测未来负载
        
        Args:
            load_history: 负载历史数据
            steps: 预测步数
            
        Returns:
            Dict[str, float]: 预测的负载值
        """
        predictions = {}
        
        for rsu_id, history in load_history.items():
            if len(history) < 2:
                predictions[rsu_id] = history[-1] if history else 0.0
                continue
            
            # 指数平滑预测
            recent_loads = list(history)
            if self.prediction_model == 'exponential_smoothing':
                predicted_load = self._exponential_smoothing_prediction(recent_loads, steps)
            else:
                predicted_load = np.mean(recent_loads[-3:])  # 简单平均
            
            predictions[rsu_id] = max(0.0, min(1.0, predicted_load))
        
        return predictions
    
    def _exponential_smoothing_prediction(self, history: List[float], steps: int) -> float:
        """指数平滑预测"""
        if len(history) < 2:
            return history[-1] if history else 0.0
        
        # 初始值
        smoothed = history[0]
        
        # 指数平滑
        for value in history[1:]:
            smoothed = self.alpha * value + (1 - self.alpha) * smoothed
        
        # 简单地返回当前平滑值作为未来预测
        return smoothed


class AllocationOptimizer:
    """🎯 分配优化器"""
    
    def __init__(self):
        self.optimization_method = 'weighted_fair_allocation'
    
    def optimize_allocation(self, 
                          current_loads: Dict[str, RSULoadInfo],
                          predicted_loads: Dict[str, float],
                          incoming_tasks: int,
                          fairness_weight: float = 0.4,
                          efficiency_weight: float = 0.3) -> Dict[str, float]:
        """
        优化任务分配
        
        Args:
            current_loads: 当前负载信息
            predicted_loads: 预测负载
            incoming_tasks: 新增任务数
            fairness_weight: 公平性权重
            efficiency_weight: 效率性权重
            
        Returns:
            Dict[str, float]: 分配比例字典
        """
        if not current_loads:
            return {}
        
        allocation = {}
        
        if self.optimization_method == 'weighted_fair_allocation':
            allocation = self._weighted_fair_allocation(
                current_loads, predicted_loads, incoming_tasks, fairness_weight, efficiency_weight
            )
        
        # 确保分配比例总和为1.0
        total_allocation = sum(allocation.values())
        if total_allocation > 0:
            allocation = {rsu_id: ratio / total_allocation for rsu_id, ratio in allocation.items()}
        
        return allocation
    
    def _weighted_fair_allocation(self, 
                                current_loads: Dict[str, RSULoadInfo],
                                predicted_loads: Dict[str, float],
                                incoming_tasks: int,
                                fairness_weight: float,
                                efficiency_weight: float) -> Dict[str, float]:
        """加权公平分配算法"""
        allocation = {}
        
        # 计算每个RSU的分配权重
        for rsu_id, load_info in current_loads.items():
            # 当前负载因子（负载越高，权重越低）
            current_load_factor = load_info.cpu_usage
            load_weight = max(0.1, 1.0 - current_load_factor)
            
            # 预测负载因子
            predicted_load_factor = predicted_loads.get(rsu_id, current_load_factor)
            prediction_weight = max(0.1, 1.0 - predicted_load_factor)
            
            # 计算能力因子（基于CPU频率和缓存命中率）
            capacity_factor = load_info.cpu_frequency / 1e10  # 标准化到0-1
            cache_factor = load_info.cache_hit_rate
            efficiency_factor = (capacity_factor * 0.7 + cache_factor * 0.3)
            
            # 综合权重
            total_weight = (
                load_weight * fairness_weight +
                prediction_weight * 0.3 +
                efficiency_factor * efficiency_weight
            )
            
            allocation[rsu_id] = max(0.05, total_weight)  # 最小分配保证
        
        return allocation


# ==================== 全局调度接口 ====================

def create_central_scheduler(central_rsu_id: str = "RSU_2") -> CentralRSUScheduler:
    """
    🏗️ 创建中央RSU调度器实例
    
    Args:
        central_rsu_id: 中央RSU ID
        
    Returns:
        CentralRSUScheduler: 调度器实例
    """
    scheduler = CentralRSUScheduler(central_rsu_id=central_rsu_id)
    logging.info(f"🏢 创建中央RSU调度器: {central_rsu_id}")
    return scheduler


if __name__ == "__main__":
    # 🧪 测试中央RSU调度器
    logging.basicConfig(level=logging.INFO)
    
    # 创建调度器
    scheduler = create_central_scheduler("RSU_2")
    
    # 模拟RSU数据
    mock_rsu_data = [
        {
            'id': 'RSU_0', 'position': [100, 100], 'cpu_usage': 0.3,
            'cpu_frequency': 8e9, 'computation_queue': [1, 2], 'cache_hit_rate': 0.6
        },
        {
            'id': 'RSU_1', 'position': [200, 200], 'cpu_usage': 0.8,
            'cpu_frequency': 6e9, 'computation_queue': [1, 2, 3, 4, 5], 'cache_hit_rate': 0.4
        },
        {
            'id': 'RSU_3', 'position': [300, 100], 'cpu_usage': 0.1,
            'cpu_frequency': 10e9, 'computation_queue': [], 'cache_hit_rate': 0.9
        },
    ]
    
    # 收集负载信息
    loads = scheduler.collect_all_rsu_loads(mock_rsu_data)
    
    # 执行调度
    decisions = scheduler.global_load_balance_scheduling(incoming_task_count=5)
    
    # 生成迁移指令
    migrations = scheduler.intelligent_migration_coordination()
    
    # 获取状态报告
    status = scheduler.get_global_scheduling_status()
    
    print("🏢 中央RSU调度器测试完成")
    print(f"📊 负载收集: {len(loads)} 个RSU")
    print(f"🎯 调度决策: {len(decisions)} 条")
    print(f"🚀 迁移指令: {len(migrations)} 条")
    print(f"📈 系统健康: {status['system_health']}")
