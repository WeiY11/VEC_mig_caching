"""
任务分类与卸载决策框架 - 对应论文第3节和第4节
实现基于延迟容忍度的任务分类和处理模式评估
"""
import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Set, Union
from dataclasses import dataclass
from enum import Enum

from models.data_structures import Task, TaskType, Position, NodeType
from config import config


class ProcessingMode(Enum):
    """处理模式枚举 - 对应论文第4节"""
    LOCAL_COMPUTING = "local"           # 模式一：本地计算
    RSU_OFFLOAD_CACHE_HIT = "rsu_hit"   # 模式二：RSU卸载(缓存命中)
    RSU_OFFLOAD_NO_CACHE = "rsu_miss"   # 模式二：RSU卸载(缓存未命中)
    RSU_MIGRATION = "rsu_migration"     # 模式三：RSU间迁移
    UAV_OFFLOAD = "uav_offload"         # 模式四：UAV卸载


@dataclass
class ProcessingOption:
    """处理选项数据结构"""
    mode: ProcessingMode
    target_node_id: str
    predicted_delay: float
    energy_cost: float
    success_probability: float
    migration_source: Optional[str] = None  # 迁移源节点 (仅迁移模式)
    cache_hit: bool = False  # 是否缓存命中
    
    @property
    def weighted_cost(self) -> float:
        """计算加权成本"""
        # 归一化因子 - 基于实际数据范围调整
        delay_norm = 0.15  # 秒 (基于平均时延0.08秒)
        energy_norm = 250.0  # 焦耳 (基于平均能耗200W左右)
        
        # 权重 (可配置) - 进一步增加能耗权重以促进卸载
        w_delay = 0.15  # 进一步降低时延权重
        w_energy = 0.7  # 进一步增加能耗权重
        w_reliability = 0.15  # 进一步降低可靠性权重
        
        normalized_delay = self.predicted_delay / delay_norm
        normalized_energy = self.energy_cost / energy_norm
        reliability_penalty = 1.0 - self.success_probability
        
        return (w_delay * normalized_delay + 
                w_energy * normalized_energy + 
                w_reliability * reliability_penalty)


class TaskClassifier:
    """
    任务分类器 - 对应论文第3.1节任务分类框架
    根据延迟容忍度对任务进行四级分类
    """
    
    def __init__(self):
        # 延迟阈值 - 对应论文τ₁, τ₂, τ₃
        self.threshold_1 = config.task.delay_threshold_1  # 极度延迟敏感
        self.threshold_2 = config.task.delay_threshold_2  # 延迟敏感  
        self.threshold_3 = config.task.delay_threshold_3  # 中度延迟容忍
        
        # 任务类型统计
        self.classification_stats: Dict[TaskType, int] = {
            TaskType.EXTREMELY_DELAY_SENSITIVE: 0,
            TaskType.DELAY_SENSITIVE: 0,
            TaskType.MODERATELY_DELAY_TOLERANT: 0,
            TaskType.DELAY_TOLERANT: 0
        }
    
    def classify_task(self, task: Task) -> TaskType:
        """
        任务分类 - 对应论文第3.1节
        根据T_max,j与阈值τ₁,τ₂,τ₃的比较确定类别
        """
        max_delay = task.max_delay_slots
        
        if max_delay <= self.threshold_1:
            task_type = TaskType.EXTREMELY_DELAY_SENSITIVE
        elif max_delay <= self.threshold_2:
            task_type = TaskType.DELAY_SENSITIVE
        elif max_delay <= self.threshold_3:
            task_type = TaskType.MODERATELY_DELAY_TOLERANT
        else:
            task_type = TaskType.DELAY_TOLERANT
        
        # 更新任务类型并统计
        task.task_type = task_type
        self.classification_stats[task_type] += 1
        
        return task_type
    
    def get_candidate_nodes(self, task: Task, all_nodes: Dict[str, Position]) -> List[str]:
        """
        获取候选节点集合 - 对应论文第3.2节卸载导向策略
        
        Args:
            task: 待处理任务
            all_nodes: 所有节点位置信息 {node_id: Position}
            
        Returns:
            候选节点ID列表
        """
        task_type = task.task_type
        source_vehicle_id = task.source_vehicle_id
        
        if task_type == TaskType.EXTREMELY_DELAY_SENSITIVE:
            # 类别1: 仅本地处理 - N_j^cand = {v_j}
            return [source_vehicle_id]
        
        elif task_type == TaskType.DELAY_SENSITIVE:
            # 类别2: 本地 + 近距离低延迟RSU + UAV - 扩展以包含UAV
            candidates = [source_vehicle_id]
            
            # 添加距离最近的几个RSU
            nearby_rsus = self._get_nearby_rsus(source_vehicle_id, all_nodes, max_distance=800.0, max_count=3)
            candidates.extend(nearby_rsus)
            
            # 添加UAV以增加卸载选择
            capable_uavs = self._get_capable_uavs(source_vehicle_id, all_nodes, max_distance=500.0)
            candidates.extend(capable_uavs)
            
            return candidates
        
        elif task_type == TaskType.MODERATELY_DELAY_TOLERANT:
            # 类别3: 本地 + 可达RSU + 近距离UAV - N_j^cand ⊆ {v_j} ∪ R_reachable ∪ U_capable,nearby
            candidates = [source_vehicle_id]
            
            # 添加可达的RSU
            reachable_rsus = self._get_reachable_rsus(source_vehicle_id, all_nodes, max_distance=800.0)
            candidates.extend(reachable_rsus)
            
            # 添加近距离有能力的UAV
            capable_uavs = self._get_capable_uavs(source_vehicle_id, all_nodes, max_distance=600.0)
            candidates.extend(capable_uavs)
            
            return candidates
        
        else:  # DELAY_TOLERANT
            # 类别4: 所有节点 - N_j^cand = N
            return list(all_nodes.keys())
    
    def _get_nearby_rsus(self, vehicle_id: str, all_nodes: Dict[str, Position], 
                        max_distance: float, max_count: int) -> List[str]:
        """获取附近的RSU"""
        if vehicle_id not in all_nodes:
            return []
        
        vehicle_pos = all_nodes[vehicle_id]
        rsu_distances = []
        
        for node_id, position in all_nodes.items():
            if node_id.startswith("rsu_"):
                distance = vehicle_pos.distance_to(position)
                if distance <= max_distance:
                    rsu_distances.append((distance, node_id))
        
        # 按距离排序并返回最近的几个
        rsu_distances.sort(key=lambda x: x[0])
        return [node_id for _, node_id in rsu_distances[:max_count]]
    
    def _get_reachable_rsus(self, vehicle_id: str, all_nodes: Dict[str, Position], 
                           max_distance: float) -> List[str]:
        """获取可达的RSU"""
        if vehicle_id not in all_nodes:
            return []
        
        vehicle_pos = all_nodes[vehicle_id]
        reachable_rsus = []
        
        for node_id, position in all_nodes.items():
            if node_id.startswith("rsu_"):
                distance = vehicle_pos.distance_to(position)
                if distance <= max_distance:
                    reachable_rsus.append(node_id)
        
        return reachable_rsus
    
    def _get_capable_uavs(self, vehicle_id: str, all_nodes: Dict[str, Position], 
                         max_distance: float) -> List[str]:
        """获取有能力的UAV"""
        if vehicle_id not in all_nodes:
            return []
        
        vehicle_pos = all_nodes[vehicle_id]
        capable_uavs = []
        
        for node_id, position in all_nodes.items():
            if node_id.startswith("uav_"):
                distance = vehicle_pos.distance_2d_to(position)  # 2D距离
                if distance <= max_distance:
                    capable_uavs.append(node_id)
        
        return capable_uavs
    
    def get_classification_distribution(self) -> Dict[TaskType, float]:
        """获取任务分类分布"""
        total = sum(self.classification_stats.values())
        if total == 0:
            return {task_type: 0.0 for task_type in TaskType}
        
        return {task_type: count / total for task_type, count in self.classification_stats.items()}


class ProcessingModeEvaluator:
    """
    处理模式评估器 - 对应论文第4节处理模式评估框架
    在候选集内评估不同处理模式的性能
    """
    
    def __init__(self):
        # 评估参数 - 优化通信开销
        self.communication_overhead = 0.0002  # 通信开销 (秒) - 降低开销
        self.cache_response_delay = 0.0001   # 缓存响应时延 (秒) - 降低缓存时延
        
        # 成本权重
        self.delay_weight = 0.6
        self.energy_weight = 0.3
        self.reliability_weight = 0.1
    
    def evaluate_all_modes(self, task: Task, candidate_nodes: List[str], 
                          node_states: Dict, node_positions: Dict[str, Position],
                          cache_states: Optional[Dict] = None) -> List[ProcessingOption]:
        """
        评估所有可行的处理模式
        
        Args:
            task: 待处理任务
            candidate_nodes: 候选节点列表
            node_states: 节点状态信息
            node_positions: 节点位置信息
            cache_states: 缓存状态信息 (可选)
            
        Returns:
            处理选项列表
        """
        options = []
        
        for node_id in candidate_nodes:
            if node_id.startswith("vehicle_"):
                # 模式一：本地计算
                option = self._evaluate_local_computing(task, node_id, node_states)
                if option:
                    options.append(option)
            
            elif node_id.startswith("rsu_"):
                # 模式二：RSU卸载 (检查缓存命中)
                cache_options = self._evaluate_rsu_offload(task, node_id, node_states, 
                                                         node_positions, cache_states or {})
                options.extend(cache_options)
                
                # 模式三：RSU间迁移 (如果适用)
                migration_options = self._evaluate_rsu_migration(task, node_id, node_states, 
                                                               node_positions)
                options.extend(migration_options)
            
            elif node_id.startswith("uav_"):
                # 模式四：UAV卸载
                option = self._evaluate_uav_offload(task, node_id, node_states, node_positions)
                if option:
                    options.append(option)
        
        return options
    
    def _evaluate_local_computing(self, task: Task, vehicle_id: str, 
                                 node_states: Dict) -> Optional[ProcessingOption]:
        """评估本地计算模式 - 对应论文第4.1节"""
        if vehicle_id not in node_states:
            return None
        
        vehicle_state = node_states[vehicle_id]
        
        # 计算处理时延 - 论文式(6)
        parallel_efficiency = config.compute.parallel_efficiency
        processing_delay = task.compute_cycles / (vehicle_state.cpu_frequency * parallel_efficiency)
        
        # 预测等待时延 (简化)
        waiting_delay = self._estimate_waiting_time(vehicle_state)
        
        total_delay = processing_delay + waiting_delay
        
        # 计算能耗 - 使用真实的本地处理能耗
        base_energy = self._calculate_local_energy(task, vehicle_state)
        energy_cost = base_energy  # 使用真实能耗，不施加惩罚
        
        # 成功概率 (提高本地处理吸引力以平衡负载)
        if total_delay <= task.max_delay_slots * config.network.time_slot_duration * 0.8:
            success_prob = 0.90  # 高成功率，在80%时间内完成
        elif total_delay <= task.max_delay_slots * config.network.time_slot_duration:
            success_prob = 0.75  # 中等成功率，在截止时间内完成
        else:
            success_prob = 0.20  # 低成功率，超过截止时间
        
        return ProcessingOption(
            mode=ProcessingMode.LOCAL_COMPUTING,
            target_node_id=vehicle_id,
            predicted_delay=total_delay,
            energy_cost=energy_cost,
            success_probability=success_prob
        )
    
    def _evaluate_rsu_offload(self, task: Task, rsu_id: str, node_states: Dict,
                             node_positions: Dict[str, Position], 
                             cache_states: Dict) -> List[ProcessingOption]:
        """评估RSU卸载模式 - 对应论文第4.2节"""
        options = []
        
        if rsu_id not in node_states or task.source_vehicle_id not in node_positions:
            return options
        
        rsu_state = node_states[rsu_id]
        vehicle_pos = node_positions[task.source_vehicle_id]
        rsu_pos = node_positions[rsu_id]
        
        # 计算通信时延
        distance = vehicle_pos.distance_to(rsu_pos)
        upload_delay = self._calculate_transmission_delay(task.data_size, distance, "upload")
        download_delay = self._calculate_transmission_delay(task.result_size, distance, "download")
        
        # 检查缓存命中
        cache_hit = self._check_cache_hit(task, rsu_id, cache_states)
        
        if cache_hit:
            # 缓存命中情况 - 使用真实的缓存命中能耗
            total_delay = self.communication_overhead + self.cache_response_delay + download_delay
            energy_cost = self._calculate_transmission_energy(task.result_size, download_delay)  # 使用真实能耗
            success_prob = 0.80
            
            options.append(ProcessingOption(
                mode=ProcessingMode.RSU_OFFLOAD_CACHE_HIT,
                target_node_id=rsu_id,
                predicted_delay=total_delay,
                energy_cost=energy_cost,
                success_probability=success_prob,
                cache_hit=True
            ))
        else:
            # 缓存未命中情况
            processing_delay = task.compute_cycles / rsu_state.cpu_frequency
            waiting_delay = self._estimate_waiting_time(rsu_state)
            
            total_delay = upload_delay + waiting_delay + processing_delay + download_delay
            
            # 计算总能耗 - 使用真实的RSU卸载能耗
            upload_energy = self._calculate_transmission_energy(task.data_size, upload_delay)
            download_energy = self._calculate_transmission_energy(task.result_size, download_delay)
            energy_cost = upload_energy + download_energy  # 使用真实传输能耗
            
            # 成功概率 (RSU卸载成功率)
            if total_delay <= task.max_delay_slots * config.network.time_slot_duration * 0.9:
                success_prob = 0.85  # RSU处理能力强，高成功率
            elif total_delay <= task.max_delay_slots * config.network.time_slot_duration:
                success_prob = 0.70  # 中等成功率
            else:
                success_prob = 0.30  # 低成功率
            
            options.append(ProcessingOption(
                mode=ProcessingMode.RSU_OFFLOAD_NO_CACHE,
                target_node_id=rsu_id,
                predicted_delay=total_delay,
                energy_cost=energy_cost,
                success_probability=success_prob,
                cache_hit=False
            ))
        
        return options
    
    def _evaluate_rsu_migration(self, task: Task, target_rsu_id: str, node_states: Dict,
                               node_positions: Dict[str, Position]) -> List[ProcessingOption]:
        """评估RSU间迁移模式 - 对应论文第4.3节"""
        options = []
        
        # 寻找可能的源RSU (当前过载的RSU)
        for source_rsu_id, state in node_states.items():
            if (source_rsu_id.startswith("rsu_") and 
                source_rsu_id != target_rsu_id and 
                state.load_factor > config.migration.rsu_overload_threshold):
                
                # 计算迁移成本和时延
                migration_delay = self._calculate_migration_delay(task, source_rsu_id, target_rsu_id, 
                                                                node_positions)
                
                # 目标RSU处理时延
                target_state = node_states.get(target_rsu_id)
                if target_state:
                    processing_delay = task.compute_cycles / target_state.cpu_frequency
                    waiting_delay = self._estimate_waiting_time(target_state)
                    
                    total_delay = migration_delay + waiting_delay + processing_delay
                    
                    # 迁移能耗 (简化)
                    energy_cost = self._calculate_migration_energy(task, source_rsu_id, target_rsu_id)
                    
                    # 成功概率 (考虑迁移风险)
                    success_prob = 0.85 if total_delay <= task.max_delay_slots * config.network.time_slot_duration else 0.2
                    
                    options.append(ProcessingOption(
                        mode=ProcessingMode.RSU_MIGRATION,
                        target_node_id=target_rsu_id,
                        predicted_delay=total_delay,
                        energy_cost=energy_cost,
                        success_probability=success_prob,
                        migration_source=source_rsu_id
                    ))
        
        return options
    
    def _evaluate_uav_offload(self, task: Task, uav_id: str, node_states: Dict,
                             node_positions: Dict[str, Position]) -> Optional[ProcessingOption]:
        """评估UAV卸载模式 - 对应论文第4.4节"""
        if uav_id not in node_states or task.source_vehicle_id not in node_positions:
            return None
        
        uav_state = node_states[uav_id]
        vehicle_pos = node_positions[task.source_vehicle_id]
        uav_pos = node_positions[uav_id]
        
        # 检查UAV是否可用 (电池、负载等)
        if (hasattr(uav_state, 'battery_level') and 
            uav_state.battery_level < config.migration.uav_min_battery):
            return None
        
        # 计算通信时延
        distance = vehicle_pos.distance_to(uav_pos)
        upload_delay = self._calculate_transmission_delay(task.data_size, distance, "upload")
        download_delay = self._calculate_transmission_delay(task.result_size, distance, "download")
        
        # 计算处理时延 - 考虑电池对性能的影响
        battery_factor = getattr(uav_state, 'battery_level', 1.0)
        effective_freq = uav_state.cpu_frequency * max(0.5, battery_factor)
        processing_delay = task.compute_cycles / effective_freq
        
        # 预测等待时延
        waiting_delay = self._estimate_waiting_time(uav_state)
        
        total_delay = upload_delay + waiting_delay + processing_delay + download_delay
        
        # 计算能耗 - 合理的UAV能耗计算
        comm_energy = self._calculate_transmission_energy(task.data_size + task.result_size, 
                                                        upload_delay + download_delay)
        compute_energy = config.compute.uav_kappa3 * (uav_state.cpu_frequency ** 2) * processing_delay
        energy_cost = comm_energy + compute_energy  # 使用真实能耗
        
        # 成功概率 (UAV卸载成功率，考虑电池和距离因素)
        if total_delay <= task.max_delay_slots * config.network.time_slot_duration * 0.8:
            success_prob = 0.80  # UAV灵活性高但受电池限制
        elif total_delay <= task.max_delay_slots * config.network.time_slot_duration:
            success_prob = 0.65  # 中等成功率
        else:
            success_prob = 0.25  # 低成功率
        
        return ProcessingOption(
            mode=ProcessingMode.UAV_OFFLOAD,
            target_node_id=uav_id,
            predicted_delay=total_delay,
            energy_cost=energy_cost,
            success_probability=success_prob
        )
    
    def _check_cache_hit(self, task: Task, rsu_id: str, cache_states: Dict) -> bool:
        """检查缓存命中"""
        if not cache_states or rsu_id not in cache_states:
            return False
        
        # 简化的缓存命中检测
        task_signature = f"{task.task_type.value}_{int(task.data_size)}_{int(task.compute_cycles)}"
        return task_signature in cache_states.get(rsu_id, {})
    
    def _calculate_transmission_delay(self, data_size: float, distance: float, direction: str) -> float:
        """计算传输时延"""
        # 优化的传输时延模型 - 提升基础速率
        base_rate = 50e6  # 50 Mbps基础速率 (5G网络)
        
        # 距离衰减 - 减少衰减影响
        path_loss_factor = 1.0 + (distance / 2000.0) ** 1.5  # 降低衰减系数
        effective_rate = base_rate / path_loss_factor
        
        transmission_delay = data_size / effective_rate
        propagation_delay = distance / 3e8  # 光速传播
        
        return transmission_delay + propagation_delay + self.communication_overhead
    
    def _calculate_transmission_energy(self, data_size: float, transmission_time: float) -> float:
        """计算传输能耗"""
        tx_power = config.communication.vehicle_tx_power
        circuit_power = config.communication.circuit_power
        
        return (tx_power + circuit_power) * transmission_time
    
    def _calculate_local_energy(self, task: Task, vehicle_state) -> float:
        """计算本地处理能耗"""
        # 处理时间
        processing_time = task.compute_cycles / (vehicle_state.cpu_frequency * config.compute.parallel_efficiency)
        
        # 动态功率 - 论文式(7)
        utilization = min(1.0, processing_time / config.network.time_slot_duration)
        dynamic_power = (config.compute.vehicle_kappa1 * (vehicle_state.cpu_frequency ** 3) +
                        config.compute.vehicle_kappa2 * (vehicle_state.cpu_frequency ** 2) * utilization +
                        config.compute.vehicle_static_power)
        
        return dynamic_power * processing_time
    
    def _calculate_migration_delay(self, task: Task, source_rsu: str, target_rsu: str,
                                  node_positions: Dict[str, Position]) -> float:
        """计算迁移时延"""
        if source_rsu not in node_positions or target_rsu not in node_positions:
            return float('inf')
        
        source_pos = node_positions[source_rsu]
        target_pos = node_positions[target_rsu]
        distance = source_pos.distance_to(target_pos)
        
        # 迁移传输时延
        migration_rate = config.migration.migration_bandwidth
        return task.data_size / migration_rate + distance / 3e8
    
    def _calculate_migration_energy(self, task: Task, source_rsu: str, target_rsu: str) -> float:
        """计算迁移能耗"""
        # 简化的迁移能耗模型
        tx_power = config.communication.rsu_tx_power
        migration_time = task.data_size / config.migration.migration_bandwidth
        
        return tx_power * migration_time
    
    def _estimate_waiting_time(self, node_state) -> float:
        """估算等待时间"""
        # 基于当前负载的简化等待时间估算
        if hasattr(node_state, 'load_factor'):
            if node_state.load_factor >= 1.0:
                return float('inf')
            else:
                # 简化的M/M/1公式: W = ρ/(μ(1-ρ))
                base_service_time = 0.1  # 基础服务时间
                return (node_state.load_factor * base_service_time) / (1 - node_state.load_factor)
        else:
            return 0.01  # 默认很小的等待时间
    
    def select_best_option(self, options: List[ProcessingOption]) -> Optional[ProcessingOption]:
        """
        选择最佳处理选项
        基于加权成本函数
        """
        if not options:
            return None
        
        # 过滤掉不可行的选项
        feasible_options = [opt for opt in options if opt.success_probability > 0.1]
        
        if not feasible_options:
            return None
        
        # 选择加权成本最小的选项
        best_option = min(feasible_options, key=lambda x: x.weighted_cost)
        
        return best_option


class OffloadingDecisionMaker:
    """
    卸载决策制定器 - 整合分类和评估
    对应论文第3-4节的完整决策流程
    """
    
    def __init__(self):
        self.classifier = TaskClassifier()
        self.evaluator = ProcessingModeEvaluator()
        
        # 决策统计
        self.decision_stats: Dict[ProcessingMode, int] = {mode: 0 for mode in ProcessingMode}
        self.total_decisions = 0
    
    def make_offloading_decision(self, task: Task, node_states: Dict, 
                               node_positions: Dict[str, Position],
                               cache_states: Optional[Dict] = None) -> Optional[ProcessingOption]:
        """
        制定卸载决策 - 完整的决策流程
        
        Args:
            task: 待处理任务
            node_states: 所有节点状态
            node_positions: 所有节点位置
            cache_states: 缓存状态 (可选)
            
        Returns:
            最佳处理选项
        """
        # 1. 任务分类 - 论文第3.1节
        task_type = self.classifier.classify_task(task)
        
        # 2. 确定候选节点集合 - 论文第3.2节
        candidate_nodes = self.classifier.get_candidate_nodes(task, node_positions)
        
        # 3. 评估所有处理模式 - 论文第4节
        processing_options = self.evaluator.evaluate_all_modes(
            task, candidate_nodes, node_states, node_positions, cache_states
        )
        
        # 4. 选择最佳选项
        best_option = self.evaluator.select_best_option(processing_options)
        
        # 5. 更新统计
        if best_option:
            self.decision_stats[best_option.mode] += 1
        self.total_decisions += 1
        
        return best_option
    
    def get_decision_statistics(self) -> Dict:
        """获取决策统计信息"""
        if self.total_decisions == 0:
            return {mode.value: 0.0 for mode in ProcessingMode}
        
        stats = {}
        for mode, count in self.decision_stats.items():
            stats[mode.value] = count / self.total_decisions
        
        # 添加分类统计
        stats['classification_distribution'] = self.classifier.get_classification_distribution()
        stats['total_decisions'] = self.total_decisions
        
        return stats