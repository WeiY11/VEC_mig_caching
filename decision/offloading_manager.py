"""
任务分类与卸载决策（优化版，200ms时隙 + 表IV 参数对齐）
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

from config import config
from utils import dbm_to_watts
from models.data_structures import Task, TaskType, Position


class ProcessingMode(Enum):
    LOCAL_COMPUTING = "local"
    RSU_OFFLOAD_CACHE_HIT = "rsu_hit"
    RSU_OFFLOAD_NO_CACHE = "rsu_miss"
    RSU_MIGRATION = "rsu_migration"
    UAV_OFFLOAD = "uav_offload"
    UAV_RELAY = "uav_relay"  # UAV中继模式：转发数据到RSU


@dataclass
class ProcessingOption:
    mode: ProcessingMode
    target_node_id: str
    predicted_delay: float
    energy_cost: float
    success_probability: float
    migration_source: Optional[str] = None
    cache_hit: bool = False
    latency_weight: float = 1.0
    reserved_processing_time: float = 0.0
    reserved_comm_time: float = 0.0
    relay_node_id: Optional[str] = None  # UAV中继时的中继节点ID
    final_target_id: Optional[str] = None  # UAV中继时的最终目标RSU

    @property
    def weighted_cost(self) -> float:
        slot = max(1e-9, float(config.network.time_slot_duration))
        energy_norm = 1000.0
        weights = getattr(ProcessingModeEvaluator, 'cost_weight_profile', {'delay': 0.5, 'energy': 0.4, 'reliability': 0.1})
        w_delay = float(weights.get('delay', 0.5))
        w_energy = float(weights.get('energy', 0.4))
        w_rel = float(weights.get('reliability', 0.1))
        total = max(1e-6, w_delay + w_energy + w_rel)
        w_delay /= total
        w_energy /= total
        w_rel /= total
        norm_delay = (self.predicted_delay / slot) * max(0.1, self.latency_weight)
        norm_energy = self.energy_cost / max(1e-9, energy_norm)
        rel_pen = 1.0 - self.success_probability
        return w_delay * norm_delay + w_energy * norm_energy + w_rel * rel_pen


class TaskClassifier:
    def __init__(self):
        th = getattr(config.task, 'delay_thresholds', None)
        if isinstance(th, dict) and th:
            self.threshold_1 = int(th.get('extremely_sensitive', 1))
            self.threshold_2 = int(th.get('sensitive', 2))
            self.threshold_3 = int(th.get('moderately_tolerant', 3))
        else:
            self.threshold_1 = int(getattr(config.task, 'delay_threshold_1', 1))
            self.threshold_2 = int(getattr(config.task, 'delay_threshold_2', 2))
            self.threshold_3 = int(getattr(config.task, 'delay_threshold_3', 3))
        self.classification_stats: Dict[TaskType, int] = {
            TaskType.EXTREMELY_DELAY_SENSITIVE: 0,
            TaskType.DELAY_SENSITIVE: 0,
            TaskType.MODERATELY_DELAY_TOLERANT: 0,
            TaskType.DELAY_TOLERANT: 0,
        }

    def classify_task(self, task: Task) -> TaskType:
        max_delay_slots = int(getattr(task, 'max_delay_slots', 1) or 1)
        data_size = float(getattr(task, 'data_size', 0.0) or 0.0)
        compute_cycles = getattr(task, 'compute_cycles', None)
        compute_density = None
        if compute_cycles and data_size:
            denom = max(data_size * 8.0, 1.0)
            compute_density = compute_cycles / denom
        system_load = self._estimate_system_load_hint()
        slot_duration = getattr(config.network, 'time_slot_duration', 0.1)
        type_value = config.task.get_task_type(
            max_delay_slots,
            data_size=data_size,
            compute_cycles=compute_cycles,
            compute_density=compute_density,
            time_slot=slot_duration,
            system_load=system_load,
            is_cacheable=self._is_task_cacheable(task),
        )
        task_type = TaskType(type_value)
        task.task_type = task_type
        self.classification_stats[task_type] += 1
        return task_type

    def _estimate_system_load_hint(self) -> Optional[float]:
        total = sum(self.classification_stats.values())
        if total < 5:
            return None
        high_priority = (
            self.classification_stats[TaskType.EXTREMELY_DELAY_SENSITIVE]
            + self.classification_stats[TaskType.DELAY_SENSITIVE]
        )
        return max(0.0, min(1.0, high_priority / total))

    def _is_task_cacheable(self, task: Task) -> bool:
        return getattr(task, 'cache_access_count', 0) > 0

    def get_candidate_nodes(self, task: Task, all_nodes: Dict[str, Position]) -> List[str]:
        t = task.task_type
        vid = task.source_vehicle_id
        if t == TaskType.EXTREMELY_DELAY_SENSITIVE:
            return [vid]
        if t == TaskType.DELAY_SENSITIVE:
            c = [vid]
            c.extend(self._get_nearby_rsus(vid, all_nodes, 600.0, 2))
            c.extend(self._get_capable_uavs(vid, all_nodes, 600.0))  # 🔧 UAV优化: 400m→600m,与RSU对齐
            return c
        if t == TaskType.MODERATELY_DELAY_TOLERANT:
            c = [vid]
            c.extend(self._get_reachable_rsus(vid, all_nodes, 800.0))
            c.extend(self._get_capable_uavs(vid, all_nodes, 800.0))  # 🔧 UAV优化: 600m→800m,扩大覆盖
            return c
        return list(all_nodes.keys())

    def _get_nearby_rsus(self, vid: str, nodes: Dict[str, Position], max_d: float, k: int) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        arr = [(v.distance_to(p), nid) for nid, p in nodes.items() if nid.startswith('rsu_') and v.distance_to(p) <= max_d]
        arr.sort(key=lambda x: x[0])
        return [nid for _, nid in arr[:k]]

    def _get_reachable_rsus(self, vid: str, nodes: Dict[str, Position], max_d: float) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        return [nid for nid, p in nodes.items() if nid.startswith('rsu_') and v.distance_to(p) <= max_d]

    def _get_capable_uavs(self, vid: str, nodes: Dict[str, Position], max_d: float) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        return [nid for nid, p in nodes.items() if nid.startswith('uav_') and v.distance_2d_to(p) <= max_d]

    def get_classification_distribution(self) -> Dict[TaskType, float]:
        total = sum(self.classification_stats.values()) or 1
        return {k: v / total for k, v in self.classification_stats.items()}


class ProcessingModeEvaluator:
    cost_weight_profile: Dict[str, float] = {'delay': 0.5, 'energy': 0.4, 'reliability': 0.1}

    def __init__(self):
        self.communication_overhead = 0.0002
        self.cache_response_delay = 0.0001
        self.scheduling_preferences = {'priority_bias': 0.5, 'deadline_bias': 0.5}
        self.tradeoff_bias = 0.5
        self.virtual_cpu_load: Dict[str, float] = {}
        self.virtual_comm_load: Dict[str, float] = {}
        self.last_slot_index: Optional[int] = None
        self._refresh_cost_weights()

    def evaluate_all_modes(self, task: Task, candidates: List[str], node_states: Dict,
                           node_positions: Dict[str, Position], cache_states: Optional[Dict] = None) -> List[ProcessingOption]:
        opts: List[ProcessingOption] = []
        for nid in candidates:
            if nid.startswith('vehicle_'):
                o = self._eval_local(task, nid, node_states)
                if o:
                    opts.append(o)
            elif nid.startswith('rsu_'):
                opts.extend(self._eval_rsu(task, nid, node_states, node_positions, cache_states or {}))
                opts.extend(self._eval_rsu_mig(task, nid, node_states, node_positions))
            elif nid.startswith('uav_'):
                # 评估UAV计算模式
                o = self._eval_uav(task, nid, node_states, node_positions)
                if o:
                    opts.append(o)
                # 评估UAV中继模式
                relay_opts = self._eval_uav_relay(task, nid, node_states, node_positions)
                opts.extend(relay_opts)
        return opts

    def _alpha(self, task: Task) -> float:
        try:
            return float(config.task.get_latency_cost_weight(int(task.task_type.value)))
        except Exception:
            return 1.0

    def _slack_prob(self, delay: float, slots: int) -> float:
        slot = config.network.time_slot_duration
        slack = slots * slot - delay
        # Require at least ~40 ms of slack (previous threshold at 0.2 s slots)
        if slack >= max(0.04, 0.2 * slot):
            return 0.9
        if slack >= 0:
            return 0.7
        return 0.25

    def _cost_delay(self, delay: float, task: Task) -> float:
        # 传给 weighted_cost 的“成本延迟”输入
        slot = max(1e-9, float(config.network.time_slot_duration))
        return delay * (0.15 / slot) * self._alpha(task)

    def _refresh_cost_weights(self) -> None:
        priority = float(self.scheduling_preferences.get('priority_bias', 0.5))
        deadline = float(self.scheduling_preferences.get('deadline_bias', 0.5))
        tradeoff = float(self.tradeoff_bias)
        delay_weight = 0.35 + 0.45 * priority
        energy_weight = max(0.1, 0.4 - 0.2 * priority)
        reliability_weight = 0.2 + 0.3 * deadline
        energy_weight *= (0.6 + 0.4 * (1.0 - tradeoff))
        weights = {
            'delay': max(1e-3, delay_weight),
            'energy': max(1e-3, energy_weight),
            'reliability': max(1e-3, reliability_weight),
        }
        total = sum(weights.values())
        if total <= 0.0:
            total = 1.0
        for key in list(weights.keys()):
            weights[key] = float(weights[key] / total)
        ProcessingModeEvaluator.cost_weight_profile = weights

    def advance_virtual_time(self, slot_index: Optional[int]) -> None:
        """Decay virtual reservations when the global slot advances."""
        if slot_index is None:
            return
        if self.last_slot_index is None or slot_index > self.last_slot_index:
            self.virtual_cpu_load = {
                node: load * 0.5 for node, load in self.virtual_cpu_load.items()
                if load * 0.5 > 1e-3
            }
            self.virtual_comm_load = {
                node: load * 0.5 for node, load in self.virtual_comm_load.items()
                if load * 0.5 > 1e-3
            }
            self.last_slot_index = slot_index

    def reserve_resources(self, option: ProcessingOption, task: Task, node_states: Dict) -> None:
        """Record the resources committed by the chosen processing option."""
        slot = max(1e-9, float(config.network.time_slot_duration))
        if option.reserved_processing_time > 0:
            normalized = option.reserved_processing_time / slot
            self._increment_cpu_reservation(option.target_node_id, normalized, node_states)
        if option.reserved_comm_time > 0:
            if option.mode == ProcessingMode.RSU_MIGRATION and option.migration_source:
                reservation_key = option.migration_source
            else:
                reservation_key = getattr(task, 'source_vehicle_id', None)
            normalized = option.reserved_comm_time / slot
            self._increment_comm_reservation(reservation_key, normalized)

    def _increment_cpu_reservation(self, node_id: Optional[str], normalized_load: float, node_states: Dict) -> None:
        if not node_id or normalized_load <= 0:
            return
        prev = self.virtual_cpu_load.get(node_id, 0.0)
        queue_cfg = getattr(config, 'queue', None)
        max_load = float(getattr(queue_cfg, 'max_load_factor', 1.25)) if queue_cfg else 1.25
        self.virtual_cpu_load[node_id] = min(max_load, prev + normalized_load)
        state = node_states.get(node_id)
        if state is not None:
            try:
                state.load_factor = min(
                    max_load,
                    float(getattr(state, 'load_factor', 0.0)) + normalized_load * 0.5
                )
            except Exception:
                setattr(
                    state,
                    'load_factor',
                    min(max_load, float(getattr(state, 'load_factor', 0.0)) + normalized_load * 0.5)
                )

    def _increment_comm_reservation(self, key: Optional[str], normalized_load: float) -> None:
        if not key or normalized_load <= 0:
            return
        prev = self.virtual_comm_load.get(key, 0.0)
        self.virtual_comm_load[key] = min(3.0, prev + normalized_load)

    def update_scheduling_preferences(self, preferences: Dict[str, float]) -> None:
        if not isinstance(preferences, dict):
            return
        priority = preferences.get('priority_bias')
        if priority is not None:
            try:
                priority_val = float(priority)
            except (TypeError, ValueError):
                priority_val = None
            else:
                self.scheduling_preferences['priority_bias'] = min(max(priority_val, 0.0), 1.0)
        deadline = preferences.get('deadline_bias')
        if deadline is not None:
            try:
                deadline_val = float(deadline)
            except (TypeError, ValueError):
                deadline_val = None
            else:
                self.scheduling_preferences['deadline_bias'] = min(max(deadline_val, 0.0), 1.0)
        self._refresh_cost_weights()

    def update_joint_tradeoff(self, joint_params: Optional[Dict[str, float]] = None) -> None:
        if joint_params is None:
            return
        if isinstance(joint_params, dict):
            value = joint_params.get('cache_migration_tradeoff')
            if value is None:
                value = joint_params.get('tradeoff')
        else:
            value = joint_params
        if value is None:
            return
        try:
            tradeoff_val = float(value)
        except (TypeError, ValueError):
            return
        self.tradeoff_bias = min(max(tradeoff_val, 0.0), 1.0)
        self._refresh_cost_weights()

    def _eval_local(self, task: Task, vid: str, states: Dict) -> Optional[ProcessingOption]:
        st = states.get(vid)
        if not st:
            return None
        pe = config.compute.parallel_efficiency
        proc = task.compute_cycles / max(1e-9, (st.cpu_frequency * pe))
        wait = self._wait(st)
        total = proc + wait
        energy = self._energy_local(task, st)
        return ProcessingOption(
            mode=ProcessingMode.LOCAL_COMPUTING,
            target_node_id=vid,
            predicted_delay=self._cost_delay(total, task),
            energy_cost=energy,
            success_probability=self._slack_prob(total, task.max_delay_slots),
            latency_weight=self._alpha(task),
            reserved_processing_time=proc
        )

    def _eval_rsu(self, task: Task, rid: str, states: Dict, pos: Dict[str, Position], caches: Dict) -> List[ProcessingOption]:
        out: List[ProcessingOption] = []
        st = states.get(rid)
        vpos = pos.get(task.source_vehicle_id)
        rpos = pos.get(rid)
        if not (st and vpos and rpos):
            return out
        d = vpos.distance_to(rpos)
        reservation_key = getattr(task, 'source_vehicle_id', None)
        up = self._tx_delay(task.data_size, d, reservation_key=reservation_key)
        down = self._tx_delay(task.result_size, d, reservation_key=reservation_key)
        if self._cache_hit(task, rid, caches):
            total = self.communication_overhead + self.cache_response_delay + down
            energy = self._tx_energy(task.result_size, down)
            out.append(ProcessingOption(
                mode=ProcessingMode.RSU_OFFLOAD_CACHE_HIT,
                target_node_id=rid,
                predicted_delay=self._cost_delay(total, task),
                energy_cost=energy,
                success_probability=self._slack_prob(total, task.max_delay_slots),
                cache_hit=True,
                latency_weight=self._alpha(task),
                reserved_comm_time=down
            ))
        else:
            proc = task.compute_cycles / max(1e-9, st.cpu_frequency)
            wait = self._wait(st)
            total = up + wait + proc + down
            comm_energy = self._tx_energy(task.data_size + task.result_size, up + down)
            rsu_dynamic_power = config.compute.rsu_kappa2 * (st.cpu_frequency ** 3)
            rsu_dynamic_energy = rsu_dynamic_power * proc
            slot_duration = config.network.time_slot_duration
            rsu_static_energy = config.compute.rsu_static_power * max(proc, slot_duration)
            energy = comm_energy + rsu_dynamic_energy + rsu_static_energy
            out.append(ProcessingOption(
                mode=ProcessingMode.RSU_OFFLOAD_NO_CACHE,
                target_node_id=rid,
                predicted_delay=self._cost_delay(total, task),
                energy_cost=energy,
                success_probability=self._slack_prob(total, task.max_delay_slots),
                latency_weight=self._alpha(task),
                reserved_processing_time=proc,
                reserved_comm_time=up + down
            ))
        return out

    def _eval_rsu_mig(self, task: Task, target: str, states: Dict, pos: Dict[str, Position]) -> List[ProcessingOption]:
        out: List[ProcessingOption] = []
        tgt = states.get(target)
        if not tgt:
            return out
        for sid, s in states.items():
            if not sid.startswith('rsu_') or sid == target:
                continue
            if float(getattr(s, 'load_factor', 0.0)) <= config.migration.rsu_overload_threshold:
                continue
            mig = (task.data_size * 8.0) / max(1e-9, config.migration.migration_bandwidth)
            proc = task.compute_cycles / max(1e-9, tgt.cpu_frequency)
            wait = self._wait(tgt)
            total = mig + wait + proc
            rsu_tx_power_w = dbm_to_watts(config.communication.rsu_tx_power)
            energy = (rsu_tx_power_w + config.communication.circuit_power) * mig
            out.append(ProcessingOption(
                mode=ProcessingMode.RSU_MIGRATION,
                target_node_id=target,
                predicted_delay=self._cost_delay(total, task),
                energy_cost=energy,
                success_probability=0.9 if total <= task.max_delay_slots * config.network.time_slot_duration else 0.25,
                migration_source=sid,
                latency_weight=self._alpha(task),
                reserved_processing_time=proc,
                reserved_comm_time=mig
            ))
        return out

    def _eval_uav(self, task: Task, uid: str, states: Dict, pos: Dict[str, Position]) -> Optional[ProcessingOption]:
        uv = states.get(uid)
        vpos = pos.get(task.source_vehicle_id)
        upos = pos.get(uid)
        if not (uv and vpos and upos):
            return None
        d = vpos.distance_to(upos)
        reservation_key = getattr(task, 'source_vehicle_id', None)
        up = self._tx_delay(task.data_size, d, reservation_key=reservation_key)
        down = self._tx_delay(task.result_size, d, reservation_key=reservation_key)
        eff = uv.cpu_frequency * max(0.5, float(getattr(uv, 'battery_level', 1.0)))
        proc = task.compute_cycles / max(1e-9, eff)
        wait = self._wait(uv)
        total = up + wait + proc + down
        comm_energy = self._tx_energy(task.data_size + task.result_size, up + down)
        dynamic_energy = config.compute.uav_kappa3 * (eff ** 2) * proc
        slot_duration = config.network.time_slot_duration
        static_energy = config.compute.uav_static_power * max(proc, slot_duration)
        # 🔧 UAV优化修正:悬停能耗持续存在,UAV空闲时也消耗(原逻辑正确)
        hover_energy = config.compute.uav_hover_power * total
        energy = comm_energy + dynamic_energy + static_energy + hover_energy
        return ProcessingOption(
            mode=ProcessingMode.UAV_OFFLOAD,
            target_node_id=uid,
            predicted_delay=self._cost_delay(total, task),
            energy_cost=energy,
            success_probability=self._slack_prob(total, task.max_delay_slots),
            latency_weight=self._alpha(task),
            reserved_processing_time=proc,
            reserved_comm_time=up + down
        )

    # helpers
    def _cache_hit(self, task: Task, rid: str, caches: Dict) -> bool:
        if not caches or rid not in caches:
            return False
        sig = f"{task.task_type.value}_{int(task.data_size)}_{int(task.compute_cycles)}"
        return sig in caches.get(rid, {})

    def _tx_delay(self, bits: float, d: float, reservation_key: Optional[str] = None) -> float:
        base = 50e6
        loss = 1.0 + (d / 2000.0) ** 1.5
        rate = base / max(1e-9, loss)
        raw_delay = bits / max(1e-9, rate) + self.communication_overhead
        if reservation_key:
            penalty = 1.0 + self.virtual_comm_load.get(reservation_key, 0.0)
            return raw_delay * penalty
        return raw_delay

    def _tx_energy(self, bits: float, t: float) -> float:
        tx_power_watts = dbm_to_watts(config.communication.vehicle_tx_power)
        return (tx_power_watts + config.communication.circuit_power) * max(0.0, t)

    def _energy_local(self, task: Task, st) -> float:
        proc = task.compute_cycles / max(1e-9, (st.cpu_frequency * config.compute.parallel_efficiency))
        util = min(1.0, proc / max(1e-9, config.network.time_slot_duration))
        dyn = (config.compute.vehicle_kappa1 * (st.cpu_frequency ** 3) +
               config.compute.vehicle_kappa2 * (st.cpu_frequency ** 2) * util +
               config.compute.vehicle_static_power)
        return dyn * proc

    def _wait(self, st) -> float:
        rho = float(getattr(st, 'load_factor', 0.0))
        node_id = getattr(st, 'node_id', None)
        if node_id is not None:
            rho += self.virtual_cpu_load.get(node_id, 0.0)
        if rho >= 0.999:
            return float('inf')
        base = 0.06
        return max(0.0, (rho * base) / max(1e-6, 1 - rho))

    def _calculate_link_quality(self, distance: float, link_type: str = 'ground') -> float:
        """
        计算链路质量（基于距离和链路类型）
        
        Args:
            distance: 距离（米）
            link_type: 'ground' (车辆-RSU) 或 'air' (车辆-UAV 或 UAV-RSU)
            
        Returns:
            信号质量 (0.0-1.0)，值越大信号越好
        """
        if link_type == 'air':
            # UAV链路：少遮挡，但距离衰减显著
            max_distance = 800.0  # UAV有效距离
            path_loss_exp = 2.2  # 空中路径损耗指数较小
        else:
            # 地面链路：遮挡多，距离衰减严重
            max_distance = 600.0  # RSU有效距离
            path_loss_exp = 3.5  # 地面路径损耗指数较大
        
        if distance >= max_distance:
            return 0.1  # 超出覆盖范围，信号极弱
        
        # 基于距离的质量计算：quality = (1 - (d/d_max)^exp)
        normalized_distance = distance / max_distance
        quality = max(0.1, 1.0 - (normalized_distance ** path_loss_exp))
        
        return quality

    def _should_use_uav_relay(self, vehicle_pos: Position, uav_pos: Position, 
                             rsu_pos: Position, uav_state: Dict, 
                             rsu_state: Dict) -> bool:
        """
        判断是否应该使用UAV中继模式
        
        决策条件（根据用户需求）：
        1. 车辆到RSU直连信号质量差 (<0.5)
        2. 通过UAV中继信号质量好 (>0.7)
        3. UAV电量充足 (>30%)
        4. UAV负载不高 (<70%)
        """
        # 1. 计算直连信号质量
        direct_distance = vehicle_pos.distance_to(rsu_pos)
        direct_quality = self._calculate_link_quality(direct_distance, 'ground')
        
        # 2. 计算中继信号质量（取两段的最小值）
        v2u_distance = vehicle_pos.distance_to(uav_pos)
        u2r_distance = uav_pos.distance_to(rsu_pos)
        v2u_quality = self._calculate_link_quality(v2u_distance, 'air')
        u2r_quality = self._calculate_link_quality(u2r_distance, 'air')
        relay_quality = min(v2u_quality, u2r_quality)
        
        # 3. UAV状态检查
        uav_battery = float(getattr(uav_state, 'battery_level', 1.0))
        uav_load = float(getattr(uav_state, 'load_factor', 0.0))
        
        # 4. RSU状态检查（过载时也可以考虑中继）
        rsu_load = float(getattr(rsu_state, 'load_factor', 0.0))
        
        # 决策逻辑（🔧 UAV优化：放宽中继激活条件）：
        # - 直连质量差 AND 中继质量好
        # - UAV电量充足（>25%，原30%）
        # - UAV负载不高（<75%，原70%）
        # - RSU未过载 (<95%，原90%)
        if (direct_quality < 0.5 and relay_quality > 0.6 and
            uav_battery > 0.25 and uav_load < 0.75 and rsu_load < 0.95):
            return True
        
        return False

    def _eval_uav_relay(self, task: Task, uav_id: str, states: Dict, 
                       pos: Dict[str, Position]) -> List[ProcessingOption]:
        """
        评估UAV中继模式：从车辆通过UAV转发到RSU
        
        Returns:
            中继选项列表（可能有多个RSU选择）
        """
        opts: List[ProcessingOption] = []
        
        uav_state = states.get(uav_id)
        vehicle_pos = pos.get(task.source_vehicle_id)
        uav_pos = pos.get(uav_id)
        
        if not (uav_state and vehicle_pos and uav_pos):
            return opts
        
        # 遍历所有RSU，找到适合中继的目标
        for rsu_id, rsu_state in states.items():
            if not rsu_id.startswith('rsu_'):
                continue
            
            rsu_pos = pos.get(rsu_id)
            if not rsu_pos:
                continue
            
            # 判断是否适合中继
            if not self._should_use_uav_relay(vehicle_pos, uav_pos, rsu_pos, uav_state, rsu_state):
                continue
            
            # 计算中继时延和能耗
            # 阶段1：车辆 -> UAV
            v2u_distance = vehicle_pos.distance_to(uav_pos)
            reservation_key = getattr(task, 'source_vehicle_id', None)
            v2u_uplink = self._tx_delay(task.data_size, v2u_distance, reservation_key)
            v2u_energy = self._tx_energy(task.data_size, v2u_uplink)
            
            # 阶段2：UAV转发延迟（很小，几毫秒）
            relay_delay = 0.002  # 2ms转发延迟
            # UAV转发能耗（低功耗）
            uav_tx_power = dbm_to_watts(config.communication.uav_tx_power)
            relay_energy = uav_tx_power * relay_delay
            
            # 阶段3：UAV -> RSU
            u2r_distance = uav_pos.distance_to(rsu_pos)
            u2r_delay = self._tx_delay(task.data_size, u2r_distance)
            u2r_energy = uav_tx_power * u2r_delay
            
            # 阶段4：RSU处理
            proc = task.compute_cycles / max(1e-9, rsu_state.cpu_frequency)
            wait = self._wait(rsu_state)
            
            # 阶段5：RSU -> UAV -> 车辆（结果返回）
            result_down_u2r = self._tx_delay(task.result_size, u2r_distance)
            result_down_v2u = self._tx_delay(task.result_size, v2u_distance)
            result_energy = (uav_tx_power * (result_down_u2r + relay_delay) + 
                           self._tx_energy(task.result_size, result_down_v2u))
            
            # 总时延
            total_delay = (v2u_uplink + relay_delay + u2r_delay + 
                          wait + proc + 
                          result_down_u2r + relay_delay + result_down_v2u)
            
            # 总能耗：车辆发送 + UAV转发 + RSU计算 + 结果返回
            rsu_dynamic_power = config.compute.rsu_kappa2 * (rsu_state.cpu_frequency ** 3)
            rsu_compute_energy = rsu_dynamic_power * proc
            rsu_static_energy = config.compute.rsu_static_power * max(proc, config.network.time_slot_duration)
            
            total_energy = (v2u_energy + relay_energy + u2r_energy + 
                          rsu_compute_energy + rsu_static_energy + result_energy)
            
            # 成功率：中继链路稳定性稍低，但总体可靠
            success_prob = self._slack_prob(total_delay, task.max_delay_slots) * 0.9  # 0.9系数考虑中继风险
            
            opts.append(ProcessingOption(
                mode=ProcessingMode.UAV_RELAY,
                target_node_id=uav_id,  # 主目标是UAV
                relay_node_id=uav_id,
                final_target_id=rsu_id,  # 最终处理节点是RSU
                predicted_delay=self._cost_delay(total_delay, task),
                energy_cost=total_energy,
                success_probability=success_prob,
                latency_weight=self._alpha(task),
                reserved_processing_time=proc,
                reserved_comm_time=v2u_uplink + u2r_delay + result_down_u2r + result_down_v2u
            ))
        
        return opts

    def select_best_option(self, opts: List[ProcessingOption]) -> Optional[ProcessingOption]:
        opts = [o for o in opts if o and o.success_probability > 0.1]
        if not opts:
            return None
        return min(opts, key=lambda x: x.weighted_cost)


class OffloadingDecisionMaker:
    def __init__(self):
        self.classifier = TaskClassifier()
        self.evaluator = ProcessingModeEvaluator()
        self.decision_stats: Dict[ProcessingMode, int] = {m: 0 for m in ProcessingMode}
        self.total_decisions = 0

    def make_offloading_decision(self, task: Task, node_states: Dict, node_positions: Dict[str, 'Position'], cache_states: Optional[Dict] = None, control_preferences: Optional[Dict[str, Dict[str, float]]] = None) -> Optional[ProcessingOption]:
        if isinstance(control_preferences, dict):
            scheduling_pref = control_preferences.get('scheduling')
            if scheduling_pref:
                self.evaluator.update_scheduling_preferences(scheduling_pref)
            joint_pref = control_preferences.get('joint')
            if joint_pref is not None:
                self.evaluator.update_joint_tradeoff(joint_pref)
        elif isinstance(cache_states, dict):
            scheduling_pref = cache_states.get('scheduling_params')
            if scheduling_pref:
                self.evaluator.update_scheduling_preferences(scheduling_pref)
            joint_pref = cache_states.get('joint_strategy_params')
            if joint_pref is not None:
                self.evaluator.update_joint_tradeoff(joint_pref)

        slot_duration = max(1e-9, float(config.network.time_slot_duration))
        slot_index = None
        gen_time = getattr(task, 'generation_time', None)
        if isinstance(gen_time, (int, float)):
            slot_index = int(gen_time / slot_duration)
        self.evaluator.advance_virtual_time(slot_index)

        self.classifier.classify_task(task)
        cands = self.classifier.get_candidate_nodes(task, node_positions)
        options = self.evaluator.evaluate_all_modes(task, cands, node_states, node_positions, cache_states)
        best = self.evaluator.select_best_option(options)
        if best:
            self.decision_stats[best.mode] = self.decision_stats.get(best.mode, 0) + 1
            self.evaluator.reserve_resources(best, task, node_states)
        self.total_decisions += 1
        return best

    def get_decision_statistics(self) -> Dict:
        if self.total_decisions == 0:
            return {m.value: 0.0 for m in ProcessingMode}
        stats: Dict = {m.value: c / self.total_decisions for m, c in self.decision_stats.items()}
        stats['classification_distribution'] = self.classifier.get_classification_distribution()
        stats['total_decisions'] = self.total_decisions
        return stats
