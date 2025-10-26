"""
任务分类与卸载决策（优化版，200ms时隙 + 表IV 参数对齐）
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from config import config
from models.data_structures import Task, TaskType


class ProcessingMode(Enum):
    LOCAL_COMPUTING = "local"
    RSU_OFFLOAD_CACHE_HIT = "rsu_hit"
    RSU_OFFLOAD_NO_CACHE = "rsu_miss"
    RSU_MIGRATION = "rsu_migration"
    UAV_OFFLOAD = "uav_offload"


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

    @property
    def weighted_cost(self) -> float:
        slot = max(1e-9, float(config.network.time_slot_duration))
        energy_norm = 1000.0
        w_delay, w_energy, w_rel = 0.5, 0.4, 0.1
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
        s = task.max_delay_slots
        if s <= self.threshold_1:
            t = TaskType.EXTREMELY_DELAY_SENSITIVE
        elif s <= self.threshold_2:
            t = TaskType.DELAY_SENSITIVE
        elif s <= self.threshold_3:
            t = TaskType.MODERATELY_DELAY_TOLERANT
        else:
            t = TaskType.DELAY_TOLERANT
        task.task_type = t
        self.classification_stats[t] += 1
        return t

    def get_candidate_nodes(self, task: Task, all_nodes: Dict[str, 'Position']) -> List[str]:
        t = task.task_type
        vid = task.source_vehicle_id
        if t == TaskType.EXTREMELY_DELAY_SENSITIVE:
            return [vid]
        if t == TaskType.DELAY_SENSITIVE:
            c = [vid]
            c.extend(self._get_nearby_rsus(vid, all_nodes, 600.0, 2))
            c.extend(self._get_capable_uavs(vid, all_nodes, 400.0))
            return c
        if t == TaskType.MODERATELY_DELAY_TOLERANT:
            c = [vid]
            c.extend(self._get_reachable_rsus(vid, all_nodes, 800.0))
            c.extend(self._get_capable_uavs(vid, all_nodes, 600.0))
            return c
        return list(all_nodes.keys())

    def _get_nearby_rsus(self, vid: str, nodes: Dict[str, 'Position'], max_d: float, k: int) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        arr = [(v.distance_to(p), nid) for nid, p in nodes.items() if nid.startswith('rsu_') and v.distance_to(p) <= max_d]
        arr.sort(key=lambda x: x[0])
        return [nid for _, nid in arr[:k]]

    def _get_reachable_rsus(self, vid: str, nodes: Dict[str, 'Position'], max_d: float) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        return [nid for nid, p in nodes.items() if nid.startswith('rsu_') and v.distance_to(p) <= max_d]

    def _get_capable_uavs(self, vid: str, nodes: Dict[str, 'Position'], max_d: float) -> List[str]:
        if vid not in nodes:
            return []
        v = nodes[vid]
        return [nid for nid, p in nodes.items() if nid.startswith('uav_') and v.distance_2d_to(p) <= max_d]

    def get_classification_distribution(self) -> Dict[TaskType, float]:
        total = sum(self.classification_stats.values()) or 1
        return {k: v / total for k, v in self.classification_stats.items()}


class ProcessingModeEvaluator:
    def __init__(self):
        self.communication_overhead = 0.0002
        self.cache_response_delay = 0.0001

    def evaluate_all_modes(self, task: Task, candidates: List[str], node_states: Dict,
                           node_positions: Dict[str, 'Position'], cache_states: Optional[Dict] = None) -> List[ProcessingOption]:
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
                o = self._eval_uav(task, nid, node_states, node_positions)
                if o:
                    opts.append(o)
        return opts

    def _alpha(self, task: Task) -> float:
        try:
            return float(config.task.get_latency_cost_weight(int(task.task_type.value)))
        except Exception:
            return 1.0

    def _slack_prob(self, delay: float, slots: int) -> float:
        slot = config.network.time_slot_duration
        slack = slots * slot - delay
        if slack >= 0.2 * slot:
            return 0.9
        if slack >= 0:
            return 0.7
        return 0.25

    def _cost_delay(self, delay: float, task: Task) -> float:
        # 传给 weighted_cost 的“成本延迟”输入
        slot = max(1e-9, float(config.network.time_slot_duration))
        return delay * (0.15 / slot) * self._alpha(task)

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
        )

    def _eval_rsu(self, task: Task, rid: str, states: Dict, pos: Dict[str, 'Position'], caches: Dict) -> List[ProcessingOption]:
        out: List[ProcessingOption] = []
        st = states.get(rid)
        vpos = pos.get(task.source_vehicle_id)
        rpos = pos.get(rid)
        if not (st and vpos and rpos):
            return out
        d = vpos.distance_to(rpos)
        up = self._tx_delay(task.data_size, d)
        down = self._tx_delay(task.result_size, d)
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
            ))
        else:
            proc = task.compute_cycles / max(1e-9, st.cpu_frequency)
            wait = self._wait(st)
            total = up + wait + proc + down
            energy = self._tx_energy(task.data_size + task.result_size, up + down)
            out.append(ProcessingOption(
                mode=ProcessingMode.RSU_OFFLOAD_NO_CACHE,
                target_node_id=rid,
                predicted_delay=self._cost_delay(total, task),
                energy_cost=energy,
                success_probability=self._slack_prob(total, task.max_delay_slots),
                latency_weight=self._alpha(task),
            ))
        return out

    def _eval_rsu_mig(self, task: Task, target: str, states: Dict, pos: Dict[str, 'Position']) -> List[ProcessingOption]:
        out: List[ProcessingOption] = []
        tgt = states.get(target)
        if not tgt:
            return out
        for sid, s in states.items():
            if not sid.startswith('rsu_') or sid == target:
                continue
            if float(getattr(s, 'load_factor', 0.0)) <= config.migration.rsu_overload_threshold:
                continue
            mig = task.data_size / max(1e-9, config.migration.migration_bandwidth)
            proc = task.compute_cycles / max(1e-9, tgt.cpu_frequency)
            wait = self._wait(tgt)
            total = mig + wait + proc
            energy = config.communication.rsu_tx_power * mig
            out.append(ProcessingOption(
                mode=ProcessingMode.RSU_MIGRATION,
                target_node_id=target,
                predicted_delay=self._cost_delay(total, task),
                energy_cost=energy,
                success_probability=0.9 if total <= task.max_delay_slots * config.network.time_slot_duration else 0.25,
                migration_source=sid,
                latency_weight=self._alpha(task),
            ))
        return out

    def _eval_uav(self, task: Task, uid: str, states: Dict, pos: Dict[str, 'Position']) -> Optional[ProcessingOption]:
        uv = states.get(uid)
        vpos = pos.get(task.source_vehicle_id)
        upos = pos.get(uid)
        if not (uv and vpos and upos):
            return None
        d = vpos.distance_to(upos)
        up = self._tx_delay(task.data_size, d)
        down = self._tx_delay(task.result_size, d)
        eff = uv.cpu_frequency * max(0.5, float(getattr(uv, 'battery_level', 1.0)))
        proc = task.compute_cycles / max(1e-9, eff)
        wait = self._wait(uv)
        total = up + wait + proc + down
        energy = self._tx_energy(task.data_size + task.result_size, up + down) + config.compute.uav_kappa3 * (uv.cpu_frequency ** 2) * proc
        return ProcessingOption(
            mode=ProcessingMode.UAV_OFFLOAD,
            target_node_id=uid,
            predicted_delay=self._cost_delay(total, task),
            energy_cost=energy,
            success_probability=self._slack_prob(total, task.max_delay_slots),
            latency_weight=self._alpha(task),
        )

    # helpers
    def _cache_hit(self, task: Task, rid: str, caches: Dict) -> bool:
        if not caches or rid not in caches:
            return False
        sig = f"{task.task_type.value}_{int(task.data_size)}_{int(task.compute_cycles)}"
        return sig in caches.get(rid, {})

    def _tx_delay(self, bits: float, d: float) -> float:
        base = 50e6
        loss = 1.0 + (d / 2000.0) ** 1.5
        rate = base / max(1e-9, loss)
        return bits / max(1e-9, rate) + self.communication_overhead

    def _tx_energy(self, bits: float, t: float) -> float:
        return (config.communication.vehicle_tx_power + config.communication.circuit_power) * max(0.0, t)

    def _energy_local(self, task: Task, st) -> float:
        proc = task.compute_cycles / max(1e-9, (st.cpu_frequency * config.compute.parallel_efficiency))
        util = min(1.0, proc / max(1e-9, config.network.time_slot_duration))
        dyn = (config.compute.vehicle_kappa1 * (st.cpu_frequency ** 3) +
               config.compute.vehicle_kappa2 * (st.cpu_frequency ** 2) * util +
               config.compute.vehicle_static_power)
        return dyn * proc

    def _wait(self, st) -> float:
        rho = float(getattr(st, 'load_factor', 0.0))
        if rho >= 0.999:
            return float('inf')
        base = 0.06
        return max(0.0, (rho * base) / max(1e-6, 1 - rho))

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

    def make_offloading_decision(self, task: Task, node_states: Dict, node_positions: Dict[str, 'Position'], cache_states: Optional[Dict] = None) -> Optional[ProcessingOption]:
        self.classifier.classify_task(task)
        cands = self.classifier.get_candidate_nodes(task, node_positions)
        options = self.evaluator.evaluate_all_modes(task, cands, node_states, node_positions, cache_states)
        best = self.evaluator.select_best_option(options)
        if best:
            self.decision_stats[best.mode] += 1
        self.total_decisions += 1
        return best

    def get_decision_statistics(self) -> Dict:
        if self.total_decisions == 0:
            return {m.value: 0.0 for m in ProcessingMode}
        stats = {m.value: c / self.total_decisions for m, c in self.decision_stats.items()}
        stats['classification_distribution'] = self.classifier.get_classification_distribution()
        stats['total_decisions'] = self.total_decisions
        return stats

