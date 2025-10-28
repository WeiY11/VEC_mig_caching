#!/usr/bin/env python3
"""
Two-stage planning utilities.

Stage 1: Coarse task assignment and resource estimation
Stage 2: Fine-grained scheduling is performed by the simulator (RL controllers,
         cache/migration, queue processing) using the Stage 1 plan as guidance.

This module introduces a lightweight planner that operates on the current
CompleteSystemSimulator state (vehicles/RSUs/UAVs dicts) and a batch of newly
generated tasks in the current step. It returns a per-task plan entry with the
chosen target (local/RSU/UAV) and rough delay/energy estimates.

Design goals:
- Keep dependencies minimal; avoid importing heavy model classes.
- Work with existing simulator dict structures (ids like 'V_0', 'RSU_0', 'UAV_0').
- Avoid mutating simulator state during planning (estimation only).

Enable with env var TWO_STAGE_MODE=1 or the train_single_agent --two-stage flag.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class PlanEntry:
    task_id: str
    source_vehicle_id: str
    target_type: str  # 'local' | 'rsu' | 'uav'
    target_idx: Optional[int]  # index into simulator.rsus/uavs; None for local
    cache_hit: bool
    # Rough estimates (seconds / joules)
    est_uplink_time: float = 0.0
    est_queue_wait: float = 0.0
    est_compute_time: float = 0.0
    est_downlink_time: float = 0.0
    est_total_delay: float = 0.0
    est_energy: float = 0.0


class TwoStagePlanner:
    """Heuristic coarse assigner working on simulator dict state."""

    def __init__(self):
        # Base link rates and power assumptions aligned with simulator's
        # _estimate_transmission fallback values.
        self.base_rate_rsu = 80e6
        self.base_rate_uav = 45e6
        self.p_tx_rsu = 0.18
        self.p_tx_uav = 0.12

    # --- Helpers -----------------------------------------------------------------
    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    @staticmethod
    def _norm01(x: float, ref: float) -> float:
        if ref <= 0:
            return 0.0
        return max(0.0, min(1.0, x / ref))

    def _estimate_tx(self, bytes_size: float, distance_m: float, link: str) -> Tuple[float, float]:
        if link == 'uav':
            base_rate = self.base_rate_uav
            pwr = self.p_tx_uav
        else:
            base_rate = self.base_rate_rsu
            pwr = self.p_tx_rsu
        attenuation = 1.0 + max(0.0, distance_m) / 800.0
        rate = base_rate / max(1e-6, attenuation)
        delay = self._clip((bytes_size * 8.0) / max(rate, 1e6), 0.01, 1.2)
        energy = pwr * delay
        return delay, energy

    def _rsu_accessible(self, simulator, vehicle_pos, rsu) -> Tuple[bool, float]:
        cov = rsu.get('coverage_radius', 300.0)
        dist = simulator.calculate_distance(vehicle_pos, rsu['position'])
        return (dist <= cov), dist

    def _uav_accessible(self, simulator, vehicle_pos, uav) -> Tuple[bool, float]:
        cov = uav.get('coverage_radius', 350.0)
        dist = simulator.calculate_distance(vehicle_pos, uav['position'])
        return (dist <= cov), dist

    def _estimate_queue_wait(self, queue_len: int, node_type: str) -> float:
        # Rough mapping of queue length to expected waiting time (seconds)
        # tuned to simulator time_slot ~0.2s and typical service capacities.
        base_slots = 0.15 if node_type == 'RSU' else 0.22
        return self._clip(queue_len * base_slots, 0.0, 3.0)

    def _estimate_compute_time(self, comp_req_mips: float, node_type: str) -> float:
        # Convert MIPS-equivalent requirement to seconds using nominal capacities
        # consistent with _estimate_remote_work_units divisors.
        if node_type == 'RSU':
            denom = 1200.0
        else:
            denom = 1600.0
        work_units = max(0.5, min(12.0, float(comp_req_mips) / max(1e-6, denom)))
        # Approximate each work unit as ~0.08s compute time on-node.
        return self._clip(work_units * 0.08, 0.02, 2.0)

    # --- Planning ----------------------------------------------------------------
    def build_plan(self, simulator, tasks_batch: List[Tuple[int, Dict, Dict]]) -> Dict[str, PlanEntry]:
        """
        Build a per-task coarse assignment plan.

        Args:
            simulator: CompleteSystemSimulator instance
            tasks_batch: list of (vehicle_index, vehicle_dict, task_dict)

        Returns:
            Dict mapping task_id -> PlanEntry
        """
        plan: Dict[str, PlanEntry] = {}

        for vidx, vehicle, task in tasks_batch:
            t_id = task.get('id') or task.get('task_id') or f"task_{vidx}"
            v_pos = vehicle.get('position')
            data_bytes = float(task.get('data_size_bytes', task.get('data_size', 1.0) * 1e6))
            comp_req = float(task.get('computation_requirement', 1500.0))
            content_id = task.get('content_id')

            # Candidate RSUs (within coverage)
            rsu_candidates = []  # (score, idx, dist, cache_hit, qlen)
            for i, rsu in enumerate(simulator.rsus):
                in_range, dist = self._rsu_accessible(simulator, v_pos, rsu)
                if not in_range:
                    continue
                cache_hit = bool(content_id and content_id in rsu.get('cache', {}))
                queue_len = len(rsu.get('computation_queue', []))
                up_t, up_e = self._estimate_tx(data_bytes, dist, 'rsu')
                wait_t = 0.0 if cache_hit else self._estimate_queue_wait(queue_len, 'RSU')
                comp_t = 0.0 if cache_hit else self._estimate_compute_time(comp_req, 'RSU')
                total_delay = up_t + wait_t + comp_t
                # Scoring: prefer cache hit and shorter delay; lightly penalize queue
                score = total_delay + 0.03 * queue_len - (0.25 if cache_hit else 0.0)
                rsu_candidates.append((score, i, dist, cache_hit, queue_len, up_t, up_e, wait_t, comp_t))

            # Candidate UAVs (within coverage)
            uav_candidates = []  # (score, idx, dist, qlen)
            for j, uav in enumerate(simulator.uavs):
                in_range, dist = self._uav_accessible(simulator, v_pos, uav)
                if not in_range:
                    continue
                queue_len = len(uav.get('computation_queue', []))
                up_t, up_e = self._estimate_tx(data_bytes, dist, 'uav')
                wait_t = self._estimate_queue_wait(queue_len, 'UAV')
                comp_t = self._estimate_compute_time(comp_req, 'UAV')
                total_delay = up_t + wait_t + comp_t
                score = total_delay + 0.04 * queue_len  # slightly harsher queue penalty
                uav_candidates.append((score, j, dist, queue_len, up_t, up_e, wait_t, comp_t))

            # Local baseline estimate
            # Mirror simulator local compute roughness (2.5GHz, ~0.03-0.8s)
            cpu_freq = 2.5e9
            proc_time = self._clip((comp_req * 1e6) / max(cpu_freq, 1e6), 0.03, 0.8)
            local_energy = 6.5 * proc_time
            best_entry = PlanEntry(
                task_id=t_id,
                source_vehicle_id=vehicle.get('id', f'V_{vidx}'),
                target_type='local',
                target_idx=None,
                cache_hit=False,
                est_uplink_time=0.0,
                est_queue_wait=0.0,
                est_compute_time=proc_time,
                est_downlink_time=0.0,
                est_total_delay=proc_time,
                est_energy=local_energy,
            )

            # Prefer best RSU if better than local
            if rsu_candidates:
                rsu_candidates.sort(key=lambda x: x[0])
                s, idx, dist, cache_hit, qlen, up_t, up_e, wait_t, comp_t = rsu_candidates[0]
                rsu_total = up_t + wait_t + comp_t
                if rsu_total + 0.02 < best_entry.est_total_delay:
                    best_entry = PlanEntry(
                        task_id=t_id,
                        source_vehicle_id=vehicle.get('id', f'V_{vidx}'),
                        target_type='rsu',
                        target_idx=idx,
                        cache_hit=cache_hit,
                        est_uplink_time=up_t,
                        est_queue_wait=wait_t,
                        est_compute_time=comp_t,
                        est_downlink_time=0.0,  # result fairly small; omit here
                        est_total_delay=rsu_total,
                        est_energy=up_e,
                    )

            # Consider UAV if still better
            if uav_candidates:
                uav_candidates.sort(key=lambda x: x[0])
                s, idx, dist, qlen, up_t, up_e, wait_t, comp_t = uav_candidates[0]
                uav_total = up_t + wait_t + comp_t
                if uav_total + 0.02 < best_entry.est_total_delay:
                    best_entry = PlanEntry(
                        task_id=t_id,
                        source_vehicle_id=vehicle.get('id', f'V_{vidx}'),
                        target_type='uav',
                        target_idx=idx,
                        cache_hit=False,
                        est_uplink_time=up_t,
                        est_queue_wait=wait_t,
                        est_compute_time=comp_t,
                        est_downlink_time=0.0,
                        est_total_delay=uav_total,
                        est_energy=up_e,
                    )

            plan[t_id] = best_entry

        return plan

