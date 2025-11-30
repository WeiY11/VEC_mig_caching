#!/usr/bin/env python3
"""
Dynamic offloading heuristic inspired by Nath & Wu (2020):
- Considers channel quality, queue length, and node load.
- Outputs continuous preference vector: [local, rsu, uav, rsu_slots..., uav_slots...].
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class NodeSnapshot:
    position: np.ndarray
    load: float
    queue: float
    channel: float


class DynamicOffloadHeuristic:
    def __init__(self, num_rsus: int, num_uavs: int, distance_scale: float = 1000.0):
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.distance_scale = distance_scale
        # Weights approximate Nath & Wu's focus on delay/energy trade-off.
        self.delay_weight = 0.6
        self.energy_weight = 0.4

    def _score_node(self, node: NodeSnapshot, weight_channel=0.5, weight_load=0.3, weight_queue=0.2) -> float:
        # Higher channel better, lower load/queue better, combines delay/energy proxy.
        ch = float(node.channel)
        load_penalty = weight_load * node.load
        queue_penalty = weight_queue * node.queue
        return weight_channel * ch - (self.delay_weight * queue_penalty + self.energy_weight * load_penalty)

    def _parse_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Robustly split flat state: vehicles(?*5) + rsu(num_rsus*5) + uav(num_uavs*5) + global(8)."""
        flat = np.array(state, dtype=np.float32).reshape(-1)
        
        # Ignore global state (last 8 dims) and potential central state
        # We assume the standard TD3Environment structure where nodes come first.
        # Vehicles (N*5) + RSUs (M*5) + UAVs (K*5) + ...
        
        node_dim = 5
        rsu_total = self.num_rsus * node_dim
        uav_total = self.num_uavs * node_dim
        
        # We don't know N (num_vehicles) exactly from state size if there's variable global/central state,
        # but we can assume the structure is [Vehicles, RSUs, UAVs, ...]
        # However, without knowing where Vehicles end, we can't easily parse if we don't know N.
        # But wait, the heuristic is initialized with num_rsus and num_uavs.
        # It doesn't know num_vehicles.
        # But usually num_vehicles is fixed or we can deduce it.
        # Let's assume the heuristic is used in a context where we can deduce N.
        # Or we can try to parse from the end, assuming we know the suffix size.
        # TD3Environment: [Vehicles, RSUs, UAVs, Central(opt), Global(8)]
        # If Central is 0, then suffix is 8.
        # But if Central is present, it's variable.
        # Benchmarks use TD3Environment with default use_central_resource=False.
        # So suffix is 8.
        
        suffix_len = 8
        useful_len = max(0, len(flat) - suffix_len)
        
        # The useful part is Vehicles + RSUs + UAVs
        # We know RSUs and UAVs sizes.
        veh_len = max(0, useful_len - rsu_total - uav_total)
        
        offset = 0
        veh = flat[offset : offset + veh_len]
        offset += veh_len
        rsu = flat[offset : offset + rsu_total]
        offset += rsu_total
        uav = flat[offset : offset + uav_total]
        
        return (
            veh.reshape(-1, node_dim) if veh_len > 0 else np.zeros((0, node_dim), dtype=np.float32),
            rsu.reshape(-1, node_dim),
            uav.reshape(-1, node_dim),
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        _, rsus, uavs = self._parse_state(state)
        candidates: list[tuple[str, int, float]] = []

        for i in range(self.num_rsus):
            if i >= rsus.shape[0]:
                break
            node = NodeSnapshot(
                position=rsus[i, :2],
                load=float(rsus[i, 3]) if rsus.shape[1] > 3 else 0.5,
                queue=float(rsus[i, 4]) if rsus.shape[1] > 4 else 0.5,
                channel=float(rsus[i, 2]) if rsus.shape[1] > 2 else 0.0,
            )
            candidates.append(("rsu", i, self._score_node(node)))

        for i in range(self.num_uavs):
            if i >= uavs.shape[0]:
                break
            node = NodeSnapshot(
                position=uavs[i, :2],
                load=float(uavs[i, 3]) if uavs.shape[1] > 3 else 0.6,
                queue=float(uavs[i, 4]) if uavs.shape[1] > 4 else 0.5,
                channel=float(uavs[i, 2]) if uavs.shape[1] > 2 else 0.0,
            )
            candidates.append(("uav", i, self._score_node(node, weight_channel=0.4)))

        # Build action vector
        action_dim = 3 + self.num_rsus + self.num_uavs
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = 0.0  # local score baseline
        action[1] = 0.0  # rsu pref
        action[2] = 0.0  # uav pref

        if not candidates:
            action[0] = 1.0
            return action

        # Softmax over scores
        scores = np.array([c[2] for c in candidates], dtype=np.float32)
        probs = np.exp(scores - scores.max())
        probs /= probs.sum()

        # Pick best for each type via argmax to avoid tuple/index pitfalls
        best_rsu_idx: Optional[int] = None
        best_uav_idx: Optional[int] = None
        best_rsu_p = -1.0
        best_uav_p = -1.0
        for (typ, idx, _score), p in zip(candidates, probs):
            if typ == "rsu" and p > best_rsu_p:
                best_rsu_idx, best_rsu_p = idx, float(p)
            if typ == "uav" and p > best_uav_p:
                best_uav_idx, best_uav_p = idx, float(p)

        if best_rsu_idx is not None:
            action[1] = best_rsu_p
            action[3 + best_rsu_idx] = 1.0
        if best_uav_idx is not None:
            action[2] = best_uav_p
            action[3 + self.num_rsus + best_uav_idx] = 1.0
        return action

    def select_action_with_dim(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """Pad/trim action to desired dimension for environments with larger action spaces."""
        act = self.select_action(state)
        if act.size < action_dim:
            padded = np.zeros(action_dim, dtype=np.float32)
            padded[: act.size] = act
            return padded
        return act[:action_dim]


__all__ = ["DynamicOffloadHeuristic"]
