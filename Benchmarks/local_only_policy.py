#!/usr/bin/env python3
"""
Local-only processing baseline: always keep tasks on the vehicle (no RSU/UAV offload).
Returns a continuous preference vector compatible with the other baseline outputs:
[local, rsu, uav, rsu_slots..., uav_slots...].
"""
from __future__ import annotations

import numpy as np


class LocalOnlyPolicy:
    def __init__(self, num_rsus: int = 0, num_uavs: int = 0, local_pref: float = 5.0):
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.local_pref = local_pref

    def select_action(self) -> np.ndarray:
        action_dim = 3 + self.num_rsus + self.num_uavs
        action = np.zeros(action_dim, dtype=np.float32)
        action[0] = self.local_pref  # strong local preference
        # rsu/uav prefs stay low, slot selections remain zero
        return action

    def select_action_with_dim(self, action_dim: int) -> np.ndarray:
        act = self.select_action()
        if act.size < action_dim:
            padded = np.zeros(action_dim, dtype=np.float32)
            padded[: act.size] = act
            return padded
        return act[:action_dim]


__all__ = ["LocalOnlyPolicy"]
