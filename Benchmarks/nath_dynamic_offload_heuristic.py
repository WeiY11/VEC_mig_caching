#!/usr/bin/env python3
"""
Dynamic offloading heuristic based on Nath & Wu 2020.

Decisions are made by:
1. Parsing the flat state vector into (Vehicles, RSUs, UAVs, Global).
2. For each vehicle, deciding offloading based on:
   - Local queue length vs. RSU/UAV queue length
   - Channel quality (distance proxy)
   - Energy levels
3. Outputting a continuous action vector compatible with OptimizedTD3.

Note: This implementation uses *global* average decisions for the centralized
agent setting. The original Nath & Wu paper is per-vehicle, which would 
require a different environment interface.
"""
import numpy as np

# 导入统一状态维度常量，确保与环境一致
try:
    from single_agent.common_state_action import (
        STATE_DIM_PER_VEHICLE,  # 5
        STATE_DIM_PER_RSU,      # 5
        STATE_DIM_PER_UAV,      # 5
        STATE_DIM_GLOBAL,       # 20
        STATE_DIM_CENTRAL,      # 16
    )
except ImportError:
    # Fallback defaults (should match common_state_action)
    STATE_DIM_PER_VEHICLE = 5
    STATE_DIM_PER_RSU = 5
    STATE_DIM_PER_UAV = 5
    STATE_DIM_GLOBAL = 20
    STATE_DIM_CENTRAL = 16


class DynamicOffloadHeuristic:
    """
    Dynamic offloading heuristic based on Nath & Wu 2020.
    """
    def __init__(self, num_rsus: int, num_uavs: int, num_vehicles: int = 12, 
                 use_central_resource: bool = True):
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.num_vehicles = num_vehicles
        self.use_central_resource = use_central_resource
        
        # State dimensions from unified constants
        self.veh_dim = STATE_DIM_PER_VEHICLE  # 5: [x, y, vel, queue, energy]
        self.rsu_dim = STATE_DIM_PER_RSU      # 5: [x, y, cache, queue, energy]
        self.uav_dim = STATE_DIM_PER_UAV      # 5: [x, y, queue, cache, energy]
        self.global_dim = STATE_DIM_GLOBAL    # 20
        self.central_dim = STATE_DIM_CENTRAL if use_central_resource else 0  # 16 or 0
        
        # Calculate expected lengths
        self.veh_total = num_vehicles * self.veh_dim
        self.rsu_total = num_rsus * self.rsu_dim
        self.uav_total = num_uavs * self.uav_dim
        self.suffix_len = self.global_dim + self.central_dim
        
        # Heuristic thresholds
        self.queue_threshold = 0.6  # If queue > 60%, try to offload
        
        # Weights for scoring
        self.delay_weight = 0.5
        self.energy_weight = 0.5

    def _parse_state(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Parse flat state vector using unified dimension constants.
        
        State layout:
            vehicles (num_vehicles * 5) + 
            rsu (num_rsus * 5) + 
            uav (num_uavs * 5) + 
            global (20) + 
            central (16, if enabled)
        
        Vehicle state (5D): [pos_x, pos_y, velocity, queue, energy]
        RSU state (5D): [pos_x, pos_y, cache_util, queue, energy]
        UAV state (5D): [pos_x, pos_y, queue, cache_util, energy]
        """
        flat = np.array(state, dtype=np.float32).reshape(-1)
        
        # 直接使用已知维度进行切分，无需猜测
        expected_len = self.veh_total + self.rsu_total + self.uav_total + self.suffix_len
        
        if len(flat) != expected_len:
            # 尝试自动检测中央资源状态是否存在
            len_with_central = self.veh_total + self.rsu_total + self.uav_total + self.global_dim + STATE_DIM_CENTRAL
            len_without_central = self.veh_total + self.rsu_total + self.uav_total + self.global_dim
            
            if len(flat) == len_with_central:
                self.suffix_len = self.global_dim + STATE_DIM_CENTRAL
            elif len(flat) == len_without_central:
                self.suffix_len = self.global_dim
            # 注: 不再打印警告，静默处理
        
        # 切分状态向量
        idx = 0
        veh = flat[idx:idx + self.veh_total]
        idx += self.veh_total
        rsu = flat[idx:idx + self.rsu_total]
        idx += self.rsu_total
        uav = flat[idx:idx + self.uav_total]
        # idx += self.uav_total  # remaining is global + central, not used in heuristic
        
        return (
            veh.reshape(self.num_vehicles, self.veh_dim),
            rsu.reshape(self.num_rsus, self.rsu_dim),
            uav.reshape(self.num_uavs, self.uav_dim),
        )

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        Generate action vector based on heuristic rules.
        
        Action structure (OptimizedTD3):
        [
          offload_prob_local, offload_prob_rsu, offload_prob_uav,  (3)
          rsu_selection_probs (num_rsus),                          (4)
          uav_selection_probs (num_uavs),                          (2)
          control_params (10)                                      (10)
        ]
        Total dim = 3 + 4 + 2 + 10 = 19.
        """
        veh_states, rsu_states, uav_states = self._parse_state(state)
        
        # Calculate average queue load across all vehicles
        # Vehicle state: [x, y, vel, queue, energy] -> queue is index 3
        avg_veh_queue = np.mean(veh_states[:, 3]) if len(veh_states) > 0 else 0
        
        # Decision logic based on Nath & Wu 2020:
        # If vehicle queue load is high, prefer offloading to RSU/UAV
        # If load is low, prefer local processing
        
        offload_prob_local = 1.0
        offload_prob_rsu = 0.0
        offload_prob_uav = 0.0
        
        if avg_veh_queue > self.queue_threshold:
            offload_prob_local = 0.2
            offload_prob_rsu = 0.5
            offload_prob_uav = 0.3
        
        # RSU selection: prefer RSU with lowest queue (index 3)
        # RSU state (5D): [pos_x, pos_y, cache_util, queue, energy]
        rsu_queues = rsu_states[:, 3]  # queue is index 3
        # Softmax over negative queue (lower queue = higher probability)
        rsu_probs = np.exp(-5.0 * rsu_queues)
        rsu_probs /= (np.sum(rsu_probs) + 1e-6)
        
        # UAV selection: prefer UAV with lowest energy consumption (index 4)
        # UAV state (5D): [pos_x, pos_y, queue, cache_util, energy]
        uav_energy = uav_states[:, 4]  # energy is index 4
        uav_probs = np.exp(-5.0 * uav_energy)
        uav_probs /= (np.sum(uav_probs) + 1e-6)
        
        # Construct action: 3 (offload) + num_rsus + num_uavs + 10 (control)
        action = np.concatenate([
            [offload_prob_local, offload_prob_rsu, offload_prob_uav],
            rsu_probs,
            uav_probs,
            np.zeros(10, dtype=np.float32)  # Control params (not used by heuristic)
        ])
        
        return action.astype(np.float32)

    def select_action_with_dim(self, state: np.ndarray, action_dim: int) -> np.ndarray:
        """Wrapper to ensure action dimension matches environment."""
        action = self.select_action(state)
        if len(action) < action_dim:
            # Pad with zeros
            padding = np.zeros(action_dim - len(action), dtype=np.float32)
            action = np.concatenate([action, padding])
        elif len(action) > action_dim:
            # Truncate
            action = action[:action_dim]
        return action


__all__ = ["DynamicOffloadHeuristic"]
