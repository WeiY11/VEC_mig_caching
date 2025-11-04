#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAM-TD3 - 自适应混合式 TD3 策略

核心思想：
1. 在 TD3 连续控制策略的基础上，引入多层级启发式权重（车辆/RSU/UAV）的融合，
   动态平衡局部执行与边缘协同，改善高负载场景下的稳定性。
2. 使用系统状态中的队列长度、缓存利用率、能耗等指标构造启发式分布，
   与策略网络输出进行加权混合，缓解训练初期的探索偏差。
3. 在训练与推理阶段使用不同融合系数（training 更强调启发式，evaluation 更依赖策略网络），
   获得更平滑的收敛表现以及更好的泛化能力。
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from single_agent.td3 import TD3Environment


class CAMTD3Environment(TD3Environment):
    """继承 TD3Environment，并融合启发式分布。"""

    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2):
        super().__init__(num_vehicles=num_vehicles, num_rsus=num_rsus, num_uavs=num_uavs)
        self.fusion_weight_training = 0.35
        self.fusion_weight_eval = 0.25
        self.algorithm_label = "CAM-TD3"

    # ------------------------------------------------------------------ #
    # 核心：策略输出与启发式分布的融合
    # ------------------------------------------------------------------ #

    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        rl_actions = super().get_actions(state, training)
        metrics = self._extract_metric_distributions(state)

        vehicle_vec = rl_actions['vehicle_agent'].copy()

        allocation = vehicle_vec[:3]
        rsu_selection = vehicle_vec[3:3 + self.num_rsus]
        uav_selection = vehicle_vec[3 + self.num_rsus:3 + self.num_rsus + self.num_uavs]

        fusion_weight = self.fusion_weight_training if training else self.fusion_weight_eval

        allocation = self._blend_vector(allocation, metrics['target_weights'], fusion_weight)
        rsu_selection = self._blend_vector(rsu_selection, metrics['rsu_weights'], fusion_weight)
        uav_selection = self._blend_vector(uav_selection, metrics['uav_weights'], fusion_weight)

        vehicle_vec[:3] = allocation
        vehicle_vec[3:3 + self.num_rsus] = rsu_selection
        vehicle_vec[3 + self.num_rsus:3 + self.num_rsus + self.num_uavs] = uav_selection

        rl_actions['vehicle_agent'] = vehicle_vec
        rl_actions['rsu_agent'] = rsu_selection
        rl_actions['uav_agent'] = uav_selection

        return rl_actions

    # ------------------------------------------------------------------ #
    # 辅助函数
    # ------------------------------------------------------------------ #

    def _extract_metric_distributions(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据状态构造启发式分布。
        
        状态结构: [vehicle×5] + [rsu×5] + [uav×5] + [global_state]
        注意：必须正确计算偏移量，避免提取到global_state
        """
        # Vehicle状态 (12×5 = 60维)
        vehicle_end = self.num_vehicles * 5
        vehicle_slice = state[:vehicle_end].reshape(self.num_vehicles, 5)
        
        # RSU状态 (4×5 = 20维)
        rsu_start = vehicle_end
        rsu_end = rsu_start + self.num_rsus * 5
        rsu_slice = state[rsu_start:rsu_end].reshape(self.num_rsus, 5)
        
        # UAV状态 (2×5 = 10维)
        uav_start = rsu_end
        uav_end = uav_start + self.num_uavs * 5
        # ✅ 修复：确保不会越界到global_state
        uav_end = min(uav_end, len(state))
        uav_slice = state[uav_start:uav_end].reshape(-1, 5) if (uav_end - uav_start) >= 5 else state[uav_start:uav_end].reshape(self.num_uavs, 5)

        vehicle_queue = vehicle_slice[:, 3] if vehicle_slice.size else np.array([0.5])
        vehicle_energy = vehicle_slice[:, 4] if vehicle_slice.size else np.array([0.5])

        rsu_queue = rsu_slice[:, 3] if rsu_slice.size else np.full(self.num_rsus, 0.5)
        rsu_cache = rsu_slice[:, 2] if rsu_slice.size else np.full(self.num_rsus, 0.5)

        uav_cache = uav_slice[:, 3] if uav_slice.size else np.full(self.num_uavs, 0.5)
        uav_energy = uav_slice[:, 4] if uav_slice.size else np.full(self.num_uavs, 0.5)

        vehicle_score = (1.0 - vehicle_queue.mean()) * 0.6 + (1.0 - vehicle_energy.mean()) * 0.4
        rsu_scores = (1.0 - rsu_queue) * 0.7 + rsu_cache * 0.3
        uav_scores = (1.0 - uav_energy) * 0.5 + uav_cache * 0.5

        target_weights = np.array([
            max(vehicle_score, 0.05),
            max(rsu_scores.mean(), 0.05),
            max(uav_scores.mean(), 0.05),
        ])

        rsu_weights = rsu_scores + 1e-3
        uav_weights = uav_scores + 1e-3

        return {
            'target_weights': self._normalize(target_weights),
            'rsu_weights': self._normalize(rsu_weights),
            'uav_weights': self._normalize(uav_weights),
        }

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 1e-4, None)
        total = clipped.sum()
        if total <= 0:
            return np.full_like(clipped, 1.0 / len(clipped))
        return clipped / total

    def _blend_vector(self, rl_vector: np.ndarray, heuristic_weights: np.ndarray, fusion_weight: float) -> np.ndarray:
        rl_prob = self._to_probability(rl_vector, heuristic_weights.shape[0])
        blended = (1.0 - fusion_weight) * rl_prob + fusion_weight * heuristic_weights
        blended = self._normalize(blended)
        return self._from_probability(blended)

    @staticmethod
    def _to_probability(vector: np.ndarray, target_dim: int) -> np.ndarray:
        padded = np.zeros(target_dim, dtype=float)
        length = min(target_dim, vector.shape[0])
        padded[:length] = np.clip(vector[:length], -1.0, 1.0)
        prob = (padded + 1.0) / 2.0
        total = prob.sum()
        if total <= 0:
            return np.full(target_dim, 1.0 / target_dim)
        return prob / total

    @staticmethod
    def _from_probability(prob: np.ndarray) -> np.ndarray:
        return np.clip(prob * 2.0 - 1.0, -0.99, 0.99)

