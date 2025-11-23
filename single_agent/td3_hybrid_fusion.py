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
import os

from single_agent.td3 import TD3Environment


class CAMTD3Environment(TD3Environment):
    """继承 TD3Environment，并融合启发式分布。"""

    def __init__(self, num_vehicles: int = 12, num_rsus: int = 4, num_uavs: int = 2, use_central_resource: bool = False):
        super().__init__(num_vehicles=num_vehicles, num_rsus=num_rsus, num_uavs=num_uavs, use_central_resource=use_central_resource)
        def _read_env(key: str, default: float) -> float:
            try:
                return float(os.environ.get(key, default))
            except Exception:
                return default
        # 优化融合权重：训练阶段大幅降低启发式干扰，让策略网络主导学习
        self.fusion_weight_training = _read_env("CAM_FUSION_TRAIN_WEIGHT", 0.05)  # 降低到5%
        self.fusion_weight_eval = _read_env("CAM_FUSION_EVAL_WEIGHT", 0.10)  # 降低到10%
        self.algorithm_label = "CAM-TD3"
        
        # 计算状态结构偏移量（支持中央资源模式）
        self._vehicle_state_end = num_vehicles * 5
        self._rsu_state_start = self._vehicle_state_end
        self._rsu_state_end = self._rsu_state_start + num_rsus * 5
        self._uav_state_start = self._rsu_state_end
        self._uav_state_end = self._uav_state_start + num_uavs * 5
        self._local_state_end = self._uav_state_end
        
        # 进度追踪用于自适应融合
        self.episode_count = 0
        self.best_reward = float('-inf')
        self.convergence_threshold = 0.95  # 收敛后减少融合

    # ------------------------------------------------------------------ #
    # 核心：策略输出与启发式分布的融合
    # ------------------------------------------------------------------ #

    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict[str, np.ndarray]:
        rl_actions = super().get_actions(state, training)
        metrics = self._extract_metric_distributions(state)

        vehicle_vec = rl_actions['vehicle_agent'].copy()

        # 解析动作段：任务分配(3) + RSU选择 + UAV选择 + 控制参数(10)
        allocation = vehicle_vec[:3]
        rsu_selection = vehicle_vec[3:3 + self.num_rsus]
        uav_selection = vehicle_vec[3 + self.num_rsus:3 + self.num_rsus + self.num_uavs]
        control_params_start = 3 + self.num_rsus + self.num_uavs
        control_params = vehicle_vec[control_params_start:control_params_start + 10]

        # 自适应融合权重：随训练进度逐步降低启发式影响
        base_fusion_weight = self.fusion_weight_training if training else self.fusion_weight_eval
        
        # 前200轮：使用基础权重；200轮后逐步降低到0
        if self.episode_count > 200:
            decay_factor = max(0.0, 1.0 - (self.episode_count - 200) / 800.0)
            fusion_weight = base_fusion_weight * decay_factor
        else:
            fusion_weight = base_fusion_weight
        
        # 只在融合权重大于阈值时才进行融合
        if fusion_weight > 0.01:
            # 融合任务分配、RSU选择、UAV选择
            allocation = self._blend_vector(allocation, metrics['target_weights'], fusion_weight)
            rsu_selection = self._blend_vector(rsu_selection, metrics['rsu_weights'], fusion_weight)
            uav_selection = self._blend_vector(uav_selection, metrics['uav_weights'], fusion_weight)
            
            # 控制参数使用更轻的融合
            if 'control_guidance' in metrics:
                control_params = self._blend_vector(control_params, metrics['control_guidance'], fusion_weight * 0.3)

        # 重新组装动作向量
        vehicle_vec[:3] = allocation
        vehicle_vec[3:3 + self.num_rsus] = rsu_selection
        vehicle_vec[3 + self.num_rsus:3 + self.num_rsus + self.num_uavs] = uav_selection
        vehicle_vec[control_params_start:control_params_start + 10] = control_params

        rl_actions['vehicle_agent'] = vehicle_vec
        rl_actions['rsu_agent'] = rsu_selection
        rl_actions['uav_agent'] = uav_selection

        return rl_actions
    
    def on_episode_end(self, episode_reward: float):
        """每个episode结束时调用，更新融合策略"""
        self.episode_count += 1
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward

    # ------------------------------------------------------------------ #
    # 辅助函数
    # ------------------------------------------------------------------ #

    def _extract_metric_distributions(self, state: np.ndarray) -> Dict[str, np.ndarray]:
        """
        根据状态构造启发式分布。
        
        状态结构: [vehicle×5] + [rsu×5] + [uav×5] + [中央资源状态(可选)] + [global_state]
        支持中央资源模式和标准模式
        """
        # 状态维度校验
        if len(state) < self._local_state_end:
            raise ValueError(
                f"状态维度不足: 需要至少{self._local_state_end}维（局部状态），实际{len(state)}维"
            )
        
        # Vehicle状态 (N×5维)
        vehicle_slice = state[:self._vehicle_state_end].reshape(self.num_vehicles, 5)
        
        # RSU状态 (M×5维)
        rsu_slice = state[self._rsu_state_start:self._rsu_state_end].reshape(self.num_rsus, 5)
        
        # UAV状态 (K×5维)
        uav_data = state[self._uav_state_start:self._uav_state_end]
        if len(uav_data) >= self.num_uavs * 5:
            uav_slice = uav_data[:self.num_uavs * 5].reshape(self.num_uavs, 5)
        else:
            # 边界情况：填充默认值
            uav_slice = np.full((self.num_uavs, 5), 0.5)
            uav_slice[:len(uav_data)//5, :] = uav_data[:len(uav_data)//5*5].reshape(-1, 5)

        # 提取关键指标（列索引含义: 0=计算资源, 1=距离, 2=缓存, 3=队列, 4=能耗）
        vehicle_queue = vehicle_slice[:, 3]
        vehicle_energy = vehicle_slice[:, 4]

        rsu_queue = rsu_slice[:, 3]
        rsu_cache = rsu_slice[:, 2]
        rsu_compute = rsu_slice[:, 0]  # 新增：RSU计算资源

        # UAV状态修正：UAV没有缓存（列2），队列在列3，能耗在列4
        uav_queue = uav_slice[:, 3]
        uav_compute = uav_slice[:, 0]
        uav_distance = uav_slice[:, 1]

        # 自适应权重计算（根据系统负载动态调整）
        avg_queue = np.concatenate([vehicle_queue, rsu_queue, uav_queue]).mean()
        load_factor = np.clip(avg_queue, 0.0, 1.0)  # 系统负载因子
        
        # 车辆得分：降低基础权重，避免过度偏向本地
        # 考虑本地计算的高能耗成本（能耗权重提高到0.6）
        vehicle_score = (1.0 - vehicle_queue.mean()) * 0.4 + (1.0 - vehicle_energy.mean()) * 0.6
        # 添加本地计算惩罚因子（模拟本地资源受限）
        local_penalty = 0.15  # 15%的基础惩罚
        vehicle_score = vehicle_score * (1.0 - local_penalty)
        
        # RSU得分：提升基础权重，强化边缘计算优势
        # 降低队列权重，提高缓存和计算资源权重
        queue_weight = 0.5 + 0.1 * load_factor  # 降低队列权重
        cache_weight = 0.3  # 固定缓存权重30%
        compute_weight = 0.2 - 0.1 * load_factor  # 计算资源权重20%
        rsu_scores = (1.0 - rsu_queue) * queue_weight + rsu_cache * cache_weight + rsu_compute * compute_weight
        # 添加RSU协同处理奖励
        rsu_bonus = 0.20  # 20%的协同奖励
        rsu_scores = rsu_scores * (1.0 + rsu_bonus)
        
        # UAV得分：提升权重，强化移动边缘计算能力
        # 降低距离惩罚，UAV的移动性是优势而非劣势
        uav_scores = (1.0 - uav_queue) * 0.5 + (1.0 - uav_distance) * 0.2 + uav_compute * 0.3
        # 添加UAV灵活性奖励
        uav_bonus = 0.25  # 25%的灵活性奖励
        uav_scores = uav_scores * (1.0 + uav_bonus)

        target_weights = np.array([
            max(vehicle_score, 0.05),
            max(rsu_scores.mean(), 0.05),
            max(uav_scores.mean(), 0.05),
        ])

        rsu_weights = rsu_scores + 1e-3
        uav_weights = uav_scores + 1e-3
        
        # 构造控制参数引导（10维：缓存阈值、迁移阈值等）
        control_guidance = self._build_control_guidance(
            vehicle_queue, rsu_cache, avg_queue, load_factor
        )

        return {
            'target_weights': self._normalize(target_weights),
            'rsu_weights': self._normalize(rsu_weights),
            'uav_weights': self._normalize(uav_weights),
            'control_guidance': control_guidance,
        }

    def _build_control_guidance(self, vehicle_queue: np.ndarray, rsu_cache: np.ndarray, 
                                avg_queue: float, load_factor: float) -> np.ndarray:
        """
        构造控制参数的启发式引导（10维）
        控制参数含义：缓存策略、迁移阈值、资源分配偏好等
        """
        # 高负载时：激进缓存淘汰、提高迁移阈值
        # 低负载时：保守缓存、降低迁移阈值
        cache_aggressiveness = load_factor  # [0,1] 高负载→激进淘汰
        migration_threshold = 0.3 + 0.4 * load_factor  # 高负载→提高迁移门槛
        
        # 资源分配偏好：大幅提升RSU/UAV偏好，降低本地偏好
        rsu_preference = 0.65 + 0.2 * load_factor  # 提高RSU基础偏好到65%
        local_preference = 0.25 - 0.15 * load_factor  # 降低本地偏好到25%
        uav_preference = 0.55 + 0.15 * load_factor  # 新增：UAV偏好55%
        
        # 队列管理：根据平均队列长度调整
        queue_sensitivity = 1.0 - avg_queue
        
        # 缓存命中率目标：根据RSU缓存状态
        cache_target = rsu_cache.mean()
        
        # 构适10维引导向量（映射到[-1, 1]）
        guidance = np.array([
            cache_aggressiveness * 2 - 1,      # 缓存淘汰激进度
            migration_threshold * 2 - 1,       # 迁移触发阈值
            rsu_preference * 2 - 1,            # RSU偏好（提高）
            local_preference * 2 - 1,          # 本地执行偏好（降低）
            queue_sensitivity * 2 - 1,         # 队列敏感度
            cache_target * 2 - 1,              # 缓存目标
            load_factor * 2 - 1,               # 负载感知因子
            uav_preference * 2 - 1,            # UAV偏好（新增）
            0.0,                               # 保留维度2
            0.0,                               # 保留维度3
        ])
        
        return np.clip(guidance, -0.99, 0.99)
    
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
