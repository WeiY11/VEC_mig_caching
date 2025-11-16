"""
Fallback baseline and meta-heuristic policies used when the
`baseline_comparison` package is unavailable.

These lightweight implementations provide deterministic heuristic behaviour
that matches the action interface expected by `SingleAgentTrainingEnvironment`.
They intentionally favour reproducibility over sophistication so that
experiment automation continues to work even in minimal environments.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def _safe_rng(seed: Optional[int] = None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(int(seed))


@dataclass
class EnvironmentSnapshot:
    num_vehicles: int
    num_rsus: int
    num_uavs: int
    action_dim: int


class HeuristicPolicy:
    """Base class for simple heuristic controllers."""

    def __init__(self, name: str, seed: Optional[int] = None) -> None:
        self.name = name
        self.rng = _safe_rng(seed)
        self.snapshot: Optional[EnvironmentSnapshot] = None
        self._env = None

    # --- interface expected by AlgorithmComparisonRunner ---
    def update_environment(self, env) -> None:  # type: ignore[override]
        action_dim = getattr(getattr(env, "agent_env", None), "action_dim", 18)
        num_vehicles = len(getattr(env.simulator, "vehicles", []) or getattr(env.simulator, "vehicles_template", []))
        num_rsus = len(getattr(env.simulator, "rsus", []))
        num_uavs = len(getattr(env.simulator, "uavs", []))
        self.snapshot = EnvironmentSnapshot(
            num_vehicles=num_vehicles,
            num_rsus=num_rsus,
            num_uavs=num_uavs,
            action_dim=action_dim,
        )
        self._env = env

    def reset(self) -> None:  # pragma: no cover - rarely needs custom logic
        """Reset controller state at the start of each episode."""
        return

    def select_action(self, state) -> np.ndarray:  # pragma: no cover - implemented by subclasses
        raise NotImplementedError

    # --- helpers ---------------------------------------------------------
    def _ensure_snapshot(self) -> EnvironmentSnapshot:
        if self.snapshot is None:
            raise RuntimeError("Environment snapshot not available. Call update_environment first.")
        return self.snapshot

    def _blank_action(self) -> np.ndarray:
        snap = self._ensure_snapshot()
        return np.zeros(snap.action_dim, dtype=np.float32)

    def _structured_state(self, state) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        snap = self._ensure_snapshot()
        state_arr = np.array(state, dtype=np.float32).reshape(-1)
        offset = 0
        veh = np.zeros((0, 5), dtype=np.float32)
        rsu = np.zeros((0, 5), dtype=np.float32)
        uav = np.zeros((0, 5), dtype=np.float32)

        if snap.num_vehicles:
            length = snap.num_vehicles * 5
            veh = state_arr[offset : offset + length].reshape(snap.num_vehicles, 5)
            offset += length
        if snap.num_rsus:
            length = snap.num_rsus * 5
            rsu = state_arr[offset : offset + length].reshape(snap.num_rsus, 5)
            offset += length
        if snap.num_uavs:
            length = snap.num_uavs * 5
            uav = state_arr[offset : offset + length].reshape(snap.num_uavs, 5)
            offset += length
        return veh, rsu, uav

    def _action_from_preference(
        self,
        local_score: float,
        rsu_score: float,
        uav_score: float,
        rsu_index: Optional[int] = None,
        uav_index: Optional[int] = None,
    ) -> np.ndarray:
        snap = self._ensure_snapshot()
        action = np.zeros(snap.action_dim, dtype=np.float32)
        action[:3] = np.array([local_score, rsu_score, uav_score], dtype=np.float32)

        if rsu_index is not None and snap.num_rsus:
            rsu_slot = 3 + int(np.clip(rsu_index, 0, snap.num_rsus - 1))
            if rsu_slot < snap.action_dim:
                action[rsu_slot] = 3.5

        if uav_index is not None and snap.num_uavs:
            uav_slot = 3 + snap.num_rsus + int(np.clip(uav_index, 0, snap.num_uavs - 1))
            if uav_slot < snap.action_dim:
                action[uav_slot] = 3.5

        return action


class RandomPolicy(HeuristicPolicy):
    """Completely random exploratory baseline."""

    def __init__(self, seed: Optional[int] = None) -> None:
        super().__init__("Random", seed=seed)

    def select_action(self, state) -> np.ndarray:
        snap = self._ensure_snapshot()
        return self.rng.normal(loc=0.0, scale=1.0, size=snap.action_dim).astype(np.float32)


class LocalOnlyPolicy(HeuristicPolicy):
    """Always favour local processing.
    
    🎯 设计目标：提供纯本地处理基线，验证边缘卸载的必要性
    
    📊 对比价值：
    - 时延：高（受限于车载CPU）
    - 能耗：中等（本地计算功耗）
    - 完成率：低（高负载下易丢弃任务）
    
    ⚠️ 重构要点：
    - 移除enforce_offload_mode依赖，仅通过策略决策实现本地处理
    - 在高负载下也坚持本地处理，体现策略特性
    """

    def __init__(self) -> None:
        super().__init__("LocalOnly")
        self.local_preference = 5.0  # 强烈偏好本地

    def select_action(self, state) -> np.ndarray:
        # 🔧 重构：始终返回强本地偏好，不依赖外部强制模式
        return self._action_from_preference(
            local_score=self.local_preference, 
            rsu_score=-5.0,  # 强烈拒绝RSU
            uav_score=-5.0   # 强烈拒绝UAV
        )


class RSUOnlyPolicy(HeuristicPolicy):
    """Always prefer edge nodes (RSU/UAV), with intelligent load balancing.
    
    🎯 设计目标：提供纯边缘处理基线，验证本地计算的价值
    
    📊 对比价值：
    - 时延：中等（受通信时延影响）
    - 能耗：高（上行传输能耗）
    - 完成率：中等（RSU过载时下降）
    
    🔧 重构要点：
    - 同时考虑RSU和UAV（原实现忽略UAV）
    - 综合负载、距离、资源能力进行决策
    - 增加通信成本感知
    """

    def __init__(self) -> None:
        super().__init__("RSUOnly")
        self.edge_preference = 5.0
        self.distance_weight = 0.3  # 距离权重

    def select_action(self, state) -> np.ndarray:
        vehicles, rsus, uavs = self._structured_state(state)
        
        # 计算车辆质心位置
        veh_center = np.mean(vehicles[:, :2], axis=0) if vehicles.size > 0 else np.zeros(2)
        
        # 🔧 重构：评估所有边缘节点（RSU + UAV）
        candidates = []
        
        # 评估RSU
        if rsus.size > 0 and rsus.ndim == 2:
            for i in range(rsus.shape[0]):
                load = rsus[i, 3] if rsus.shape[1] > 3 else 0.5
                pos = rsus[i, :2] if rsus.shape[1] >= 2 else veh_center
                distance = np.linalg.norm(pos - veh_center)
                # 综合评分：负载 + 距离惩罚
                score = load + self.distance_weight * (distance / 1000.0)
                candidates.append(('rsu', i, score))
        
        # 评估UAV
        if uavs.size > 0 and uavs.ndim == 2:
            for i in range(uavs.shape[0]):
                load = uavs[i, 3] if uavs.shape[1] > 3 else 0.6
                pos = uavs[i, :2] if uavs.shape[1] >= 2 else veh_center
                distance = np.linalg.norm(pos - veh_center)
                # UAV距离惩罚稍高（空中通信衰减）
                score = load + (self.distance_weight * 1.2) * (distance / 800.0)
                candidates.append(('uav', i, score))
        
        # 选择最佳边缘节点
        if not candidates:
            # 无边缘节点可用，被迫本地处理
            return self._action_from_preference(
                local_score=0.0, 
                rsu_score=-5.0, 
                uav_score=-5.0
            )
        
        kind, idx, _ = min(candidates, key=lambda x: x[2])
        
        if kind == 'rsu':
            return self._action_from_preference(
                local_score=-self.edge_preference,
                rsu_score=self.edge_preference,
                uav_score=-3.0,
                rsu_index=idx
            )
        else:  # UAV
            return self._action_from_preference(
                local_score=-self.edge_preference,
                rsu_score=-3.0,
                uav_score=self.edge_preference,
                uav_index=idx
            )


class RoundRobinPolicy(HeuristicPolicy):
    """Cycle through local, RSU, UAV processing targets."""

    def __init__(self) -> None:
        super().__init__("RoundRobin")
        self._counter = 0

    def reset(self) -> None:
        self._counter = 0

    def select_action(self, state) -> np.ndarray:
        snap = self._ensure_snapshot()
        total_targets = 1 + snap.num_rsus + snap.num_uavs
        if total_targets <= 0:
            return self._blank_action()

        choice = self._counter % total_targets
        self._counter += 1

        if choice == 0 or snap.num_rsus + snap.num_uavs == 0:
            return self._action_from_preference(local_score=3.5, rsu_score=-2.0, uav_score=-2.0)

        choice -= 1
        if choice < snap.num_rsus:
            return self._action_from_preference(local_score=-2.0, rsu_score=3.5, uav_score=-2.0, rsu_index=choice)

        choice -= snap.num_rsus
        return self._action_from_preference(local_score=-2.0, rsu_score=-2.0, uav_score=3.5, uav_index=choice)


class SimulatedAnnealingPolicy(HeuristicPolicy):
    """Lightweight simulated annealing policy for global offloading preference."""

    def __init__(self, initial_temperature: float = 1.2, cooling_rate: float = 0.95, seed: Optional[int] = None) -> None:
        super().__init__("SimulatedAnnealing", seed=seed)
        self.initial_temperature = float(initial_temperature)
        self.cooling_rate = float(cooling_rate)
        self.temperature = self.initial_temperature
        self.current_choice: Tuple[str, Optional[int]] = ("local", None)

    def reset(self) -> None:
        self.temperature = self.initial_temperature
        self.current_choice = ("local", None)

    def _candidate_choices(self, snap: EnvironmentSnapshot) -> Dict[str, Optional[int]]:
        choices: Dict[str, Optional[int]] = {"local": None}
        for idx in range(snap.num_rsus):
            choices[f"rsu_{idx}"] = idx
        for idx in range(snap.num_uavs):
            choices[f"uav_{idx}"] = idx
        return choices

    def _evaluate_choice(self, choice: Tuple[str, Optional[int]], state) -> float:
        vehicles, rsus, uavs = self._structured_state(state)
        kind, index = choice

        # Use queue length + energy footprint as a proxy cost.
        if kind == "local":
            if vehicles.size == 0:
                return 1.0
            queues = vehicles[:, 3] if vehicles.shape[1] >= 4 else np.zeros(vehicles.shape[0])
            energy = vehicles[:, 4] if vehicles.shape[1] >= 5 else np.zeros(vehicles.shape[0])
            return float(np.mean(queues) * 1.5 + np.mean(energy) * 0.5)

        if kind.startswith("rsu"):
            if rsus.size == 0 or index is None or index >= rsus.shape[0]:
                return 2.0
            metrics = rsus[int(index)]
            queue = metrics[3] if metrics.size >= 4 else 0.5
            cache = metrics[2] if metrics.size >= 3 else 0.5
            energy = metrics[4] if metrics.size >= 5 else 0.5
            return float(queue * 1.2 + cache * 0.6 + energy * 0.4)

        if kind.startswith("uav"):
            if uavs.size == 0 or index is None or index >= uavs.shape[0]:
                return 3.0
            metrics = uavs[int(index)]
            queue = metrics[3] if metrics.size >= 4 else 0.6
            energy = metrics[4] if metrics.size >= 5 else 0.6
            altitude = metrics[2] if metrics.size >= 3 else 0.6
            return float(queue * 1.4 + energy * 0.5 + altitude * 0.3)

        return 4.0

    def _choice_to_action(self, choice: Tuple[str, Optional[int]]) -> np.ndarray:
        kind, index = choice
        if kind == "local":
            return self._action_from_preference(local_score=4.0, rsu_score=-2.0, uav_score=-2.0)
        if kind.startswith("rsu"):
            return self._action_from_preference(local_score=-1.5, rsu_score=4.0, uav_score=-1.5, rsu_index=index)
        return self._action_from_preference(local_score=-1.0, rsu_score=-1.0, uav_score=4.0, uav_index=index)

    def select_action(self, state) -> np.ndarray:
        snap = self._ensure_snapshot()
        choices = list(self._candidate_choices(snap).items())
        if not choices:
            return self._blank_action()

        current_cost = self._evaluate_choice(self.current_choice, state)

        # Propose neighbour
        key, idx = choices[self.rng.integers(0, len(choices))]
        proposal = ("local", None) if key == "local" else (key.split("_")[0], idx)
        proposal_cost = self._evaluate_choice(proposal, state)

        accept = False
        if proposal_cost < current_cost:
            accept = True
        else:
            # Boltzmann acceptance probability
            delta = proposal_cost - current_cost
            temperature = max(self.temperature, 1e-3)
            prob = math.exp(-delta / temperature)
            if self.rng.random() < prob:
                accept = True

        if accept:
            self.current_choice = proposal
            current_cost = proposal_cost

        self.temperature *= self.cooling_rate
        self.temperature = max(self.temperature, 0.05)

        return self._choice_to_action(self.current_choice)


class WeightedPreferencePolicy(HeuristicPolicy):
    """Weighted multi-objective heuristic that reacts to queue, cache and energy metrics."""

    def __init__(
        self,
        queue_weight: float = 1.6,
        energy_weight: float = 1.0,
        cache_weight: float = 0.6,
        jitter: float = 0.05,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__("WeightedPreference", seed=seed)
        self.queue_weight = float(queue_weight)
        self.energy_weight = float(energy_weight)
        self.cache_weight = float(cache_weight)
        self.jitter = max(float(jitter), 0.0)

    @staticmethod
    def _mean_col(arr: np.ndarray, idx: int, default: float) -> float:
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] <= idx:
            return default
        return float(np.mean(arr[:, idx]))

    @staticmethod
    def _value_col(arr: np.ndarray, row: int, idx: int, default: float) -> float:
        if arr.size == 0 or arr.ndim != 2 or arr.shape[1] <= idx:
            return default
        row = max(0, min(arr.shape[0] - 1, row))
        return float(arr[row, idx])

    def _score(self, kind: str, idx: Optional[int], veh: np.ndarray, rsu: np.ndarray, uav: np.ndarray) -> float:
        if kind == "local":
            queue = self._mean_col(veh, 3, 0.6)
            energy = self._mean_col(veh, 4, 0.6)
            cache = self._mean_col(veh, 2, 0.5)
        elif kind == "rsu":
            index = 0 if idx is None else int(idx)
            queue = self._value_col(rsu, index, 3, 0.7)
            energy = self._value_col(rsu, index, 4, 0.7)
            cache = self._value_col(rsu, index, 2, 0.4)
        else:  # "uav"
            index = 0 if idx is None else int(idx)
            queue = self._value_col(uav, index, 3, 0.75)
            energy = self._value_col(uav, index, 4, 0.8)
            cache = self._value_col(uav, index, 2, 0.35)

        penalty = self.queue_weight * queue + self.energy_weight * energy - self.cache_weight * cache
        if self.jitter > 0.0:
            penalty += float(self.rng.normal(0.0, self.jitter))
        return penalty

    def select_action(self, state) -> np.ndarray:
        veh, rsu, uav = self._structured_state(state)
        snap = self._ensure_snapshot()

        candidates: list[Tuple[str, Optional[int]]] = [("local", None)]
        if rsu.size:
            candidates.extend(("rsu", idx) for idx in range(min(snap.num_rsus, rsu.shape[0])))
        if uav.size:
            candidates.extend(("uav", idx) for idx in range(min(snap.num_uavs, uav.shape[0])))

        scores = [
            (self._score(kind, idx, veh, rsu, uav), kind, idx)
            for kind, idx in candidates
        ]
        _, best_kind, best_idx = min(scores, key=lambda item: item[0])

        if best_kind == "local":
            return self._action_from_preference(local_score=4.0, rsu_score=-1.5, uav_score=-1.5)
        if best_kind == "rsu":
            return self._action_from_preference(local_score=-1.0, rsu_score=4.0, uav_score=-1.0, rsu_index=best_idx)
        return self._action_from_preference(local_score=-0.5, rsu_score=-0.5, uav_score=4.0, uav_index=best_idx)


class GreedyPolicy(HeuristicPolicy):
    """Intelligent offloading policy with multi-factor awareness.
    
    🎯 设计目标：提供智能卸载基线，验证TD3学习的必要性
    
    📊 对比价值：
    - 时延：中等（考虑负载和通信）
    - 能耗：中等（动态平衡本地和卸载）
    - 完成率：中等（基于贪心决策）
    
    🔧 重构要点：
    - 综合考虑：队列负载、通信成本、计算能力
    - 支持RSU资源变化适应（通过状态感知）
    - 增加任务特性感知（通过能耗列）
    """

    def __init__(self) -> None:
        super().__init__("Greedy")
        # 多因素权重
        self.queue_weight = 1.5      # 队列负载权重
        self.comm_weight = 0.8       # 通信成本权重
        self.energy_weight = 0.6     # 能耗权重

    def select_action(self, state) -> np.ndarray:
        veh, rsu, uav = self._structured_state(state)
        
        # 计算车辆质心
        veh_center = np.mean(veh[:, :2], axis=0) if veh.size > 0 else np.zeros(2)
        
        candidates = []
        
        # 🔧 重构：评估本地处理（考虑队列和能耗）
        local_score = self._evaluate_local(veh)
        candidates.append(('local', None, local_score))
        
        # 🔧 重构：评估所有RSU（负载+距离+能耗）
        if rsu.size > 0 and rsu.ndim == 2:
            for i in range(rsu.shape[0]):
                score = self._evaluate_rsu(rsu[i], veh_center)
                candidates.append(('rsu', i, score))
        
        # 🔧 重构：评估所有UAV（负载+距离+悬停能耗）
        if uav.size > 0 and uav.ndim == 2:
            for i in range(uav.shape[0]):
                score = self._evaluate_uav(uav[i], veh_center)
                candidates.append(('uav', i, score))
        
        # 选择成本最低的方案
        kind, idx, _ = min(candidates, key=lambda x: x[2])
        
        if kind == 'local':
            return self._action_from_preference(
                local_score=4.0, 
                rsu_score=-2.0, 
                uav_score=-2.0
            )
        elif kind == 'rsu':
            return self._action_from_preference(
                local_score=-1.5, 
                rsu_score=4.0, 
                uav_score=-1.5, 
                rsu_index=idx
            )
        else:  # UAV
            return self._action_from_preference(
                local_score=-1.0, 
                rsu_score=-1.0, 
                uav_score=4.0, 
                uav_index=idx
            )
    
    def _evaluate_local(self, veh: np.ndarray) -> float:
        """评估本地处理成本：队列负载 + 能耗"""
        if veh.size == 0 or veh.ndim != 2:
            return 0.6
        
        # 队列负载（列3）
        queue = float(np.mean(veh[:, 3])) if veh.shape[1] > 3 else 0.5
        # 能耗状态（列4）
        energy = float(np.mean(veh[:, 4])) if veh.shape[1] > 4 else 0.5
        
        return float(self.queue_weight * queue + self.energy_weight * energy)
    
    def _evaluate_rsu(self, rsu_state: np.ndarray, veh_pos: np.ndarray) -> float:
        """评估RSU卸载成本：队列 + 通信距离 + 能耗"""
        # 队列负载
        queue = float(rsu_state[3]) if rsu_state.size > 3 else 0.6
        
        # 通信成本（基于距离）
        rsu_pos = rsu_state[:2] if rsu_state.size >= 2 else veh_pos
        distance = float(np.linalg.norm(rsu_pos - veh_pos))
        comm_cost = distance / 1000.0  # 归一化到[0, 1]范围
        
        # 能耗状态
        energy = float(rsu_state[4]) if rsu_state.size > 4 else 0.5
        
        return float(
            self.queue_weight * queue +
            self.comm_weight * comm_cost +
            self.energy_weight * energy * 0.5  # RSU能耗权重降低
        )
    
    def _evaluate_uav(self, uav_state: np.ndarray, veh_pos: np.ndarray) -> float:
        """评估UAV卸载成本：队列 + 通信距离 + 悬停能耗"""
        # 队列负载
        queue = float(uav_state[3]) if uav_state.size > 3 else 0.7
        
        # 通信成本（UAV距离衰减更快）
        uav_pos = uav_state[:2] if uav_state.size >= 2 else veh_pos
        distance = float(np.linalg.norm(uav_pos - veh_pos))
        comm_cost = distance / 800.0  # UAV通信范围较小
        
        # 悬停能耗
        energy = float(uav_state[4]) if uav_state.size > 4 else 0.7
        
        return float(
            self.queue_weight * queue +
            self.comm_weight * comm_cost * 1.2 +  # 空中通信惩罚
            self.energy_weight * energy * 0.8  # UAV能耗权重较高
        )


def create_baseline_algorithm(name: str, **kwargs):
    """Return a heuristic controller by name."""
    key = name.strip().lower()
    if key in {"random"}:
        return RandomPolicy(seed=kwargs.get("seed"))
    if key in {"greedy"}:
        return GreedyPolicy()
    if key in {"localonly", "local_only"}:
        return LocalOnlyPolicy()
    if key in {"rsuonly", "rsu_only"}:
        return RSUOnlyPolicy()
    if key in {"roundrobin", "round_robin"}:
        return RoundRobinPolicy()
    if key in {"simulatedannealing", "sim_anneal", "sa"}:
        return SimulatedAnnealingPolicy(
            initial_temperature=kwargs.get("initial_temperature", 1.2),
            cooling_rate=kwargs.get("cooling_rate", 0.95),
            seed=kwargs.get("seed"),
        )
    if key in {"weighted", "hybrid", "multiobjective"}:
        return WeightedPreferencePolicy(
            queue_weight=kwargs.get("queue_weight", 1.6),
            energy_weight=kwargs.get("energy_weight", 1.0),
            cache_weight=kwargs.get("cache_weight", 0.6),
            jitter=kwargs.get("jitter", 0.05),
            seed=kwargs.get("seed"),
        )
    raise ValueError(f"Unsupported baseline algorithm '{name}'.")


__all__ = [
    "create_baseline_algorithm",
    "RandomPolicy",
    "LocalOnlyPolicy",
    "RSUOnlyPolicy",
    "RoundRobinPolicy",
    "SimulatedAnnealingPolicy",
    "WeightedPreferencePolicy",
]
