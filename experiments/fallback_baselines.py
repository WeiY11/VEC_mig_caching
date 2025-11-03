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
    """Always favour local processing."""

    def __init__(self) -> None:
        super().__init__("LocalOnly")

    def select_action(self, state) -> np.ndarray:
        return self._action_from_preference(local_score=4.0, rsu_score=-4.0, uav_score=-4.0)


class RSUOnlyPolicy(HeuristicPolicy):
    """Always prefer the least-loaded RSU when available."""

    def __init__(self) -> None:
        super().__init__("RSUOnly")

    def select_action(self, state) -> np.ndarray:
        vehicles, rsus, _ = self._structured_state(state)
        if rsus.size == 0:
            return self._action_from_preference(local_score=3.0, rsu_score=-3.0, uav_score=-3.0)

        loads = rsus[:, 3] if rsus.ndim == 2 and rsus.shape[1] >= 4 else np.full(rsus.shape[0], 0.5, dtype=np.float32)
        target = int(np.argmin(loads))
        pref = self._action_from_preference(local_score=-3.0, rsu_score=4.0, uav_score=-3.0, rsu_index=target)

        # If vehicles are extremely congested, slightly increase RSU preference.
        if vehicles.size and float(np.mean(vehicles[:, 3])) > 0.6:
            pref[1] += 1.0
        return pref


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
    """Greedy baseline: pick the least-loaded compute target.

    Uses the 4th column (index 3) of node feature vectors as a proxy for queue/load.
    Falls back to 0.5 when unavailable. Chooses among local aggregate, per-RSU, per-UAV.
    """

    def __init__(self) -> None:
        super().__init__("Greedy")

    def select_action(self, state) -> np.ndarray:
        veh, rsu, uav = self._structured_state(state)

        def _mean_col(arr, idx: int, default: float) -> float:
            if arr.size == 0 or arr.ndim != 2 or arr.shape[1] <= idx:
                return default
            return float(np.mean(arr[:, idx]))

        def _argmin_col(arr, idx: int) -> int:
            if arr.size == 0 or arr.ndim != 2 or arr.shape[1] <= idx:
                return -1
            return int(np.argmin(arr[:, idx]))

        local_load = _mean_col(veh, 3, 0.5)
        best_rsu_idx = _argmin_col(rsu, 3)
        rsu_load = float(rsu[best_rsu_idx, 3]) if best_rsu_idx >= 0 else 0.7
        best_uav_idx = _argmin_col(uav, 3)
        uav_load = float(uav[best_uav_idx, 3]) if best_uav_idx >= 0 else 0.8

        # Select the minimum-load family
        loads = [("local", local_load), ("rsu", rsu_load), ("uav", uav_load)]
        kind, _ = min(loads, key=lambda kv: kv[1])

        if kind == "rsu" and best_rsu_idx >= 0:
            return self._action_from_preference(local_score=-1.5, rsu_score=4.0, uav_score=-1.5, rsu_index=best_rsu_idx)
        if kind == "uav" and best_uav_idx >= 0:
            return self._action_from_preference(local_score=-1.0, rsu_score=-1.0, uav_score=4.0, uav_index=best_uav_idx)
        return self._action_from_preference(local_score=4.0, rsu_score=-2.0, uav_score=-2.0)


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
