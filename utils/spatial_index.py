#!/usr/bin/env python3
"""
空间索引工具
使用KD-tree优化最近节点查找性能
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence, Tuple, overload, Literal

import numpy as np

try:
    from scipy.spatial import cKDTree  # type: ignore
except Exception:  # pragma: no cover - SciPy may be unavailable in minimal envs
    cKDTree = None  # type: ignore


def _to_point(coord: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    将坐标标准化为长度为3的浮点向量，方便统一的距离计算。
    - 如果只有二维坐标，自动补齐 z=0
    - 如果超过3维，仅保留前三个分量
    """
    arr = np.asarray(coord, dtype=float).reshape(-1)
    if arr.size == 2:
        arr = np.append(arr, 0.0)
    elif arr.size == 1:
        arr = np.array([arr[0], 0.0, 0.0], dtype=float)
    elif arr.size == 0:
        arr = np.zeros(3, dtype=float)
    elif arr.size > 3:
        arr = arr[:3]
    return arr.astype(float, copy=False)


class SpatialIndex:
    """
    空间索引系统
    通过KD-tree加速距离查询，支持RSU/UAV/车辆的邻域检索。
    """

    def __init__(self) -> None:
        self._rsu_data: List[Dict] = []
        self._uav_data: List[Dict] = []
        self._vehicle_data: List[Dict] = []

        self._rsu_positions = np.empty((0, 3), dtype=float)
        self._uav_positions = np.empty((0, 3), dtype=float)
        self._vehicle_positions = np.empty((0, 3), dtype=float)

        self._rsu_tree: Optional[cKDTree] = None
        self._uav_tree: Optional[cKDTree] = None
        self._vehicle_tree: Optional[cKDTree] = None

        self._rsu_max_radius: float = 0.0
        self._uav_max_radius: float = 0.0

        self.query_count = 0
        self.total_query_time = 0.0

    # ------------------------------------------------------------------
    # Public update APIs
    # ------------------------------------------------------------------
    def update_static_nodes(self, rsus: Sequence[Dict], uavs: Sequence[Dict]) -> None:
        """更新静态节点（RSU、UAV）的空间索引。"""
        self._rsu_data = list(rsus) if rsus is not None else []
        self._uav_data = list(uavs) if uavs is not None else []

        self._rsu_positions = self._build_positions(self._rsu_data)
        self._uav_positions = self._build_positions(self._uav_data)

        self._rsu_tree = self._build_tree(self._rsu_positions)
        self._uav_tree = self._build_tree(self._uav_positions)

        self._rsu_max_radius = max(
            (float(rsu.get('coverage_radius', 0.0)) for rsu in self._rsu_data),
            default=0.0,
        )
        self._uav_max_radius = max(
            (float(uav.get('coverage_radius', 0.0)) for uav in self._uav_data),
            default=0.0,
        )

    def update_vehicle_nodes(self, vehicles: Sequence[Dict]) -> None:
        """更新车辆节点的空间索引（车辆为动态节点，需要高频刷新）。"""
        self._vehicle_data = list(vehicles) if vehicles is not None else []
        self._vehicle_positions = self._build_positions(self._vehicle_data)
        self._vehicle_tree = self._build_tree(self._vehicle_positions)

    def update_nodes(
        self,
        vehicles: Sequence[Dict],
        rsus: Optional[Sequence[Dict]] = None,
        uavs: Optional[Sequence[Dict]] = None,
    ) -> None:
        """
        统一更新接口：
        - 若提供 RSU/UAV 列表，则重建静态索引
        - 始终刷新车辆索引
        """
        if rsus is not None or uavs is not None:
            self.update_static_nodes(rsus or [], uavs or [])
        self.update_vehicle_nodes(vehicles)

    # ------------------------------------------------------------------
    # 查询接口
    # ------------------------------------------------------------------
    @overload
    def find_nearest_rsu(
        self,
        position: Sequence[float] | np.ndarray,
        return_distance: Literal[False] = False,
    ) -> Optional[Dict]:
        ...

    @overload
    def find_nearest_rsu(
        self,
        position: Sequence[float] | np.ndarray,
        *,
        return_distance: Literal[True],
    ) -> Optional[Tuple[int, Dict, float]]:
        ...

    def find_nearest_rsu(
        self,
        position: Sequence[float] | np.ndarray,
        return_distance: bool = False,
    ):
        start = time.perf_counter()
        result = self._find_nearest(position, self._rsu_positions, self._rsu_tree, self._rsu_data)
        self._record_query(start)
        if result is None:
            return None
        idx, node, dist = result
        return (idx, node, dist) if return_distance else node

    @overload
    def find_nearest_uav(
        self,
        position: Sequence[float] | np.ndarray,
        return_distance: Literal[False] = False,
    ) -> Optional[Dict]:
        ...

    @overload
    def find_nearest_uav(
        self,
        position: Sequence[float] | np.ndarray,
        *,
        return_distance: Literal[True],
    ) -> Optional[Tuple[int, Dict, float]]:
        ...

    def find_nearest_uav(
        self,
        position: Sequence[float] | np.ndarray,
        return_distance: bool = False,
    ):
        start = time.perf_counter()
        result = self._find_nearest(position, self._uav_positions, self._uav_tree, self._uav_data)
        self._record_query(start)
        if result is None:
            return None
        idx, node, dist = result
        return (idx, node, dist) if return_distance else node

    def query_rsus_within_radius(
        self,
        position: Sequence[float] | np.ndarray,
        radius: float,
    ) -> List[Tuple[int, Dict, float]]:
        start = time.perf_counter()
        result = self._query_within(position, radius, self._rsu_positions, self._rsu_tree, self._rsu_data)
        self._record_query(start)
        return result

    def query_uavs_within_radius(
        self,
        position: Sequence[float] | np.ndarray,
        radius: float,
    ) -> List[Tuple[int, Dict, float]]:
        start = time.perf_counter()
        result = self._query_within(position, radius, self._uav_positions, self._uav_tree, self._uav_data)
        self._record_query(start)
        return result

    def query_vehicles_within_radius(
        self,
        position: Sequence[float] | np.ndarray,
        radius: float,
    ) -> List[Tuple[int, Dict, float]]:
        start = time.perf_counter()
        result = self._query_within(position, radius, self._vehicle_positions, self._vehicle_tree, self._vehicle_data)
        self._record_query(start)
        return result

    # ------------------------------------------------------------------
    # 统计信息
    # ------------------------------------------------------------------
    def get_performance_stats(self) -> Dict[str, float]:
        if self.query_count == 0:
            return {
                'query_count': 0,
                'avg_query_time': 0.0,
                'total_query_time': 0.0,
                'rsu_count': float(len(self._rsu_data)),
                'uav_count': float(len(self._uav_data)),
                'vehicle_count': float(len(self._vehicle_data)),
            }
        return {
            'query_count': float(self.query_count),
            'avg_query_time': self.total_query_time / self.query_count,
            'total_query_time': self.total_query_time,
            'rsu_count': float(len(self._rsu_data)),
            'uav_count': float(len(self._uav_data)),
            'vehicle_count': float(len(self._vehicle_data)),
        }

    def reset_stats(self) -> None:
        self.query_count = 0
        self.total_query_time = 0.0

    @property
    def rsu_max_radius(self) -> float:
        return self._rsu_max_radius

    @property
    def uav_max_radius(self) -> float:
        return self._uav_max_radius

    # ------------------------------------------------------------------
    # 内部工具方法
    # ------------------------------------------------------------------
    def _build_positions(self, nodes: List[Dict]) -> np.ndarray:
        if not nodes:
            return np.empty((0, 3), dtype=float)
        return np.vstack([_to_point(node.get('position', (0.0, 0.0))) for node in nodes]).astype(float)

    def _build_tree(self, positions: np.ndarray) -> Optional[cKDTree]:
        if cKDTree is None or positions.size == 0:
            return None
        return cKDTree(positions)

    def _find_nearest(
        self,
        position: Sequence[float] | np.ndarray,
        positions: np.ndarray,
        tree: Optional[cKDTree],
        data: List[Dict],
    ) -> Optional[Tuple[int, Dict, float]]:
        if not data or positions.size == 0:
            return None

        point = _to_point(position)
        if tree is not None:
            dist, idx = tree.query(point, k=1)
            if isinstance(dist, np.ndarray):
                dist = float(dist[0])
                idx = int(idx[0])
            else:
                dist = float(dist)
                idx = int(idx)
            if np.isinf(dist):
                return None
        else:
            diffs = positions - point
            dist_sq = np.einsum('ij,ij->i', diffs, diffs)
            idx = int(np.argmin(dist_sq))
            dist = float(np.sqrt(dist_sq[idx]))

        return idx, data[idx], dist

    def _query_within(
        self,
        position: Sequence[float] | np.ndarray,
        radius: float,
        positions: np.ndarray,
        tree: Optional[cKDTree],
        data: List[Dict],
    ) -> List[Tuple[int, Dict, float]]:
        if not data or positions.size == 0 or radius <= 0.0:
            return []

        point = _to_point(position)
        radius = float(radius)
        results: List[Tuple[int, Dict, float]] = []

        if tree is not None:
            indices = tree.query_ball_point(point, radius)
            if not indices:
                return []
            subset = positions[np.asarray(indices, dtype=int)]
            diffs = subset - point
            dists = np.sqrt(np.einsum('ij,ij->i', diffs, diffs))
            for local_idx, dist in zip(indices, dists):
                results.append((int(local_idx), data[int(local_idx)], float(dist)))
            return results

        diffs = positions - point
        dist_sq = np.einsum('ij,ij->i', diffs, diffs)
        mask = dist_sq <= radius * radius
        if not np.any(mask):
            return []
        candidate_indices = np.where(mask)[0]
        dists = np.sqrt(dist_sq[candidate_indices])
        for idx, dist in zip(candidate_indices, dists):
            results.append((int(idx), data[int(idx)], float(dist)))
        return results

    def _record_query(self, start: float) -> None:
        elapsed = time.perf_counter() - start
        self.query_count += 1
        self.total_query_time += max(0.0, elapsed)
