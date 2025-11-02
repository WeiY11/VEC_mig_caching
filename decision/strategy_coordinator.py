"""联合策略协调器。

负责在缓存控制器与迁移控制器之间建立显式通信通道，
实现迁移-缓存参数共享、预取同步和指标驱动的自适应调节。
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Optional, Tuple
import copy

from utils.unified_time_manager import get_simulation_time


class StrategyCoordinator:
    """桥接缓存与迁移控制，让单智能体的决策具备联动语义。"""

    def __init__(
        self,
        cache_controller,
        migration_controller=None,
        history_size: int = 200,
    ) -> None:
        self.cache_controller = cache_controller
        self.migration_controller = migration_controller
        self._joint_params: Dict[str, float] = {
            "prefetch_lead_time": getattr(cache_controller, "joint_params", {}).get("prefetch_lead_time", 0.4),
            "migration_backoff": 0.2,
        }
        self.simulator = None
        self.migration_events: deque = deque(maxlen=history_size)
        self.prefetch_events: deque = deque(maxlen=history_size)

    # ------------------------------------------------------------------ #
    # Registration / parameter updates
    # ------------------------------------------------------------------ #

    def register_simulator(self, simulator) -> None:
        """关联底层仿真器，便于在迁移阶段直接操作缓存副本。"""
        self.simulator = simulator

    def update_joint_params(self, joint_params: Dict[str, float]) -> None:
        """更新由智能体输出的联合控制参数。"""
        if not isinstance(joint_params, dict):
            return
        updated = False
        for key in ("prefetch_lead_time", "migration_backoff"):
            if key in joint_params:
                try:
                    value = float(joint_params[key])
                except (TypeError, ValueError):
                    continue
                self._joint_params[key] = value
                updated = True

        if not updated:
            return

        if self.cache_controller is not None:
            self.cache_controller.apply_joint_params(self._joint_params)
        if self.migration_controller is not None:
            self.migration_controller.apply_joint_params(self._joint_params)

    # ------------------------------------------------------------------ #
    # Feedback processing
    # ------------------------------------------------------------------ #

    def observe_step(
        self,
        system_metrics: Optional[Dict],
        cache_metrics: Optional[Dict],
        migration_metrics: Optional[Dict],
        step_summary: Optional[Dict] = None,
    ) -> None:
        """根据仿真输出的指标动态调节阈值联动。"""
        if self.migration_controller is not None and cache_metrics:
            hit_rate = float(cache_metrics.get("hit_rate", 0.0) or 0.0)
            miss_rate = float(cache_metrics.get("miss_rate", 1.0) or 1.0)
            self.migration_controller.ingest_cache_feedback(hit_rate, miss_rate)

        if step_summary and self.cache_controller is not None:
            utilization = step_summary.get("rsu_cache_utilization")
            if utilization is not None:
                try:
                    self.cache_controller.cache_stats["current_utilization"] = float(utilization)
                except Exception:
                    pass

    # ------------------------------------------------------------------ #
    # Migration-cache handshake
    # ------------------------------------------------------------------ #

    def notify_migration_triggered(
        self,
        node_id: str,
        reason: str,
        urgency: float,
        node_snapshot: Optional[Dict] = None,
    ) -> None:
        """在迁移触发瞬间记录上下文，便于调试与后续分析。"""
        event = {
            "time": get_simulation_time(),
            "node": node_id,
            "reason": reason,
            "urgency": float(urgency),
            "snapshot": node_snapshot or {},
        }
        self.migration_events.append(event)

    def notify_migration_result(
        self,
        node_id: str,
        success: bool,
        metadata: Optional[Dict] = None,
    ) -> None:
        """记录迁移结果，供后续统计使用。"""
        event = {
            "time": get_simulation_time(),
            "node": node_id,
            "success": bool(success),
            "metadata": metadata or {},
        }
        self.migration_events.append(event)

    def prepare_prefetch(
        self,
        source_node: Dict,
        target_node: Dict,
        candidate_tasks: Iterable[Dict],
        urgency: float,
    ) -> None:
        """在迁移前将热点内容同步到目标节点缓存。"""
        if self.cache_controller is None:
            return
        if not isinstance(source_node, dict) or not isinstance(target_node, dict):
            return

        source_cache = source_node.get("cache") or {}
        target_cache = target_node.setdefault("cache", {})
        if not source_cache:
            return

        target_capacity = float(target_node.get("cache_capacity", 0.0) or 0.0)
        if target_capacity <= 0.0:
            return

        pending_ids = self._collect_candidate_contents(source_cache, candidate_tasks)
        if not pending_ids:
            return

        available = self._calculate_available_capacity(target_cache, target_capacity)
        if available <= 0.0:
            return

        prefetch_lead = max(0.0, float(self._joint_params.get("prefetch_lead_time", 0.4)))
        urgency_weight = min(1.0, max(0.2, urgency + 0.1))
        budget = available * min(1.0, urgency_weight)

        copied = 0
        consumed = 0.0
        for content_id in pending_ids:
            if content_id in target_cache:
                continue
            meta = source_cache.get(content_id)
            if not isinstance(meta, dict):
                continue
            size_mb = float(meta.get("size", 1.0) or 1.0)
            if consumed + size_mb > budget:
                break

            target_cache[content_id] = self._clone_cache_entry(meta, prefetch_lead)
            consumed += size_mb
            copied += 1

        if copied > 0:
            self.cache_controller.register_prefetch_event(copied)
            self.prefetch_events.append(
                {
                    "time": get_simulation_time(),
                    "source": source_node.get("id"),
                    "target": target_node.get("id"),
                    "count": copied,
                    "lead_time": prefetch_lead,
                }
            )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _collect_candidate_contents(
        source_cache: Dict,
        candidate_tasks: Iterable[Dict],
    ) -> List[str]:
        """根据待迁移任务收集需要同步的内容ID。"""
        ids: List[str] = []
        seen = set()
        for task in candidate_tasks or []:
            if not isinstance(task, dict):
                continue
            content_id = task.get("content_id") or task.get("input_content_id")
            if not content_id or content_id in seen:
                continue
            if content_id not in source_cache:
                continue
            ids.append(content_id)
            seen.add(content_id)
        return ids

    @staticmethod
    def _calculate_available_capacity(cache_snapshot: Dict, capacity_mb: float) -> float:
        used = 0.0
        for meta in cache_snapshot.values():
            try:
                used += float(meta.get("size", 0.0) or 0.0)
            except Exception:
                used += 0.0
        return max(0.0, capacity_mb - used)

    @staticmethod
    def _clone_cache_entry(meta: Dict, prefetch_lead: float) -> Dict:
        """复制缓存条目并标记预取信息。"""
        cloned = copy.deepcopy(meta)
        cloned["prefetched_at"] = get_simulation_time()
        cloned["prefetch_valid_for"] = prefetch_lead
        cloned["prefetch_origin"] = meta.get("origin_node")
        return cloned

    def get_joint_params(self) -> Dict[str, float]:
        """外部查询接口，返回当前联动参数。"""
        return dict(self._joint_params)
