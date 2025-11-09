"""Task migration manager module.

Provides utilities for planning and executing task migrations."""
import logging
import numpy as np
import uuid
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

from models.data_structures import Task, Position
from config import config
from utils.unified_time_manager import get_simulation_time


class MigrationType(Enum):
    """Migration type enumeration."""
    RSU_TO_RSU = "rsu_to_rsu"
    RSU_TO_UAV = "rsu_to_uav"
    UAV_TO_RSU = "uav_to_rsu"
    VEHICLE_FOLLOW = "vehicle_follow"
    PREEMPTIVE = "preemptive"


@dataclass
class MigrationPlan:
    """Migration plan data structure."""
    migration_id: str
    migration_type: MigrationType
    source_node_id: str
    target_node_id: str
    migration_cost: float = 0.0
    migration_delay: float = 0.0
    success_probability: float = 0.0
    is_completed: bool = False
    downtime: float = 0.001  # Keep-Before-Break downtime (seconds)
    tasks_moved: int = 0



class TaskMigrationManager:
    """High-level task migration manager."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 瑙﹀彂闃堝€?
        self.rsu_overload_threshold = config.migration.rsu_overload_threshold
        self.uav_overload_threshold = config.migration.uav_overload_threshold
        self.uav_min_battery = config.migration.uav_min_battery
        
        # 鎴愭湰鍙傛暟
        self.alpha_comp = config.migration.migration_alpha_comp
        self.alpha_tx = config.migration.migration_alpha_tx
        self.alpha_lat = config.migration.migration_alpha_lat
        
        # 缁熻淇℃伅
        self.migration_stats = {
            'total_attempts': 0,
            'successful_migrations': 0,
            'total_downtime': 0.0,
            'avg_cost': 0.0,
            'total_tasks_migrated': 0
        }
        
        # 鍐峰嵈绠＄悊
        self.node_last_migration: Dict[str, float] = {}
        self.cooldown_period = config.migration.cooldown_period
        # Retry/backoff configuration
        self.retry_backoff_base = float(getattr(config.migration, 'retry_backoff_base', 0.5))
        self.retry_backoff_max = float(getattr(config.migration, 'retry_backoff_max', 6.0))
        self.max_retry_attempts = int(getattr(config.migration, 'max_retry_attempts', 3))
        self.retry_queue: Dict[str, Dict[str, Any]] = {}
    
    def check_migration_needs(self, node_states: Dict, node_positions: Dict[str, Position]) -> List[MigrationPlan]:
        """Check nodes and create migration plans."""
        migration_plans = []
        current_time = get_simulation_time()
        
        for node_id, state in node_states.items():
            # 妫€鏌ュ喎鍗存湡
            if (node_id in self.node_last_migration and 
                current_time - self.node_last_migration[node_id] < self.cooldown_period):
                continue
            
            if node_id.startswith("rsu_") and state.load_factor > self.rsu_overload_threshold:
                # RSU杩囪浇锛屽鎵捐縼绉荤洰鏍?
                target_node = self._find_best_target(node_id, "rsu", node_states, node_positions)
                if target_node:
                    plan = self._create_migration_plan(node_id, target_node, node_states, node_positions)
                    if plan:
                        migration_plans.append(plan)
            
            elif node_id.startswith("uav_"):
                battery_level = getattr(state, 'battery_level', 1.0)
                if (battery_level < self.uav_min_battery or 
                    state.load_factor > self.uav_overload_threshold):
                    # UAV闇€瑕佽縼绉?
                    target_node = self._find_best_target(node_id, "uav", node_states, node_positions)
                    if target_node:
                        plan = self._create_migration_plan(node_id, target_node, node_states, node_positions)
                        if plan:
                            migration_plans.append(plan)
        
        migration_plans.extend(
            self._collect_retry_plans(current_time, node_states, node_positions)
        )
        
        return migration_plans
    
    def _find_best_target(self, source_node_id: str, source_type: str, 
                         node_states: Dict, node_positions: Dict[str, Position]) -> Optional[str]:
        """Find the best migration target for a source node."""
        candidates = []
        
        if source_type == "rsu":
            # 馃敡 淇锛氭斁瀹借縼绉荤洰鏍囬€夋嫨鏉′欢锛屽鍔犺縼绉绘満浼?
            for node_id, state in node_states.items():
                if node_id.startswith("rsu_") and node_id != source_node_id:
                    if state.load_factor < self.rsu_overload_threshold * 0.9:  # 浠?.8鎻愰珮鍒?.9
                        candidates.append(node_id)
                elif node_id.startswith("uav_"):
                    battery_level = getattr(state, 'battery_level', 1.0)
                    if (battery_level > self.uav_min_battery * 1.2 and   # 浠?.5闄嶈嚦1.2
                        state.load_factor < self.uav_overload_threshold * 0.9):  # 浠?.8鎻愰珮鍒?.9
                        candidates.append(node_id)
        
        elif source_type == "uav":
            # 馃敡 淇锛歎AV杩佺Щ鏉′欢涔熼€傚害鏀惧
            for node_id, state in node_states.items():
                if node_id.startswith("rsu_"):
                    if state.load_factor < self.rsu_overload_threshold * 0.9:  # 浠?.8鎻愰珮鍒?.9
                        candidates.append(node_id)
        
        # 閫夋嫨璺濈鏈€杩戠殑鍊欓€?
        if candidates and source_node_id in node_positions:
            source_pos = node_positions[source_node_id]
            best_candidate = min(candidates, 
                               key=lambda x: source_pos.distance_to(node_positions.get(x, source_pos)))
            return best_candidate
        
        return None

    def _collect_retry_plans(self, current_time: float,
                             node_states: Dict,
                             node_positions: Dict[str, Position]) -> List[MigrationPlan]:
        """Generate migration plans for entries waiting in the retry queue."""
        ready_plans: List[MigrationPlan] = []
        pending_keys = list(self.retry_queue.keys())
        for source_id in pending_keys:
            entry = self.retry_queue.get(source_id, {})
            if not entry:
                continue
            if current_time < entry.get('next_retry_time', 0.0):
                continue
            target_id = entry.get('target_node_id')
            plan = None
            if target_id:
                plan = self._create_migration_plan(source_id, target_id, node_states, node_positions)
            if plan is None:
                target_id = self._find_best_target(source_id, entry.get('source_type', ''), node_states, node_positions)
                if target_id:
                    plan = self._create_migration_plan(source_id, target_id, node_states, node_positions)
            if plan:
                ready_plans.append(plan)
                self.retry_queue.pop(source_id, None)
            else:
                # Could not create plan now; push next retry window
                entry_attempts = entry.get('attempts', 1)
                backoff = min(self.retry_backoff_max, self.retry_backoff_base * (2 ** max(0, entry_attempts - 1)))
                entry['next_retry_time'] = current_time + backoff
                self.retry_queue[source_id] = entry
        return ready_plans
    
    def _create_migration_plan(self, source_node_id: str, target_node_id: str,
                             node_states: Dict, node_positions: Dict[str, Position]) -> Optional[MigrationPlan]:
        """Create a migration plan."""
        distance = 0.0
        if source_node_id in node_positions and target_node_id in node_positions:
            distance = node_positions[source_node_id].distance_to(node_positions[target_node_id])

        transmission_cost = distance / 1000.0  # 传输成本近似按公里计算
        computation_cost = 1.0  # 固定计算成本占位

        migration_bandwidth = max(1e-9, getattr(config.migration, 'migration_bandwidth', 1e6))
        data_range = getattr(config.task, 'task_data_size_range', getattr(config.task, 'data_size_range', (1.0, 1.0)))
        if isinstance(data_range, (list, tuple)) and len(data_range) >= 2:
            avg_data_size = (float(data_range[0]) + float(data_range[1])) / 2.0
        elif isinstance(data_range, (list, tuple)):
            avg_data_size = float(data_range[0])
        else:
            avg_data_size = float(data_range)
        data_size_bits = max(avg_data_size * 8.0, 1.0)
        migration_delay = max(0.01, data_size_bits / migration_bandwidth)
        latency_cost = migration_delay / max(1e-9, config.network.time_slot_duration)  # 延迟成本

        total_cost = (
            self.alpha_comp * computation_cost +
            self.alpha_tx * transmission_cost +
            self.alpha_lat * latency_cost
        )

        success_prob = max(0.5, 0.9 - distance / 10000.0)  # 距离越远成功率越低

        if source_node_id.startswith("rsu_") and target_node_id.startswith("rsu_"):
            migration_type = MigrationType.RSU_TO_RSU
        elif source_node_id.startswith("rsu_") and target_node_id.startswith("uav_"):
            migration_type = MigrationType.RSU_TO_UAV
        elif source_node_id.startswith("uav_") and target_node_id.startswith("rsu_"):
            migration_type = MigrationType.UAV_TO_RSU
        else:
            migration_type = MigrationType.PREEMPTIVE

        return MigrationPlan(
            migration_id=str(uuid.uuid4()),
            migration_type=migration_type,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            migration_cost=total_cost,
            migration_delay=migration_delay,
            success_probability=success_prob
        )

    def execute_migration(self, migration_plan: MigrationPlan,
                          node_states: Optional[Dict] = None,
                          system_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
                          tasks_to_move: Optional[int] = None) -> bool:
        """Execute a Keep-Before-Break migration cycle and return whether it succeeded."""
        self.migration_stats['total_attempts'] += 1
        migration_plan.tasks_moved = 0

        # 模拟Keep-Before-Break过程阶段划分
        preparation_time = migration_plan.migration_delay * 0.7
        sync_time = migration_plan.migration_delay * 0.25
        migration_plan.downtime = migration_plan.migration_delay * 0.05

        success = np.random.random() < migration_plan.success_probability

        if success:
            self.migration_stats['successful_migrations'] += 1
            self.migration_stats['total_downtime'] += migration_plan.downtime
            migration_plan.is_completed = True

            # 更新冷却时间
            self.node_last_migration[migration_plan.source_node_id] = get_simulation_time()

            # Update average migration cost
            self._update_avg_cost(migration_plan.migration_cost)

            # 应用迁移对节点的实际影响
            self._apply_migration_effects(
                migration_plan, node_states=node_states, system_nodes=system_nodes, tasks_to_move=tasks_to_move
            )
            if migration_plan.tasks_moved:
                self.migration_stats['total_tasks_migrated'] = (
                    self.migration_stats.get('total_tasks_migrated', 0) + migration_plan.tasks_moved
                )
        else:
            self._schedule_retry(migration_plan)

        return success

    def _schedule_retry(self, migration_plan: MigrationPlan) -> None:
        """Schedule a retry with exponential backoff for a failed migration."""
        source_id = migration_plan.source_node_id
        source_type = ''
        if source_id.startswith('rsu_'):
            source_type = 'rsu'
        elif source_id.startswith('uav_'):
            source_type = 'uav'
        entry = self.retry_queue.get(source_id, {
            'attempts': 0,
            'source_type': source_type
        })
        attempts = entry.get('attempts', 0) + 1
        if attempts > self.max_retry_attempts:
            self.retry_queue.pop(source_id, None)
            self.logger.debug(
                "Dropping migration retries for %s after %d attempts",
                source_id, attempts
            )
            return
        backoff = min(self.retry_backoff_max, self.retry_backoff_base * (2 ** (attempts - 1)))
        next_retry = get_simulation_time() + backoff
        self.retry_queue[source_id] = {
            'attempts': attempts,
            'source_type': entry.get('source_type'),
            'target_node_id': migration_plan.target_node_id,
            'next_retry_time': next_retry
        }


    def _apply_migration_effects(self, migration_plan: MigrationPlan,
                              node_states: Optional[Dict] = None,
                              system_nodes: Optional[Dict[str, Dict[str, Any]]] = None,
                              tasks_to_move: Optional[int] = None) -> None:
        """Update source and target nodes after a successful migration."""
        if system_nodes is None:
            return

        source_node = system_nodes.get(migration_plan.source_node_id)
        target_node = system_nodes.get(migration_plan.target_node_id)
        if not source_node or not target_node:
            return

        source_queue = None
        for key in ('computation_queue', 'task_queue', 'tasks'):
            candidate = source_node.get(key)
            if candidate is not None:
                source_queue = candidate
                break

        target_queue = None
        target_queue_key = None
        for key in ('computation_queue', 'task_queue', 'tasks'):
            candidate = target_node.get(key)
            if candidate is not None:
                target_queue = candidate
                target_queue_key = key
                break

        if source_queue is None or len(source_queue) == 0:
            return

        if target_queue is None:
            target_queue = []
            target_queue_key = target_queue_key or 'computation_queue'
            target_node[target_queue_key] = target_queue

        if tasks_to_move is None:
            tasks_to_move = max(1, int(len(source_queue) * 0.2))
        tasks_to_move = max(1, min(tasks_to_move, len(source_queue)))

        moved_tasks: List[Task] = []
        scored_candidates = [t for t in list(source_queue) if isinstance(t, Task)]
        if scored_candidates:
            scored_candidates.sort(key=self._score_task_for_migration)
            desired = scored_candidates[:tasks_to_move]
            for task in desired:
                if self._detach_task_from_queue(source_queue, task):
                    moved_tasks.append(task)
        # Fall back to FIFO if we still need to move tasks
        while len(moved_tasks) < tasks_to_move and source_queue:
            if hasattr(source_queue, 'popleft'):
                try:
                    moved_tasks.append(source_queue.popleft())
                    continue
                except IndexError:
                    break
            try:
                moved_tasks.append(source_queue.pop(0))
            except (IndexError, AttributeError):
                break

        if not moved_tasks:
            return

        if hasattr(target_queue, 'extend'):
            target_queue.extend(moved_tasks)
        else:
            for task in moved_tasks:
                if hasattr(target_queue, 'append'):
                    target_queue.append(task)

        migration_plan.tasks_moved = len(moved_tasks)

        if node_states is not None:
            source_state = node_states.get(migration_plan.source_node_id)
            target_state = node_states.get(migration_plan.target_node_id)
            self._update_node_state_metrics(source_state, len(source_queue))
            self._update_node_state_metrics(target_state, len(target_queue))

    def _update_node_state_metrics(self, node_state, queue_length: int) -> None:
        """Refresh queue length and load factor for a node state."""
        if node_state is None:
            return

        try:
            node_state.queue_length = max(0, int(queue_length))
        except Exception:
            setattr(node_state, 'queue_length', max(0, int(queue_length)))

        capacity = self._get_nominal_capacity(node_state)
        if capacity <= 0:
            return

        load_factor = queue_length / capacity
        try:
            node_state.load_factor = float(load_factor)
        except Exception:
            setattr(node_state, 'load_factor', float(load_factor))

    def _get_nominal_capacity(self, node_state) -> float:
        label = getattr(node_state, 'node_type', None)
        if isinstance(label, str):
            label_value = label.lower()
        elif label is not None and hasattr(label, 'value'):
            label_value = str(label.value).lower()
        else:
            label_value = str(label).lower() if label is not None else ''

        queue_cfg = getattr(config, 'queue', None)
        if 'rsu' in label_value:
            return float(getattr(queue_cfg, 'rsu_nominal_capacity', 20.0)) if queue_cfg else 20.0
        if 'uav' in label_value:
            return float(getattr(queue_cfg, 'uav_nominal_capacity', 10.0)) if queue_cfg else 10.0
        if 'vehicle' in label_value:
            return 5.0
        return 10.0

    def _update_avg_cost(self, new_cost: float):
        """Update average migration cost."""
        current_avg = self.migration_stats['avg_cost']
        success_count = self.migration_stats['successful_migrations']
        
        if success_count == 1:
            self.migration_stats['avg_cost'] = new_cost
        else:
            # 绉诲姩骞冲潎
            alpha = 0.1
            self.migration_stats['avg_cost'] = alpha * new_cost + (1 - alpha) * current_avg
    
    def get_migration_statistics(self) -> Dict:
        """Return migration statistics."""
        total_attempts = self.migration_stats['total_attempts']
        successful = self.migration_stats['successful_migrations']
        
        return {
            'total_attempts': total_attempts,
            'successful_migrations': successful,
            'success_rate': successful / max(1, total_attempts),
            'total_downtime': self.migration_stats['total_downtime'],
            'avg_downtime_per_migration': self.migration_stats['total_downtime'] / max(1, successful),
            'avg_cost': self.migration_stats['avg_cost'],
            'total_tasks_migrated': self.migration_stats.get('total_tasks_migrated', 0)
        }
    
    def step(self, node_states: Dict, node_positions: Dict[str, Position],
            system_nodes: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict:
        """Run one migration-planning step and return aggregated statistics."""
        migration_plans = self.check_migration_needs(node_states, node_positions)

        step_stats = {
            'migrations_planned': len(migration_plans),
            'migrations_executed': 0,
            'migrations_successful': 0,
            'tasks_migrated': 0
        }

        for plan in migration_plans:
            step_stats['migrations_executed'] += 1
            success = self.execute_migration(
                plan,
                node_states=node_states,
                system_nodes=system_nodes
            )
            if success:
                step_stats['migrations_successful'] += 1
                step_stats['tasks_migrated'] += plan.tasks_moved

        return step_stats

    def _score_task_for_migration(self, task: Task) -> Tuple[int, int]:
        """Lower score means higher priority for migration."""
        priority = getattr(task, 'priority', getattr(config.task, 'num_priority_levels', 4))
        try:
            remaining = int(task.remaining_lifetime_slots)
        except Exception:
            remaining = getattr(task, 'max_delay_slots', 0)
        return (priority, remaining)

    def _detach_task_from_queue(self, queue, task: Task) -> bool:
        """Remove a specific task object from the given queue-like container."""
        if queue is None or task is None:
            return False
        if hasattr(queue, 'remove_task'):
            try:
                queue.remove_task(task.task_id)
                return True
            except Exception:
                pass
        if hasattr(queue, 'remove'):
            try:
                queue.remove(task)
                return True
            except ValueError:
                return False
        # Manual scan fallback
        try:
            for idx, item in enumerate(queue):
                if item is task:
                    del queue[idx]
                    return True
        except Exception:
            return False
        return False


