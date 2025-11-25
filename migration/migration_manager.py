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
    urgency_score: float = 0.5  # 🆕 创新:迁移紧急度评分



class TaskMigrationManager:
    """High-level task migration manager."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # 瑙﹀彂闃堝€?
        self.rsu_overload_threshold = config.migration.rsu_overload_threshold
        self.uav_overload_threshold = config.migration.uav_overload_threshold
        self.uav_min_battery = config.migration.uav_min_battery
        
        # 🆕 创新:自适应阈值调整机制
        self.adaptive_threshold_enabled = True
        self.rsu_threshold_min = 0.70  # 最小阈值(激进迁移)
        self.rsu_threshold_max = 0.90  # 最大阈值(保守迁移)
        self.threshold_adjustment_rate = 0.02  # 每次调整幅度
        
        # 🆕 创新:性能反馈指标(用于阈值自适应)
        self.recent_migration_success_rate = 0.0
        self.recent_avg_delay_improvement = 0.0
        self.threshold_adjustment_interval = 50  # 每50次迁移调整一次
        self.migration_counter = 0
        
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
        """🚀 创新优化:智能迁移需求检测 + 自适应阈值调整"""
        migration_plans = []
        current_time = get_simulation_time()
        
        # 🆕 创新:定期调整阈值(基于性能反馈)
        self.migration_counter += 1
        if self.adaptive_threshold_enabled and self.migration_counter % self.threshold_adjustment_interval == 0:
            self._adjust_threshold_based_on_performance()
        
        for node_id, state in node_states.items():
            # 妫€鏌ュ喎鍗存湡
            if (node_id in self.node_last_migration and 
                current_time - self.node_last_migration[node_id] < self.cooldown_period):
                continue
            
            # 🆕 创新:综合评估迁移必要性(不仅看负载,还看队列趋势)
            if node_id.startswith("rsu_"):
                should_migrate, urgency_score = self._evaluate_rsu_migration_need(
                    node_id, state, node_states
                )
                # 🔧 修复：提高迁移触发阈值，减少频繁迁移
                if should_migrate and urgency_score > 1.2:
                    # 瀵绘壘杩佺Щ鐩爣
                    target_node = self._find_best_target(node_id, "rsu", node_states, node_positions)
                    if target_node:
                        plan = self._create_migration_plan(node_id, target_node, node_states, node_positions)
                        if plan:
                            # 🆕 创新:根据紧急度调整迁移优先级
                            plan.urgency_score = urgency_score
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
        
        # 🎯 P3优化：批量迁移优化
        migration_plans = self._batch_migrate_optimization(migration_plans)
        
        # 🆕 创新:按紧急度排序迁移计划
        migration_plans.sort(key=lambda p: getattr(p, 'urgency_score', 0.5), reverse=True)
        
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
            best_candidate = max(candidates,  # 🎯 P1-1: 使用max和评分函数
                               key=lambda x: self._score_target_node(x, source_node_id, source_pos, node_states, node_positions))
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
        
        # 安全地解析数据大小范围
        if isinstance(data_range, (list, tuple)) and len(data_range) >= 2:
            avg_data_size = (float(data_range[0]) + float(data_range[1])) / 2.0
        elif isinstance(data_range, (list, tuple)) and len(data_range) == 1:
            avg_data_size = float(data_range[0])
        elif isinstance(data_range, (int, float)):
            avg_data_size = float(data_range)
        else:
            # 默认值：1MB
            avg_data_size = 1e6
        data_size_bits = max(avg_data_size * 8.0, 1.0)
        migration_delay = max(0.01, data_size_bits / migration_bandwidth)
        latency_cost = migration_delay / max(1e-9, config.network.time_slot_duration)  # 延迟成本

        total_cost = (
            self.alpha_comp * computation_cost +
            self.alpha_tx * transmission_cost +
            self.alpha_lat * latency_cost
        )

        success_prob = self._calculate_success_probability(distance, node_states, source_node_id, target_node_id)  # 🔧 优化：多因素成功率

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

        # 🔧 优化：Keep-Before-Break阶段自适应划分
        prep_ratio, sync_ratio, down_ratio = self._adaptive_kbb_phases(migration_plan)
        preparation_time = migration_plan.migration_delay * prep_ratio
        sync_time = migration_plan.migration_delay * sync_ratio
        migration_plan.downtime = migration_plan.migration_delay * down_ratio

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
            # 🎯 P2-2优化：智能任务选择
            intelligent_tasks = self._select_tasks_for_intelligent_migration(scored_candidates, tasks_to_move)
            for task in intelligent_tasks:
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
        
        # 🔧 修复：迁移前同步缓存内容，避免数据丢失
        self._sync_cache_before_migration(source_node, target_node, moved_tasks)

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

    def _calculate_success_probability(self, distance: float, node_states: Dict,
                                      source_node_id: str, target_node_id: str) -> float:
        """
        🎯 优化：多因素迁移成功率计算
        
        考虑因素：
        1. 距离惩罚
        2. 源节点负载惩罚（过载时迁移更难）
        3. 目标节点容量奖励
        4. 网络拥塞惩罚
        """
        # 基础成功率
        base_prob = 0.9
        
        # 💡 距离惩罚
        distance_penalty = min(0.3, distance / 10000.0)
        
        # 💡 源节点负载惩罚（过载时迁移更难）
        source_state = node_states.get(source_node_id)
        source_penalty = 0.0
        if source_state and hasattr(source_state, 'load_factor'):
            if source_state.load_factor > 0.8:
                source_penalty = (source_state.load_factor - 0.8) * 0.5
        
        # 💡 目标节点容量奖励
        target_state = node_states.get(target_node_id)
        target_bonus = 0.0
        if target_state and hasattr(target_state, 'load_factor'):
            target_bonus = (1.0 - target_state.load_factor) * 0.1
        
        # 💡 网络拥塞惩罚
        network_penalty = 0.0
        if source_state and hasattr(source_state, 'bandwidth_utilization'):
            network_penalty = source_state.bandwidth_utilization * 0.1
        
        # 🎯 综合成功率
        success_prob = base_prob - distance_penalty - source_penalty + target_bonus - network_penalty
        return float(np.clip(success_prob, 0.4, 0.95))
    
    def _adaptive_kbb_phases(self, migration_plan: MigrationPlan) -> Tuple[float, float, float]:
        """
        🔧 优化：自适应Keep-Before-Break阶段划分
        
        根据迁移类型动态调整三个阶段的时间分配：
        - 准备阶段：资源预留、状态同步
        - 同步阶段：数据传输
        - 静默切换：downtime
        
        Returns:
            (prep_ratio, sync_ratio, downtime_ratio)
        """
        migration_type = migration_plan.migration_type
        
        if migration_type == MigrationType.RSU_TO_RSU:
            # RSU间有线迁移，准备时间短
            return (0.5, 0.4, 0.1)
        elif migration_type == MigrationType.RSU_TO_UAV:
            # 到UAV无线迁移，同步时间长
            return (0.6, 0.35, 0.05)
        elif migration_type == MigrationType.UAV_TO_RSU:
            # UAV到RSU，平衡配置
            return (0.55, 0.35, 0.1)
        else:
            # 默认配置（VEHICLE_FOLLOW, PREEMPTIVE等）
            return (0.7, 0.25, 0.05)

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

    # ========== 🎯 P1-P3 全面优化方法 ==========
    
    def _score_target_node(self, target_id: str, source_id: str, source_pos: Position,
                          node_states: Dict, node_positions: Dict[str, Position]) -> float:
        """
        🎯 P1-1: 多维度目标节点评分（轻量“注意力”融合）

        - 基于负载/距离/队列/带宽的旧评分保留
        - 增加“缓解收益”(source->target负载差) 与 “历史可靠性” 作为动态权重
        - 使用 softmax 计算轻量权重，突出最具收益的特征，兼顾简单性
        """
        target_state = node_states.get(target_id)
        source_state = node_states.get(source_id)
        if not target_state:
            return 0.0

        # 1. 负载评分：越空闲越好
        load_score = 1.0 - min(1.0, getattr(target_state, 'load_factor', 1.0))

        # 2. 距离评分：越近越好
        target_pos = node_positions.get(target_id)
        if target_pos:
            distance = source_pos.distance_to(target_pos)
            distance_score = 1.0 / (1.0 + distance / 1000.0)
        else:
            distance_score = 0.5

        # 3. 队列评分：队列越短越好
        queue_length = getattr(target_state, 'queue_length', 0)
        queue_capacity = 20.0 if target_id.startswith("rsu_") else 10.0
        queue_score = 1.0 - min(1.0, queue_length / queue_capacity)

        # 4. 带宽评分：带宽越空闲越好
        bandwidth_util = getattr(target_state, 'bandwidth_utilization', 0.5)
        bandwidth_score = 1.0 - min(1.0, bandwidth_util)

        # 5. 缓解收益：源节点与目标节点的负载差（越大越好）
        source_load = getattr(source_state, 'load_factor', 1.0) if source_state else 1.0
        relief_score = max(0.0, source_load - getattr(target_state, 'load_factor', 0.0))
        relief_score = min(1.0, relief_score)

        # 6. 历史可靠性：近期迁移成功率，避免频繁失败的路径
        success_rate = self.migration_stats['successful_migrations'] / max(1, self.migration_stats['total_attempts'])
        reliability_score = float(np.clip(success_rate + 0.05, 0.0, 1.0))  # 加一个轻微的先验

        # 旧版静态权重（保持兼容）
        legacy_score = 0.4 * load_score + 0.3 * distance_score + 0.2 * queue_score + 0.1 * bandwidth_score

        # 轻量注意力：让收益/可靠性自动“抬权重”
        attn_features = np.array([
            load_score,
            queue_score,
            distance_score,
            relief_score,
            reliability_score,
            bandwidth_score
        ], dtype=np.float32)
        attn_logits = attn_features * np.array([1.0, 1.0, 0.8, 1.5, 1.2, 0.6], dtype=np.float32)  # 偏向缓解收益与可靠性
        attn_weights = np.exp(attn_logits - np.max(attn_logits))
        attn_weights_sum = float(attn_weights.sum()) if np.isfinite(attn_weights.sum()) and attn_weights.sum() > 0 else 1.0
        attn_weights = attn_weights / attn_weights_sum
        attention_score = float(np.dot(attn_weights, attn_features))

        # 融合：保持旧逻辑的稳定性，同时让注意力突出高收益目标
        return 0.55 * attention_score + 0.45 * legacy_score
    
    def _sync_cache_before_migration(self, source_node: Dict, target_node: Dict, tasks: List[Task]) -> None:
        """
        🔧 修复：迁移前同步缓存内容，确保数据不丢失
        
        将待迁移任务相关的缓存内容预先复制到目标节点
        """
        source_cache = source_node.get('cache', {})
        if not source_cache or not tasks:
            return
        
        target_cache = target_node.setdefault('cache', {})
        target_capacity = target_node.get('cache_capacity', 1000.0)
        
        # 计算目标缓存可用空间
        target_used = sum(float(item.get('size', 0) or 0) for item in target_cache.values())
        target_available = max(0, target_capacity - target_used)
        
        # 收集需要同步的内容ID
        content_ids_to_sync = set()
        for task in tasks:
            if not isinstance(task, Task):
                continue
            content_id = getattr(task, 'content_id', None) or getattr(task, 'input_content_id', None)
            if content_id and content_id in source_cache:
                content_ids_to_sync.add(content_id)
        
        # 同步缓存内容
        synced_count = 0
        synced_size = 0.0
        for content_id in content_ids_to_sync:
            if content_id in target_cache:
                continue
            
            cache_item = source_cache.get(content_id)
            if not cache_item:
                continue
            
            item_size = float(cache_item.get('size', 1.0) or 1.0)
            if target_available < item_size:
                break
            
            # 复制缓存条目
            import copy
            target_cache[content_id] = copy.deepcopy(cache_item)
            target_cache[content_id]['migrated'] = True
            target_available -= item_size
            synced_size += item_size
            synced_count += 1
        
        if synced_count > 0:
            self.logger.debug(f"🔄 迁移前同步缓存: {synced_count}项, {synced_size:.1f}MB")
    
    def _select_tasks_for_intelligent_migration(self, source_queue, max_count: int) -> List[Task]:
        """
        🎯 P2-2: 智能任务选择 - 优先迁移高优先级+紧急任务
        """
        tasks_scored = []
        for task in source_queue:
            if not isinstance(task, Task):
                continue
            
            # 计算紧急度
            try:
                remaining_slots = int(task.remaining_lifetime_slots)
                urgency = 1.0 / max(1.0, remaining_slots)
            except:
                urgency = 0.5
            
            # 优先级权重（优先级1最高）
            priority = getattr(task, 'priority', 4)
            priority_weight = (5 - priority) / 4.0
            
            # 大小惩罚（大任务迁移成本高）
            data_size = getattr(task, 'data_size', 0)
            size_penalty = data_size / 1e6  # MB
            
            # 综合评分
            score = urgency * 0.5 + priority_weight * 0.3 - size_penalty * 0.2
            tasks_scored.append((task, score))
        
        # 按评分排序，选择top-K
        tasks_scored.sort(key=lambda x: x[1], reverse=True)
        return [task for task, _ in tasks_scored[:max_count]]
    
    def _batch_migrate_optimization(self, migration_plans: List[MigrationPlan]) -> List[MigrationPlan]:
        """
        🎯 P3: 批量迁移优化
        
        合并同源同目标的迁移计划，减少20%开销
        """
        from collections import defaultdict
        batches = defaultdict(list)
        
        # 按(source, target)分组
        for plan in migration_plans:
            key = (plan.source_node_id, plan.target_node_id)
            batches[key].append(plan)
        
        optimized_plans = []
        for (source, target), plans in batches.items():
            if len(plans) > 1:
                # 合并为批量迁移，减少20%开销
                merged_plan = plans[0]
                merged_plan.migration_delay *= 0.8
                merged_plan.migration_cost *= 0.8
                self.logger.info(f"🚀 批量迁移优化: {source}->{target} 合并{len(plans)}个计划")
                optimized_plans.append(merged_plan)
            else:
                optimized_plans.extend(plans)
        
        return optimized_plans
    
    def _calculate_precise_migration_cost(self, migration_plan: MigrationPlan, 
                                         task_list: List[Task],
                                         node_states: Dict) -> float:
        """
        🎯 P3: 精确迁移成本计算
        
        考虑：传输成本 + 计算成本 + 网络拥塞成本
        """
        # 1. 实际传输成本
        total_data_size = sum(getattr(t, 'data_size', 0) for t in task_list)
        data_size_bits = total_data_size * 8
        migration_bw = max(1e-9, getattr(config.migration, 'migration_bandwidth', 1e6))
        transmission_time = data_size_bits / migration_bw
        transmission_cost = transmission_time * self.alpha_tx
        
        # 2. 实际计算成本（状态同步、上下文切换）
        num_tasks = len(task_list)
        computation_cost = num_tasks * 0.05 * self.alpha_comp
        
        # 3. 网络拥塞成本
        source_state = node_states.get(migration_plan.source_node_id)
        if source_state and hasattr(source_state, 'bandwidth_utilization'):
            source_bw_util = source_state.bandwidth_utilization
            latency_penalty = transmission_time * (1 + source_bw_util) * self.alpha_lat
        else:
            latency_penalty = transmission_time * self.alpha_lat
        
        total_cost = transmission_cost + computation_cost + latency_penalty
        return total_cost

    def _evaluate_rsu_migration_need(self, node_id: str, state, node_states: Dict) -> Tuple[bool, float]:
        """
        🆕 创新:综合评估RSU迁移必要性
        
        基于多个因素判断:
        1. 负载因子(当前负载 vs 阈值)
        2. 负载趋势(是否持续上升)
        3. 队列长度增长速度
        
        Returns:
            (should_migrate, urgency_score): 是否迁移和紧急度评分[0,1]
        """
        load_factor = state.load_factor
        
        # 1. 基础判断:负载是否超阈值
        if load_factor <= self.rsu_overload_threshold:
            return False, 0.0
        
        # 2. 计算超载程度
        overload_ratio = (load_factor - self.rsu_overload_threshold) / max(0.1, 1.0 - self.rsu_overload_threshold)
        urgency_score = min(1.0, overload_ratio)
        
        # 3. 负载趋势判断(如果有历史数据)
        # 简化版:基于队列长度估计趋势
        queue_length = getattr(state, 'queue_length', 0)
        if queue_length > 15:  # 队列过长,增加紧急度
            urgency_score *= 1.2
        
        urgency_score = min(1.0, urgency_score)
        return True, urgency_score
    
    def _adjust_threshold_based_on_performance(self) -> None:
        """
        🆕 创新:基于性能反馈调整迁移阈值
        
        策略:
        - 如果迁移成功率高且效果好 -> 降低阈值(更激进)
        - 如果迁移成功率低或效果差 -> 提高阈值(更保守)
        """
        success_rate = self.migration_stats['successful_migrations'] / max(1, self.migration_stats['total_attempts'])
        self.recent_migration_success_rate = success_rate
        
        # 成功率高,且迁移有效 -> 降低阈值
        if success_rate > 0.85:
            self.rsu_overload_threshold = max(
                self.rsu_threshold_min,
                self.rsu_overload_threshold - self.threshold_adjustment_rate
            )
            self.logger.info(f"🔧 调整迁移阈值: {self.rsu_overload_threshold:.3f} (更激进,成功率={success_rate:.2%})")
        # 成功率低 -> 提高阈值
        elif success_rate < 0.65:
            self.rsu_overload_threshold = min(
                self.rsu_threshold_max,
                self.rsu_overload_threshold + self.threshold_adjustment_rate
            )
            self.logger.info(f"🔧 调整迁移阈值: {self.rsu_overload_threshold:.3f} (更保守,成功率={success_rate:.2%})")


