"""
任务迁移管理器 - 对应论文第6节
实现Keep-Before-Break任务迁移机制和低中断切换
"""
import numpy as np
import time
import uuid
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from models.data_structures import Task, Position
from config import config


class MigrationType(Enum):
    """迁移类型枚举"""
    RSU_TO_RSU = "rsu_to_rsu"
    RSU_TO_UAV = "rsu_to_uav"
    UAV_TO_RSU = "uav_to_rsu"
    VEHICLE_FOLLOW = "vehicle_follow"
    PREEMPTIVE = "preemptive"


@dataclass
class MigrationPlan:
    """迁移计划数据结构"""
    migration_id: str
    migration_type: MigrationType
    source_node_id: str
    target_node_id: str
    migration_cost: float = 0.0
    migration_delay: float = 0.0
    success_probability: float = 0.0
    is_completed: bool = False
    downtime: float = 0.001  # Keep-Before-Break的中断时间


class TaskMigrationManager:
    """
    任务迁移管理器 - 整合迁移功能
    """
    
    def __init__(self):
        # 触发阈值
        self.rsu_overload_threshold = config.migration.rsu_overload_threshold
        self.uav_overload_threshold = config.migration.uav_overload_threshold
        self.uav_min_battery = config.migration.uav_min_battery
        
        # 成本参数
        self.alpha_comp = config.migration.migration_alpha_comp
        self.alpha_tx = config.migration.migration_alpha_tx
        self.alpha_lat = config.migration.migration_alpha_lat
        
        # 统计信息
        self.migration_stats = {
            'total_attempts': 0,
            'successful_migrations': 0,
            'total_downtime': 0.0,
            'avg_cost': 0.0
        }
        
        # 冷却管理
        self.node_last_migration: Dict[str, float] = {}
        self.cooldown_period = config.migration.cooldown_period
    
    def check_migration_needs(self, node_states: Dict, node_positions: Dict[str, Position]) -> List[MigrationPlan]:
        """检查并创建迁移计划"""
        migration_plans = []
        current_time = time.time()
        
        for node_id, state in node_states.items():
            # 检查冷却期
            if (node_id in self.node_last_migration and 
                current_time - self.node_last_migration[node_id] < self.cooldown_period):
                continue
            
            if node_id.startswith("rsu_") and state.load_factor > self.rsu_overload_threshold:
                # RSU过载，寻找迁移目标
                target_node = self._find_best_target(node_id, "rsu", node_states, node_positions)
                if target_node:
                    plan = self._create_migration_plan(node_id, target_node, node_states, node_positions)
                    if plan:
                        migration_plans.append(plan)
            
            elif node_id.startswith("uav_"):
                battery_level = getattr(state, 'battery_level', 1.0)
                if (battery_level < self.uav_min_battery or 
                    state.load_factor > self.uav_overload_threshold):
                    # UAV需要迁移
                    target_node = self._find_best_target(node_id, "uav", node_states, node_positions)
                    if target_node:
                        plan = self._create_migration_plan(node_id, target_node, node_states, node_positions)
                        if plan:
                            migration_plans.append(plan)
        
        return migration_plans
    
    def _find_best_target(self, source_node_id: str, source_type: str, 
                         node_states: Dict, node_positions: Dict[str, Position]) -> Optional[str]:
        """寻找最佳迁移目标"""
        candidates = []
        
        if source_type == "rsu":
            # RSU可以迁移到其他RSU或UAV
            for node_id, state in node_states.items():
                if node_id.startswith("rsu_") and node_id != source_node_id:
                    if state.load_factor < self.rsu_overload_threshold * 0.8:
                        candidates.append(node_id)
                elif node_id.startswith("uav_"):
                    battery_level = getattr(state, 'battery_level', 1.0)
                    if (battery_level > self.uav_min_battery * 1.5 and 
                        state.load_factor < self.uav_overload_threshold * 0.8):
                        candidates.append(node_id)
        
        elif source_type == "uav":
            # UAV主要迁移到RSU
            for node_id, state in node_states.items():
                if node_id.startswith("rsu_"):
                    if state.load_factor < self.rsu_overload_threshold * 0.8:
                        candidates.append(node_id)
        
        # 选择距离最近的候选
        if candidates and source_node_id in node_positions:
            source_pos = node_positions[source_node_id]
            best_candidate = min(candidates, 
                               key=lambda x: source_pos.distance_to(node_positions.get(x, source_pos)))
            return best_candidate
        
        return None
    
    def _create_migration_plan(self, source_node_id: str, target_node_id: str,
                             node_states: Dict, node_positions: Dict[str, Position]) -> Optional[MigrationPlan]:
        """创建迁移计划"""
        # 计算迁移成本
        distance = 0.0
        if source_node_id in node_positions and target_node_id in node_positions:
            distance = node_positions[source_node_id].distance_to(node_positions[target_node_id])
        
        # 简化的成本计算
        transmission_cost = distance / 1000.0  # 距离成本
        computation_cost = 1.0  # 固定计算成本
        latency_cost = distance * 0.001  # 延迟成本
        
        total_cost = (self.alpha_comp * computation_cost + 
                     self.alpha_tx * transmission_cost + 
                     self.alpha_lat * latency_cost)
        
        # 计算迁移时延
        migration_delay = max(0.01, distance / config.migration.migration_bandwidth)
        
        # 计算成功概率
        success_prob = max(0.5, 0.9 - distance / 10000.0)  # 距离越远成功率越低
        
        # 确定迁移类型
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
    
    def execute_migration(self, migration_plan: MigrationPlan) -> bool:
        """
        执行Keep-Before-Break迁移
        返回是否成功
        """
        self.migration_stats['total_attempts'] += 1
        
        # 模拟Keep-Before-Break过程
        # 1. 准备阶段 (70%时间)
        preparation_time = migration_plan.migration_delay * 0.7
        
        # 2. 同步阶段 (25%时间)
        sync_time = migration_plan.migration_delay * 0.25
        
        # 3. 静默切换阶段 (5%时间) - 这是实际的downtime
        migration_plan.downtime = migration_plan.migration_delay * 0.05
        
        # 判断是否成功
        success = np.random.random() < migration_plan.success_probability
        
        if success:
            self.migration_stats['successful_migrations'] += 1
            self.migration_stats['total_downtime'] += migration_plan.downtime
            migration_plan.is_completed = True
            
            # 更新冷却时间
            self.node_last_migration[migration_plan.source_node_id] = time.time()
            
            # 更新平均成本
            self._update_avg_cost(migration_plan.migration_cost)
        
        return success
    
    def _update_avg_cost(self, new_cost: float):
        """更新平均成本"""
        current_avg = self.migration_stats['avg_cost']
        success_count = self.migration_stats['successful_migrations']
        
        if success_count == 1:
            self.migration_stats['avg_cost'] = new_cost
        else:
            # 移动平均
            alpha = 0.1
            self.migration_stats['avg_cost'] = alpha * new_cost + (1 - alpha) * current_avg
    
    def get_migration_statistics(self) -> Dict:
        """获取迁移统计信息"""
        total_attempts = self.migration_stats['total_attempts']
        successful = self.migration_stats['successful_migrations']
        
        return {
            'total_attempts': total_attempts,
            'successful_migrations': successful,
            'success_rate': successful / max(1, total_attempts),
            'total_downtime': self.migration_stats['total_downtime'],
            'avg_downtime_per_migration': self.migration_stats['total_downtime'] / max(1, successful),
            'avg_cost': self.migration_stats['avg_cost']
        }
    
    def step(self, node_states: Dict, node_positions: Dict[str, Position]) -> Dict:
        """迁移管理器单步更新"""
        # 检查迁移需求
        migration_plans = self.check_migration_needs(node_states, node_positions)
        
        step_stats = {
            'migrations_planned': len(migration_plans),
            'migrations_executed': 0,
            'migrations_successful': 0
        }
        
        # 执行迁移计划
        for plan in migration_plans:
            step_stats['migrations_executed'] += 1
            success = self.execute_migration(plan)
            if success:
                step_stats['migrations_successful'] += 1
        
        return step_stats