"""
浠诲姟杩佺Щ绠＄悊鍣?- 瀵瑰簲璁烘枃绗?鑺?
瀹炵幇Keep-Before-Break浠诲姟杩佺Щ鏈哄埗鍜屼綆涓柇鍒囨崲
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
    """杩佺Щ绫诲瀷鏋氫妇"""
    RSU_TO_RSU = "rsu_to_rsu"
    RSU_TO_UAV = "rsu_to_uav"
    UAV_TO_RSU = "uav_to_rsu"
    VEHICLE_FOLLOW = "vehicle_follow"
    PREEMPTIVE = "preemptive"


@dataclass
class MigrationPlan:
    """杩佺Щ璁″垝鏁版嵁缁撴瀯"""
    migration_id: str
    migration_type: MigrationType
    source_node_id: str
    target_node_id: str
    migration_cost: float = 0.0
    migration_delay: float = 0.0
    success_probability: float = 0.0
    is_completed: bool = False
    downtime: float = 0.001  # Keep-Before-Break鐨勪腑鏂椂闂?


class TaskMigrationManager:
    """
    浠诲姟杩佺Щ绠＄悊鍣?- 鏁村悎杩佺Щ鍔熻兘
    """
    
    def __init__(self):
        # 瑙﹀彂闃堝€?
        self.rsu_overload_threshold = config.migration.rsu_overload_threshold
        self.uav_overload_threshold = config.migration.uav_overload_threshold
        self.uav_min_battery = config.migration.uav_min_battery
        
        # 鎴愭湰鍙傛暟
        self.alpha_comp = config.migration.migration_alpha_comp
        self.alpha_tx = config.migration.migration_alpha_tx
        self.alpha_lat = config.migration.migration_alpha_lat
        
        # 缁熻淇℃伅
        self.migration_stats = {
            'total_attempts': 0,
            'successful_migrations': 0,
            'total_downtime': 0.0,
            'avg_cost': 0.0
        }
        
        # 鍐峰嵈绠＄悊
        self.node_last_migration: Dict[str, float] = {}
        self.cooldown_period = config.migration.cooldown_period
    
    def check_migration_needs(self, node_states: Dict, node_positions: Dict[str, Position]) -> List[MigrationPlan]:
        """妫€鏌ュ苟鍒涘缓杩佺Щ璁″垝"""
        migration_plans = []
        current_time = time.time()
        
        for node_id, state in node_states.items():
            # 妫€鏌ュ喎鍗存湡
            if (node_id in self.node_last_migration and 
                current_time - self.node_last_migration[node_id] < self.cooldown_period):
                continue
            
            if node_id.startswith("rsu_") and state.load_factor > self.rsu_overload_threshold:
                # RSU杩囪浇锛屽鎵捐縼绉荤洰鏍?
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
        
        return migration_plans
    
    def _find_best_target(self, source_node_id: str, source_type: str, 
                         node_states: Dict, node_positions: Dict[str, Position]) -> Optional[str]:
        """瀵绘壘鏈€浣宠縼绉荤洰鏍?""
        candidates = []
        
        if source_type == "rsu":
            # 馃敡 淇锛氭斁瀹借縼绉荤洰鏍囬€夋嫨鏉′欢锛屽鍔犺縼绉绘満浼?
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
            # 馃敡 淇锛歎AV杩佺Щ鏉′欢涔熼€傚害鏀惧
            for node_id, state in node_states.items():
                if node_id.startswith("rsu_"):
                    if state.load_factor < self.rsu_overload_threshold * 0.9:  # 浠?.8鎻愰珮鍒?.9
                        candidates.append(node_id)
        
        # 閫夋嫨璺濈鏈€杩戠殑鍊欓€?
        if candidates and source_node_id in node_positions:
            source_pos = node_positions[source_node_id]
            best_candidate = min(candidates, 
                               key=lambda x: source_pos.distance_to(node_positions.get(x, source_pos)))
            return best_candidate
        
        return None
    
    def _create_migration_plan(self, source_node_id: str, target_node_id: str,
                             node_states: Dict, node_positions: Dict[str, Position]) -> Optional[MigrationPlan]:
        """鍒涘缓杩佺Щ璁″垝"""
        # 璁＄畻杩佺Щ鎴愭湰
        distance = 0.0
        if source_node_id in node_positions and target_node_id in node_positions:
            distance = node_positions[source_node_id].distance_to(node_positions[target_node_id])
        
        # 绠€鍖栫殑鎴愭湰璁＄畻
        transmission_cost = distance / 1000.0  # 璺濈鎴愭湰
        computation_cost = 1.0  # 鍥哄畾璁＄畻鎴愭湰
        latency_cost = migration_delay / max(1e-9, config.network.time_slot_duration)  # 寤惰繜鎴愭湰
        
        total_cost = (self.alpha_comp * computation_cost + 
                     self.alpha_tx * transmission_cost + 
                     self.alpha_lat * latency_cost)
        
        # 璁＄畻杩佺Щ鏃跺欢
        migration_delay = max(0.01, distance / config.migration.migration_bandwidth)
        
        # 璁＄畻鎴愬姛姒傜巼
        success_prob = max(0.5, 0.9 - distance / 10000.0)  # 璺濈瓒婅繙鎴愬姛鐜囪秺浣?
        
        # 纭畾杩佺Щ绫诲瀷
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
        鎵цKeep-Before-Break杩佺Щ
        杩斿洖鏄惁鎴愬姛
        """
        self.migration_stats['total_attempts'] += 1
        
        # 妯℃嫙Keep-Before-Break杩囩▼
        # 1. 鍑嗗闃舵 (70%鏃堕棿)
        preparation_time = migration_plan.migration_delay * 0.7
        
        # 2. 鍚屾闃舵 (25%鏃堕棿)
        sync_time = migration_plan.migration_delay * 0.25
        
        # 3. 闈欓粯鍒囨崲闃舵 (5%鏃堕棿) - 杩欐槸瀹為檯鐨刣owntime
        migration_plan.downtime = migration_plan.migration_delay * 0.05
        
        # 鍒ゆ柇鏄惁鎴愬姛
        success = np.random.random() < migration_plan.success_probability
        
        if success:
            self.migration_stats['successful_migrations'] += 1
            self.migration_stats['total_downtime'] += migration_plan.downtime
            migration_plan.is_completed = True
            
            # 鏇存柊鍐峰嵈鏃堕棿
            self.node_last_migration[migration_plan.source_node_id] = time.time()
            
            # 鏇存柊骞冲潎鎴愭湰
            self._update_avg_cost(migration_plan.migration_cost)
        
        return success
    
    def _update_avg_cost(self, new_cost: float):
        """鏇存柊骞冲潎鎴愭湰"""
        current_avg = self.migration_stats['avg_cost']
        success_count = self.migration_stats['successful_migrations']
        
        if success_count == 1:
            self.migration_stats['avg_cost'] = new_cost
        else:
            # 绉诲姩骞冲潎
            alpha = 0.1
            self.migration_stats['avg_cost'] = alpha * new_cost + (1 - alpha) * current_avg
    
    def get_migration_statistics(self) -> Dict:
        """鑾峰彇杩佺Щ缁熻淇℃伅"""
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
        """杩佺Щ绠＄悊鍣ㄥ崟姝ユ洿鏂?""
        # 妫€鏌ヨ縼绉婚渶姹?
        migration_plans = self.check_migration_needs(node_states, node_positions)
        
        step_stats = {
            'migrations_planned': len(migration_plans),
            'migrations_executed': 0,
            'migrations_successful': 0
        }
        
        # 鎵ц杩佺Щ璁″垝
        for plan in migration_plans:
            step_stats['migrations_executed'] += 1
            success = self.execute_migration(plan)
            if success:
                step_stats['migrations_successful'] += 1
        
        return step_stats
