#!/usr/bin/env python3
"""
瀹屾暣绯荤粺浠跨湡鍣?
鐢ㄤ簬娴嬭瘯瀹屾暣鐨勮溅鑱旂綉杈圭紭缂撳瓨绯荤粺
"""

import numpy as np
import torch
import random
from typing import Dict, List, Tuple, Any, Optional
import json
from datetime import datetime
# 馃敡 淇锛氬鍏ョ粺涓€鏃堕棿绠＄悊鍣?
from utils.unified_time_manager import get_simulation_time, advance_simulation_time, reset_simulation_time
# 馃敡 淇锛氬鍏ealistic鍐呭鐢熸垚鍣?
from utils.realistic_content_generator import generate_realistic_content, get_realistic_content_size

class CompleteSystemSimulator:
    """瀹屾暣绯荤粺浠跨湡鍣?""
    
    def __init__(self, config: Dict = None):
        """鍒濆鍖栦豢鐪熷櫒"""
        self.config = config or self.get_default_config()
        self.override_topology = self.config.get('override_topology', False)
        # 缁熶竴绯荤粺閰嶇疆鍏ュ彛锛堣嫢鍙敤锛?
        try:
            from config import config as sys_config
            self.sys_config = sys_config
        except Exception:
            self.sys_config = None
        
        # 缃戠粶鎷撴墤
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.num_vehicles = getattr(self.sys_config.network, 'num_vehicles', 12)
            self.num_rsus = getattr(self.sys_config.network, 'num_rsus', 6)
            self.num_uavs = getattr(self.sys_config.network, 'num_uavs', 2)
        else:
            self.num_vehicles = self.config.get('num_vehicles', 12)
            self.num_rsus = self.config.get('num_rsus', 4)  # 馃敡 淇锛氫娇鐢ㄦ纭殑榛樿鍊?
            self.num_uavs = self.config.get('num_uavs', 2)
        if self.sys_config is not None and not self.override_topology:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300)
        else:
            default_radius = getattr(self.sys_config.network, 'coverage_radius', 300) if self.sys_config is not None else 300
        self.coverage_radius = self.config.get('coverage_radius', default_radius)

        
        # 浠跨湡鍙傛暟
        if self.sys_config is not None and not self.config.get('override_topology', False):
            self.simulation_time = getattr(self.sys_config, 'simulation_time', 1000)
            self.time_slot = getattr(self.sys_config.network, 'time_slot_duration', 0.2)  # 馃殌 閫傚簲楂樿礋杞芥椂闅?
            self.task_arrival_rate = getattr(self.sys_config.task, 'arrival_rate', 2.5)  # 馃殌 楂樿礋杞藉埌杈剧巼
        else:
            self.simulation_time = self.config.get('simulation_time', 1000)
            self.time_slot = self.config.get('time_slot', 0.2)  # 馃殌 楂樿礋杞介粯璁ゆ椂闅?
            self.task_arrival_rate = self.config.get('task_arrival_rate', 2.5)  # 馃殌 楂樿礋杞介粯璁ゅ埌杈剧巼
        
        self.task_config = getattr(self.sys_config, 'task', None) if self.sys_config is not None else None
        self.service_config = getattr(self.sys_config, 'service', None) if self.sys_config is not None else None
        self.stats_config = getattr(self.sys_config, 'stats', None) if self.sys_config is not None else None
        
        # 鎬ц兘缁熻涓庤繍琛屾€?
        self.stats = self._fresh_stats_dict()
        self.active_tasks: List[Dict] = []  # 姣忛」: {id, vehicle_id, arrival_time, deadline, work_remaining, node_type, node_idx}
        self.task_counter = 0
        self.current_step = 0
        self.current_time = 0.0
        
        # 鍒濆鍖栫粍浠?
        self.initialize_components()
        self._reset_runtime_states()
    
    def get_default_config(self) -> Dict:
        """鑾峰彇榛樿閰嶇疆"""
        return {
            'num_vehicles': 12,
            'num_rsus': 6,
            'num_uavs': 2,
            'simulation_time': 1000,
            'time_slot': 0.1,
            'task_arrival_rate': 0.8,
            'cache_capacity': 100,
            'computation_capacity': 1000,  # MIPS
            'bandwidth': 20,  # MHz
            'transmission_power': 0.1,  # W
            'computation_power': 1.0,  # W
            'rsu_base_service': 4,
            'rsu_max_service': 9,
            'rsu_work_capacity': 2.5,
            'uav_base_service': 3,
            'uav_max_service': 6,
            'uav_work_capacity': 1.7,
            'drop_log_interval': 200,
            'task_report_interval': 100,
            'task_compute_density': 400,
        }
    
    def initialize_components(self):
        """鍒濆鍖栫郴缁熺粍浠?""
        # 馃殾 涓诲共閬?鍙岃矾鍙ｅ垵濮嬪寲
        # 鍧愭爣绯荤粺 0..1000锛屼富骞查亾娌?x 杞翠腑绾?y=500锛屼粠宸﹀悜鍙筹紱涓ゅ璺彛浣嶄簬 x=300 涓?x=700
        self.road_y = 500.0
        self.intersections = {  # 淇″彿鐏浉浣? 鍛ㄦ湡 T锛岀豢鐏瘮渚?g
            'L': {'x': 300.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 0.0},
            'R': {'x': 700.0, 'cycle_T': 60.0, 'green_ratio': 0.5, 'phase_offset': 15.0},
        }

        # 杞﹁締鍒濆鍖栵細钀藉湪閬撹矾涓婏紝鏂瑰悜涓轰笢(0)鎴栬タ(pi)锛岃溅閬撳唴寰壈
        self.vehicles = []
        for i in range(self.num_vehicles):
            go_east = np.random.rand() < 0.6  # 60% 鍚戜笢
            base_dir = 0.0 if go_east else np.pi
            x0 = np.random.uniform(100.0, 900.0)
            y0 = self.road_y + np.random.uniform(-6.0, 6.0)  # 绠€鍗曚袱杞﹂亾璺箙
            v0 = np.random.uniform(12.0, 22.0)
            vehicle = {
                'id': f'V_{i}',
                'position': np.array([x0, y0], dtype=float),
                'velocity': v0,
                'direction': base_dir,
                'lane_bias': y0 - self.road_y,
                'tasks': [],
                'energy_consumed': 0.0,
                'device_cache': {},
                'device_cache_capacity': 32.0
            }
            self.vehicles.append(vehicle)
        print("馃殾 杞﹁締鍒濆鍖栧畬鎴愶細涓诲共閬撳弻璺彛鍦烘櫙")
        
        # RSU鑺傜偣
        self.rsus = []
        # 馃敡 鍔ㄦ€丷SU閮ㄧ讲锛氭牴鎹畁um_rsus鍧囧寑鍒嗗竷鍦ㄩ亾璺笂
        if self.num_rsus <= 4:
            # 鍘熷鍥哄畾4涓猂SU鐨勯儴缃?
            rsu_positions = [
                np.array([300.0, 500.0]),
                np.array([500.0, 500.0]),
                np.array([700.0, 500.0]),
                np.array([900.0, 500.0]),
            ]
        else:
            # 鍔ㄦ€佺敓鎴怰SU浣嶇疆锛屽潎鍖€鍒嗗竷鍦?00-900涔嬮棿
            rsu_positions = []
            spacing = 700.0 / (self.num_rsus - 1)  # 鍧囧寑闂撮殧
            for i in range(self.num_rsus):
                x_pos = 200.0 + i * spacing
                rsu_positions.append(np.array([x_pos, 500.0]))
        
        # 鍒涘缓RSU
        for i in range(self.num_rsus):
            rsu = {
                'id': f'RSU_{i}',
                'position': rsu_positions[i],
                'coverage_radius': self.coverage_radius,
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'rsu_cache_capacity', 10e9) if self.sys_config is not None else 10e9),
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.rsus.append(rsu)
        
        # UAV鑺傜偣
        self.uavs = []
        # 馃敡 鍔ㄦ€乁AV閮ㄧ讲锛氭牴鎹畁um_uavs鍧囧寑鍒嗗竷
        if self.num_uavs <= 2:
            # 鍘熷2鏋禪AV鐨勯儴缃?
            uav_positions = [
                np.array([300.0, 500.0, 120.0]),
                np.array([700.0, 500.0, 120.0]),
            ]
        else:
            # 鍔ㄦ€佺敓鎴怳AV浣嶇疆锛屽潎鍖€鍒嗗竷鍦ㄩ亾璺笂鏂?
            uav_positions = []
            spacing = 600.0 / (self.num_uavs - 1)  # 鍧囧寑闂撮殧
            for i in range(self.num_uavs):
                x_pos = 200.0 + i * spacing
                uav_positions.append(np.array([x_pos, 500.0, 120.0]))
        
        # 鍒涘缓UAV
        for i in range(self.num_uavs):
            uav = {
                'id': f'UAV_{i}',
                'position': uav_positions[i],  # 鍥哄畾鎮仠浣嶇疆
                'velocity': 0.0,
                'coverage_radius': 350.0,
                'cache': {},
                'cache_capacity': self.config['cache_capacity'],
                'cache_capacity_bytes': (getattr(self.sys_config.cache, 'uav_cache_capacity', 2e9) if self.sys_config is not None else 2e9),
                'computation_queue': [],
                'energy_consumed': 0.0
            }
            self.uavs.append(uav)
        
        print(f"鉁?鍒涘缓浜?{self.num_vehicles} 杞﹁締, {self.num_rsus} RSU, {self.num_uavs} UAV")
        
        # 馃彚 鍒濆鍖栦腑澶甊SU璋冨害鍣?(閫夋嫨RSU_2浣滀负涓ぎ璋冨害涓績)
        try:
            from utils.central_rsu_scheduler import create_central_scheduler
            central_rsu_id = f"RSU_{2 if self.num_rsus > 2 else 0}"
            self.central_scheduler = create_central_scheduler(central_rsu_id)
            print(f"馃彚 涓ぎRSU璋冨害鍣ㄥ凡鍚敤: {central_rsu_id}")
        except Exception as e:
            print(f"鈿狅笍 涓ぎ璋冨害鍣ㄥ姞杞藉け璐? {e}")
            self.central_scheduler = None
        
        # 鎳掑姞杞借縼绉荤鐞嗗櫒
        try:
            from migration.migration_manager import TaskMigrationManager
            if not hasattr(self, 'migration_manager') or self.migration_manager is None:
                self.migration_manager = TaskMigrationManager()
        except Exception:
            self.migration_manager = None
        
        # 涓€鑷存€ц嚜妫€锛堜笉寮哄埗缁堟锛屼粎鎻愮ず锛?
        try:
            expected_rsus, expected_uavs = 4, 2
            if self.num_rsus != expected_rsus or self.num_uavs != expected_uavs:
                print(f"鈿狅笍 鎷撴墤涓€鑷存€ф彁绀? 褰撳墠 num_rsus={self.num_rsus}, num_uavs={self.num_uavs}, 寤鸿涓?{expected_rsus}/{expected_uavs} 浠ュ尮閰嶈鏂囧浘绀?)
            print("馃彚 涓ぎRSU璁惧畾: RSU_2 (浣滀负璋冨害涓庡洖浼犳眹鑱氳妭鐐?")
        except Exception:
            pass
    
    def _setup_scenario(self):
        """璁剧疆浠跨湡鍦烘櫙"""
        # 閲嶆柊鍒濆鍖栫粍浠讹紙濡傛灉闇€瑕侊級
        self.initialize_components()
        self._reset_runtime_states()
        print("鉁?鍒濆鍖栦簡 6 涓紦瀛樼鐞嗗櫒")

    def _fresh_stats_dict(self) -> Dict[str, float]:
        """鍒涘缓鏂扮殑缁熻瀛楀吀锛屼繚璇佸叧閿寚鏍囬綈鍏?""
        return {
            'total_tasks': 0,
            'processed_tasks': 0,
            'completed_tasks': 0,
            'dropped_tasks': 0,
            'generated_data_bytes': 0.0,
            'dropped_data_bytes': 0.0,
            'total_delay': 0.0,
            'total_energy': 0.0,
            'energy_uplink': 0.0,
            'energy_downlink': 0.0,
            'local_cache_hits': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'migrations_executed': 0,
            'migrations_successful': 0,
            'rsu_migration_delay': 0.0,
            'rsu_migration_energy': 0.0,
            'rsu_migration_data': 0.0,
            'uav_migration_distance': 0.0,
            'uav_migration_count': 0,
            'task_generation': {'total': 0, 'by_type': {}, 'by_scenario': {}},
            'drop_stats': {
                'total': 0,
                'wait_time_sum': 0.0,
                'queue_sum': 0,
                'by_type': {},
                'by_scenario': {}
            }
        }

    def _reset_runtime_states(self):
        """閲嶇疆杩愯鏃剁姸鎬侊紙鐢ㄤ簬episode閲嶅惎锛?""
        reset_simulation_time()
        self.current_step = 0
        self.current_time = 0.0
        self.task_counter = 0
        self.stats = self._fresh_stats_dict()
        self.active_tasks = []
        self._last_app_name = 'unknown'

        # 閲嶇疆杞﹁締/鑺傜偣鐘舵€?
        for vehicle in self.vehicles:
            vehicle.setdefault('tasks', [])
            vehicle['tasks'].clear()
            vehicle['energy_consumed'] = 0.0
            vehicle['device_cache'] = {}
            vehicle['device_cache_capacity'] = vehicle.get('device_cache_capacity', 32.0)

        for rsu in self.rsus:
            rsu.setdefault('cache', {})
            rsu['computation_queue'] = []
            rsu['energy_consumed'] = 0.0

        for uav in self.uavs:
            uav.setdefault('cache', {})
            uav['computation_queue'] = []
            uav['energy_consumed'] = 0.0
    
    def _get_realistic_content_size(self, content_id: str) -> float:
        """
        馃敡 淇锛氫娇鐢╮ealistic鍐呭鐢熸垚鍣ㄨ幏鍙栧ぇ灏?
        """
        return get_realistic_content_size(content_id)
    
    def _calculate_available_cache_capacity(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        馃敡 淇锛氭纭绠楀彲鐢ㄧ紦瀛樺閲?MB)
        """
        if not cache or cache_capacity_mb <= 0:
            return cache_capacity_mb
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                # 鍏煎鏃ф牸寮?
                total_used_mb += 1.0
        
        available_mb = cache_capacity_mb - total_used_mb
        return max(0.0, available_mb)
    
    def _infer_content_type(self, content_id: str) -> str:
        """
        馃敡 淇锛氭牴鎹唴瀹笽D鎺ㄦ柇鍐呭绫诲瀷
        """
        content_id_lower = content_id.lower()
        
        if 'traffic' in content_id_lower:
            return 'traffic_info'
        elif 'nav' in content_id_lower or 'route' in content_id_lower:
            return 'navigation'
        elif 'safety' in content_id_lower or 'alert' in content_id_lower:
            return 'safety_alert'
        elif 'park' in content_id_lower:
            return 'parking_info'
        elif 'weather' in content_id_lower:
            return 'weather_info'
        elif 'map' in content_id_lower:
            return 'map_data'
        elif 'video' in content_id_lower or 'entertainment' in content_id_lower:
            return 'entertainment'
        elif 'sensor' in content_id_lower:
            return 'sensor_data'
        else:
            return 'general'
    
    def generate_task(self, vehicle_id: str) -> Dict:
        """鐢熸垚璁＄畻浠诲姟 - 浣跨敤閰嶇疆椹卞姩鐨勪换鍔″満鏅畾涔?""
        self.task_counter += 1

        task_cfg = getattr(self.sys_config, 'task', None) if self.sys_config is not None else None
        time_slot = getattr(self.sys_config.network, 'time_slot_duration', self.time_slot) if self.sys_config is not None else self.time_slot

        scenario_name = 'fallback'
        relax_factor_applied = self.config.get('deadline_relax_fallback', 1.3)
        task_type = 3

        if task_cfg is not None:
            scenario = task_cfg.sample_scenario()
            scenario_name = scenario.name
            relax_factor_applied = scenario.relax_factor or task_cfg.deadline_relax_default

            deadline_duration = np.random.uniform(scenario.min_deadline, scenario.max_deadline)
            deadline_duration *= relax_factor_applied
            max_delay_slots = max(1, int(deadline_duration / max(time_slot, 1e-6)))
            task_type = scenario.task_type or task_cfg.get_task_type(max_delay_slots)

            profile = task_cfg.get_profile(task_type)
            data_min, data_max = profile.data_range
            data_size_bytes = float(np.random.uniform(data_min, data_max))
            compute_density = profile.compute_density
        else:
            deadline_duration = np.random.uniform(0.5, 3.0) * relax_factor_applied
            task_type = int(np.random.randint(1, 5))
            data_size_mb = np.random.exponential(0.5)
            data_size_bytes = data_size_mb * 1e6
            compute_density = self.config.get('task_compute_density', 400)
            max_delay_slots = max(1, int(deadline_duration / max(self.config.get('time_slot', self.time_slot), 0.1)))

        # 浠诲姟澶嶆潅搴︽帶鍒?
        data_size_mb = data_size_bytes / 1e6
        effective_density = compute_density
        complexity_multiplier = 1.0

        if self.config.get('high_load_mode', False):
            complexity_multiplier = self.config.get('task_complexity_multiplier', 1.5)
            data_size_mb = min(data_size_mb * 1.1, 2.0)
            data_size_bytes = data_size_mb * 1e6
            effective_density = min(effective_density * 1.05, 200)

        total_bits = data_size_bytes * 8
        computation_cycles = total_bits * effective_density
        computation_mips = (computation_cycles / 1e6) * complexity_multiplier

        task = {
            'id': f'task_{self.task_counter}',
            'vehicle_id': vehicle_id,
            'arrival_time': self.current_time,
            'data_size': data_size_mb,
            'data_size_bytes': data_size_bytes,
            'computation_requirement': computation_mips,
            'deadline': self.current_time + deadline_duration,
            'content_id': f'content_{np.random.randint(0, 100)}',
            'priority': np.random.uniform(0.1, 1.0),
            'task_type': task_type,
            'app_scenario': scenario_name,
            'app_name': scenario_name,
            'compute_density': effective_density,
            'complexity_multiplier': complexity_multiplier,
            'max_delay_slots': max_delay_slots,
            'deadline_relax_factor': relax_factor_applied
        }

        self._last_app_name = scenario_name

        # 馃搳 浠诲姟缁熻鏀堕泦
        gen_stats = self.stats.setdefault('task_generation', {'total': 0, 'by_type': {}, 'by_scenario': {}})
        gen_stats['total'] += 1
        by_type = gen_stats.setdefault('by_type', {})
        by_type[task_type] = by_type.get(task_type, 0) + 1
        by_scenario = gen_stats.setdefault('by_scenario', {})
        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1

        report_interval = self.stats_config.task_report_interval if getattr(self, 'stats_config', None) else self.config.get('task_report_interval', 100)
        report_interval = max(1, int(report_interval))
        if gen_stats['total'] % report_interval == 0:
            total_classified = sum(by_type.values()) or 1
            type1_pct = by_type.get(1, 0) / total_classified * 100
            type2_pct = by_type.get(2, 0) / total_classified * 100
            type3_pct = by_type.get(3, 0) / total_classified * 100
            type4_pct = by_type.get(4, 0) / total_classified * 100
            print(
                f"馃搳 浠诲姟鍒嗙被缁熻({gen_stats['total']}): "
                f"绫诲瀷1={type1_pct:.1f}%, 绫诲瀷2={type2_pct:.1f}%, 绫诲瀷3={type3_pct:.1f}%, 绫诲瀷4={type4_pct:.1f}%"
            )
            print(
                f"   褰撳墠浠诲姟: {scenario_name}, {deadline_duration:.2f}s 鈫?"
                f"绫诲瀷{task_type}, 鏁版嵁{data_size_mb:.2f}MB"
            )

        return task
    
    def calculate_distance(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """璁＄畻涓ょ偣闂磋窛绂?""
        if len(pos1) == 3 and len(pos2) == 2:
            pos2 = np.append(pos2, 0)  # 2D杞?D
        elif len(pos1) == 2 and len(pos2) == 3:
            pos1 = np.append(pos1, 0)
        
        return np.linalg.norm(pos1 - pos2)
    
    def _find_least_loaded_node(self, node_type: str, exclude_node: Dict = None) -> Dict:
        """瀵绘壘璐熻浇鏈€杞荤殑鑺傜偣"""
        if node_type == 'RSU':
            candidates = [rsu for rsu in self.rsus if rsu != exclude_node]
        elif node_type == 'UAV':
            candidates = [uav for uav in self.uavs if uav != exclude_node]
        else:
            return None
        
        if not candidates:
            return None
        
        # 鎵惧埌闃熷垪闀垮害鏈€鐭殑鑺傜偣
        best_node = min(candidates, key=lambda n: len(n.get('computation_queue', [])))
        return best_node
    
    def _process_node_queues(self):
        """馃敡 鍏抽敭淇锛氬鐞哛SU鍜孶AV闃熷垪涓殑浠诲姟锛岄槻姝换鍔″爢绉?""
        # 澶勭悊鎵€鏈塕SU闃熷垪
        for rsu in self.rsus:
            self._process_single_node_queue(rsu, 'RSU')
        
        # 澶勭悊鎵€鏈塙AV闃熷垪
        for uav in self.uavs:
            self._process_single_node_queue(uav, 'UAV')
    

    def _process_single_node_queue(self, node: Dict, node_type: str):
        "澶勭悊鍗曚釜鑺傜偣鐨勮绠楅槦鍒?
        queue = node.get('computation_queue', []) or []
        queue_len = len(queue)
        if queue_len == 0:
            return

        if node_type == 'RSU':
            if self.service_config:
                base_capacity = self.service_config.rsu_base_service
                max_service = self.service_config.rsu_max_service
                boost_divisor = self.service_config.rsu_queue_boost_divisor
            else:
                base_capacity = int(self.config.get('rsu_base_service', 4))
                max_service = int(self.config.get('rsu_max_service', 9))
                boost_divisor = 5.0
        elif node_type == 'UAV':
            if self.service_config:
                base_capacity = self.service_config.uav_base_service
                max_service = self.service_config.uav_max_service
                boost_divisor = self.service_config.uav_queue_boost_divisor
            else:
                base_capacity = int(self.config.get('uav_base_service', 3))
                max_service = int(self.config.get('uav_max_service', 6))
                boost_divisor = 4.0
        else:
            base_capacity = 2
            max_service = 4
            boost_divisor = 5.0

        dynamic_boost = 0
        if queue_len > base_capacity:
            dynamic_boost = int(np.ceil((queue_len - base_capacity) / boost_divisor))

        max_tasks_per_slot = min(queue_len, base_capacity + dynamic_boost)
        max_tasks_per_slot = min(max_tasks_per_slot, max_service)
        max_tasks_per_slot = max(max_tasks_per_slot, min(queue_len, base_capacity))
        tasks_to_process = max_tasks_per_slot

        new_queue: List[Dict] = []
        current_time = getattr(self, 'current_time', 0.0)

        for idx, task in enumerate(queue):
            if current_time - task.get('queued_at', -1e9) < self.time_slot:
                new_queue.append(task)
                continue

            if idx >= tasks_to_process:
                new_queue.append(task)
                continue

            remaining_work = float(task.get('work_remaining', 0.5))
            if node_type == 'RSU':
                if self.service_config:
                    work_capacity = self.time_slot * self.service_config.rsu_work_capacity
                else:
                    work_capacity = self.time_slot * self.config.get('rsu_work_capacity', 2.5)
            elif node_type == 'UAV':
                if self.service_config:
                    work_capacity = self.time_slot * self.service_config.uav_work_capacity
                else:
                    work_capacity = self.time_slot * self.config.get('uav_work_capacity', 1.7)
            else:
                work_capacity = self.time_slot * 1.2

            remaining_work -= work_capacity
            task['work_remaining'] = max(0.0, remaining_work)

            if task['work_remaining'] > 0.0:
                new_queue.append(task)
                continue

            self.stats['completed_tasks'] += 1
            self.stats['processed_tasks'] = self.stats.get('processed_tasks', 0) + 1

            actual_delay = current_time - task.get('arrival_time', current_time)
            actual_delay = max(0.001, min(actual_delay, 20.0))
            self.stats['total_delay'] += actual_delay

            vehicle_id = task.get('vehicle_id', 'V_0')
            vehicle = next((v for v in self.vehicles if v['id'] == vehicle_id), None)
            if vehicle is not None:
                node_pos = node.get('position', np.zeros(3))
                if len(node_pos) == 2:
                    node_pos = np.append(node_pos, 0.0)
                vehicle_pos = vehicle.get('position', np.zeros(3))
                if len(vehicle_pos) == 2:
                    vehicle_pos = np.append(vehicle_pos, 0.0)
                distance = np.linalg.norm(node_pos - vehicle_pos)
                result_size = task.get('data_size_bytes', task.get('data_size', 1.0) * 1e6) * 0.1
                down_delay, down_energy = self._estimate_transmission(result_size, distance, node_type.lower())
                self.stats['energy_downlink'] = self.stats.get('energy_downlink', 0.0) + down_energy
                self.stats['total_delay'] += down_delay
                self.stats['total_energy'] += down_energy

            if node_type == 'RSU':
                processing_power = 50.0
            elif node_type == 'UAV':
                processing_power = 20.0
            else:
                processing_power = 10.0

            task_energy = processing_power * work_capacity
            self.stats['total_energy'] += task_energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + task_energy

            task['completed'] = True

        node['computation_queue'] = new_queue

    def find_nearest_rsu(self, vehicle_pos: np.ndarray) -> Dict:
        """鎵惧埌鏈€杩戠殑RSU"""
        min_distance = float('inf')
        nearest_rsu = None
        
        for rsu in self.rsus:
            distance = self.calculate_distance(vehicle_pos, rsu['position'])
            if distance < min_distance and distance <= rsu['coverage_radius']:
                min_distance = distance
                nearest_rsu = rsu
        
        return nearest_rsu
    
    def find_nearest_uav(self, vehicle_pos: np.ndarray) -> Dict:
        """鎵惧埌鏈€杩戠殑UAV"""
        min_distance = float('inf')
        nearest_uav = None
        
        for uav in self.uavs:
            distance = self.calculate_distance(vehicle_pos, uav['position'])
            if distance < min_distance:
                min_distance = distance
                nearest_uav = uav
        
        return nearest_uav
    
    def check_cache_hit(self, content_id: str, node: Dict) -> bool:
        """妫€鏌ョ紦瀛樺懡涓?""
        if content_id in node.get('cache', {}):
            self.stats['cache_hits'] += 1
            return True
        else:
            self.stats['cache_misses'] += 1
            return False
    
    def check_cache_hit_adaptive(
        self,
        content_id: str,
        node: Dict,
        agents_actions: Dict = None,
        node_type: str = 'RSU'
    ) -> bool:
        """馃 鏅鸿兘浣撴帶鍒剁殑鑷€傚簲缂撳瓨妫€鏌?""
        # 鍩虹缂撳瓨妫€鏌?
        cache_hit = content_id in node.get('cache', {})
        
        # 鏇存柊缁熻
        if cache_hit:
            self.stats['cache_hits'] += 1
            if node_type == 'RSU':
                self._propagate_cache_after_hit(content_id, node, agents_actions)
        else:
            self.stats['cache_misses'] += 1
            
            # 馃 濡傛灉鏈夋櫤鑳戒綋鎺у埗鍣紝鎵ц鑷€傚簲缂撳瓨绛栫暐
            if agents_actions and 'cache_controller' in agents_actions:
                cache_controller = agents_actions['cache_controller']
                
                # 鏇存柊鍐呭鐑害
                cache_controller.update_content_heat(content_id)
                cache_controller.record_cache_result(content_id, was_hit=False)
                
                # 馃敡 淇锛氫娇鐢╮ealistic鍐呭澶у皬鍜屾纭閲忚绠?
                data_size = self._get_realistic_content_size(content_id)
                capacity_limit = node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
                available_capacity = self._calculate_available_cache_capacity(
                    node.get('cache', {}), capacity_limit
                )
                
                should_cache, reason, evictions = cache_controller.should_cache_content(
                    content_id,
                    data_size,
                    available_capacity,
                    node.get('cache', {}),
                    capacity_limit
                )
                
                if should_cache:
                    if 'cache' not in node:
                        node['cache'] = {}
                    cache_dict = node['cache']
                    reclaimed = 0.0
                    for evict_id in evictions:
                        removed = cache_dict.pop(evict_id, None)
                        if removed:
                            reclaimed += float(removed.get('size', 0.0) or 0.0)
                            cache_controller.cache_stats['evicted_items'] += 1
                    if reclaimed > 0.0:
                        available_capacity += reclaimed
                    if available_capacity < data_size:
                        return cache_hit
                    cache_dict[content_id] = {
                        'size': data_size,
                        'timestamp': self.current_time,
                        'reason': reason,
                        'content_type': self._infer_content_type(content_id)
                    }
                    if 'Collaborative cache' in reason:
                        cache_controller.cache_stats['collaborative_writes'] += 1
        
        # 璁板綍缂撳瓨鎺у埗鍣ㄧ粺璁?
        if agents_actions and 'cache_controller' in agents_actions and cache_hit:
            cache_controller = agents_actions['cache_controller'] 
            cache_controller.record_cache_result(content_id, was_hit=True)
            cache_controller.update_content_heat(content_id)
            
        return cache_hit
    
    def _calculate_enhanced_load_factor(self, node: Dict, node_type: str) -> float:
        """
        馃敡 淇锛氱粺涓€鍜宺ealistic鐨勮礋杞藉洜瀛愯绠?
        鍩轰簬瀹為檯闃熷垪璐熻浇锛屼笉浣跨敤铏氬亣鐨勯檺鍒?
        """
        queue_length = len(node.get('computation_queue', []))
        
        # 馃敡 鍩轰簬瀹為檯瑙傚療璋冩暣瀹归噺鍩哄噯
        if node_type == 'RSU':
            # 鍩轰簬瀹為檯娴嬭瘯锛孯SU澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            base_capacity = 20.0  
            queue_factor = queue_length / base_capacity
        else:  # UAV
            # UAV澶勭悊鑳藉姏绾?0涓换鍔′负婊¤礋杞?
            base_capacity = 10.0
            queue_factor = queue_length / base_capacity
        
        # 馃敡 淇锛氫娇鐢ㄦ纭殑缂撳瓨璁＄畻
        cache_utilization = self._calculate_correct_cache_utilization(
            node.get('cache', {}), 
            node.get('cache_capacity', 1000.0 if node_type == 'RSU' else 200.0)
        )
        
        # 馃敡 绠€鍖栦絾鍑嗙‘鐨勮礋杞借绠?
        load_factor = (
            0.8 * queue_factor +           # 闃熷垪鏄富瑕佽礋杞芥寚鏍?0%
            0.2 * cache_utilization       # 缂撳瓨鍒╃敤鐜?0%
        )
        
        # 馃敡 涓嶉檺鍒跺湪1.0锛屽厑璁告樉绀虹湡瀹炶繃杞界▼搴?
        return max(0.0, load_factor)
    
    def _summarize_task_types(self) -> Dict[str, Any]:
        """Aggregate per-task-type queues, active counts, and deadline slack."""
        num_types = 4
        queue_counts = np.zeros(num_types, dtype=float)
        active_counts = np.zeros(num_types, dtype=float)
        deadline_sums = np.zeros(num_types, dtype=float)
        deadline_counts = np.zeros(num_types, dtype=float)

        current_time = getattr(self, "current_time", 0.0)

        def _record(entry: Dict[str, Any]) -> Optional[int]:
            task_type = int(entry.get("task_type", 0) or 0) - 1
            if 0 <= task_type < num_types:
                remaining = max(0.0, entry.get("deadline", current_time) - current_time)
                deadline_sums[task_type] += remaining
                deadline_counts[task_type] += 1.0
                return task_type
            return None

        for node in list(self.rsus) + list(self.uavs):
            for task in node.get("computation_queue", []):
                idx = _record(task)
                if idx is not None:
                    queue_counts[idx] += 1.0

        for task in self.active_tasks:
            idx = _record(task)
            if idx is not None:
                active_counts[idx] += 1.0

        if self.task_config is not None and hasattr(self.task_config, "deadline_range"):
            deadline_upper = float(getattr(self.task_config, "deadline_range", (1.0, 10.0))[1])
        else:
            fallback_range = self.config.get("deadline_range", (1.0, 10.0))
            if isinstance(fallback_range, (list, tuple)) and len(fallback_range) >= 2:
                deadline_upper = float(fallback_range[1])
            else:
                deadline_upper = float(self.config.get("deadline_range_max", 10.0))
        deadline_upper = max(deadline_upper, 1.0)

        queue_total = float(queue_counts.sum())
        active_total = float(active_counts.sum())

        def _normalize(counts: np.ndarray, total: float) -> List[float]:
            if total <= 0.0:
                return [0.0] * num_types
            return [float(np.clip(val / total, 0.0, 1.0)) for val in counts]

        deadline_features = []
        for idx in range(num_types):
            if deadline_counts[idx] > 0.0:
                avg_remaining = deadline_sums[idx] / deadline_counts[idx]
                deadline_features.append(float(np.clip(avg_remaining / deadline_upper, 0.0, 1.0)))
            else:
                deadline_features.append(0.0)

        return {
            "task_type_queue_distribution": _normalize(queue_counts, queue_total),
            "task_type_active_distribution": _normalize(active_counts, active_total),
            "task_type_deadline_remaining": deadline_features,
            "task_type_queue_counts": [float(c) for c in queue_counts],
            "task_type_active_counts": [float(c) for c in active_counts],
        }
    
    def _calculate_correct_cache_utilization(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        馃敡 璁＄畻姝ｇ‘鐨勭紦瀛樺埄鐢ㄧ巼
        """
        if not cache or cache_capacity_mb <= 0:
            return 0.0
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                total_used_mb += 1.0  # 鍏煎鏃ф牸寮?
        
        utilization = total_used_mb / cache_capacity_mb
        return min(1.0, max(0.0, utilization))

    # ==================== 鏂板锛氫竴姝ヤ豢鐪熸秹鍙婄殑鏍稿績杈呭姪鍑芥暟 ====================

    def _update_vehicle_positions(self):
        """绠€鍗曟洿鏂拌溅杈嗕綅缃紝妯℃嫙杞﹁締娌夸富骞查亾绉诲姩"""
        for vehicle in self.vehicles:
            position = vehicle.get('position')
            if position is None or len(position) < 2:
                continue

            # === 1) 鏇存柊閫熷害锛堢紦鎱㈠姞鍑忛€?+ 浜ゅ弶鍙ｅ噺閫燂級 ===
            base_speed = float(vehicle.get('velocity', 15.0))
            accel_state = vehicle.setdefault('speed_accel', 0.0)
            accel_state = 0.7 * accel_state + np.random.uniform(-0.4, 0.4)

            # 鍦ㄦ帴杩戣矾鍙ｆ椂闄嶄綆閫熷害锛岄伩鍏嶉珮閫熷啿杩囦氦鍙夊彛
            for intersection in self.intersections.values():
                dist_to_signal = abs(position[0] - intersection['x'])
                if dist_to_signal < 40.0:
                    accel_state = min(accel_state, -0.8)
                    break

            new_speed = np.clip(base_speed + accel_state, 5.0, 32.0)
            vehicle['speed_accel'] = accel_state
            vehicle['velocity'] = new_speed

            # === 2) 鏂瑰悜淇濇寔锛屽悓鏃跺厑璁歌交寰姈鍔?===
            direction = vehicle.get('direction', 0.0)
            heading_jitter = vehicle.setdefault('heading_jitter', 0.0)
            heading_jitter = 0.6 * heading_jitter + np.random.uniform(-0.01, 0.01)
            direction = (direction + heading_jitter) % (2 * np.pi)
            vehicle['direction'] = direction
            vehicle['heading_jitter'] = heading_jitter

            dx = np.cos(direction) * new_speed * self.time_slot
            dy = np.sin(direction) * new_speed * self.time_slot

            # === 3) 渚у悜婕傜Щ锛堟ā鎷熻交寰崲閬擄級 ===
            lane_bias = vehicle.get('lane_bias', position[1] - self.road_y)
            lane_switch_timer = vehicle.setdefault('lane_switch_timer', np.random.randint(80, 160))
            lane_switch_timer -= 1
            if lane_switch_timer <= 0 and np.random.rand() < 0.1:
                lane_bias = np.clip(lane_bias + np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.5),
                                    -6.0, 6.0)
                lane_switch_timer = np.random.randint(120, 220)
            vehicle['lane_switch_timer'] = lane_switch_timer
            vehicle['lane_bias'] = lane_bias

            lateral_state = vehicle.setdefault('lateral_state', 0.0)
            lateral_state = 0.5 * lateral_state + np.random.uniform(-0.25, 0.25)
            vehicle['lateral_state'] = np.clip(lateral_state, -2.0, 2.0)

            # === 4) 搴旂敤浣嶇疆鏇存柊锛坸 鐜矾锛寉 鍙?lane_bias 涓庢紓绉诲奖鍝嶏級 ===
            new_x = (position[0] + dx) % 1000.0
            baseline_lane_y = float(self.road_y + lane_bias)
            new_y = baseline_lane_y + vehicle['lateral_state']
            new_y = np.clip(new_y, self.road_y - 6.5, self.road_y + 6.5)

            vehicle['position'][0] = new_x
            vehicle['position'][1] = new_y

    def _sample_arrivals(self) -> int:
        """鎸夋硦鏉捐繃绋嬮噰鏍锋瘡杞︽瘡鏃堕殭鐨勪换鍔″埌杈炬暟"""
        lam = max(1e-6, float(self.task_arrival_rate) * float(self.time_slot))
        return int(np.random.poisson(lam))

    def _choose_offload_target(self, actions: Dict, rsu_available: bool, uav_available: bool) -> str:
        """鏍规嵁鏅鸿兘浣撴彁渚涚殑鍋忓ソ閫夋嫨鍗歌浇鐩爣"""
        prefs = actions.get('vehicle_offload_pref') or {}
        probs = np.array([
            max(0.0, float(prefs.get('local', 0.0))),
            max(0.0, float(prefs.get('rsu', 0.0))) if rsu_available else 0.0,
            max(0.0, float(prefs.get('uav', 0.0))) if uav_available else 0.0,
        ], dtype=float)

        if probs.sum() <= 0:
            probs = np.array([
                0.34,
                0.33 if rsu_available else 0.0,
                0.33 if uav_available else 0.0
            ], dtype=float)

        if probs.sum() <= 0:
            return 'local'

        probs = probs / probs.sum()
        target_labels = np.array(['local', 'rsu', 'uav'])
        return str(np.random.choice(target_labels, p=probs))

    def _estimate_remote_work_units(self, task: Dict, node_type: str) -> float:
        """浼拌杩滅▼鑺傜偣鐨勫伐浣滈噺鍗曚綅锛堜緵闃熷垪璋冨害浣跨敤锛?""
        requirement = float(task.get('computation_requirement', 1500.0))
        base_divisor = 1200.0 if node_type == 'RSU' else 1600.0
        work_units = requirement / base_divisor
        return float(np.clip(work_units, 0.5, 12.0))

    def _estimate_local_processing(self, task: Dict, vehicle: Dict) -> Tuple[float, float]:
        """浼拌鏈湴澶勭悊鐨勫欢杩熶笌鑳借€?""
        cpu_freq = 2.5e9
        power = 6.5
        if self.sys_config is not None:
            cpu_freq = getattr(self.sys_config.compute, 'vehicle_cpu_freq', cpu_freq)
            power = getattr(self.sys_config.compute, 'vehicle_static_power', power)
        else:
            cpu_freq = float(self.config.get('vehicle_cpu_freq', cpu_freq))
            power = float(self.config.get('vehicle_static_power', power))

        requirement = float(task.get('computation_requirement', 1500.0)) * 1e6  # cycles
        processing_time = requirement / max(cpu_freq, 1e6)
        processing_time = float(np.clip(processing_time, 0.03, 0.8))
        energy = float(power) * processing_time
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + energy
        return processing_time, energy

    def _estimate_transmission(self, data_size_bytes: float, distance: float, link: str) -> Tuple[float, float]:
        """浼拌涓婁紶鑰楁椂涓庤兘鑰?""
        # 鏈夋晥鍚炲悙閲?(bit/s)
        if link == 'uav':
            base_rate = 45e6
            power_w = 0.12
        else:
            base_rate = 80e6
            power_w = 0.18

        attenuation = 1.0 + max(0.0, distance) / 800.0
        rate = base_rate / attenuation
        delay = (float(data_size_bytes) * 8.0) / max(rate, 1e6)
        delay = float(np.clip(delay, 0.01, 1.2))
        energy = power_w * delay
        return delay, energy

    def _append_active_task(self, task_entry: Dict):
        """灏嗕换鍔¤褰曞姞鍏ユ椿璺冨垪琛?""
        self.active_tasks.append(task_entry)

    def _cleanup_active_tasks(self):
        """绉婚櫎宸茬粡瀹屾垚鎴栦涪寮冪殑浠诲姟"""
        self.active_tasks = [
            task for task in self.active_tasks
            if not task.get('completed') and not task.get('dropped')
        ]

    def _handle_deadlines(self):
        """妫€鏌ラ槦鍒椾换鍔℃槸鍚﹁秴鏈熷苟涓㈠純"""
        for node_list, node_type in ((self.rsus, 'RSU'), (self.uavs, 'UAV')):
            for idx, node in enumerate(node_list):
                queue = node.get('computation_queue', [])
                if not queue:
                    continue

                remaining = []
                drop_stats = self.stats.setdefault('drop_stats', {
                    'total': 0,
                    'wait_time_sum': 0.0,
                    'queue_sum': 0,
                    'by_type': {},
                    'by_scenario': {}
                })
                by_type = drop_stats.setdefault('by_type', {})
                by_scenario = drop_stats.setdefault('by_scenario', {})
                log_interval = self.stats_config.drop_log_interval if getattr(self, 'stats_config', None) else self.config.get('drop_log_interval', 200)
                log_interval = max(1, int(log_interval))
                for task in queue:
                    if self.current_time > task.get('deadline', float('inf')):
                        task['dropped'] = True
                        self.stats['dropped_tasks'] += 1
                        self.stats['dropped_data_bytes'] += float(task.get('data_size_bytes', 0.0))

                        drop_stats['total'] += 1
                        wait_time = max(0.0, self.current_time - task.get('queued_at', task.get('arrival_time', self.current_time)))
                        drop_stats['wait_time_sum'] += wait_time
                        drop_stats['queue_sum'] += len(queue)
                        task_type = task.get('task_type', 'unknown')
                        by_type[task_type] = by_type.get(task_type, 0) + 1
                        scenario_name = task.get('app_scenario', 'unknown')
                        by_scenario[scenario_name] = by_scenario.get(scenario_name, 0) + 1

                        if drop_stats['total'] % log_interval == 0:
                            avg_wait = drop_stats['wait_time_sum'] / max(1, drop_stats['total'])
                            avg_queue = drop_stats['queue_sum'] / max(1, drop_stats['total'])
                            print(
                                f"鈿狅笍 Dropped tasks: {drop_stats['total']} "
                                f"(avg wait {avg_wait:.2f}s, avg queue {avg_queue:.1f}) "
                                f"latest type {task_type}, scenario {scenario_name}"
                            )
                        continue
                    remaining.append(task)
                node['computation_queue'] = remaining

    def _store_in_vehicle_cache(self, vehicle: Dict, content_id: str, size_mb: float,
                                cache_controller: Optional[Any] = None):
        """灏嗗唴瀹规帹閫佸埌杞﹁浇缂撳瓨锛屼娇鐢ㄧ畝鍗昄RU娣樻卑"""
        if size_mb <= 0.0:
            return
        capacity = float(vehicle.get('device_cache_capacity', 32.0))
        if size_mb > capacity:
            return
        cache = vehicle.setdefault('device_cache', {})
        total_used = sum(float(meta.get('size', 0.0) or 0.0) for meta in cache.values())
        if total_used + size_mb > capacity:
            # LRU娣樻卑
            ordered = sorted(cache.items(), key=lambda item: item[1].get('timestamp', 0.0))
            for cid, meta in ordered:
                removed_size = float(meta.get('size', 0.0) or 0.0)
                cache.pop(cid, None)
                total_used -= removed_size
                if cache_controller:
                    cache_controller.cache_stats['evicted_items'] += 1
                if total_used + size_mb <= capacity:
                    break
        if total_used + size_mb > capacity:
            return
        cache[content_id] = {
            'size': size_mb,
            'timestamp': self.current_time,
            'source': 'rsu_push'
        }
        if cache_controller:
            cache_controller.cache_stats['collaborative_writes'] += 1

    def _store_in_neighbor_rsu_cache(self, neighbor: Dict, content_id: str, size_mb: float,
                                     content_meta: Dict, cache_controller: Optional[Any]):
        """灏濊瘯灏嗗唴瀹规帹閫佸埌閭昏繎RSU"""
        if size_mb <= 0.0:
            return
        cache = neighbor.setdefault('cache', {})
        if content_id in cache:
            return
        capacity = neighbor.get('cache_capacity', 1000.0)
        available = self._calculate_available_cache_capacity(cache, capacity)
        cache_snapshot = dict(cache)
        should_store = available >= size_mb
        evictions: List[str] = []
        reason = 'RSU_push_neighbor'
        if cache_controller is not None:
            should_store, reason, evictions = cache_controller.should_cache_content(
                content_id, size_mb, available, cache_snapshot, capacity
            )
        if not should_store:
            return
        for cid in evictions:
            removed = cache.pop(cid, None)
            if removed:
                available += float(removed.get('size', 0.0) or 0.0)
                if cache_controller:
                    cache_controller.cache_stats['evicted_items'] += 1
        if available < size_mb:
            return
        cache[content_id] = {
            'size': size_mb,
            'timestamp': self.current_time,
            'reason': reason,
            'source': content_meta.get('source', 'rsu_hit')
        }
        if cache_controller:
            cache_controller.cache_stats['collaborative_writes'] += 1

    def _propagate_cache_after_hit(self, content_id: str, rsu_node: Dict, agents_actions: Optional[Dict]):
        """RSU鍛戒腑鍚庡悜杞﹁締鍜岄偦杩慠SU鎺ㄩ€佸唴瀹?""
        cache_meta = rsu_node.get('cache', {}).get(content_id)
        if not cache_meta:
            return
        size_mb = float(cache_meta.get('size', 0.0) or self._get_realistic_content_size(content_id))
        cache_controller = None
        if agents_actions:
            cache_controller = agents_actions.get('cache_controller')

        # 鎺ㄩ€佸埌瑕嗙洊鑼冨洿鍐呯殑杞﹁締
        coverage = rsu_node.get('coverage_radius', 300.0)
        for vehicle in self.vehicles:
            distance = self.calculate_distance(vehicle.get('position', np.zeros(2)), rsu_node['position'])
            if distance <= coverage * 0.8:
                self._store_in_vehicle_cache(vehicle, content_id, size_mb, cache_controller)

        # 鎺ㄩ€佸埌閭昏繎RSU
        for neighbor in self.rsus:
            if neighbor is rsu_node:
                continue
            distance = self.calculate_distance(neighbor['position'], rsu_node['position'])
            if distance <= coverage * 1.2:
                self._store_in_neighbor_rsu_cache(neighbor, content_id, size_mb, cache_meta, cache_controller)

    def _dispatch_task(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict):
        """鏍规嵁鍔ㄤ綔鍒嗛厤浠诲姟"""
        cache_controller = None
        if isinstance(actions, dict):
            cache_controller = actions.get('cache_controller')
        if cache_controller is None:
            cache_controller = getattr(self, 'adaptive_cache_controller', None)

        content_id = task.get('content_id')
        vehicle_cache = vehicle.setdefault('device_cache', {})
        if content_id and content_id in vehicle_cache:
            vehicle_cache[content_id]['timestamp'] = self.current_time
            local_delay = 0.02
            local_energy = 0.0
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self.stats['total_delay'] += local_delay
            self.stats['total_energy'] += local_energy
            self.stats['cache_hits'] += 1
            self.stats['local_cache_hits'] = self.stats.get('local_cache_hits', 0) + 1
            vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + local_energy
            step_summary['local_cache_hits'] = step_summary.get('local_cache_hits', 0) + 1
            if cache_controller is not None:
                cache_controller.record_cache_result(content_id, True)
                cache_controller.update_content_heat(content_id)
            return

        rsu_available = len(self.rsus) > 0
        uav_available = len(self.uavs) > 0
        target = self._choose_offload_target(actions, rsu_available, uav_available)

        assigned = False
        if target == 'rsu' and rsu_available:
            assigned = self._assign_to_rsu(vehicle, task, actions, step_summary)
        elif target == 'uav' and uav_available:
            assigned = self._assign_to_uav(vehicle, task, actions, step_summary)

        if not assigned:
            self._handle_local_processing(vehicle, task, step_summary)

    def _assign_to_rsu(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict) -> bool:
        """鍒嗛厤鑷砇SU"""
        if not self.rsus:
            return False

        vehicle_pos = np.array(vehicle.get('position', [0.0, 0.0]))
        distances = []
        in_range_mask = []
        for rsu in self.rsus:
            dist = self.calculate_distance(vehicle_pos, rsu['position'])
            distances.append(dist)
            in_range_mask.append(1.0 if dist <= rsu.get('coverage_radius', 300.0) else 0.0)

        accessible = np.array(in_range_mask, dtype=float)
        if accessible.sum() == 0:
            # 娌℃湁瑕嗙洊鐨凴SU
            return False

        probs = np.ones(len(self.rsus), dtype=float)
        rsu_pref = actions.get('rsu_selection_probs')
        if isinstance(rsu_pref, (list, tuple, np.ndarray)) and len(rsu_pref) == len(self.rsus):
            probs = np.array([max(0.0, float(v)) for v in rsu_pref], dtype=float)

        weights = probs * accessible
        if weights.sum() <= 0:
            weights = accessible

        weights = weights / weights.sum()
        rsu_idx = int(np.random.choice(np.arange(len(self.rsus)), p=weights))
        distance = distances[rsu_idx]
        node = self.rsus[rsu_idx]
        success = self._handle_remote_assignment(vehicle, task, node, 'RSU', rsu_idx, distance, actions, step_summary)
        if success:
            step_summary['remote_tasks'] += 1
        return success

    def _assign_to_uav(self, vehicle: Dict, task: Dict, actions: Dict, step_summary: Dict) -> bool:
        """鍒嗛厤鑷砋AV"""
        if not self.uavs:
            return False

        vehicle_pos = np.array(vehicle.get('position', [0.0, 0.0]))
        distances = []
        in_range_mask = []
        for uav in self.uavs:
            dist = self.calculate_distance(vehicle_pos, uav['position'])
            distances.append(dist)
            in_range_mask.append(1.0 if dist <= uav.get('coverage_radius', 350.0) else 0.0)

        accessible = np.array(in_range_mask, dtype=float)
        if accessible.sum() == 0:
            return False

        probs = np.ones(len(self.uavs), dtype=float)
        uav_pref = actions.get('uav_selection_probs')
        if isinstance(uav_pref, (list, tuple, np.ndarray)) and len(uav_pref) == len(self.uavs):
            probs = np.array([max(0.0, float(v)) for v in uav_pref], dtype=float)

        weights = probs * accessible
        if weights.sum() <= 0:
            weights = accessible

        weights = weights / weights.sum()
        uav_idx = int(np.random.choice(np.arange(len(self.uavs)), p=weights))
        distance = distances[uav_idx]
        node = self.uavs[uav_idx]
        success = self._handle_remote_assignment(vehicle, task, node, 'UAV', uav_idx, distance, actions, step_summary)
        if success:
            step_summary['remote_tasks'] += 1
        return success

    def _handle_remote_assignment(
        self,
        vehicle: Dict,
        task: Dict,
        node: Dict,
        node_type: str,
        node_idx: int,
        distance: float,
        actions: Dict,
        step_summary: Dict
    ) -> bool:
        """鎵ц杩滅▼鍗歌浇锛氱紦瀛樺垽瀹氥€佸缓绔嬮槦鍒楀苟璁板綍缁熻"""
        actions = actions or {}
        cache_hit = False

        if node_type == 'RSU':
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='RSU')
        else:
            cache_hit = self.check_cache_hit_adaptive(task['content_id'], node, actions, node_type='UAV')

        if cache_hit:
            # 缂撳瓨鍛戒腑锛氬揩閫熷畬鎴?
            delay = max(0.02, 0.2 * self.time_slot)
            power = 18.0 if node_type == 'RSU' else 12.0
            energy = power * delay * 0.1
            self.stats['processed_tasks'] += 1
            self.stats['completed_tasks'] += 1
            self.stats['total_delay'] += delay
            self.stats['total_energy'] += energy
            node['energy_consumed'] = node.get('energy_consumed', 0.0) + energy
            return True

        upload_delay, upload_energy = self._estimate_transmission(task.get('data_size_bytes', 1e6), distance, node_type.lower())
        self.stats['total_delay'] += upload_delay
        self.stats['energy_uplink'] += upload_energy
        self.stats['total_energy'] += upload_energy
        vehicle['energy_consumed'] = vehicle.get('energy_consumed', 0.0) + upload_energy

        work_units = self._estimate_remote_work_units(task, node_type)
        task_entry = {
            'id': task['id'],
            'vehicle_id': task['vehicle_id'],
            'arrival_time': self.current_time + upload_delay,
            'deadline': task['deadline'],
            'data_size': task.get('data_size', 1.0),
            'data_size_bytes': task.get('data_size_bytes', 1e6),
            'content_id': task.get('content_id'),
            'computation_requirement': task.get('computation_requirement', 1500.0),
            'work_remaining': work_units,
            'queued_at': self.current_time,
            'node_type': node_type,
            'node_idx': node_idx,
            'upload_delay': upload_delay,
            'priority': task.get('priority', 0.5),
            'task_type': task.get('task_type'),
            'app_scenario': task.get('app_scenario'),
            'deadline_relax_factor': task.get('deadline_relax_factor', 1.0)
        }

        queue = node.setdefault('computation_queue', [])
        queue.append(task_entry)
        self._append_active_task(task_entry)
        return True

    def _handle_local_processing(self, vehicle: Dict, task: Dict, step_summary: Dict):
        """鏈湴澶勭悊浠诲姟"""
        processing_delay, energy = self._estimate_local_processing(task, vehicle)
        self.stats['processed_tasks'] += 1
        self.stats['completed_tasks'] += 1
        self.stats['total_delay'] += processing_delay
        self.stats['total_energy'] += energy
        step_summary['local_tasks'] += 1

    
    def check_adaptive_migration(self, agents_actions: Dict = None):
        """馃幆 澶氱淮搴︽櫤鑳借縼绉绘鏌?(闃堝€艰Е鍙?璐熻浇宸Е鍙?璺熼殢杩佺Щ)"""
        if not agents_actions or 'migration_controller' not in agents_actions:
            return
        
        migration_controller = agents_actions['migration_controller']
        
        hotspot_map: Dict[str, float] = {}
        collaborative_system = getattr(self, 'collaborative_cache', None)
        if collaborative_system is not None and hasattr(collaborative_system, 'get_hotspot_intensity'):
            try:
                hotspot_map = collaborative_system.get_hotspot_intensity()
            except Exception:
                hotspot_map = {}
        
        # 馃攳 鏀堕泦鎵€鏈夎妭鐐圭姸鎬佺敤浜庨偦灞呮瘮杈?
        all_node_states = {}
        
        # RSU鐘舵€佹敹闆?
        for i, rsu in enumerate(self.rsus):
            queue = rsu.get('computation_queue', [])
            queue_len = len(queue)
            cache_capacity = rsu.get('cache_capacity', 1000.0)
            available_cache = self._calculate_available_cache_capacity(rsu.get('cache', {}), cache_capacity)
            storage_load = 0.0 if cache_capacity <= 0 else 1.0 - (available_cache / max(1.0, cache_capacity))
            total_data = sum(task.get('data_size', 1.0) for task in queue)
            bandwidth_capacity = rsu.get('bandwidth_capacity', 50.0)
            bandwidth_load = float(np.clip(total_data / max(1.0, bandwidth_capacity), 0.0, 0.99))
            cpu_load = float(np.clip(queue_len / 10.0, 0.0, 0.99))

            all_node_states[f'rsu_{i}'] = {
                'cpu_load': cpu_load,
                'bandwidth_load': bandwidth_load,
                'storage_load': float(np.clip(storage_load, 0.0, 0.99)),
                'load_factor': self._calculate_enhanced_load_factor(rsu, 'RSU'),
                'battery_level': 1.0,
                'node_type': 'RSU',
                'queue_length': queue_len,
                'cache_capacity': cache_capacity,
                'cache_available': available_cache,
                'hotspot_intensity': float(np.clip(hotspot_map.get(f'RSU_{i}', 0.0), 0.0, 1.0))
            }

        # UAV鐘舵€佹敹闆?
        for i, uav in enumerate(self.uavs):
            queue = uav.get('computation_queue', [])
            queue_len = len(queue)
            cache_capacity = uav.get('cache_capacity', 200.0)
            available_cache = self._calculate_available_cache_capacity(uav.get('cache', {}), cache_capacity)
            storage_load = 0.0 if cache_capacity <= 0 else 1.0 - (available_cache / max(1.0, cache_capacity))
            total_data = sum(task.get('data_size', 1.0) for task in queue)
            bandwidth_capacity = uav.get('bandwidth_capacity', 15.0)
            bandwidth_load = float(np.clip(total_data / max(1.0, bandwidth_capacity), 0.0, 0.99))
            cpu_load = float(np.clip(queue_len / 12.0, 0.0, 0.99))

            all_node_states[f'uav_{i}'] = {
                'cpu_load': cpu_load,
                'bandwidth_load': bandwidth_load,
                'storage_load': float(np.clip(storage_load, 0.0, 0.99)),
                'load_factor': self._calculate_enhanced_load_factor(uav, 'UAV'),
                'battery_level': uav.get('battery_level', 1.0),
                'node_type': 'UAV',
                'queue_length': queue_len,
                'cache_capacity': cache_capacity,
                'cache_available': available_cache,
                'hotspot_intensity': 0.0
            }
        
        # 馃彚 RSU杩佺Щ妫€鏌?(闃堝€?璐熻浇宸Е鍙?
        for i, rsu in enumerate(self.rsus):
            node_id = f'rsu_{i}'
            current_state = all_node_states[node_id]
            
            # 鏇存柊璐熻浇鍘嗗彶
            migration_controller.update_node_load(node_id, current_state['load_factor'])
            
            # 馃幆 澶氱淮搴﹁縼绉昏Е鍙戞鏌?
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"馃幆 {node_id} 瑙﹀彂杩佺Щ: {reason} (绱ф€ュ害:{urgency:.3f})")
                
                # 鎵цRSU闂磋縼绉?
                result = self.execute_rsu_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
        # 馃殎 UAV杩佺Щ妫€鏌?
        for i, uav in enumerate(self.uavs):
            node_id = f'uav_{i}'
            current_state = all_node_states[node_id]
            
            # 鏇存柊璐熻浇鍘嗗彶
            migration_controller.update_node_load(node_id, current_state['load_factor'], current_state['battery_level'])
            
            # 馃幆 澶氱淮搴﹁縼绉昏Е鍙戞鏌?
            should_migrate, reason, urgency = migration_controller.should_trigger_migration(
                node_id, current_state, all_node_states
            )
            
            if should_migrate:
                self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                print(f"馃幆 {node_id} 瑙﹀彂杩佺Щ: {reason} (绱ф€ュ害:{urgency:.3f})")
                
                # UAV杩佺Щ鍒癛SU
                result = self.execute_uav_migration(i, urgency)
                if result.get('success'):
                    self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                    migration_controller.record_migration_result(True, cost=result.get('cost', 0.0), delay_saved=result.get('delay_saved', 0.0))
                else:
                    migration_controller.record_migration_result(False)
        
        # 馃殫 杞﹁締璺熼殢杩佺Щ妫€鏌?
        self._check_vehicle_handover_migration(migration_controller)
    
    def _check_vehicle_handover_migration(self, migration_controller):
        """馃殫 杞﹁締璺熼殢杩佺Щ锛氬綋杞﹁締绉诲姩瓒呭嚭褰撳墠杈圭紭鑺傜偣閫氫俊瑕嗙洊鏃惰Е鍙戣縼绉?""
        handover_count = 0
        
        # 妫€鏌ユ瘡涓椿璺冧换鍔＄殑杞﹁締浣嶇疆
        for task in list(self.active_tasks):
            if task.get('node_type') not in ['RSU', 'UAV']:
                continue  # 鍙鏌ヨ竟缂樿妭鐐逛换鍔?
            
            try:
                # 鎵惧埌瀵瑰簲杞﹁締
                vehicle = next(v for v in self.vehicles if v['id'] == task['vehicle_id'])
                current_pos = vehicle['position']
                
                # 鑾峰彇褰撳墠鏈嶅姟鑺傜偣
                current_node = None
                if task['node_type'] == 'RSU' and task.get('node_idx') is not None:
                    current_node = self.rsus[task['node_idx']]
                elif task['node_type'] == 'UAV' and task.get('node_idx') is not None:
                    current_node = self.uavs[task['node_idx']]
                
                if current_node is None:
                    continue
                
                # 馃攳 妫€鏌ラ€氫俊瑕嗙洊鍜岃窡闅忚縼绉昏Е鍙?
                distance_to_current = self.calculate_distance(current_pos, current_node['position'])
                coverage_radius = current_node.get('coverage_radius', 500.0)  # 榛樿500m瑕嗙洊
                
                # 馃敡 鏅鸿兘璺熼殢杩佺Щ瑙﹀彂鏈哄埗锛?
                # 1. 鍩虹闃堝€硷細85%瑕嗙洊鍗婂緞锛堜俊鍙疯川閲忓紑濮嬫槑鏄句笅闄嶏級
                # 2. 鑰冭檻杞﹁締閫熷害锛氶珮閫熻溅杈嗘彁鍓嶈Е鍙?
                # 3. 鑰冭檻棰勬祴锛氳溅杈嗘槸鍚﹀湪蹇€熻繙绂诲綋鍓嶈妭鐐?
                
                vehicle_speed = np.linalg.norm(vehicle.get('velocity', [0, 0]))
                
                # 馃敡 浼樺寲鐨勯€熷害璋冩暣鍥犲瓙锛氶€熷害瓒婂揩锛岃秺鏃╄Е鍙?
                # 30 m/s 鈫?0.85 (425m瑙﹀彂)
                # 45 m/s 鈫?0.775 (387m瑙﹀彂)  
                # 60 m/s 鈫?0.70 (350m瑙﹀彂)
                speed_factor = max(0.70, 1.0 - (vehicle_speed / 200.0))
                
                # 鍔ㄦ€佽Е鍙戦槇鍊?
                trigger_threshold = coverage_radius * speed_factor
                
                # 瓒呭嚭鍔ㄦ€侀槇鍊硷紝瑙﹀彂璺熼殢杩佺Щ
                if distance_to_current > trigger_threshold:
                    # 馃攳 瀵绘壘鏈€浣虫柊鏈嶅姟鑺傜偣
                    best_new_node = None
                    best_distance = float('inf')
                    best_node_idx = None
                    best_node_type = None
                    
                    # 妫€鏌ユ墍鏈塕SU - 浼樺厛閫夋嫨RSU锛堢ǔ瀹氭€ф洿濂斤級
                    for i, rsu in enumerate(self.rsus):
                        dist = self.calculate_distance(current_pos, rsu['position'])
                        if dist <= rsu.get('coverage_radius', 500.0):
                            queue_len = len(rsu.get('computation_queue', []))
                            cpu_load = rsu.get('cpu_usage', 0.5)
                            
                            # 馃敡 缁煎悎璇勫垎锛氳窛绂?+ 闃熷垪 + 璐熻浇
                            score = dist * 1.0 + queue_len * 30 + cpu_load * 200
                            
                            if score < best_distance:
                                best_new_node = rsu
                                best_distance = score
                                best_node_idx = i
                                best_node_type = 'RSU'
                    
                    # 妫€鏌ユ墍鏈塙AV锛堜綔涓哄閫夛級
                    if best_new_node is None or best_distance > 500:  # RSU涓嶇悊鎯虫椂鑰冭檻UAV
                        for i, uav in enumerate(self.uavs):
                            dist = self.calculate_distance(current_pos, uav['position'])
                            if dist <= uav.get('coverage_radius', 300.0):
                                queue_len = len(uav.get('computation_queue', []))
                                cpu_load = uav.get('cpu_usage', 0.5)
                                
                                # UAV璇勫垎鐣ユ湁涓嶅悓锛堣€冭檻绉诲姩鎬э級
                                score = dist * 1.2 + queue_len * 20 + cpu_load * 150
                                
                                if score < best_distance:
                                    best_new_node = uav
                                    best_distance = score
                                    best_node_idx = i
                                    best_node_type = 'UAV'
                    
                    # 馃殌 鎵ц璺熼殢杩佺Щ锛堝彧鍦ㄦ壘鍒版槑鏄炬洿濂界殑鑺傜偣鏃讹級
                    # 蹇呴』婊¤冻锛?) 鎵惧埌鏂拌妭鐐? 2) 鏂拌妭鐐逛笉鍚? 3) 鏂拌妭鐐规槑鏄炬洿浼?
                    current_queue = len(current_node.get('computation_queue', []))
                    current_score = distance_to_current * 1.0 + current_queue * 30
                    origin_queue_before = current_queue
                    
                    should_migrate = (
                        best_new_node is not None and 
                        (best_node_idx != task.get('node_idx') or best_node_type != task['node_type']) and
                        best_distance < current_score * 0.7  # 鏂拌妭鐐硅嚦灏戝ソ30%鎵嶈縼绉?
                    )
                    
                    if should_migrate:
                        origin_node_type = task['node_type']
                        origin_node_idx = task.get('node_idx')
                        
                        # 浠庡師鑺傜偣绉婚櫎浠诲姟
                        if origin_node_type == 'RSU':
                            old_queue = self.rsus[origin_node_idx].get('computation_queue', [])
                            updated_queue = [t for t in old_queue if t.get('id') != task['id']]
                            self.rsus[origin_node_idx]['computation_queue'] = updated_queue
                            current_node = self.rsus[origin_node_idx]
                            origin_queue_after = len(updated_queue)
                        elif origin_node_type == 'UAV':
                            old_queue = self.uavs[origin_node_idx].get('computation_queue', [])
                            updated_queue = [t for t in old_queue if t.get('id') != task['id']]
                            self.uavs[origin_node_idx]['computation_queue'] = updated_queue
                            current_node = self.uavs[origin_node_idx]
                            origin_queue_after = len(updated_queue)
                        else:
                            origin_queue_after = origin_queue_before
                        
                        # 娣诲姞鍒版柊鑺傜偣
                        if 'computation_queue' not in best_new_node:
                            best_new_node['computation_queue'] = []
                        target_queue_before = len(best_new_node['computation_queue'])
                        
                        # 鍒涘缓鏂颁换鍔￠」
                        migrated_task = {
                            'id': task['id'],
                            'vehicle_id': task['vehicle_id'],
                            'arrival_time': task['arrival_time'],
                            'deadline': task['deadline'],
                            'data_size': task.get('data_size', 2.0),
                            'computation_requirement': task.get('computation_requirement', 1000),
                            'content_id': task['content_id'],
                            'compute_time_needed': task.get('compute_time_needed', 1.0),
                            'work_remaining': task.get('work_remaining', 0.5),
                            'cache_hit': task.get('cache_hit', False),
                            'queued_at': self.current_time,
                            'migrated_from': f"{task['node_type']}_{task.get('node_idx')}",
                            'task_type': task.get('task_type'),
                            'app_scenario': task.get('app_scenario'),
                            'deadline_relax_factor': task.get('deadline_relax_factor', 1.0)
                        }
                        best_new_node['computation_queue'].append(migrated_task)
                        target_queue_after = len(best_new_node['computation_queue'])
                        
                        # 鏇存柊浠诲姟淇℃伅
                        task['node_type'] = best_node_type
                        task['node_idx'] = best_node_idx
                        
                        handover_count += 1
                        migration_label = handover_count
                        
                        # 馃敡 澧炲己鏃ュ織锛氭樉绀鸿Е鍙戝師鍥犲拰杩佺Щ鏀剁泭
                        print(
                            f"馃殫 杞﹁締璺熼殢杩佺Щ[{migration_label}]: 杞﹁締 {task['vehicle_id']} 浠诲姟 {task['id']} "
                            f"浠?{origin_node_type}_{origin_node_idx} 鈫?{best_node_type}_{best_node_idx}"
                        )
                        print(f"   瑙﹀彂鍘熷洜: 璺濈{distance_to_current:.1f}m > 闃堝€納trigger_threshold:.1f}m (杞﹂€焮vehicle_speed:.1f}m/s)")
                        print(f"   杩佺Щ鏀剁泭: 褰撳墠璇勫垎{current_score:.1f} 鈫?鏂拌瘎鍒唟best_distance:.1f} (鏀瑰杽{(1-best_distance/current_score)*100:.1f}%)")
                        print(
                            f"   闃熷垪瓒嬪娍: {origin_node_type}_{origin_node_idx}: {origin_queue_before} 鈫?{origin_queue_after}, "
                            f"{best_node_type}_{best_node_idx}: {target_queue_before} 鈫?{target_queue_after}"
                        )
                        
                        # 璁板綍璺熼殢杩佺Щ缁熻
                        self.stats['migrations_executed'] = self.stats.get('migrations_executed', 0) + 1
                        self.stats['migrations_successful'] = self.stats.get('migrations_successful', 0) + 1
                        self.stats['handover_migrations'] = self.stats.get('handover_migrations', 0) + 1
                        migration_controller.record_migration_result(True, cost=5.0, delay_saved=0.3)
                
            except Exception as e:
                continue  # 蹇界暐閿欒锛岀户缁鐞嗕笅涓€涓换鍔?
        
        if handover_count > 0:
            print(f"馃殫 鏈椂闅欐墽琛屼簡 {handover_count} 娆¤溅杈嗚窡闅忚縼绉?)
    
    def run_simulation_step(self, step: int, actions: Optional[Dict] = None) -> Dict[str, Any]:
        """鎵ц鍗曚釜浠跨湡姝ワ紝杩斿洖鎴嚦褰撳墠鐨勭疮璁＄粺璁℃暟鎹?""
        actions = actions or {}

        advance_simulation_time()
        self.current_step += 1
        self.current_time = get_simulation_time()

        step_summary = {
            'generated_tasks': 0,
            'local_tasks': 0,
            'remote_tasks': 0,
            'local_cache_hits': 0
        }

        # 1. 鏇存柊杞﹁締浣嶇疆
        self._update_vehicle_positions()

        # 2. 鐢熸垚浠诲姟骞跺垎閰?
        for vehicle in self.vehicles:
            arrivals = self._sample_arrivals()
            if arrivals <= 0:
                continue

            vehicle_id = vehicle['id']
            for _ in range(arrivals):
                task = self.generate_task(vehicle_id)
                step_summary['generated_tasks'] += 1
                self.stats['total_tasks'] += 1
                self.stats['generated_data_bytes'] += float(task.get('data_size_bytes', 0.0))
                self._dispatch_task(vehicle, task, actions, step_summary)

        # 3. 鏅鸿兘杩佺Щ绛栫暐
        if actions:
            self.check_adaptive_migration(actions)

        # 4. 澶勭悊闃熷垪涓殑浠诲姟
        self._process_node_queues()

        # 5. 妫€鏌ヨ秴鏃跺苟娓呯悊
        self._handle_deadlines()
        self._cleanup_active_tasks()

        # 姹囨€讳俊鎭?
        step_summary.update({
            'current_time': self.current_time,
            'rsu_queue_lengths': [len(rsu.get('computation_queue', [])) for rsu in self.rsus],
            'uav_queue_lengths': [len(uav.get('computation_queue', [])) for uav in self.uavs],
            'active_tasks': len(self.active_tasks)
        })

        step_summary.update(self._summarize_task_types())

        cumulative_stats = dict(self.stats)
        cumulative_stats.update(step_summary)
        return cumulative_stats
    
    def execute_rsu_migration(self, source_rsu_idx: int, urgency: float) -> Dict[str, float]:
        """Execute RSU-to-RSU migration and return cost/delay metrics."""
        source_rsu = self.rsus[source_rsu_idx]
        source_queue = source_rsu.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        candidates = []
        for i, rsu in enumerate(self.rsus):
            if i == source_rsu_idx:
                continue
            queue_len = len(rsu.get('computation_queue', []))
            cpu_load = min(0.99, queue_len / 10.0)
            score = queue_len + cpu_load * 10.0
            candidates.append((i, queue_len, cpu_load, score))

        if not candidates:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_idx, target_queue_len, target_cpu_load, _ = min(candidates, key=lambda x: x[3])
        source_queue_len = len(source_queue)
        queue_diff = target_queue_len - source_queue_len

        all_queue_lens = [len(rsu.get('computation_queue', [])) for rsu in self.rsus]
        system_queue_variance = np.var(all_queue_lens)
        if system_queue_variance > 50:
            migration_tolerance = 8
        elif system_queue_variance > 20:
            migration_tolerance = 5
        else:
            migration_tolerance = 3
        if queue_diff > migration_tolerance:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        migration_ratio = max(0.1, min(0.5, urgency))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        source_rsu_id = source_rsu['id']
        target_rsu_id = target_rsu['id']
        avg_task_size = 2.0
        total_data_size = tasks_to_migrate * avg_task_size

        migrated_tasks = source_queue[:tasks_to_migrate]
        source_rsu['computation_queue'] = source_queue[tasks_to_migrate:]
        target_rsu['computation_queue'].extend(migrated_tasks)

        delay_saved = max(0.0, (source_queue_len - target_queue_len) * self.time_slot)
        migration_cost = 0.0
        try:
            from utils.wired_backhaul_model import calculate_rsu_to_rsu_delay, calculate_rsu_to_rsu_energy
            wired_delay = calculate_rsu_to_rsu_delay(total_data_size, source_rsu_id, target_rsu_id)
            wired_energy = calculate_rsu_to_rsu_energy(total_data_size, source_rsu_id, target_rsu_id, wired_delay)
            self.stats['rsu_migration_delay'] = self.stats.get('rsu_migration_delay', 0.0) + wired_delay
            self.stats['rsu_migration_energy'] = self.stats.get('rsu_migration_energy', 0.0) + wired_energy
            self.stats['rsu_migration_data'] = self.stats.get('rsu_migration_data', 0.0) + total_data_size
            migration_cost = wired_energy + wired_delay * 1000.0
        except Exception:
            migration_cost = total_data_size * 0.2

        return {'success': True, 'cost': migration_cost, 'delay_saved': delay_saved}
    def execute_uav_migration(self, source_uav_idx: int, urgency: float) -> Dict[str, float]:
        """Execute UAV-to-RSU migration and return cost/delay metrics."""
        source_uav = self.uavs[source_uav_idx]
        source_queue = source_uav.get('computation_queue', [])
        if not source_queue:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        uav_position = source_uav['position']
        candidates = []
        for i, rsu in enumerate(self.rsus):
            queue_len = len(rsu.get('computation_queue', []))
            distance = self.calculate_distance(uav_position, rsu['position'])
            cpu_load = min(0.99, queue_len / 10.0)
            score = distance * 0.01 + queue_len + cpu_load * 10.0
            candidates.append((i, queue_len, cpu_load, distance, score))

        if not candidates:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        target_idx, target_queue_len, target_cpu_load, distance, _ = min(candidates, key=lambda x: x[4])
        target_rsu = self.rsus[target_idx]
        if 'computation_queue' not in target_rsu:
            target_rsu['computation_queue'] = []

        source_queue_len = len(source_queue)
        migration_ratio = max(0.2, min(0.6, urgency + 0.1))
        tasks_to_migrate = max(1, int(source_queue_len * migration_ratio))
        tasks_to_migrate = min(tasks_to_migrate, source_queue_len)
        if tasks_to_migrate <= 0:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        base_success_rate = 0.75
        distance_penalty = min(0.35, distance / 1200.0)
        load_penalty = min(0.25, target_queue_len / 16.0)
        urgency_bonus = min(0.2, urgency)
        actual_success_rate = np.clip(base_success_rate - distance_penalty - load_penalty + urgency_bonus, 0.35, 0.95)
        if np.random.random() > actual_success_rate:
            return {'success': False, 'cost': 0.0, 'delay_saved': 0.0}

        migrated_tasks = source_queue[:tasks_to_migrate]
        source_uav['computation_queue'] = source_queue[tasks_to_migrate:]
        target_rsu['computation_queue'].extend(migrated_tasks)

        total_data_size = sum(task.get('data_size', 1.0) for task in migrated_tasks) or (tasks_to_migrate * 1.0)
        # Estimate wireless transfer characteristics
        wireless_rate = 12.0  # MB/s
        wireless_delay = (total_data_size / wireless_rate)
        wireless_energy = total_data_size * 0.15 + distance * 0.01
        delay_saved = max(0.0, (source_queue_len - target_queue_len) * self.time_slot)

        self.stats['uav_migration_distance'] = self.stats.get('uav_migration_distance', 0.0) + distance
        self.stats['uav_migration_count'] = self.stats.get('uav_migration_count', 0) + 1

        migration_cost = wireless_energy + wireless_delay * 800.0
        return {'success': True, 'cost': migration_cost, 'delay_saved': delay_saved}

