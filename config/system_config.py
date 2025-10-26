#!/usr/bin/env python3
"""
绯荤粺閰嶇疆
"""

import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List


@dataclass(frozen=True)
class TaskProfileSpec:
    """鎻忚堪鍗曠被浠诲姟鐨勬暟鎹寖鍥淬€佽绠楀瘑搴﹀強寤惰繜灞炴€?""
    task_type: int
    data_range: Tuple[float, float]
    compute_density: float
    max_latency_slots: int
    latency_weight: float


@dataclass(frozen=True)
class TaskScenarioSpec:
    """搴旂敤鍦烘櫙鍙婂叾瀵瑰簲鐨勪换鍔＄被鍨嬩笌棰濆鍙傛暟"""
    name: str
    min_deadline: float
    max_deadline: float
    task_type: int
    relax_factor: float
    weight: float

class ExperimentConfig:
    """瀹為獙閰嶇疆绫?""
    
    def __init__(self):
        self.num_episodes = 1000
        self.num_runs = 3
        self.save_interval = 100
        self.eval_interval = 50
        self.log_interval = 10
        self.max_steps_per_episode = 200
        self.warmup_episodes = 10
        self.use_timestamp = True
        self.timestamp_format = "%Y%m%d_%H%M%S"

class RLConfig:
    """寮哄寲瀛︿範閰嶇疆绫?""
    
    def __init__(self):
        self.num_agents = 3
        self.state_dim = 20
        self.action_dim = 10
        self.hidden_dim = 256
        self.lr = 0.0003
        self.actor_lr = 0.0003
        self.critic_lr = 0.0003
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.memory_size = 100000
        self.noise_std = 0.05          # 闄嶄綆鍣０鏍囧噯宸?
        self.policy_delay = 2
        self.noise_clip = 0.3           # 闄嶄綆鍣０瑁佸壀
        self.exploration_noise = 0.05   # 闄嶄綆鎺㈢储鍣０
        self.policy_noise = 0.1         # 闄嶄綆绛栫暐鍣０
        self.target_noise = 0.1         # 闄嶄綆鐩爣鍣０
        self.update_freq = 1
        self.buffer_size = 100000
        self.warmup_steps = 1000
        
        # 馃幆 鏍稿績濂栧姳鏉冮噸锛堢粺涓€濂栧姳鍑芥暟锛?
        # Objective = 蠅_T 脳 鏃跺欢 + 蠅_E 脳 鑳借€?
        self.reward_weight_delay = 2.0     # 蠅_T: 鏃跺欢鏉冮噸锛堢洰鏍団増0.4s锛?        self.reward_weight_energy = 1.2    # 蠅_E: 鑳借€楁潈閲?
        self.reward_penalty_dropped = 0.02 # 杞诲井鎯╃綒锛堜繚璇佸畬鎴愮巼绾︽潫锛?
        
        # 鉂?宸插純鐢ㄥ弬鏁帮紙淇濈暀浠ュ吋瀹规棫浠ｇ爜锛?
        self.reward_weight_loss = 0.0      # 宸茬Щ闄わ細data_loss鏄椂寤剁殑琛嶇敓鎸囨爣
        self.reward_weight_completion = 0.0  # 宸查泦鎴愬埌dropped_penalty
        self.reward_weight_cache = 0.3       # 缂撳瓨鍛戒腑鐜?/ 娣樻卑鎴愭湰鏉冮噸
        self.reward_weight_migration = 0.2   # 杩佺Щ鏀剁泭 / 鎴愭湰鏉冮噸

        # 馃幆 寤舵椂-鑳借€椾紭鍖栫洰鏍囬槇鍊硷紙渚涚畻娉曞姩鎬佽皟鏁达級
        self.latency_target = 0.40          # 鐩爣骞冲潎寤舵椂锛堢锛?        self.latency_upper_tolerance = 0.80 # 瓒呰繃姝ゅ€艰Е鍙戝己鍖栨儵缃?        self.energy_target = 1200.0         # 鐩爣鑳借€楋紙鐒﹁€筹級
        self.energy_upper_tolerance = 1800.0# 瓒呰繃姝ゅ€艰Е鍙戝己鍖栨儵缃?
class QueueConfig:
    """闃熷垪閰嶇疆绫?""
    
    def __init__(self):
        # 涓?鈥?涓椂闅欏悓姝ワ細鐢熷懡鍛ㄦ湡鏍煎彇4
        self.max_lifetime = 4
        self.max_queue_size = 100
        self.priority_levels = 4
        # 鑰佸寲鍥犲瓙閫傞厤鐭椂闅欙紙姣忔鏄捐憲琛板噺锛?        self.aging_factor = 0.25

class TaskConfig:
    """浠诲姟閰嶇疆绫?""
    
    def __init__(self):
        self.num_priority_levels = 4
        self.task_compute_density = 120  # cycles/bit锛屼綔涓虹己鐪佸€?        self.arrival_rate = 2.5   # tasks/second - 馃殌 12杞﹁締鏋侀珮璐熻浇浼樺寲
        
        # 馃敡 閲嶆柊璁捐锛氫换鍔″弬鏁?- 鍒嗗眰璁捐涓嶅悓澶嶆潅搴︿换鍔?
        self.data_size_range = (0.5e6/8, 15e6/8)  # 0.5-15 Mbits = 0.0625-1.875 MB
        self.task_data_size_range = self.data_size_range  # 鍏煎鎬у埆鍚?

        # 璁＄畻鍛ㄦ湡閰嶇疆 (鑷姩璁＄畻锛岀‘淇濅竴鑷存€?
        self.compute_cycles_range = (1e8, 1e10)  # cycles
        
        # 鎴鏃堕棿閰嶇疆
        self.deadline_range = (0.2, 0.8)  # seconds锛屽搴?-4涓椂闅?        
        # 杈撳嚭姣斾緥閰嶇疆
        self.task_output_ratio = 0.05  # 杈撳嚭澶у皬鏄緭鍏ュぇ灏忕殑5%
        
        # 馃敡 閲嶆柊璁捐锛氫换鍔＄被鍨嬮槇鍊?- 鍩轰簬12GHz RSU瀹為檯澶勭悊鑳藉姏
        self.delay_thresholds = {
            'extremely_sensitive': 1,    # 蟿鈧?= 1 涓椂闅?= 0.2s
            'sensitive': 2,              # 蟿鈧?= 2 涓椂闅?= 0.4s
            'moderately_tolerant': 3,    # 蟿鈧?= 3 涓椂闅?= 0.6s
        }

        # 寤惰繜鎴愭湰鏉冮噸锛堝弬鑰冭鏂囪〃IV锛?        self.latency_cost_weights = {
            1: 1.0,
            2: 0.4,
            3: 0.4,
            4: 0.4,
        }

        # Deadline 鏀炬澗鍙傛暟
        self.deadline_relax_default = 1.0
        self.deadline_relax_fallback = 1.0

        # 浠诲姟绫诲瀷鐗瑰寲鍙傛暟锛圖ataclass褰㈠紡锛?        self.task_profiles: Dict[int, TaskProfileSpec] = {
            1: TaskProfileSpec(1, (0.5e6/8, 2e6/8), 60, 1, 1.0),
            2: TaskProfileSpec(2, (1.5e6/8, 5e6/8), 90, 2, 0.4),
            3: TaskProfileSpec(3, (4e6/8, 9e6/8), 120, 3, 0.4),
            4: TaskProfileSpec(4, (7e6/8, 15e6/8), 150, 4, 0.4),
        }
        # 鍏煎鏃у瓧娈垫牸寮?        self.task_type_specs = {
            k: {
                'data_range': v.data_range,
                'compute_density': v.compute_density,
                'max_latency_slots': v.max_latency_slots,
                'latency_weight': v.latency_weight,
            }
            for k, v in self.task_profiles.items()
        }

        # 鍦烘櫙瀹氫箟
        self.scenarios: List[TaskScenarioSpec] = [
            TaskScenarioSpec('emergency_brake', 0.18, 0.22, 1, 1.0, 0.08),
            TaskScenarioSpec('collision_avoid', 0.18, 0.24, 1, 1.0, 0.07),
            TaskScenarioSpec('navigation', 0.38, 0.42, 2, 1.0, 0.25),
            TaskScenarioSpec('traffic_signal', 0.38, 0.44, 2, 1.0, 0.15),
            TaskScenarioSpec('video_process', 0.58, 0.64, 3, 1.0, 0.20),
            TaskScenarioSpec('image_recognition', 0.58, 0.66, 3, 1.0, 0.15),
            TaskScenarioSpec('data_analysis', 0.78, 0.84, 4, 1.0, 0.08),
            TaskScenarioSpec('ml_training', 0.78, 0.86, 4, 1.0, 0.02),
        ]
        self._scenario_weights = [scenario.weight for scenario in self.scenarios]
        self._scenario_lookup = {scenario.name: scenario for scenario in self.scenarios}
        self.type_priority_weights = self._compute_type_priority_weights()
    
    def get_task_type(self, max_delay_slots: int) -> int:
        """
        鏍规嵁鏈€澶у欢杩熸椂闅欐暟纭畾浠诲姟绫诲瀷
        瀵瑰簲璁烘枃绗?.1鑺備换鍔″垎绫绘鏋?
        
        Args:
            max_delay_slots: 浠诲姟鏈€澶у彲瀹瑰繊寤惰繜鏃堕殭鏁?
            
        Returns:
            浠诲姟绫诲瀷鍊?(1-4)
        """
        if max_delay_slots <= self.delay_thresholds['extremely_sensitive']:
            return 1  # EXTREMELY_DELAY_SENSITIVE
        elif max_delay_slots <= self.delay_thresholds['sensitive']:
            return 2  # DELAY_SENSITIVE
        elif max_delay_slots <= self.delay_thresholds['moderately_tolerant']:
            return 3  # MODERATELY_DELAY_TOLERANT
        else:
            return 4  # DELAY_TOLERANT

    def sample_scenario(self) -> TaskScenarioSpec:
        """鎸夐璁炬潈閲嶉殢鏈洪€夋嫨涓€涓换鍔″満鏅€?""
        return random.choices(self.scenarios, weights=self._scenario_weights, k=1)[0]

    def get_profile(self, task_type: int) -> TaskProfileSpec:
        """鑾峰彇浠诲姟绫诲瀷瀵瑰簲鐨勬暟鎹寖鍥翠笌璁＄畻瀵嗗害閰嶇疆銆?""
        return self.task_profiles.get(
            task_type,
            TaskProfileSpec(task_type, self.data_size_range, self.task_compute_density)
        )

    def get_relax_factor(self, task_type: int) -> float:
        """鏍规嵁浠诲姟绫诲瀷杩斿洖榛樿鐨刣eadline鏀炬澗绯绘暟銆?""
        for scenario in self.scenarios:
            if scenario.task_type == task_type:
                return scenario.relax_factor
        return self.deadline_relax_default

    def _compute_type_priority_weights(self) -> Dict[int, float]:
        """鏍规嵁鍦烘櫙鏉冮噸姹囨€讳换鍔＄被鍨嬮噸瑕佹€э紝鐢ㄤ簬鍗忓悓浼樺寲鏉冮噸銆?""
        totals = defaultdict(float)
        for scenario in self.scenarios:
            profile = self.task_profiles.get(scenario.task_type)
            latency_weight = profile.latency_weight if profile else 1.0
            totals[scenario.task_type] += scenario.weight * latency_weight

        for task_type, profile in self.task_profiles.items():
            totals[task_type] = max(totals[task_type], profile.latency_weight)

        # 纭繚姣忎釜浠诲姟绫诲瀷鑷冲皯鍏峰鍩虹嚎鏉冮噸
        for task_type in self.task_profiles.keys():
            totals.setdefault(task_type, 1.0)

        values = list(totals.values())
        mean_val = sum(values) / len(values) if values else 1.0
        if mean_val <= 0:
            mean_val = 1.0

        priority_weights = {
            task_type: float(max(0.1, totals[task_type] / mean_val))
            for task_type in self.task_profiles.keys()
        }
        return priority_weights

    def get_latency_cost_weight(self, task_type: int) -> float:
        """杩斿洖浠诲姟绫诲瀷鐨勫欢杩熸垚鏈潈閲?""
        return float(self.latency_cost_weights.get(task_type, 1.0))

    def get_priority_weight(self, task_type: int) -> float:
        """杩斿洖鎸囧畾浠诲姟绫诲瀷鐨勪紭鍏堢骇鏉冮噸銆?""
        return float(self.type_priority_weights.get(task_type, 1.0))


class ServiceConfig:
    """鏈嶅姟鑳藉姏閰嶇疆锛氭帶鍒惰妭鐐规瘡涓椂闅欏彲澶勭悊鐨勪换鍔℃暟閲忎笌宸ヤ綔閲?""

    def __init__(self):
        # RSU 鏈嶅姟鑳藉姏
        self.rsu_base_service = 4
        self.rsu_max_service = 9
        self.rsu_work_capacity = 2.5  # 鐩稿綋浜庢瘡涓椂闅欑殑宸ヤ綔鍗曚綅
        self.rsu_queue_boost_divisor = 5.0

        # UAV 鏈嶅姟鑳藉姏
        self.uav_base_service = 3
        self.uav_max_service = 6
        self.uav_work_capacity = 1.7
        self.uav_queue_boost_divisor = 4.0


class StatsConfig:
    """缁熻涓庣洃鎺ч厤缃?""

    def __init__(self):
        self.drop_log_interval = 200
        # 鏇寸煭鐢熷懡鍛ㄦ湡涓嬫彁楂樿娴嬬矑搴?        self.task_report_interval = 50

class ComputeConfig:
    """璁＄畻閰嶇疆绫?""
    
    def __init__(self):
        self.parallel_efficiency = 0.8
        
        # 馃敡 淇锛氳溅杈嗚兘鑰楀弬鏁?- 鍩轰簬瀹為檯纭欢鏍″噯
        self.vehicle_kappa1 = 5.12e-31  # 鍩轰簬Intel NUC i7瀹為檯鏍″噯
        self.vehicle_kappa2 = 2.40e-20  # 棰戠巼骞虫柟椤圭郴鏁?
        self.vehicle_static_power = 8.0  # W (鐜板疄杞﹁浇鑺墖闈欐€佸姛鑰?
        self.vehicle_idle_power = 3.5   # W (绌洪棽鍔熻€?
        
        # 馃敡 淇锛歊SU鑳借€楀弬鏁?- 鍩轰簬12GHz杈圭紭鏈嶅姟鍣ㄦ牎鍑?
        self.rsu_kappa = 2.8e-31  # 12GHz楂樻€ц兘CPU鐨勫姛鑰楃郴鏁?
        self.rsu_kappa2 = 2.8e-31
        self.rsu_static_power = 25.0  # W (12GHz杈圭紭鏈嶅姟鍣ㄩ潤鎬佸姛鑰?
        
        # 馃敡 淇锛歎AV鑳借€楀弬鏁?- 鍩轰簬瀹為檯UAV纭欢鏍″噯
        self.uav_kappa = 8.89e-31  # 鍔熻€楀彈闄愮殑UAV鑺墖
        self.uav_kappa3 = 8.89e-31  # 淇鍚庡弬鏁?
        self.uav_static_power = 2.5  # W (杞婚噺鍖栬璁?
        self.uav_hover_power = 25.0  # W (鏇村悎鐞嗙殑鎮仠鍔熻€?
        
        # CPU棰戠巼鑼冨洿 - 绗﹀悎鍐呭瓨瑙勮寖
        self.vehicle_cpu_freq_range = (8e9, 25e9)  # 8-25 GHz
        self.rsu_cpu_freq_range = (45e9, 55e9)  # 50 GHz宸﹀彸
        self.uav_cpu_freq_range = (1.5e9, 9e9)  # 1.5-9 GHz锛屽寘鍚紭鍖栧悗鐨?.8GHz
        
        # 馃敡 淇锛氫紭鍖朥AV璁＄畻鑳藉姏浠ュ钩琛＄郴缁熻礋杞?
        self.vehicle_default_freq = 2.5e9  # 2.5 GHz (淇濇寔杞﹁浇鑺墖)
        self.rsu_default_freq = 12e9  # 鎭㈠12GHz - 楂樻€ц兘杈圭紭璁＄畻
        self.uav_default_freq = 1.8e9  # 馃敡 浼樺寲鑷?.8GHz - 骞宠　璐熻浇涓庤兘鑰?
        
        # 鑺傜偣CPU棰戠巼锛堢敤浜庡垵濮嬪寲锛?
        self.vehicle_cpu_freq = self.vehicle_default_freq
        self.rsu_cpu_freq = self.rsu_default_freq
        self.uav_cpu_freq = self.uav_default_freq
        
        # 鍐呭瓨閰嶇疆
        self.vehicle_memory_size = 8e9  # 8 GB
        self.rsu_memory_size = 32e9  # 32 GB
        self.uav_memory_size = 4e9  # 4 GB
        
        # UAV鐗规畩閰嶇疆
        self.uav_hover_power = 50.0  # W

class NetworkConfig:
    """缃戠粶閰嶇疆绫?""
    
    def __init__(self):
        self.time_slot_duration = 0.2  # seconds - 浼樺寲涓烘洿鍚堢悊鐨勬椂闅欓暱搴?
        self.bandwidth = 20e6  # Hz
        self.carrier_frequency = 2.4e9  # Hz
        self.noise_power = -174  # dBm/Hz
        self.path_loss_exponent = 2.0
        self.coverage_radius = 1000  # meters
        self.interference_threshold = 0.1
        self.handover_threshold = 0.2
        
        # 鑺傜偣鏁伴噺閰嶇疆
        self.num_vehicles = 12  # 鎭㈠鍒板師濮嬭缃?
        self.num_rsus = 4       # 鏇存柊涓?涓猂SU锛堝崟涓€璺鍙岃矾鍙ｅ満鏅級
        self.num_uavs = 2       # 鎭㈠鍒板師濮嬭缃紝绗﹀悎璁烘枃瑕佹眰
        
        # 缃戠粶鎷撴墤鍙傛暟
        self.area_width = 2500  # meters - 缂╁皬浠跨湡鍖哄煙
        self.area_height = 2500  # meters
        self.min_distance = 50  # meters
        
        # 杩炴帴鍙傛暟
        self.max_connections_per_node = 10
        self.connection_timeout = 30  # seconds

class CommunicationConfig:
    """3GPP鏍囧噯閫氫俊閰嶇疆绫?""
    
    def __init__(self):
        # 3GPP鏍囧噯鍙戝皠鍔熺巼
        self.vehicle_tx_power = 23.0  # dBm (200mW) - 3GPP鏍囧噯
        self.rsu_tx_power = 46.0      # dBm (40W) - 3GPP鏍囧噯
        self.uav_tx_power = 30.0      # dBm (1W) - 3GPP鏍囧噯
        self.circuit_power = 0.1      # W
        self.noise_figure = 9.0       # dB - 3GPP鏍囧噯
        
        # 3GPP鏍囧噯甯﹀閰嶇疆
        self.total_bandwidth = 20e6   # 20 MHz - 3GPP鏍囧噯
        self.channel_bandwidth = 1e6  # 1 MHz per channel
        self.uplink_bandwidth = 10e6  # 10 MHz
        self.downlink_bandwidth = 10e6  # 10 MHz
        
        # 3GPP鏍囧噯浼犳挱鍙傛暟
        self.carrier_frequency = 2.0e9  # 2 GHz - 3GPP鏍囧噯棰戠巼
        self.speed_of_light = 3e8       # m/s
        self.thermal_noise_density = -174.0  # dBm/Hz - 3GPP鏍囧噯
        
        # 3GPP鏍囧噯澶╃嚎澧炵泭
        self.antenna_gain_rsu = 15.0     # dBi
        self.antenna_gain_uav = 5.0      # dBi
        self.antenna_gain_vehicle = 3.0  # dBi
        
        # 3GPP鏍囧噯璺緞鎹熻€楀弬鏁?
        self.los_threshold = 50.0        # m - 3GPP TS 38.901
        self.los_decay_factor = 100.0    # m
        self.shadowing_std_los = 4.0     # dB
        self.shadowing_std_nlos = 8.0    # dB
        
        # 璋冨埗鍙傛暟
        self.modulation_order = 4  # QPSK
        self.coding_rate = 0.5

class MigrationConfig:
    """杩佺Щ閰嶇疆绫?""
    
    def __init__(self):
        self.migration_bandwidth = 100e6  # bps
        self.migration_threshold = 0.8
        self.migration_cost_factor = 0.1
        
        # 馃敡 璋冩暣锛氬悎鐞嗙殑杩佺Щ瑙﹀彂闃堝€?
        self.rsu_overload_threshold = 0.85   # 鎭㈠鍒?0%锛屾洿鍚堢悊鐨勮Е鍙戠偣
        self.uav_overload_threshold = 0.85  # UAV 75%璐熻浇瑙﹀彂锛岀暐鏃╀簬RSU
        self.rsu_underload_threshold = 0.3
        # 闃熷垪/鍒囨崲闃堝€硷紙鐢ㄤ簬杞﹁締璺熼殢涓庤繃杞藉垏鎹級
        self.follow_handover_distance = 30.0  # meters锛岃溅杈嗚窡闅忚Е鍙戠殑鏈€灏忚窛绂绘敼鍠?
        # 馃敡 鏈€缁堜紭鍖栵細缁熶竴闃熷垪绠＄悊鏍囧噯
        self.queue_switch_diff = 3            # 涓紝鐩爣RSU杈冨綋鍓峈SU闃熷垪鑷冲皯灏?涓墠鍒囨崲  
        self.rsu_queue_overload_len = 10      # 涓紝鍩轰簬瀹為檯瑙傚療鎻愰珮鍒?5涓换鍔¤繃杞介槇鍊?
        self.service_jitter_ratio = 0.2       # 鏈嶅姟閫熺巼卤20%鎶栧姩
        
        # UAV杩佺Щ鍙傛暟
        self.uav_min_battery = 0.2  # 20%
        self.migration_delay_threshold = 1.0  # seconds
        self.max_migration_distance = 1000  # meters
        
        # 杩佺Щ鎴愭湰鍙傛暟
        self.migration_alpha_comp = 0.4  # 璁＄畻鎴愭湰鏉冮噸
        self.migration_alpha_tx = 0.3    # 浼犺緭鎴愭湰鏉冮噸
        self.migration_alpha_lat = 0.3   # 寤惰繜鎴愭湰鏉冮噸
        
        self.migration_energy_cost = 0.1  # J per bit
        self.migration_time_penalty = 0.05  # seconds
        
        # 馃敡 鐢ㄦ埛瑕佹眰锛氭瘡绉掕Е鍙戜竴娆¤縼绉诲喅绛?
        self.cooldown_period = 1.0  # 1绉掑喎鍗存湡锛屽疄鐜版瘡绉掓渶澶氫竴娆¤縼绉?

class CacheConfig:
    """缂撳瓨閰嶇疆绫?""
    
    def __init__(self):
        # 缂撳瓨瀹归噺閰嶇疆
        self.vehicle_cache_capacity = 1e9  # 1 GB
        self.rsu_cache_capacity = 10e9  # 10 GB
        self.uav_cache_capacity = 2e9  # 2 GB
        
        # 缂撳瓨绛栫暐閰嶇疆
        self.cache_replacement_policy = 'LRU'  # LRU, LFU, RANDOM
        self.cache_hit_threshold = 0.8
        self.cache_update_interval = 1.0  # seconds
        
        # 缂撳瓨棰勬祴鍙傛暟
        self.prediction_window = 10  # time slots
        self.popularity_decay_factor = 0.9
        self.request_history_size = 100

class SystemConfig:
    """绯荤粺閰嶇疆绫?""
    
    def __init__(self):
        # 鍩烘湰绯荤粺閰嶇疆
        self.device = 'cuda' if os.environ.get('CUDA_AVAILABLE', 'false').lower() == 'true' else 'cpu'
        self.num_threads = int(os.environ.get('NUM_THREADS', '4'))
        self.random_seed = int(os.environ.get('RANDOM_SEED', '42'))
        
        # 馃殌 12杞﹁締楂樿礋杞藉満鏅綉缁滈厤缃?
        self.num_vehicles = 12  # 淇濇寔12杞﹁締锛岄€氳繃鍏朵粬鏂瑰紡鍒涢€犻珮璐熻浇
        self.num_rsus = 4       # 鏇存柊涓?涓猂SU
        self.num_uavs = 2       # 淇濇寔UAV鏁伴噺
        
        # 浠跨湡閰嶇疆
        self.simulation_time = 1000
        self.time_slot = 0.2
        
        # 鎬ц兘閰嶇疆
        self.enable_performance_optimization = True
        self.batch_size_optimization = True
        self.parallel_environments = 6
        
        # 瀛愰厤缃ā鍧?
        self.queue = QueueConfig()
        self.task = TaskConfig()
        self.compute = ComputeConfig()
        self.network = NetworkConfig()
        self.communication = CommunicationConfig()
        self.migration = MigrationConfig()
        self.cache = CacheConfig()
        self.service = ServiceConfig()
        self.stats = StatsConfig()
        
        # 瀹為獙閰嶇疆
        self.experiment = ExperimentConfig()
        
        # 寮哄寲瀛︿範閰嶇疆
        self.rl = RLConfig()
        
    def get_config_dict(self) -> Dict[str, Any]:
        """鑾峰彇閰嶇疆瀛楀吀"""
        return {
            'device': self.device,
            'num_threads': self.num_threads,
            'random_seed': self.random_seed,
            'num_vehicles': self.num_vehicles,
            'num_rsus': self.num_rsus,
            'num_uavs': self.num_uavs,
            'simulation_time': self.simulation_time,
            'time_slot': self.time_slot,
            'enable_performance_optimization': self.enable_performance_optimization,
            'batch_size_optimization': self.batch_size_optimization,
            'parallel_environments': self.parallel_environments
        }
    
    def update_config(self, **kwargs):
        """鏇存柊閰嶇疆"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

# 鍏ㄥ眬閰嶇疆瀹炰緥
config = SystemConfig()

