"""
馃幆 CAMTD3璁粌鑴氭湰锛圕ache-Aware Migration with Twin Delayed DDPG锛?

銆愮郴缁熸灦鏋勩€?
CAMTD3 = 鍩轰簬涓ぎ璧勬簮鍒嗛厤鐨勭紦瀛樻劅鐭ヤ换鍔¤縼绉荤郴缁?
鈹溾攢鈹€ Phase 1: 涓ぎ鏅鸿兘浣撹祫婧愬垎閰嶅喅绛栵紙鏍稿績鍒涙柊锛?
鈹?  鈹溾攢鈹€ 鐘舵€佺┖闂? 80缁达紙杞﹁締+RSU+UAV鍏ㄥ眬鐘舵€侊級
鈹?  鈹溾攢鈹€ 鍔ㄤ綔绌洪棿: 30缁达紙甯﹀+璁＄畻璧勬簮鍒嗛厤鍚戦噺锛?
鈹?  鈹斺攢鈹€ 绠楁硶: TD3/SAC/DDPG/PPO
鈹溾攢鈹€ Phase 2: 鏈湴浠诲姟鎵ц
鈹?  鈹溾攢鈹€ 缂撳瓨鍐崇瓥锛圕ache-Aware锛?
鈹?  鈹溾攢鈹€ 浠诲姟杩佺Щ锛圡igration锛?
鈹?  鈹斺攢鈹€ 浠诲姟璋冨害

python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 1000 --num-vehicles 12 --seed 42

Queue-aware Replay
鉁?璁粌鏁堢巼鎻愬崌5鍊?
鉁?蹇€熷涔犻珮璐熻浇鍦烘櫙
鉁?閽堝VEC闃熷垪绠＄悊鐥涚偣
GNN Attention
鉁?缂撳瓨鍛戒腑鐜囨彁鍗?20鍊?
鉁?鏅鸿兘瀛︿範鑺傜偣鍗忎綔鍏崇郴
鉁?閫傚簲鍔ㄦ€佹嫇鎵戝彉鍖?
銆愪娇鐢ㄦ柟娉曘€?
# CAMTD3鏍囧噯璁粌锛堥粯璁ゆā寮忥級
python train_single_agent.py --algorithm TD3 --episodes 200
python train_single_agent.py --algorithm SAC --episodes 200

鉁? 鍙惎鐢ㄥ姩鎬佸甫瀹藉垎閰?
python train_single_agent.py --algorithm TD3 --episodes 200 --dynamic-bandwidth


# 濡傞渶绂佺敤涓ぎ璧勬簮鍒嗛厤锛堜笉鎺ㄨ崘锛屼粎鐢ㄤ簬娑堣瀺瀹為獙锛?
python train_single_agent.py --algorithm TD3 --episodes 200 --no-central-resource

馃悕馃枼锔忦煋?

鍗曟櫤鑳戒綋绠楁硶璁粌鑴氭湰
鏀寔DDPG銆乀D3銆乀D3-LE銆丏QN銆丳PO銆丼AC绛夌畻娉曠殑璁粌鍜屾瘮杈?
python train_single_agent.py --compare --episodes 200  # 姣旇緝鎵€鏈夌畻娉?
馃殌 澧炲己缂撳瓨妯″紡 (榛樿鍚敤 - 鍒嗗眰L1/L2 + 鑷€傚簲鐑害绛栫暐 + RSU鍗忎綔):
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 8
python train_single_agent.py --algorithm TD3 --episodes 1000 --num-vehicles 12
python train_single_agent.py --algorithm TD3 --episodes 800 --num-vehicles 12 --silent-mode  # 闈欓粯淇濆瓨缁撴灉
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 16
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 20
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 24
python train_single_agent.py --algorithm TD3-LE --episodes 1600 --num-vehicles 12
python train_single_agent.py --algorithm SAC --episodes 800
python train_single_agent.py --algorithm PPO --episodes 800

馃寪 瀹炴椂鍙鍖?
python train_single_agent.py --algorithm DDPG --episodes 100 --realtime-vis --vis-port 8080

馃悕 鐢熸垚瀛︽湳鍥捐〃:
python generate_academic_charts.py results/single_agent/td3/training_results_20251007_220900.json

鍒拌揪鐜囧姣旓細python experiments/arrival_rate_analysis/run_td3_arrival_rate_sweep_silent.py --rates 1.0 1.5 2.0 2.5 3.0 3.5 --episodes 800


""" 
import os
import sys
import random

# 馃敡 淇Windows缂栫爜闂
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
        elif hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')  # type: ignore[attr-defined]
        elif hasattr(sys.stderr, 'buffer'):
            import io
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass

import argparse
import json
from tools.fixed_topology_optimizer import FixedTopologyOptimizer
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 瀵煎叆鏍稿績妯″潡
from config import config
from evaluation.system_simulator import CompleteSystemSimulator
try:
    from evaluation.enhanced_system_simulator import EnhancedSystemSimulator
    ENHANCED_CACHE_AVAILABLE = True
except ImportError:
    ENHANCED_CACHE_AVAILABLE = False
    print("[Warning] Enhanced cache system not available, using standard simulator")
from utils import MovingAverage
from utils.normalization_utils import (
    normalize_distribution,
    normalize_feature_vector,
    normalize_ratio,
    normalize_scalar,
)
# 馃 瀵煎叆鑷€傚簲鎺у埗缁勪欢
from utils.adaptive_control import AdaptiveCacheController, AdaptiveMigrationController, map_agent_actions_to_params
from decision.strategy_coordinator import StrategyCoordinator
from utils.unified_reward_calculator import update_reward_targets

# 瀵煎叆鍚勭鍗曟櫤鑳戒綋绠楁硶
from single_agent.ddpg import DDPGEnvironment
from single_agent.td3 import TD3Environment
from single_agent.td3_hybrid_fusion import CAMTD3Environment
from single_agent.td3_latency_energy import TD3LatencyEnergyEnvironment
from single_agent.dqn import DQNEnvironment
from single_agent.ppo import PPOEnvironment
from single_agent.sac import SACEnvironment
# 瀵煎叆绮剧畝浼樺寲TD3 (浠匭ueue-aware + GNN)
from single_agent.optimized_td3_wrapper import OptimizedTD3Environment

# 瀵煎叆HTML鎶ュ憡鐢熸垚鍣?
from utils.html_report_generator import HTMLReportGenerator

# 瀵煎叆璁粌缁撴灉淇濆瓨鍜岀粯鍥惧伐鍏?
from utils.training_results import save_single_training_results, plot_single_training_curves

# 馃寪 瀵煎叆瀹炴椂鍙鍖栨ā鍧?
try:
    from scripts.visualize.realtime_visualization import create_visualizer
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("鈿狅笍  瀹炴椂鍙鍖栧姛鑳戒笉鍙敤锛岃杩愯: pip install flask flask-socketio")

# 馃帹 瀵煎叆楂樼璁粌鍙鍖栧櫒
try:
    from utils.advanced_training_visualizer import create_visualizer as create_advanced_visualizer
    ADVANCED_VIS_AVAILABLE = True
except ImportError:
    ADVANCED_VIS_AVAILABLE = False
    print("鈿狅笍  楂樼鍙鍖栧姛鑳戒笉鍙敤")

# 灏濊瘯瀵煎叆PyTorch浠ヨ缃殢鏈虹瀛愶紱濡傛灉涓嶅彲鐢ㄥ垯璺宠繃
try:
    import torch
except ImportError:  # pragma: no cover -瀹归敊澶勭悊
    torch = None


def _apply_global_seed_from_env():
    """鏍规嵁鐜鍙橀噺RANDOM_SEED璁剧疆闅忔満绉嶅瓙锛岀‘淇濆彲閲嶅鎬?""
    seed_env = os.environ.get('RANDOM_SEED')
    if not seed_env:
        return
    try:
        seed = int(seed_env)
    except ValueError:
        print(f"鈿狅笍  RANDOM_SEED 鐜鍙橀噺鏃犳晥: {seed_env}")
        return

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU鍙€?
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    config.random_seed = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"馃攼 鍏ㄥ眬闅忔満绉嶅瓙宸茶缃负 {seed}")


def _maybe_apply_reward_smoothing_from_env() -> None:
    """Optionally enable reward smoothing via environment variables.

    RL_SMOOTH_DELAY, RL_SMOOTH_ENERGY, RL_SMOOTH_ALPHA can be provided.
    """
    try:
        d = os.environ.get('RL_SMOOTH_DELAY')
        e = os.environ.get('RL_SMOOTH_ENERGY')
        a = os.environ.get('RL_SMOOTH_ALPHA')
        if d is not None:
            setattr(config.rl, 'reward_smooth_delay_weight', float(d))
        if e is not None:
            setattr(config.rl, 'reward_smooth_energy_weight', float(e))
        if a is not None:
            setattr(config.rl, 'reward_smooth_alpha', float(a))
    except Exception:
        pass


def _apply_reward_overrides_from_env() -> None:
    """Allow quick reward/target retuning via environment variables for hard scenarios."""
    latency_target = os.environ.get("RL_LATENCY_TARGET")
    energy_target = os.environ.get("RL_ENERGY_TARGET")
    disable_dynamic_targets = os.environ.get("RL_DISABLE_DYNAMIC_TARGETS", "0") != "0"
    # 鏂板锛氭敮鎸佸揩閫熸彁楂樹涪鍖?瀹屾垚鐜囨儵缃氾紝榛樿鏇撮噸绾︽潫楂樿礋杞戒笉鍙潬琛屼负
    default_loss_weight = 1.4
    default_completion_gap = 0.7
    default_drop_penalty = 0.18
    if not getattr(config.rl, "reward_weight_loss_ratio", 0.0):
        setattr(config.rl, "reward_weight_loss_ratio", default_loss_weight)
    if not getattr(config.rl, "reward_weight_completion_gap", 0.0):
        setattr(config.rl, "reward_weight_completion_gap", default_completion_gap)
    if getattr(config.rl, "reward_penalty_dropped", 0.0) < default_drop_penalty:
        setattr(config.rl, "reward_penalty_dropped", default_drop_penalty)

    weight_overrides = {
        "RL_WEIGHT_DELAY": "reward_weight_delay",
        "RL_WEIGHT_ENERGY": "reward_weight_energy",
        "RL_WEIGHT_CACHE": "reward_weight_cache",
        "RL_WEIGHT_CACHE_PRESSURE": "reward_weight_cache_pressure",
        "RL_WEIGHT_MIGRATION": "reward_weight_migration",
        "RL_WEIGHT_QUEUE_OVERLOAD": "reward_weight_queue_overload",
        "RL_WEIGHT_REMOTE_REJECT": "reward_weight_remote_reject",
        "RL_WEIGHT_LOSS_RATIO": "reward_weight_loss_ratio",
        "RL_WEIGHT_COMPLETION_GAP": "reward_weight_completion_gap",
        "RL_PENALTY_DROPPED": "reward_penalty_dropped",
    }

    def _try_set(env_key: str, attr: str) -> None:
        val = os.environ.get(env_key)
        if not val:
            return
        try:
            setattr(config.rl, attr, float(val))
            print(f"[RewardOverride] {attr} <- {val}")
        except Exception:
            pass

    for env_key, attr in weight_overrides.items():
        _try_set(env_key, attr)

    # Targets need to sync with the reward calculator singletons
    target_changed = False
    try:
        if latency_target is not None:
            config.rl.latency_target = float(latency_target)
            target_changed = True
            print(f"[RewardOverride] latency_target <- {latency_target}")
    except Exception:
        pass
    try:
        if energy_target is not None:
            config.rl.energy_target = float(energy_target)
            target_changed = True
            print(f"[RewardOverride] energy_target <- {energy_target}")
    except Exception:
        pass
    if disable_dynamic_targets:
        # 璁剧疆鍏ㄥ眬绂佺敤鍔ㄦ€佹斁瀹?
        os.environ['DYNAMIC_TARGET_DISABLE'] = '1'
        print("[RewardOverride] 鍔ㄦ€佺洰鏍囨斁瀹藉凡绂佺敤 (RL_DISABLE_DYNAMIC_TARGETS=1)")
    if target_changed:
        try:
            update_reward_targets(
                latency_target=config.rl.latency_target,
                energy_target=config.rl.energy_target,
            )
        except Exception:
            # keep training even if sync fails
            pass

def _build_scenario_config() -> Dict[str, Any]:
    """鏋勫缓妯℃嫙鐜閰嶇疆锛屽厑璁搁€氳繃鐜鍙橀噺瑕嗙洊榛樿鍊?""
    # 馃敡 鏀寔浠庣幆澧冨彉閲忚鐩栦换鍔″埌杈剧巼锛堢敤浜庡弬鏁版晱鎰熸€у垎鏋愶級
    task_arrival_rate = getattr(getattr(config, "task", None), "arrival_rate", 1.8)
    if os.environ.get('TASK_ARRIVAL_RATE'):
        try:
            arrival_rate_str = os.environ.get('TASK_ARRIVAL_RATE')
            if arrival_rate_str is not None:
                task_arrival_rate = float(arrival_rate_str)
                print(f"馃敡 浠庣幆澧冨彉閲忚鐩栦换鍔″埌杈剧巼: {task_arrival_rate} tasks/s")
        except ValueError:
            print(f"鈿狅笍  鐜鍙橀噺TASK_ARRIVAL_RATE鏃犳晥锛屼娇鐢ㄩ粯璁ゅ€?)

    def _get_or_default(obj: Optional[Any], attr: str, default: Any) -> Any:
        return getattr(obj, attr, default) if obj is not None else default

    network_cfg = getattr(config, "network", None)
    vehicle_cfg = getattr(network_cfg, "vehicle_config", {}) if network_cfg else {}
    rsu_cfg = getattr(network_cfg, "rsu_config", {}) if network_cfg else {}
    uav_cfg = getattr(network_cfg, "uav_config", {}) if network_cfg else {}
    comm_cfg = getattr(network_cfg, "communication_config", {}) if network_cfg else {}
    compute_cfg = getattr(config, "compute", None)
    service_cfg = getattr(config, "service", None)
    communication_cfg = getattr(config, "communication", None)

    def _normalize_bandwidth(value: Optional[float], fallback: float) -> float:
        if value is None:
            return fallback
        bw = float(value)
        if bw < 1e3:  # assume MHz 鈫?Hz
            bw *= 1e6
        return bw

    scenario = {
        "num_vehicles": getattr(config, "num_vehicles", vehicle_cfg.get('num_vehicles', 12)),
        "num_rsus": getattr(config, "num_rsus", rsu_cfg.get('num_rsus', 4)),
        "num_uavs": getattr(config, "num_uavs", uav_cfg.get('num_uavs', 2)),
        "task_arrival_rate": task_arrival_rate,
        "time_slot": getattr(config, "time_slot", _get_or_default(network_cfg, 'time_slot_duration', 0.1)),
        "simulation_time": getattr(config, "simulation_time", 1000),
        "computation_capacity": float(vehicle_cfg.get('computation_capacity', 1000)),
        "bandwidth": _normalize_bandwidth(
            comm_cfg.get('bandwidth'),
            _get_or_default(communication_cfg, 'total_bandwidth', 50e6),
        ),
        "coverage_radius": float(rsu_cfg.get('coverage_radius', 300)),
        "cache_capacity": float(rsu_cfg.get('cache_capacity', 120)),
        "transmission_power": float(vehicle_cfg.get('transmission_power', 0.15)),
        "computation_power": float(_get_or_default(compute_cfg, 'vehicle_static_power', 1.2)),
        "thermal_noise_density": float(comm_cfg.get('thermal_noise_density', -174.0)),
        "noise_figure": float(_get_or_default(communication_cfg, 'noise_figure', 9.0)),
        "high_load_mode": getattr(getattr(config, "task", None), "high_load_mode", False),
        "task_complexity_multiplier": float(
            getattr(getattr(config, "task", None), "complexity_multiplier", 1.1)
        ),
        "rsu_load_divisor": float(_get_or_default(service_cfg, 'rsu_queue_boost_divisor', 4.0)),
        "uav_load_divisor": float(_get_or_default(service_cfg, 'uav_queue_boost_divisor', 2.0)),
        "enhanced_task_generation": True,
    }

    override_env = os.environ.get('TRAINING_SCENARIO_OVERRIDES')
    if override_env:
        try:
            overrides = json.loads(override_env)
            if isinstance(overrides, dict):
                scenario.update(overrides)
            else:
                print("鈿狅笍  TRAINING_SCENARIO_OVERRIDES 闇€涓篔SON瀵硅薄锛屽凡蹇界暐銆?)
        except json.JSONDecodeError as exc:
            print(f"鈿狅笍  TRAINING_SCENARIO_OVERRIDES 瑙ｆ瀽澶辫触: {exc}")

    return scenario


_apply_global_seed_from_env()
_maybe_apply_reward_smoothing_from_env()


def generate_timestamp() -> str:
    """鐢熸垚鏃堕棿鎴?""
    if config.experiment.use_timestamp:
        return datetime.now().strftime(config.experiment.timestamp_format)
    else:
        return ""

def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """鑾峰彇甯︽椂闂存埑鐨勬枃浠跺悕"""
    timestamp = generate_timestamp()
    if timestamp:
        name_parts = base_name.split('.')
        if len(name_parts) > 1:
            base = '.'.join(name_parts[:-1])
            return f"{base}_{timestamp}{extension}"
        else:
            return f"{base_name}_{timestamp}{extension}"
    else:
        return f"{base_name}{extension}"


class SingleAgentTrainingEnvironment:
    """鍗曟櫤鑳戒綋璁粌鐜鍩虹被"""
    
    def _apply_optimized_td3_defaults(self) -> None:
        """涓篛PTIMIZED_TD3璁剧疆鏇村己鐨勫彲闈犳€?鎺㈢储榛樿鍊硷紙鍙鐜鍙橀噺瑕嗙洊锛夈€?""
        if not hasattr(self, 'algorithm') and hasattr(self, 'input_algorithm'):
            alg = str(self.input_algorithm).upper()
        else:
            alg = getattr(self, 'algorithm', '').upper()
        if alg != "OPTIMIZED_TD3":
            return
        rl = getattr(config, "rl", None)
        if rl is None:
            return

        def _set_if_absent(env_key: str, attr: str, value: float, use_max: bool = False) -> None:
            if os.environ.get(env_key) is not None:
                return
            current = float(getattr(rl, attr, 0.0) or 0.0)
            val_to_set = max(current, value) if use_max else (current if current else value)
            setattr(rl, attr, val_to_set)

        def _force_override(env_key: str, attr: str, value: float) -> None:
            """瀵筄PTIMIZED_TD3浣跨敤鏇存俯鍜岀殑鏉冮噸/鐩爣锛岄檷浣庡鍔辨柟宸€?""
            if os.environ.get(env_key) is not None:
                return
            setattr(rl, attr, float(value))

        # 鍙潬鎬ф潈閲嶏紙鏀舵暃鍙嬪ソ鐗堟湰锛?        _set_if_absent("RL_WEIGHT_LOSS_RATIO", "reward_weight_loss_ratio", 1.2)
        _set_if_absent("RL_WEIGHT_COMPLETION_GAP", "reward_weight_completion_gap", 0.7)
        _set_if_absent("RL_PENALTY_DROPPED", "reward_penalty_dropped", 0.15, use_max=True)
        _set_if_absent("RL_WEIGHT_QUEUE_OVERLOAD", "reward_weight_queue_overload", 0.8, use_max=True)
        _set_if_absent("RL_WEIGHT_REMOTE_REJECT", "reward_weight_remote_reject", 0.25, use_max=True)

        # 鏍稿績鏉冮噸锛氱洿鎺ヨ鐩栧叏灞€杈冩縺杩涚殑榛樿鍊?        _force_override("RL_WEIGHT_CACHE", "reward_weight_cache", 0.2)
        _force_override("RL_WEIGHT_CACHE_BONUS", "reward_weight_cache_bonus", 0.3)
        _force_override("RL_WEIGHT_DELAY", "reward_weight_delay", 1.8)
        _force_override("RL_WEIGHT_ENERGY", "reward_weight_energy", 1.2)
        _force_override("RL_WEIGHT_OFFLOAD_BONUS", "reward_weight_offload_bonus", 3.0)
        _force_override("RL_WEIGHT_LOCAL_PENALTY", "reward_weight_local_penalty", 1.0)

        # 鐩爣鍊硷紙鏀惧锛屽尮閰嶅浘琛ㄤ娇鐢ㄧ殑2.3s/9600J鍩哄噯锛屽噺灏戝綊涓€鍖栨姈鍔級
        _force_override("RL_LATENCY_TARGET", "latency_target", 2.3)
        _force_override("RL_LATENCY_UPPER_TOL", "latency_upper_tolerance", 3.5)
        _force_override("RL_ENERGY_TARGET", "energy_target", 9600.0)
        _force_override("RL_ENERGY_UPPER_TOL", "energy_upper_tolerance", 14000.0)
        try:
            update_reward_targets(
                latency_target=float(getattr(rl, "latency_target", 2.3)),
                energy_target=float(getattr(rl, "energy_target", 9600.0)),
            )
        except Exception:
            pass

    def __init__(
        self,
        algorithm: str,
        override_scenario: Optional[Dict[str, Any]] = None,
        use_enhanced_cache: bool = False,
        disable_migration: bool = False,
        enforce_offload_mode: Optional[str] = None,
        fixed_offload_policy: Optional[str] = None,
        joint_controller: bool = False,
    ):
        self.input_algorithm = algorithm
        normalized_algorithm = algorithm.upper().replace('-', '_')
        alias_map = {
            "TD3LE": "TD3_LATENCY_ENERGY",
            "TD3_LE": "TD3_LATENCY_ENERGY",
            "TD3LATENCY": "TD3_LATENCY_ENERGY",
            "TD3_LATENCY": "TD3_LATENCY_ENERGY",
            "TD3_LATENCY_ENERGY": "TD3_LATENCY_ENERGY",
            "CAMTD3": "CAM_TD3",
            "CAM_TD3": "CAM_TD3",
            "HYBRID_EDGE-TD3": "CAM_TD3",
        }
        alias_key = normalized_algorithm.replace('_', '')
        self.algorithm = alias_map.get(normalized_algorithm, alias_map.get(alias_key, normalized_algorithm))
        self._apply_optimized_td3_defaults()
        scenario_config = _build_scenario_config()
        # 搴旂敤澶栭儴瑕嗙洊
        central_env_value = os.environ.get('CENTRAL_RESOURCE', '')
        self.central_resource_enabled = central_env_value.strip() in {'1', 'true', 'True'}
        self.joint_controller = bool(joint_controller)
        if self.joint_controller and not self.central_resource_enabled:
            os.environ['CENTRAL_RESOURCE'] = '1'
            self.central_resource_enabled = True

        if override_scenario:
            scenario_config.update(override_scenario)
            scenario_config['override_topology'] = True
            
            # 馃敡 鍏抽敭淇锛氬姩鎬佷慨鏀瑰叏灞€config浠ユ敮鎸佸弬鏁拌鐩?
            # 鍘熷洜锛歂ode绫讳娇鐢ㄥ叏灞€config鑰岄潪scenario_config
            network_cfg = getattr(config, "network", None)

            def _sync_topology(attr_name: str, component_attr: str, dict_key: str, value: int) -> None:
                setattr(config, attr_name, value)
                if network_cfg is not None:
                    setattr(network_cfg, attr_name, value)
                    component_cfg = getattr(network_cfg, component_attr, None)
                    if isinstance(component_cfg, dict):
                        component_cfg[dict_key] = value
            
            # 鎷撴墤鏁伴噺鍙傛暟
            if 'num_vehicles' in override_scenario:
                num_vehicles_override = int(override_scenario['num_vehicles'])
                _sync_topology('num_vehicles', 'vehicle_config', 'num_vehicles', num_vehicles_override)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃溅杈嗘暟閲? {num_vehicles_override}")
                
                # 馃敡 鏂板锛氭牴鎹溅杈嗘暟鍔ㄦ€佽皟鏁寸洰鏍囧€?
                # 浼扮畻锛氭瘡杞﹁締绾?0.3s 鏃跺欢, 1000J 鑳借€?
                if os.environ.get('RL_LATENCY_TARGET') is None:  # 浠呭綋鏈墜鍔ㄦ寚瀹氭椂
                    auto_latency_target = 0.5 + num_vehicles_override * 0.15  # 6杞︹増1.4s, 12杞︹増2.3s, 20杞︹増3.5s
                    config.rl.latency_target = auto_latency_target
                    config.rl.latency_upper_tolerance = auto_latency_target * 2.5
                    print(f"  鈫?鑷姩璋冩暣 latency_target: {auto_latency_target:.2f}s")
                
                if os.environ.get('RL_ENERGY_TARGET') is None:  # 浠呭綋鏈墜鍔ㄦ寚瀹氭椂
                    auto_energy_target = num_vehicles_override * 800.0  # 6杞︹増4800J, 12杞︹増9600J, 20杞︹墾16000J
                    config.rl.energy_target = auto_energy_target
                    config.rl.energy_upper_tolerance = auto_energy_target * 2.0
                    print(f"  鈫?鑷姩璋冩暣 energy_target: {auto_energy_target:.0f}J")
                
                # 鍚屾鍒板叏灞€濂栧姳璁＄畻鍣?
                try:
                    from utils.unified_reward_calculator import update_reward_targets
                    update_reward_targets(
                        latency_target=float(config.rl.latency_target),
                        energy_target=float(config.rl.energy_target)
                    )
                except Exception as e:
                    print(f"  鈿狅笍  濂栧姳鐩爣鍚屾澶辫触: {e}")
            if 'num_rsus' in override_scenario:
                num_rsus_override = int(override_scenario['num_rsus'])
                _sync_topology('num_rsus', 'rsu_config', 'num_rsus', num_rsus_override)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甊SU鏁伴噺: {num_rsus_override}")
            if 'num_uavs' in override_scenario:
                num_uav_override = int(override_scenario['num_uavs'])
                _sync_topology('num_uavs', 'uav_config', 'num_uavs', num_uav_override)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甎AV鏁伴噺: {num_uav_override}")

            # 甯﹀鍙傛暟
            if 'bandwidth' in override_scenario or 'total_bandwidth' in override_scenario:
                bw_value = override_scenario.get('total_bandwidth') or override_scenario.get('bandwidth')
                if bw_value:
                    config.communication.total_bandwidth = float(bw_value)
                    network_comm_cfg = getattr(network_cfg, "communication_config", None)
                    if isinstance(network_comm_cfg, dict):
                        network_comm_cfg['bandwidth'] = float(bw_value)
                    # 馃敡 鍏抽敭淇锛氬悓姝ュ埌scenario_config锛岀‘淇濅豢鐪熷櫒浣跨敤姝ｇ‘鐨勫甫瀹?
                    scenario_config['total_bandwidth'] = float(bw_value)
                    scenario_config['bandwidth'] = float(bw_value)  # 鍏煎涓ょ鍛藉悕
                    print(f"馃敡 [Override] 鍔ㄦ€佽缃甫瀹? {float(bw_value)/1e6:.1f} MHz")
            
            # 馃幆 鎬昏祫婧愭睜鍙傛暟锛堜紭鍏堢骇楂樹簬鍗曡妭鐐归鐜囷級
        if override_scenario is not None and 'total_vehicle_compute' in override_scenario:
            total_compute = float(override_scenario['total_vehicle_compute'])
            config.compute.total_vehicle_compute = total_compute
            # 鑷姩璁＄畻姣忚溅骞冲潎棰戠巼
            avg_freq = total_compute / config.num_vehicles
            config.compute.vehicle_initial_freq = avg_freq
            config.compute.vehicle_default_freq = avg_freq
            config.compute.vehicle_cpu_freq = avg_freq
            config.compute.vehicle_cpu_freq_range = (avg_freq, avg_freq)
            # 鍚屾 scenario_config锛屼豢鐪熷櫒 override_topology=True 鏃剁洿鎺ヨ鍙栬繖浜涘€?
            scenario_config['total_vehicle_compute'] = total_compute
            scenario_config['vehicle_cpu_freq'] = avg_freq
            scenario_config['vehicle_default_freq'] = avg_freq
            scenario_config['vehicle_initial_freq'] = avg_freq
            print(f"馃敡 [Override] 鍔ㄦ€佽缃€绘湰鍦拌绠? {total_compute/1e9:.1f} GHz (姣忚溅{avg_freq/1e9:.3f} GHz)")

        if override_scenario is not None and 'total_rsu_compute' in override_scenario:
            total_compute = float(override_scenario['total_rsu_compute'])
            config.compute.total_rsu_compute = total_compute
            avg_freq = total_compute / config.num_rsus
            config.compute.rsu_initial_freq = avg_freq
            config.compute.rsu_default_freq = avg_freq
            config.compute.rsu_cpu_freq = avg_freq
            config.compute.rsu_cpu_freq_range = (avg_freq, avg_freq)
            scenario_config['total_rsu_compute'] = total_compute
            scenario_config['rsu_cpu_freq'] = avg_freq
            scenario_config['rsu_default_freq'] = avg_freq
            scenario_config['rsu_initial_freq'] = avg_freq
            print(f"馃敡 [Override] 鍔ㄦ€佽缃€籖SU璁＄畻: {total_compute/1e9:.1f} GHz (姣廟SU{avg_freq/1e9:.1f} GHz)")

        if override_scenario is not None and 'total_uav_compute' in override_scenario:
            total_compute = float(override_scenario['total_uav_compute'])
            config.compute.total_uav_compute = total_compute
            avg_freq = total_compute / config.num_uavs
            config.compute.uav_initial_freq = avg_freq
            config.compute.uav_default_freq = avg_freq
            config.compute.uav_cpu_freq = avg_freq
            config.compute.uav_cpu_freq_range = (avg_freq, avg_freq)
            scenario_config['total_uav_compute'] = total_compute
            scenario_config['uav_cpu_freq'] = avg_freq
            scenario_config['uav_default_freq'] = avg_freq
            scenario_config['uav_initial_freq'] = avg_freq
            print(f"馃敡 [Override] 鍔ㄦ€佽缃€籙AV璁＄畻: {total_compute/1e9:.1f} GHz (姣廢AV{avg_freq/1e9:.1f} GHz)")

        # CPU棰戠巼鍙傛暟锛堝崟鑺傜偣棰戠巼锛屽吋瀹规棫浠ｇ爜锛?
        if override_scenario is not None and 'vehicle_cpu_freq' in override_scenario and 'total_vehicle_compute' not in override_scenario:
            freq_value = override_scenario['vehicle_cpu_freq']
            # 鏇存柊鑼冨洿鍜岄粯璁ゅ€?
            config.compute.vehicle_cpu_freq_range = (freq_value, freq_value)
            config.compute.vehicle_cpu_freq = freq_value
            scenario_config['vehicle_cpu_freq'] = freq_value
            scenario_config.setdefault('vehicle_default_freq', freq_value)
            scenario_config.setdefault('vehicle_initial_freq', freq_value)
            print(f"馃敡 [Override] 鍔ㄦ€佽缃溅杈咰PU棰戠巼: {float(freq_value)/1e9:.2f} GHz")

        if override_scenario is not None and 'rsu_cpu_freq' in override_scenario and 'total_rsu_compute' not in override_scenario:
            freq_value = override_scenario['rsu_cpu_freq']
            config.compute.rsu_cpu_freq_range = (freq_value, freq_value)
            config.compute.rsu_cpu_freq = freq_value
            scenario_config['rsu_cpu_freq'] = freq_value
            scenario_config.setdefault('rsu_default_freq', freq_value)
            scenario_config.setdefault('rsu_initial_freq', freq_value)
            print(f"馃敡 [Override] 鍔ㄦ€佽缃甊SU CPU棰戠巼: {float(freq_value)/1e9:.2f} GHz")

        if override_scenario is not None and 'uav_cpu_freq' in override_scenario and 'total_uav_compute' not in override_scenario:
            freq_value = override_scenario['uav_cpu_freq']
            config.compute.uav_cpu_freq_range = (freq_value, freq_value)
            config.compute.uav_cpu_freq = freq_value
            scenario_config['uav_cpu_freq'] = freq_value
            scenario_config.setdefault('uav_default_freq', freq_value)
            scenario_config.setdefault('uav_initial_freq', freq_value)
            print(f"馃敡 [Override] 鍔ㄦ€佽缃甎AV CPU棰戠巼: {float(freq_value)/1e9:.2f} GHz")
            
            # 浠诲姟鏁版嵁澶у皬鍙傛暟
            if override_scenario is not None and ('task_data_size_min_kb' in override_scenario or 'task_data_size_max_kb' in override_scenario):
                min_kb = override_scenario.get('task_data_size_min_kb')
                max_kb = override_scenario.get('task_data_size_max_kb')
                if min_kb is not None and max_kb is not None:
                    # 杞崲涓哄瓧鑺?
                    min_bytes = float(min_kb) * 1024
                    max_bytes = float(max_kb) * 1024
                    config.task.data_size_range = (min_bytes, max_bytes)
                    config.task.task_data_size_range = (min_bytes, max_bytes)
                    print(f"馃敡 [Override] 鍔ㄦ€佽缃换鍔℃暟鎹ぇ灏? {min_kb}-{max_kb} KB")
            
            # 浠诲姟澶嶆潅搴﹀弬鏁?
            if override_scenario is not None and 'task_complexity_multiplier' in override_scenario:
                multiplier = override_scenario['task_complexity_multiplier']
                # 閫氳繃鐜鍙橀噺浼犻€掔粰TaskConfig
                os.environ['TASK_COMPLEXITY_MULTIPLIER'] = str(multiplier)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃换鍔″鏉傚害鍊嶆暟: {multiplier}x")
            
            if override_scenario is not None and 'task_compute_density' in override_scenario:
                density = override_scenario['task_compute_density']
                config.task.task_compute_density = int(float(density))  # type: ignore
                print(f"馃敡 [Override] 鍔ㄦ€佽缃换鍔¤绠楀瘑搴? {density} cycles/bit")
            
            # 缂撳瓨瀹归噺鍙傛暟
            if override_scenario is not None and 'cache_capacity' in override_scenario:
                capacity_mb = override_scenario['cache_capacity']
                # 閫氳繃鐜鍙橀噺浼犻€掞紙褰卞搷鎵€鏈夎妭鐐癸級
                os.environ['CACHE_CAPACITY_MB'] = str(capacity_mb)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃紦瀛樺閲? {capacity_mb} MB")

            # 鏈嶅姟鑳藉姏鍙傛暟
            if override_scenario is not None and 'rsu_base_service' in override_scenario:
                value = int(override_scenario['rsu_base_service'])
                config.service.rsu_base_service = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甊SU鍩虹鏈嶅姟鑳藉姏: {value}")
            if override_scenario is not None and 'rsu_max_service' in override_scenario:
                value = int(override_scenario['rsu_max_service'])
                config.service.rsu_max_service = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甊SU鏈€澶ф湇鍔¤兘鍔? {value}")
            if override_scenario is not None and 'rsu_work_capacity' in override_scenario:
                value = float(override_scenario['rsu_work_capacity'])
                config.service.rsu_work_capacity = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甊SU宸ヤ綔瀹归噺: {value}")
            if override_scenario is not None and 'uav_base_service' in override_scenario:
                value = int(override_scenario['uav_base_service'])
                config.service.uav_base_service = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甎AV鍩虹鏈嶅姟鑳藉姏: {value}")
            if override_scenario is not None and 'uav_max_service' in override_scenario:
                value = int(override_scenario['uav_max_service'])
                config.service.uav_max_service = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甎AV鏈€澶ф湇鍔¤兘鍔? {value}")
            if override_scenario is not None and 'uav_work_capacity' in override_scenario:
                value = float(override_scenario['uav_work_capacity'])
                config.service.uav_work_capacity = value
                print(f"馃敡 [Override] 鍔ㄦ€佽缃甎AV宸ヤ綔瀹归噺: {value}")
            
            # 浠诲姟鍒拌揪鐜囧弬鏁?
            if override_scenario is not None and 'task_arrival_rate' in override_scenario:
                arrival_rate = override_scenario['task_arrival_rate']
                config.task.arrival_rate = float(arrival_rate)
                # 鍚屾椂璁剧疆鐜鍙橀噺浠ュ吋瀹规棫浠ｇ爜
                os.environ['TASK_ARRIVAL_RATE'] = str(arrival_rate)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃换鍔″埌杈剧巼: {arrival_rate} tasks/s")
            
            # 鍗曚竴浠诲姟鏁版嵁澶у皬鍙傛暟锛堢敤浜庢贩鍚堣礋杞藉疄楠岋級
            if override_scenario is not None and 'task_data_size_kb' in override_scenario:
                size_kb = override_scenario['task_data_size_kb']
                size_bytes = float(size_kb) * 1024
                config.task.data_size_range = (size_bytes, size_bytes)
                config.task.task_data_size_range = (size_bytes, size_bytes)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃换鍔℃暟鎹ぇ灏? {size_kb} KB")
            
            # 閫氫俊鍙傛暟锛堝櫔澹板姛鐜囥€佽矾寰勬崯鑰楋級
            if override_scenario is not None and 'noise_power_dbm' in override_scenario:
                noise_power = override_scenario['noise_power_dbm']
                setattr(config.communication, 'noise_power_dbm', float(noise_power))  # type: ignore
                print(f"馃敡 [Override] 鍔ㄦ€佽缃櫔澹板姛鐜? {noise_power} dBm")
            
            if override_scenario is not None and 'path_loss_exponent' in override_scenario:
                exponent = override_scenario['path_loss_exponent']
                setattr(config.communication, 'path_loss_exponent', float(exponent))  # type: ignore
                print(f"馃敡 [Override] 鍔ㄦ€佽缃矾寰勬崯鑰楁寚鏁? {exponent}")
            
            # 璧勬簮寮傛瀯鎬у弬鏁?
            if override_scenario is not None and 'heterogeneity_level' in override_scenario:
                hetero_level = override_scenario['heterogeneity_level']
                os.environ['HETEROGENEITY_LEVEL'] = str(hetero_level)
                print(f"馃敡 [Override] 鍔ㄦ€佽缃祫婧愬紓鏋勬€х骇鍒? {hetero_level}")
        
        mode_aliases = {
            'local': 'local_only',
            'local_only': 'local_only',
            'remote': 'remote_only',
            'remote_only': 'remote_only',
            '': ''
        }
        forced_mode_input = (
            enforce_offload_mode
            or scenario_config.get('forced_offload_mode')
            or os.environ.get('FORCE_OFFLOAD_MODE', '')
        )
        requested_mode = mode_aliases.get(str(forced_mode_input).strip().lower(), '')
        if requested_mode not in {'', 'local_only', 'remote_only'}:
            print(f"鈿狅笍 鏈瘑鍒殑寮哄埗鍗歌浇妯″紡: {forced_mode_input}, 灏嗗拷鐣ャ€?)
            requested_mode = ''
        self.enforce_offload_mode = requested_mode
        if self.enforce_offload_mode:
            scenario_config['forced_offload_mode'] = self.enforce_offload_mode
            if self.enforce_offload_mode == 'remote_only':
                scenario_config.setdefault('allow_local_processing', False)
            elif self.enforce_offload_mode == 'local_only':
                scenario_config.setdefault('allow_local_processing', True)

        if self.enforce_offload_mode == 'local_only':
            print("馃Х 寮哄埗鍗歌浇妯″紡: 鍏ㄩ儴鏈湴澶勭悊锛圠ocal-Only锛?)
        elif self.enforce_offload_mode == 'remote_only':
            print("馃Х 寮哄埗鍗歌浇妯″紡: 鍏ㄩ儴杩滅鎵ц锛圧emote-Only锛?)
        
        # 馃幆 鍥哄畾鍗歌浇绛栫暐鍒濆鍖?
        self.fixed_offload_policy = None
        self.fixed_policy_name = None
        if fixed_offload_policy:
            try:
                import sys
                import importlib.util
                from pathlib import Path
                
                # 鍔ㄦ€佹坊鍔?experiments 鐩綍鍒?Python 璺緞
                exp_path = Path(__file__).parent / 'experiments'
                if str(exp_path) not in sys.path:
                    sys.path.insert(0, str(exp_path))
                
                # 浣跨敤 importlib 鍔ㄦ€佸鍏ユā鍧楋紙閬垮厤闈欐€佸垎鏋愯鍛婏級
                module_path = exp_path / 'fallback_baselines.py'
                if module_path.exists():
                    spec = importlib.util.spec_from_file_location("fallback_baselines", module_path)
                    if spec and spec.loader:
                        fallback_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(fallback_module)
                        create_baseline_algorithm = fallback_module.create_baseline_algorithm
                    else:
                        raise ImportError(f"鏃犳硶鍔犺浇妯″潡 {module_path}")
                else:
                    raise ImportError(f"妯″潡鏂囦欢涓嶅瓨鍦? {module_path}")
                
                self.fixed_offload_policy = create_baseline_algorithm(fixed_offload_policy)
                self.fixed_policy_name = fixed_offload_policy
                print(f"馃幉 鍥哄畾鍗歌浇绛栫暐: {fixed_offload_policy} (鍗歌浇鍐崇瓥涓嶇敱鏅鸿兘浣撳涔?")
                print(f"   鍏朵粬鍐崇瓥锛堢紦瀛樸€佽縼绉汇€佽祫婧愬垎閰嶏級浠嶇敱鏅鸿兘浣撳涔?)
            except Exception as e:
                print(f"鈿狅笍  鏃犳硶鍒涘缓鍥哄畾绛栫暐 '{fixed_offload_policy}': {e}")
                print(f"   灏嗕娇鐢ㄦ櫤鑳戒綋瀛︿範鍗歌浇鍐崇瓥")
                self.fixed_offload_policy = None
        
        # 閫夋嫨浠跨湡鍣ㄧ被鍨?
        self.use_enhanced_cache = use_enhanced_cache and ENHANCED_CACHE_AVAILABLE
        env_disable_migration = os.environ.get("DISABLE_MIGRATION", "").strip() == "1"
        self.disable_migration = disable_migration or env_disable_migration
        
        # 馃敡 鏂板锛氬鏋滄湭閫氳繃override璁剧疆鐩爣鍊硷紝鏍规嵁褰撳墠杞﹁締鏁拌嚜鍔ㄨ皟鏁?
        if 'num_vehicles' not in (override_scenario or {}):
            current_num_vehicles = scenario_config.get('num_vehicles', config.num_vehicles)
            if os.environ.get('RL_LATENCY_TARGET') is None:
                auto_latency_target = 0.5 + current_num_vehicles * 0.15
                config.rl.latency_target = auto_latency_target
                config.rl.latency_upper_tolerance = auto_latency_target * 2.5
                print(f"馃幆 鑷姩璋冩暣 latency_target: {auto_latency_target:.2f}s (鍩轰簬{current_num_vehicles}杈嗚溅)")
            
            if os.environ.get('RL_ENERGY_TARGET') is None:
                auto_energy_target = current_num_vehicles * 800.0
                config.rl.energy_target = auto_energy_target
                config.rl.energy_upper_tolerance = auto_energy_target * 2.0
                print(f"馃幆 鑷姩璋冩暣 energy_target: {auto_energy_target:.0f}J (鍩轰簬{current_num_vehicles}杈嗚溅)")
            
            # 鍚屾鍒板叏灞€濂栧姳璁＄畻鍣?
            try:
                from utils.unified_reward_calculator import update_reward_targets
                update_reward_targets(
                    latency_target=float(config.rl.latency_target),
                    energy_target=float(config.rl.energy_target)
                )
            except Exception:
                pass
        
        simulator: CompleteSystemSimulator
        if self.use_enhanced_cache:
            print("馃殌 [Training] Using Enhanced Cache System (Default) with:")
            print("   - Hierarchical L1/L2 caching (3GB + 7GB)")
            print("   - Adaptive HeatBasedCacheStrategy")
            print("   - Inter-RSU collaboration")
            simulator = EnhancedSystemSimulator(scenario_config)  # type: ignore[assignment]
        else:
            simulator = CompleteSystemSimulator(scenario_config)
        self.simulator: CompleteSystemSimulator = simulator
        
        # 馃 鍒濆鍖栬嚜閫傚簲鎺у埗缁勪欢
        self.adaptive_cache_controller = AdaptiveCacheController()
        self.adaptive_migration_controller = AdaptiveMigrationController()
        if self.disable_migration:
            print("馃 鑷€傚簲缂撳瓨宸插惎鐢紱杩佺Щ鎺у埗宸茬鐢紙DISABLE_MIGRATION 妯″紡锛?)
        else:
            print(f"馃 宸插惎鐢ㄨ嚜閫傚簲缂撳瓨鍜岃縼绉绘帶鍒跺姛鑳?)

        self.strategy_coordinator = StrategyCoordinator(
            self.adaptive_cache_controller,
            None if self.disable_migration else self.adaptive_migration_controller
        )
        self.strategy_coordinator.register_simulator(self.simulator)
        setattr(self.simulator, 'strategy_coordinator', self.strategy_coordinator)
        
        # 浠庝豢鐪熷櫒鑾峰彇瀹為檯缃戠粶鎷撴墤鍙傛暟
        num_vehicles = len(self.simulator.vehicles)
        num_rsus = len(self.simulator.rsus)
        num_uavs = len(self.simulator.uavs)
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
        # 馃幆 鏇存柊鍥哄畾绛栫暐鐨勭幆澧冧俊鎭?
        if self.fixed_offload_policy is not None:
            try:
                # 鍒涘缓涓€涓畝鍖栫殑鐜瀵硅薄渚涘浐瀹氱瓥鐣ヤ娇鐢?
                class SimpleEnv:
                    def __init__(self, simulator):
                        self.simulator = simulator
                        self.agent_env = type('obj', (object,), {
                            'action_dim': 18,  # 榛樿action缁村害
                        })()
                
                simple_env = SimpleEnv(self.simulator)
                self.fixed_offload_policy.update_environment(simple_env)
                print(f"   鍥哄畾绛栫暐宸叉洿鏂扮幆澧冧俊鎭? {num_vehicles}杞﹁締, {num_rsus}RSU, {num_uavs}UAV")
            except Exception as e:
                print(f"鈿狅笍  鍥哄畾绛栫暐鏇存柊鐜澶辫触: {e}")
        
        # 搴旂敤鍥哄畾鎷撴墤鐨勫弬鏁颁紭鍖栵紙淇濇寔4 RSU + 2 UAV锛?
        if self.algorithm in {"TD3", "TD3_LATENCY_ENERGY"}:
            topology_optimizer = FixedTopologyOptimizer()
            opt_params = topology_optimizer.get_optimized_params(num_vehicles)
            
            # 搴旂敤浼樺寲鐨勮秴鍙傛暟鍒癟D3閰嶇疆
            os.environ['TD3_HIDDEN_DIM'] = str(opt_params.get('hidden_dim', 400))
            os.environ['TD3_ACTOR_LR'] = str(opt_params.get('actor_lr', 1e-4))
            os.environ['TD3_CRITIC_LR'] = str(opt_params.get('critic_lr', 8e-5))
            os.environ['TD3_BATCH_SIZE'] = str(opt_params.get('batch_size', 256))
            
            print(f"[FIXED-TOPOLOGY] 杞﹁締鏁?{num_vehicles} 鈫?Hidden:{opt_params['hidden_dim']}, LR:{opt_params['actor_lr']:.1e}, Batch:{opt_params['batch_size']}")
            print(f"[FIXED-TOPOLOGY] 淇濇寔鍥哄畾: RSU=4, UAV=2锛堥獙璇佺畻娉曠瓥鐣ユ湁鏁堟€э級")
        
        # 馃敡 浼樺寲锛氭墍鏈夌畻娉曠粺涓€浼犲叆鎷撴墤鍙傛暟锛屽疄鐜板姩鎬侀€傞厤
        if self.algorithm == "DDPG":
            self.agent_env = DDPGEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "TD3":
            # TD3榛樿鍚敤涓ぎ璧勬簮妯″紡锛堝彲閫氳繃鐜鍙橀噺CENTRAL_RESOURCE=0绂佺敤锛?
            if not self.central_resource_enabled:
                central_env_override = os.environ.get('CENTRAL_RESOURCE', '1')  # 榛樿鍚敤
                self.central_resource_enabled = central_env_override.strip() in {'1', 'true', 'True'}
            self.agent_env = TD3Environment(
                num_vehicles,
                num_rsus,
                num_uavs,
                use_central_resource=self.central_resource_enabled,
            )
        elif self.algorithm == "TD3_LATENCY_ENERGY":
            self.agent_env = TD3LatencyEnergyEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "CAM_TD3":
            # CAM_TD3榛樿鍚敤涓ぎ璧勬簮妯″紡锛堝彲閫氳繃鐜鍙橀噺CENTRAL_RESOURCE=0绂佺敤锛?
            if not self.central_resource_enabled:
                central_env_override = os.environ.get('CENTRAL_RESOURCE', '1')  # 榛樿鍚敤
                self.central_resource_enabled = central_env_override.strip() in {'1', 'true', 'True'}
            self.agent_env = CAMTD3Environment(
                num_vehicles, num_rsus, num_uavs,
                use_central_resource=self.central_resource_enabled
            )
        elif self.algorithm == "DQN":
            self.agent_env = DQNEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "PPO":
            self.agent_env = PPOEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "SAC":
            self.agent_env = SACEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "OPTIMIZED_TD3":
            # 绮剧畝浼樺寲TD3 (Queue-aware Replay + GNN Attention)
            self.agent_env = OptimizedTD3Environment(
                num_vehicles,
                num_rsus,
                num_uavs,
                use_central_resource=self.central_resource_enabled
            )
            print(f"[OptimizedTD3] 浣跨敤绮剧畝浼樺寲閰嶇疆 (Queue+GNN)")
        else:
            raise ValueError(f"涓嶆敮鎸佺殑绠楁硶: {algorithm}")

        # 馃幆 涓ぎ璧勬簮鍒嗛厤妯″紡鏃ュ織
        import sys
        print(f"\n[璧勬簮鍒嗛厤妯″紡妫€鏌", file=sys.stderr)
        print(f"  CENTRAL_RESOURCE 鐜鍙橀噺: '{central_env_value}'", file=sys.stderr)
        print(f"  use_central_resource: {self.central_resource_enabled}", file=sys.stderr)
        
        self.central_resource_action_dim = getattr(self.agent_env, 'central_resource_action_dim', 0)
        self.central_resource_state_dim = getattr(self.agent_env, 'central_state_dim', 0)
        self.base_action_dim = getattr(self.agent_env, 'base_action_dim', getattr(self.agent_env, 'action_dim', 0) - self.central_resource_action_dim)
        
        if self.central_resource_enabled and self.central_resource_action_dim > 0:
            print(f"鉁?鍚敤涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛歅hase 1(鍐崇瓥) + Phase 2(鎵ц)", file=sys.stderr)
            print(f"   鐜绫诲瀷: {type(self.agent_env).__name__}", file=sys.stderr)
            print(f"   鍩虹鍔ㄤ綔缁村害: {self.base_action_dim}", file=sys.stderr)
            print(f"   涓ぎ璧勬簮鍔ㄤ綔缁村害: {self.central_resource_action_dim}", file=sys.stderr)
            if self.central_resource_state_dim:
                print(f"   鐘舵€佹墿灞曠淮搴? +{self.central_resource_state_dim}", file=sys.stderr)
        else:
            print(f"  浣跨敤鏍囧噯妯″紡锛堝潎鍖€璧勬簮鍒嗛厤锛?, file=sys.stderr)
        
        # 馃 鑻ユ寚瀹氫簡闃舵涓€绠楁硶锛堥€氳繃鐜鍙橀噺锛夛紝鐢―ualStage灏佽鍣ㄧ粍鍚堜袱涓樁娈?
        stage1_alg = os.environ.get('STAGE1_ALG', '').strip().lower()
        if stage1_alg:
            try:
                from single_agent.dual_stage_controller import DualStageControllerEnv
                self.agent_env = DualStageControllerEnv(self.agent_env, self.simulator, stage1_strategy=stage1_alg)
                print(f"馃 鍚敤涓ら樁娈垫帶鍒讹細Stage1={stage1_alg} + Stage2={self.algorithm}")
                # Two-stage planner inside simulator becomes redundant
                os.environ['TWO_STAGE_MODE'] = '0'
            except Exception as e:
                print(f"鈿狅笍 涓ら樁娈垫帶鍒跺皝瑁呭け璐ワ紝鍥為€€鍒板崟绠楁硶: {e}")
        
        # 璁粌缁熻
        self.episode_rewards = []
        self.episode_losses = {}
        self.episode_metrics = {
            'avg_delay': [],
            'total_energy': [],
            'data_loss_bytes': [],
            'data_loss_ratio_bytes': [],
            'task_completion_rate': [],
            'cache_hit_rate': [],
            'cache_utilization': [],
            'cache_evictions': [],
            'cache_eviction_rate': [],
            'cache_requests': [],
            'cache_collaborative_writes': [],
            'local_cache_hits': [],
            'migration_avg_cost': [],
            'migration_avg_delay_saved': [],
            'migration_success_rate': [],
            'queue_rho_sum': [],
            'queue_rho_max': [],
            'queue_overload_flag': [],  # 馃敡 淇锛氱‘淇濊褰曚簩鍊艰繃杞芥爣蹇?
            'queue_overload_events': [],  # 馃敡 淇锛氳褰曠疮璁¤繃杞戒簨浠舵暟
            'episode_steps': [],  # 馃敡 鏂板锛氳褰曟瘡涓猠pisode鐨勫疄闄呮鏁?
            'avg_step_reward': [],  # 馃敡 淇锛氳褰曞钩鍧囨瘡姝ュ鍔?
            'task_type_queue_share_1': [],
            'task_type_queue_share_2': [],
            'task_type_queue_share_3': [],
            'task_type_queue_share_4': [],
            'task_type_deadline_norm_1': [],
            'task_type_deadline_norm_2': [],
            'task_type_deadline_norm_3': [],
            'task_type_deadline_norm_4': [],
            'task_type_drop_rate_1': [],
            'task_type_drop_rate_2': [],
            'task_type_drop_rate_3': [],
            'task_type_drop_rate_4': [],
            'task_type_queue_share_ep_1': [],
            'task_type_queue_share_ep_2': [],
            'task_type_queue_share_ep_3': [],
            'task_type_queue_share_ep_4': [],
            'rsu_hotspot_mean': [],  # 馃敡 淇锛氳褰曟瘡涓猠pisode鐨凴SU鐑偣骞冲潎寮哄害
            'rsu_hotspot_peak': [],  # 馃敡 淇锛氳褰曟瘡涓猠pisode鐨凴SU鐑偣宄板€煎己搴?
            'rsu_hotspot_mean_series': [],
            'rsu_hotspot_peak_series': [],
            'mm1_queue_error': [],
            'mm1_delay_error': [],
            'normalized_delay': [],
            'normalized_energy': [],
            'normalized_reward': [],
            # 馃幆 鏂板锛歊SU璧勬簮鍒╃敤鐜囧拰鍗歌浇鐜囩粺璁★紙淇bug锛?
            'rsu_utilization': [],
            'offload_ratio': [],  # remote_execution_ratio (rsu+uav)
            'rsu_offload_ratio': [],
            'uav_offload_ratio': [],
            'local_offload_ratio': [],
            # 馃殌 鏂板锛氳縼绉昏兘鑰楁寚鏍?
            'rsu_migration_energy': [],
            'uav_migration_energy': [],
        }
        
        # 鎬ц兘杩借釜鍣?
        self.performance_tracker = {
            'recent_rewards': MovingAverage(100),
            'recent_step_rewards': MovingAverage(100),
            'recent_delays': MovingAverage(100),
            'recent_energy': MovingAverage(100),
            'recent_completion': MovingAverage(100)
        }
        self._reward_baseline: Dict[str, float] = {}
        self._energy_target_per_vehicle = float(os.environ.get('ENERGY_TARGET_PER_VEHICLE', 180.0))
        self._dynamic_energy_target = float(getattr(config.rl, 'energy_target', 2200.0))
        heuristic_energy_target = max(
            self._dynamic_energy_target,
            self.num_vehicles * self._energy_target_per_vehicle
        )
        if heuristic_energy_target > self._dynamic_energy_target * 1.02:
            self._dynamic_energy_target = heuristic_energy_target
            update_reward_targets(energy_target=heuristic_energy_target)
            print(
                f"鈿栵笍 鍔ㄦ€佽皟鏁磋兘鑰楃洰鏍? {heuristic_energy_target:.1f}J "
                f"(杞﹁締鏁?{self.num_vehicles}, 姣忚溅棰勭畻={self._energy_target_per_vehicle:.1f}J)"
            )
        self._energy_target_ema = self._dynamic_energy_target
        self._energy_target_warmup = max(40, int(config.experiment.num_episodes * 0.1))
        self._last_energy_target_update = 0
        # 鑷€傚簲寤惰繜鐩爣锛堥珮璐熻浇鍦烘櫙鑷姩鏀惧锛岄伩鍏嶅鍔遍ケ鍜岋級
        self._dynamic_latency_target = float(getattr(config.rl, 'latency_target', 0.4))
        self._delay_target_ema = self._dynamic_latency_target
        self._last_delay_target_update = 0
        self._reward_smoothing_alpha = float(getattr(config.rl, 'reward_smooth_alpha', 0.35))
        self._reward_ema_delay: Optional[float] = None
        self._reward_ema_energy: Optional[float] = None
        self._episode_counters_initialized = False
        
        print(f"鉁?{self.algorithm}璁粌鐜鍒濆鍖栧畬鎴?)
        print(f"鉁?绠楁硶绫诲瀷: 鍗曟櫤鑳戒綋")
    
    def _calculate_correct_cache_utilization(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        馃敡 淇锛氭纭绠楃紦瀛樺埄鐢ㄧ巼
        
        Args:
            cache: 缂撳瓨瀛楀吀
            cache_capacity_mb: 缂撳瓨瀹归噺(MB)
        Returns:
            缂撳瓨鍒╃敤鐜?[0.0, 1.0]
        """
        if not cache or cache_capacity_mb <= 0:
            return 0.0
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                # 鍏煎鏃ф牸寮忥紝浣跨敤realistic澶у皬
                total_used_mb += 1.0  # 榛樿1MB
        
        utilization = total_used_mb / cache_capacity_mb
        return min(1.0, max(0.0, utilization))
    
    def _initialize_episode_counters(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """Reset per-episode baseline counters to avoid carrying over cumulative stats."""
        stats_dict: Dict[str, Any]
        if stats is None:
            stats_dict = {}
        else:
            try:
                stats_dict = dict(stats)
            except Exception:
                stats_dict = {}

        self._episode_energy_base = float(stats_dict.get('total_energy', 0.0) or 0.0)
        self._episode_processed_base = int(stats_dict.get('processed_tasks', 0) or 0)
        self._episode_dropped_base = int(stats_dict.get('dropped_tasks', 0) or 0)
        self._episode_generated_bytes_base = float(stats_dict.get('generated_data_bytes', 0.0) or 0.0)
        self._episode_dropped_bytes_base = float(stats_dict.get('dropped_data_bytes', 0.0) or 0.0)
        remote_stats = stats_dict.get('remote_rejections', {})
        if isinstance(remote_stats, dict):
            self._episode_remote_reject_base = int(remote_stats.get('total', 0) or 0)
        else:
            self._episode_remote_reject_base = 0

        # Cache controllers keep their own cumulative counters; snapshot them as the new baseline
        if hasattr(self, 'adaptive_cache_controller'):
            cache_metrics = self.adaptive_cache_controller.get_cache_metrics()
            self._episode_cache_requests_base = int(cache_metrics.get('total_requests', 0) or 0)
            self._episode_cache_evictions_base = int(cache_metrics.get('evicted_items', 0) or 0)
            self._episode_cache_collab_base = int(cache_metrics.get('collaborative_writes', 0) or 0)
        else:
            self._episode_cache_requests_base = 0
            self._episode_cache_evictions_base = 0
            self._episode_cache_collab_base = 0

        self._episode_queue_overload_events_base = int(stats_dict.get('queue_overload_events', 0) or 0)
        delay_buckets = ('delay_processing', 'delay_uplink', 'delay_downlink', 'delay_cache', 'delay_waiting')
        energy_buckets = ('energy_compute', 'energy_transmit_uplink', 'energy_transmit_downlink', 'energy_cache')
        self._episode_delay_component_base = {
            bucket: float(stats_dict.get(bucket, 0.0) or 0.0) for bucket in delay_buckets
        }
        self._episode_energy_component_base = {
            bucket: float(stats_dict.get(bucket, 0.0) or 0.0) for bucket in energy_buckets
        }
        self._episode_queue_overflow_base = int(stats_dict.get('queue_overflow_drops', 0) or 0)
        
        # 馃敡 淇锛氭坊鍔犲欢杩熸€婚噺鍩虹嚎锛岀敤浜庤绠梕pisode骞冲潎寤惰繜
        self._episode_delay_base = float(stats_dict.get('total_delay', 0.0) or 0.0)
        
        # Initialize task count accumulators
        self._episode_local_tasks = 0
        self._episode_rsu_tasks = 0
        self._episode_uav_tasks = 0
        
        self._episode_counters_initialized = True

    def _reset_reward_baseline(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """鍒濆鍖?閲嶇疆濂栧姳澧為噺鍩虹嚎銆?""
        def _safe_scalar(value: Any, default: float = 0.0) -> float:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return default
            return val if np.isfinite(val) else default

        def _safe_int(value: Any, default: int = 0) -> int:
            try:
                val = float(value)
            except (TypeError, ValueError):
                return default
            return int(val) if np.isfinite(val) else default

        base = stats or {}
        self._reward_baseline = {
            'processed': _safe_int(base.get('processed_tasks', 0)),
            'dropped': _safe_int(base.get('dropped_tasks', 0)),
            'delay': _safe_scalar(base.get('total_delay', 0.0)),
            'energy': _safe_scalar(base.get('total_energy', 0.0)),
            'generated_bytes': _safe_scalar(base.get('generated_data_bytes', 0.0)),
            'dropped_bytes': _safe_scalar(base.get('dropped_data_bytes', 0.0)),
        }
        self._reward_ema_delay = None
        self._reward_ema_energy = None

    def _build_reward_snapshot(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """鍩轰簬绱缁熻璁＄畻鍗曟濂栧姳鎵€闇€鐨勫閲忔寚鏍囥€?""
        safe_scalar = lambda v, d=0.0: float(v) if isinstance(v, (int, float, np.floating, np.integer)) and np.isfinite(float(v)) else d  # type: ignore[arg-type]
        def safe_int(v: Any, default: int = 0) -> int:
            try:
                val = float(v)
            except (TypeError, ValueError):
                return default
            return int(val) if np.isfinite(val) else default

        raw_baseline = getattr(self, '_reward_baseline', None) or {}
        baseline = {
            'processed': safe_int(raw_baseline.get('processed', 0)),
            'dropped': safe_int(raw_baseline.get('dropped', 0)),
            'delay': safe_scalar(raw_baseline.get('delay', 0.0)),
            'energy': safe_scalar(raw_baseline.get('energy', 0.0)),
            'generated_bytes': safe_scalar(raw_baseline.get('generated_bytes', 0.0)),
            'dropped_bytes': safe_scalar(raw_baseline.get('dropped_bytes', 0.0)),
        }

        total_processed = safe_int(stats.get('processed_tasks', 0))
        total_dropped = safe_int(stats.get('dropped_tasks', 0))
        total_delay = safe_scalar(stats.get('total_delay', 0.0))
        total_energy = safe_scalar(stats.get('total_energy', 0.0))
        total_generated = safe_scalar(stats.get('generated_data_bytes', 0.0))
        total_dropped_bytes = safe_scalar(stats.get('dropped_data_bytes', 0.0))

        delta_processed = max(0, total_processed - baseline['processed'])
        delta_dropped = max(0, total_dropped - baseline['dropped'])
        delta_delay = max(0.0, total_delay - baseline['delay'])
        delta_energy = max(0.0, total_energy - baseline['energy'])
        delta_generated = max(0.0, total_generated - baseline['generated_bytes'])
        delta_loss_bytes = max(0.0, total_dropped_bytes - baseline['dropped_bytes'])

        tasks_for_delay = delta_processed if delta_processed > 0 else max(1, total_processed)
        avg_delay_increment = delta_delay / max(1, tasks_for_delay)

        completion_total = delta_processed + delta_dropped
        completion_rate = normalize_ratio(delta_processed, completion_total, default=1.0)
        loss_ratio = normalize_ratio(delta_loss_bytes, delta_generated)

        avg_delay_increment = safe_scalar(avg_delay_increment)
        avg_delay_for_reward = avg_delay_increment if avg_delay_increment > 0 else safe_scalar(stats.get('avg_task_delay', 0.0))
        energy_per_task = delta_energy / max(1, delta_processed) if delta_processed > 0 else 0.0
        energy_per_task = safe_scalar(energy_per_task)
        smoothed_delay, smoothed_energy_per_task = self._apply_reward_smoothing(
            avg_delay_for_reward,
            energy_per_task
        )
        smoothed_energy_total = smoothed_energy_per_task * max(1, delta_processed)

        reward_snapshot = {
            'avg_task_delay': smoothed_delay,
            'total_energy_consumption': smoothed_energy_total if smoothed_energy_total > 0 else safe_scalar(stats.get('total_energy_consumption', 0.0)),
            'dropped_tasks': delta_dropped,
            'task_completion_rate': completion_rate,
            'data_loss_bytes': delta_loss_bytes,
            'data_loss_ratio_bytes': loss_ratio,
        }

        self._reward_baseline = {
            'processed': total_processed,
            'dropped': total_dropped,
            'delay': total_delay,
            'energy': total_energy,
            'generated_bytes': total_generated,
            'dropped_bytes': total_dropped_bytes,
        }

        return reward_snapshot

    def _apply_reward_smoothing(self, delay_value: float, energy_per_task: float) -> Tuple[float, float]:
        """瀵瑰鍔卞叧閿寚鏍囪繘琛屾寚鏁板钩婊戯紝鍑忓皬TD3璁粌鍣０銆?""
        delay_value = float(delay_value) if np.isfinite(delay_value) else 0.0
        energy_per_task = float(energy_per_task) if np.isfinite(energy_per_task) else 0.0
        if self._reward_smoothing_alpha <= 0.0:
            return delay_value, energy_per_task
        alpha = self._reward_smoothing_alpha
        if self._reward_ema_delay is None or not np.isfinite(self._reward_ema_delay):
            self._reward_ema_delay = delay_value
        else:
            self._reward_ema_delay = (1.0 - alpha) * self._reward_ema_delay + alpha * delay_value
            if not np.isfinite(self._reward_ema_delay):
                self._reward_ema_delay = delay_value
        if self._reward_ema_energy is None or not np.isfinite(self._reward_ema_energy):
            self._reward_ema_energy = energy_per_task
        else:
            self._reward_ema_energy = (1.0 - alpha) * self._reward_ema_energy + alpha * energy_per_task
            if not np.isfinite(self._reward_ema_energy):
                self._reward_ema_energy = energy_per_task
        return self._reward_ema_delay, self._reward_ema_energy

    def _maybe_update_dynamic_energy_target(self, episode: int, episode_energy: float) -> None:
        """鏍规嵁瀹為檯鑳借€楄嚜鍔ㄦ斁瀹界洰鏍囷紝閬垮厤涓嶅彲杈剧害鏉熷鑷存尟鑽°€?""
        if os.environ.get('DYNAMIC_TARGET_DISABLE', '0') != '0':
            return
        if episode_energy <= 0:
            return
        decay = 0.9
        self._energy_target_ema = decay * self._energy_target_ema + (1.0 - decay) * episode_energy
        if episode < self._energy_target_warmup:
            return
        if episode - self._last_energy_target_update < 5:
            return
        target = self._dynamic_energy_target
        ema = self._energy_target_ema
        if ema > target * 1.2:
            new_target = min(ema * 0.95, target * 3.0)
            self._dynamic_energy_target = new_target
            self._last_energy_target_update = episode
            update_reward_targets(energy_target=new_target)
            print(
                f"[DynamicTarget] Energy EMA {ema:.1f}J > target {target:.1f}J -> {new_target:.1f}J (Episode {episode})"
            )

    def _maybe_update_dynamic_latency_target(self, episode: int, episode_delay: float) -> None:
        """鏍规嵁瀹為檯鏃跺欢鑷姩鏀惧鐩爣锛岄伩鍏嶉珮璐熻浇鍦烘櫙濂栧姳楗卞拰銆?""
        if os.environ.get('DYNAMIC_TARGET_DISABLE', '0') != '0':
            return
        if episode_delay <= 0:
            return
        decay = 0.9
        self._delay_target_ema = decay * self._delay_target_ema + (1.0 - decay) * episode_delay
        warmup = max(20, int(config.experiment.num_episodes * 0.05))
        if episode < warmup:
            return
        if episode - self._last_delay_target_update < 5:
            return
        target = self._dynamic_latency_target
        ema = self._delay_target_ema
        if ema > target * 1.2:
            new_target = min(ema * 0.95, target * 3.0)
            self._dynamic_latency_target = new_target
            self._last_delay_target_update = episode
            update_reward_targets(latency_target=new_target)
            print(
                f"[DynamicTarget] Delay EMA {ema:.3f}s > target {target:.3f}s -> {new_target:.3f}s (Episode {episode})"
            )

    def reset_environment(self) -> np.ndarray:
        """閲嶇疆鐜骞惰繑鍥炲垵濮嬬姸鎬?""
        # 閲嶇疆浠跨湡鍣ㄧ姸鎬?
        self._episode_counters_initialized = False
        self.simulator._setup_scenario()
        
        # 鏀堕泦绯荤粺鐘舵€?
        node_states = {}
        
        # 杞﹁締鐘舵€侊紙涓巗tep淇濇寔涓€鑷寸殑褰掍竴鍖栨柟寮忥級
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                normalize_scalar(vehicle['position'][0], 'vehicle_position_range', 2060.0),
                normalize_scalar(vehicle['position'][1], 'vehicle_position_range', 2060.0),
                normalize_scalar(vehicle.get('velocity', 0.0), 'vehicle_speed_range', 50.0),
                normalize_scalar(len(vehicle.get('tasks', [])), 'vehicle_queue_capacity', 20.0),
                normalize_scalar(vehicle.get('energy_consumed', 0.0), 'vehicle_energy_reference', 1000.0),
            ])
            node_states[f'vehicle_{i}'] = vehicle_state

        # RSU鐘舵€侊紙缁熶竴褰掍竴鍖?瑁佸壀锛?
        for i, rsu in enumerate(self.simulator.rsus):
            # 馃敡 淇锛氭坊鍔燙PU棰戠巼鐗瑰緛锛岃鏅鸿兘浣撶煡閬揜SU鐨勮绠楀閲忎紭鍔?
            cpu_freq_norm = normalize_scalar(rsu.get('cpu_freq', 12.5e9), 'cpu_frequency_range', 20e9)  # 褰掍竴鍖栧埌[0,1]
            rsu_state = np.array([
                normalize_scalar(rsu['position'][0], 'rsu_position_range', 2060.0),
                normalize_scalar(rsu['position'][1], 'rsu_position_range', 2060.0),
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),
                normalize_scalar(len(rsu.get('computation_queue', [])), 'rsu_queue_capacity', 20.0),
                normalize_scalar(rsu.get('energy_consumed', 0.0), 'rsu_energy_reference', 1000.0),
                cpu_freq_norm,  # 馃敡 鏂板锛氱6缁?- CPU棰戠巼 (RSU绾?2.5GHz/20GHz=0.625)
            ])
            node_states[f'rsu_{i}'] = rsu_state

        # UAV鐘舵€侊紙缁熶竴褰掍竴鍖?瑁佸壀锛?
        for i, uav in enumerate(self.simulator.uavs):
            # 馃敡 淇锛氭坊鍔燙PU棰戠巼鐗瑰緛锛岃鏅鸿兘浣撶煡閬揢AV鐨勮绠楀閲忕浉瀵硅緝寮?
            cpu_freq_norm = normalize_scalar(uav.get('cpu_freq', 5.0e9), 'cpu_frequency_range', 20e9)  # 褰掍竴鍖栧埌[0,1]
            uav_state = np.array([
                normalize_scalar(uav['position'][0], 'uav_position_range', 2060.0),
                normalize_scalar(uav['position'][1], 'uav_position_range', 2060.0),
                normalize_scalar(uav['position'][2], 'uav_altitude_range', 200.0),
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),
                normalize_scalar(uav.get('energy_consumed', 0.0), 'uav_energy_reference', 1000.0),
                cpu_freq_norm,  # 馃敡 鏂板锛氱6缁?- CPU棰戠巼 (UAV绾?.0GHz/20GHz=0.25)
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 鍒濆绯荤粺鎸囨爣
        system_metrics = {
            'avg_task_delay': 0.0,
            'total_energy_consumption': 0.0,
            'data_loss_bytes': 0.0,
            'data_loss_ratio_bytes': 0.0,
            'cache_hit_rate': 0.0,
            'migration_success_rate': 0.0
        }
        
        # 馃敡 淇锛氶噸缃兘鑰楄拷韪櫒锛岄伩鍏嶈法episode绱Н
        if hasattr(self, '_last_total_energy'):
            delattr(self, '_last_total_energy')

        # 鑾峰彇鍒濆鐘舵€佸悜閲?
        if isinstance(self.agent_env, (TD3Environment, TD3LatencyEnergyEnvironment, CAMTD3Environment)):
            state = self.agent_env.get_state_vector(node_states, system_metrics, {'vehicles': [], 'rsus': [], 'uavs': []})  # type: ignore[call-arg]
        else:
            state = self.agent_env.get_state_vector(node_states, system_metrics)  # type: ignore[call-arg]
        
        return state
    
    def step(self, action, state, actions_dict: Optional[Dict] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """鎵ц涓€姝ヤ豢鐪燂紝搴旂敤鏅鸿兘浣撳姩浣滃埌浠跨湡鍣?""
        # 馃幆 浣跨敤鍥哄畾鍗歌浇绛栫暐锛堝鏋滆缃級
        if self.fixed_offload_policy is not None and actions_dict is not None:
            try:
                # 浣跨敤鍥哄畾绛栫暐鐢熸垚鍗歌浇鍐崇瓥
                fixed_action = self.fixed_offload_policy.select_action(state)
                
                # 灏嗗浐瀹氱瓥鐣ョ殑action杞崲涓簅ffload preference
                # 鍥哄畾绛栫暐杩斿洖鐨刟ction鏍煎紡: [local_score, rsu_score, uav_score, ...]
                if isinstance(fixed_action, np.ndarray) and len(fixed_action) >= 3:
                    local_pref = float(fixed_action[0])
                    rsu_pref = float(fixed_action[1])
                    uav_pref = float(fixed_action[2])
                    
                    # 褰掍竴鍖栦负姒傜巼鍒嗗竷
                    total = abs(local_pref) + abs(rsu_pref) + abs(uav_pref)
                    if total > 1e-6:
                        local_pref = abs(local_pref) / total
                        rsu_pref = abs(rsu_pref) / total
                        uav_pref = abs(uav_pref) / total
                    else:
                        local_pref, rsu_pref, uav_pref = 0.33, 0.33, 0.34
                    
                    # 瑕嗙洊鏅鸿兘浣撶殑鍗歌浇鍐崇瓥锛屼繚鐣欏叾浠栧喅绛栵紙缂撳瓨銆佽縼绉荤瓑锛?
                    if 'offload_preference' in actions_dict:
                        actions_dict['offload_preference'] = {
                            'local': local_pref,
                            'rsu': rsu_pref,
                            'uav': uav_pref
                        }
            except Exception as e:
                # 濡傛灉鍥哄畾绛栫暐澶辫触锛屽洖閫€鍒版櫤鑳戒綋鍐崇瓥
                pass
        
        # 鏋勯€犱紶閫掔粰浠跨湡鍣ㄧ殑鍔ㄤ綔锛堝皢杩炵画鍔ㄤ綔鏄犲皠涓烘湰鍦?RSU/UAV鍋忓ソ锛?
        sim_actions = self._build_simulator_actions(actions_dict)
        
        # 鎵ц浠跨湡姝ラ锛堜紶鍏ュ姩浣滐級
        step_stats = self.simulator.run_simulation_step(0, sim_actions)
        resource_state = self._collect_resource_state()
        
        # 鏀堕泦涓嬩竴姝ョ姸鎬?
        node_states = {}
        
        # 杞﹁締鐘舵€?(5缁?- 缁熶竴褰掍竴鍖?
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                normalize_scalar(vehicle['position'][0], 'vehicle_position_range', 2060.0),  # 浣嶇疆x
                normalize_scalar(vehicle['position'][1], 'vehicle_position_range', 2060.0),  # 浣嶇疆y
                normalize_scalar(vehicle.get('velocity', 0.0), 'vehicle_speed_range', 50.0),  # 閫熷害
                normalize_scalar(len(vehicle.get('tasks', [])), 'vehicle_queue_capacity', 20.0),  # 闃熷垪
                normalize_scalar(vehicle.get('energy_consumed', 0.0), 'vehicle_energy_reference', 1000.0),  # 鑳借€?
            ])
            node_states[f'vehicle_{i}'] = vehicle_state

        # RSU鐘舵€?(6缁?- 娣诲姞CPU棰戠巼)
        for i, rsu in enumerate(self.simulator.rsus):
            # 鏍囧噯鍖栧綊涓€鍖栵細纭繚鎵€鏈夊€煎湪[0,1]鑼冨洿
            # 馃敡 淇锛氭坊鍔燙PU棰戠巼鐗瑰緛
            cpu_freq_norm = normalize_scalar(rsu.get('cpu_freq', 12.5e9), 'cpu_frequency_range', 20e9)
            rsu_state = np.array([
                normalize_scalar(rsu['position'][0], 'rsu_position_range', 2060.0),  # 浣嶇疆x
                normalize_scalar(rsu['position'][1], 'rsu_position_range', 2060.0),  # 浣嶇疆y
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),  # 缂撳瓨鍒╃敤鐜?
                normalize_scalar(len(rsu.get('computation_queue', [])), 'rsu_queue_capacity', 20.0),  # 闃熷垪鍒╃敤鐜?
                normalize_scalar(rsu.get('energy_consumed', 0.0), 'rsu_energy_reference', 1000.0),  # 鑳借€?
                cpu_freq_norm,  # 馃敡 鏂板锛氱6缁?- CPU棰戠巼
            ])
            node_states[f'rsu_{i}'] = rsu_state

        # UAV鐘舵€?(6缁?- 娣诲姞CPU棰戠巼)
        for i, uav in enumerate(self.simulator.uavs):
            # 鏍囧噯鍖栧綊涓€鍖栵細纭繚鎵€鏈夊€煎湪[0,1]鑼冨洿
            # 馃敡 淇锛氭坊鍔燙PU棰戠巼鐗瑰緛
            cpu_freq_norm = normalize_scalar(uav.get('cpu_freq', 5.0e9), 'cpu_frequency_range', 20e9)
            uav_state = np.array([
                normalize_scalar(uav['position'][0], 'uav_position_range', 2060.0),  # 浣嶇疆x
                normalize_scalar(uav['position'][1], 'uav_position_range', 2060.0),  # 浣嶇疆y
                normalize_scalar(uav['position'][2], 'uav_altitude_range', 200.0),   # 浣嶇疆z锛堥珮搴︼級
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),  # 缂撳瓨鍒╃敤鐜?
                normalize_scalar(uav.get('energy_consumed', 0.0), 'uav_energy_reference', 1000.0),  # 鑳借€?
                cpu_freq_norm,  # 馃敡 鏂板锛氱6缁?- CPU棰戠巼
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 璁＄畻绯荤粺鎸囨爣
        system_metrics = self._calculate_system_metrics(step_stats)
        
        # 鑾峰彇涓嬩竴鐘舵€?
        if isinstance(self.agent_env, (TD3Environment, TD3LatencyEnergyEnvironment, CAMTD3Environment)):
            next_state = self.agent_env.get_state_vector(node_states, system_metrics, resource_state)  # type: ignore[call-arg]
        else:
            next_state = self.agent_env.get_state_vector(node_states, system_metrics)  # type: ignore[call-arg]
        
        # 馃敡 澧炲己锛氳绠楀寘鍚瓙绯荤粺鎸囨爣鐨勫鍔?
        cache_metrics = self.adaptive_cache_controller.get_cache_metrics()
        migration_metrics = self.adaptive_migration_controller.get_migration_metrics()
        if hasattr(self, 'strategy_coordinator') and self.strategy_coordinator is not None:
            try:
                self.strategy_coordinator.observe_step(
                    system_metrics,
                    cache_metrics,
                    migration_metrics,
                    step_stats,
                )
            except Exception as exc:
                print(f"鈿狅笍 鑱斿悎绛栫暐鍗忚皟鍣ㄨ娴嬪紓甯? {exc}")

        # 鍙嶉鍏抽敭绯荤粺鎸囨爣缁橳D3绛栫暐鎸囧妯″潡锛岄┍鍔ㄨ兘鑰?寤惰繜娓╁害鑷€傚簲
        agent_core = getattr(self.agent_env, 'agent', None)
        if agent_core is not None and hasattr(agent_core, 'update_guidance_feedback'):
            try:
                agent_core.update_guidance_feedback(system_metrics, cache_metrics, migration_metrics)
            except Exception as exc:
                if getattr(self, '_current_episode', 0) % 200 == 0:
                    print(f"鈿狅笍 鎸囧鍙嶉鏇存柊澶辫触: {exc}")

        reward_source = system_metrics.get('reward_snapshot', system_metrics)
        reward = self.agent_env.calculate_reward(reward_source, cache_metrics, migration_metrics)
        if not np.isfinite(reward):
            reward = 0.0
        try:
            system_metrics['normalized_reward'] = self._normalize_reward_value(reward)
        except Exception:
            system_metrics['normalized_reward'] = 0.0
        
        task_type_queue = system_metrics.get('task_type_queue_distribution', [])
        task_type_deadline = system_metrics.get('task_type_deadline_remaining', [])
        task_type_drop = system_metrics.get('task_type_drop_rate', [])
        for idx in range(4):
            queue_val = float(task_type_queue[idx]) if idx < len(task_type_queue) else 0.0
            deadline_val = float(task_type_deadline[idx]) if idx < len(task_type_deadline) else 0.0
            drop_val = float(task_type_drop[idx]) if idx < len(task_type_drop) else 0.0
            self.episode_metrics[f'task_type_queue_share_{idx+1}'].append(queue_val)
            self.episode_metrics[f'task_type_deadline_norm_{idx+1}'].append(deadline_val)
            self.episode_metrics[f'task_type_drop_rate_{idx+1}'].append(drop_val)

        hotspot_mean = float(system_metrics.get('rsu_hotspot_mean', 0.0))
        hotspot_peak = float(system_metrics.get('rsu_hotspot_peak', 0.0))
        self.episode_metrics['rsu_hotspot_mean_series'].append(hotspot_mean)
        self.episode_metrics['rsu_hotspot_peak_series'].append(hotspot_peak)
        self.episode_metrics['mm1_queue_error'].append(float(system_metrics.get('mm1_queue_error', 0.0)))
        self.episode_metrics['mm1_delay_error'].append(float(system_metrics.get('mm1_delay_error', 0.0)))
        
        # 馃殌 鏂板锛氳褰曡縼绉昏兘鑰?
        self.episode_metrics['rsu_migration_energy'].append(float(system_metrics.get('rsu_migration_energy', 0.0)))
        self.episode_metrics['uav_migration_energy'].append(float(system_metrics.get('uav_migration_energy', 0.0)))
        
        # 鍒ゆ柇鏄惁缁撴潫
        done = False  # 鍗曟櫤鑳戒綋鐜閫氬父涓嶄細鎻愬墠缁撴潫
        
        # 闄勫姞淇℃伅
        info = {
            'step_stats': step_stats,
            'system_metrics': system_metrics
        }
        
        return next_state, reward, done, info
    
    def _calculate_system_metrics(self, step_stats: Dict) -> Dict:
        """璁＄畻绯荤粺鎬ц兘鎸囨爣 - 鏈€缁堜慨澶嶇増锛岀‘淇濇暟鍊煎湪鍚堢悊鑼冨洿"""
        import numpy as np
        
        # 瀹夊叏鑾峰彇鏁板€?
        def safe_get(key: str, default: float = 0.0) -> float:
            value = step_stats.get(key, default)
            if np.isnan(value) or np.isinf(value):
                return default
            return max(0.0, value)  # 纭繚闈炶礋
        
        # 馃敡 淇锛氫娇鐢╡pisode绾у埆缁熻鑰岄潪绱Н缁熻锛岄伩鍏嶅鍔辩疮绉伓鍖?
        # 璁＄畻鏈琫pisode鐨勫閲忕粺璁?
        total_processed = int(safe_get('processed_tasks', 0))  # 绱瀹屾垚
        total_dropped = int(safe_get('dropped_tasks', 0))  # 绱涓㈠純锛堟暟閲忥級
        
        # 璁＄畻鏈琫pisode澧為噺
        episode_processed = total_processed - getattr(self, '_episode_processed_base', 0)
        episode_dropped = total_dropped - getattr(self, '_episode_dropped_base', 0)
        
        # 鏁版嵁涓㈠け閲忥細浣跨敤鏈琫pisode澧為噺
        current_generated_bytes = float(step_stats.get('generated_data_bytes', 0.0))
        current_dropped_bytes = float(step_stats.get('dropped_data_bytes', 0.0))
        episode_generated_bytes = current_generated_bytes - getattr(self, '_episode_generated_bytes_base', 0.0)
        episode_dropped_bytes = current_dropped_bytes - getattr(self, '_episode_dropped_bytes_base', 0.0)
        
        # 璁＄畻鏈琫pisode浠诲姟鎬绘暟鍜屽畬鎴愮巼锛堥伩鍏嶇疮绉晥搴旓級
        episode_total = episode_processed + episode_dropped
        completion_rate = normalize_ratio(episode_processed, episode_total, default=0.5)
        
        cache_hits = int(safe_get('cache_hits', 0))
        cache_misses = int(safe_get('cache_misses', 0))
        cache_requests_total = cache_hits + cache_misses
        reported_requests = int(step_stats.get('cache_requests', cache_requests_total) or cache_requests_total)
        reported_hit_rate = step_stats.get('cache_hit_rate')
        if reported_requests > 0:
            cache_requests_total = reported_requests
        cache_hit_rate = normalize_ratio(cache_hits, cache_requests_total)
        if isinstance(reported_hit_rate, (int, float)):
            cache_hit_rate = float(np.clip(reported_hit_rate, 0.0, 1.0))
        # 馃敡 鍥為€€鍒扮紦瀛樻帶鍒跺櫒鐨勭粺璁★紝閬垮厤鏃ュ織缂哄け瀵艰嚧0鍛戒腑
        cache_metrics = getattr(self, "adaptive_cache_controller", None)
        if cache_metrics is not None:
            cache_metrics = cache_metrics.get_cache_metrics()
            cm_requests = int(cache_metrics.get('total_requests', 0) or 0)
            cm_hit_rate = float(cache_metrics.get('hit_rate', 0.0) or 0.0)
            if cm_requests > 0 and cache_hit_rate <= 0.0:
                cache_requests_total = cm_requests
                cache_hit_rate = float(np.clip(cm_hit_rate, 0.0, 1.0))
        local_cache_hits = int(safe_get('local_cache_hits', 0))
        
        # 馃敡 淇锛氭纭绠楀钩鍧囧欢杩?- 浣跨敤episode绾у埆澧為噺锛岃€岄潪绱Н鍊?
        total_delay_绱Н = safe_get('total_delay', 0.0)
        # 璁＄畻鏈琫pisode鐨勫欢杩熷閲?
        delay_base_value = getattr(self, '_episode_delay_base', 0.0)
        episode_delay = max(0.0, total_delay_绱Н - delay_base_value)
        # 浣跨敤鏈琫pisode鐨勪换鍔℃暟
        processed_for_delay = max(1, episode_processed) if episode_processed > 0 else max(1, total_processed)
        # 璁＄畻鏈琫pisode鐨勫钩鍧囧欢杩?
        avg_delay = episode_delay / processed_for_delay if processed_for_delay > 0 else 0.0
        
        # 馃敡 淇锛氱Щ闄ら敊璇殑clip锛屽欢杩熷簲璇ユ牴鎹疄闄呮儏鍐佃嚜鐒跺睍鐜?
        # 鍙湪鏄庢樉寮傚父鏃舵墠杩涜瑁佸壀锛堜緥濡傝秴杩?0绉掞紝璇存槑璁＄畻鏈夎锛?
        if avg_delay > 60.0 or not np.isfinite(avg_delay):
            print(f"鈿狅笍 寮傚父寤惰繜妫€娴? {avg_delay:.2f}s锛岄噸缃负0.0s")
            avg_delay = 0.0
        avg_delay = max(0.0, avg_delay)  # 纭繚闈炶礋

        delay_base = getattr(self, '_episode_delay_component_base', {})
        delay_processing_total = safe_get('delay_processing', 0.0)
        delay_uplink_total = safe_get('delay_uplink', 0.0)
        delay_downlink_total = safe_get('delay_downlink', 0.0)
        delay_cache_total = safe_get('delay_cache', 0.0)
        delay_wait_total = safe_get('delay_waiting', 0.0)
        def _episode_delay(bucket_total: float, bucket_key: str) -> float:
            return max(0.0, bucket_total - delay_base.get(bucket_key, 0.0))
        episode_delay_processing = _episode_delay(delay_processing_total, 'delay_processing')
        episode_delay_uplink = _episode_delay(delay_uplink_total, 'delay_uplink')
        episode_delay_downlink = _episode_delay(delay_downlink_total, 'delay_downlink')
        episode_delay_cache = _episode_delay(delay_cache_total, 'delay_cache')
        episode_delay_wait = _episode_delay(delay_wait_total, 'delay_waiting')
        delay_denominator = max(1, episode_processed) if episode_processed > 0 else max(1, processed_for_delay)
        avg_processing_delay_component = episode_delay_processing / delay_denominator
        avg_uplink_delay_component = episode_delay_uplink / delay_denominator
        avg_downlink_delay_component = episode_delay_downlink / delay_denominator
        avg_cache_delay_component = episode_delay_cache / delay_denominator
        avg_wait_delay_component = episode_delay_wait / delay_denominator
        
        # 馃敡 淇鑳借€楄绠楋細浣跨敤鐪熷疄episode澧為噺鑳借€?
        current_total_energy = safe_get('total_energy', 0.0)

        if not getattr(self, '_episode_counters_initialized', False):
            self._initialize_episode_counters(step_stats)

        # 鑷€傚簲鎺у埗鍣ㄧ粺璁★紙鐢ㄤ簬濂栧姳涓庢寚鏍囧綊涓€鍖栵級
        cache_metrics = self.adaptive_cache_controller.get_cache_metrics()
        migration_metrics = self.adaptive_migration_controller.get_migration_metrics()
        cache_total_requests = int(cache_metrics.get('total_requests', 0) or 0)
        cache_total_evictions = int(cache_metrics.get('evicted_items', 0) or 0)
        cache_total_collab = int(cache_metrics.get('collaborative_writes', 0) or 0)

        queue_rho_sum = float(step_stats.get('queue_rho_sum', 0.0) or 0.0)
        queue_rho_max = float(step_stats.get('queue_rho_max', 0.0) or 0.0)
        queue_overload_flag = 1.0 if bool(step_stats.get('queue_overload_flag', False)) else 0.0
        queue_rho_by_node = step_stats.get('queue_rho_by_node', {}) or {}
        queue_overloaded_nodes = step_stats.get('queue_overloaded_nodes', {}) or {}
        queue_warning_nodes = step_stats.get('queue_warning_nodes', {}) or {}
        queue_overload_events_total = int(step_stats.get('queue_overload_events', 0) or 0)
        queue_overload_events = max(0, queue_overload_events_total - getattr(self, '_episode_queue_overload_events_base', 0))
        queue_overflow_total = int(step_stats.get('queue_overflow_drops', 0) or 0)
        queue_overflow_drops = max(0, queue_overflow_total - getattr(self, '_episode_queue_overflow_base', 0))
        remote_stats = step_stats.get('remote_rejections', {}) or {}
        remote_total = int(remote_stats.get('total', 0) or 0)
        episode_remote_rejects = max(0, remote_total - getattr(self, '_episode_remote_reject_base', 0))
        remote_rejection_rate = normalize_ratio(episode_remote_rejects, episode_total, default=0.0)

        mm1_predictions_raw = step_stats.get('mm1_predictions', {}) or {}
        mm1_predictions: Dict[str, Dict[str, float]] = {}
        mm1_queue_errors: List[float] = []
        mm1_delay_errors: List[float] = []
        if isinstance(mm1_predictions_raw, dict):
            for node_key, pred in mm1_predictions_raw.items():
                if not isinstance(pred, dict):
                    continue
                arrival_rate = float(pred.get('arrival_rate', 0.0) or 0.0)
                service_rate = float(pred.get('service_rate', 0.0) or 0.0)
                rho_val = pred.get('rho')
                if rho_val is None:
                    rho = np.inf if service_rate <= 0.0 and arrival_rate > 0.0 else 0.0
                else:
                    try:
                        rho = float(rho_val)
                    except (TypeError, ValueError):
                        rho = np.inf
                stable = bool(pred.get('stable', False))
                rho_storable = float(rho) if np.isfinite(rho) else float('inf')
                theoretical_queue = pred.get('theoretical_queue')
                actual_queue = float(pred.get('actual_queue', 0.0) or 0.0)
                theoretical_delay = pred.get('theoretical_delay')
                actual_delay_obs = float(pred.get('actual_delay', 0.0) or 0.0)

                if theoretical_queue is not None:
                    try:
                        theo_queue_val = float(theoretical_queue)
                    except (TypeError, ValueError):
                        theo_queue_val = None
                else:
                    theo_queue_val = None

                if theoretical_delay is not None:
                    try:
                        theo_delay_val = float(theoretical_delay)
                    except (TypeError, ValueError):
                        theo_delay_val = None
                else:
                    theo_delay_val = None

                mm1_predictions[node_key] = {  # type: ignore[assignment]
                    'arrival_rate': float(arrival_rate),
                    'service_rate': float(service_rate),
                    'rho': float(rho_storable) if rho_storable is not None else 0.0,
                    'stable': bool(stable),
                    'theoretical_queue': float(theo_queue_val) if theo_queue_val is not None else 0.0,
                    'actual_queue': float(actual_queue),
                    'theoretical_delay': float(theo_delay_val) if theo_delay_val is not None else 0.0,
                    'actual_delay': float(actual_delay_obs),
                }

                if theo_queue_val is not None:
                    mm1_queue_errors.append(abs(actual_queue - theo_queue_val))
                if theo_delay_val is not None:
                    mm1_delay_errors.append(abs(actual_delay_obs - theo_delay_val))

        mm1_queue_error = float(np.mean(mm1_queue_errors)) if mm1_queue_errors else 0.0
        mm1_delay_error = float(np.mean(mm1_delay_errors)) if mm1_delay_errors else 0.0

        
        # 馃敡 P0淇锛氱Щ闄よ兘鑰椾及绠楅瓟娉曟暟瀛楋紝濡傛灉涓?鍒欐樉绀鸿鍛婁絾涓嶄娇鐢ㄨ櫄鍋囧€?
        if current_total_energy <= 0.0:
            # 浣跨敤涓婁竴episode鐨勮兘鑰椾綔涓哄熀绾匡紙鏇村悎鐞嗭級
            episode_incremental_energy = 0.0
            total_energy = 0.0
            print(f"鈿狅笍 浠跨湡鍣ㄨ兘鑰椾负0锛岃妫€鏌ヤ豢鐪熷櫒鑳借€楁ā鍨嬶紒")
        else:
            episode_incremental_energy = max(0.0, current_total_energy - getattr(self, '_episode_energy_base', 0.0))
            total_energy = episode_incremental_energy

        energy_base = getattr(self, '_episode_energy_component_base', {})
        def _episode_energy(bucket_key: str) -> float:
            return max(0.0, safe_get(bucket_key, 0.0) - energy_base.get(bucket_key, 0.0))
        energy_compute_component = _episode_energy('energy_compute')
        energy_tx_uplink_component = _episode_energy('energy_transmit_uplink')
        energy_tx_downlink_component = _episode_energy('energy_transmit_downlink')
        energy_cache_component = _episode_energy('energy_cache')
        energy_denominator = max(1, episode_processed) if episode_processed > 0 else 1
        avg_energy_compute_component = energy_compute_component / energy_denominator
        avg_energy_uplink_component = energy_tx_uplink_component / energy_denominator
        avg_energy_downlink_component = energy_tx_downlink_component / energy_denominator
        avg_energy_cache_component = energy_cache_component / energy_denominator
        
        # 馃敡 淇锛氫娇鐢╡pisode绾у埆鏁版嵁涓㈠け閲忥紝閬垮厤绱Н鏁堝簲
        data_loss_bytes = max(0.0, episode_dropped_bytes)
        data_generated_bytes = max(0.0, episode_generated_bytes)
        data_loss_ratio_bytes = normalize_ratio(data_loss_bytes, data_generated_bytes)
        
        # 馃敟 鏂板锛氳绠楀嵏杞芥瘮渚嬶紙local/rsu/uav锛?
        # Accumulate task counts from per-step stats
        self._episode_local_tasks += int(step_stats.get('local_tasks', 0))
        self._episode_rsu_tasks += int(step_stats.get('rsu_tasks', 0))
        self._episode_uav_tasks += int(step_stats.get('uav_tasks', 0))

        local_tasks_count = self._episode_local_tasks
        rsu_tasks_count = self._episode_rsu_tasks
        uav_tasks_count = self._episode_uav_tasks
        total_offload_tasks = local_tasks_count + rsu_tasks_count + uav_tasks_count
        
        if total_offload_tasks > 0:
            local_offload_ratio = float(local_tasks_count) / float(total_offload_tasks)
            rsu_offload_ratio = float(rsu_tasks_count) / float(total_offload_tasks)
            uav_offload_ratio = float(uav_tasks_count) / float(total_offload_tasks)
            # 馃幆 淇锛氳绠楁€昏繙绋嬪嵏杞芥瘮渚嬶紙RSU+UAV锛?
            remote_execution_ratio = rsu_offload_ratio + uav_offload_ratio
        else:
            # 榛樿鍊硷細鍏ㄩ儴鏈湴澶勭悊
            local_offload_ratio = 1.0
            rsu_offload_ratio = 0.0
            uav_offload_ratio = 0.0
            remote_execution_ratio = 0.0
        
        # 馃幆 淇锛氳绠桼SU璧勬簮鍒╃敤鐜囷紙璁＄畻闃熷垪鍗犵敤鐜囷級
        rsu_total_utilization = 0.0
        rsu_count = len(self.simulator.rsus) if hasattr(self.simulator, 'rsus') else 0
        if rsu_count > 0:
            for rsu in self.simulator.rsus:
                queue_len = len(rsu.get('computation_queue', []))
                queue_capacity = rsu.get('queue_capacity', 20)  # 榛樿瀹归噺20
                rsu_total_utilization += float(queue_len) / max(1.0, float(queue_capacity))
            rsu_utilization = rsu_total_utilization / float(rsu_count)
        else:
            rsu_utilization = 0.0
        
        # 杩佺Щ鎴愬姛鐜囷紙鏉ヨ嚜浠跨湡鍣ㄧ粺璁★級
        migrations_executed = int(safe_get('migrations_executed', 0))
        migrations_successful = int(safe_get('migrations_successful', 0))
        migration_success_rate = normalize_ratio(migrations_successful, migrations_executed)
        
        # 馃敡 璋冭瘯杩佺Щ缁熻
        if migrations_executed > 0:
            print(f"馃攳 杩佺Щ缁熻: 鎵ц{migrations_executed}娆? 鎴愬姛{migrations_successful}娆? 鎴愬姛鐜噞migration_success_rate:.1%}")

        episode_cache_requests = max(
            0,
            cache_total_requests - getattr(self, '_episode_cache_requests_base', 0)
        )
        episode_cache_evictions = max(
            0,
            cache_total_evictions - getattr(self, '_episode_cache_evictions_base', 0)
        )
        episode_cache_collab = max(
            0,
            cache_total_collab - getattr(self, '_episode_cache_collab_base', 0)
        )
        cache_eviction_rate = normalize_ratio(episode_cache_evictions, episode_cache_requests)

        def _normalize_vector(key: str, length: int = 4, clip: bool = True) -> List[float]:
            raw = step_stats.get(key)
            if isinstance(raw, (np.ndarray, list, tuple)):
                values = [float(v) for v in raw]  # 鏄庣‘杞崲涓篺loat鍒楄〃
            else:
                values = []
            return normalize_feature_vector(values, length, clip=clip)  # type: ignore[arg-type]

        queue_distribution = _normalize_vector('task_type_queue_distribution')
        active_distribution = _normalize_vector('task_type_active_distribution')
        deadline_remaining = _normalize_vector('task_type_deadline_remaining')
        queue_counts = _normalize_vector('task_type_queue_counts', clip=False)
        active_counts = _normalize_vector('task_type_active_counts', clip=False)
        hotspot_list = _normalize_vector(
            'rsu_hotspot_intensity',
            length=getattr(self.simulator, 'num_rsus', 0) or 4
        )
        rsu_hotspot_mean = float(np.mean(hotspot_list)) if hotspot_list else 0.0
        rsu_hotspot_peak = float(np.max(hotspot_list)) if hotspot_list else 0.0

        task_generation_stats = step_stats.get('task_generation')
        gen_by_type = task_generation_stats.get('by_type', {}) if isinstance(task_generation_stats, dict) else {}
        drop_stats = step_stats.get('drop_stats')
        drop_by_type = drop_stats.get('by_type', {}) if isinstance(drop_stats, dict) else {}

        generated_counts: List[float] = []
        drop_rate: List[float] = []
        for task_type in range(1, 5):
            generated = float(gen_by_type.get(task_type, 0.0))
            dropped = float(drop_by_type.get(task_type, 0.0))
            generated_counts.append(generated)
            drop_rate.append(normalize_ratio(dropped, generated))
        generated_share = normalize_distribution(generated_counts) if generated_counts else []

        # 馃攳 璋冭瘯鏃ュ織锛氳兘鑰椾笌杩佺Щ鏁忔劅鍖洪棿
        current_episode = getattr(self, '_current_episode', 0)
        if current_episode > 0 and (current_episode % 50 == 0 or avg_delay > 0.2 or migration_success_rate < 0.9):
            # 馃敡 淇锛氳绠椾换鍔℃暟閲忔崯澶辩巼锛屼笌瀹屾垚鐜囧搴?
            task_drop_rate = normalize_ratio(episode_dropped, episode_total)
            print(
                f"[璋冭瘯] Episode {current_episode:04d}: 寤惰繜 {avg_delay:.3f}s, 鑳借€?{total_energy:.2f}J, "
                f"瀹屾垚鐜?{completion_rate:.1%}, 杩佺Щ鎴愬姛鐜?{migration_success_rate:.1%}, "
                f"缂撳瓨鍛戒腑 {cache_hit_rate:.1%}, 鏁版嵁鎹熷け {data_loss_ratio_bytes:.1%}, "
                f"缂撳瓨娣樻卑鐜?{cache_eviction_rate:.1%}"
            )
            # 馃敟 鏂板锛氭樉绀哄嵏杞藉垎甯冪粺璁″拰鎹熷け鐜囧姣?
            print(
                f"  浠诲姟鍒嗗竷: 鏈湴 {local_tasks_count}涓?{local_offload_ratio:.1%}), "
                f"RSU {rsu_tasks_count}涓?{rsu_offload_ratio:.1%}), "
                f"UAV {uav_tasks_count}涓?{uav_offload_ratio:.1%}), "
                f"涓㈠純 {episode_dropped}涓?
            )
            # 馃啎 娣诲姞锛氫换鍔℃暟閲弙s鏁版嵁閲忕殑瀵规瘮璇存槑
            if abs(task_drop_rate - data_loss_ratio_bytes) > 0.1:  # 宸紓>10%鏃舵彁绀?
                print(
                    f"  鈿狅笍 娉ㄦ剰: 浠诲姟鏁伴噺涓㈠け鐜噞task_drop_rate:.1%} vs 鏁版嵁閲忎涪澶辩巼{data_loss_ratio_bytes:.1%} "
                    f"(宸紓{abs(task_drop_rate - data_loss_ratio_bytes)*100:.1f}%锛岃鏄庝涪寮冧换鍔＄殑鏁版嵁閲忚緝澶?"
                )

        # 馃 鏇存柊缂撳瓨鎺у埗鍣ㄧ粺璁★紙濡傛灉鏈夊疄闄呮暟鎹級
        if cache_hit_rate > 0:
            # 馃敡 淇锛氭纭绠楃紦瀛樼粺璁?
            total_utilization = 0.0
            for rsu in self.simulator.rsus:
                utilization = self._calculate_correct_cache_utilization(
                    rsu.get('cache', {}), 
                    rsu.get('cache_capacity', 1000.0)
                )
                total_utilization += utilization
            
            self.adaptive_cache_controller.cache_stats['current_utilization'] = (
                total_utilization / max(1, len(self.simulator.rsus))
            )
        
        latency_target = max(1e-6, getattr(config.rl, 'latency_target', 0.4))
        energy_target = max(1e-6, getattr(config.rl, 'energy_target', 1200.0))

        reward_snapshot = self._build_reward_snapshot(step_stats)

        return {
            'avg_task_delay': avg_delay,
            'total_energy_consumption': total_energy,
            'data_loss_bytes': data_loss_bytes,
            'data_loss_ratio_bytes': data_loss_ratio_bytes,
            'task_completion_rate': completion_rate,
            'cache_hit_rate': cache_hit_rate,
            'local_cache_hits': local_cache_hits,
            'migration_success_rate': migration_success_rate,
            'dropped_tasks': episode_dropped,
            'avg_processing_delay': avg_processing_delay_component,
            'avg_uplink_delay': avg_uplink_delay_component,
            'avg_downlink_delay': avg_downlink_delay_component,
            'avg_cache_delay': avg_cache_delay_component,
            'avg_waiting_delay': avg_wait_delay_component,
            'energy_compute': energy_compute_component,
            'energy_transmit_uplink': energy_tx_uplink_component,
            'energy_transmit_downlink': energy_tx_downlink_component,
            'energy_cache': energy_cache_component,
            'avg_energy_compute': avg_energy_compute_component,
            'avg_energy_uplink': avg_energy_uplink_component,
            'avg_energy_downlink': avg_energy_downlink_component,
            'avg_energy_cache': avg_energy_cache_component,
            # 馃 鏂板鑷€傚簲鎺у埗鎸囨爣
            'adaptive_cache_effectiveness': cache_metrics.get('effectiveness', 0.0),
            'adaptive_migration_effectiveness': migration_metrics.get('effectiveness', 0.0),
            'migration_avg_cost': migration_metrics.get('avg_cost', 0.0),
            'migration_avg_delay_saved': migration_metrics.get('avg_delay_saved', 0.0),
            'cache_utilization': cache_metrics.get('utilization', 0.0),
            'cache_evictions': episode_cache_evictions,
            'cache_eviction_rate': cache_eviction_rate,
            'cache_requests': episode_cache_requests,
            'cache_collaborative_writes': episode_cache_collab,
            'adaptive_cache_params': cache_metrics.get('agent_params', {}),
            'adaptive_migration_params': migration_metrics.get('agent_params', {}),
            'task_type_queue_distribution': queue_distribution,
            'task_type_active_distribution': active_distribution,
            'task_type_deadline_remaining': deadline_remaining,
            'task_type_queue_counts': queue_counts,
            'task_type_active_counts': active_counts,
            'task_type_drop_rate': drop_rate,
            'task_type_generated_share': generated_share,
            'queue_rho_sum': queue_rho_sum,
            'queue_rho_max': queue_rho_max,
            'queue_overload_flag': queue_overload_flag,
            'queue_overload_events': queue_overload_events,
            'queue_rho_by_node': queue_rho_by_node,
            'queue_overloaded_nodes': queue_overloaded_nodes,
            'queue_warning_nodes': queue_warning_nodes,
            'queue_overflow_drops': queue_overflow_drops,
            'mm1_queue_error': mm1_queue_error,
            'mm1_delay_error': mm1_delay_error,
            'mm1_predictions': mm1_predictions,
            'rsu_hotspot_intensity_list': hotspot_list,
            'rsu_hotspot_mean': rsu_hotspot_mean,
            'rsu_hotspot_peak': rsu_hotspot_peak,
            'remote_rejection_count': episode_remote_rejects,
            'remote_rejection_rate': remote_rejection_rate,
            'normalized_delay': avg_delay / latency_target,
            'normalized_energy': total_energy / energy_target,
            'reward_snapshot': reward_snapshot,
            # 馃敟 鏂板锛氬嵏杞芥瘮渚嬬粺璁?
            'local_offload_ratio': local_offload_ratio,
            'rsu_offload_ratio': rsu_offload_ratio,
            'uav_offload_ratio': uav_offload_ratio,
            'local_tasks_count': local_tasks_count,
            'rsu_tasks_count': rsu_tasks_count,
            'uav_tasks_count': uav_tasks_count,
            # 馃幆 淇bug锛氭坊鍔犲叧閿寚鏍?
            'rsu_utilization': rsu_utilization,  # RSU璧勬簮鍒╃敤鐜?
            'offload_ratio': remote_execution_ratio,  # 鎬昏繙绋嬪嵏杞芥瘮渚嬶紙RSU+UAV锛?
            'remote_execution_ratio': remote_execution_ratio,  # 鍒悕锛屽吋瀹规棫浠ｇ爜
            # 馃殌 鏂板锛氳縼绉昏兘鑰楁寚鏍?
            'rsu_migration_energy': _episode_energy('rsu_migration_energy'),
            'uav_migration_energy': _episode_energy('uav_migration_energy'),
        }

    def _normalize_reward_value(self, reward: float) -> float:
        """灏嗗鍔卞€艰浆鎹负鏃犻噺绾叉瘮渚嬶紝渚夸簬涓庡叾浠栨寚鏍囧姣斻€?""
        import numpy as np
        rl_config = getattr(config, 'rl', None)
        reward_scale = float(
            getattr(
                rl_config,
                'reward_normalizer',
                getattr(rl_config, 'reward_weight_delay', 1.0)
                + getattr(rl_config, 'reward_weight_energy', 1.0)
            )
        )
        reward_scale = max(reward_scale, 1e-6)
        normalized = -reward / reward_scale
        return float(np.clip(normalized, -5.0, 5.0))
    
    def _record_episode_metrics(self, system_metrics: Dict, episode_steps: Optional[int] = None) -> None:
        """灏嗙郴缁熸寚鏍囧啓鍏pisode_metrics锛屾柟渚垮悗缁姤鍛?鍙鍖栦娇鐢ㄣ€?""
        import numpy as np

        metric_mapping = {
            'avg_task_delay': 'avg_delay',
            'total_energy_consumption': 'total_energy',
            'data_loss_bytes': 'data_loss_bytes',
            'data_loss_ratio_bytes': 'data_loss_ratio_bytes',
            'task_completion_rate': 'task_completion_rate',
            'cache_hit_rate': 'cache_hit_rate',
            'cache_utilization': 'cache_utilization',
            'cache_evictions': 'cache_evictions',
            'cache_eviction_rate': 'cache_eviction_rate',
            'cache_requests': 'cache_requests',
            'cache_collaborative_writes': 'cache_collaborative_writes',
            'local_cache_hits': 'local_cache_hits',
            'migration_success_rate': 'migration_success_rate',
            'queue_rho_sum': 'queue_rho_sum',
            'queue_rho_max': 'queue_rho_max',
            'queue_overload_flag': 'queue_overload_flag',
            'queue_overload_events': 'queue_overload_events',
            'avg_step_reward': 'avg_step_reward',  # 馃敡 淇锛氭坊鍔犲钩鍧囨瘡姝ュ鍔辨槧灏?
            'migration_avg_cost': 'migration_avg_cost',
            'migration_avg_delay_saved': 'migration_avg_delay_saved',
            'rsu_hotspot_mean': 'rsu_hotspot_mean',  # 馃敡 淇锛氱‘淇濊褰昬pisode绾у埆鐑偣骞冲潎
            'rsu_hotspot_peak': 'rsu_hotspot_peak',  # 馃敡 淇锛氱‘淇濊褰昬pisode绾у埆鐑偣宄板€?
            'normalized_delay': 'normalized_delay',
            'normalized_energy': 'normalized_energy',
            'normalized_reward': 'normalized_reward',
            'avg_processing_delay': 'avg_processing_delay',
            'avg_uplink_delay': 'avg_uplink_delay',
            'avg_downlink_delay': 'avg_downlink_delay',
            'avg_cache_delay': 'avg_cache_delay',
            'avg_waiting_delay': 'avg_waiting_delay',
            'energy_compute': 'energy_compute',
            'energy_transmit_uplink': 'energy_transmit_uplink',
            'energy_transmit_downlink': 'energy_transmit_downlink',
            'energy_cache': 'energy_cache',
            'avg_energy_compute': 'avg_energy_compute',
            'avg_energy_uplink': 'avg_energy_uplink',
            'avg_energy_downlink': 'avg_energy_downlink',
            'avg_energy_cache': 'avg_energy_cache',
            'queue_overflow_drops': 'queue_overflow_drops',
            # 馃幆 淇bug锛氭坊鍔犲叧閿寚鏍囨槧灏?
            'rsu_utilization': 'rsu_utilization',
            'offload_ratio': 'offload_ratio',
            'rsu_offload_ratio': 'rsu_offload_ratio',
            'uav_offload_ratio': 'uav_offload_ratio',
            'local_offload_ratio': 'local_offload_ratio',
        }

        def _coerce_scalar(value: Any) -> Optional[float]:
            if isinstance(value, (list, tuple, dict)):
                return None
            try:
                return float(value)
            except (TypeError, ValueError):
                if isinstance(value, np.ndarray) and value.size == 1:
                    return float(value.item())
                return None

        for system_key, episode_key in metric_mapping.items():
            if episode_key not in self.episode_metrics:
                continue
            scalar_value = _coerce_scalar(system_metrics.get(system_key))
            if scalar_value is None:
                continue
            self.episode_metrics[episode_key].append(scalar_value)

        queue_distribution_ep = system_metrics.get('task_type_queue_distribution')
        if isinstance(queue_distribution_ep, (list, tuple, np.ndarray)):
            for idx, value in enumerate(queue_distribution_ep):
                key = f'task_type_queue_share_ep_{idx+1}'
                if key in self.episode_metrics:
                    coerced = _coerce_scalar(value)
                    if coerced is not None:
                        self.episode_metrics[key].append(coerced)

        if episode_steps is not None and 'episode_steps' in self.episode_metrics:
            self.episode_metrics['episode_steps'].append(int(episode_steps))
        
        # 馃敡 淇锛氳绠楀苟璁板綍骞冲潎姣忔濂栧姳
        if episode_steps and episode_steps > 0:
            # 浠庢渶鍚庝竴涓猠pisode_reward璁＄畻avg_step_reward
            if hasattr(self, 'episode_rewards') and self.episode_rewards:
                last_episode_reward = self.episode_rewards[-1]
                avg_step_reward = last_episode_reward / episode_steps
                system_metrics['avg_step_reward'] = avg_step_reward
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None) -> Dict:
        """杩愯涓€涓畬鏁寸殑璁粌杞"""
        # 浣跨敤閰嶇疆涓殑鏈€澶ф鏁?
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 閲嶇疆鐜
        self._episode_counters_initialized = False
        state = self.reset_environment()
        
        # 馃敡 P1淇锛氬己鍒跺垵濮嬪寲episode璁℃暟鍣紝纭繚绗竴涓猠pisode缁熻姝ｇ‘
        if hasattr(self, 'simulator') and hasattr(self.simulator, 'stats'):
            self._initialize_episode_counters(self.simulator.stats)
        
        # 馃敡 淇濆瓨褰撳墠episode缂栧彿
        self._current_episode = episode
        
        # 馃敡 閲嶇疆episode姝ユ暟璺熻釜锛屼慨澶嶈兘鑰楄绠?
        self._current_episode_step = 0
        
        # 馃幆 鍒濆鍖栨湰episode鐨剆tep缁熻鍒楄〃
        episode_step_stats = []
        
        episode_reward = 0.0
        episode_info = {}
        step = 0
        info = {}  # 鍒濆鍖杋nfo鍙橀噺
        
        # PPO闇€瑕佺壒娈婂鐞?
        if self.algorithm == "PPO":
            return self._run_ppo_episode(episode, max_steps)
        
        for step in range(max_steps):
            # 閫夋嫨鍔ㄤ綔
            if self.algorithm == "DQN":
                # DQN杩斿洖绂绘暎鍔ㄤ綔
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 澶勭悊鍙兘鐨勫厓缁勮繑鍥?
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                        
                # 闇€瑕佸皢鍔ㄤ綔鏄犲皠鍥炲叏灞€鍔ㄤ綔绱㈠紩
                action_idx = self._encode_discrete_action(actions_dict)
                action = action_idx
            else:
                # 杩炵画鍔ㄤ綔绠楁硶
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 澶勭悊鍙兘鐨勫厓缁勮繑鍥?
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = self._encode_continuous_action(actions_dict)
            
            # 馃敡 鏇存柊episode姝ユ暟璁℃暟鍣?
            self._current_episode_step += 1
            
            # 鎵ц鍔ㄤ綔锛堝皢鍔ㄤ綔瀛楀吀浼犲叆浠ュ奖鍝嶄豢鐪熷櫒鍗歌浇鍋忓ソ锛?
            next_state, reward, done, info = self.step(action, state, actions_dict)
            
            # 馃幆 淇濆瓨鏈鐨剆tep_stats渚涗换鍔″垎甯冪粺璁′娇鐢?
            step_stats = info.get('step_stats', {})
            episode_step_stats.append(step_stats)

            # 灏唖tep绾у埆鐨勯槦鍒楁寚鏍囧悓姝ョ粰鏀寔鐨勬櫤鑳戒綋锛圦ueue-aware Replay锛?
            if hasattr(self.agent_env, 'update_queue_metrics'):
                try:
                    self.agent_env.update_queue_metrics(step_stats)  # type: ignore[attr-defined]
                except Exception:
                    pass

            # 灏嗛槦鍒?缂撳瓨鍘嬪姏浼犻€掔粰鏀寔鐨勬櫤鑳戒綋鐢ㄤ簬PER浼樺厛搴︽斁澶?
            if hasattr(self.agent_env, 'update_priority_signal'):
                try:
                    queue_pressure = float(max(
                        step_stats.get('queue_overload_flag', 0.0),
                        step_stats.get('queue_rho_max', 0.0),
                        step_stats.get('cache_eviction_rate', 0.0),
                    ))
                    self.agent_env.update_priority_signal(queue_pressure)  # type: ignore[attr-defined]
                except Exception:
                    pass
            
            # 鍒濆鍖杢raining_info
            training_info = {}
            
            # 璁粌鏅鸿兘浣?- 鎵€鏈夌畻娉曠幇鍦ㄩ兘鏀寔Union绫诲瀷缁熶竴鎺ュ彛
            # 纭繚action绫诲瀷瀹夊叏杞崲
            if self.algorithm == "DQN":
                # DQN棣栭€夋暣鏁板姩浣滐紝浣嗘帴鍙桿nion绫诲瀷
                safe_action = self._safe_int_conversion(action)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm in ["DDPG", "TD3", "TD3_LATENCY_ENERGY", "SAC", "OPTIMIZED_TD3"]:
                # 杩炵画鍔ㄤ綔绠楁硶棣栭€塶umpy鏁扮粍锛屼絾鎺ュ彈Union绫诲瀷
                if isinstance(action, np.ndarray):
                    safe_action = action
                elif isinstance(action, (int, float)):
                    safe_action = np.array([float(action)], dtype=np.float32)
                else:
                    safe_action = np.array(action, dtype=np.float32)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)  # type: ignore[arg-type]
            elif self.algorithm == "PPO":
                # PPO浣跨敤鐗规畩鐨別pisode绾у埆璁粌锛宼rain_step涓哄崰浣嶇
                # 淇濇寔鍘焌ction绫诲瀷鍗冲彲锛屽洜涓篜PO鐨則rain_step涓嶅仛瀹為檯澶勭悊
                training_info = self.agent_env.train_step(state, action, reward, next_state, done)  # type: ignore[arg-type]
            else:
                # 鍏朵粬绠楁硶鐨勯粯璁ゅ鐞?
                training_info = {'message': f'Unknown algorithm: {self.algorithm}'}
            
            episode_info = training_info
            
            # 鏇存柊鐘舵€?
            state = next_state
            episode_reward += reward
            
            # 妫€鏌ユ槸鍚︾粨鏉?
            if done:
                break
        
        episode_reward = float(episode_reward)
        if not np.isfinite(episode_reward):
            episode_reward = 0.0

        # 璁板綍杞缁熻
        system_metrics = info.get('system_metrics', {})
        self._record_episode_metrics(system_metrics, episode_steps=step + 1)
        self._maybe_update_dynamic_energy_target(
            episode,
            float(system_metrics.get('total_energy_consumption', 0.0) or 0.0)
        )
        self._maybe_update_dynamic_latency_target(
            episode,
            float(system_metrics.get('avg_task_delay', 0.0) or 0.0)
        )
        
        # 璋冪敤CAM-TD3 episode缁撴潫鍥炶皟锛屾洿鏂拌瀺鍚堢瓥鐣?
        if isinstance(self.agent_env, CAMTD3Environment) and hasattr(self.agent_env, 'on_episode_end'):
            self.agent_env.on_episode_end(episode_reward)
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': step + 1,
            'step_stats_list': episode_step_stats  # 馃幆 杩斿洖姣忎釜step鐨勭粺璁℃暟鎹?
        }
    
    def _run_ppo_episode(self, episode: int, max_steps: int = 100) -> Dict:
        """杩愯PPO涓撶敤episode"""
        state = self.reset_environment()
        episode_reward = 0.0
        
        # 鍒濆鍖栧彉閲?
        done = False
        step = 0
        info = {}
        
        for step in range(max_steps):
            # 鑾峰彇鍔ㄤ綔銆佸鏁版鐜囧拰浠峰€?
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, tuple) and len(actions_result) == 3:
                    actions_dict, log_prob, value = actions_result
                else:
                    # 濡傛灉涓嶆槸鍏冪粍锛屽氨浣跨敤榛樿鍊?
                    actions_dict = actions_result if isinstance(actions_result, dict) else {}
                    log_prob = 0.0
                    value = 0.0
            else:
                actions_dict = {}
                log_prob = 0.0
                value = 0.0
                
            action = self._encode_continuous_action(actions_dict)
            
            # 鎵ц鍔ㄤ綔
            next_state, reward, done, info = self.step(action, state, actions_dict)
            
            # 瀛樺偍缁忛獙 - 鎵€鏈夌畻娉曢兘鏀寔缁熶竴鎺ュ彛
            # 纭繚鍙傛暟绫诲瀷姝ｇ‘
            log_prob_float = float(log_prob) if not isinstance(log_prob, float) else log_prob
            value_float = float(value) if not isinstance(value, float) else value
            # 浣跨敤鍛藉悕鍙傛暟閬垮厤浣嶇疆鍙傛暟椤哄簭闂
            self.agent_env.store_experience(
                state=state, 
                action=action, 
                reward=reward, 
                next_state=next_state, 
                done=done, 
                log_prob=log_prob_float, 
                value=value_float
            )
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 馃敡 PPO鏇存柊绛栫暐淇锛氱疮绉涓猠pisode鍚庡啀鏇存柊
        last_value = 0.0
        if not done:
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, tuple) and len(actions_result) >= 3:
                    _, _, last_value = actions_result
                else:
                    last_value = 0.0
        
        # 纭繚 last_value 涓?float 绫诲瀷
        last_value_float = float(last_value) if not isinstance(last_value, float) else last_value
        
        # 妫€鏌ユ槸鍚﹀簲璇ユ洿鏂帮紙姣廚涓猠pisode鎴朾uffer蹇弧鏃讹級
        ppo_config = self.agent_env.config
        update_freq = getattr(ppo_config, 'update_frequency', 1)
        buffer_size = getattr(ppo_config, 'buffer_size', 1000)
        agent_obj = getattr(self.agent_env, 'agent', None)
        if agent_obj is not None:
            agent_buffer = getattr(agent_obj, 'buffer', None)  # type: ignore[union-attr]
            buffer_current_size = agent_buffer.size if agent_buffer is not None else 0
        else:
            buffer_current_size = 0
        should_update = (
            episode % max(1, update_freq) == 0 or  # 姣廚涓猠pisode
            buffer_current_size >= buffer_size * 0.9  # buffer鎺ヨ繎婊?
        )
        
        # 杩涜鏇存柊
        # PPOEnvironment.update鍙帴鍙條ast_value鍙傛暟
        training_info: Dict = {}
        if agent_obj is not None:
            training_info = agent_obj.update(last_value_float)  # type: ignore[call-arg]
        
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': training_info,
            'system_metrics': system_metrics,
            'steps': step + 1
        }

    def _build_simulator_actions(self, actions_dict: Optional[Dict]) -> Optional[Dict]:
        """灏嗙畻娉曞姩浣滃瓧鍏歌浆鎹负浠跨湡鍣ㄥ彲娑堣垂鐨勭畝鍗曟帶鍒朵俊鍙枫€?
        馃 鎵╁睍鏀寔鑱斿悎鍔ㄤ綔绌洪棿锛?
        - vehicle_agent 鍓?缁?鈫?鍘熸湁浠诲姟鍒嗛厤鍋忓ソ
        - 涓棿 num_rsus/num_uavs 缁?鈫?鑺傜偣閫夋嫨鏉冮噸
        - 鏈熬10缁?鈫?缂撳瓨銆佽縼绉诲強鑱斿姩鎺у埗鍙傛暟
        """
        if not isinstance(actions_dict, dict):
            return None
        vehicle_action = actions_dict.get('vehicle_agent')
        if vehicle_action is None:
            return None
        try:
            import numpy as np
            
            vehicle_action_array = np.array(vehicle_action, dtype=np.float32).reshape(-1)
            expected_dim = getattr(self.agent_env, 'action_dim', vehicle_action_array.size)
            if vehicle_action_array.size < expected_dim:
                padded = np.zeros(expected_dim, dtype=np.float32)
                padded[:vehicle_action_array.size] = vehicle_action_array
                vehicle_action_array = padded
            else:
                vehicle_action_array = vehicle_action_array[:expected_dim]
            
            # =============== 鍘熸湁浠诲姟鍒嗛厤閫昏緫 (淇濇寔鍏煎) ===============
            raw = vehicle_action_array[:3]
            raw = np.clip(raw, -5.0, 5.0)
            
            # 鉁?绉婚櫎鍋忕疆锛岃鏅鸿兘浣撻€氳繃濂栧姳淇″彿鐪熸瀛︿範
            # 濂栧姳鍑芥暟宸茬粡寮哄寲锛歊SU=8.0, UAV=1.0, Local penalty=4.0
            # 杩欎細鎻愪緵娓呮櫚鐨勫涔犱俊鍙凤紝寮曞鏅鸿兘浣撳悜RSU鍗歌浇
            
            exp = np.exp(raw - np.max(raw))
            probs = exp / np.sum(exp)
            
            sim_actions = {
                'vehicle_offload_pref': {
                    'local': float(probs[0]),
                    'rsu': float(probs[1] if probs.size > 1 else 0.33),
                    'uav': float(probs[2] if probs.size > 2 else 0.34)
                },
                # 璁板綍鍘熷softmax鐢ㄤ簬璇婃柇
                'offload_probs_raw': probs.tolist()
            }
            # RSU閫夋嫨姒傜巼
            num_rsus = self.num_rsus
            rsu_action = actions_dict.get('rsu_agent')
            if isinstance(rsu_action, (list, tuple, np.ndarray)) and num_rsus > 0:
                rsu_raw = np.array(rsu_action[:num_rsus], dtype=np.float32)
            else:
                rsu_raw = vehicle_action_array[3:3 + num_rsus]
            if num_rsus > 0:
                rsu_raw = np.clip(rsu_raw, -5.0, 5.0)
                rsu_exp = np.exp(rsu_raw - np.max(rsu_raw))
                rsu_probs = rsu_exp / np.sum(rsu_exp)
                sim_actions['rsu_selection_probs'] = rsu_probs.tolist()  # type: ignore[assignment]
            
            # UAV閫夋嫨姒傜巼
            num_uavs = self.num_uavs
            uav_action = actions_dict.get('uav_agent')
            if isinstance(uav_action, (list, tuple, np.ndarray)) and num_uavs > 0:
                uav_raw = np.array(uav_action[:num_uavs], dtype=np.float32)
            else:
                uav_raw = vehicle_action_array[3 + num_rsus:3 + num_rsus + num_uavs]
            if num_uavs > 0:
                uav_raw = np.clip(uav_raw, -5.0, 5.0)
                uav_exp = np.exp(uav_raw - np.max(uav_raw))
                uav_probs = uav_exp / np.sum(uav_exp)
                sim_actions['uav_selection_probs'] = uav_probs.tolist()  # type: ignore[assignment]
            
            # 馃 =============== 鏂板鑱斿悎缂撳瓨-杩佺Щ鎺у埗鍙傛暟 ===============
            control_start = 3 + num_rsus + num_uavs
            control_end = control_start + 10
            cache_migration_actions = vehicle_action_array[control_start:control_end]
            if cache_migration_actions.size < 10:
                padded = np.zeros(10, dtype=np.float32)
                padded[:cache_migration_actions.size] = cache_migration_actions
                cache_migration_actions = padded
            cache_migration_actions = np.clip(cache_migration_actions, -1.0, 1.0)

            cache_params, migration_params, joint_params = map_agent_actions_to_params(cache_migration_actions)

            self.adaptive_cache_controller.update_agent_params(cache_params)
            if not self.disable_migration:
                self.adaptive_migration_controller.update_agent_params(migration_params)
            if getattr(self, 'strategy_coordinator', None) is not None:
                self.strategy_coordinator.update_joint_params(joint_params)

            payload = {
                'adaptive_cache_params': cache_params,
                'cache_controller': self.adaptive_cache_controller,
                'joint_strategy_params': joint_params,
            }
            if not self.disable_migration:
                payload.update({
                    'adaptive_migration_params': migration_params,
                    'migration_controller': self.adaptive_migration_controller
                })
            sim_actions.update(payload)

            # 馃攣 璁╃郴缁熸ā鎷熷櫒鎺ユ敹Actor瀵煎嚭鐨勬寚瀵间俊鍙凤紙缁熶竴閿悕涓簉l_guidance锛?
            guidance_payload = actions_dict.get('guidance') if isinstance(actions_dict, dict) else None
            if isinstance(guidance_payload, dict) and guidance_payload:
                sim_actions['rl_guidance'] = guidance_payload

            # 馃幆 =============== 涓ぎ璧勬簮鍒嗛厤鍔ㄤ綔 (Phase 1) ===============
            if self.central_resource_enabled and self.central_resource_action_dim > 0:
                central_start = self.base_action_dim
                central_end = central_start + self.central_resource_action_dim
                central_vector = vehicle_action_array[central_start:central_end]
                allocations = self._decode_central_resource_actions(central_vector)
                if allocations:
                    try:
                        self.simulator.apply_resource_allocation(allocations)
                        sim_actions['central_resource_allocation'] = allocations  # type: ignore[assignment]
                    except Exception as exc:
                        print(f"鈿狅笍 涓ぎ璧勬簮鍒嗛厤搴旂敤澶辫触: {exc}")
            
            forced_mode = getattr(self, 'enforce_offload_mode', '')
            if forced_mode == 'local_only':
                sim_actions['vehicle_offload_pref'] = {'local': 1.0, 'rsu': 0.0, 'uav': 0.0}
            elif forced_mode == 'remote_only':
                if num_rsus == 0 and num_uavs == 0:
                    sim_actions['vehicle_offload_pref'] = {'local': 1.0, 'rsu': 0.0, 'uav': 0.0}
                elif num_rsus == 0:
                    sim_actions['vehicle_offload_pref'] = {'local': 0.0, 'rsu': 0.0, 'uav': 1.0}
                elif num_uavs == 0:
                    sim_actions['vehicle_offload_pref'] = {'local': 0.0, 'rsu': 1.0, 'uav': 0.0}
                else:
                    sim_actions['vehicle_offload_pref'] = {'local': 0.0, 'rsu': 0.5, 'uav': 0.5}

            # Attach distance-cache tradeoff gate for heuristic guidance (if actor exposes it)
            try:
                import numpy as _np  # safe local import
                actor_obj = getattr(self.agent_env, 'agent', None)
                if actor_obj is not None:
                    actor_obj = getattr(actor_obj, 'actor', None)
                gate = None
                if actor_obj is not None:
                    gate = getattr(actor_obj, 'last_tradeoff_gate', None)
                    if gate is None:
                        enc = getattr(actor_obj, 'encoder', None)
                        if enc is not None:
                            gate = getattr(enc, 'last_gate', None)
                if gate is not None:
                    try:
                        sim_actions['dc_tradeoff_gate'] = float(_np.clip(gate, 0.0, 1.0))  # type: ignore[assignment]
                    except Exception:
                        pass
            except Exception:
                pass

            return sim_actions
        except Exception as e:
            print(f"鈿狅笍 鍔ㄤ綔鏋勯€犲紓甯? {e}")
            return None
    
    def _collect_resource_state(self) -> Optional[Dict[str, Any]]:
        if not self.central_resource_enabled:
            return None
        resource_pool = getattr(self.simulator, 'resource_pool', None)
        if resource_pool is None:
            return None
        try:
            return resource_pool.get_resource_state()
        except Exception:
            return None
    
    @staticmethod
    def _normalize_allocation(vector: np.ndarray, size: int) -> np.ndarray:
        if size <= 0:
            return np.zeros(0, dtype=np.float32)
        vec = np.array(vector, dtype=np.float32).reshape(-1)
        if vec.size < size:
            vec = np.pad(vec, (0, size - vec.size), constant_values=0.0)
        elif vec.size > size:
            vec = vec[:size]
        vec = np.clip(vec, 0.0, 1.0)
        total = float(np.sum(vec))
        if total <= 1e-6:
            return np.full(size, 1.0 / size, dtype=np.float32)
        return (vec / total).astype(np.float32)
    
    def _decode_central_resource_actions(
        self, central_vector: np.ndarray
    ) -> Optional[Dict[str, np.ndarray]]:
        if not self.central_resource_enabled or self.central_resource_action_dim <= 0:
            return None
        vector = np.array(central_vector, dtype=np.float32).reshape(-1)
        expected = self.central_resource_action_dim
        if vector.size < expected:
            padded = np.zeros(expected, dtype=np.float32)
            padded[:vector.size] = vector
            vector = padded
        elif vector.size > expected:
            vector = vector[:expected]
        vector = np.clip(vector, 0.0, 1.0)
        
        idx = 0
        bandwidth = self._normalize_allocation(
            vector[idx:idx + self.num_vehicles], self.num_vehicles
        )
        idx += self.num_vehicles
        vehicle_compute = self._normalize_allocation(
            vector[idx:idx + self.num_vehicles], self.num_vehicles
        )
        idx += self.num_vehicles
        rsu_compute = self._normalize_allocation(
            vector[idx:idx + self.num_rsus], self.num_rsus
        )
        idx += self.num_rsus
        uav_compute = self._normalize_allocation(
            vector[idx:idx + self.num_uavs], self.num_uavs
        )
        
        return {
            'bandwidth': bandwidth,
            'vehicle_compute': vehicle_compute,
            'rsu_compute': rsu_compute,
            'uav_compute': uav_compute,
        }
    
    def _encode_continuous_action(self, actions_dict) -> np.ndarray:
        """
        馃 灏嗗姩浣滃瓧鍏哥紪鐮佷负杩炵画鍔ㄤ綔鍚戦噺 - 鍔ㄦ€侀€傞厤鍔ㄤ綔缁村害
        """
        # 澶勭悊鍙兘鐨勪笉鍚岃緭鍏ョ被鍨?
        action_dim = getattr(self.agent_env, 'action_dim', 18)
        if not isinstance(actions_dict, dict):
            # 濡傛灉涓嶆槸瀛楀吀锛岃繑鍥為粯璁ゅ姩浣滅淮搴?
            return np.zeros(action_dim, dtype=np.float32)

        # 馃 鍙娇鐢╲ehicle_agent鐨勫畬鏁村姩浣滃悜閲?
        vehicle_action = actions_dict.get('vehicle_agent')
        if isinstance(vehicle_action, (list, tuple, np.ndarray)):
            vehicle_action = np.array(vehicle_action, dtype=np.float32)
            if vehicle_action.size >= action_dim:
                return vehicle_action[:action_dim]
            action = np.zeros(action_dim, dtype=np.float32)
            action[:vehicle_action.size] = vehicle_action
            return action

        # 榛樿杩斿洖鍏ㄩ浂鍔ㄤ綔
        return np.zeros(action_dim, dtype=np.float32)
    
    def _build_actions_from_vector(self, action_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """灏嗚繛缁姩浣滃悜閲忔仮澶嶄负浠跨湡鍣ㄩ渶瑕佺殑鍔ㄤ綔瀛楀吀锛堝姩鎬佺淮搴︼級"""
        import numpy as np

        if not isinstance(action_vector, np.ndarray):
            action_vector = np.array(action_vector, dtype=np.float32)

        action_dim = getattr(self.agent_env, 'action_dim', action_vector.size)
        if action_vector.size < action_dim:
            padded = np.zeros(action_dim, dtype=np.float32)
            padded[:action_vector.size] = action_vector
            action_vector = padded
        else:
            action_vector = action_vector.astype(np.float32)[:action_dim]

        num_rsus = len(getattr(self.simulator, 'rsus', []))
        num_uavs = len(getattr(self.simulator, 'uavs', []))
        rsu_start = 3
        rsu_end = rsu_start + num_rsus
        uav_end = rsu_end + num_uavs

        return {
            'vehicle_agent': action_vector,
            'rsu_agent': action_vector[rsu_start:rsu_end],
            'uav_agent': action_vector[rsu_end:uav_end]
        }

    def _encode_discrete_action(self, actions_dict) -> int:
        """灏嗗姩浣滃瓧鍏哥紪鐮佷负绂绘暎鍔ㄤ綔绱㈠紩"""
        # 澶勭悊鍙兘鐨勪笉鍚岃緭鍏ョ被鍨?
        if not isinstance(actions_dict, dict):
            return 0  # 榛樿鍔ㄤ綔绱㈠紩
        
        # 绠€鍖栧疄鐜帮細灏嗘瘡涓櫤鑳戒綋鐨勫姩浣滅粍鍚堟垚涓€涓储寮?
        vehicle_action = actions_dict.get('vehicle_agent', 0)
        rsu_action = actions_dict.get('rsu_agent', 0)
        uav_action = actions_dict.get('uav_agent', 0)
        
        # 瀹夊叏鍦板皢鍔ㄤ綔杞崲涓烘暣鏁?
        def safe_int_conversion(value):
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    return int(value.item())
                else:
                    return int(value[0])  # 鍙栫涓€涓厓绱?
            elif isinstance(value, (float, np.floating)):
                return int(value)
            else:
                return 0
        
        vehicle_action = safe_int_conversion(vehicle_action)
        rsu_action = safe_int_conversion(rsu_action)
        uav_action = safe_int_conversion(uav_action)
        
        # 5^3 = 125 绉嶇粍鍚?
        return vehicle_action * 25 + rsu_action * 5 + uav_action
    
    def _safe_int_conversion(self, value) -> int:
        """瀹夊叏鍦板皢涓嶅悓绫诲瀷杞崲涓烘暣鏁?""
        if isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return int(value.item())
            else:
                return int(value[0])  # 鍙栫涓€涓厓绱?
        elif isinstance(value, (float, np.floating)):
            return int(round(value))
        else:
            return 0  # 瀹夊叏鍥為€€鍊?


def train_single_algorithm(algorithm: str, num_episodes: Optional[int] = None, eval_interval: Optional[int] = None,
                          save_interval: Optional[int] = None, enable_realtime_vis: bool = False,
                          vis_port: int = 5000, silent_mode: bool = False, override_scenario: Optional[Dict[str, Any]] = None,
                          use_enhanced_cache: bool = False, disable_migration: bool = False,
                          enforce_offload_mode: Optional[str] = None, fixed_offload_policy: Optional[str] = None,
                          resume_from: Optional[str] = None, resume_lr_scale: Optional[float] = None,
                          joint_controller: bool = False, enable_advanced_vis: bool = False) -> Dict:
    """璁粌鍗曚釜绠楁硶
    
    Args:
        algorithm: 绠楁硶鍚嶇О
        num_episodes: 璁粌杞
        eval_interval: 璇勪及闂撮殧
        save_interval: 淇濆瓨闂撮殧
        enable_realtime_vis: 鏄惁鍚敤瀹炴椂鍙鍖?
        vis_port: 鍙鍖栨湇鍔″櫒绔彛
        silent_mode: 闈欓粯妯″紡锛岃烦杩囩敤鎴蜂氦浜掞紙鐢ㄤ簬鎵归噺瀹為獙锛?
        resume_from: 宸茶缁冩ā鍨嬭矾寰勶紙.pth 鎴栫洰褰曞墠缂€锛夛紝鐢ㄤ簬warm-start缁х画璁粌
        resume_lr_scale: Warm-start鍚庡瀛︿範鐜囩殑缂╂斁绯绘暟锛堥粯璁?.5锛孨one琛ㄧず淇濇寔鍘熷€硷級
        enable_advanced_vis: 鏄惁鍚敤楂樼璁粌鍙鍖?
    """
    # 瀵煎叆浠诲姟鍒嗗竷缁熻妯″潡
    from utils.training_analytics_integration import TaskAnalyticsTracker
    
    # 浣跨敤閰嶇疆涓殑榛樿鍊?
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes

    # 鍏佽鐢ㄧ幆澧冨彉閲忓揩閫熼噸璁惧鍔辨潈閲?鐩爣锛屼究浜庨珮璐熻浇鍦烘櫙鏀舵暃
    _apply_reward_overrides_from_env()
    
    # 馃敡 鑷姩璋冩暣璇勪及闂撮殧鍜屼繚瀛橀棿闅?
    def auto_adjust_intervals(total_episodes: int):
        """鏍规嵁鎬昏疆鏁拌嚜鍔ㄨ皟鏁撮棿闅?""
        # 璇勪及闂撮殧锛氭€昏疆鏁扮殑5-8%锛岃寖鍥碵10, 100]
        auto_eval = max(10, min(100, int(total_episodes * 0.06)))
        
        # 淇濆瓨闂撮殧锛氭€昏疆鏁扮殑15-20%锛岃寖鍥碵50, 500]  
        auto_save = max(50, min(500, int(total_episodes * 0.18)))
        
        return auto_eval, auto_save
    
    # 搴旂敤鑷姩璋冩暣锛堜粎褰撶敤鎴锋湭鎸囧畾鏃讹級
    if eval_interval is None or save_interval is None:
        auto_eval, auto_save = auto_adjust_intervals(num_episodes)
        if eval_interval is None:
            eval_interval = auto_eval
        if save_interval is None:
            save_interval = auto_save
    
    # 鏈€缁堝洖閫€鍒伴厤缃粯璁ゅ€?
    if eval_interval is None:
        eval_interval = config.experiment.eval_interval
    if save_interval is None:
        save_interval = config.experiment.save_interval
    
    print(f"\n>> 寮€濮媨algorithm}鍗曟櫤鑳戒綋绠楁硶璁粌")
    print("=" * 60)
    
    # 鍒涘缓璁粌鐜锛堝簲鐢ㄩ澶栧満鏅鐩栵級
    training_env = SingleAgentTrainingEnvironment(
        algorithm,
        override_scenario=override_scenario,
        use_enhanced_cache=use_enhanced_cache,
        disable_migration=disable_migration,
        enforce_offload_mode=enforce_offload_mode,
        fixed_offload_policy=fixed_offload_policy,
        joint_controller=joint_controller,
    )
    canonical_algorithm = training_env.algorithm
    if canonical_algorithm != algorithm:
        print(f"鈿欙笍  瑙勮寖鍖栫畻娉曟爣璇? {canonical_algorithm}")
    algorithm = canonical_algorithm

    resume_loaded = False
    resume_target_path = None
    if resume_from:
        loader = getattr(training_env.agent_env, 'load_models', None)
        if callable(loader):
            try:
                resume_target_path = loader(resume_from) or resume_from
                resume_loaded = True
                print(f"鈾伙笍  浠庡凡鏈夋ā鍨嬪姞杞芥垚鍔? {resume_target_path}")
            except Exception as exc:  # pragma: no cover - 瀹归敊璺緞
                print(f"鈿狅笍  鍔犺浇宸叉湁妯″瀷澶辫触 ({resume_from}): {exc}")
        else:
            print("鈿狅笍  褰撳墠绠楁硶鐜涓嶆敮鎸佸姞杞藉凡鏈夋ā鍨嬶紝蹇界暐 --resume-from")

        if resume_loaded:
            agent_obj = getattr(training_env.agent_env, 'agent', None)
            warmup_adjusted = False
            original_warmup = 0
            new_warmup = 0
            if agent_obj and hasattr(agent_obj, 'config') and hasattr(agent_obj.config, 'warmup_steps'):
                original_warmup = int(getattr(agent_obj.config, 'warmup_steps', 0) or 0)
                new_warmup = max(500, original_warmup // 4) if original_warmup else 500
                if original_warmup and new_warmup < original_warmup:
                    agent_obj.config.warmup_steps = new_warmup
                    warmup_adjusted = True
            if warmup_adjusted:
                print(f"   鈥?Warm-up 姝ユ暟鐢?{original_warmup} 缂╁噺鑷?{new_warmup}锛屽姞閫熺粡楠岀紦鍐查噸鏂板～鍏?)

            lr_scale_value = resume_lr_scale if resume_lr_scale is not None else 0.5
            lr_info = None
            lr_callback = getattr(training_env.agent_env, 'apply_late_stage_lr', None)
            if callable(lr_callback) and lr_scale_value:
                try:
                    lr_info = lr_callback(factor=lr_scale_value, min_lr=5e-5)
                except Exception:
                    lr_info = None
            elif agent_obj and hasattr(agent_obj, 'apply_lr_schedule') and lr_scale_value:
                try:
                    lr_info = agent_obj.apply_lr_schedule(factor=lr_scale_value, min_lr=5e-5)
                except Exception:
                    lr_info = None
            if lr_info and isinstance(lr_info, dict):
                print(f"   鈥?瀛︿範鐜囩缉鏀? actor_lr={lr_info.get('actor_lr', 0):.2e}, critic_lr={lr_info.get('critic_lr', 0):.2e}")
            elif resume_lr_scale:
                print("   鈥?瀛︿範鐜囩缉鏀捐姹傛湭鎵ц锛堝綋鍓嶇畻娉曠幆澧冩湭瀹炵幇 apply_lr_schedule锛?)

    lr_decay_episode: Optional[int] = None
    late_stage_lr_factor = 0.5
    lr_decay_applied = resume_loaded  # warm-start 宸茬粡缂╂斁杩囦竴娆″涔犵巼
    if algorithm.upper() == 'TD3' and num_episodes >= 1200:
        lr_decay_episode = 1200

    # 馃寪 鍒涘缓瀹炴椂鍙鍖栧櫒锛堝鏋滃惎鐢級
    visualizer = None
    advanced_visualizer = None
    
    # 馃帹 浼樺厛浣跨敤楂樼鍙鍖栵紙鏇村ソ鐨勬樉绀烘晥鏋滐級
    if enable_advanced_vis and ADVANCED_VIS_AVAILABLE:
        print("馃帹 鍚姩楂樼璁粌鍙鍖?Dashboard")
        advanced_visualizer = create_advanced_visualizer(max_history=min(500, num_episodes))  # type: ignore[name-defined]
        advanced_visualizer.start(interval=1000)  # 姣忕鍒锋柊涓€娆?
        print("鉁?楂樼鍙鍖栧凡鍚敤")
        print("   - 鎸?'p' 鏆傚仠/缁х画")
        print("   - 鎸?'s' 淇濆瓨鎴浘")
        print("   - 鎸?'q' 閫€鍑?)
    elif enable_advanced_vis and not ADVANCED_VIS_AVAILABLE:
        print("鈿狅笍  楂樼鍙鍖栨湭鍚敤锛堢己灏戜緷璧栧寘锛?)
    
    # 馃寪 Fallback鍒癢eb鍙鍖?
    if enable_realtime_vis and REALTIME_AVAILABLE and not advanced_visualizer:
        print(f"馃寪 鍚姩瀹炴椂鍙鍖栨湇鍔″櫒 (绔彛: {vis_port})")
        # 鍏佽閫氳繃鐜鍙橀噺瑕嗙洊鍙鍖栧睍绀哄悕锛堢敤浜庝袱闃舵鏍囩锛?
        display_name = os.environ.get('ALGO_DISPLAY_NAME', algorithm)
        visualizer = create_visualizer(  # type: ignore[name-defined]
            algorithm=display_name,
            total_episodes=num_episodes,
            port=vis_port,
            auto_open=True
        )
        print(f"鉁?瀹炴椂鍙鍖栧凡鍚敤锛岃闂?http://localhost:{vis_port}")
    elif enable_realtime_vis and not REALTIME_AVAILABLE:
        print("鈿狅笍  瀹炴椂鍙鍖栨湭鍚敤锛堢己灏戜緷璧栧寘锛?)
    
    print(f"璁粌閰嶇疆:")
    print(f"  绠楁硶: {algorithm}")
    print(f"  鎬昏疆娆? {num_episodes}")
    print(f"  璇勪及闂撮殧: {eval_interval} (鑷姩璋冩暣)" if eval_interval != config.experiment.eval_interval else f"  璇勪及闂撮殧: {eval_interval}")
    print(f"  淇濆瓨闂撮殧: {save_interval} (鑷姩璋冩暣)" if save_interval != config.experiment.save_interval else f"  淇濆瓨闂撮殧: {save_interval}")
    print(f"  楂樼鍙鍖? {'鍚敤 鉁? if advanced_visualizer else '绂佺敤'}")
    print(f"  瀹炴椂鍙鍖? {'鍚敤 鉁? if visualizer else '绂佺敤'}")
    if hasattr(config, 'rl'):
        print(
            f"  濂栧姳鏉冮噸: 寤惰繜={getattr(config.rl, 'reward_weight_delay', 0.0):.2f}, "
            f"鑳借€?{getattr(config.rl, 'reward_weight_energy', 0.0):.2f}, "
            f"涓㈠純={getattr(config.rl, 'reward_penalty_dropped', 0.0):.2f}"
        )
        print(
            f"  鐩爣绾︽潫: 鏃跺欢鈮getattr(config.rl, 'latency_target', 0.0):.2f}s, "
            f"鑳借€椻墹{getattr(config.rl, 'energy_target', 0.0):.0f}J"
        )
    print("-" * 60)
    
    # 鍒涘缓缁撴灉鐩綍
    os.makedirs(f"results/single_agent/{algorithm.lower()}", exist_ok=True)
    os.makedirs(f"results/models/single_agent/{algorithm.lower()}", exist_ok=True)
    
    # 馃幆 鍒濆鍖栦换鍔″鐞嗘柟寮忓垎甯冪粺璁¤窡韪櫒
    # 鏍规嵁episode鏁拌嚜鍔ㄨ皟鏁存棩蹇楄緭鍑洪棿闅?
    log_interval = max(1, num_episodes // 20) if num_episodes > 0 else 10
    analytics_tracker = TaskAnalyticsTracker(
        enable_logging=True,
        log_interval=log_interval
    )
    print(f"\n馃搳 宸插惎鐢ㄤ换鍔″鐞嗘柟寮忓垎甯冪粺璁★紙姣弡log_interval}涓猠pisode杈撳嚭涓€娆★級")
    
    # 璁粌寰幆
    # 馃敡 淇锛歱er-step濂栧姳鑼冨洿绾︿负-2.0鍒?0.5锛屽垵濮嬪€煎簲鐩稿簲璋冩暣
    best_avg_reward = -10.0  # per-step濂栧姳鍒濆闃堝€硷紙璐熷€艰秺澶ц秺濂斤級
    training_start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 馃幆 寮€濮嬭褰曡episode鐨勪换鍔″垎甯冪粺璁?
        analytics_tracker.start_episode(episode)
        
        # 杩愯璁粌杞
        episode_result = training_env.run_episode(episode)
        avg_reward_safe = float(episode_result.get('avg_reward', 0.0) or 0.0)
        if not np.isfinite(avg_reward_safe):
            avg_reward_safe = 0.0
        episode_result['avg_reward'] = avg_reward_safe
        episode_result['episode_reward'] = float(episode_result.get('episode_reward', avg_reward_safe) or avg_reward_safe)
        
        # 馃幆 璁板綍鏈琫pisode鍐呮墍鏈夌殑step缁熻
        step_stats_list = episode_result.get('step_stats_list', [])
        for step_idx, step_stats in enumerate(step_stats_list):
            analytics_tracker.record_step(step_idx, step_stats)
        
        # 馃幆 缁撴潫璇pisode鐨勪换鍔″垎甯冪粺璁?
        episode_stats = analytics_tracker.end_episode()
        
        # 璁板綍璁粌鏁版嵁
        training_env.episode_rewards.append(episode_result['avg_reward'])
        
        episode_steps = episode_result.get('steps', config.experiment.max_steps_per_episode)
        
        # 馃攧 閲嶈锛氬浜嶰PTIMIZED_TD3锛屾洿鏂癮gent鐨別pisode璁℃暟锛堝府鍔╅伩鍏嶅眬閮ㄦ渶浼橈級
        if algorithm.upper() == 'OPTIMIZED_TD3' and hasattr(training_env.agent_env, 'agent'):
            agent = training_env.agent_env.agent
            if hasattr(agent, 'set_episode_count'):
                agent.set_episode_count(episode, episode_result['avg_reward'])
            
            # 馃敟 鏂板锛氭鏌ユ槸鍚﹀簲璇ユ彁鍓嶇粓姝㈣缁?(600杞悗)
            if hasattr(agent, 'check_early_stopping'):
                if agent.check_early_stopping():
                    print(f"\n鉁?璁粌鍦‥pisode {episode}鎻愬墠缁堟锛屽凡鏀舵暃")
                    # 鏇存柊num_episodes浠ユ彁鍓嶉€€鍑?
                    num_episodes = episode
                    break
        
        # 馃攧 閽堝OPTIMIZED_TD3鐗瑰垾澶勭悊锛氭洿鏂版帰绱㈤噸鍚厤缃?
        if algorithm.upper() == 'OPTIMIZED_TD3' and hasattr(training_env.agent_env, 'agent'):
            agent = training_env.agent_env.agent
            if hasattr(agent, 'exploration_reset_interval'):
                # 鎺㈢储閲嶅惎闂撮殧鍙互鏍规嵁闇€瑕佽皟鏁达紙鐩墠100episode锛?
                pass
        per_step_reward = episode_result['avg_reward'] / max(1, episode_steps)
        training_env.performance_tracker['recent_step_rewards'].update(per_step_reward)
        
        system_metrics = episode_result['system_metrics']
        training_env.performance_tracker['recent_delays'].update(system_metrics.get('avg_task_delay', 0))
        training_env.performance_tracker['recent_energy'].update(system_metrics.get('total_energy_consumption', 0))
        training_env.performance_tracker['recent_completion'].update(system_metrics.get('task_completion_rate', 0))
        
        # 馃帹 鏇存柊楂樼鍙鍖?
        if advanced_visualizer:
            # 鏀堕泦璇︾粏鎸囨爣
            vis_metrics = {
                'reward': episode_result['avg_reward'],
                'loss': episode_result.get('loss', 0),  # 濡傛灉鏈夋崯澶卞€?
                'hit_rate': system_metrics.get('cache_hit_rate', 0),
                'delay': system_metrics.get('avg_task_delay', 0) * 1000,  # 杞崲涓簃s
                'energy': system_metrics.get('total_energy_consumption', 0),
                'success_rate': system_metrics.get('task_completion_rate', 0),
                'action': episode_result.get('last_action'),  # 鏈€鍚庝竴涓姩浣?
                'gradient_norm': episode_result.get('gradient_norm')  # 濡傛灉鏈夋搴﹁寖鏁?
            }
            advanced_visualizer.update(episode, vis_metrics)
            
            # 瀹氭湡淇濆瓨鍙鍖栨埅鍥?
            if episode % save_interval == 0:
                advanced_visualizer.save(f"results/single_agent/{algorithm.lower()}/viz_checkpoint_{episode}.png")
        
        # 馃寪 鏇存柊瀹炴椂鍙鍖?
        if visualizer:
            vis_metrics = {
                'avg_delay': system_metrics.get('avg_task_delay', 0),
                'total_energy': system_metrics.get('total_energy_consumption', 0),
                'task_completion_rate': system_metrics.get('task_completion_rate', 0),
                'cache_hit_rate': system_metrics.get('cache_hit_rate', 0),
                'data_loss_ratio_bytes': system_metrics.get('data_loss_ratio_bytes', 0),
                'migration_success_rate': system_metrics.get('migration_success_rate', 0),
                'local_tasks_count': system_metrics.get('local_tasks_count', 0),
                'rsu_tasks_count': system_metrics.get('rsu_tasks_count', 0),
                'uav_tasks_count': system_metrics.get('uav_tasks_count', 0)
            }
            visualizer.update(episode, episode_result['avg_reward'], vis_metrics)
        
        episode_time = time.time() - episode_start_time
        
        # 瀹氭湡杈撳嚭杩涘害
        if episode % 10 == 0:
            avg_reward_step = training_env.performance_tracker['recent_step_rewards'].get_average()
            avg_delay = training_env.performance_tracker['recent_delays'].get_average()
            avg_completion = training_env.performance_tracker['recent_completion'].get_average()
            
            print(f"杞 {episode:4d}/{num_episodes}:")
            print(f"  骞冲潎姣忔濂栧姳: {avg_reward_step:8.3f}")
            print(f"  骞冲潎鏃跺欢: {avg_delay:8.3f}s")
            print(f"  瀹屾垚鐜?   {avg_completion:8.1%}")
            print(f"  杞鐢ㄦ椂: {episode_time:6.3f}s")
        
        # 璇勪及妯″瀷
        if episode % eval_interval == 0:
            eval_result = evaluate_single_model(algorithm, training_env, episode)
            print(f"\n馃搳 杞 {episode} 璇勪及缁撴灉:")
            print(f"  Per-Step濂栧姳: {eval_result['avg_reward']:.3f}")
            print(f"  璇勪及鏃跺欢: {eval_result['avg_delay']:.3f}s")
            print(f"  璇勪及瀹屾垚鐜? {eval_result['completion_rate']:.1%}")
            
            # 淇濆瓨鏈€浣虫ā鍨?
            if eval_result['avg_reward'] > best_avg_reward:
                best_avg_reward = eval_result['avg_reward']
                best_model_base = f"results/models/single_agent/{algorithm.lower()}/best_model"
                saved_target = training_env.agent_env.save_models(best_model_base)
                saved_display = saved_target or best_model_base
                print(f"  馃捑 淇濆瓨鏈€浣虫ā鍨?-> {saved_display} (Per-Step濂栧姳: {best_avg_reward:.3f})")
        
        # 杈惧埌鍚庢湡闃舵鏃剁缉鏀綯D3瀛︿範鐜囷紙涓€娆℃€э級
        if (lr_decay_episode is not None and not lr_decay_applied and episode >= lr_decay_episode):
            lr_info = None
            lr_callback = getattr(training_env.agent_env, 'apply_late_stage_lr', None)
            if callable(lr_callback):
                lr_info = lr_callback(factor=late_stage_lr_factor, min_lr=5e-5)
                lr_decay_applied = True
            elif hasattr(training_env.agent_env, 'agent'):
                agent_obj = getattr(training_env.agent_env, 'agent')
                if hasattr(agent_obj, 'apply_lr_schedule'):
                    lr_info = agent_obj.apply_lr_schedule(factor=late_stage_lr_factor, min_lr=5e-5)
                    lr_decay_applied = True
            if lr_info and isinstance(lr_info, dict):
                print(
                    f"馃敡 绗瑊episode}杞Е鍙慣D3瀛︿範鐜囩缉鏀?-> "
                    f"actor_lr={lr_info.get('actor_lr', 0):.2e}, critic_lr={lr_info.get('critic_lr', 0):.2e}"
                )

        # 瀹氭湡淇濆瓨妯″瀷
        if episode % save_interval == 0:
            checkpoint_base = f"results/models/single_agent/{algorithm.lower()}/checkpoint_{episode}"
            checkpoint_path = training_env.agent_env.save_models(checkpoint_base)
            checkpoint_display = checkpoint_path or checkpoint_base
            print(f"馃捑 淇濆瓨妫€鏌ョ偣: {checkpoint_display}")
    
    # 璁粌瀹屾垚
    total_training_time = time.time() - training_start_time
    
    # 馃帹 淇濆瓨楂樼鍙鍖栨渶缁堢粨鏋?
    if advanced_visualizer:
        final_viz_path = f"results/single_agent/{algorithm.lower()}/final_training_viz.png"
        advanced_visualizer.save(final_viz_path)
        print(f"馃捑 楂樼鍙鍖栧凡淇濆瓨: {final_viz_path}")
    
    # 馃寪 鏍囪瀹炴椂鍙鍖栧畬鎴?
    if visualizer:
        visualizer.complete()
        print(f"鉁?瀹炴椂鍙鍖栧凡鏍囪瀹屾垚")
    
    print("\n" + "=" * 60)
    print(f"馃帀 {algorithm}璁粌瀹屾垚!")
    print(f"鈴憋笍  鎬昏缁冩椂闂? {total_training_time/3600:.2f} 灏忔椂")
    print(f"馃弳 鏈€浣砅er-Step濂栧姳: {best_avg_reward:.3f}")
    
    # 馃搳 杈撳嚭浠诲姟澶勭悊鏂瑰紡鍒嗗竷缁熻
    print("\n" + "=" * 60)
    print("馃搳 浠诲姟澶勭悊鏂瑰紡鍒嗗竷缁熻")
    print("=" * 60)
    
    # 鎵撳嵃璁粌姹囨€荤粺璁?
    analytics_tracker.print_training_summary()
    
    # 鎵撳嵃鏈€杩慛涓猠pisode鐨勮缁嗙粺璁?
    analytics_tracker.print_summary(top_n=min(20, num_episodes))
    
    # 瀵煎嚭CSV鏁版嵁鐢ㄤ簬鍚庣画鍒嗘瀽
    csv_export_path = f"results/single_agent/{algorithm.lower()}/task_distribution_analysis.csv"
    analytics_tracker.export_csv(csv_export_path)
    
    # 鑾峰彇婕斿寲瓒嬪娍
    evolution_trends = analytics_tracker.get_evolution_trend()
    if evolution_trends and evolution_trends.get('episodes'):
        print(f"\n馃搱 浠诲姟澶勭悊鏂瑰紡婕斿寲瓒嬪娍鍒嗘瀽:")
        print(f"   - 鏈湴澶勭悊鍗犳瘮: {evolution_trends['local_ratio'][-1]:.1%} (鍒濆: {evolution_trends['local_ratio'][0]:.1%})")
        print(f"   - RSU澶勭悊鍗犳瘮: {evolution_trends['rsu_ratio'][-1]:.1%} (鍒濆: {evolution_trends['rsu_ratio'][0]:.1%})")
        print(f"   - UAV澶勭悊鍗犳瘮: {evolution_trends['uav_ratio'][-1]:.1%} (鍒濆: {evolution_trends['uav_ratio'][0]:.1%})")
        print(f"   - 浠诲姟鎴愬姛鐜? {evolution_trends['success_ratio'][-1]:.1%} (鍒濆: {evolution_trends['success_ratio'][0]:.1%})")
    
    # 鏀堕泦绯荤粺缁熻淇℃伅鐢ㄤ簬鎶ュ憡
    simulator_stats = {}
    
    # 馃彚 鏄剧ず涓ぎRSU璋冨害鍣ㄦ姤鍛?
    try:
        central_report = training_env.simulator.get_central_scheduling_report()
        if central_report.get('status') != 'not_available' and central_report.get('status') != 'error':
            print(f"\n馃彚 涓ぎRSU楠ㄥ共璋冨害鍣ㄦ€荤粨:")
            print(f"   馃搳 璋冨害璋冪敤娆℃暟: {central_report.get('scheduling_calls', 0)}")
            
            scheduler_status = central_report.get('central_scheduler_status', {})
            if 'global_metrics' in scheduler_status:
                metrics = scheduler_status['global_metrics']
                print(f"   鈿栵笍 璐熻浇鍧囪　鎸囨暟: {metrics.get('load_balance_index', 0.0):.3f}")
                print(f"   馃挌 绯荤粺鍋ュ悍鐘舵€? {scheduler_status.get('system_health', 'N/A')}")
                
                # 鏀堕泦璋冨害鍣ㄧ粺璁′俊鎭?
                simulator_stats['scheduling_calls'] = central_report.get('scheduling_calls', 0)
                simulator_stats['load_balance_index'] = metrics.get('load_balance_index', 0.0)
                simulator_stats['system_health'] = scheduler_status.get('system_health', 'N/A')
            
            # 鏄剧ず鍚凴SU璐熻浇鍒嗗竷
            rsu_details = central_report.get('rsu_details', {})
            if rsu_details:
                print(f"   馃摗 鍚凴SU璐熻浇鐘舵€?")
                for rsu_id, details in rsu_details.items():
                    print(f"      {rsu_id}: CPU璐熻浇={details['cpu_usage']:.1%}, 浠诲姟闃熷垪={details['queue_length']}")
        else:
            print(f"馃搵 涓ぎ璋冨害鍣ㄧ姸鎬? {central_report.get('message', '鏈惎鐢?)}")
        
        # 馃攲 鏄剧ず鏈夌嚎鍥炰紶缃戠粶缁熻
        rsu_migration_delay = training_env.simulator.stats.get('rsu_migration_delay', 0.0)
        rsu_migration_energy = training_env.simulator.stats.get('rsu_migration_energy', 0.0)
        rsu_migration_data = training_env.simulator.stats.get('rsu_migration_data', 0.0)
        backhaul_collection_delay = training_env.simulator.stats.get('backhaul_collection_delay', 0.0)
        backhaul_command_delay = training_env.simulator.stats.get('backhaul_command_delay', 0.0)
        backhaul_total_energy = training_env.simulator.stats.get('backhaul_total_energy', 0.0)
        
        # 馃殫 鏄剧ず鍚勭杩佺Щ缁熻
        handover_migrations = training_env.simulator.stats.get('handover_migrations', 0)
        uav_migration_count = training_env.simulator.stats.get('uav_migration_count', 0)
        uav_migration_distance = training_env.simulator.stats.get('uav_migration_distance', 0.0)
        
        # 鏀堕泦杩佺Щ缁熻淇℃伅
        simulator_stats['rsu_migration_delay'] = rsu_migration_delay
        simulator_stats['rsu_migration_energy'] = rsu_migration_energy
        simulator_stats['rsu_migration_data'] = rsu_migration_data
        simulator_stats['backhaul_total_energy'] = backhaul_total_energy
        simulator_stats['handover_migrations'] = handover_migrations
        simulator_stats['uav_migration_count'] = uav_migration_count
        
        if rsu_migration_data > 0 or backhaul_total_energy > 0 or handover_migrations > 0 or uav_migration_count > 0:
            print(f"\n馃攲 鏈夌嚎鍥炰紶缃戠粶涓庤縼绉荤粺璁?")
            print(f"   馃摗 RSU杩佺Щ鏁版嵁: {rsu_migration_data:.1f}MB")
            print(f"   鈴憋笍 RSU杩佺Щ寤惰繜: {rsu_migration_delay*1000:.1f}ms")
            print(f"   鈿?RSU杩佺Щ鑳借€? {rsu_migration_energy:.2f}J")
            print(f"   馃搳 淇℃伅鏀堕泦寤惰繜: {backhaul_collection_delay*1000:.1f}ms")
            print(f"   馃摛 鎸囦护鍒嗗彂寤惰繜: {backhaul_command_delay*1000:.1f}ms")
            print(f"   馃攱 鍥炰紶缃戠粶鎬昏兘鑰? {backhaul_total_energy:.2f}J")
            if handover_migrations > 0:
                print(f"   馃殫 杞﹁締璺熼殢杩佺Щ: {handover_migrations} 娆?)
            if uav_migration_count > 0:
                avg_distance = uav_migration_distance / uav_migration_count if uav_migration_count > 0 else 0
                print(f"   馃殎 UAV杩佺Щ: {uav_migration_count} 娆? 骞冲潎璺濈{avg_distance:.1f}m")
    except Exception as e:
        print(f"鈿狅笍 涓ぎ璋冨害鎶ュ憡鑾峰彇澶辫触: {e}")
    
    # 淇濆瓨璁粌缁撴灉
    results = save_single_training_results(algorithm, training_env, total_training_time, override_scenario=override_scenario)
    
    # 缁樺埗璁粌鏇茬嚎
    plot_single_training_curves(algorithm, training_env)
    
    # 鐢熸垚HTML璁粌鎶ュ憡
    print("\n" + "=" * 60)
    print("馃摑 鐢熸垚璁粌鎶ュ憡...")
    
    try:
        report_generator = HTMLReportGenerator()
        html_content = report_generator.generate_full_report(
            algorithm=algorithm,
            training_env=training_env,
            training_time=total_training_time,
            results=results,
            simulator_stats=simulator_stats
        )
        
        # 鐢熸垚鎶ュ憡鏂囦欢鍚?
        timestamp = generate_timestamp()
        report_filename = f"training_report_{timestamp}.html" if timestamp else "training_report.html"
        report_path = f"results/single_agent/{algorithm.lower()}/{report_filename}"
        
        print(f"鉁?璁粌鎶ュ憡宸茬敓鎴?)
        print(f"馃搫 鎶ュ憡鍖呭惈:")
        print(f"   - 鎵ц鎽樿涓庡叧閿寚鏍?)
        print(f"   - 璁粌閰嶇疆璇︽儏")
        print(f"   - 鎬ц兘鎸囨爣鍙鍖栧浘琛?)
        print(f"   - 璇︾粏鐨勭郴缁熺粺璁′俊鎭?)
        print(f"   - 鑷€傚簲鎺у埗鍣ㄥ垎鏋?)
        print(f"   - 浼樺寲寤鸿涓庣粨璁?)
        
        # 璇㈤棶鐢ㄦ埛鏄惁淇濆瓨鎶ュ憡锛堥潤榛樻ā寮忎笅鑷姩淇濆瓨锛?
        if silent_mode:
            # 闈欓粯妯″紡锛氳嚜鍔ㄤ繚瀛橈紝涓嶆墦寮€娴忚鍣?
            if report_generator.save_report(html_content, report_path):
                print(f"鉁?鎶ュ憡宸茶嚜鍔ㄤ繚瀛樺埌: {report_path}")
            else:
                print("鉂?鎶ュ憡淇濆瓨澶辫触")
        else:
            # 浜や簰妯″紡锛氳闂敤鎴?
            print("\n" + "-" * 60)
            save_choice = input("馃捑 鏄惁淇濆瓨HTML璁粌鎶ュ憡? (y/n, 榛樿y): ").strip().lower()
            
            if save_choice in ['', 'y', 'yes', '鏄?]:
                if report_generator.save_report(html_content, report_path):
                    print(f"鉁?鎶ュ憡宸蹭繚瀛樺埌: {report_path}")
                    print(f"馃挕 鎻愮ず: 浣跨敤娴忚鍣ㄦ墦寮€璇ユ枃浠跺嵆鍙煡鐪嬪畬鏁存姤鍛?)
                    
                    # 灏濊瘯鑷姩鎵撳紑鎶ュ憡锛堝彲閫夛級
                    auto_open = input("馃寪 鏄惁鍦ㄦ祻瑙堝櫒涓墦寮€鎶ュ憡? (y/n, 榛樿n): ").strip().lower()
                    if auto_open in ['y', 'yes', '鏄?]:
                        import webbrowser
                        abs_path = os.path.abspath(report_path)
                        webbrowser.open(f'file://{abs_path}')
                        print("鉁?鎶ュ憡宸插湪娴忚鍣ㄤ腑鎵撳紑")
                else:
                    print("鉂?鎶ュ憡淇濆瓨澶辫触")
            else:
                print("鈩癸笍 鎶ュ憡鏈繚瀛?)
                print(f"馃挕 濡傞渶鏌ョ湅锛岃鎵嬪姩杩愯鎶ュ憡鐢熸垚鍔熻兘")
    
    except Exception as e:
        print(f"鈿狅笍 鐢熸垚璁粌鎶ュ憡鏃跺嚭閿? {e}")
        print("璁粌鏁版嵁宸叉甯镐繚瀛橈紝鍙◢鍚庢墜鍔ㄧ敓鎴愭姤鍛?)
    
    return results


def evaluate_single_model(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                         episode: int, num_eval_episodes: int = 5) -> Dict:
    """璇勪及鍗曟櫤鑳戒綋妯″瀷鎬ц兘 - 淇鐗堬紝闃叉inf鍜宯an"""
    import numpy as np
    
    eval_rewards = []
    eval_delays = []
    eval_completions = []
    
    def safe_value(value: float, default: float = 0.0, max_val: float = 1e6) -> float:
        """瀹夊叏澶勭悊鏁板€硷紝闃叉inf鍜宯an"""
        if np.isnan(value) or np.isinf(value):
            return default
        return np.clip(value, -max_val, max_val)
    
    eval_max_steps = getattr(config.experiment, 'max_steps_per_episode', 200)
    eval_max_steps = max(50, int(eval_max_steps))
    
    for _ in range(num_eval_episodes):
        state = training_env.reset_environment()
        episode_reward = 0.0
        episode_delay = 0.0
        episode_completion = 0.0
        steps = 0
        
        for step in range(eval_max_steps):
            if algorithm == "DQN":
                actions_result = training_env.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = training_env._encode_discrete_action(actions_dict)
            else:
                actions_result = training_env.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, tuple):  # PPO杩斿洖鍏冪粍
                    actions_dict = actions_result[0]
                elif isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = {}
                action = training_env._encode_continuous_action(actions_dict)
            
            # 璇勪及鏃朵篃浼犲叆鍔ㄤ綔瀛楀吀锛岀‘淇濆亸濂界敓鏁?
            next_state, reward, done, info = training_env.step(action, state, actions_dict)
            
            # 瀹夊叏澶勭悊濂栧姳鍜屾寚鏍?
            safe_reward = safe_value(reward, -10.0, 120.0)
            episode_reward += safe_reward
            
            system_metrics = info['system_metrics']
            safe_delay = safe_value(system_metrics.get('avg_task_delay', 0), 0.0, 10.0)
            safe_completion = safe_value(system_metrics.get('task_completion_rate', 0), 0.0, 1.0)
            
            episode_delay += safe_delay
            episode_completion += safe_completion
            steps += 1
            
            state = next_state
            
            if done:
                break
        
        # 瀹夊叏璁＄畻骞冲潎鍊?
        steps = max(1, steps)  # 闃叉闄ら浂
        eval_rewards.append(safe_value(episode_reward / steps, -20.0, 80.0))
        eval_delays.append(safe_value(episode_delay / steps, 0.0, 10.0))
        eval_completions.append(safe_value(episode_completion / steps, 0.0, 1.0))
    
    # 瀹夊叏璁＄畻鏈€缁堢粨鏋?
    if len(eval_rewards) == 0:
        return {'avg_reward': -1.0, 'avg_delay': 1.0, 'completion_rate': 0.0}
    
    avg_reward = safe_value(float(np.mean(eval_rewards)), -20.0, 80.0)
    avg_delay = safe_value(float(np.mean(eval_delays)), 0.0, 10.0)
    avg_completion = safe_value(float(np.mean(eval_completions)), 0.0, 1.0)
    
    return {
        'avg_reward': avg_reward,
        'avg_delay': avg_delay,
        'completion_rate': avg_completion
    }

 
def _finite_mean(values: List[float], default: float = 0.0) -> float:
    """璁＄畻鏈夐檺鍊肩殑鍧囧€硷紝杩囨护鎺塏aN/Inf銆?""
    finite_values: List[float] = []
    for v in values:
        try:
            val = float(v)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            finite_values.append(val)
    return float(np.mean(finite_values)) if finite_values else default


def _calculate_stable_delay_average(training_env: SingleAgentTrainingEnvironment) -> float:
    """
    璁＄畻绋冲畾鐨勬椂寤跺钩鍧囧€硷紝閬垮厤MovingAverage(100)鐨勮缁冩尝鍔ㄥ奖鍝?
    
    绛栫暐锛?
    1. 浼樺厛浣跨敤episode_metrics涓殑瀹屾暣鏁版嵁锛堝鏋滃彲鐢級
    2. 浣跨敤鍚?0%鐨勬暟鎹紙鎺掗櫎鍓嶆湡瀛︿範闃舵锛?
    3. 濡傛灉鏁版嵁涓嶈冻锛屽洖閫€鍒癕ovingAverage(100)
    
    Returns:
        float: 绋冲畾鐨勫钩鍧囨椂寤?
    """
    # 灏濊瘯浠巈pisode_metrics鑾峰彇瀹屾暣鏃跺欢鏁版嵁
    if hasattr(training_env, 'episode_metrics') and 'avg_delay' in training_env.episode_metrics:
        delay_history = training_env.episode_metrics['avg_delay']
        
        if len(delay_history) >= 100:
            # 浣跨敤鍚?0%鐨勬暟鎹紙鏇存垚鐔熺殑绛栫暐锛?
            half_point = len(delay_history) // 2
            converged_delays = delay_history[half_point:]
            return _finite_mean(converged_delays, training_env.performance_tracker['recent_delays'].get_average())
        elif len(delay_history) >= 50:
            # 濡傛灉涓嶈冻100杞紝浣跨敤鍚?0杞?
            return _finite_mean(delay_history[-30:], training_env.performance_tracker['recent_delays'].get_average())
        elif len(delay_history) > 0:
            # 鏁版嵁寰堝皯锛屼娇鐢ㄥ叏閮?
            return _finite_mean(delay_history, training_env.performance_tracker['recent_delays'].get_average())
    
    # 鍥為€€锛氫娇鐢∕ovingAverage
    recent_delay = training_env.performance_tracker['recent_delays'].get_average()
    return _finite_mean([recent_delay], 0.0)


def _calculate_stable_completion_average(training_env: SingleAgentTrainingEnvironment) -> float:
    """
    璁＄畻绋冲畾鐨勫畬鎴愮巼骞冲潎鍊?
    
    Returns:
        float: 绋冲畾鐨勫钩鍧囧畬鎴愮巼
    """
    # 灏濊瘯浠巈pisode_metrics鑾峰彇瀹屾暣瀹屾垚鐜囨暟鎹?
    if hasattr(training_env, 'episode_metrics') and 'task_completion_rate' in training_env.episode_metrics:
        completion_history = training_env.episode_metrics['task_completion_rate']
        
        if len(completion_history) >= 100:
            # 浣跨敤鍚?0%鐨勬暟鎹?
            half_point = len(completion_history) // 2
            converged_completions = completion_history[half_point:]
            return _finite_mean(converged_completions, training_env.performance_tracker['recent_completion'].get_average())
        elif len(completion_history) >= 50:
            # 濡傛灉涓嶈冻100杞紝浣跨敤鍚?0杞?
            return _finite_mean(completion_history[-30:], training_env.performance_tracker['recent_completion'].get_average())
        elif len(completion_history) > 0:
            # 鏁版嵁寰堝皯锛屼娇鐢ㄥ叏閮?
            return _finite_mean(completion_history, training_env.performance_tracker['recent_completion'].get_average())
    
    # 鍥為€€锛氫娇鐢∕ovingAverage
    recent_completion = training_env.performance_tracker['recent_completion'].get_average()
    return _finite_mean([recent_completion], 0.0)


def _calculate_raw_cost_for_training(training_env: SingleAgentTrainingEnvironment) -> float:
    """
    浠庤缁冨鍔辫绠梤aw_cost锛堝鍔辨湰韬氨鏄礋鎴愭湰锛?
    
    璁粌鏃讹細reward = -cost锛堟垚鏈秺浣庯紝濂栧姳瓒婇珮锛?
    鍥犳锛歳aw_cost = -reward
    
    Returns:
        float: raw_cost锛堟鍊硷紝瓒婂皬瓒婂ソ锛?
    """
    # 鑾峰彇鏀舵暃鍚庣殑骞冲潎濂栧姳
    if hasattr(training_env, 'episode_rewards') and len(training_env.episode_rewards) > 0:
        rewards = []
        for r in training_env.episode_rewards:
            try:
                val = float(r)
            except (TypeError, ValueError):
                continue
            if np.isfinite(val):
                rewards.append(val)
        if not rewards:
            rewards = [training_env.performance_tracker['recent_rewards'].get_average()]
        if len(rewards) >= 100:
            # 浣跨敤鍚?0%鏁版嵁锛堟敹鏁涘悗锛?
            half_point = len(rewards) // 2
            converged_rewards = rewards[half_point:]
            avg_reward = _finite_mean(converged_rewards, 0.0)
        elif len(rewards) >= 50:
            avg_reward = _finite_mean(rewards[-30:], 0.0)
        else:
            avg_reward = _finite_mean(rewards, 0.0)
    else:
        recent_avg = training_env.performance_tracker['recent_rewards'].get_average()
        avg_reward = _finite_mean([recent_avg], 0.0)
    
    # reward鏄礋鎴愭湰锛屾墍浠aw_cost = -reward
    raw_cost = -avg_reward
    return float(raw_cost)


def save_single_training_results(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                                training_time: float,
                                override_scenario: Optional[Dict[str, Any]] = None) -> Dict:
    """淇濆瓨璁粌缁撴灉"""
    # 鐢熸垚鏃堕棿鎴?
    timestamp = generate_timestamp()
    
    # 馃敡 鍚屾椂鎻愪緵Episode鎬诲鍔卞拰Per-Step骞冲潎濂栧姳
    reward_samples = list(training_env.episode_rewards[-100:]) if hasattr(training_env, 'episode_rewards') else []
    reward_samples.append(training_env.performance_tracker['recent_rewards'].get_average())
    recent_episode_reward = _finite_mean(reward_samples, 0.0)
    
    # 馃敡 浼樺寲锛氫娇鐢ㄥ疄闄呭钩鍧囨鏁拌绠?avg_step_reward
    if 'episode_steps' in training_env.episode_metrics and training_env.episode_metrics['episode_steps']:
        # 浣跨敤鏈€杩?00涓猠pisode鐨勫钩鍧囨鏁?
        recent_steps = training_env.episode_metrics['episode_steps'][-100:]
        avg_steps_per_episode = sum(recent_steps) / len(recent_steps)
    else:
        # 鍥為€€鍒伴厤缃殑榛樿鍊?
        avg_steps_per_episode = config.experiment.max_steps_per_episode
    
    avg_step_reward = recent_episode_reward / avg_steps_per_episode if avg_steps_per_episode else 0.0
    
    # 鑾峰彇缃戠粶鎷撴墤淇℃伅
    num_vehicles = len(training_env.simulator.vehicles)
    num_rsus = len(training_env.simulator.rsus)
    num_uavs = len(training_env.simulator.uavs)
    state_dim = getattr(training_env.agent_env, 'state_dim', 'N/A')
    
    # 馃啎 淇锛氭敹闆嗗畬鏁寸殑绯荤粺閰嶇疆鍙傛暟锛堢敤浜嶩TML鎶ュ憡鏄剧ず锛?
    # 鐩存帴浣跨敤宸插鍏ョ殑config瀵硅薄
    
    results = {
        'algorithm': algorithm,
        'agent_type': 'single_agent',
        'timestamp': timestamp,
        'training_start_time': datetime.now().isoformat(),
        'network_topology': {
            'num_vehicles': num_vehicles,
            'num_rsus': num_rsus,
            'num_uavs': num_uavs,
        },
        'state_dim': state_dim,
        'override_scenario': override_scenario,
        'training_config': {
            'num_episodes': len(training_env.episode_rewards),
            'training_time_hours': training_time / 3600,
            'max_steps_per_episode': config.experiment.max_steps_per_episode
        },
        # 馃啎 娣诲姞绯荤粺閰嶇疆鍙傛暟锛圚TML鎶ュ憡闇€瑕侊級
        'system_config': {
            'num_vehicles': num_vehicles,
            'num_rsus': num_rsus,
            'num_uavs': num_uavs,
            'simulation_time': config.simulation_time,
            'time_slot': config.time_slot,
            'device': str(config.device),
            'random_seed': config.random_seed,
        },
        # 馃啎 娣诲姞缃戠粶閰嶇疆鍙傛暟
        'network_config': {
            'bandwidth': config.network.bandwidth,
            'carrier_frequency': config.communication.carrier_frequency,
            'coverage_radius': config.network.coverage_radius,
        },
        # 馃啎 娣诲姞閫氫俊閰嶇疆鍙傛暟
        'communication_config': {
            'vehicle_tx_power': config.communication.vehicle_tx_power,
            'rsu_tx_power': config.communication.rsu_tx_power,
            'uav_tx_power': config.communication.uav_tx_power,
            'antenna_gain_vehicle': config.communication.antenna_gain_vehicle,
            'antenna_gain_rsu': config.communication.antenna_gain_rsu,
            'antenna_gain_uav': config.communication.antenna_gain_uav,
        },
        # 馃啎 娣诲姞璁＄畻鑳藉姏鍙傛暟
        'compute_config': {
            'vehicle_cpu_freq': config.compute.vehicle_cpu_freq,
            'rsu_cpu_freq': config.compute.rsu_cpu_freq,
            'uav_cpu_freq': config.compute.uav_cpu_freq,
            'vehicle_memory': getattr(config.compute, 'vehicle_memory', 4e9),
            'rsu_memory': getattr(config.compute, 'rsu_memory', 32e9),
            'uav_memory': getattr(config.compute, 'uav_memory', 16e9),
            'vehicle_static_power': config.compute.vehicle_static_power,
            'rsu_static_power': config.compute.rsu_static_power,
            'uav_static_power': getattr(config.compute, 'uav_static_power', 20.0),
        },
        # 馃啎 娣诲姞浠诲姟鍜岃縼绉诲弬鏁?
        'task_migration_config': {
            'task_arrival_rate': config.task.arrival_rate,
            'task_size_mean': sum(config.task.data_size_range) / 2,
            'task_size_std': (config.task.data_size_range[1] - config.task.data_size_range[0]) / 4,
            'task_cpu_cycles_mean': sum(config.task.compute_cycles_range) / 2,
            'task_cpu_cycles_std': (config.task.compute_cycles_range[1] - config.task.compute_cycles_range[0]) / 4,
            'task_deadline_mean': sum(config.task.deadline_range) / 2,
            'cache_capacity_rsu': config.cache.rsu_cache_capacity,
            'cache_capacity_uav': config.cache.uav_cache_capacity,
            'migration_threshold': getattr(config.migration, 'threshold', 0.8),
        },
        'episode_rewards': training_env.episode_rewards,
        'episode_metrics': training_env.episode_metrics,
        'final_performance': {
            # 鎻愪緵涓ょ濂栧姳鎸囨爣锛岀敤閫斾笉鍚?
            'avg_episode_reward': recent_episode_reward,  # Episode鎬诲鍔憋紙璁粌鐩爣锛?
            'avg_step_reward': avg_step_reward,           # 姣忔骞冲潎濂栧姳锛堝姣旇瘎浼帮級
            'avg_reward': avg_step_reward,  # 鍚戝悗鍏煎锛氶粯璁や娇鐢╬er-step锛堜笌鍙鍖栦竴鑷达級
            
            # 馃敡 淇锛氫娇鐢ㄦ洿绋冲畾鐨勫钩鍧囨柟娉曪紝閬垮厤MovingAverage(100)鐨勬尝鍔ㄥ奖鍝?
            'avg_delay': _calculate_stable_delay_average(training_env),
            'avg_completion': _calculate_stable_completion_average(training_env),
            
            # 馃幆 鏂板锛氭坊鍔燼vg_energy鍜宺aw_cost锛岀敤浜庝笌瀵规瘮瀹為獙涓€鑷?
            'avg_energy': _finite_mean(
                training_env.episode_metrics['total_energy'][len(training_env.episode_metrics['total_energy'])//2:]
                if training_env.episode_metrics.get('total_energy') else [],
                0.0
            ),
            'raw_cost': _calculate_raw_cost_for_training(training_env),
        }
    }
    
    print(f"馃搳 鏀堕泦鐨勯厤缃弬鏁?")
    print(f"   绯荤粺鎷撴墤: {num_vehicles}杞﹁締, {num_rsus}RSU, {num_uavs}UAV")
    print(f"   缃戠粶閰嶇疆: 甯﹀{config.network.bandwidth/1e6:.0f}MHz, 棰戠巼{config.communication.carrier_frequency/1e9:.1f}GHz")
    print(f"   浠诲姟鍙傛暟: 鍒拌揪鐜噞config.task.arrival_rate:.1f}, 鏁版嵁閲弡sum(config.task.data_size_range)/2/1e6:.1f}MB")
    
    # 馃幆 鎵撳嵃鍏抽敭鎬ц兘鎸囨爣
    final_perf = results['final_performance']
    print(f"\n馃幆 鏈€缁堟€ц兘鎸囨爣:")
    print(f"   Raw Cost: {final_perf.get('raw_cost', 'N/A'):.4f} (= -avg_reward锛屼笌瀵规瘮瀹為獙涓€鑷?")
    print(f"   Avg Reward: {final_perf.get('avg_reward', 0):.4f} (= -raw_cost锛岃缁冧紭鍖栫洰鏍?")
    print(f"   Avg Delay: {final_perf.get('avg_delay', 0):.4f}s")
    print(f"   Avg Energy: {final_perf.get('avg_energy', 0):.2f}J")
    print(f"   Completion Rate: {final_perf.get('avg_completion', 0):.1%}")
    
    # 浣跨敤鏃堕棿鎴虫枃浠跺悕
    filename = get_timestamped_filename("training_results")
    filepath = f"results/single_agent/{algorithm.lower()}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"馃捑 {algorithm}璁粌缁撴灉宸蹭繚瀛樺埌 {filepath}")
    
    return results


def plot_single_training_curves(algorithm: str, training_env: SingleAgentTrainingEnvironment):
    """缁樺埗璁粌鏇茬嚎 - 绠€娲佷紭缇庣増"""
    
    # 馃帹 浣跨敤鏂扮殑绠€娲佸彲瑙嗗寲绯荤粺
    from visualization.clean_charts import create_training_chart, cleanup_old_charts, plot_objective_function_breakdown
    
    # 鍒涘缓绠楁硶鐩綍
    algorithm_dir = f"results/single_agent/{algorithm.lower()}"
    
    # 娓呯悊鏃х殑鍐椾綑鍥捐〃
    cleanup_old_charts(algorithm_dir)
    
    # 鐢熸垚鏍稿績鍥捐〃
    chart_path = f"{algorithm_dir}/training_overview.png"
    create_training_chart(training_env, algorithm, chart_path)
    
    # 馃幆 鐢熸垚鐩爣鍑芥暟鍒嗚В鍥撅紙鏄剧ず鏃跺欢銆佽兘鑰椾袱椤规牳蹇冪洰鏍囩殑鏉冮噸璐＄尞锛?
    objective_path = f"{algorithm_dir}/objective_analysis.png"
    plot_objective_function_breakdown(training_env, algorithm, objective_path)
    
    print(f"馃搱 {algorithm} 璁粌鍙鍖栧凡瀹屾垚")
    print(f"   璁粌鎬昏: {chart_path}")
    print(f"   鐩爣鍒嗘瀽: {objective_path}")
    
    # 鐢熸垚璁粌鎬荤粨
    from visualization.clean_charts import get_summary_text
    summary = get_summary_text(training_env, algorithm)
    print(f"\n{summary}")


def compare_single_algorithms(algorithms: List[str], num_episodes: Optional[int] = None) -> Dict:
    """姣旇緝澶氫釜鍗曟櫤鑳戒綋绠楁硶鐨勬€ц兘"""
    # 浣跨敤閰嶇疆涓殑榛樿鍊?
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    print("\n馃敟 寮€濮嬪崟鏅鸿兘浣撶畻娉曟€ц兘姣旇緝")
    print("=" * 60)
    
    results = {}
    
    # 璁粌鎵€鏈夌畻娉?
    for algorithm in algorithms:
        print(f"\n寮€濮嬭缁?{algorithm}...")
        results[algorithm] = train_single_algorithm(algorithm, num_episodes)
    
    # 馃帹 鐢熸垚绠€娲佺殑瀵规瘮鍥捐〃
    from visualization.clean_charts import create_comparison_chart
    timestamp = generate_timestamp()
    comparison_chart_path = f"results/single_agent_comparison_{timestamp}.png" if timestamp else "results/single_agent_comparison.png"
    create_comparison_chart(results, comparison_chart_path)
    
    # 淇濆瓨姣旇緝缁撴灉
    timestamp = generate_timestamp()
    comparison_results = {
        'algorithms': algorithms,
        'agent_type': 'single_agent',
        'num_episodes': num_episodes,
        'timestamp': timestamp,
        'comparison_time': datetime.now().isoformat(),
        'results': results,
        'summary': {}
    }
    
    # 璁＄畻姹囨€荤粺璁?
    for algorithm, result in results.items():
        final_perf = result['final_performance']
        comparison_results['summary'][algorithm] = {
            'final_avg_reward': final_perf['avg_reward'],
            'final_avg_delay': final_perf['avg_delay'],
            'final_completion_rate': final_perf['avg_completion'],
            'training_time_hours': result['training_config']['training_time_hours']
        }
    
    # 浣跨敤鏃堕棿鎴虫枃浠跺悕
    comparison_filename = get_timestamped_filename("single_agent_comparison")
    with open(f"results/{comparison_filename}", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print("\n馃幆 鍗曟櫤鑳戒綋绠楁硶姣旇緝瀹屾垚锛?)
    print(f"馃搫 姣旇緝缁撴灉宸蹭繚瀛樺埌 results/{comparison_filename}")
    print(f"馃搳 瀵规瘮鍥捐〃宸蹭繚瀛樺埌 {comparison_chart_path}")
    
    return comparison_results




def main():
    """涓诲嚱鏁?""
    parser = argparse.ArgumentParser(description='鍗曟櫤鑳戒綋绠楁硶璁粌鑴氭湰')
    parser.add_argument('--algorithm', type=str, choices=['DDPG', 'TD3', 'TD3-LE', 'TD3_LE', 'TD3_LATENCY_ENERGY', 'DQN', 'PPO', 'SAC', 'CAM_TD3', 'OPTIMIZED_TD3'],
                       help='閫夋嫨璁粌绠楁硶')
    parser.add_argument('--episodes', type=int, default=None, help=f'璁粌杞 (榛樿: {config.experiment.num_episodes})')
    parser.add_argument('--eval_interval', type=int, default=None, help=f'璇勪及闂撮殧 (榛樿: {config.experiment.eval_interval})')
    parser.add_argument('--save_interval', type=int, default=None, help=f'淇濆瓨闂撮殧 (榛樿: {config.experiment.save_interval})')
    parser.add_argument('--compare', action='store_true', help='姣旇緝鎵€鏈夌畻娉?)
    parser.add_argument('--seed', type=int, default=None, help='瑕嗙洊闅忔満绉嶅瓙 (榛樿璇诲彇config鎴栫幆澧冨彉閲?')
    parser.add_argument('--num-vehicles', type=int, default=None, help='瑕嗙洊杞﹁締鏁伴噺鐢ㄤ簬瀹為獙')
    parser.add_argument('--force-offload', type=str, choices=['local', 'remote', 'local_only', 'remote_only'],
                        help='寮哄埗鍗歌浇妯″紡锛歭ocal/local_only 鎴?remote/remote_only')
    parser.add_argument('--fixed-offload-policy', type=str, 
                        choices=['random', 'greedy', 'local_only', 'rsu_only', 'round_robin', 'weighted'],
                        help='鍥哄畾鍗歌浇绛栫暐锛堜笉浣跨敤鏅鸿兘浣撳涔狅級锛歳andom/greedy/local_only/rsu_only/round_robin/weighted')
    # 馃寪 瀹炴椂鍙鍖栧弬鏁?(榛樿寮€鍚?
    parser.add_argument('--realtime-vis', action='store_true', default=True, help='鍚敤瀹炴椂鍙鍖?(榛樿寮€鍚?')
    parser.add_argument('--no-realtime-vis', action='store_false', dest='realtime_vis', help='绂佺敤瀹炴椂鍙鍖?)
    parser.add_argument('--vis-port', type=int, default=5000, help='瀹炴椂鍙鍖栨湇鍔″櫒绔彛 (榛樿: 5000)')
    # 馃帹 楂樼璁粌鍙鍖栧弬鏁?
    parser.add_argument('--advanced-vis', action='store_true', help='鍚敤楂樼璁粌鍙鍖?Dashboard')
    # 馃殌 澧炲己缂撳瓨鍙傛暟锛堥粯璁ゅ惎鐢級
    parser.add_argument('--no-enhanced-cache', action='store_true', 
                       help='绂佺敤澧炲己缂撳瓨绯荤粺锛堥粯璁ゅ惎鐢ㄥ垎灞侺1/L2 + 鐑害绛栫暐 + RSU鍗忎綔锛?)
    # 馃Л 涓ら樁娈电绾垮紑鍏筹紙Stage-1 棰勫垎閰?+ Stage-2 绮剧粏璋冨害锛?
    parser.add_argument('--two-stage', action='store_true', help='鍚敤涓ら樁娈垫眰瑙ｏ紙棰勫垎閰?绮剧粏璋冨害锛?)
    # 馃 鎸囧畾涓や釜闃舵鐨勭畻娉?
    parser.add_argument('--stage1-alg', type=str, default=None,
                        help='闃舵涓€绠楁硶锛坥ffloading 澶达級锛歨euristic|greedy|cache_first|distance_first')
    parser.add_argument('--stage2-alg', type=str, default=None,
                        help='闃舵浜岀畻娉曪紙缂撳瓨/杩佺Щ鎺у埗鐨凴L锛夛細TD3|SAC|DDPG|PPO|DQN|TD3-LE')
    # 馃幆 涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛圥hase 1 + Phase 2锛? 榛樿鍚敤
    parser.add_argument('--central-resource', action='store_true', default=True,
                        help='鍚敤涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛圥hase 1鍐崇瓥 + Phase 2鎵ц锛夛紝鎵╁睍鐘舵€?鍔ㄤ綔绌洪棿 [榛樿鍚敤]')
    parser.add_argument('--no-central-resource', action='store_false', dest='central_resource',
                        help='绂佺敤涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛屼娇鐢ㄦ爣鍑嗗潎鍖€璧勬簮鍒嗛厤')
    parser.add_argument('--silent-mode', action='store_true',
                        help='鍚敤闈欓粯妯″紡锛岃烦杩囪缁冪粨鏉熷悗鐨勪氦浜掓彁绀?)
    parser.add_argument('--resume-from', type=str,
                        help='浠庡凡鏈夋ā鍨?(.pth 鎴栫洰褰曞墠缂€) 缁х画璁粌锛屽鐢ㄥ凡瀛︾瓥鐣?)
    parser.add_argument('--resume-lr-scale', type=float, default=None,
                        help='Warm-start 鍚庣殑瀛︿範鐜囩缉鏀剧郴鏁?(榛樿0.5锛岃涓?鍙繚鐣欏師鍊?')
    
    # 馃啎 閫氫俊妯″瀷浼樺寲鍙傛暟锛?GPP鏍囧噯澧炲己锛?
    parser.add_argument('--comm-enhancements', action='store_true',
                        help='鍚敤鎵€鏈夐€氫俊妯″瀷浼樺寲锛堝揩琛拌惤+绯荤粺绾у共鎵?鍔ㄦ€佸甫瀹斤級Enable all communication model enhancements')
    parser.add_argument('--fast-fading', action='store_true',
                        help='鍚敤闅忔満蹇“钀斤紙Rayleigh/Rician锛塃nable fast fading')
    parser.add_argument('--system-interference', action='store_true',
                        help='鍚敤绯荤粺绾у共鎵拌绠?Enable system-level interference calculation')
    parser.add_argument('--dynamic-bandwidth', action='store_true',
                        help='鍚敤鍔ㄦ€佸甫瀹藉垎閰?Enable dynamic bandwidth allocation')
    # 馃啎 姝ｄ氦淇￠亾鍒嗛厤
    parser.add_argument('--channel-allocation', action='store_true',
                        help='鍚敤姝ｄ氦淇￠亾鍒嗛厤锛堝噺灏戝悓棰戝共鎵帮級Enable orthogonal channel allocation')
    
    args = parser.parse_args()

    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)
        _apply_global_seed_from_env()

    # 馃幆 涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛堥粯璁ゅ惎鐢級
    if args.central_resource:
        os.environ['CENTRAL_RESOURCE'] = '1'
        print("馃幆 鍚敤涓ぎ璧勬簮鍒嗛厤鏋舵瀯锛圥hase 1 + Phase 2锛塠榛樿妯″紡]")
    else:
        os.environ.pop('CENTRAL_RESOURCE', None)
        print("鈿狅笍  浣跨敤鏍囧噯鍧囧寑璧勬簮鍒嗛厤妯″紡锛堝凡閫氳繃 --no-central-resource 绂佺敤涓ぎ璧勬簮锛?)
    
    # 馃啎 閫氫俊妯″瀷浼樺寲閰嶇疆
    if args.comm_enhancements or args.fast_fading or args.system_interference or args.dynamic_bandwidth or args.channel_allocation:
        print("\n" + "="*70)
        print("馃寪 閫氫俊妯″瀷浼樺寲閰嶇疆锛?GPP鏍囧噯澧炲己锛?)
        print("="*70)
        
        # 濡傛灉鍚敤浜?-comm-enhancements锛屽垯鍚敤鎵€鏈変紭鍖?
        if args.comm_enhancements:
            config.communication.enable_fast_fading = True
            config.communication.use_system_interference = True
            config.communication.use_bandwidth_allocator = True
            config.communication.use_channel_allocation = True  # 馃啎 鍖呭惈淇￠亾鍒嗛厤
            config.communication.use_communication_enhancements = True
            print("鉁?鍚敤鎵€鏈夐€氫俊妯″瀷浼樺寲锛堝畬鏁?GPP鏍囧噯妯″紡锛?)
        else:
            # 鍗曠嫭閰嶇疆鍚勯」浼樺寲
            if args.fast_fading:
                config.communication.enable_fast_fading = True
                print("鉁?鍚敤闅忔満蹇“钀斤紙Rayleigh/Rician鍒嗗竷锛?)
            
            if args.system_interference:
                config.communication.use_system_interference = True
                print("鉁?鍚敤绯荤粺绾у共鎵拌绠?)
            
            if args.dynamic_bandwidth:
                config.communication.use_bandwidth_allocator = True
                print("鉁?鍚敤鍔ㄦ€佸甫瀹藉垎閰嶈皟搴﹀櫒")
            
            # 馃啎 姝ｄ氦淇￠亾鍒嗛厤
            if args.channel_allocation:
                config.communication.use_channel_allocation = True
                print("鉁?鍚敤姝ｄ氦淇￠亾鍒嗛厤锛堝噺灏戝悓棰戝共鎵帮級")
        
        # 鏄剧ず閰嶇疆璇︽儏
        print("\n閰嶇疆璇︽儏锛?)
        print(f"  - 蹇“钀? {'鍚敤' if config.communication.enable_fast_fading else '绂佺敤'}")
        print(f"  - 绯荤粺绾у共鎵? {'鍚敤' if config.communication.use_system_interference else '绂佺敤'}")
        print(f"  - 鍔ㄦ€佸甫瀹藉垎閰? {'鍚敤' if config.communication.use_bandwidth_allocator else '绂佺敤'}")
        print(f"  - 姝ｄ氦淇￠亾鍒嗛厤: {'鍚敤' if config.communication.use_channel_allocation else '绂佺敤'}")
        print(f"  - 杞芥尝棰戠巼: {config.communication.carrier_frequency/1e9:.1f} GHz")
        print(f"  - 缂栫爜鏁堢巼: {config.communication.coding_efficiency}")
        if config.communication.enable_fast_fading:
            print(f"  - 蹇“钀藉弬鏁? 蟽={config.communication.fast_fading_std}, K={config.communication.rician_k_factor}dB")
        if config.communication.use_channel_allocation:
            num_channels = int(config.communication.total_bandwidth / config.communication.channel_bandwidth)
            print(f"  - 鎬讳俊閬撴暟: {num_channels}涓?({config.communication.total_bandwidth/1e6:.0f}MHz / {config.communication.channel_bandwidth/1e6:.0f}MHz)")
        print("="*70 + "\n")
    
    # Toggle two-stage pipeline via environment for the simulator
    if args.two_stage:
        os.environ['TWO_STAGE_MODE'] = '1'
    # Stage1/Stage2 algorithm selections (env-based for env init)
    if args.stage1_alg:
        os.environ['STAGE1_ALG'] = args.stage1_alg
    if args.stage2_alg:
        # 鍏佽瑕嗙洊涓荤畻娉曢€夋嫨
        if not args.algorithm:
            args.algorithm = args.stage2_alg
        else:
            # 瑕嗗啓涓洪樁娈典簩閫夋嫨
            args.algorithm = args.stage2_alg

    # 馃敡 淇锛氭纭瀯寤簅verride_scenario鍙傛暟
    override_scenario = None
    if args.num_vehicles is not None:
        override_scenario = {
            "num_vehicles": args.num_vehicles,
        }
        # 鍚屾椂璁剧疆鐜鍙橀噺锛堝悜鍚庡吋瀹癸級
        os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(override_scenario)
        print(f"馃搵 瑕嗙洊鍙傛暟: 杞﹁締鏁?= {args.num_vehicles}")
    
    enforce_mode = None
    if getattr(args, 'force_offload', None):
        if args.force_offload in ('local', 'local_only'):
            enforce_mode = 'local_only'
        elif args.force_offload in ('remote', 'remote_only'):
            enforce_mode = 'remote_only'
    
    # 鍒涘缓缁撴灉鐩綍
    os.makedirs("results/single_agent", exist_ok=True)
    
    # 馃幆 鏄剧ずCAMTD3绯荤粺淇℃伅
    if args.algorithm and not args.compare:
        print("\n" + "="*80)
        print("馃殌 CAMTD3 璁粌绯荤粺鍚姩")
        print("="*80)
        print(f"绯荤粺鍚嶇О: CAMTD3 (Cache-Aware Migration with Twin Delayed DDPG)")
        print(f"浣跨敤绠楁硶: {args.algorithm}")
        print(f"绯荤粺鏋舵瀯: Phase 1 (涓ぎ璧勬簮鍒嗛厤) + Phase 2 (浠诲姟鎵ц)")
        print(f"璁粌杞暟: {args.episodes}")
        if args.seed:
            print(f"闅忔満绉嶅瓙: {args.seed}")
        print(f"瀹屾暣鍚嶇О: CAMTD3-{args.algorithm}")
        print("="*80 + "\n")
    
    if args.compare:
        # 姣旇緝鎵€鏈夌畻娉?
        algorithms = ['DDPG', 'TD3', 'TD3-LE', 'DQN', 'PPO', 'SAC']
        compare_single_algorithms(algorithms, args.episodes)
    elif args.algorithm:
        # 璁粌鍗曚釜绠楁硶 - 馃敡 浼犻€抩verride_scenario鍙傛暟
        train_single_algorithm(
            args.algorithm, 
            args.episodes, 
            args.eval_interval, 
            args.save_interval,
            enable_realtime_vis=args.realtime_vis,
            vis_port=args.vis_port,
            override_scenario=override_scenario,  # 馃敡 鏂板锛氫紶閫掕鐩栧弬鏁?
            use_enhanced_cache=not args.no_enhanced_cache,  # 馃殌 榛樿鍚敤澧炲己缂撳瓨
            enforce_offload_mode=enforce_mode,
            fixed_offload_policy=getattr(args, 'fixed_offload_policy', None),  # 馃幆 鍥哄畾鍗歌浇绛栫暐
            silent_mode=args.silent_mode,
            resume_from=args.resume_from,
            resume_lr_scale=args.resume_lr_scale,
            enable_advanced_vis=args.advanced_vis  # 馃帹 楂樼鍙鍖?
        )
    else:
        print("璇锋寚瀹?--algorithm 鎴栦娇鐢?--compare 鏍囧織")
        print("浣跨敤 python train_single_agent.py --help 鏌ョ湅甯姪")


if __name__ == "__main__":
    main()
    
"""

馃攧 瀹屾暣鎵ц娴佺▼锛堝垎5涓樁娈碉級
馃搶 闃舵1: 绯荤粺鍒濆鍖?(train_single_agent.py: main鍑芥暟)
1.1 鍙傛暟瑙ｆ瀽涓庨厤缃?
鈹溾攢 瑙ｆ瀽鍛戒护琛屽弬鏁?
鈹? 鈹溾攢 algorithm = "TD3"
鈹? 鈹溾攢 episodes = 800  
鈹? 鈹溾攢 num_vehicles = 12
鈹? 鈹斺攢 enhanced_cache = True (榛樿)
鈹?
鈹溾攢 璁剧疆闅忔満绉嶅瓙
鈹? 鈹斺攢 浠巆onfig鎴栫幆澧冨彉閲忚鍙栫瀛?
鈹?
鈹斺攢 鏋勫缓鍦烘櫙閰嶇疆 override_scenario
   鈹斺攢 {'num_vehicles': 12, 'override_topology': True}
   
1.2 鍒涘缓璁粌鐜 (SingleAgentTrainingEnvironment)
鐜鍒濆鍖栨祦绋?
鈹溾攢 1) 閫夋嫨浠跨湡鍣ㄧ被鍨?
鈹? 鈹溾攢 use_enhanced_cache=True
鈹? 鈹斺攢 simulator = EnhancedSystemSimulator(scenario_config)
鈹?
鈹溾攢 2) 鍒濆鍖栦豢鐪熷櫒缁勪欢 (system_simulator.py)
鈹? 鈹溾攢 杞﹁締鍒濆鍖? 12杈嗚溅
鈹? 鈹? 鈹溾攢 浣嶇疆: 闅忔満鍒嗗竷鍦ㄩ亾璺笂
鈹? 鈹? 鈹溾攢 閫熷害: 30-50 km/h
鈹? 鈹? 鈹斺攢 缂撳瓨: L1(200MB) + L2(300MB)
鈹? 鈹?
鈹? 鈹溾攢 RSU閮ㄧ讲: 4涓矾渚у崟鍏?(鍥哄畾鎷撴墤)
鈹? 鈹? 鈹溾攢 浣嶇疆: 绛夐棿璺濆垎甯?
鈹? 鈹? 鈹溾攢 瑕嗙洊鍗婂緞: 150m
鈹? 鈹? 鈹溾攢 缂撳瓨瀹归噺: 1000MB
鈹? 鈹? 鈹斺攢 璁＄畻鑳藉姏: 50 GHz
鈹? 鈹?
鈹? 鈹斺攢 UAV閮ㄧ讲: 2涓棤浜烘満
鈹?    鈹溾攢 浣嶇疆: 鍔ㄦ€佸贰鑸?
鈹?    鈹溾攢 楂樺害: 100m
鈹?    鈹溾攢 缂撳瓨瀹归噺: 200MB
鈹?    鈹斺攢 璁＄畻鑳藉姏: 20 GHz
鈹?
鈹溾攢 3) 鍒濆鍖栬嚜閫傚簲鎺у埗鍣?
鈹? 鈹溾攢 AdaptiveCacheController (鏅鸿兘缂撳瓨鎺у埗)
鈹? 鈹? 鈹溾攢 鍒嗗眰L1/L2缂撳瓨绛栫暐
鈹? 鈹? 鈹溾攢 鐑害杩借釜 (HeatBasedStrategy)
鈹? 鈹? 鈹斺攢 RSU鍗忎綔缂撳瓨
鈹? 鈹?
鈹? 鈹斺攢 AdaptiveMigrationController (杩佺Щ鍐崇瓥鎺у埗)
鈹?    鈹溾攢 璐熻浇鍘嗗彶杩借釜
鈹?    鈹溾攢 澶氱淮瑙﹀彂鏉′欢
鈹?    鈹斺攢 鎴愭湰鏁堢泭鍒嗘瀽
鈹?
鈹斺攢 4) 鎷撴墤浼樺寲 (FixedTopologyOptimizer)
   鈹溾攢 鏍规嵁杞﹁締鏁颁紭鍖栬秴鍙傛暟
   鈹溾攢 num_vehicles=12 鈫?hidden_dim=512
   鈹溾攢 actor_lr=1e-4, critic_lr=8e-5
   鈹斺攢 batch_size=256
   
1.3 鍒涘缓TD3鏅鸿兘浣?(TD3Environment)
TD3绠楁硶鍒濆鍖?
鈹溾攢 缃戠粶缁撴瀯
鈹? 鈹溾攢 Actor缃戠粶 (绛栫暐缃戠粶)
鈹? 鈹? 鈹溾攢 杈撳叆: state_dim = 杞﹁締(12脳5) + RSU(4脳5) + UAV(2脳5) + 鍏ㄥ眬(16) = 106缁?
鈹? 鈹? 鈹溾攢 闅愯棌灞? 512 鈫?512 鈫?256
鈹? 鈹? 鈹斺攢 杈撳嚭: action_dim = 3(浠诲姟鍒嗛厤) + 4(RSU閫夋嫨) + 2(UAV閫夋嫨) + 8(鎺у埗鍙傛暟) = 17缁?
鈹? 鈹?
鈹? 鈹溾攢 Twin Critic缃戠粶 (浠峰€肩綉缁溍?)
鈹? 鈹? 鈹溾攢 Critic1: 璇勪及鐘舵€?鍔ㄤ綔浠峰€?
鈹? 鈹? 鈹溾攢 Critic2: 鍑忓皯杩囦及璁″亸宸?
鈹? 鈹? 鈹斺攢 杈撳叆: state(106缁? + action(17缁? 鈫?杈撳嚭: Q鍊?
鈹? 鈹?
鈹? 鈹斺攢 Target缃戠粶 (鐩爣缃戠粶)
鈹?    鈹溾攢 Target Actor: 鐢熸垚鐩爣鍔ㄤ綔
鈹?    鈹溾攢 Target Critic1 & Critic2: 璁＄畻鐩爣Q鍊?
鈹?    鈹斺攢 杞洿鏂板弬鏁? 蟿=0.005
鈹?
鈹溾攢 缁忛獙鍥炴斁缂撳啿鍖?
鈹? 鈹溾攢 瀹归噺: 100,000鏉＄粡楠?
鈹? 鈹溾攢 鎵规澶у皬: 256
鈹? 鈹斺攢 浼樺厛绾х粡楠屽洖鏀?(PER)
鈹?    鈹溾攢 伪=0.6 (浼樺厛绾ф寚鏁?
鈹?    鈹斺攢 尾=0.4鈫?.0 (閲嶈鎬ч噰鏍?
鈹?
鈹斺攢 TD3鐗规湁鏈哄埗
   鈹溾攢 绛栫暐寤惰繜鏇存柊: policy_delay=2 (姣?姝ユ洿鏂癆ctor)
   鈹溾攢 鐩爣绛栫暐骞虫粦: target_noise=0.05
   鈹溾攢 鎺㈢储鍣０: exploration_noise=0.2 (鎸囨暟琛板噺)
   鈹斺攢 姊害瑁佸壀: gradient_clip=0.7
   
馃搶 闃舵2: Episode寰幆 (璁粌800涓猠pisode)
2.1 Episode閲嶇疆
姣忎釜Episode寮€濮嬫椂:
鈹溾攢 1) 閲嶇疆浠跨湡鍣?(system_simulator.py: initialize_components)
鈹? 鈹溾攢 娓呯┖鎵€鏈夐槦鍒?
鈹? 鈹溾攢 閲嶇疆杞﹁締浣嶇疆鍜岄€熷害
鈹? 鈹溾攢 娓呯┖缂撳瓨鍐呭
鈹? 鈹溾攢 閲嶇疆缁熻鏁版嵁
鈹? 鈹斺攢 閲嶆柊鐢熸垚鍐呭搴?(1000涓唴瀹?
鈹?
鈹溾攢 2) 鏋勫缓鍒濆鐘舵€?
鈹? 鈹溾攢 杞﹁締鐘舵€?(12脳5缁?
鈹? 鈹? 鈹溾攢 浣嶇疆(x,y): 褰掍竴鍖栧埌[0,1]
鈹? 鈹? 鈹溾攢 閫熷害: 褰掍竴鍖栧埌[0,1]
鈹? 鈹? 鈹溾攢 浠诲姟闃熷垪闀垮害: 褰掍竴鍖?
鈹? 鈹? 鈹斺攢 鑳借€? 褰掍竴鍖?
鈹? 鈹?
鈹? 鈹溾攢 RSU鐘舵€?(4脳5缁?
鈹? 鈹? 鈹溾攢 浣嶇疆(x,y)
鈹? 鈹? 鈹溾攢 缂撳瓨鍒╃敤鐜?
鈹? 鈹? 鈹溾攢 闃熷垪璐熻浇
鈹? 鈹? 鈹斺攢 鑳借€?
鈹? 鈹?
鈹? 鈹溾攢 UAV鐘舵€?(2脳5缁?
鈹? 鈹? 鈹溾攢 浣嶇疆(x,y,z)
鈹? 鈹? 鈹溾攢 缂撳瓨鍒╃敤鐜?
鈹? 鈹? 鈹斺攢 鑳借€?
鈹? 鈹?
鈹? 鈹斺攢 鍏ㄥ眬鐘舵€?(16缁?
鈹?    鈹溾攢 骞冲潎闃熷垪闀垮害
鈹?    鈹溾攢 骞冲潎缂撳瓨鍒╃敤鐜?
鈹?    鈹溾攢 绯荤粺璐熻浇
鈹?    鈹溾攢 浠诲姟绫诲瀷鍒嗗竷 (4缁?
鈹?    鈹溾攢 浠诲姟绫诲瀷闃熷垪鍗犳瘮 (4缁?
鈹?    鈹斺攢 浠诲姟绫诲瀷鎴鏈?(4缁?
鈹?
鈹斺攢 3) 閲嶇疆鎺у埗鍣ㄧ姸鎬?
   鈹溾攢 缂撳瓨鎺у埗鍣? 娓呯┖鐑害杩借釜
   鈹斺攢 杩佺Щ鎺у埗鍣? 娓呯┖璐熻浇鍘嗗彶

2.2 鏃堕棿姝ュ惊鐜?(姣忎釜Episode绾?00-300姝?
姣忎釜鏃堕棿姝ョ殑鎵ц娴佺▼:

鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 姝ラ1: TD3閫夋嫨鍔ㄤ綔 (td3.py: select_action)        鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 杈撳叆: state (106缁村悜閲?                            鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 鍓嶅悜浼犳挱閫氳繃Actor缃戠粶                          鈹?
鈹? 鈹? 鈹斺攢 杈撳嚭鍘熷鍔ㄤ綔: action_raw (17缁?             鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 娣诲姞鎺㈢储鍣０ (楂樻柉鍣０)                        鈹?
鈹? 鈹? 鈹溾攢 noise = N(0, exploration_noise)              鈹?
鈹? 鈹? 鈹斺攢 action = action_raw + noise                  鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 鍔ㄤ綔瑁佸壀鍒癧-1, 1]                              鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹斺攢 鍔ㄤ綔鍒嗚В (decompose_action)                    鈹?
鈹?    鈹溾攢 浠诲姟鍒嗛厤鍋忓ソ [0:3]                          鈹?
鈹?    鈹? 鈹斺攢 softmax([local, rsu, uav])               鈹?
鈹?    鈹溾攢 RSU閫夋嫨鏉冮噸 [3:7]                           鈹?
鈹?    鈹? 鈹斺攢 softmax(4涓猂SU鐨勬潈閲?                    鈹?
鈹?    鈹溾攢 UAV閫夋嫨鏉冮噸 [7:9]                           鈹?
鈹?    鈹? 鈹斺攢 softmax(2涓猆AV鐨勬潈閲?                    鈹?
鈹?    鈹斺攢 鎺у埗鍙傛暟 [9:17]                             鈹?
鈹?       鈹溾攢 缂撳瓨鎺у埗 (4缁?                           鈹?
鈹?       鈹? 鈹溾攢 鐑害闃堝€艰皟鏁?                         鈹?
鈹?       鈹? 鈹溾攢 娣樻卑绛栫暐鏉冮噸                          鈹?
鈹?       鈹? 鈹溾攢 鍗忎綔寮哄害                              鈹?
鈹?       鈹? 鈹斺攢 L1/L2姣斾緥                             鈹?
鈹?       鈹斺攢 杩佺Щ鎺у埗 (4缁?                           鈹?
鈹?          鈹溾攢 璐熻浇闃堝€?                             鈹?
鈹?          鈹溾攢 鎴愭湰鏁忔劅搴?                           鈹?
鈹?          鈹溾攢 寤惰繜鏉冮噸                              鈹?
鈹?          鈹斺攢 鑳借€楁潈閲?                             鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 姝ラ2: 鏄犲皠鍔ㄤ綔鍒拌嚜閫傚簲鎺у埗鍣?                    鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? (train_single_agent.py: _build_simulator_actions)  鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 瑙ｆ瀽鎺у埗鍙傛暟 (鍚?缁村姩浣?                       鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 璋冪敤 map_agent_actions_to_params()             鈹?
鈹? 鈹? 鈹溾攢 灏哰-1,1]鑼冨洿鏄犲皠鍒板叿浣撳弬鏁拌寖鍥?            鈹?
鈹? 鈹? 鈹斺攢 鍒嗙缂撳瓨鍙傛暟鍜岃縼绉诲弬鏁?                    鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 鏇存柊 AdaptiveCacheController                   鈹?
鈹? 鈹? 鈹溾攢 heat_threshold = action[0] * 50 + 50        鈹?
鈹? 鈹? 鈹溾攢 eviction_strategy_weight = sigmoid(action[1])鈹?
鈹? 鈹? 鈹溾攢 collaboration_strength = action[2] * 0.5 + 0.5鈹?
鈹? 鈹? 鈹斺攢 l1_l2_ratio = action[3] * 0.3 + 0.4         鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹斺攢 鏇存柊 AdaptiveMigrationController               鈹?
鈹?    鈹溾攢 load_threshold = action[4] * 0.3 + 0.6      鈹?
鈹?    鈹溾攢 cost_sensitivity = action[5] * 0.5 + 0.5    鈹?
鈹?    鈹溾攢 delay_weight = action[6] * 0.4 + 0.4        鈹?
鈹?    鈹斺攢 energy_weight = action[7] * 0.4 + 0.4       鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 姝ラ3: 浠跨湡鍣ㄦ墽琛屼竴姝?                             鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? (system_simulator.py: run_simulation_step)         鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.1 鏇存柊杞﹁締浣嶇疆                               鈹?
鈹? 鈹? 鈹溾攢 鏍规嵁閫熷害鍜屾柟鍚戠Щ鍔?                        鈹?
鈹? 鈹? 鈹溾攢 澶勭悊璺彛杞悜                               鈹?
鈹? 鈹? 鈹斺攢 娣诲姞闅忔満鎵板姩                               鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.2 鐢熸垚浠诲姟                                   鈹?
鈹? 鈹? 鈹溾攢 娉婃澗杩囩▼閲囨牱 (位=杞﹁締鏁懊椾换鍔＄巼)            鈹?
鈹? 鈹? 鈹溾攢 涓烘瘡杈嗚溅鐢熸垚浠诲姟                           鈹?
鈹? 鈹? 鈹? 鈹溾攢 浠诲姟绫诲瀷 (1-4): 鏍规嵁鍦烘櫙鍒嗗竷           鈹?
鈹? 鈹? 鈹? 鈹溾攢 鏁版嵁澶у皬: 0.5-2.0 MB                    鈹?
鈹? 鈹? 鈹? 鈹溾攢 璁＄畻闇€姹? 500-3000 CPU鍛ㄦ湡              鈹?
鈹? 鈹? 鈹? 鈹斺攢 鎴鏈? 0.5-3.0绉?                      鈹?
鈹? 鈹? 鈹斺攢 娣诲姞鍒拌溅杈嗕换鍔￠槦鍒?                        鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.3 浠诲姟鍒嗛厤涓庤皟搴?                            鈹?
鈹? 鈹? 鈹溾攢 瀵规瘡涓换鍔″喅绛栧嵏杞界洰鏍?                    鈹?
鈹? 鈹? 鈹? 鈹溾攢 鏈湴澶勭悊 (姒傜巼: local_pref)             鈹?
鈹? 鈹? 鈹? 鈹溾攢 RSU鍗歌浇 (姒傜巼: rsu_pref)                鈹?
鈹? 鈹? 鈹? 鈹? 鈹斺攢 鏍规嵁RSU閫夋嫨鏉冮噸閫夋嫨鍏蜂綋RSU          鈹?
鈹? 鈹? 鈹? 鈹斺攢 UAV鍗歌浇 (姒傜巼: uav_pref)                鈹?
鈹? 鈹? 鈹?    鈹斺攢 鏍规嵁UAV閫夋嫨鏉冮噸閫夋嫨鍏蜂綋UAV          鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 缂撳瓨鍛戒腑妫€鏌?                              鈹?
鈹? 鈹? 鈹? 鈹斺攢 check_cache_hit_adaptive()              鈹?
鈹? 鈹? 鈹?    鈹溾攢 妫€鏌ュ唴瀹规槸鍚﹀湪鑺傜偣缂撳瓨涓?           鈹?
鈹? 鈹? 鈹?    鈹溾攢 鍛戒腑: 鍑忓皯浼犺緭鏃跺欢                  鈹?
鈹? 鈹? 鈹?    鈹斺攢 鏈懡涓? 鏅鸿兘缂撳瓨鍐崇瓥                鈹?
鈹? 鈹? 鈹?       鈹溾攢 璋冪敤缂撳瓨鎺у埗鍣?should_cache_content鈹?
鈹? 鈹? 鈹?       鈹溾攢 鍩轰簬鐑害鍐冲畾鏄惁缂撳瓨             鈹?
鈹? 鈹? 鈹?       鈹斺攢 鎵ц娣樻卑鍜屽崗浣滅紦瀛?              鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹斺攢 浠诲姟浼犺緭涓庡叆闃?                            鈹?
鈹? 鈹?    鈹溾攢 璁＄畻涓婅浼犺緭鏃跺欢鍜岃兘鑰?                鈹?
鈹? 鈹?    鈹溾攢 灏嗕换鍔″姞鍏ヨ妭鐐硅绠楅槦鍒?                鈹?
鈹? 鈹?    鈹斺攢 璁板綍浠诲姟鍏冩暟鎹?                        鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.4 澶勭悊璁＄畻闃熷垪                               鈹?
鈹? 鈹? 鈹斺攢 _process_node_queues()                      鈹?
鈹? 鈹?    鈹溾攢 閬嶅巻鎵€鏈塕SU鍜孶AV                        鈹?
鈹? 鈹?    鈹溾攢 瀵规瘡涓妭鐐?                             鈹?
鈹? 鈹?    鈹? 鈹溾攢 鑾峰彇闃熷垪闀垮害                         鈹?
鈹? 鈹?    鈹? 鈹溾攢 鍔ㄦ€佽皟鏁村鐞嗚兘鍔?                    鈹?
鈹? 鈹?    鈹? 鈹? 鈹斺攢 capacity = base + boost(闃熷垪闀垮害) 鈹?
鈹? 鈹?    鈹? 鈹溾攢 澶勭悊浠诲姟宸ヤ綔閲?                      鈹?
鈹? 鈹?    鈹? 鈹? 鈹斺攢 work_remaining -= capacity        鈹?
鈹? 鈹?    鈹? 鈹溾攢 瀹屾垚鐨勪换鍔?                          鈹?
鈹? 鈹?    鈹? 鈹? 鈹溾攢 璁＄畻涓嬭浼犺緭                      鈹?
鈹? 鈹?    鈹? 鈹? 鈹溾攢 鏇存柊缁熻(寤惰繜銆佽兘鑰?             鈹?
鈹? 鈹?    鈹? 鈹? 鈹斺攢 鏍囪瀹屾垚                          鈹?
鈹? 鈹?    鈹? 鈹斺攢 澶勭悊瓒呮湡浠诲姟                         鈹?
鈹? 鈹?    鈹斺攢 鏇存柊鑺傜偣鐘舵€?                            鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.5 鑷€傚簲杩佺Щ妫€鏌?                            鈹?
鈹? 鈹? 鈹斺攢 check_adaptive_migration()                  鈹?
鈹? 鈹?    鈹溾攢 璁＄畻鎵€鏈夎妭鐐硅礋杞藉洜瀛?                   鈹?
鈹? 鈹?    鈹? 鈹斺攢 load = 0.8脳闃熷垪璐熻浇 + 0.2脳缂撳瓨鍒╃敤鐜団攤
鈹? 鈹?    鈹溾攢 鏇存柊杩佺Щ鎺у埗鍣ㄨ礋杞藉巻鍙?                 鈹?
鈹? 鈹?    鈹溾攢 鍒ゆ柇鏄惁瑙﹀彂杩佺Щ                        鈹?
鈹? 鈹?    鈹? 鈹溾攢 璐熻浇瓒呴槇鍊?                          鈹?
鈹? 鈹?    鈹? 鈹溾攢 鎸佺画鏃堕棿瓒冲                         鈹?
鈹? 鈹?    鈹? 鈹斺攢 鎴愭湰鏁堢泭鍒嗘瀽閫氳繃                     鈹?
鈹? 鈹?    鈹斺攢 鎵ц杩佺Щ                                鈹?
鈹? 鈹?       鈹溾攢 RSU鈫扲SU (鏈夌嚎杩佺Щ)                  鈹?
鈹? 鈹?       鈹? 鈹溾攢 閫夋嫨鐩爣RSU (璐熻浇鏈€杞?           鈹?
鈹? 鈹?       鈹? 鈹溾攢 璁＄畻杩佺Щ鎴愭湰                      鈹?
鈹? 鈹?       鈹? 鈹溾攢 浼犺緭浠诲姟                          鈹?
鈹? 鈹?       鈹? 鈹斺攢 鏇存柊缁熻                          鈹?
鈹? 鈹?       鈹斺攢 UAV鈫扲SU (鏃犵嚎杩佺Щ)                  鈹?
鈹? 鈹?          鈹斺攢 绫讳技娴佺▼                          鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 3.6 鏇存柊缁熻鎸囨爣                               鈹?
鈹? 鈹? 鈹溾攢 绱瀹屾垚浠诲姟鏁?                            鈹?
鈹? 鈹? 鈹溾攢 绱寤惰繜                                   鈹?
鈹? 鈹? 鈹溾攢 绱鑳借€?                                  鈹?
鈹? 鈹? 鈹溾攢 缂撳瓨鍛戒腑鐜?                                鈹?
鈹? 鈹? 鈹溾攢 杩佺Щ鎴愬姛鐜?                                鈹?
鈹? 鈹? 鈹斺攢 浠诲姟绫诲瀷鍒嗗竷缁熻                           鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹斺攢 杩斿洖 step_stats (鏈缁熻鏁版嵁)                鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 姝ラ4: 璁＄畻濂栧姳鍜屼笅涓€鐘舵€?                        鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? (train_single_agent.py: step 鏂规硶)                 鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 4.1 鎻愬彇绯荤粺鎸囨爣                               鈹?
鈹? 鈹? 鈹溾攢 骞冲潎寤惰繜: avg_delay (绉?                   鈹?
鈹? 鈹? 鈹溾攢 鎬昏兘鑰? total_energy (鐒﹁€?                鈹?
鈹? 鈹? 鈹溾攢 浠诲姟瀹屾垚鐜? completion_rate                鈹?
鈹? 鈹? 鈹溾攢 缂撳瓨鍛戒腑鐜? cache_hit_rate                 鈹?
鈹? 鈹? 鈹溾攢 鏁版嵁涓㈠け鐜? data_loss_ratio                鈹?
鈹? 鈹? 鈹斺攢 杩佺Щ鎴愬姛鐜? migration_success_rate         鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 4.2 璋冪敤缁熶竴濂栧姳璁＄畻鍣?                        鈹?
鈹? 鈹? 鈹斺攢 unified_reward_calculator.calculate_reward()鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹溾攢 寤惰繜鎯╃綒: -伪 脳 log(avg_delay + 蔚)       鈹?
鈹? 鈹?    鈹? 鈹斺攢 伪=15.0, 寮鸿皟浣庡欢杩?                 鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹溾攢 鑳借€楁儵缃? -尾 脳 log(total_energy + 蔚)    鈹?
鈹? 鈹?    鈹? 鈹斺攢 尾=0.01, 骞宠　鑳芥晥                    鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹溾攢 瀹屾垚鐜囧鍔? +纬 脳 completion_rate        鈹?
鈹? 鈹?    鈹? 鈹斺攢 纬=200.0, 榧撳姳浠诲姟瀹屾垚               鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹溾攢 缂撳瓨鍛戒腑濂栧姳: +未 脳 cache_hit_rate       鈹?
鈹? 鈹?    鈹? 鈹斺攢 未=10.0, 榧撳姳楂樺懡涓巼                鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹溾攢 鏁版嵁涓㈠け鎯╃綒: -蔚 脳 data_loss_ratio      鈹?
鈹? 鈹?    鈹? 鈹斺攢 蔚=50.0, 閬垮厤涓㈠寘                    鈹?
鈹? 鈹?    鈹?                                           鈹?
鈹? 鈹?    鈹斺攢 杩佺Щ鎴愬姛濂栧姳: +味 脳 migration_success    鈹?
鈹? 鈹?       鈹斺攢 味=5.0, 榧撳姳鏈夋晥杩佺Щ                 鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹?    鏈€缁堝鍔?= 危(鍚勯」濂栧姳/鎯╃綒)                鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 4.3 鏋勫缓涓嬩竴鐘舵€佸悜閲?(106缁?                   鈹?
鈹? 鈹? 鈹斺攢 涓庡垵濮嬬姸鎬佺浉鍚岀殑缁撴瀯                       鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹斺攢 4.4 鍒ゆ柇Episode鏄惁缁撴潫                        鈹?
鈹?    鈹溾攢 杈惧埌鏈€澶ф鏁?(200-300姝?                   鈹?
鈹?    鈹溾攢 绯荤粺宕╂簝 (鎵€鏈夎妭鐐硅繃杞?                     鈹?
鈹?    鈹斺攢 瀹屾垚鐜囪繃浣?(<20%)                          鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

鈹屸攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 姝ラ5: TD3瀛︿範鏇存柊 (td3.py: update)               鈹?
鈹溾攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 5.1 瀛樺偍缁忛獙鍒板洖鏀剧紦鍐插尯                       鈹?
鈹? 鈹? 鈹斺攢 buffer.add(state, action, reward, next_state, done)鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 5.2 閲囨牱鎵规鏁版嵁 (batch_size=256)              鈹?
鈹? 鈹? 鈹斺攢 浣跨敤PER浼樺厛绾ч噰鏍?                         鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 5.3 璁＄畻Critic鎹熷け                             鈹?
鈹? 鈹? 鈹溾攢 鐢熸垚鐩爣鍔ㄤ綔 (Target Actor)                鈹?
鈹? 鈹? 鈹? 鈹斺攢 target_action = target_actor(next_state) 鈹?
鈹? 鈹? 鈹?    + clipped_noise  # 鐩爣绛栫暐骞虫粦        鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 璁＄畻鐩爣Q鍊?(Twin Target Critics)          鈹?
鈹? 鈹? 鈹? 鈹溾攢 q1_target = target_critic1(next_state, target_action)鈹?
鈹? 鈹? 鈹? 鈹溾攢 q2_target = target_critic2(next_state, target_action)鈹?
鈹? 鈹? 鈹? 鈹斺攢 target_q = min(q1, q2)  # 鍑忓皯杩囦及璁?   鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 璁＄畻TD鐩爣                                 鈹?
鈹? 鈹? 鈹? 鈹斺攢 y = reward + 纬 脳 (1-done) 脳 target_q   鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 璁＄畻褰撳墠Q鍊?                               鈹?
鈹? 鈹? 鈹? 鈹溾攢 current_q1 = critic1(state, action)     鈹?
鈹? 鈹? 鈹? 鈹斺攢 current_q2 = critic2(state, action)     鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 Critic鎹熷け                                 鈹?
鈹? 鈹? 鈹? 鈹斺攢 loss = MSE(current_q1, y) + MSE(current_q2, y)鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹斺攢 鍙嶅悜浼犳挱鏇存柊Critic                         鈹?
鈹? 鈹?    鈹溾攢 critic_optimizer.zero_grad()             鈹?
鈹? 鈹?    鈹溾攢 loss.backward()                          鈹?
鈹? 鈹?    鈹溾攢 姊害瑁佸壀 (norm=0.7)                     鈹?
鈹? 鈹?    鈹斺攢 critic_optimizer.step()                  鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹溾攢 5.4 寤惰繜Actor鏇存柊 (姣弍olicy_delay=2姝?        鈹?
鈹? 鈹? 鈹溾攢 璁＄畻Actor鎹熷け                              鈹?
鈹? 鈹? 鈹? 鈹溾攢 new_action = actor(state)                鈹?
鈹? 鈹? 鈹? 鈹斺攢 actor_loss = -critic1(state, new_action).mean()鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹溾攢 鍙嶅悜浼犳挱鏇存柊Actor                          鈹?
鈹? 鈹? 鈹? 鈹溾攢 actor_optimizer.zero_grad()              鈹?
鈹? 鈹? 鈹? 鈹溾攢 actor_loss.backward()                    鈹?
鈹? 鈹? 鈹? 鈹溾攢 姊害瑁佸壀                                鈹?
鈹? 鈹? 鈹? 鈹斺攢 actor_optimizer.step()                   鈹?
鈹? 鈹? 鈹?                                             鈹?
鈹? 鈹? 鈹斺攢 杞洿鏂扮洰鏍囩綉缁?                            鈹?
鈹? 鈹?    鈹溾攢 target_actor = 蟿脳actor + (1-蟿)脳target_actor鈹?
鈹? 鈹?    鈹斺攢 target_critics = 蟿脳critics + (1-蟿)脳target_critics鈹?
鈹? 鈹?                                                  鈹?
鈹? 鈹斺攢 5.5 鏇存柊PER浼樺厛绾?                            鈹?
鈹?    鈹斺攢 鏍规嵁TD璇樊鏇存柊鏍锋湰浼樺厛绾?                  鈹?
鈹斺攢鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹?

馃搶 闃舵3: Episode缁撴潫涓庣粺璁?
Episode缁撴潫鍚?
鈹溾攢 璁板綍Episode缁熻
鈹? 鈹溾攢 鎬诲鍔?
鈹? 鈹溾攢 骞冲潎寤惰繜
鈹? 鈹溾攢 鎬昏兘鑰?
鈹? 鈹溾攢 瀹屾垚鐜?
鈹? 鈹溾攢 缂撳瓨鍛戒腑鐜?
鈹? 鈹斺攢 杩佺Щ缁熻
鈹?
鈹溾攢 琛板噺鎺㈢储鍣０
鈹? 鈹斺攢 exploration_noise *= noise_decay (0.9997)
鈹?
鈹斺攢 鎵撳嵃杩涘害淇℃伅
   鈹斺攢 姣?0涓狤pisode鎵撳嵃涓€娆¤缁嗙粺璁?

馃搶 闃舵4: 鍛ㄦ湡鎬ц瘎浼?(姣廵val_interval=50涓猠pisode)
璇勪及娴佺▼:
鈹溾攢 鍏抽棴鎺㈢储鍣０
鈹溾攢 杩愯10涓祴璇旹pisode
鈹溾攢 璁＄畻骞冲潎鎬ц兘鎸囨爣
鈹? 鈹溾攢 骞冲潎濂栧姳
鈹? 鈹溾攢 骞冲潎寤惰繜
鈹? 鈹溾攢 骞冲潎鑳借€?
鈹? 鈹斺攢 骞冲潎瀹屾垚鐜?
鈹斺攢 淇濆瓨鎬ц兘鏇茬嚎

馃搶 闃舵5: 璁粌缁撴潫涓庝繚瀛?(800涓猠pisode瀹屾垚鍚?
淇濆瓨缁撴灉:
鈹溾攢 1) 妯″瀷鏉冮噸
鈹? 鈹斺攢 results/models/single_agent/td3/
鈹?    鈹溾攢 actor_final.pth
鈹?    鈹溾攢 critic1_final.pth
鈹?    鈹溾攢 critic2_final.pth
鈹?    鈹斺攢 target_networks_final.pth
鈹?
鈹溾攢 2) 璁粌鏁版嵁
鈹? 鈹斺攢 results/single_agent/td3/training_results_YYYYMMDD_HHMMSS.json
鈹?    鈹溾攢 rewards: [...]
鈹?    鈹溾攢 delays: [...]
鈹?    鈹溾攢 energies: [...]
鈹?    鈹溾攢 completion_rates: [...]
鈹?    鈹斺攢 cache_metrics: {...}
鈹?
鈹斺攢 3) 鍙鍖栧浘琛?
   鈹斺攢 results/single_agent/td3/training_chart_YYYYMMDD_HHMMSS.png
      鈹溾攢 濂栧姳鏇茬嚎
      鈹溾攢 寤惰繜鏇茬嚎
      鈹溾攢 鑳借€楁洸绾?
      鈹斺攢 瀹屾垚鐜囨洸绾?
      
馃攽 鏍稿績鎶€鏈寒鐐?
1. Twin Delayed DDPG (TD3)
    鍙孋ritic缃戠粶鍑忓皯Q鍊艰繃浼拌
    寤惰繜绛栫暐鏇存柊鎻愰珮绋冲畾鎬?
    鐩爣绛栫暐骞虫粦鍖栧噺灏戞柟宸?
2. 鑷€傚簲鎺у埗鏈哄埗
    鏅鸿兘缂撳瓨鎺у埗锛氱儹搴﹁拷韪?+ 鍒嗗眰缂撳瓨
    鏅鸿兘杩佺Щ鎺у埗锛氬缁磋Е鍙?+ 鎴愭湰鏁堢泭
3. 缁熶竴濂栧姳鍑芥暟
    澶氱洰鏍囦紭鍖栵細寤惰繜銆佽兘鑰椼€佸畬鎴愮巼
    瀵规暟鎯╃綒锛氶伩鍏嶆瀬绔€煎奖鍝?
    骞宠　鏉冮噸锛氱‘淇濆悇椤规寚鏍囧崗璋?
4. 鍔ㄦ€佺綉缁滄嫇鎵?
    杞﹁締绉诲姩妯″瀷锛氱湡瀹為亾璺満鏅?
    鍥哄畾RSU/UAV锛氶獙璇佺畻娉曟湁鏁堟€?
    鑷€傚簲璁＄畻璧勬簮鍒嗛厤

"""
