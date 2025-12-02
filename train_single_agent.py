"""
🎯 CAMTD3训练脚本（Cache-Aware Migration with Twin Delayed DDPG）

【系统架构】
CAMTD3 = 基于中央资源分配的缓存感知任务迁移系统
├── Phase 1: 中央智能体资源分配决策（核心创新）
│   ├── 状态空间: 80维（车辆+RSU+UAV全局状态）
│   ├── 动作空间: 30维（带宽+计算资源分配向量）
│   └── 算法: TD3/SAC/DDPG/PPO
├── Phase 2: 本地任务执行
│   ├── 缓存决策（Cache-Aware）
│   ├── 任务迁移（Migration）
│   └── 任务调度
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 1000 --num-vehicles 12 --seed 42

Queue-aware Replay
•训练效率提升35%
•快速学习高负载场景
•针对VEC队列管理痛点
GNN Attention
•缓存命中率提升20%
•智能学习节点协作关系
•适应动态拓扑变化

【使用方法】
# CAMTD3标准训练（默认模式）
python train_single_agent.py --algorithm TD3 --episodes 200
python train_single_agent.py --algorithm SAC --episodes 200

🐍🖥️📚

单智能体算法训练脚本
支持DDPG、TD3、TD3-LE、DQN、PPO、SAC等算法的训练和比较
python train_single_agent.py --compare --episodes 200  # 比较所有算法
🚀 增强缓存模式 (默认启用 - 分层L1/L2 + 自适应热度策略 + RSU协作):
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 8
python train_single_agent.py --algorithm TD3 --episodes 1000 --num-vehicles 12
python train_single_agent.py --algorithm TD3 --episodes 800 --num-vehicles 12 --silent-mode  # 静默保存结果
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 16
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 20
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 24
python train_single_agent.py --algorithm TD3-LE --episodes 1600 --num-vehicles 12
python train_single_agent.py --algorithm SAC --episodes 800
python train_single_agent.py --algorithm PPO --episodes 800

🌐 实时可视化:
python train_single_agent.py --algorithm DDPG --episodes 100 --realtime-vis --vis-port 8080

🐍 生成学术图表:
python generate_academic_charts.py results/single_agent/td3/training_results_20251007_220900.json

到达率对比：python experiments/arrival_rate_analysis/run_td3_arrival_rate_sweep_silent.py --rates 1.0 1.5 2.0 2.5 3.0 3.5 --episodes 800


""" 
import os
import sys
import random

# 🔧 修复Windows编码问题
if sys.platform == 'win32':
    try:
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        elif hasattr(sys.stdout, 'buffer'):
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)
    except Exception:
        pass
    try:
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
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

# 导入核心模块
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
# 🤖 导入自适应控制组件
from utils.adaptive_control import AdaptiveCacheController, AdaptiveMigrationController, map_agent_actions_to_params
from decision.strategy_coordinator import StrategyCoordinator
from utils.unified_reward_calculator import update_reward_targets, _general_reward_calculator

# 导入各种单智能体算法
from single_agent.ddpg import DDPGEnvironment
from single_agent.td3 import TD3Environment
from single_agent.td3_hybrid_fusion import CAMTD3Environment
from single_agent.td3_latency_energy import TD3LatencyEnergyEnvironment
from single_agent.dqn import DQNEnvironment
from single_agent.ppo import PPOEnvironment
from single_agent.sac import SACEnvironment
from single_agent.optimized_td3_wrapper import OptimizedTD3Environment

# 导入HTML报告生成器
from utils.html_report_generator import HTMLReportGenerator

# 🌐 导入实时可视化模块
# try:
#     from scripts.visualize.realtime_visualization import create_visualizer
#     REALTIME_AVAILABLE = True
# except ImportError:
#     try:
#         from scripts.visualize.realtime_visualization_simple import create_visualizer
#         REALTIME_AVAILABLE = True
#     except ImportError:
#         REALTIME_AVAILABLE = False
#     print("⚠️  实时可视化功能不可用，请运行: pip install flask flask-socketio")
REALTIME_AVAILABLE = False

# 尝试导入PyTorch以设置随机种子；如果不可用则跳过
try:
    import torch
except ImportError:  # pragma: no cover - 容错处理
    torch = None


def _apply_global_seed_from_env():
    """根据环境变量RANDOM_SEED设置随机种子，确保可重复性"""
    seed_env = os.environ.get('RANDOM_SEED')
    if not seed_env:
        return
    try:
        seed = int(seed_env)
    except ValueError:
        print(f"⚠️  RANDOM_SEED 环境变量无效: {seed_env}")
        return

    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - GPU可选
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    config.random_seed = seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🔐 全局随机种子已设置为 {seed}")


def _maybe_apply_reward_smoothing_from_env():
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

def _build_scenario_config() -> Dict[str, Any]:
    """构建模拟环境配置，允许通过环境变量覆盖默认值"""
    # 🔧 支持从环境变量覆盖任务到达率（用于参数敏感性分析）
    task_arrival_rate = getattr(getattr(config, "task", None), "arrival_rate", 1.8)
    if os.environ.get('TASK_ARRIVAL_RATE'):
        try:
            task_arrival_rate = float(os.environ.get('TASK_ARRIVAL_RATE'))
            print(f"🔧 从环境变量覆盖任务到达率: {task_arrival_rate} tasks/s")
        except ValueError:
            print(f"⚠️  环境变量TASK_ARRIVAL_RATE无效，使用默认值")

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
        if bw < 1e3:  # assume MHz → Hz
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
                print("⚠️  TRAINING_SCENARIO_OVERRIDES 需为JSON对象，已忽略。")
        except json.JSONDecodeError as exc:
            print(f"⚠️  TRAINING_SCENARIO_OVERRIDES 解析失败: {exc}")

    return scenario


_apply_global_seed_from_env()
_maybe_apply_reward_smoothing_from_env()


def generate_timestamp() -> str:
    """生成时间戳"""
    if config.experiment.use_timestamp:
        return datetime.now().strftime(config.experiment.timestamp_format)
    else:
        return ""

def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """获取带时间戳的文件名"""
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
    """单智能体训练环境基类"""
    
    def _apply_optimized_td3_defaults(self) -> None:
        """
        【功能】若当前算法为OPTIMIZED_TD3，则放宽奖励权重与目标，降低训练振荡。
        【说明】仅在未通过环境变量覆盖时生效，确保兼容论文统一奖励。
        """
        if not hasattr(self, 'algorithm') and hasattr(self, 'input_algorithm'):
            alg = str(self.input_algorithm).upper()
        else:
            alg = getattr(self, 'algorithm', '').upper()
        if alg != "OPTIMIZED_TD3":
            return
        rl = getattr(config, "rl", None)
        if rl is None:
            return

        overridden_keys = []

        def _set_if_absent(env_key: str, attr: str, value: float, use_max: bool = False) -> None:
            if os.environ.get(env_key) is not None:
                return
            current = float(getattr(rl, attr, 0.0) or 0.0)
            if use_max:
                if value > current:
                    setattr(rl, attr, value)
                    overridden_keys.append(f"{attr}={value} (max)")
            elif current == 0.0:
                setattr(rl, attr, value)
                overridden_keys.append(f"{attr}={value} (default)")

        def _force_override(env_key: str, attr: str, value: float) -> None:
            if os.environ.get(env_key) is not None:
                return
            setattr(rl, attr, float(value))
            overridden_keys.append(f"{attr}={value}")

        # 🚫 禁用所有覆盖，使用system_config.py中的优化权重
        # _force_override("RL_USE_DYNAMIC_REWARD_NORMALIZATION", "use_dynamic_reward_normalization", 0.0)
        # _force_override("RL_WEIGHT_LOSS_RATIO", "reward_weight_loss_ratio", 1.0)
        # _force_override("RL_WEIGHT_CACHE", "reward_weight_cache", 0.35)
        # _force_override("RL_WEIGHT_CACHE_BONUS", "reward_weight_cache_bonus", 0.8)
        # _force_override("RL_WEIGHT_CACHE_PRESSURE", "reward_weight_cache_pressure", 0.8)
        # _force_override("RL_WEIGHT_OFFLOAD_BONUS", "reward_weight_offload_bonus", 0.8)
        # _force_override("RL_WEIGHT_COMPLETION_GAP", "reward_weight_completion_gap", 0.95)
        # _force_override("RL_PENALTY_DROPPED", "reward_penalty_dropped", 0.35)
        # _force_override("RL_WEIGHT_QUEUE_OVERLOAD", "reward_weight_queue_overload", 1.2)
        # _force_override("RL_WEIGHT_REMOTE_REJECT", "reward_weight_remote_reject", 0.45)
        # _force_override("RL_LATENCY_TARGET", "latency_target", 2.5)
        # _force_override("RL_LATENCY_UPPER_TOL", "latency_upper_tolerance", 5.0)
        # _force_override("RL_ENERGY_TARGET", "energy_target", 20000.0)
        # _force_override("RL_ENERGY_UPPER_TOL", "energy_upper_tolerance", 35000.0)
        # _force_override("RL_SMOOTH_DELAY", "reward_smooth_delay_weight", 0.35)
        # _force_override("RL_SMOOTH_ENERGY", "reward_smooth_energy_weight", 0.45)
        # _force_override("RL_SMOOTH_ALPHA", "reward_smooth_alpha", 0.12)

        # 🚫 禁用这些覆盖，使用system_config.py中的优化值
        # _set_if_absent("RL_WEIGHT_COMPLETION_GAP", "reward_weight_completion_gap", 0.7)
        # _set_if_absent("RL_PENALTY_DROPPED", "reward_penalty_dropped", 0.15, use_max=True)
        # _set_if_absent("RL_WEIGHT_QUEUE_OVERLOAD", "reward_weight_queue_overload", 0.8, use_max=True)
        # _set_if_absent("RL_WEIGHT_REMOTE_REJECT", "reward_weight_remote_reject", 0.25, use_max=True)

        if overridden_keys:
            print(f"\n⚡ OPTIMIZED_TD3 Configuration Overrides:")
            for k in overridden_keys:
                print(f"   - {k}")
            print("")

        # ✅ 启用update_reward_targets，使用system_config.py中的优化目标值
        # 确保全局单例计算器使用正确的归一化目标
        # 🔧 2024-12-02 激进简化：降低归一化目标，增强核心信号
        try:
            update_reward_targets(
                latency_target=float(getattr(rl, "latency_target", 1.5)),
                energy_target=float(getattr(rl, "energy_target", 200.0)),
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
        simulation_only: bool = False,
    ):
        self.input_algorithm = algorithm
        self.simulation_only = simulation_only
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
            "OPTIMIZEDTD3": "OPTIMIZED_TD3",
            "OPTIMIZED-TD3": "OPTIMIZED_TD3",
        }
        alias_key = normalized_algorithm.replace('_', '')
        self.algorithm = alias_map.get(normalized_algorithm, alias_map.get(alias_key, normalized_algorithm))
        self._apply_optimized_td3_defaults()
        scenario_config = _build_scenario_config()
        # 应用外部覆盖
        central_env_value = os.environ.get('CENTRAL_RESOURCE', '')
        self.central_resource_enabled = central_env_value.strip() in {'1', 'true', 'True'}
        self.joint_controller = bool(joint_controller)
        if self.joint_controller and not self.central_resource_enabled:
            os.environ['CENTRAL_RESOURCE'] = '1'
            self.central_resource_enabled = True

        if override_scenario:
            scenario_config.update(override_scenario)
            scenario_config['override_topology'] = True
            
            # 🔧 关键修复：动态修改全局config以支持参数覆盖
            # 原因：Node类使用全局config而非scenario_config
            network_cfg = getattr(config, "network", None)

            def _sync_topology(attr_name: str, component_attr: str, dict_key: str, value: int) -> None:
                setattr(config, attr_name, value)
                if network_cfg is not None:
                    setattr(network_cfg, attr_name, value)
                    component_cfg = getattr(network_cfg, component_attr, None)
                    if isinstance(component_cfg, dict):
                        component_cfg[dict_key] = value
            
            # 拓扑数量参数
            if 'num_vehicles' in override_scenario:
                num_vehicles_override = int(override_scenario['num_vehicles'])
                _sync_topology('num_vehicles', 'vehicle_config', 'num_vehicles', num_vehicles_override)
                print(f"🔧 [Override] 动态设置车辆数量: {num_vehicles_override}")
            if 'num_rsus' in override_scenario:
                num_rsus_override = int(override_scenario['num_rsus'])
                _sync_topology('num_rsus', 'rsu_config', 'num_rsus', num_rsus_override)
                print(f"🔧 [Override] 动态设置RSU数量: {num_rsus_override}")
            if 'num_uavs' in override_scenario:
                num_uav_override = int(override_scenario['num_uavs'])
                _sync_topology('num_uavs', 'uav_config', 'num_uavs', num_uav_override)
                print(f"🔧 [Override] 动态设置UAV数量: {num_uav_override}")

            # 带宽参数
            if 'bandwidth' in override_scenario or 'total_bandwidth' in override_scenario:
                bw_value = override_scenario.get('total_bandwidth') or override_scenario.get('bandwidth')
                if bw_value:
                    config.communication.total_bandwidth = float(bw_value)
                    network_comm_cfg = getattr(network_cfg, "communication_config", None)
                    if isinstance(network_comm_cfg, dict):
                        network_comm_cfg['bandwidth'] = float(bw_value)
                    print(f"🔧 [Override] 动态设置带宽: {float(bw_value)/1e6:.1f} MHz")
            
            # 🎯 总资源池参数（优先级高于单节点频率）
            if 'total_vehicle_compute' in override_scenario:
                total_compute = float(override_scenario['total_vehicle_compute'])
                config.compute.total_vehicle_compute = total_compute
                # 自动计算每车平均频率
                avg_freq = total_compute / config.num_vehicles
                config.compute.vehicle_initial_freq = avg_freq
                config.compute.vehicle_default_freq = avg_freq
                config.compute.vehicle_cpu_freq = avg_freq
                config.compute.vehicle_cpu_freq_range = (avg_freq, avg_freq)
                print(f"🔧 [Override] 动态设置总本地计算: {total_compute/1e9:.1f} GHz (每车{avg_freq/1e9:.3f} GHz)")
            
            if 'total_rsu_compute' in override_scenario:
                total_compute = float(override_scenario['total_rsu_compute'])
                config.compute.total_rsu_compute = total_compute
                avg_freq = total_compute / config.num_rsus
                config.compute.rsu_initial_freq = avg_freq
                config.compute.rsu_default_freq = avg_freq
                config.compute.rsu_cpu_freq = avg_freq
                config.compute.rsu_cpu_freq_range = (avg_freq, avg_freq)
                print(f"🔧 [Override] 动态设置总RSU计算: {total_compute/1e9:.1f} GHz (每RSU{avg_freq/1e9:.1f} GHz)")
            
            if 'total_uav_compute' in override_scenario:
                total_compute = float(override_scenario['total_uav_compute'])
                config.compute.total_uav_compute = total_compute
                avg_freq = total_compute / config.num_uavs
                config.compute.uav_initial_freq = avg_freq
                config.compute.uav_default_freq = avg_freq
                config.compute.uav_cpu_freq = avg_freq
                config.compute.uav_cpu_freq_range = (avg_freq, avg_freq)
                print(f"🔧 [Override] 动态设置总UAV计算: {total_compute/1e9:.1f} GHz (每UAV{avg_freq/1e9:.1f} GHz)")
            
            # CPU频率参数（单节点频率，兼容旧代码）
            if 'vehicle_cpu_freq' in override_scenario and 'total_vehicle_compute' not in override_scenario:
                freq_value = override_scenario['vehicle_cpu_freq']
                # 更新范围和默认值
                config.compute.vehicle_cpu_freq_range = (freq_value, freq_value)
                config.compute.vehicle_cpu_freq = freq_value
                print(f"🔧 [Override] 动态设置车辆CPU频率: {float(freq_value)/1e9:.2f} GHz")
            
            if 'rsu_cpu_freq' in override_scenario and 'total_rsu_compute' not in override_scenario:
                freq_value = override_scenario['rsu_cpu_freq']
                config.compute.rsu_cpu_freq_range = (freq_value, freq_value)
                config.compute.rsu_cpu_freq = freq_value
                print(f"🔧 [Override] 动态设置RSU CPU频率: {float(freq_value)/1e9:.2f} GHz")
            
            if 'uav_cpu_freq' in override_scenario and 'total_uav_compute' not in override_scenario:
                freq_value = override_scenario['uav_cpu_freq']
                config.compute.uav_cpu_freq_range = (freq_value, freq_value)
                config.compute.uav_cpu_freq = freq_value
                print(f"🔧 [Override] 动态设置UAV CPU频率: {float(freq_value)/1e9:.2f} GHz")
            
            # 任务数据大小参数
            if 'task_data_size_min_kb' in override_scenario or 'task_data_size_max_kb' in override_scenario:
                min_kb = override_scenario.get('task_data_size_min_kb')
                max_kb = override_scenario.get('task_data_size_max_kb')
                if min_kb is not None and max_kb is not None:
                    # 转换为字节
                    min_bytes = float(min_kb) * 1024
                    max_bytes = float(max_kb) * 1024
                    config.task.data_size_range = (min_bytes, max_bytes)
                    config.task.task_data_size_range = (min_bytes, max_bytes)
                    print(f"🔧 [Override] 动态设置任务数据大小: {min_kb}-{max_kb} KB")
            
            # 任务复杂度参数
            if 'task_complexity_multiplier' in override_scenario:
                multiplier = override_scenario['task_complexity_multiplier']
                # 通过环境变量传递给TaskConfig
                os.environ['TASK_COMPLEXITY_MULTIPLIER'] = str(multiplier)
                print(f"🔧 [Override] 动态设置任务复杂度倍数: {multiplier}x")
            
            if 'task_compute_density' in override_scenario:
                density = override_scenario['task_compute_density']
                config.task.task_compute_density = float(density)
                print(f"🔧 [Override] 动态设置任务计算密度: {density} cycles/bit")
            
            # 缓存容量参数
            if 'cache_capacity' in override_scenario:
                capacity_mb = override_scenario['cache_capacity']
                # 通过环境变量传递（影响所有节点）
                os.environ['CACHE_CAPACITY_MB'] = str(capacity_mb)
                print(f"🔧 [Override] 动态设置缓存容量: {capacity_mb} MB")

            # 服务能力参数
            if 'rsu_base_service' in override_scenario:
                value = int(override_scenario['rsu_base_service'])
                config.service.rsu_base_service = value
                print(f"🔧 [Override] 动态设置RSU基础服务能力: {value}")
            if 'rsu_max_service' in override_scenario:
                value = int(override_scenario['rsu_max_service'])
                config.service.rsu_max_service = value
                print(f"🔧 [Override] 动态设置RSU最大服务能力: {value}")
            if 'rsu_work_capacity' in override_scenario:
                value = float(override_scenario['rsu_work_capacity'])
                config.service.rsu_work_capacity = value
                print(f"🔧 [Override] 动态设置RSU工作容量: {value}")
            if 'uav_base_service' in override_scenario:
                value = int(override_scenario['uav_base_service'])
                config.service.uav_base_service = value
                print(f"🔧 [Override] 动态设置UAV基础服务能力: {value}")
            if 'uav_max_service' in override_scenario:
                value = int(override_scenario['uav_max_service'])
                config.service.uav_max_service = value
                print(f"🔧 [Override] 动态设置UAV最大服务能力: {value}")
            if 'uav_work_capacity' in override_scenario:
                value = float(override_scenario['uav_work_capacity'])
                config.service.uav_work_capacity = value
                print(f"🔧 [Override] 动态设置UAV工作容量: {value}")
            
            # 任务到达率参数
            if 'task_arrival_rate' in override_scenario:
                arrival_rate = override_scenario['task_arrival_rate']
                config.task.arrival_rate = float(arrival_rate)
                # 同时设置环境变量以兼容旧代码
                os.environ['TASK_ARRIVAL_RATE'] = str(arrival_rate)
                print(f"🔧 [Override] 动态设置任务到达率: {arrival_rate} tasks/s")
            
            # 单一任务数据大小参数（用于混合负载实验）
            if 'task_data_size_kb' in override_scenario:
                size_kb = override_scenario['task_data_size_kb']
                size_bytes = float(size_kb) * 1024
                config.task.data_size_range = (size_bytes, size_bytes)
                config.task.task_data_size_range = (size_bytes, size_bytes)
                print(f"🔧 [Override] 动态设置任务数据大小: {size_kb} KB")
            
            # 通信参数（噪声功率、路径损耗）
            if 'noise_power_dbm' in override_scenario:
                noise_power = override_scenario['noise_power_dbm']
                config.communication.noise_power_dbm = float(noise_power)
                print(f"🔧 [Override] 动态设置噪声功率: {noise_power} dBm")
            
            if 'path_loss_exponent' in override_scenario:
                exponent = override_scenario['path_loss_exponent']
                config.communication.path_loss_exponent = float(exponent)
                print(f"🔧 [Override] 动态设置路径损耗指数: {exponent}")
            
            # 资源异构性参数
            if 'heterogeneity_level' in override_scenario:
                hetero_level = override_scenario['heterogeneity_level']
                os.environ['HETEROGENEITY_LEVEL'] = str(hetero_level)
                print(f"🔧 [Override] 动态设置资源异构性级别: {hetero_level}")
        
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
            print(f"⚠️ 未识别的强制卸载模式: {forced_mode_input}, 将忽略。")
            requested_mode = ''
        self.enforce_offload_mode = requested_mode
        if self.enforce_offload_mode:
            scenario_config['forced_offload_mode'] = self.enforce_offload_mode
            if self.enforce_offload_mode == 'remote_only':
                scenario_config.setdefault('allow_local_processing', False)
            elif self.enforce_offload_mode == 'local_only':
                scenario_config.setdefault('allow_local_processing', True)

        if self.enforce_offload_mode == 'local_only':
            print("🧷 强制卸载模式: 全部本地处理（Local-Only）")
        elif self.enforce_offload_mode == 'remote_only':
            print("🧷 强制卸载模式: 全部远端执行（Remote-Only）")
        
        # 🎯 固定卸载策略初始化
        self.fixed_offload_policy = None
        self.fixed_policy_name = None
        if fixed_offload_policy:
            try:
                import sys
                import importlib.util
                from pathlib import Path
                
                # 动态添加 experiments 目录到 Python 路径
                exp_path = Path(__file__).parent / 'experiments'
                if str(exp_path) not in sys.path:
                    sys.path.insert(0, str(exp_path))
                
                # 使用 importlib 动态导入模块（避免静态分析警告）
                module_path = exp_path / 'fallback_baselines.py'
                if module_path.exists():
                    spec = importlib.util.spec_from_file_location("fallback_baselines", module_path)
                    if spec and spec.loader:
                        fallback_module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(fallback_module)
                        create_baseline_algorithm = fallback_module.create_baseline_algorithm
                    else:
                        raise ImportError(f"无法加载模块 {module_path}")
                else:
                    raise ImportError(f"模块文件不存在: {module_path}")
                
                self.fixed_offload_policy = create_baseline_algorithm(fixed_offload_policy)
                self.fixed_policy_name = fixed_offload_policy
                print(f"🎲 固定卸载策略: {fixed_offload_policy} (卸载决策不由智能体学习)")
                print(f"   其他决策（缓存、迁移、资源分配）仍由智能体学习")
            except Exception as e:
                print(f"⚠️  无法创建固定策略 '{fixed_offload_policy}': {e}")
                print(f"   将使用智能体学习卸载决策")
                self.fixed_offload_policy = None
        
        # 选择仿真器类型
        self.use_enhanced_cache = use_enhanced_cache and ENHANCED_CACHE_AVAILABLE
        env_disable_migration = os.environ.get("DISABLE_MIGRATION", "").strip() == "1"
        self.disable_migration = disable_migration or env_disable_migration
        if self.use_enhanced_cache:
            print("🚀 [Training] Using Enhanced Cache System (Default) with:")
            print("   - Hierarchical L1/L2 caching (3GB + 7GB)")
            print("   - Adaptive HeatBasedCacheStrategy")
            print("   - Inter-RSU collaboration")
            self.simulator = EnhancedSystemSimulator(scenario_config)
        else:
            self.simulator = CompleteSystemSimulator(scenario_config)
        
        # 🤖 初始化自适应控制组件
        self.adaptive_cache_controller = AdaptiveCacheController()
        self.adaptive_migration_controller = AdaptiveMigrationController()
        if self.disable_migration:
            print("🤖 自适应缓存已启用；迁移控制已禁用（DISABLE_MIGRATION 模式）")
        else:
            print(f"🤖 已启用自适应缓存和迁移控制功能")

        self.strategy_coordinator = StrategyCoordinator(
            self.adaptive_cache_controller,
            None if self.disable_migration else self.adaptive_migration_controller
        )
        self.strategy_coordinator.register_simulator(self.simulator)
        setattr(self.simulator, 'strategy_coordinator', self.strategy_coordinator)
        
        # 从仿真器获取实际网络拓扑参数
        num_vehicles = len(self.simulator.vehicles)
        num_rsus = len(self.simulator.rsus)
        num_uavs = len(self.simulator.uavs)
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        
        # 🎯 更新固定策略的环境信息
        if self.fixed_offload_policy is not None:
            try:
                # 创建一个简化的环境对象供固定策略使用
                class SimpleEnv:
                    def __init__(self, simulator):
                        self.simulator = simulator
                        self.agent_env = type('obj', (object,), {
                            'action_dim': 18,  # 默认action维度
                        })()
                
                simple_env = SimpleEnv(self.simulator)
                self.fixed_offload_policy.update_environment(simple_env)
                print(f"   固定策略已更新环境信息: {num_vehicles}车辆, {num_rsus}RSU, {num_uavs}UAV")
            except Exception as e:
                print(f"⚠️  固定策略更新环境失败: {e}")
        
        # 应用固定拓扑的参数优化（保持4 RSU + 2 UAV）
        if self.algorithm in {"TD3", "TD3_LATENCY_ENERGY"}:
            topology_optimizer = FixedTopologyOptimizer()
            opt_params = topology_optimizer.get_optimized_params(num_vehicles)
            
            # 应用优化的超参数到TD3配置
            os.environ['TD3_HIDDEN_DIM'] = str(opt_params.get('hidden_dim', 400))
            os.environ['TD3_ACTOR_LR'] = str(opt_params.get('actor_lr', 1e-4))
            os.environ['TD3_CRITIC_LR'] = str(opt_params.get('critic_lr', 8e-5))
            os.environ['TD3_BATCH_SIZE'] = str(opt_params.get('batch_size', 256))
            
            print(f"[FIXED-TOPOLOGY] 车辆数:{num_vehicles} → Hidden:{opt_params['hidden_dim']}, LR:{opt_params['actor_lr']:.1e}, Batch:{opt_params['batch_size']}")
            print(f"[FIXED-TOPOLOGY] 保持固定: RSU=4, UAV=2（验证算法策略有效性）")
        
        # 🔧 优化：所有算法统一传入拓扑参数，实现动态适配
        if self.algorithm == "DDPG":
            self.agent_env = DDPGEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "TD3":
            self.agent_env = TD3Environment(
                num_vehicles,
                num_rsus,
                num_uavs,
                use_central_resource=self.central_resource_enabled,
            )
        elif self.algorithm == "TD3_LATENCY_ENERGY":
            self.agent_env = TD3LatencyEnergyEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "CAM_TD3":
            self.agent_env = CAMTD3Environment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "DQN":
            self.agent_env = DQNEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "PPO":
            self.agent_env = PPOEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "SAC":
            self.agent_env = SACEnvironment(num_vehicles, num_rsus, num_uavs)
        elif self.algorithm == "OPTIMIZED_TD3":
            self.agent_env = OptimizedTD3Environment(
                num_vehicles,
                num_rsus,
                num_uavs,
                use_central_resource=self.central_resource_enabled,
                simulation_only=self.simulation_only
            )
            if not self.simulation_only:
                print(f"[OptimizedTD3] 使用精简优化配置 (Queue-aware Replay + GNN Attention)")
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

        # 🎯 中央资源分配模式日志
        import sys
        print(f"\n[资源分配模式检查]", file=sys.stderr)
        print(f"  CENTRAL_RESOURCE 环境变量: '{central_env_value}'", file=sys.stderr)
        print(f"  use_central_resource: {self.central_resource_enabled}", file=sys.stderr)
        
        self.central_resource_action_dim = getattr(self.agent_env, 'central_resource_action_dim', 0)
        self.central_resource_state_dim = getattr(self.agent_env, 'central_state_dim', 0)
        self.base_action_dim = getattr(self.agent_env, 'base_action_dim', getattr(self.agent_env, 'action_dim', 0) - self.central_resource_action_dim)
        
        if self.central_resource_enabled and self.central_resource_action_dim > 0:
            print(f"✅ 启用中央资源分配架构：Phase 1(决策) + Phase 2(执行)", file=sys.stderr)
            print(f"   环境类型: {type(self.agent_env).__name__}", file=sys.stderr)
            print(f"   基础动作维度: {self.base_action_dim}", file=sys.stderr)
            print(f"   中央资源动作维度: {self.central_resource_action_dim}", file=sys.stderr)
            if self.central_resource_state_dim:
                print(f"   状态扩展维度: +{self.central_resource_state_dim}", file=sys.stderr)
        else:
            print(f"  使用标准模式（均匀资源分配）", file=sys.stderr)
        
        # 🧠 若指定了阶段一算法（通过环境变量），用DualStage封装器组合两个阶段
        stage1_alg = os.environ.get('STAGE1_ALG', '').strip().lower()
        if stage1_alg:
            try:
                from single_agent.dual_stage_controller import DualStageControllerEnv
                self.agent_env = DualStageControllerEnv(self.agent_env, self.simulator, stage1_strategy=stage1_alg)
                print(f"🧠 启用两阶段控制：Stage1={stage1_alg} + Stage2={self.algorithm}")
                # Two-stage planner inside simulator becomes redundant
                os.environ['TWO_STAGE_MODE'] = '0'
            except Exception as e:
                print(f"⚠️ 两阶段控制封装失败，回退到单算法: {e}")
        
        # 训练统计
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
            'queue_overload_flag': [],
            'queue_overload_events': [],
            'episode_steps': [],  # 🔧 新增：记录每个episode的实际步数
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
            'rsu_hotspot_mean': [],
            'rsu_hotspot_peak': [],
            'rsu_hotspot_mean_series': [],
            'rsu_hotspot_peak_series': [],
            'mm1_queue_error': [],
            'mm1_delay_error': [],
            'normalized_delay': [],
            'normalized_energy': [],
            'normalized_reward': []
        }
        
        # 性能追踪器
        self.performance_tracker = {
            'recent_rewards': MovingAverage(100),
            'recent_step_rewards': MovingAverage(100),
            'recent_delays': MovingAverage(100),
            'recent_energy': MovingAverage(100),
            'recent_completion': MovingAverage(100)
        }
        self._reward_baseline: Dict[str, float] = {}
        self._energy_target_per_vehicle = float(os.environ.get('ENERGY_TARGET_PER_VEHICLE', '75.0'))  # 🔧 220 → 75 (使启发式目标 = 75×12 = 900J)
        self._dynamic_energy_target = float(getattr(config.rl, 'energy_target', 1200.0))
        heuristic_energy_target = max(
            self._dynamic_energy_target,
            self.num_vehicles * self._energy_target_per_vehicle
        )
        if heuristic_energy_target > self._dynamic_energy_target * 1.05:
            self._dynamic_energy_target = heuristic_energy_target
            update_reward_targets(energy_target=heuristic_energy_target)
            print(
                f"⚖️ 动态调整能耗目标: {heuristic_energy_target:.1f}J "
                f"(车辆数={self.num_vehicles}, 每车预算={self._energy_target_per_vehicle:.1f}J)"
            )
        self._energy_target_ema = self._dynamic_energy_target
        self._energy_target_warmup = max(40, int(config.experiment.num_episodes * 0.1))
        self._last_energy_target_update = 0
        self._reward_smoothing_alpha = float(getattr(config.rl, 'reward_smooth_alpha', 0.35))
        self._reward_ema_delay: Optional[float] = None
        self._reward_ema_energy: Optional[float] = None
        self._episode_counters_initialized = False
        
        print(f"✓ {self.algorithm}训练环境初始化完成")
        print(f"✓ 算法类型: 单智能体")
        

    
    def _calculate_correct_cache_utilization(self, cache: Dict, cache_capacity_mb: float) -> float:
        """
        🔧 修复：正确计算缓存利用率
        
        Args:
            cache: 缓存字典
            cache_capacity_mb: 缓存容量(MB)
        Returns:
            缓存利用率 [0.0, 1.0]
        """
        if not cache or cache_capacity_mb <= 0:
            return 0.0
        
        total_used_mb = 0.0
        for item in cache.values():
            if isinstance(item, dict) and 'size' in item:
                total_used_mb += float(item.get('size', 0.0))
            else:
                # 兼容旧格式，使用realistic大小
                total_used_mb += 1.0  # 默认1MB
        
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
        self._episode_counters_initialized = True

    def _reset_reward_baseline(self, stats: Optional[Dict[str, Any]] = None) -> None:
        """初始化/重置奖励增量基线。"""
        base = stats or {}
        self._reward_baseline = {
            'processed': int(base.get('processed_tasks', 0) or 0),
            'dropped': int(base.get('dropped_tasks', 0) or 0),
            'delay': float(base.get('total_delay', 0.0) or 0.0),
            'energy': float(base.get('total_energy', 0.0) or 0.0),
            'generated_bytes': float(base.get('generated_data_bytes', 0.0) or 0.0),
            'dropped_bytes': float(base.get('dropped_data_bytes', 0.0) or 0.0),
        }
        self._reward_ema_delay = None
        self._reward_ema_energy = None

    def _build_reward_snapshot(self, stats: Dict[str, Any]) -> Dict[str, float]:
        """基于累计统计计算单步奖励所需的增量指标。"""
        baseline = getattr(self, '_reward_baseline', None) or {
            'processed': 0,
            'dropped': 0,
            'delay': 0.0,
            'energy': 0.0,
            'generated_bytes': 0.0,
            'dropped_bytes': 0.0,
        }

        total_processed = int(stats.get('processed_tasks', 0) or 0)
        total_dropped = int(stats.get('dropped_tasks', 0) or 0)
        total_delay = float(stats.get('total_delay', 0.0) or 0.0)
        total_energy = float(stats.get('total_energy', 0.0) or 0.0)
        total_generated = float(stats.get('generated_data_bytes', 0.0) or 0.0)
        total_dropped_bytes = float(stats.get('dropped_data_bytes', 0.0) or 0.0)

        delta_processed = max(0, total_processed - baseline['processed'])
        delta_dropped = max(0, total_dropped - baseline['dropped'])
        delta_delay = max(0.0, total_delay - baseline['delay'])
        delta_energy = max(0.0, total_energy - baseline['energy'])
        
        # 🔧 修复：减去静态能耗，只奖励动态能耗
        # 静态功率 = RSU静态 * num_rsus + UAV静态 * num_uavs
        rsu_static = getattr(config.compute, 'rsu_static_power', 25.0)
        uav_static = getattr(config.compute, 'uav_static_power', 2.5)
        static_power = (self.num_rsus * rsu_static) + (self.num_uavs * uav_static)
        time_slot = getattr(config.experiment, 'time_slot', 0.1)
        static_energy_step = static_power * time_slot
        
        # 确保不减成负数
        dynamic_delta_energy = max(0.0, delta_energy - static_energy_step)
        
        delta_generated = max(0.0, total_generated - baseline['generated_bytes'])
        delta_loss_bytes = max(0.0, total_dropped_bytes - baseline['dropped_bytes'])

        if delta_processed > 0:
            avg_delay_for_reward = delta_delay / delta_processed
        else:
            avg_delay_for_reward = 0.0

        completion_total = delta_processed + delta_dropped
        completion_rate = normalize_ratio(delta_processed, completion_total, default=1.0)
        loss_ratio = normalize_ratio(delta_loss_bytes, delta_generated)
        # 🔧 修复：直接使用delta_energy，移除平滑和回退逻辑
        # 之前的回退导致在无任务处理的step使用了累积能耗（~900J），导致奖励崩塌
        reward_snapshot = {
            'avg_task_delay': avg_delay_for_reward,
            'total_energy_consumption': dynamic_delta_energy,
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
        """对奖励关键指标进行指数平滑，减小TD3训练噪声。"""
        if self._reward_smoothing_alpha <= 0.0:
            return delay_value, energy_per_task
        alpha = self._reward_smoothing_alpha
        if self._reward_ema_delay is None:
            self._reward_ema_delay = delay_value
        else:
            self._reward_ema_delay = (1.0 - alpha) * self._reward_ema_delay + alpha * delay_value
        if self._reward_ema_energy is None:
            self._reward_ema_energy = energy_per_task
        else:
            self._reward_ema_energy = (1.0 - alpha) * self._reward_ema_energy + alpha * energy_per_task
        return self._reward_ema_delay, self._reward_ema_energy

    def _maybe_update_dynamic_energy_target(self, episode: int, episode_energy: float) -> None:
        """根据实际能耗自动放宽目标，避免不可达约束导致振荡。"""
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
            new_target = min(ema * 0.95, target * 1.8)
            self._dynamic_energy_target = new_target
            self._last_energy_target_update = episode
            update_reward_targets(energy_target=new_target)
            print(
                f"⚙️ 能耗EMA {ema:.1f}J 超过目标 {target:.1f}J，"
                f"自动上调奖励阈值 -> {new_target:.1f}J (Episode {episode})"
            )

    def reset_environment(self) -> np.ndarray:
        """重置环境并返回初始状态"""
        # 重置仿真器状态
        self._episode_counters_initialized = False
        self.simulator._setup_scenario()
        
        # 收集系统状态
        node_states = {}
        
        # 车辆状态（与step保持一致的归一化方式）
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                normalize_scalar(vehicle['position'][0], 'vehicle_position_range', 1000.0),
                normalize_scalar(vehicle['position'][1], 'vehicle_position_range', 1000.0),
                normalize_scalar(vehicle.get('velocity', 0.0), 'vehicle_speed_range', 50.0),
                normalize_scalar(len(vehicle.get('tasks', [])), 'vehicle_queue_capacity', 20.0),
                normalize_scalar(vehicle.get('energy_consumed', 0.0), 'vehicle_energy_reference', 1000.0),
            ])
            node_states[f'vehicle_{i}'] = vehicle_state

        # RSU状态（统一归一化/裁剪）
        for i, rsu in enumerate(self.simulator.rsus):
            rsu_state = np.array([
                normalize_scalar(rsu['position'][0], 'rsu_position_range', 1000.0),
                normalize_scalar(rsu['position'][1], 'rsu_position_range', 1000.0),
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),
                normalize_scalar(len(rsu.get('computation_queue', [])), 'rsu_queue_capacity', 20.0),
                normalize_scalar(rsu.get('energy_consumed', 0.0), 'rsu_energy_reference', 1000.0),
            ])
            node_states[f'rsu_{i}'] = rsu_state

        # UAV状态（统一归一化/裁剪）
        for i, uav in enumerate(self.simulator.uavs):
            uav_state = np.array([
                normalize_scalar(uav['position'][0], 'uav_position_range', 1000.0),
                normalize_scalar(uav['position'][1], 'uav_position_range', 1000.0),
                normalize_scalar(uav['position'][2], 'uav_altitude_range', 200.0),
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),
                normalize_scalar(uav.get('energy_consumed', 0.0), 'uav_energy_reference', 1000.0),
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 初始系统指标
        system_metrics = {
            'avg_task_delay': 0.0,
            'total_energy_consumption': 0.0,
            'data_loss_bytes': 0.0,
            'data_loss_ratio_bytes': 0.0,
            'cache_hit_rate': 0.0,
            'migration_success_rate': 0.0
        }
        
        # 🔧 修复：重置能耗追踪器，避免跨episode累积
        if hasattr(self, '_last_total_energy'):
            delattr(self, '_last_total_energy')

        stats_snapshot = getattr(self.simulator, 'stats', None)
        self._initialize_episode_counters(stats_snapshot)
        self._reset_reward_baseline(stats_snapshot)
        
        resource_state = self._collect_resource_state()
        state = self.agent_env.get_state_vector(node_states, system_metrics, resource_state)
        
        return state

    def step(self, action, state: Optional[np.ndarray] = None, actions_dict: Optional[Dict] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步仿真，应用智能体动作到仿真器"""
        # 🎯 如果未提供actions_dict，尝试从action分解
        if actions_dict is None and hasattr(self.agent_env, 'decompose_action'):
            try:
                actions_dict = self.agent_env.decompose_action(action)
            except Exception:
                pass


        # 🎯 使用固定卸载策略（如果设置）
        if self.fixed_offload_policy is not None and actions_dict is not None:
            try:
                # 使用固定策略生成卸载决策
                fixed_action = self.fixed_offload_policy.select_action(state)
                
                # 将固定策略的action转换为offload preference
                # 固定策略返回的action格式: [local_score, rsu_score, uav_score, ...]
                if isinstance(fixed_action, np.ndarray) and len(fixed_action) >= 3:
                    local_pref = float(fixed_action[0])
                    rsu_pref = float(fixed_action[1])
                    uav_pref = float(fixed_action[2])
                    
                    # 归一化为概率分布
                    total = abs(local_pref) + abs(rsu_pref) + abs(uav_pref)
                    if total > 1e-6:
                        local_pref = abs(local_pref) / total
                        rsu_pref = abs(rsu_pref) / total
                        uav_pref = abs(uav_pref) / total
                    else:
                        local_pref, rsu_pref, uav_pref = 0.33, 0.33, 0.34
                    
                    # 覆盖智能体的卸载决策，保留其他决策（缓存、迁移等）
                    if 'offload_preference' in actions_dict:
                        actions_dict['offload_preference'] = {
                            'local': local_pref,
                            'rsu': rsu_pref,
                            'uav': uav_pref
                        }
            except Exception as e:
                # 如果固定策略失败，回退到智能体决策
                pass
        
        # 🔍 诊断日志：监控卸载决策分布
        if actions_dict is not None and 'offload_preference' in actions_dict:
            step_count = getattr(self, '_step_counter', 0)
            self._step_counter = step_count + 1
            
            if step_count % 50 == 0:
                offload_pref = actions_dict['offload_preference']
                local_val = offload_pref.get('local', 0.0)
                rsu_val = offload_pref.get('rsu', 0.0)
                uav_val = offload_pref.get('uav', 0.0)
                print(f"🔍 [Step {step_count}] 卸载偏好 → Local:{local_val:.3f}, RSU:{rsu_val:.3f}, UAV:{uav_val:.3f}")
        
        # 构造传递给仿真器的动作（将连续动作映射为本地/RSU/UAV偏好）
        sim_actions = self._build_simulator_actions(actions_dict)
        
        # 执行仿真步骤（传入动作）
        step_stats = self.simulator.run_simulation_step(0, sim_actions)
        
        # 🔧 实时可视化：发射任务事件
        if getattr(self, 'visualizer', None) is not None:
            step_events = step_stats.get('step_events', [])
            for event in step_events:
                try:
                    self.visualizer.emit_task_event(
                        event_type=event['type'],
                        vehicle_id=event['vehicle_id'],
                        target_id=event['target_id']
                    )
                except Exception:
                    pass
            
            # 🔧 实时可视化：更新车辆位置拓扑
            vehicle_positions = step_stats.get('vehicle_positions', [])
            if vehicle_positions:
                try:
                    self.visualizer.emit_topology_update(vehicle_positions)
                except Exception:
                    pass
        
        resource_state = self._collect_resource_state()

        
        # 收集下一步状态
        node_states = {}
        
        # 车辆状态 (5维 - 统一归一化)
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                normalize_scalar(vehicle['position'][0], 'vehicle_position_range', 1000.0),  # 位置x
                normalize_scalar(vehicle['position'][1], 'vehicle_position_range', 1000.0),  # 位置y
                normalize_scalar(vehicle.get('velocity', 0.0), 'vehicle_speed_range', 50.0),  # 速度
                normalize_scalar(len(vehicle.get('tasks', [])), 'vehicle_queue_capacity', 20.0),  # 队列
                normalize_scalar(vehicle.get('energy_consumed', 0.0), 'vehicle_energy_reference', 1000.0),  # 能耗
            ])
            node_states[f'vehicle_{i}'] = vehicle_state

        # RSU状态 (5维 - 清理版，移除控制参数)
        for i, rsu in enumerate(self.simulator.rsus):
            # 标准化归一化：确保所有值在[0,1]范围
            rsu_state = np.array([
                normalize_scalar(rsu['position'][0], 'rsu_position_range', 1000.0),  # 位置x
                normalize_scalar(rsu['position'][1], 'rsu_position_range', 1000.0),  # 位置y
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),  # 缓存利用率
                normalize_scalar(len(rsu.get('computation_queue', [])), 'rsu_queue_capacity', 20.0),  # 队列利用率
                normalize_scalar(rsu.get('energy_consumed', 0.0), 'rsu_energy_reference', 1000.0),  # 能耗
            ])
            node_states[f'rsu_{i}'] = rsu_state

        # UAV状态 (5维 - 清理版，移除控制参数)
        for i, uav in enumerate(self.simulator.uavs):
            # 标准化归一化：确保所有值在[0,1]范围
            uav_state = np.array([
                normalize_scalar(uav['position'][0], 'uav_position_range', 1000.0),  # 位置x
                normalize_scalar(uav['position'][1], 'uav_position_range', 1000.0),  # 位置y
                # 🔧 修复：使用队列利用率代替高度（高度对决策影响小，队列负载关键）
                normalize_scalar(len(uav.get('computation_queue', [])), 'uav_queue_capacity', 20.0),   # 队列利用率
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),  # 缓存利用率
                normalize_scalar(uav.get('energy_consumed', 0.0), 'uav_energy_reference', 1000.0),  # 能耗
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 计算系统指标
        system_metrics = self._calculate_system_metrics(step_stats)
        
        # 获取下一状态
        next_state = self.agent_env.get_state_vector(node_states, system_metrics, resource_state)
        
        # 🔧 增强：计算包含子系统指标的奖励
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
                print(f"⚠️ 联合策略协调器观测异常: {exc}")

        # 反馈关键系统指标给TD3策略指导模块，驱动能耗/延迟温度自适应
        agent_core = getattr(self.agent_env, 'agent', None)
        if agent_core is not None and hasattr(agent_core, 'update_guidance_feedback'):
            try:
                agent_core.update_guidance_feedback(system_metrics, cache_metrics, migration_metrics)
            except Exception as exc:
                if getattr(self, '_current_episode', 0) % 200 == 0:
                    print(f"⚠️ 指导反馈更新失败: {exc}")

        reward_source = system_metrics.get('reward_snapshot', system_metrics)
        reward, reward_components = self.agent_env.calculate_reward(reward_source, cache_metrics, migration_metrics)
        
        # 将奖励组件添加到step_stats供调试使用
        step_stats['reward_components'] = reward_components
        
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
        
        # 判断是否结束
        done = False  # 单智能体环境通常不会提前结束
        
        # 附加信息
        info = {
            'step_stats': step_stats,
            'system_metrics': system_metrics
        }
        
        return next_state, reward, done, info
    
    def _calculate_system_metrics(self, step_stats: Dict) -> Dict:
        """计算系统性能指标 - 最终修复版，确保数值在合理范围"""
        import numpy as np
        
        # 安全获取数值
        def safe_get(key: str, default: float = 0.0) -> float:
            value = step_stats.get(key, default)
            if np.isnan(value) or np.isinf(value):
                return default
            return max(0.0, value)  # 确保非负
        
        # 🔧 修复：使用episode级别统计而非累积统计，避免奖励累积恶化
        # 计算本episode的增量统计
        total_processed = int(safe_get('processed_tasks', 0))  # 累计完成
        total_dropped = int(safe_get('dropped_tasks', 0))  # 累计丢弃（数量）
        
        # 计算本episode增量
        episode_processed = total_processed - getattr(self, '_episode_processed_base', 0)
        episode_dropped = total_dropped - getattr(self, '_episode_dropped_base', 0)
        
        # 数据丢失量：使用本episode增量
        current_generated_bytes = float(step_stats.get('generated_data_bytes', 0.0))
        current_dropped_bytes = float(step_stats.get('dropped_data_bytes', 0.0))
        episode_generated_bytes = current_generated_bytes - getattr(self, '_episode_generated_bytes_base', 0.0)
        episode_dropped_bytes = current_dropped_bytes - getattr(self, '_episode_dropped_bytes_base', 0.0)
        
        # 计算本episode任务总数和完成率（避免累积效应）
        episode_total = episode_processed + episode_dropped
        completion_rate = normalize_ratio(episode_processed, episode_total, default=0.5)
        
        cache_hits = int(safe_get('cache_hits', 0))
        cache_misses = int(safe_get('cache_misses', 0))
        cache_requests_total = cache_hits + cache_misses
        reported_requests = int(step_stats.get('cache_requests', cache_requests_total) or cache_requests_total)
        reported_hit_rate = step_stats.get('cache_hit_rate')
        if reported_requests > 0:
            cache_requests_total = reported_requests
        if isinstance(reported_hit_rate, (int, float)):
            cache_hit_rate = float(np.clip(reported_hit_rate, 0.0, 1.0))
        else:
            cache_hit_rate = normalize_ratio(cache_hits, cache_requests_total)
        local_cache_hits = int(safe_get('local_cache_hits', 0))
        
        # 🔧 修复：安全计算平均延迟 - 使用累计统计
        total_delay = safe_get('total_delay', 0.0)
        processed_for_delay = max(1, total_processed)  # 使用累计完成数
        avg_delay = total_delay / processed_for_delay
        
        # 限制延迟在合理范围内（关键修复）
        avg_delay = np.clip(avg_delay, 0.01, 5.0)  # 扩大到0.01-5.0秒范围，适应跨时隙处理

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
        
        # 🔧 修复能耗计算：使用真实累积能耗并转换为本episode增量
        current_total_energy = safe_get('total_energy', 0.0)

        if not getattr(self, '_episode_counters_initialized', False):
            self._initialize_episode_counters(step_stats)

        # 自适应控制器统计（用于奖励与指标归一化）
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

                mm1_predictions[node_key] = {
                    'arrival_rate': arrival_rate,
                    'service_rate': service_rate,
                    'rho': rho_storable,
                    'stable': bool(stable),
                    'theoretical_queue': theo_queue_val,
                    'actual_queue': actual_queue,
                    'theoretical_delay': theo_delay_val,
                    'actual_delay': actual_delay_obs,
                }

                if theo_queue_val is not None:
                    mm1_queue_errors.append(abs(actual_queue - theo_queue_val))
                if theo_delay_val is not None:
                    mm1_delay_errors.append(abs(actual_delay_obs - theo_delay_val))

        mm1_queue_error = float(np.mean(mm1_queue_errors)) if mm1_queue_errors else 0.0
        mm1_delay_error = float(np.mean(mm1_delay_errors)) if mm1_delay_errors else 0.0

        
        # 计算本episode增量能耗（防止负值与异常）
        if current_total_energy <= 0.0:
            # 仿真器能耗异常时的保底估算
            completed_tasks = self.simulator.stats.get('completed_tasks', 0) if hasattr(self, 'simulator') else 0
            estimated_energy = max(0.0, completed_tasks * 15.0)
            total_energy = estimated_energy
            print(f"⚠️ 仿真器能耗为0，使用估算能耗: {total_energy:.1f}J")
        else:
            episode_incremental_energy = max(0.0, current_total_energy - getattr(self, '_episode_energy_base', 0.0))
            
            # 🔧 关键修复：移除静态能耗基线，只奖励动态能耗优化
            # 静态能耗 = (RSU静态功率 * RSU数量 + UAV静态功率 * UAV数量) * 持续时间
            # 这样可以让智能体专注于优化那 ~200J 的动态能耗，而不是被 ~2000J 的静态能耗淹没
            rsu_static = getattr(config.compute, 'rsu_static_power', 25.0)
            uav_static = getattr(config.compute, 'uav_static_power', 2.5)
            # 车辆静态能耗通常不计入系统运营成本（属于用户设备），但为了严谨也可以减去
            # 这里主要关注基础设施能耗
            static_power_total = (self.num_rsus * rsu_static) + (self.num_uavs * uav_static)
            
            # 计算当前episode已运行时间的静态能耗
            # 使用仿真器当前时间作为持续时间
            current_duration = self.simulator.current_time
            static_energy_baseline = static_power_total * current_duration
            
            # 动态能耗 = 总能耗 - 静态基线
            # 限制为非负，防止因浮点误差出现负值
            dynamic_energy = max(0.0, episode_incremental_energy - static_energy_baseline)
            
            # ⚠️ 仍然记录总能耗用于展示，但使用动态能耗用于奖励计算
            total_energy = dynamic_energy
            # print(f"DEBUG: Total={episode_incremental_energy:.1f}J, Static={static_energy_baseline:.1f}J, Dynamic={dynamic_energy:.1f}J")

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
        
        # 🔧 修复：使用episode级别数据丢失量，避免累积效应
        data_loss_bytes = max(0.0, episode_dropped_bytes)
        data_generated_bytes = max(0.0, episode_generated_bytes)
        data_loss_ratio_bytes = normalize_ratio(data_loss_bytes, data_generated_bytes)
        
        # 迁移成功率（来自仿真器统计）
        migrations_executed = int(safe_get('migrations_executed', 0))
        migrations_successful = int(safe_get('migrations_successful', 0))
        migration_success_rate = normalize_ratio(migrations_successful, migrations_executed)
        
        # 🔧 调试迁移统计
        if migrations_executed > 0:
            print(f"🔍 迁移统计: 执行{migrations_executed}次, 成功{migrations_successful}次, 成功率{migration_success_rate:.1%}")

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
                values = raw
            else:
                values = []
            return normalize_feature_vector(values, length, clip=clip)

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

        # 🔍 调试日志：能耗与迁移敏感区间
        current_episode = getattr(self, '_current_episode', 0)
        if current_episode > 0 and (current_episode % 50 == 0 or avg_delay > 0.2 or migration_success_rate < 0.9):
            print(
                f"[调试] Episode {current_episode:04d}: 延迟 {avg_delay:.3f}s, 能耗 {total_energy:.2f}J, "
                f"完成率 {completion_rate:.1%}, 迁移成功率 {migration_success_rate:.1%}, "
                f"缓存命中 {cache_hit_rate:.1%}, 数据损失 {data_loss_ratio_bytes:.1%}, "
                f"缓存淘汰率 {cache_eviction_rate:.1%}"
            )

        # 🤖 更新缓存控制器统计（如果有实际数据）
        if cache_hit_rate > 0:
            # 🔧 修复：正确计算缓存统计
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
            # 🤖 新增自适应控制指标
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
        }

    def _normalize_reward_value(self, reward: float) -> float:
        """将奖励值转换为无量纲比例，便于与其他指标对比。"""
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
        """将系统指标写入episode_metrics，方便后续报告/可视化使用。"""
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
            'migration_avg_cost': 'migration_avg_cost',
            'migration_avg_delay_saved': 'migration_avg_delay_saved',
            'rsu_hotspot_mean': 'rsu_hotspot_mean',
            'rsu_hotspot_peak': 'rsu_hotspot_peak',
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
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None, visualizer: Optional[Any] = None) -> Dict:
        """运行一个完整的训练轮次"""
        # 使用配置中的最大步数
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 重置环境
        self._episode_counters_initialized = False
        state = self.reset_environment()
        
        # 🔧 设置可视化器
        self.visualizer = visualizer
        
        # 🔧 保存当前episode编号
        self._current_episode = episode
        
        # 🔧 重置episode步数跟踪，修复能耗计算
        self._current_episode_step = 0
        
        episode_reward = 0.0
        episode_info = {}
        step = 0
        info = {}  # 初始化info变量
        
        # PPO需要特殊处理
        if self.algorithm == "PPO":
            return self._run_ppo_episode(episode, max_steps, visualizer)
        
        for step in range(max_steps):
            # 选择动作
            if self.algorithm == "DQN":
                # DQN返回离散动作
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 处理可能的元组返回
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                        
                # 需要将动作映射回全局动作索引
                action_idx = self._encode_discrete_action(actions_dict)
                action = action_idx
            else:
                # 连续动作算法
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    # 处理可能的元组返回
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = self._encode_continuous_action(actions_dict)
            
            # 🔧 更新episode步数计数器
            self._current_episode_step += 1
            
            # 将向量动作恢复为字典供模拟器消费（避免动作被忽略）
            sim_actions_dict = actions_dict if isinstance(actions_dict, dict) else self._build_actions_from_vector(action)
            
            # 执行动作（将动作字典传入以影响仿真器卸载偏好）
            next_state, reward, done, info = self.step(action, state, sim_actions_dict)
            
            # 🔧 修复1：更新队列指标（驱动Queue-aware Replay）
            if hasattr(self.agent_env, 'update_queue_metrics'):
                step_stats = info.get('step_stats', {})
                try:
                    self.agent_env.update_queue_metrics(step_stats)
                except Exception as e:
                    if self._current_episode % 100 == 0:  # 仅每100轮报告一次
                        print(f"⚠️ 队列指标更新失败: {e}")
            
            # 初始化training_info
            training_info = {}
            
            # 训练智能体 - 所有算法现在都支持Union类型统一接口
            # 确保action类型安全转换
            if self.algorithm == "DQN":
                # DQN首选整数动作，但接受Union类型
                safe_action = self._safe_int_conversion(action)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm in ["DDPG", "TD3", "TD3_LATENCY_ENERGY", "SAC", "OPTIMIZED_TD3"]:
                # 连续动作算法首选numpy数组，但接受Union类型
                safe_action = action if isinstance(action, np.ndarray) else np.array([action], dtype=np.float32)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm == "PPO":
                # PPO使用特殊的episode级别训练，train_step为占位符
                # 保持原action类型即可，因为PPO的train_step不做实际处理
                training_info = self.agent_env.train_step(state, action, reward, next_state, done)
            else:
                # 其他算法的默认处理
                training_info = {'message': f'Unknown algorithm: {self.algorithm}'}
            
            # 累积奖励并保存最新的训练信息
            episode_reward += reward
            episode_info = training_info

            # 更新状态；如未来引入提前结束，这里兼容 done 标志
            state = next_state
            if done:
                break
            
        # ?? ?????system_metrics?????????episode???
        steps_taken = step + 1  # range ? 0 ??
        system_metrics = info.get('system_metrics', {})
        self._record_episode_metrics(system_metrics, episode_steps=steps_taken)
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': steps_taken
        }
    
    def _run_ppo_episode(self, episode: int, max_steps: int = 100, visualizer: Optional[Any] = None) -> Dict:
        """运行PPO专用episode"""
        state = self.reset_environment()
        self.visualizer = visualizer
        episode_reward = 0.0
        
        # 初始化变量
        done = False
        step = 0
        info = {}
        
        for step in range(max_steps):
            # 获取动作、对数概率和价值
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, tuple) and len(actions_result) == 3:
                    actions_dict, log_prob, value = actions_result
                else:
                    # 如果不是元组，就使用默认值
                    actions_dict = actions_result if isinstance(actions_result, dict) else {}
                    log_prob = 0.0
                    value = 0.0
            else:
                actions_dict = {}
                log_prob = 0.0
                value = 0.0
                
            action = self._encode_continuous_action(actions_dict)
            
            # 执行动作
            next_state, reward, done, info = self.step(action, state, actions_dict)
            
            # 存储经验 - 所有算法都支持统一接口
            # 确保参数类型正确
            log_prob_float = float(log_prob) if not isinstance(log_prob, float) else log_prob
            value_float = float(value) if not isinstance(value, float) else value
            # 使用命名参数避免位置参数顺序问题
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
        
        # 🔧 PPO更新策略修复：累积多个episode后再更新
        last_value = 0.0
        if not done:
            if hasattr(self.agent_env, 'get_actions'):
                actions_result = self.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, tuple) and len(actions_result) >= 3:
                    _, _, last_value = actions_result
                else:
                    last_value = 0.0
        
        # 确保 last_value 为 float 类型
        last_value_float = float(last_value) if not isinstance(last_value, float) else last_value
        
        # 检查是否应该更新（每N个episode或buffer快满时）
        ppo_config = self.agent_env.config
        should_update = (
            episode % ppo_config.update_frequency == 0 or  # 每N个episode
            self.agent_env.agent.buffer.size >= ppo_config.buffer_size * 0.9  # buffer接近满
        )
        
        # 进行更新
        # PPOEnvironment.update只接受last_value参数，force_update在agent内部处理
        if should_update:
            training_info = self.agent_env.agent.update(last_value_float, force_update=True)
        else:
            training_info = self.agent_env.agent.update(last_value_float, force_update=False)
        
        steps_taken = step + 1  # range ? 0 ??
        system_metrics = info.get('system_metrics', {})
        self._record_episode_metrics(system_metrics, episode_steps=steps_taken)
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': training_info,
            'system_metrics': system_metrics,
            'steps': steps_taken
        }

    def _build_simulator_actions(self, actions_dict: Optional[Dict]) -> Optional[Dict]:
        """将算法动作字典转换为仿真器可消费的简单控制信号。
        🤖 扩展支持联合动作空间：
        - vehicle_agent 前3维 → 原有任务分配偏好
        - 中间 num_rsus/num_uavs 维 → 节点选择权重
        - 末尾10维 → 缓存、迁移及联动控制参数
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
            
            # =============== 原有任务分配逻辑 (保持兼容) ===============
            raw = vehicle_action_array[:3]
            # 🔧 修复：将[-1,1]范围的动作值放大到[-5,5]，使softmax更敏感
            # Actor输出是[-1,1]，需要放大才能产生明显的偏好差异
            raw = np.clip(raw, -1.0, 1.0) * 5.0  # 放大5倍：[-1,1] -> [-5,5]
            raw = np.clip(raw, -5.0, 5.0)  # 确保在[-5,5]范围内
            exp = np.exp(raw - np.max(raw))
            probs = exp / np.sum(exp)
            sim_actions = {
                'vehicle_offload_pref': {
                    'local': float(probs[0]),
                    'rsu': float(probs[1] if probs.size > 1 else 0.33),
                    'uav': float(probs[2] if probs.size > 2 else 0.34)
                }
            }
            # RSU选择概率
            num_rsus = self.num_rsus
            rsu_action = actions_dict.get('rsu_agent')
            if isinstance(rsu_action, (list, tuple, np.ndarray)) and num_rsus > 0:
                rsu_raw = np.array(rsu_action[:num_rsus], dtype=np.float32)
            else:
                rsu_raw = vehicle_action_array[3:3 + num_rsus]
            if num_rsus > 0:
                # 🔧 修复：同样放大RSU选择权重
                rsu_raw = np.clip(rsu_raw, -1.0, 1.0) * 5.0  # 放大5倍
                rsu_raw = np.clip(rsu_raw, -5.0, 5.0)
                rsu_exp = np.exp(rsu_raw - np.max(rsu_raw))
                rsu_probs = rsu_exp / np.sum(rsu_exp)
                sim_actions['rsu_selection_probs'] = [float(x) for x in rsu_probs]
            
            # UAV选择概率
            num_uavs = self.num_uavs
            uav_action = actions_dict.get('uav_agent')
            if isinstance(uav_action, (list, tuple, np.ndarray)) and num_uavs > 0:
                uav_raw = np.array(uav_action[:num_uavs], dtype=np.float32)
            else:
                uav_raw = vehicle_action_array[3 + num_rsus:3 + num_rsus + num_uavs]
            if num_uavs > 0:
                # 🔧 修复：同样放大UAV选择权重
                uav_raw = np.clip(uav_raw, -1.0, 1.0) * 5.0  # 放大5倍
                uav_raw = np.clip(uav_raw, -5.0, 5.0)
                uav_exp = np.exp(uav_raw - np.max(uav_raw))
                uav_probs = uav_exp / np.sum(uav_exp)
                sim_actions['uav_selection_probs'] = [float(x) for x in uav_probs]
            
            # 🤖 =============== 新增联合缓存-迁移控制参数 ===============
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

            # 🔁 让系统模拟器接收Actor导出的指导信号（统一键名为rl_guidance）
            guidance_payload = actions_dict.get('guidance') if isinstance(actions_dict, dict) else None
            if isinstance(guidance_payload, dict) and guidance_payload:
                sim_actions['rl_guidance'] = guidance_payload

            # 🎯 =============== 中央资源分配动作 (Phase 1) ===============
            if self.central_resource_enabled and self.central_resource_action_dim > 0:
                central_start = self.base_action_dim
                central_end = central_start + self.central_resource_action_dim
                central_vector = vehicle_action_array[central_start:central_end]
                allocations = self._decode_central_resource_actions(central_vector)
                if allocations:
                    try:
                        self.simulator.apply_resource_allocation(allocations)
                        sim_actions['central_resource_allocation'] = allocations
                    except Exception as exc:
                        print(f"⚠️ 中央资源分配应用失败: {exc}")
            
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
                        sim_actions['dc_tradeoff_gate'] = float(_np.clip(gate, 0.0, 1.0))
                    except Exception:
                        pass
            except Exception:
                pass

            return sim_actions
        except Exception as e:
            print(f"⚠️ 动作构造异常: {e}")
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
        🤖 将动作字典编码为连续动作向量 - 动态适配动作维度
        """
        # 处理可能的不同输入类型
        action_dim = getattr(self.agent_env, 'action_dim', 18)
        if not isinstance(actions_dict, dict):
            # 如果不是字典，返回默认动作维度
            return np.zeros(action_dim, dtype=np.float32)

        # 🤖 只使用vehicle_agent的完整动作向量
        vehicle_action = actions_dict.get('vehicle_agent')
        if isinstance(vehicle_action, (list, tuple, np.ndarray)):
            vehicle_action = np.array(vehicle_action, dtype=np.float32)
            if vehicle_action.size >= action_dim:
                return vehicle_action[:action_dim]
            action = np.zeros(action_dim, dtype=np.float32)
            action[:vehicle_action.size] = vehicle_action
            return action

        # 默认返回全零动作
        return np.zeros(action_dim, dtype=np.float32)
    
    def _build_actions_from_vector(self, action_vector: np.ndarray) -> Dict[str, np.ndarray]:
        """将连续动作向量恢复为仿真器需要的动作字典（动态维度）"""
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
        """将动作字典编码为离散动作索引"""
        # 处理可能的不同输入类型
        if not isinstance(actions_dict, dict):
            return 0  # 默认动作索引
        
        # 简化实现：将每个智能体的动作组合成一个索引
        vehicle_action = actions_dict.get('vehicle_agent', 0)
        rsu_action = actions_dict.get('rsu_agent', 0)
        uav_action = actions_dict.get('uav_agent', 0)
        
        # 安全地将动作转换为整数
        def safe_int_conversion(value):
            if isinstance(value, (int, np.integer)):
                return int(value)
            elif isinstance(value, np.ndarray):
                if value.size == 1:
                    return int(value.item())
                else:
                    return int(value[0])  # 取第一个元素
            elif isinstance(value, (float, np.floating)):
                return int(value)
            else:
                return 0
        
        vehicle_action = safe_int_conversion(vehicle_action)
        rsu_action = safe_int_conversion(rsu_action)
        uav_action = safe_int_conversion(uav_action)
        
        # 5^3 = 125 种组合
        return vehicle_action * 25 + rsu_action * 5 + uav_action
    
    def _safe_int_conversion(self, value) -> int:
        """安全地将不同类型转换为整数"""
        if isinstance(value, (int, np.integer)):
            return int(value)
        elif isinstance(value, np.ndarray):
            if value.size == 1:
                return int(value.item())
            else:
                return int(value[0])  # 取第一个元素
        elif isinstance(value, (float, np.floating)):
            return int(round(value))
        else:
            return 0  # 安全回退值


def train_single_algorithm(algorithm: str, num_episodes: Optional[int] = None, eval_interval: Optional[int] = None,
                          save_interval: Optional[int] = None, enable_realtime_vis: bool = False,
                          vis_port: int = 5000, silent_mode: bool = False, override_scenario: Optional[Dict[str, Any]] = None,
                          use_enhanced_cache: bool = False, disable_migration: bool = False,
                          enforce_offload_mode: Optional[str] = None, fixed_offload_policy: Optional[str] = None,
                          resume_from: Optional[str] = None, resume_lr_scale: Optional[float] = None,
                          joint_controller: bool = False, num_envs: int = 1) -> Dict:
    """训练单个算法
    
    Args:
        algorithm: 算法名称
        num_episodes: 训练轮次
        eval_interval: 评估间隔
        save_interval: 保存间隔
        enable_realtime_vis: 是否启用实时可视化
        vis_port: 可视化服务器端口
        silent_mode: 静默模式，跳过用户交互（用于批量实验）
        resume_from: 已训练模型路径（.pth 或目录前缀），用于warm-start继续训练
        resume_lr_scale: Warm-start后对学习率的缩放系数（默认0.5，None表示保持原值）
    """
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    # 🔧 自动调整评估间隔和保存间隔
    def auto_adjust_intervals(total_episodes: int):
        """根据总轮数自动调整间隔"""
        # 评估间隔：总轮数的5-8%，范围[10, 100]
        auto_eval = max(10, min(100, int(total_episodes * 0.06)))
        
        # 保存间隔：总轮数的15-20%，范围[50, 500]  
        auto_save = max(50, min(500, int(total_episodes * 0.18)))
        
        return auto_eval, auto_save
    
    # 应用自动调整（仅当用户未指定时）
    if eval_interval is None or save_interval is None:
        auto_eval, auto_save = auto_adjust_intervals(num_episodes)
        if eval_interval is None:
            eval_interval = auto_eval
        if save_interval is None:
            save_interval = auto_save
    
    # 最终回退到配置默认值
    if eval_interval is None:
        eval_interval = config.experiment.eval_interval
    if save_interval is None:
        save_interval = config.experiment.save_interval
    
    print(f"\n>> 开始{algorithm}单智能体算法训练")
    print(f"DEBUG: config.rl.energy_target = {getattr(config.rl, 'energy_target', 'N/A')}")
    print("=" * 60)
    

    

    
    # 创建训练环境（应用额外场景覆盖）
    if num_envs > 1:
        print(f"DEBUG: Entering parallel training block with num_envs={num_envs}")
        print(f"🚀 启动并行训练: {num_envs} 个环境进程")
        from utils.vectorized_env import VectorizedSingleAgentEnvironment
        
        def make_env():
            return SingleAgentTrainingEnvironment(
                algorithm,
                override_scenario=override_scenario,
                use_enhanced_cache=use_enhanced_cache,
                disable_migration=disable_migration,
                enforce_offload_mode=enforce_offload_mode,
                fixed_offload_policy=fixed_offload_policy,
                joint_controller=joint_controller,
                simulation_only=True  # 关键：子进程只跑仿真
            )
        
        # 主环境用于保存模型和评估（加载完整Agent）
        main_env = SingleAgentTrainingEnvironment(
            algorithm,
            override_scenario=override_scenario,
            use_enhanced_cache=use_enhanced_cache,
            disable_migration=disable_migration,
            enforce_offload_mode=enforce_offload_mode,
            fixed_offload_policy=fixed_offload_policy,
            joint_controller=joint_controller,
            simulation_only=False
        )
        
        # 向量化环境用于收集经验
        vec_env = VectorizedSingleAgentEnvironment([make_env for _ in range(num_envs)])
        training_env = main_env  # 保持接口兼容，主要操作main_env
        print(f"✅ 并行环境初始化完成")
    else:
        training_env = SingleAgentTrainingEnvironment(
            algorithm,
            override_scenario=override_scenario,
            use_enhanced_cache=use_enhanced_cache,
            disable_migration=disable_migration,
            enforce_offload_mode=enforce_offload_mode,
            fixed_offload_policy=fixed_offload_policy,
            joint_controller=joint_controller,
        )
        vec_env = None

    canonical_algorithm = training_env.algorithm
    if canonical_algorithm != algorithm:
        print(f"⚙️  规范化算法标识: {canonical_algorithm}")
    algorithm = canonical_algorithm

    resume_loaded = False
    resume_target_path = None
    if resume_from:
        loader = getattr(training_env.agent_env, 'load_models', None)
        if callable(loader):
            try:
                resume_target_path = loader(resume_from) or resume_from
                resume_loaded = True
                print(f"♻️  从已有模型加载成功: {resume_target_path}")
            except Exception as exc:  # pragma: no cover - 容错路径
                print(f"⚠️  加载已有模型失败 ({resume_from}): {exc}")
        else:
            print("⚠️  当前算法环境不支持加载已有模型，忽略 --resume-from")

        if resume_loaded:
            agent_obj = getattr(training_env.agent_env, 'agent', None)
            warmup_adjusted = False
            if agent_obj and hasattr(agent_obj, 'config') and hasattr(agent_obj.config, 'warmup_steps'):
                original_warmup = int(getattr(agent_obj.config, 'warmup_steps', 0) or 0)
                new_warmup = max(500, original_warmup // 4) if original_warmup else 500
                if original_warmup and new_warmup < original_warmup:
                    agent_obj.config.warmup_steps = new_warmup
                    warmup_adjusted = True
            if warmup_adjusted:
                print(f"   • Warm-up 步数由 {original_warmup} 缩减至 {new_warmup}，加速经验缓冲重新填充")

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
            if lr_info:
                print(f"   • 学习率缩放: actor_lr={lr_info.get('actor_lr', 0):.2e}, critic_lr={lr_info.get('critic_lr', 0):.2e}")
            elif resume_lr_scale:
                print("   • 学习率缩放请求未执行（当前算法环境未实现 apply_lr_schedule）")

    lr_decay_episode: Optional[int] = None
    late_stage_lr_factor = 0.5
    lr_decay_applied = resume_loaded  # warm-start 已经缩放过一次学习率
    if algorithm.upper() == 'TD3' and num_episodes >= 1200:
        lr_decay_episode = 1200

    # 🌐 创建实时可视化器（如果启用）
    visualizer = None
    if enable_realtime_vis and REALTIME_AVAILABLE:
        print(f"🌐 启动实时可视化服务器 (端口: {vis_port})")
        # 允许通过环境变量覆盖可视化展示名（用于两阶段标签）
        display_name = os.environ.get('ALGO_DISPLAY_NAME', algorithm)
        visualizer = create_visualizer(
            algorithm=display_name,
            total_episodes=num_episodes,
            port=vis_port,
            auto_open=True
        )
        print(f"✅ 实时可视化已启用，访问 http://localhost:{vis_port}")
    elif enable_realtime_vis and not REALTIME_AVAILABLE:
        print("⚠️  实时可视化未启用（缺少依赖包）")
    
    print(f"训练配置:")
    print(f"  算法: {algorithm}")
    print(f"  总轮次: {num_episodes}")
    print(f"  评估间隔: {eval_interval} (自动调整)" if eval_interval != config.experiment.eval_interval else f"  评估间隔: {eval_interval}")
    print(f"  保存间隔: {save_interval} (自动调整)" if save_interval != config.experiment.save_interval else f"  保存间隔: {save_interval}")
    print(f"  实时可视化: {'启用 ✓' if visualizer else '禁用'}")
    if hasattr(config, 'rl'):
        print(
            f"  奖励权重: 延迟={getattr(config.rl, 'reward_weight_delay', 0.0):.2f}, "
            f"能耗={getattr(config.rl, 'reward_weight_energy', 0.0):.2f}, "
            f"丢弃={getattr(config.rl, 'reward_penalty_dropped', 0.0):.2f}"
        )
        print(f"  【配置目标】")
        print(f"    - latency_target:    {getattr(config.rl, 'latency_target', 'N/A')}s")
        print(f"    - energy_target:     {getattr(config.rl, 'energy_target', 'N/A')}J")
        print(f"  【权重】")
        print(f"    - ω_T (delay):       {_general_reward_calculator.weight_delay:.2f}")
        print(f"    - ω_E (energy):      {_general_reward_calculator.weight_energy:.2f}")
        print(f"  【其他配置】")
        print(f"    - 丢弃惩罚:          {_general_reward_calculator.penalty_dropped:.2f}")
        print(f"    - 奖励裁剪范围:      {_general_reward_calculator.reward_clip_range}")
        print("=" * 60 + "\n")
    
    # 创建结果目录
    os.makedirs(f"results/single_agent/{algorithm.lower()}", exist_ok=True)
    os.makedirs(f"results/models/single_agent/{algorithm.lower()}", exist_ok=True)
    
    # 训练循环
    # 🔧 修复：per-step奖励范围约为-2.0到-0.5，初始值应相应调整
    best_avg_reward = -10.0  # per-step奖励初始阈值（负值越大越好）
    training_start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start_time = time.time()
        
        # 运行训练轮次
        if vec_env is not None:
            # 并行训练逻辑
            # 1. 重置所有环境
            states = vec_env.reset()
            episode_rewards = np.zeros(num_envs)
            episode_steps_count = np.zeros(num_envs)
            active_envs = np.ones(num_envs, dtype=bool)
            infos = []
            
            # 2. 步进循环 (以max_steps为准)
            max_steps = config.experiment.max_steps_per_episode
            
            for step in range(max_steps):
                # 批量选择动作
                # 注意：main_env.agent_env 必须加载了 Agent
                # states: (num_envs, state_dim)
                # actions: (num_envs, action_dim)
                actions = training_env.agent_env.select_action(states, training=True)
                
                # 批量执行动作
                # vec_env.step 接受 actions 数组，返回 (next_states, rewards, dones, infos)
                next_states, rewards, dones, step_infos = vec_env.step(actions)
                
                # 更新episode统计
                episode_rewards[active_envs] += rewards[active_envs]
                episode_steps_count[active_envs] += 1
                infos.extend([info for i, info in enumerate(step_infos) if active_envs[i]])
                
                # 标记已完成的环境
                active_envs = active_envs & ~dones
                
                # 更新Agent（使用主环境的Agent）
                training_env.agent_env.update()
                
                # 更新状态
                states = next_states
                
                if not np.any(active_envs):
                    break
            
            # 记录平均奖励
            avg_ep_reward = np.mean(episode_rewards)
            
            # 聚合多环境的system_metrics
            aggregated_metrics = {}
            if len(infos) > 0 and 'system_metrics' in infos[0]:
                keys = infos[0]['system_metrics'].keys()
                for key in keys:
                    values = [info['system_metrics'].get(key, 0) for info in infos]
                    try:
                        aggregated_metrics[key] = np.mean([float(v) for v in values])
                    except:
                        aggregated_metrics[key] = values[0]
            
            episode_result = {
                'avg_reward': avg_ep_reward,
                'steps': int(np.mean(episode_steps_count)),
                'system_metrics': aggregated_metrics,
                'step_stats': infos[0].get('step_stats', {}) if len(infos) > 0 else {}
            }

            # 记录 episode 级指标（并行环境聚合后的均值）
            training_env._record_episode_metrics(aggregated_metrics, episode_steps=episode_result['steps'])
            
            # 记录到主环境用于统计
            training_env.episode_rewards.append(avg_ep_reward)
            
        else:
            # 原始串行训练
            episode_result = training_env.run_episode(episode, visualizer=visualizer)
            training_env.episode_rewards.append(episode_result['avg_reward'])
        
        episode_steps = episode_result.get('steps', config.experiment.max_steps_per_episode)

        if algorithm.upper() == 'OPTIMIZED_TD3' and hasattr(training_env.agent_env, 'agent'):
            agent_ref = training_env.agent_env.agent
            if hasattr(agent_ref, 'set_episode_count'):
                try:
                    agent_ref.set_episode_count(episode, episode_result['avg_reward'])
                except Exception:
                    pass
        
        # 更新性能追踪器
        training_env.performance_tracker['recent_rewards'].update(episode_result['avg_reward'])
        per_step_reward = episode_result['avg_reward'] / max(1, episode_steps)
        training_env.performance_tracker['recent_step_rewards'].update(per_step_reward)
        
        system_metrics = episode_result['system_metrics']
        training_env.performance_tracker['recent_delays'].update(system_metrics.get('avg_task_delay', 0))
        training_env.performance_tracker['recent_energy'].update(system_metrics.get('total_energy_consumption', 0))
        training_env.performance_tracker['recent_completion'].update(system_metrics.get('task_completion_rate', 0))
        # 🌐 更新实时可视化
        if visualizer:
            step_stats = episode_result.get('step_stats', {}) # Assuming step_stats is part of episode_result
            vis_metrics = {
                'avg_delay': float(system_metrics.get('avg_task_delay', 0)),
                'total_energy': float(system_metrics.get('total_energy_consumption', 0)),
                'task_completion_rate': float(system_metrics.get('task_completion_rate', 0)),
                'cache_hit_rate': float(system_metrics.get('cache_hit_rate', 0)),
                'data_loss_ratio_bytes': float(system_metrics.get('data_loss_ratio_bytes', 0)),
                'migration_success_rate': float(system_metrics.get('migration_success_rate', 0)),
                'vehicle_positions': step_stats.get('vehicle_positions', []) # 🔧 传递车辆位置
            }
            visualizer.update(episode, float(episode_result['avg_reward']), vis_metrics)
        
        episode_time = time.time() - episode_start_time
        
        # 🔧 修复：每个episode都输出简化日志，显示关键指标
        avg_reward_step = training_env.performance_tracker['recent_step_rewards'].get_average()
        avg_delay = training_env.performance_tracker['recent_delays'].get_average()
        avg_energy = training_env.performance_tracker['recent_energy'].get_average()
        avg_completion = training_env.performance_tracker['recent_completion'].get_average()
        
        # 每个episode显示一行简化信息
        print(f"Episode {episode:4d}/{num_episodes} | "
              f"奖励:{avg_reward_step:7.3f} | "
              f"延迟:{avg_delay:6.3f}s | "
              f"能耗:{avg_energy:7.1f}J | "
              f"完成率:{avg_completion:5.1%} | "
              f"用时:{episode_time:5.2f}s")
        
        # 定期输出详细进度
        if episode % 10 == 0:
            print(f"\n{'='*70}")
            print(f"轮次 {episode:4d}/{num_episodes} 详细统计:")
            print(f"  平均每步奖励: {avg_reward_step:8.3f}")
            print(f"  平均时延: {avg_delay:8.3f}s")
            print(f"  平均能耗: {avg_energy:8.1f}J")
            print(f"  完成率:   {avg_completion:8.1%}")
            print(f"  轮次用时: {episode_time:6.3f}s")
            print(f"{'='*70}\n")
        
        # 评估模型
        if episode % eval_interval == 0:
            eval_result = evaluate_single_model(algorithm, training_env, episode)
            print(f"\n📊 轮次 {episode} 评估结果:")
            print(f"  Per-Step奖励: {eval_result['avg_reward']:.3f}")
            print(f"  评估时延: {eval_result['avg_delay']:.3f}s")
            print(f"  评估完成率: {eval_result['completion_rate']:.1%}")
            
            # 保存最佳模型
            if eval_result['avg_reward'] > best_avg_reward:
                best_avg_reward = eval_result['avg_reward']
                best_model_base = f"results/models/single_agent/{algorithm.lower()}/best_model"
                saved_target = training_env.agent_env.save_models(best_model_base)
                saved_display = saved_target or best_model_base
                print(f"  💾 保存最佳模型 -> {saved_display} (Per-Step奖励: {best_avg_reward:.3f})")
        
        # 达到后期阶段时缩放TD3学习率（一次性）
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
            if lr_info:
                print(
                    f"🔧 第{episode}轮触发TD3学习率缩放 -> "
                    f"actor_lr={lr_info['actor_lr']:.2e}, critic_lr={lr_info['critic_lr']:.2e}"
                )

        # 定期保存模型
        if episode % save_interval == 0:
            checkpoint_base = f"results/models/single_agent/{algorithm.lower()}/checkpoint_{episode}"
            checkpoint_path = training_env.agent_env.save_models(checkpoint_base)
            checkpoint_display = checkpoint_path or checkpoint_base
            print(f"💾 保存检查点: {checkpoint_display}")
    
    # 训练完成
    total_training_time = time.time() - training_start_time
    
    # 🌐 标记实时可视化完成
    if visualizer:
        visualizer.complete()
        print(f"✅ 实时可视化已标记完成")
    
    print("\n" + "=" * 60)
    print(f"🎉 {algorithm}训练完成!")
    print(f"⏱️  总训练时间: {total_training_time/3600:.2f} 小时")
    print(f"🏆 最佳Per-Step奖励: {best_avg_reward:.3f}")
    
    # 收集系统统计信息用于报告
    simulator_stats = {}
    
    # 🏢 显示中央RSU调度器报告
    try:
        central_report = training_env.simulator.get_central_scheduling_report()
        if central_report.get('status') != 'not_available' and central_report.get('status') != 'error':
            print(f"\n🏢 中央RSU骨干调度器总结:")
            print(f"   📊 调度调用次数: {central_report.get('scheduling_calls', 0)}")
            
            scheduler_status = central_report.get('central_scheduler_status', {})
            if 'global_metrics' in scheduler_status:
                metrics = scheduler_status['global_metrics']
                print(f"   ⚖️ 负载均衡指数: {metrics.get('load_balance_index', 0.0):.3f}")
                print(f"   💚 系统健康状态: {scheduler_status.get('system_health', 'N/A')}")
                
                # 收集调度器统计信息
                simulator_stats['scheduling_calls'] = central_report.get('scheduling_calls', 0)
                simulator_stats['load_balance_index'] = metrics.get('load_balance_index', 0.0)
                simulator_stats['system_health'] = scheduler_status.get('system_health', 'N/A')
            
            # 显示各RSU负载分布
            rsu_details = central_report.get('rsu_details', {})
            if rsu_details:
                print(f"   📡 各RSU负载状态:")
                for rsu_id, details in rsu_details.items():
                    print(f"      {rsu_id}: CPU负载={details['cpu_usage']:.1%}, 任务队列={details['queue_length']}")
        else:
            print(f"📋 中央调度器状态: {central_report.get('message', '未启用')}")
        
        # 🔌 显示有线回传网络统计
        rsu_migration_delay = training_env.simulator.stats.get('rsu_migration_delay', 0.0)
        rsu_migration_energy = training_env.simulator.stats.get('rsu_migration_energy', 0.0)
        rsu_migration_data = training_env.simulator.stats.get('rsu_migration_data', 0.0)
        backhaul_collection_delay = training_env.simulator.stats.get('backhaul_collection_delay', 0.0)
        backhaul_command_delay = training_env.simulator.stats.get('backhaul_command_delay', 0.0)
        backhaul_total_energy = training_env.simulator.stats.get('backhaul_total_energy', 0.0)
        
        # 🚗 显示各种迁移统计
        handover_migrations = training_env.simulator.stats.get('handover_migrations', 0)
        uav_migration_count = training_env.simulator.stats.get('uav_migration_count', 0)
        uav_migration_distance = training_env.simulator.stats.get('uav_migration_distance', 0.0)
        
        # 收集迁移统计信息
        simulator_stats['rsu_migration_delay'] = rsu_migration_delay
        simulator_stats['rsu_migration_energy'] = rsu_migration_energy
        simulator_stats['rsu_migration_data'] = rsu_migration_data
        simulator_stats['backhaul_total_energy'] = backhaul_total_energy
        simulator_stats['handover_migrations'] = handover_migrations
        simulator_stats['uav_migration_count'] = uav_migration_count
        
        if rsu_migration_data > 0 or backhaul_total_energy > 0 or handover_migrations > 0 or uav_migration_count > 0:
            print(f"\n🔌 有线回传网络与迁移统计:")
            print(f"   📡 RSU迁移数据: {rsu_migration_data:.1f}MB")
            print(f"   ⏱️ RSU迁移延迟: {rsu_migration_delay*1000:.1f}ms")
            print(f"   ⚡ RSU迁移能耗: {rsu_migration_energy:.2f}J")
            print(f"   📊 信息收集延迟: {backhaul_collection_delay*1000:.1f}ms")
            print(f"   📤 指令分发延迟: {backhaul_command_delay*1000:.1f}ms")
            print(f"   🔋 回传网络总能耗: {backhaul_total_energy:.2f}J")
            if handover_migrations > 0:
                print(f"   🚗 车辆跟随迁移: {handover_migrations} 次")
            if uav_migration_count > 0:
                avg_distance = uav_migration_distance / uav_migration_count if uav_migration_count > 0 else 0
                print(f"   🚁 UAV迁移: {uav_migration_count} 次, 平均距离{avg_distance:.1f}m")
    except Exception as e:
        print(f"⚠️ 中央调度报告获取失败: {e}")
    
    # 保存训练结果
    results = save_single_training_results(algorithm, training_env, total_training_time, override_scenario=override_scenario)
    
    # 绘制训练曲线
    plot_single_training_curves(algorithm, training_env)
    
    # 生成HTML训练报告
    print("\n" + "=" * 60)
    print("📝 生成训练报告...")
    
    try:
        report_generator = HTMLReportGenerator()
        html_content = report_generator.generate_full_report(
            algorithm=algorithm,
            training_env=training_env,
            training_time=total_training_time,
            results=results,
            simulator_stats=simulator_stats
        )
        
        # 生成报告文件名
        timestamp = generate_timestamp()
        report_filename = f"training_report_{timestamp}.html" if timestamp else "training_report.html"
        report_path = f"results/single_agent/{algorithm.lower()}/{report_filename}"
        
        print(f"✅ 训练报告已生成")
        print(f"📄 报告包含:")
        print(f"   - 执行摘要与关键指标")
        print(f"   - 训练配置详情")
        print(f"   - 性能指标可视化图表")
        print(f"   - 详细的系统统计信息")
        print(f"   - 自适应控制器分析")
        print(f"   - 优化建议与结论")
        
        # 询问用户是否保存报告（静默模式下自动保存）
        # 🔧 强制自动保存，不询问用户
        if report_generator.save_report(html_content, report_path):
            print(f"✅ 报告已自动保存到: {report_path}")
        else:
            print("❌ 报告保存失败")
    
    except Exception as e:
        print(f"⚠️ 生成训练报告时出错: {e}")
        print("训练数据已正常保存，可稍后手动生成报告")
    
    return results


def evaluate_single_model(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                         episode: int, num_eval_episodes: int = 5) -> Dict:
    """评估单智能体模型性能 - 修复版，防止inf和nan"""
    import numpy as np
    
    eval_rewards = []
    eval_delays = []
    eval_completions = []
    
    def safe_value(value: float, default: float = 0.0, max_val: float = 1e6) -> float:
        """安全处理数值，防止inf和nan"""
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
                if isinstance(actions_result, tuple):  # PPO返回元组
                    actions_dict = actions_result[0]
                elif isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = {}
                action = training_env._encode_continuous_action(actions_dict)
            
            # 评估时也传入动作字典，确保偏好生效
            next_state, reward, done, info = training_env.step(action, state, actions_dict)
            
            # 安全处理奖励和指标
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
        
        # 安全计算平均值
        steps = max(1, steps)  # 防止除零
        eval_rewards.append(safe_value(episode_reward / steps, -20.0, 80.0))
        eval_delays.append(safe_value(episode_delay / steps, 0.0, 10.0))
        eval_completions.append(safe_value(episode_completion / steps, 0.0, 1.0))
    
    # 安全计算最终结果
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


def _calculate_stable_delay_average(training_env: SingleAgentTrainingEnvironment) -> float:
    """
    计算稳定的时延平均值，避免MovingAverage(100)的训练波动影响
    
    策略：
    1. 优先使用episode_metrics中的完整数据（如果可用）
    2. 使用后50%的数据（排除前期学习阶段）
    3. 如果数据不足，回退到MovingAverage(100)
    
    Returns:
        float: 稳定的平均时延
    """
    # 尝试从episode_metrics获取完整时延数据
    if hasattr(training_env, 'episode_metrics') and 'avg_delay' in training_env.episode_metrics:
        delay_history = training_env.episode_metrics['avg_delay']
        
        if len(delay_history) >= 100:
            # 使用后50%的数据（更成熟的策略）
            half_point = len(delay_history) // 2
            converged_delays = delay_history[half_point:]
            return float(np.mean(converged_delays))
        elif len(delay_history) >= 50:
            # 如果不足100轮，使用后30轮
            return float(np.mean(delay_history[-30:]))
        elif len(delay_history) > 0:
            # 数据很少，使用全部
            return float(np.mean(delay_history))
    
    # 回退：使用MovingAverage
    return training_env.performance_tracker['recent_delays'].get_average()


def _calculate_stable_completion_average(training_env: SingleAgentTrainingEnvironment) -> float:
    """
    计算稳定的完成率平均值
    
    Returns:
        float: 稳定的平均完成率
    """
    # 尝试从episode_metrics获取完整完成率数据
    if hasattr(training_env, 'episode_metrics') and 'task_completion_rate' in training_env.episode_metrics:
        completion_history = training_env.episode_metrics['task_completion_rate']
        
        if len(completion_history) >= 100:
            # 使用后50%的数据
            half_point = len(completion_history) // 2
            converged_completions = completion_history[half_point:]
            return float(np.mean(converged_completions))
        elif len(completion_history) >= 50:
            # 如果不足100轮，使用后30轮
            return float(np.mean(completion_history[-30:]))
        elif len(completion_history) > 0:
            # 数据很少，使用全部
            return float(np.mean(completion_history))
    
    # 回退：使用MovingAverage
    return training_env.performance_tracker['recent_completion'].get_average()


def save_single_training_results(algorithm: str, training_env: SingleAgentTrainingEnvironment, 
                                training_time: float,
                                override_scenario: Optional[Dict[str, Any]] = None) -> Dict:
    """保存训练结果"""
    # 生成时间戳
    timestamp = generate_timestamp()
    
    # 🔧 同时提供Episode总奖励和Per-Step平均奖励
    recent_episode_reward = training_env.performance_tracker['recent_rewards'].get_average()
    
    # 🔧 优化：使用实际平均步数计算 avg_step_reward
    if 'episode_steps' in training_env.episode_metrics and training_env.episode_metrics['episode_steps']:
        # 使用最近100个episode的平均步数
        recent_steps = training_env.episode_metrics['episode_steps'][-100:]
        avg_steps_per_episode = sum(recent_steps) / len(recent_steps)
    else:
        # 回退到配置的默认值
        avg_steps_per_episode = config.experiment.max_steps_per_episode
    
    avg_step_reward = recent_episode_reward / avg_steps_per_episode
    
    # 获取网络拓扑信息
    num_vehicles = len(training_env.simulator.vehicles)
    num_rsus = len(training_env.simulator.rsus)
    num_uavs = len(training_env.simulator.uavs)
    state_dim = getattr(training_env.agent_env, 'state_dim', 'N/A')
    
    # 🆕 修复：收集完整的系统配置参数（用于HTML报告显示）
    # 直接使用已导入的config对象
    
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
        # 🆕 添加系统配置参数（HTML报告需要）
        'system_config': {
            'num_vehicles': num_vehicles,
            'num_rsus': num_rsus,
            'num_uavs': num_uavs,
            'simulation_time': config.simulation_time,
            'time_slot': config.time_slot,
            'device': str(config.device),
            'random_seed': config.random_seed,
        },
        # 🆕 添加网络配置参数
        'network_config': {
            'bandwidth': config.network.bandwidth,
            'carrier_frequency': config.communication.carrier_frequency,
            'coverage_radius': config.network.coverage_radius,
        },
        # 🆕 添加通信配置参数
        'communication_config': {
            'vehicle_tx_power': config.communication.vehicle_tx_power,
            'rsu_tx_power': config.communication.rsu_tx_power,
            'uav_tx_power': config.communication.uav_tx_power,
            'antenna_gain_vehicle': config.communication.antenna_gain_vehicle,
            'antenna_gain_rsu': config.communication.antenna_gain_rsu,
            'antenna_gain_uav': config.communication.antenna_gain_uav,
        },
        # 🆕 添加计算能力参数
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
        # 🆕 添加任务和迁移参数
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
            # 提供两种奖励指标，用途不同
            'avg_episode_reward': recent_episode_reward,  # Episode总奖励（训练目标）
            'avg_step_reward': avg_step_reward,           # 每步平均奖励（对比评估）
            'avg_reward': avg_step_reward,  # 向后兼容：默认使用per-step（与可视化一致）
            
            # 🔧 修复：使用更稳定的平均方法，避免MovingAverage(100)的波动影响
            'avg_delay': _calculate_stable_delay_average(training_env),
            'avg_completion': _calculate_stable_completion_average(training_env)
        }
    }
    
    print(f"📊 收集的配置参数:")
    print(f"   系统拓扑: {num_vehicles}车辆, {num_rsus}RSU, {num_uavs}UAV")
    print(f"   网络配置: 带宽{config.network.bandwidth/1e6:.0f}MHz, 频率{config.communication.carrier_frequency/1e9:.1f}GHz")
    print(f"   任务参数: 到达率{config.task.arrival_rate:.1f}, 数据量{sum(config.task.data_size_range)/2/1e6:.1f}MB")
    
    # 使用时间戳文件名
    filename = get_timestamped_filename("training_results")
    filepath = f"results/single_agent/{algorithm.lower()}/{filename}"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 {algorithm}训练结果已保存到 {filepath}")
    
    return results


def plot_single_training_curves(algorithm: str, training_env: SingleAgentTrainingEnvironment):
    """绘制训练曲线 - 简洁优美版"""
    
    # 🎨 使用新的简洁可视化系统
    from visualization.clean_charts import create_training_chart, cleanup_old_charts, plot_objective_function_breakdown
    
    # 创建算法目录
    algorithm_dir = f"results/single_agent/{algorithm.lower()}"
    
    # 清理旧的冗余图表
    cleanup_old_charts(algorithm_dir)
    
    # 生成核心图表
    chart_path = f"{algorithm_dir}/training_overview.png"
    create_training_chart(training_env, algorithm, chart_path)
    
    # 🎯 生成目标函数分解图（显示时延、能耗两项核心目标的权重贡献）
    objective_path = f"{algorithm_dir}/objective_analysis.png"
    plot_objective_function_breakdown(training_env, algorithm, objective_path)
    
    print(f"📈 {algorithm} 训练可视化已完成")
    print(f"   训练总览: {chart_path}")
    print(f"   目标分析: {objective_path}")
    
    # 生成训练总结
    from visualization.clean_charts import get_summary_text
    summary = get_summary_text(training_env, algorithm)
    print(f"\n{summary}")


def compare_single_algorithms(algorithms: List[str], num_episodes: Optional[int] = None) -> Dict:
    """比较多个单智能体算法的性能"""
    # 使用配置中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    print("\n🔥 开始单智能体算法性能比较")
    print("=" * 60)
    
    results = {}
    
    # 训练所有算法
    for algorithm in algorithms:
        print(f"\n开始训练 {algorithm}...")
        results[algorithm] = train_single_algorithm(algorithm, num_episodes)
    
    # 🎨 生成简洁的对比图表
    from visualization.clean_charts import create_comparison_chart
    timestamp = generate_timestamp()
    comparison_chart_path = f"results/single_agent_comparison_{timestamp}.png" if timestamp else "results/single_agent_comparison.png"
    create_comparison_chart(results, comparison_chart_path)
    
    # 保存比较结果
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
    
    # 计算汇总统计
    for algorithm, result in results.items():
        final_perf = result['final_performance']
        comparison_results['summary'][algorithm] = {
            'final_avg_reward': final_perf['avg_reward'],
            'final_avg_delay': final_perf['avg_delay'],
            'final_completion_rate': final_perf['avg_completion'],
            'training_time_hours': result['training_config']['training_time_hours']
        }
    
    # 使用时间戳文件名
    comparison_filename = get_timestamped_filename("single_agent_comparison")
    with open(f"results/{comparison_filename}", "w", encoding="utf-8") as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print("\n🎯 单智能体算法比较完成！")
    print(f"📄 比较结果已保存到 results/{comparison_filename}")
    print(f"📊 对比图表已保存到 {comparison_chart_path}")
    
    return comparison_results




def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='单智能体算法训练脚本')
    parser.add_argument('--algorithm', type=str, choices=['DDPG', 'TD3', 'TD3-LE', 'TD3_LE', 'TD3_LATENCY_ENERGY', 'DQN', 'PPO', 'SAC', 'CAM_TD3', 'OPTIMIZED_TD3'],
                       help='选择训练算法')
    parser.add_argument('--episodes', type=int, default=None, help=f'训练轮次 (默认: {config.experiment.num_episodes})')
    parser.add_argument('--eval_interval', type=int, default=None, help=f'评估间隔 (默认: {config.experiment.eval_interval})')
    parser.add_argument('--save_interval', type=int, default=None, help=f'保存间隔 (默认: {config.experiment.save_interval})')
    parser.add_argument('--compare', action='store_true', help='比较所有算法')
    parser.add_argument('--quick-test', action='store_true', help='快速基准测试，仅运行少量 episodes')
    parser.add_argument('--seed', type=int, default=None, help='覆盖随机种子 (默认读取config或环境变量)')
    parser.add_argument('--num-vehicles', type=int, default=None, help='覆盖车辆数量用于实验')
    parser.add_argument('--force-offload', type=str, choices=['local', 'remote', 'local_only', 'remote_only'],
                        help='强制卸载模式：local/local_only 或 remote/remote_only')
    parser.add_argument('--fixed-offload-policy', type=str, 
                        choices=['random', 'greedy', 'local_only', 'rsu_only', 'round_robin', 'weighted'],
                        help='固定卸载策略（不使用智能体学习）：random/greedy/local_only/rsu_only/round_robin/weighted')
    # 🌐 实时可视化参数（默认开启，可通过 --no-realtime-vis 关闭）
    parser.add_argument(
        '--realtime-vis',
        action='store_true',
        dest='realtime_vis',
        default=True,
        help='启用实时可视化（默认已开启）'
    )
    parser.add_argument(
        '--no-realtime-vis',
        action='store_false',
        dest='realtime_vis',
        help='禁用实时可视化（覆盖默认开启行为）'
    )
    parser.add_argument('--vis-port', type=int, default=5000, help='实时可视化服务器端口 (默认: 5000)')
    # 🚀 增强缓存参数（默认启用）
    parser.add_argument('--no-enhanced-cache', action='store_true', 
                       help='禁用增强缓存系统（默认启用分层L1/L2 + 热度策略 + RSU协作）')
    # 🧭 两阶段管线开关（Stage-1 预分配 + Stage-2 精细调度）
    parser.add_argument('--two-stage', action='store_true', help='启用两阶段求解（预分配+精细调度）')
    # 🧠 指定两个阶段的算法
    parser.add_argument('--stage1-alg', type=str, default=None,
                        help='阶段一算法（offloading 头）：heuristic|greedy|cache_first|distance_first')
    parser.add_argument('--stage2-alg', type=str, default=None,
                        help='阶段二算法（缓存/迁移控制的RL）：TD3|SAC|DDPG|PPO|DQN|TD3-LE|OPTIMIZED_TD3')
    # 🎯 中央资源分配架构（Phase 1 + Phase 2）- 默认启用
    parser.add_argument('--central-resource', action='store_true', default=True,
                        help='启用中央资源分配架构（Phase 1决策 + Phase 2执行），扩展状态/动作空间 [默认启用]')
    parser.add_argument('--no-central-resource', action='store_false', dest='central_resource',
                        help='禁用中央资源分配架构，使用标准均匀资源分配')
    parser.add_argument('--silent-mode', action='store_true',
                        help='启用静默模式，跳过训练结束后的交互提示')
    parser.add_argument('--resume-from', type=str,
                        help='从已有模型 (.pth 或目录前缀) 继续训练，复用已学策略')
    parser.add_argument('--resume-lr-scale', type=float, default=None,
                        help='Warm-start 后的学习率缩放系数 (默认0.5，设为1可保留原值)')
    parser.add_argument('--num-envs', type=int, default=4,
                        help='并行训练环境数量 (默认: 4)')
    
    # 🆕 通信模型优化参数（3GPP标准增强）
    parser.add_argument('--comm-enhancements', action='store_true',
                        help='启用所有通信模型优化（快衰落+系统级干扰+动态带宽）Enable all communication model enhancements')
    parser.add_argument('--fast-fading', action='store_true',
                        help='启用随机快衰落（Rayleigh/Rician）Enable fast fading')
    parser.add_argument('--system-interference', action='store_true',
                        help='启用系统级干扰计算 Enable system-level interference calculation')
    parser.add_argument('--dynamic-bandwidth', action='store_true',
                        help='启用动态带宽分配 Enable dynamic bandwidth allocation')
    
    args = parser.parse_args()

    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)
        _apply_global_seed_from_env()

    # 设置默认超参数（可通过环境变量覆盖）
    os.environ.setdefault('TD3_ACTOR_LR', '5e-5')
    os.environ.setdefault('TD3_CRITIC_LR', '8e-5')
    os.environ.setdefault('TD3_BATCH_SIZE', '512')
    os.environ.setdefault('RL_SMOOTH_DELAY', '0.6')
    os.environ.setdefault('RL_SMOOTH_ENERGY', '0.6')
    os.environ.setdefault('RL_SMOOTH_ALPHA', '0.25')

    # 快速基准测试模式
    if args.quick_test:
        print("=== QUICK TEST (Baseline Fixed Policy) ===")
        # 创建环境并强制使用本地策略
        env = SingleAgentTrainingEnvironment('OPTIMIZED_TD3', enforce_offload_mode='local_only')
        for ep in range(5):
            state = env.reset_environment()
            total_reward = 0.0
            for step in range(100):
                # 获取动作（虽然被强制本地策略覆盖，但仍需传入）
                actions_result = env.agent_env.get_actions(state, training=False)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                
                # 编码动作
                if hasattr(env, '_encode_continuous_action'):
                    action = env._encode_continuous_action(actions_dict)
                else:
                    # Fallback for simple envs
                    action = np.zeros(env.agent_env.action_dim)

                next_state, reward, done, info = env.step(action, state, actions_dict)
                total_reward += reward
                state = next_state
                if done:
                    break
            print(f"Baseline Episode {ep}: Reward = {total_reward:.4f}")
        print("=== QUICK TEST DONE ===")
        return
    # 🎯 中央资源分配架构（默认启用）
    if args.central_resource:
        os.environ['CENTRAL_RESOURCE'] = '1'
        print("🎯 启用中央资源分配架构（Phase 1 + Phase 2）[默认模式]")
    else:
        os.environ.pop('CENTRAL_RESOURCE', None)
        print("⚠️  使用标准均匀资源分配模式（已通过 --no-central-resource 禁用中央资源）")
    
    # 🆕 通信模型优化配置
    if args.comm_enhancements or args.fast_fading or args.system_interference or args.dynamic_bandwidth:
        print("\n" + "="*70)
        print("🌐 通信模型优化配置（3GPP标准增强）")
        print("="*70)
        
        # 如果启用了--comm-enhancements，则启用所有优化
        if args.comm_enhancements:
            config.communication.enable_fast_fading = True
            config.communication.use_system_interference = True
            config.communication.use_bandwidth_allocator = True
            config.communication.use_communication_enhancements = True
            print("✅ 启用所有通信模型优化（完整3GPP标准模式）")
        else:
            # 单独配置各项优化
            if args.fast_fading:
                config.communication.enable_fast_fading = True
                print("✅ 启用随机快衰落（Rayleigh/Rician分布）")
            
            if args.system_interference:
                config.communication.use_system_interference = True
                print("✅ 启用系统级干扰计算")
            
            if args.dynamic_bandwidth:
                config.communication.use_bandwidth_allocator = True
                print("✅ 启用动态带宽分配调度器")
        
        # 显示配置详情
        print("\n配置详情：")
        print(f"  - 快衰落: {'启用' if config.communication.enable_fast_fading else '禁用'}")
        print(f"  - 系统级干扰: {'启用' if config.communication.use_system_interference else '禁用'}")
        print(f"  - 动态带宽分配: {'启用' if config.communication.use_bandwidth_allocator else '禁用'}")
        print(f"  - 载波频率: {config.communication.carrier_frequency/1e9:.1f} GHz")
        print(f"  - 编码效率: {config.communication.coding_efficiency}")
        if config.communication.enable_fast_fading:
            print(f"  - 快衰落参数: σ={config.communication.fast_fading_std}, K={config.communication.rician_k_factor}dB")
        print("="*70 + "\n")
    
    # Toggle two-stage pipeline via environment for the simulator
    if args.two_stage:
        os.environ['TWO_STAGE_MODE'] = '1'
    # Stage1/Stage2 algorithm selections (env-based for env init)
    if args.stage1_alg:
        os.environ['STAGE1_ALG'] = args.stage1_alg
    if args.stage2_alg:
        # 允许覆盖主算法选择
        if not args.algorithm:
            args.algorithm = args.stage2_alg
        else:
            # 覆写为阶段二选择
            args.algorithm = args.stage2_alg

    # 🔧 修复：正确构建override_scenario参数
    override_scenario = None
    if args.num_vehicles is not None:
        override_scenario = {
            "num_vehicles": args.num_vehicles,
        }
        # 同时设置环境变量（向后兼容）
        os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(override_scenario)
        print(f"📋 覆盖参数: 车辆数 = {args.num_vehicles}")
   
    enforce_mode = None
    if getattr(args, 'force_offload', None):
        if args.force_offload in ('local', 'local_only'):
            enforce_mode = 'local_only'
        elif args.force_offload in ('remote', 'remote_only'):
            enforce_mode = 'remote_only'
    
    # 创建结果目录
    os.makedirs("results/single_agent", exist_ok=True)
    
    # 🎯 显示CAMTD3系统信息
    if args.algorithm and not args.compare:
        print("\n" + "="*80)
        print("🚀 CAMTD3 训练系统启动")
        print("="*80)
        print(f"系统名称: CAMTD3 (Cache-Aware Migration with Twin Delayed DDPG)")
        print(f"使用算法: {args.algorithm}")
        print(f"系统架构: Phase 1 (中央资源分配) + Phase 2 (任务执行)")
        print(f"训练轮数: {args.episodes}")
        if args.seed:
            print(f"随机种子: {args.seed}")
        print(f"完整名称: CAMTD3-{args.algorithm}")
        print("="*80 + "\n")
    
    if args.compare:
        # 比较所有算法
        algorithms = ['DDPG', 'TD3', 'TD3-LE', 'DQN', 'PPO', 'SAC']
        compare_single_algorithms(algorithms, args.episodes)
    elif args.algorithm:
        # 训练单个算法 - 🔧 传递override_scenario参数
        train_single_algorithm(
            args.algorithm, 
            args.episodes, 
            args.eval_interval, 
            args.save_interval,
            enable_realtime_vis=args.realtime_vis,
            vis_port=args.vis_port,
            override_scenario=override_scenario,  # 🔧 新增：传递覆盖参数
            use_enhanced_cache=not args.no_enhanced_cache,  # 🚀 默认启用增强缓存
            enforce_offload_mode=enforce_mode,
            fixed_offload_policy=getattr(args, 'fixed_offload_policy', None),  # 🎯 固定卸载策略
            silent_mode=args.silent_mode,
            resume_from=args.resume_from,
            resume_lr_scale=args.resume_lr_scale,
            num_envs=args.num_envs
        )
    else:
        print("请指定 --algorithm 或使用 --compare 标志")
        print("使用 python train_single_agent.py --help 查看帮助")


if __name__ == "__main__":
    main()
    
"""

🔄 完整执行流程（分5个阶段）
📌 阶段1: 系统初始化 (train_single_agent.py: main函数)
1.1 参数解析与配置
├─ 解析命令行参数
│  ├─ algorithm = "TD3"
│  ├─ episodes = 800  
│  ├─ num_vehicles = 12
│  └─ enhanced_cache = True (默认)
│
├─ 设置随机种子
│  └─ 从config或环境变量读取种子
│
└─ 构建场景配置 override_scenario
   └─ {'num_vehicles': 12, 'override_topology': True}
   
1.2 创建训练环境 (SingleAgentTrainingEnvironment)
环境初始化流程:
├─ 1) 选择仿真器类型
│  ├─ use_enhanced_cache=True
│  └─ simulator = EnhancedSystemSimulator(scenario_config)
│
├─ 2) 初始化仿真器组件 (system_simulator.py)
│  ├─ 车辆初始化: 12辆车
│  │  ├─ 位置: 随机分布在道路上
│  │  ├─ 速度: 30-50 km/h
│  │  └─ 缓存: L1(200MB) + L2(300MB)
│  │
│  ├─ RSU部署: 4个路侧单元 (固定拓扑)
│  │  ├─ 位置: 等间距分布
│  │  ├─ 覆盖半径: 150m
│  │  ├─ 缓存容量: 1000MB
│  │  └─ 计算能力: 50 GHz
│  │
│  └─ UAV部署: 2个无人机
│     ├─ 位置: 动态巡航
│     ├─ 高度: 100m
│     ├─ 缓存容量: 200MB
│     └─ 计算能力: 20 GHz
│
├─ 3) 初始化自适应控制器
│  ├─ AdaptiveCacheController (智能缓存控制)
│  │  ├─ 分层L1/L2缓存策略
│  │  ├─ 热度追踪 (HeatBasedStrategy)
│  │  └─ RSU协作缓存
│  │
│  └─ AdaptiveMigrationController (迁移决策控制)
│     ├─ 负载历史追踪
│     ├─ 多维触发条件
│     └─ 成本效益分析
│
└─ 4) 拓扑优化 (FixedTopologyOptimizer)
   ├─ 根据车辆数优化超参数
   ├─ num_vehicles=12 → hidden_dim=512
   ├─ actor_lr=1e-4, critic_lr=8e-5
   └─ batch_size=256
   
1.3 创建TD3智能体 (TD3Environment)
TD3算法初始化:
├─ 网络结构
│  ├─ Actor网络 (策略网络)
│  │  ├─ 输入: state_dim = 车辆(12×5) + RSU(4×5) + UAV(2×5) + 全局(16) = 106维
│  │  ├─ 隐藏层: 512 → 512 → 256
│  │  └─ 输出: action_dim = 3(任务分配) + 4(RSU选择) + 2(UAV选择) + 8(控制参数) = 17维
│  │
│  ├─ Twin Critic网络 (价值网络×2)
│  │  ├─ Critic1: 评估状态-动作价值
│  │  ├─ Critic2: 减少过估计偏差
│  │  └─ 输入: state(106维) + action(17维) → 输出: Q值
│  │
│  └─ Target网络 (目标网络)
│     ├─ Target Actor: 生成目标动作
│     ├─ Target Critic1 & Critic2: 计算目标Q值
│     └─ 软更新参数: τ=0.005
│
├─ 经验回放缓冲区
│  ├─ 容量: 100,000条经验
│  ├─ 批次大小: 256
│  └─ 优先级经验回放 (PER)
│     ├─ α=0.6 (优先级指数)
│     └─ β=0.4→1.0 (重要性采样)
│
└─ TD3特有机制
   ├─ 策略延迟更新: policy_delay=2 (每2步更新Actor)
   ├─ 目标策略平滑: target_noise=0.05
   ├─ 探索噪声: exploration_noise=0.2 (指数衰减)
   └─ 梯度裁剪: gradient_clip=0.7
   
📌 阶段2: Episode循环 (训练800个episode)
2.1 Episode重置
每个Episode开始时:
├─ 1) 重置仿真器 (system_simulator.py: initialize_components)
│  ├─ 清空所有队列
│  ├─ 重置车辆位置和速度
│  ├─ 清空缓存内容
│  ├─ 重置统计数据
│  └─ 重新生成内容库 (1000个内容)
│
├─ 2) 构建初始状态
│  ├─ 车辆状态 (12×5维)
│  │  ├─ 位置(x,y): 归一化到[0,1]
│  │  ├─ 速度: 归一化到[0,1]
│  │  ├─ 任务队列长度: 归一化
│  │  └─ 能耗: 归一化
│  │
│  ├─ RSU状态 (4×5维)
│  │  ├─ 位置(x,y)
│  │  ├─ 缓存利用率
│  │  ├─ 队列负载
│  │  └─ 能耗
│  │
│  ├─ UAV状态 (2×5维)
│  │  ├─ 位置(x,y,z)
│  │  ├─ 缓存利用率
│  │  └─ 能耗
│  │
│  └─ 全局状态 (16维)
│     ├─ 平均队列长度
│     ├─ 平均缓存利用率
│     ├─ 系统负载
│     ├─ 任务类型分布 (4维)
│     ├─ 任务类型队列占比 (4维)
│     └─ 任务类型截止期 (4维)
│
└─ 3) 重置控制器状态
   ├─ 缓存控制器: 清空热度追踪
   └─ 迁移控制器: 清空负载历史

2.2 时间步循环 (每个Episode约200-300步)
每个时间步的执行流程:

┌─────────────────────────────────────────────────────┐
│  步骤1: TD3选择动作 (td3.py: select_action)        │
├─────────────────────────────────────────────────────┤
│  输入: state (106维向量)                            │
│  │                                                   │
│  ├─ 前向传播通过Actor网络                          │
│  │  └─ 输出原始动作: action_raw (17维)             │
│  │                                                   │
│  ├─ 添加探索噪声 (高斯噪声)                        │
│  │  ├─ noise = N(0, exploration_noise)              │
│  │  └─ action = action_raw + noise                  │
│  │                                                   │
│  ├─ 动作裁剪到[-1, 1]                              │
│  │                                                   │
│  └─ 动作分解 (decompose_action)                    │
│     ├─ 任务分配偏好 [0:3]                          │
│     │  └─ softmax([local, rsu, uav])               │
│     ├─ RSU选择权重 [3:7]                           │
│     │  └─ softmax(4个RSU的权重)                    │
│     ├─ UAV选择权重 [7:9]                           │
│     │  └─ softmax(2个UAV的权重)                    │
│     └─ 控制参数 [9:17]                             │
│        ├─ 缓存控制 (4维)                           │
│        │  ├─ 热度阈值调整                          │
│        │  ├─ 淘汰策略权重                          │
│        │  ├─ 协作强度                              │
│        │  └─ L1/L2比例                             │
│        └─ 迁移控制 (4维)                           │
│           ├─ 负载阈值                              │
│           ├─ 成本敏感度                            │
│           ├─ 延迟权重                              │
│           └─ 能耗权重                              │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  步骤2: 映射动作到自适应控制器                     │
├─────────────────────────────────────────────────────┤
│  (train_single_agent.py: _build_simulator_actions)  │
│  │                                                   │
│  ├─ 解析控制参数 (后8维动作)                       │
│  │                                                   │
│  ├─ 调用 map_agent_actions_to_params()             │
│  │  ├─ 将[-1,1]范围映射到具体参数范围             │
│  │  └─ 分离缓存参数和迁移参数                     │
│  │                                                   │
│  ├─ 更新 AdaptiveCacheController                   │
│  │  ├─ heat_threshold = action[0] * 50 + 50        │
│  │  ├─ eviction_strategy_weight = sigmoid(action[1])│
│  │  ├─ collaboration_strength = action[2] * 0.5 + 0.5│
│  │  └─ l1_l2_ratio = action[3] * 0.3 + 0.4         │
│  │                                                   │
│  └─ 更新 AdaptiveMigrationController               │
│     ├─ load_threshold = action[4] * 0.3 + 0.6      │
│     ├─ cost_sensitivity = action[5] * 0.5 + 0.5    │
│     ├─ delay_weight = action[6] * 0.4 + 0.4        │
│     └─ energy_weight = action[7] * 0.4 + 0.4       │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  步骤3: 仿真器执行一步                              │
├─────────────────────────────────────────────────────┤
│  (system_simulator.py: run_simulation_step)         │
│  │                                                   │
│  ├─ 3.1 更新车辆位置                               │
│  │  ├─ 根据速度和方向移动                         │
│  │  ├─ 处理路口转向                               │
│  │  └─ 添加随机扰动                               │
│  │                                                   │
│  ├─ 3.2 生成任务                                   │
│  │  ├─ 泊松过程采样 (λ=车辆数×任务率)            │
│  │  ├─ 为每辆车生成任务                           │
│  │  │  ├─ 任务类型 (1-4): 根据场景分布           │
│  │  │  ├─ 数据大小: 0.5-2.0 MB                    │
│  │  │  ├─ 计算需求: 500-3000 CPU周期              │
│  │  │  └─ 截止期: 0.5-3.0秒                       │
│  │  └─ 添加到车辆任务队列                         │
│  │                                                   │
│  ├─ 3.3 任务分配与调度                             │
│  │  ├─ 对每个任务决策卸载目标                     │
│  │  │  ├─ 本地处理 (概率: local_pref)             │
│  │  │  ├─ RSU卸载 (概率: rsu_pref)                │
│  │  │  │  └─ 根据RSU选择权重选择具体RSU          │
│  │  │  └─ UAV卸载 (概率: uav_pref)                │
│  │  │     └─ 根据UAV选择权重选择具体UAV          │
│  │  │                                              │
│  │  ├─ 缓存命中检查                               │
│  │  │  └─ check_cache_hit_adaptive()              │
│  │  │     ├─ 检查内容是否在节点缓存中            │
│  │  │     ├─ 命中: 减少传输时延                  │
│  │  │     └─ 未命中: 智能缓存决策                │
│  │  │        ├─ 调用缓存控制器.should_cache_content│
│  │  │        ├─ 基于热度决定是否缓存             │
│  │  │        └─ 执行淘汰和协作缓存               │
│  │  │                                              │
│  │  └─ 任务传输与入队                             │
│  │     ├─ 计算上行传输时延和能耗                 │
│  │     ├─ 将任务加入节点计算队列                 │
│  │     └─ 记录任务元数据                         │
│  │                                                   │
│  ├─ 3.4 处理计算队列                               │
│  │  └─ _process_node_queues()                      │
│  │     ├─ 遍历所有RSU和UAV                        │
│  │     ├─ 对每个节点:                             │
│  │     │  ├─ 获取队列长度                         │
│  │     │  ├─ 动态调整处理能力                     │
│  │     │  │  └─ capacity = base + boost(队列长度) │
│  │     │  ├─ 处理任务工作量                       │
│  │     │  │  └─ work_remaining -= capacity        │
│  │     │  ├─ 完成的任务:                          │
│  │     │  │  ├─ 计算下行传输                      │
│  │     │  │  ├─ 更新统计(延迟、能耗)             │
│  │     │  │  └─ 标记完成                          │
│  │     │  └─ 处理超期任务                         │
│  │     └─ 更新节点状态                             │
│  │                                                   │
│  ├─ 3.5 自适应迁移检查                             │
│  │  └─ check_adaptive_migration()                  │
│  │     ├─ 计算所有节点负载因子                    │
│  │     │  └─ load = 0.8×队列负载 + 0.2×缓存利用率│
│  │     ├─ 更新迁移控制器负载历史                  │
│  │     ├─ 判断是否触发迁移                        │
│  │     │  ├─ 负载超阈值                           │
│  │     │  ├─ 持续时间足够                         │
│  │     │  └─ 成本效益分析通过                     │
│  │     └─ 执行迁移                                │
│  │        ├─ RSU→RSU (有线迁移)                  │
│  │        │  ├─ 选择目标RSU (负载最轻)           │
│  │        │  ├─ 计算迁移成本                      │
│  │        │  ├─ 传输任务                          │
│  │        │  └─ 更新统计                          │
│  │        └─ UAV→RSU (无线迁移)                  │
│  │           └─ 类似流程                          │
│  │                                                   │
│  ├─ 3.6 更新统计指标                               │
│  │  ├─ 累计完成任务数                             │
│  │  ├─ 累计延迟                                   │
│  │  ├─ 累计能耗                                   │
│  │  ├─ 缓存命中率                                 │
│  │  ├─ 迁移成功率                                 │
│  │  └─ 任务类型分布统计                           │
│  │                                                   │
│  └─ 返回 step_stats (本步统计数据)                │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  步骤4: 计算奖励和下一状态                         │
├─────────────────────────────────────────────────────┤
│  (train_single_agent.py: step 方法)                 │
│  │                                                   │
│  ├─ 4.1 提取系统指标                               │
│  │  ├─ 平均延迟: avg_delay (秒)                   │
│  │  ├─ 总能耗: total_energy (焦耳)                │
│  │  ├─ 任务完成率: completion_rate                │
│  │  ├─ 缓存命中率: cache_hit_rate                 │
│  │  ├─ 数据丢失率: data_loss_ratio                │
│  │  └─ 迁移成功率: migration_success_rate         │
│  │                                                   │
│  ├─ 4.2 调用统一奖励计算器                         │
│  │  └─ unified_reward_calculator.calculate_reward()│
│  │     │                                            │
│  │     ├─ 延迟惩罚: -α × log(avg_delay + ε)       │
│  │     │  └─ α=15.0, 强调低延迟                  │
│  │     │                                            │
│  │     ├─ 能耗惩罚: -β × log(total_energy + ε)    │
│  │     │  └─ β=0.01, 平衡能效                    │
│  │     │                                            │
│  │     ├─ 完成率奖励: +γ × completion_rate        │
│  │     │  └─ γ=200.0, 鼓励任务完成               │
│  │     │                                            │
│  │     ├─ 缓存命中奖励: +δ × cache_hit_rate       │
│  │     │  └─ δ=10.0, 鼓励高命中率                │
│  │     │                                            │
│  │     ├─ 数据丢失惩罚: -ε × data_loss_ratio      │
│  │     │  └─ ε=50.0, 避免丢包                    │
│  │     │                                            │
│  │     └─ 迁移成功奖励: +ζ × migration_success    │
│  │        └─ ζ=5.0, 鼓励有效迁移                 │
│  │                                                   │
│  │     最终奖励 = Σ(各项奖励/惩罚)                │
│  │                                                   │
│  ├─ 4.3 构建下一状态向量 (106维)                   │
│  │  └─ 与初始状态相同的结构                       │
│  │                                                   │
│  └─ 4.4 判断Episode是否结束                        │
│     ├─ 达到最大步数 (200-300步)                   │
│     ├─ 系统崩溃 (所有节点过载)                     │
│     └─ 完成率过低 (<20%)                          │
└─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────┐
│  步骤5: TD3学习更新 (td3.py: update)               │
├─────────────────────────────────────────────────────┤
│  │                                                   │
│  ├─ 5.1 存储经验到回放缓冲区                       │
│  │  └─ buffer.add(state, action, reward, next_state, done)│
│  │                                                   │
│  ├─ 5.2 采样批次数据 (batch_size=256)              │
│  │  └─ 使用PER优先级采样                          │
│  │                                                   │
│  ├─ 5.3 计算Critic损失                             │
│  │  ├─ 生成目标动作 (Target Actor)                │
│  │  │  └─ target_action = target_actor(next_state) │
│  │  │     + clipped_noise  # 目标策略平滑        │
│  │  │                                              │
│  │  ├─ 计算目标Q值 (Twin Target Critics)          │
│  │  │  ├─ q1_target = target_critic1(next_state, target_action)│
│  │  │  ├─ q2_target = target_critic2(next_state, target_action)│
│  │  │  └─ target_q = min(q1, q2)  # 减少过估计    │
│  │  │                                              │
│  │  ├─ 计算TD目标                                 │
│  │  │  └─ y = reward + γ × (1-done) × target_q   │
│  │  │                                              │
│  │  ├─ 计算当前Q值                                │
│  │  │  ├─ current_q1 = critic1(state, action)     │
│  │  │  └─ current_q2 = critic2(state, action)     │
│  │  │                                              │
│  │  ├─ Critic损失                                 │
│  │  │  └─ loss = MSE(current_q1, y) + MSE(current_q2, y)│
│  │  │                                              │
│  │  └─ 反向传播更新Critic                         │
│  │     ├─ critic_optimizer.zero_grad()             │
│  │     ├─ loss.backward()                          │
│  │     ├─ 梯度裁剪 (norm=0.7)                     │
│  │     └─ critic_optimizer.step()                  │
│  │                                                   │
│  ├─ 5.4 延迟Actor更新 (每policy_delay=2步)        │
│  │  ├─ 计算Actor损失                              │
│  │  │  ├─ new_action = actor(state)                │
│  │  │  └─ actor_loss = -critic1(state, new_action).mean()│
│  │  │                                              │
│  │  ├─ 反向传播更新Actor                          │
│  │  │  ├─ actor_optimizer.zero_grad()              │
│  │  │  ├─ actor_loss.backward()                    │
│  │  │  ├─ 梯度裁剪                                │
│  │  │  └─ actor_optimizer.step()                   │
│  │  │                                              │
│  │  └─ 软更新目标网络                             │
│  │     ├─ target_actor = τ×actor + (1-τ)×target_actor│
│  │     └─ target_critics = τ×critics + (1-τ)×target_critics│
│  │                                                   │
│  └─ 5.5 更新PER优先级                             │
│     └─ 根据TD误差更新样本优先级                   │
└─────────────────────────────────────────────────────┘

📌 阶段3: Episode结束与统计
Episode结束后:
├─ 记录Episode统计
│  ├─ 总奖励
│  ├─ 平均延迟
│  ├─ 总能耗
│  ├─ 完成率
│  ├─ 缓存命中率
│  └─ 迁移统计
│
├─ 衰减探索噪声
│  └─ exploration_noise *= noise_decay (0.9997)
│
└─ 打印进度信息
   └─ 每50个Episode打印一次详细统计

📌 阶段4: 周期性评估 (每eval_interval=50个episode)
评估流程:
├─ 关闭探索噪声
├─ 运行10个测试Episode
├─ 计算平均性能指标
│  ├─ 平均奖励
│  ├─ 平均延迟
│  ├─ 平均能耗
│  └─ 平均完成率
└─ 保存性能曲线

📌 阶段5: 训练结束与保存 (800个episode完成后)
保存结果:
├─ 1) 模型权重
│  └─ results/models/single_agent/td3/
│     ├─ actor_final.pth
│     ├─ critic1_final.pth
│     ├─ critic2_final.pth
│     └─ target_networks_final.pth
│
├─ 2) 训练数据
│  └─ results/single_agent/td3/training_results_YYYYMMDD_HHMMSS.json
│     ├─ rewards: [...]
│     ├─ delays: [...]
│     ├─ energies: [...]
│     ├─ completion_rates: [...]
│     └─ cache_metrics: {...}
│
└─ 3) 可视化图表
   └─ results/single_agent/td3/training_chart_YYYYMMDD_HHMMSS.png
      ├─ 奖励曲线
      ├─ 延迟曲线
      ├─ 能耗曲线
      └─ 完成率曲线
      
🔑 核心技术亮点
1. Twin Delayed DDPG (TD3)
    双Critic网络减少Q值过估计
    延迟策略更新提高稳定性
    目标策略平滑化减少方差
2. 自适应控制机制
    智能缓存控制：热度追踪 + 分层缓存
    智能迁移控制：多维触发 + 成本效益
3. 统一奖励函数
    多目标优化：延迟、能耗、完成率
    对数惩罚：避免极端值影响
    平衡权重：确保各项指标协调
4. 动态网络拓扑
    车辆移动模型：真实道路场景
    固定RSU/UAV：验证算法有效性
    自适应计算资源分配

"""

