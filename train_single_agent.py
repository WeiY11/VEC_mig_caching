"""


平滑时延能耗”和“注意力权衡距离与缓存”的优化
python train_single_agent.py --algorithm TD3 --episodes 800 --num-vehicles 12 --two-stage
🐍🖥️📚
cd offloading_strategy_comparison
# 1. 测试（1分钟）
python test_offloading_strategies.py
# 2. 快速实验（10分钟）
python run_offloading_comparison.py --mode vehicle --episodes 5
# 3. 完整实验（3-4小时，论文用）
python run_offloading_comparison.py --mode all --episodes 50
# 4. 生成图表
python visualize_offloading_comparison.py --results all_experiments_*.json --mode all

单智能体算法训练脚本
支持DDPG、TD3、TD3-LE、DQN、PPO、SAC等算法的训练和比较
使用方法:
python train_single_agent.py --algorithm TD3 --episodes 200
python train_single_agent.py --algorithm TD3 --episodes 200 --seed 123 --num-vehicles 16
python train_single_agent.py --algorithm DDPG --episodes 200
python train_single_agent.py --algorithm PPO --episodes 150 --seed 3407
python train_single_agent.py --algorithm TD3-LE --episodes 200  # 延时-能耗协同优化
python train_single_agent.py --compare --episodes 200  # 比较所有算法
🚀 增强缓存模式 (默认启用 - 分层L1/L2 + 自适应热度策略 + RSU协作):
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 8
python train_single_agent.py --algorithm TD3 --episodes 800 --num-vehicles 12
python train_single_agent.py --algorithm TD3 --episodes 800 --num-vehicles 12 --silent-mode  # 静默保存结果
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 16
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 20
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 24
python train_single_agent.py --algorithm TD3-LE --episodes 1600 --num-vehicles 12
python train_single_agent.py --algorithm SAC --episodes 800
python train_single_agent.py --algorithm PPO --episodes 800
🔧 禁用增强缓存 (如需baseline对比):
python train_single_agent.py --algorithm TD3 --episodes 1600 --num-vehicles 20 --no-enhanced-cache

🌐 实时可视化:
python train_single_agent.py --algorithm TD3 --episodes 200 --realtime-vis
python train_single_agent.py --algorithm DDPG --episodes 100 --realtime-vis --vis-port 8080

📊 批量实验脚本:
python experiments/run_td3_seed_sweep.py --seeds 42 2025 3407 --episodes 200
python experiments/run_td3_vehicle_sweep.py --vehicles 8 12 16 --episodes 200
python experiments/run_td3_vehicle_sweep.py --vehicles 8 12 16 20 24 --episodes 800
🐍 生成学术图表:
python generate_academic_charts.py results/single_agent/td3/training_results_20251007_220900.json


train_single_agent.py (主入口)
    ↓
├─ 解析参数: --algorithm TD3, --episodes 800, --num-vehicles 12
├─ 加载配置: config/
├─ 初始化环境: TD3Environment (single_agent/td3.py)
│   ├─ 创建仿真器: CompleteSystemSimulator (evaluation/system_simulator.py)
│   ├─ 初始化拓扑: FixedTopologyOptimizer
│   ├─ 设置车辆数: 12辆
│   └─ 创建控制器: AdaptiveCacheController, AdaptiveMigrationController
├─ 训练循环 (800轮):
│   ├─ 重置环境 → system_simulator.initialize_components()
│   ├─ 每一步:
│   │   ├─ TD3选择动作
│   │   ├─ 环境执行: system_simulator.run_simulation_step()
│   │   ├─ 计算奖励: unified_reward_calculator.calculate_reward()
│   │   └─ TD3学习更新
│   └─ 评估性能
└─ 保存结果:
    ├─ 模型: results/models/single_agent/td3/
    ├─ 训练数据: results/single_agent/td3/training_results_*.json
    └─ 图表: results/single_agent/td3/training_chart_*.png
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
# 🤖 导入自适应控制组件
from utils.adaptive_control import AdaptiveCacheController, AdaptiveMigrationController, map_agent_actions_to_params

# 导入各种单智能体算法
from single_agent.ddpg import DDPGEnvironment
from single_agent.td3 import TD3Environment
from single_agent.td3_hybrid_fusion import CAMTD3Environment
from single_agent.td3_latency_energy import TD3LatencyEnergyEnvironment
from single_agent.dqn import DQNEnvironment
from single_agent.ppo import PPOEnvironment
from single_agent.sac import SACEnvironment

# 导入HTML报告生成器
from utils.html_report_generator import HTMLReportGenerator

# 🌐 导入实时可视化模块
try:
    from scripts.visualize.realtime_visualization import create_visualizer
    REALTIME_AVAILABLE = True
except ImportError:
    REALTIME_AVAILABLE = False
    print("⚠️  实时可视化功能不可用，请运行: pip install flask flask-socketio")

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
    scenario = {
        "num_vehicles": getattr(config, "num_vehicles", 12),
        "num_rsus": getattr(config, "num_rsus", 4),
        "num_uavs": getattr(config, "num_uavs", 2),
        "task_arrival_rate": getattr(getattr(config, "task", None), "arrival_rate", 1.8),
        "time_slot": getattr(config, "time_slot", 0.2),
        "simulation_time": getattr(config, "simulation_time", 1000),
        "computation_capacity": 800,
        "bandwidth": 15,
        "coverage_radius": 300,
        "cache_capacity": 80,
        "transmission_power": 0.15,
        "computation_power": 1.2,
        "thermal_noise_density": -174.0,
        "noise_figure": 9.0,
        "high_load_mode": True,
        "task_complexity_multiplier": 1.5,
        "rsu_load_divisor": 4.0,
        "uav_load_divisor": 2.0,
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
    
    def __init__(self, algorithm: str, override_scenario: Optional[Dict[str, Any]] = None, 
                 use_enhanced_cache: bool = False, disable_migration: bool = False,
                 enforce_offload_mode: Optional[str] = None):
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
        scenario_config = _build_scenario_config()
        # 应用外部覆盖
        if override_scenario:
            scenario_config.update(override_scenario)
            scenario_config['override_topology'] = True
        
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
        
        # 从仿真器获取实际网络拓扑参数
        num_vehicles = len(self.simulator.vehicles)
        num_rsus = len(self.simulator.rsus)
        num_uavs = len(self.simulator.uavs)
        
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
            self.agent_env = TD3Environment(num_vehicles, num_rsus, num_uavs)
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
        else:
            raise ValueError(f"不支持的算法: {algorithm}")

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
            'rsu_hotspot_peak_series': []
        }
        
        # 性能追踪器
        self.performance_tracker = {
            'recent_rewards': MovingAverage(100),
            'recent_step_rewards': MovingAverage(100),
            'recent_delays': MovingAverage(100),
            'recent_energy': MovingAverage(100),
            'recent_completion': MovingAverage(100)
        }
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
        self._episode_counters_initialized = True

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
                np.clip(vehicle['position'][0] / 1000, 0.0, 1.0),
                np.clip(vehicle['position'][1] / 1000, 0.0, 1.0),
                np.clip(vehicle['velocity'] / 50, 0.0, 1.0),
                np.clip(len(vehicle.get('tasks', [])) / 20.0, 0.0, 1.0),
                np.clip(vehicle.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)
            ])
            node_states[f'vehicle_{i}'] = vehicle_state

        # RSU状态（统一归一化/裁剪）
        for i, rsu in enumerate(self.simulator.rsus):
            rsu_state = np.array([
                np.clip(rsu['position'][0] / 1000, 0.0, 1.0),
                np.clip(rsu['position'][1] / 1000, 0.0, 1.0),
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),
                np.clip(len(rsu.get('computation_queue', [])) / 20.0, 0.0, 1.0),
                np.clip(rsu.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)
            ])
            node_states[f'rsu_{i}'] = rsu_state

        # UAV状态（统一归一化/裁剪）
        for i, uav in enumerate(self.simulator.uavs):
            uav_state = np.array([
                np.clip(uav['position'][0] / 1000, 0.0, 1.0),
                np.clip(uav['position'][1] / 1000, 0.0, 1.0),
                np.clip(uav['position'][2] / 200, 0.0, 1.0),
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),
                np.clip(uav.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)
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

        self._initialize_episode_counters(getattr(self.simulator, 'stats', None))

        state = self.agent_env.get_state_vector(node_states, system_metrics)

        return state

    def step(self, action, state, actions_dict: Optional[Dict] = None) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步仿真，应用智能体动作到仿真器"""
        # 构造传递给仿真器的动作（将连续动作映射为本地/RSU/UAV偏好）
        sim_actions = self._build_simulator_actions(actions_dict)
        
        # 执行仿真步骤（传入动作）
        step_stats = self.simulator.run_simulation_step(0, sim_actions)
        
        # 收集下一步状态
        node_states = {}
        
        # 车辆状态 (5维 - 统一归一化)
        for i, vehicle in enumerate(self.simulator.vehicles):
            vehicle_state = np.array([
                np.clip(vehicle['position'][0] / 1000, 0.0, 1.0),  # 位置x
                np.clip(vehicle['position'][1] / 1000, 0.0, 1.0),  # 位置y
                np.clip(vehicle['velocity'] / 50, 0.0, 1.0),  # 速度
                np.clip(len(vehicle.get('tasks', [])) / 20.0, 0.0, 1.0),  # 队列（扩大范围到20）
                np.clip(vehicle.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)  # 能耗
            ])
            node_states[f'vehicle_{i}'] = vehicle_state
        
        # RSU状态 (5维 - 清理版，移除控制参数)
        for i, rsu in enumerate(self.simulator.rsus):
            # 标准化归一化：确保所有值在[0,1]范围
            rsu_state = np.array([
                np.clip(rsu['position'][0] / 1000, 0.0, 1.0),  # 位置x
                np.clip(rsu['position'][1] / 1000, 0.0, 1.0),  # 位置y
                self._calculate_correct_cache_utilization(rsu.get('cache', {}), rsu.get('cache_capacity', 1000.0)),  # 缓存利用率
                np.clip(len(rsu.get('computation_queue', [])) / 20.0, 0.0, 1.0),  # 队列利用率（扩大范围到20）
                np.clip(rsu.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)  # 能耗
            ])
            node_states[f'rsu_{i}'] = rsu_state
        
        # UAV状态 (5维 - 清理版，移除控制参数)
        for i, uav in enumerate(self.simulator.uavs):
            # 标准化归一化：确保所有值在[0,1]范围
            uav_state = np.array([
                np.clip(uav['position'][0] / 1000, 0.0, 1.0),  # 位置x
                np.clip(uav['position'][1] / 1000, 0.0, 1.0),  # 位置y
                np.clip(uav['position'][2] / 200, 0.0, 1.0),   # 位置z（高度）
                self._calculate_correct_cache_utilization(uav.get('cache', {}), uav.get('cache_capacity', 200.0)),  # 缓存利用率
                np.clip(uav.get('energy_consumed', 0) / 1000.0, 0.0, 1.0)  # 能耗
            ])
            node_states[f'uav_{i}'] = uav_state
        
        # 计算系统指标
        system_metrics = self._calculate_system_metrics(step_stats)
        
        # 获取下一状态
        next_state = self.agent_env.get_state_vector(node_states, system_metrics)
        
        # 🔧 增强：计算包含子系统指标的奖励
        cache_metrics = self.adaptive_cache_controller.get_cache_metrics()
        migration_metrics = self.adaptive_migration_controller.get_migration_metrics()
        
        reward = self.agent_env.calculate_reward(system_metrics, cache_metrics, migration_metrics)
        
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
        completion_rate = episode_processed / max(1, episode_total) if episode_total > 0 else 0.5
        
        cache_hits = int(safe_get('cache_hits', 0))
        cache_misses = int(safe_get('cache_misses', 0))
        cache_requests = max(1, cache_hits + cache_misses)
        cache_hit_rate = cache_hits / cache_requests
        local_cache_hits = int(safe_get('local_cache_hits', 0))
        
        # 🔧 修复：安全计算平均延迟 - 使用累计统计
        total_delay = safe_get('total_delay', 0.0)
        processed_for_delay = max(1, total_processed)  # 使用累计完成数
        avg_delay = total_delay / processed_for_delay
        
        # 限制延迟在合理范围内（关键修复）
        avg_delay = np.clip(avg_delay, 0.01, 5.0)  # 扩大到0.01-5.0秒范围，适应跨时隙处理
        
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

        
        # 计算本episode增量能耗（防止负值与异常）
        if current_total_energy <= 0.0:
            # 仿真器能耗异常时的保底估算
            completed_tasks = self.simulator.stats.get('completed_tasks', 0) if hasattr(self, 'simulator') else 0
            estimated_energy = max(0.0, completed_tasks * 15.0)
            total_energy = estimated_energy
            print(f"⚠️ 仿真器能耗为0，使用估算能耗: {total_energy:.1f}J")
        else:
            episode_incremental_energy = max(0.0, current_total_energy - getattr(self, '_episode_energy_base', 0.0))
            total_energy = episode_incremental_energy
        
        # 🔧 修复：使用episode级别数据丢失量，避免累积效应
        data_loss_bytes = max(0.0, episode_dropped_bytes)
        data_generated_bytes = max(1.0, episode_generated_bytes)
        data_loss_ratio_bytes = min(1.0, data_loss_bytes / data_generated_bytes) if data_generated_bytes > 0 else 0.0
        
        # 迁移成功率（来自仿真器统计）
        migrations_executed = int(safe_get('migrations_executed', 0))
        migrations_successful = int(safe_get('migrations_successful', 0))
        migration_success_rate = (migrations_successful / migrations_executed) if migrations_executed > 0 else 0.0
        
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
        cache_eviction_rate = (
            episode_cache_evictions / episode_cache_requests
            if episode_cache_requests > 0 else 0.0
        )

        def _normalize_vector(key: str, length: int = 4, clip: bool = True) -> List[float]:
            raw = step_stats.get(key)
            if isinstance(raw, np.ndarray):
                values = raw.tolist()
            elif isinstance(raw, (list, tuple)):
                values = list(raw)
            else:
                values = []
            values = [float(v) for v in values[:length]]
            if len(values) < length:
                values.extend([0.0] * (length - len(values)))
            if clip:
                values = [float(np.clip(v, 0.0, 1.0)) for v in values]
            else:
                values = [float(max(0.0, v)) for v in values]
            return values

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

        total_generated_by_type = sum(float(gen_by_type.get(t, 0.0)) for t in range(1, 5))
        generated_share: List[float] = []
        drop_rate: List[float] = []
        for task_type in range(1, 5):
            generated = float(gen_by_type.get(task_type, 0.0))
            dropped = float(drop_by_type.get(task_type, 0.0))
            drop_rate.append(float(np.clip(dropped / generated, 0.0, 1.0)) if generated > 0.0 else 0.0)
            generated_share.append(
                float(np.clip(generated / total_generated_by_type, 0.0, 1.0)) if total_generated_by_type > 0.0 else 0.0
            )

        # 🔍 调试日志：能耗与迁移敏感区间
        current_episode = getattr(self, '_current_episode', 0)
        if current_episode > 0 and (current_episode % 50 == 0 or avg_delay > 0.2 or migration_success_rate < 0.9):
            print(
                f"[调试] Episode {current_episode:04d}: 延迟 {avg_delay:.3f}s, 能耗 {total_energy:.2f}J, "
                f"完成率 {completion_rate:.1%}, 迁移成功率 {migration_success_rate:.1%}, "
                f"缓存命中 {cache_hit_rate:.1%}"
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
            'rsu_hotspot_intensity_list': hotspot_list,
            'rsu_hotspot_mean': rsu_hotspot_mean,
            'rsu_hotspot_peak': rsu_hotspot_peak
        }
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None) -> Dict:
        """运行一个完整的训练轮次"""
        # 使用配置中的最大步数
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 重置环境
        self._episode_counters_initialized = False
        state = self.reset_environment()
        
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
            return self._run_ppo_episode(episode, max_steps)
        
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
            
            # 执行动作（将动作字典传入以影响仿真器卸载偏好）
            next_state, reward, done, info = self.step(action, state, actions_dict)
            
            # 初始化training_info
            training_info = {}
            
            # 训练智能体 - 所有算法现在都支持Union类型统一接口
            # 确保action类型安全转换
            if self.algorithm == "DQN":
                # DQN首选整数动作，但接受Union类型
                safe_action = self._safe_int_conversion(action)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm in ["DDPG", "TD3", "TD3_LATENCY_ENERGY", "SAC"]:
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
            
            episode_info = training_info
            
            # 更新状态
            state = next_state
            episode_reward += reward
            
            # 检查是否结束
            if done:
                break
        
        # 记录轮次统计
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': step + 1
        }
    
    def _run_ppo_episode(self, episode: int, max_steps: int = 100) -> Dict:
        """运行PPO专用episode"""
        state = self.reset_environment()
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
        
        system_metrics = info.get('system_metrics', {})
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': episode_reward,
            'episode_info': training_info,
            'system_metrics': system_metrics,
            'steps': step + 1
        }

    def _build_simulator_actions(self, actions_dict: Optional[Dict]) -> Optional[Dict]:
        """将算法动作字典转换为仿真器可消费的简单控制信号。
        🤖 扩展支持18维动作空间：
        - vehicle_agent 前11维 → 原有任务分配和节点选择
        - vehicle_agent 后8维 → 缓存迁移参数控制
        """
        if not isinstance(actions_dict, dict):
            return None
        vehicle_action = actions_dict.get('vehicle_agent')
        if vehicle_action is None:
            return None
        try:
            import numpy as np
            
            # =============== 原有11维动作逻辑 (保持兼容) ===============
            # 取前三维，映射到[0,1]并softmax为概率
            raw = np.array(vehicle_action[:3], dtype=np.float32).reshape(-1)
            # 数值安全
            raw = np.clip(raw, -5.0, 5.0)
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
            num_rsus = len(getattr(self.simulator, 'rsus', []))
            rsu_action = actions_dict.get('rsu_agent')
            if isinstance(rsu_action, (list, tuple, np.ndarray)) and num_rsus > 0:
                rsu_raw = np.array(rsu_action[:num_rsus], dtype=np.float32)
                rsu_raw = np.clip(rsu_raw, -5.0, 5.0)
                rsu_exp = np.exp(rsu_raw - np.max(rsu_raw))
                rsu_probs = rsu_exp / np.sum(rsu_exp)
                sim_actions['rsu_selection_probs'] = [float(x) for x in rsu_probs]
            # UAV选择概率
            num_uavs = len(getattr(self.simulator, 'uavs', []))
            uav_action = actions_dict.get('uav_agent')
            if isinstance(uav_action, (list, tuple, np.ndarray)) and num_uavs > 0:
                uav_raw = np.array(uav_action[:num_uavs], dtype=np.float32)
                uav_raw = np.clip(uav_raw, -5.0, 5.0)
                uav_exp = np.exp(uav_raw - np.max(uav_raw))
                uav_probs = uav_exp / np.sum(uav_exp)
                sim_actions['uav_selection_probs'] = [float(x) for x in uav_probs]
            
            # 🤖 =============== 新增7维缓存迁移控制 ===============
            if isinstance(vehicle_action, (list, tuple, np.ndarray)):
                vehicle_action_array = np.array(vehicle_action, dtype=np.float32)
                control_start = 3 + num_rsus + num_uavs
                control_end = control_start + 8
                if vehicle_action_array.size >= control_end:
                    cache_migration_actions = vehicle_action_array[control_start:control_end]
                elif vehicle_action_array.size > control_start:
                    # 若长度不足7维，做安全补零
                    cache_migration_actions = np.zeros(8, dtype=np.float32)
                    available = vehicle_action_array[control_start:]
                    cache_migration_actions[:min(available.size, 8)] = available[:8]
                else:
                    cache_migration_actions = np.zeros(8, dtype=np.float32)

                cache_migration_actions = np.clip(cache_migration_actions, -1.0, 1.0)

                # 映射为参数字典
                cache_params, migration_params = map_agent_actions_to_params(cache_migration_actions)

                # 更新自适应控制器参数
                self.adaptive_cache_controller.update_agent_params(cache_params)
                if not self.disable_migration:
                    self.adaptive_migration_controller.update_agent_params(migration_params)

                # ������Ӧ�������ݸ�������
                payload = {
                    'adaptive_cache_params': cache_params,
                    'cache_controller': self.adaptive_cache_controller,
                }
                if not self.disable_migration:
                    payload.update({
                        'adaptive_migration_params': migration_params,
                        'migration_controller': self.adaptive_migration_controller
                    })
                sim_actions.update(payload)

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
                          enforce_offload_mode: Optional[str] = None) -> Dict:
    """训练单个算法
    
    Args:
        algorithm: 算法名称
        num_episodes: 训练轮次
        eval_interval: 评估间隔
        save_interval: 保存间隔
        enable_realtime_vis: 是否启用实时可视化
        vis_port: 可视化服务器端口
        silent_mode: 静默模式，跳过用户交互（用于批量实验）
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
    print("=" * 60)
    
    # 创建训练环境（应用额外场景覆盖）
    training_env = SingleAgentTrainingEnvironment(
        algorithm,
        override_scenario=override_scenario,
        use_enhanced_cache=use_enhanced_cache,
        disable_migration=disable_migration,
        enforce_offload_mode=enforce_offload_mode
    )
    canonical_algorithm = training_env.algorithm
    if canonical_algorithm != algorithm:
        print(f"⚙️  规范化算法标识: {canonical_algorithm}")
    algorithm = canonical_algorithm
    
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
    print("-" * 60)
    
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
        episode_result = training_env.run_episode(episode)
        
        # 记录训练数据
        training_env.episode_rewards.append(episode_result['avg_reward'])
        
        # 🔧 新增：记录实际步数
        episode_steps = episode_result.get('steps', config.experiment.max_steps_per_episode)
        training_env.episode_metrics['episode_steps'].append(episode_steps)
        
        # 更新性能追踪器
        training_env.performance_tracker['recent_rewards'].update(episode_result['avg_reward'])
        per_step_reward = episode_result['avg_reward'] / max(1, episode_steps)
        training_env.performance_tracker['recent_step_rewards'].update(per_step_reward)
        
        system_metrics = episode_result['system_metrics']
        training_env.performance_tracker['recent_delays'].update(system_metrics.get('avg_task_delay', 0))
        training_env.performance_tracker['recent_energy'].update(system_metrics.get('total_energy_consumption', 0))
        training_env.performance_tracker['recent_completion'].update(system_metrics.get('task_completion_rate', 0))
        
        # 🔧 修复：记录指标 - 解决键名不匹配问题
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
            'rsu_hotspot_peak': 'rsu_hotspot_peak'
        }
        
        for system_key, episode_key in metric_mapping.items():
            if system_key in system_metrics and episode_key in training_env.episode_metrics:
                training_env.episode_metrics[episode_key].append(system_metrics[system_key])
                # print(f"✅ 记录指标 {episode_key}: {system_metrics[system_key]:.3f}")  # 调试信息（减少输出）
        
        queue_distribution_ep = system_metrics.get('task_type_queue_distribution')
        if isinstance(queue_distribution_ep, (list, tuple)):
            for idx, value in enumerate(queue_distribution_ep):
                key = f'task_type_queue_share_ep_{idx+1}'
                if key in training_env.episode_metrics:
                    training_env.episode_metrics[key].append(float(value))
        # 🌐 更新实时可视化
        if visualizer:
            vis_metrics = {
                'avg_delay': system_metrics.get('avg_task_delay', 0),
                'total_energy': system_metrics.get('total_energy_consumption', 0),
                'task_completion_rate': system_metrics.get('task_completion_rate', 0),
                'cache_hit_rate': system_metrics.get('cache_hit_rate', 0),
                'data_loss_ratio_bytes': system_metrics.get('data_loss_ratio_bytes', 0),
                'migration_success_rate': system_metrics.get('migration_success_rate', 0)
            }
            visualizer.update(episode, episode_result['avg_reward'], vis_metrics)
        
        episode_time = time.time() - episode_start_time
        
        # 定期输出进度
        if episode % 10 == 0:
            avg_reward_step = training_env.performance_tracker['recent_step_rewards'].get_average()
            avg_delay = training_env.performance_tracker['recent_delays'].get_average()
            avg_completion = training_env.performance_tracker['recent_completion'].get_average()
            
            print(f"轮次 {episode:4d}/{num_episodes}:")
            print(f"  平均每步奖励: {avg_reward_step:8.3f}")
            print(f"  平均时延: {avg_delay:8.3f}s")
            print(f"  完成率:   {avg_completion:8.1%}")
            print(f"  轮次用时: {episode_time:6.3f}s")
        
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
                training_env.agent_env.save_models(f"results/models/single_agent/{algorithm.lower()}/best_model")
                print(f"  💾 保存最佳模型 (Per-Step奖励: {best_avg_reward:.3f})")
        
        # 定期保存模型
        if episode % save_interval == 0:
            training_env.agent_env.save_models(f"results/models/single_agent/{algorithm.lower()}/checkpoint_{episode}")
            print(f"💾 保存检查点: checkpoint_{episode}")
    
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
        if silent_mode:
            # 静默模式：自动保存，不打开浏览器
            if report_generator.save_report(html_content, report_path):
                print(f"✅ 报告已自动保存到: {report_path}")
            else:
                print("❌ 报告保存失败")
        else:
            # 交互模式：询问用户
            print("\n" + "-" * 60)
            save_choice = input("💾 是否保存HTML训练报告? (y/n, 默认y): ").strip().lower()
            
            if save_choice in ['', 'y', 'yes', '是']:
                if report_generator.save_report(html_content, report_path):
                    print(f"✅ 报告已保存到: {report_path}")
                    print(f"💡 提示: 使用浏览器打开该文件即可查看完整报告")
                    
                    # 尝试自动打开报告（可选）
                    auto_open = input("🌐 是否在浏览器中打开报告? (y/n, 默认n): ").strip().lower()
                    if auto_open in ['y', 'yes', '是']:
                        import webbrowser
                        abs_path = os.path.abspath(report_path)
                        webbrowser.open(f'file://{abs_path}')
                        print("✅ 报告已在浏览器中打开")
                else:
                    print("❌ 报告保存失败")
            else:
                print("ℹ️ 报告未保存")
                print(f"💡 如需查看，请手动运行报告生成功能")
    
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
    
    for _ in range(num_eval_episodes):
        state = training_env.reset_environment()
        episode_reward = 0.0
        episode_delay = 0.0
        episode_completion = 0.0
        steps = 0
        
        for step in range(50):  # 较短的评估轮次
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
    parser.add_argument('--algorithm', type=str, choices=['DDPG', 'TD3', 'TD3-LE', 'TD3_LE', 'TD3_LATENCY_ENERGY', 'DQN', 'PPO', 'SAC'],
                       help='选择训练算法')
    parser.add_argument('--episodes', type=int, default=None, help=f'训练轮次 (默认: {config.experiment.num_episodes})')
    parser.add_argument('--eval_interval', type=int, default=None, help=f'评估间隔 (默认: {config.experiment.eval_interval})')
    parser.add_argument('--save_interval', type=int, default=None, help=f'保存间隔 (默认: {config.experiment.save_interval})')
    parser.add_argument('--compare', action='store_true', help='比较所有算法')
    parser.add_argument('--seed', type=int, default=None, help='覆盖随机种子 (默认读取config或环境变量)')
    parser.add_argument('--num-vehicles', type=int, default=None, help='覆盖车辆数量用于实验')
    parser.add_argument('--force-offload', type=str, choices=['local', 'remote', 'local_only', 'remote_only'],
                        help='强制卸载模式：local/local_only 或 remote/remote_only')
    # 🌐 实时可视化参数
    parser.add_argument('--realtime-vis', action='store_true', help='启用实时可视化')
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
                        help='阶段二算法（缓存/迁移控制的RL）：TD3|SAC|DDPG|PPO|DQN|TD3-LE')
    parser.add_argument('--silent-mode', action='store_true',
                        help='启用静默模式，跳过训练结束后的交互提示')
    
    args = parser.parse_args()

    if args.seed is not None:
        os.environ['RANDOM_SEED'] = str(args.seed)
        _apply_global_seed_from_env()

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
            silent_mode=args.silent_mode
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
