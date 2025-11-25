#!/usr/bin/env python3
"""
TD3 Strategy Training Runner
--------------------------------

This module orchestrates the TD3 ablation / comparison suites. It activates or disables
unloading, resource allocation, and migration so that each component's contribution can be
quantified. The comparison now focuses on the six strategies requested for the paper:
  1. local-only (pure on-board execution)
  2. remote-only (single RSU enforced, no local execution)
  3. offloading-only (layered policy where RSU decides the destination)
  4. resource-only (multi-RSU resource allocation without local processing)
  5. comprehensive-no-migration (full TD3 stack with migration disabled)
  6. comprehensive-migration (your original TD3 pipeline; identical to running
     `python train_single_agent.py --algorithm TD3 --episodes 2000 --num-vehicles 12`)

Example usage:
```bash
python experiments/td3_strategy_suite/run_strategy_training.py \\
    --strategy local-only --episodes 800 --seed 42

for strategy in local-only remote-only offloading-only resource-only comprehensive-no-migration comprehensive-migration; do
    python experiments/td3_strategy_suite/run_strategy_training.py \\
        --strategy $strategy --suite-id ablation_20231029 --episodes 800
done
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# 娣诲姞椤圭洰鏍圭洰褰曞埌Python璺緞
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from train_single_agent import (
    _apply_global_seed_from_env,
    _build_scenario_config,
    SingleAgentTrainingEnvironment,
    train_single_algorithm,
)
from utils.unified_reward_calculator import UnifiedRewardCalculator
from experiments.fallback_baselines import (
    HeuristicPolicy,
    LocalOnlyPolicy,
    RSUOnlyPolicy,
    GreedyPolicy,
    create_baseline_algorithm,
)

StrategyPreset = Dict[str, Any]  # 绛栫暐棰勮閰嶇疆绫诲瀷

# ========== 初始化统一奖励计算器 ==========
# 使用统一奖励计算器确保与训练时的奖励函数一致
_reward_calculator: Optional[UnifiedRewardCalculator] = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    """鑾峰彇鍏ㄥ眬濂栧姳璁＄畻鍣ㄥ疄渚嬶紙寤惰繜鍒濆鍖栵級"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator

# ========== 默认实验参数 ==========
DEFAULT_EPISODES = 1500  # 默认训练轮数（建议≥1500确保TD3充分收敛）
DEFAULT_SEED = 42        # 默认随机种子（保证实验可重复性）

# ========== 绛栫暐鎵ц椤哄簭 ==========
# 鎸夌収澶嶆潅搴﹂€掑鎺掑垪锛氫粠鍗曚竴鍔熻兘鍒板畬鏁寸郴缁?
# 杩欎釜椤哄簭涔熺敤浜庣敓鎴愬姣斿浘琛ㄦ椂鐨勫睍绀洪『搴?
STRATEGY_ORDER = [
    "local-only",
    "remote-only",
    "offloading-only",
    "resource-only",
    "comprehensive-no-migration",
    "comprehensive-migration",
]



def _build_override(
    num_rsus: Optional[int],
    num_uavs: Optional[int],
    allow_local: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    构建策略专用的场景覆盖配置（在默认配置基础上做最小修改）

    【功能】避免与默认命令 `python train_single_agent.py --algorithm TD3`
    出现配置漂移，仅调整与策略严格相关的参数，确保对比公平。

    【参数】
    num_rsus: Optional[int] - RSU 数量（None 表示沿用默认配置）
    num_uavs: Optional[int] - UAV 数量（None 表示沿用默认配置）
    allow_local: Optional[bool] - 是否允许本地处理（None 表示沿用默认值）

    【返回值】Dict[str, Any] - 覆盖字典，仅包含被修改的键

    【设计原则】
    - 继承默认的车辆规模、覆盖半径等基础参数
    - 仅调整 RSU/UAV 数量或本地执行开关
    - 固定拓扑，减少跨策略比较的随机性
    """
    _ = _build_scenario_config()  # 调用以确保配置加载，与默认训练保持同步
    override: Dict[str, Any] = {}

    if num_rsus is not None:
        override["num_rsus"] = num_rsus
    if num_uavs is not None:
        override["num_uavs"] = num_uavs
    if allow_local is not None:
        override["allow_local_processing"] = allow_local

    override["override_topology"] = True
    return override

@dataclass(frozen=True)
class ScenarioProfile:
    """Descriptor for the scenario tweaks applied to a strategy."""

    key: str
    label: str
    num_rsus: Optional[int]
    num_uavs: Optional[int]
    allow_local: Optional[bool]
    extra_overrides: Optional[Dict[str, Any]] = None
    env_options: Optional[Dict[str, Any]] = None


SCENARIO_PROFILES: Dict[str, ScenarioProfile] = {
    "shared_edge": ScenarioProfile(
        key="shared_edge",
        label="Shared scenario: 4 RSU + 2 UAV (local allowed)",
        num_rsus=4,
        num_uavs=2,
        allow_local=True,
    ),
    "baseline_single_rsu": ScenarioProfile(
        key="baseline_single_rsu",
        label="Single RSU baseline (local allowed, no UAV)",
        num_rsus=1,
        num_uavs=0,
        allow_local=True,
    ),
    "baseline_single_rsu_remote": ScenarioProfile(
        key="baseline_single_rsu_remote",
        label="Single RSU baseline (remote enforced, no UAV)",
        num_rsus=1,
        num_uavs=0,
        allow_local=False,
    ),
    "layered_multi_edge": ScenarioProfile(
        key="layered_multi_edge",
        label="Layered multi-edge (4 RSU + 2 UAV, local allowed)",
        num_rsus=4,
        num_uavs=2,
        allow_local=True,
    ),
    "layered_multi_edge_remote": ScenarioProfile(
        key="layered_multi_edge_remote",
        label="Layered multi-edge (remote enforced, no local execution)",
        num_rsus=4,
        num_uavs=2,
        allow_local=False,
    ),
}


def _scenario_override(profile_key: str) -> Optional[Dict[str, Any]]:
    """Convert a scenario profile into the override dict consumed by training."""
    profile = SCENARIO_PROFILES[profile_key]
    if (
        profile.num_rsus is None
        and profile.num_uavs is None
        and profile.allow_local is None
        and not profile.extra_overrides
    ):
        return None
    override = _build_override(
        num_rsus=profile.num_rsus,
        num_uavs=profile.num_uavs,
        allow_local=profile.allow_local,
    )
    if profile.extra_overrides:
        override.update(profile.extra_overrides)
    return override


def _make_preset(
    *,
    description: str,
    scenario_key: str,
    use_enhanced_cache: bool,
    disable_migration: bool,
    enforce_offload_mode: Optional[str],
    algorithm: str = "TD3",
    flags: Optional[Sequence[str]] = None,
    heuristic_name: Optional[str] = None,
    group: str = "baseline",
    central_resource: bool = False,
    env_options: Optional[Dict[str, Any]] = None,
) -> StrategyPreset:
    """Factory keeping strategy definitions concise and consistent."""
    scenario = SCENARIO_PROFILES[scenario_key]
    merged_env_options: Dict[str, Any] = {}
    if scenario.env_options:
        merged_env_options.update(scenario.env_options)
    if env_options:
        merged_env_options.update(env_options)
    preset: StrategyPreset = {
        "description": description,
        "algorithm": algorithm,
        "episodes": DEFAULT_EPISODES,
        "use_enhanced_cache": use_enhanced_cache,
        "disable_migration": disable_migration,
        "enforce_offload_mode": enforce_offload_mode,
        "override_scenario": _scenario_override(scenario_key),
        "scenario_key": scenario.key,
        "scenario_label": scenario.label,
        "flags": list(flags or ()),
        "heuristic_name": heuristic_name,
        "group": group,
        "central_resource": bool(central_resource),
        "env_options": merged_env_options or None,
    }
    return preset


STRATEGY_PRESETS: "OrderedDict[str, StrategyPreset]" = OrderedDict(
    [
        (
            "random",
            _make_preset(
                description="Random baseline",
                scenario_key="layered_multi_edge",
                use_enhanced_cache=False,
                disable_migration=True,
                enforce_offload_mode=None,
                algorithm="heuristic",
                heuristic_name="random",
                flags=("cache_off", "migration_off", "random"),
                group="heuristic",
            ),
        ),
        (
            "round-robin",
            _make_preset(
                description="Round-robin baseline",
                scenario_key="layered_multi_edge",
                use_enhanced_cache=False,
                disable_migration=True,
                enforce_offload_mode=None,
                algorithm="heuristic",
                heuristic_name="round_robin",
                flags=("cache_off", "migration_off", "round_robin"),
                group="heuristic",
            ),
        ),
        (
            "local-only",
            _make_preset(
                description="Local-only baseline",
                scenario_key="layered_multi_edge",  # 保持相同场景以保证对比公平
                use_enhanced_cache=False,
                disable_migration=True,
                enforce_offload_mode=None,  # 🔧 移除强制模式，纯策略决策
                algorithm="heuristic",
                heuristic_name="local_only",
                flags=("cache_off", "migration_off", "local_only"),
                group="baseline",
            ),
        ),
        (
            "remote-only",
            _make_preset(
                description="Remote-only baseline",
                scenario_key="layered_multi_edge",  # 🔧 改为通用场景
                use_enhanced_cache=False,
                disable_migration=True,
                enforce_offload_mode=None,  # 🔧 移除强制模式，由RSUOnlyPolicy实现
                algorithm="heuristic",
                heuristic_name="rsu_only",  # 使用重构后RSUOnlyPolicy
                flags=("cache_off", "migration_off", "edge_only"),
                group="baseline",
            ),
        ),
        (
            "offloading-only",
            _make_preset(
                description="Offloading-only",
                scenario_key="layered_multi_edge",
                use_enhanced_cache=False,
                disable_migration=True,
                enforce_offload_mode=None,
                algorithm="heuristic",
                heuristic_name="greedy",  # 使用重构后GreedyPolicy
                flags=("cache_off", "migration_off", "smart_offload"),
                group="layered",
            ),
        ),
        (
            "resource-only",
            _make_preset(
                description="Resource-only",
                scenario_key="layered_multi_edge",  # 🔧 改为通用场景
                use_enhanced_cache=True,
                disable_migration=True,
                enforce_offload_mode=None,  # 🔧 移除强制模式，RemoteGreedyPolicy会拒绝本地
                algorithm="heuristic",
                heuristic_name="remote_greedy",  # 使用重构后RemoteGreedyPolicy
                flags=("cache_on", "migration_off", "resource_alloc"),
                group="layered",
            ),
        ),
        (
            "comprehensive-no-migration",
            _make_preset(
                description="TD3noMIG",
                scenario_key="layered_multi_edge",
                use_enhanced_cache=True,
                disable_migration=True,
                enforce_offload_mode=None,
                algorithm="OPTIMIZED_TD3",  # 🎯 使用OPTIMIZED_TD3保持与CAMTD3一致
                flags=("cache_on", "migration_off", "multi_edge"),
                group="layered",
            ),
        ),
        (
            "comprehensive-migration",
            _make_preset(
                description="CAMTD3",
                scenario_key="layered_multi_edge",
                use_enhanced_cache=True,
                disable_migration=False,
                enforce_offload_mode=None,
                algorithm="OPTIMIZED_TD3",  # 🎯 修复：使用OPTIMIZED_TD3代替TD3
                flags=("cache_on", "migration_on", "multi_edge"),
                group="layered",
            ),
        ),
    ]
)


class RemoteGreedyPolicy(HeuristicPolicy):
    """Intelligent resource allocation policy for edge nodes.
    
    🎯 设计目标：提供真正的资源分配基线，验证CAMTD3的缓存和迁移优势
    
    📊 对比价值：
    - 时延：中低（边缘计算+负载均衡）
    - 能耗：中等（优化通信+计算）
    - 完成率：中高（智能资源匹配）
    
    🔧 重构要点：
    - 真正的多维资源评估：计算、缓存、带宽、队列
    - 支持RSU资源变化适应（通过状态负载感知）
    - 充分利用缓存状态（use_enhanced_cache=True）
    """

    def __init__(self) -> None:
        super().__init__("RemoteGreedy")
        # 🔧 多维资源权重（体现“资源分配”核心）
        self.queue_weight = 1.8      # 队列负载权重
        self.cache_weight = 1.2      # 缓存命中权重（负利益）
        self.comm_weight = 1.0       # 通信成本权重
        self.energy_weight = 0.7     # 能耗权重

    def select_action(self, state) -> np.ndarray:
        veh, rsu, uav = self._structured_state(state)
        
        # 计算车辆质心位置
        anchor = np.mean(veh[:, :2], axis=0) if veh.size > 0 else np.zeros(2, dtype=np.float32)
        
        candidates = []
        
        # 🔧 重构：评估所有RSU（资源感知）
        if rsu.size > 0 and rsu.ndim == 2:
            for i in range(rsu.shape[0]):
                score = self._evaluate_rsu_resource(rsu[i], anchor)
                candidates.append(('rsu', i, score))
        
        # 🔧 重构：评估所有UAV（资源感知）
        if uav.size > 0 and uav.ndim == 2:
            for i in range(uav.shape[0]):
                score = self._evaluate_uav_resource(uav[i], anchor)
                candidates.append(('uav', i, score))
        
        if not candidates:
            # 无边缘节点，极强拒绝本地（与remote-only语义一致）
            return self._action_from_preference(
                local_score=-5.0, 
                rsu_score=0.0, 
                uav_score=0.0
            )
        
        # 选择资源成本最低的边缘节点
        kind, idx, _ = min(candidates, key=lambda x: x[2])
        
        if kind == 'rsu':
            return self._action_from_preference(
                local_score=-5.0,
                rsu_score=5.0,
                uav_score=-3.0,
                rsu_index=idx,
            )
        else:  # UAV
            return self._action_from_preference(
                local_score=-5.0,
                rsu_score=-3.0,
                uav_score=5.0,
                uav_index=idx,
            )
    
    def _evaluate_rsu_resource(self, rsu_state: np.ndarray, veh_pos: np.ndarray) -> float:
        """🔧 多维度RSU资源评估：队列 + 缓存 + 通信 + 能耗"""
        # 队列负载（列3）
        queue_load = float(rsu_state[3]) if rsu_state.size > 3 else 0.6
        
        # 缓存利用率（列2）- 缓存命中为负成本
        cache_util = float(rsu_state[2]) if rsu_state.size > 2 else 0.5
        cache_benefit = -(1.0 - cache_util)  # 命中越高，成本越低
        
        # 通信成本（基于距离）
        rsu_pos = rsu_state[:2] if rsu_state.size >= 2 else veh_pos
        distance = float(np.linalg.norm(rsu_pos - veh_pos))
        comm_cost = distance / 1000.0
        
        # 能耗状态（列4）
        energy = float(rsu_state[4]) if rsu_state.size > 4 else 0.5
        
        # 🎯 综合资源成本
        total_cost = (
            self.queue_weight * queue_load +
            self.cache_weight * cache_benefit +  # 缓存是负成本
            self.comm_weight * comm_cost +
            self.energy_weight * energy * 0.5
        )
        
        return float(total_cost)
    
    def _evaluate_uav_resource(self, uav_state: np.ndarray, veh_pos: np.ndarray) -> float:
        """🔧 多维度UAV资源评估：队列 + 通信 + 悬停能耗"""
        # 队列负载
        queue_load = float(uav_state[3]) if uav_state.size > 3 else 0.7
        
        # 通信成本（UAV空中信道衰减更快）
        uav_pos = uav_state[:2] if uav_state.size >= 2 else veh_pos
        distance = float(np.linalg.norm(uav_pos - veh_pos))
        comm_cost = distance / 800.0  # UAV通信范围较小
        
        # 悬停能耗（列4）
        energy = float(uav_state[4]) if uav_state.size > 4 else 0.8
        
        # UAV无缓存，能耗权重更高
        total_cost = (
            self.queue_weight * queue_load +
            self.comm_weight * comm_cost * 1.3 +  # 空中通信惩罚
            self.energy_weight * energy * 1.2  # UAV能耗惩罚更高
        )
        
        return float(total_cost)


def _resolve_heuristic_policy(name: Optional[str], seed: int) -> HeuristicPolicy:
    key = (name or "").strip().lower()
    if key in {"random"}:
        from experiments.fallback_baselines import RandomPolicy
        return RandomPolicy(seed=seed)
    if key in {"round_robin", "roundrobin", "round-robin"}:
        from experiments.fallback_baselines import RoundRobinPolicy
        return RoundRobinPolicy()
    if key in {"local_only", "localonly"}:
        return LocalOnlyPolicy()
    if key in {"rsu_only", "remote_only"}:
        return RSUOnlyPolicy()
    if key in {"remote_greedy"}:
        return RemoteGreedyPolicy()
    if key in {"greedy"}:
        return GreedyPolicy()

    policy = create_baseline_algorithm(key or "greedy", seed=seed)
    if not isinstance(policy, HeuristicPolicy):
        raise TypeError(f"Heuristic factory for '{name}' did not return a HeuristicPolicy.")
    return policy


def _run_heuristic_strategy(
    preset: StrategyPreset,
    episodes: int,
    seed: int,
    extra_override: Optional[Dict[str, Any]] = None,
    env_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute deterministic heuristic policies under the shared scenario."""

    controller = _resolve_heuristic_policy(preset.get("heuristic_name"), seed)
    override = dict(preset.get("override_scenario") or {})
    if extra_override:
        override.update(extra_override)
    env_kwargs = dict(env_options or {})
    env = SingleAgentTrainingEnvironment(
        "TD3",
        override_scenario=override,
        use_enhanced_cache=preset["use_enhanced_cache"],
        disable_migration=preset["disable_migration"],
        enforce_offload_mode=preset["enforce_offload_mode"],
        joint_controller=env_kwargs.get("joint_controller", False),
    )
    if hasattr(controller, "update_environment"):
        controller.update_environment(env)

    max_steps = int(config.experiment.max_steps_per_episode)
    delay_records: List[float] = []
    energy_records: List[float] = []
    completion_records: List[float] = []
    cache_records: List[float] = []
    migration_records: List[float] = []
    reward_records: List[float] = []  # 🎯 新增：收集奖励

    for _ in range(episodes):
        state = env.reset_environment()
        if hasattr(controller, "reset"):
            controller.reset()

        last_info: Dict[str, Any] = {}
        episode_reward = 0.0  # 🎯 新增：累积episode奖励
        for _ in range(max_steps):
            action_vec = controller.select_action(state)
            actions_dict = env._build_actions_from_vector(action_vec)
            next_state, reward, done, info = env.step(action_vec, state, actions_dict)
            episode_reward += reward  # 🎯 新增：累积奖励
            state = next_state
            last_info = info
            if done:
                break

        metrics = last_info.get("system_metrics", {})
        delay_records.append(float(metrics.get("avg_task_delay", 0.0)))
        energy_records.append(float(metrics.get("total_energy_consumption", 0.0)))
        completion_records.append(float(metrics.get("task_completion_rate", 0.0)))
        cache_records.append(float(metrics.get("cache_hit_rate", 0.0)))
        migration_records.append(float(metrics.get("migration_success_rate", 0.0)))
        reward_records.append(episode_reward)  # 🎯 新增：记录episode奖励

    episode_metrics = {
        "avg_delay": delay_records,
        "total_energy": energy_records,
        "task_completion_rate": completion_records,
        "cache_hit_rate": cache_records,
        "migration_success_rate": migration_records,
    }
    if hasattr(env, "episode_metrics"):
        env_metrics: Dict[str, Any] = getattr(env, "episode_metrics", {}) or {}

        def _coerce_numeric_series(series: Any) -> List[float]:
            if series is None:
                return []
            if not isinstance(series, list):
                series = [series]
            cleaned: List[float] = []
            for item in series:
                if isinstance(item, (list, tuple)):
                    for sub_item in item:
                        try:
                            cleaned.append(float(sub_item))
                        except (TypeError, ValueError):
                            continue
                    continue
                try:
                    cleaned.append(float(item))
                except (TypeError, ValueError):
                    if isinstance(item, np.ndarray) and item.size == 1:
                        cleaned.append(float(item.item()))
            return cleaned

        for key, values in env_metrics.items():
            if key in episode_metrics:
                continue
            numeric_values = _coerce_numeric_series(values)
            if numeric_values:
                episode_metrics[key] = numeric_values

    return {
        "algorithm": "heuristic",
        "timestamp": datetime.now().isoformat(),
        "episode_metrics": episode_metrics,
        "episode_rewards": reward_records,  # 🎯 新增：返回奖励列表
        "artifacts": {},
    }


def tail_mean(values: Any) -> float:
    """
    璁＄畻搴忓垪鍚庡崐閮ㄥ垎鐨勭ǔ瀹氬潎鍊?    
    銆愬姛鑳姐€?    浣跨敤璁粌鍚庢湡鏁版嵁璁＄畻鎬ц兘鎸囨爣鐨勭ǔ瀹氬潎鍊硷紝閬垮厤鍓嶆湡鎺㈢储闃舵鐨勯珮鏂瑰樊骞叉壈銆?    杩欐槸璇勪及鏀舵暃鍚庢€ц兘鐨勬爣鍑嗘柟娉曘€?    
    銆愬弬鏁般€?    values: Any - 鎬ц兘鎸囨爣搴忓垪锛堝姣忚疆鐨勬椂寤躲€佽兘鑰楃瓑锛?    
    銆愯繑鍥炲€笺€?    float - 鍚庢湡绋冲畾闃舵鐨勫潎鍊?    
    銆愯绠楃瓥鐣ャ€?    - 搴忓垪闀垮害 >= 100: 浣跨敤鍚?0%鏁版嵁锛堝厖鍒嗘敹鏁涳級
    - 搴忓垪闀垮害 >= 50: 浣跨敤鏈€鍚?0杞暟鎹?    - 搴忓垪闀垮害 < 50: 浣跨敤鍏ㄩ儴鏁版嵁锛堝揩閫熸祴璇曟ā寮忥級
    
    銆愯鏂囧搴斻€?    璇勪及鏀舵暃鎬ц兘鏃讹紝閫氬父浣跨敤璁粌鍚庢湡鐨勫钩鍧囧€间綔涓烘渶缁堟€ц兘鎸囨爣
    """
    if not values:
        return 0.0
    seq = list(map(float, values))
    length = len(seq)
    if length >= 100:
        subset = seq[length // 2 :]
    elif length >= 50:
        subset = seq[-30:]
    else:
        subset = seq
    return float(sum(subset) / max(1, len(subset)))


# ⚠️ 已废弃：请使用 strategy_runner.py::compute_cost 代替
# 该函数仅做手动计算，不使用avg_reward，已统一到compute_cost（优先使用-reward）
def compute_raw_cost(delay_mean: float, energy_mean: float, completion_rate: Optional[float] = None) -> float:
    """
    璁＄畻缁熶竴浠ｄ环鍑芥暟鐨勫師濮嬪€?    
    銆愬姛鑳姐€?    浣跨敤缁熶竴濂栧姳璁＄畻鍣ㄨ绠椾唬浠凤紝纭繚涓庤缁冩椂浣跨敤鐨勫鍔卞嚱鏁板畬鍏ㄤ竴鑷淬€?    璇ュ嚱鏁扮敤浜庣瓥鐣ラ棿鐨勫叕骞冲姣斻€?    
    銆愬弬鏁般€?    delay_mean: float - 骞冲潎鏃跺欢锛堢锛?    energy_mean: float - 骞冲潎鑳借€楋紙鐒﹁€筹級
    
    銆愯繑鍥炲€笺€?    float - 褰掍竴鍖栧悗鐨勫姞鏉冧唬浠?    
    銆愯绠楀叕寮忋€?    Raw Cost = 蠅_T 路 (T / T_target) + 蠅_E 路 (E / E_target)
    鍏朵腑锛?    - 蠅_T = 2.0锛堟椂寤舵潈閲嶏級
    - 蠅_E = 1.2锛堣兘鑰楁潈閲嶏級
    - T_target = 0.4s锛堟椂寤剁洰鏍囧€硷紝鐢ㄤ簬褰掍竴鍖栵級
    - E_target = 1200J锛堣兘鑰楃洰鏍囧€硷紝鐢ㄤ簬褰掍竴鍖栵級
    
    銆愯鏂囧搴斻€?    浼樺寲鐩爣锛歮inimize 蠅_T路鏃跺欢 + 蠅_E路鑳借€?    璇ユ寚鏍囪秺灏忥紝绯荤粺鎬ц兘瓒婂ソ
    
    銆愪慨澶嶈鏄庛€?    鉁?淇鍚庯細浣跨敤latency_target鍜宔nergy_target锛屼笌璁粌鏃剁殑濂栧姳璁＄畻瀹屽叏涓€鑷?    鉁?淇鍓嶏細閿欒浣跨敤浜哾elay_normalizer(0.2)鍜宔nergy_normalizer(1000)
    鉁?澶嶇敤缁熶竴妯″潡锛岄伒寰狣RY鍘熷垯
    """
    weight_delay = float(config.rl.reward_weight_delay)      # 蠅_T = 2.0
    weight_energy = float(config.rl.reward_weight_energy)    # 蠅_E = 1.2
    
    # 鉁?淇锛氫娇鐢ㄤ笌璁粌鏃跺畬鍏ㄤ竴鑷寸殑褰掍竴鍖栧洜瀛?
    reward_calc = _get_reward_calculator()
    delay_normalizer = reward_calc.latency_target  # 0.4锛堜笌璁粌涓€鑷达級
    energy_normalizer = reward_calc.energy_target  # 1200.0锛堜笌璁粌涓€鑷达級
    
    
    base_cost = (
        weight_delay * (delay_mean / max(delay_normalizer, 1e-6))
        + weight_energy * (energy_mean / max(energy_normalizer, 1e-6))
    )
    
    if completion_rate is not None and completion_rate > 0:
        import math
        completion_penalty = 1.0 + 0.5 * math.log(1.0 / max(completion_rate, 0.5))
        return base_cost * completion_penalty
    
    return base_cost


def update_summary(
    suite_path: Path,
    strategy: str,
    preset: StrategyPreset,
    result: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Dict[str, str],
    episodes: int,
    seed: int,
) -> None:
    """
    鏇存柊绛栫暐瀹為獙鎽樿JSON鏂囦欢
    
    銆愬姛鑳姐€?    灏嗗崟涓瓥鐣ョ殑璁粌缁撴灉杩藉姞鍒皊uite绾у埆鐨剆ummary.json涓€?    璇ユ枃浠舵眹鎬绘墍鏈夌瓥鐣ョ殑鎬ц兘鎸囨爣锛岀敤浜庡悗缁殑瀵规瘮鍒嗘瀽鍜屽彲瑙嗗寲銆?    
    銆愬弬鏁般€?    suite_path: Path - Suite鏍圭洰褰曡矾寰?    strategy: str - 绛栫暐鍚嶇О锛堝"local-only"锛?    preset: StrategyPreset - 绛栫暐棰勮閰嶇疆
    result: Dict[str, Any] - 璁粌杩斿洖鐨勫畬鏁寸粨鏋?    metrics: Dict[str, float] - 璁＄畻鍚庣殑鎬ц兘鎸囨爣
    artifacts: Dict[str, str] - 鐢熸垚鐨勬枃浠惰矾寰?    episodes: int - 瀹為檯璁粌杞暟
    seed: int - 浣跨敤鐨勯殢鏈虹瀛?    
    銆愯繑鍥炲€笺€?    None锛堢洿鎺ュ啓鍏ユ枃浠讹級
    
    銆恠ummary.json缁撴瀯銆?    {
      "suite_id": "20231029_123456",
      "created_at": "2023-10-29T12:34:56",
      "updated_at": "2023-10-29T13:45:00",
      "strategies": {
        "local-only": {
          "description": "...",
          "metrics": {"delay_mean": 0.15, ...},
          "controls": {...},
          "artifacts": {...}
        },
        ...
      }
    }
    
    銆愪娇鐢ㄥ満鏅€?    - 姣忎釜绛栫暐璁粌瀹屾垚鍚庤皟鐢ㄤ竴娆?    - 鏀寔澧為噺鏇存柊锛堝彲澶氭杩愯涓嶅悓绛栫暐锛?    - 鍚庣画鍙敤浜庣敓鎴愬姣斿浘琛?    """
    summary_path = suite_path / "summary.json"
    
    # ========== 鍔犺浇鎴栧垱寤簊ummary ==========
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "suite_id": suite_path.name,
            "created_at": datetime.now().isoformat(),
            "strategies": {},
        }
    
    # ========== 鏇存柊绛栫暐淇℃伅 ==========
    summary["updated_at"] = datetime.now().isoformat()
    summary["strategies"][strategy] = {
        "description": preset["description"],
        "timestamp": result.get("timestamp"),
        "algorithm": result.get("algorithm"),
        "episodes": episodes,
        "seed": seed,
        "controls": {
            "use_enhanced_cache": preset["use_enhanced_cache"],
            "disable_migration": preset["disable_migration"],
            "enforce_offload_mode": preset["enforce_offload_mode"],
            "scenario_key": preset.get("scenario_key"),
            "scenario_label": preset.get("scenario_label"),
            "flags": preset.get("flags", []),
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }
    
    # ========== 鎸佷箙鍖栦繚瀛?==========
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_artifacts(
    result: Dict[str, Any],
    strategy_dir: Path,
) -> Dict[str, str]:
    """
    澶嶅埗璁粌浜х敓鐨勬牳蹇冩枃浠跺埌绛栫暐涓撳睘鐩綍
    
    銆愬姛鑳姐€?    灏唗rain_single_agent.py鐢熸垚鐨勭粨鏋滄枃浠讹紙JSON/鍥捐〃/鎶ュ憡锛夊鍒跺埌
    绛栫暐涓撳睘鏂囦欢澶癸紝渚夸簬鍚庣画鍒嗘瀽鍜屽綊妗ｃ€?    
    銆愬弬鏁般€?    result: Dict[str, Any] - 璁粌缁撴灉瀛楀吀锛堝寘鍚玜lgorithm銆乼imestamp绛夛級
    strategy_dir: Path - 绛栫暐涓撳睘鐩綍锛堝results/td3_strategy_suite/suite_id/local-only/锛?    
    銆愯繑鍥炲€笺€?    Dict[str, str] - 澶嶅埗鍚庣殑鏂囦欢璺緞瀛楀吀
        {
          "training_json": "path/to/training_results.json",
          "training_chart": "path/to/training_overview.png",
          "training_report": "path/to/training_report.html"
        }
    
    銆愬鍒剁殑鏂囦欢銆?    1. training_results_{timestamp}.json - 瀹屾暣璁粌鏁版嵁
    2. training_overview.png - 璁粌鏇茬嚎鍥捐〃
    3. training_report_{timestamp}.html - 璁粌鎶ュ憡
    
    銆愭簮鏂囦欢浣嶇疆銆?    results/single_agent/{algorithm}/
    
    銆愮洰鏍囦綅缃€?    results/td3_strategy_suite/{suite_id}/{strategy}/
    """
    algorithm = str(result.get("algorithm", "")).lower()
    timestamp = result.get("timestamp")
    artifacts: Dict[str, str] = {}

    # ========== 纭畾婧愭枃浠惰矾寰?==========
    src_root = Path("results") / "single_agent" / algorithm
    if timestamp:
        json_name = f"training_results_{timestamp}.json"
        report_name = f"training_report_{timestamp}.html"
    else:
        json_name = "training_results.json"
        report_name = "training_report.html"
    chart_name = "training_overview.png"

    # ========== 瀹氫箟澶嶅埗娓呭崟 ==========
    copies = [
        ("training_json", src_root / json_name),
        ("training_chart", src_root / chart_name),
        ("training_report", src_root / report_name),
    ]
    
    # ========== 鎵ц澶嶅埗 ==========
    strategy_dir.mkdir(parents=True, exist_ok=True)
    for key, src in copies:
        if src.exists():
            dst = strategy_dir / src.name
            shutil.copy2(src, dst)
            artifacts[key] = str(dst)
    
    return artifacts


def run_strategy(strategy: str, args: argparse.Namespace) -> None:
    """
    鎵ц鍗曚釜绛栫暐鐨勫畬鏁磋缁冩祦绋?    
    銆愬姛鑳姐€?    杩欐槸涓绘墽琛屽嚱鏁帮紝瀹屾垚浠ヤ笅浠诲姟锛?    1. 鍔犺浇绛栫暐閰嶇疆
    2. 璁剧疆闅忔満绉嶅瓙
    3. 璋冪敤train_single_algorithm杩涜璁粌
    4. 璁＄畻绋冲畾鎬ц兘鎸囨爣
    5. 澶嶅埗缁撴灉鏂囦欢
    6. 鏇存柊summary.json
    7. 鎵撳嵃缁撴灉鎽樿
    
    銆愬弬鏁般€?    strategy: str - 绛栫暐鍚嶇О锛堝繀椤诲湪STRATEGY_PRESETS涓畾涔夛級
    args: argparse.Namespace - 鍛戒护琛屽弬鏁?    
    銆愬伐浣滄祦绋嬨€?    姝ラ1: 楠岃瘉绛栫暐鍚嶇О
    姝ラ2: 璁剧疆闅忔満绉嶅瓙锛堜繚璇佸彲閲嶅鎬э級
    姝ラ3: 璋冪敤璁粌鍑芥暟锛堜娇鐢ㄧ瓥鐣ヤ笓灞為厤缃級
    姝ラ4: 浠庤缁冪粨鏋滀腑鎻愬彇鎬ц兘鎸囨爣
    姝ラ5: 璁＄畻绋冲畾鍧囧€硷紙浣跨敤tail_mean锛?    姝ラ6: 澶嶅埗鐢熸垚鐨勬枃浠跺埌绛栫暐鐩綍
    姝ラ7: 鏇存柊姹囨€籎SON
    姝ラ8: 鎵撳嵃缁撴灉
    
    銆愯緭鍑烘枃浠剁粨鏋勩€?    results/td3_strategy_suite/{suite_id}/
    鈹溾攢鈹€ summary.json                    # 姹囨€绘枃浠讹紙鎵€鏈夌瓥鐣ワ級
    鈹溾攢鈹€ local-only/
    鈹?  鈹溾攢鈹€ training_results_*.json
    鈹?  鈹溾攢鈹€ training_overview.png
    鈹?  鈹斺攢鈹€ training_report_*.html
    鈹溾攢鈹€ remote-only/
    鈹?  鈹斺攢鈹€ ...
    鈹斺攢鈹€ ...
    
    銆愭€ц兘鎸囨爣銆?    - delay_mean: 骞冲潎浠诲姟鏃跺欢锛堢锛?    - energy_mean: 骞冲潎鎬昏兘鑰楋紙鐒﹁€筹級
    - completion_mean: 浠诲姟瀹屾垚鐜囷紙0-1锛?    - raw_cost: 缁熶竴浠ｄ环鍑芥暟锛堣秺灏忚秺濂斤級
    """
    # ========== 姝ラ1: 鍔犺浇绛栫暐閰嶇疆 ==========
    if strategy not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {strategy}")
    preset = STRATEGY_PRESETS[strategy]
    scenario_label = preset.get("scenario_label", "Simulator defaults")
    control_flags = ", ".join(preset.get("flags", [])) or "none"

    # ========== 姝ラ2: 纭畾璁粌鍙傛暟 ==========
    episodes = args.episodes or preset["episodes"]
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    # ========== 姝ラ3: 璁剧疆闅忔満绉嶅瓙 ==========
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()

    # ========== 步骤4: 执行策略 ==========
    # TD3 继续调用训练接口，启发式策略走轻量评估
    env_options = dict(preset.get("env_options") or {})
    if preset.get("central_resource"):
        os.environ['CENTRAL_RESOURCE'] = '1'
    else:
        os.environ.pop('CENTRAL_RESOURCE', None)

    algorithm_kind = str(preset["algorithm"]).lower()
    if algorithm_kind == "heuristic":
        silent = True
        results = _run_heuristic_strategy(preset, episodes, seed, env_options=env_options)
    else:
        silent = getattr(args, "silent", True)
        results = train_single_algorithm(
            preset["algorithm"],
            num_episodes=episodes,
            silent_mode=silent,
            override_scenario=preset["override_scenario"],
            use_enhanced_cache=preset["use_enhanced_cache"],
            disable_migration=preset["disable_migration"],
            enforce_offload_mode=preset["enforce_offload_mode"],
            joint_controller=env_options.get("joint_controller", False),
        )

    # ========== 步骤5: 提取性能指标 ==========
    episode_metrics: Dict[str, Any] = results.get("episode_metrics", {})
    delay_mean = tail_mean(episode_metrics.get("avg_delay", []))
    energy_mean = tail_mean(episode_metrics.get("total_energy", []))
    completion_mean = tail_mean(episode_metrics.get("task_completion_rate", []))
    
    # 🎯 修复：优先使用奖励计算成本（与strategy_runner.py一致）
    episode_rewards = results.get("episode_rewards", [])
    avg_reward: Optional[float] = None
    if episode_rewards and len(episode_rewards) > 0:
        # 使用后50%数据（收敛后）
        if len(episode_rewards) >= 100:
            half_point = len(episode_rewards) // 2
            avg_reward = float(np.mean(episode_rewards[half_point:]))
        elif len(episode_rewards) >= 50:
            avg_reward = float(np.mean(episode_rewards[-30:]))
        else:
            avg_reward = float(np.mean(episode_rewards))
    
    # 导入统一的compute_cost函数（自动处理reward优先逻辑）
    from experiments.td3_strategy_suite.strategy_runner import compute_cost
    raw_cost = compute_cost(delay_mean, energy_mean, avg_reward, completion_mean)

    # ========== 姝ラ6: 鍑嗗杈撳嚭鐩綍 ==========
    suite_id = args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_path = Path(args.output_root) / suite_id
    strategy_dir = suite_path / strategy
    suite_path.mkdir(parents=True, exist_ok=True)

    # ========== 姝ラ7: 澶嶅埗缁撴灉鏂囦欢 ==========
    artifacts = copy_artifacts(results, strategy_dir)

    # ========== 姝ラ8: 姹囨€绘€ц兘鎸囨爣 ==========
    metrics = {
        "delay_mean": delay_mean,
        "energy_mean": energy_mean,
        "completion_mean": completion_mean,
        "raw_cost": raw_cost,
    }
    
    # ========== 姝ラ9: 鏇存柊summary.json ==========
    update_summary(suite_path, strategy, preset, results, metrics, artifacts, episodes, seed)

    # ========== 姝ラ10: 鎵撳嵃缁撴灉鎽樿 ==========
    print("\n=== Strategy Run Completed ===")
    print(f"Suite ID        : {suite_id}")
    print(f"Strategy        : {strategy}")
    print(f"Episodes        : {episodes}")
    print(f"Seed            : {seed}")
    print(f"Scenario Profile: {scenario_label}")
    print(f"Toggles         : {control_flags}")
    print(f"Average Delay   : {delay_mean:.4f} s")
    print(f"Average Energy  : {energy_mean:.2f} J")
    print(f"Completion Rate : {completion_mean:.3f}")
    print(f"Raw Cost        : {raw_cost:.4f}")
    if artifacts:
        print("Artifacts:")
        for key, path in artifacts.items():
            print(f"  - {key}: {path}")
    summary_path = suite_path / "summary.json"
    print(f"Summary updated : {summary_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    鏋勫缓鍛戒护琛屽弬鏁拌В鏋愬櫒
    
    銆愬姛鑳姐€?    瀹氫箟鑴氭湰鐨勫懡浠よ鎺ュ彛锛屾敮鎸佺伒娲婚厤缃缁冨弬鏁般€?    
    銆愯繑鍥炲€笺€?    argparse.ArgumentParser - 閰嶇疆濂界殑鍙傛暟瑙ｆ瀽鍣?    
    銆愬懡浠よ鍙傛暟銆?    --strategy: str (蹇呴渶)
        - 绛栫暐鍚嶇О锛屽彲閫夊€? local-only, remote-only, offloading-only, 
          resource-only, comprehensive-no-migration, comprehensive-migration
    
    --episodes: int (鍙€?
        - 璁粌杞暟锛岄粯璁?00
        - 蹇€熸祴璇曞彲鐢?0-100锛屽畬鏁村疄楠屽缓璁?00-1000
    
    --seed: int (鍙€?
        - 闅忔満绉嶅瓙锛岄粯璁?2
        - 鐢ㄤ簬淇濊瘉瀹為獙鍙噸澶嶆€?    
    --suite-id: str (鍙€?
        - Suite鏍囪瘑绗︼紝鐢ㄤ簬灏嗗涓瓥鐣ュ綊涓哄悓涓€缁勫疄楠?        - 鏈寚瀹氭椂鑷姩鐢熸垚鏃堕棿鎴筹紙YYYYMMDD_HHMMSS锛?    
    --output-root: str (鍙€?
        - 杈撳嚭鏍圭洰褰曪紝榛樿"results/td3_strategy_suite"
    
    --silent: bool (鍙€?
        - 闈欓粯妯″紡锛屽噺灏戣缁冭繃绋嬬殑杈撳嚭
        - 鉁?娉ㄦ剰锛氭壒閲忓疄楠岃剼鏈粯璁ゅ凡鍚敤闈欓粯妯″紡锛屾棤闇€鎵嬪姩浜や簰
    
    銆愪娇鐢ㄧず渚嬨€?    # 鉁?榛樿闈欓粯杩愯锛堟棤闇€鎵嬪姩浜や簰锛屾帹鑽愶級
    # 鍩烘湰鐢ㄦ硶
    python run_strategy_training.py --strategy local-only
    
    # 鎸囧畾鍙傛暟 - 鑷姩淇濆瓨鎶ュ憡锛屾棤浜哄€煎畧杩愯
    python run_strategy_training.py --strategy comprehensive-migration \\
        --episodes 1000 --seed 123 --suite-id exp_ablation_v1
    
    # 蹇€熸祴璇曪紙宸查粯璁ら潤榛橈級
    python run_strategy_training.py --strategy offloading-only \\
        --episodes 50
    
    # 馃挕 濡傞渶浜や簰寮忕'璁や繚瀛樻姤鍛婏紝娣诲姞 --interactive 鍙傛暟
    python run_strategy_training.py --strategy td3-full \\
        --episodes 500 --interactive
    """
    parser = argparse.ArgumentParser(
        description="Run TD3 under a specific strategy baseline and collect results."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGY_PRESETS.keys()),
        help="Select which strategy preset to train.",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        help="Override number of training episodes (default 800)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed (default 42)."
    )
    parser.add_argument(
        "--suite-id", 
        type=str, 
        help="Suite identifier to group multiple runs."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/td3_strategy_suite",
        help="Root folder where per-strategy results will be stored.",
    )
    parser.add_argument(
        "--silent", 
        action="store_true", 
        help="Run training in silent mode."
    )
    return parser


def main() -> None:
    """
    鑴氭湰涓诲叆鍙ｅ嚱鏁?    
    銆愬姛鑳姐€?    瑙ｆ瀽鍛戒护琛屽弬鏁板苟鍚姩绛栫暐璁粌娴佺▼銆?    
    銆愭墽琛屾祦绋嬨€?    1. 鏋勫缓鍙傛暟瑙ｆ瀽鍣?    2. 瑙ｆ瀽鍛戒护琛屽弬鏁?    3. 璋冪敤run_strategy鎵ц璁粌
    
    銆愰敊璇鐞嗐€?    - 鏈煡绛栫暐鍚嶇О锛歏alueError
    - 鍙傛暟缂哄け锛歛rgparse鑷姩鎻愮ず
    - 璁粌杩囩▼閿欒锛氱敱train_single_algorithm澶勭悊
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    run_strategy(args.strategy, args)


if __name__ == "__main__":
    main()
