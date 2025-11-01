#!/usr/bin/env python3
"""
CAMTD3 Strategy Training Runner
--------------------------------

【功能】
CAMTD3消融实验训练运行器，用于系统地评估各决策模块的独立贡献。
通过禁用/启用不同的系统组件（卸载、资源分配、迁移等），量化每个模块对整体性能的影响。

【论文对应】
- 消融实验（Ablation Study）：评估系统各模块的必要性
- 对比以下6种策略配置：
  1. local-only: 仅本地执行（无卸载）
  2. remote-only（单RSU远程执行）
  3. offloading-only: 卸载决策（本地vs单RSU）
  4. resource-only: 多节点资源分配（无迁移）
  5. comprehensive-no-migration: 完整系统（无迁移）
  6. comprehensive-migration: 完整CAMTD3系统

【工作流程】
1. 每次调用运行单个策略配置
2. 训练结果（JSON/图表/报告）被复制到策略专属文件夹
3. 维护滚动更新的summary.json，记录所有策略的性能指标
4. 后续可使用summary.json生成策略对比图表

【使用示例】
```bash
# 运行单个策略
python experiments/camtd3_strategy_suite/run_strategy_training.py \\
    --strategy local-only --episodes 800 --seed 42

# 批量运行所有策略（需外部脚本）
for strategy in local-only remote-only offloading-only resource-only \\
                comprehensive-no-migration comprehensive-migration; do
    python experiments/camtd3_strategy_suite/run_strategy_training.py \\
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
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from train_single_agent import _apply_global_seed_from_env, train_single_algorithm
from utils.unified_reward_calculator import UnifiedRewardCalculator

StrategyPreset = Dict[str, Any]  # 策略预设配置类型

# ========== 初始化统一奖励计算器 ==========
# 使用统一奖励计算器确保与训练时的奖励函数一致
_reward_calculator = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    """获取全局奖励计算器实例（延迟初始化）"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator

# ========== 默认实验参数 ==========
DEFAULT_EPISODES = 800   # 默认训练轮数（平衡收敛质量与时间成本）
DEFAULT_SEED = 42        # 默认随机种子（保证实验可重复性）

# ========== 策略执行顺序 ==========
# 按照复杂度递增排列：从单一功能到完整系统
# 这个顺序也用于生成对比图表时的展示顺序
STRATEGY_ORDER = [
    "local-only",                    # 基准1：纯本地计算
    "remote-only",                   # 基准2：强制远程卸载
    "offloading-only",               # 模块1：卸载决策
    "resource-only",                 # 模块2：资源分配
    "comprehensive-no-migration",    # 模块3：完整系统（无迁移）
    "comprehensive-migration",       # 完整系统：所有模块启用
]


def _base_override(num_rsus: int, num_uavs: int, allow_local: bool = True) -> Dict[str, Any]:
    """
    构建一致的场景覆盖配置
    
    【功能】
    为不同策略生成统一的基础场景参数，确保消融实验的对照变量一致性。
    所有策略使用相同的车辆数量、覆盖半径等基础参数，仅改变边缘节点配置。
    
    【参数】
    num_rsus: int - RSU（路边单元）数量
        - 0: 无边缘节点（仅本地）
        - 1: 单节点卸载场景
        - 4: 多节点资源分配场景
    num_uavs: int - UAV（无人机）数量
        - 0: 不使用UAV
        - 2: 典型的UAV辅助场景
    allow_local: bool - 是否允许本地处理（默认True）
        - True: 可以选择本地执行
        - False: 强制卸载
    
    【返回值】
    Dict[str, Any] - 场景覆盖配置字典
    
    【设计原则】
    - 固定车辆数12：典型的VEC场景规模
    - 覆盖半径600m：符合5G NR城市场景
    - override_topology=True：使用固定拓扑，避免随机性干扰
    """
    return {
        "num_vehicles": 12,
        "num_rsus": num_rsus,
        "num_uavs": num_uavs,
        "coverage_radius": 600.0,
        "override_topology": True,
        "allow_local_processing": allow_local,
    }


# ========== 策略预设配置字典 ==========
# 定义6种策略的完整配置，用于系统化的消融实验
# 每个策略都是CAMTD3系统的一个简化版本，用于评估特定模块的贡献
STRATEGY_PRESETS: "OrderedDict[str, StrategyPreset]" = OrderedDict(
    [
        # ===== 策略1: 纯本地执行基准 =====
        (
            "local-only",
            {
                "description": "All tasks execute locally; edge nodes and migration are disabled.",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 无缓存协作
                "disable_migration": True,        # 禁用迁移
                "enforce_offload_mode": "local_only",  # 强制本地执行
                "override_scenario": _base_override(num_rsus=0, num_uavs=0, allow_local=True),  # 无边缘节点
            },
        ),
        # ===== 策略2: 强制远程卸载基准 =====
        (
            "remote-only",
            {
                "description": "Edge-Only (single RSU offload)",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 无缓存协作
                "disable_migration": True,        # 禁用迁移
                "enforce_offload_mode": "remote_only",  # 强制远程卸载
                "override_scenario": _base_override(num_rsus=1, num_uavs=0, allow_local=False),  # 单RSU，禁用本地
            },
        ),
        # ===== 策略3: 卸载决策模块 =====
        (
            "offloading-only",
            {
                "description": "Agent decides between local execution and a single RSU; migration and multi-node balancing are disabled.",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 无缓存协作
                "disable_migration": True,        # 禁用迁移
                "enforce_offload_mode": None,     # 允许智能卸载决策
                "override_scenario": _base_override(num_rsus=1, num_uavs=0, allow_local=True),  # 单RSU，可选本地
            },
        ),
        # ===== 策略4: 资源分配模块 =====
        (
            "resource-only",
            {
                "description": "All tasks must offload; the agent only balances load across RSUs/UAVs (no migration).",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 启用缓存协作
                "disable_migration": True,        # 禁用迁移
                "enforce_offload_mode": "remote_only",  # 强制卸载
                "override_scenario": _base_override(num_rsus=4, num_uavs=2, allow_local=False),  # 多节点，禁用本地
            },
        ),
        # ===== 策略5: 完整系统（无迁移） =====
        (
            "comprehensive-no-migration",
            {
                "description": "Full offloading and resource allocation with migration disabled.",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 启用缓存协作
                "disable_migration": True,        # 禁用迁移
                "enforce_offload_mode": None,     # 允许智能决策
                "override_scenario": _base_override(num_rsus=4, num_uavs=2, allow_local=True),  # 完整配置
            },
        ),
        # ===== 策略6: 完整CAMTD3系统 =====
        (
            "comprehensive-migration",
            {
                "description": "Complete CAMTD3 strategy: offloading, resource allocation, and migration all enabled.",
                "algorithm": "CAMTD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 启用缓存协作
                "disable_migration": False,       # 启用迁移（关键差异）
                "enforce_offload_mode": None,     # 允许智能决策
                "override_scenario": _base_override(num_rsus=4, num_uavs=2, allow_local=True),  # 完整配置
            },
        ),
    ]
)


def tail_mean(values: Any) -> float:
    """
    计算序列后半部分的稳定均值
    
    【功能】
    使用训练后期数据计算性能指标的稳定均值，避免前期探索阶段的高方差干扰。
    这是评估收敛后性能的标准方法。
    
    【参数】
    values: Any - 性能指标序列（如每轮的时延、能耗等）
    
    【返回值】
    float - 后期稳定阶段的均值
    
    【计算策略】
    - 序列长度 >= 100: 使用后50%数据（充分收敛）
    - 序列长度 >= 50: 使用最后30轮数据
    - 序列长度 < 50: 使用全部数据（快速测试模式）
    
    【论文对应】
    评估收敛性能时，通常使用训练后期的平均值作为最终性能指标
    """
    if not values:
        return 0.0
    seq = list(map(float, values))
    length = len(seq)
    if length >= 100:
        subset = seq[length // 2 :]  # 后50%
    elif length >= 50:
        subset = seq[-30:]           # 最后30轮
    else:
        subset = seq                 # 全部数据
    return float(sum(subset) / max(1, len(subset)))


def compute_raw_cost(delay_mean: float, energy_mean: float) -> float:
    """
    计算统一代价函数的原始值
    
    【功能】
    使用统一奖励计算器计算代价，确保与训练时使用的奖励函数完全一致。
    该函数用于策略间的公平对比。
    
    【参数】
    delay_mean: float - 平均时延（秒）
    energy_mean: float - 平均能耗（焦耳）
    
    【返回值】
    float - 归一化后的加权代价
    
    【计算公式】
    Raw Cost = ω_T · (T / T_norm) + ω_E · (E / E_norm)
    其中：
    - ω_T = 2.0（时延权重）
    - ω_E = 1.2（能耗权重）
    - T_norm = 0.2s（时延归一化因子）
    - E_norm = 1000J（能耗归一化因子）
    
    【论文对应】
    优化目标：minimize ω_T·时延 + ω_E·能耗
    该指标越小，系统性能越好
    
    【修复说明】
    ✅ 修复后：使用unified_reward_calculator，确保与训练一致
    ✅ 复用统一模块，遵循DRY原则
    """
    weight_delay = float(config.rl.reward_weight_delay)      # ω_T = 2.0
    weight_energy = float(config.rl.reward_weight_energy)    # ω_E = 1.2
    
    # 使用与统一奖励计算器相同的归一化因子
    calc = _get_reward_calculator()
    delay_normalizer = calc.delay_normalizer  # 0.2
    energy_normalizer = calc.energy_normalizer  # 1000.0
    
    return (
        weight_delay * (delay_mean / max(delay_normalizer, 1e-6))
        + weight_energy * (energy_mean / max(energy_normalizer, 1e-6))
    )


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
    更新策略实验摘要JSON文件
    
    【功能】
    将单个策略的训练结果追加到suite级别的summary.json中。
    该文件汇总所有策略的性能指标，用于后续的对比分析和可视化。
    
    【参数】
    suite_path: Path - Suite根目录路径
    strategy: str - 策略名称（如"local-only"）
    preset: StrategyPreset - 策略预设配置
    result: Dict[str, Any] - 训练返回的完整结果
    metrics: Dict[str, float] - 计算后的性能指标
    artifacts: Dict[str, str] - 生成的文件路径
    episodes: int - 实际训练轮数
    seed: int - 使用的随机种子
    
    【返回值】
    None（直接写入文件）
    
    【summary.json结构】
    {
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
    
    【使用场景】
    - 每个策略训练完成后调用一次
    - 支持增量更新（可多次运行不同策略）
    - 后续可用于生成对比图表
    """
    summary_path = suite_path / "summary.json"
    
    # ========== 加载或创建summary ==========
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "suite_id": suite_path.name,
            "created_at": datetime.now().isoformat(),
            "strategies": {},
        }
    
    # ========== 更新策略信息 ==========
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
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }
    
    # ========== 持久化保存 ==========
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_artifacts(
    result: Dict[str, Any],
    strategy_dir: Path,
) -> Dict[str, str]:
    """
    复制训练产生的核心文件到策略专属目录
    
    【功能】
    将train_single_agent.py生成的结果文件（JSON/图表/报告）复制到
    策略专属文件夹，便于后续分析和归档。
    
    【参数】
    result: Dict[str, Any] - 训练结果字典（包含algorithm、timestamp等）
    strategy_dir: Path - 策略专属目录（如results/camtd3_strategy_suite/suite_id/local-only/）
    
    【返回值】
    Dict[str, str] - 复制后的文件路径字典
        {
          "training_json": "path/to/training_results.json",
          "training_chart": "path/to/training_overview.png",
          "training_report": "path/to/training_report.html"
        }
    
    【复制的文件】
    1. training_results_{timestamp}.json - 完整训练数据
    2. training_overview.png - 训练曲线图表
    3. training_report_{timestamp}.html - 训练报告
    
    【源文件位置】
    results/single_agent/{algorithm}/
    
    【目标位置】
    results/camtd3_strategy_suite/{suite_id}/{strategy}/
    """
    algorithm = str(result.get("algorithm", "")).lower()
    timestamp = result.get("timestamp")
    artifacts: Dict[str, str] = {}

    # ========== 确定源文件路径 ==========
    src_root = Path("results") / "single_agent" / algorithm
    if timestamp:
        json_name = f"training_results_{timestamp}.json"
        report_name = f"training_report_{timestamp}.html"
    else:
        json_name = "training_results.json"
        report_name = "training_report.html"
    chart_name = "training_overview.png"

    # ========== 定义复制清单 ==========
    copies = [
        ("training_json", src_root / json_name),
        ("training_chart", src_root / chart_name),
        ("training_report", src_root / report_name),
    ]
    
    # ========== 执行复制 ==========
    strategy_dir.mkdir(parents=True, exist_ok=True)
    for key, src in copies:
        if src.exists():
            dst = strategy_dir / src.name
            shutil.copy2(src, dst)  # copy2保留元数据
            artifacts[key] = str(dst)
    
    return artifacts


def run_strategy(strategy: str, args: argparse.Namespace) -> None:
    """
    执行单个策略的完整训练流程
    
    【功能】
    这是主执行函数，完成以下任务：
    1. 加载策略配置
    2. 设置随机种子
    3. 调用train_single_algorithm进行训练
    4. 计算稳定性能指标
    5. 复制结果文件
    6. 更新summary.json
    7. 打印结果摘要
    
    【参数】
    strategy: str - 策略名称（必须在STRATEGY_PRESETS中定义）
    args: argparse.Namespace - 命令行参数
    
    【工作流程】
    步骤1: 验证策略名称
    步骤2: 设置随机种子（保证可重复性）
    步骤3: 调用训练函数（使用策略专属配置）
    步骤4: 从训练结果中提取性能指标
    步骤5: 计算稳定均值（使用tail_mean）
    步骤6: 复制生成的文件到策略目录
    步骤7: 更新汇总JSON
    步骤8: 打印结果
    
    【输出文件结构】
    results/camtd3_strategy_suite/{suite_id}/
    ├── summary.json                    # 汇总文件（所有策略）
    ├── local-only/
    │   ├── training_results_*.json
    │   ├── training_overview.png
    │   └── training_report_*.html
    ├── remote-only/
    │   └── ...
    └── ...
    
    【性能指标】
    - delay_mean: 平均任务时延（秒）
    - energy_mean: 平均总能耗（焦耳）
    - completion_mean: 任务完成率（0-1）
    - raw_cost: 统一代价函数（越小越好）
    """
    # ========== 步骤1: 加载策略配置 ==========
    if strategy not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {strategy}")
    preset = STRATEGY_PRESETS[strategy]

    # ========== 步骤2: 确定训练参数 ==========
    episodes = args.episodes or preset["episodes"]
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    # ========== 步骤3: 设置随机种子 ==========
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()

    # ========== 步骤4: 执行训练 ==========
    # 使用args.silent参数控制静默模式（批量实验推荐开启）
    silent = getattr(args, 'silent', True)  # 默认静默模式，避免交互卡住
    results = train_single_algorithm(
        preset["algorithm"],
        num_episodes=episodes,
        silent_mode=silent,
        override_scenario=preset["override_scenario"],
        use_enhanced_cache=preset["use_enhanced_cache"],
        disable_migration=preset["disable_migration"],
        enforce_offload_mode=preset["enforce_offload_mode"],
    )

    # ========== 步骤5: 提取性能指标 ==========
    episode_metrics: Dict[str, Any] = results.get("episode_metrics", {})
    delay_mean = tail_mean(episode_metrics.get("avg_delay", []))
    energy_mean = tail_mean(episode_metrics.get("total_energy", []))
    completion_mean = tail_mean(episode_metrics.get("task_completion_rate", []))
    raw_cost = compute_raw_cost(delay_mean, energy_mean)

    # ========== 步骤6: 准备输出目录 ==========
    suite_id = args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_path = Path(args.output_root) / suite_id
    strategy_dir = suite_path / strategy
    suite_path.mkdir(parents=True, exist_ok=True)

    # ========== 步骤7: 复制结果文件 ==========
    artifacts = copy_artifacts(results, strategy_dir)

    # ========== 步骤8: 汇总性能指标 ==========
    metrics = {
        "delay_mean": delay_mean,
        "energy_mean": energy_mean,
        "completion_mean": completion_mean,
        "raw_cost": raw_cost,
    }
    
    # ========== 步骤9: 更新summary.json ==========
    update_summary(suite_path, strategy, preset, results, metrics, artifacts, episodes, seed)

    # ========== 步骤10: 打印结果摘要 ==========
    print("\n=== Strategy Run Completed ===")
    print(f"Suite ID        : {suite_id}")
    print(f"Strategy        : {strategy}")
    print(f"Episodes        : {episodes}")
    print(f"Seed            : {seed}")
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
    构建命令行参数解析器
    
    【功能】
    定义脚本的命令行接口，支持灵活配置训练参数。
    
    【返回值】
    argparse.ArgumentParser - 配置好的参数解析器
    
    【命令行参数】
    --strategy: str (必需)
        - 策略名称，可选值: local-only, remote-only, offloading-only, 
          resource-only, comprehensive-no-migration, comprehensive-migration
    
    --episodes: int (可选)
        - 训练轮数，默认800
        - 快速测试可用50-100，完整实验建议800-1000
    
    --seed: int (可选)
        - 随机种子，默认42
        - 用于保证实验可重复性
    
    --suite-id: str (可选)
        - Suite标识符，用于将多个策略归为同一组实验
        - 未指定时自动生成时间戳（YYYYMMDD_HHMMSS）
    
    --output-root: str (可选)
        - 输出根目录，默认"results/camtd3_strategy_suite"
    
    --silent: bool (可选)
        - 静默模式，减少训练过程的输出
        - ✅ 注意：批量实验脚本默认已启用静默模式，无需手动交互
    
    【使用示例】
    # ✅ 默认静默运行（无需手动交互，推荐）
    # 基本用法
    python run_strategy_training.py --strategy local-only
    
    # 指定参数 - 自动保存报告，无人值守运行
    python run_strategy_training.py --strategy comprehensive-migration \\
        --episodes 1000 --seed 123 --suite-id exp_ablation_v1
    
    # 快速测试（已默认静默）
    python run_strategy_training.py --strategy offloading-only \\
        --episodes 50
    
    # 💡 如需交互式确认保存报告，添加 --interactive 参数
    python run_strategy_training.py --strategy camtd3-full \\
        --episodes 500 --interactive
    """
    parser = argparse.ArgumentParser(
        description="Run CAMTD3 under a specific strategy baseline and collect results."
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
        default="results/camtd3_strategy_suite",
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
    脚本主入口函数
    
    【功能】
    解析命令行参数并启动策略训练流程。
    
    【执行流程】
    1. 构建参数解析器
    2. 解析命令行参数
    3. 调用run_strategy执行训练
    
    【错误处理】
    - 未知策略名称：ValueError
    - 参数缺失：argparse自动提示
    - 训练过程错误：由train_single_algorithm处理
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    run_strategy(args.strategy, args)


if __name__ == "__main__":
    main()
