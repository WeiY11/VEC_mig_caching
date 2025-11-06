#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
三种方案核心对比实验
===================

【对比方案】
1. CAMTD3(Avg)      - 固定均匀资源分配
2. CAMTD3(Agent)    - 中央智能体动态资源分配
3. CAMTD3 no mig    - 固定资源 + 禁用任务迁移（仅本地计算）

【对比维度】
- 不同任务到达率（1.5, 2.0, 2.5, 3.0 tasks/s/车）
- 不同本地计算资源（4, 6, 8, 10 GHz总资源）

【输出】
- 对比图表：平均成本 (ω_T·时延 + ω_E·能耗)
- JSON结果：供进一步分析

【使用示例】
```bash
# 默认运行（400轮）
python experiments/td3_strategy_suite/run_three_mode_comparison.py

# 快速测试（10轮）
python experiments/td3_strategy_suite/run_three_mode_comparison.py --episodes 10

# 完整实验（800轮）
python experiments/td3_strategy_suite/run_three_mode_comparison.py --episodes 800

# 只对比到达率
python experiments/td3_strategy_suite/run_three_mode_comparison.py --dimension arrival

# 只对比本地计算
python experiments/td3_strategy_suite/run_three_mode_comparison.py --dimension compute
```
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np

# ========== 添加项目根目录到路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train_single_agent import train_single_algorithm, _apply_global_seed_from_env
from config import config

# ========== 三种运行模式配置 ==========
MODES = [
    {
        "name": "CAMTD3(Avg)",
        "key": "standard",
        "description": "固定均匀资源分配",
        "flags": [],  # 不加任何特殊参数
        "disable_migration": False,
        "color": "#1f77b4",
        "marker": "o",
    },
    {
        "name": "CAMTD3(Agent)",
        "key": "central",
        "description": "中央智能体动态资源分配",
        "flags": ["--central-resource"],
        "disable_migration": False,
        "color": "#ff7f0e",
        "marker": "s",
    },
    {
        "name": "CAMTD3 no mig",
        "key": "nomig",
        "description": "固定资源 + 禁用任务迁移（仅本地计算）",
        "flags": [],  # 资源分配和标准模式一样
        "disable_migration": True,  # 只禁用迁移
        "color": "#2ca02c",
        "marker": "^",
    },
]

# ========== 实验配置 ==========
# 任务到达率配置
ARRIVAL_RATES = [1.5, 2.0, 2.5, 3.0]  # tasks/s/车

# 本地计算资源配置
COMPUTE_RESOURCES = [4.0, 6.0, 8.0, 10.0]  # GHz 总资源

DEFAULT_EPISODES = 400
DEFAULT_SEED = 42


def run_single_training(
    mode: Dict,
    override_config: Dict,
    episodes: int,
    silent: bool,
    seed: int,
) -> Dict[str, Any]:
    """
    运行单次训练
    
    Args:
        mode: 运行模式配置
        override_config: 场景覆盖配置
        episodes: 训练轮数
        silent: 是否静默
        seed: 随机种子
    
    Returns:
        训练结果字典
    """
    print(f"  运行: {mode['name']}")
    
    try:
        # 🎯 设置环境变量（模式控制）
        if "--central-resource" in mode["flags"]:
            os.environ['CENTRAL_RESOURCE'] = '1'
        else:
            os.environ.pop('CENTRAL_RESOURCE', None)
        
        # 设置随机种子
        os.environ['RANDOM_SEED'] = str(seed)
        _apply_global_seed_from_env()
        
        # 🔧 获取是否禁用迁移（从mode配置中读取）
        disable_migration = mode.get("disable_migration", False)
        
        # 调用训练函数
        result = train_single_algorithm(
            algorithm="TD3",
            num_episodes=episodes,
            silent_mode=silent,
            override_scenario=override_config if override_config else None,
            disable_migration=disable_migration,  # 只有无迁移模式为True
            enforce_offload_mode=None,  # 不强制卸载模式
        )
        
        # 提取关键指标
        episode_metrics = result.get("episode_metrics", {})
        
        # 计算平均值（取后50%的数据）
        def tail_mean(values):
            if not values:
                return 0.0
            return float(np.mean(values[len(values)//2:]))
        
        # 计算平均成本（核心评价指标）
        avg_delay = tail_mean(episode_metrics.get("avg_delay", []))
        avg_energy = tail_mean(episode_metrics.get("total_energy", []))
        
        # 使用配置中的权重计算平均成本
        weight_delay = config.rl.reward_weight_delay
        weight_energy = config.rl.reward_weight_energy
        avg_cost = weight_delay * avg_delay + weight_energy * avg_energy
        
        metrics = {
            "success": True,
            "mode": mode["key"],
            "avg_delay": avg_delay,
            "avg_energy": avg_energy,
            "avg_cost": avg_cost,  # 🎯 核心对比指标
            "completion_rate": tail_mean(episode_metrics.get("task_completion_rate", [])),
            "cache_hit_rate": tail_mean(episode_metrics.get("cache_hit_rate", [])),
            "final_reward": result.get("final_episode_reward", 0.0),
        }
        
        print(f"  ✅ 完成 - 平均成本:{avg_cost:.2f} (时延:{avg_delay:.3f}s×{weight_delay:.1f} + 能耗:{avg_energy:.0f}J×{weight_energy:.1f}), 完成率:{metrics['completion_rate']*100:.1f}%")
        
        return metrics
        
    except Exception as e:
        print(f"  ❌ 异常: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "mode": mode["key"], "error": str(e)}


def run_arrival_rate_comparison(
    modes: List[Dict],
    arrival_rates: List[float],
    episodes: int,
    silent: bool,
    seed: int,
) -> List[Dict]:
    """运行任务到达率对比实验"""
    
    print("\n" + "=" * 80)
    print("任务到达率对比实验")
    print("=" * 80)
    
    results = []
    total = len(modes) * len(arrival_rates)
    counter = 0
    
    for rate in arrival_rates:
        print(f"\n配置: 任务到达率 {rate} tasks/s/车 (总{rate*12:.0f} tasks/s)")
        print("-" * 80)
        
        config_results = {
            "arrival_rate": rate,
            "total_arrival_rate": rate * 12,
            "modes": {}
        }
        
        for mode in modes:
            counter += 1
            print(f"[{counter}/{total}] {mode['name']}...")
            
            override_config = {
                "task_arrival_rate": rate,
                "override_topology": True,
            }
            
            result = run_single_training(
                mode=mode,
                override_config=override_config,
                episodes=episodes,
                silent=silent,
                seed=seed,
            )
            
            config_results["modes"][mode["key"]] = result
        
        results.append(config_results)
    
    return results


def run_compute_resource_comparison(
    modes: List[Dict],
    compute_resources: List[float],
    episodes: int,
    silent: bool,
    seed: int,
) -> List[Dict]:
    """运行本地计算资源对比实验"""
    
    print("\n" + "=" * 80)
    print("本地计算资源对比实验")
    print("=" * 80)
    
    results = []
    total = len(modes) * len(compute_resources)
    counter = 0
    
    for compute_ghz in compute_resources:
        avg_per_vehicle = compute_ghz / 12
        print(f"\n配置: 总本地计算 {compute_ghz:.1f} GHz (每车{avg_per_vehicle:.3f} GHz)")
        print("-" * 80)
        
        config_results = {
            "total_compute_ghz": compute_ghz,
            "avg_per_vehicle_ghz": avg_per_vehicle,
            "modes": {}
        }
        
        for mode in modes:
            counter += 1
            print(f"[{counter}/{total}] {mode['name']}...")
            
            override_config = {
                "total_vehicle_compute": compute_ghz * 1e9,  # 转换为Hz
                "override_topology": True,
            }
            
            result = run_single_training(
                mode=mode,
                override_config=override_config,
                episodes=episodes,
                silent=silent,
                seed=seed,
            )
            
            config_results["modes"][mode["key"]] = result
        
        results.append(config_results)
    
    return results


def plot_comparison_results(
    arrival_results: List[Dict],
    compute_results: List[Dict],
    output_dir: Path,
):
    """生成对比图表（平均成本对比）"""
    
    print("\n📊 生成对比图表...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    # ========== 1. 任务到达率对比图（平均成本）==========
    if arrival_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        arrival_rates = [r["arrival_rate"] for r in arrival_results]
        
        # 平均成本对比
        for mode in MODES:
            costs = [r["modes"][mode["key"]]["avg_cost"] for r in arrival_results 
                    if r["modes"][mode["key"]].get("success", False)]
            if costs:
                ax.plot(arrival_rates[:len(costs)], costs, 
                       marker=mode["marker"], color=mode["color"], 
                       linewidth=2.5, markersize=10, label=mode["name"],
                       markeredgewidth=1.5, markeredgecolor='white')
        
        ax.set_xlabel('任务到达率 (tasks/s/车)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均成本 (ω_T·时延 + ω_E·能耗)', fontsize=13, fontweight='bold')
        ax.set_title('三种方案平均成本对比 - 任务到达率影响', fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / "arrival_rate_cost_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 已生成: arrival_rate_cost_comparison.png")
    
    # ========== 2. 本地计算资源对比图（平均成本）==========
    if compute_results:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        compute_ghz = [r["total_compute_ghz"] for r in compute_results]
        
        # 平均成本对比
        for mode in MODES:
            costs = [r["modes"][mode["key"]]["avg_cost"] for r in compute_results 
                    if r["modes"][mode["key"]].get("success", False)]
            if costs:
                ax.plot(compute_ghz[:len(costs)], costs, 
                       marker=mode["marker"], color=mode["color"], 
                       linewidth=2.5, markersize=10, label=mode["name"],
                       markeredgewidth=1.5, markeredgecolor='white')
        
        ax.set_xlabel('总本地计算资源 (GHz)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均成本 (ω_T·时延 + ω_E·能耗)', fontsize=13, fontweight='bold')
        ax.set_title('三种方案平均成本对比 - 本地计算资源影响', fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=11)
        
        plt.tight_layout()
        plt.savefig(output_dir / "compute_resource_cost_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  ✅ 已生成: compute_resource_cost_comparison.png")


def main():
    parser = argparse.ArgumentParser(
        description="三种方案核心对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
对比方案：
  1. CAMTD3(Avg) - 固定均匀资源分配
  2. CAMTD3(Agent) - 中央智能体动态资源分配
  3. CAMTD3 no mig - 固定资源 + 禁用任务迁移（仅本地计算）

对比维度：
  - 任务到达率: 1.5-3.0 tasks/s/车
  - 本地计算: 4-10 GHz总资源

示例：
  python %(prog)s --episodes 400
  python %(prog)s --episodes 10 --dimension arrival
  python %(prog)s --episodes 800 --dimension compute
        """
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=DEFAULT_EPISODES,
        help=f"每个配置的训练轮数 (默认: {DEFAULT_EPISODES})",
    )
    parser.add_argument(
        "--dimension",
        type=str,
        choices=["all", "arrival", "compute"],
        default="all",
        help="对比维度: all(全部), arrival(到达率), compute(本地计算)",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=True,
        help="静默模式 (默认)",
    )
    parser.add_argument(
        "--no-silent",
        action="store_false",
        dest="silent",
        help="显示详细日志",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"随机种子 (默认: {DEFAULT_SEED})",
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    import os
    os.environ['RANDOM_SEED'] = str(args.seed)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = project_root / "results" / "three_mode_comparison" / f"suite_{timestamp}"
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 打印实验信息
    print("=" * 80)
    print("三种方案核心对比实验")
    print("=" * 80)
    print(f"\n开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"输出目录: {output_root}")
    print(f"训练轮数: {args.episodes} episodes")
    print(f"对比维度: {args.dimension}")
    print(f"随机种子: {args.seed}")
    
    print("\n对比方案:")
    for idx, mode in enumerate(MODES, 1):
        print(f"  {idx}. {mode['name']} - {mode['description']}")
    
    # 运行实验
    arrival_results = []
    compute_results = []
    
    start_time = datetime.now()
    
    if args.dimension in ["all", "arrival"]:
        print("\n" + "=" * 80)
        print("维度1: 任务到达率对比")
        print("=" * 80)
        print(f"配置: {len(ARRIVAL_RATES)} × {len(MODES)} = {len(ARRIVAL_RATES) * len(MODES)} 个训练")
        
        arrival_results = run_arrival_rate_comparison(
            modes=MODES,
            arrival_rates=ARRIVAL_RATES,
            episodes=args.episodes,
            silent=args.silent,
            seed=args.seed,
        )
    
    if args.dimension in ["all", "compute"]:
        print("\n" + "=" * 80)
        print("维度2: 本地计算资源对比")
        print("=" * 80)
        print(f"配置: {len(COMPUTE_RESOURCES)} × {len(MODES)} = {len(COMPUTE_RESOURCES) * len(MODES)} 个训练")
        
        compute_results = run_compute_resource_comparison(
            modes=MODES,
            compute_resources=COMPUTE_RESOURCES,
            episodes=args.episodes,
            silent=args.silent,
            seed=args.seed,
        )
    
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # 生成对比图表
    plot_comparison_results(arrival_results, compute_results, output_root)
    
    # 保存结果
    summary = {
        "experiment_type": "three_mode_comparison",
        "timestamp": timestamp,
        "created_at": datetime.now().isoformat(),
        "episodes": args.episodes,
        "seed": args.seed,
        "dimension": args.dimension,
        "modes": [{"name": m["name"], "key": m["key"], "description": m["description"]} for m in MODES],
        "arrival_rate_results": arrival_results,
        "compute_resource_results": compute_results,
        "elapsed_time": str(elapsed),
    }
    
    summary_path = output_root / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    # 打印总结
    print("\n" + "=" * 80)
    print("实验完成！")
    print("=" * 80)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed}")
    
    print("\n实验统计:")
    if arrival_results:
        success_count = sum(
            1 for r in arrival_results 
            for m in r["modes"].values() 
            if m.get("success", False)
        )
        total_count = len(arrival_results) * len(MODES)
        print(f"  任务到达率维度: {success_count}/{total_count} 成功")
    
    if compute_results:
        success_count = sum(
            1 for r in compute_results 
            for m in r["modes"].values() 
            if m.get("success", False)
        )
        total_count = len(compute_results) * len(MODES)
        print(f"  本地计算维度: {success_count}/{total_count} 成功")
    
    print(f"\n结果保存在: {output_root}")
    print(f"  - summary.json: 实验总结")
    print(f"  - arrival_dimension/: 到达率对比结果")
    print(f"  - compute_dimension/: 计算资源对比结果")
    
    print("\n下一步:")
    print("  1. 查看训练结果: results/single_agent/td3/")
    print("  2. 分析性能指标")
    print("  3. 生成对比图表")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

