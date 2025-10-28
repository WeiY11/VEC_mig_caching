#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3聚焦对比实验 - 启动脚本
python run_td3_focused.py --mode standard --experiment all
快速使用：
    # 快速测试（1-2小时，验证流程）
    python run_td3_focused.py --mode quick
    
    # 标准实验（24-30小时，论文标准）
    python run_td3_focused.py --mode standard
    
    # 只运行某个实验组
    python run_td3_focused.py --mode standard --experiment baseline

用途：
- 启动“聚焦对比方案”（experiments/td3_focused_comparison.py）：
  仅包含最核心的三组实验（算法对比、车辆规模、网络条件），快速产出论文必需图表。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.td3_focused_comparison import TD3FocusedComparison


def print_plan():
    """打印实验计划"""
    print("\n" + "="*80)
    print("🎯 TD3聚焦对比实验方案")
    print("="*80)
    print("\n【核心目标】")
    print("  证明CAM-TD3方案有效降低时延和能耗")
    print("\n【实验设计】")
    print("  1️⃣ Baseline对比 (4个算法)")
    print("     CAM-TD3 vs DDPG vs SAC vs Greedy")
    print("     → 论文Table 1: 算法性能对比")
    print()
    print("  2️⃣ 车辆规模扫描 (5个规模点)")
    print("     8, 12, 16, 20, 24辆车")
    print("     → 论文Figure 1: 车辆规模影响曲线")
    print()
    print("  3️⃣ 网络条件对比 (3个维度)")
    print("     - 带宽: 10, 15, 20, 25 MHz")
    print("     - RSU密度: 2, 4, 6 个")
    print("     - 极端场景: 低带宽+高负载")
    print("     → 论文Figure 2: 网络条件影响")
    print("\n【实验模式】")
    print("  Quick模式:    80 episodes × 1 seed  ≈  2小时   (验证流程)")
    print("  Standard模式: 800 episodes × 1 seed  ≈ 24-30小时 (论文标准)")
    print("\n【预期产出】")
    print("  ✓ table1_algorithm_comparison.csv     (算法对比表)")
    print("  ✓ figure1_vehicle_scaling.json        (车辆规模曲线数据)")
    print("  ✓ figure2_bandwidth_impact.json       (带宽影响曲线数据)")
    print("  ✓ 30个详细结果JSON文件")
    print("="*80 + "\n")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TD3聚焦对比实验启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "standard"],
        help="实验模式: quick(快速测试) 或 standard(论文标准)"
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "baseline", "vehicle", "network"],
        help="实验选择: all(全部), baseline(算法对比), vehicle(车辆规模), network(网络条件)"
    )
    
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="只显示实验计划，不执行"
    )
    
    args = parser.parse_args()
    
    # 显示实验计划
    print_plan()
    
    if args.show_plan:
        print("📋 仅显示实验计划，不执行实验")
        return
    
    # 确认执行
    if args.mode == "standard":
        print("⚠️  Standard模式预计需要24-30小时")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
    
    # 运行实验
    print("\n🚀 开始运行实验...\n")
    
    runner = TD3FocusedComparison()
    
    if args.experiment == "all":
        runner.run_all_experiments(mode=args.mode)
    else:
        # 运行单个实验组
        if args.experiment == "baseline":
            configs = runner.define_baseline_comparison()
            print("\n📊 运行Baseline对比实验...")
        elif args.experiment == "vehicle":
            configs = runner.define_vehicle_scaling()
            print("\n📈 运行车辆规模扫描实验...")
        else:  # network
            configs = runner.define_network_conditions()
            print("\n🌐 运行网络条件对比实验...")
        
        for config in configs:
            if args.mode == "quick":
                config.episodes = int(config.episodes * 0.1)
            # 始终单种子运行，确保与计划一致
            config.seeds = config.seeds[:1]
            result = runner.run_experiment(config)
            runner.results[config.name] = result
        
        runner._save_summary()
        runner._generate_paper_materials()
    
    print("\n" + "="*80)
    print("🎉 实验完成！")
    print("="*80)
    print(f"📁 结果目录: {runner.experiment_dir}")
    print("\n📊 论文素材:")
    print(f"   - {runner.experiment_dir}/table1_algorithm_comparison.csv")
    print(f"   - {runner.experiment_dir}/figure1_vehicle_scaling.json")
    print(f"   - {runner.experiment_dir}/figure2_bandwidth_impact.json")
    print("\n💡 下一步:")
    print("   1. 查看结果: cat results/td3_focused/*/experiment_summary.json")
    print("   2. 生成图表: python tools/plot_td3_results.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()

