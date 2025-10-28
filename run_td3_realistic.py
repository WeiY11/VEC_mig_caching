#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3真实对比实验 - 启动脚本

特点：
✅ 所有算法都是真实的
✅ 所有算法你已经有了
✅ 不编造任何内容
✅ 立即可以开始

使用方法：
    # 快速测试（2小时）
    python run_td3_realistic.py --mode quick
    
    # 标准实验（14-16小时）
    python run_td3_realistic.py --mode standard
    
    # 只运行某一组
    python run_td3_realistic.py --mode standard --group drl
    python run_td3_realistic.py --mode standard --group ablation

用途：
- 启动“真实可用对比方案”（experiments/td3_realistic_comparison.py）：
  仅使用项目内已有算法（DRL/启发式/消融），不依赖外部复现，快速得到可靠对比基线。
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from experiments.td3_realistic_comparison import (
    RealisticComparisonAlgorithms,
    print_realistic_plan
)
from experiments.td3_focused_comparison import TD3FocusedComparison


def run_realistic_experiments(group_name: str = "all", mode: str = "standard"):
    """运行真实对比实验"""
    
    all_configs = RealisticComparisonAlgorithms.define_all_algorithms()
    groups = RealisticComparisonAlgorithms.get_algorithm_groups()
    
    # 选择对应组的算法
    group_map = {
        "drl": "A_DRL",
        "heuristic": "B_Heuristic",
        "ablation": "C_Ablation"
    }
    
    if group_name == "all":
        selected_configs = all_configs
        output_suffix = "all"
    else:
        group_key = group_map.get(group_name)
        if not group_key:
            print(f"❌ 未知的组名: {group_name}")
            print(f"   可选: {list(group_map.keys())} 或 'all'")
            return
        
        alg_names = groups[group_key]
        selected_configs = [c for c in all_configs if c.name in alg_names]
        output_suffix = group_name
    
    print(f"\n🚀 运行 {group_name.upper()} 组实验...")
    print(f"   包含算法: {[c.name for c in selected_configs]}")
    print(f"   模式: {mode.upper()}")
    
    # 创建实验执行器
    runner = TD3FocusedComparison(
        output_dir=f"results/td3_realistic_{output_suffix}"
    )
    
    # 根据模式调整参数
    for config in selected_configs:
        if mode == "quick":
            config.episodes = int(config.episodes * 0.1)
            config.seeds = config.seeds[:1]
        
        # 运行实验
        try:
            result = runner.run_experiment(config)
            runner.results[config.name] = result
        except Exception as e:
            print(f"❌ 算法 {config.name} 运行失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 保存结果
    runner._save_summary()
    runner._generate_paper_materials()
    
    print(f"\n✅ {group_name.upper()} 组实验完成！")
    print(f"   结果保存在: {runner.experiment_dir}")
    
    return runner.experiment_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TD3真实对比实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 查看实验计划
  python run_td3_realistic.py --show-plan
  
  # 快速测试（2小时）
  python run_td3_realistic.py --mode quick
  
  # 标准实验（14-16小时）
  python run_td3_realistic.py --mode standard
  
  # 分组运行
  python run_td3_realistic.py --mode standard --group drl
  python run_td3_realistic.py --mode standard --group ablation
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "standard"],
        help="实验模式: quick(快速) 或 standard(标准)"
    )
    
    parser.add_argument(
        "--group",
        type=str,
        default="all",
        choices=["all", "drl", "heuristic", "ablation"],
        help="实验组: all(全部), drl(DRL对比), heuristic(启发式), ablation(消融)"
    )
    
    parser.add_argument(
        "--show-plan",
        action="store_true",
        help="只显示实验计划"
    )
    
    args = parser.parse_args()
    
    # 显示实验计划
    print_realistic_plan()
    
    if args.show_plan:
        print("\n📋 仅显示实验计划，不执行实验")
        
        # 显示论文描述模板
        print(RealisticComparisonAlgorithms.get_paper_template())
        return
    
    # 确认执行
    if args.mode == "standard":
        if args.group == "all":
            time_estimate = "14-16小时"
        elif args.group == "drl":
            time_estimate = "8-10小时"
        elif args.group == "ablation":
            time_estimate = "5-6小时"
        else:
            time_estimate = "1-2小时"
        
        print(f"\n⚠️  Standard模式预计需要 {time_estimate}")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
    
    # 运行实验
    print("\n🚀 开始运行实验...\n")
    output_dir = run_realistic_experiments(args.group, args.mode)
    
    print("\n" + "="*80)
    print("🎉 实验完成！")
    print("="*80)
    print(f"\n📁 结果目录: {output_dir}")
    print("\n📊 生成的文件:")
    print("  - experiment_summary.json      (实验总结)")
    print("  - table1_algorithm_comparison.csv  (对比表格)")
    print("  - table1_latex.tex             (LaTeX表格)")
    print("  - statistical_analysis.txt     (统计分析)")
    print("  - figures/                     (所有图表)")
    print("\n💡 下一步:")
    print(f"  1. 查看结果: cat {output_dir}/experiment_summary.json")
    print(f"  2. 查看图表: explorer {output_dir}\\figures")
    print(f"  3. 查看LaTeX: cat {output_dir}/table1_latex.tex")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()



