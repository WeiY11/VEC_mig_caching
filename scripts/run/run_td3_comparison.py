#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3对比实验启动脚本
快速运行TD3综合对比实验套件

使用方法：
    # 快速测试（验证流程，约2小时）
    python run_td3_comparison.py --mode quick
    
    # 标准实验（论文标准，约48小时）
    python run_td3_comparison.py --mode standard
    
    # 扩展实验（最全面，约96小时）
    python run_td3_comparison.py --mode extensive
    
    # 仅运行特定维度
    python run_td3_comparison.py --mode standard --dimension ablation
    
    # 自定义配置
    python run_td3_comparison.py --config config/td3_experiment_config.json --mode standard

用途：
- 启动“综合对比实验套件”（experiments/td3_comprehensive_comparison.py）的统一入口。
- 支持快速/标准/扩展模式与按维度选择运行，自动组织结果与图表数据。
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from experiments.td3_comprehensive_comparison import TD3ComprehensiveComparison


def load_config(config_path: str = "config/td3_experiment_config.json") -> dict:
    """加载实验配置"""
    if not os.path.exists(config_path):
        print(f"⚠️ 配置文件不存在: {config_path}")
        print("使用默认配置")
        return {}
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def print_experiment_plan(config: dict, mode: str, dimension: str):
    """打印实验计划"""
    print("\n" + "="*80)
    print("🔬 TD3综合对比实验计划")
    print("="*80)
    
    mode_config = config.get("execution_modes", {}).get(mode, {})
    print(f"\n模式: {mode.upper()}")
    print(f"描述: {mode_config.get('description', 'N/A')}")
    print(f"预计时间: {mode_config.get('estimated_time_hours', 'N/A')} 小时")
    print(f"Episode缩放: {mode_config.get('episode_factor', 1.0)}x")
    print(f"随机种子数: {mode_config.get('seed_count', 3)}")
    
    print(f"\n实验维度: {dimension.upper()}")
    
    dimensions = config.get("dimensions", {})
    
    if dimension == "all":
        print("\n启用的维度:")
        for dim_name, dim_config in dimensions.items():
            if dim_config.get("enabled", False):
                priority = dim_config.get("priority", 99)
                desc = dim_config.get("description", "")
                print(f"  [{priority}] {dim_name}: {desc}")
    else:
        dim_config = dimensions.get(dimension, {})
        print(f"  描述: {dim_config.get('description', 'N/A')}")
        print(f"  优先级: {dim_config.get('priority', 'N/A')}")
    
    print("\n默认场景配置:")
    default_scenario = config.get("default_scenario", {})
    for key, value in default_scenario.items():
        print(f"  {key}: {value}")
    
    print("\n评估指标:")
    metrics = config.get("metrics", [])
    for metric in metrics:
        print(f"  - {metric}")
    
    print("\n" + "="*80)


def run_quick_test():
    """运行快速测试（验证流程）"""
    print("\n🚀 快速测试模式")
    print("这将运行一个简化的实验来验证整个流程")
    print("预计时间: 1-2小时\n")
    
    runner = TD3ComprehensiveComparison(output_dir="results/td3_comprehensive_quick")
    
    # 只运行消融实验的2个配置
    from experiments.td3_comprehensive_comparison import TD3ExperimentConfig
    
    configs = [
        TD3ExperimentConfig(
            name="Full-System-Quick",
            description="完整系统快速测试",
            episodes=80,  # 10%轮次
            seeds=[42],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2
        ),
        TD3ExperimentConfig(
            name="No-Cache-Quick",
            description="无缓存快速测试",
            episodes=80,
            seeds=[42],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2,
            enable_cache=False
        )
    ]
    
    results = {}
    for config in configs:
        result = runner.run_experiment(config, algorithm="TD3")
        results[config.name] = result
    
    runner.results = results
    runner._save_summary()
    
    print("\n✅ 快速测试完成！")
    print(f"结果保存在: {runner.experiment_dir}")
    
    return runner.experiment_dir


def run_dimension_experiments(dimension: str, mode: str, config: dict):
    """运行特定维度的实验"""
    runner = TD3ComprehensiveComparison(
        output_dir=f"results/td3_{dimension}_{mode}"
    )
    
    print(f"\n🎯 运行维度: {dimension.upper()}")
    
    if dimension == "ablation":
        print("消融实验: 验证各模块有效性")
        ablation_configs = runner.define_ablation_study()
        
        # 根据模式调整参数
        mode_config = config.get("execution_modes", {}).get(mode, {})
        episode_factor = mode_config.get("episode_factor", 1.0)
        seed_count = mode_config.get("seed_count", 3)
        
        for cfg in ablation_configs:
            cfg.episodes = int(cfg.episodes * episode_factor)
            cfg.seeds = cfg.seeds[:seed_count]
            result = runner.run_experiment(cfg, algorithm="TD3")
            runner.results[cfg.name] = result
    
    elif dimension == "sensitivity":
        print("参数敏感性分析")
        sensitivity_experiments = runner.define_parameter_sensitivity()
        
        mode_config = config.get("execution_modes", {}).get(mode, {})
        episode_factor = mode_config.get("episode_factor", 1.0)
        seed_count = mode_config.get("seed_count", 3)
        
        # 根据配置选择启用的参数
        param_config = config.get("dimensions", {}).get("parameter_sensitivity", {}).get("parameters", {})
        
        for param_name, configs in sensitivity_experiments.items():
            if param_config.get(param_name, {}).get("enabled", True):
                print(f"\n  → 参数: {param_name}")
                for cfg in configs:
                    cfg.episodes = int(cfg.episodes * episode_factor)
                    cfg.seeds = cfg.seeds[:seed_count]
                    result = runner.run_experiment(cfg, algorithm="TD3")
                    runner.results[cfg.name] = result
    
    elif dimension == "robustness":
        print("鲁棒性测试: 极端场景验证")
        robustness_configs = runner.define_robustness_tests()
        
        mode_config = config.get("execution_modes", {}).get(mode, {})
        episode_factor = mode_config.get("episode_factor", 1.0)
        seed_count = mode_config.get("seed_count", 3)
        
        for cfg in robustness_configs:
            cfg.episodes = int(cfg.episodes * episode_factor)
            cfg.seeds = cfg.seeds[:seed_count]
            result = runner.run_experiment(cfg, algorithm="TD3")
            runner.results[cfg.name] = result
    
    elif dimension == "convergence":
        print("收敛性分析: 训练稳定性评估")
        convergence_configs = runner.define_convergence_analysis()
        
        for cfg in convergence_configs[:1]:  # 只运行多种子实验
            result = runner.run_experiment(cfg, algorithm="TD3")
            runner.results[cfg.name] = result
    
    elif dimension == "scalability":
        print("可扩展性测试: 大规模场景性能")
        scalability_configs = runner.define_scalability_tests()
        
        mode_config = config.get("execution_modes", {}).get(mode, {})
        episode_factor = mode_config.get("episode_factor", 1.0)
        seed_count = mode_config.get("seed_count", 3)
        
        for cfg in scalability_configs[:3]:  # 运行前3个规模
            cfg.episodes = int(cfg.episodes * episode_factor)
            cfg.seeds = cfg.seeds[:seed_count]
            result = runner.run_experiment(cfg, algorithm="TD3")
            runner.results[cfg.name] = result
    
    else:
        print(f"⚠️ 未知维度: {dimension}")
        return None
    
    runner._save_summary()
    
    print(f"\n✅ 维度 {dimension} 实验完成！")
    print(f"结果保存在: {runner.experiment_dir}")
    
    return runner.experiment_dir


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="TD3综合对比实验启动脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（验证流程）
  python run_td3_comparison.py --mode quick
  
  # 标准实验（论文标准）
  python run_td3_comparison.py --mode standard
  
  # 仅运行消融实验
  python run_td3_comparison.py --mode standard --dimension ablation
  
  # 仅运行参数敏感性分析
  python run_td3_comparison.py --mode standard --dimension sensitivity
  
  # 使用自定义配置
  python run_td3_comparison.py --config my_config.json --mode standard
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="quick",
        choices=["quick", "standard", "extensive"],
        help="实验模式: quick(快速测试), standard(标准实验), extensive(扩展实验)"
    )
    
    parser.add_argument(
        "--dimension",
        type=str,
        default="all",
        choices=["all", "ablation", "sensitivity", "robustness", 
                "convergence", "scalability", "algorithm"],
        help="实验维度选择"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/td3_experiment_config.json",
        help="实验配置文件路径"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印实验计划，不实际运行"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 打印实验计划
    print_experiment_plan(config, args.mode, args.dimension)
    
    if args.dry_run:
        print("\n🔍 Dry-run模式: 仅显示实验计划，不执行")
        return
    
    # 确认执行
    if args.mode in ["standard", "extensive"]:
        mode_config = config.get("execution_modes", {}).get(args.mode, {})
        estimated_hours = mode_config.get("estimated_time_hours", "未知")
        
        print(f"\n⚠️ 注意: 该实验预计需要 {estimated_hours} 小时")
        response = input("是否继续? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return
    
    # 运行实验
    start_time = datetime.now()
    
    if args.mode == "quick" and args.dimension == "all":
        # 快速测试模式
        output_dir = run_quick_test()
    elif args.dimension == "all":
        # 运行完整套件
        runner = TD3ComprehensiveComparison()
        runner.run_full_suite(mode=args.mode)
        output_dir = runner.experiment_dir
    else:
        # 运行特定维度
        output_dir = run_dimension_experiments(args.dimension, args.mode, config)
    
    # 计算总时间
    elapsed_time = datetime.now() - start_time
    hours = elapsed_time.total_seconds() / 3600
    
    print("\n" + "="*80)
    print("🎉 实验完成！")
    print("="*80)
    print(f"总耗时: {hours:.2f} 小时")
    print(f"结果目录: {output_dir}")
    print(f"配置文件: {args.config}")
    print("="*80 + "\n")
    
    # 提示下一步
    print("📊 下一步建议:")
    print(f"  1. 查看结果摘要: {output_dir}/experiment_summary.json")
    print(f"  2. 生成可视化图表: python tools/visualize_td3_results.py --input {output_dir}")
    print(f"  3. 生成LaTeX表格: python tools/generate_latex_tables.py --input {output_dir}")
    print()


if __name__ == "__main__":
    main()

