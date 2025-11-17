#!/usr/bin/env python3
"""
TD3 四个核心参数敏感性实验批量运行脚本
==========================================

【实验列表】
1. 带宽成本对比 (10-50MHz，5个配置)
2. 任务到达率对比 (1.0-2.5 tasks/s，4个配置)
3. 数据大小对比 (100-600KB，3个配置)
4. 本地计算资源对比 (1.2-2.8GHz，3个配置)

【运行模式】
- 默认：400 episodes/配置，静默模式
- 可自定义轮数和是否静默
- 自动生成唯一的suite-id

【使用示例】
```bash
# 默认运行（400轮）
python experiments/td3_strategy_suite/run_four_key_experiments.py

# 快速测试（10轮）
python experiments/td3_strategy_suite/run_four_key_experiments.py --episodes 10

# 完整实验（800轮）
python experiments/td3_strategy_suite/run_four_key_experiments.py --episodes 800

# 显示详细日志
python experiments/td3_strategy_suite/run_four_key_experiments.py --no-silent
```
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# ========== 实验配置 ==========
EXPERIMENTS = [
    {
        "name": "任务到达率对比",
        "script": "run_task_arrival_comparison.py",
        "extra_args": [],  # 使用默认的 1.0,1.5,2.0,2.5
        "suite_prefix": "arrival",
        "description": "4个到达率 (1.0-2.5 tasks/s)",
    },
    {
        "name": "数据大小对比",
        "script": "run_data_size_comparison.py",
        "extra_args": [],  # 使用默认的 Light/Standard/Heavy
        "suite_prefix": "datasize",
        "description": "3个数据大小 (100-600KB)",
    },
    {
        "name": "本地计算资源对比",
        "script": "run_local_compute_resource_comparison.py",
        "extra_args": [],  # 使用默认的 1.2,2.0,2.8 GHz
        "suite_prefix": "local",
        "description": "3个CPU频率 (1.2-2.8GHz)",
    },
    {
        "name": "带宽成本对比",
        "script": "run_bandwidth_cost_comparison.py",
        "extra_args": ["--bandwidths", "10,20,30,40,50"],
        "suite_prefix": "bw",
        "description": "5个带宽配置 (10-50MHz)",
    },
]


def run_experiment(
    script_name: str,
    extra_args: List[str],
    episodes: int,
    silent: bool,
    suite_id: str,
    experiment_dir: Path,
    central_resource: bool = False,
) -> bool:
    """
    运行单个实验
    
    Args:
        central_resource: 是否启用中央资源分配架构
    
    返回：
        True 表示成功，False 表示失败
    """
    script_path = experiment_dir / script_name
    
    cmd = [
        sys.executable,  # 使用当前Python解释器
        str(script_path),
        "--episodes", str(episodes),
        "--suite-id", suite_id,
    ]
    
    # 添加额外参数
    cmd.extend(extra_args)
    
    # 🎯 添加中央资源分配模式
    if central_resource:
        cmd.append("--central-resource")
    
    # 添加静默模式
    if silent:
        cmd.append("--silent")
    else:
        cmd.append("--no-silent")
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验执行失败！错误码: {e.returncode}")
        return False
    except Exception as e:
        print(f"❌ 实验执行异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量运行TD3核心参数敏感性实验",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
实验列表：
  1. 带宽成本对比 (10-50MHz，5个配置)
  2. 任务到达率对比 (1.0-2.5 tasks/s，4个配置)
  3. 数据大小对比 (100-600KB，3个配置)
  4. 本地计算资源对比 (1.2-2.8GHz，3个配置)

示例：
  python %(prog)s                    # 运行400轮
  python %(prog)s --episodes 10      # 快速测试10轮
  python %(prog)s --no-silent        # 显示详细日志
        """
    )
    
    parser.add_argument(
        "--episodes",
        type=int,
        default=1500,
        help="每个实验的训练轮数 (默认: 1500，建议≥1500确保TD3充分收敛)",
    )
    parser.add_argument(
        "--silent",
        action="store_true",
        default=True,
        help="静默模式，不显示详细训练日志 (默认)",
    )
    parser.add_argument(
        "--no-silent",
        action="store_false",
        dest="silent",
        help="显示详细训练日志",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1,2,3,4",
        help="要运行的实验编号，逗号分隔 (默认: 1,2,3,4)",
    )
    parser.add_argument(
        "--central-resource",
        action="store_true",
        help="启用中央资源分配架构（Phase 1决策 + Phase 2执行），对比分层模式 vs 标准模式",
    )
    parser.add_argument(
        "--compare-modes",
        action="store_true",
        help="对比运行两种模式：标准模式 + 分层模式（自动运行2倍实验）",
    )
    
    args = parser.parse_args()
    
    # 🎯 训练轮数检查：确保策略充分收敛
    if args.episodes < 1500:
        print("\n" + "="*70)
        print("⚠️  训练轮数警告")
        print("="*70)
        print(f"当前配置轮数: {args.episodes}")
        print(f"建议最低轮数: 1500")
        print()
        print("【风险提示】")
        print("  - TD3算法收敛较慢，<1500轮可能导致策略未充分收敛")
        print("  - 在不同RSU资源配置下，低轮数影响策略质量和结果稳定性")
        print("  - 实验结果可能出现性能异常或波动过大")
        print()
        print("【推荐配置】")
        print("  - 正式实验: --episodes 1500 或更高")
        print("  - 快速验证: --episodes 500（仅用于代码调试）")
        print()
        print("示例命令:")
        print(f"  python {Path(__file__).name} --episodes 1500")
        print("="*70)
        
        # 倒计时确认（给用户5秒考虑）
        import time
        for i in range(5, 0, -1):
            print(f"\r将在 {i} 秒后继续执行...", end="", flush=True)
            time.sleep(1)
        print("\r执行中...                    ")
        print()
    
    # 解析要运行的实验
    exp_indices = [int(x.strip()) - 1 for x in args.experiments.split(",") if x.strip()]
    selected_experiments = [EXPERIMENTS[i] for i in exp_indices if 0 <= i < len(EXPERIMENTS)]
    
    if not selected_experiments:
        print("❌ 没有选择有效的实验！")
        return 1
    
    # 生成时间戳
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 获取实验脚本目录
    script_dir = Path(__file__).resolve().parent
    
    # 打印实验信息
    print("=" * 70)
    print("TD3 四个核心参数敏感性实验")
    print("=" * 70)
    print()
    print(f"实验轮数: {args.episodes} episodes/配置")
    print(f"运行模式: {'静默模式' if args.silent else '详细日志模式'}")
    
    # 🎯 显示架构模式
    if args.compare_modes:
        print(f"架构模式: 🔄 对比模式（标准 + 分层）")
        print(f"  ├─ 标准模式: 固定资源分配")
        print(f"  └─ 分层模式: 中央智能体动态资源分配")
    elif args.central_resource:
        print(f"架构模式: 🎯 分层模式（中央资源分配）")
    else:
        print(f"架构模式: 📊 标准模式（固定资源分配）")
    
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Suite标识: *_{timestamp}")
    print()
    print("实验列表:")
    for idx, exp in enumerate(selected_experiments, 1):
        print(f"  [{idx}/{len(selected_experiments)}] {exp['name']} - {exp['description']}")
    print()
    print("=" * 70)
    print()
    
    # 记录开始时间
    start_time = datetime.now()
    
    # 🎯 确定要运行的模式
    modes_to_run = []
    if args.compare_modes:
        # 对比模式：运行标准 + 分层
        modes_to_run = [
            {"name": "标准模式", "central_resource": False, "suffix": "standard"},
            {"name": "分层模式", "central_resource": True, "suffix": "central"},
        ]
    else:
        # 单模式
        modes_to_run = [
            {
                "name": "分层模式" if args.central_resource else "标准模式",
                "central_resource": args.central_resource,
                "suffix": "central" if args.central_resource else "standard",
            }
        ]
    
    # 运行实验
    results = []
    total_experiments = len(selected_experiments) * len(modes_to_run)
    exp_counter = 0
    
    for mode in modes_to_run:
        if len(modes_to_run) > 1:
            print(f"\n{'='*70}")
            print(f"🔄 开始运行: {mode['name']}")
            print(f"{'='*70}\n")
        
        for idx, exp in enumerate(selected_experiments, 1):
            exp_counter += 1
            print(f"[{exp_counter}/{total_experiments}] 运行 {exp['name']} ({mode['name']})...")
            print("-" * 70)
            
            suite_id = f"{exp['suite_prefix']}_{mode['suffix']}_{timestamp}"
            
            success = run_experiment(
                script_name=exp["script"],
                extra_args=exp["extra_args"],
                episodes=args.episodes,
                silent=args.silent,
                suite_id=suite_id,
                experiment_dir=script_dir,
                central_resource=mode["central_resource"],
            )
            
            results.append({
                "name": exp["name"],
                "mode": mode["name"],
                "success": success,
                "suite_id": suite_id,
            })
            
            if success:
                print(f"✅ [{exp_counter}/{total_experiments}] 完成！")
            else:
                print(f"❌ [{exp_counter}/{total_experiments}] 失败！")
            
            print()
    
    # 计算耗时
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    # 打印总结
    print("=" * 70)
    print("实验批量运行完成！")
    print("=" * 70)
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总耗时: {elapsed}")
    print()
    print("实验结果:")
    
    # 按模式分组显示
    if args.compare_modes:
        for mode in modes_to_run:
            mode_name = mode["name"]
            mode_results = [r for r in results if r["mode"] == mode_name]
            print(f"\n  {mode_name}:")
            for idx, result in enumerate(mode_results, 1):
                status = "✅ 成功" if result["success"] else "❌ 失败"
                print(f"    [{idx}] {result['name']}: {status}")
                print(f"        Suite ID: {result['suite_id']}")
    else:
        for idx, result in enumerate(results, 1):
            status = "✅ 成功" if result["success"] else "❌ 失败"
            print(f"  [{idx}] {result['name']}: {status}")
            print(f"      Suite ID: {result['suite_id']}")
    
    print()
    print("结果保存在: results/parameter_sensitivity/")
    
    # 🎯 如果是对比模式，提示对比分析
    if args.compare_modes:
        print()
        print("💡 对比分析提示:")
        print("  可以使用相同的suite_prefix但不同的suffix来对比：")
        print("  - *_standard_* 文件：标准模式结果")
        print("  - *_central_* 文件：分层模式结果")
    for result in results:
        print(f"  - {result['suite_id']}/")
    print()
    print("=" * 70)
    
    # 检查是否全部成功
    all_success = all(r["success"] for r in results)
    if all_success:
        print("🎉 所有实验均成功完成！")
        return 0
    else:
        failed_count = sum(1 for r in results if not r["success"])
        print(f"⚠️  有 {failed_count} 个实验失败，请检查日志！")
        return 1


if __name__ == "__main__":
    sys.exit(main())

