#!/usr/bin/env python3
"""
CAMTD3 参数敏感性分析 - 批量运行工具
====================================

【功能】
智能化的批量实验运行工具，支持：
- 交互式选择要运行的实验
- 预设的快速/中等/完整模式
- 实时进度显示
- 自动结果汇总
- 错误处理和日志记录

【使用方法】
```bash
# 交互式模式（推荐）
python experiments/camtd3_strategy_suite/run_batch_experiments.py

# 快速测试所有实验（10轮）
python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode quick --all

# 完整实验指定实验
python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode full --experiments 1,2,6,7,8

# 并行运行（如果有多GPU）
python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode medium --parallel 3

# 静默模式
python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode quick --all --silent
```
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import threading
from queue import Queue

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ========== 实验配置 ==========
EXPERIMENTS = {
    1: {
        "name": "数据大小对比",
        "script": "run_data_size_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.5-2h",
        "configs": 5,
    },
    2: {
        "name": "车辆数量对比",
        "script": "run_vehicle_count_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.5-2h",
        "configs": 5,
    },
    3: {
        "name": "本地资源对卸载影响",
        "script": "run_local_resource_offload_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "1.5-2h",
        "configs": 5,
    },
    4: {
        "name": "本地资源对成本影响",
        "script": "run_local_resource_cost_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "2-3h",
        "configs": 7,
    },
    5: {
        "name": "带宽对成本影响",
        "script": "run_bandwidth_cost_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "2-3h",
        "configs": 7,
    },
    6: {
        "name": "边缘节点配置对比",
        "script": "run_edge_node_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.8-2.5h",
        "configs": 6,
        "new": True,
    },
    7: {
        "name": "任务到达率对比",
        "script": "run_task_arrival_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.8-2.5h",
        "configs": 6,
        "new": True,
    },
    8: {
        "name": "移动速度对比",
        "script": "run_mobility_speed_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.8-2.5h",
        "configs": 6,
        "new": True,
    },
}

MODES = {
    "quick": {"episodes": 10, "desc": "快速测试（10轮，约10-20分钟/实验）"},
    "medium": {"episodes": 100, "desc": "中等测试（100轮，约1.5-3小时/实验）"},
    "full": {"episodes": 500, "desc": "完整实验（500轮，约7-15小时/实验）"},
}

# ========== 颜色输出（可选）==========
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    USE_COLOR = True
except ImportError:
    USE_COLOR = False

def colored(text: str, color: str = "") -> str:
    """带颜色的文本输出"""
    if not USE_COLOR or not color:
        return text
    color_map = {
        "red": Fore.RED if USE_COLOR else "",
        "green": Fore.GREEN if USE_COLOR else "",
        "yellow": Fore.YELLOW if USE_COLOR else "",
        "blue": Fore.BLUE if USE_COLOR else "",
        "cyan": Fore.CYAN if USE_COLOR else "",
        "magenta": Fore.MAGENTA if USE_COLOR else "",
    }
    reset = Style.RESET_ALL if USE_COLOR else ""
    return f"{color_map.get(color, '')}{text}{reset}"


def print_banner(text: str, char: str = "="):
    """打印横幅"""
    width = 70
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)


def print_experiments_table():
    """打印实验列表表格"""
    print("\n可用的实验:")
    print("-" * 90)
    print(f"{'#':<4} {'实验名称':<25} {'优先级':<8} {'配置数':<8} {'预计时间(100轮)':<18} {'标记':<6}")
    print("-" * 90)
    
    for exp_id, exp in EXPERIMENTS.items():
        new_tag = colored("⭐NEW", "yellow") if exp.get("new") else ""
        priority_color = "green" if exp["priority"] == "高" else "yellow"
        priority = colored(exp["priority"], priority_color)
        
        print(f"{exp_id:<4} {exp['name']:<25} {priority:<8} {exp['configs']:<8} "
              f"{exp['time_estimate_100ep']:<18} {new_tag:<6}")
    
    print("-" * 90)


def select_experiments_interactive() -> List[int]:
    """交互式选择实验"""
    print_experiments_table()
    
    print("\n选择方式:")
    print("  1. 输入实验编号（用逗号分隔，如: 1,2,6）")
    print("  2. 输入 'all' 运行所有实验")
    print("  3. 输入 'high' 运行高优先级实验")
    print("  4. 输入 'new' 运行新增实验")
    
    while True:
        choice = input("\n请选择要运行的实验: ").strip().lower()
        
        if choice == "all":
            return list(EXPERIMENTS.keys())
        elif choice == "high":
            return [k for k, v in EXPERIMENTS.items() if v["priority"] == "高"]
        elif choice == "new":
            return [k for k, v in EXPERIMENTS.items() if v.get("new")]
        elif choice:
            try:
                selected = [int(x.strip()) for x in choice.split(",")]
                invalid = [x for x in selected if x not in EXPERIMENTS]
                if invalid:
                    print(colored(f"错误: 无效的实验编号 {invalid}", "red"))
                    continue
                return selected
            except ValueError:
                print(colored("错误: 请输入有效的编号或选项", "red"))
                continue
        else:
            print(colored("错误: 请输入选择", "red"))


def select_mode_interactive() -> Dict[str, Any]:
    """交互式选择运行模式"""
    print("\n运行模式:")
    for mode_key, mode_info in MODES.items():
        print(f"  {mode_key}: {mode_info['desc']}")
    
    while True:
        choice = input("\n选择运行模式 (quick/medium/full): ").strip().lower()
        if choice in MODES:
            return {"key": choice, **MODES[choice]}
        print(colored("错误: 请选择 quick, medium 或 full", "red"))


def run_single_experiment(
    exp_id: int,
    mode: Dict[str, Any],
    suite_id: str,
    seed: int,
    silent: bool,
    output_queue: Optional[Queue] = None,
) -> Dict[str, Any]:
    """
    运行单个实验
    
    【参数】
    exp_id: int - 实验编号
    mode: Dict - 运行模式配置
    suite_id: str - Suite标识符
    seed: int - 随机种子
    silent: bool - 静默模式
    output_queue: Queue - 输出队列（用于并行运行）
    
    【返回值】
    Dict - 运行结果
    """
    exp = EXPERIMENTS[exp_id]
    script_name = exp["script"]
    script_path = script_dir / script_name
    
    # 构建命令
    cmd = [
        "python",
        str(script_path),
        "--episodes", str(mode["episodes"]),
        "--seed", str(seed),
        "--suite-id", f"{suite_id}_{exp_id}_{exp['name'].replace(' ', '_')}",
    ]
    
    if silent:
        cmd.append("--silent")
    
    # 运行实验
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=7200,  # 2小时超时
        )
        
        elapsed = time.time() - start_time
        
        status = {
            "exp_id": exp_id,
            "name": exp["name"],
            "success": result.returncode == 0,
            "elapsed_time": elapsed,
            "returncode": result.returncode,
            "stdout": result.stdout if not silent else "",
            "stderr": result.stderr if result.returncode != 0 else "",
        }
        
        if output_queue:
            output_queue.put(status)
        
        return status
        
    except subprocess.TimeoutExpired:
        status = {
            "exp_id": exp_id,
            "name": exp["name"],
            "success": False,
            "elapsed_time": 7200,
            "returncode": -1,
            "error": "Timeout (>2 hours)",
        }
        
        if output_queue:
            output_queue.put(status)
        
        return status
    
    except Exception as e:
        status = {
            "exp_id": exp_id,
            "name": exp["name"],
            "success": False,
            "elapsed_time": time.time() - start_time,
            "returncode": -2,
            "error": str(e),
        }
        
        if output_queue:
            output_queue.put(status)
        
        return status


def run_experiments_sequential(
    exp_ids: List[int],
    mode: Dict[str, Any],
    suite_id: str,
    seed: int,
    silent: bool,
) -> List[Dict[str, Any]]:
    """顺序运行实验"""
    results = []
    total = len(exp_ids)
    
    print_banner("开始顺序运行实验")
    
    for i, exp_id in enumerate(exp_ids, 1):
        exp = EXPERIMENTS[exp_id]
        
        print(f"\n[{i}/{total}] 运行实验: {colored(exp['name'], 'cyan')}")
        print(f"  脚本: {exp['script']}")
        print(f"  配置数: {exp['configs']}, 每配置轮数: {mode['episodes']}")
        print(f"  预计时间: {exp['time_estimate_100ep']} (基于100轮)")
        print("-" * 70)
        
        result = run_single_experiment(exp_id, mode, suite_id, seed, silent)
        results.append(result)
        
        # 打印结果
        if result["success"]:
            print(colored(f"✓ 完成! 用时: {result['elapsed_time']/60:.1f} 分钟", "green"))
        else:
            print(colored(f"✗ 失败! 错误: {result.get('error', result.get('stderr', 'Unknown'))}", "red"))
    
    return results


def run_experiments_parallel(
    exp_ids: List[int],
    mode: Dict[str, Any],
    suite_id: str,
    seed: int,
    silent: bool,
    max_parallel: int = 2,
) -> List[Dict[str, Any]]:
    """并行运行实验"""
    print_banner(f"并行运行实验（最多{max_parallel}个同时）")
    
    results = []
    output_queue = Queue()
    
    def worker(exp_id: int):
        run_single_experiment(exp_id, mode, suite_id, seed, silent, output_queue)
    
    # 分批运行
    batches = [exp_ids[i:i+max_parallel] for i in range(0, len(exp_ids), max_parallel)]
    
    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n批次 {batch_idx}/{len(batches)}: 运行 {len(batch)} 个实验")
        
        threads = []
        for exp_id in batch:
            exp = EXPERIMENTS[exp_id]
            print(f"  启动: {exp['name']}")
            
            thread = threading.Thread(target=worker, args=(exp_id,))
            thread.start()
            threads.append(thread)
        
        # 等待批次完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        while not output_queue.empty():
            result = output_queue.get()
            results.append(result)
            
            status_icon = colored("✓", "green") if result["success"] else colored("✗", "red")
            print(f"  {status_icon} {result['name']}: {result['elapsed_time']/60:.1f} 分钟")
    
    return results


def print_summary(results: List[Dict[str, Any]], mode: Dict[str, Any], suite_id: str):
    """打印运行摘要"""
    print_banner("运行摘要", "=")
    
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    total_time = sum(r["elapsed_time"] for r in results)
    
    print(f"\nSuite ID: {colored(suite_id, 'cyan')}")
    print(f"运行模式: {mode['key']} ({mode['episodes']} 轮/配置)")
    print(f"\n总实验数: {total}")
    print(f"  {colored('✓', 'green')} 成功: {success}")
    if failed > 0:
        print(f"  {colored('✗', 'red')} 失败: {failed}")
    print(f"\n总用时: {total_time/3600:.2f} 小时 ({total_time/60:.1f} 分钟)")
    
    # 详细结果表
    print("\n详细结果:")
    print("-" * 80)
    print(f"{'#':<4} {'实验名称':<28} {'状态':<10} {'用时':<15}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: x["exp_id"]):
        status = colored("成功", "green") if r["success"] else colored("失败", "red")
        time_str = f"{r['elapsed_time']/60:.1f} min"
        print(f"{r['exp_id']:<4} {r['name']:<28} {status:<10} {time_str:<15}")
    
    print("-" * 80)
    
    # 失败详情
    failed_results = [r for r in results if not r["success"]]
    if failed_results:
        print(f"\n{colored('失败实验详情:', 'red')}")
        for r in failed_results:
            print(f"\n实验 {r['exp_id']}: {r['name']}")
            print(f"  错误: {r.get('error', r.get('stderr', 'Unknown'))}")
    
    # 结果位置
    print(f"\n{colored('结果位置:', 'green')}")
    print(f"  results/parameter_sensitivity/{suite_id}_*/")
    print(f"  - summary.json (汇总数据)")
    print(f"  - *.png (对比图表)")
    
    # 保存摘要
    summary_file = Path("results/parameter_sensitivity") / f"{suite_id}_batch_summary.json"
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    
    summary_data = {
        "suite_id": suite_id,
        "mode": mode["key"],
        "episodes": mode["episodes"],
        "total_experiments": total,
        "successful": success,
        "failed": failed,
        "total_time_seconds": total_time,
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }
    
    summary_file.write_text(json.dumps(summary_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n批量运行摘要已保存: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="CAMTD3 参数敏感性分析 - 批量运行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 交互式模式
  python run_batch_experiments.py
  
  # 快速测试所有实验
  python run_batch_experiments.py --mode quick --all
  
  # 完整实验（高优先级）
  python run_batch_experiments.py --mode full --high-priority
  
  # 指定实验
  python run_batch_experiments.py --mode medium --experiments 1,2,6,7,8
  
  # 并行运行
  python run_batch_experiments.py --mode medium --all --parallel 3
        """
    )
    
    parser.add_argument("--mode", choices=["quick", "medium", "full"], 
                       help="运行模式 (quick=10轮, medium=100轮, full=500轮)")
    parser.add_argument("--experiments", type=str,
                       help="指定实验编号（逗号分隔，如: 1,2,6）")
    parser.add_argument("--all", action="store_true",
                       help="运行所有实验")
    parser.add_argument("--high-priority", action="store_true",
                       help="运行高优先级实验")
    parser.add_argument("--new-only", action="store_true",
                       help="仅运行新增实验")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--suite-id", type=str,
                       help="Suite标识符前缀 (默认: 自动生成时间戳)")
    parser.add_argument("--parallel", type=int, metavar="N",
                       help="并行运行最多N个实验（需要多GPU支持）")
    parser.add_argument("--silent", action="store_true",
                       help="静默模式（减少输出）")
    parser.add_argument("--non-interactive", action="store_true",
                       help="非交互模式（必须指定所有参数）")
    
    args = parser.parse_args()
    
    # ========== 交互式或命令行模式 ==========
    if args.non_interactive or (args.mode and (args.all or args.experiments or args.high_priority or args.new_only)):
        # 非交互模式
        if not args.mode:
            print(colored("错误: 非交互模式必须指定 --mode", "red"))
            sys.exit(1)
        
        mode = {"key": args.mode, **MODES[args.mode]}
        
        # 选择实验
        if args.all:
            exp_ids = list(EXPERIMENTS.keys())
        elif args.high_priority:
            exp_ids = [k for k, v in EXPERIMENTS.items() if v["priority"] == "高"]
        elif args.new_only:
            exp_ids = [k for k, v in EXPERIMENTS.items() if v.get("new")]
        elif args.experiments:
            exp_ids = [int(x.strip()) for x in args.experiments.split(",")]
        else:
            print(colored("错误: 必须指定 --all, --high-priority, --new-only 或 --experiments", "red"))
            sys.exit(1)
    else:
        # 交互模式
        print_banner("CAMTD3 参数敏感性分析 - 批量运行工具")
        exp_ids = select_experiments_interactive()
        mode = select_mode_interactive()
    
    # 生成suite_id
    if args.suite_id:
        suite_id = args.suite_id
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suite_id = f"batch_{mode['key']}_{timestamp}"
    
    # 确认运行
    print(f"\n{colored('即将运行:', 'yellow')}")
    print(f"  实验数量: {len(exp_ids)}")
    print(f"  实验列表: {', '.join([EXPERIMENTS[i]['name'] for i in exp_ids])}")
    print(f"  运行模式: {mode['key']} ({mode['episodes']} 轮/配置)")
    print(f"  Suite ID: {suite_id}")
    print(f"  随机种子: {args.seed}")
    if args.parallel:
        print(f"  并行数: {args.parallel}")
    
    # 估算时间
    total_configs = sum(EXPERIMENTS[i]["configs"] for i in exp_ids)
    est_time_min = total_configs * mode["episodes"] * 0.5 / 60  # 粗略估算
    est_time_max = total_configs * mode["episodes"] * 1.0 / 60
    print(f"  预计时间: {est_time_min:.1f}-{est_time_max:.1f} 小时")
    
    if not args.non_interactive:
        confirm = input(f"\n{colored('确认开始运行? (y/n): ', 'yellow')}").strip().lower()
        if confirm != 'y':
            print("已取消")
            sys.exit(0)
    
    # 运行实验
    start_time = time.time()
    
    if args.parallel and args.parallel > 1:
        results = run_experiments_parallel(exp_ids, mode, suite_id, args.seed, args.silent, args.parallel)
    else:
        results = run_experiments_sequential(exp_ids, mode, suite_id, args.seed, args.silent)
    
    # 打印摘要
    print_summary(results, mode, suite_id)
    
    # 返回码
    failed_count = sum(1 for r in results if not r["success"])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()

