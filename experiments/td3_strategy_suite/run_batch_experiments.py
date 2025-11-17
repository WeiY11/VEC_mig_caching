#!/usr/bin/env python3
"""
TD3 参数敏感性分析 - 批量运行工具
====================================

【功能】
智能化的批量实验运行工具，支持：
- 交互式选择要运行的实验
- 预设的快速/中等/完整模式
- 实时进度显示
- 自动结果汇总
- 错误处理和日志记录

【默认运行模式】
✅ 自动静默运行，无需手动交互
✅ 自动保存所有训练报告
✅ 支持长时间无人值守运行

【使用方法】
```bash
# ✅ 默认静默运行（无需手动交互，推荐）
# 快速测试所有实验（10轮）
python experiments/td3_strategy_suite/run_batch_experiments.py --mode quick --all

# 完整实验指定实验 - 自动保存报告，无人值守运行
python experiments/td3_strategy_suite/run_batch_experiments.py --mode full --experiments 1,2,6,7,8

# 并行运行（如果有多GPU）
python experiments/td3_strategy_suite/run_batch_experiments.py --mode medium --parallel 3 --all

# 💡 如需交互式确认，添加 --interactive 参数
python experiments/td3_strategy_suite/run_batch_experiments.py --mode full --all --interactive
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
# 🔄 已整合：18个实验 → 14个实验（合并7个→3个）
# ✅ 优化效果：配置数 95→37 (节省61%), 训练时间 285h→111h (节省61%)

EXPERIMENTS = {
    # ========== 系统规模与拓扑 (4个) ==========
    1: {
        "name": "车辆数量对比",
        "script": "run_vehicle_count_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "0.9-1.2h",
        "configs": 3,
        "category": "系统规模",
    },
    2: {
        "name": "边缘节点配置对比",
        "script": "run_edge_node_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "0.8-1.3h",
        "configs": 3,
        "category": "系统规模",
    },
    3: {
        "name": "网络与拓扑综合对比",  # 🆕 合并实验
        "script": "run_network_topology_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.2-1.6h",
        "configs": 6,  # 6个综合场景
        "category": "系统规模",
        "merged": True,
        "merged_from": ["带宽", "信道质量", "拓扑密度"],
    },
    4: {
        "name": "移动速度对比",
        "script": "run_mobility_speed_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "0.8-1.3h",
        "configs": 3,
        "category": "系统规模",
    },
    
    # ========== 任务特性 (3个) ==========
    5: {
        "name": "任务到达率对比",
        "script": "run_task_arrival_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "0.8-1.3h",
        "configs": 3,
        "category": "任务特性",
    },
    6: {
        "name": "任务复杂度对比",
        "script": "run_task_complexity_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "1.0-1.4h",
        "configs": 3,
        "category": "任务特性",
    },
    7: {
        "name": "数据大小对比",
        "script": "run_data_size_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "0.9-1.2h",
        "configs": 3,
        "category": "任务特性",
    },
    
    # ========== 资源配置 (3个) ==========
    8: {
        "name": "本地计算资源综合对比",  # 🆕 合并实验
        "script": "run_local_compute_resource_comparison.py",
        "priority": "中",
        "time_estimate_100ep": "0.9-1.3h",
        "configs": 3,
        "category": "资源配置",
        "merged": True,
        "merged_from": ["本地资源成本", "本地资源卸载"],
    },
    9: {
        "name": "边缘基础设施综合对比",  # 🆕 合并实验
        "script": "run_edge_infrastructure_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.0-1.4h",
        "configs": 5,  # 5个综合场景
        "category": "资源配置",
        "merged": True,
        "merged_from": ["边缘计算能力", "边缘通信资源"],
    },
    10: {
        "name": "缓存容量对比",
        "script": "run_cache_capacity_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.0-1.4h",
        "configs": 3,
        "category": "资源配置",
    },
    
    # ========== 综合场景 (2个) ==========
    11: {
        "name": "混合负载场景对比",
        "script": "run_mixed_workload_comparison.py",
        "priority": "高",
        "time_estimate_100ep": "1.0-1.4h",
        "configs": 3,
        "category": "综合场景",
    },
    12: {
        "name": "Pareto权重分析",
        "script": "run_pareto_weight_analysis.py",
        "priority": "高",
        "time_estimate_100ep": "1.2-1.8h",
        "configs": 3,
        "category": "综合场景",
    },
    
    # ========== 其他实验 (2个, 优先级较低) ==========
    13: {
        "name": "服务能力扩展对比",
        "script": "run_service_capacity_comparison.py",
        "priority": "低",
        "time_estimate_100ep": "1.0-1.4h",
        "configs": 3,
        "category": "其他",
    },
    14: {
        "name": "资源异构性对比",
        "script": "run_resource_heterogeneity_comparison.py",
        "priority": "低",
        "time_estimate_100ep": "0.9-1.4h",
        "configs": 3,
        "category": "其他",
    },
}


MODES = {
    "quick": {"episodes": 500, "desc": "快速验证（500轮，仅用于代码调试，约20-40分钟/实验）"},
    "medium": {"episodes": 1000, "desc": "中等测试（1000轮，约2-4小时/实验）"},
    "full": {"episodes": 1500, "desc": "完整实验（1500轮，建议轮数，约3-6小时/实验）"},
}

# ========== 颜色输出（可选）==========
try:
    from colorama import init, Fore, Style
    init(autoreset=True)
    USE_COLOR = True
except ImportError:
    USE_COLOR = False
    # 为类型检查器提供占位类
    class Fore:  # type: ignore
        RED = GREEN = YELLOW = BLUE = CYAN = MAGENTA = ""
    class Style:  # type: ignore
        RESET_ALL = ""

def colored(text: str, color: str = "") -> str:
    """
    为文本添加颜色（如果终端支持）
    
    【功能】
    - 使用colorama库为终端输出添加颜色
    - 如果colorama未安装，则返回原始文本
    - 支持多种预定义颜色
    
    【参数】
    - text: str - 要着色的文本
    - color: str - 颜色名称（red/green/yellow/blue/cyan/magenta）
    
    【返回值】
    - str - 带颜色的文本（或原始文本）
    
    【使用示例】
    >>> colored("成功", "green")
    >>> colored("警告", "yellow")
    >>> colored("错误", "red")
    """
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
    """
    打印带边框的横幅标题
    
    【功能】
    - 创建视觉上突出的标题横幅
    - 用于区分不同的执行阶段
    
    【参数】
    - text: str - 横幅文本
    - char: str - 边框字符（默认"="）
    
    【使用示例】
    >>> print_banner("开始训练")
    >>> print_banner("运行摘要", "#")
    """
    width = 70
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)


def print_experiments_table():
    """
    打印实验列表表格
    
    【功能】
    - 以表格形式展示所有可用实验
    - 包含实验编号、名称、优先级、配置数、预计时间等信息
    - 使用颜色高亮不同优先级
    - 显示特殊标记（NEW、SOLID、MERGED）
    
    【显示内容】
    - #: 实验编号
    - 实验名称: 实验描述
    - 优先级: 高/中/低
    - 配置数: 参数配置数量
    - 预计时间: 基于100轮的时间估算
    - 标记: 特殊标记
    
    【使用场景】
    - 交互式选择实验时展示
    - 帮助用户了解可用实验
    """
    print("\n可用的实验:")
    print("-" * 100)
    print(f"{'#':<4} {'实验名称':<25} {'优先级':<8} {'配置数':<8} {'预计时间(100轮)':<18} {'标记':<10}")
    print("-" * 100)
    
    for exp_id, exp in EXPERIMENTS.items():
        tags = []
        if exp.get("new"):
            tags.append(colored("[NEW]", "yellow"))
        if exp.get("solid"):
            tags.append(colored("[SOLID]", "green"))
        tag_str = " ".join(tags)
        
        priority_color = "green" if exp["priority"] == "高" else "yellow"
        priority = colored(exp["priority"], priority_color)
        
        print(f"{exp_id:<4} {exp['name']:<25} {priority:<8} {exp['configs']:<8} "
              f"{exp['time_estimate_100ep']:<18} {tag_str:<10}")
    
    print("-" * 100)


def select_experiments_interactive() -> List[int]:
    """
    交互式选择要运行的实验
    
    【功能】
    - 显示实验列表表格
    - 提供多种选择方式（编号/全部/筛选）
    - 验证用户输入的有效性
    - 支持快捷选择（all/high/new/solid/merged）
    
    【返回值】
    - List[int] - 选中的实验编号列表
    
    【选择方式】
    1. 输入实验编号（用逗号分隔，如: 1,2,6）
    2. 输入 'all' 运行所有14个实验
    3. 输入 'high' 运行高优先级实验（核心实验）
    4. 输入 'new' 运行新增实验（仅限新开发的）
    5. 输入 'solid' 运行SOLID标记实验（学术化对比方案）
    6. 输入 'merged' 运行合并实验（3个综合对比实验）
    
    【使用示例】
    >>> select_experiments_interactive()  # 交互式选择
    请选择要运行的实验: 1,2,6  # 运行实验1, 2, 6
    请选择要运行的实验: all   # 运行所有实验
    请选择要运行的实验: high  # 运行高优先级实验
    """
    print_experiments_table()
    
    print("\n选择方式:")
    print("  1. 输入实验编号（用逗号分隔，如: 1,2,6）")
    print("  2. 输入 'all' 运行所有14个实验")
    print("  3. 输入 'high' 运行高优先级实验（核心12个）")
    print("  4. 输入 'new' 运行新增实验")
    print("  5. 输入 'solid' 运行SOLID标记实验（学术化对比）")
    print("  6. 输入 'merged' 运行3个合并实验（综合对比）")
    
    while True:
        choice = input("\n请选择要运行的实验: ").strip().lower()
        
        if choice == "all":
            return list(EXPERIMENTS.keys())
        elif choice == "high":
            return [k for k, v in EXPERIMENTS.items() if v["priority"] == "高"]
        elif choice == "new":
            return [k for k, v in EXPERIMENTS.items() if v.get("new")]
        elif choice == "solid":
            return [k for k, v in EXPERIMENTS.items() if v.get("solid")]
        elif choice == "merged":
            # 返回3个合并实验 (ID: 3, 8, 9)
            return [k for k, v in EXPERIMENTS.items() if v.get("merged")]
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
    """
    交互式选择运行模式
    
    【功能】
    - 显示可用运行模式
    - 接受用户选择
    - 返回模式配置
    
    【返回值】
    - Dict[str, Any] - 模式配置
      - key: str - 模式名称 (quick/medium/full)
      - episodes: int - 训练轮数
      - desc: str - 模式描述
    
    【可用模式】
    - quick: 10轮 - 快速测试（约10-20分钟/实验）
    - medium: 100轮 - 中等测试（约1.5-3小时/实验）
    - full: 500轮 - 完整实验（约7-15小时/实验）
    
    【使用示例】
    >>> mode = select_mode_interactive()
    >>> print(mode["episodes"])  # 10 / 100 / 500
    """
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
    运行单个实验脚本
    
    【功能】
    - 构建Python命令调用指定实验脚本
    - 传递必要参数（episodes、seed、suite-id）
    - 设置正确的PYTHONPATH环境变量
    - 实时显示训练输出
    - 处理超时和异常情况
    - 记录运行状态和耗时
    
    【参数】
    - exp_id: int - 实验编号（1-14）
    - mode: Dict[str, Any] - 运行模式配置
      - episodes: int - 训练轮数
      - key: str - 模式名称
    - suite_id: str - Suite标识符（用于组织结果）
    - seed: int - 随机种子（保证可重复性）
    - silent: bool - 静默模式（减少输出）
    - output_queue: Optional[Queue] - 输出队列（并行运行时使用）
    
    【返回值】
    - Dict[str, Any] - 运行结果
      - exp_id: int - 实验编号
      - name: str - 实验名称
      - success: bool - 是否成功
      - elapsed_time: float - 耗时（秒）
      - returncode: int - 返回码
      - error: str - 错误信息（如果失败）
    
    【超时设置】
    - 单个实验最长运行时间: 2小时
    - 超时后自动终止并标记失败
    
    【使用示例】
    >>> result = run_single_experiment(
    ...     exp_id=1,
    ...     mode={"key": "quick", "episodes": 10},
    ...     suite_id="test_20251103",
    ...     seed=42,
    ...     silent=True
    ... )
    >>> print(result["success"])  # True/False
    >>> print(result["elapsed_time"])  # 耗时（秒）
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
    
    # 设置环境变量，确保子进程能找到项目模块
    import os
    env = os.environ.copy()
    env['PYTHONPATH'] = str(project_root)
    
    # 运行实验
    start_time = time.time()
    
    try:
        # ========== 实时输出模式 ==========
        # 不使用 capture_output，而是实时显示输出
        print(f"\n{'='*70}")
        print(f"开始训练: {exp['name']}")
        print(f"{'='*70}")
        
        result = subprocess.run(
            cmd,
            env=env,  # 传递PYTHONPATH
            text=True,
            encoding='utf-8',  # 修复Windows编码问题
            errors='replace',   # 替换无法解码的字符
            timeout=7200,  # 2小时超时
        )
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"完成: {exp['name']} (用时: {elapsed/60:.1f} 分钟)")
        print(f"{'='*70}\n")
        
        status = {
            "exp_id": exp_id,
            "name": exp["name"],
            "success": result.returncode == 0,
            "elapsed_time": elapsed,
            "returncode": result.returncode,
            "stdout": "",  # 已经实时输出，不需要保存
            "stderr": "",
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
    """
    顺序运行多个实验（串行执行）
    
    【功能】
    - 按顺序依次运行每个实验
    - 显示详细的进度信息
    - 实时报告每个实验的结果
    - 显示剩余待运行实验数量
    - 收集所有实验的运行结果
    
    【参数】
    - exp_ids: List[int] - 要运行的实验编号列表
    - mode: Dict[str, Any] - 运行模式配置
    - suite_id: str - Suite标识符
    - seed: int - 随机种子
    - silent: bool - 静默模式
    
    【返回值】
    - List[Dict[str, Any]] - 所有实验的运行结果列表
    
    【优点】
    - 稳定可靠，不需要多GPU
    - 输出清晰，易于调试
    - 资源占用稳定
    
    【缺点】
    - 总耗时较长
    - GPU利用率可能不高（如果有多GPU）
    
    【推荐场景】
    - 单GPU环境
    - 需要详细输出日志
    - 调试实验脚本
    
    【使用示例】
    >>> results = run_experiments_sequential(
    ...     exp_ids=[1, 2, 3],
    ...     mode={"key": "quick", "episodes": 10},
    ...     suite_id="test",
    ...     seed=42,
    ...     silent=False
    ... )
    >>> print(f"成功: {sum(1 for r in results if r['success'])}/{len(results)}")
    """
    results = []
    total = len(exp_ids)
    
    print_banner("开始顺序运行实验")
    
    # 显示实验计划
    print(f"\n{'='*70}")
    print(f"[计划] 实验计划总览")
    print(f"{'='*70}")
    for idx, exp_id in enumerate(exp_ids, 1):
        exp = EXPERIMENTS[exp_id]
        print(f"  {idx}. {exp['name']} ({exp['configs']}个配置 x {mode['episodes']}轮)")
    print(f"{'='*70}\n")
    
    for i, exp_id in enumerate(exp_ids, 1):
        exp = EXPERIMENTS[exp_id]
        
        print(f"\n{'#'*70}")
        print(f"[进度] [{i}/{total}] - {colored(exp['name'], 'cyan')}")
        print(f"{'#'*70}")
        print(f"  [脚本] {exp['script']}")
        print(f"  [配置] {exp['configs']}个配置, 每配置{mode['episodes']}轮")
        print(f"  [时间] 预计: {exp['time_estimate_100ep']} (基于100轮)")
        print(f"  [种子] {seed}")
        print(f"{'#'*70}\n")
        
        result = run_single_experiment(exp_id, mode, suite_id, seed, silent)
        results.append(result)
        
        # 打印结果
        print(f"\n{'='*70}")
        if result["success"]:
            print(colored(f"[OK] [{i}/{total}] 实验完成: {exp['name']}", "green"))
            print(colored(f"     用时: {result['elapsed_time']/60:.1f} 分钟", "green"))
        else:
            print(colored(f"[FAIL] [{i}/{total}] 实验失败: {exp['name']}", "red"))
            print(colored(f"       错误: {result.get('error', 'Unknown')}", "red"))
        print(f"{'='*70}\n")
        
        # 显示剩余进度
        remaining = total - i
        if remaining > 0:
            print(colored(f"[进度] 还剩 {remaining} 个实验待运行...\n", "yellow"))
    
    return results


def run_experiments_parallel(
    exp_ids: List[int],
    mode: Dict[str, Any],
    suite_id: str,
    seed: int,
    silent: bool,
    max_parallel: int = 2,
) -> List[Dict[str, Any]]:
    """
    并行运行多个实验（多线程执行）
    
    【功能】
    - 同时运行多个实验（最多max_parallel个）
    - 分批执行，每批完成后再启动下一批
    - 使用线程池管理并发
    - 通过队列收集运行结果
    
    【参数】
    - exp_ids: List[int] - 要运行的实验编号列表
    - mode: Dict[str, Any] - 运行模式配置
    - suite_id: str - Suite标识符
    - seed: int - 随机种子
    - silent: bool - 静默模式
    - max_parallel: int - 最大并行数（默认2）
    
    【返回值】
    - List[Dict[str, Any]] - 所有实验的运行结果列表
    
    【优点】
    - 显著缩短总耗时
    - 充分利用多GPU资源
    
    【缺点】
    - 需要多GPU支持
    - 输出可能交错（不如串行清晰）
    - 资源占用峰值较高
    
    【推荐场景】
    - 多GPU环境（2个或以上）
    - 时间紧迫需要加速
    - 已验证的稳定实验
    
    【注意事项】
    - max_parallel不应超过GPU数量
    - 注意显存占用，避免OOM
    - 某个实验失败不影响其他实验
    
    【使用示例】
    >>> results = run_experiments_parallel(
    ...     exp_ids=[1, 2, 3, 4],
    ...     mode={"key": "quick", "episodes": 10},
    ...     suite_id="test",
    ...     seed=42,
    ...     silent=True,
    ...     max_parallel=2  # 同时运行2个
    ... )
    """
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
            
            status_icon = colored("[OK]", "green") if result["success"] else colored("[FAIL]", "red")
            print(f"  {status_icon} {result['name']}: {result['elapsed_time']/60:.1f} 分钟")
    
    return results


def print_summary(results: List[Dict[str, Any]], mode: Dict[str, Any], suite_id: str):
    """
    打印批量运行摘要报告
    
    【功能】
    - 统计成功/失败实验数量
    - 计算总耗时
    - 生成详细结果表格
    - 显示失败实验详情
    - 提示结果保存位置
    - 保存JSON格式摘要文件
    
    【参数】
    - results: List[Dict[str, Any]] - 所有实验的运行结果
    - mode: Dict[str, Any] - 运行模式配置
    - suite_id: str - Suite标识符
    
    【显示内容】
    1. Suite ID和运行模式
    2. 成功/失败统计
    3. 总耗时（小时和分钟）
    4. 详细结果表格（编号、名称、状态、耗时）
    5. 失败实验的错误详情
    6. 结果文件位置
    
    【输出文件】
    - results/parameter_sensitivity/{suite_id}_batch_summary.json
      包含完整的运行统计和每个实验的详细结果
    
    【使用示例】
    >>> print_summary(
    ...     results=[...],
    ...     mode={"key": "quick", "episodes": 10},
    ...     suite_id="batch_quick_20251103"
    ... )
    """
    print_banner("运行摘要", "=")
    
    total = len(results)
    success = sum(1 for r in results if r["success"])
    failed = total - success
    total_time = sum(r["elapsed_time"] for r in results)
    
    print(f"\nSuite ID: {colored(suite_id, 'cyan')}")
    print(f"运行模式: {mode['key']} ({mode['episodes']} 轮/配置)")
    print(f"\n总实验数: {total}")
    print(f"  {colored('[OK]', 'green')} 成功: {success}")
    if failed > 0:
        print(f"  {colored('[FAIL]', 'red')} 失败: {failed}")
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
    """
    主函数 - 批量运行工具入口
    
    【功能】
    - 解析命令行参数
    - 支持交互式和命令行两种模式
    - 选择实验和运行模式
    - 执行实验（串行或并行）
    - 生成运行摘要
    - 返回正确的退出码
    
    【运行模式】
    1. 交互式模式: 不带参数运行，逐步选择
    2. 命令行模式: 指定参数直接运行（推荐）
    
    【命令行参数】
    --mode: 运行模式 (quick/medium/full)
    --experiments: 实验编号（逗号分隔）
    --all: 运行所有14个实验
    --high-priority: 运行高优先级实验
    --merged: 运行3个合并实验
    --seed: 随机种子（默认42）
    --parallel: 并行数（需要多GPU）
    --silent: 静默模式（默认开启）
    --interactive: 交互模式
    
    【退出码】
    - 0: 所有实验成功
    - 1: 有实验失败
    
    【完整示例】见下方epilog
    """
    parser = argparse.ArgumentParser(
        description="TD3 参数敏感性分析 - 批量运行工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # ========== 推荐：命令行模式（无需交互）==========
  
  # 快速验证3个合并实验（10轮，10-15分钟）
  python run_batch_experiments.py --mode quick --experiments 3,8,9 --silent
  
  # 快速测试所有14个实验（10轮，约2-3小时）
  python run_batch_experiments.py --mode quick --all
  
  # 运行核心12个高优先级实验（500轮，2-3天）
  python run_batch_experiments.py --mode full --high-priority
  
  # 运行指定实验（中等模式，100轮）
  python run_batch_experiments.py --mode medium --experiments 1,2,5,6,7,10
  
  # 并行运行（需要多GPU）
  python run_batch_experiments.py --mode medium --all --parallel 2
  
  # ========== 交互式模式 ==========
  
  # 逐步选择实验和模式
  python run_batch_experiments.py --interactive
  
  # ========== 高级用法 ==========
  
  # 自定义训练轮数
  python run_batch_experiments.py --mode quick --all --episodes 5
  
  # 指定Suite ID
  python run_batch_experiments.py --mode full --all --suite-id my_experiment
  
  # 更改随机种子
  python run_batch_experiments.py --mode quick --all --seed 123
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
    parser.add_argument("--solid-only", action="store_true",
                       help="仅运行SOLID标记实验（学术化对比方案）")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    parser.add_argument("--suite-id", type=str,
                       help="Suite标识符前缀 (默认: 自动生成时间戳)")
    parser.add_argument("--parallel", type=int, metavar="N",
                       help="并行运行最多N个实验（需要多GPU支持）")
    parser.add_argument("--silent", action="store_true", default=True,
                       help="静默模式（减少输出，默认开启）")
    parser.add_argument("--non-interactive", action="store_true", default=True,
                       help="非交互模式（默认开启，使用 --interactive 覆盖）")
    parser.add_argument("--interactive", action="store_true",
                       help="启用交互模式（覆盖 --non-interactive 和 --silent）")
    parser.add_argument("--episodes", type=int,
                       help="覆盖模式默认的训练轮数（用于快速测试）")
    
    args = parser.parse_args()
    
    # 如果指定了 --interactive，则启用交互模式
    if args.interactive:
        args.non_interactive = False
        args.silent = False
    
    # ========== 交互式或命令行模式 ==========
    if args.non_interactive or (args.mode and (args.all or args.experiments or args.high_priority or args.new_only or args.solid_only)):
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
        elif args.solid_only:
            exp_ids = [k for k, v in EXPERIMENTS.items() if v.get("solid")]
        elif args.experiments:
            exp_ids = [int(x.strip()) for x in args.experiments.split(",")]
        else:
            print(colored("错误: 必须指定 --all, --high-priority, --new-only, --solid-only 或 --experiments", "red"))
            sys.exit(1)
    else:
        # 交互模式
        print_banner("TD3 参数敏感性分析 - 批量运行工具")
        exp_ids = select_experiments_interactive()
        mode = select_mode_interactive()
    
    # 覆盖episodes（如果指定）
    if args.episodes:
        mode["episodes"] = args.episodes
    
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
