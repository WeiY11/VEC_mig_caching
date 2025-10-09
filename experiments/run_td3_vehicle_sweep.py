#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 多车辆数量灵敏度实验脚本

示例用法：
    python experiments/run_td3_vehicle_sweep.py --vehicles 8 12 16 --episodes 200
    python experiments/run_td3_vehicle_sweep.py --vehicle-range 8 16 4 --episodes 150

脚本会针对不同车辆数量运行 TD3 训练，
通过环境变量 `TRAINING_SCENARIO_OVERRIDES` 覆盖仿真器配置，
并将关键结果汇总保存。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from train_single_agent import _apply_global_seed_from_env, train_single_algorithm
from utils.vehicle_sweep_html_generator import VehicleSweepHTMLGenerator
import matplotlib
matplotlib.use('Agg')  # 必须在导入pyplot之前设置
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行TD3不同车辆数量实验")
    parser.add_argument(
        "--vehicles",
        type=int,
        nargs="*",
        help="显式指定车辆数量列表 (优先级高于 --vehicle-range)",
    )
    parser.add_argument(
        "--vehicle-range",
        type=int,
        nargs=3,
        metavar=("START", "END", "STEP"),
        help="使用范围生成车辆数量 (含起始, 不含终止)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=200,
        help="每个车辆设置的训练轮次 (默认: 200)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="实验统一随机种子 (默认: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/experiments/td3_vehicle_sweep"),
        help="实验结果输出目录 (默认: results/experiments/td3_vehicle_sweep)",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=None,
        help="评估间隔 (透传给train_single_algorithm)",
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=None,
        help="保存间隔 (透传给train_single_algorithm)",
    )
    parser.add_argument(
        "--generate-charts",
        action="store_true",
        default=True,
        help="自动生成对比图表 (默认: True)",
    )
    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="禁用图表生成",
    )
    parser.add_argument(
        "--generate-html",
        action="store_true",
        default=True,
        help="生成HTML报告 (默认: True)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="禁用HTML报告生成",
    )
    return parser.parse_args()


def _create_empty_result(num_vehicles: int, episodes: int) -> Dict:
    """
    创建空的训练结果（用于失败情况）
    
    Args:
        num_vehicles: 车辆数量
        episodes: 预期的训练轮次
        
    Returns:
        空的结果字典
    """
    return {
        'algorithm': 'TD3',
        'network_topology': {
            'num_vehicles': num_vehicles,
            'num_rsus': 4,
            'num_uavs': 2,
        },
        'state_dim': 'N/A',
        'training_config': {
            'num_episodes': episodes,
            'training_time_hours': 0.0
        },
        'episode_rewards': [],
        'episode_metrics': {},
        'final_performance': {
            'avg_step_reward': -999.0,  # 明显的失败标记
            'avg_delay': 999.0,
            'avg_completion': 0.0
        }
    }


def _build_vehicle_list(args: argparse.Namespace) -> List[int]:
    """
    构建车辆数量列表
    
    Args:
        args: 命令行参数
        
    Returns:
        车辆数量列表
    """
    if args.vehicles:
        return args.vehicles
    if args.vehicle_range:
        start, end, step = args.vehicle_range
        if step <= 0:
            raise ValueError("vehicle-range 的步长必须为正数")
        return list(range(start, end, step))
    
    # 🔧 优化：使用配置中的默认车辆数量作为基准
    from config import config
    default_vehicles = config.num_vehicles
    return [default_vehicles // 2, default_vehicles, default_vehicles + 4]  # 例如：[6, 12, 16]


def _run_single_setting(num_vehicles: int, seed: int, episodes: int, eval_interval: int | None, save_interval: int | None) -> Dict:
    """
    运行单个车辆配置的训练
    
    Args:
        num_vehicles: 车辆数量
        seed: 随机种子
        episodes: 训练轮次
        eval_interval: 评估间隔
        save_interval: 保存间隔
        
    Returns:
        训练结果字典
    """
    # 设置随机种子
    os.environ['RANDOM_SEED'] = str(seed)
    _apply_global_seed_from_env()
    
    # 构建场景覆盖配置
    overrides = {
        "num_vehicles": num_vehicles,
        "override_topology": True
    }
    
    try:
        # 🔧 优化：直接通过override_scenario参数传递，无需环境变量
        result = train_single_algorithm(
            "TD3",
            num_episodes=episodes,
            eval_interval=eval_interval,
            save_interval=save_interval,
            silent_mode=True,  # 启用静默模式，避免用户交互阻塞批量实验
            override_scenario=overrides
        )
        
        # 验证结果完整性
        if not result or 'final_performance' not in result:
            print(f"⚠️  训练结果不完整 (vehicles={num_vehicles})")
            return _create_empty_result(num_vehicles, episodes)
            
        return result
        
    except Exception as e:
        print(f"❌ 训练失败 (vehicles={num_vehicles}): {e}")
        import traceback
        traceback.print_exc()
        return _create_empty_result(num_vehicles, episodes)
    finally:
        # 清理环境变量（如果有设置的话）
        os.environ.pop('TRAINING_SCENARIO_OVERRIDES', None)


def _extract_summary(num_vehicles: int, run_result: Dict) -> Dict:
    """
    提取训练结果摘要
    
    Args:
        num_vehicles: 车辆数量
        run_result: 训练结果字典
        
    Returns:
        摘要字典
    """
    final_perf = run_result.get("final_performance", {})
    training_cfg = run_result.get("training_config", {})
    
    # 🔧 优化：获取实际状态维度，支持多种格式
    state_dim = run_result.get("state_dim", "N/A")
    if state_dim == "N/A" and "network_topology" in run_result:
        # 尝试计算状态维度
        topo = run_result["network_topology"]
        num_v = topo.get("num_vehicles", num_vehicles)
        num_r = topo.get("num_rsus", 4)
        num_u = topo.get("num_uavs", 2)
        # 使用统一的状态维度计算公式
        state_dim = num_v * 5 + num_r * 5 + num_u * 5 + 8
    
    # 保存完整的episode数据用于图表生成
    episode_rewards = run_result.get("episode_rewards", [])
    episode_metrics = run_result.get("episode_metrics", {})
    
    # 🔧 优化：确保avg_step_reward存在（兼容旧版本结果）
    avg_step_reward = final_perf.get("avg_step_reward")
    if avg_step_reward is None:
        # 尝试从avg_reward获取（旧版本）
        avg_step_reward = final_perf.get("avg_reward", 0.0)
    
    return {
        "num_vehicles": num_vehicles,
        "state_dim": state_dim,
        "episodes": training_cfg.get("num_episodes", 0),
        "training_time_hours": training_cfg.get("training_time_hours", 0.0),
        "avg_step_reward": avg_step_reward,
        "avg_delay": final_perf.get("avg_delay", 0.0),
        "avg_completion": final_perf.get("avg_completion", 0.0),
        # 保存时间序列数据
        "episode_rewards": episode_rewards,
        "episode_metrics": episode_metrics,
    }


def _save_results(output_dir: Path, summaries: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_path = output_dir / f"td3_vehicle_sweep_summary_{timestamp}.json"
    with summary_path.open("w", encoding="utf-8") as fp:
        json.dump(summaries, fp, indent=2, ensure_ascii=False)

    md_path = output_dir / f"td3_vehicle_sweep_summary_{timestamp}.md"
    with md_path.open("w", encoding="utf-8") as fp:
        fp.write("# TD3 不同车辆数量实验结果\n\n")
        fp.write("| Vehicles | State Dim | Episodes | Training Hours | Avg Step Reward | Avg Delay (s) | Completion Rate |\n")
        fp.write("| -------- | --------- | -------- | --------------- | ---------------- | ------------- | ---------------- |\n")
        for item in summaries:
            fp.write(
                f"| {item['num_vehicles']} | {item['state_dim']} | {item['episodes']} | {item['training_time_hours']:.3f} |"
                f" {item['avg_step_reward']:.4f} | {item['avg_delay']:.4f} | {item['avg_completion']:.2%} |\n"
            )

    print(f"[OK] Results saved: {summary_path}")
    print(f"[OK] Markdown report: {md_path}")


def _generate_comparison_charts(output_dir: Path, summaries: List[Dict], timestamp: str) -> None:
    """
    生成车辆数量对比图表
    """
    print("\n" + "=" * 80)
    print("[Chart Generation] Generating comparison charts...")
    print("=" * 80 + "\n")
    
    # 设置学术风格
    plt.style.use('seaborn-v0_8-paper')
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.dpi'] = 300
    
    # 提取数据
    vehicles = [s['num_vehicles'] for s in summaries]
    avg_delays = [s['avg_delay'] for s in summaries]
    avg_completions = [s['avg_completion'] for s in summaries]
    avg_step_rewards = [s['avg_step_reward'] for s in summaries]
    training_times = [s['training_time_hours'] for s in summaries]
    
    # 创建4×2布局的综合对比图
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 平均时延 vs 车辆数
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(vehicles, avg_delays, 'o-', color='#D55E00', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Number of Vehicles')
    ax1.set_ylabel('Average Delay (s)')
    ax1.set_title('(a) Average Delay vs Vehicle Count')
    ax1.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(vehicles, avg_delays)):
        ax1.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 2. 任务完成率 vs 车辆数
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(vehicles, [c*100 for c in avg_completions], 'o-', color='#029E73', linewidth=2.5, markersize=8)
    ax2.set_xlabel('Number of Vehicles')
    ax2.set_ylabel('Task Completion Rate (%)')
    ax2.set_title('(b) Completion Rate vs Vehicle Count')
    ax2.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(vehicles, avg_completions)):
        ax2.text(x, y*100, f'{y*100:.1f}%', ha='center', va='bottom', fontsize=8)
    
    # 3. 平均每步奖励 vs 车辆数
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(vehicles, avg_step_rewards, 'o-', color='#0173B2', linewidth=2.5, markersize=8)
    ax3.set_xlabel('Number of Vehicles')
    ax3.set_ylabel('Average Step Reward')
    ax3.set_title('(c) Reward vs Vehicle Count')
    ax3.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(vehicles, avg_step_rewards)):
        ax3.text(x, y, f'{y:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. 训练时间 vs 车辆数
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(vehicles, training_times, 'o-', color='#CC78BC', linewidth=2.5, markersize=8)
    ax4.set_xlabel('Number of Vehicles')
    ax4.set_ylabel('Training Time (hours)')
    ax4.set_title('(d) Training Time vs Vehicle Count')
    ax4.grid(True, alpha=0.3, linestyle='--')
    for i, (x, y) in enumerate(zip(vehicles, training_times)):
        ax4.text(x, y, f'{y:.2f}h', ha='center', va='bottom', fontsize=8)
    
    # 5. 收敛性对比（如果有episode数据）
    ax5 = plt.subplot(2, 3, 5)
    has_episode_data = all('episode_rewards' in s and s['episode_rewards'] for s in summaries)
    if has_episode_data:
        colors = ['#0173B2', '#DE8F05', '#029E73', '#D55E00', '#CC78BC', '#CA9161']
        for i, summary in enumerate(summaries):
            rewards = summary['episode_rewards']
            if len(rewards) > 0:
                # 移动平均
                window = min(20, len(rewards) // 5) if len(rewards) >= 5 else 1
                if window > 1:
                    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
                    episodes = range(window, len(rewards) + 1)
                else:
                    moving_avg = rewards
                    episodes = range(1, len(rewards) + 1)
                
                label = f'{summary["num_vehicles"]} vehicles'
                ax5.plot(episodes, moving_avg, label=label, 
                        color=colors[i % len(colors)], linewidth=2, alpha=0.8)
        
        ax5.set_xlabel('Episode')
        ax5.set_ylabel('Average Step Reward')
        ax5.set_title('(e) Convergence Comparison')
        ax5.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax5.grid(True, alpha=0.3, linestyle='--')
    else:
        ax5.text(0.5, 0.5, 'No episode data available', 
                ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('(e) Convergence Comparison')
    
    # 6. 综合性能雷达图
    ax6 = plt.subplot(2, 3, 6, projection='polar')
    
    # 归一化指标到[0, 1]
    def normalize(data):
        min_val, max_val = min(data), max(data)
        if max_val == min_val:
            return [0.5] * len(data)
        return [(x - min_val) / (max_val - min_val) for x in data]
    
    # 时延归一化（越小越好，所以取反）
    norm_delays = [1 - x for x in normalize(avg_delays)]
    norm_completions = normalize(avg_completions)
    # 奖励归一化（负值，需要特殊处理）
    if all(r < 0 for r in avg_step_rewards):
        # 都是负值，越大（接近0）越好
        norm_rewards = normalize(avg_step_rewards)
    else:
        norm_rewards = normalize(avg_step_rewards)
    
    # 选择中间的配置作为示例
    mid_idx = len(summaries) // 2
    radar_data = [norm_completions[mid_idx], norm_delays[mid_idx], norm_rewards[mid_idx]]
    
    metrics = ['Completion\nRate', 'Low\nDelay', 'Reward']
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    radar_data += radar_data[:1]  # 闭合
    angles += angles[:1]
    
    ax6.plot(angles, radar_data, 'o-', linewidth=2, color='#0173B2')
    ax6.fill(angles, radar_data, alpha=0.25, color='#0173B2')
    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(metrics)
    ax6.set_ylim(0, 1)
    ax6.set_title(f'(f) Performance Radar\n({vehicles[mid_idx]} vehicles)', y=1.08)
    ax6.grid(True)
    
    plt.suptitle('TD3 Vehicle Count Sensitivity Analysis', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    # 保存图表
    chart_path = output_dir / f"vehicle_sweep_comparison_{timestamp}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Comparison chart saved: {chart_path}")
    
    # 生成单独的详细图表
    _generate_detailed_charts(output_dir, summaries, timestamp)
    
    print("\n" + "=" * 80)
    print("[SUCCESS] All charts generated!")
    print("=" * 80 + "\n")


def _generate_detailed_charts(output_dir: Path, summaries: List[Dict], timestamp: str) -> None:
    """
    生成更详细的单独图表
    """
    vehicles = [s['num_vehicles'] for s in summaries]
    
    # 1. 柱状图对比 - 关键指标
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 时延和完成率
    x = np.arange(len(vehicles))
    width = 0.35
    
    delays = [s['avg_delay'] for s in summaries]
    completions = [s['avg_completion'] * 100 for s in summaries]
    
    ax1.bar(x, delays, width, label='Avg Delay (s)', color='#D55E00', alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Number of Vehicles')
    ax1.set_ylabel('Average Delay (s)')
    ax1.set_title('Average Delay Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(vehicles)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 添加数值标注
    for i, v in enumerate(delays):
        ax1.text(i, v, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax2.bar(x, completions, width, label='Completion Rate (%)', color='#029E73', alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Number of Vehicles')
    ax2.set_ylabel('Task Completion Rate (%)')
    ax2.set_title('Task Completion Rate Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(vehicles)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # 添加数值标注
    for i, v in enumerate(completions):
        ax2.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    chart_path = output_dir / f"vehicle_sweep_bar_comparison_{timestamp}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[OK] Bar chart saved: {chart_path}")


def main() -> None:
    args = parse_args()
    vehicle_list = _build_vehicle_list(args)
    
    # 处理图表生成标志
    generate_charts = args.generate_charts and not args.no_charts
    generate_html = args.generate_html and not args.no_html

    print("=" * 80)
    print("[TD3 Vehicle Sweep Experiment]")
    print(f"Vehicle counts: {vehicle_list}")
    print(f"Random seed: {args.seed}")
    print(f"Episodes per run: {args.episodes}")
    print(f"Generate charts: {generate_charts}")
    print(f"Generate HTML report: {generate_html}")
    print("=" * 80)

    summaries: List[Dict] = []
    for num_vehicles in vehicle_list:
        print("-" * 60)
        print(f"[Running] num_vehicles = {num_vehicles}")
        result = _run_single_setting(num_vehicles, args.seed, args.episodes, args.eval_interval, args.save_interval)
        summary = _extract_summary(num_vehicles, result)
        summaries.append(summary)
        print(
            f"[OK] num_vehicles={num_vehicles} completed: "
            f"Delay={summary['avg_delay']:.4f}s, Completion={summary['avg_completion']:.2%}"
        )

    # 🔧 统一时间戳（确保文件名一致）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存结果
    _save_results(args.output_dir, summaries)
    
    # 定义图表路径
    chart1_path = args.output_dir / f"vehicle_sweep_comparison_{timestamp}.png"
    chart2_path = args.output_dir / f"vehicle_sweep_bar_comparison_{timestamp}.png"
    
    # 🆕 生成对比图表
    if generate_charts:
        try:
            _generate_comparison_charts(args.output_dir, summaries, timestamp)
        except Exception as e:
            print(f"[WARNING] Failed to generate charts: {e}")
            print("Results are still saved successfully.")
    
    # 🆕 生成增强版HTML报告（在图表生成之后）
    if generate_html:
        try:
            print("\n" + "=" * 80)
            print("[HTML Report] Generating enhanced HTML report...")
            print("=" * 80 + "\n")
            
            # 使用增强版HTML生成器
            generator = VehicleSweepHTMLGenerator()
            html_content = generator.generate_enhanced_report(summaries, timestamp, chart1_path, chart2_path)
            
            html_path = args.output_dir / f"vehicle_sweep_report_{timestamp}.html"
            if generator.save_report(html_content, html_path):
                print(f"[OK] HTML report saved: {html_path}")
                
                # 自动打开HTML报告
                import webbrowser
                abs_path = html_path.resolve()
                webbrowser.open(f'file://{abs_path}')
                print(f"[OK] HTML report opened in browser")
                
                print("\n" + "=" * 80)
                print("[SUCCESS] Enhanced HTML report generated!")
                print("=" * 80 + "\n")
        except Exception as e:
            print(f"[WARNING] Failed to generate HTML report: {e}")
            print("Results and charts are still saved successfully.")
    
    print("=" * 80)
    print("[SUCCESS] Vehicle sweep experiment completed!")
    print(f"Output directory: {args.output_dir.resolve()}")
    print("=" * 80)


if __name__ == "__main__":
    main()


