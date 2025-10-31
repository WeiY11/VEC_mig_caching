#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3任务到达率敏感性分析实验 - 静默模式版本

【功能】
测试不同任务到达率(arrival_rate)对TD3算法性能的影响
静默模式：只显示关键信息，不显示详细训练日志

【使用方法】
# 从项目根目录运行
cd d:/VEC_mig_caching

# 快速测试（50轮）
python experiments/arrival_rate_analysis/run_td3_arrival_rate_sweep_silent.py --episodes 50

# 完整实验（800轮）
python experiments/arrival_rate_analysis/run_td3_arrival_rate_sweep_silent.py --episodes 800

# 自定义到达率范围
python experiments/arrival_rate_analysis/run_td3_arrival_rate_sweep_silent.py --rates 1.5 2.5 3.5 --episodes 200

【输出】
- 结果保存: results/parameter_sensitivity/arrival_rate/
- 对比图表: arrival_rate_comparison_[timestamp].png
- 汇总数据: arrival_rate_summary_[timestamp].json
"""

import os
import sys

# 修复Windows编码问题
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import argparse
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
sys.path.insert(0, project_root)

from config import config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TD3任务到达率敏感性分析（静默模式）')
    
    parser.add_argument('--rates', type=float, nargs='+',
                        default=[1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                        help='任务到达率列表 (tasks/s)，默认: 1.0-3.5')
    
    parser.add_argument('--episodes', type=int, default=200,
                        help='每个到达率的训练轮次，默认: 200')
    
    parser.add_argument('--num-vehicles', type=int, default=12,
                        help='车辆数量，默认: 12')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子，默认: 42')
    
    parser.add_argument('--skip-training', action='store_true',
                        help='跳过训练，仅从现有结果生成图表')
    
    parser.add_argument('--output-dir', type=str,
                        default='results/parameter_sensitivity/arrival_rate',
                        help='结果输出目录')
    
    parser.add_argument('--verify-first', action='store_true',
                        help='首先运行1轮验证环境变量是否生效')
    
    return parser.parse_args()


def verify_env_var(arrival_rate: float, num_vehicles: int, seed: int) -> bool:
    """
    验证环境变量是否能正确传递（运行1轮测试）
    
    【返回】
    True: 环境变量生效
    False: 环境变量未生效
    """
    print(f"\n{'='*80}")
    print(f"[VERIFY] Testing environment variable for arrival_rate={arrival_rate}")
    print(f"{'='*80}")
    
    train_script = os.path.join(project_root, 'train_single_agent.py')
    cmd = [
        sys.executable,
        train_script,
        '--algorithm', 'TD3',
        '--episodes', '1',  # 只运行1轮
        '--num-vehicles', str(num_vehicles),
        '--seed', str(seed)
    ]
    
    env = os.environ.copy()
    env['TASK_ARRIVAL_RATE'] = str(arrival_rate)
    
    print(f"Running: {' '.join(cmd)}")
    print(f"With TASK_ARRIVAL_RATE={arrival_rate}")
    print("Checking if environment variable is picked up...\n")
    
    # 不使用静默模式，捕获输出以检查
    result = subprocess.run(
        cmd,
        env=env,
        cwd=project_root,
        capture_output=True,
        text=True,
        encoding='utf-8',
        errors='replace'
    )
    
    # 检查输出中是否包含环境变量覆盖的提示
    success = False
    if f'{arrival_rate}' in result.stdout:
        for line in result.stdout.split('\n'):
            if '从环境变量覆盖任务到达率' in line or f'{arrival_rate} tasks/s' in line:
                print(f"[OK] Found: {line.strip()}")
                success = True
                break
    
    if success:
        print(f"[SUCCESS] Environment variable is working correctly!")
    else:
        print(f"[WARNING] Could not verify environment variable in output.")
        print(f"Searching for arrival_rate references in output...")
        found_any = False
        for line in result.stdout.split('\n'):
            if 'arrival' in line.lower() or 'task_arrival_rate' in line.lower():
                print(f"  {line.strip()}")
                found_any = True
        if not found_any:
            print("  (No arrival rate references found)")
    
    print(f"{'='*80}\n")
    return success


def run_training(arrival_rate: float, episodes: int, num_vehicles: int, 
                 seed: int, output_dir: str) -> Dict[str, Any]:
    """
    运行单个到达率的TD3训练（静默模式）
    
    【参数】
    - arrival_rate: 任务到达率 (tasks/s)
    - episodes: 训练轮次
    - num_vehicles: 车辆数量
    - seed: 随机种子
    - output_dir: 输出目录
    
    【返回】
    训练结果字典，包含指标和路径信息
    """
    print(f"\n{'='*80}")
    print(f"[TRAINING] Starting: arrival_rate={arrival_rate} tasks/s, episodes={episodes}")
    print(f"{'='*80}")
    
    # 构建训练命令
    train_script = os.path.join(project_root, 'train_single_agent.py')
    cmd = [
        sys.executable,
        train_script,
        '--algorithm', 'TD3',
        '--episodes', str(episodes),
        '--num-vehicles', str(num_vehicles),
        '--seed', str(seed),
        '--silent-mode'  # 静默模式
    ]
    
    # 设置环境变量
    env = os.environ.copy()
    env['TASK_ARRIVAL_RATE'] = str(arrival_rate)
    
    # 恢复config（避免影响）
    original_rate = config.task.arrival_rate
    config.task.arrival_rate = arrival_rate
    
    try:
        print(f"Command: python train_single_agent.py --algorithm TD3 --episodes {episodes} --silent-mode")
        print(f"Environment: TASK_ARRIVAL_RATE={arrival_rate}")
        print(f"Training in progress... (this may take a while)")
        
        # 静默模式运行
        result = subprocess.run(
            cmd,
            env=env,
            cwd=project_root,
            capture_output=True,  # 捕获输出，避免显示
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Training failed! Return code: {result.returncode}")
            if result.stderr:
                print(f"Error output:\n{result.stderr[:500]}")
            return None
        
        print(f"[OK] Training completed: arrival_rate={arrival_rate}")
        
        # 查找最新的训练结果文件
        results_dir = Path(os.path.join(project_root, 'results/single_agent/td3'))
        if not results_dir.exists():
            print(f"[WARNING] Results directory not found: {results_dir}")
            return None
        
        result_files = list(results_dir.glob('training_results_*.json'))
        if not result_files:
            print(f"[WARNING] No training result files found")
            return None
        
        latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
        
        # 读取结果
        with open(latest_file, 'r', encoding='utf-8') as f:
            training_results = json.load(f)
        
        # 提取关键指标
        metrics = extract_metrics(training_results, arrival_rate)
        
        # 保存到指定输出目录
        save_result(metrics, arrival_rate, output_dir)
        
        return metrics
        
    except Exception as e:
        print(f"[ERROR] Training exception: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        config.task.arrival_rate = original_rate


def extract_metrics(training_results: Dict[str, Any], arrival_rate: float) -> Dict[str, Any]:
    """从训练结果中提取关键指标"""
    metrics = {
        'arrival_rate': arrival_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    last_n = 50
    
    # 提取episode奖励
    if 'episode_rewards' in training_results:
        rewards = training_results['episode_rewards']
        if len(rewards) > 0:
            metrics['ave_reward_per_step_final'] = float(np.mean(rewards[-last_n:]))
            metrics['ave_reward_per_step_std'] = float(np.std(rewards[-last_n:]))
            metrics['ave_reward_per_step_all'] = [float(r) for r in rewards]
    
    # 提取episode指标
    if 'episode_metrics' in training_results:
        ep_metrics = training_results['episode_metrics']
        
        if 'avg_delay' in ep_metrics and len(ep_metrics['avg_delay']) > 0:
            delays = ep_metrics['avg_delay']
            metrics['avg_delay_final'] = float(np.mean(delays[-last_n:]))
            metrics['avg_delay_std'] = float(np.std(delays[-last_n:]))
            metrics['avg_delay_all'] = [float(d) for d in delays]
        
        if 'total_energy' in ep_metrics and len(ep_metrics['total_energy']) > 0:
            energies = ep_metrics['total_energy']
            metrics['avg_energy_final'] = float(np.mean(energies[-last_n:]))
            metrics['avg_energy_std'] = float(np.std(energies[-last_n:]))
            metrics['avg_energy_all'] = [float(e) for e in energies]
        
        if 'task_completion_rate' in ep_metrics and len(ep_metrics['task_completion_rate']) > 0:
            completion_rates = ep_metrics['task_completion_rate']
            dropped_tasks = [(1.0 - rate) * 100 if rate <= 1.0 else 0 for rate in completion_rates]
            metrics['dropped_tasks_final'] = float(np.mean(dropped_tasks[-last_n:]))
            metrics['dropped_tasks_std'] = float(np.std(dropped_tasks[-last_n:]))
            metrics['dropped_tasks_all'] = [float(d) for d in dropped_tasks]
    
    return metrics


def save_result(metrics: Dict[str, Any], arrival_rate: float, output_dir: str):
    """保存单个实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"arrival_rate_{arrival_rate:.1f}_results.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"[SAVED] {filepath}")


def load_existing_results(output_dir: str) -> List[Dict[str, Any]]:
    """从输出目录加载已有的实验结果"""
    results = []
    
    if not os.path.exists(output_dir):
        return results
    
    result_files = Path(output_dir).glob('arrival_rate_*_results.json')
    
    for filepath in result_files:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
                results.append(metrics)
        except Exception as e:
            print(f"[WARNING] Failed to load {filepath}: {e}")
    
    return results


def generate_comparison_plots(all_results: List[Dict[str, Any]], output_dir: str):
    """生成对比图表"""
    if not all_results:
        print("[WARNING] No results to plot!")
        return
    
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    arrival_rates = [r['arrival_rate'] for r in all_results]
    
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TD3 Algorithm - Task Arrival Rate Sensitivity Analysis', 
                 fontsize=16, fontweight='bold')
    
    # 图1: 平均步奖励 (Cost)
    ax1 = axes[0, 0]
    rewards_mean = [-r.get('ave_reward_per_step_final', 0) for r in all_results]  # Cost = -Reward
    rewards_std = [r.get('ave_reward_per_step_std', 0) for r in all_results]
    ax1.plot(arrival_rates, rewards_mean, 'o-', linewidth=2, markersize=8, color='#2ecc71')
    ax1.fill_between(arrival_rates, 
                     np.array(rewards_mean) - np.array(rewards_std),
                     np.array(rewards_mean) + np.array(rewards_std),
                     alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax1.set_ylabel('Average Cost per Step', fontsize=12)
    ax1.set_title('(a) Average Cost vs Arrival Rate', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 图2: 平均时延
    ax2 = axes[0, 1]
    delays_mean = [r.get('avg_delay_final', 0) for r in all_results]
    delays_std = [r.get('avg_delay_std', 0) for r in all_results]
    ax2.plot(arrival_rates, delays_mean, 's-', linewidth=2, markersize=8, color='#e74c3c')
    ax2.fill_between(arrival_rates,
                     np.array(delays_mean) - np.array(delays_std),
                     np.array(delays_mean) + np.array(delays_std),
                     alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax2.set_ylabel('Average Delay (s)', fontsize=12)
    ax2.set_title('(b) Average Delay vs Arrival Rate', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 图3: 平均能耗
    ax3 = axes[1, 0]
    energies_mean = [r.get('avg_energy_final', 0) for r in all_results]
    energies_std = [r.get('avg_energy_std', 0) for r in all_results]
    ax3.plot(arrival_rates, energies_mean, '^-', linewidth=2, markersize=8, color='#3498db')
    ax3.fill_between(arrival_rates,
                     np.array(energies_mean) - np.array(energies_std),
                     np.array(energies_mean) + np.array(energies_std),
                     alpha=0.2, color='#3498db')
    ax3.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax3.set_ylabel('Average Energy (J)', fontsize=12)
    ax3.set_title('(c) Average Energy vs Arrival Rate', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 图4: 丢弃任务数
    ax4 = axes[1, 1]
    dropped_mean = [r.get('dropped_tasks_final', 0) for r in all_results]
    dropped_std = [r.get('dropped_tasks_std', 0) for r in all_results]
    ax4.plot(arrival_rates, dropped_mean, 'd-', linewidth=2, markersize=8, color='#f39c12')
    ax4.fill_between(arrival_rates,
                     np.array(dropped_mean) - np.array(dropped_std),
                     np.array(dropped_mean) + np.array(dropped_std),
                     alpha=0.2, color='#f39c12')
    ax4.set_xlabel('Arrival Rate (tasks/s)', fontsize=12)
    ax4.set_ylabel('Dropped Tasks', fontsize=12)
    ax4.set_title('(d) Dropped Tasks vs Arrival Rate', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"arrival_rate_comparison_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT SAVED] {plot_path}")
    plt.close()


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    """生成汇总报告"""
    if not all_results:
        return
    
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    
    summary = {
        'experiment_info': {
            'algorithm': 'TD3',
            'parameter': 'arrival_rate',
            'unit': 'tasks/s',
            'num_experiments': len(all_results),
            'timestamp': datetime.now().isoformat()
        },
        'results': []
    }
    
    print(f"\n{'='*80}")
    print("Experiment Results Summary")
    print(f"{'='*80}")
    print(f"{'Rate':>10} | {'Avg Cost':>12} | {'Avg Delay':>10} | {'Avg Energy':>11} | {'Dropped':>10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        rate = result['arrival_rate']
        reward = result.get('ave_reward_per_step_final', 0)
        cost = -reward  # Cost = -Reward
        delay = result.get('avg_delay_final', 0)
        energy = result.get('avg_energy_final', 0)
        dropped = result.get('dropped_tasks_final', 0)
        
        print(f"{rate:>10.1f} | {cost:>12.4f} | {delay:>10.4f} | {energy:>11.2f} | {dropped:>10.2f}")
        
        summary['results'].append({
            'arrival_rate': rate,
            'ave_cost_per_step': cost,
            'avg_delay': delay,
            'avg_energy': energy,
            'dropped_tasks': dropped
        })
    
    print(f"{'='*80}\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"arrival_rate_summary_{timestamp}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"[SUMMARY SAVED] {summary_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("TD3 Task Arrival Rate Sensitivity Analysis (Silent Mode)")
    print("="*80)
    print(f"Configuration:")
    print(f"   - Algorithm: TD3")
    print(f"   - Arrival rates: {args.rates} tasks/s")
    print(f"   - Episodes: {args.episodes}")
    print(f"   - Vehicles: {args.num_vehicles}")
    print(f"   - Random seed: {args.seed}")
    print(f"   - Output directory: {args.output_dir}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    if not args.skip_training:
        # 可选：首先验证环境变量
        if args.verify_first:
            print("\n[VERIFY MODE] Testing with 1 episode first...")
            success = verify_env_var(args.rates[0], args.num_vehicles, args.seed)
            if success:
                print("\n[OK] Environment variable verified. Proceeding with full training...")
            else:
                print("\n[WARNING] Environment variable verification unclear. Continue anyway...")
            # 自动继续，不等待用户输入
        
        # 运行所有实验
        for i, rate in enumerate(args.rates, 1):
            print(f"\n{'='*80}")
            print(f"Experiment {i}/{len(args.rates)}")
            print(f"{'='*80}")
            
            result = run_training(
                arrival_rate=rate,
                episodes=args.episodes,
                num_vehicles=args.num_vehicles,
                seed=args.seed,
                output_dir=args.output_dir
            )
            
            if result:
                all_results.append(result)
                print(f"[PROGRESS] Completed {i}/{len(args.rates)} experiments")
    else:
        print("\n[SKIP TRAINING] Loading existing results...")
        all_results = load_existing_results(args.output_dir)
    
    if not all_results:
        print("\n[ERROR] No results available!")
        return
    
    # 生成对比图表
    print(f"\n{'='*80}")
    print("Generating comparison plots...")
    print(f"{'='*80}")
    generate_comparison_plots(all_results, args.output_dir)
    
    # 生成汇总报告
    generate_summary_report(all_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("[COMPLETED] Experiment finished!")
    print(f"{'='*80}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total experiments: {len(all_results)}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

