#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3任务到达率敏感性分析实验

【功能】
测试不同任务到达率(arrival_rate)对TD3算法性能的影响

【实验设计】
- 算法: TD3
- 测试到达率: 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 tasks/s
- 车辆数: 12辆（固定）
- 评估指标: ave_reward_per_step（平均步奖励）

【使用方法】
# 快速测试（50轮）
python experiments/run_td3_arrival_rate_sweep.py --episodes 50

# 完整实验（800轮）
python experiments/run_td3_arrival_rate_sweep.py --episodes 800

# 自定义到达率范围
python experiments/run_td3_arrival_rate_sweep.py --rates 1.0 2.0 3.0 --episodes 200

【输出】
- 结果保存: results/parameter_sensitivity/arrival_rate/
- 对比图表: arrival_rate_comparison_[timestamp].png
- 汇总数据: arrival_rate_summary_[timestamp].json
"""

import os
import sys
import json
import argparse
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TD3任务到达率敏感性分析')
    
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
    
    return parser.parse_args()


def run_training(arrival_rate: float, episodes: int, num_vehicles: int, 
                 seed: int, output_dir: str) -> Dict[str, Any]:
    """
    运行单个到达率的TD3训练
    
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
    print(f"🚀 开始训练: arrival_rate={arrival_rate} tasks/s")
    print(f"{'='*80}")
    
    # 构建训练命令
    cmd = [
        sys.executable,  # python解释器
        'train_single_agent.py',
        '--algorithm', 'TD3',
        '--episodes', str(episodes),
        '--num-vehicles', str(num_vehicles),
        '--seed', str(seed),
        '--silent-mode'  # 静默模式，避免过多输出
    ]
    
    # 设置环境变量来覆盖arrival_rate
    env = os.environ.copy()
    env['TASK_ARRIVAL_RATE'] = str(arrival_rate)
    
    # 临时修改config（运行时）
    original_rate = config.task.arrival_rate
    config.task.arrival_rate = arrival_rate
    
    try:
        # 运行训练
        print(f"📝 执行命令: {' '.join(cmd)}")
        print(f"📊 任务到达率: {arrival_rate} tasks/s")
        
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        
        if result.returncode != 0:
            print(f"❌ 训练失败! 返回码: {result.returncode}")
            print(f"错误输出:\n{result.stderr}")
            return None
        
        print(f"✅ 训练完成: arrival_rate={arrival_rate}")
        
        # 查找最新的训练结果文件
        results_dir = Path('results/single_agent/td3')
        if not results_dir.exists():
            print(f"⚠️  结果目录不存在: {results_dir}")
            return None
        
        # 查找最新的training_results文件
        result_files = list(results_dir.glob('training_results_*.json'))
        if not result_files:
            print(f"⚠️  未找到训练结果文件")
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
        print(f"❌ 训练过程异常: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        # 恢复原始arrival_rate
        config.task.arrival_rate = original_rate


def extract_metrics(training_results: Dict[str, Any], arrival_rate: float) -> Dict[str, Any]:
    """
    从训练结果中提取关键指标
    
    【参数】
    - training_results: 训练结果字典
    - arrival_rate: 任务到达率
    
    【返回】
    包含关键指标的字典
    """
    metrics = {
        'arrival_rate': arrival_rate,
        'timestamp': datetime.now().isoformat()
    }
    
    # 提取episode历史数据
    if 'episode_history' in training_results:
        history = training_results['episode_history']
        
        # 计算最后50轮的平均值（稳定期性能）
        last_n = 50
        
        if 'ave_reward_per_step' in history and len(history['ave_reward_per_step']) > 0:
            rewards = history['ave_reward_per_step']
            metrics['ave_reward_per_step_final'] = np.mean(rewards[-last_n:])
            metrics['ave_reward_per_step_std'] = np.std(rewards[-last_n:])
            metrics['ave_reward_per_step_all'] = rewards
        
        if 'avg_delay' in history and len(history['avg_delay']) > 0:
            delays = history['avg_delay']
            metrics['avg_delay_final'] = np.mean(delays[-last_n:])
            metrics['avg_delay_std'] = np.std(delays[-last_n:])
            metrics['avg_delay_all'] = delays
        
        if 'avg_energy' in history and len(history['avg_energy']) > 0:
            energies = history['avg_energy']
            metrics['avg_energy_final'] = np.mean(energies[-last_n:])
            metrics['avg_energy_std'] = np.std(energies[-last_n:])
            metrics['avg_energy_all'] = energies
        
        if 'total_dropped_tasks' in history and len(history['total_dropped_tasks']) > 0:
            dropped = history['total_dropped_tasks']
            metrics['dropped_tasks_final'] = np.mean(dropped[-last_n:])
            metrics['dropped_tasks_std'] = np.std(dropped[-last_n:])
            metrics['dropped_tasks_all'] = dropped
    
    # 提取最终评估结果
    if 'final_evaluation' in training_results:
        final_eval = training_results['final_evaluation']
        metrics['final_evaluation'] = final_eval
    
    return metrics


def save_result(metrics: Dict[str, Any], arrival_rate: float, output_dir: str):
    """保存单个实验结果"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存详细结果
    filename = f"arrival_rate_{arrival_rate:.1f}_results.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"💾 结果已保存: {filepath}")


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
            print(f"⚠️  加载结果文件失败: {filepath}, 错误: {e}")
    
    return results


def generate_comparison_plots(all_results: List[Dict[str, Any]], output_dir: str):
    """
    生成到达率对比图表
    
    【参数】
    - all_results: 所有实验结果列表
    - output_dir: 输出目录
    """
    if not all_results:
        print("⚠️  没有可用的实验结果")
        return
    
    # 按arrival_rate排序
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    
    arrival_rates = [r['arrival_rate'] for r in all_results]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TD3算法 - 任务到达率敏感性分析', fontsize=16, fontweight='bold')
    
    # ========== 图1: 平均步奖励 vs 到达率 ==========
    ax1 = axes[0, 0]
    
    rewards_mean = [r.get('ave_reward_per_step_final', 0) for r in all_results]
    rewards_std = [r.get('ave_reward_per_step_std', 0) for r in all_results]
    
    ax1.plot(arrival_rates, rewards_mean, 'o-', linewidth=2, markersize=8, 
             color='#2ecc71', label='平均步奖励')
    ax1.fill_between(arrival_rates, 
                     np.array(rewards_mean) - np.array(rewards_std),
                     np.array(rewards_mean) + np.array(rewards_std),
                     alpha=0.2, color='#2ecc71')
    ax1.set_xlabel('任务到达率 (tasks/s)', fontsize=12)
    ax1.set_ylabel('平均步奖励', fontsize=12)
    ax1.set_title('(a) 平均步奖励 vs 任务到达率', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # ========== 图2: 平均时延 vs 到达率 ==========
    ax2 = axes[0, 1]
    
    delays_mean = [r.get('avg_delay_final', 0) for r in all_results]
    delays_std = [r.get('avg_delay_std', 0) for r in all_results]
    
    ax2.plot(arrival_rates, delays_mean, 's-', linewidth=2, markersize=8,
             color='#e74c3c', label='平均时延')
    ax2.fill_between(arrival_rates,
                     np.array(delays_mean) - np.array(delays_std),
                     np.array(delays_mean) + np.array(delays_std),
                     alpha=0.2, color='#e74c3c')
    ax2.set_xlabel('任务到达率 (tasks/s)', fontsize=12)
    ax2.set_ylabel('平均时延 (s)', fontsize=12)
    ax2.set_title('(b) 平均时延 vs 任务到达率', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # ========== 图3: 平均能耗 vs 到达率 ==========
    ax3 = axes[1, 0]
    
    energies_mean = [r.get('avg_energy_final', 0) for r in all_results]
    energies_std = [r.get('avg_energy_std', 0) for r in all_results]
    
    ax3.plot(arrival_rates, energies_mean, '^-', linewidth=2, markersize=8,
             color='#3498db', label='平均能耗')
    ax3.fill_between(arrival_rates,
                     np.array(energies_mean) - np.array(energies_std),
                     np.array(energies_mean) + np.array(energies_std),
                     alpha=0.2, color='#3498db')
    ax3.set_xlabel('任务到达率 (tasks/s)', fontsize=12)
    ax3.set_ylabel('平均能耗 (J)', fontsize=12)
    ax3.set_title('(c) 平均能耗 vs 任务到达率', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # ========== 图4: 丢弃任务数 vs 到达率 ==========
    ax4 = axes[1, 1]
    
    dropped_mean = [r.get('dropped_tasks_final', 0) for r in all_results]
    dropped_std = [r.get('dropped_tasks_std', 0) for r in all_results]
    
    ax4.plot(arrival_rates, dropped_mean, 'd-', linewidth=2, markersize=8,
             color='#f39c12', label='丢弃任务数')
    ax4.fill_between(arrival_rates,
                     np.array(dropped_mean) - np.array(dropped_std),
                     np.array(dropped_mean) + np.array(dropped_std),
                     alpha=0.2, color='#f39c12')
    ax4.set_xlabel('任务到达率 (tasks/s)', fontsize=12)
    ax4.set_ylabel('丢弃任务数', fontsize=12)
    ax4.set_title('(d) 丢弃任务数 vs 任务到达率', fontsize=13, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    plot_filename = f"arrival_rate_comparison_{timestamp}.png"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"📊 对比图表已保存: {plot_path}")
    
    plt.close()


def generate_summary_report(all_results: List[Dict[str, Any]], output_dir: str):
    """
    生成汇总报告
    
    【参数】
    - all_results: 所有实验结果列表
    - output_dir: 输出目录
    """
    if not all_results:
        return
    
    # 按arrival_rate排序
    all_results = sorted(all_results, key=lambda x: x['arrival_rate'])
    
    # 创建汇总数据
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
    print("📊 实验结果汇总")
    print(f"{'='*80}")
    print(f"{'到达率':>10} | {'平均步奖励':>12} | {'平均时延':>10} | {'平均能耗':>10} | {'丢弃任务':>10}")
    print(f"{'-'*80}")
    
    for result in all_results:
        rate = result['arrival_rate']
        reward = result.get('ave_reward_per_step_final', 0)
        delay = result.get('avg_delay_final', 0)
        energy = result.get('avg_energy_final', 0)
        dropped = result.get('dropped_tasks_final', 0)
        
        print(f"{rate:>10.1f} | {reward:>12.4f} | {delay:>10.4f} | {energy:>10.4f} | {dropped:>10.2f}")
        
        summary['results'].append({
            'arrival_rate': rate,
            'ave_reward_per_step': reward,
            'avg_delay': delay,
            'avg_energy': energy,
            'dropped_tasks': dropped
        })
    
    print(f"{'='*80}\n")
    
    # 保存汇总JSON
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_filename = f"arrival_rate_summary_{timestamp}.json"
    summary_path = os.path.join(output_dir, summary_filename)
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"💾 汇总报告已保存: {summary_path}")


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("🔬 TD3任务到达率敏感性分析实验")
    print("="*80)
    print(f"📋 实验配置:")
    print(f"   - 算法: TD3")
    print(f"   - 到达率范围: {args.rates} tasks/s")
    print(f"   - 训练轮次: {args.episodes}")
    print(f"   - 车辆数: {args.num_vehicles}")
    print(f"   - 随机种子: {args.seed}")
    print(f"   - 输出目录: {args.output_dir}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    if not args.skip_training:
        # 运行所有实验
        for rate in args.rates:
            result = run_training(
                arrival_rate=rate,
                episodes=args.episodes,
                num_vehicles=args.num_vehicles,
                seed=args.seed,
                output_dir=args.output_dir
            )
            
            if result:
                all_results.append(result)
    else:
        print("⏭️  跳过训练，加载已有结果...")
        all_results = load_existing_results(args.output_dir)
    
    if not all_results:
        print("❌ 没有可用的实验结果!")
        return
    
    # 生成对比图表
    print(f"\n{'='*80}")
    print("📊 生成对比图表...")
    print(f"{'='*80}")
    generate_comparison_plots(all_results, args.output_dir)
    
    # 生成汇总报告
    generate_summary_report(all_results, args.output_dir)
    
    print(f"\n{'='*80}")
    print("✅ 实验完成!")
    print(f"{'='*80}")
    print(f"📁 结果目录: {args.output_dir}")
    print(f"📊 共完成 {len(all_results)} 个实验")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

