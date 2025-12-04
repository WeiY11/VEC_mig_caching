#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
带宽敏感性分析实验
在不同带宽情况下运行OPTIMIZED_TD3算法训练
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional


def run_single_bandwidth_experiment(
    bandwidth_mhz: float,
    episodes: int = 1000,
    num_vehicles: int = 12,
    seed: int = 42,
    output_dir: Path = None,
) -> Dict[str, Any]:
    """
    在指定带宽下运行单次实验
    
    Args:
        bandwidth_mhz: 带宽(MHz)
        episodes: 训练轮次
        num_vehicles: 车辆数量
        seed: 随机种子
        output_dir: 输出目录
    
    Returns:
        实验结果字典
    """
    bandwidth_hz = bandwidth_mhz * 1e6
    run_start_ts = datetime.now().timestamp()
    
    print(f"\n{'='*80}")
    print(f"🔹 带宽实验: {bandwidth_mhz:.1f} MHz")
    print(f"{'='*80}")
    print(f"  训练轮次: {episodes}")
    print(f"  车辆数量: {num_vehicles}")
    print(f"  随机种子: {seed}")
    print(f"{'='*80}\n")
    
    # 设置环境变量
    env = os.environ.copy()
    env['RANDOM_SEED'] = str(seed)
    
    # 构建训练命令
    cmd = [
        sys.executable,
        'train_single_agent.py',
        '--algorithm', 'OPTIMIZED_TD3',
        '--episodes', str(episodes),
        '--num-vehicles', str(num_vehicles),
        '--seed', str(seed),
    ]
    
    # 准备场景配置覆盖（通过环境变量）
    scenario_override = {
        'num_vehicles': num_vehicles,
        'bandwidth': bandwidth_hz,
        'total_bandwidth': bandwidth_hz,
    }
    env['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(scenario_override)
    
    # 运行训练
    print(f"执行命令: {' '.join(cmd)}")
    print(f"带宽配置: {bandwidth_mhz:.1f} MHz = {bandwidth_hz:.0f} Hz\n")
    
    try:
        result = subprocess.run(
            cmd,
            env=env,
            check=True,
            capture_output=False,
            text=True,
        )
        
        # 查找本次运行生成的最新结果文件（按时间过滤）
        results_dir = Path('results/single_agent/optimized_td3')
        if results_dir.exists():
            result_files = sorted(
                [
                    f for f in results_dir.glob('training_results_*.json')
                    if f.stat().st_mtime >= run_start_ts
                ],
                key=lambda p: p.stat().st_mtime,
            )
            if result_files:
                latest_result = result_files[-1]
                with open(latest_result, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)

                # 优先使用最终性能摘要，再回退到 episode 度量
                final_perf = training_data.get('final_performance', {})
                episode_metrics = training_data.get('episode_metrics', {})

                def _last_metric(name: str, default: Optional[float] = None) -> Optional[float]:
                    seq = episode_metrics.get(name)
                    if isinstance(seq, list) and seq:
                        return seq[-1]
                    return default

                avg_delay = final_perf.get('avg_delay', _last_metric('avg_delay', -1))
                total_energy = _last_metric('total_energy', -1)
                completion_rate = final_perf.get('avg_completion', _last_metric('task_completion_rate', -1))
                cache_hit_rate = _last_metric('cache_hit_rate', -1)
                data_loss_ratio = _last_metric('data_loss_ratio_bytes', -1)
                avg_reward = final_perf.get('avg_reward', _last_metric('normalized_reward', -1))

                # 可选：将结果文件复制到 sweep 专属目录，便于后续查看
                stored_result = latest_result
                if output_dir:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    stored_result = output_dir / latest_result.name
                    try:
                        shutil.copy2(latest_result, stored_result)
                    except OSError:
                        stored_result = latest_result

                return {
                    'bandwidth_mhz': bandwidth_mhz,
                    'bandwidth_hz': bandwidth_hz,
                    'episodes': episodes,
                    'num_vehicles': num_vehicles,
                    'seed': seed,
                    'avg_delay': avg_delay,
                    'total_energy': total_energy,
                    'completion_rate': completion_rate,
                    'cache_hit_rate': cache_hit_rate,
                    'data_loss_ratio': data_loss_ratio,
                    'avg_reward': avg_reward,
                    'result_file': str(stored_result),
                    'status': 'success',
                }
        
        return {
            'bandwidth_mhz': bandwidth_mhz,
            'status': 'success_no_data',
        }
    
    except subprocess.CalledProcessError as e:
        print(f"❌ 实验失败: {e}")
        return {
            'bandwidth_mhz': bandwidth_mhz,
            'status': 'failed',
            'error': str(e),
        }


def run_bandwidth_sweep(
    bandwidths: List[float],
    episodes: int = 1000,
    num_vehicles: int = 12,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    运行带宽扫描实验
    
    Args:
        bandwidths: 带宽列表(MHz)
        episodes: 训练轮次
        num_vehicles: 车辆数量
        seed: 随机种子
    
    Returns:
        所有实验结果
    """
    # 创建输出目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(f'results/bandwidth_sweep/sweep_{timestamp}')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print(f"🎯 带宽敏感性分析实验")
    print(f"{'='*80}")
    print(f"  带宽范围: {min(bandwidths):.1f} - {max(bandwidths):.1f} MHz")
    print(f"  配置点数: {len(bandwidths)}")
    print(f"  每点轮次: {episodes}")
    print(f"  输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    results = []
    
    for i, bw_mhz in enumerate(bandwidths):
        print(f"\n进度: [{i+1}/{len(bandwidths)}]")
        result = run_single_bandwidth_experiment(
            bandwidth_mhz=bw_mhz,
            episodes=episodes,
            num_vehicles=num_vehicles,
            seed=seed,
            output_dir=output_dir,
        )
        results.append(result)
        
        # 保存中间结果
        interim_file = output_dir / 'interim_results.json'
        with open(interim_file, 'w', encoding='utf-8') as f:
            json.dump({
                'completed': i + 1,
                'total': len(bandwidths),
                'results': results,
            }, f, indent=2, ensure_ascii=False)
    
    # 汇总结果
    summary = {
        'experiment': 'bandwidth_sweep',
        'timestamp': timestamp,
        'config': {
            'bandwidths_mhz': bandwidths,
            'episodes': episodes,
            'num_vehicles': num_vehicles,
            'seed': seed,
        },
        'results': results,
    }
    
    # 保存最终结果
    summary_file = output_dir / 'summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*80}")
    print(f"✅ 实验完成！")
    print(f"{'='*80}")
    print(f"  结果保存至: {summary_file}")
    print(f"{'='*80}\n")
    
    # 打印性能对比
    print_performance_comparison(results)
    
    return summary


def print_performance_comparison(results: List[Dict[str, Any]]):
    """打印性能对比"""
    print(f"\n{'='*80}")
    print(f"📊 性能对比")
    print(f"{'='*80}")
    print(f"{'带宽(MHz)':<12} {'时延(s)':<12} {'能耗(J)':<12} {'完成率':<12} {'数据丢失率':<12}")
    print(f"{'-'*80}")
    
    for r in results:
        if r.get('status') == 'success':
            bw = r.get('bandwidth_mhz', -1)
            delay = r.get('avg_delay', -1)
            energy = r.get('total_energy', -1)
            comp_rate = r.get('completion_rate', -1)
            loss_rate = r.get('data_loss_ratio', -1)
            
            print(f"{bw:<12.1f} {delay:<12.3f} {energy:<12.1f} {comp_rate:<12.3f} {loss_rate:<12.4f}")
    
    print(f"{'='*80}\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='带宽敏感性分析实验',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--bandwidths',
        type=str,
        default='20,30,40,50,60',
        help='带宽列表(MHz)，逗号分隔。例如: 20,30,40,50,60',
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='每个配置的训练轮次',
    )
    
    parser.add_argument(
        '--num-vehicles',
        type=int,
        default=12,
        help='车辆数量',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子',
    )
    
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='快速验证模式（3个配置点，500轮训练）',
    )
    
    args = parser.parse_args()
    
    # 解析带宽列表
    bandwidths = [float(x.strip()) for x in args.bandwidths.split(',')]
    
    # 快速模式
    if args.fast_mode:
        print("\n🚀 快速验证模式已启用")
        bandwidths = [20.0, 40.0, 60.0]  # 3个配置点
        episodes = 500
        print(f"  带宽配置: {bandwidths} MHz")
        print(f"  训练轮次: {episodes}")
        print("")
    else:
        episodes = args.episodes
    
    # 运行实验
    summary = run_bandwidth_sweep(
        bandwidths=bandwidths,
        episodes=episodes,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
    )
    
    return summary


if __name__ == '__main__':
    main()
