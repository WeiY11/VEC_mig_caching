#!/usr/bin/env python3
"""
🎯 OPTIMIZED_TD3参数扫描实验

在不同RSU计算资源和带宽组合下，测试OPTIMIZED_TD3算法的性能

使用方法：
# 完整实验（RSU计算 + 带宽）
python run_optimized_td3_parameter_sweep.py

# 仅RSU计算资源扫描
python run_optimized_td3_parameter_sweep.py --experiments rsu_compute

# 仅带宽扫描
python run_optimized_td3_parameter_sweep.py --experiments bandwidth

# 快速测试（3个配置点）
python run_optimized_td3_parameter_sweep.py --fast-mode

# 自定义参数范围
python run_optimized_td3_parameter_sweep.py --rsu-levels "30.0,40.0,50.0,60.0,70.0" --bandwidths "20.0,30.0,40.0,50.0,60.0"
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from train_single_agent import train_single_algorithm


def parse_float_list(value: str, default: List[float]) -> List[float]:
    """解析浮点数列表"""
    if not value or value.strip().lower() == "default":
        return default
    return [float(x.strip()) for x in value.split(',') if x.strip()]


def run_single_experiment(
    rsu_compute_ghz: float,
    bandwidth_mhz: float,
    episodes: int,
    seed: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """运行单个参数配置的实验"""
    
    # 构建场景配置
    override_scenario = {
        'total_rsu_compute': rsu_compute_ghz * 1e9,
        'total_bandwidth': bandwidth_mhz * 1e6,
        'num_vehicles': 12,
        'num_rsus': 4,
        'num_uavs': 2,
        'override_topology': True,
    }
    
    exp_key = f"rsu{rsu_compute_ghz:.1f}ghz_bw{bandwidth_mhz:.1f}mhz"
    print(f"\n{'='*80}")
    print(f"运行实验: {exp_key}")
    print(f"  RSU计算资源: {rsu_compute_ghz:.1f} GHz")
    print(f"  带宽: {bandwidth_mhz:.1f} MHz")
    print(f"  训练轮次: {episodes}")
    print(f"  随机种子: {seed}")
    print(f"{'='*80}\n")
    
    # 设置环境变量
    os.environ['RANDOM_SEED'] = str(seed)
    
    # 训练
    try:
        results = train_single_algorithm(
            algorithm='OPTIMIZED_TD3',
            num_episodes=episodes,
            silent_mode=True,
            override_scenario=override_scenario,
            use_enhanced_cache=True,
            disable_migration=False,
        )
        
        # 从results提取关键性能指标
        final_perf = results.get('final_performance', {})
        episode_metrics = results.get('episode_metrics', {})
        
        # 收集指标
        metrics = {
            'rsu_compute_ghz': rsu_compute_ghz,
            'bandwidth_mhz': bandwidth_mhz,
            'episodes': episodes,
            'seed': seed,
            'status': 'success',
            'avg_delay': final_perf.get('avg_delay', 0.0),
            'avg_energy': final_perf.get('avg_energy', 0.0),
            'completion_rate': final_perf.get('avg_completion', 0.0),
            'avg_reward': final_perf.get('avg_reward', 0.0),
            'raw_cost': final_perf.get('raw_cost', 0.0),
        }
        
        # 从episode_metrics提取RSU利用率和卸载率（后半段平均值）
        for key in ['cache_hit_rate', 'rsu_utilization', 'offload_ratio']:
            if key in episode_metrics and episode_metrics[key]:
                values = episode_metrics[key]
                # 取后半部分的平均值
                half_idx = len(values) // 2
                if len(values) > half_idx:
                    metrics[key] = float(sum(values[half_idx:]) / len(values[half_idx:]))
        
        return metrics
        
    except Exception as e:
        print(f"❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return {
            'rsu_compute_ghz': rsu_compute_ghz,
            'bandwidth_mhz': bandwidth_mhz,
            'status': 'failed',
            'error': str(e),
        }


def run_rsu_compute_sweep(
    rsu_levels: List[float],
    episodes: int,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """RSU计算资源扫描（固定带宽）"""
    
    fixed_bandwidth = 50.0  # 固定带宽50MHz（系统默认）
    results = []
    
    print(f"\n{'='*80}")
    print(f"RSU计算资源扫描实验")
    print(f"  固定带宽: {fixed_bandwidth} MHz")
    print(f"  RSU计算档位: {rsu_levels}")
    print(f"{'='*80}")
    
    for rsu_ghz in rsu_levels:
        result = run_single_experiment(
            rsu_compute_ghz=rsu_ghz,
            bandwidth_mhz=fixed_bandwidth,
            episodes=episodes,
            seed=seed,
            output_dir=output_dir,
        )
        results.append(result)
    
    return results


def run_bandwidth_sweep(
    bandwidths: List[float],
    episodes: int,
    seed: int,
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """带宽扫描（固定RSU计算）"""
    
    fixed_rsu_compute = 50.0  # 固定RSU计算50GHz
    results = []
    
    print(f"\n{'='*80}")
    print(f"带宽扫描实验")
    print(f"  固定RSU计算: {fixed_rsu_compute} GHz")
    print(f"  带宽档位: {bandwidths}")
    print(f"{'='*80}")
    
    for bw_mhz in bandwidths:
        result = run_single_experiment(
            rsu_compute_ghz=fixed_rsu_compute,
            bandwidth_mhz=bw_mhz,
            episodes=episodes,
            seed=seed,
            output_dir=output_dir,
        )
        results.append(result)
    
    return results


def print_experiment_summary(all_results: Dict[str, List[Dict[str, Any]]]):
    """打印实验结果摘要"""
    print(f"\n{'='*80}")
    print("实验结果汇总")
    print(f"{'='*80}")
    
    for exp_type, exp_results in all_results.items():
        if exp_type == 'rsu_compute':
            print(f"\n📊 RSU计算资源扫描结果:")
            print(f"{'RSU(GHz)':<10} {'带宽(MHz)':<12} {'时延(s)':<10} {'能耗(J)':<10} {'完成率':<10} {'成本':<10}")
            print("-" * 80)
            for result in exp_results:
                if result.get('status') == 'success':
                    rsu = result.get('rsu_compute_ghz', 0)
                    bw = result.get('bandwidth_mhz', 0)
                    delay = result.get('avg_delay', 0)
                    energy = result.get('avg_energy', 0)
                    comp = result.get('completion_rate', 0)
                    cost = result.get('raw_cost', 0)
                    print(f"{rsu:<10.1f} {bw:<12.1f} {delay:<10.4f} {energy:<10.2f} {comp:<10.2%} {cost:<10.4f}")
                else:
                    print(f"  ❌ {result.get('rsu_compute_ghz', 0):.1f} GHz: 失败")
        
        elif exp_type == 'bandwidth':
            print(f"\n📊 带宽扫描结果:")
            print(f"{'带宽(MHz)':<12} {'RSU(GHz)':<10} {'时延(s)':<10} {'能耗(J)':<10} {'完成率':<10} {'成本':<10}")
            print("-" * 80)
            for result in exp_results:
                if result.get('status') == 'success':
                    bw = result.get('bandwidth_mhz', 0)
                    rsu = result.get('rsu_compute_ghz', 0)
                    delay = result.get('avg_delay', 0)
                    energy = result.get('avg_energy', 0)
                    comp = result.get('completion_rate', 0)
                    cost = result.get('raw_cost', 0)
                    print(f"{bw:<12.1f} {rsu:<10.1f} {delay:<10.4f} {energy:<10.2f} {comp:<10.2%} {cost:<10.4f}")
                else:
                    print(f"  ❌ {result.get('bandwidth_mhz', 0):.1f} MHz: 失败")


def main():
    parser = argparse.ArgumentParser(
        description='OPTIMIZED_TD3参数扫描实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # 实验类型
    parser.add_argument(
        '--experiments',
        type=str,
        default='rsu_compute,bandwidth',
        help='实验类型（逗号分隔）: rsu_compute, bandwidth, all。默认: rsu_compute,bandwidth',
    )
    
    # RSU计算资源档位
    parser.add_argument(
        '--rsu-levels',
        type=str,
        default='default',
        help='RSU计算资源档位(GHz)，逗号分隔。默认: 30.0,40.0,50.0,60.0,70.0',
    )
    
    # 带宽档位
    parser.add_argument(
        '--bandwidths',
        type=str,
        default='default',
        help='带宽档位(MHz)，逗号分隔。默认: 20.0,30.0,40.0,50.0,60.0',
    )
    
    # 训练参数
    parser.add_argument(
        '--episodes',
        type=int,
        default=800,
        help='每个配置的训练轮次。默认: 800',
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子。默认: 42',
    )
    
    # 快速模式
    parser.add_argument(
        '--fast-mode',
        action='store_true',
        help='快速验证模式（3个配置点，500轮训练）',
    )
    
    # 输出目录
    parser.add_argument(
        '--output-dir',
        type=str,
        default='',
        help='输出目录。默认: results/optimized_td3_sweep/sweep_<timestamp>',
    )
    
    args = parser.parse_args()
    
    # 解析实验类型
    exp_types = [x.strip().lower() for x in args.experiments.split(',')]
    if 'all' in exp_types:
        exp_types = ['rsu_compute', 'bandwidth']
    
    # 快速模式配置
    if args.fast_mode:
        default_rsu_levels = [30.0, 50.0, 70.0]
        default_bandwidths = [20.0, 40.0, 60.0]
        if args.episodes == 800:  # 用户未自定义
            args.episodes = 500
        print(f"\n🚀 快速验证模式已启用")
        print(f"  配置点: 3个")
        print(f"  训练轮次: {args.episodes}")
    else:
        default_rsu_levels = [30.0, 40.0, 50.0, 60.0, 70.0]
        default_bandwidths = [20.0, 30.0, 40.0, 50.0, 60.0]
    
    # 解析参数档位
    rsu_levels = parse_float_list(args.rsu_levels, default_rsu_levels)
    bandwidths = parse_float_list(args.bandwidths, default_bandwidths)
    
    # 创建输出目录
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / 'optimized_td3_sweep' / f'sweep_{timestamp}'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("OPTIMIZED_TD3 参数扫描实验")
    print(f"{'='*80}")
    print(f"实验类型: {', '.join(exp_types)}")
    print(f"RSU计算档位: {rsu_levels}")
    print(f"带宽档位: {bandwidths}")
    print(f"训练轮次: {args.episodes}")
    print(f"随机种子: {args.seed}")
    print(f"输出目录: {output_dir}")
    print(f"{'='*80}\n")
    
    # 运行实验
    all_results = {}
    
    if 'rsu_compute' in exp_types:
        print("\n开始RSU计算资源扫描...")
        rsu_results = run_rsu_compute_sweep(
            rsu_levels=rsu_levels,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=output_dir / 'rsu_compute',
        )
        all_results['rsu_compute'] = rsu_results
    
    if 'bandwidth' in exp_types:
        print("\n开始带宽扫描...")
        bw_results = run_bandwidth_sweep(
            bandwidths=bandwidths,
            episodes=args.episodes,
            seed=args.seed,
            output_dir=output_dir / 'bandwidth',
        )
        all_results['bandwidth'] = bw_results
    
    # 保存总结
    summary = {
        'experiment_type': 'optimized_td3_parameter_sweep',
        'algorithm': 'OPTIMIZED_TD3',
        'created_at': datetime.now().isoformat(),
        'episodes': args.episodes,
        'seed': args.seed,
        'experiments': list(exp_types),
        'results': all_results,
    }
    
    summary_path = output_dir / 'summary.json'
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ 实验总结已保存: {summary_path}")
    print_experiment_summary(all_results)
    
    print(f"\n✅ 所有实验完成！")
    print(f"   结果目录: {output_dir}")


if __name__ == '__main__':
    main()
