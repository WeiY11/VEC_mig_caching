#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对比已有的训练结果并生成图表
"""

import os
import sys
import json
import numpy as np
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入可视化模块
from experiments.visualize_weight_comparison import (
    plot_cost_comparison,
    plot_cost_curves,
    plot_reward_curves,
    plot_radar_comparison,
    plot_metrics_comparison,
    plot_convergence_comparison,
    plot_pareto_frontier
)


def load_and_process_results(json_files):
    """加载并处理多个训练结果JSON文件"""
    
    results = []
    
    print(f"\n{'='*70}")
    print(f"加载 {len(json_files)} 个训练结果文件...")
    print(f"{'='*70}\n")
    
    for i, json_file in enumerate(json_files, 1):
        print(f"[{i}/{len(json_files)}] 读取: {os.path.basename(json_file)}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取配置信息
            config_info = data.get('config', {})
            experiment_name = data.get('experiment_name', f'config_{i}')
            
            # 提取权重配置
            weights = {
                'reward_weight_delay': config_info.get('reward_weight_delay', 2.0),
                'reward_weight_energy': config_info.get('reward_weight_energy', 1.2),
                'reward_weight_cache': config_info.get('reward_weight_cache', 0.15),
                'reward_penalty_dropped': config_info.get('reward_penalty_dropped', 0.05),
                'energy_target': config_info.get('energy_target', 1200.0),
                'latency_target': config_info.get('latency_target', 0.40),
            }
            
            # 提取指标数据
            metrics = data.get('episode_metrics', {})
            
            # 计算后100轮平均指标
            last_100 = min(100, len(metrics.get('total_energy', [])))
            
            if last_100 > 0:
                avg_energy = np.mean(metrics['total_energy'][-last_100:])
                avg_delay = np.mean(metrics['avg_delay'][-last_100:])
                avg_cache_hit = np.mean(metrics['cache_hit_rate'][-last_100:])
                avg_completion = np.mean(metrics['task_completion_rate'][-last_100:])
                
                result = {
                    'name': experiment_name,
                    'file': os.path.basename(json_file),
                    'data': data,
                    'metrics': metrics,
                    'weights': weights,
                    'avg_energy': avg_energy,
                    'avg_delay': avg_delay,
                    'avg_cache_hit': avg_cache_hit,
                    'avg_completion': avg_completion,
                }
                
                results.append(result)
                
                print(f"  ✓ {experiment_name}")
                print(f"    能耗: {avg_energy:.1f}J, 时延: {avg_delay:.4f}s, "
                      f"缓存: {avg_cache_hit:.2%}, 完成率: {avg_completion:.2%}")
            else:
                print(f"  ✗ 数据不足")
                
        except Exception as e:
            print(f"  ✗ 读取失败: {e}")
    
    print(f"\n成功加载 {len(results)} 个结果\n")
    
    return results


def generate_all_charts(results, output_dir):
    """生成所有对比图表"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print("生成对比图表...")
    print(f"{'='*70}\n")
    
    try:
        # 1. 成本对比图
        print("  [1/7] 生成成本对比图...")
        plot_cost_comparison(results, os.path.join(output_dir, "cost_comparison.png"))
        
        # 2. 成本曲线对比
        print("  [2/7] 生成成本曲线对比图...")
        plot_cost_curves(results, os.path.join(output_dir, "cost_curves.png"))
        
        # 3. 奖励曲线对比
        print("  [3/7] 生成奖励曲线对比图...")
        plot_reward_curves(results, os.path.join(output_dir, "reward_curves.png"))
        
        # 4. 雷达图
        print("  [4/7] 生成综合性能雷达图...")
        plot_radar_comparison(results, os.path.join(output_dir, "radar_comparison.png"))
        
        # 5. 详细指标对比
        print("  [5/7] 生成详细指标对比图...")
        plot_metrics_comparison(results, os.path.join(output_dir, "metrics_comparison.png"))
        
        # 6. 收敛曲线
        print("  [6/7] 生成收敛曲线对比图...")
        plot_convergence_comparison(results, os.path.join(output_dir, "convergence_comparison.png"))
        
        # 7. Pareto前沿
        print("  [7/7] 生成Pareto前沿分析图...")
        plot_pareto_frontier(results, os.path.join(output_dir, "pareto_frontier.png"))
        
        print(f"\n✅ 所有图表已保存到: {output_dir}/")
        print("\n生成的图表列表:")
        print("  1. cost_comparison.png   - 成本对比（柱状图+堆叠图）⭐")
        print("  2. cost_curves.png       - 成本曲线（训练过程）⭐")
        print("  3. reward_curves.png     - 奖励曲线（训练过程）⭐")
        print("  4. radar_comparison.png  - 综合性能雷达图")
        print("  5. metrics_comparison.png - 6指标详细对比")
        print("  6. convergence_comparison.png - 4维收敛曲线")
        print("  7. pareto_frontier.png   - 时延-能耗Pareto前沿")
        
    except Exception as e:
        print(f"\n❌ 图表生成失败: {e}")
        import traceback
        traceback.print_exc()


def print_summary(results):
    """打印结果摘要"""
    
    print(f"\n{'='*70}")
    print("实验结果摘要")
    print(f"{'='*70}\n")
    
    # 按成本排序（计算总成本）
    for result in results:
        weights = result['weights']
        w_delay = weights['reward_weight_delay']
        w_energy = weights['reward_weight_energy']
        w_cache = weights['reward_weight_cache']
        target_delay = weights['latency_target']
        target_energy = weights['energy_target']
        
        norm_delay = result['avg_delay'] / target_delay
        norm_energy = result['avg_energy'] / target_energy
        cache_miss = 1 - result['avg_cache_hit']
        
        total_cost = w_delay * norm_delay + w_energy * norm_energy + w_cache * cache_miss
        result['total_cost'] = total_cost
    
    # 按成本排序
    sorted_results = sorted(results, key=lambda x: x['total_cost'])
    
    print(f"{'配置名称':30s} | {'总成本':>8s} | {'能耗(J)':>10s} | {'时延(s)':>8s} | {'缓存率':>8s} | {'完成率':>8s}")
    print("-"*90)
    
    for i, result in enumerate(sorted_results, 1):
        marker = "🏆" if i == 1 else f"{i:2d}"
        print(f"{marker} {result['name']:27s} | {result['total_cost']:8.2f} | "
              f"{result['avg_energy']:10.1f} | {result['avg_delay']:8.4f} | "
              f"{result['avg_cache_hit']:7.2%} | {result['avg_completion']:7.2%}")
    
    print(f"\n最优配置: {sorted_results[0]['name']} (总成本: {sorted_results[0]['total_cost']:.2f})")


def main():
    # 14个训练结果文件
    json_files = [
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_201444.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_193909.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_190111.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_182026.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_174023.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_170758.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_163726.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_160246.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_153226.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_143208.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_150158.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_140220.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_133219.json",
        r"D:\VEC_mig_caching\results\single_agent\td3\权重对比实验\training_results_20251102_130212.json",
    ]
    
    # 加载结果
    results = load_and_process_results(json_files)
    
    if not results:
        print("❌ 没有成功加载任何结果")
        return
    
    # 打印摘要
    print_summary(results)
    
    # 生成图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results/weight_comparison/comparison_{timestamp}"
    
    generate_all_charts(results, output_dir)
    
    print(f"\n{'='*70}")
    print("🎉 对比分析完成！")
    print(f"{'='*70}")
    print(f"\n查看结果:")
    print(f"  📊 图表目录: {output_dir}/")
    print(f"  🏆 最优配置: {sorted(results, key=lambda x: x['total_cost'])[0]['name']}")
    print()


if __name__ == "__main__":
    main()

