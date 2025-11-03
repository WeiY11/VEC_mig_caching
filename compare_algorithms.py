#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速算法性能对比工具
"""

import json
import os
from pathlib import Path

# 要对比的算法
ALGORITHMS = ['cam_td3', 'td3', 'ddpg', 'sac', 'ppo', 'ltd3']

def get_latest_results():
    """获取每个算法的最新训练结果"""
    results = {}
    
    for algo in ALGORITHMS:
        algo_path = Path('results/single_agent') / algo
        if not algo_path.exists():
            continue
            
        # 找到最新的训练结果文件
        json_files = list(algo_path.rglob('training_results_*.json'))
        if not json_files:
            continue
            
        latest_file = max(json_files, key=os.path.getmtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if 'episodes' not in data or not data['episodes']:
                continue
                
            # 取最后一个episode的指标
            last_episode = data['episodes'][-1]
            
            results[algo] = {
                'delay': last_episode.get('avg_delay', 0),
                'energy': last_episode.get('avg_energy', 0),
                'completion_rate': last_episode.get('completion_rate', 0),
                'reward': last_episode.get('avg_step_reward', 0),
                'episodes': len(data['episodes']),
                'file': str(latest_file.name)
            }
        except Exception as e:
            print(f"警告: 无法读取{algo}的结果: {e}")
            continue
    
    return results

def print_comparison(results):
    """打印对比表格"""
    print("\n" + "="*100)
    print("算法性能对比 (基于最新训练结果)")
    print("="*100)
    print(f"{'算法':<12} {'平均时延(s)':<15} {'平均能耗(J)':<15} {'完成率':<12} {'奖励/步':<15} {'训练轮数':<10}")
    print("-"*100)
    
    # 按奖励排序
    sorted_results = sorted(results.items(), key=lambda x: x[1]['reward'], reverse=True)
    
    for i, (algo, metrics) in enumerate(sorted_results):
        rank_symbol = "🏆" if i == 0 else f"{i+1}. "
        print(f"{rank_symbol} {algo.upper():<9} "
              f"{metrics['delay']:<15.4f} "
              f"{metrics['energy']:<15.2f} "
              f"{metrics['completion_rate']:<12.2%} "
              f"{metrics['reward']:<15.4f} "
              f"{metrics['episodes']:<10}")
    
    print("="*100)
    
    # 找出各项指标的最佳算法
    if results:
        best_delay = min(results.items(), key=lambda x: x[1]['delay'])
        best_energy = min(results.items(), key=lambda x: x[1]['energy'])
        best_completion = max(results.items(), key=lambda x: x[1]['completion_rate'])
        best_reward = max(results.items(), key=lambda x: x[1]['reward'])
        
        print("\n各项指标最佳算法:")
        print(f"  最低时延:   {best_delay[0].upper()} ({best_delay[1]['delay']:.4f}s)")
        print(f"  最低能耗:   {best_energy[0].upper()} ({best_energy[1]['energy']:.2f}J)")
        print(f"  最高完成率: {best_completion[0].upper()} ({best_completion[1]['completion_rate']:.2%})")
        print(f"  最高奖励:   {best_reward[0].upper()} ({best_reward[1]['reward']:.4f})")
        
        # 判断CAMTD3表现
        if 'cam_td3' in results:
            cam_metrics = results['cam_td3']
            print(f"\nCAMTD3 性能分析:")
            print(f"  总体排名: {[algo for algo, _ in sorted_results].index('cam_td3') + 1}/{len(sorted_results)}")
            
            rankings = []
            if cam_metrics['delay'] == best_delay[1]['delay']:
                rankings.append("时延第一")
            if cam_metrics['energy'] == best_energy[1]['energy']:
                rankings.append("能耗第一")
            if cam_metrics['completion_rate'] == best_completion[1]['completion_rate']:
                rankings.append("完成率第一")
            if cam_metrics['reward'] == best_reward[1]['reward']:
                rankings.append("综合奖励第一")
            
            if rankings:
                print(f"  优势指标: {', '.join(rankings)}")
            else:
                print(f"  与最佳差距:")
                print(f"    时延: +{(cam_metrics['delay'] - best_delay[1]['delay'])*1000:.2f}ms ({(cam_metrics['delay']/best_delay[1]['delay']-1)*100:+.2f}%)")
                print(f"    能耗: +{cam_metrics['energy'] - best_energy[1]['energy']:.2f}J ({(cam_metrics['energy']/best_energy[1]['energy']-1)*100:+.2f}%)")
                print(f"    完成率: {(cam_metrics['completion_rate'] - best_completion[1]['completion_rate'])*100:+.2f}%")
                print(f"    奖励: {cam_metrics['reward'] - best_reward[1]['reward']:+.4f}")

if __name__ == '__main__':
    results = get_latest_results()
    
    if not results:
        print("错误: 未找到任何训练结果!")
        print("请先运行训练: python train_single_agent.py --algorithm CAM_TD3 --episodes 200")
    else:
        print_comparison(results)

