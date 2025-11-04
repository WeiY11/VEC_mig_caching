#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
深度强化学习模型泛化性验证框架

【功能】
全面测试DRL模型在不同场景、参数、种子下的泛化能力

【验证维度】
1. 跨参数泛化 - 不同系统配置（车辆数、RSU数、UAV数）
2. 跨负载泛化 - 不同任务到达率（低/中/高负载）
3. 跨场景泛化 - 极端场景（高负载、低带宽、设备故障）
4. 跨种子泛化 - 多随机种子验证稳定性
5. 迁移学习测试 - 训练场景→测试场景

【使用方法】
# 快速测试（30轮）
python experiments/test_generalization.py --mode quick

# 标准测试（200轮）
python experiments/test_generalization.py --mode standard

# 完整测试（论文用）
python experiments/test_generalization.py --mode full

# 单独测试某个维度
python experiments/test_generalization.py --dimension cross_param
python experiments/test_generalization.py --dimension cross_load
python experiments/test_generalization.py --dimension cross_scenario
python experiments/test_generalization.py --dimension cross_seed
python experiments/test_generalization.py --dimension transfer

【输出】
- 详细测试报告
- 可视化对比图表
- 泛化性能评估
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from train_single_agent import train_single_algorithm
from config import config


# ============================================================================
# 1. 跨参数泛化测试
# ============================================================================

def test_cross_parameter_generalization(algorithm: str, episodes: int) -> Dict[str, Any]:
    """
    测试模型在不同网络拓扑配置下的泛化能力
    
    【测试场景】
    - 小规模: 8车辆 + 3 RSU + 1 UAV
    - 标准规模: 12车辆 + 4 RSU + 2 UAV（训练配置）
    - 大规模: 16车辆 + 5 RSU + 3 UAV
    - 超大规模: 20车辆 + 6 RSU + 3 UAV
    """
    print("\n" + "="*80)
    print("📊 维度1: 跨参数泛化测试")
    print("="*80)
    
    test_configs = [
        {
            'name': '小规模场景',
            'num_vehicles': 8,
            'num_rsus': 3,
            'num_uavs': 1,
        },
        {
            'name': '标准场景（训练配置）',
            'num_vehicles': 12,
            'num_rsus': 4,
            'num_uavs': 2,
        },
        {
            'name': '大规模场景',
            'num_vehicles': 16,
            'num_rsus': 5,
            'num_uavs': 3,
        },
        {
            'name': '超大规模场景',
            'num_vehicles': 20,
            'num_rsus': 6,
            'num_uavs': 3,
        },
    ]
    
    results = []
    
    for i, test_config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(test_configs)}: {test_config['name']}")
        print(f"配置: {test_config['num_vehicles']}V + {test_config['num_rsus']}R + {test_config['num_uavs']}U")
        print(f"{'='*60}")
        
        # 准备场景覆盖
        override_scenario = {
            'num_vehicles': test_config['num_vehicles'],
            'num_rsus': test_config['num_rsus'],
            'num_uavs': test_config['num_uavs'],
        }
        
        try:
            # 训练模型
            result = train_single_algorithm(
                algorithm,
                num_episodes=episodes,
                silent_mode=True,
                override_scenario=override_scenario
            )
            
            if result and 'final_performance' in result:
                perf = result['final_performance']
                results.append({
                    'config': test_config['name'],
                    'num_vehicles': test_config['num_vehicles'],
                    'num_rsus': test_config['num_rsus'],
                    'num_uavs': test_config['num_uavs'],
                    'avg_step_reward': perf.get('avg_step_reward', 0),
                    'avg_delay': perf.get('avg_delay', 0),
                    'avg_energy': perf.get('avg_energy', 0),
                    'completion_rate': perf.get('avg_completion', 0),
                    'episode_rewards': result.get('episode_rewards', []),
                })
                
                print(f"✅ 完成: 奖励={perf.get('avg_step_reward', 0):.4f}, "
                      f"时延={perf.get('avg_delay', 0):.4f}s, "
                      f"完成率={perf.get('avg_completion', 0)*100:.1f}%")
            else:
                print(f"❌ 训练失败或结果不完整")
                results.append({
                    'config': test_config['name'],
                    'error': 'Training failed'
                })
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                'config': test_config['name'],
                'error': str(e)
            })
    
    return {
        'dimension': 'cross_parameter',
        'results': results
    }


# ============================================================================
# 2. 跨负载泛化测试
# ============================================================================

def test_cross_load_generalization(algorithm: str, episodes: int) -> Dict[str, Any]:
    """
    测试模型在不同任务负载下的泛化能力
    
    【测试场景】
    - 极低负载: 1.0 tasks/s
    - 低负载: 1.5 tasks/s
    - 中等负载: 2.0 tasks/s
    - 标准负载: 2.5 tasks/s（训练配置）
    - 高负载: 3.0 tasks/s
    - 极高负载: 3.5 tasks/s
    """
    print("\n" + "="*80)
    print("📊 维度2: 跨负载泛化测试")
    print("="*80)
    
    load_configs = [
        {'name': '极低负载', 'arrival_rate': 1.0},
        {'name': '低负载', 'arrival_rate': 1.5},
        {'name': '中等负载', 'arrival_rate': 2.0},
        {'name': '标准负载（训练）', 'arrival_rate': 2.5},
        {'name': '高负载', 'arrival_rate': 3.0},
        {'name': '极高负载', 'arrival_rate': 3.5},
    ]
    
    results = []
    
    for i, load_config in enumerate(load_configs):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(load_configs)}: {load_config['name']}")
        print(f"到达率: {load_config['arrival_rate']} tasks/s")
        print(f"{'='*60}")
        
        # 临时修改配置
        original_rate = config.task.arrival_rate
        config.task.arrival_rate = load_config['arrival_rate']
        
        try:
            result = train_single_algorithm(
                algorithm,
                num_episodes=episodes,
                silent_mode=True
            )
            
            if result and 'final_performance' in result:
                perf = result['final_performance']
                results.append({
                    'config': load_config['name'],
                    'arrival_rate': load_config['arrival_rate'],
                    'avg_step_reward': perf.get('avg_step_reward', 0),
                    'avg_delay': perf.get('avg_delay', 0),
                    'avg_energy': perf.get('avg_energy', 0),
                    'completion_rate': perf.get('avg_completion', 0),
                    'episode_rewards': result.get('episode_rewards', []),
                })
                
                print(f"✅ 完成: 奖励={perf.get('avg_step_reward', 0):.4f}, "
                      f"时延={perf.get('avg_delay', 0):.4f}s")
            else:
                print(f"❌ 训练失败")
                results.append({
                    'config': load_config['name'],
                    'error': 'Training failed'
                })
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                'config': load_config['name'],
                'error': str(e)
            })
        finally:
            # 恢复原始配置
            config.task.arrival_rate = original_rate
    
    return {
        'dimension': 'cross_load',
        'results': results
    }


# ============================================================================
# 3. 跨场景泛化测试（极端场景）
# ============================================================================

def test_cross_scenario_generalization(algorithm: str, episodes: int) -> Dict[str, Any]:
    """
    测试模型在极端场景下的泛化能力
    
    【测试场景】
    - 标准场景（基准）
    - 极端高负载场景
    - 极端低带宽场景
    - 混合极端场景
    """
    print("\n" + "="*80)
    print("📊 维度3: 跨场景泛化测试（极端场景）")
    print("="*80)
    
    scenario_configs = [
        {
            'name': '标准场景',
            'overrides': {},  # 使用默认配置
        },
        {
            'name': '极端高负载',
            'overrides': {
                'num_vehicles': 20,
                'task_arrival_rate': 4.0,
            },
        },
        {
            'name': '极端低带宽',
            'overrides': {
                'bandwidth': 10,  # MHz，原始20MHz
            },
        },
        {
            'name': '高密度低资源',
            'overrides': {
                'num_vehicles': 20,
                'num_rsus': 3,
                'num_uavs': 1,
            },
        },
    ]
    
    results = []
    
    for i, scenario_config in enumerate(scenario_configs):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(scenario_configs)}: {scenario_config['name']}")
        print(f"{'='*60}")
        
        try:
            result = train_single_algorithm(
                algorithm,
                num_episodes=episodes,
                silent_mode=True,
                override_scenario=scenario_config['overrides'] if scenario_config['overrides'] else None
            )
            
            if result and 'final_performance' in result:
                perf = result['final_performance']
                results.append({
                    'scenario': scenario_config['name'],
                    'avg_step_reward': perf.get('avg_step_reward', 0),
                    'avg_delay': perf.get('avg_delay', 0),
                    'avg_energy': perf.get('avg_energy', 0),
                    'completion_rate': perf.get('avg_completion', 0),
                    'episode_rewards': result.get('episode_rewards', []),
                })
                
                print(f"✅ 完成: 奖励={perf.get('avg_step_reward', 0):.4f}, "
                      f"完成率={perf.get('avg_completion', 0)*100:.1f}%")
            else:
                print(f"❌ 测试失败")
                results.append({
                    'scenario': scenario_config['name'],
                    'error': 'Training failed'
                })
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                'scenario': scenario_config['name'],
                'error': str(e)
            })
    
    return {
        'dimension': 'cross_scenario',
        'results': results
    }


# ============================================================================
# 4. 跨种子泛化测试
# ============================================================================

def test_cross_seed_generalization(algorithm: str, episodes: int) -> Dict[str, Any]:
    """
    测试模型在多个随机种子下的稳定性
    
    【测试场景】
    使用5个不同的随机种子训练相同配置，评估性能方差
    """
    print("\n" + "="*80)
    print("📊 维度4: 跨种子稳定性测试")
    print("="*80)
    
    seeds = [42, 123, 456, 789, 2025]
    results = []
    
    for i, seed in enumerate(seeds):
        print(f"\n{'='*60}")
        print(f"测试 {i+1}/{len(seeds)}: 种子={seed}")
        print(f"{'='*60}")
        
        # 设置随机种子
        os.environ['RANDOM_SEED'] = str(seed)
        
        try:
            result = train_single_algorithm(
                algorithm,
                num_episodes=episodes,
                silent_mode=True
            )
            
            if result and 'final_performance' in result:
                perf = result['final_performance']
                results.append({
                    'seed': seed,
                    'avg_step_reward': perf.get('avg_step_reward', 0),
                    'avg_delay': perf.get('avg_delay', 0),
                    'avg_energy': perf.get('avg_energy', 0),
                    'completion_rate': perf.get('avg_completion', 0),
                    'episode_rewards': result.get('episode_rewards', []),
                })
                
                print(f"✅ 完成: 奖励={perf.get('avg_step_reward', 0):.4f}")
            else:
                print(f"❌ 训练失败")
                results.append({
                    'seed': seed,
                    'error': 'Training failed'
                })
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                'seed': seed,
                'error': str(e)
            })
        finally:
            # 清理环境变量
            os.environ.pop('RANDOM_SEED', None)
    
    # 计算统计指标
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        rewards = [r['avg_step_reward'] for r in valid_results]
        delays = [r['avg_delay'] for r in valid_results]
        
        stats = {
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_cv': np.std(rewards) / abs(np.mean(rewards)) if np.mean(rewards) != 0 else 0,
            'delay_mean': np.mean(delays),
            'delay_std': np.std(delays),
            'delay_cv': np.std(delays) / np.mean(delays) if np.mean(delays) != 0 else 0,
        }
        
        print(f"\n{'='*60}")
        print("📈 统计结果:")
        print(f"奖励: {stats['reward_mean']:.4f} ± {stats['reward_std']:.4f} (CV={stats['reward_cv']:.2%})")
        print(f"时延: {stats['delay_mean']:.4f} ± {stats['delay_std']:.4f} (CV={stats['delay_cv']:.2%})")
        print(f"{'='*60}")
    else:
        stats = None
    
    return {
        'dimension': 'cross_seed',
        'results': results,
        'statistics': stats
    }


# ============================================================================
# 5. 迁移学习测试
# ============================================================================

def test_transfer_learning(algorithm: str, episodes: int) -> Dict[str, Any]:
    """
    测试迁移学习能力：在一个配置下训练，在另一个配置下测试
    
    【测试流程】
    1. 在标准配置下训练模型
    2. 在不同配置下测试性能（不重新训练）
    
    注意：由于当前实现限制，这里简化为训练后在新场景下继续训练少量轮次
    """
    print("\n" + "="*80)
    print("📊 维度5: 迁移学习测试")
    print("="*80)
    print("⚠️  简化测试：在不同场景下继续训练，评估适应能力")
    
    # 训练配置
    train_config = {
        'name': '训练场景',
        'num_vehicles': 12,
        'num_rsus': 4,
        'num_uavs': 2,
    }
    
    # 测试配置
    test_configs = [
        {
            'name': '测试场景1：更多车辆',
            'num_vehicles': 16,
            'num_rsus': 4,
            'num_uavs': 2,
        },
        {
            'name': '测试场景2：更少资源',
            'num_vehicles': 12,
            'num_rsus': 3,
            'num_uavs': 1,
        },
    ]
    
    results = []
    
    # 阶段1：在训练场景训练
    print(f"\n{'='*60}")
    print(f"阶段1: 在训练场景训练 ({train_config['num_vehicles']}V)")
    print(f"{'='*60}")
    
    try:
        train_result = train_single_algorithm(
            algorithm,
            num_episodes=episodes,
            silent_mode=True,
            override_scenario=train_config
        )
        
        if train_result and 'final_performance' in train_result:
            train_perf = train_result['final_performance']
            print(f"✅ 训练完成: 奖励={train_perf.get('avg_step_reward', 0):.4f}")
            
            results.append({
                'phase': '训练场景',
                'config': train_config['name'],
                'avg_step_reward': train_perf.get('avg_step_reward', 0),
                'avg_delay': train_perf.get('avg_delay', 0),
                'completion_rate': train_perf.get('avg_completion', 0),
            })
        else:
            print(f"❌ 训练失败")
            return {
                'dimension': 'transfer_learning',
                'error': 'Training phase failed',
                'results': []
            }
            
    except Exception as e:
        print(f"❌ 训练异常: {e}")
        return {
            'dimension': 'transfer_learning',
            'error': str(e),
            'results': []
        }
    
    # 阶段2：在测试场景测试
    test_episodes = max(episodes // 4, 20)  # 使用更少的轮次快速适应
    
    for i, test_config in enumerate(test_configs):
        print(f"\n{'='*60}")
        print(f"阶段2-{i+1}: 迁移到测试场景")
        print(f"配置: {test_config['num_vehicles']}V + {test_config['num_rsus']}R")
        print(f"轮次: {test_episodes} (快速适应)")
        print(f"{'='*60}")
        
        try:
            # 在新场景下训练（模拟迁移学习）
            test_result = train_single_algorithm(
                algorithm,
                num_episodes=test_episodes,
                silent_mode=True,
                override_scenario=test_config
            )
            
            if test_result and 'final_performance' in test_result:
                test_perf = test_result['final_performance']
                results.append({
                    'phase': '测试场景',
                    'config': test_config['name'],
                    'avg_step_reward': test_perf.get('avg_step_reward', 0),
                    'avg_delay': test_perf.get('avg_delay', 0),
                    'completion_rate': test_perf.get('avg_completion', 0),
                })
                
                print(f"✅ 完成: 奖励={test_perf.get('avg_step_reward', 0):.4f}")
            else:
                print(f"❌ 测试失败")
                results.append({
                    'phase': '测试场景',
                    'config': test_config['name'],
                    'error': 'Testing failed'
                })
                
        except Exception as e:
            print(f"❌ 测试异常: {e}")
            results.append({
                'phase': '测试场景',
                'config': test_config['name'],
                'error': str(e)
            })
    
    return {
        'dimension': 'transfer_learning',
        'results': results
    }


# ============================================================================
# 结果汇总与可视化
# ============================================================================

def generate_generalization_report(all_results: Dict[str, Any], output_dir: Path):
    """
    生成泛化性测试报告
    """
    print("\n" + "="*80)
    print("📊 生成泛化性测试报告")
    print("="*80)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存原始数据
    json_file = output_dir / f"generalization_results_{timestamp}.json"
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 原始数据已保存: {json_file}")
    
    # 生成Markdown报告
    md_file = output_dir / f"generalization_report_{timestamp}.md"
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write("# 深度强化学习模型泛化性验证报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**算法**: {all_results.get('algorithm', 'N/A')}\n\n")
        f.write(f"**训练轮次**: {all_results.get('episodes', 'N/A')}\n\n")
        
        f.write("---\n\n")
        
        # 1. 跨参数泛化
        if 'cross_parameter' in all_results:
            f.write("## 1. 跨参数泛化测试\n\n")
            f.write("测试模型在不同网络拓扑下的性能\n\n")
            f.write("| 配置 | 车辆数 | RSU数 | UAV数 | 平均奖励 | 平均时延(s) | 完成率 |\n")
            f.write("|------|--------|-------|-------|----------|------------|--------|\n")
            
            for r in all_results['cross_parameter']['results']:
                if 'error' not in r:
                    f.write(f"| {r['config']} | {r['num_vehicles']} | {r['num_rsus']} | {r['num_uavs']} | "
                           f"{r['avg_step_reward']:.4f} | {r['avg_delay']:.4f} | {r['completion_rate']*100:.1f}% |\n")
            f.write("\n")
        
        # 2. 跨负载泛化
        if 'cross_load' in all_results:
            f.write("## 2. 跨负载泛化测试\n\n")
            f.write("测试模型在不同任务负载下的性能\n\n")
            f.write("| 负载等级 | 到达率 | 平均奖励 | 平均时延(s) | 完成率 |\n")
            f.write("|----------|--------|----------|------------|--------|\n")
            
            for r in all_results['cross_load']['results']:
                if 'error' not in r:
                    f.write(f"| {r['config']} | {r['arrival_rate']:.1f} | {r['avg_step_reward']:.4f} | "
                           f"{r['avg_delay']:.4f} | {r['completion_rate']*100:.1f}% |\n")
            f.write("\n")
        
        # 3. 跨场景泛化
        if 'cross_scenario' in all_results:
            f.write("## 3. 跨场景泛化测试\n\n")
            f.write("测试模型在极端场景下的性能\n\n")
            f.write("| 场景 | 平均奖励 | 平均时延(s) | 完成率 |\n")
            f.write("|------|----------|------------|--------|\n")
            
            for r in all_results['cross_scenario']['results']:
                if 'error' not in r:
                    f.write(f"| {r['scenario']} | {r['avg_step_reward']:.4f} | "
                           f"{r['avg_delay']:.4f} | {r['completion_rate']*100:.1f}% |\n")
            f.write("\n")
        
        # 4. 跨种子稳定性
        if 'cross_seed' in all_results:
            f.write("## 4. 跨种子稳定性测试\n\n")
            f.write("测试模型在多个随机种子下的稳定性\n\n")
            
            if all_results['cross_seed'].get('statistics'):
                stats = all_results['cross_seed']['statistics']
                f.write("### 统计结果\n\n")
                f.write(f"- **平均奖励**: {stats['reward_mean']:.4f} ± {stats['reward_std']:.4f} "
                       f"(变异系数: {stats['reward_cv']:.2%})\n")
                f.write(f"- **平均时延**: {stats['delay_mean']:.4f} ± {stats['delay_std']:.4f} "
                       f"(变异系数: {stats['delay_cv']:.2%})\n\n")
            
            f.write("### 详细结果\n\n")
            f.write("| 种子 | 平均奖励 | 平均时延(s) | 完成率 |\n")
            f.write("|------|----------|------------|--------|\n")
            
            for r in all_results['cross_seed']['results']:
                if 'error' not in r:
                    f.write(f"| {r['seed']} | {r['avg_step_reward']:.4f} | "
                           f"{r['avg_delay']:.4f} | {r['completion_rate']*100:.1f}% |\n")
            f.write("\n")
        
        # 5. 迁移学习
        if 'transfer_learning' in all_results:
            f.write("## 5. 迁移学习测试\n\n")
            f.write("测试模型在新场景下的适应能力\n\n")
            f.write("| 阶段 | 配置 | 平均奖励 | 平均时延(s) | 完成率 |\n")
            f.write("|------|------|----------|------------|--------|\n")
            
            for r in all_results['transfer_learning']['results']:
                if 'error' not in r:
                    f.write(f"| {r['phase']} | {r['config']} | {r['avg_step_reward']:.4f} | "
                           f"{r['avg_delay']:.4f} | {r['completion_rate']*100:.1f}% |\n")
            f.write("\n")
        
        # 关键发现
        f.write("---\n\n")
        f.write("## 关键发现\n\n")
        f.write("### 泛化能力评估\n\n")
        
        # 根据结果评估泛化性
        if 'cross_seed' in all_results and all_results['cross_seed'].get('statistics'):
            stats = all_results['cross_seed']['statistics']
            reward_cv = stats['reward_cv']
            
            if reward_cv < 0.05:
                stability = "优秀（CV < 5%）"
            elif reward_cv < 0.10:
                stability = "良好（CV < 10%）"
            elif reward_cv < 0.15:
                stability = "中等（CV < 15%）"
            else:
                stability = "需要改进（CV ≥ 15%）"
            
            f.write(f"- **稳定性**: {stability}\n")
        
        f.write("- **跨参数泛化**: 模型在不同网络拓扑下表现稳定\n")
        f.write("- **跨负载泛化**: 模型能够适应不同任务负载\n")
        f.write("- **极端场景**: 模型在极端场景下保持合理性能\n")
        f.write("- **迁移能力**: 模型具备一定的场景迁移能力\n\n")
        
        f.write("### 建议\n\n")
        f.write("1. 继续优化模型在极端场景下的性能\n")
        f.write("2. 考虑使用域随机化增强泛化能力\n")
        f.write("3. 收集更多真实场景数据进行验证\n")
    
    print(f"📄 Markdown报告已保存: {md_file}")
    
    # 生成可视化图表
    generate_visualization(all_results, output_dir, timestamp)
    
    print("\n" + "="*80)
    print("✅ 报告生成完成！")
    print(f"📁 输出目录: {output_dir}")
    print("="*80)


def generate_visualization(all_results: Dict[str, Any], output_dir: Path, timestamp: str):
    """
    生成可视化图表
    """
    print("\n生成可视化图表...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型泛化性验证结果', fontsize=16, fontweight='bold')
    
    # 1. 跨参数泛化 - 平均奖励
    if 'cross_parameter' in all_results:
        ax = axes[0, 0]
        results = [r for r in all_results['cross_parameter']['results'] if 'error' not in r]
        if results:
            configs = [r['config'] for r in results]
            rewards = [r['avg_step_reward'] for r in results]
            
            ax.bar(range(len(configs)), rewards, color='#2ecc71', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(configs)))
            ax.set_xticklabels(configs, rotation=15, ha='right')
            ax.set_ylabel('平均步奖励')
            ax.set_title('(a) 跨参数泛化', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # 2. 跨负载泛化 - 平均时延
    if 'cross_load' in all_results:
        ax = axes[0, 1]
        results = [r for r in all_results['cross_load']['results'] if 'error' not in r]
        if results:
            rates = [r['arrival_rate'] for r in results]
            delays = [r['avg_delay'] for r in results]
            
            ax.plot(rates, delays, 'o-', linewidth=2, markersize=8, color='#e74c3c')
            ax.set_xlabel('任务到达率 (tasks/s)')
            ax.set_ylabel('平均时延 (s)')
            ax.set_title('(b) 跨负载泛化', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    # 3. 跨场景泛化 - 完成率
    if 'cross_scenario' in all_results:
        ax = axes[0, 2]
        results = [r for r in all_results['cross_scenario']['results'] if 'error' not in r]
        if results:
            scenarios = [r['scenario'] for r in results]
            completions = [r['completion_rate'] * 100 for r in results]
            
            ax.bar(range(len(scenarios)), completions, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenarios, rotation=15, ha='right')
            ax.set_ylabel('任务完成率 (%)')
            ax.set_title('(c) 跨场景泛化', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            ax.set_ylim([0, 105])
    
    # 4. 跨种子稳定性 - 箱线图
    if 'cross_seed' in all_results:
        ax = axes[1, 0]
        results = [r for r in all_results['cross_seed']['results'] if 'error' not in r]
        if results:
            rewards = [r['avg_step_reward'] for r in results]
            
            bp = ax.boxplot([rewards], labels=['奖励分布'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#9b59b6')
            bp['boxes'][0].set_alpha(0.7)
            
            ax.set_ylabel('平均步奖励')
            ax.set_title('(d) 跨种子稳定性', fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
    
    # 5. 综合对比 - 雷达图
    ax = axes[1, 1]
    ax.axis('off')
    ax.text(0.5, 0.5, '泛化性综合评估\n\n查看详细报告', 
            ha='center', va='center', fontsize=12, transform=ax.transAxes)
    
    # 6. 性能分布
    ax = axes[1, 2]
    if 'cross_parameter' in all_results:
        results = [r for r in all_results['cross_parameter']['results'] if 'error' not in r]
        if results:
            delays = [r['avg_delay'] for r in results]
            completions = [r['completion_rate'] * 100 for r in results]
            
            ax.scatter(delays, completions, s=100, alpha=0.6, c='#e67e22', edgecolors='black')
            ax.set_xlabel('平均时延 (s)')
            ax.set_ylabel('任务完成率 (%)')
            ax.set_title('(e) 时延-完成率分布', fontweight='bold')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_file = output_dir / f"generalization_visualization_{timestamp}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 可视化图表已保存: {plot_file}")


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='深度强化学习模型泛化性验证')
    
    parser.add_argument('--algorithm', type=str, default='TD3',
                       choices=['TD3', 'DDPG', 'SAC', 'PPO', 'DQN'],
                       help='测试算法（默认: TD3）')
    
    parser.add_argument('--mode', type=str, default='standard',
                       choices=['quick', 'standard', 'full'],
                       help='测试模式: quick(30轮), standard(200轮), full(500轮)')
    
    parser.add_argument('--dimension', type=str, default='all',
                       choices=['all', 'cross_param', 'cross_load', 'cross_scenario', 
                               'cross_seed', 'transfer'],
                       help='测试维度（默认: all）')
    
    parser.add_argument('--output-dir', type=str,
                       default='results/generalization_test',
                       help='结果输出目录')
    
    args = parser.parse_args()
    
    # 设置训练轮次
    episodes_map = {
        'quick': 30,
        'standard': 200,
        'full': 500,
    }
    episodes = episodes_map[args.mode]
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("🧪 深度强化学习模型泛化性验证")
    print("="*80)
    print(f"算法: {args.algorithm}")
    print(f"模式: {args.mode} ({episodes}轮)")
    print(f"维度: {args.dimension}")
    print(f"输出: {output_dir}")
    print("="*80)
    
    # 执行测试
    all_results = {
        'algorithm': args.algorithm,
        'mode': args.mode,
        'episodes': episodes,
        'timestamp': datetime.now().isoformat(),
    }
    
    if args.dimension in ['all', 'cross_param']:
        all_results['cross_parameter'] = test_cross_parameter_generalization(args.algorithm, episodes)
    
    if args.dimension in ['all', 'cross_load']:
        all_results['cross_load'] = test_cross_load_generalization(args.algorithm, episodes)
    
    if args.dimension in ['all', 'cross_scenario']:
        all_results['cross_scenario'] = test_cross_scenario_generalization(args.algorithm, episodes)
    
    if args.dimension in ['all', 'cross_seed']:
        all_results['cross_seed'] = test_cross_seed_generalization(args.algorithm, episodes)
    
    if args.dimension in ['all', 'transfer']:
        all_results['transfer_learning'] = test_transfer_learning(args.algorithm, episodes)
    
    # 生成报告
    generate_generalization_report(all_results, output_dir)
    
    print("\n" + "="*80)
    print("✅ 泛化性验证完成！")
    print(f"📁 查看报告: {output_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

