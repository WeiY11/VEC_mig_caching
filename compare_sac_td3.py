#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json

# 读取SAC结果
with open('results/single_agent/sac/1/training_results_20251028_222636.json', 'r', encoding='utf-8') as f:
    sac = json.load(f)

# 读取TD3结果
with open('results/single_agent/td3/12/5/training_results_20251028_200556.json', 'r', encoding='utf-8') as f:
    td3 = json.load(f)

print("="*80)
print("SAC vs TD3 性能对比分析")
print("="*80)

# 基本信息
print("\n【训练配置】")
print(f"{'项目':<20} {'SAC':<20} {'TD3':<20}")
print("-"*60)
print(f"{'训练轮次':<20} {sac['training_config']['num_episodes']:<20} {td3['training_config']['num_episodes']:<20}")
print(f"{'训练时长(小时)':<20} {sac['training_config']['training_time_hours']:<20.2f} {td3['training_config']['training_time_hours']:<20.2f}")
print(f"{'车辆数':<20} {sac['network_topology']['num_vehicles']:<20} {td3['network_topology']['num_vehicles']:<20}")

# 性能指标
sac_perf = sac['final_performance']
td3_perf = td3['final_performance']

print("\n【关键性能指标】")
print(f"{'指标':<25} {'SAC':<15} {'TD3':<15} {'优胜':<10} {'改进':<10}")
print("-"*80)

# 延迟
sac_delay = sac_perf.get('avg_delay', 0)
td3_delay = td3_perf.get('avg_delay', 0)
winner = 'SAC' if sac_delay < td3_delay else 'TD3'
improvement = abs((sac_delay - td3_delay) / td3_delay * 100) if td3_delay > 0 else 0
print(f"{'平均延迟 (s)':<25} {sac_delay:<15.4f} {td3_delay:<15.4f} {winner:<10} {improvement:>6.1f}%")

# 能耗
sac_energy = sac_perf.get('total_energy', 0)
td3_energy = td3_perf.get('total_energy', 0)
winner = 'SAC' if sac_energy < td3_energy else 'TD3'
improvement = abs((sac_energy - td3_energy) / td3_energy * 100) if td3_energy > 0 else 0
print(f"{'总能耗 (J)':<25} {sac_energy:<15.1f} {td3_energy:<15.1f} {winner:<10} {improvement:>6.1f}%")

# 完成率
sac_comp = sac_perf.get('task_completion_rate', 0)
td3_comp = td3_perf.get('task_completion_rate', 0)
winner = 'SAC' if sac_comp > td3_comp else 'TD3'
improvement = abs((sac_comp - td3_comp) / td3_comp * 100) if td3_comp > 0 else 0
print(f"{'任务完成率 (%)':<25} {sac_comp*100:<15.2f} {td3_comp*100:<15.2f} {winner:<10} {improvement:>6.1f}%")

# 缓存命中率
sac_cache = sac_perf.get('cache_hit_rate', 0)
td3_cache = td3_perf.get('cache_hit_rate', 0)
winner = 'SAC' if sac_cache > td3_cache else 'TD3'
improvement = abs((sac_cache - td3_cache) / td3_cache * 100) if td3_cache > 0 else 0
print(f"{'缓存命中率 (%)':<25} {sac_cache*100:<15.2f} {td3_cache*100:<15.2f} {winner:<10} {improvement:>6.1f}%")

# 迁移成功率
sac_mig = sac_perf.get('migration_success_rate', 0)
td3_mig = td3_perf.get('migration_success_rate', 0)
winner = 'SAC' if sac_mig > td3_mig else 'TD3'
print(f"{'迁移成功率 (%)':<25} {sac_mig*100:<15.2f} {td3_mig*100:<15.2f} {winner:<10}")

# 计算Objective值（统一的优化目标）
weight_delay = 2.0
weight_energy = 1.2

sac_obj = weight_delay * sac_delay + weight_energy * sac_energy / 1000.0
td3_obj = weight_delay * td3_delay + weight_energy * td3_energy / 1000.0

print("\n【目标函数值】(越小越好)")
print(f"{'指标':<25} {'SAC':<15} {'TD3':<15} {'优胜':<10} {'改进':<10}")
print("-"*80)
print(f"{'Objective':<25} {sac_obj:<15.3f} {td3_obj:<15.3f} {'SAC' if sac_obj < td3_obj else 'TD3':<10} {abs((sac_obj - td3_obj) / td3_obj * 100):>6.1f}%")

# 奖励值（说明不可直接比较）
print("\n【奖励值】(不可直接比较，仅供参考)")
print(f"{'指标':<25} {'SAC':<15} {'TD3':<15} {'说明':<30}")
print("-"*80)
sac_reward = sac.get('best_avg_reward', 0)
td3_reward = td3.get('best_avg_reward', 0)
print(f"{'Best Avg Reward':<25} {sac_reward:<15.3f} {td3_reward:<15.3f} {'奖励范围不同，不能直接比较':<30}")

# 综合结论
print("\n" + "="*80)
print("【综合结论】")
print("="*80)

if sac_obj < td3_obj:
    print(f"🏆 SAC 性能更优！")
    print(f"   - Objective值: {sac_obj:.3f} < {td3_obj:.3f}")
    print(f"   - 综合改进: {abs((sac_obj - td3_obj) / td3_obj * 100):.1f}%")
else:
    print(f"🏆 TD3 性能更优！")
    print(f"   - Objective值: {td3_obj:.3f} < {sac_obj:.3f}")
    print(f"   - 综合改进: {abs((td3_obj - sac_obj) / sac_obj * 100):.1f}%")

print("\n具体优势分析:")
if sac_delay < td3_delay:
    print(f"  ✓ SAC延迟更低: {sac_delay:.4f}s vs {td3_delay:.4f}s (-{abs((sac_delay - td3_delay) / td3_delay * 100):.1f}%)")
else:
    print(f"  ✓ TD3延迟更低: {td3_delay:.4f}s vs {sac_delay:.4f}s (-{abs((td3_delay - sac_delay) / sac_delay * 100):.1f}%)")

if sac_energy < td3_energy:
    print(f"  ✓ SAC能耗更低: {sac_energy:.1f}J vs {td3_energy:.1f}J (-{abs((sac_energy - td3_energy) / td3_energy * 100):.1f}%)")
else:
    print(f"  ✓ TD3能耗更低: {td3_energy:.1f}J vs {sac_energy:.1f}J (-{abs((td3_energy - sac_energy) / sac_energy * 100):.1f}%)")

if sac_comp > td3_comp:
    print(f"  ✓ SAC完成率更高: {sac_comp*100:.2f}% vs {td3_comp*100:.2f}% (+{abs((sac_comp - td3_comp) / td3_comp * 100):.1f}%)")
else:
    print(f"  ✓ TD3完成率更高: {td3_comp*100:.2f}% vs {sac_comp*100:.2f}% (+{abs((td3_comp - sac_comp) / sac_comp * 100):.1f}%)")

print("\n" + "="*80)

