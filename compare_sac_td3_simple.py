#!/usr/bin/env python3
import json
import numpy as np

# 读取数据
with open('results/single_agent/sac/1/training_results_20251028_222636.json', 'r') as f:
    sac = json.load(f)
with open('results/single_agent/td3/12/5/training_results_20251028_200556.json', 'r') as f:
    td3 = json.load(f)

# 提取最后50轮的平均值
def get_final_avg(metrics, key, last_n=50):
    values = metrics.get(key, [])
    if not values:
        return 0.0
    return np.mean(values[-last_n:])

sac_m = sac['episode_metrics']
td3_m = td3['episode_metrics']

print("="*80)
print("SAC vs TD3 Performance Comparison (Final 50 Episodes Avg)")
print("="*80)

# 提取指标
metrics_to_compare = [
    ('avg_delay', 's', 'Avg Delay', 'lower'),
    ('total_energy', 'J', 'Total Energy', 'lower'),
    ('task_completion_rate', '%', 'Completion Rate', 'higher'),
    ('cache_hit_rate', '%', 'Cache Hit Rate', 'higher'),
    ('migration_success_rate', '%', 'Migration Success', 'higher'),
]

print(f"\n{'Metric':<25} {'SAC':<15} {'TD3':<15} {'Winner':<10} {'Diff':<10}")
print("-"*80)

results = {}
for key, unit, name, better in metrics_to_compare:
    sac_val = get_final_avg(sac_m, key)
    td3_val = get_final_avg(td3_m, key)
    
    # 判断胜者
    if better == 'lower':
        winner = 'SAC' if sac_val < td3_val else 'TD3'
        improvement = abs((sac_val - td3_val) / td3_val * 100) if td3_val > 0 else 0
    else:
        winner = 'SAC' if sac_val > td3_val else 'TD3'
        improvement = abs((sac_val - td3_val) / td3_val * 100) if td3_val > 0 else 0
    
    # 格式化
    if unit == '%':
        print(f"{name:<25} {sac_val*100:<15.2f} {td3_val*100:<15.2f} {winner:<10} {improvement:>6.1f}%")
    elif unit == 'J':
        print(f"{name:<25} {sac_val:<15.1f} {td3_val:<15.1f} {winner:<10} {improvement:>6.1f}%")
    else:
        print(f"{name:<25} {sac_val:<15.4f} {td3_val:<15.4f} {winner:<10} {improvement:>6.1f}%")
    
    results[key] = {'sac': sac_val, 'td3': td3_val, 'winner': winner}

# 计算Objective
weight_delay = 2.0
weight_energy = 1.2

sac_obj = weight_delay * results['avg_delay']['sac'] + weight_energy * results['total_energy']['sac'] / 1000.0
td3_obj = weight_delay * results['avg_delay']['td3'] + weight_energy * results['total_energy']['td3'] / 1000.0

print("\n" + "="*80)
print("OBJECTIVE FUNCTION (The True Optimization Goal)")
print("="*80)
print(f"\nFormula: 2.0 * Delay + 1.2 * Energy/1000")
print(f"\nSAC Objective: {sac_obj:.4f}")
print(f"TD3 Objective: {td3_obj:.4f}")
print(f"\n{'*' * 80}")

if sac_obj < td3_obj:
    print(f"WINNER: SAC (Objective {sac_obj:.4f} < {td3_obj:.4f})")
    print(f"Improvement: {abs((sac_obj - td3_obj) / td3_obj * 100):.2f}%")
else:
    print(f"WINNER: TD3 (Objective {td3_obj:.4f} < {sac_obj:.4f})")
    print(f"Improvement: {abs((td3_obj - sac_obj) / sac_obj * 100):.2f}%")

print(f"{'*' * 80}")

# 奖励说明
print("\n" + "="*80)
print("REWARD VALUES (For Reference Only - Different Scales)")
print("="*80)
print(f"SAC Best Reward: {sac.get('best_avg_reward', 0):.3f} (range: -15 to +3)")
print(f"TD3 Best Reward: {td3.get('best_avg_reward', 0):.3f} (range: -80 to -0.005)")
print(f"\n! CANNOT compare reward values directly - use Objective instead !")
print("="*80)

