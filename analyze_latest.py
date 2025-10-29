import json
import numpy as np

# 读取最新结果
latest = json.load(open('results/single_agent/td3/training_results_20251029_183047.json'))
original = json.load(open('results/single_agent/td3/12/5/training_results_20251028_200556.json'))
sac = json.load(open('results/single_agent/sac/1/training_results_20251028_222636.json'))

def avg(metrics, key):
    vals = metrics.get(key, [])
    if not vals:
        return 0.0
    return np.mean(vals[-50:])

print("="*80)
print("LATEST TD3 TRAINING RESULTS ANALYSIS")
print("="*80)

# 基本信息
print("\n[Training Configuration]")
print(f"  Algorithm: {latest.get('algorithm')}")
print(f"  Episodes: {latest['training_config']['num_episodes']}")
print(f"  Training Time: {latest['training_config']['training_time_hours']:.2f} hours")
print(f"  Vehicles: {latest['network_topology']['num_vehicles']}")
print(f"  State Dim: {latest.get('state_dim', 'N/A')}")

# 提取指标
latest_m = latest['episode_metrics']
orig_m = original['episode_metrics']
sac_m = sac['episode_metrics']

latest_delay = avg(latest_m, 'avg_delay')
orig_delay = avg(orig_m, 'avg_delay')
sac_delay = avg(sac_m, 'avg_delay')

latest_energy = avg(latest_m, 'total_energy')
orig_energy = avg(orig_m, 'total_energy')
sac_energy = avg(sac_m, 'total_energy')

latest_cache = avg(latest_m, 'cache_hit_rate')
orig_cache = avg(orig_m, 'cache_hit_rate')
sac_cache = avg(sac_m, 'cache_hit_rate')

latest_comp = avg(latest_m, 'task_completion_rate')
orig_comp = avg(orig_m, 'task_completion_rate')
sac_comp = avg(sac_m, 'task_completion_rate')

print("\n[Performance Metrics - Last 50 Episodes Average]")
print(f"  Delay:      {latest_delay:.4f}s")
print(f"  Energy:     {latest_energy:.1f}J")
print(f"  Cache Hit:  {latest_cache*100:.2f}%")
print(f"  Completion: {latest_comp*100:.2f}%")

# Objective
latest_obj = 2.0*latest_delay + 1.2*latest_energy/1000
orig_obj = 2.0*orig_delay + 1.2*orig_energy/1000
sac_obj = 2.0*sac_delay + 1.2*sac_energy/1000

print(f"  Objective:  {latest_obj:.4f}")

print("\n" + "="*80)
print("COMPARISON WITH BASELINE")
print("="*80)

print(f"\n{'Metric':<20} {'Latest':<15} {'TD3-Orig':<15} {'SAC':<15} {'vs Orig':<12} {'vs SAC':<12}")
print("-"*80)
print(f"{'Delay (s)':<20} {latest_delay:<15.4f} {orig_delay:<15.4f} {sac_delay:<15.4f} {(latest_delay-orig_delay)/orig_delay*100:>+10.1f}% {(latest_delay-sac_delay)/sac_delay*100:>+10.1f}%")
print(f"{'Energy (J)':<20} {latest_energy:<15.1f} {orig_energy:<15.1f} {sac_energy:<15.1f} {(latest_energy-orig_energy)/orig_energy*100:>+10.1f}% {(latest_energy-sac_energy)/sac_energy*100:>+10.1f}%")
print(f"{'Cache Hit (%)':<20} {latest_cache*100:<15.2f} {orig_cache*100:<15.2f} {sac_cache*100:<15.2f} {(latest_cache-orig_cache)/orig_cache*100:>+10.1f}% {(latest_cache-sac_cache)/sac_cache*100:>+10.1f}%")
print(f"{'Completion (%)':<20} {latest_comp*100:<15.2f} {orig_comp*100:<15.2f} {sac_comp*100:<15.2f} {(latest_comp-orig_comp)/orig_comp*100:>+10.1f}% {(latest_comp-sac_comp)/sac_comp*100:>+10.1f}%")
print(f"{'Objective':<20} {latest_obj:<15.4f} {orig_obj:<15.4f} {sac_obj:<15.4f} {(latest_obj-orig_obj)/orig_obj*100:>+10.1f}% {(latest_obj-sac_obj)/sac_obj*100:>+10.1f}%")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

if latest_obj < orig_obj and latest_obj < sac_obj:
    print("\n*** GREAT SUCCESS! ***")
    print(f"Latest training BEATS both TD3-Original and SAC!")
    print(f"  - vs TD3-Orig: {abs((latest_obj-orig_obj)/orig_obj*100):.2f}% better")
    print(f"  - vs SAC:      {abs((latest_obj-sac_obj)/sac_obj*100):.2f}% better")
elif latest_obj < orig_obj:
    print("\nSUCCESS! Latest training improves TD3-Original")
    print(f"  - Improvement: {abs((latest_obj-orig_obj)/orig_obj*100):.2f}%")
    print(f"  - But SAC is better by: {abs((sac_obj-latest_obj)/latest_obj*100):.2f}%")
elif latest_obj < sac_obj:
    print("\nPartial success - beats SAC but not TD3-Original")
    print(f"  - vs SAC: {abs((latest_obj-sac_obj)/sac_obj*100):.2f}% better")
    print(f"  - vs TD3-Orig: {abs((orig_obj-latest_obj)/latest_obj*100):.2f}% worse")
else:
    print("\nDid not improve over baseline algorithms")
    print(f"  - TD3-Orig is better by: {abs((latest_obj-orig_obj)/orig_obj*100):.2f}%")
    print(f"  - SAC is better by: {abs((latest_obj-sac_obj)/sac_obj*100):.2f}%")

print("\n" + "="*80)


