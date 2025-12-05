"""验证最新训练结果"""
import json
import glob
import os

# 找最新的结果文件
files = sorted(glob.glob('results/**/training_results_*.json', recursive=True), 
               key=os.path.getmtime, reverse=True)

if not files:
    print("未找到训练结果文件")
    exit(1)

latest = files[0]
print(f"读取文件: {latest}")

with open(latest, 'r', encoding='utf-8') as f:
    data = json.load(f)

last = data['episode_stats'][-1]
print("=" * 40)
print("最新训练结果")
print("=" * 40)
print(f"Episode: {last.get('episode', 'N/A')}")
print(f"Total Energy: {last.get('total_energy', 0):.2f} J")
print(f"Avg Delay: {last.get('avg_delay', 0):.4f} s") 
print(f"Completion Rate: {last.get('task_completion_rate', 0)*100:.1f}%")
print(f"Reward: {last.get('episode_reward', 0):.2f}")
print("=" * 40)
print("验证完成 ✓")
