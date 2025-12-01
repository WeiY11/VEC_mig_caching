"""
训练结果分析脚本
分析OPTIMIZED_TD3训练不收敛的原因
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from pathlib import Path

# 读取训练结果
results_file = 'd:/VEC_mig_caching/results/single_agent/optimized_td3/training_results_20251202_002525.json'
with open(results_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

episode_rewards = np.array(data['episode_rewards'])
num_episodes = len(episode_rewards)

# 计算滑动平均
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

window_sizes = [10, 50, 100]
ma_rewards = {w: moving_average(episode_rewards, w) for w in window_sizes}

# 统计分析
print("=" * 80)
print("OPTIMIZED_TD3 训练结果分析报告")
print("=" * 80)
print(f"\n【基本信息】")
print(f"总Episodes: {num_episodes}")
print(f"训练时长: {data['training_config']['training_time_hours']:.2f} 小时")
print(f"设备: {data['system_config']['device']}")
print(f"网络拓扑: {data['network_topology']}")
print(f"状态维度: {data['state_dim']}")

print(f"\n【奖励统计】")
print(f"平均奖励: {np.mean(episode_rewards):.4f}")
print(f"标准差: {np.std(episode_rewards):.4f}")
print(f"最小值: {np.min(episode_rewards):.4f}")
print(f"最大值: {np.max(episode_rewards):.4f}")
print(f"前50轮均值: {np.mean(episode_rewards[:50]):.4f}")
print(f"后50轮均值: {np.mean(episode_rewards[-50:]):.4f}")
print(f"最佳50轮均值: {np.mean(sorted(episode_rewards, reverse=True)[:50]):.4f}")

# 计算收敛性指标
print(f"\n【收敛性分析】")
# 划分训练阶段
phase1 = episode_rewards[:250]  # 前25%
phase2 = episode_rewards[250:500]  # 中期25%-50%
phase3 = episode_rewards[500:750]  # 中后期50%-75%
phase4 = episode_rewards[750:]  # 后25%

print(f"阶段1 (0-250):   均值={np.mean(phase1):.4f}, 标准差={np.std(phase1):.4f}")
print(f"阶段2 (250-500): 均值={np.mean(phase2):.4f}, 标准差={np.std(phase2):.4f}")
print(f"阶段3 (500-750): 均值={np.mean(phase3):.4f}, 标准差={np.std(phase3):.4f}")
print(f"阶段4 (750-1000):均值={np.mean(phase4):.4f}, 标准差={np.std(phase4):.4f}")

# 趋势分析
improvement = np.mean(phase4) - np.mean(phase1)
print(f"\n阶段1→阶段4改进: {improvement:.4f} ({improvement/np.mean(phase1)*100:.2f}%)")

# 计算波动系数
cv = np.std(episode_rewards) / abs(np.mean(episode_rewards))
print(f"变异系数 (CV): {cv:.4f} (越小越稳定)")

# 异常值分析
print(f"\n【异常值分析】")
threshold = -3.0
outliers = episode_rewards[episode_rewards < threshold]
print(f"低于{threshold}的异常值数量: {len(outliers)} ({len(outliers)/num_episodes*100:.2f}%)")
if len(outliers) > 0:
    print(f"异常值均值: {np.mean(outliers):.4f}")
    outlier_indices = np.where(episode_rewards < threshold)[0]
    print(f"异常值出现位置 (前10个): {outlier_indices[:10].tolist()}")

# 创建可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('OPTIMIZED_TD3 Training Analysis - 模型未收敛诊断', fontsize=16, fontweight='bold')

# 图1: 原始奖励曲线
ax1 = axes[0, 0]
ax1.plot(episode_rewards, alpha=0.3, label='Episode Reward', color='blue')
for w in window_sizes:
    ma = ma_rewards[w]
    x = np.arange(w-1, num_episodes)
    ax1.plot(x, ma, label=f'MA-{w}', linewidth=2)
ax1.axhline(y=np.mean(episode_rewards), color='red', linestyle='--', label='Mean')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Reward')
ax1.set_title('训练奖励曲线 (无明显上升趋势)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2: 奖励分布直方图
ax2 = axes[0, 1]
ax2.hist(episode_rewards, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
ax2.axvline(np.mean(episode_rewards), color='red', linestyle='--', linewidth=2, label=f'Mean={np.mean(episode_rewards):.3f}')
ax2.axvline(np.median(episode_rewards), color='green', linestyle='--', linewidth=2, label=f'Median={np.median(episode_rewards):.3f}')
ax2.set_xlabel('Reward')
ax2.set_ylabel('Frequency')
ax2.set_title('奖励分布直方图 (高方差)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3: 阶段性表现对比
ax3 = axes[1, 0]
phases = ['Phase1\n(0-250)', 'Phase2\n(250-500)', 'Phase3\n(500-750)', 'Phase4\n(750-1000)']
means = [np.mean(phase1), np.mean(phase2), np.mean(phase3), np.mean(phase4)]
stds = [np.std(phase1), np.std(phase2), np.std(phase3), np.std(phase4)]
x_pos = np.arange(len(phases))
bars = ax3.bar(x_pos, means, yerr=stds, capsize=5, color=['#ff7f0e', '#2ca02c', '#d62728', '#9467bd'], alpha=0.7)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(phases)
ax3.set_ylabel('Mean Reward')
ax3.set_title('分阶段性能对比 (无稳定改进)')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.grid(True, alpha=0.3, axis='y')
# 添加数值标签
for i, (m, s) in enumerate(zip(means, stds)):
    ax3.text(i, m, f'{m:.3f}', ha='center', va='bottom' if m > 0 else 'top')

# 图4: 滑动标准差 (波动性分析)
ax4 = axes[1, 1]
window = 50
rolling_std = [np.std(episode_rewards[max(0, i-window):i+1]) for i in range(num_episodes)]
ax4.plot(rolling_std, color='purple', linewidth=1.5)
ax4.axhline(y=np.mean(rolling_std), color='red', linestyle='--', label=f'Mean Std={np.mean(rolling_std):.3f}')
ax4.set_xlabel('Episode')
ax4.set_ylabel('Rolling Std (window=50)')
ax4.set_title('训练波动性分析 (高波动性)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
output_path = 'd:/VEC_mig_caching/results/single_agent/optimized_td3/training_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n可视化图表已保存至: {output_path}")

# 诊断结论
print("\n" + "=" * 80)
print("【诊断结论】")
print("=" * 80)
print("\n❌ 模型未能收敛，主要问题：")
print("   1. 奖励无上升趋势: 前后期均值相近，改进幅度微小")
print(f"   2. 高方差: 标准差={np.std(episode_rewards):.4f}，变异系数={cv:.4f}")
print(f"   3. 频繁异常值: {len(outliers)}个极低奖励 (<{threshold})")
print("   4. 持续波动: 后期仍存在大幅震荡，未稳定")

print("\n🔍 可能原因分析：")
print("   ① 探索噪声过高: exploration_noise=0.15 可能导致后期探索过度")
print("   ② 学习率不匹配: actor_lr=3e-5, critic_lr=8e-5 可能过小或不平衡")
print("   ③ 批量大小过大: batch_size=768 可能导致更新频率不足")
print("   ④ 奖励函数尺度: 负值奖励可能影响梯度传播")
print("   ⑤ 网络容量: hidden_dim=512, GAT heads=6 可能过拟合或欠拟合")
print("   ⑥ 预热不足: warmup_steps=2000 (约20 episodes) 可能不够")

print("\n" + "=" * 80)
