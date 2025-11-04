#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3 版本对比脚本 - v2.0 vs v3.0
用于可视化对比优化前后的训练效果
"""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_results(filepath):
    """加载训练结果"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def moving_average(data, window=20):
    """计算移动平均"""
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window)/window, mode='valid')

def calculate_variance(data, window=100):
    """计算滑动窗口方差"""
    variances = []
    for i in range(len(data)):
        start = max(0, i - window + 1)
        window_data = data[start:i+1]
        if len(window_data) >= 10:  # 至少10个数据点
            mean = np.mean(window_data)
            var = np.mean((np.array(window_data) - mean)**2)
            variances.append(var)
        else:
            variances.append(0)
    return variances

def compare_versions(v2_file, v3_file, output_dir="results/single_agent/td3"):
    """对比两个版本的训练结果"""
    print("📊 加载训练数据...")
    v2_data = load_results(v2_file)
    v3_data = load_results(v3_file)
    
    v2_episodes = v2_data['episodes']
    v3_episodes = v3_data['episodes']
    
    # 提取关键指标
    v2_rewards = [ep['reward'] for ep in v2_episodes]
    v3_rewards = [ep['reward'] for ep in v3_episodes]
    
    v2_delays = [ep['avg_delay'] for ep in v2_episodes]
    v3_delays = [ep['avg_delay'] for ep in v3_episodes]
    
    v2_energy = [ep['avg_energy'] for ep in v2_episodes]
    v3_energy = [ep['avg_energy'] for ep in v3_episodes]
    
    v2_completion = [ep['completion_rate'] for ep in v2_episodes]
    v3_completion = [ep['completion_rate'] for ep in v3_episodes]
    
    v2_noise = [ep['training_stats']['exploration_noise'] for ep in v2_episodes]
    v3_noise = [ep['training_stats']['exploration_noise'] for ep in v3_episodes]
    
    print("📈 生成对比图表...")
    
    # 创建对比图表
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Reward对比
    ax1 = plt.subplot(3, 3, 1)
    ax1.plot(v2_rewards, alpha=0.3, color='#E74C3C', linewidth=0.5)
    ax1.plot(moving_average(v2_rewards, 20), color='#E74C3C', linewidth=2, label='v2.0')
    ax1.plot(v3_rewards, alpha=0.3, color='#27AE60', linewidth=0.5)
    ax1.plot(moving_average(v3_rewards, 20), color='#27AE60', linewidth=2, label='v3.0')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Reward Convergence Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Reward Variance对比
    ax2 = plt.subplot(3, 3, 2)
    v2_var = calculate_variance(v2_rewards, 100)
    v3_var = calculate_variance(v3_rewards, 100)
    ax2.plot(v2_var, color='#E74C3C', linewidth=2, label='v2.0')
    ax2.plot(v3_var, color='#27AE60', linewidth=2, label='v3.0')
    ax2.axhline(y=0.15, color='orange', linestyle='--', alpha=0.5, label='Target (<0.15)')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward Variance')
    ax2.set_title('Stability Comparison (100-Episode Window)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Exploration Noise对比
    ax3 = plt.subplot(3, 3, 3)
    ax3.plot(v2_noise, color='#E74C3C', linewidth=2, label='v2.0')
    ax3.plot(v3_noise, color='#27AE60', linewidth=2, label='v3.0')
    ax3.set_xlabel('Episode')
    ax3.set_ylabel('Exploration Noise')
    ax3.set_title('Exploration Strategy Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 时延对比
    ax4 = plt.subplot(3, 3, 4)
    ax4.plot(moving_average(v2_delays, 20), color='#E74C3C', linewidth=2, label='v2.0')
    ax4.plot(moving_average(v3_delays, 20), color='#27AE60', linewidth=2, label='v3.0')
    ax4.set_xlabel('Episode')
    ax4.set_ylabel('Average Delay (s)')
    ax4.set_title('Delay Performance')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 5. 能耗对比
    ax5 = plt.subplot(3, 3, 5)
    ax5.plot(moving_average(v2_energy, 20), color='#E74C3C', linewidth=2, label='v2.0')
    ax5.plot(moving_average(v3_energy, 20), color='#27AE60', linewidth=2, label='v3.0')
    ax5.set_xlabel('Episode')
    ax5.set_ylabel('Average Energy (J)')
    ax5.set_title('Energy Performance')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 完成率对比
    ax6 = plt.subplot(3, 3, 6)
    ax6.plot(moving_average(v2_completion, 20), color='#E74C3C', linewidth=2, label='v2.0')
    ax6.plot(moving_average(v3_completion, 20), color='#27AE60', linewidth=2, label='v3.0')
    ax6.set_xlabel('Episode')
    ax6.set_ylabel('Completion Rate')
    ax6.set_title('Task Completion Rate')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. 后期稳定性对比（最后200轮）
    ax7 = plt.subplot(3, 3, 7)
    late_v2 = v2_rewards[-200:] if len(v2_rewards) >= 200 else v2_rewards
    late_v3 = v3_rewards[-200:] if len(v3_rewards) >= 200 else v3_rewards
    ax7.boxplot([late_v2, late_v3], labels=['v2.0', 'v3.0'])
    ax7.set_ylabel('Reward')
    ax7.set_title('Late-Stage Stability (Last 200 Episodes)')
    ax7.grid(True, alpha=0.3)
    
    # 8. 统计对比表
    ax8 = plt.subplot(3, 3, 8)
    ax8.axis('off')
    
    # 计算统计指标
    v2_final_reward = np.mean(v2_rewards[-100:])
    v3_final_reward = np.mean(v3_rewards[-100:])
    v2_final_var = np.var(v2_rewards[-100:])
    v3_final_var = np.var(v3_rewards[-100:])
    v2_final_noise = v2_noise[-1]
    v3_final_noise = v3_noise[-1]
    v2_final_delay = np.mean(v2_delays[-100:])
    v3_final_delay = np.mean(v3_delays[-100:])
    
    stats_text = f"""
    📊 最终性能对比 (最后100轮)
    
    指标              v2.0        v3.0        改进
    ─────────────────────────────────────────────
    平均奖励       {v2_final_reward:7.4f}   {v3_final_reward:7.4f}   {(v3_final_reward-v2_final_reward)/abs(v2_final_reward)*100:+.1f}%
    奖励方差       {v2_final_var:7.4f}   {v3_final_var:7.4f}   {(v3_final_var-v2_final_var)/v2_final_var*100:+.1f}%
    探索噪声       {v2_final_noise:7.4f}   {v3_final_noise:7.4f}   {(v3_final_noise-v2_final_noise)/v2_final_noise*100:+.1f}%
    平均时延       {v2_final_delay:7.4f}s  {v3_final_delay:7.4f}s  {(v3_final_delay-v2_final_delay)/v2_final_delay*100:+.1f}%
    
    ✅ 稳定性评估:
    v2.0: {"优秀" if v2_final_var < 0.15 else "良好" if v2_final_var < 0.25 else "需改进"}
    v3.0: {"优秀" if v3_final_var < 0.15 else "良好" if v3_final_var < 0.25 else "需改进"}
    """
    ax8.text(0.1, 0.5, stats_text, fontsize=10, family='monospace', 
             verticalalignment='center')
    
    # 9. 收敛速度对比
    ax9 = plt.subplot(3, 3, 9)
    # 计算达到目标性能的轮次
    target_reward = max(np.mean(v2_rewards), np.mean(v3_rewards)) * 0.9
    v2_converge = next((i for i, r in enumerate(moving_average(v2_rewards, 20)) if r >= target_reward), len(v2_rewards))
    v3_converge = next((i for i, r in enumerate(moving_average(v3_rewards, 20)) if r >= target_reward), len(v3_rewards))
    
    ax9.bar(['v2.0', 'v3.0'], [v2_converge, v3_converge], color=['#E74C3C', '#27AE60'])
    ax9.set_ylabel('Episodes to Converge')
    ax9.set_title(f'Convergence Speed (Target: {target_reward:.2f})')
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('TD3 Version Comparison: v2.0 vs v3.0 (Optimization Validation)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    output_path = Path(output_dir) / 'td3_version_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 对比图表已保存: {output_path}")
    
    # 保存统计报告
    report = {
        'v2.0': {
            'final_reward_mean': float(v2_final_reward),
            'final_reward_variance': float(v2_final_var),
            'final_exploration_noise': float(v2_final_noise),
            'final_delay': float(v2_final_delay),
            'convergence_episode': int(v2_converge)
        },
        'v3.0': {
            'final_reward_mean': float(v3_final_reward),
            'final_reward_variance': float(v3_final_var),
            'final_exploration_noise': float(v3_final_noise),
            'final_delay': float(v3_final_delay),
            'convergence_episode': int(v3_converge)
        },
        'improvements': {
            'reward_mean': f"{(v3_final_reward-v2_final_reward)/abs(v2_final_reward)*100:+.2f}%",
            'reward_variance': f"{(v3_final_var-v2_final_var)/v2_final_var*100:+.2f}%",
            'exploration_noise': f"{(v3_final_noise-v2_final_noise)/v2_final_noise*100:+.2f}%",
            'delay': f"{(v3_final_delay-v2_final_delay)/v2_final_delay*100:+.2f}%"
        }
    }
    
    report_path = Path(output_dir) / 'td3_version_comparison.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"✅ 统计报告已保存: {report_path}")
    
    plt.show()
    
    return report

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("使用方法: python compare_td3_versions.py <v2.0结果文件> <v3.0结果文件>")
        print("\n示例:")
        print("python scripts/compare_td3_versions.py \\")
        print("    results/single_agent/td3/training_results_v2.json \\")
        print("    results/single_agent/td3/training_results_v3.json")
        sys.exit(1)
    
    v2_file = sys.argv[1]
    v3_file = sys.argv[2]
    
    if not Path(v2_file).exists():
        print(f"❌ 文件不存在: {v2_file}")
        sys.exit(1)
    
    if not Path(v3_file).exists():
        print(f"❌ 文件不存在: {v3_file}")
        sys.exit(1)
    
    report = compare_versions(v2_file, v3_file)
    
    print("\n" + "="*60)
    print("📊 优化效果总结:")
    print("="*60)
    for key, value in report['improvements'].items():
        print(f"  {key:20s}: {value}")
    print("="*60)

