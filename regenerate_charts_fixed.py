#!/usr/bin/env python3
"""
使用最新的训练结果重新生成图表，验证修复效果
"""

import sys
sys.path.insert(0, 'd:/VEC_mig_caching')

import json
from pathlib import Path

# 加载最新训练结果
result_file = Path('results/single_agent/td3/training_results_20251122_041137.json')

print("=" * 60)
print("🔧 重新生成训练图表（使用修复后的可视化代码）")
print("=" * 60)

if not result_file.exists():
    print(f"❌ 文件不存在: {result_file}")
    exit(1)

print(f"\n📖 加载训练结果: {result_file}")
with open(result_file, 'r') as f:
    data = json.load(f)

# 创建一个模拟的training_env来重新生成图表
class MockTrainingEnv:
    """模拟训练环境，用于重新生成图表"""
    def __init__(self, data):
        self.episode_rewards = data.get('episode_rewards', [])
        self.episode_metrics = data.get('episode_metrics', {})
        self.max_steps_per_episode = 200
        
        # 如果有episode_steps记录，使用它
        if 'episode_steps' in self.episode_metrics:
            self.episode_steps = self.episode_metrics['episode_steps']

print(f"  总episode数: {len(data.get('episode_rewards', []))}")
print(f"  指标数据: {len(data.get('episode_metrics', {}))} 个指标")

# 创建模拟环境
mock_env = MockTrainingEnv(data)

# 使用修复后的可视化代码重新生成图表
print("\n🎨 生成训练总览图表...")
from visualization.clean_charts import create_training_chart

output_path = 'results/single_agent/td3/training_overview_FIXED.png'
create_training_chart(mock_env, 'TD3', output_path)

print(f"\n✅ 修复后的图表已生成:")
print(f"   {output_path}")
print(f"   (同时生成了热点分析图)")

# 生成目标函数分解图
print("\n🎨 生成目标函数分解图...")
from visualization.clean_charts import plot_objective_function_breakdown

objective_path = 'results/single_agent/td3/objective_analysis_FIXED.png'
plot_objective_function_breakdown(mock_env, 'TD3', objective_path)

print(f"\n✅ 目标函数分解图已生成:")
print(f"   {objective_path}")

# 显示训练总结
print("\n📊 训练总结:")
from visualization.clean_charts import get_summary_text
summary = get_summary_text(mock_env, 'TD3')
print(summary)

print("\n" + "=" * 60)
print("✅ 图表生成完成！请查看:")
print("   1. training_overview_FIXED.png - 完整的训练总览")
print("   2. training_overview_FIXED_hotspot.png - 热点分析")
print("   3. objective_analysis_FIXED.png - 目标函数分解")
print("=" * 60)
