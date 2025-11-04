#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练监控脚本 - 实时查看TD3训练进度
"""
import os
import json
import time
import glob
from datetime import datetime

def find_latest_results():
    """查找最新的训练结果文件"""
    results_dir = "results/single_agent/td3"
    if not os.path.exists(results_dir):
        return None
    
    # 查找最新的JSON文件
    json_files = glob.glob(os.path.join(results_dir, "training_results_*.json"))
    if not json_files:
        return None
    
    latest = max(json_files, key=os.path.getmtime)
    return latest

def display_progress(filepath):
    """显示训练进度"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        episodes = data.get('episodes', [])
        if not episodes:
            print("⏳ 等待训练数据...")
            return
        
        latest = episodes[-1]
        episode_num = latest.get('episode', 0)
        total_episodes = data.get('config', {}).get('num_episodes', 800)
        
        # 清屏（Windows）
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("=" * 80)
        print(f"🎯 TD3 训练监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        print(f"\n📊 进度: {episode_num}/{total_episodes} ({episode_num/total_episodes*100:.1f}%)")
        print(f"{'█' * int(episode_num/total_episodes*50)}{'░' * (50-int(episode_num/total_episodes*50))}")
        
        print(f"\n📈 最新指标 (Episode {episode_num}):")
        print(f"  • Reward:        {latest.get('reward', 0):.4f}")
        print(f"  • 平均时延:      {latest.get('avg_delay', 0):.4f}s")
        print(f"  • 平均能耗:      {latest.get('avg_energy', 0):.2f}J")
        print(f"  • 任务完成率:    {latest.get('completion_rate', 0)*100:.2f}%")
        print(f"  • 迁移成功率:    {latest.get('migration_success_rate', 0)*100:.2f}%")
        
        training_stats = latest.get('training_stats', {})
        print(f"\n🎓 训练统计:")
        print(f"  • Actor Loss:    {training_stats.get('actor_loss_avg', 0):.6f}")
        print(f"  • Critic Loss:   {training_stats.get('critic_loss_avg', 0):.6f}")
        print(f"  • Exploration:   {training_stats.get('exploration_noise', 0):.4f}")
        print(f"  • Buffer Size:   {training_stats.get('buffer_size', 0)}")
        
        # 计算最近100轮的趋势
        if len(episodes) >= 100:
            recent_rewards = [ep.get('reward', 0) for ep in episodes[-100:]]
            avg_reward = sum(recent_rewards) / len(recent_rewards)
            variance = sum((r - avg_reward)**2 for r in recent_rewards) / len(recent_rewards)
            
            print(f"\n📉 最近100轮趋势:")
            print(f"  • 平均奖励:      {avg_reward:.4f}")
            print(f"  • 奖励方差:      {variance:.4f}")
            print(f"  • 稳定性:        {'✅ 优秀' if variance < 0.15 else '⚠️ 一般' if variance < 0.25 else '❌ 需优化'}")
        
        print(f"\n📁 结果文件: {filepath}")
        print("=" * 80)
        print("按 Ctrl+C 停止监控")
        
    except json.JSONDecodeError:
        print("⚠️  JSON文件读取中...")
    except Exception as e:
        print(f"❌ 错误: {e}")

def main():
    """主函数"""
    print("🔍 正在搜索训练结果...")
    
    while True:
        filepath = find_latest_results()
        
        if filepath:
            display_progress(filepath)
        else:
            print("⏳ 等待训练开始...")
        
        time.sleep(5)  # 每5秒刷新一次

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 监控已停止")

