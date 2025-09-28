#!/usr/bin/env python3
"""
快速VEC演示 - 展示基于时间的用户行为模式效果
"""

import numpy as np
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from collections import defaultdict

# 导入组件
from user_behavior_patterns import TimeBasedBehaviorModel, UserType
from realistic_vec_cache import RealisticVECCacheSimulator, ContentType

def quick_demo():
    """快速演示基于时间的用户行为"""
    print("🚀 VEC用户行为模式演示")
    print("="*50)
    
    # 创建行为模型
    behavior_model = TimeBasedBehaviorModel()
    
    # 创建RSU缓存
    rsu_cache = RealisticVECCacheSimulator("demo_rsu", "rsu", (39.9042, 116.4074))
    
    # 生成不同类型用户
    users = behavior_model.generate_realistic_user_profiles(5)
    print(f"👥 生成了 {len(users)} 个用户:")
    for user in users:
        print(f"  - {user.user_id}: {user.user_type.value}")
    
    # 模拟一天中不同时间段的行为
    time_scenarios = [
        (datetime(2024, 1, 15, 8, 0), "早高峰"),
        (datetime(2024, 1, 15, 12, 30), "午休时间"),
        (datetime(2024, 1, 15, 18, 0), "晚高峰"),
        (datetime(2024, 1, 15, 22, 0), "夜间"),
        (datetime(2024, 1, 13, 15, 0), "周末下午"),
    ]
    
    total_stats = defaultdict(int)
    user_type_stats = defaultdict(lambda: defaultdict(int))
    time_pattern_stats = defaultdict(lambda: defaultdict(int))
    
    print(f"\n🎬 开始行为模拟...")
    
    for sim_time, time_desc in time_scenarios:
        print(f"\n⏰ {time_desc} ({sim_time.strftime('%Y-%m-%d %H:%M')})")
        print("-" * 30)
        
        scenario_requests = 0
        scenario_hits = 0
        scenario_content_types = defaultdict(int)
        
        # 每个用户在此时间段发起多个请求
        for user in users:
            for _ in range(np.random.randint(3, 8)):  # 每用户3-8个请求
                # 计算用户需求概率
                demand_probs = behavior_model.calculate_content_demand_probability(
                    user, user.home_location, sim_time
                )
                
                # 选择内容类型
                if demand_probs:
                    content_types = list(demand_probs.keys())
                    probabilities = list(demand_probs.values())
                    selected_type = np.random.choice(content_types, p=probabilities)
                    
                    # 生成实际内容请求
                    content = rsu_cache.generate_realistic_content_request(user.home_location)
                    
                    # 强制设置为选中的内容类型
                    content.content_type = ContentType(selected_type) if selected_type in [ct.value for ct in ContentType] else ContentType.TRAFFIC_INFO
                    
                    # 处理请求
                    hit, action, metrics = rsu_cache.request_content(content, user.home_location)
                    
                    # 统计
                    scenario_requests += 1
                    total_stats['total_requests'] += 1
                    
                    if hit:
                        scenario_hits += 1
                        total_stats['cache_hits'] += 1
                    
                    scenario_content_types[selected_type] += 1
                    user_type_stats[user.user_type.value][selected_type] += 1
                    time_pattern_stats[time_desc][selected_type] += 1
        
        # 输出场景统计
        hit_rate = scenario_hits / max(1, scenario_requests)
        print(f"📊 请求数: {scenario_requests}, 命中数: {scenario_hits}, 命中率: {hit_rate:.1%}")
        
        # 显示内容类型分布
        print("📱 内容类型需求:")
        sorted_content = sorted(scenario_content_types.items(), key=lambda x: x[1], reverse=True)
        for content_type, count in sorted_content[:5]:
            percentage = count / max(1, scenario_requests) * 100
            print(f"  {content_type}: {count} ({percentage:.1f}%)")
    
    # 最终统计报告
    print(f"\n📈 最终统计报告")
    print("="*50)
    
    overall_hit_rate = total_stats['cache_hits'] / max(1, total_stats['total_requests'])
    print(f"🎯 总请求数: {total_stats['total_requests']}")
    print(f"💎 总命中数: {total_stats['cache_hits']}")
    print(f"📊 整体命中率: {overall_hit_rate:.1%}")
    
    # 用户类型行为分析
    print(f"\n👥 用户类型行为分析:")
    for user_type, behaviors in user_type_stats.items():
        if behaviors:
            total_user_requests = sum(behaviors.values())
            top_content = max(behaviors.items(), key=lambda x: x[1])
            print(f"  {user_type}:")
            print(f"    总请求: {total_user_requests}")
            print(f"    最爱: {top_content[0]} ({top_content[1]}次, {top_content[1]/total_user_requests:.1%})")
    
    # 时间模式分析
    print(f"\n⏰ 时间模式分析:")
    for time_pattern, behaviors in time_pattern_stats.items():
        if behaviors:
            total_time_requests = sum(behaviors.values())
            top_content = max(behaviors.items(), key=lambda x: x[1])
            print(f"  {time_pattern}:")
            print(f"    主要需求: {top_content[0]} ({top_content[1]/total_time_requests:.1%})")
    
    # 缓存性能分析
    cache_stats = rsu_cache.get_comprehensive_statistics()
    print(f"\n🏢 缓存性能分析:")
    print(f"  缓存利用率: {cache_stats['resource_utilization']['usage_ratio']:.1%}")
    print(f"  缓存项目数: {cache_stats['resource_utilization']['cached_items']}")
    print(f"  新鲜内容命中率: {cache_stats['cache_performance']['fresh_hit_rate']:.1%}")
    print(f"  地理相关命中率: {cache_stats['cache_performance']['location_hit_rate']:.1%}")
    
    content_dist = cache_stats['content_distribution']
    if content_dist:
        print(f"  缓存内容分布:")
        for content_type, count in sorted(content_dist.items(), key=lambda x: x[1], reverse=True):
            print(f"    {content_type}: {count}项")
    
    print(f"\n✅ 演示完成！")

if __name__ == "__main__":
    quick_demo()
