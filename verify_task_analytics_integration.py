#!/usr/bin/env python3
"""
验证任务处理方式分布统计功能集成是否正确

此脚本检查：
1. 必要的模块是否存在
2. TaskAnalyticsTracker是否可以正常导入
3. 基本功能是否可用
"""

import sys
import os

def verify_modules():
    """验证所有必要的模块"""
    print("=" * 80)
    print("🔍 验证任务处理方式分布统计功能集成")
    print("=" * 80)
    
    modules_to_check = [
        ('utils.task_offloading_analytics', ['TaskDistribution', 'EpisodeTaskStatistics', 'TaskOffloadingAnalytics']),
        ('utils.training_analytics_integration', ['TaskAnalyticsTracker', 'create_analytics_callback']),
    ]
    
    all_ok = True
    
    for module_name, classes in modules_to_check:
        print(f"\n📦 检查模块: {module_name}")
        try:
            module = __import__(module_name, fromlist=classes)
            print(f"   ✓ 模块存在")
            
            for cls_name in classes:
                if hasattr(module, cls_name):
                    print(f"   ✓ 类 '{cls_name}' 存在")
                else:
                    print(f"   ✗ 类 '{cls_name}' 不存在")
                    all_ok = False
        except ImportError as e:
            print(f"   ✗ 模块导入失败: {e}")
            all_ok = False
    
    return all_ok


def verify_basic_functionality():
    """验证基本功能"""
    print("\n" + "=" * 80)
    print("🧪 验证基本功能")
    print("=" * 80)
    
    try:
        from utils.task_offloading_analytics import TaskOffloadingAnalytics, TaskDistribution
        from utils.training_analytics_integration import TaskAnalyticsTracker
        
        # 测试 TaskOffloadingAnalytics
        print("\n📊 测试 TaskOffloadingAnalytics...")
        analytics = TaskOffloadingAnalytics()
        
        # 模拟一个episode
        analytics.start_episode(1)
        print("   ✓ start_episode() 正常")
        
        # 模拟几个step
        for step in range(5):
            step_result = {
                'generated_tasks': 10,
                'local_tasks': 3,
                'remote_tasks': 6,
                'dropped_tasks': 1,
                'local_cache_hits': 2,
            }
            analytics.record_step(step, step_result)
        print("   ✓ record_step() 正常")
        
        # 结束episode
        stats = analytics.finalize_episode()
        print("   ✓ finalize_episode() 正常")
        
        if stats:
            print(f"\n   📈 Episode统计:")
            print(f"      - 总生成任务: {stats.total_generated}")
            print(f"      - 本地处理: {stats.total_local} ({stats.local_ratio:.1%})")
            print(f"      - RSU处理: {stats.total_rsu} ({stats.rsu_ratio:.1%})")
            print(f"      - UAV处理: {stats.total_uav} ({stats.uav_ratio:.1%})")
            print(f"      - 被丢弃: {stats.total_dropped} ({stats.drop_ratio:.1%})")
            print(f"      - 成功率: {stats.success_ratio:.1%}")
        
        # 测试 TaskAnalyticsTracker
        print("\n📊 测试 TaskAnalyticsTracker...")
        tracker = TaskAnalyticsTracker(enable_logging=False)
        
        # 模拟多个episodes
        for ep in range(1, 4):
            tracker.start_episode(ep)
            for step in range(10):
                step_result = {
                    'generated_tasks': 8 + ep,
                    'local_tasks': 3 + ep // 2,
                    'remote_tasks': 4 + ep // 3,
                    'dropped_tasks': 1,
                }
                tracker.record_step(step, step_result)
            tracker.end_episode()
        
        print("   ✓ start_episode() 正常")
        print("   ✓ record_step() 正常")
        print("   ✓ end_episode() 正常")
        
        # 获取汇总
        summary = tracker.get_training_summary()
        if summary and 'error' not in summary:
            print(f"\n   📊 训练汇总:")
            print(f"      - 总Episode数: {summary['total_episodes']}")
            print(f"      - 总步数: {summary['total_steps']}")
            print(f"      - 本地处理占比: {summary['local_ratio_avg']:.1%}")
            print(f"      - RSU处理占比: {summary['rsu_ratio_avg']:.1%}")
            print(f"      - UAV处理占比: {summary['uav_ratio_avg']:.1%}")
            print(f"      - 平均成功率: {summary['success_rate_avg']:.1%}")
        
        # 获取演化趋势
        trends = tracker.get_evolution_trend()
        if trends and trends.get('episodes'):
            print(f"\n   📈 演化趋势 (最后1个episode):")
            last_idx = -1
            print(f"      - 本地处理占比: {trends['local_ratio'][last_idx]:.1%}")
            print(f"      - RSU处理占比: {trends['rsu_ratio'][last_idx]:.1%}")
            print(f"      - UAV处理占比: {trends['uav_ratio'][last_idx]:.1%}")
            print(f"      - 成功率: {trends['success_ratio'][last_idx]:.1%}")
        
        # 测试CSV导出
        print("\n📊 测试CSV导出...")
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            csv_path = f.name
        
        try:
            tracker.export_csv(csv_path)
            if os.path.exists(csv_path):
                print(f"   ✓ CSV导出成功: {csv_path}")
                os.remove(csv_path)
            else:
                print(f"   ✗ CSV导出失败")
                return False
        except Exception as e:
            print(f"   ✗ CSV导出异常: {e}")
            if os.path.exists(csv_path):
                os.remove(csv_path)
            return False
        
        return True
        
    except Exception as e:
        print(f"   ✗ 基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def verify_train_single_agent_integration():
    """验证train_single_agent.py的集成"""
    print("\n" + "=" * 80)
    print("🔗 验证 train_single_agent.py 集成")
    print("=" * 80)
    
    try:
        with open('train_single_agent.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = [
            ('导入TaskAnalyticsTracker', 'from utils.training_analytics_integration import TaskAnalyticsTracker'),
            ('创建analytics_tracker', 'analytics_tracker = TaskAnalyticsTracker('),
            ('start_episode调用', 'analytics_tracker.start_episode(episode)'),
            ('record_step调用', 'analytics_tracker.record_step('),
            ('end_episode调用', 'analytics_tracker.end_episode()'),
            ('print_training_summary调用', 'analytics_tracker.print_training_summary()'),
            ('print_summary调用', 'analytics_tracker.print_summary('),
            ('export_csv调用', 'analytics_tracker.export_csv('),
        ]
        
        all_ok = True
        for check_name, pattern in checks:
            if pattern in content:
                print(f"   ✓ {check_name}")
            else:
                print(f"   ✗ {check_name} - 未找到")
                all_ok = False
        
        # 额外检查：step_stats_list返回
        if 'step_stats_list' in content:
            print(f"   ✓ run_episode返回step_stats_list")
        else:
            print(f"   ✗ run_episode返回step_stats_list - 未找到")
            all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"   ✗ 检查失败: {e}")
        return False


def main():
    """主函数"""
    print("\n")
    
    # 检查modules
    modules_ok = verify_modules()
    
    # 检查基本功能
    functionality_ok = verify_basic_functionality()
    
    # 检查train_single_agent.py集成
    integration_ok = verify_train_single_agent_integration()
    
    # 总结
    print("\n" + "=" * 80)
    print("📋 验证总结")
    print("=" * 80)
    
    results = [
        ("模块检查", modules_ok),
        ("基本功能", functionality_ok),
        ("train_single_agent.py集成", integration_ok),
    ]
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:30s} {status}")
    
    all_passed = all(r for _, r in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✅ 所有验证均已通过！任务分布统计功能已成功集成。")
        print("\n📖 使用指南:")
        print("   1. 运行训练: python train_single_agent.py --algorithm TD3")
        print("   2. 查看日志输出中的任务分布统计")
        print("   3. 训练完成后查看 results/single_agent/td3/task_distribution_analysis.csv")
        print("   4. 详细说明请参考 docs/TASK_DISTRIBUTION_STATISTICS_USAGE.md")
        return 0
    else:
        print("❌ 部分验证失败。请检查集成是否完整。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
