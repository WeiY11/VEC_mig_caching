#!/usr/bin/env python3
"""
缓存系统快速测试脚本
===================

【功能】
快速验证策略模型缓存系统是否正常工作

【测试内容】
1. 第1次运行: 训练模型并保存到缓存
2. 第2次运行: 从缓存加载（应该快很多）
3. 验证结果一致性

【使用方式】
```bash
# 快速测试（10轮训练）
python experiments/camtd3_strategy_suite/test_cache_system.py

# 完整测试（100轮训练，更可靠但耗时更长）
python experiments/camtd3_strategy_suite/test_cache_system.py --episodes 100
```
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ========== 添加项目根目录到Python路径 ==========
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.camtd3_strategy_suite.strategy_runner import run_strategy_suite
from experiments.camtd3_strategy_suite.strategy_model_cache import get_global_cache


def print_banner(text: str, char: str = "="):
    """打印横幅"""
    width = 70
    print("\n" + char * width)
    print(f"{text:^{width}}")
    print(char * width)


def test_cache_system(episodes: int = 10, seed: int = 42):
    """
    测试缓存系统
    
    【参数】
    episodes: int - 测试训练轮数（默认10，快速测试）
    seed: int - 随机种子
    """
    
    print_banner("🧪 策略模型缓存系统测试")
    
    # 测试场景配置
    test_scenario = {
        "num_vehicles": 12,
        "num_rsus": 4,
        "num_uavs": 2,
        "override_topology": True,
    }
    
    # 仅测试2个策略（节省时间）
    test_strategies = ["local-only", "comprehensive-migration"]
    
    print(f"\n📋 测试配置:")
    print(f"  - 训练轮数: {episodes}")
    print(f"  - 随机种子: {seed}")
    print(f"  - 测试策略: {test_strategies}")
    print(f"  - 场景参数: {test_scenario}")
    
    # ========== 第1次运行: 训练并缓存 ==========
    print_banner("第1次运行: 训练并保存到缓存", "-")
    
    print("\n⏳ 开始训练（预计 1-2 分钟）...\n")
    
    start_time_1 = time.time()
    results_1 = run_strategy_suite(
        override_scenario=test_scenario,
        episodes=episodes,
        seed=seed,
        silent=True,
        strategies=test_strategies,
    )
    elapsed_1 = time.time() - start_time_1
    
    print(f"\n✅ 第1次运行完成")
    print(f"   用时: {elapsed_1:.2f} 秒 ({elapsed_1/60:.2f} 分钟)")
    
    # 显示结果
    print(f"\n   结果:")
    for strategy, metrics in results_1.items():
        print(f"     - {strategy}:")
        print(f"         Cost: {metrics['raw_cost']:.4f}")
        print(f"         From Cache: {metrics.get('from_cache', False)}")
    
    # ========== 第2次运行: 从缓存加载 ==========
    print_banner("第2次运行: 从缓存加载", "-")
    
    print("\n⏳ 开始加载缓存（应该很快）...\n")
    
    start_time_2 = time.time()
    results_2 = run_strategy_suite(
        override_scenario=test_scenario,
        episodes=episodes,
        seed=seed,
        silent=True,
        strategies=test_strategies,
    )
    elapsed_2 = time.time() - start_time_2
    
    print(f"\n✅ 第2次运行完成")
    print(f"   用时: {elapsed_2:.2f} 秒")
    
    # 显示结果
    print(f"\n   结果:")
    for strategy, metrics in results_2.items():
        print(f"     - {strategy}:")
        print(f"         Cost: {metrics['raw_cost']:.4f}")
        print(f"         From Cache: {metrics.get('from_cache', False)}")
    
    # ========== 结果对比 ==========
    print_banner("测试结果对比", "-")
    
    # 计算加速比
    speedup = elapsed_1 / max(elapsed_2, 0.001)
    
    print(f"\n⏱️ 性能对比:")
    print(f"   第1次运行（训练）: {elapsed_1:.2f} 秒 ({elapsed_1/60:.2f} 分钟)")
    print(f"   第2次运行（缓存）: {elapsed_2:.2f} 秒")
    print(f"   加速比: {speedup:.1f}x")
    
    if speedup > 10:
        print(f"   🚀 缓存效果显著！节省了 {100*(1-1/speedup):.1f}% 的时间")
    elif speedup > 2:
        print(f"   ✅ 缓存有效，节省了 {100*(1-1/speedup):.1f}% 的时间")
    else:
        print(f"   ⚠️ 缓存效果不明显，可能没有正确使用缓存")
    
    # 验证结果一致性
    print(f"\n📊 结果一致性:")
    all_consistent = True
    for strategy in test_strategies:
        cost_1 = results_1[strategy]['raw_cost']
        cost_2 = results_2[strategy]['raw_cost']
        diff = abs(cost_1 - cost_2)
        consistent = diff < 1e-6
        
        status = "✅" if consistent else "❌"
        print(f"   {status} {strategy}:")
        print(f"      第1次 Cost: {cost_1:.6f}")
        print(f"      第2次 Cost: {cost_2:.6f}")
        print(f"      差异: {diff:.6e}")
        
        all_consistent = all_consistent and consistent
    
    # ========== 缓存统计 ==========
    print_banner("缓存统计", "-")
    cache = get_global_cache()
    cache.print_cache_stats()
    
    # ========== 最终总结 ==========
    print_banner("🎉 测试总结", "=")
    
    success = True
    
    # 检查1: 第2次运行是否使用了缓存
    cache_used = all(results_2[s].get('from_cache', False) for s in test_strategies)
    if cache_used:
        print("✅ 缓存加载: 成功")
    else:
        print("❌ 缓存加载: 失败（第2次运行未使用缓存）")
        success = False
    
    # 检查2: 结果是否一致
    if all_consistent:
        print("✅ 结果一致性: 通过")
    else:
        print("❌ 结果一致性: 失败（两次运行结果不一致）")
        success = False
    
    # 检查3: 性能提升
    if speedup > 5:
        print(f"✅ 性能提升: 显著 ({speedup:.1f}x)")
    elif speedup > 2:
        print(f"⚠️ 性能提升: 一般 ({speedup:.1f}x)")
    else:
        print(f"❌ 性能提升: 不明显 ({speedup:.1f}x)")
        success = False
    
    # 最终结论
    print("\n" + "=" * 70)
    if success:
        print("🎉 缓存系统测试通过！")
        print("   可以正常使用缓存系统来加速对比实验。")
    else:
        print("⚠️ 缓存系统测试未完全通过")
        print("   请检查上述失败项，或联系开发者。")
    print("=" * 70)
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="测试策略模型缓存系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 快速测试（10轮训练，1-2分钟）
  python test_cache_system.py
  
  # 完整测试（100轮训练，10-20分钟）
  python test_cache_system.py --episodes 100
  
  # 使用不同随机种子
  python test_cache_system.py --episodes 10 --seed 123
        """
    )
    
    parser.add_argument("--episodes", type=int, default=10,
                       help="测试训练轮数 (默认: 10，快速测试)")
    parser.add_argument("--seed", type=int, default=42,
                       help="随机种子 (默认: 42)")
    
    args = parser.parse_args()
    
    # 运行测试
    success = test_cache_system(episodes=args.episodes, seed=args.seed)
    
    # 返回码
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

