#!/usr/bin/env python3
"""
快速查看策略模型缓存状态
"""
from pathlib import Path
import json

cache_dir = Path("results/strategy_model_cache")

if not cache_dir.exists():
    print("❌ 缓存目录不存在，还没有运行过任何实验")
else:
    print("📦 策略模型缓存状态\n" + "="*50)
    
    total_cached = 0
    for strategy_dir in sorted(cache_dir.iterdir()):
        if strategy_dir.is_dir() and not strategy_dir.name.startswith('.'):
            cache_count = len(list(strategy_dir.glob("ep*")))
            if cache_count > 0:
                print(f"\n策略: {strategy_dir.name}")
                print(f"  缓存数量: {cache_count}")
                
                # 显示前3个缓存
                for i, cache_path in enumerate(sorted(strategy_dir.glob("ep*"))[:3]):
                    config_file = cache_path / "config.json"
                    if config_file.exists():
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = json.load(f)
                        episodes = config.get('episodes', '?')
                        seed = config.get('seed', '?')
                        print(f"    [{i+1}] {cache_path.name} (ep={episodes}, seed={seed})")
                
                total_cached += cache_count
    
    print(f"\n{'='*50}")
    print(f"总计: {total_cached} 个缓存模型")

