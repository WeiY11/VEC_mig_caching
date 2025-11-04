#!/usr/bin/env python3
"""
TD3 策略模型缓存系统
======================

【功能】
避免对比实验中重复训练相同策略配置，通过缓存训练好的模型实现：
- 自动生成唯一的缓存键（基于策略配置、seed、episodes等）
- 训练后自动保存模型到共享缓存目录
- 实验开始前自动检查缓存，如有则直接加载评估
- 支持缓存管理（清理、统计、导出）

【使用方式】
```python
# 在strategy_runner.py中自动使用，无需手动调用
cache = StrategyModelCache()
cached = cache.get_cached_model(strategy_key, episodes, seed, overrides)
if cached:
    # 使用缓存的模型
else:
    # 训练新模型，然后保存
    cache.save_model(strategy_key, episodes, seed, overrides, model_data)
```

【缓存目录结构】
```
results/strategy_model_cache/
  ├── cache_metadata.json        # 缓存元数据
  ├── local-only/
  │   ├── ep800_seed42_hash1234/
  │   │   ├── model.pth          # 模型参数
  │   │   ├── config.json        # 配置信息
  │   │   └── metrics.json       # 训练指标
  │   └── ...
  ├── comprehensive-migration/
  │   └── ...
  └── ...
```
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# ========== 添加项目根目录到Python路径 ==========
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# ========== 缓存配置 ==========
DEFAULT_CACHE_DIR = Path("results/strategy_model_cache")
CACHE_METADATA_FILE = "cache_metadata.json"


class StrategyModelCache:
    """
    策略模型缓存管理器
    
    【功能】
    - 生成唯一的缓存标识
    - 保存和加载训练好的模型
    - 管理缓存元数据
    - 提供缓存清理和统计功能
    """
    
    def __init__(self, cache_dir: Optional[Path] = None):
        """
        初始化缓存管理器
        
        【参数】
        cache_dir: Path - 缓存根目录（默认：results/strategy_model_cache）
        """
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / CACHE_METADATA_FILE
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """加载缓存元数据"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except Exception as e:
                print(f"⚠️ 加载缓存元数据失败: {e}，使用空元数据")
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """保存缓存元数据"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠️ 保存缓存元数据失败: {e}")
    
    def _generate_cache_key(
        self,
        strategy_key: str,
        episodes: int,
        seed: int,
        overrides: Dict[str, Any],
    ) -> str:
        """
        生成唯一的缓存键
        
        【原理】
        基于策略配置的关键参数生成哈希值，确保相同配置使用相同缓存
        
        【参数】
        strategy_key: str - 策略标识（如 "comprehensive-migration"）
        episodes: int - 训练轮数
        seed: int - 随机种子
        overrides: Dict - 场景覆盖参数（如车辆数、RSU数等）
        
        【返回值】
        str - 8位哈希缓存键（如 "a3b7c9d1"）
        """
        # 提取影响模型性能的关键参数
        key_params = {
            "strategy": strategy_key,
            "episodes": episodes,
            "seed": seed,
            # 场景参数（排序确保一致性）
            "overrides": sorted(
                [(k, v) for k, v in (overrides or {}).items()],
                key=lambda x: x[0]
            ),
        }
        
        # 生成JSON字符串并计算哈希
        json_str = json.dumps(key_params, sort_keys=True, ensure_ascii=False)
        hash_obj = hashlib.md5(json_str.encode('utf-8'))
        return hash_obj.hexdigest()[:8]  # 使用前8位（足够区分）
    
    def _get_cache_path(
        self,
        strategy_key: str,
        episodes: int,
        seed: int,
        cache_hash: str,
    ) -> Path:
        """
        获取缓存目录路径
        
        【返回值】
        Path - 缓存目录路径（如 "results/strategy_model_cache/local-only/ep800_seed42_hash1234/"）
        """
        cache_name = f"ep{episodes}_seed{seed}_{cache_hash}"
        return self.cache_dir / strategy_key / cache_name
    
    def has_cached_model(
        self,
        strategy_key: str,
        episodes: int,
        seed: int,
        overrides: Dict[str, Any],
    ) -> bool:
        """
        检查是否存在缓存的模型
        
        【参数】
        strategy_key: str - 策略标识
        episodes: int - 训练轮数
        seed: int - 随机种子
        overrides: Dict - 场景覆盖参数
        
        【返回值】
        bool - 是否存在有效缓存
        """
        cache_hash = self._generate_cache_key(strategy_key, episodes, seed, overrides)
        cache_path = self._get_cache_path(strategy_key, episodes, seed, cache_hash)
        
        # 检查必需文件是否存在
        required_files = [
            cache_path / "config.json",
            cache_path / "metrics.json",
        ]
        
        return all(f.exists() for f in required_files)
    
    def get_cached_model(
        self,
        strategy_key: str,
        episodes: int,
        seed: int,
        overrides: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """
        获取缓存的模型数据
        
        【参数】
        strategy_key: str - 策略标识
        episodes: int - 训练轮数
        seed: int - 随机种子
        overrides: Dict - 场景覆盖参数
        
        【返回值】
        Dict | None - 如果存在缓存，返回包含配置和指标的字典；否则返回None
        """
        if not self.has_cached_model(strategy_key, episodes, seed, overrides):
            return None
        
        cache_hash = self._generate_cache_key(strategy_key, episodes, seed, overrides)
        cache_path = self._get_cache_path(strategy_key, episodes, seed, cache_hash)
        
        try:
            # 加载配置
            with open(cache_path / "config.json", 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            # 加载指标
            with open(cache_path / "metrics.json", 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            # 更新访问时间
            cache_id = f"{strategy_key}/{cache_hash}"
            if cache_id in self.metadata:
                self.metadata[cache_id]["last_accessed"] = datetime.now().isoformat()
                self.metadata[cache_id]["access_count"] = self.metadata[cache_id].get("access_count", 0) + 1
                self._save_metadata()
            
            print(f"✅ 使用缓存模型: {strategy_key} (ep{episodes}_seed{seed}_{cache_hash})")
            
            return {
                "config": config_data,
                "metrics": metrics_data,
                "cache_path": str(cache_path),
                "cache_hash": cache_hash,
            }
            
        except Exception as e:
            print(f"⚠️ 加载缓存失败: {e}，将重新训练")
            return None
    
    def save_model(
        self,
        strategy_key: str,
        episodes: int,
        seed: int,
        overrides: Dict[str, Any],
        outcome: Dict[str, Any],
        model_state: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """
        保存模型到缓存
        
        【参数】
        strategy_key: str - 策略标识
        episodes: int - 训练轮数
        seed: int - 随机种子
        overrides: Dict - 场景覆盖参数
        outcome: Dict - 训练结果（包含episode_metrics等）
        model_state: Dict | None - 模型参数（可选，用于后续加载）
        
        【返回值】
        Path - 缓存目录路径
        """
        cache_hash = self._generate_cache_key(strategy_key, episodes, seed, overrides)
        cache_path = self._get_cache_path(strategy_key, episodes, seed, cache_hash)
        cache_path.mkdir(parents=True, exist_ok=True)
        
        try:
            # 保存配置
            config_data = {
                "strategy_key": strategy_key,
                "episodes": episodes,
                "seed": seed,
                "overrides": overrides,
                "cache_hash": cache_hash,
                "created_at": datetime.now().isoformat(),
            }
            with open(cache_path / "config.json", 'w', encoding='utf-8') as f:
                json.dump(config_data, f, indent=2, ensure_ascii=False)
            
            # 保存指标（包括episode_metrics）
            metrics_data = {
                "episode_metrics": outcome.get("episode_metrics", {}),
                "final_metrics": {
                    "avg_reward": outcome.get("avg_reward", 0.0),
                    "total_episodes": episodes,
                },
            }
            with open(cache_path / "metrics.json", 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2, ensure_ascii=False)
            
            # 保存模型参数（如果提供）
            if model_state is not None:
                torch.save(model_state, cache_path / "model.pth")
            
            # 更新元数据
            cache_id = f"{strategy_key}/{cache_hash}"
            self.metadata[cache_id] = {
                "strategy_key": strategy_key,
                "episodes": episodes,
                "seed": seed,
                "cache_hash": cache_hash,
                "cache_path": str(cache_path),
                "created_at": datetime.now().isoformat(),
                "last_accessed": datetime.now().isoformat(),
                "access_count": 0,
                "size_mb": self._get_dir_size_mb(cache_path),
            }
            self._save_metadata()
            
            print(f"💾 模型已缓存: {strategy_key} (ep{episodes}_seed{seed}_{cache_hash})")
            
            return cache_path
            
        except Exception as e:
            print(f"⚠️ 保存缓存失败: {e}")
            # 清理部分保存的文件
            if cache_path.exists():
                shutil.rmtree(cache_path, ignore_errors=True)
            raise
    
    def _get_dir_size_mb(self, path: Path) -> float:
        """计算目录大小（MB）"""
        total_size = 0
        for file in path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        return total_size / (1024 * 1024)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        【返回值】
        Dict - 缓存统计（策略数量、总大小、命中率等）
        """
        total_size_mb = sum(item.get("size_mb", 0) for item in self.metadata.values())
        total_access = sum(item.get("access_count", 0) for item in self.metadata.values())
        
        by_strategy = {}
        for cache_id, item in self.metadata.items():
            strategy = item["strategy_key"]
            if strategy not in by_strategy:
                by_strategy[strategy] = {"count": 0, "size_mb": 0, "total_access": 0}
            by_strategy[strategy]["count"] += 1
            by_strategy[strategy]["size_mb"] += item.get("size_mb", 0)
            by_strategy[strategy]["total_access"] += item.get("access_count", 0)
        
        return {
            "total_cached_models": len(self.metadata),
            "total_size_mb": round(total_size_mb, 2),
            "total_access_count": total_access,
            "by_strategy": by_strategy,
            "cache_dir": str(self.cache_dir),
        }
    
    def print_cache_stats(self) -> None:
        """打印缓存统计信息"""
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 70)
        print("📊 策略模型缓存统计")
        print("=" * 70)
        print(f"缓存目录: {stats['cache_dir']}")
        print(f"缓存模型数: {stats['total_cached_models']}")
        print(f"总缓存大小: {stats['total_size_mb']:.2f} MB")
        print(f"总访问次数: {stats['total_access_count']}")
        
        if stats['by_strategy']:
            print("\n按策略统计:")
            print(f"{'策略':<35} {'模型数':<10} {'大小(MB)':<12} {'访问次数':<10}")
            print("-" * 70)
            for strategy, info in sorted(stats['by_strategy'].items()):
                print(f"{strategy:<35} {info['count']:<10} {info['size_mb']:<12.2f} {info['total_access']:<10}")
        
        print("=" * 70 + "\n")
    
    def clear_cache(
        self,
        strategy_key: Optional[str] = None,
        older_than_days: Optional[int] = None,
    ) -> int:
        """
        清理缓存
        
        【参数】
        strategy_key: str | None - 仅清理指定策略（None表示全部）
        older_than_days: int | None - 仅清理N天前的缓存（None表示全部）
        
        【返回值】
        int - 清理的缓存数量
        """
        from datetime import timedelta
        
        removed_count = 0
        to_remove = []
        
        for cache_id, item in self.metadata.items():
            # 策略过滤
            if strategy_key and item["strategy_key"] != strategy_key:
                continue
            
            # 时间过滤
            if older_than_days:
                created_at = datetime.fromisoformat(item["created_at"])
                age = datetime.now() - created_at
                if age < timedelta(days=older_than_days):
                    continue
            
            # 删除缓存目录
            cache_path = Path(item["cache_path"])
            if cache_path.exists():
                shutil.rmtree(cache_path)
                removed_count += 1
            
            to_remove.append(cache_id)
        
        # 更新元数据
        for cache_id in to_remove:
            del self.metadata[cache_id]
        
        self._save_metadata()
        
        print(f"🗑️ 清理了 {removed_count} 个缓存模型")
        
        return removed_count


# ========== 全局缓存实例 ==========
_global_cache: Optional[StrategyModelCache] = None


def get_global_cache() -> StrategyModelCache:
    """获取全局缓存实例（单例模式）"""
    global _global_cache
    if _global_cache is None:
        _global_cache = StrategyModelCache()
    return _global_cache


# ========== 命令行工具 ==========
def main():
    """命令行缓存管理工具"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="TD3 策略模型缓存管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 查看缓存统计
  python strategy_model_cache.py --stats
  
  # 清理所有缓存
  python strategy_model_cache.py --clear-all
  
  # 清理指定策略的缓存
  python strategy_model_cache.py --clear --strategy local-only
  
  # 清理7天前的缓存
  python strategy_model_cache.py --clear --older-than 7
        """
    )
    
    parser.add_argument("--stats", action="store_true",
                       help="显示缓存统计信息")
    parser.add_argument("--clear", action="store_true",
                       help="清理缓存")
    parser.add_argument("--clear-all", action="store_true",
                       help="清理所有缓存")
    parser.add_argument("--strategy", type=str,
                       help="指定策略（用于清理）")
    parser.add_argument("--older-than", type=int, metavar="DAYS",
                       help="清理N天前的缓存")
    parser.add_argument("--cache-dir", type=str,
                       help="指定缓存目录（默认：results/strategy_model_cache）")
    
    args = parser.parse_args()
    
    # 初始化缓存
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    cache = StrategyModelCache(cache_dir=cache_dir)
    
    # 执行操作
    if args.stats or (not args.clear and not args.clear_all):
        cache.print_cache_stats()
    
    if args.clear or args.clear_all:
        strategy = args.strategy if args.clear else None
        older_than = args.older_than
        
        if args.clear_all and not args.older_than and not args.strategy:
            confirm = input("⚠️ 确认清理所有缓存? (y/n): ").strip().lower()
            if confirm != 'y':
                print("已取消")
                return
        
        cache.clear_cache(strategy_key=strategy, older_than_days=older_than)
        print("\n清理后的统计:")
        cache.print_cache_stats()


if __name__ == "__main__":
    main()

