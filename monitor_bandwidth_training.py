#!/usr/bin/env python3
"""
实时监控带宽对比实验训练进度
"""
import json
import time
from pathlib import Path
from datetime import datetime
import sys

def find_latest_suite():
    """查找最新的实验套件目录"""
    base = Path("results/parameter_sensitivity")
    if not base.exists():
        return None
    
    bandwidth_dirs = sorted(base.glob("bandwidth*"), key=lambda p: p.stat().st_mtime, reverse=True)
    return bandwidth_dirs[0] if bandwidth_dirs else None

def parse_training_log(log_path):
    """解析训练日志获取最新进度"""
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 查找最后的episode信息
        for line in reversed(lines[-100:]):  # 只查看最后100行
            if 'Episode' in line and '/' in line:
                return line.strip()
        return None
    except Exception as e:
        return f"读取失败: {e}"

def check_training_metrics(suite_dir):
    """检查训练指标文件"""
    bandwidth_dir = suite_dir / "bandwidth"
    if not bandwidth_dir.exists():
        return None
    
    # 查找所有配置目录
    config_dirs = sorted(bandwidth_dir.glob("*mhz"))
    
    results = {}
    for config_dir in config_dirs:
        config_name = config_dir.name
        strategy_dirs = list(config_dir.glob("*"))
        
        config_results = {}
        for strat_dir in strategy_dirs:
            if not strat_dir.is_dir():
                continue
            
            strat_name = strat_dir.name
            metrics_file = strat_dir / "training_metrics.json"
            
            if metrics_file.exists():
                try:
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        metrics = json.load(f)
                    
                    episodes = metrics.get('episodes', [])
                    if episodes:
                        latest = episodes[-1]
                        config_results[strat_name] = {
                            'episode': latest.get('episode', 0),
                            'cost': latest.get('raw_cost', 0),
                            'delay': latest.get('avg_delay', 0),
                            'total_episodes': len(episodes)
                        }
                except Exception as e:
                    config_results[strat_name] = {'error': str(e)}
        
        if config_results:
            results[config_name] = config_results
    
    return results if results else None

def monitor_training():
    """主监控循环"""
    print("="*80)
    print("🔍 带宽对比实验训练监控")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("按 Ctrl+C 停止监控\n")
    
    last_update = None
    check_count = 0
    
    try:
        while True:
            check_count += 1
            suite_dir = find_latest_suite()
            
            if not suite_dir:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ 等待实验开始... (检查 {check_count})")
                time.sleep(10)
                continue
            
            print(f"\n{'='*80}")
            print(f"📊 监控报告 #{check_count} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'='*80}")
            print(f"实验目录: {suite_dir.name}")
            
            # 检查训练指标
            metrics = check_training_metrics(suite_dir)
            
            if metrics:
                print("\n📈 当前训练进度:")
                print("-"*80)
                
                for config_name, strategies in metrics.items():
                    print(f"\n配置: {config_name}")
                    for strat_name, info in strategies.items():
                        if 'error' in info:
                            print(f"  ❌ {strat_name:35s} - 错误: {info['error']}")
                        else:
                            ep = info['episode']
                            cost = info['cost']
                            delay = info['delay']
                            total = info['total_episodes']
                            progress = (total / 1500.0) * 100 if total > 0 else 0
                            print(f"  ✅ {strat_name:35s} - Episode {ep:4d} | Cost: {cost:8.2f} | Delay: {delay:6.3f}s | 进度: {progress:5.1f}%")
                
                # 🚨 异常检测
                print(f"\n{'='*80}")
                print("🔍 异常检测:")
                print("-"*80)
                
                warnings = []
                for config_name, strategies in metrics.items():
                    for strat_name, info in strategies.items():
                        if 'error' in info:
                            continue
                        
                        # 检测1: 成本异常高
                        if info['cost'] > 100:
                            warnings.append(f"⚠️  {config_name}/{strat_name}: 成本过高 ({info['cost']:.2f})")
                        
                        # 检测2: 延迟异常高
                        if info['delay'] > 5.0:
                            warnings.append(f"⚠️  {config_name}/{strat_name}: 延迟过高 ({info['delay']:.3f}s)")
                        
                        # 检测3: 训练轮数不足
                        if 'comprehensive' in strat_name and info['total_episodes'] < 1500:
                            remaining = 1500 - info['total_episodes']
                            warnings.append(f"📊 {config_name}/{strat_name}: 还需 {remaining} 轮")
                        
                        # 检测4: 进度停滞
                        if info['total_episodes'] > 100 and info['cost'] > 50:
                            warnings.append(f"🐌 {config_name}/{strat_name}: 可能未收敛 (Episode {info['total_episodes']}, Cost {info['cost']:.2f})")
                
                if warnings:
                    for w in warnings:
                        print(f"  {w}")
                else:
                    print("  ✅ 暂无异常检测到")
                
                last_update = datetime.now()
            else:
                print("  ⏳ 暂无训练数据（可能刚开始）")
            
            print(f"\n下次检查: 30秒后...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("⏹️  监控已停止")
        print("="*80)
        if last_update:
            print(f"最后更新: {last_update.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

if __name__ == "__main__":
    monitor_training()
