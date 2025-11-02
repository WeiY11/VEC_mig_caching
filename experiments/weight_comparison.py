#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
奖励权重对比实验脚本

功能：
1. 定义多组权重配置方案
2. 自动运行快速评估实验
3. 生成详细对比报告
4. 可视化不同权重的效果
# 步骤1: 快速验证（5分钟）
python experiments/weight_comparison.py --mode full --config balanced --episodes 10

# 步骤2: 如果成功，运行完整版（2-3小时）
python experiments/weight_comparison.py --mode full --config balanced --episodes 500

# 步骤3: 运行Top 3配置（6-9小时）
experiments\run_top3_configs.bat

# 步骤4: 生成对比图表
python experiments/visualize_weight_comparison.py

# 步骤5: 查看分析报告
python experiments/weight_comparison.py --mode analyze

使用方法：
  python experiments/weight_comparison.py --mode quick  # 快速评估（100轮）
  python experiments/weight_comparison.py --mode full   # 完整实验（500轮）
  python experiments/weight_comparison.py --mode generate  # 仅生成配置文件
"""

import os
import sys
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config


class WeightConfiguration:
    """权重配置类"""
    
    def __init__(self, name: str, description: str, weights: Dict[str, float]):
        self.name = name
        self.description = description
        self.weights = weights
    
    def to_dict(self):
        return {
            "name": self.name,
            "description": self.description,
            "weights": self.weights
        }


# ========== 定义权重配置方案 ==========

WEIGHT_CONFIGS = [
    # 1. 当前配置（基线）
    WeightConfiguration(
        name="current",
        description="当前配置 - 能耗主导（能耗归一化值大）",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 2. 时延优先配置
    WeightConfiguration(
        name="delay_priority",
        description="时延优先 - 时延权重加倍",
        weights={
            "reward_weight_delay": 3.0,  # 增加时延权重
            "reward_weight_energy": 1.0,  # 降低能耗权重
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 3. 能耗优先配置
    WeightConfiguration(
        name="energy_priority",
        description="能耗优先 - 能耗权重加倍",
        weights={
            "reward_weight_delay": 1.5,  # 降低时延权重
            "reward_weight_energy": 2.0,  # 增加能耗权重
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 4. 平衡配置（时延能耗等权重）
    WeightConfiguration(
        name="balanced",
        description="平衡配置 - 时延能耗归一化后等权重",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 3500.0,  # 调整能耗目标使归一化值接近时延
            "latency_target": 0.40,
        }
    ),
    
    # 5. 缓存增强配置
    WeightConfiguration(
        name="cache_enhanced",
        description="缓存增强 - 提高缓存权重",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.35,  # 缓存权重提高
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 6. 高可靠性配置
    WeightConfiguration(
        name="high_reliability",
        description="高可靠性 - 强调任务完成率",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.10,  # 大幅增加丢弃惩罚
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 7. 激进配置（高权重，挑战极限）
    WeightConfiguration(
        name="aggressive",
        description="激进配置 - 同时优化所有目标",
        weights={
            "reward_weight_delay": 3.0,
            "reward_weight_energy": 2.0,
            "reward_weight_cache": 0.25,
            "reward_penalty_dropped": 0.08,
            "energy_target": 1200.0,
            "latency_target": 0.35,  # 更严格的时延目标
        }
    ),
    
    # 8. 保守配置（低权重，稳定收敛）
    WeightConfiguration(
        name="conservative",
        description="保守配置 - 平滑权重，易于收敛",
        weights={
            "reward_weight_delay": 1.5,
            "reward_weight_energy": 1.0,
            "reward_weight_cache": 0.10,
            "reward_penalty_dropped": 0.03,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 9. 时延能耗平衡v2（调整归一化目标）
    WeightConfiguration(
        name="balanced_v2",
        description="平衡v2 - 通过目标值平衡时延能耗权重",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 2000.0,  # 中间值
            "latency_target": 0.40,
        }
    ),
    
    # 10. 缓存激进配置
    WeightConfiguration(
        name="cache_aggressive",
        description="缓存激进 - 大幅提高缓存权重",
        weights={
            "reward_weight_delay": 2.0,
            "reward_weight_energy": 1.2,
            "reward_weight_cache": 0.50,  # 非常高的缓存权重
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.40,
        }
    ),
    
    # 11. 最小成本配置
    WeightConfiguration(
        name="min_cost",
        description="最小成本 - 平衡权重+合理目标",
        weights={
            "reward_weight_delay": 1.8,
            "reward_weight_energy": 1.5,
            "reward_weight_cache": 0.12,
            "reward_penalty_dropped": 0.04,
            "energy_target": 2500.0,
            "latency_target": 0.38,
        }
    ),
    
    # 12. 严格时延配置
    WeightConfiguration(
        name="strict_latency",
        description="严格时延 - 更严格的时延目标",
        weights={
            "reward_weight_delay": 3.5,
            "reward_weight_energy": 1.0,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 1200.0,
            "latency_target": 0.35,  # 更严格的时延目标
        }
    ),
    
    # 13. 节能优先v2
    WeightConfiguration(
        name="energy_saver",
        description="节能优先v2 - 极低能耗目标",
        weights={
            "reward_weight_delay": 1.5,
            "reward_weight_energy": 2.5,
            "reward_weight_cache": 0.15,
            "reward_penalty_dropped": 0.05,
            "energy_target": 800.0,  # 极低能耗目标
            "latency_target": 0.40,
        }
    ),
    
    # 14. 综合最优（基于前期分析）
    WeightConfiguration(
        name="comprehensive",
        description="综合最优 - 基于经验的综合配置",
        weights={
            "reward_weight_delay": 2.2,
            "reward_weight_energy": 1.5,
            "reward_weight_cache": 0.20,
            "reward_penalty_dropped": 0.06,
            "energy_target": 1800.0,
            "latency_target": 0.38,
        }
    ),
]


def generate_config_files(output_dir: str = "experiments/weight_configs"):
    """生成所有权重配置文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"生成权重配置文件...")
    print(f"{'='*70}\n")
    
    config_files = []
    for cfg in WEIGHT_CONFIGS:
        filename = f"{cfg.name}_weights.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(cfg.to_dict(), f, indent=2, ensure_ascii=False)
        
        config_files.append(filepath)
        print(f"[OK] {cfg.name:20s} - {cfg.description}")
    
    print(f"\n配置文件已保存到: {output_dir}/")
    return config_files


def run_single_experiment(config_name: str, config_weights: Dict, 
                         episodes: int = 100, output_dir: str = None):
    """运行单个权重配置实验"""
    
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"results/weight_comparison/{config_name}_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存配置
    config_file = os.path.join(output_dir, "weights_config.json")
    with open(config_file, 'w') as f:
        json.dump(config_weights, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"运行实验: {config_name}")
    print(f"配置: {config_weights}")
    print(f"训练轮数: {episodes}")
    print(f"{'='*70}\n")
    
    # 构建命令行参数
    cmd = [
        sys.executable,
        "train_single_agent.py",
        "--algorithm", "TD3",
        "--episodes", str(episodes),
        "--num-vehicles", "12",
        "--silent-mode",  # 🔧 静默模式，避免交互式输入卡住
    ]
    
    # 设置环境变量传递权重配置
    env = os.environ.copy()
    env['WEIGHT_CONFIG'] = json.dumps(config_weights)
    env['EXPERIMENT_NAME'] = config_name
    
    try:
        # 运行训练
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=False,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        
        if result.returncode == 0:
            print(f"\n[OK] Experiment {config_name} completed!")
            return True
        else:
            print(f"\n[FAIL] Experiment {config_name} failed!")
            return False
            
    except Exception as e:
        print(f"\n[ERROR] Experiment {config_name} error: {e}")
        return False


def create_batch_script(episodes: int = 500, output_file: str = None):
    """创建批处理脚本，用于依次运行所有配置"""
    
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"experiments/run_weight_comparison_{timestamp}.bat"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("@echo off\n")
        f.write("REM 权重对比实验批处理脚本\n")
        f.write(f"REM 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("REM 每个配置训练 {} 轮\n\n".format(episodes))
        
        for i, cfg in enumerate(WEIGHT_CONFIGS, 1):
            f.write(f"echo.\n")
            f.write(f"echo ============================================================\n")
            f.write(f"echo 实验 {i}/{len(WEIGHT_CONFIGS)}: {cfg.name}\n")
            f.write(f"echo {cfg.description}\n")
            f.write(f"echo ============================================================\n")
            f.write(f"echo.\n\n")
            
            # 设置环境变量
            for key, value in cfg.weights.items():
                f.write(f"set WEIGHT_{key.upper()}={value}\n")
            
            f.write(f"set EXPERIMENT_NAME={cfg.name}\n\n")
            
            # 运行训练（添加 --silent-mode 避免交互式输入）
            f.write(f"python train_single_agent.py --algorithm TD3 --episodes {episodes} --num-vehicles 12 --silent-mode\n\n")
            
            f.write(f"if errorlevel 1 (\n")
            f.write(f"    echo 实验 {cfg.name} 失败！\n")
            f.write(f"    pause\n")
            f.write(f"    exit /b 1\n")
            f.write(f")\n\n")
        
        f.write("echo.\n")
        f.write("echo 所有实验完成！\n")
        f.write("echo.\n")
        f.write("pause\n")
    
    print(f"\n批处理脚本已生成: {output_file}")
    print(f"运行方式: {output_file}")
    return output_file


def analyze_results(results_dir: str = "results/weight_comparison"):
    """分析所有实验结果并生成对比报告"""
    
    if not os.path.exists(results_dir):
        print(f"结果目录不存在: {results_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f"分析权重对比实验结果...")
    print(f"{'='*70}\n")
    
    results = []
    
    # 扫描所有实验结果
    for config_name in os.listdir(results_dir):
        config_path = os.path.join(results_dir, config_name)
        if not os.path.isdir(config_path):
            continue
        
        # 查找训练结果文件
        result_files = [f for f in os.listdir(config_path) if f.startswith('training_results') and f.endswith('.json')]
        
        if not result_files:
            print(f"[WARN] No result file found: {config_name}")
            continue
        
        # 读取最新结果
        result_file = sorted(result_files)[-1]
        result_path = os.path.join(config_path, result_file)
        
        try:
            with open(result_path, 'r') as f:
                data = json.load(f)
            
            metrics = data.get('episode_metrics', {})
            
            # 提取后100轮平均指标
            last_100 = min(100, len(metrics.get('total_energy', [])))
            
            if last_100 == 0:
                continue
            
            result = {
                'config_name': config_name,
                'avg_energy': np.mean(metrics['total_energy'][-last_100:]),
                'avg_cache_hit': np.mean(metrics['cache_hit_rate'][-last_100:]),
                'avg_completion': np.mean(metrics['task_completion_rate'][-last_100:]),
                'avg_delay': np.mean(metrics['avg_delay'][-last_100:]),
                'avg_loss': np.mean(metrics['data_loss_ratio_bytes'][-last_100:]),
                'std_energy': np.std(metrics['total_energy'][-last_100:]),
                'std_cache_hit': np.std(metrics['cache_hit_rate'][-last_100:]),
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"[ERROR] Failed to read {config_name}: {e}")
    
    if not results:
        print("没有找到有效的实验结果")
        return
    
    # 生成对比报告
    print("\n" + "="*70)
    print("权重配置对比结果")
    print("="*70)
    print(f"{'配置名称':20s} | {'能耗(J)':>10s} | {'缓存率':>8s} | {'完成率':>8s} | {'时延(s)':>8s} | {'丢失率':>8s}")
    print("-"*70)
    
    for r in sorted(results, key=lambda x: x['avg_completion'], reverse=True):
        print(f"{r['config_name']:20s} | {r['avg_energy']:10.1f} | {r['avg_cache_hit']:7.1%} | {r['avg_completion']:7.1%} | {r['avg_delay']:8.4f} | {r['avg_loss']:7.1%}")
    
    # 找出最佳配置
    print("\n" + "="*70)
    print("最佳配置推荐")
    print("="*70)
    
    best_completion = max(results, key=lambda x: x['avg_completion'])
    best_cache = max(results, key=lambda x: x['avg_cache_hit'])
    best_energy = min(results, key=lambda x: x['avg_energy'])
    best_delay = min(results, key=lambda x: x['avg_delay'])
    
    print(f"最高完成率: {best_completion['config_name']} ({best_completion['avg_completion']:.2%})")
    print(f"最高缓存命中率: {best_cache['config_name']} ({best_cache['avg_cache_hit']:.2%})")
    print(f"最低能耗: {best_energy['config_name']} ({best_energy['avg_energy']:.1f}J)")
    print(f"最低时延: {best_delay['config_name']} ({best_delay['avg_delay']:.4f}s)")
    
    # 计算综合得分（归一化后加权平均）
    print("\n" + "="*70)
    print("综合评分（归一化加权平均）")
    print("="*70)
    
    # 归一化各指标
    max_completion = max(r['avg_completion'] for r in results)
    max_cache = max(r['avg_cache_hit'] for r in results)
    min_energy = min(r['avg_energy'] for r in results)
    max_energy = max(r['avg_energy'] for r in results)
    min_delay = min(r['avg_delay'] for r in results)
    max_delay = max(r['avg_delay'] for r in results)
    
    for r in results:
        # 归一化得分（越高越好）
        score_completion = r['avg_completion'] / max_completion if max_completion > 0 else 0
        score_cache = r['avg_cache_hit'] / max_cache if max_cache > 0 else 0
        score_energy = 1 - (r['avg_energy'] - min_energy) / (max_energy - min_energy) if max_energy > min_energy else 1
        score_delay = 1 - (r['avg_delay'] - min_delay) / (max_delay - min_delay) if max_delay > min_delay else 1
        
        # 综合得分（权重：完成率30%，缓存20%，能耗25%，时延25%）
        r['综合得分'] = 0.30 * score_completion + 0.20 * score_cache + 0.25 * score_energy + 0.25 * score_delay
    
    for r in sorted(results, key=lambda x: x['综合得分'], reverse=True):
        print(f"{r['config_name']:20s} | 综合得分: {r['综合得分']:.3f}")
    
    # 保存对比结果
    comparison_file = os.path.join(results_dir, f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n对比报告已保存: {comparison_file}")


def main():
    parser = argparse.ArgumentParser(description="权重对比实验工具")
    parser.add_argument("--mode", type=str, default="generate",
                       choices=["quick", "full", "generate", "analyze"],
                       help="运行模式: quick(快速100轮), full(完整500轮), generate(仅生成配置), analyze(分析结果)")
    parser.add_argument("--config", type=str, default=None,
                       help="指定单个配置运行")
    parser.add_argument("--episodes", type=int, default=None,
                       help="训练轮数（覆盖模式默认值）")
    
    args = parser.parse_args()
    
    if args.mode == "generate":
        # 生成配置文件
        generate_config_files()
        
        # 生成批处理脚本
        episodes = args.episodes if args.episodes else 500
        create_batch_script(episodes=episodes)
        
        print("\n" + "="*70)
        print("下一步:")
        print("="*70)
        print("1. 检查生成的配置文件: experiments/weight_configs/")
        print("2. 运行批处理脚本: experiments/run_weight_comparison_*.bat")
        print("3. 或手动运行单个配置（修改config/system_config.py中的权重）")
        print("4. 实验完成后运行: python experiments/weight_comparison.py --mode analyze")
        
    elif args.mode == "analyze":
        # 分析结果
        analyze_results()
        
    elif args.mode in ["quick", "full"]:
        # 运行实验
        episodes = args.episodes if args.episodes else (100 if args.mode == "quick" else 500)
        
        if args.config:
            # 运行指定配置
            cfg = next((c for c in WEIGHT_CONFIGS if c.name == args.config), None)
            if cfg:
                run_single_experiment(cfg.name, cfg.weights, episodes)
            else:
                print(f"错误: 未找到配置 '{args.config}'")
                print(f"可用配置: {', '.join(c.name for c in WEIGHT_CONFIGS)}")
        else:
            # 运行所有配置
            print(f"\n{'='*70}")
            print(f"开始权重对比实验 - {args.mode.upper()} 模式")
            print(f"共 {len(WEIGHT_CONFIGS)} 个配置，每个配置训练 {episodes} 轮")
            print(f"{'='*70}\n")
            
            for i, cfg in enumerate(WEIGHT_CONFIGS, 1):
                print(f"\n[{i}/{len(WEIGHT_CONFIGS)}] 运行配置: {cfg.name}")
                run_single_experiment(cfg.name, cfg.weights, episodes)
            
            print(f"\n{'='*70}")
            print("所有实验完成！")
            print(f"{'='*70}\n")
            
            # 自动分析结果
            analyze_results()
            
            # 🎨 自动生成对比图表
            print(f"\n{'='*70}")
            print("开始生成对比图表...")
            print(f"{'='*70}\n")
            
            try:
                viz_script = os.path.join(os.path.dirname(__file__), "visualize_weight_comparison.py")
                subprocess.run([sys.executable, viz_script], check=True)
                print("\n✅ 对比图表生成完成！")
            except Exception as e:
                print(f"\n⚠️ 图表生成失败: {e}")
                print("可手动运行: python experiments/visualize_weight_comparison.py")


if __name__ == "__main__":
    main()

