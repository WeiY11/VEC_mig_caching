#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3算法消融实验主脚本
完全独立的实验环境，不影响原始项目文件

【功能】
1. 运行7种消融配置
2. 收集性能指标
3. 生成对比分析
4. 输出论文图表

【使用】
快速测试: python run_ablation_td3.py --episodes 30 --quick
标准实验: python run_ablation_td3.py --episodes 200
完整实验: python run_ablation_td3.py --episodes 500 --full
单独配置: python run_ablation_td3.py --config No-Cache --episodes 100
cd ablation_experiments
python run_ablation_td3.py --episodes 200
"""

import os
import sys
import json
import time
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List
from pathlib import Path

# 修复Windows编码问题
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加父目录到路径，以导入主项目模块
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# 导入项目模块
from config import config
from single_agent.td3 import TD3Environment
from evaluation.system_simulator import CompleteSystemSimulator
from ablation_experiments.ablation_configs import get_all_ablation_configs, get_config_by_name


class TD3AblationExperiment:
    """
    TD3消融实验执行器
    
    【职责】
    1. 管理实验流程
    2. 收集实验数据
    3. 生成分析报告
    """
    
    def __init__(self, save_dir: str = None):
        """初始化实验环境"""
        if save_dir is None:
            self.save_dir = Path(__file__).parent / "results"
        else:
            self.save_dir = Path(save_dir)
        
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 创建分析目录
        self.analysis_dir = Path(__file__).parent / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        # 实验结果存储
        self.results = {}
        
        print("="*80)
        print("🔬 TD3消融实验环境初始化")
        print("="*80)
        print(f"✓ 结果保存目录: {self.save_dir}")
        print(f"✓ 分析保存目录: {self.analysis_dir}")
        print("="*80)
    
    def run_single_config(self, 
                         ablation_config,
                         num_episodes: int = 200,
                         random_seed: int = 42) -> Dict:
        """
        运行单个消融配置的实验
        
        【参数】
        - ablation_config: 消融配置对象
        - num_episodes: 训练轮次
        - random_seed: 随机种子
        
        【返回】实验结果字典
        """
        print(f"\n{'='*80}")
        print(f"🎯 开始实验: {ablation_config.name}")
        print(f"{'='*80}")
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        # 应用消融配置
        ablation_config.apply_to_system()
        
        # 创建训练环境（使用SingleAgentTrainingEnvironment）
        from train_single_agent import SingleAgentTrainingEnvironment
        training_env = SingleAgentTrainingEnvironment("TD3")
        
        # 训练统计
        episode_rewards = []
        episode_delays = []
        episode_energies = []
        episode_completion_rates = []
        episode_cache_hits = []
        episode_migration_success = []
        
        start_time = time.time()
        
        # ========== 训练循环 ==========
        for episode in range(1, num_episodes + 1):
            # 运行一个完整的Episode
            episode_result = training_env.run_episode(episode)
            
            # 收集指标
            episode_rewards.append(episode_result['avg_reward'])
            
            system_metrics = episode_result.get('system_metrics', {})
            episode_delays.append(system_metrics.get('avg_task_delay', 0))
            episode_energies.append(system_metrics.get('total_energy_consumption', 0))
            episode_completion_rates.append(system_metrics.get('task_completion_rate', 0))
            
            # 缓存和迁移统计
            cache_hit_rate = system_metrics.get('cache_hit_rate', 0)
            migration_success_rate = system_metrics.get('migration_success_rate', 0)
            
            episode_cache_hits.append(cache_hit_rate)
            episode_migration_success.append(migration_success_rate)
            
            # 打印进度
            if episode % 20 == 0 or episode == num_episodes:
                print(f"  Episode {episode}/{num_episodes}: "
                      f"Reward={episode_rewards[-1]:.3f}, "
                      f"Delay={episode_delays[-1]:.3f}s, "
                      f"Energy={episode_energies[-1]:.1f}J, "
                      f"Completion={episode_completion_rates[-1]:.1%}")
        
        experiment_time = time.time() - start_time
        
        # ========== 计算平均指标（后50%数据，避免初期不稳定）==========
        stable_start = num_episodes // 2
        
        result = {
            'config_name': ablation_config.name,
            'description': ablation_config.description,
            'num_episodes': num_episodes,
            'random_seed': random_seed,
            'experiment_time': experiment_time,
            
            # 核心指标（稳定期平均）
            'avg_delay': float(np.mean(episode_delays[stable_start:])),
            'std_delay': float(np.std(episode_delays[stable_start:])),
            'avg_energy': float(np.mean(episode_energies[stable_start:])),
            'std_energy': float(np.std(episode_energies[stable_start:])),
            'avg_completion_rate': float(np.mean(episode_completion_rates[stable_start:])),
            'avg_cache_hit_rate': float(np.mean(episode_cache_hits[stable_start:])),
            'avg_migration_success_rate': float(np.mean(episode_migration_success[stable_start:])),
            
            # 完整历史数据
            'episode_rewards': [float(r) for r in episode_rewards],
            'episode_delays': [float(d) for d in episode_delays],
            'episode_energies': [float(e) for e in episode_energies],
            'episode_completion_rates': [float(c) for c in episode_completion_rates],
            
            # 配置信息
            'config': ablation_config.to_dict()
        }
        
        print(f"\n{'='*80}")
        print(f"✓ {ablation_config.name} 实验完成")
        print(f"{'='*80}")
        print(f"  平均时延: {result['avg_delay']:.3f}±{result['std_delay']:.3f}s")
        print(f"  平均能耗: {result['avg_energy']:.1f}±{result['std_energy']:.1f}J")
        print(f"  任务完成率: {result['avg_completion_rate']:.2%}")
        print(f"  缓存命中率: {result['avg_cache_hit_rate']:.2%}")
        print(f"  迁移成功率: {result['avg_migration_success_rate']:.2%}")
        print(f"  实验耗时: {experiment_time:.1f}秒")
        print(f"{'='*80}\n")
        
        # 保存单个配置的结果
        config_save_dir = self.save_dir / ablation_config.name
        config_save_dir.mkdir(exist_ok=True, parents=True)
        
        result_file = config_save_dir / f"result_{ablation_config.name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 结果已保存: {result_file}\n")
        
        return result
    
    def run_all_configs(self, num_episodes: int = 200, random_seed: int = 42):
        """
        运行所有消融配置
        
        【参数】
        - num_episodes: 每个配置的训练轮次
        - random_seed: 随机种子
        """
        configs = get_all_ablation_configs()
        
        print("\n" + "="*80)
        print("🚀 开始运行所有消融实验")
        print("="*80)
        print(f"  配置数量: {len(configs)}")
        print(f"  每配置轮次: {num_episodes}")
        print(f"  预计总耗时: ~{len(configs) * num_episodes * 2 / 60:.1f}分钟")
        print("="*80)
        
        total_start = time.time()
        
        for i, ablation_config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] 正在运行: {ablation_config.name}")
            
            result = self.run_single_config(
                ablation_config,
                num_episodes=num_episodes,
                random_seed=random_seed
            )
            
            self.results[ablation_config.name] = result
            
            # 打印进度
            elapsed = time.time() - total_start
            estimated_total = elapsed / i * len(configs)
            remaining = estimated_total - elapsed
            print(f"⏱️  进度: {i}/{len(configs)}, "
                  f"已用时: {elapsed/60:.1f}分钟, "
                  f"剩余: {remaining/60:.1f}分钟")
        
        total_time = time.time() - total_start
        
        print("\n" + "="*80)
        print("🎉 所有消融实验完成!")
        print("="*80)
        print(f"  总耗时: {total_time/60:.1f}分钟")
        print(f"  结果数: {len(self.results)}")
        print("="*80)
    
    def analyze_results(self):
        """分析实验结果"""
        if not self.results:
            print("⚠️ 没有实验结果可分析")
            return
        
        print("\n" + "="*80)
        print("📊 消融实验结果分析")
        print("="*80)
        
        # 获取基准（Full-System）
        baseline = self.results.get('Full-System')
        if not baseline:
            print("⚠️ 未找到Full-System基准结果")
            return
        
        print(f"\n【基准配置】Full-System")
        print(f"  平均时延: {baseline['avg_delay']:.3f}s")
        print(f"  平均能耗: {baseline['avg_energy']:.1f}J")
        print(f"  任务完成率: {baseline['avg_completion_rate']:.2%}")
        
        # 对比分析
        print(f"\n{'配置名称':<20} {'时延变化':<15} {'能耗变化':<15} {'完成率变化':<15} {'综合影响'}")
        print("-"*80)
        
        analysis = {'baseline': baseline, 'comparisons': {}}
        
        for config_name, result in self.results.items():
            if config_name == 'Full-System':
                continue
            
            # 计算相对变化
            delay_change = (result['avg_delay'] - baseline['avg_delay']) / baseline['avg_delay'] * 100
            energy_change = (result['avg_energy'] - baseline['avg_energy']) / baseline['avg_energy'] * 100
            completion_change = (result['avg_completion_rate'] - baseline['avg_completion_rate']) * 100
            
            # 综合影响评分
            impact_score = abs(delay_change) * 0.4 + abs(energy_change) * 0.3 + abs(completion_change) * 0.3
            
            analysis['comparisons'][config_name] = {
                'delay_change_pct': delay_change,
                'energy_change_pct': energy_change,
                'completion_change_pct': completion_change,
                'impact_score': impact_score
            }
            
            print(f"{config_name:<20} {delay_change:>+12.1f}% {energy_change:>+12.1f}% "
                  f"{completion_change:>+12.1f}% {impact_score:>12.1f}")
        
        # 模块重要性排序
        print("\n【模块重要性排序】(影响力从高到低)")
        sorted_impacts = sorted(analysis['comparisons'].items(),
                               key=lambda x: x[1]['impact_score'],
                               reverse=True)
        
        for i, (config_name, data) in enumerate(sorted_impacts, 1):
            module_name = config_name.replace('No-', '').replace('Minimal-', '')
            print(f"  {i}. {module_name:<15} (影响力: {data['impact_score']:.1f})")
        
        # 保存分析结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = self.analysis_dir / f"ablation_analysis_{timestamp}.json"
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 分析结果已保存: {analysis_file}")
        
        return analysis
    
    def generate_plots(self):
        """生成对比图表"""
        if not self.results:
            print("⚠️ 没有结果可绘图")
            return
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
            matplotlib.rcParams['axes.unicode_minus'] = False
        except ImportError:
            print("⚠️ 缺少matplotlib，跳过绘图")
            return
        
        print("\n📈 生成对比图表...")
        
        # 提取数据
        configs = []
        delays = []
        energies = []
        completions = []
        
        # 确保Full-System在第一个
        if 'Full-System' in self.results:
            configs.append('Full-System')
            delays.append(self.results['Full-System']['avg_delay'])
            energies.append(self.results['Full-System']['avg_energy'])
            completions.append(self.results['Full-System']['avg_completion_rate'] * 100)
        
        # 添加其他配置
        for config_name, result in self.results.items():
            if config_name != 'Full-System':
                configs.append(config_name.replace('-', '\n'))
                delays.append(result['avg_delay'])
                energies.append(result['avg_energy'])
                completions.append(result['avg_completion_rate'] * 100)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 时延对比
        bars1 = axes[0].bar(range(len(configs)), delays, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].set_title('平均任务时延对比', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('时延 (秒)', fontsize=12)
        axes[0].set_xticks(range(len(configs)))
        axes[0].set_xticklabels(configs, rotation=45, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars1, delays)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # 能耗对比
        bars2 = axes[1].bar(range(len(configs)), energies, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[1].set_title('系统总能耗对比', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('能耗 (焦耳)', fontsize=12)
        axes[1].set_xticks(range(len(configs)))
        axes[1].set_xticklabels(configs, rotation=45, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars2, energies)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # 完成率对比
        bars3 = axes[2].bar(range(len(configs)), completions, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[2].set_title('任务完成率对比', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('完成率 (%)', fontsize=12)
        axes[2].set_xticks(range(len(configs)))
        axes[2].set_xticklabels(configs, rotation=45, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].set_ylim([min(completions)-5, 100])
        # 标注数值
        for i, (bar, val) in enumerate(zip(bars3, completions)):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                        f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plot_path = self.analysis_dir / 'ablation_comparison.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 消融对比图: {plot_path}")
        plt.close()
        
        # 生成训练曲线图
        self._generate_training_curves()
    
    def _smooth_curve(self, data, window_size=20):
        """
        滑动平均平滑曲线
        
        【参数】
        - data: 原始数据
        - window_size: 滑动窗口大小
        
        【返回】平滑后的数据
        """
        if len(data) < window_size:
            return data
        
        smoothed = []
        for i in range(len(data)):
            start = max(0, i - window_size + 1)
            end = i + 1
            smoothed.append(np.mean(data[start:end]))
        
        return smoothed
    
    def _generate_training_curves(self):
        """生成训练曲线图（添加滑动平滑）"""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 滑动窗口大小
        window_size = 20
        
        for config_name, result in self.results.items():
            # 时延曲线（原始+平滑）
            delays_raw = result['episode_delays']
            delays_smooth = self._smooth_curve(delays_raw, window_size)
            axes[0, 0].plot(delays_smooth, label=config_name, alpha=0.8, linewidth=1.5)
            
            # 能耗曲线（原始+平滑）
            energies_raw = result['episode_energies']
            energies_smooth = self._smooth_curve(energies_raw, window_size)
            axes[0, 1].plot(energies_smooth, label=config_name, alpha=0.8, linewidth=1.5)
            
            # 完成率曲线（原始+平滑）
            completions_raw = result['episode_completion_rates']
            completions_smooth = self._smooth_curve(completions_raw, window_size)
            axes[1, 0].plot(completions_smooth, label=config_name, alpha=0.8, linewidth=1.5)
            
            # 奖励曲线（原始+平滑）⭐ 重点优化
            rewards_raw = result['episode_rewards']
            rewards_smooth = self._smooth_curve(rewards_raw, window_size)
            axes[1, 1].plot(rewards_smooth, label=config_name, alpha=0.8, linewidth=1.5)
        
        axes[0, 0].set_title('时延训练曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[0, 0].set_xlabel('Episode', fontsize=10)
        axes[0, 0].set_ylabel('平均时延 (s)', fontsize=10)
        axes[0, 0].legend(fontsize=8, loc='upper right')
        axes[0, 0].grid(alpha=0.3)
        
        axes[0, 1].set_title('能耗训练曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[0, 1].set_xlabel('Episode', fontsize=10)
        axes[0, 1].set_ylabel('总能耗 (J)', fontsize=10)
        axes[0, 1].legend(fontsize=8, loc='upper right')
        axes[0, 1].grid(alpha=0.3)
        
        axes[1, 0].set_title('完成率训练曲线 (滑动平均)', fontweight='bold', fontsize=12)
        axes[1, 0].set_xlabel('Episode', fontsize=10)
        axes[1, 0].set_ylabel('完成率', fontsize=10)
        axes[1, 0].legend(fontsize=8, loc='lower right')
        axes[1, 0].grid(alpha=0.3)
        
        axes[1, 1].set_title('奖励训练曲线 (滑动平均, 窗口=20)', fontweight='bold', fontsize=12)
        axes[1, 1].set_xlabel('Episode', fontsize=10)
        axes[1, 1].set_ylabel('平均奖励', fontsize=10)
        axes[1, 1].legend(fontsize=8, loc='lower right')
        axes[1, 1].grid(alpha=0.3)
        
        plt.tight_layout()
        curve_path = self.analysis_dir / 'training_curves.png'
        plt.savefig(curve_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 训练曲线图: {curve_path}")
        plt.close()
    
    def save_all_results(self):
        """保存所有实验结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存汇总JSON
        summary_file = self.save_dir / f"ablation_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 实验结果汇总已保存: {summary_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TD3消融实验')
    parser.add_argument('--episodes', type=int, default=200, help='训练轮次 (默认200)')
    parser.add_argument('--config', type=str, default=None, help='单独运行某个配置')
    parser.add_argument('--seed', type=int, default=42, help='随机种子 (默认42)')
    parser.add_argument('--quick', action='store_true', help='快速测试模式 (30轮)')
    parser.add_argument('--full', action='store_true', help='完整实验模式 (500轮)')
    
    args = parser.parse_args()
    
    # 确定训练轮次
    if args.quick:
        num_episodes = 30
    elif args.full:
        num_episodes = 500
    else:
        num_episodes = args.episodes
    
    # 创建实验环境
    experiment = TD3AblationExperiment()
    
    # 运行实验
    if args.config:
        # 单独运行某个配置
        config_obj = get_config_by_name(args.config)
        result = experiment.run_single_config(config_obj, num_episodes, args.seed)
        experiment.results[config_obj.name] = result
    else:
        # 运行所有配置
        experiment.run_all_configs(num_episodes, args.seed)
    
    # 分析结果
    experiment.analyze_results()
    
    # 生成图表
    experiment.generate_plots()
    
    # 保存结果
    experiment.save_all_results()
    
    print("\n" + "="*80)
    print("🎉 实验全部完成!")
    print("="*80)
    print(f"  结果目录: {experiment.save_dir}")
    print(f"  分析目录: {experiment.analysis_dir}")
    print("="*80)


if __name__ == "__main__":
    main()

