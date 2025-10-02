#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验框架
用于验证系统各模块的有效性

消融实验设计：
1. No-Cache: 禁用缓存模块
2. No-Migration: 禁用迁移模块
3. No-Priority: 禁用任务优先级
4. No-Adaptive: 禁用自适应控制
5. Full-System: 完整系统（对照组）
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class AblationConfig:
    """消融实验配置"""
    name: str
    description: str
    enable_cache: bool = True
    enable_migration: bool = True
    enable_priority: bool = True
    enable_adaptive: bool = True
    enable_collaboration: bool = True


@dataclass
class AblationResult:
    """消融实验结果"""
    config_name: str
    avg_delay: float
    total_energy: float
    data_loss_ratio: float
    task_completion_rate: float
    cache_hit_rate: float
    migration_success_rate: float
    experiment_time: float
    
    def to_dict(self):
        return asdict(self)


class AblationStudy:
    """消融实验执行器"""
    
    def __init__(self):
        self.results = {}
        self.configs = self._create_ablation_configs()
        
    def _create_ablation_configs(self) -> List[AblationConfig]:
        """创建消融实验配置"""
        configs = []
        
        # 1. 完整系统（对照组）
        configs.append(AblationConfig(
            name="Full-System",
            description="完整系统（所有模块启用）",
            enable_cache=True,
            enable_migration=True,
            enable_priority=True,
            enable_adaptive=True,
            enable_collaboration=True
        ))
        
        # 2. 无缓存
        configs.append(AblationConfig(
            name="No-Cache",
            description="禁用边缘缓存模块",
            enable_cache=False,
            enable_migration=True,
            enable_priority=True,
            enable_adaptive=True,
            enable_collaboration=True
        ))
        
        # 3. 无迁移
        configs.append(AblationConfig(
            name="No-Migration",
            description="禁用任务迁移模块",
            enable_cache=True,
            enable_migration=False,
            enable_priority=True,
            enable_adaptive=True,
            enable_collaboration=True
        ))
        
        # 4. 无优先级
        configs.append(AblationConfig(
            name="No-Priority",
            description="禁用任务优先级队列",
            enable_cache=True,
            enable_migration=True,
            enable_priority=False,
            enable_adaptive=True,
            enable_collaboration=True
        ))
        
        # 5. 无自适应控制
        configs.append(AblationConfig(
            name="No-Adaptive",
            description="禁用自适应缓存和迁移控制",
            enable_cache=True,
            enable_migration=True,
            enable_priority=True,
            enable_adaptive=False,
            enable_collaboration=True
        ))
        
        # 6. 无协作
        configs.append(AblationConfig(
            name="No-Collaboration",
            description="禁用RSU间协作缓存",
            enable_cache=True,
            enable_migration=True,
            enable_priority=True,
            enable_adaptive=True,
            enable_collaboration=False
        ))
        
        # 7. 最小系统
        configs.append(AblationConfig(
            name="Minimal-System",
            description="最小系统（仅基础功能）",
            enable_cache=False,
            enable_migration=False,
            enable_priority=False,
            enable_adaptive=False,
            enable_collaboration=False
        ))
        
        return configs
    
    def run_ablation_experiment(self, algorithm: str = "TD3", num_episodes: int = 100):
        """运行消融实验"""
        from config import config
        from train_single_agent import SingleAgentTrainingEnvironment
        
        print("=" * 80)
        print("🔬 消融实验开始")
        print("=" * 80)
        print(f"算法: {algorithm}")
        print(f"训练轮次: {num_episodes}")
        print(f"实验配置数: {len(self.configs)}")
        print("")
        
        results = {}
        
        for i, ablation_config in enumerate(self.configs, 1):
            print(f"\n[{i}/{len(self.configs)}] 运行配置: {ablation_config.name}")
            print(f"  描述: {ablation_config.description}")
            print(f"  缓存: {'✓' if ablation_config.enable_cache else '✗'}")
            print(f"  迁移: {'✓' if ablation_config.enable_migration else '✗'}")
            print(f"  优先级: {'✓' if ablation_config.enable_priority else '✗'}")
            print(f"  自适应: {'✓' if ablation_config.enable_adaptive else '✗'}")
            print(f"  协作: {'✓' if ablation_config.enable_collaboration else '✗'}")
            
            # 应用消融配置
            self._apply_ablation_config(ablation_config)
            
            # 创建训练环境
            training_env = SingleAgentTrainingEnvironment(algorithm)
            
            # 运行训练
            start_time = time.time()
            episode_rewards = []
            episode_delays = []
            episode_energies = []
            episode_losses = []
            
            for episode in range(1, num_episodes + 1):
                episode_result = training_env.run_episode(episode)
                
                episode_rewards.append(episode_result['avg_reward'])
                
                metrics = episode_result['system_metrics']
                episode_delays.append(metrics.get('avg_task_delay', 0))
                episode_energies.append(metrics.get('total_energy_consumption', 0))
                episode_losses.append(metrics.get('data_loss_ratio_bytes', 0))
                
                if episode % 20 == 0:
                    print(f"    Episode {episode}/{num_episodes}: "
                          f"Reward={episode_result['avg_reward']:.3f}, "
                          f"Delay={metrics.get('avg_task_delay', 0):.3f}s")
            
            experiment_time = time.time() - start_time
            
            # 计算平均指标（后50%数据，避免初期不稳定）
            stable_start = num_episodes // 2
            
            result = AblationResult(
                config_name=ablation_config.name,
                avg_delay=np.mean(episode_delays[stable_start:]),
                total_energy=np.mean(episode_energies[stable_start:]),
                data_loss_ratio=np.mean(episode_losses[stable_start:]),
                task_completion_rate=1.0 - np.mean(episode_losses[stable_start:]),
                cache_hit_rate=training_env.simulator.stats.get('cache_hits', 0) / 
                              max(1, training_env.simulator.stats.get('cache_requests', 1)),
                migration_success_rate=training_env.simulator.stats.get('migrations_successful', 0) / 
                                     max(1, training_env.simulator.stats.get('migrations_executed', 1)),
                experiment_time=experiment_time
            )
            
            results[ablation_config.name] = result
            
            print(f"  ✓ 完成 - 平均时延: {result.avg_delay:.3f}s, "
                  f"平均能耗: {result.total_energy:.1f}J, "
                  f"完成率: {result.task_completion_rate:.1%}")
        
        self.results = results
        return results
    
    def _apply_ablation_config(self, ablation_config: AblationConfig):
        """应用消融配置到系统"""
        from config import config
        
        # 这里需要修改全局配置以禁用相应模块
        # 实际实现中需要在系统中添加相应的开关
        
        # 示例：设置配置标志
        if hasattr(config, 'ablation'):
            config.ablation.enable_cache = ablation_config.enable_cache
            config.ablation.enable_migration = ablation_config.enable_migration
            config.ablation.enable_priority = ablation_config.enable_priority
            config.ablation.enable_adaptive = ablation_config.enable_adaptive
            config.ablation.enable_collaboration = ablation_config.enable_collaboration
    
    def analyze_results(self) -> Dict:
        """分析消融实验结果"""
        if not self.results:
            print("⚠️ 没有实验结果可分析")
            return {}
        
        print("\n" + "=" * 80)
        print("📊 消融实验结果分析")
        print("=" * 80)
        
        # 获取Full-System作为基准
        baseline = self.results.get('Full-System')
        if not baseline:
            print("⚠️ 未找到Full-System基准结果")
            return {}
        
        analysis = {
            'baseline': baseline.to_dict(),
            'comparisons': {}
        }
        
        print(f"\n基准配置 (Full-System):")
        print(f"  平均时延: {baseline.avg_delay:.3f}s")
        print(f"  平均能耗: {baseline.total_energy:.1f}J")
        print(f"  数据丢失率: {baseline.data_loss_ratio:.2%}")
        print(f"  任务完成率: {baseline.task_completion_rate:.2%}")
        print(f"  缓存命中率: {baseline.cache_hit_rate:.2%}")
        print(f"  迁移成功率: {baseline.migration_success_rate:.2%}")
        
        print("\n各配置相对Full-System的性能变化:")
        print("-" * 80)
        print(f"{'配置名称':<20} {'时延变化':<12} {'能耗变化':<12} {'完成率变化':<12} {'综合影响'}")
        print("-" * 80)
        
        for config_name, result in self.results.items():
            if config_name == 'Full-System':
                continue
            
            # 计算相对变化（正值表示性能下降）
            delay_change = (result.avg_delay - baseline.avg_delay) / baseline.avg_delay * 100
            energy_change = (result.total_energy - baseline.total_energy) / baseline.total_energy * 100
            completion_change = (baseline.task_completion_rate - result.task_completion_rate) * 100
            
            # 综合影响评分（越高表示该模块越重要）
            impact_score = abs(delay_change) * 0.4 + abs(energy_change) * 0.3 + abs(completion_change) * 0.3
            
            analysis['comparisons'][config_name] = {
                'delay_change_pct': delay_change,
                'energy_change_pct': energy_change,
                'completion_change_pct': completion_change,
                'impact_score': impact_score
            }
            
            print(f"{config_name:<20} {delay_change:>+10.1f}% {energy_change:>+10.1f}% "
                  f"{completion_change:>+10.1f}% {impact_score:>10.1f}")
        
        # 按影响力排序
        print("\n模块重要性排序 (影响力从高到低):")
        sorted_impacts = sorted(analysis['comparisons'].items(), 
                               key=lambda x: x[1]['impact_score'], 
                               reverse=True)
        
        for i, (config_name, data) in enumerate(sorted_impacts, 1):
            module_name = config_name.replace('No-', '').replace('Minimal-', '')
            print(f"  {i}. {module_name:<15} (影响力: {data['impact_score']:.1f})")
        
        return analysis
    
    def generate_ablation_plots(self, save_dir: str = "results/ablation"):
        """生成消融实验图表"""
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        os.makedirs(save_dir, exist_ok=True)
        
        if not self.results:
            print("⚠️ 没有结果可绘图")
            return
        
        # 提取数据
        configs = []
        delays = []
        energies = []
        completions = []
        
        for config_name, result in self.results.items():
            configs.append(config_name.replace('-', '\n'))
            delays.append(result.avg_delay)
            energies.append(result.total_energy)
            completions.append(result.task_completion_rate * 100)
        
        # 创建对比图
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 时延对比
        axes[0].bar(configs, delays, color='skyblue', edgecolor='navy', alpha=0.7)
        axes[0].set_title('平均任务时延对比', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('时延 (秒)', fontsize=12)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(axis='y', alpha=0.3)
        
        # 能耗对比
        axes[1].bar(configs, energies, color='lightcoral', edgecolor='darkred', alpha=0.7)
        axes[1].set_title('系统总能耗对比', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('能耗 (焦耳)', fontsize=12)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(axis='y', alpha=0.3)
        
        # 完成率对比
        axes[2].bar(configs, completions, color='lightgreen', edgecolor='darkgreen', alpha=0.7)
        axes[2].set_title('任务完成率对比', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('完成率 (%)', fontsize=12)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(axis='y', alpha=0.3)
        axes[2].set_ylim([0, 105])
        
        plt.tight_layout()
        plot_path = os.path.join(save_dir, 'ablation_comparison.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"✓ 消融对比图已保存: {plot_path}")
        plt.close()
        
        # 生成雷达图（模块重要性）
        self._generate_radar_chart(save_dir)
    
    def _generate_radar_chart(self, save_dir: str):
        """生成雷达图展示各模块影响"""
        import matplotlib.pyplot as plt
        import matplotlib
        from math import pi
        
        matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        matplotlib.rcParams['axes.unicode_minus'] = False
        
        # 数据准备
        baseline = self.results.get('Full-System')
        if not baseline:
            return
        
        categories = []
        delay_impacts = []
        energy_impacts = []
        
        for config_name, result in self.results.items():
            if config_name in ['Full-System', 'Minimal-System']:
                continue
            
            module_name = config_name.replace('No-', '')
            categories.append(module_name)
            
            # 计算影响百分比
            delay_impact = abs((result.avg_delay - baseline.avg_delay) / baseline.avg_delay * 100)
            energy_impact = abs((result.total_energy - baseline.total_energy) / baseline.total_energy * 100)
            
            delay_impacts.append(delay_impact)
            energy_impacts.append(energy_impact)
        
        # 创建雷达图
        N = len(categories)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        
        delay_impacts += delay_impacts[:1]
        energy_impacts += energy_impacts[:1]
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        ax.plot(angles, delay_impacts, 'o-', linewidth=2, label='时延影响', color='blue')
        ax.fill(angles, delay_impacts, alpha=0.25, color='blue')
        
        ax.plot(angles, energy_impacts, 'o-', linewidth=2, label='能耗影响', color='red')
        ax.fill(angles, energy_impacts, alpha=0.25, color='red')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=12)
        ax.set_ylim(0, max(max(delay_impacts), max(energy_impacts)) * 1.2)
        ax.set_title('各模块对系统性能的影响\n(百分比变化)', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True)
        
        radar_path = os.path.join(save_dir, 'module_impact_radar.png')
        plt.savefig(radar_path, dpi=150, bbox_inches='tight')
        print(f"✓ 模块影响雷达图已保存: {radar_path}")
        plt.close()
    
    def save_results(self, save_dir: str = "results/ablation"):
        """保存消融实验结果"""
        os.makedirs(save_dir, exist_ok=True)
        
        # 保存JSON格式结果
        results_dict = {
            config_name: result.to_dict() 
            for config_name, result in self.results.items()
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(save_dir, f'ablation_results_{timestamp}.json')
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_dict, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 消融实验结果已保存: {json_path}")
        
        # 保存分析报告
        analysis = self.analyze_results()
        analysis_path = os.path.join(save_dir, f'ablation_analysis_{timestamp}.json')
        
        with open(analysis_path, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 消融分析报告已保存: {analysis_path}")

