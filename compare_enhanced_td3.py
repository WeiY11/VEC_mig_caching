#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3系列算法对比实验脚本

自动运行并对比以下4个算法：
1. TD3 - 标准Twin Delayed DDPG
2. CAM_TD3 - Cache-Aware Migration TD3
3. ENHANCED_TD3 - 增强型TD3（所有5项优化）
4. ENHANCED_CAM_TD3 - 增强型CAM_TD3（队列焦点优化）

使用方法:
    # 快速测试（50轮）
    python compare_enhanced_td3.py --mode quick
    
    # 标准实验（500轮）
    python compare_enhanced_td3.py --mode standard
    
    # 自定义配置
    python compare_enhanced_td3.py --episodes 800 --num-vehicles 12
"""

import os
import sys
import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 防止终端编码导致的输出异常（特别是包含 emoji 时）
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


class TD3ComparisonRunner:
    """TD3系列算法对比实验运行器"""
    
    # 实验配置
    ALGORITHMS = ['TD3', 'CAM_TD3', 'ENHANCED_TD3', 'ENHANCED_CAM_TD3']
    
    ALGORITHM_LABELS = {
        'TD3': 'TD3 (标准)',
        'CAM_TD3': 'CAM-TD3 (缓存感知)',
        'ENHANCED_TD3': 'Enhanced TD3 (全优化)',
        'ENHANCED_CAM_TD3': 'Enhanced CAM-TD3 (队列焦点)',
    }
    
    ALGORITHM_COLORS = {
        'TD3': '#1f77b4',
        'CAM_TD3': '#ff7f0e',
        'ENHANCED_TD3': '#2ca02c',
        'ENHANCED_CAM_TD3': '#d62728',
    }
    
    def __init__(
        self,
        episodes: int = 500,
        num_vehicles: int = 12,
        seed: int = 42,
        output_dir: str = 'results/td3_comparison',
        silent_mode: bool = True,
    ):
        self.episodes = episodes
        self.num_vehicles = num_vehicles
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.silent_mode = silent_mode
        
        # 创建输出目录
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.output_dir / f'run_{self.timestamp}'
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # 结果存储
        self.results = {}
        self.training_data = {}
        
        print("=" * 80)
        print("🚀 TD3系列算法对比实验")
        print("=" * 80)
        print(f"实验配置:")
        print(f"  训练轮次: {episodes}")
        print(f"  车辆数量: {num_vehicles}")
        print(f"  随机种子: {seed}")
        print(f"  输出目录: {self.run_dir}")
        print("=" * 80)
    
    def run_single_algorithm(self, algorithm: str) -> Optional[Dict]:
        """运行单个算法"""
        print(f"\n{'='*80}")
        print(f"🔬 开始训练: {self.ALGORITHM_LABELS[algorithm]}")
        print(f"{'='*80}")
        
        # 构建命令
        cmd = [
            sys.executable,
            'train_single_agent.py',
            '--algorithm', algorithm,
            '--episodes', str(self.episodes),
            '--num-vehicles', str(self.num_vehicles),
            '--seed', str(self.seed),
        ]
        
        if self.silent_mode:
            cmd.append('--silent-mode')
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行训练
            print(f"执行命令: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # 记录结束时间
            elapsed_time = time.time() - start_time
            
            print(f"✅ {self.ALGORITHM_LABELS[algorithm]} 训练完成!")
            print(f"⏱️  用时: {elapsed_time/60:.1f} 分钟")
            
            # 查找训练结果文件
            result_file = self._find_latest_result_file(algorithm)
            if result_file:
                print(f"📁 结果文件: {result_file}")
                with open(result_file, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                
                # 保存结果
                self.training_data[algorithm] = training_data
                self.results[algorithm] = {
                    'success': True,
                    'elapsed_time': elapsed_time,
                    'result_file': str(result_file),
                    'final_metrics': self._extract_final_metrics(training_data)
                }
                
                return self.results[algorithm]
            else:
                print(f"⚠️  未找到结果文件")
                return None
                
        except subprocess.CalledProcessError as e:
            elapsed_time = time.time() - start_time
            print(f"❌ {self.ALGORITHM_LABELS[algorithm]} 训练失败!")
            print(f"错误信息: {e.stderr[:500]}")
            
            self.results[algorithm] = {
                'success': False,
                'elapsed_time': elapsed_time,
                'error': str(e)
            }
            return None
        except Exception as e:
            print(f"❌ 运行过程中出现异常: {e}")
            return None
    
    def _find_latest_result_file(self, algorithm: str) -> Optional[Path]:
        """查找最新的训练结果文件"""
        algo_dir = Path('results') / 'single_agent' / algorithm.lower()
        
        if not algo_dir.exists():
            return None
        
        # 查找所有 training_results*.json 文件
        result_files = list(algo_dir.glob('training_results_*.json'))
        
        if not result_files:
            return None
        
        # 返回最新的文件
        return max(result_files, key=lambda p: p.stat().st_mtime)
    
    def _extract_final_metrics(self, training_data: Dict) -> Dict:
        """??????"""
        metrics: Dict[str, float] = {}
        episode_metrics = training_data.get('episode_metrics', {})
        final_perf = training_data.get('final_performance', {})

        def _get_series(key: str) -> Optional[List]:
            # ???episode_metrics??????????
            if isinstance(episode_metrics.get(key), list):
                return episode_metrics[key]
            if isinstance(training_data.get(key), list):
                return training_data[key]
            return None

        def _mean_last(series: List[float]) -> float:
            last_n = max(1, len(series) // 10)
            return float(np.mean(series[-last_n:]))

        # Episode reward
        rewards = _get_series('episode_rewards') or training_data.get('episode_rewards')
        if isinstance(rewards, list) and len(rewards) > 0:
            metrics['final_reward'] = _mean_last(rewards)
            metrics['reward_std'] = float(np.std(rewards[-max(1, len(rewards) // 10):]))
        elif isinstance(final_perf.get('avg_episode_reward'), (int, float)):
            metrics['final_reward'] = float(final_perf['avg_episode_reward'])

        # ????
        metric_keys = [
            'avg_delay', 'total_energy', 'task_completion_rate',
            'cache_hit_rate', 'data_loss_ratio_bytes',
            'migration_success_rate', 'queue_overload_rate'
        ]

        for key in metric_keys:
            series = _get_series(key)
            if isinstance(series, list) and len(series) > 0:
                metrics[f'final_{key}'] = _mean_last(series)

        # ??????
        if isinstance(final_perf.get('avg_delay'), (int, float)):
            metrics.setdefault('final_avg_delay', float(final_perf['avg_delay']))
        if isinstance(final_perf.get('avg_energy'), (int, float)):
            metrics.setdefault('final_total_energy', float(final_perf['avg_energy']))
        if isinstance(final_perf.get('avg_completion'), (int, float)):
            metrics.setdefault('final_task_completion_rate', float(final_perf['avg_completion']))

        return metrics

    def run_all_algorithms(self):
        """运行所有算法"""
        print(f"\n🎯 将依次运行 {len(self.ALGORITHMS)} 个算法")
        print(f"预计总用时: {len(self.ALGORITHMS) * self.episodes * 0.5 / 60:.0f} - {len(self.ALGORITHMS) * self.episodes * 1.0 / 60:.0f} 分钟\n")
        
        for i, algorithm in enumerate(self.ALGORITHMS, 1):
            print(f"\n进度: {i}/{len(self.ALGORITHMS)}")
            self.run_single_algorithm(algorithm)
            
            # 等待一下，确保文件写入完成
            time.sleep(2)
        
        print(f"\n{'='*80}")
        print("✅ 所有算法训练完成!")
        print(f"{'='*80}")
    
    def generate_comparison_report(self):
        """生成对比报告"""
        print(f"\n📊 生成对比报告...")
        
        # 1. 生成对比表格
        self._generate_comparison_table()
        
        # 2. 生成训练曲线对比图
        self._generate_training_curves()
        
        # 3. 生成性能雷达图
        self._generate_radar_chart()
        
        # 4. 生成文本摘要
        self._generate_text_summary()
        
        print(f"\n✅ 报告生成完成!")
        print(f"📁 输出目录: {self.run_dir}")
    
    def _generate_comparison_table(self):
        """生成对比表格"""
        # 收集数据
        table_data = []
        for algo in self.ALGORITHMS:
            if algo in self.results and self.results[algo]['success']:
                metrics = self.results[algo]['final_metrics']
                row = {
                    '算法': self.ALGORITHM_LABELS[algo],
                    '最终奖励': f"{metrics.get('final_reward', 0):.2f}",
                    '任务完成率': f"{metrics.get('final_task_completion_rate', 0)*100:.1f}%",
                    '平均时延(s)': f"{metrics.get('final_avg_delay', 0):.3f}",
                    '总能耗(J)': f"{metrics.get('final_total_energy', 0):.1f}",
                    '缓存命中率': f"{metrics.get('final_cache_hit_rate', 0)*100:.1f}%",
                    '数据丢失率': f"{metrics.get('final_data_loss_ratio_bytes', 0)*100:.2f}%",
                    '训练用时(分)': f"{self.results[algo]['elapsed_time']/60:.1f}",
                }
                table_data.append(row)
        
        # 创建DataFrame
        df = pd.DataFrame(table_data)
        
        # 保存为CSV
        csv_path = self.run_dir / 'comparison_table.csv'
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"  ✓ 对比表格: {csv_path}")
        
        # 打印到控制台
        print(f"\n{'='*100}")
        print("📊 性能对比表")
        print(f"{'='*100}")
        print(df.to_string(index=False))
        print(f"{'='*100}\n")
    
    def _generate_training_curves(self):
        """?????????"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('??????', fontsize=16, fontweight='bold')

        # ?????
        metrics_to_plot = [
            ('episode_rewards', '????', axes[0, 0]),
            ('avg_delay', '???? (s)', axes[0, 1]),
            ('cache_hit_rate', '?????', axes[1, 0]),
            ('task_completion_rate', '?????', axes[1, 1]),
        ]

        for metric_key, metric_label, ax in metrics_to_plot:
            for algo in self.ALGORITHMS:
                data = self.training_data.get(algo)
                if not data:
                    continue

                # ???episode_metrics?????????
                if metric_key == 'episode_rewards':
                    values = data.get('episode_rewards')
                else:
                    values = data.get('episode_metrics', {}).get(metric_key) or data.get(metric_key)

                if not isinstance(values, list) or len(values) == 0:
                    continue

                episodes = range(1, len(values) + 1)

                # ???????????
                ax.plot(
                    episodes,
                    values,
                    color=self.ALGORITHM_COLORS[algo],
                    alpha=0.2,
                    linewidth=1,
                )

                # ??????
                if len(values) > 10:
                    window = min(50, len(values) // 10)
                    smoothed = pd.Series(values).rolling(window=window, min_periods=1).mean()
                    ax.plot(
                        episodes,
                        smoothed,
                        color=self.ALGORITHM_COLORS[algo],
                        label=self.ALGORITHM_LABELS[algo],
                        linewidth=2,
                    )

            ax.set_xlabel('Episode')
            ax.set_ylabel(metric_label)
            ax.set_title(metric_label)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # ????
        fig_path = self.run_dir / 'training_curves.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ?????: {fig_path}")

    def _generate_radar_chart(self):
        """生成性能雷达图"""
        # 定义指标（归一化到0-1）
        metrics_config = [
            ('final_task_completion_rate', '任务完成率', 1.0),
            ('final_cache_hit_rate', '缓存命中率', 1.0),
            ('final_avg_delay', '时延', 0.0),  # 越低越好，需要反转
            ('final_total_energy', '能耗', 0.0),  # 越低越好，需要反转
            ('final_data_loss_ratio_bytes', '数据丢失', 0.0),  # 越低越好，需要反转
        ]
        
        # 收集数据
        radar_data = {}
        for algo in self.ALGORITHMS:
            if algo in self.results and self.results[algo]['success']:
                radar_data[algo] = []
        
        # 归一化数据
        for metric_key, metric_label, better_higher in metrics_config:
            values = []
            for algo in radar_data.keys():
                val = self.results[algo]['final_metrics'].get(metric_key, 0)
                values.append(val)
            
            # 归一化到0-1
            if len(values) > 0:
                min_val = min(values)
                max_val = max(values)
                range_val = max_val - min_val if max_val > min_val else 1.0
                
                for algo, val in zip(radar_data.keys(), values):
                    normalized = (val - min_val) / range_val
                    # 如果越低越好，反转
                    if not better_higher:
                        normalized = 1.0 - normalized
                    radar_data[algo].append(normalized)
        
        # 绘制雷达图
        labels = [label for _, label, _ in metrics_config]
        num_vars = len(labels)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        for algo, values in radar_data.items():
            values_plot = values + values[:1]  # 闭合
            ax.plot(angles, values_plot, 
                   color=self.ALGORITHM_COLORS[algo],
                   linewidth=2, label=self.ALGORITHM_LABELS[algo])
            ax.fill(angles, values_plot, 
                   color=self.ALGORITHM_COLORS[algo], alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('性能雷达图（值越大越好）', fontsize=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        # 保存
        fig_path = self.run_dir / 'performance_radar.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ 性能雷达图: {fig_path}")
    
    def _generate_text_summary(self):
        """生成文本摘要"""
        summary_path = self.run_dir / 'summary.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("TD3系列算法对比实验总结\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"实验时间: {self.timestamp}\n")
            f.write(f"训练轮次: {self.episodes}\n")
            f.write(f"车辆数量: {self.num_vehicles}\n")
            f.write(f"随机种子: {self.seed}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("算法性能排名\n")
            f.write("=" * 80 + "\n\n")
            
            # 按最终奖励排序
            ranking = []
            for algo in self.ALGORITHMS:
                if algo in self.results and self.results[algo]['success']:
                    reward = self.results[algo]['final_metrics'].get('final_reward', -float('inf'))
                    ranking.append((algo, reward))
            
            ranking.sort(key=lambda x: x[1], reverse=True)
            
            for rank, (algo, reward) in enumerate(ranking, 1):
                f.write(f"{rank}. {self.ALGORITHM_LABELS[algo]}\n")
                f.write(f"   最终奖励: {reward:.2f}\n")
                
                metrics = self.results[algo]['final_metrics']
                f.write(f"   任务完成率: {metrics.get('final_task_completion_rate', 0)*100:.1f}%\n")
                f.write(f"   平均时延: {metrics.get('final_avg_delay', 0):.3f}s\n")
                f.write(f"   缓存命中率: {metrics.get('final_cache_hit_rate', 0)*100:.1f}%\n")
                f.write(f"   训练用时: {self.results[algo]['elapsed_time']/60:.1f}分钟\n\n")
        
        print(f"  ✓ 文本摘要: {summary_path}")
    
    def save_experiment_config(self):
        """保存实验配置"""
        config_path = self.run_dir / 'experiment_config.json'
        
        config = {
            'timestamp': self.timestamp,
            'episodes': self.episodes,
            'num_vehicles': self.num_vehicles,
            'seed': self.seed,
            'algorithms': self.ALGORITHMS,
            'results_summary': {
                algo: {
                    'success': self.results[algo]['success'],
                    'elapsed_time': self.results[algo]['elapsed_time'],
                }
                for algo in self.ALGORITHMS
                if algo in self.results
            }
        }
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"  ✓ 实验配置: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='TD3系列算法对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 快速测试（50轮，约5-10分钟）
  python compare_enhanced_td3.py --mode quick
  
  # 标准实验（500轮，约1-2小时）
  python compare_enhanced_td3.py --mode standard
  
  # 完整实验（1500轮，约3-5小时）
  python compare_enhanced_td3.py --mode full
  
  # 自定义配置
  python compare_enhanced_td3.py --episodes 800 --num-vehicles 16 --seed 123
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['quick', 'standard', 'full'],
                       help='实验模式（快速/标准/完整）')
    parser.add_argument('--episodes', type=int, help='训练轮次')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--output-dir', type=str, default='results/td3_comparison',
                       help='输出目录')
    parser.add_argument('--no-silent', action='store_true',
                       help='禁用静默模式（显示训练详情）')
    
    args = parser.parse_args()
    
    # 根据模式设置episodes
    if args.mode:
        mode_episodes = {'quick': 50, 'standard': 500, 'full': 1500}
        episodes = mode_episodes[args.mode]
    else:
        episodes = args.episodes if args.episodes else 500
    
    # 创建运行器
    runner = TD3ComparisonRunner(
        episodes=episodes,
        num_vehicles=args.num_vehicles,
        seed=args.seed,
        output_dir=args.output_dir,
        silent_mode=not args.no_silent,
    )
    
    # 运行实验
    runner.run_all_algorithms()
    
    # 生成报告
    runner.generate_comparison_report()
    
    # 保存配置
    runner.save_experiment_config()
    
    print(f"\n{'='*80}")
    print("🎉 实验完成!")
    print(f"{'='*80}")
    print(f"📁 查看结果: {runner.run_dir}")
    print(f"📊 对比表格: {runner.run_dir / 'comparison_table.csv'}")
    print(f"📈 训练曲线: {runner.run_dir / 'training_curves.png'}")
    print(f"🎯 性能雷达图: {runner.run_dir / 'performance_radar.png'}")
    print(f"📝 文本摘要: {runner.run_dir / 'summary.txt'}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
