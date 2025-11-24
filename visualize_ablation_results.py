#!/usr/bin/env python3
"""
📊 消融实验结果可视化
分析和对比7组实验的效果
"""

import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pathlib import Path
from typing import Dict, List

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


class AblationVisualizer:
    """消融实验结果可视化器"""
    
    def __init__(self, results_dir: str):
        self.results_dir = Path(results_dir)
        self.experiments = [
            "baseline",
            "opt1_distributional",
            "opt2_entropy", 
            "opt3_model",
            "opt4_queue",
            "opt5_gnn",
            "full"
        ]
        
        self.exp_names = {
            "baseline": "TD3\nBaseline",
            "opt1_distributional": "TD3+\nDistributional",
            "opt2_entropy": "TD3+\nEntropy",
            "opt3_model": "TD3+\nModel",
            "opt4_queue": "TD3+\nQueue",
            "opt5_gnn": "TD3+\nGNN",
            "full": "TD3+\nAll"
        }
    
    def load_results(self) -> Dict:
        """加载所有实验结果"""
        results = {}
        for exp_id in self.experiments:
            result_file = self.results_dir / f"{exp_id}_result.json"
            if result_file.exists():
                with open(result_file, 'r', encoding='utf-8') as f:
                    results[exp_id] = json.load(f)
        return results
    
    def plot_comparison_bar(self, results: Dict, output_file: str = "ablation_comparison.png"):
        """绘制对比柱状图"""
        metrics = ['final_reward', 'avg_delay', 'cache_hit_rate', 'training_time']
        metric_names = ['总奖励', '平均延迟(s)', '缓存命中率(%)', '训练时间(分钟)']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx]
            
            values = []
            labels = []
            colors = []
            
            for exp_id in self.experiments:
                if exp_id in results:
                    val = results[exp_id].get(metric, 0)
                    
                    # 特殊处理
                    if metric == 'cache_hit_rate':
                        val *= 100  # 转换为百分比
                    elif metric == 'training_time':
                        val /= 60  # 转换为分钟
                    
                    values.append(val)
                    labels.append(self.exp_names[exp_id])
                    
                    # 颜色编码
                    if exp_id == 'baseline':
                        colors.append('#808080')  # 灰色
                    elif exp_id == 'full':
                        colors.append('#2E7D32')  # 深绿色
                    else:
                        colors.append('#1976D2')  # 蓝色
            
            bars = ax.bar(range(len(values)), values, color=colors, alpha=0.8, edgecolor='black')
            
            # 添加数值标签
            for i, (bar, val) in enumerate(zip(bars, values)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}',
                       ha='center', va='bottom', fontsize=9)
            
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=15, ha='right')
            ax.set_ylabel(metric_name, fontsize=11)
            ax.set_title(f'{metric_name}对比', fontsize=12, fontweight='bold')
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            
            # 延迟越低越好，其他越高越好
            if metric == 'avg_delay' or metric == 'training_time':
                ax.invert_yaxis()
        
        plt.tight_layout()
        output_path = self.results_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 对比柱状图已保存: {output_path}")
        plt.close()
    
    def plot_improvement_radar(self, results: Dict, output_file: str = "improvement_radar.png"):
        """绘制改进雷达图"""
        baseline = results.get('baseline', {})
        full = results.get('full', {})
        
        if not baseline or not full:
            print("⚠️  缺少baseline或full结果，跳过雷达图")
            return
        
        # 计算每个优化的相对贡献
        metrics = {
            'reward': '奖励',
            'delay': '延迟',
            'cache': '缓存',
            'completion': '完成率',
            'efficiency': '效率'
        }
        
        improvements = {}
        for exp_id in self.experiments[1:6]:  # 只看单优化
            if exp_id not in results:
                continue
            
            exp_data = results[exp_id]
            
            # 计算改进度 = (Opt - Baseline) / (Full - Baseline) * 100%
            impr = {}
            
            # 奖励改进
            baseline_reward = baseline.get('final_reward', -500)
            full_reward = full.get('final_reward', -300)
            opt_reward = exp_data.get('final_reward', -400)
            if full_reward != baseline_reward:
                impr['reward'] = max(0, (opt_reward - baseline_reward) / (full_reward - baseline_reward) * 100)
            else:
                impr['reward'] = 0
            
            # 延迟改进（越低越好）
            baseline_delay = baseline.get('avg_delay', 2.0)
            full_delay = full.get('avg_delay', 1.5)
            opt_delay = exp_data.get('avg_delay', 1.8)
            if baseline_delay != full_delay:
                impr['delay'] = max(0, (baseline_delay - opt_delay) / (baseline_delay - full_delay) * 100)
            else:
                impr['delay'] = 0
            
            # 缓存改进
            baseline_cache = baseline.get('cache_hit_rate', 0.002)
            full_cache = full.get('cache_hit_rate', 0.24)
            opt_cache = exp_data.get('cache_hit_rate', 0.1)
            if full_cache != baseline_cache:
                impr['cache'] = max(0, (opt_cache - baseline_cache) / (full_cache - baseline_cache) * 100)
            else:
                impr['cache'] = 0
            
            # 完成率改进
            baseline_comp = baseline.get('completion_rate', 0.98)
            full_comp = full.get('completion_rate', 0.99)
            opt_comp = exp_data.get('completion_rate', 0.985)
            if full_comp != baseline_comp:
                impr['completion'] = max(0, (opt_comp - baseline_comp) / (full_comp - baseline_comp) * 100)
            else:
                impr['completion'] = 0
            
            # 效率改进（训练时间，越短越好）
            baseline_time = baseline.get('training_time', 150)
            full_time = full.get('training_time', 30)
            opt_time = exp_data.get('training_time', 100)
            if baseline_time != full_time:
                impr['efficiency'] = max(0, (baseline_time - opt_time) / (baseline_time - full_time) * 100)
            else:
                impr['efficiency'] = 0
            
            improvements[exp_id] = impr
        
        # 绘制雷达图
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        categories = list(metrics.values())
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=12)
        ax.set_ylim(0, 120)
        ax.set_yticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.grid(True)
        
        # 添加每个优化的线
        opt_labels = {
            'opt1_distributional': 'Distributional',
            'opt2_entropy': 'Entropy',
            'opt3_model': 'Model',
            'opt4_queue': 'Queue',
            'opt5_gnn': 'GNN'
        }
        
        colors = ['#E53935', '#1E88E5', '#43A047', '#FB8C00', '#8E24AA']
        
        for (exp_id, impr), color in zip(improvements.items(), colors):
            values = [impr.get(m, 0) for m in metrics.keys()]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=opt_labels[exp_id], color=color)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        ax.set_title('各优化方法的相对贡献度\n(相对于Baseline→Full的改进)', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.results_dir / output_file
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ 改进雷达图已保存: {output_path}")
        plt.close()
    
    def generate_summary_table(self, results: Dict, output_file: str = "summary_table.md"):
        """生成汇总表格"""
        output_path = self.results_dir / output_file
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# 消融实验结果汇总\n\n")
            
            f.write("## 性能对比\n\n")
            f.write("| 实验 | 总奖励 | 延迟(s) | 缓存命中率(%) | 完成率(%) | 训练时间(分) |\n")
            f.write("|------|--------|---------|---------------|-----------|-------------|\n")
            
            baseline_reward = results.get('baseline', {}).get('final_reward', -500)
            
            for exp_id in self.experiments:
                if exp_id not in results:
                    continue
                
                data = results[exp_id]
                reward = data.get('final_reward', 0)
                delay = data.get('avg_delay', 0)
                cache = data.get('cache_hit_rate', 0) * 100
                comp = data.get('completion_rate', 0) * 100
                time = data.get('training_time', 0) / 60
                
                # 计算相对baseline的改进
                reward_diff = reward - baseline_reward
                reward_str = f"{reward:.1f} ({reward_diff:+.1f})"
                
                f.write(f"| {self.exp_names[exp_id].replace(chr(10), ' ')} | {reward_str} | {delay:.3f} | {cache:.1f} | {comp:.1f} | {time:.1f} |\n")
            
            f.write("\n## 关键发现\n\n")
            
            # 分析哪个优化最有效
            best_single = None
            best_improvement = -float('inf')
            
            for exp_id in self.experiments[1:6]:
                if exp_id not in results:
                    continue
                reward = results[exp_id].get('final_reward', -500)
                improvement = reward - baseline_reward
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_single = exp_id
            
            if best_single:
                f.write(f"1. **最有效的单项优化**: {self.exp_names[best_single].replace(chr(10), ' ')}\n")
                f.write(f"   - 奖励提升: {best_improvement:+.1f}\n\n")
            
            # 检查组合效果
            if 'full' in results and baseline_reward != 0:
                full_reward = results['full'].get('final_reward', -300)
                full_improvement = full_reward - baseline_reward
                f.write(f"2. **全优化组合效果**: 奖励提升 {full_improvement:+.1f}\n\n")
                
                # 计算协同效应
                single_sum = sum([
                    results[exp_id].get('final_reward', baseline_reward) - baseline_reward
                    for exp_id in self.experiments[1:6] if exp_id in results
                ])
                synergy = full_improvement - single_sum
                f.write(f"3. **协同效应**: {synergy:+.1f}\n")
                if synergy > 0:
                    f.write("   - ✅ 存在正协同效应（组合>简单叠加）\n\n")
                else:
                    f.write("   - ⚠️  存在负交互（可能需要调参）\n\n")
        
        print(f"✅ 汇总表格已生成: {output_path}")
    
    def analyze_all(self):
        """执行完整分析"""
        print("\n📊 开始分析消融实验结果...\n")
        
        results = self.load_results()
        
        if not results:
            print("❌ 未找到实验结果文件")
            return
        
        print(f"✅ 加载了 {len(results)} 组实验结果")
        
        # 生成可视化
        self.plot_comparison_bar(results)
        self.plot_improvement_radar(results)
        self.generate_summary_table(results)
        
        print("\n✅ 分析完成！")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='消融实验结果分析')
    parser.add_argument('--results-dir', type=str, required=True,
                       help='实验结果目录路径')
    
    args = parser.parse_args()
    
    visualizer = AblationVisualizer(args.results_dir)
    visualizer.analyze_all()


if __name__ == '__main__':
    main()
