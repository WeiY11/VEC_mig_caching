#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融实验结果分析脚本
用于分析和可视化已完成的实验结果

【用途】
1. 重新分析已有实验结果
2. 生成论文图表
3. 导出LaTeX表格
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class AblationResultAnalyzer:
    """消融实验结果分析器"""
    
    def __init__(self, results_dir: str = None):
        """初始化分析器"""
        if results_dir is None:
            self.results_dir = Path(__file__).parent / "results"
        else:
            self.results_dir = Path(results_dir)
        
        self.analysis_dir = Path(__file__).parent / "analysis"
        self.analysis_dir.mkdir(exist_ok=True, parents=True)
        
        self.results = {}
    
    def load_results(self, summary_file: str = None):
        """加载实验结果"""
        if summary_file:
            # 加载指定的汇总文件
            with open(summary_file, 'r', encoding='utf-8') as f:
                self.results = json.load(f)
            print(f"✓ 加载实验结果: {summary_file}")
        else:
            # 加载所有配置目录中的结果
            for config_dir in self.results_dir.iterdir():
                if config_dir.is_dir():
                    result_file = config_dir / f"result_{config_dir.name}.json"
                    if result_file.exists():
                        with open(result_file, 'r', encoding='utf-8') as f:
                            result = json.load(f)
                            self.results[config_dir.name] = result
            
            print(f"✓ 加载了 {len(self.results)} 个配置的结果")
    
    def generate_latex_table(self):
        """生成LaTeX表格"""
        if not self.results:
            print("⚠️ 没有结果可生成表格")
            return
        
        print("\n生成LaTeX表格...")
        
        latex_code = r"""\begin{table}[h]
\centering
\caption{消融实验结果对比}
\label{tab:ablation_results}
\begin{tabular}{lccccc}
\hline
配置 & 平均时延(s) & 总能耗(J) & 完成率(\%) & 缓存命中率(\%) & 迁移成功率(\%) \\
\hline
"""
        
        # 确保Full-System在第一行
        if 'Full-System' in self.results:
            result = self.results['Full-System']
            latex_code += f"Full-System & {result['avg_delay']:.3f} & {result['avg_energy']:.1f} & "
            latex_code += f"{result['avg_completion_rate']*100:.1f} & "
            latex_code += f"{result['avg_cache_hit_rate']*100:.1f} & "
            latex_code += f"{result['avg_migration_success_rate']*100:.1f} \\\\\n"
        
        # 添加其他配置
        for config_name, result in self.results.items():
            if config_name != 'Full-System':
                latex_code += f"{config_name} & {result['avg_delay']:.3f} & {result['avg_energy']:.1f} & "
                latex_code += f"{result['avg_completion_rate']*100:.1f} & "
                latex_code += f"{result['avg_cache_hit_rate']*100:.1f} & "
                latex_code += f"{result['avg_migration_success_rate']*100:.1f} \\\\\n"
        
        latex_code += r"""\hline
\end{tabular}
\end{table}
"""
        
        # 保存LaTeX代码
        latex_file = self.analysis_dir / "ablation_table.tex"
        with open(latex_file, 'w', encoding='utf-8') as f:
            f.write(latex_code)
        
        print(f"✓ LaTeX表格已保存: {latex_file}")
        
        # 同时打印到控制台
        print("\nLaTeX表格代码:")
        print("-"*60)
        print(latex_code)
        print("-"*60)
    
    def generate_comparison_report(self):
        """生成对比报告"""
        if not self.results:
            print("⚠️ 没有结果可生成报告")
            return
        
        baseline = self.results.get('Full-System')
        if not baseline:
            print("⚠️ 未找到Full-System基准")
            return
        
        print("\n生成对比报告...")
        
        report = "# 消融实验对比报告\n\n"
        report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        report += "## 基准配置 (Full-System)\n\n"
        report += f"- 平均时延: {baseline['avg_delay']:.3f}±{baseline['std_delay']:.3f}s\n"
        report += f"- 平均能耗: {baseline['avg_energy']:.1f}±{baseline['std_energy']:.1f}J\n"
        report += f"- 任务完成率: {baseline['avg_completion_rate']:.2%}\n"
        report += f"- 缓存命中率: {baseline['avg_cache_hit_rate']:.2%}\n"
        report += f"- 迁移成功率: {baseline['avg_migration_success_rate']:.2%}\n\n"
        
        report += "## 各配置对比\n\n"
        report += "| 配置 | 时延变化 | 能耗变化 | 完成率变化 | 综合影响 |\n"
        report += "|------|----------|----------|------------|----------|\n"
        
        impacts = []
        for config_name, result in self.results.items():
            if config_name == 'Full-System':
                continue
            
            delay_change = (result['avg_delay'] - baseline['avg_delay']) / baseline['avg_delay'] * 100
            energy_change = (result['avg_energy'] - baseline['avg_energy']) / baseline['avg_energy'] * 100
            completion_change = (result['avg_completion_rate'] - baseline['avg_completion_rate']) * 100
            impact_score = abs(delay_change) * 0.4 + abs(energy_change) * 0.3 + abs(completion_change) * 0.3
            
            impacts.append((config_name, impact_score, delay_change, energy_change, completion_change))
            
            report += f"| {config_name} | {delay_change:+.1f}% | {energy_change:+.1f}% | "
            report += f"{completion_change:+.1f}% | {impact_score:.1f} |\n"
        
        report += "\n## 模块重要性排序\n\n"
        impacts.sort(key=lambda x: x[1], reverse=True)
        for i, (name, score, delay_ch, energy_ch, comp_ch) in enumerate(impacts, 1):
            module = name.replace('No-', '').replace('Minimal-', '')
            report += f"{i}. **{module}** (影响力: {score:.1f})\n"
            report += f"   - 时延影响: {delay_ch:+.1f}%\n"
            report += f"   - 能耗影响: {energy_ch:+.1f}%\n"
            report += f"   - 完成率影响: {comp_ch:+.1f}%\n\n"
        
        # 保存报告
        report_file = self.analysis_dir / "comparison_report.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✓ 对比报告已保存: {report_file}")
    
    def print_summary(self):
        """打印结果摘要"""
        if not self.results:
            print("⚠️ 没有结果")
            return
        
        print("\n" + "="*80)
        print("📊 消融实验结果摘要")
        print("="*80)
        
        print(f"\n{'配置':<20} {'时延(s)':<12} {'能耗(J)':<12} {'完成率':<10} {'缓存率':<10}")
        print("-"*80)
        
        # Full-System在前
        if 'Full-System' in self.results:
            r = self.results['Full-System']
            print(f"{'Full-System':<20} {r['avg_delay']:<12.3f} {r['avg_energy']:<12.1f} "
                  f"{r['avg_completion_rate']*100:<10.1f} {r['avg_cache_hit_rate']*100:<10.1f}")
        
        # 其他配置
        for name, r in self.results.items():
            if name != 'Full-System':
                print(f"{name:<20} {r['avg_delay']:<12.3f} {r['avg_energy']:<12.1f} "
                      f"{r['avg_completion_rate']*100:<10.1f} {r['avg_cache_hit_rate']*100:<10.1f}")
        
        print("="*80)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分析消融实验结果')
    parser.add_argument('--summary', type=str, default=None, 
                       help='指定汇总结果文件路径')
    parser.add_argument('--latex', action='store_true', 
                       help='生成LaTeX表格')
    parser.add_argument('--report', action='store_true', 
                       help='生成对比报告')
    
    args = parser.parse_args()
    
    # 创建分析器
    analyzer = AblationResultAnalyzer()
    
    # 加载结果
    analyzer.load_results(args.summary)
    
    # 打印摘要
    analyzer.print_summary()
    
    # 生成LaTeX表格
    if args.latex or not (args.latex or args.report):
        analyzer.generate_latex_table()
    
    # 生成对比报告
    if args.report or not (args.latex or args.report):
        analyzer.generate_comparison_report()
    
    print("\n✓ 分析完成")


if __name__ == "__main__":
    main()

