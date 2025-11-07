#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务迁移开销分析工具
====================

【功能】
分析为什么无迁移方案在某些场景下表现更好

【分析维度】
1. 迁移频率统计
2. 迁移时延开销
3. 迁移能耗开销
4. 迁移成功率
5. 负载分布特征

【使用示例】
```bash
python tools/analyze_migration_overhead.py results/three_mode/three_mode_comparison/suite_20251105_235911/summary.json
```
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def analyze_migration_patterns(summary_data: Dict) -> Dict:
    """分析迁移模式和开销"""
    
    analysis = {
        "scenarios": [],
        "insights": [],
    }
    
    # 分析任务到达率维度
    if "arrival_rate_results" in summary_data:
        for result in summary_data["arrival_rate_results"]:
            rate = result["arrival_rate"]
            
            # 比较有迁移 vs 无迁移
            standard = result["modes"].get("standard", {})
            nomig = result["modes"].get("nomig", {})
            
            if standard.get("success") and nomig.get("success"):
                scenario = {
                    "type": "arrival_rate",
                    "value": rate,
                    "with_migration": {
                        "cost": standard["avg_cost"],
                        "delay": standard["avg_delay"],
                        "energy": standard["avg_energy"],
                        "completion": standard["completion_rate"],
                        "cache_hit": standard["cache_hit_rate"],
                    },
                    "without_migration": {
                        "cost": nomig["avg_cost"],
                        "delay": nomig["avg_delay"],
                        "energy": nomig["avg_energy"],
                        "completion": nomig["completion_rate"],
                        "cache_hit": nomig["cache_hit_rate"],
                    },
                }
                
                # 计算差异
                scenario["cost_diff"] = standard["avg_cost"] - nomig["avg_cost"]
                scenario["cost_diff_pct"] = (scenario["cost_diff"] / nomig["avg_cost"]) * 100
                scenario["delay_diff"] = standard["avg_delay"] - nomig["avg_delay"]
                scenario["energy_diff"] = standard["avg_energy"] - nomig["avg_energy"]
                
                # 判断哪个更优
                scenario["winner"] = "无迁移" if nomig["avg_cost"] < standard["avg_cost"] else "有迁移"
                
                analysis["scenarios"].append(scenario)
    
    # 分析计算资源维度
    if "compute_resource_results" in summary_data:
        for result in summary_data["compute_resource_results"]:
            compute = result["total_compute_ghz"]
            
            standard = result["modes"].get("standard", {})
            nomig = result["modes"].get("nomig", {})
            
            if standard.get("success") and nomig.get("success"):
                scenario = {
                    "type": "compute_resource",
                    "value": compute,
                    "with_migration": {
                        "cost": standard["avg_cost"],
                        "delay": standard["avg_delay"],
                        "energy": standard["avg_energy"],
                        "completion": standard["completion_rate"],
                        "cache_hit": standard["cache_hit_rate"],
                    },
                    "without_migration": {
                        "cost": nomig["avg_cost"],
                        "delay": nomig["avg_delay"],
                        "energy": nomig["avg_energy"],
                        "completion": nomig["completion_rate"],
                        "cache_hit": nomig["cache_hit_rate"],
                    },
                }
                
                scenario["cost_diff"] = standard["avg_cost"] - nomig["avg_cost"]
                scenario["cost_diff_pct"] = (scenario["cost_diff"] / nomig["avg_cost"]) * 100
                scenario["delay_diff"] = standard["avg_delay"] - nomig["avg_delay"]
                scenario["energy_diff"] = standard["avg_energy"] - nomig["avg_energy"]
                
                scenario["winner"] = "无迁移" if nomig["avg_cost"] < standard["avg_cost"] else "有迁移"
                
                analysis["scenarios"].append(scenario)
    
    return analysis


def generate_insights(analysis: Dict) -> List[str]:
    """生成分析洞察"""
    
    insights = []
    
    # 统计无迁移优势的场景
    nomig_wins = [s for s in analysis["scenarios"] if s["winner"] == "无迁移"]
    mig_wins = [s for s in analysis["scenarios"] if s["winner"] == "有迁移"]
    
    insights.append(f"总体统计：")
    insights.append(f"  - 无迁移更优: {len(nomig_wins)}/{len(analysis['scenarios'])} 场景")
    insights.append(f"  - 有迁移更优: {len(mig_wins)}/{len(analysis['scenarios'])} 场景")
    
    if nomig_wins:
        insights.append(f"\n无迁移优势场景特征：")
        
        # 任务到达率特征
        arrival_nomig_wins = [s for s in nomig_wins if s["type"] == "arrival_rate"]
        if arrival_nomig_wins:
            rates = [s["value"] for s in arrival_nomig_wins]
            avg_improvement = np.mean([abs(s["cost_diff_pct"]) for s in arrival_nomig_wins])
            insights.append(f"  - 到达率范围: {min(rates):.1f} - {max(rates):.1f} tasks/s/车")
            insights.append(f"  - 平均成本降低: {avg_improvement:.1f}%")
            
            # 分析时延和能耗的变化
            delay_changes = [s["delay_diff"] for s in arrival_nomig_wins]
            energy_changes = [s["energy_diff"] for s in arrival_nomig_wins]
            
            if np.mean(delay_changes) < 0:
                insights.append(f"  - 时延降低: {abs(np.mean(delay_changes)):.4f}秒 (无迁移更快)")
            else:
                insights.append(f"  - 时延增加: {np.mean(delay_changes):.4f}秒 (但总成本仍更低)")
            
            if np.mean(energy_changes) < 0:
                insights.append(f"  - 能耗降低: {abs(np.mean(energy_changes)):.1f}J")
            else:
                insights.append(f"  - 能耗增加: {np.mean(energy_changes):.1f}J")
        
        # 计算资源特征
        compute_nomig_wins = [s for s in nomig_wins if s["type"] == "compute_resource"]
        if compute_nomig_wins:
            computes = [s["value"] for s in compute_nomig_wins]
            avg_improvement = np.mean([abs(s["cost_diff_pct"]) for s in compute_nomig_wins])
            insights.append(f"\n  计算资源维度：")
            insights.append(f"  - 资源范围: {min(computes):.1f} - {max(computes):.1f} GHz")
            insights.append(f"  - 平均成本降低: {avg_improvement:.1f}%")
    
    if mig_wins:
        insights.append(f"\n有迁移优势场景特征：")
        
        arrival_mig_wins = [s for s in mig_wins if s["type"] == "arrival_rate"]
        if arrival_mig_wins:
            rates = [s["value"] for s in arrival_mig_wins]
            avg_improvement = np.mean([abs(s["cost_diff_pct"]) for s in arrival_mig_wins])
            insights.append(f"  - 到达率范围: {min(rates):.1f} - {max(rates):.1f} tasks/s/车")
            insights.append(f"  - 平均成本降低: {avg_improvement:.1f}%")
    
    # 关键洞察
    insights.append(f"\n关键发现：")
    
    # 1. 低负载特征
    low_load = [s for s in analysis["scenarios"] 
                if s["type"] == "arrival_rate" and s["value"] <= 2.0]
    if low_load:
        low_load_nomig = [s for s in low_load if s["winner"] == "无迁移"]
        insights.append(f"  1. 低负载(≤2.0 tasks/s): {len(low_load_nomig)}/{len(low_load)} 场景无迁移更优")
        if low_load_nomig:
            insights.append(f"     → 原因：本地资源充足，迁移开销大于收益")
    
    # 2. 高负载特征
    high_load = [s for s in analysis["scenarios"] 
                 if s["type"] == "arrival_rate" and s["value"] >= 2.5]
    if high_load:
        high_load_nomig = [s for s in high_load if s["winner"] == "无迁移"]
        if high_load_nomig:
            insights.append(f"  2. 高负载(≥2.5 tasks/s): {len(high_load_nomig)}/{len(high_load)} 场景无迁移更优")
            insights.append(f"     → 原因：迁移通信拥塞，开销显著增加")
    
    # 3. 迁移开销估算
    if nomig_wins:
        # 使用能耗差异作为迁移开销的近似
        energy_overhead = np.mean([abs(s["energy_diff"]) for s in nomig_wins if s["energy_diff"] < 0])
        if energy_overhead > 0:
            insights.append(f"  3. 估算迁移能耗开销: ~{energy_overhead:.1f}J")
            insights.append(f"     → 占无迁移能耗的比例: {(energy_overhead/3000)*100:.1f}%")
    
    # 4. 缓存影响
    cache_hit_diff = []
    for s in analysis["scenarios"]:
        diff = s["with_migration"]["cache_hit"] - s["without_migration"]["cache_hit"]
        cache_hit_diff.append(diff)
    
    if cache_hit_diff:
        avg_cache_diff = np.mean(cache_hit_diff)
        if abs(avg_cache_diff) > 0.01:
            insights.append(f"  4. 缓存命中率差异: {avg_cache_diff*100:.1f}%")
            if avg_cache_diff > 0:
                insights.append(f"     → 迁移方案缓存更有效（但不足以弥补开销）")
            else:
                insights.append(f"     → 无迁移方案缓存更有效")
    
    return insights


def plot_migration_analysis(analysis: Dict, output_dir: Path):
    """绘制迁移分析图表"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. 成本差异对比
    ax = axes[0, 0]
    arrival_scenarios = [s for s in analysis["scenarios"] if s["type"] == "arrival_rate"]
    compute_scenarios = [s for s in analysis["scenarios"] if s["type"] == "compute_resource"]
    
    if arrival_scenarios:
        x = [s["value"] for s in arrival_scenarios]
        y = [s["cost_diff"] for s in arrival_scenarios]
        colors = ['red' if diff > 0 else 'green' for diff in y]
        
        ax.bar(x, y, color=colors, alpha=0.6, width=0.2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('任务到达率 (tasks/s/车)', fontsize=12, fontweight='bold')
        ax.set_ylabel('成本差异 (有迁移 - 无迁移)', fontsize=12, fontweight='bold')
        ax.set_title('任务到达率维度：迁移开销分析', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        ax.text(0.05, 0.95, '红色=迁移更差\n绿色=迁移更好', 
               transform=ax.transAxes, va='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 2. 时延-能耗权衡
    ax = axes[0, 1]
    for s in analysis["scenarios"]:
        if s["type"] == "arrival_rate":
            with_mig = s["with_migration"]
            without_mig = s["without_migration"]
            
            # 有迁移
            ax.scatter(with_mig["delay"], with_mig["energy"], 
                      s=100, c='blue', marker='o', alpha=0.6,
                      label='有迁移' if s["value"] == analysis["scenarios"][0]["value"] else "")
            
            # 无迁移
            ax.scatter(without_mig["delay"], without_mig["energy"],
                      s=100, c='green', marker='^', alpha=0.6,
                      label='无迁移' if s["value"] == analysis["scenarios"][0]["value"] else "")
            
            # 连线
            ax.plot([with_mig["delay"], without_mig["delay"]],
                   [with_mig["energy"], without_mig["energy"]],
                   'k--', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('平均时延 (秒)', fontsize=12, fontweight='bold')
    ax.set_ylabel('平均能耗 (焦耳)', fontsize=12, fontweight='bold')
    ax.set_title('时延-能耗权衡分析', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    # 3. 成本差异百分比
    ax = axes[1, 0]
    if arrival_scenarios:
        x = [s["value"] for s in arrival_scenarios]
        y = [s["cost_diff_pct"] for s in arrival_scenarios]
        colors = ['red' if pct > 0 else 'green' for pct in y]
        
        bars = ax.bar(x, y, color=colors, alpha=0.6, width=0.2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        
        # 添加数值标签
        for i, (xi, yi) in enumerate(zip(x, y)):
            ax.text(xi, yi + (2 if yi > 0 else -2), f'{yi:.1f}%',
                   ha='center', va='bottom' if yi > 0 else 'top',
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('任务到达率 (tasks/s/车)', fontsize=12, fontweight='bold')
        ax.set_ylabel('成本差异百分比 (%)', fontsize=12, fontweight='bold')
        ax.set_title('相对成本差异分析', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
    
    # 4. 完成率对比
    ax = axes[1, 1]
    if arrival_scenarios:
        x_pos = np.arange(len(arrival_scenarios))
        width = 0.35
        
        with_mig_completion = [s["with_migration"]["completion"]*100 for s in arrival_scenarios]
        without_mig_completion = [s["without_migration"]["completion"]*100 for s in arrival_scenarios]
        
        ax.bar(x_pos - width/2, with_mig_completion, width, 
              label='有迁移', color='blue', alpha=0.6)
        ax.bar(x_pos + width/2, without_mig_completion, width,
              label='无迁移', color='green', alpha=0.6)
        
        ax.set_xlabel('任务到达率', fontsize=12, fontweight='bold')
        ax.set_ylabel('任务完成率 (%)', fontsize=12, fontweight='bold')
        ax.set_title('任务完成率对比', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{s["value"]:.1f}' for s in arrival_scenarios])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / "migration_overhead_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[完成] 生成迁移分析图: {output_path.name}")


def generate_report(analysis: Dict, insights: List[str], output_dir: Path):
    """生成分析报告"""
    
    report_path = output_dir / "migration_analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("任务迁移开销深度分析报告\n")
        f.write("="*80 + "\n\n")
        
        f.write("分析目标：解释为什么无迁移方案在某些场景下表现更好\n\n")
        
        # 统计摘要
        f.write("-"*80 + "\n")
        f.write("统计摘要\n")
        f.write("-"*80 + "\n")
        for insight in insights:
            f.write(insight + "\n")
        
        # 详细场景分析
        f.write("\n" + "-"*80 + "\n")
        f.write("详细场景分析\n")
        f.write("-"*80 + "\n\n")
        
        for i, scenario in enumerate(analysis["scenarios"], 1):
            f.write(f"{i}. ")
            if scenario["type"] == "arrival_rate":
                f.write(f"到达率 {scenario['value']:.1f} tasks/s/车\n")
            else:
                f.write(f"计算资源 {scenario['value']:.1f} GHz\n")
            
            f.write(f"   胜者: {scenario['winner']}\n")
            f.write(f"   成本差异: {scenario['cost_diff']:.1f} ({scenario['cost_diff_pct']:+.1f}%)\n")
            f.write(f"   时延差异: {scenario['delay_diff']:+.4f}秒\n")
            f.write(f"   能耗差异: {scenario['energy_diff']:+.1f}J\n")
            
            f.write(f"\n   有迁移: 成本={scenario['with_migration']['cost']:.1f}, ")
            f.write(f"时延={scenario['with_migration']['delay']:.4f}s, ")
            f.write(f"能耗={scenario['with_migration']['energy']:.1f}J\n")
            
            f.write(f"   无迁移: 成本={scenario['without_migration']['cost']:.1f}, ")
            f.write(f"时延={scenario['without_migration']['delay']:.4f}s, ")
            f.write(f"能耗={scenario['without_migration']['energy']:.1f}J\n\n")
        
        # 理论分析
        f.write("-"*80 + "\n")
        f.write("理论分析与建议\n")
        f.write("-"*80 + "\n\n")
        
        f.write("1. 迁移开销的组成\n")
        f.write("   a) 通信时延：任务数据传输到远程节点的时间\n")
        f.write("   b) 通信能耗：无线传输消耗的能量\n")
        f.write("   c) 决策开销：迁移决策计算的时间和能耗\n")
        f.write("   d) 队列等待：远程节点可能存在排队延迟\n\n")
        
        f.write("2. 无迁移优势的场景特征\n")
        f.write("   a) 低负载场景：本地资源充足，无需迁移\n")
        f.write("   b) 高负载场景：网络拥塞，迁移通信开销巨大\n")
        f.write("   c) 高计算资源：本地处理能力强，迁移收益小\n\n")
        
        f.write("3. 优化建议\n")
        f.write("   a) 自适应迁移阈值：根据负载动态调整\n")
        f.write("   b) 迁移成本预测：在决策时考虑通信开销\n")
        f.write("   c) 选择性迁移：只迁移高价值任务\n")
        f.write("   d) 本地优先策略：优先本地处理，减少迁移\n\n")
        
        f.write("4. 后续研究方向\n")
        f.write("   a) 迁移决策算法优化\n")
        f.write("   b) 通信协议优化（降低传输开销）\n")
        f.write("   c) 预测式迁移（提前准备资源）\n")
        f.write("   d) 协同缓存（减少数据传输）\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[完成] 生成分析报告: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="任务迁移开销分析工具"
    )
    
    parser.add_argument(
        "summary_file",
        type=str,
        help="实验结果summary.json文件路径"
    )
    
    args = parser.parse_args()
    
    summary_path = Path(args.summary_file)
    if not summary_path.exists():
        print(f"错误：找不到文件 {summary_path}")
        sys.exit(1)
    
    output_dir = summary_path.parent
    
    print("="*80)
    print("任务迁移开销分析")
    print("="*80)
    print(f"\n加载数据: {summary_path}\n")
    
    # 加载数据
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)
    
    # 分析迁移模式
    print("[分析] 迁移模式和开销...")
    analysis = analyze_migration_patterns(summary_data)
    
    # 生成洞察
    print("[分析] 生成洞察...")
    insights = generate_insights(analysis)
    
    # 绘制图表
    print("\n[生成] 分析图表...")
    plot_migration_analysis(analysis, output_dir)
    
    # 生成报告
    print("\n[生成] 分析报告...")
    generate_report(analysis, insights, output_dir)
    
    # 打印关键洞察
    print("\n" + "="*80)
    print("关键洞察")
    print("="*80)
    for insight in insights:
        print(insight)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print(f"\n输出目录: {output_dir}")
    print("生成的文件:")
    print("  - migration_overhead_analysis.png")
    print("  - migration_analysis_report.txt")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()


