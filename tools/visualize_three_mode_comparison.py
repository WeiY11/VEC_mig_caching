#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CAMTD3三变体对比实验可视化工具
================================

【系统】CAMTD3 = Cache-Aware Migration with Twin Delayed DDPG

【功能】
对CAMTD3三变体对比实验结果进行深度可视化分析
- CAMTD3: 标准版（智能资源分配）
- CAMTD3-Avg: 简化版（固定均匀分配）
- CAMTD3-NoMig: 对比版（禁用迁移）

【输出】
1. 平均成本对比图（含数据标注）
2. 时延-能耗散点图
3. 完成率-缓存命中率对比
4. 详细数据对比表

【使用示例】
```bash
python tools/visualize_three_mode_comparison.py results/three_mode_comparison/suite_20251105_235911
```
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# CAMTD3模式配置（与实验脚本保持一致）
MODES = {
    "camtd3": {
        "name": "CAMTD3",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "description": "CAMTD3标准版：智能体学习资源分配"
    },
    "camtd3_avg": {
        "name": "CAMTD3-Avg",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "--",
        "description": "CAMTD3简化版：固定均匀资源分配"
    },
    "camtd3_nomig": {
        "name": "CAMTD3-NoMig",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "description": "CAMTD3对比版：禁用任务迁移"
    },
    # 兼容旧的key名称
    "standard": {
        "name": "CAMTD3-Avg",
        "color": "#1f77b4",
        "marker": "o",
        "linestyle": "--",
        "description": "CAMTD3简化版：固定均匀资源分配"
    },
    "central": {
        "name": "CAMTD3",
        "color": "#ff7f0e",
        "marker": "s",
        "linestyle": "-",
        "description": "CAMTD3标准版：智能体学习资源分配"
    },
    "nomig": {
        "name": "CAMTD3-NoMig",
        "color": "#2ca02c",
        "marker": "^",
        "linestyle": "-.",
        "description": "CAMTD3对比版：禁用任务迁移"
    },
}


def load_results(result_dir: Path) -> Dict:
    """加载实验结果"""
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到summary.json: {summary_path}")
    
    with open(summary_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_cost_comparison(results: Dict, output_dir: Path):
    """绘制平均成本对比图（改进版）"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== 1. 任务到达率对比 ==========
    if results.get("arrival_rate_results"):
        ax = axes[0]
        arrival_results = results["arrival_rate_results"]
        arrival_rates = [r["arrival_rate"] for r in arrival_results]
        
        # 绘制三条线
        lines = []
        for mode_key, mode_info in MODES.items():
            costs = []
            for r in arrival_results:
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    costs.append(r["modes"][mode_key]["avg_cost"])
            
            if costs:
                line, = ax.plot(arrival_rates[:len(costs)], costs,
                               marker=mode_info["marker"], 
                               color=mode_info["color"],
                               linewidth=2.5, 
                               markersize=10,
                               label=mode_info["name"],
                               markeredgewidth=1.5, 
                               markeredgecolor='white',
                               alpha=0.85)
                lines.append((mode_info["name"], costs))
                
                # 数据标注（显示具体数值）
                for i, (x, y) in enumerate(zip(arrival_rates[:len(costs)], costs)):
                    ax.annotate(f'{y:.0f}', 
                               xy=(x, y), 
                               xytext=(0, 8),
                               textcoords='offset points',
                               ha='center',
                               fontsize=8,
                               color=mode_info["color"],
                               alpha=0.7)
        
        # 检查是否有重叠数据
        if len(lines) >= 2:
            standard_costs = lines[0][1]
            central_costs = lines[1][1]
            if np.allclose(standard_costs, central_costs, rtol=1e-6):
                ax.text(0.5, 0.98, '⚠️ 注意：Avg和Agent数据完全相同（重叠显示）',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('任务到达率 (tasks/s/车)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均成本 (ω_T·时延 + ω_E·能耗)', fontsize=13, fontweight='bold')
        ax.set_title('三种方案平均成本对比 - 任务到达率影响', 
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=11)
    
    # ========== 2. 本地计算资源对比 ==========
    if results.get("compute_resource_results"):
        ax = axes[1]
        compute_results = results["compute_resource_results"]
        compute_ghz = [r["total_compute_ghz"] for r in compute_results]
        
        # 绘制三条线
        lines = []
        for mode_key, mode_info in MODES.items():
            costs = []
            for r in compute_results:
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    costs.append(r["modes"][mode_key]["avg_cost"])
            
            if costs:
                line, = ax.plot(compute_ghz[:len(costs)], costs,
                               marker=mode_info["marker"], 
                               color=mode_info["color"],
                               linewidth=2.5, 
                               markersize=10,
                               label=mode_info["name"],
                               markeredgewidth=1.5, 
                               markeredgecolor='white',
                               alpha=0.85)
                lines.append((mode_info["name"], costs))
                
                # 数据标注
                for i, (x, y) in enumerate(zip(compute_ghz[:len(costs)], costs)):
                    ax.annotate(f'{y:.0f}', 
                               xy=(x, y), 
                               xytext=(0, 8),
                               textcoords='offset points',
                               ha='center',
                               fontsize=8,
                               color=mode_info["color"],
                               alpha=0.7)
        
        # 检查重叠
        if len(lines) >= 2:
            standard_costs = lines[0][1]
            central_costs = lines[1][1]
            if np.allclose(standard_costs, central_costs, rtol=1e-6):
                ax.text(0.5, 0.98, '⚠️ 注意：Avg和Agent数据完全相同（重叠显示）',
                       transform=ax.transAxes,
                       ha='center', va='top',
                       bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5),
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('总本地计算资源 (GHz)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均成本 (ω_T·时延 + ω_E·能耗)', fontsize=13, fontweight='bold')
        ax.set_title('三种方案平均成本对比 - 本地计算资源影响',
                    fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "cost_comparison_enhanced.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[完成] 生成成本对比图: {output_path.name}")


def plot_delay_energy_scatter(results: Dict, output_dir: Path):
    """绘制时延-能耗散点图"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # ========== 1. 任务到达率维度 ==========
    if results.get("arrival_rate_results"):
        ax = axes[0]
        arrival_results = results["arrival_rate_results"]
        
        for mode_key, mode_info in MODES.items():
            delays = []
            energies = []
            rates = []
            
            for r in arrival_results:
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    delays.append(r["modes"][mode_key]["avg_delay"])
                    energies.append(r["modes"][mode_key]["avg_energy"])
                    rates.append(r["arrival_rate"])
            
            if delays:
                scatter = ax.scatter(delays, energies, 
                                    s=150, 
                                    c=rates,
                                    cmap='viridis',
                                    marker=mode_info["marker"],
                                    edgecolors=mode_info["color"],
                                    linewidths=2,
                                    alpha=0.7,
                                    label=mode_info["name"])
        
        ax.set_xlabel('平均时延 (秒)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均能耗 (焦耳)', fontsize=13, fontweight='bold')
        ax.set_title('时延-能耗权衡 (任务到达率维度)', fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('任务到达率 (tasks/s/车)', fontsize=11)
    
    # ========== 2. 本地计算资源维度 ==========
    if results.get("compute_resource_results"):
        ax = axes[1]
        compute_results = results["compute_resource_results"]
        
        for mode_key, mode_info in MODES.items():
            delays = []
            energies = []
            computes = []
            
            for r in compute_results:
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    delays.append(r["modes"][mode_key]["avg_delay"])
                    energies.append(r["modes"][mode_key]["avg_energy"])
                    computes.append(r["total_compute_ghz"])
            
            if delays:
                scatter = ax.scatter(delays, energies,
                                    s=150,
                                    c=computes,
                                    cmap='plasma',
                                    marker=mode_info["marker"],
                                    edgecolors=mode_info["color"],
                                    linewidths=2,
                                    alpha=0.7,
                                    label=mode_info["name"])
        
        ax.set_xlabel('平均时延 (秒)', fontsize=13, fontweight='bold')
        ax.set_ylabel('平均能耗 (焦耳)', fontsize=13, fontweight='bold')
        ax.set_title('时延-能耗权衡 (本地计算资源维度)', fontsize=14, fontweight='bold', pad=15)
        ax.grid(alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best', framealpha=0.9)
        
        # 添加colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('本地计算资源 (GHz)', fontsize=11)
    
    plt.tight_layout()
    output_path = output_dir / "delay_energy_scatter.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[完成] 生成时延-能耗散点图: {output_path.name}")


def generate_comparison_table(results: Dict, output_dir: Path):
    """生成详细对比数据表"""
    
    tables = []
    
    # ========== 1. 任务到达率对比表 ==========
    if results.get("arrival_rate_results"):
        arrival_results = results["arrival_rate_results"]
        
        rows = []
        for r in arrival_results:
            rate = r["arrival_rate"]
            for mode_key, mode_info in MODES.items():
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    m = r["modes"][mode_key]
                    rows.append({
                        "到达率": f"{rate:.1f}",
                        "方案": mode_info["name"],
                        "平均时延(s)": f"{m['avg_delay']:.4f}",
                        "平均能耗(J)": f"{m['avg_energy']:.1f}",
                        "平均成本": f"{m['avg_cost']:.1f}",
                        "完成率": f"{m['completion_rate']*100:.1f}%",
                        "缓存命中率": f"{m['cache_hit_rate']*100:.1f}%",
                    })
        
        df_arrival = pd.DataFrame(rows)
        tables.append(("任务到达率对比", df_arrival))
    
    # ========== 2. 本地计算资源对比表 ==========
    if results.get("compute_resource_results"):
        compute_results = results["compute_resource_results"]
        
        rows = []
        for r in compute_results:
            compute = r["total_compute_ghz"]
            for mode_key, mode_info in MODES.items():
                if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                    m = r["modes"][mode_key]
                    rows.append({
                        "计算资源(GHz)": f"{compute:.1f}",
                        "方案": mode_info["name"],
                        "平均时延(s)": f"{m['avg_delay']:.4f}",
                        "平均能耗(J)": f"{m['avg_energy']:.1f}",
                        "平均成本": f"{m['avg_cost']:.1f}",
                        "完成率": f"{m['completion_rate']*100:.1f}%",
                        "缓存命中率": f"{m['cache_hit_rate']*100:.1f}%",
                    })
        
        df_compute = pd.DataFrame(rows)
        tables.append(("本地计算资源对比", df_compute))
    
    # ========== 保存为CSV和文本 ==========
    for table_name, df in tables:
        # CSV格式
        csv_path = output_dir / f"{table_name}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"[完成] 生成数据表: {csv_path.name}")
        
        # 文本格式（美化）
        txt_path = output_dir / f"{table_name}.txt"
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"{'='*80}\n")
            f.write(f"{table_name}\n")
            f.write(f"{'='*80}\n\n")
            f.write(df.to_string(index=False))
            f.write(f"\n\n{'='*80}\n")
        print(f"[完成] 生成数据表: {txt_path.name}")


def check_data_consistency(results: Dict) -> Dict[str, Any]:
    """检查数据一致性（检测Avg和Agent是否相同）"""
    
    issues = {
        "has_issues": False,
        "identical_results": [],
        "missing_data": [],
    }
    
    # 检查任务到达率维度
    if results.get("arrival_rate_results"):
        for r in results["arrival_rate_results"]:
            rate = r["arrival_rate"]
            standard = r["modes"].get("standard", {})
            central = r["modes"].get("central", {})
            
            if standard.get("success") and central.get("success"):
                # 检查是否完全相同
                if (abs(standard["avg_cost"] - central["avg_cost"]) < 1e-6 and
                    abs(standard["avg_delay"] - central["avg_delay"]) < 1e-9 and
                    abs(standard["avg_energy"] - central["avg_energy"]) < 1e-3):
                    issues["has_issues"] = True
                    issues["identical_results"].append(f"到达率 {rate:.1f}")
    
    # 检查本地计算资源维度
    if results.get("compute_resource_results"):
        for r in results["compute_resource_results"]:
            compute = r["total_compute_ghz"]
            standard = r["modes"].get("standard", {})
            central = r["modes"].get("central", {})
            
            if standard.get("success") and central.get("success"):
                if (abs(standard["avg_cost"] - central["avg_cost"]) < 1e-6 and
                    abs(standard["avg_delay"] - central["avg_delay"]) < 1e-9 and
                    abs(standard["avg_energy"] - central["avg_energy"]) < 1e-3):
                    issues["has_issues"] = True
                    issues["identical_results"].append(f"计算资源 {compute:.1f} GHz")
    
    return issues


def generate_analysis_report(results: Dict, issues: Dict, output_dir: Path):
    """生成分析报告"""
    
    report_path = output_dir / "analysis_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("三种方案对比实验分析报告\n")
        f.write("="*80 + "\n\n")
        
        # 实验信息
        f.write(f"实验时间: {results.get('created_at', 'N/A')}\n")
        f.write(f"训练轮数: {results.get('episodes', 'N/A')}\n")
        f.write(f"随机种子: {results.get('seed', 'N/A')}\n")
        f.write(f"耗时: {results.get('elapsed_time', 'N/A')}\n\n")
        
        # 对比方案
        f.write("-"*80 + "\n")
        f.write("对比方案:\n")
        f.write("-"*80 + "\n")
        for idx, mode in enumerate(results.get("modes", []), 1):
            f.write(f"  {idx}. {mode['name']} - {mode['description']}\n")
        f.write("\n")
        
        # 数据一致性检查
        f.write("-"*80 + "\n")
        f.write("数据一致性检查:\n")
        f.write("-"*80 + "\n")
        if issues["has_issues"]:
            f.write("[警告] 发现问题：以下配置中，Avg和Agent的结果完全相同！\n\n")
            for item in issues["identical_results"]:
                f.write(f"  - {item}\n")
            f.write("\n这可能意味着：\n")
            f.write("  1. CENTRAL_RESOURCE环境变量没有正确生效\n")
            f.write("  2. CentralResourceEnvWrapper没有正常工作\n")
            f.write("  3. 训练过程中资源分配策略没有区分\n\n")
            f.write("建议：\n")
            f.write("  1. 检查train_single_agent.py中的环境变量处理\n")
            f.write("  2. 验证CentralResourceEnvWrapper是否被正确调用\n")
            f.write("  3. 添加调试日志确认两种模式的差异\n")
        else:
            f.write("[完成] 数据一致性正常，三种方案结果存在差异\n")
        f.write("\n")
        
        # 性能总结
        f.write("-"*80 + "\n")
        f.write("性能总结:\n")
        f.write("-"*80 + "\n")
        
        if results.get("arrival_rate_results"):
            f.write("\n【任务到达率维度】\n")
            for r in results["arrival_rate_results"]:
                rate = r["arrival_rate"]
                f.write(f"\n到达率 {rate:.1f} tasks/s/车:\n")
                
                costs = []
                for mode_key, mode_info in MODES.items():
                    if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                        m = r["modes"][mode_key]
                        costs.append((mode_info["name"], m["avg_cost"]))
                        f.write(f"  {mode_info['name']:20s}: 成本={m['avg_cost']:7.1f}, "
                               f"时延={m['avg_delay']:.4f}s, 能耗={m['avg_energy']:6.1f}J, "
                               f"完成率={m['completion_rate']*100:.1f}%\n")
                
                # 最优方案
                if costs:
                    best = min(costs, key=lambda x: x[1])
                    f.write(f"  ➤ 最优: {best[0]} (成本: {best[1]:.1f})\n")
        
        if results.get("compute_resource_results"):
            f.write("\n【本地计算资源维度】\n")
            for r in results["compute_resource_results"]:
                compute = r["total_compute_ghz"]
                f.write(f"\n计算资源 {compute:.1f} GHz:\n")
                
                costs = []
                for mode_key, mode_info in MODES.items():
                    if mode_key in r["modes"] and r["modes"][mode_key].get("success"):
                        m = r["modes"][mode_key]
                        costs.append((mode_info["name"], m["avg_cost"]))
                        f.write(f"  {mode_info['name']:20s}: 成本={m['avg_cost']:7.1f}, "
                               f"时延={m['avg_delay']:.4f}s, 能耗={m['avg_energy']:6.1f}J, "
                               f"完成率={m['completion_rate']*100:.1f}%\n")
                
                # 最优方案
                if costs:
                    best = min(costs, key=lambda x: x[1])
                    f.write(f"  ➤ 最优: {best[0]} (成本: {best[1]:.1f})\n")
        
        f.write("\n" + "="*80 + "\n")
    
    print(f"[完成] 生成分析报告: {report_path.name}")


def main():
    parser = argparse.ArgumentParser(
        description="三种方案对比实验可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "result_dir",
        type=str,
        help="实验结果目录（包含summary.json）"
    )
    
    args = parser.parse_args()
    
    # 加载结果
    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"❌ 目录不存在: {result_dir}")
        sys.exit(1)
    
    print("="*80)
    print("三种方案对比实验可视化")
    print("="*80)
    print(f"\n加载结果: {result_dir}\n")
    
    results = load_results(result_dir)
    
    # 检查数据一致性
    print("[检查] 数据一致性...")
    issues = check_data_consistency(results)
    
    if issues["has_issues"]:
        print("[警告] 发现问题：Avg和Agent的结果完全相同！")
        print("       详见生成的分析报告\n")
    else:
        print("[完成] 数据一致性正常\n")
    
    # 生成可视化
    print("[生成] 可视化图表...")
    plot_cost_comparison(results, result_dir)
    plot_delay_energy_scatter(results, result_dir)
    
    # 生成数据表
    print("\n[生成] 数据对比表...")
    generate_comparison_table(results, result_dir)
    
    # 生成分析报告
    print("\n[生成] 分析报告...")
    generate_analysis_report(results, issues, result_dir)
    
    print("\n" + "="*80)
    print("可视化完成！")
    print("="*80)
    print(f"\n输出目录: {result_dir}")
    print("\n生成的文件:")
    print("  [图] cost_comparison_enhanced.png - 成本对比图（改进版）")
    print("  [图] delay_energy_scatter.png - 时延-能耗散点图")
    print("  [表] 任务到达率对比.csv/.txt - 数据对比表")
    print("  [表] 本地计算资源对比.csv/.txt - 数据对比表")
    print("  [报告] analysis_report.txt - 详细分析报告")
    print("\n" + "="*80)


if __name__ == "__main__":
    main()

