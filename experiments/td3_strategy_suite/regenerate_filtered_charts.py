#!/usr/bin/env python3
"""
重新生成可视化图表 - 过滤版本
==================================

功能：
- 过滤掉 resource-only 和 remote-only 策略
- 移除标签中的括号及其内容
- 使用现有的 summary.json 数据重新生成图表
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, cast

import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.td3_strategy_suite.strategy_runner import (
    strategy_label,
    strategy_group,
)

# 过滤掉的策略
EXCLUDED_STRATEGIES = ["resource-only", "remote-only"]

# 策略颜色配置
STRATEGY_COLORS = {
    "local-only": "#1f77b4",
    "offloading-only": "#2ca02c",
    "comprehensive-no-migration": "#9467bd",
    "comprehensive-migration": "#8c564b",
    "random": "#e377c2",
    "round-robin": "#7f7f7f",
}

# 分组样式
GROUP_STYLE = {
    "baseline": {"color": "#1f77b4", "linestyle": "--"},
    "layered": {"color": "#ff7f0e", "linestyle": "-"},
    "heuristic": {"color": "#7f7f7f", "linestyle": ":"},
}
GROUP_STYLE["default"] = {"color": "#7f7f7f", "linestyle": ":"}


def clean_label(label: str) -> str:
    """移除标签中的括号及其内容，并去掉baseline字样"""
    import re
    # 移除括号及其内容
    cleaned = re.sub(r'\s*\([^)]*\)', '', label)
    # 移除 "baseline" 字样（不区分大小写）
    cleaned = re.sub(r'\s*baseline\s*', ' ', cleaned, flags=re.IGNORECASE)
    # 清理多余空格
    cleaned = ' '.join(cleaned.split())
    return cleaned.strip()


# 不需要清理标签函数，保持原样


def plot_filtered_results(
    results: List[Dict[str, object]],
    suite_dir: Path,
    strategy_keys: List[str],
    *,
    chart_prefix: str,
    title_prefix: str,
    x_label: str,
) -> List[Path]:
    """绘制过滤后的结果图表"""
    labels = [str(record["label"]) for record in results]
    x_positions = range(len(results))
    saved_paths: List[Path] = []

    def make_chart(metric: str, ylabel: str, suffix: str, highlight_adaptive: bool = False) -> None:
        # 设置白色背景样式
        plt.style.use('default')
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['savefig.facecolor'] = 'white'
        
        plt.figure(figsize=(12, 7))
        
        # 分组策略
        adaptive_strategies = ['comprehensive-no-migration', 'comprehensive-migration']
        baseline_strategies = ['local-only', 'offloading-only']
        heuristic_strategies = ['random', 'round-robin']
        
        for strat_key in strategy_keys:
            values: List[float] = []
            for r in results:
                strategies_dict = cast(Dict[str, object], r["strategies"])
                strat_dict = cast(Dict[str, object], strategies_dict[strat_key])
                values.append(float(cast(float, strat_dict[metric])))
            
            group_name = strategy_group(strat_key)
            style = GROUP_STYLE.get(group_name, GROUP_STYLE["default"])
            
            # 获取原始标签并清理括号
            raw_label = f"{strategy_label(strat_key)} ({group_name})"
            label = clean_label(raw_label)
            
            color = STRATEGY_COLORS.get(strat_key, style.get("color"))
            linestyle = style.get("linestyle", "-")
            
            # 突出显示TD3策略
            if highlight_adaptive and strat_key in adaptive_strategies:
                linewidth = 3.0
                markersize = 10
                alpha = 1.0
            else:
                linewidth = 2.0 if strat_key in baseline_strategies else 1.5
                markersize = 8 if strat_key in baseline_strategies else 6
                alpha = 0.7 if strat_key in heuristic_strategies else 1.0
            
            plt.plot(
                x_positions,
                values,
                marker="o",
                linewidth=linewidth,
                markersize=markersize,
                label=label,
                color=color,
                linestyle=linestyle,
                alpha=alpha,
            )
        
        plt.xticks(x_positions, cast(List[str], labels), fontsize=11)
        plt.xlabel(x_label, fontsize=13, fontweight='bold')
        plt.ylabel(ylabel, fontsize=13, fontweight='bold')
        plt.title(f"Impact of {title_prefix} on {ylabel}", fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3, linestyle='--')
        plt.legend(fontsize=10, loc='best', framealpha=0.95)
        plt.tight_layout()
        filename = f"{chart_prefix}_vs_{suffix}.png"
        out_path = suite_dir / filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()
        saved_paths.append(out_path)

    # 基础性能指标
    make_chart("raw_cost", "Average Cost", "total_cost")
    make_chart("avg_delay", "Average Delay (s)", "delay")
    make_chart("normalized_cost", "Normalized Cost", "normalized_cost", highlight_adaptive=True)
    make_chart("avg_throughput_mbps", "Average Throughput (Mbps)", "throughput")
    
    # 资源利用率图表
    make_chart("avg_rsu_utilization", "RSU Utilization", "rsu_utilization")
    make_chart("avg_offload_ratio", "Offload Ratio", "offload_ratio")
    make_chart("avg_queue_length", "Average Queue Length", "queue_length")
    make_chart("resource_efficiency", "Resource Efficiency", "efficiency")

    return saved_paths


def regenerate_charts(summary_path: Path) -> None:
    """从 summary.json 重新生成图表"""
    
    if not summary_path.exists():
        print(f"错误：找不到文件 {summary_path}")
        return
    
    # 读取 summary.json
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)
    
    # 获取策略列表并过滤
    all_strategies = summary.get("strategies", "").split(", ")
    filtered_strategies = [s.strip() for s in all_strategies if s.strip() not in EXCLUDED_STRATEGIES]
    
    print(f"\n原始策略数量: {len(all_strategies)}")
    print(f"过滤后策略数量: {len(filtered_strategies)}")
    print(f"\n已排除的策略: {', '.join(EXCLUDED_STRATEGIES)}")
    print(f"保留的策略: {', '.join(filtered_strategies)}")
    
    # 过滤结果数据
    results = summary.get("results", [])
    for record in results:
        strategies_dict = record.get("strategies", {})
        # 从每个配置中移除被排除的策略
        for excluded in EXCLUDED_STRATEGIES:
            strategies_dict.pop(excluded, None)
    
    # 获取输出目录
    suite_dir = summary_path.parent
    
    # 提取图表参数
    experiment_key = summary.get("experiment_key", "")
    title_prefix = summary.get("title_prefix", "")
    axis_label = summary.get("axis_label", "")
    chart_prefix = experiment_key  # 通常 experiment_key 就是 chart_prefix
    
    print(f"\n正在重新生成图表...")
    print(f"输出目录: {suite_dir}")
    
    # 绘制过滤后的图表
    saved_paths = plot_filtered_results(
        results,
        suite_dir,
        filtered_strategies,
        chart_prefix=chart_prefix,
        title_prefix=title_prefix,
        x_label=axis_label,
    )
    
    print("\n✅ 图表已重新生成:")
    for path in saved_paths:
        print(f"  - {path.name}")
    
    print(f"\n总计生成 {len(saved_paths)} 个图表文件")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="重新生成过滤后的可视化图表（移除 resource-only 和 remote-only）"
    )
    parser.add_argument(
        "summary_path",
        type=str,
        help="summary.json 文件的路径",
    )
    
    args = parser.parse_args()
    summary_path = Path(args.summary_path)
    
    regenerate_charts(summary_path)


if __name__ == "__main__":
    main()
