#!/usr/bin/env python3
"""
CAMTD3 完整策略套件批量运行器
=============================

【功能】
一键执行完整的CAMTD3消融实验套件，自动化以下流程：
1. 顺序训练所有（或指定的）策略配置
2. 收集所有训练结果到统一的suite目录
3. 计算归一化代价（normalized cost）
4. 生成策略对比图表

这是进行系统化消融实验的推荐方式，确保所有策略使用一致的参数和环境。

【论文对应】
- 消融实验（Ablation Study）：系统性评估各模块的贡献
- 归一化对比：将不同策略的性能归一化到[0,1]区间，便于可视化
- 生成论文图表：适合直接用于论文的高质量对比图

【与run_strategy_training.py的区别】
- run_strategy_training.py：单策略训练工具（手动调用多次）
- run_full_suite.py：批量自动化工具（一次运行所有策略）

【工作流程】
1. 解析命令行参数（选择策略、设置episodes/seed等）
2. 循环调用run_strategy()训练每个策略
3. 加载summary.json
4. 计算归一化代价（相对于最差策略）
5. 生成对比折线图
6. 输出最终结果摘要

【输出结构】
results/camtd3_strategy_suite/{suite_id}/
├── summary.json                          # 汇总文件（含归一化代价）
├── camtd3_strategy_cost.png              # 策略对比图表
├── local-only/                           # 策略1结果
│   ├── training_results_*.json
│   └── ...
├── remote-only/                          # 策略2结果
└── ...                                   # 其他策略

【使用示例】
```bash
# 运行所有6种策略（默认800轮）
python experiments/camtd3_strategy_suite/run_full_suite.py

# 快速测试（50轮）
python experiments/camtd3_strategy_suite/run_full_suite.py \\
    --episodes 50 --suite-id quick_test

# 运行部分策略
python experiments/camtd3_strategy_suite/run_full_suite.py \\
    --strategies "local-only,remote-only,comprehensive-migration" \\
    --episodes 800 --seed 42 --suite-id partial_ablation

# 只训练不绘图（后续手动绘图）
python experiments/camtd3_strategy_suite/run_full_suite.py \\
    --skip-plot --silent
```

【预计运行时间】
- 快速测试（50轮 × 6策略）：约30-60分钟
- 完整实验（800轮 × 6策略）：约8-12小时
- 建议在服务器上运行完整实验

【论文价值】
该脚本生成的对比图可直接用于论文的消融实验章节，展示：
- CAMTD3各模块的独立贡献
- 从简单基准到完整系统的性能提升路径
- 迁移管理模块的额外收益
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List

import json

from experiments.camtd3_strategy_suite.plot_strategy_comparison import (
    compute_normalized_costs,
    load_summary,
    plot_line,
)
from experiments.camtd3_strategy_suite.run_strategy_training import STRATEGY_ORDER, run_strategy


def parse_strategy_list(value: str) -> List[str]:
    """
    解析策略列表字符串
    
    【功能】
    将用户输入的逗号分隔字符串解析为策略名称列表，同时：
    - 支持"all"关键字（运行所有策略）
    - 验证策略名称的有效性
    - 保持STRATEGY_ORDER定义的顺序
    
    【参数】
    value: str - 输入字符串
        - "all" 或空字符串：返回所有策略
        - "strategy1,strategy2,..."：返回指定策略（逗号分隔）
    
    【返回值】
    List[str] - 策略名称列表（按STRATEGY_ORDER顺序排列）
    
    【错误处理】
    如果包含未知策略名称，抛出ValueError
    
    【使用示例】
    parse_strategy_list("all")  
    # -> ["local-only", "remote-only", ..., "comprehensive-migration"]
    
    parse_strategy_list("local-only,comprehensive-migration")
    # -> ["local-only", "comprehensive-migration"]  # 保持定义顺序
    
    parse_strategy_list("local-only,unknown-strategy")
    # -> ValueError: Unknown strategies: unknown-strategy
    """
    # ========== 处理"all"情况 ==========
    if not value or value.strip().lower() == "all":
        return list(STRATEGY_ORDER)
    
    # ========== 解析逗号分隔列表 ==========
    items = [item.strip().lower() for item in value.split(",") if item.strip()]
    
    # ========== 验证策略名称 ==========
    unknown = [item for item in items if item not in STRATEGY_ORDER]
    if unknown:
        raise ValueError(f"Unknown strategies: {', '.join(unknown)}")
    
    # ========== 保持原始顺序 ==========
    # 按STRATEGY_ORDER定义的顺序返回，确保结果的一致性
    return [strategy for strategy in STRATEGY_ORDER if strategy in items]


def run_suite(strategies: Iterable[str], args: argparse.Namespace) -> None:
    """
    批量执行多个策略的训练流程
    
    【功能】
    顺序执行指定的策略列表，每个策略使用相同的训练参数（episodes、seed等）。
    所有结果会汇总到同一个suite目录，便于后续对比分析。
    
    【参数】
    strategies: Iterable[str] - 要执行的策略名称列表
    args: argparse.Namespace - 命令行参数对象（包含episodes、seed等）
    
    【返回值】
    None（结果通过文件系统持久化）
    
    【执行流程】
    对于每个策略：
    1. 打印当前执行的策略名称
    2. 构建策略专属的参数对象
    3. 调用run_strategy()执行训练
    4. 结果自动追加到{suite_id}/summary.json
    
    【设计考虑】
    - 顺序执行（非并行）：避免GPU资源竞争
    - 统一参数：确保公平对比（相同episodes/seed）
    - 共享suite_id：所有策略结果归档到同一目录
    
    【错误处理】
    如果某个策略训练失败，会抛出异常中断后续策略。
    建议先用小episodes测试，确认流程正常后再运行完整实验。
    """
    for strategy in strategies:
        print(f"\n=== Running strategy: {strategy} ===")
        
        # ========== 构建策略参数 ==========
        # 使用SimpleNamespace传递参数给run_strategy
        strategy_args = SimpleNamespace(
            episodes=args.episodes,
            seed=args.seed,
            suite_id=args.suite_id,
            output_root=args.output_root,
            silent=args.silent,
        )
        
        # ========== 执行策略训练 ==========
        run_strategy(strategy, strategy_args)


def main() -> None:
    """
    脚本主入口函数
    
    【功能】
    完整的消融实验自动化流程：
    1. 解析命令行参数
    2. 批量执行策略训练
    3. 计算归一化代价
    4. 生成对比图表
    5. 输出结果摘要
    
    【执行流程】
    步骤1: 构建并解析命令行参数
    步骤2: 解析策略列表（支持"all"或指定列表）
    步骤3: 调用run_suite批量执行训练
    步骤4: 加载汇总的summary.json
    步骤5: 计算归一化代价（normalize to [0,1]）
    步骤6: 回写归一化结果到summary.json
    步骤7: 生成对比图表（折线图）
    步骤8: 打印最终结果
    
    【命令行参数】
    --strategies: str (可选)
        - 策略列表（逗号分隔）或"all"
        - 默认: "all"（运行所有6种策略）
    
    --episodes: int (可选)
        - 每个策略的训练轮数
        - 默认: 使用策略预设值（800）
    
    --seed: int (可选)
        - 随机种子（所有策略共享）
        - 默认: 42
    
    --suite-id: str (可选)
        - Suite标识符（结果目录名）
        - 默认: "CAMTD3_suite"
    
    --output-root: str (可选)
        - 输出根目录
        - 默认: "results/camtd3_strategy_suite"
    
    --silent: bool (可选)
        - 静默模式（减少训练输出）
    
    --output-name: str (可选)
        - 对比图文件名
        - 默认: "camtd3_strategy_cost.png"
    
    --skip-plot: bool (可选)
        - 跳过绘图步骤（仅训练）
        - 适合先训练，后续单独绘图
    
    【归一化代价计算】
    Normalized Cost = (Raw Cost - Min Cost) / (Max Cost - Min Cost)
    - 结果范围: [0, 1]
    - 0 = 最优策略（代价最小）
    - 1 = 最差策略（代价最大）
    
    【输出示例】
    === Suite Completed ===
    Strategies: local-only, remote-only, ..., comprehensive-migration
    local-only                  : 1.0000    # 最差（纯本地）
    remote-only                 : 0.8234
    offloading-only             : 0.6512
    resource-only               : 0.4123
    comprehensive-no-migration  : 0.2456
    comprehensive-migration     : 0.0000    # 最优（完整系统）
    Chart saved to: results/.../camtd3_strategy_cost.png
    Summary updated: results/.../summary.json
    """
    # ========== 步骤1: 构建参数解析器 ==========
    parser = argparse.ArgumentParser(
        description="Execute multiple CAMTD3 strategy runs and produce the comparison plot."
    )
    parser.add_argument(
        "--strategies",
        type=parse_strategy_list,
        default="all",
        help="Comma-separated strategy list (default: all).",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        help="Override training episodes for every strategy."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed applied to every run."
    )
    parser.add_argument(
        "--suite-id",
        type=str,
        default="CAMTD3_suite",
        help="Suite identifier used for result aggregation (default: CAMTD3_suite).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/camtd3_strategy_suite",
        help="Directory where strategy folders are stored.",
    )
    parser.add_argument(
        "--silent", 
        action="store_true", 
        help="Run underlying training in silent mode."
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="camtd3_strategy_cost.png",
        help="Filename for the generated comparison chart.",
    )
    parser.add_argument(
        "--skip-plot",
        action="store_true",
        help="Skip plotting after the runs (summary.json will still be updated).",
    )
    args = parser.parse_args()

    # ========== 步骤2: 解析策略列表 ==========
    # argparse已通过custom type自动解析，此处做兼容处理
    strategies: List[str] = (
        args.strategies 
        if isinstance(args.strategies, list) 
        else parse_strategy_list(args.strategies)
    )

    # ========== 步骤3: 批量执行策略训练 ==========
    run_suite(strategies, args)

    # ========== 步骤4: 检查是否跳过绘图 ==========
    if args.skip_plot:
        return

    # ========== 步骤5: 加载训练结果 ==========
    suite_path = Path(args.output_root) / args.suite_id
    summary_path = suite_path / "summary.json"

    summary = load_summary(summary_path)
    
    # ========== 步骤6: 计算归一化代价 ==========
    labels, values = compute_normalized_costs(summary)

    # ========== 步骤7: 回写归一化结果 ==========
    # compute_normalized_costs已经修改了summary对象，此处持久化
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), 
        encoding="utf-8"
    )

    # ========== 步骤8: 生成对比图表 ==========
    chart_path = suite_path / args.output_name
    plot_line(labels, values, chart_path)

    # ========== 步骤9: 打印最终结果摘要 ==========
    print("\n=== Suite Completed ===")
    print(f"Strategies: {', '.join(strategies)}")
    for label, value in zip(labels, values):
        print(f"{label:28s}: {value:.4f}")
    print(f"Chart saved to: {chart_path}")
    print(f"Summary updated: {summary_path}")


if __name__ == "__main__":
    main()
