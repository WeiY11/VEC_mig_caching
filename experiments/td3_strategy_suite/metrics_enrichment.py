#!/usr/bin/env python3
"""统一的指标增强模块"""

from typing import Dict, List
import numpy as np


def enrich_strategy_metrics(
    strategy_key: str,
    metrics: Dict[str, float],
    config: Dict[str, object],
    episode_metrics: Dict[str, List[float]],
) -> None:
    """增强策略指标，添加吞吐量、利用率、稳定性等关键指标
    
    Args:
        strategy_key: 策略标识
        metrics: 基础指标字典（会被就地修改）
        config: 配置信息
        episode_metrics: 每轮次的详细指标
    """
    # 吞吐量计算
    throughput_series = episode_metrics.get("throughput_mbps") or episode_metrics.get("avg_throughput_mbps")
    avg_throughput = 0.0
    if throughput_series:
        values = list(map(float, throughput_series))
        if values:
            half = values[len(values) // 2 :] if len(values) >= 100 else values
            avg_throughput = float(sum(half) / max(len(half), 1))

    if avg_throughput <= 0:
        avg_task_size_mb = 0.35
        num_tasks_per_step = int(config.get("assumed_tasks_per_step", 12))
        avg_delay = metrics.get("avg_delay", 0.0)
        if avg_delay > 0:
            avg_throughput = (avg_task_size_mb * num_tasks_per_step) / avg_delay

    metrics["avg_throughput_mbps"] = max(avg_throughput, 0.0)
    
    # RSU利用率
    rsu_util_series = episode_metrics.get("rsu_utilization") or episode_metrics.get("avg_rsu_utilization")
    if rsu_util_series:
        values = list(map(float, rsu_util_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_rsu_utilization"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_rsu_utilization"] = 0.0
    
    # 卸载率
    offload_series = episode_metrics.get("offload_ratio") or episode_metrics.get("remote_execution_ratio")
    if offload_series:
        values = list(map(float, offload_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_offload_ratio"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_offload_ratio"] = 0.0
    
    # 队列长度
    queue_series = episode_metrics.get("queue_rho_mean") or episode_metrics.get("avg_queue_length")
    if queue_series:
        values = list(map(float, queue_series))
        if values:
            half = values[len(values) // 2:] if len(values) >= 100 else values
            metrics["avg_queue_length"] = float(sum(half) / max(len(half), 1))
    else:
        metrics["avg_queue_length"] = 0.0
    
    # 性能稳定性（时延标准差和变异系数）
    delay_series = episode_metrics.get("avg_delay")
    if delay_series:
        values = list(map(float, delay_series))
        if len(values) >= 100:
            half = values[len(values) // 2:]
            if half:
                metrics["delay_std"] = float(np.std(half))
                metrics["delay_cv"] = float(np.std(half) / max(np.mean(half), 1e-6))
    
    # 资源利用效率（任务完成率 / 能耗）
    completion_rate = metrics.get("completion_rate", 0.0)
    avg_energy = metrics.get("avg_energy", 1.0)
    if avg_energy > 0:
        metrics["resource_efficiency"] = completion_rate / avg_energy * 1000


def print_metrics_comparison_table(
    results: List[Dict[str, object]],
    strategy_keys: List[str],
    axis_field: str,
    axis_label: str,
) -> None:
    """打印关键指标对比表
    
    Args:
        results: 实验结果列表
        strategy_keys: 策略列表
        axis_field: X轴字段名
        axis_label: X轴标签
    """
    from experiments.td3_strategy_suite.strategy_runner import strategy_label
    
    print("\n" + "="*80)
    print("📊 关键指标对比 (RSU利用率 | 卸载率 | 队列长度)")
    print("="*80)
    
    for record in results:
        axis_value = record.get(axis_field, record.get("label", "N/A"))
        if isinstance(axis_value, float):
            config_label = f"{axis_value:.1f}"
        else:
            config_label = str(axis_value)
        print(f"\n配置: {config_label}")
        print("-" * 80)
        
        for strat_key in strategy_keys:
            strategies_dict = record.get("strategies", {})
            if not isinstance(strategies_dict, dict):
                continue
            strat_dict = strategies_dict.get(strat_key, {})
            if not isinstance(strat_dict, dict):
                continue
            
            rsu_util = strat_dict.get("avg_rsu_utilization", 0.0)
            offload = strat_dict.get("avg_offload_ratio", 0.0)
            queue = strat_dict.get("avg_queue_length", 0.0)
            
            label = strategy_label(strat_key)
            print(f"  {label:40s} | RSU: {rsu_util:5.2f} | Offload: {offload:5.2f} | Queue: {queue:6.3f}")
