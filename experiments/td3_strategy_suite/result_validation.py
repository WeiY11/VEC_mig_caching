#!/usr/bin/env python3
"""实验结果验证模块"""

from typing import Dict, List, cast
import numpy as np


def validate_experiment_results(
    results: List[Dict[str, object]],
    experiment_name: str,
) -> None:
    """验证实验结果的合理性
    
    Args:
        results: 实验结果列表
        experiment_name: 实验名称（用于报告）
    """
    from experiments.td3_strategy_suite.strategy_runner import strategy_label
    
    print("\n" + "="*80)
    print("✅ 结果验证检查")
    print("="*80)
    print(f"\n🔍 验证实验: {experiment_name}")
    print("-" * 80)
    
    # 验证1: local-only策略性能一致性
    _validate_local_only_consistency(results)
    
    # 验证2: 资源增加时性能改善
    _validate_resource_scaling(results, experiment_name)
    
    # 验证3: 高资源配置下完成率
    _validate_completion_rates(results)


def _validate_local_only_consistency(results: List[Dict[str, object]]) -> None:
    """验证local-only策略在不同配置下的一致性"""
    from experiments.td3_strategy_suite.strategy_runner import strategy_label
    
    local_only_costs = []
    for result in results:
        strategies = result.get('strategies', {})
        if not isinstance(strategies, dict):
            continue
        local_strategy = strategies.get('local-only', {})
        if isinstance(local_strategy, dict):
            cost_val = local_strategy.get('raw_cost', 0.0)
            if isinstance(cost_val, (int, float)):
                local_only_costs.append(float(cost_val))
    
    if len(local_only_costs) > 1:
        cost_std = float(np.std(local_only_costs))
        cost_mean = float(np.mean(local_only_costs))
        cv = cost_std / max(cost_mean, 1e-6)
        
        if cv < 0.1:
            print(f"  ✅ local-only 策略性能一致性: CV={cv:.3f} (< 0.1)")
        else:
            print(f"  ⚠️  local-only 策略性能变异较大: CV={cv:.3f}")


def _validate_resource_scaling(
    results: List[Dict[str, object]],
    experiment_name: str,
) -> None:
    """验证资源增加时CAMTD3性能改善"""
    from experiments.td3_strategy_suite.strategy_runner import strategy_label
    
    if "rsu" not in experiment_name.lower() and "compute" not in experiment_name.lower():
        return
    
    camtd3_costs: List[float] = []
    config_values: List[float] = []
    
    # 提取配置值字段
    config_field = None
    if "rsu" in experiment_name.lower():
        config_field = "rsu_compute_ghz"
    elif "uav" in experiment_name.lower():
        config_field = "uav_compute_ghz"
    elif "bandwidth" in experiment_name.lower():
        config_field = "bandwidth_mhz"
    
    if not config_field:
        return
    
    for result in results:
        config_val = result.get(config_field)
        if isinstance(config_val, (int, float)):
            config_values.append(float(config_val))
            
        strategies = result.get('strategies', {})
        if not isinstance(strategies, dict):
            continue
        camtd3_strategy = strategies.get('comprehensive-migration', {})
        if isinstance(camtd3_strategy, dict):
            cost_val = camtd3_strategy.get('raw_cost', 0.0)
            if isinstance(cost_val, (int, float)):
                camtd3_costs.append(float(cost_val))
    
    if len(camtd3_costs) >= 3 and len(config_values) >= 3:
        sorted_indices = np.argsort(config_values)
        sorted_costs = [camtd3_costs[i] for i in sorted_indices]
        
        # 检查是否单调递减或保持稳定
        increasing_count = sum(1 for i in range(len(sorted_costs)-1) if sorted_costs[i+1] > sorted_costs[i])
        
        if increasing_count <= 1:
            print(f"  ✅ CAMTD3 性能随资源增加而改善")
        else:
            print(f"  ⚠️  CAMTD3 性能未能随资源一致改善 (上升{increasing_count}次)")


def _validate_completion_rates(results: List[Dict[str, object]]) -> None:
    """验证高资源配置下的任务完成率"""
    from experiments.td3_strategy_suite.strategy_runner import strategy_label
    
    if len(results) == 0:
        return
    
    last_config = results[-1]
    strategies = last_config.get('strategies', {})
    
    if isinstance(strategies, dict):
        low_completion_strategies: List[tuple] = []
        for key, metrics_obj in strategies.items():
            if not isinstance(metrics_obj, dict):
                continue
            completion_val = metrics_obj.get('completion_rate', 0.0)
            if isinstance(completion_val, (int, float)):
                completion = float(completion_val)
                if completion < 0.95:
                    low_completion_strategies.append((str(key), completion))
        
        if not low_completion_strategies:
            print(f"  ✅ 高资源配置下所有策略完成率 ≥ 95%")
        else:
            print(f"  ⚠️  以下策略完成率较低:")
            for key, completion in low_completion_strategies:
                print(f"      - {strategy_label(key)}: {completion:.2%}")
