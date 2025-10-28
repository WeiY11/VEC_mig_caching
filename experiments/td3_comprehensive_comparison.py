#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TD3综合对比试验框架
为CAM-TD3算法设计的完整实验方案，符合顶级会议/期刊标准

实验维度：
1. 算法对比（Baseline Comparison）：与DRL、启发式、元启发式算法对比
2. 消融实验（Ablation Study）：验证各模块有效性
3. 参数敏感性（Parameter Sensitivity）：分析关键参数影响
4. 鲁棒性测试（Robustness Test）：极端场景下的性能
5. 收敛性分析（Convergence Analysis）：训练稳定性评估
6. 可扩展性测试（Scalability Test）：大规模场景性能

论文对应：
- 算法对比 → Section 5.1: Performance Comparison
- 消融实验 → Section 5.2: Ablation Study
- 参数敏感性 → Section 5.3: Parameter Analysis
- 鲁棒性测试 → Section 5.4: Robustness Evaluation
- 收敛性分析 → Section 5.5: Convergence Study
- 可扩展性 → Section 5.6: Scalability Analysis

用途：
- 期刊/会议级的完整实验套件，包括：算法对比、消融、参数敏感性、鲁棒性、收敛性、可扩展性。
- 自动组织并保存结果，产出论文可用的数据与摘要。

运行命令：
- 完整套件（快速）:   python run_td3_comparison.py --mode quick --dimension all
- 完整套件（标准）:   python run_td3_comparison.py --mode standard --dimension all
- 仅某一维度:         python run_td3_comparison.py --mode standard --dimension ablation|sensitivity|robustness|convergence|scalability
"""

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict

# ============================================================
# 实验配置数据类
# ============================================================

@dataclass
class TD3ExperimentConfig:
    """TD3实验配置"""
    name: str
    description: str
    episodes: int = 200
    seeds: List[int] = field(default_factory=lambda: [42, 2025, 3407])
    num_vehicles: int = 12
    num_rsus: int = 4
    num_uavs: int = 2
    max_steps: int = 200
    
    # 消融控制
    enable_cache: bool = True
    enable_migration: bool = True
    enable_collaborative_cache: bool = True
    enable_priority: bool = True
    
    # 场景配置
    bandwidth: float = 20.0  # MHz
    task_arrival_rate: float = 0.5  # tasks/step
    task_size_range: Tuple[float, float] = (1.0, 5.0)  # MB
    
    # 其他参数
    extra_params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return asdict(self)


@dataclass
class ExperimentResult:
    """实验结果"""
    config_name: str
    algorithm: str
    seeds: List[int]
    episodes: int
    
    # 性能指标（均值 ± 标准差）
    avg_reward: Tuple[float, float]  # (mean, std)
    avg_delay: Tuple[float, float]
    avg_energy: Tuple[float, float]
    task_completion_rate: Tuple[float, float]
    cache_hit_rate: Tuple[float, float]
    migration_success_rate: Tuple[float, float]
    
    # 额外统计
    convergence_episode: Optional[int] = None
    training_time_hours: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


# ============================================================
# 实验套件定义
# ============================================================

class TD3ComprehensiveComparison:
    """TD3综合对比实验执行器"""
    
    def __init__(self, output_dir: str = "results/td3_comprehensive"):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_dir = self.output_dir / self.timestamp
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: Dict[str, Dict[str, Any]] = {}
        
    # ========================================================
    # 维度1: 算法对比实验
    # ========================================================
    
    def define_algorithm_comparison(self) -> List[Dict[str, Any]]:
        """
        定义算法对比实验
        
        对比组：
        1. DRL算法组：TD3, DDPG, SAC, PPO, DQN
        2. 启发式算法组：Random, Greedy, RoundRobin, LoadBalanced, NearestNode, LocalFirst
        3. 元启发式算法组：GA, PSO, SimulatedAnnealing
        
        评估指标：
        - 时延（主要）、能耗（主要）
        - 任务完成率、缓存命中率、迁移成功率
        """
        algorithms = []
        
        # ===== DRL算法组 =====
        drl_algorithms = ["TD3", "DDPG", "SAC", "PPO", "DQN"]
        for alg in drl_algorithms:
            algorithms.append({
                "name": alg,
                "label": f"CAM-{alg}" if alg == "TD3" else alg,
                "category": "drl",
                "episodes": 800,  # 充分训练
                "seeds": [42, 2025, 3407],  # 3个随机种子
                "params": {}
            })
        
        # ===== 启发式算法组 =====
        heuristic_algorithms = [
            ("Random", "随机选择"),
            ("Greedy", "贪心最小负载"),
            ("RoundRobin", "轮询分配"),
            ("LoadBalanced", "负载均衡"),
            ("NearestNode", "最近节点"),
            ("LocalFirst", "本地优先")
        ]
        for alg_name, desc in heuristic_algorithms:
            algorithms.append({
                "name": alg_name,
                "label": alg_name,
                "category": "heuristic",
                "episodes": 200,  # 启发式算法不需要训练，但需要评估
                "seeds": [42, 2025, 3407],
                "params": {},
                "description": desc
            })
        
        # ===== 元启发式算法组 =====
        meta_algorithms = [
            ("GA", "遗传算法", {"population_size": 50, "generations": 100}),
            ("PSO", "粒子群算法", {"swarm_size": 40, "iterations": 100}),
            ("SimulatedAnnealing", "模拟退火", {"initial_temp": 1000, "cooling_rate": 0.95})
        ]
        for alg_name, desc, params in meta_algorithms:
            algorithms.append({
                "name": alg_name,
                "label": alg_name,
                "category": "meta",
                "episodes": 200,
                "seeds": [42, 2025, 3407],
                "params": params,
                "description": desc
            })
        
        return algorithms
    
    # ========================================================
    # 维度2: 消融实验
    # ========================================================
    
    def define_ablation_study(self) -> List[TD3ExperimentConfig]:
        """
        定义消融实验配置
        
        目的：验证每个模块对系统性能的贡献
        
        配置组：
        1. Full-System: 完整系统（CAM-TD3）- 基准
        2. No-Cache: 禁用边缘缓存模块
        3. No-Migration: 禁用任务迁移模块
        4. No-Collaborative-Cache: 禁用协作缓存（RSU间）
        5. No-Priority: 禁用优先级队列
        6. Basic-TD3: 基础TD3（无缓存、无迁移）
        7. Minimal-System: 最小系统（所有优化模块禁用）
        """
        configs = []
        
        # 1. 完整系统（基准）
        configs.append(TD3ExperimentConfig(
            name="Full-System",
            description="完整CAM-TD3系统（所有模块启用）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],  # 5个种子，更可靠
            enable_cache=True,
            enable_migration=True,
            enable_collaborative_cache=True,
            enable_priority=True
        ))
        
        # 2. 无缓存
        configs.append(TD3ExperimentConfig(
            name="No-Cache",
            description="禁用边缘缓存模块（验证缓存有效性）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=False,
            enable_migration=True,
            enable_collaborative_cache=False,  # 缓存禁用，协作缓存也无效
            enable_priority=True
        ))
        
        # 3. 无迁移
        configs.append(TD3ExperimentConfig(
            name="No-Migration",
            description="禁用任务迁移模块（验证迁移有效性）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=True,
            enable_migration=False,
            enable_collaborative_cache=True,
            enable_priority=True
        ))
        
        # 4. 无协作缓存
        configs.append(TD3ExperimentConfig(
            name="No-Collaborative-Cache",
            description="禁用RSU间协作缓存（验证协作有效性）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=True,
            enable_migration=True,
            enable_collaborative_cache=False,
            enable_priority=True
        ))
        
        # 5. 无优先级队列
        configs.append(TD3ExperimentConfig(
            name="No-Priority",
            description="禁用任务优先级队列（FIFO队列）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=True,
            enable_migration=True,
            enable_collaborative_cache=True,
            enable_priority=False
        ))
        
        # 6. 基础TD3（无缓存、无迁移）
        configs.append(TD3ExperimentConfig(
            name="Basic-TD3",
            description="基础TD3（仅卸载决策，无缓存迁移）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=False,
            enable_migration=False,
            enable_collaborative_cache=False,
            enable_priority=True
        ))
        
        # 7. 最小系统
        configs.append(TD3ExperimentConfig(
            name="Minimal-System",
            description="最小系统（所有优化模块禁用）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890],
            enable_cache=False,
            enable_migration=False,
            enable_collaborative_cache=False,
            enable_priority=False
        ))
        
        return configs
    
    # ========================================================
    # 维度3: 参数敏感性分析
    # ========================================================
    
    def define_parameter_sensitivity(self) -> Dict[str, List[TD3ExperimentConfig]]:
        """
        定义参数敏感性分析实验
        
        目的：分析关键参数对系统性能的影响
        
        参数维度：
        1. 车辆规模 (num_vehicles): 4, 8, 12, 16, 20, 24, 30
        2. RSU密度 (num_rsus): 2, 4, 6, 8, 10
        3. UAV数量 (num_uavs): 0, 1, 2, 3, 4
        4. 带宽水平 (bandwidth): 10, 15, 20, 25, 30 MHz
        5. 任务到达率 (task_arrival_rate): 0.2, 0.4, 0.6, 0.8, 1.0
        6. 任务规模 (task_size): 小(0.5-2MB), 中(1-5MB), 大(3-10MB)
        """
        sensitivity_experiments = {}
        
        # ===== 1. 车辆规模敏感性 =====
        vehicle_counts = [4, 8, 12, 16, 20, 24, 30]
        sensitivity_experiments["vehicle_scaling"] = []
        for num_vehicles in vehicle_counts:
            sensitivity_experiments["vehicle_scaling"].append(TD3ExperimentConfig(
                name=f"TD3_vehicles_{num_vehicles}",
                description=f"车辆数量: {num_vehicles}",
                episodes=400,  # 参数分析可以适当减少轮次
                seeds=[42, 2025, 3407],
                num_vehicles=num_vehicles,
                num_rsus=4,
                num_uavs=2
            ))
        
        # ===== 2. RSU密度敏感性 =====
        rsu_counts = [2, 4, 6, 8, 10]
        sensitivity_experiments["rsu_density"] = []
        for num_rsus in rsu_counts:
            sensitivity_experiments["rsu_density"].append(TD3ExperimentConfig(
                name=f"TD3_rsus_{num_rsus}",
                description=f"RSU数量: {num_rsus}",
                episodes=400,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                num_rsus=num_rsus,
                num_uavs=2
            ))
        
        # ===== 3. UAV数量敏感性 =====
        uav_counts = [0, 1, 2, 3, 4]
        sensitivity_experiments["uav_count"] = []
        for num_uavs in uav_counts:
            sensitivity_experiments["uav_count"].append(TD3ExperimentConfig(
                name=f"TD3_uavs_{num_uavs}",
                description=f"UAV数量: {num_uavs}",
                episodes=400,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                num_rsus=4,
                num_uavs=num_uavs
            ))
        
        # ===== 4. 带宽水平敏感性 =====
        bandwidth_levels = [10, 15, 20, 25, 30]  # MHz
        sensitivity_experiments["bandwidth"] = []
        for bw in bandwidth_levels:
            sensitivity_experiments["bandwidth"].append(TD3ExperimentConfig(
                name=f"TD3_bw_{bw}MHz",
                description=f"带宽: {bw} MHz",
                episodes=400,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                num_rsus=4,
                num_uavs=2,
                bandwidth=float(bw)
            ))
        
        # ===== 5. 任务到达率敏感性 =====
        arrival_rates = [0.2, 0.4, 0.6, 0.8, 1.0]
        sensitivity_experiments["task_arrival_rate"] = []
        for rate in arrival_rates:
            sensitivity_experiments["task_arrival_rate"].append(TD3ExperimentConfig(
                name=f"TD3_arrival_{rate:.1f}",
                description=f"任务到达率: {rate} tasks/step",
                episodes=400,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                task_arrival_rate=rate
            ))
        
        # ===== 6. 任务规模敏感性 =====
        task_sizes = [
            ("small", (0.5, 2.0), "小任务(0.5-2MB)"),
            ("medium", (1.0, 5.0), "中任务(1-5MB)"),
            ("large", (3.0, 10.0), "大任务(3-10MB)")
        ]
        sensitivity_experiments["task_size"] = []
        for size_name, size_range, desc in task_sizes:
            sensitivity_experiments["task_size"].append(TD3ExperimentConfig(
                name=f"TD3_tasksize_{size_name}",
                description=desc,
                episodes=400,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                task_size_range=size_range
            ))
        
        return sensitivity_experiments
    
    # ========================================================
    # 维度4: 鲁棒性测试
    # ========================================================
    
    def define_robustness_tests(self) -> List[TD3ExperimentConfig]:
        """
        定义鲁棒性测试实验
        
        目的：验证算法在极端/异常场景下的表现
        
        场景：
        1. 极端高负载：车辆30辆 + 高任务到达率
        2. 极端低带宽：带宽5MHz（拥塞场景）
        3. 高移动性：车辆高速移动（频繁切换连接）
        4. RSU失效：部分RSU随机失效
        5. 动态拓扑：拓扑结构动态变化
        6. 突发流量：任务突发到达
        """
        configs = []
        
        # 1. 极端高负载
        configs.append(TD3ExperimentConfig(
            name="Extreme-High-Load",
            description="极端高负载场景（30车辆+高任务率）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=30,
            num_rsus=6,
            num_uavs=3,
            task_arrival_rate=1.2,
            extra_params={"scenario": "high_load"}
        ))
        
        # 2. 极端低带宽
        configs.append(TD3ExperimentConfig(
            name="Extreme-Low-Bandwidth",
            description="极端低带宽场景（5MHz拥塞）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=16,
            num_rsus=4,
            num_uavs=2,
            bandwidth=5.0,
            extra_params={"scenario": "low_bandwidth"}
        ))
        
        # 3. 高移动性
        configs.append(TD3ExperimentConfig(
            name="High-Mobility",
            description="高移动性场景（车辆高速120km/h+）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=12,
            num_rsus=6,  # 需要更多RSU覆盖
            num_uavs=2,
            extra_params={
                "scenario": "high_mobility",
                "vehicle_speed_range": (80, 140)  # km/h
            }
        ))
        
        # 4. RSU失效
        configs.append(TD3ExperimentConfig(
            name="RSU-Failure",
            description="RSU失效场景（随机RSU失效30%概率）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=12,
            num_rsus=6,
            num_uavs=2,
            extra_params={
                "scenario": "rsu_failure",
                "failure_probability": 0.3
            }
        ))
        
        # 5. 动态拓扑
        configs.append(TD3ExperimentConfig(
            name="Dynamic-Topology",
            description="动态拓扑场景（车辆进出、节点变化）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2,
            extra_params={
                "scenario": "dynamic_topology",
                "vehicle_join_leave": True
            }
        ))
        
        # 6. 突发流量
        configs.append(TD3ExperimentConfig(
            name="Bursty-Traffic",
            description="突发流量场景（任务突发到达）",
            episodes=500,
            seeds=[42, 2025, 3407],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2,
            extra_params={
                "scenario": "bursty_traffic",
                "burst_interval": 50,  # 每50步突发一次
                "burst_size": 20  # 突发20个任务
            }
        ))
        
        return configs
    
    # ========================================================
    # 维度5: 收敛性分析
    # ========================================================
    
    def define_convergence_analysis(self) -> List[TD3ExperimentConfig]:
        """
        定义收敛性分析实验
        
        目的：分析TD3算法的收敛速度和稳定性
        
        实验：
        1. 多随机种子实验（10个种子）：评估收敛一致性
        2. 长期训练实验（1500轮）：观察长期稳定性
        3. 不同学习率实验：分析学习率对收敛的影响
        """
        configs = []
        
        # 1. 多随机种子（10个）
        configs.append(TD3ExperimentConfig(
            name="Convergence-MultiSeed",
            description="多随机种子收敛性分析（10种子）",
            episodes=800,
            seeds=[42, 2025, 3407, 12345, 67890, 11111, 22222, 33333, 44444, 55555],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2
        ))
        
        # 2. 长期训练
        configs.append(TD3ExperimentConfig(
            name="Convergence-Long-Term",
            description="长期训练收敛性分析（1500轮）",
            episodes=1500,
            seeds=[42, 2025, 3407],
            num_vehicles=12,
            num_rsus=4,
            num_uavs=2
        ))
        
        # 3. 不同学习率
        learning_rates = [1e-4, 3e-4, 5e-4, 1e-3]
        for lr in learning_rates:
            configs.append(TD3ExperimentConfig(
                name=f"Convergence-LR-{lr:.0e}",
                description=f"学习率{lr}的收敛性",
                episodes=800,
                seeds=[42, 2025, 3407],
                num_vehicles=12,
                num_rsus=4,
                num_uavs=2,
                extra_params={"learning_rate": lr}
            ))
        
        return configs
    
    # ========================================================
    # 维度6: 可扩展性测试
    # ========================================================
    
    def define_scalability_tests(self) -> List[TD3ExperimentConfig]:
        """
        定义可扩展性测试实验
        
        目的：验证算法在大规模场景下的性能
        
        规模：
        1. 小规模：5车 + 2RSU + 1UAV
        2. 中规模：12车 + 4RSU + 2UAV（标准）
        3. 大规模：30车 + 8RSU + 4UAV
        4. 超大规模：50车 + 12RSU + 6UAV
        5. 极限规模：100车 + 20RSU + 10UAV
        """
        scales = [
            ("Small", 5, 2, 1, "小规模"),
            ("Medium", 12, 4, 2, "中规模（标准）"),
            ("Large", 30, 8, 4, "大规模"),
            ("XLarge", 50, 12, 6, "超大规模"),
            ("XXLarge", 100, 20, 10, "极限规模")
        ]
        
        configs = []
        for scale_name, num_v, num_r, num_u, desc in scales:
            configs.append(TD3ExperimentConfig(
                name=f"Scalability-{scale_name}",
                description=f"{desc}: {num_v}车+{num_r}RSU+{num_u}UAV",
                episodes=500 if num_v <= 30 else 300,  # 大规模减少轮次
                seeds=[42, 2025, 3407],
                num_vehicles=num_v,
                num_rsus=num_r,
                num_uavs=num_u
            ))
        
        return configs
    
    # ========================================================
    # 实验执行核心
    # ========================================================
    
    def run_experiment(self, config: TD3ExperimentConfig, algorithm: str = "TD3") -> Dict[str, Any]:
        """
        运行单个实验配置
        
        参数：
            config: 实验配置
            algorithm: 算法名称
        
        返回：
            实验结果字典
        """
        from train_single_agent import train_single_algorithm
        
        print(f"\n{'='*80}")
        print(f"实验: {config.name}")
        print(f"描述: {config.description}")
        print(f"算法: {algorithm}")
        print(f"轮次: {config.episodes}")
        print(f"种子: {config.seeds}")
        print(f"{'='*80}\n")
        
        # 准备场景覆盖配置
        scenario_overrides = {
            "num_vehicles": config.num_vehicles,
            "num_rsus": config.num_rsus,
            "num_uavs": config.num_uavs,
            "max_steps_per_episode": config.max_steps,
            "override_topology": True
        }
        
        # 添加额外参数
        if config.bandwidth != 20.0:
            scenario_overrides["bandwidth"] = config.bandwidth
        
        scenario_overrides.update(config.extra_params)
        
        # 多种子实验
        seed_results = []
        for seed in config.seeds:
            print(f"  → 运行种子: {seed}")
            start_time = time.time()
            
            # 设置随机种子
            import random
            random.seed(seed)
            np.random.seed(seed)
            try:
                import torch
                torch.manual_seed(seed)
            except ImportError:
                pass
            
            # 运行训练
            result = train_single_algorithm(
                algorithm,
                num_episodes=config.episodes,
                silent_mode=True,
                override_scenario=scenario_overrides,
                use_enhanced_cache=config.enable_cache,
                disable_migration=(not config.enable_migration)
            )
            
            elapsed_time = time.time() - start_time
            
            # 提取指标（后20%稳定期）
            stable_start = int(config.episodes * 0.8)
            episode_rewards = result.get("episode_rewards", [])
            episode_metrics = result.get("episode_metrics", {})
            
            seed_result = {
                "seed": seed,
                "training_time_hours": elapsed_time / 3600.0,
                "avg_reward": np.mean(episode_rewards[stable_start:]) if episode_rewards else 0,
                "avg_delay": np.mean(episode_metrics.get("avg_delay", [])[stable_start:]) if episode_metrics.get("avg_delay") else 0,
                "avg_energy": np.mean(episode_metrics.get("total_energy", [])[stable_start:]) if episode_metrics.get("total_energy") else 0,
                "task_completion_rate": np.mean(episode_metrics.get("task_completion_rate", [])[stable_start:]) if episode_metrics.get("task_completion_rate") else 0,
                "cache_hit_rate": np.mean(episode_metrics.get("cache_hit_rate", [])[stable_start:]) if episode_metrics.get("cache_hit_rate") else 0,
                "migration_success_rate": np.mean(episode_metrics.get("migration_success_rate", [])[stable_start:]) if episode_metrics.get("migration_success_rate") else 0,
            }
            
            seed_results.append(seed_result)
            print(f"     完成 - 奖励: {seed_result['avg_reward']:.3f}, 时延: {seed_result['avg_delay']:.3f}s")
        
        # 聚合多种子结果
        aggregated = self._aggregate_seed_results(seed_results)
        aggregated["config"] = config.to_dict()
        aggregated["algorithm"] = algorithm
        
        # 保存结果
        result_file = self.experiment_dir / f"{config.name}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(aggregated, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 结果已保存: {result_file}")
        
        return aggregated
    
    def _aggregate_seed_results(self, seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多个种子的实验结果"""
        metrics = ["avg_reward", "avg_delay", "avg_energy", "task_completion_rate", 
                   "cache_hit_rate", "migration_success_rate", "training_time_hours"]
        
        aggregated = {
            "num_seeds": len(seed_results),
            "seeds": [r["seed"] for r in seed_results]
        }
        
        for metric in metrics:
            values = [r[metric] for r in seed_results if r.get(metric) is not None]
            if values:
                aggregated[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "values": values
                }
            else:
                aggregated[metric] = None
        
        return aggregated
    
    # ========================================================
    # 完整实验套件运行
    # ========================================================
    
    def run_full_suite(self, mode: str = "quick"):
        """
        运行完整实验套件
        
        参数：
            mode: 实验模式
                - "quick": 快速测试（减少轮次和种子）
                - "standard": 标准实验（论文标准配置）
                - "extensive": 扩展实验（最全面）
        """
        print("\n" + "="*80)
        print("🔬 TD3综合对比实验套件")
        print("="*80)
        print(f"模式: {mode.upper()}")
        print(f"输出目录: {self.experiment_dir}")
        print("="*80 + "\n")
        
        # 根据模式调整参数
        if mode == "quick":
            episode_factor = 0.25
            seed_count = 1
        elif mode == "standard":
            episode_factor = 1.0
            seed_count = 3
        else:  # extensive
            episode_factor = 1.5
            seed_count = 5
        
        # 保存实验配置
        suite_config = {
            "mode": mode,
            "timestamp": self.timestamp,
            "episode_factor": episode_factor,
            "seed_count": seed_count
        }
        
        config_file = self.experiment_dir / "suite_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(suite_config, f, indent=2)
        
        # 1. 算法对比
        print("\n" + "="*80)
        print("📊 维度1: 算法对比实验")
        print("="*80)
        # （这里可以调用算法对比执行）
        
        # 2. 消融实验
        print("\n" + "="*80)
        print("🔍 维度2: 消融实验")
        print("="*80)
        ablation_configs = self.define_ablation_study()
        for config in ablation_configs:
            # 根据模式调整
            config.episodes = int(config.episodes * episode_factor)
            config.seeds = config.seeds[:seed_count]
            
            result = self.run_experiment(config, algorithm="TD3")
            self.results[f"ablation_{config.name}"] = result
        
        # 3. 参数敏感性（选择部分维度）
        print("\n" + "="*80)
        print("📈 维度3: 参数敏感性分析")
        print("="*80)
        sensitivity_experiments = self.define_parameter_sensitivity()
        
        # 选择关键维度（车辆规模 + 带宽）
        for dim_name in ["vehicle_scaling", "bandwidth"]:
            print(f"\n→ 参数维度: {dim_name}")
            for config in sensitivity_experiments[dim_name]:
                config.episodes = int(config.episodes * episode_factor)
                config.seeds = config.seeds[:seed_count]
                
                result = self.run_experiment(config, algorithm="TD3")
                self.results[f"sensitivity_{config.name}"] = result
        
        # 4. 鲁棒性测试（选择2个场景）
        print("\n" + "="*80)
        print("🛡️ 维度4: 鲁棒性测试")
        print("="*80)
        robustness_configs = self.define_robustness_tests()
        for config in robustness_configs[:2]:  # 先运行前2个
            config.episodes = int(config.episodes * episode_factor)
            config.seeds = config.seeds[:seed_count]
            
            result = self.run_experiment(config, algorithm="TD3")
            self.results[f"robustness_{config.name}"] = result
        
        # 5. 收敛性分析
        if mode in ["standard", "extensive"]:
            print("\n" + "="*80)
            print("📉 维度5: 收敛性分析")
            print("="*80)
            convergence_configs = self.define_convergence_analysis()
            for config in convergence_configs[:1]:  # 多种子实验
                result = self.run_experiment(config, algorithm="TD3")
                self.results[f"convergence_{config.name}"] = result
        
        # 6. 可扩展性测试
        if mode == "extensive":
            print("\n" + "="*80)
            print("📏 维度6: 可扩展性测试")
            print("="*80)
            scalability_configs = self.define_scalability_tests()
            for config in scalability_configs:
                result = self.run_experiment(config, algorithm="TD3")
                self.results[f"scalability_{config.name}"] = result
        
        # 保存总结
        self._save_summary()
        
        print("\n" + "="*80)
        print("✅ TD3综合对比实验套件完成！")
        print(f"结果保存在: {self.experiment_dir}")
        print("="*80 + "\n")
    
    def _save_summary(self):
        """保存实验总结"""
        summary = {
            "timestamp": self.timestamp,
            "total_experiments": len(self.results),
            "results_overview": {}
        }
        
        for exp_name, result in self.results.items():
            summary["results_overview"][exp_name] = {
                "avg_reward": result.get("avg_reward", {}).get("mean"),
                "avg_delay": result.get("avg_delay", {}).get("mean"),
                "avg_energy": result.get("avg_energy", {}).get("mean"),
                "task_completion_rate": result.get("task_completion_rate", {}).get("mean")
            }
        
        summary_file = self.experiment_dir / "experiment_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ 实验总结已保存: {summary_file}")


# ============================================================
# 命令行入口
# ============================================================

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="TD3综合对比实验框架")
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["quick", "standard", "extensive"],
                       help="实验模式：quick(快速测试), standard(标准), extensive(扩展)")
    parser.add_argument("--dimension", type=str, default="all",
                       choices=["all", "algorithm", "ablation", "sensitivity", 
                               "robustness", "convergence", "scalability"],
                       help="实验维度选择")
    parser.add_argument("--output-dir", type=str, default="results/td3_comprehensive",
                       help="输出目录")
    
    args = parser.parse_args()
    
    # 创建实验执行器
    runner = TD3ComprehensiveComparison(output_dir=args.output_dir)
    
    # 运行实验
    if args.dimension == "all":
        runner.run_full_suite(mode=args.mode)
    else:
        # 单独运行某个维度
        print(f"运行单个维度: {args.dimension}")
        # TODO: 实现单独维度运行
    
    print("\n🎉 实验完成！")


if __name__ == "__main__":
    main()

