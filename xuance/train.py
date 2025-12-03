#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XuanCe风格训练脚本 - VEC边缘计算系统

本脚本采用XuanCe框架的标准训练流程，支持：
- YAML配置文件管理
- 命令行参数覆盖
- 多种DRL算法（TD3, SAC, PPO, DDPG）
- TensorBoard/WandB可视化
- 模型保存与加载

使用方式：
    # 使用默认OPTIMIZED_TD3配置训练
    python xuance/train.py
    
    # 指定算法和配置文件
    python xuance/train.py --method td3 --config xuance/configs/td3_vec.yaml
    
    # 命令行覆盖参数
    python xuance/train.py --method sac --episodes 500 --device cuda:0
    
    # 运行对比方案
    python xuance/train.py --method local --episodes 50

作者: VEC Team
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# 设置Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))  # 添加项目根目录

# 导入YAML解析
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠️ PyYAML未安装，使用默认配置")

# 导入PyTorch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("❌ PyTorch未安装，无法运行")
    sys.exit(1)

# 导入XuanCe
try:
    import xuance
    from xuance.common import get_configs, recursive_dict_update
    from xuance.environment import make_envs
    from xuance.torch.utils.operations import set_seed
    HAS_XUANCE = True
except ImportError:
    HAS_XUANCE = False
    print("⚠️ XuanCe未安装，将使用本地实现")

# 导入VEC环境
try:
    from xuance.vec_env import VECEnv, register_vec_env
except ImportError:
    # 允许直接运行
    from vec_env import VECEnv, register_vec_env


def parse_args() -> argparse.Namespace:
    """解析命令行参数（XuanCe风格）"""
    parser = argparse.ArgumentParser(
        description="VEC边缘计算系统 - XuanCe训练脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 算法选择 - 支持所有原有算法和对比方案
    parser.add_argument("--method", "-m", type=str, default="optimized_td3",
                        choices=[
                            # DRL算法
                            "td3", "sac", "ppo", "ddpg", "dqn",
                            "optimized_td3", "cam_td3", "td3_le", "td3-le",
                            # 对比方案 (Benchmarks)
                            "local", "heuristic", "sa",
                            "benchmark_td3", "benchmark_ddpg", "benchmark_sac"
                        ],
                        help="训练算法: DRL(optimized_td3,td3,sac,ppo,ddpg,dqn,cam_td3,td3_le) 或 Baseline(local,heuristic,sa,benchmark_*)")
    
    # 配置文件
    parser.add_argument("--config", "-c", type=str, default=None,
                        help="YAML配置文件路径")
    
    # 环境配置
    parser.add_argument("--env-name", type=str, default="VEC",
                        help="环境名称")
    parser.add_argument("--env-id", type=str, default="VEC-v1",
                        help="环境ID")
    
    # VEC特定参数
    parser.add_argument("--num-vehicles", type=int, default=None,
                        help="车辆数量")
    parser.add_argument("--num-rsus", type=int, default=None,
                        help="RSU数量")
    parser.add_argument("--num-uavs", type=int, default=None,
                        help="UAV数量")
    parser.add_argument("--arrival-rate", type=float, default=None,
                        help="任务到达率")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=None,
                        help="训练轮次（自动转换为running_steps）")
    parser.add_argument("--max-steps", type=int, default=200,
                        help="每轮最大步数")
    parser.add_argument("--running-steps", type=int, default=None,
                        help="总训练步数")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # 硬件配置
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="计算设备")
    parser.add_argument("--parallels", type=int, default=1,
                        help="并行环境数")
    
    # 评估与日志
    parser.add_argument("--eval-interval", type=int, default=10000,
                        help="评估间隔（步数）")
    parser.add_argument("--test-episode", type=int, default=5,
                        help="测试轮次")
    parser.add_argument("--logger", type=str, default="tensorboard",
                        choices=["tensorboard", "wandb"],
                        help="日志工具")
    
    # 模式选择
    parser.add_argument("--test", action="store_true",
                        help="测试模式")
    parser.add_argument("--benchmark", action="store_true",
                        help="基准测试模式")
    parser.add_argument("--model-path", type=str, default=None,
                        help="模型加载路径")
    
    # 输出配置
    parser.add_argument("--log-dir", type=str, default="./logs/",
                        help="日志目录")
    parser.add_argument("--model-dir", type=str, default="./models/",
                        help="模型保存目录")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="详细输出")
    
    return parser.parse_args()


def get_default_config(method: str) -> Dict[str, Any]:
    """获取默认配置"""
    base_config = {
        "dl_toolbox": "torch",
        "project_name": "VEC_Edge_Computing",
        "logger": "tensorboard",
        "render": False,
        "render_mode": "rgb_array",
        "test_mode": False,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "distributed_training": False,
        
        "env_name": "VEC",
        "env_id": "VEC-v1",
        "env_seed": 42,
        "vectorize": "DummyVecEnv",
        
        "representation": "Basic_MLP",
        "representation_hidden_size": [256, 256],
        "actor_hidden_size": [256, 256],
        "critic_hidden_size": [256, 256],
        "activation": "relu",
        "activation_action": "tanh",
        
        "seed": 42,
        "parallels": 1,
        "running_steps": 200000,
        
        "gamma": 0.99,
        "use_grad_clip": True,
        "grad_clip_norm": 0.5,
        
        "use_obsnorm": False,
        "use_rewnorm": False,
        "obsnorm_range": 5,
        "rewnorm_range": 5,
        
        "test_steps": 10000,
        "eval_interval": 10000,
        "test_episode": 5,
        
        "log_dir": "./logs/",
        "model_dir": "./models/",
        
        "vec_config": {
            "num_vehicles": 12,
            "num_rsus": 4,
            "num_uavs": 2,
            "arrival_rate": 3.5,
            "max_episode_steps": 200,
            "use_enhanced_cache": True,
            "disable_migration": False,
            "reward_weight_delay": 0.5,
            "reward_weight_energy": 0.5,
            "reward_penalty_dropped": 1.0,
        }
    }
    
    # 算法特定配置
    if method.lower() == "td3":
        base_config.update({
            "agent": "TD3",
            "learner": "TD3_Learner",
            "policy": "DeterministicPolicy",
            "actor_learning_rate": 9e-5,
            "critic_learning_rate": 9e-5,
            "tau": 0.005,
            "batch_size": 384,
            "buffer_size": 100000,
            "start_training": 1000,
            "training_frequency": 1,
            "actor_update_delay": 2,
            # TD3 noise parameters (xuance required)
            "start_noise": 0.18,
            "end_noise": 0.05,
            "explore_noise": 0.18,
            "target_noise": 0.05,
            "noise_clip": 0.2,
        })
    elif method.lower() == "sac":
        base_config.update({
            "agent": "SAC",
            "learner": "SAC_Learner",
            "policy": "Gaussian_SAC",
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "alpha_learning_rate": 3e-4,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 100000,
            "start_training": 1000,
            "training_frequency": 1,
            "alpha": 0.2,
            "use_automatic_entropy_tuning": True,
        })
    elif method.lower() == "ppo":
        base_config.update({
            "agent": "PPO_Clip",
            "learner": "PPOCLIP_Learner",
            "policy": "Gaussian_AC",
            "learning_rate": 3e-4,
            "horizon_size": 256,
            "n_epochs": 10,
            "n_minibatch": 4,
            "clip_range": 0.2,
            "vf_coef": 0.5,
            "ent_coef": 0.01,
            "use_gae": True,
            "gae_lambda": 0.95,
            "use_advnorm": True,
            "use_obsnorm": True,
            "use_rewnorm": True,
            "parallels": 4,
        })
    elif method.lower() == "ddpg":
        base_config.update({
            "agent": "DDPG",
            "learner": "DDPG_Learner",
            "policy": "Deterministic_Policy",
            "actor_learning_rate": 1e-4,
            "critic_learning_rate": 1e-3,
            "tau": 0.005,
            "batch_size": 256,
            "buffer_size": 100000,
            "start_training": 1000,
            "training_frequency": 1,
            "explore_noise": 0.1,
        })
    
    return base_config


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """加载YAML配置文件"""
    if not HAS_YAML:
        return {}
    
    path = Path(config_path)
    if not path.exists():
        print(f"⚠️ 配置文件不存在: {config_path}")
        return {}
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def merge_configs(base: Dict, *updates: Dict) -> Dict:
    """递归合并配置"""
    result = deepcopy(base)
    for update in updates:
        if update:
            for key, value in update.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = merge_configs(result[key], value)
                elif value is not None:
                    result[key] = value
    return result


def apply_args_to_config(config: Dict, args: argparse.Namespace) -> Dict:
    """应用命令行参数到配置"""
    # 直接映射
    direct_mappings = {
        'device': 'device',
        'seed': 'seed',
        'parallels': 'parallels',
        'running_steps': 'running_steps',
        'eval_interval': 'eval_interval',
        'test_episode': 'test_episode',
        'logger': 'logger',
        'log_dir': 'log_dir',
        'model_dir': 'model_dir',
    }
    
    for arg_name, config_key in direct_mappings.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            config[config_key] = value
    
    # 计算running_steps
    if args.episodes is not None:
        config['running_steps'] = args.episodes * args.max_steps * config.get('parallels', 1)
    
    # VEC特定参数
    vec_config = config.setdefault('vec_config', {})
    if args.num_vehicles is not None:
        vec_config['num_vehicles'] = args.num_vehicles
    if args.num_rsus is not None:
        vec_config['num_rsus'] = args.num_rsus
    if args.num_uavs is not None:
        vec_config['num_uavs'] = args.num_uavs
    if args.arrival_rate is not None:
        vec_config['arrival_rate'] = args.arrival_rate
    vec_config['max_episode_steps'] = args.max_steps
    
    # 测试模式
    config['test_mode'] = args.test
    
    # 更新路径
    method = args.method.lower()
    if config['log_dir'] == "./logs/":
        config['log_dir'] = f"./logs/{method}_vec/"
    if config['model_dir'] == "./models/":
        config['model_dir'] = f"./models/{method}_vec/"
    
    return config


def create_vec_envs(config: Dict, parallels: int = 1):
    """创建VEC环境"""
    from argparse import Namespace
    
    config_ns = Namespace(**config)
    
    if parallels == 1:
        return VECEnv(config=config_ns)
    else:
        # 创建向量化环境
        def make_env():
            return VECEnv(config=config_ns)
        
        envs = [make_env() for _ in range(parallels)]
        # 简单包装
        return envs[0]  # 暂时返回单个环境


def train_with_xuance(config: Dict, args: argparse.Namespace):
    """使用XuanCe框架训练"""
    from argparse import Namespace
    
    print("\n" + "="*60)
    print("🚀 使用XuanCe框架训练")
    print("="*60)
    
    # 注册VEC环境
    register_vec_env()
    
    # 设置随机种子
    set_seed(config['seed'])
    
    # 创建环境
    config_ns = Namespace(**config)
    envs = make_envs(config_ns)
    
    # 获取Agent类
    from xuance.torch.agents import REGISTRY_Agents
    agent_name = config['agent']
    if agent_name not in REGISTRY_Agents:
        print(f"❌ 不支持的算法: {agent_name}")
        print(f"   可用算法: {list(REGISTRY_Agents.keys())}")
        return None
    
    Agent_cls = REGISTRY_Agents[agent_name]
    agent = Agent_cls(config=config_ns, envs=envs)
    
    # 打印训练信息
    print(f"\n📋 训练配置:")
    print(f"   算法: {config['agent']}")
    print(f"   设备: {config['device']}")
    print(f"   环境: {config['env_name']} / {config['env_id']}")
    print(f"   训练步数: {config['running_steps']}")
    print(f"   并行环境: {config['parallels']}")
    
    # 训练或测试
    if args.test:
        print("\n🧪 测试模式")
        if args.model_path:
            agent.load_model(path=args.model_path)
        
        def env_fn():
            return make_envs(config_ns)
        
        scores = agent.test(env_fn, config['test_episode'])
        print(f"   平均得分: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    elif args.benchmark:
        print("\n📊 基准测试模式")
        
        def env_fn():
            cfg_test = deepcopy(config_ns)
            cfg_test.parallels = config['test_episode']
            return make_envs(cfg_test)
        
        train_steps = config['running_steps'] // config['parallels']
        eval_interval = config['eval_interval'] // config['parallels']
        num_epochs = train_steps // eval_interval
        
        best_score = -float('inf')
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            agent.train(eval_interval)
            
            scores = agent.test(env_fn, config['test_episode'])
            mean_score = np.mean(scores)
            print(f"   评估得分: {mean_score:.4f} ± {np.std(scores):.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                agent.save_model(model_name="best_model.pth")
                print(f"   ✅ 保存最佳模型 (得分: {best_score:.4f})")
        
        print(f"\n🏆 最佳得分: {best_score:.4f}")
    
    else:
        print("\n🎯 训练模式")
        start_time = time.time()
        
        train_steps = config['running_steps'] // config['parallels']
        agent.train(train_steps)
        agent.save_model("final_model.pth")
        
        training_time = time.time() - start_time
        print(f"\n✅ 训练完成!")
        print(f"   耗时: {training_time/3600:.2f}小时")
        print(f"   模型保存至: {config['model_dir']}")
    
    agent.finish()
    envs.close()
    
    return agent


def train_with_benchmark(config: Dict, args: argparse.Namespace):
    """使用对比方案训练/评估 (Benchmarks)"""
    print("\n" + "="*60)
    print("[Benchmark] 对比方案评估")
    print("="*60)
    
    method = args.method.lower()
    episodes = config['running_steps'] // config['vec_config']['max_episode_steps']
    seed = config.get('seed', 42)
    
    # 环境配置 (使用VecEnvWrapper的参数名)
    env_cfg = {
        'num_vehicles': config['vec_config']['num_vehicles'],
        'num_rsus': config['vec_config']['num_rsus'],
        'num_uavs': config['vec_config']['num_uavs'],
        'task_arrival_rate': config['vec_config'].get('arrival_rate', 3.5),
    }
    
    print(f"\n[Config] 对比方案配置:")
    print(f"   算法: {method}")
    print(f"   评估轮次: {episodes}")
    print(f"   环境: {env_cfg}")
    
    # 导入Benchmarks模块
    try:
        from Benchmarks.run_benchmarks_vs_optimized_td3 import (
            run_rl, run_local, run_heuristic, run_sa, set_global_seeds
        )
        
        set_global_seeds(seed)
        
        if method == 'local':
            results = run_local(env_cfg, episodes, seed)
            alg_name = "Local-Only"
        elif method == 'heuristic':
            results = run_heuristic(env_cfg, episodes, seed)
            alg_name = "Dynamic-Heuristic"
        elif method == 'sa':
            results = run_sa(env_cfg, episodes, seed)
            alg_name = "Simulated-Annealing"
        elif method == 'benchmark_td3':
            results = run_rl('td3', episodes, seed, env_cfg)
            alg_name = "Benchmark-TD3"
        elif method == 'benchmark_ddpg':
            results = run_rl('ddpg', episodes, seed, env_cfg)
            alg_name = "Benchmark-DDPG"
        elif method == 'benchmark_sac':
            results = run_rl('sac', episodes, seed, env_cfg)
            alg_name = "Benchmark-SAC"
        else:
            print(f"[Error] 未知的对比方案: {method}")
            return None
        
        # 打印结果
        print(f"\n[OK] {alg_name} 评估完成!")
        if results and 'episode_rewards' in results:
            rewards = results['episode_rewards']
            print(f"   轮次: {len(rewards)}")
            print(f"   平均奖励: {np.mean(rewards):.4f}")
            print(f"   最终奖励: {rewards[-1]:.4f}")
            
            if 'episode_metrics' in results:
                metrics = results['episode_metrics']
                if 'avg_task_delay' in metrics:
                    print(f"   平均延迟: {np.mean(metrics['avg_task_delay']):.4f}s")
                if 'task_completion_rate' in metrics:
                    print(f"   完成率: {np.mean(metrics['task_completion_rate'])*100:.2f}%")
        
        return results
        
    except ImportError as e:
        print(f"[Error] 无法导入Benchmarks模块: {e}")
        print("   请确保 Benchmarks/ 目录存在")
        return None
    except Exception as e:
        print(f"[Error] 对比方案运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def train_with_local(config: Dict, args: argparse.Namespace):
    """使用本地实现训练（当XuanCe不可用时）"""
    print("\n" + "="*60)
    print("🔧 使用本地实现训练")
    print("="*60)
    
    # 注册环境
    register_vec_env()
    
    # 设置随机种子
    np.random.seed(config['seed'])
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # 导入本地算法
    method = args.method.lower()
    
    # 算法名称映射（小写 -> train_single_agent要求的格式）
    algorithm_map = {
        'td3': 'TD3',
        'sac': 'SAC',
        'ppo': 'PPO',
        'ddpg': 'DDPG',
        'dqn': 'DQN',
        'optimized_td3': 'OPTIMIZED_TD3',
        'cam_td3': 'CAM_TD3',
        'td3_le': 'TD3-LE',
        'td3-le': 'TD3-LE',
    }
    
    algorithm_name = algorithm_map.get(method)
    if not algorithm_name:
        # 检查是否是对比方案
        if method in ['local', 'heuristic', 'sa', 'benchmark_td3', 'benchmark_ddpg', 'benchmark_sac']:
            return train_with_benchmark(config, args)
        print(f"[Error] 不支持的算法: {method}")
        print(f"   DRL算法: {list(algorithm_map.keys())}")
        print(f"   对比方案: local, heuristic, sa, benchmark_td3, benchmark_ddpg, benchmark_sac")
        return None
    
    # 计算训练轮次
    episodes = config['running_steps'] // config['vec_config']['max_episode_steps']
    
    print(f"\n[Config] 训练配置:")
    print(f"   算法: {algorithm_name}")
    print(f"   训练轮次: {episodes}")
    print(f"   每轮步数: {config['vec_config']['max_episode_steps']}")
    
    # 使用现有训练脚本
    from train_single_agent import train_single_algorithm
    
    override_scenario = {
        'num_vehicles': config['vec_config']['num_vehicles'],
        'num_rsus': config['vec_config']['num_rsus'],
        'num_uavs': config['vec_config']['num_uavs'],
    }
    
    # 设置随机种子
    import random
    seed = config.get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    
    results = train_single_algorithm(
        algorithm=algorithm_name,
        num_episodes=episodes,
        override_scenario=override_scenario,
        use_enhanced_cache=config['vec_config']['use_enhanced_cache'],
        disable_migration=config['vec_config']['disable_migration'],
    )
    
    print(f"\n[OK] 训练完成!")
    if results:
        print(f"   最终奖励: {results.get('final_reward', 'N/A')}")
    
    return results


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    print("\n" + "="*60)
    print("🎮 VEC边缘计算系统 - XuanCe训练脚本")
    print("="*60)
    
    # 获取默认配置
    config = get_default_config(args.method)
    
    # 加载YAML配置
    if args.config:
        yaml_config = load_yaml_config(args.config)
        config = merge_configs(config, yaml_config)
    else:
        # 尝试加载默认配置文件
        # 支持两种路径: xuance/configs/ 或 xuance_configs/
        script_dir = Path(__file__).parent
        config_paths = [
            script_dir / "configs" / f"{args.method.lower()}_vec.yaml",
            Path(f"xuance/configs/{args.method.lower()}_vec.yaml"),
            Path(f"xuance_configs/{args.method.lower()}_vec.yaml"),
        ]
        for config_path in config_paths:
            if config_path.exists():
                yaml_config = load_yaml_config(str(config_path))
                config = merge_configs(config, yaml_config)
                break
    
    # 应用命令行参数
    config = apply_args_to_config(config, args)
    
    # 创建目录
    Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['model_dir']).mkdir(parents=True, exist_ok=True)
    
    # 打印配置摘要
    if args.verbose:
        print("\n📋 完整配置:")
        for key, value in config.items():
            if isinstance(value, dict):
                print(f"   {key}:")
                for k, v in value.items():
                    print(f"      {k}: {v}")
            else:
                print(f"   {key}: {value}")
    
    # 选择训练方式
    # 本地DRL算法列表
    local_drl_algorithms = ['td3', 'ddpg', 'sac', 'ppo', 'dqn', 
                            'optimized_td3', 'cam_td3', 'td3_le', 'td3-le']
    # 对比方案列表
    benchmark_algorithms = ['local', 'heuristic', 'sa', 
                           'benchmark_td3', 'benchmark_ddpg', 'benchmark_sac']
    
    method = args.method.lower()
    
    if method in benchmark_algorithms:
        # 运行对比方案
        train_with_benchmark(config, args)
    elif not HAS_XUANCE or method in local_drl_algorithms:
        # 使用本地DRL实现
        train_with_local(config, args)
    else:
        # 使用XuanCe框架
        train_with_xuance(config, args)


if __name__ == "__main__":
    main()
