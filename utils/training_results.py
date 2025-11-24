#!/usr/bin/env python3
"""
训练结果保存和可视化工具
Utilities for saving training results and plotting training curves
"""
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from config import config


def generate_timestamp() -> str:
    """生成时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """获取带时间戳的文件名"""
    timestamp = generate_timestamp()
    return f"{base_name}_{timestamp}{extension}"


def save_single_training_results(
    algorithm: str,
    training_env: Any,
    total_training_time: float,
    override_scenario: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    保存单智能体训练结果到JSON文件
    
    Args:
        algorithm: 算法名称
        training_env: 训练环境对象
        total_training_time: 总训练时间（秒）
        override_scenario: 覆盖的场景配置
        
    Returns:
        包含所有训练结果的字典
    """
    print("\n" + "=" * 60)
    print("💾 保存训练结果...")
    
    # 创建结果目录
    results_dir = Path(f"results/single_agent/{algorithm.lower()}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集训练结果
    results = {
        "algorithm": algorithm,
        "training_time": total_training_time,
        "timestamp": generate_timestamp(),
        "episode_rewards": training_env.episode_rewards,
        "episode_metrics": {},
        # 记录关键目标，便于后续分析与画图保持一致
        "network_config": {
            "delay_target": getattr(training_env, 'config', None) and getattr(training_env.config, 'rl', config.rl).latency_target or config.rl.latency_target,
            "energy_target": getattr(training_env, 'config', None) and getattr(training_env.config, 'rl', config.rl).energy_target or config.rl.energy_target,
        },
        "config": {
            "num_vehicles": training_env.simulator.num_vehicles,
            "num_rsus": training_env.simulator.num_rsus,
            "num_uavs": training_env.simulator.num_uavs,
            "total_episodes": len(training_env.episode_rewards),
        }
    }
    
    # 添加覆盖场景配置
    if override_scenario:
        results["config"]["override_scenario"] = override_scenario
    
    # 收集episode级别的指标
    for metric_name, metric_values in training_env.episode_metrics.items():
        if isinstance(metric_values, list) and len(metric_values) > 0:
            results["episode_metrics"][metric_name] = metric_values
    
    # 计算统计信息
    if len(training_env.episode_rewards) > 0:
        last_n = min(50, len(training_env.episode_rewards))
        results["statistics"] = {
            "final_avg_reward": float(np.mean(training_env.episode_rewards[-last_n:])),
            "final_avg_delay": float(np.mean(training_env.episode_metrics.get("avg_delay", [0])[-last_n:])),
            "final_avg_energy": float(np.mean(training_env.episode_metrics.get("total_energy", [0])[-last_n:])),
            "final_completion_rate": float(np.mean(training_env.episode_metrics.get("task_completion_rate", [0])[-last_n:])),
            "final_cache_hit_rate": float(np.mean(training_env.episode_metrics.get("cache_hit_rate", [0])[-last_n:])),
        }
    
    # 保存到JSON文件
    filename = get_timestamped_filename("training_results", ".json")
    filepath = results_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 训练结果已保存到: {filepath}")
        print(f"📊 包含 {len(training_env.episode_rewards)} 个episode的数据")
        
        # 打印关键统计
        if "statistics" in results:
            print("\n📈 最终性能（最后50个episode平均）:")
            stats = results["statistics"]
            print(f"  • 平均奖励: {stats['final_avg_reward']:.2f}")
            print(f"  • 平均延迟: {stats['final_avg_delay']:.4f}s")
            print(f"  • 平均能耗: {stats['final_avg_energy']:.2f}J")
            print(f"  • 完成率: {stats['final_completion_rate']:.2%}")
            print(f"  • 缓存命中率: {stats['final_cache_hit_rate']:.2%}")
        
        return results
        
    except Exception as e:
        print(f"❌ 保存训练结果失败: {e}")
        return results


def plot_single_training_curves(algorithm: str, training_env: Any) -> None:
    """
    绘制单智能体训练曲线
    
    Args:
        algorithm: 算法名称
        training_env: 训练环境对象
    """
    print("\n" + "=" * 60)
    print("📊 绘制训练曲线...")
    
    # 创建结果目录
    results_dir = Path(f"results/single_agent/{algorithm.lower()}")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置matplotlib
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.rcParams['font.size'] = 10
    
    # 创建4x2子图布局
    fig, axes = plt.subplots(4, 2, figsize=(16, 16))
    fig.suptitle(f'{algorithm} Training Curves', fontsize=16, fontweight='bold')
    
    # 获取数据
    episodes = list(range(1, len(training_env.episode_rewards) + 1))
    
    def smooth_curve(data: List[float], window: int = 20) -> np.ndarray:
        """使用移动平均平滑曲线"""
        if len(data) < window:
            return np.array(data)
        smoothed = np.convolve(data, np.ones(window)/window, mode='valid')
        # 填充前面的值
        padding = np.array(data[:window-1])
        return np.concatenate([padding, smoothed])
    
    # 1. Episode Reward
    ax = axes[0, 0]
    if training_env.episode_rewards:
        rewards = training_env.episode_rewards
        ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')
        ax.plot(episodes, smooth_curve(rewards), color='blue', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Reward')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Average Delay
    ax = axes[0, 1]
    delays = training_env.episode_metrics.get("avg_delay", [])
    if delays:
        ax.plot(episodes[:len(delays)], delays, alpha=0.3, color='red', label='Raw')
        ax.plot(episodes[:len(delays)], smooth_curve(delays), color='red', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delay (s)')
        ax.set_title('Average Task Delay')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Total Energy
    ax = axes[1, 0]
    energies = training_env.episode_metrics.get("total_energy", [])
    if energies:
        ax.plot(episodes[:len(energies)], energies, alpha=0.3, color='green', label='Raw')
        ax.plot(episodes[:len(energies)], smooth_curve(energies), color='green', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Total Energy Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Task Completion Rate
    ax = axes[1, 1]
    completion_rates = training_env.episode_metrics.get("task_completion_rate", [])
    if completion_rates:
        ax.plot(episodes[:len(completion_rates)], completion_rates, alpha=0.3, color='purple', label='Raw')
        ax.plot(episodes[:len(completion_rates)], smooth_curve(completion_rates), color='purple', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Completion Rate')
        ax.set_title('Task Completion Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # 5. Cache Hit Rate
    ax = axes[2, 0]
    cache_hits = training_env.episode_metrics.get("cache_hit_rate", [])
    if cache_hits:
        ax.plot(episodes[:len(cache_hits)], cache_hits, alpha=0.3, color='orange', label='Raw')
        ax.plot(episodes[:len(cache_hits)], smooth_curve(cache_hits), color='orange', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Cache Hit Rate')
        ax.set_title('Cache Hit Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # 6. Migration Success Rate
    ax = axes[2, 1]
    migration_rates = training_env.episode_metrics.get("migration_success_rate", [])
    if migration_rates:
        ax.plot(episodes[:len(migration_rates)], migration_rates, alpha=0.3, color='brown', label='Raw')
        ax.plot(episodes[:len(migration_rates)], smooth_curve(migration_rates), color='brown', linewidth=2, label='Smoothed')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Migration Success Rate')
        ax.set_title('Migration Success Rate')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])
    
    # 7. Energy Components Breakdown
    ax = axes[3, 0]
    energy_compute = training_env.episode_metrics.get("energy_compute", [])
    energy_uplink = training_env.episode_metrics.get("energy_transmit_uplink", [])
    energy_downlink = training_env.episode_metrics.get("energy_transmit_downlink", [])
    
    if energy_compute and energy_uplink and energy_downlink:
        min_len = min(len(energy_compute), len(energy_uplink), len(energy_downlink))
        x = episodes[:min_len]
        ax.plot(x, smooth_curve(energy_compute[:min_len]), label='Compute', linewidth=2)
        ax.plot(x, smooth_curve(energy_uplink[:min_len]), label='Uplink TX', linewidth=2)
        ax.plot(x, smooth_curve(energy_downlink[:min_len]), label='Downlink TX', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Energy (J)')
        ax.set_title('Energy Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 8. Delay Components Breakdown
    ax = axes[3, 1]
    delay_processing = training_env.episode_metrics.get("avg_processing_delay", [])
    delay_waiting = training_env.episode_metrics.get("avg_waiting_delay", [])
    delay_uplink = training_env.episode_metrics.get("avg_uplink_delay", [])
    delay_downlink = training_env.episode_metrics.get("avg_downlink_delay", [])
    
    if delay_processing and delay_uplink:
        min_len = min(len(delay_processing), len(delay_uplink))
        x = episodes[:min_len]
        ax.plot(x, smooth_curve(delay_processing[:min_len]), label='Processing', linewidth=2)
        if delay_waiting and len(delay_waiting) >= min_len:
            ax.plot(x, smooth_curve(delay_waiting[:min_len]), label='Waiting', linewidth=2)
        ax.plot(x, smooth_curve(delay_uplink[:min_len]), label='Uplink', linewidth=2)
        if delay_downlink and len(delay_downlink) >= min_len:
            ax.plot(x, smooth_curve(delay_downlink[:min_len]), label='Downlink', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delay (s)')
        ax.set_title('Delay Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    filename = get_timestamped_filename("training_curves", ".png")
    filepath = results_dir / filename
    
    try:
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ 训练曲线已保存到: {filepath}")
        
        # 可选：显示图表（在非静默模式下）
        # plt.show()
        
        plt.close(fig)
        
    except Exception as e:
        print(f"❌ 保存训练曲线失败: {e}")
        plt.close(fig)


if __name__ == "__main__":
    print("📊 训练结果工具模块")
    print("此模块提供以下功能：")
    print("  - save_single_training_results(): 保存训练结果到JSON")
    print("  - plot_single_training_curves(): 绘制训练曲线")
