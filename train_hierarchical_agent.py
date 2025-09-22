"""
分层强化学习训练脚本
支持战略层(SAC)、战术层(MATD3/MAPPO)、执行层(TD3/DDPG)的分层训练

使用方法:
python train_hierarchical_agent.py --episodes 200 --mode hierarchical
python train_hierarchical_agent.py --episodes 200 --mode strategic_only
python train_hierarchical_agent.py --episodes 200 --mode tactical_only
python train_hierarchical_agent.py --episodes 200 --mode operational_only
"""

# 性能优化 - 必须在其他导入之前
try:
    from tools.performance_optimization import *
except ImportError:
    print("警告: 无法导入性能优化模块")
    OPTIMIZED_BATCH_SIZES = {}
    PARALLEL_ENVS = 1
    NUM_WORKERS = 0

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入核心模块
from evaluation.test_complete_system import CompleteSystemSimulator
from utils import MovingAverage
from config import config

# 导入分层学习模块
from hierarchical_learning.core.hierarchical_environment import HierarchicalEnvironment
from hierarchical_learning.core.strategic_layer import StrategicLayer
from hierarchical_learning.core.tactical_layer import TacticalLayer
from hierarchical_learning.core.operational_layer import OperationalLayer

# 导入现有算法（用于对比）
from algorithms.matd3 import MATD3Environment
from algorithms.mappo import MAPPOEnvironment
from single_agent.sac import SACEnvironment
from single_agent.td3 import TD3Environment


def generate_timestamp() -> str:
    """生成时间戳"""
    if config.experiment.use_timestamp:
        return datetime.now().strftime(config.experiment.timestamp_format)
    else:
        return ""


def get_timestamped_filename(base_name: str, extension: str = ".json") -> str:
    """获取带时间戳的文件名"""
    timestamp = generate_timestamp()
    if timestamp:
        name_parts = base_name.split('.')
        if len(name_parts) > 1:
            base = '.'.join(name_parts[:-1])
            return f"{base}_{timestamp}{extension}"
        else:
            return f"{base_name}_{timestamp}{extension}"
    else:
        return f"{base_name}{extension}"


class HierarchicalTrainingEnvironment:
    """分层强化学习训练环境"""
    
    def __init__(self, training_mode: str = "hierarchical"):
        """
        初始化分层训练环境
        
        Args:
            training_mode: 训练模式 - "hierarchical", "strategic_only", "tactical_only", "operational_only"
        """
        self.training_mode = training_mode.lower()
        self.simulator = CompleteSystemSimulator()
        
        # 获取优化后的批次大小
        self.optimized_batch_size = self._get_optimized_batch_size()
        print(f"🚀 使用优化批次大小: {self.optimized_batch_size}")
        
        # 分层环境配置
        hierarchical_config = {
            'num_rsus': config.num_rsus,
            'num_uavs': config.num_uavs,
            'num_vehicles': config.num_vehicles,
            'area_size': (1000, 1000),  # 默认区域大小
            'max_episode_steps': config.experiment.max_steps_per_episode,
            'strategic_config': {
                'state_dim': 50,  # 战略层状态维度
                'action_dim': 10,  # 战略层动作维度
                'hidden_dim': 256,
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'alpha': 0.2,
                'batch_size': self.optimized_batch_size
            },
            'tactical_config': {
                'num_agents': config.num_rsus + config.num_uavs,
                'state_dim': 30,  # 战术层状态维度
                'action_dim': 8,   # 战术层动作维度
                'hidden_dim': 128,
                'lr': 1e-4,
                'gamma': 0.95,
                'tau': 0.01,
                'batch_size': self.optimized_batch_size
            },
            'operational_config': {
                'num_agents': config.num_rsus + config.num_uavs,
                'state_dim': 40,  # 执行层状态维度
                'action_dim': 6,   # 执行层动作维度
                'hidden_dim': 128,
                'lr': 1e-4,
                'gamma': 0.9,
                'tau': 0.005,
                'batch_size': self.optimized_batch_size
            }
        }
        
        # 创建分层环境
        self.hierarchical_env = HierarchicalEnvironment(hierarchical_config)
        
        # 训练统计
        self.episode_rewards = {
            'strategic': [],
            'tactical': [],
            'operational': [],
            'total': []
        }
        
        self.episode_losses = {
            'strategic': [],
            'tactical': [],
            'operational': []
        }
        
        self.episode_metrics = {
            'avg_task_delay': [],
            'total_energy_consumption': [],
            'task_completion_rate': [],
            'cache_hit_rate': [],
            'migration_success_rate': [],
            'data_loss_rate': [],
            'strategic_decision_quality': [],
            'tactical_coordination_efficiency': [],
            'operational_control_precision': []
        }
        
        # 分层性能统计
        self.layer_performance = {
            'strategic': {'updates': 0, 'avg_loss': 0.0, 'avg_reward': 0.0},
            'tactical': {'updates': 0, 'avg_loss': 0.0, 'avg_reward': 0.0},
            'operational': {'updates': 0, 'avg_loss': 0.0, 'avg_reward': 0.0}
        }
        
        print(f"🎯 分层训练环境初始化完成 - 模式: {self.training_mode}")
        print(f"📊 战略层状态维度: {hierarchical_config['strategic_config']['state_dim']}")
        print(f"📊 战术层智能体数量: {hierarchical_config['tactical_config']['num_agents']}")
        print(f"📊 执行层智能体数量: {hierarchical_config['operational_config']['num_agents']}")
    
    def _get_optimized_batch_size(self) -> int:
        """获取优化后的批次大小"""
        try:
            return OPTIMIZED_BATCH_SIZES.get('hierarchical', config.rl.batch_size)
        except:
            return config.rl.batch_size
    
    def reset_environment(self) -> Dict[str, Dict[str, np.ndarray]]:
        """重置环境并返回初始状态"""
        # 重置分层环境
        hierarchical_states = self.hierarchical_env.reset()
        
        # 重置模拟器
        self.simulator.reset()
        
        return hierarchical_states
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None) -> Dict:
        """运行一个训练回合"""
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        # 重置环境
        states = self.reset_environment()
        
        episode_rewards = {'strategic': 0.0, 'tactical': 0.0, 'operational': 0.0, 'total': 0.0}
        episode_losses = {'strategic': [], 'tactical': [], 'operational': []}
        episode_metrics = []
        
        step_count = 0
        done = False
        
        print(f"🎮 开始第 {episode + 1} 回合训练 (模式: {self.training_mode})")
        
        while not done and step_count < max_steps:
            # 执行环境步骤
            next_states, rewards, done, info = self.hierarchical_env.step()
            
            # 存储经验
            self.hierarchical_env.store_experience(
                states, {}, rewards, next_states, {'strategic': done, 'tactical': done, 'operational': done}
            )
            
            # 根据训练模式执行训练
            training_results = {}
            if self.training_mode == "hierarchical":
                # 完整分层训练
                training_results = self.hierarchical_env.train_step()
            elif self.training_mode == "strategic_only":
                # 仅训练战略层
                if len(self.hierarchical_env.strategic_layer.sac_agent.replay_buffer) >= 32:
                    strategic_stats = self.hierarchical_env.strategic_layer.train()
                    if strategic_stats:
                        training_results['strategic'] = strategic_stats
            elif self.training_mode == "tactical_only":
                # 仅训练战术层
                tactical_stats = self.hierarchical_env.tactical_layer.train()
                if tactical_stats:
                    training_results['tactical'] = tactical_stats
            elif self.training_mode == "operational_only":
                # 仅训练执行层
                operational_stats = self.hierarchical_env.operational_layer.train()
                if operational_stats:
                    training_results['operational'] = operational_stats
            
            # 记录损失
            for layer, stats in training_results.items():
                if 'loss' in stats:
                    episode_losses[layer].append(stats['loss'])
                    self.layer_performance[layer]['updates'] += 1
                    self.layer_performance[layer]['avg_loss'] = (
                        self.layer_performance[layer]['avg_loss'] * 0.9 + stats['loss'] * 0.1
                    )
            
            # 累积奖励
            for layer, reward in rewards.items():
                if isinstance(reward, (int, float)):
                    episode_rewards[layer] += reward
                    self.layer_performance[layer]['avg_reward'] = (
                        self.layer_performance[layer]['avg_reward'] * 0.9 + reward * 0.1
                    )
            
            # 计算当前累计总和
            episode_rewards['total'] = sum([r for k, r in episode_rewards.items() 
                                          if k != 'total' and isinstance(r, (int, float))])
            
            # 记录系统指标
            episode_metrics.append(info.get('performance_metrics', {}))
            
            # 更新状态
            states = next_states
            step_count += 1
            
            # 每50步打印一次进度（按步平均，口径稳定）
            if step_count % 50 == 0:
                avg_total_so_far = episode_rewards['total'] / max(1, step_count)
                print(f"  步骤 {step_count}/{max_steps}, 平均奖励/步: {avg_total_so_far:.2f}")
            
            # 每200步打印一次诊断信息：各层buffer大小与更新计数
            if step_count % 200 == 0:
                try:
                    strat_buf = len(self.hierarchical_env.strategic_layer.sac_agent.replay_buffer)
                except Exception:
                    strat_buf = -1
                try:
                    tac_bufs = {aid: len(ag.replay_buffer) for aid, ag in self.hierarchical_env.tactical_layer.agents.items()}
                    tac_total = sum(tac_bufs.values())
                except Exception:
                    tac_bufs, tac_total = {}, -1
                try:
                    op_bufs = {aid: len(ag.replay_buffer) for aid, ag in self.hierarchical_env.operational_layer.agents.items()}
                    op_total = sum(op_bufs.values())
                except Exception:
                    op_bufs, op_total = {}, -1
                ts = self.hierarchical_env.training_stats
                print(f"  诊断: SAC缓冲={strat_buf}, MATD3总缓冲={tac_total}, TD3总缓冲={op_total}; 更新计数 S/T/O = {ts['strategic_updates']}/{ts['tactical_updates']}/{ts['operational_updates']}")
        
        # 计算回合平均指标
        avg_metrics = {}
        if episode_metrics:
            for key in episode_metrics[0].keys():
                values = [m.get(key, 0) for m in episode_metrics if key in m]
                if values:
                    avg_metrics[key] = np.mean(values)
        
        # 记录回合结果（统一为按步平均口径）
        if step_count > 0:
            strategic_avg = episode_rewards['strategic'] / step_count
            tactical_avg = episode_rewards['tactical'] / step_count
            operational_avg = episode_rewards['operational'] / step_count
            total_avg = strategic_avg + tactical_avg + operational_avg
        else:
            strategic_avg = tactical_avg = operational_avg = total_avg = 0.0

        # 打印总结时附带总和，主显示为均值
        print(f"   总奖励(均值/步): {total_avg:.2f}")
        print(f"   战略层奖励(均值/步): {strategic_avg:.2f}")
        print(f"   战术层奖励(均值/步): {tactical_avg:.2f}")
        print(f"   执行层奖励(均值/步): {operational_avg:.2f}")

        for layer, value in [('strategic', strategic_avg), ('tactical', tactical_avg), ('operational', operational_avg), ('total', total_avg)]:
            self.episode_rewards[layer].append(value)
        
        for layer in ['strategic', 'tactical', 'operational']:
            if episode_losses[layer]:
                self.episode_losses[layer].append(np.mean(episode_losses[layer]))
            else:
                self.episode_losses[layer].append(0.0)
        
        # 记录系统指标
        for key, value in avg_metrics.items():
            if key in self.episode_metrics:
                self.episode_metrics[key].append(value)
        
        # 添加分层特有指标
        self.episode_metrics['strategic_decision_quality'].append(
            self.layer_performance['strategic']['avg_reward']
        )
        self.episode_metrics['tactical_coordination_efficiency'].append(
            self.layer_performance['tactical']['avg_reward']
        )
        self.episode_metrics['operational_control_precision'].append(
            self.layer_performance['operational']['avg_reward']
        )
        
        print(f"✅ 第 {episode + 1} 回合完成:")
        print(f"   总步数: {step_count}")
        print(f"   总奖励: {episode_rewards['total']:.2f}")
        print(f"   战略层奖励: {episode_rewards['strategic']:.2f}")
        print(f"   战术层奖励: {episode_rewards['tactical']:.2f}")
        print(f"   执行层奖励: {episode_rewards['operational']:.2f}")
        
        return {
            'episode': episode,
            'steps': step_count,
            'rewards': episode_rewards,
            'losses': episode_losses,
            'metrics': avg_metrics,
            'layer_performance': self.layer_performance.copy()
        }
    
    def evaluate_model(self, num_eval_episodes: int = 5) -> Dict:
        """评估分层模型性能"""
        print(f"🔍 开始模型评估 ({num_eval_episodes} 回合)")
        
        eval_rewards = {'strategic': [], 'tactical': [], 'operational': [], 'total': []}
        eval_metrics = []
        
        for eval_episode in range(num_eval_episodes):
            # 重置环境
            states = self.reset_environment()
            
            episode_rewards = {'strategic': 0.0, 'tactical': 0.0, 'operational': 0.0, 'total': 0.0}
            episode_metrics = []
            
            step_count = 0
            done = False
            max_steps = config.experiment.max_steps_per_episode
            
            while not done and step_count < max_steps:
                # 执行环境步骤（评估模式，不训练）
                next_states, rewards, done, info = self.hierarchical_env.step()
                
                # 累积奖励（评估同样按总和累加，最后转均值）
                for layer, reward in rewards.items():
                    if isinstance(reward, (int, float)):
                        episode_rewards[layer] += reward
                
                episode_rewards['total'] = sum([r for k, r in episode_rewards.items() 
                                               if k != 'total' and isinstance(r, (int, float))])
                
                # 记录指标
                episode_metrics.append(info.get('performance_metrics', {}))
                
                states = next_states
                step_count += 1
            
            # 记录评估结果（统一为按步平均口径）
            if step_count > 0:
                strategic_avg = episode_rewards['strategic'] / step_count
                tactical_avg = episode_rewards['tactical'] / step_count
                operational_avg = episode_rewards['operational'] / step_count
                total_avg = strategic_avg + tactical_avg + operational_avg
            else:
                strategic_avg = tactical_avg = operational_avg = total_avg = 0.0

            for layer, value in [('strategic', strategic_avg), ('tactical', tactical_avg), ('operational', operational_avg), ('total', total_avg)]:
                eval_rewards[layer].append(value)
            
            # 计算平均指标
            if episode_metrics:
                avg_metrics = {}
                for key in episode_metrics[0].keys():
                    values = [m.get(key, 0) for m in episode_metrics if key in m]
                    if values:
                        avg_metrics[key] = np.mean(values)
                eval_metrics.append(avg_metrics)
        
        # 计算评估统计
        eval_stats = {}
        for layer in eval_rewards.keys():
            if eval_rewards[layer]:
                eval_stats[f'{layer}_reward_mean'] = np.mean(eval_rewards[layer])
                eval_stats[f'{layer}_reward_std'] = np.std(eval_rewards[layer])
        
        # 计算系统指标统计
        if eval_metrics:
            for key in eval_metrics[0].keys():
                values = [m.get(key, 0) for m in eval_metrics if key in m]
                if values:
                    eval_stats[f'{key}_mean'] = np.mean(values)
                    eval_stats[f'{key}_std'] = np.std(values)
        
        print(f"📊 评估完成:")
        print(f"   平均总奖励(均值/步): {eval_stats.get('total_reward_mean', 0):.2f} ± {eval_stats.get('total_reward_std', 0):.2f}")
        print(f"   平均任务延迟: {eval_stats.get('total_latency_mean', 0):.2f} ms")
        print(f"   平均成功率: {eval_stats.get('success_rate_mean', 0):.3f}")
        
        return eval_stats
    
    def save_models(self, save_path: str):
        """保存分层模型"""
        timestamp = generate_timestamp()
        if timestamp:
            save_path = f"{save_path}_{timestamp}"
        
        self.hierarchical_env.save_models(save_path)
        print(f"💾 分层模型已保存到: {save_path}")
    
    def load_models(self, load_path: str):
        """加载分层模型"""
        self.hierarchical_env.load_models(load_path)
        print(f"📂 分层模型已从 {load_path} 加载")


def train_hierarchical_algorithm(training_mode: str = "hierarchical", 
                                num_episodes: Optional[int] = None,
                                eval_interval: Optional[int] = None,
                                save_interval: Optional[int] = None) -> Dict:
    """训练分层强化学习算法"""
    
    # 使用配置文件中的默认值
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    if eval_interval is None:
        eval_interval = config.experiment.eval_interval
    if save_interval is None:
        save_interval = config.experiment.save_interval
    
    print(f"🚀 开始分层强化学习训练")
    print(f"📋 训练模式: {training_mode}")
    print(f"📋 训练回合数: {num_episodes}")
    print(f"📋 评估间隔: {eval_interval}")
    print(f"📋 保存间隔: {save_interval}")
    
    # 创建训练环境
    training_env = HierarchicalTrainingEnvironment(training_mode)
    
    # 训练统计
    training_start_time = time.time()
    best_performance = -float('inf')
    
    # 训练循环
    for episode in range(num_episodes):
        episode_start_time = time.time()
        
        # 运行训练回合
        episode_result = training_env.run_episode(episode)
        
        episode_time = time.time() - episode_start_time
        
        # 定期评估
        if (episode + 1) % eval_interval == 0:
            eval_stats = training_env.evaluate_model()
            current_performance = eval_stats.get('total_reward_mean', -float('inf'))
            
            # 保存最佳模型
            if current_performance > best_performance:
                best_performance = current_performance
                training_env.save_models(f"models/hierarchical_best_{training_mode}")
                print(f"🏆 发现更好的模型! 性能: {current_performance:.2f}")
        
        # 定期保存检查点
        if (episode + 1) % save_interval == 0:
            training_env.save_models(f"models/hierarchical_checkpoint_{training_mode}_ep{episode+1}")
        
        # 打印训练进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(training_env.episode_rewards['total'][-10:])
            print(f"📈 回合 {episode + 1}/{num_episodes}, 最近10回合平均奖励: {avg_reward:.2f}, 用时: {episode_time:.2f}s")
    
    training_time = time.time() - training_start_time
    
    # 最终评估
    final_eval_stats = training_env.evaluate_model(num_eval_episodes=10)
    
    # 保存最终模型
    training_env.save_models(f"models/hierarchical_final_{training_mode}")
    
    # 保存训练结果
    training_results = save_hierarchical_training_results(training_mode, training_env, training_time)
    
    # 绘制训练曲线
    plot_hierarchical_training_curves(training_mode, training_env)
    
    print(f"🎉 分层训练完成!")
    print(f"⏱️  总训练时间: {training_time:.2f} 秒")
    print(f"📊 最终性能: {final_eval_stats.get('total_reward_mean', 0):.2f}")
    
    return {
        'training_mode': training_mode,
        'num_episodes': num_episodes,
        'training_time': training_time,
        'final_performance': final_eval_stats,
        'training_env': training_env,
        'results': training_results
    }


def save_hierarchical_training_results(training_mode: str, 
                                     training_env: HierarchicalTrainingEnvironment,
                                     training_time: float) -> Dict:
    """保存分层训练结果"""
    
    results = {
        'training_mode': training_mode,
        'training_time': training_time,
        'num_episodes': len(training_env.episode_rewards['total']),
        'episode_rewards': training_env.episode_rewards,
        'episode_losses': training_env.episode_losses,
        'episode_metrics': training_env.episode_metrics,
        'layer_performance': training_env.layer_performance,
        'final_stats': training_env.hierarchical_env.get_training_stats()
    }
    
    # 保存到文件
    os.makedirs('results', exist_ok=True)
    filename = get_timestamped_filename(f'results/hierarchical_{training_mode}_training_results', '.json')
    
    # 转换numpy数组为列表以便JSON序列化
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_numpy(results)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results_serializable, f, indent=2, ensure_ascii=False)
    
    print(f"💾 训练结果已保存到: {filename}")
    
    return results


def plot_hierarchical_training_curves(training_mode: str, 
                                     training_env: HierarchicalTrainingEnvironment):
    """绘制分层训练曲线"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'分层强化学习训练曲线 - {training_mode.upper()}', fontsize=16)
    
    # 奖励曲线
    axes[0, 0].plot(training_env.episode_rewards['total'], label='总奖励', color='blue')
    axes[0, 0].plot(training_env.episode_rewards['strategic'], label='战略层', color='red', alpha=0.7)
    axes[0, 0].plot(training_env.episode_rewards['tactical'], label='战术层', color='green', alpha=0.7)
    axes[0, 0].plot(training_env.episode_rewards['operational'], label='执行层', color='orange', alpha=0.7)
    axes[0, 0].set_title('分层奖励曲线')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 损失曲线
    axes[0, 1].plot(training_env.episode_losses['strategic'], label='战略层', color='red')
    axes[0, 1].plot(training_env.episode_losses['tactical'], label='战术层', color='green')
    axes[0, 1].plot(training_env.episode_losses['operational'], label='执行层', color='orange')
    axes[0, 1].set_title('分层损失曲线')
    axes[0, 1].set_xlabel('回合')
    axes[0, 1].set_ylabel('损失')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 系统性能指标
    if training_env.episode_metrics['avg_task_delay']:
        axes[0, 2].plot(training_env.episode_metrics['avg_task_delay'], label='平均延迟', color='purple')
        axes[0, 2].set_title('平均任务延迟')
        axes[0, 2].set_xlabel('回合')
        axes[0, 2].set_ylabel('延迟 (ms)')
        axes[0, 2].grid(True)
    
    if training_env.episode_metrics['task_completion_rate']:
        axes[1, 0].plot(training_env.episode_metrics['task_completion_rate'], label='任务完成率', color='cyan')
        axes[1, 0].set_title('任务完成率')
        axes[1, 0].set_xlabel('回合')
        axes[1, 0].set_ylabel('完成率')
        axes[1, 0].grid(True)
    
    if training_env.episode_metrics['total_energy_consumption']:
        axes[1, 1].plot(training_env.episode_metrics['total_energy_consumption'], label='总能耗', color='brown')
        axes[1, 1].set_title('总能耗')
        axes[1, 1].set_xlabel('回合')
        axes[1, 1].set_ylabel('能耗 (J)')
        axes[1, 1].grid(True)
    
    # 分层决策质量
    axes[1, 2].plot(training_env.episode_metrics['strategic_decision_quality'], 
                   label='战略决策质量', color='red', alpha=0.8)
    axes[1, 2].plot(training_env.episode_metrics['tactical_coordination_efficiency'], 
                   label='战术协调效率', color='green', alpha=0.8)
    axes[1, 2].plot(training_env.episode_metrics['operational_control_precision'], 
                   label='执行控制精度', color='orange', alpha=0.8)
    axes[1, 2].set_title('分层决策质量')
    axes[1, 2].set_xlabel('回合')
    axes[1, 2].set_ylabel('质量指标')
    axes[1, 2].legend()
    axes[1, 2].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    os.makedirs('plots', exist_ok=True)
    filename = get_timestamped_filename(f'plots/hierarchical_{training_mode}_training_curves', '.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 训练曲线已保存到: {filename}")


def compare_hierarchical_modes(modes: List[str], num_episodes: Optional[int] = None) -> Dict:
    """比较不同分层训练模式"""
    
    if num_episodes is None:
        num_episodes = config.experiment.num_episodes
    
    print(f"🔄 开始比较分层训练模式: {modes}")
    
    results = {}
    
    for mode in modes:
        print(f"\n🎯 训练模式: {mode}")
        mode_results = train_hierarchical_algorithm(mode, num_episodes)
        results[mode] = mode_results
    
    # 绘制比较图
    plot_hierarchical_mode_comparison(results)
    
    # 保存比较结果
    comparison_results = {
        'modes': modes,
        'num_episodes': num_episodes,
        'results': {mode: {
            'final_performance': result['final_performance'],
            'training_time': result['training_time']
        } for mode, result in results.items()}
    }
    
    filename = get_timestamped_filename('results/hierarchical_mode_comparison', '.json')
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, indent=2, ensure_ascii=False)
    
    print(f"💾 比较结果已保存到: {filename}")
    
    return results


def plot_hierarchical_mode_comparison(results: Dict):
    """绘制分层模式比较图"""
    
    modes = list(results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('分层训练模式比较', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    
    # 总奖励比较
    for i, (mode, result) in enumerate(results.items()):
        training_env = result['training_env']
        axes[0, 0].plot(training_env.episode_rewards['total'], 
                       label=mode, color=colors[i % len(colors)])
    
    axes[0, 0].set_title('总奖励比较')
    axes[0, 0].set_xlabel('回合')
    axes[0, 0].set_ylabel('总奖励')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # 训练时间比较
    training_times = [result['training_time'] for result in results.values()]
    axes[0, 1].bar(modes, training_times, color=colors[:len(modes)])
    axes[0, 1].set_title('训练时间比较')
    axes[0, 1].set_xlabel('训练模式')
    axes[0, 1].set_ylabel('时间 (秒)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 最终性能比较
    final_performances = [result['final_performance'].get('total_reward_mean', 0) 
                         for result in results.values()]
    axes[1, 0].bar(modes, final_performances, color=colors[:len(modes)])
    axes[1, 0].set_title('最终性能比较')
    axes[1, 0].set_xlabel('训练模式')
    axes[1, 0].set_ylabel('平均奖励')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # 系统指标比较（以任务完成率为例）
    for i, (mode, result) in enumerate(results.items()):
        training_env = result['training_env']
        if training_env.episode_metrics['task_completion_rate']:
            axes[1, 1].plot(training_env.episode_metrics['task_completion_rate'], 
                           label=mode, color=colors[i % len(colors)])
    
    axes[1, 1].set_title('任务完成率比较')
    axes[1, 1].set_xlabel('回合')
    axes[1, 1].set_ylabel('完成率')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    # 保存图像
    filename = get_timestamped_filename('plots/hierarchical_mode_comparison', '.png')
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 比较图已保存到: {filename}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='分层强化学习训练脚本')
    parser.add_argument('--mode', type=str, default='hierarchical',
                       choices=['hierarchical', 'strategic_only', 'tactical_only', 'operational_only'],
                       help='训练模式')
    parser.add_argument('--episodes', type=int, default=None,
                       help='训练回合数')
    parser.add_argument('--eval_interval', type=int, default=None,
                       help='评估间隔')
    parser.add_argument('--save_interval', type=int, default=None,
                       help='保存间隔')
    parser.add_argument('--compare', action='store_true',
                       help='比较所有训练模式')
    
    args = parser.parse_args()
    
    # 创建必要的目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    if args.compare:
        # 比较所有模式
        modes = ['hierarchical', 'strategic_only', 'tactical_only', 'operational_only']
        compare_hierarchical_modes(modes, args.episodes)
    else:
        # 训练指定模式
        train_hierarchical_algorithm(
            training_mode=args.mode,
            num_episodes=args.episodes,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval
        )


if __name__ == "__main__":
    main()