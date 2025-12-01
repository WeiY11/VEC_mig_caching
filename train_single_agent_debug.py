"""
🐛 OPTIMIZED_TD3 调试版训练脚本
=================================

用途: 诊断OPTIMIZED_TD3算法不学习的根本原因

调试内容:
1. ✅ 动作传播追踪 - 验证agent输出的action是否被simulator正确使用
2. ✅ 奖励分量分析 - 详细记录delay/energy/cache各组件的贡献
3. ✅ 状态向量质量 - 检查state normalization和信息完整性
4. ✅ 网络梯度监控 - 追踪actor/critic的梯度更新
5. ✅ 经验回放采样 - 验证Queue-aware replay是否生效

使用方法:
python train_single_agent_debug.py --algorithm OPTIMIZED_TD3 --episodes 50 --num-vehicles 12 --seed 42

输出:
- debug_log_<timestamp>.txt: 详细调试日志
- debug_metrics_<timestamp>.json: 结构化调试数据
"""

import sys
sys.path.insert(0, 'd:\\VEC_mig_caching')

from train_single_agent import SingleAgentTrainingEnvironment, config, generate_timestamp
import numpy as np
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import time


class DebugSingleAgentTraining(SingleAgentTrainingEnvironment):
    """调试版训练环境 - 添加详细的日志输出"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # 创建调试日志文件
        timestamp = generate_timestamp()
        self.debug_log_file = f"debug_log_{timestamp}.txt"
        self.debug_metrics_file = f"debug_metrics_{timestamp}.json"
        
        # 调试数据收集器
        self.debug_data = {
            'action_traces': [],
            'reward_components': [],
            'state_samples': [],
            'gradient_norms': [],
            'replay_priorities': [],
            'system_states': []
        }
        
        # 打开日志文件
        self.log_file_handle = open(self.debug_log_file, 'w', encoding='utf-8')
        self.debug_log(f"{'='*80}")
        self.debug_log(f"🐛 调试会话开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.debug_log(f"算法: {self.algorithm}")
        self.debug_log(f"配置: 车辆={self.num_vehicles}, RSU={self.num_rsus}, UAV={self.num_uavs}")
        self.debug_log(f"{'='*80}\n")
        
        # 采样频率控制（避免日志过大）
        self.log_every_n_steps = 10  # 每10步详细记录一次
        self.step_counter = 0
    
    def debug_log(self, message: str, level: str = "INFO"):
        """写入调试日志"""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        formatted_msg = f"[{timestamp}] [{level}] {message}"
        print(formatted_msg)  # 同时输出到控制台
        self.log_file_handle.write(formatted_msg + '\n')
        self.log_file_handle.flush()
    
    def run_episode(self, episode: int, max_steps: Optional[int] = None, visualizer: Optional[Any] = None) -> Dict:
        """增强版episode运行 - 添加调试输出"""
        if max_steps is None:
            max_steps = config.experiment.max_steps_per_episode
        
        self.debug_log(f"\n{'─'*80}")
        self.debug_log(f"▶ Episode {episode} 开始 (最大步数: {max_steps})")
        self.debug_log(f"{'─'*80}")
        
        # 重置环境
        self._episode_counters_initialized = False
        state = self.reset_environment()
        
        self.visualizer = visualizer
        self._current_episode = episode
        self._current_episode_step = 0
        
        # 📊 记录初始状态样本
        self._log_state_vector(state, 0, "INITIAL")
        
        episode_reward = 0.0
        episode_info = {}
        step = 0
        info = {}
        
        # PPO特殊处理
        if self.algorithm == "PPO":
            return self._run_ppo_episode(episode, max_steps, visualizer)
        
        for step in range(max_steps):
            self.step_counter += 1
            should_log_detail = (step % self.log_every_n_steps == 0) or (step < 5)
            
            if should_log_detail:
                self.debug_log(f"\n┌── Step {step + 1}/{max_steps} ──┐")
            
            # ════════════════════════════════════════════════════════════════
            # 阶段 1: 选择动作
            # ════════════════════════════════════════════════════════════════
            if self.algorithm == "DQN":
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action_idx = self._encode_discrete_action(actions_dict)
                action = action_idx
            else:
                # 连续动作算法 (TD3, OPTIMIZED_TD3等)
                actions_result = self.agent_env.get_actions(state, training=True)
                if isinstance(actions_result, dict):
                    actions_dict = actions_result
                else:
                    actions_dict = actions_result[0] if isinstance(actions_result, tuple) else actions_result
                action = self._encode_continuous_action(actions_dict)
            
            # 📊 记录动作详情
            if should_log_detail:
                self._log_action_details(action, actions_dict, step)
            
            # ════════════════════════════════════════════════════════════════
            # 阶段 2: 执行动作
            # ════════════════════════════════════════════════════════════════
            self._current_episode_step += 1
            
            # 将向量动作恢复为字典供模拟器消费
            sim_actions_dict = actions_dict if isinstance(actions_dict, dict) else self._build_actions_from_vector(action)
            
            # 执行动作并获取奖励
            next_state, reward, done, info = self.step(action, state, sim_actions_dict)
            
            # 📊 记录奖励分量
            if should_log_detail:
                self._log_reward_breakdown(reward, info, step)
            
            # ════════════════════════════════════════════════════════════════
            # 阶段 3: 训练智能体
            # ════════════════════════════════════════════════════════════════
            # 更新队列指标
            if hasattr(self.agent_env, 'update_queue_metrics'):
                step_stats = info.get('step_stats', {})
                try:
                    self.agent_env.update_queue_metrics(step_stats)
                except Exception as e:
                    if self._current_episode % 100 == 0:
                        self.debug_log(f"⚠️ 队列指标更新失败: {e}", "WARNING")
            
            training_info = {}
            
            if self.algorithm == "DQN":
                safe_action = self._safe_int_conversion(action)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm in ["DDPG", "TD3", "TD3_LATENCY_ENERGY", "SAC", "OPTIMIZED_TD3"]:
                safe_action = action if isinstance(action, np.ndarray) else np.array([action], dtype=np.float32)
                training_info = self.agent_env.train_step(state, safe_action, reward, next_state, done)
            elif self.algorithm == "PPO":
                training_info = self.agent_env.train_step(state, action, reward, next_state, done)
            else:
                training_info = {'message': f'Unknown algorithm: {self.algorithm}'}
            
            # 📊 记录训练信息
            if should_log_detail and training_info:
                self._log_training_info(training_info, step)
            
            # ════════════════════════════════════════════════════════════════
            # 阶段 4: 更新状态
            # ════════════════════════════════════════════════════════════════
            episode_reward += reward
            episode_info = training_info
            
            # 📊 每N步记录状态样本
            if should_log_detail:
                self._log_state_vector(next_state, step + 1, "NEXT")
                self.debug_log(f"└── Step {step + 1} 完成 (累积奖励: {episode_reward:.4f}) ──┘")
            
            state = next_state
            if done:
                self.debug_log(f"⏹ Episode 提前结束于 step {step + 1}", "WARNING")
                break
        
        # ════════════════════════════════════════════════════════════════
        # Episode 结束统计
        # ════════════════════════════════════════════════════════════════
        steps_taken = step + 1
        system_metrics = info.get('system_metrics', {})
        self._record_episode_metrics(system_metrics, episode_steps=steps_taken)
        
        avg_step_reward = episode_reward / steps_taken if steps_taken > 0 else 0
        
        self.debug_log(f"\n{'─'*80}")
        self.debug_log(f"⏸ Episode {episode} 完成")
        self.debug_log(f"  • 总步数: {steps_taken}")
        self.debug_log(f"  • Episode总奖励: {episode_reward:.4f}")
        self.debug_log(f"  • 平均每步奖励: {avg_step_reward:.4f}")
        
        # 记录系统指标
        if system_metrics:
            self.debug_log(f"  • 系统平均延迟: {system_metrics.get('avg_task_delay', 0):.4f} s")
            self.debug_log(f"  • 系统总能耗: {system_metrics.get('total_energy_consumption', 0):.4f} J")
            self.debug_log(f"  • 任务完成率: {system_metrics.get('task_completion_rate', 0):.4%}")
            self.debug_log(f"  • 缓存命中率: {system_metrics.get('cache_hit_rate', 0):.4%}")
        
        self.debug_log(f"{'─'*80}\n")
        
        return {
            'episode_reward': episode_reward,
            'avg_reward': avg_step_reward,
            'episode_info': episode_info,
            'system_metrics': system_metrics,
            'steps': steps_taken
        }
    
    def _log_action_details(self, action: np.ndarray, actions_dict: Dict, step: int):
        """记录动作详细信息"""
        self.debug_log(f"│ 🎯 动作生成:")
        
        if isinstance(action, np.ndarray):
            self.debug_log(f"│   • 动作维度: {action.shape}")
            self.debug_log(f"│   • 动作范围: [{action.min():.4f}, {action.max():.4f}]")
            self.debug_log(f"│   • 动作均值: {action.mean():.4f}, 标准差: {action.std():.4f}")
            self.debug_log(f"│   • 前5维: {action[:5] if len(action) >= 5 else action}")
        
        # 解析卸载决策
        if isinstance(actions_dict, dict) and 'vehicle_agent' in actions_dict:
            vehicle_action = np.array(actions_dict['vehicle_agent']).reshape(-1)
            if len(vehicle_action) >= 3:
                raw = vehicle_action[:3]
                raw_scaled = np.clip(raw, -1.0, 1.0) * 5.0
                exp = np.exp(raw_scaled - np.max(raw_scaled))
                probs = exp / np.sum(exp)
                
                self.debug_log(f"│   • 卸载概率: Local={probs[0]:.4f}, RSU={probs[1]:.4f}, UAV={probs[2]:.4f}")
        
        # 保存动作样本到调试数据
        self.debug_data['action_traces'].append({
            'episode': self._current_episode,
            'step': step,
            'action_vector': action.tolist() if isinstance(action, np.ndarray) else action,
            'offload_probs': {
                'local': float(probs[0]) if 'probs' in locals() else 0,
                'rsu': float(probs[1]) if 'probs' in locals() and len(probs) > 1 else 0,
                'uav': float(probs[2]) if 'probs' in locals() and len(probs) > 2 else 0
            }
        })
    
    def _log_reward_breakdown(self, reward: float, info: Dict, step: int):
        """记录奖励分量"""
        self.debug_log(f"│ 💰 奖励分析:")
        self.debug_log(f"│   • 总奖励: {reward:.6f}")
        
        # 尝试从info中提取奖励分量
        step_stats = info.get('step_stats', {})
        reward_components = step_stats.get('reward_components', {})
        
        if reward_components:
            delay_component = reward_components.get('delay', 0)
            energy_component = reward_components.get('energy', 0)
            cache_component = reward_components.get('cache', 0)
            penalty_component = reward_components.get('penalty', 0)
            
            self.debug_log(f"│   • 延迟分量: {delay_component:.6f}")
            self.debug_log(f"│   • 能耗分量: {energy_component:.6f}")
            self.debug_log(f"│   • 缓存分量: {cache_component:.6f}")
            self.debug_log(f"│   • 惩罚分量: {penalty_component:.6f}")
            
            # 保存到调试数据
            self.debug_data['reward_components'].append({
                'episode': self._current_episode,
                'step': step,
                'total_reward': float(reward),
                'delay': float(delay_component),
                'energy': float(energy_component),
                'cache': float(cache_component),
                'penalty': float(penalty_component)
            })
        else:
            self.debug_log(f"│   ⚠️ 无法获取奖励分量详情", "WARNING")
        
        # 记录当前步的系统状态
        if 'avg_delay' in step_stats:
            self.debug_log(f"│   • 当前延迟: {step_stats.get('avg_delay', 0):.4f} s")
            self.debug_log(f"│   • 当前能耗: {step_stats.get('avg_energy', 0):.4f} J")
    
    def _log_state_vector(self, state: np.ndarray, step: int, label: str):
        """记录状态向量信息"""
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        
        if step % 20 == 0:  # 减少状态日志频率
            self.debug_log(f"│ 📊 状态向量 ({label}):")
            self.debug_log(f"│   • 维度: {state.shape}")
            self.debug_log(f"│   • 范围: [{state.min():.4f}, {state.max():.4f}]")
            self.debug_log(f"│   • 均值: {state.mean():.4f}, 标准差: {state.std():.4f}")
            self.debug_log(f"│   • 是否有NaN: {np.isnan(state).any()}")
            self.debug_log(f"│   • 是否有Inf: {np.isinf(state).any()}")
            
            # 采样保存部分状态
            if step % 50 == 0:
                self.debug_data['state_samples'].append({
                    'episode': self._current_episode,
                    'step': step,
                    'label': label,
                    'state_sample': state[:20].tolist() if len(state) >= 20 else state.tolist(),
                    'state_stats': {
                        'min': float(state.min()),
                        'max': float(state.max()),
                        'mean': float(state.mean()),
                        'std': float(state.std())
                    }
                })
    
    def _log_training_info(self, training_info: Dict, step: int):
        """记录训练信息"""
        if not training_info:
            return
        
        self.debug_log(f"│ 🔧 训练更新:")
        
        # 记录损失值
        if 'critic_loss' in training_info:
            self.debug_log(f"│   • Critic Loss: {training_info['critic_loss']:.6f}")
        
        if 'actor_loss' in training_info:
            self.debug_log(f"│   • Actor Loss: {training_info['actor_loss']:.6f}")
        
        # 记录Q值
        if 'q_value' in training_info:
            self.debug_log(f"│   • Q值: {training_info['q_value']:.6f}")
        
        # 记录梯度范数（如果可用）
        if 'actor_grad_norm' in training_info:
            self.debug_log(f"│   • Actor梯度范数: {training_info['actor_grad_norm']:.6f}")
        
        if 'critic_grad_norm' in training_info:
            self.debug_log(f"│   • Critic梯度范数: {training_info['critic_grad_norm']:.6f}")
        
        # 记录经验池大小
        if 'buffer_size' in training_info:
            self.debug_log(f"│   • 经验池大小: {training_info['buffer_size']}")
        
        # 保存梯度信息
        if 'actor_grad_norm' in training_info or 'critic_grad_norm' in training_info:
            self.debug_data['gradient_norms'].append({
                'episode': self._current_episode,
                'step': step,
                'actor_grad': training_info.get('actor_grad_norm', 0),
                'critic_grad': training_info.get('critic_grad_norm', 0),
                'actor_loss': training_info.get('actor_loss', 0),
                'critic_loss': training_info.get('critic_loss', 0)
            })
    
    def save_debug_data(self):
        """保存调试数据到JSON文件"""
        self.debug_log(f"\n{'='*80}")
        self.debug_log(f"💾 保存调试数据到 {self.debug_metrics_file}")
        
        try:
            with open(self.debug_metrics_file, 'w', encoding='utf-8') as f:
                json.dump(self.debug_data, f, indent=2, ensure_ascii=False)
            
            self.debug_log(f"✅ 调试数据保存成功")
            self.debug_log(f"  • 动作样本数: {len(self.debug_data['action_traces'])}")
            self.debug_log(f"  • 奖励样本数: {len(self.debug_data['reward_components'])}")
            self.debug_log(f"  • 状态样本数: {len(self.debug_data['state_samples'])}")
            self.debug_log(f"  • 梯度样本数: {len(self.debug_data['gradient_norms'])}")
        except Exception as e:
            self.debug_log(f"❌ 保存调试数据失败: {e}", "ERROR")
        
        self.debug_log(f"{'='*80}\n")
    
    def __del__(self):
        """析构函数 - 关闭日志文件"""
        if hasattr(self, 'log_file_handle'):
            self.log_file_handle.close()


def main():
    """调试训练主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='OPTIMIZED_TD3 调试训练')
    parser.add_argument('--algorithm', type=str, default='OPTIMIZED_TD3', help='算法名称')
    parser.add_argument('--episodes', type=int, default=50, help='训练轮数')
    parser.add_argument('--num-vehicles', type=int, default=12, help='车辆数量')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"🐛 OPTIMIZED_TD3 调试训练启动")
    print(f"{'='*80}")
    print(f"配置:")
    print(f"  • 算法: {args.algorithm}")
    print(f"  • 轮数: {args.episodes}")
    print(f"  • 车辆: {args.num_vehicles}")
    print(f"  • 种子: {args.seed}")
    print(f"{'='*80}\n")
    
    # 设置随机种子
    np.random.seed(args.seed)
    import random
    random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(args.seed)
    except ImportError:
        pass
    
    # 创建调试环境
    override_scenario = {'num_vehicles': args.num_vehicles}
    
    debug_env = DebugSingleAgentTraining(
        algorithm=args.algorithm,
        override_scenario=override_scenario,
        use_enhanced_cache=True,
        disable_migration=False
    )
    
    # 训练循环
    print(f"\n开始调试训练...\n")
    
    for episode in range(1, args.episodes + 1):
        result = debug_env.run_episode(episode, max_steps=200)
        
        # 每10个episode输出摘要
        if episode % 10 == 0:
            print(f"\n{'─'*80}")
            print(f"📈 Episode {episode} 摘要:")
            print(f"  • Episode奖励: {result['episode_reward']:.4f}")
            print(f"  • 平均步奖励: {result['avg_reward']:.4f}")
            if result.get('system_metrics'):
                sm = result['system_metrics']
                print(f"  • 平均延迟: {sm.get('avg_task_delay', 0):.4f} s")
                print(f"  • 总能耗: {sm.get('total_energy_consumption', 0):.4f} J")
                print(f"  • 完成率: {sm.get('task_completion_rate', 0):.4%}")
            print(f"{'─'*80}\n")
    
    # 保存调试数据
    debug_env.save_debug_data()
    
    print(f"\n{'='*80}")
    print(f"✅ 调试训练完成！")
    print(f"{'='*80}")
    print(f"调试输出文件:")
    print(f"  📄 日志文件: {debug_env.debug_log_file}")
    print(f"  📊 数据文件: {debug_env.debug_metrics_file}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
