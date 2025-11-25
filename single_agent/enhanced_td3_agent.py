"""
增强型TD3智能体 - 集成5项高级优化

整合了以下优化技术：
1. 队列约束的分布式Critic（QR-DQN风格）
2. 带熵正则的SAC特性（自适应温度）
3. 模型化队列预测（Dreamer/MBPO风格）
4. 队列感知的优先经验回放
5. GNN路由器的聚合特性（GAT风格）

所有优化均可通过配置参数独立启用/禁用。

作者：VEC_mig_caching Team
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import os

# 导入基础TD3组件
from .td3 import TD3Actor, TD3Critic, GraphFeatureExtractor

# 导入增强组件
from .enhanced_td3_config import EnhancedTD3Config
from .quantile_critic import DistributionalCritic
from .queue_aware_replay import QueueAwareReplayBuffer
from .queue_dynamics_model import QueueDynamicsModel, ModelBasedRollout, ModelTrainer
from .gat_router import GATRouterActor


class EnhancedTD3Agent:
    """
    增强型TD3智能体
    
    相比标准TD3，增加了5项可选优化：
    1. 分布式Critic - 抑制尾部时延
    2. 熵正则化 - 维持探索
    3. 模型化队列预测 - 加速收敛
    4. 队列感知回放 - 智能采样
    5. GAT路由器 - 协同缓存
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        config: EnhancedTD3Config,
        num_vehicles: Optional[int] = None,
        num_rsus: Optional[int] = None,
        num_uavs: Optional[int] = None,
        global_dim: int = 8,
        central_state_dim: Optional[int] = None,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # 拓扑信息
        self.num_vehicles = num_vehicles or 12
        self.num_rsus = num_rsus or 4
        self.num_uavs = num_uavs or 2
        self.global_dim = global_dim
        self.central_state_dim = central_state_dim or 0
        
        # ========== 构建Actor网络 ==========
        if config.use_gat_router:
            # 使用GAT路由器
            print("[EnhancedTD3] 使用GAT路由器构建Actor")
            self.graph_encoder = GATRouterActor(
                num_vehicles=self.num_vehicles,
                num_rsus=self.num_rsus,
                num_uavs=self.num_uavs,
                global_feature_dim=self.global_dim,
                hidden_dim=config.gat_hidden_dim,
                num_heads=config.num_attention_heads,
                edge_feature_dim=config.edge_feature_dim,
                central_state_dim=self.central_state_dim,  # 添加中央状态维度
            ).to(self.device)
            actor_input_dim = config.gat_hidden_dim
        else:
            # 使用标准图编码器，传入central_dim参数
            # 🎯 修复: 让GraphFeatureExtractor处理中央资源状态
            self.graph_encoder = GraphFeatureExtractor(
                num_vehicles=self.num_vehicles,
                num_rsus=self.num_rsus,
                num_uavs=self.num_uavs,
                embed_dim=config.graph_embed_dim,
                central_dim=self.central_state_dim,  # 添加中央资源维度
            ).to(self.device)
            # GraphFeatureExtractor输出已包含中央资源编码
            actor_input_dim = self.graph_encoder.output_dim
        
        # Actor主网络（不再需要手动添加central_state_dim）
        # 🎯 修复: 直接使用graph_encoder的输出维度
        self.actor = nn.Sequential(
            nn.Linear(actor_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, action_dim),
            nn.Tanh(),
        ).to(self.device)
        
        # Target Actor
        self.target_graph_encoder = self._clone_network(self.graph_encoder)
        self.target_actor = self._clone_network(self.actor)
        
        # ========== 构建Critic网络 ==========
        if config.use_distributional_critic:
            # 使用分布式Critic
            print(f"[EnhancedTD3] 使用分布式Critic (n_quantiles={config.n_quantiles})")
            self.critic = DistributionalCritic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config.hidden_dim,
                n_quantiles=config.n_quantiles,
                quantile_embedding_dim=config.quantile_embedding_dim,
                kappa=config.quantile_kappa,
            ).to(self.device)
            self.target_critic = self._clone_network(self.critic)
        else:
            # 使用标准Twin Critic
            self.critic = TD3Critic(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dim=config.hidden_dim,
            ).to(self.device)
            self.target_critic = self._clone_network(self.critic)
        
        # ========== 优化器 ==========
        self.actor_optimizer = optim.Adam(
            list(self.graph_encoder.parameters()) + list(self.actor.parameters()),
            lr=config.actor_lr
        )
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        
        # ========== 熵正则化 ==========
        if config.use_entropy_reg:
            print(f"[EnhancedTD3] 启用熵正则化 (initial_alpha={config.initial_alpha})")
            self.use_entropy_reg = True
            self.log_alpha = torch.tensor(
                np.log(config.initial_alpha), requires_grad=True, device=self.device
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.target_entropy = -config.target_entropy_ratio * action_dim
            self.auto_tune_alpha = config.auto_tune_alpha
        else:
            self.use_entropy_reg = False
            self.log_alpha = None
        
        # ========== 经验回放缓冲区 ==========
        if config.use_queue_aware_replay:
            print(f"[EnhancedTD3] 使用队列感知回放 (queue_priority_weight={config.queue_priority_weight})")
            self.replay_buffer = QueueAwareReplayBuffer(
                capacity=config.buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                alpha=config.alpha,
                queue_priority_weight=config.queue_priority_weight,
                queue_metrics_ema_decay=config.queue_metrics_ema_decay,
            )
        else:
            # 使用标准优先回放（从td3.py导入）
            from .td3 import TD3ReplayBuffer
            self.replay_buffer = TD3ReplayBuffer(
                capacity=config.buffer_size,
                state_dim=state_dim,
                action_dim=action_dim,
                alpha=config.alpha,
            )
        
        # ========== 模型化队列预测 ==========
        if config.use_model_based_rollout:
            print(f"[EnhancedTD3] 启用模型化队列预测 (rollout_horizon={config.rollout_horizon})")
            self.use_model_based = True
            self.dynamics_model = QueueDynamicsModel(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_dims=config.model_hidden_dims,
            ).to(self.device)
            self.model_rollout = ModelBasedRollout(
                dynamics_model=self.dynamics_model,
                rollout_horizon=config.rollout_horizon,
                imagined_reward_weight=config.imagined_reward_weight,
                overflow_penalty=config.overflow_penalty,
                device=self.device,
            )
            self.model_trainer = ModelTrainer(
                model=self.dynamics_model,
                learning_rate=config.model_lr,
                batch_size=config.batch_size,
                train_iterations=config.model_train_iterations,
                device=self.device,
            )
            self.model_train_freq = config.model_train_freq
            self.model_step_count = 0
        else:
            self.use_model_based = False
            self.dynamics_model = None
        
        # ========== PER参数 ==========
        self.beta = config.beta_start
        self.beta_increment = config.beta_increment
        
        # ========== 探索噪声 ==========
        self.exploration_noise = config.exploration_noise
        self.step_count = 0
        self.update_count = 0
        
        # ========== 训练统计 ==========
        self.actor_losses = []
        self.critic_losses = []
        self.entropy_values = []
        self.alpha_values = []
        
        print(f"[EnhancedTD3] 初始化完成")
        print(f"  - 分布式Critic: {config.use_distributional_critic}")
        print(f"  - 熵正则化: {config.use_entropy_reg}")
        print(f"  - 模型化预测: {config.use_model_based_rollout}")
        print(f"  - 队列感知回放: {config.use_queue_aware_replay}")
        print(f"  - GAT路由器: {config.use_gat_router}")
    
    def _clone_network(self, network: nn.Module) -> nn.Module:
        """克隆网络用于创建target网络"""
        import copy
        clone = copy.deepcopy(network)
        clone.to(self.device)
        return clone
    
    @property
    def alpha(self) -> float:
        """获取当前熵温度参数"""
        if self.use_entropy_reg:
            return self.log_alpha.exp().item()
        return 0.0
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """选择动作
        
        🎯 修复: 状态向量已经包含中央资源状态（来自EnhancedTD3Wrapper）
        不再需要手动添加全零中央状态
        
        Args:
            state: 状态向量（已包含中央资源状态）
            training: 是否训练模式
        
        Returns:
            action: 动作向量
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # 直接使用状态，不需要手动添加中央状态
            encoded_state = self.graph_encoder(state_tensor)
            
            # 生成动作
            action_tensor = self.actor(encoded_state)
        
        action = action_tensor.cpu().numpy()[0]
        
        # 添加探索噪声
        if training:
            noise = np.random.normal(0, self.exploration_noise, size=action.shape)
            action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        queue_metrics: Optional[Dict[str, float]] = None,
    ):
        """存储经验"""
        if self.config.use_queue_aware_replay:
            # 队列感知回放需要队列指标
            self.replay_buffer.push(state, action, reward, next_state, done, queue_metrics)
        else:
            # 标准回放
            self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """更新网络参数"""
        if len(self.replay_buffer) < self.config.batch_size:
            return {}
        
        self.step_count += 1
        
        # 预热期不更新
        if self.step_count < self.config.warmup_steps:
            return {}
        
        self.update_count += 1
        
        # 采样经验批次
        batch = self.replay_buffer.sample(self.config.batch_size, self.beta)
        states, actions, rewards, next_states, dones, indices, weights = batch
        
        # 移动到设备
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        
        # 更新beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # ========== 更新Critic ==========
        critic_loss, td_errors = self._update_critic(
            states, actions, rewards, next_states, dones, weights
        )
        
        # 更新优先级
        self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        training_info = {'critic_loss': critic_loss}
        
        # ========== 延迟策略更新 ==========
        if self.update_count % self.config.policy_delay == 0:
            actor_loss, entropy_info = self._update_actor(states)
            training_info['actor_loss'] = actor_loss
            training_info.update(entropy_info)
            
            # 软更新目标网络
            self._soft_update(self.target_graph_encoder, self.graph_encoder, self.config.tau)
            self._soft_update(self.target_actor, self.actor, self.config.tau)
            self._soft_update(self.target_critic, self.critic, self.config.tau)
        
        # ========== 模型化队列预测 ==========
        if self.use_model_based:
            self.model_step_count += 1
            
            # 定期训练动态模型
            if self.model_step_count % self.model_train_freq == 0:
                model_stats = self.model_trainer.train(
                    self.replay_buffer,
                    min_buffer_size=self.config.min_model_buffer_size
                )
                training_info.update({f'model_{k}': v for k, v in model_stats.items()})
                
                # 生成合成transitions
                if len(self.replay_buffer) >= self.config.min_model_buffer_size:
                    self._generate_synthetic_data()
        
        # 衰减噪声
        self.exploration_noise = max(
            self.config.min_noise,
            self.exploration_noise * self.config.noise_decay
        )
        training_info['exploration_noise'] = self.exploration_noise
        
        return training_info
    
    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """更新Critic网络"""
        if self.config.use_distributional_critic:
            return self._update_distributional_critic(
                states, actions, rewards, next_states, dones, weights
            )
        else:
            return self._update_standard_critic(
                states, actions, rewards, next_states, dones, weights
            )
    
    def _update_distributional_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """更新分布式Critic"""
        with torch.no_grad():
            # 生成目标动作
            next_encoded = self.target_graph_encoder(next_states)
            next_actions = self.target_actor(next_encoded)
            
            # 添加目标噪声
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            # 获取目标分位数Q值
            target_q1_quantiles, target_q2_quantiles = self.target_critic(next_states, next_actions)
            target_q_quantiles = torch.min(target_q1_quantiles, target_q2_quantiles)
            
            # Bootstrap
            target_quantiles = rewards + (1 - dones) * self.config.gamma * target_q_quantiles
        
        #  计算损失
        loss, td_errors = self.critic.compute_loss(
            states, actions, target_quantiles, weights
        )
        
        # 反向传播
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.config.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.gradient_clip_norm
            )
        self.critic_optimizer.step()
        
        self.critic_losses.append(loss.item())
        return loss.item(), td_errors
    
    def _update_standard_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> Tuple[float, torch.Tensor]:
        """更新标准Twin Critic"""
        with torch.no_grad():
            next_encoded = self.target_graph_encoder(next_states)
            next_actions = self.target_actor(next_encoded)
            
            noise = torch.randn_like(next_actions) * self.config.target_noise
            noise = torch.clamp(noise, -self.config.noise_clip, self.config.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1.0, 1.0)
            
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.config.gamma * target_q
        
        current_q1, current_q2 = self.critic(states, actions)
        td_errors_q1 = current_q1 - target_q
        td_errors_q2 = current_q2 - target_q
        
        critic_loss = (weights * td_errors_q1.pow(2)).mean() + (weights * td_errors_q2.pow(2)).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if self.config.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.config.gradient_clip_norm
            )
        self.critic_optimizer.step()
        
        self.critic_losses.append(critic_loss.item())
        td_errors = td_errors_q1.detach().abs().squeeze()
        return critic_loss.item(), td_errors
    
    def _update_actor(self, states: torch.Tensor) -> Tuple[float, Dict[str, float]]:
        """更新Actor网络"""
        # 直接编码状态，不需要手动添加中央状态
        # 🎯 修复: 状态向量已经包含中央资源状态（来自EnhancedTD3Wrapper）
        encoded_states = self.graph_encoder(states)
        
        # 生成动作
        actions = self.actor(encoded_states)
        
        # 计算Q值
        if self.config.use_distributional_critic:
            q_values = self.critic.q1(states, actions)
        else:
            q_values, _ = self.critic(states, actions)
            q_values = q_values[:, :1]  # 只用Q1
        
        actor_loss = -q_values.mean()
        
        entropy_info = {}
        
        # ========== 熵正则化 ==========
        if self.use_entropy_reg:
            # 简单估计：基于动作方差
            action_std = actions.std(dim=0).mean()
            entropy = torch.log(action_std + 1e-6)
            
            self.entropy_values.append(entropy.item())
            entropy_info['entropy'] = entropy.item()
            
            # 添加熵bonus
            actor_loss = actor_loss - self.alpha * entropy
            
            # 自动调节温度
            if self.auto_tune_alpha:
                alpha_loss = -(self.log_alpha * (entropy - self.target_entropy).detach()).mean()
                
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                
                self.alpha_values.append(self.alpha)
                entropy_info['alpha'] = self.alpha
                entropy_info['alpha_loss'] = alpha_loss.item()
        
        # 更新Actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.config.use_gradient_clip:
            torch.nn.utils.clip_grad_norm_(
                list(self.graph_encoder.parameters()) + list(self.actor.parameters()),
                self.config.gradient_clip_norm
            )
        self.actor_optimizer.step()
        
        self.actor_losses.append(actor_loss.item())
        return actor_loss.item(), entropy_info
    
    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float):
        """软更新目标网络"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
    
    def _generate_synthetic_data(self):
        """使用模型生成合成数据"""
        # 从replay buffer采样真实状态
        batch_size = min(self.config.rollout_batch_size, len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        real_states = torch.FloatTensor(self.replay_buffer.states[indices])
        
        # 执行rollout
        synthetic_transitions = self.model_rollout.generate_synthetic_transitions(
            real_states,
            self.actor,
            num_rollouts_per_state=self.config.num_rollouts_per_state,
        )
        
        # 将合成transitions加入replay buffer
        for s, a, r, s_next, done in synthetic_transitions:
            # 注意：合成数据可能需要降低优先级或标记
            self.store_experience(s, a, r, s_next, done, queue_metrics=None)
    
    def save_model(self, filepath: str) -> str:
        """保存模型"""
        save_dict = {
            'graph_encoder_state_dict': self.graph_encoder.state_dict(),
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'target_graph_encoder_state_dict': self.target_graph_encoder.state_dict(),
            'target_actor_state_dict': self.target_actor.state_dict(),
            'target_critic_state_dict': self.target_critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'exploration_noise': self.exploration_noise,
            'step_count': self.step_count,
            'update_count': self.update_count,
        }
        
        if self.use_entropy_reg:
            save_dict['log_alpha'] = self.log_alpha
            save_dict['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
        
        if self.use_model_based:
            save_dict['dynamics_model_state_dict'] = self.dynamics_model.state_dict()
        
        os.makedirs(os.path.dirname(filepath) or '.', exist_ok=True)
        torch.save(save_dict, filepath)
        print(f"[EnhancedTD3] 模型已保存: {filepath}")
        return filepath
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.graph_encoder.load_state_dict(checkpoint['graph_encoder_state_dict'])
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.target_graph_encoder.load_state_dict(checkpoint['target_graph_encoder_state_dict'])
        self.target_actor.load_state_dict(checkpoint['target_actor_state_dict'])
        self.target_critic.load_state_dict(checkpoint['target_critic_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        
        if 'log_alpha' in checkpoint and self.use_entropy_reg:
            self.log_alpha = checkpoint['log_alpha']
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
        
        if 'dynamics_model_state_dict' in checkpoint and self.use_model_based:
            self.dynamics_model.load_state_dict(checkpoint['dynamics_model_state_dict'])
        
        self.exploration_noise = checkpoint['exploration_noise']
        self.step_count = checkpoint['step_count']
        self.update_count = checkpoint['update_count']
        
        print(f"[EnhancedTD3] 模型已加载: {filepath}")
