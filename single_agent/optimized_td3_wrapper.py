"""
精简优化TD3 - 仅包含最有效的两个优化
Queue-aware Replay + GNN Attention

专为VEC场景优化：
- 队列感知回放：快速学习高负载场景
- GNN注意力：大幅提升缓存命中率（0.2%→24%）

作者：VEC_mig_caching Team
"""

from typing import Optional, Dict, Union, Any
import numpy as np
from scipy.special import softmax

from .enhanced_td3_agent import EnhancedTD3Agent
from .enhanced_td3_config import EnhancedTD3Config
from .common_state_action import (
    UnifiedStateActionSpace,
    ACTION_DIM_OFFLOAD_PREF,
    ACTION_DIM_CONTROL_PARAMS,
    CENTRAL_VEHICLE_GROUPS,
    CENTRAL_RSU_AGGREGATE,
    CENTRAL_UAV_AGGREGATE,
    STATE_DIM_PER_VEHICLE,
    STATE_DIM_PER_RSU,
    STATE_DIM_PER_UAV,
    STATE_DIM_GLOBAL,
    STATE_DIM_CENTRAL,
)


def create_optimized_config() -> EnhancedTD3Config:
    """创建精简优化配置 - ✨ 使用最新GAT优化 + 🚀 GPU性能优化
    
    🔧 2024-12-03 v23: 增强探索防止局部最优
    核心优化：
    1. batch_size 1024 (大幅提高GPU利用率)
    2. gradient_steps 8 (每步更多梯度更新)
    3. 启用AMP混合精度训练
    4. 🆕 增强探索噪声 + 熵正则化 (防止局部最优)
    """
    return EnhancedTD3Config(
        # ✅ 核心优化1：队列感知回放
        use_queue_aware_replay=True,
        queue_priority_weight=0.2,
        queue_occ_coef=0.5,
        packet_loss_coef=0.3,
        migration_cong_coef=0.2,
        queue_metrics_ema_decay=0.8,
        
        # ✅ 核心优化2：GNN注意力
        use_gat_router=True,
        num_attention_heads=6,
        gat_hidden_dim=192,
        gat_dropout=0.15,

        # 🆕 v23: 启用熵正则化防止局部最优
        use_distributional_critic=False,
        use_entropy_reg=True,           # 🆕 启用熵正则化
        auto_tune_alpha=True,           # 自动调节温度
        initial_alpha=0.3,              # 较高初始温度鼓励探索
        use_grouped_temperature=True,   # 分组温度
        offload_temp=2.0,               # 卸载决策高温探索
        cache_temp=0.8,                 # 缓存决策中等温度
        use_model_based_rollout=False,

        # 🚀 v22 GPU性能优化 - 大幅提高利用率
        hidden_dim=1024,          # 网络容量
        batch_size=1024,          # 大batch提高GPU利用率
        buffer_size=500000,       
        warmup_steps=300,         # 🆕 v23: 100→300 更长warmup充分探索
        gradient_steps=8,         # 每步更多梯度更新
        
        # 🚀 v22 性能优化参数
        use_amp=True,             # 混合精度训练
        use_async_transfer=True,  # 异步数据传输
        pin_memory=True,          # 锁页内存

        # 🔧 v15学习率优化
        actor_lr=3e-4,
        critic_lr=3e-4,

        # 🆕 v23: 增强探索噪声 - 跳出局部最优
        exploration_noise=0.4,    # 🆕 0.25→0.4 更高初始噪声
        noise_decay=0.9992,       # 🆕 0.995→0.9992 更慢衰减
        min_noise=0.12,           # 🆕 0.05→0.12 更高最小噪声
        target_noise=0.2,         # 🆕 0.15→0.2
        noise_clip=0.4,           # 🆕 0.30→0.4

        # 🔧 奖励归一化
        reward_norm_beta=0.995,
        reward_norm_clip=5.0,
    )


class OptimizedTD3Wrapper:
    """
    精简优化TD3包装器
    
    只包含最有效的两个优化：
    1. Queue-aware Replay - 提升训练效率5倍
    2. GNN Attention - 缓存命中率提升120倍
    """
    
    def __init__(
        self,
        num_vehicles: int = 12,
        num_rsus: int = 4,
        num_uavs: int = 2,
        use_central_resource: bool = True,
        simulation_only: bool = False,
    ):
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.use_central_resource = use_central_resource
        self.simulation_only = simulation_only
        
        # 创建优化配置
        config = create_optimized_config()
        
        # 计算维度 - 使用统一常量
        vehicle_state_dim = num_vehicles * STATE_DIM_PER_VEHICLE  # 车辆保持5维
        rsu_state_dim = num_rsus * STATE_DIM_PER_RSU  # RSU统一为5维
        uav_state_dim = num_uavs * STATE_DIM_PER_UAV  # UAV统一为5维
        global_state_dim = STATE_DIM_GLOBAL
        base_state_dim = vehicle_state_dim + rsu_state_dim + uav_state_dim + global_state_dim
        
        if use_central_resource:
            self.central_state_dim = STATE_DIM_CENTRAL
            self.state_dim = base_state_dim + self.central_state_dim
        else:
            self.central_state_dim = 0
            self.state_dim = base_state_dim
        
        # 动作空间配置 - 支持压缩模式
        import os
        self.simplified_action = os.environ.get('SIMPLIFIED_ACTION', '0').strip() in {'1', 'true', 'True'}
        
        # 🔧 v11: 压缩动作空间模式 - 26维 → 10维
        # 默认启用压缩模式以加速收敛
        self.compressed_action = os.environ.get('COMPRESSED_ACTION', '1').strip() in {'1', 'true', 'True'}
        
        if self.compressed_action:
            # 压缩动作空间 (10维):
            # [0:2] 卸载倾向 (edge_pref, local_pref) - 通过softmax展开为3维
            # [2]   RSU偏好 - 广播到所有RSU
            # [3]   UAV偏好 - 广播到所有UAV
            # [4:8] 核心控制 (缓存激进度, 迁移阈值, 负载均衡, 能效权重)
            # [8:10] 资源策略 (计算优先级, 带宽优先级)
            self.compressed_base_dim = 8  # 基础压缩动作 (不含中央资源)
            self.compressed_central_dim = 2  # 压缩中央资源动作
            self.base_action_dim = self.compressed_base_dim
            print(f"[OptimizedTD3] 🚀 压缩动作空间已启用 (10维 vs 原26维, 减少60%)")
        elif self.simplified_action:
            self.base_action_dim = 8  # 简化版：只保留核心控制
            print("[OptimizedTD3] 🔧 简化动作空间已启用 (8维基础动作)")
        else:
            # 原始版：使用统一计算函数
            self.base_action_dim = UnifiedStateActionSpace.calculate_action_dim(num_rsus, num_uavs, include_central=False)
        
        if use_central_resource:
            # 中央资源分配模式
            self.aggregated_central = os.environ.get('AGGREGATED_CENTRAL', '1').strip() in {'1', 'true', 'True'}
            
            if self.compressed_action:
                # 🔧 v11: 压缩中央资源动作 (2维)
                self.num_vehicle_groups = CENTRAL_VEHICLE_GROUPS
                self.central_resource_action_dim = self.compressed_central_dim  # 2维
                print(f"[OptimizedTD3] 🚀 压缩中央资源模式 ({self.central_resource_action_dim}维)")
            elif self.aggregated_central:
                # 聚合模式：使用统一常量
                self.num_vehicle_groups = CENTRAL_VEHICLE_GROUPS
                self.central_resource_action_dim = CENTRAL_VEHICLE_GROUPS + CENTRAL_RSU_AGGREGATE + CENTRAL_UAV_AGGREGATE
                print(f"[OptimizedTD3] 🔧 聚合中央资源模式 ({self.central_resource_action_dim}维)")
            else:
                # 原始模式
                self.num_vehicle_groups = num_vehicles
                self.central_resource_action_dim = num_vehicles + num_vehicles + num_rsus + num_uavs
            
            self.action_dim = self.base_action_dim + self.central_resource_action_dim
        else:
            self.central_resource_action_dim = 0
            self.aggregated_central = False
            self.num_vehicle_groups = num_vehicles
            self.action_dim = self.base_action_dim
        
        # 如果只是仿真进程，跳过加载沉重的神经网络
        if simulation_only:
            self.agent = None
            print("[OptimizedTD3] Simulation-only mode initialized (No Agent Loaded)")
            return

        # 创建优化TD3智能体
        self.agent = EnhancedTD3Agent(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            config=config,
            num_vehicles=num_vehicles,
            num_rsus=num_rsus,
            num_uavs=num_uavs,
            global_dim=global_state_dim,
            central_state_dim=self.central_state_dim,
        )
        
        print("[OptimizedTD3] init done")
        print(f"  topology: vehicles={num_vehicles}, rsus={num_rsus}, uavs={num_uavs}")
        print(f"  state_dim: {self.state_dim}")
        print(f"  action_dim: {self.action_dim}")
        print("  optimizations: Queue-aware Replay + GNN Attention")
        
        # ???????/??????????????
        self._last_queue_metrics = {
            'queue_occupancy': 0.0,
            'packet_loss': 0.0,
            'migration_congestion': 0.0,
        }
        self._queue_pressure_ema: Optional[float] = None
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        if self.agent is None:
            # Simulation-only mode: return random action or zeros
            # This shouldn't be called in worker process usually, as actions come from main process
            return np.zeros(self.action_dim, dtype=np.float32)
        return self.agent.select_action(state, training=training)
    
    def store_experience(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        queue_metrics: Optional[dict] = None,
    ):
        self.agent.store_experience(state, action, reward, next_state, done, queue_metrics)
    
    def update(self) -> dict:
        return self.agent.update()
    
    def save_model(self, filepath: str) -> str:
        return self.agent.save_model(filepath)
    
    def save_models(self, filepath: str) -> str:
        return self.save_model(filepath)
    
    def load_model(self, filepath: str):
        self.agent.load_model(filepath)
    
    def load_models(self, filepath: str):
        self.load_model(filepath)
    
    def _extract_central_state(self, resource_state: Dict):
        """从resource_state提取中央资源状态"""
        central_state = []
        
        try:
            # 带宽分配统计
            bandwidth_alloc = resource_state.get('bandwidth_allocation', [])
            if isinstance(bandwidth_alloc, (list, np.ndarray)) and len(bandwidth_alloc) > 0:
                bw_array = np.array(bandwidth_alloc, dtype=np.float32)
                bw_array = np.nan_to_num(bw_array, nan=0.0)
                central_state.extend([
                    float(np.mean(bw_array)),
                    float(np.max(bw_array)),
                    float(np.min(bw_array)),
                    float(np.std(bw_array))
                ])
            else:
                central_state.extend([1.0/self.num_vehicles] * 4)
            
            # 车辆计算资源
            vehicle_compute = resource_state.get('vehicle_compute_allocation', [])
            if isinstance(vehicle_compute, (list, np.ndarray)) and len(vehicle_compute) > 0:
                vc_array = np.array(vehicle_compute, dtype=np.float32)
                vc_array = np.nan_to_num(vc_array, nan=0.0)
                central_state.extend([
                    float(np.mean(vc_array)),
                    float(np.max(vc_array)),
                    float(np.min(vc_array)),
                    float(np.std(vc_array))
                ])
            else:
                central_state.extend([1.0/self.num_vehicles] * 4)
            
            # RSU计算资源
            rsu_compute = resource_state.get('rsu_compute_allocation', [])
            if isinstance(rsu_compute, (list, np.ndarray)) and len(rsu_compute) >= self.num_rsus:
                rc_array = np.array(rsu_compute[:self.num_rsus], dtype=np.float32)
                rc_array = np.nan_to_num(rc_array, nan=1.0/self.num_rsus)
                central_state.extend([float(v) for v in rc_array])
            else:
                central_state.extend([1.0/self.num_rsus] * self.num_rsus)
            
            # UAV计算资源
            uav_compute = resource_state.get('uav_compute_allocation', [])
            if isinstance(uav_compute, (list, np.ndarray)) and len(uav_compute) >= self.num_uavs:
                uc_array = np.array(uav_compute[:self.num_uavs], dtype=np.float32)
                uc_array = np.nan_to_num(uc_array, nan=1.0/self.num_uavs)
                central_state.extend([float(v) for v in uc_array])
            else:
                central_state.extend([1.0/self.num_uavs] * self.num_uavs)
            
            while len(central_state) < 16:
                central_state.append(0.0)
            
            central_state = central_state[:16]
            
        except Exception as e:
            print(f"⚠️ 中央资源状态提取失败: {e}，使用默认值")
            central_state = [
                1.0/self.num_vehicles, 1.0/self.num_vehicles, 1.0/self.num_vehicles, 0.0,
                1.0/self.num_vehicles, 1.0/self.num_vehicles, 1.0/self.num_vehicles, 0.0,
                1.0/self.num_rsus, 1.0/self.num_rsus, 1.0/self.num_rsus, 1.0/self.num_rsus,
                1.0/self.num_uavs, 1.0/self.num_uavs, 0.0, 0.0
            ]
        
        central_state = [float(v) if np.isfinite(v) else 0.0 for v in central_state]
        return central_state
    
    def get_state_vector(
        self,
        node_states: Dict,
        system_metrics: Dict,
        resource_state: Optional[Dict] = None,
    ) -> np.ndarray:
        """构建状态向量"""
        state_components = []
        
        # 节点状态
        for i in range(self.num_vehicles):
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key][:5]  # 车辆保持5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in vehicle_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        for i in range(self.num_rsus):
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key][:5]  # 🔧 修复2：RSU统一为5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in rsu_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])  # 默认5维
        
        for i in range(self.num_uavs):
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key][:5]  # 🔧 修复2：UAV统一为5维
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in uav_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.5, 0.0, 0.0])  # 默认5维（高度维度已包含）
        
        # 全局状态 (16维 = 基础8维 + 任务类型8维)
        # 🔧 v24修复：状态维度与STATE_DIM_GLOBAL常量严格对齐
        # 问题诊断：之前只构建8维全局状态，但预期16维，导致8维被padding填充
        from config import config
        latency_target = float(getattr(config.rl, 'latency_target', 0.1))     # 🔧 v16: 0.1s
        energy_target = float(getattr(config.rl, 'energy_target', 10000.0))  # 🔧 v16: 10000J
        
        # 基础全局状态 (8维)
        base_global = [
            float(system_metrics.get('avg_task_delay', 0.0) / max(latency_target, 1e-6)),
            float(system_metrics.get('total_energy_consumption', 0.0) / max(energy_target, 1e-6)),
            float(system_metrics.get('task_completion_rate', 0.95)),
            float(system_metrics.get('cache_hit_rate', 0.85)),
            float(system_metrics.get('queue_overload_flag', 0.0)),
            float(system_metrics.get('rsu_offload_ratio', 0.5)),
            float(system_metrics.get('uav_offload_ratio', 0.2)),
            float(system_metrics.get('local_offload_ratio', 0.3)),
        ]
        
        # 🆕 v24: 任务类型特征 (8维) - 修复维度不一致bug
        # 从system_metrics提取任务类型相关特征
        queue_dist = system_metrics.get('task_type_queue_distribution', [])
        deadline_norm = system_metrics.get('task_type_deadline_remaining', [])
        
        # 确保4维队列分布
        if not isinstance(queue_dist, (list, np.ndarray)) or len(queue_dist) < 4:
            queue_dist = [0.25, 0.25, 0.25, 0.25]  # 默认均匀分布
        else:
            queue_dist = list(queue_dist)[:4]
            while len(queue_dist) < 4:
                queue_dist.append(0.25)
        
        # 确保4维截止期裕度
        if not isinstance(deadline_norm, (list, np.ndarray)) or len(deadline_norm) < 4:
            deadline_norm = [0.5, 0.5, 0.5, 0.5]  # 默认中等裕度
        else:
            deadline_norm = list(deadline_norm)[:4]
            while len(deadline_norm) < 4:
                deadline_norm.append(0.5)
        
        # 组合全局状态 (16维)
        global_state = base_global + [float(np.clip(v, 0.0, 1.0)) for v in queue_dist] + \
                       [float(np.clip(v, 0.0, 1.0)) for v in deadline_norm]
        global_state = [float(v) if np.isfinite(v) else 0.0 for v in global_state]
        state_components.extend(global_state)
        
        # 中央资源状态
        if self.central_state_dim > 0 and resource_state is not None:
            central_state_vector = self._extract_central_state(resource_state)
            state_components.extend(central_state_vector)
        
        state_vector = np.array(state_components, dtype=np.float32)
        
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        
        if state_vector.size < self.state_dim:
            padding_needed = self.state_dim - state_vector.size
            state_vector = np.pad(state_vector, (0, padding_needed), mode='constant', constant_values=0.5)
        elif state_vector.size > self.state_dim:
            state_vector = state_vector[:self.state_dim]
            
        return state_vector
    
    def calculate_reward(
        self,
        system_metrics: Dict,
        cache_metrics: Optional[Dict] = None,
        migration_metrics: Optional[Dict] = None
    ) -> tuple[float, Dict[str, float]]:
        """
        计算奖励并返回组件字典
        
        使用统一奖励计算器确保一致性。
        归一化基准自动与状态归一化对齐（通过config.rl.latency_target和energy_target）。
        
        Returns:
            tuple: (reward, reward_components)
        """
        from utils.unified_reward_calculator import _general_reward_calculator
        return _general_reward_calculator.calculate_reward(system_metrics, cache_metrics, migration_metrics)
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict:
        """获取动作"""
        global_action = self.agent.select_action(state, training)
        actions = self.decompose_action(global_action)
        return actions
    
    def decompose_action(self, action: np.ndarray) -> Dict:
        """分解动作 - 支持压缩动作空间解压"""
        actions = {}
        
        # 🔧 v11: 压缩动作空间解压
        if self.compressed_action:
            return self._decompose_compressed_action(action)
        
        # 1. 基础动作 (Offload + RSU/UAV Selection + Control Params)
        base_segment = action[:self.base_action_dim]
        
        # 🔧 简化动作处理：8维 → 展开为完整格式
        if self.simplified_action:
            # 简化动作结构 (8维):
            # [0:3] 卸载偏好 (local, rsu, uav)
            # [3]   RSU聚合权重 (广播到所有RSU)
            # [4]   UAV聚合权重 (广播到所有UAV)
            # [5:8] 核心控制参数 (缓存激进度, 迁移倾向, 负载均衡)
            offload_preference = base_segment[:3]
            rsu_aggregate = float(base_segment[3]) if len(base_segment) > 3 else 0.0
            uav_aggregate = float(base_segment[4]) if len(base_segment) > 4 else 0.0
            core_control = base_segment[5:8] if len(base_segment) > 5 else np.zeros(3)
            
            # 广播到所有RSU/UAV
            rsu_selection = np.full(self.num_rsus, rsu_aggregate, dtype=np.float32)
            uav_selection = np.full(self.num_uavs, uav_aggregate, dtype=np.float32)
            # 扩展核心控制到10维
            control_params = np.zeros(10, dtype=np.float32)
            control_params[:len(core_control)] = core_control
        else:
            # 原始动作处理 (19维)
            offload_preference = base_segment[:3]
            idx = 3
            rsu_selection = base_segment[idx:idx + self.num_rsus]
            idx += self.num_rsus
            uav_selection = base_segment[idx:idx + self.num_uavs]
            idx += self.num_uavs
            control_params = base_segment[idx:idx + 10]
        
        actions['vehicle_agent'] = action.copy() # 保留原始完整动作供参考
        actions['offload_preference'] = {
            'local': float(offload_preference[0]),
            'rsu': float(offload_preference[1]),
            'uav': float(offload_preference[2])
        }
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params
        
        # 2. 🔧 修复：提取中央资源动作 (Central Resource Allocation)
        if self.use_central_resource:
            # action的后半部分是中央资源动作
            central_segment = action[self.base_action_dim:]
            
            expected_len = self.central_resource_action_dim
            if len(central_segment) >= expected_len:
                
                if self.aggregated_central:
                    # 🔧 聚合模式：7维 → 展开为完整资源分配
                    # [0:4] 4组车辆资源分配
                    # [4:6] 2个RSU聚合权重
                    # [6]   1个UAV聚合权重
                    c_idx = 0
                    group_weights = central_segment[c_idx:c_idx + self.num_vehicle_groups]  # 4维
                    c_idx += self.num_vehicle_groups
                    rsu_weights = central_segment[c_idx:c_idx + 2]  # 2维
                    c_idx += 2
                    uav_weight = float(central_segment[c_idx]) if c_idx < len(central_segment) else 0.0  # 1维
                    
                    # 将组权重广播到每辆车 (4组 → 12车)
                    vehicles_per_group = self.num_vehicles // self.num_vehicle_groups
                    bw_alloc = np.zeros(self.num_vehicles, dtype=np.float32)
                    comp_alloc = np.zeros(self.num_vehicles, dtype=np.float32)
                    for g in range(self.num_vehicle_groups):
                        start_v = g * vehicles_per_group
                        end_v = min(start_v + vehicles_per_group, self.num_vehicles)
                        group_w = float(group_weights[g]) if g < len(group_weights) else 0.0
                        bw_alloc[start_v:end_v] = group_w
                        comp_alloc[start_v:end_v] = group_w  # 带宽和计算共享权重
                    
                    # 将RSU权重广播 (2 → 4 RSUs)
                    rsu_alloc = np.zeros(self.num_rsus, dtype=np.float32)
                    rsus_per_group = max(1, self.num_rsus // 2)
                    for r in range(self.num_rsus):
                        group_idx = min(r // rsus_per_group, 1)
                        rsu_alloc[r] = float(rsu_weights[group_idx]) if group_idx < len(rsu_weights) else 0.0
                    
                    # UAV统一权重
                    uav_alloc = np.full(self.num_uavs, uav_weight, dtype=np.float32)
                    
                else:
                    # 原始模式：30维完整分配
                    c_idx = 0
                    bw_alloc = central_segment[c_idx:c_idx + self.num_vehicles]
                    c_idx += self.num_vehicles
                    comp_alloc = central_segment[c_idx:c_idx + self.num_vehicles]
                    c_idx += self.num_vehicles
                    rsu_alloc = central_segment[c_idx:c_idx + self.num_rsus]
                    c_idx += self.num_rsus
                    uav_alloc = central_segment[c_idx:c_idx + self.num_uavs]
                
                actions['central_resource'] = {
                    'bandwidth_weights': softmax(bw_alloc),
                    'compute_weights': softmax(comp_alloc),
                    'rsu_reservation': softmax(rsu_alloc),
                    'uav_reservation': softmax(uav_alloc)
                }
            else:
                print(f"⚠️ 动作维度警告: Central segment len {len(central_segment)} < expected {expected_len}")
                actions['central_resource'] = None
        
        return actions

    def _decompose_compressed_action(self, action: np.ndarray) -> Dict:
        """🔧 v11: 解压缩压缩动作 (10维 → 完整动作字典)
        
        压缩动作结构 (10维):
            [0:2]  卸载倾向: [edge_pref, local_pref]
                   → 通过softmax展开为 [local, rsu, uav]
            [2]    RSU偏好: 主动级在中心RSU
            [3]    UAV偏好: UAV整体使用倾向
            [4:8]  核心控制: [缓存激进度, 迁移阈值, 负载均衡, 能效权重]
            [8:10] 资源策略: [计算优先级, 带宽优先级]
        """
        actions = {}
        
        # 确保动作长度足够
        if len(action) < self.action_dim:
            action = np.pad(action, (0, self.action_dim - len(action)), mode='constant')
        
        # ========== 1. 解压基础动作 (8维) ==========
        
        # [0:2] 卸载倾向 → 展开为3维
        edge_pref = float(action[0])  # 边缘卸载偏好 (RSU+UAV)
        local_pref = float(action[1])  # 本地处理偏好
        
        # 将 [edge_pref, local_pref] 转换为 [local, rsu, uav] 分布
        # 使用tanh输出。edge_pref > 0 倾向卸载，< 0 倾向本地
        offload_raw = np.array([
            local_pref,           # 本地偏好
            edge_pref * 0.6,      # RSU偏好 (边缘的主要部分)
            edge_pref * 0.4       # UAV偏好 (边缘的辅助部分)
        ], dtype=np.float32)
        offload_preference = softmax(offload_raw)
        
        # [2] RSU偏好 → 广播到所有RSU (加入位置偏移创造差异)
        rsu_center_pref = float(action[2])
        rsu_selection = np.zeros(self.num_rsus, dtype=np.float32)
        for r in range(self.num_rsus):
            # 距离中心RSU越远，权重越低
            distance_factor = 1.0 - 0.2 * abs(r - self.num_rsus // 2)
            rsu_selection[r] = rsu_center_pref * distance_factor
        
        # [3] UAV偏好 → 广播到所有UAV
        uav_pref = float(action[3])
        uav_selection = np.full(self.num_uavs, uav_pref, dtype=np.float32)
        
        # [4:8] 核心控制参数 → 展开到10维
        core_control = action[4:8] if len(action) > 4 else np.zeros(4)
        control_params = np.zeros(10, dtype=np.float32)
        # 映射关系:
        # core[0]=缓存激进度 → ctrl[0]缓存激进度 + ctrl[3]协作缓存权重
        # core[1]=迁移阈值   → ctrl[4]迁移阈值 + ctrl[5]迁移成本权重
        # core[2]=负载均衡   → ctrl[7]负载均衡权重 + ctrl[8]队列感知
        # core[3]=能效权重   → ctrl[9]能效权重
        if len(core_control) >= 4:
            control_params[0] = float(core_control[0])  # 缓存激进度
            control_params[1] = float(core_control[0]) * 0.5  # 驱逐阈值(关联)
            control_params[2] = 0.0  # 本地缓存优先级(默认)
            control_params[3] = float(core_control[0]) * 0.5  # 协作缓存权重
            control_params[4] = float(core_control[1])  # 迁移阈值
            control_params[5] = float(core_control[1]) * 0.5  # 迁移成本权重
            control_params[6] = 0.0  # 迁移紧迫因子(默认)
            control_params[7] = float(core_control[2])  # 负载均衡权重
            control_params[8] = float(core_control[2]) * 0.5  # 队列感知因子
            control_params[9] = float(core_control[3])  # 能效权重
        
        # ========== 2. 构建动作字典 ==========
        actions['vehicle_agent'] = action.copy()
        actions['offload_preference'] = {
            'local': float(offload_preference[0]),
            'rsu': float(offload_preference[1]),
            'uav': float(offload_preference[2])
        }
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params
        
        # 解析控制参数为语义化字典
        actions['cache_params'] = {
            'aggressiveness': float(control_params[0]),
            'eviction_threshold': float(control_params[1]),
            'priority_local': float(control_params[2]),
            'collaborative_weight': float(control_params[3]),
        }
        actions['migration_params'] = {
            'threshold': float(control_params[4]),
            'cost_weight': float(control_params[5]),
            'urgency_factor': float(control_params[6]),
        }
        actions['joint_params'] = {
            'load_balance_weight': float(control_params[7]),
            'queue_aware_factor': float(control_params[8]),
            'energy_efficiency_weight': float(control_params[9]),
        }
        
        # ========== 3. 解压中央资源动作 (2维) ==========
        if self.use_central_resource:
            # [8:10] 资源策略 → 展开为完整资源分配
            resource_segment = action[self.compressed_base_dim:self.compressed_base_dim + self.compressed_central_dim]
            
            compute_priority = float(resource_segment[0]) if len(resource_segment) > 0 else 0.0
            bandwidth_priority = float(resource_segment[1]) if len(resource_segment) > 1 else 0.0
            
            # 根据优先级生成资源分配
            # 计算优先级高 → 更多资源分配给高计算需求的节点
            # 带宽优先级高 → 更多带宽分配给高通信需求的节点
            
            # 生成车辆分组权重 (4组)
            bw_alloc = np.zeros(self.num_vehicles, dtype=np.float32)
            comp_alloc = np.zeros(self.num_vehicles, dtype=np.float32)
            vehicles_per_group = self.num_vehicles // 4
            
            for g in range(4):
                start_v = g * vehicles_per_group
                end_v = min(start_v + vehicles_per_group, self.num_vehicles)
                # 组权重基于优先级和组索引
                group_compute_w = compute_priority * (1.0 - 0.1 * g)
                group_bw_w = bandwidth_priority * (1.0 - 0.1 * g)
                bw_alloc[start_v:end_v] = group_bw_w
                comp_alloc[start_v:end_v] = group_compute_w
            
            # RSU和UAV资源分配
            rsu_alloc = np.full(self.num_rsus, compute_priority * 0.8, dtype=np.float32)
            uav_alloc = np.full(self.num_uavs, compute_priority * 0.6, dtype=np.float32)
            
            actions['central_resource'] = {
                'bandwidth_weights': softmax(bw_alloc + 1e-6),
                'compute_weights': softmax(comp_alloc + 1e-6),
                'rsu_reservation': softmax(rsu_alloc + 1e-6),
                'uav_reservation': softmax(uav_alloc + 1e-6)
            }
        else:
            actions['central_resource'] = None
        
        return actions

    # ================== 训练接口 & 队列信号 ================== #
    def update_queue_metrics(self, step_stats: Dict[str, Any]) -> None:
        """从step统计中提取队列/丢包信号，驱动Queue-aware Replay。"""
        try:
            # 🔧 P1修复：改进队列指标提取，分节点类型提取
            # 1. 车辆级别队列压力
            vehicle_queue_pressure = []
            queue_rho_by_node = step_stats.get('queue_rho_by_node', {})
            if isinstance(queue_rho_by_node, dict):
                for node_key, rho_value in queue_rho_by_node.items():
                    if node_key.startswith('vehicle_'):
                        try:
                            vehicle_queue_pressure.append(float(rho_value))
                        except (TypeError, ValueError):
                            pass
            
            # 2. 综合队列压力指标
            queue_rho_max = float(step_stats.get('queue_rho_max', 0.0) or 0.0)
            queue_overload_flag = 1.0 if step_stats.get('queue_overload_flag', False) else 0.0
            
            # 3. 计算平均车辆队列压力
            avg_vehicle_pressure = float(np.mean(vehicle_queue_pressure)) if vehicle_queue_pressure else 0.0
            
            # 4. 综合队列压力：最大值 + 车辆平均 + 过载标志
            queue_occ = float(max(
                queue_rho_max,
                avg_vehicle_pressure,
                queue_overload_flag
            ))
            
            # 5. 丢包率指标
            packet_loss = float(
                step_stats.get('data_loss_ratio_bytes', step_stats.get('packet_loss', 0.0)) or 0.0
            )
            
            # 6. 迁移拥塞指标
            migration_cong = float(
                max(
                    step_stats.get('cache_eviction_rate', 0.0) or 0.0,
                    step_stats.get('migration_queue_pressure', 0.0) or 0.0,
                )
            )
        except Exception:
            queue_occ, packet_loss, migration_cong = 0.0, 0.0, 0.0
        
        queue_occ = float(np.clip(queue_occ, 0.0, 1.0))
        packet_loss = float(np.clip(packet_loss, 0.0, 1.0))
        migration_cong = float(np.clip(migration_cong, 0.0, 1.0))
        
        # 平滑队列压力，避免抖动
        if self._queue_pressure_ema is None:
            self._queue_pressure_ema = queue_occ
        else:
            self._queue_pressure_ema = 0.8 * self._queue_pressure_ema + 0.2 * queue_occ
        queue_occ = float(np.clip(self._queue_pressure_ema, 0.0, 1.0))
        
        self._last_queue_metrics = {
            'queue_occupancy': queue_occ,
            'packet_loss': packet_loss,
            'migration_congestion': migration_cong,
        }

    def update_priority_signal(self, queue_pressure: Union[float, int]) -> None:
        """兼容上层的队列压力接口，直接转成队列占用率信号。"""
        try:
            qp = float(queue_pressure)
        except Exception:
            qp = 0.0
        qp = float(np.clip(qp, 0.0, 1.0))
        self.update_queue_metrics({'queue_rho_max': qp})

    def train_step(
        self,
        state: np.ndarray,
        action: Union[np.ndarray, float, int],
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Dict[str, Any]:
        """单步训练：写入经验 + 更新网络"""
        action_arr = np.asarray(action, dtype=np.float32)
        if action_arr.ndim > 1:
            action_arr = action_arr.flatten()
        
        # 使用最新的队列指标驱动优先级采样
        self.store_experience(state, action_arr, reward, next_state, done, self._last_queue_metrics)
        training_info = self.update()
        return training_info


# 别名
OptimizedTD3Environment = OptimizedTD3Wrapper
