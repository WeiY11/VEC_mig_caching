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
    """创建精简优化配置 - ✨ 使用最新GAT优化
    
    🔧 2024-12-02 v3修复：增强探索+学习率优化
    核心修复：
    1. 增加初始探索噪声 0.15 → 0.25 (更强的初始探索)
    2. 加快噪声衰减 0.9995 → 0.999 (更快收敛)
    3. 提高Critic学习率 3e-4 → 5e-4 (加快值函数学习)
    4. 增加梯度更新次数
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

        # ❌ 禁用其他优化
        use_distributional_critic=False,
        use_entropy_reg=False,
        use_model_based_rollout=False,

        # 🔧 基础参数优化
        hidden_dim=384,
        batch_size=512,
        buffer_size=100000,
        warmup_steps=3000,    # 🔧 5000 → 3000 (更快开始学习)

        # 🔧 学习率优化 - 加快Critic学习
        actor_lr=1e-4,
        critic_lr=5e-4,       # 🔧 3e-4 → 5e-4 (加快Q网络学习)

        # 🔧 探索噪声优化 - 更强的初始探索，更快的衰减
        exploration_noise=0.25,   # 🔧 0.15 → 0.25 (更强的初始探索)
        noise_decay=0.999,        # 🔧 0.9995 → 0.999 (更快衰减)
        min_noise=0.03,           # 🔧 0.02 → 0.03 (保持最低探索)
        target_noise=0.05,        # 🔧 0.03 → 0.05 (适当的目标噪声)
        noise_clip=0.15,          # 🔧 0.1 → 0.15 (增大裁剪范围)

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
        
        # 动作空间配置 - 使用统一常量
        import os
        self.simplified_action = os.environ.get('SIMPLIFIED_ACTION', '0').strip() in {'1', 'true', 'True'}
        if self.simplified_action:
            self.base_action_dim = 8  # 简化版：只保留核心控制
            print("[OptimizedTD3] 🔧 简化动作空间已启用 (8维基础动作)")
        else:
            # 原始版：使用统一计算函数
            self.base_action_dim = UnifiedStateActionSpace.calculate_action_dim(num_rsus, num_uavs, include_central=False)
        
        if use_central_resource:
            # 中央资源分配模式
            self.aggregated_central = os.environ.get('AGGREGATED_CENTRAL', '1').strip() in {'1', 'true', 'True'}
            
            if self.aggregated_central:
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
        
        # 全局状态
        # 🔧 P0修复：归一化因子必须与UnifiedRewardCalculator严格对齐
        # 从配置读取目标值，确保状态归一化与奖励计算使用相同基准
        # 🔧 2024-12-02 修复：默认值对齐实际系统性能
        from config import config
        latency_target = float(getattr(config.rl, 'latency_target', 1.5))     # 🔧 0.4 → 1.5 (对齐实际延迟)
        energy_target = float(getattr(config.rl, 'energy_target', 1000.0))    # 🔧 3500 → 1000 (对齐实际能耗)
        
        global_state = [
            float(system_metrics.get('avg_task_delay', 0.0) / max(latency_target, 1e-6)),
            float(system_metrics.get('total_energy_consumption', 0.0) / max(energy_target, 1e-6)),
            float(system_metrics.get('task_completion_rate', 0.95)),
            float(system_metrics.get('cache_hit_rate', 0.85)),
            float(system_metrics.get('queue_overload_flag', 0.0)),
            float(system_metrics.get('rsu_offload_ratio', 0.5)),
            float(system_metrics.get('uav_offload_ratio', 0.2)),
            float(system_metrics.get('local_offload_ratio', 0.3)),
        ]
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
        """分解动作"""
        actions = {}
        
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
