"""
增强型TD3环境包装器

为EnhancedTD3Agent创建兼容train_single_agent.py的环境接口。
允许在训练脚本中无缝使用所有5项高级优化。

用法:
    在train_single_agent.py中:
    if algorithm == "ENHANCED_TD3":
        from single_agent.enhanced_td3_wrapper import EnhancedTD3Wrapper
        agent_env = EnhancedTD3Wrapper(num_vehicles, num_rsus, num_uavs, use_all_features=True)

作者：VEC_mig_caching Team
"""

from typing import Optional, Dict, List
import numpy as np

from .enhanced_td3_agent import EnhancedTD3Agent
from .enhanced_td3_config import (
    EnhancedTD3Config,
    create_full_enhanced_config,
    create_queue_focused_config
)


class EnhancedTD3Wrapper:
    """
    EnhancedTD3的环境包装器
    
    提供与TD3Environment相同的接口，但内部使用EnhancedTD3Agent
    """
    
    def __init__(
        self,
        num_vehicles: int = 12,
        num_rsus: int = 4,
        num_uavs: int = 2,
        use_central_resource: bool = True,
        use_all_features: bool = True,
        config_preset: str = 'full',  # 'full', 'queue_focused', 'baseline'
    ):
        """
        Args:
            num_vehicles: 车辆数量
            num_rsus: RSU数量
            num_uavs: UAV数量
            use_central_resource: 是否使用中央资源分配
            use_all_features: 是否启用所有5项优化
            config_preset: 配置预设 ('full', 'queue_focused', 'baseline')
        """
        self.num_vehicles = num_vehicles
        self.num_rsus = num_rsus
        self.num_uavs = num_uavs
        self.use_central_resource = use_central_resource
        
        # 创建配置
        if config_preset == 'full':
            config = create_full_enhanced_config()
        elif config_preset == 'queue_focused':
            config = create_queue_focused_config()
        else:
            config = EnhancedTD3Config()  # baseline
            if use_all_features:
                config.use_distributional_critic = True
                config.use_entropy_reg = True
                config.use_model_based_rollout = True
                config.use_queue_aware_replay = True
                config.use_gat_router = True
        
        # 计算状态和动作维度
        # 车辆状态：每车5维
        vehicle_state_dim = num_vehicles * 5
        # RSU状态：每RSU 5维
        rsu_state_dim = num_rsus * 5
        # UAV状态：每UAV 5维
        uav_state_dim = num_uavs * 5
        # 全局状态：8维
        global_state_dim = 8
        
        # 基础状态维度
        base_state_dim = vehicle_state_dim + rsu_state_dim + uav_state_dim + global_state_dim
        
        # 如果启用中央资源，增加中央资源状态维度
        if use_central_resource:
            self.central_state_dim = 16  # 资源池状态
            self.state_dim = base_state_dim  # 实际上不需要增加，因为中央资源状态是分开处理的
        else:
            self.central_state_dim = 0
            self.state_dim = base_state_dim
        
        # 动作维度：3(任务分配) + num_rsus(RSU选择) + num_uavs(UAV选择) + 10(控制参数)
        self.base_action_dim = 3 + num_rsus + num_uavs + 10
        
        # 如果启用中央资源，增加动作维度
        if use_central_resource:
            # 中央资源动作：车辆带宽 + 车辆计算 + RSU计算 + UAV计算
            self.central_resource_action_dim = num_vehicles + num_vehicles + num_rsus + num_uavs
            self.action_dim = self.base_action_dim + self.central_resource_action_dim
        else:
            self.central_resource_action_dim = 0
            self.action_dim = self.base_action_dim
        
        # 创建EnhancedTD3Agent
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
        
        print(f"[EnhancedTD3Wrapper] 初始化完成")
        print(f"  拓扑: {num_vehicles}车辆, {num_rsus}RSU, {num_uavs}UAV")
        print(f"  状态维度: {self.state_dim}")
        print(f"  动作维度: {self.action_dim}")
        print(f"  中央资源: {use_central_resource}")
        print(f"  配置预设: {config_preset}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> np.ndarray:
        """
        选择动作
        
        Args:
            state: 状态向量
            training: 是否训练模式
            
        Returns:
            action: 动作向量
        """
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
        """
        存储经验
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否结束
            queue_metrics: 队列指标（可选）
        """
        self.agent.store_experience(state, action, reward, next_state, done, queue_metrics)
    
    def update(self) -> dict:
        """
        更新网络参数
        
        Returns:
            training_info: 训练信息字典
        """
        return self.agent.update()
    
    def save_model(self, filepath: str) -> str:
        """保存模型"""
        return self.agent.save_model(filepath)
    
    def save_models(self, filepath: str) -> str:
        """保存模型（兼容方法）"""
        return self.save_model(filepath)
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.agent.load_model(filepath)
    
    def load_models(self, filepath: str):
        """加载模型（兼容方法）"""
        self.load_model(filepath)
    
    def _extract_central_state(self, resource_state: Dict) -> List[float]:
        """
        从resource_state中提取中央资源分配状态向量
        
        资源状态包括:
        - bandwidth_allocation: 带宽分配 (12维，每个车辆)
        - vehicle_compute_allocation: 车辆计算资源分配 (12维)
        - rsu_compute_allocation: RSU计算资源分配 (4维)
        - uav_compute_allocation: UAV计算资源分配 (2维)
        
        总计: 12 + 12 + 4 + 2 = 30维，但我们的central_state_dim=16
        所以需要聚合压缩
        
        Args:
            resource_state: 资源状态字典
            
        Returns:
            central_state_vector: 16维中央资源状态向量
        """
        central_state = []
        
        try:
            # 1. 车辆带宽分配统计 (4维): 均值、最大、最小、标准差
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
                central_state.extend([1.0/self.num_vehicles] * 4)  # 均匀分配
            
            # 2. 车辆计算资源分配统计 (4维)
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
            
            # 3. RSU计算资源分配 (4维，直接使用原始值)
            rsu_compute = resource_state.get('rsu_compute_allocation', [])
            if isinstance(rsu_compute, (list, np.ndarray)) and len(rsu_compute) >= self.num_rsus:
                rc_array = np.array(rsu_compute[:self.num_rsus], dtype=np.float32)
                rc_array = np.nan_to_num(rc_array, nan=1.0/self.num_rsus)
                central_state.extend([float(v) for v in rc_array])
            else:
                central_state.extend([1.0/self.num_rsus] * self.num_rsus)
            
            # 4. UAV计算资源分配 (4维: 2个真实值 + 2个填充)
            uav_compute = resource_state.get('uav_compute_allocation', [])
            if isinstance(uav_compute, (list, np.ndarray)) and len(uav_compute) >= self.num_uavs:
                uc_array = np.array(uav_compute[:self.num_uavs], dtype=np.float32)
                uc_array = np.nan_to_num(uc_array, nan=1.0/self.num_uavs)
                central_state.extend([float(v) for v in uc_array])
            else:
                central_state.extend([1.0/self.num_uavs] * self.num_uavs)
            
            # 补充到4维 (如果UAV < 4个)
            while len(central_state) < 16:
                central_state.append(0.0)
            
            # 确保正好16维
            central_state = central_state[:16]
            
        except Exception as e:
            # 如果提取失败，返回默认均匀分配状态
            print(f"⚠️ 中央资源状态提取失败: {e}，使用默认值")
            # 默认值：所有资源均匀分配
            central_state = [
                # 带宽统计 (4维)
                1.0/self.num_vehicles, 1.0/self.num_vehicles, 1.0/self.num_vehicles, 0.0,
                # 车辆计算统计 (4维)
                1.0/self.num_vehicles, 1.0/self.num_vehicles, 1.0/self.num_vehicles, 0.0,
                # RSU计算 (4维)
                1.0/self.num_rsus, 1.0/self.num_rsus, 1.0/self.num_rsus, 1.0/self.num_rsus,
                # UAV计算 (4维)
                1.0/self.num_uavs, 1.0/self.num_uavs, 0.0, 0.0
            ]
        
        # 最终验证
        central_state = [float(v) if np.isfinite(v) else 0.0 for v in central_state]
        
        return central_state
    
    def get_state_vector(
        self,
        node_states: Dict,
        system_metrics: Dict,
        resource_state: Optional[Dict] = None,
    ) -> np.ndarray:
        """
        构建状态向量
        
        Args:
            node_states: 节点状态字典
            system_metrics: 系统指标字典
            resource_state: 资源状态（可选）
            
        Returns:
            state_vector: 状态向量
        """
        state_components = []
        
        # 1. 节点状态 (车辆 + RSU + UAV)
        for i in range(self.num_vehicles):
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in vehicle_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        for i in range(self.num_rsus):
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in rsu_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        for i in range(self.num_uavs):
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in uav_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.5, 0.0, 0.0])
        
        # 2. 全局系统状态 (8维)
        global_state = [
            float(system_metrics.get('avg_task_delay', 0.0) / 1.0),
            float(system_metrics.get('total_energy_consumption', 0.0) / 1000.0),
            float(system_metrics.get('task_completion_rate', 0.95)),  # 使用正确的键名
            float(system_metrics.get('cache_hit_rate', 0.85)),
            float(system_metrics.get('queue_overload_flag', 0.0)),
            float(system_metrics.get('rsu_offload_ratio', 0.5)),
            float(system_metrics.get('uav_offload_ratio', 0.2)),
            float(system_metrics.get('local_offload_ratio', 0.3)),
        ]
        # 确保全局状态值有效
        global_state = [float(v) if np.isfinite(v) else 0.0 for v in global_state]
        state_components.extend(global_state)
        
        # 🎯 3. 中央资源状态（如果启用，添加16维资源分配信息）
        if self.central_state_dim > 0 and resource_state is not None:
            central_state_vector = self._extract_central_state(resource_state)
            state_components.extend(central_state_vector)
        
        # 转换为numpy数组
        state_vector = np.array(state_components, dtype=np.float32)
        
        # 检查并处理NaN值
        if np.any(np.isnan(state_vector)) or np.any(np.isinf(state_vector)):
            state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        
        # 维度对齐
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
    ) -> float:
        """
        计算奖励
        
        Args:
            system_metrics: 系统指标
            cache_metrics: 缓存指标
            migration_metrics: 迁移指标
            
        Returns:
            reward: 奖励值
        """
        from utils.unified_reward_calculator import calculate_unified_reward
        return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, algorithm="general")
    
    def get_actions(self, state: np.ndarray, training: bool = True) -> Dict:
        """
        获取动作
        
        Args:
            state: 状态向量
            training: 是否训练模式
            
        Returns:
            actions: 动作字典
        """
        global_action = self.agent.select_action(state, training)
        actions = self.decompose_action(global_action)
        return actions
    
    def decompose_action(self, action: np.ndarray) -> Dict:
        """
        将全局动作分解为各节点动作
        
        Args:
            action: 全局动作向量
            
        Returns:
            actions: 分解后的动作字典
        """
        actions = {}
        idx = 0
        
        # 基础动作段
        base_segment = action[:self.base_action_dim]
        
        # 任务分配偏好 (3维)
        offload_preference = base_segment[:3]
        idx = 3
        
        # RSU选择 (num_rsus维)
        rsu_selection = base_segment[idx:idx + self.num_rsus]
        idx += self.num_rsus
        
        # UAV选择 (num_uavs维)
        uav_selection = base_segment[idx:idx + self.num_uavs]
        idx += self.num_uavs
        
        # 控制参数 (10维)
        control_params = base_segment[idx:idx + 10]
        
        actions['vehicle_agent'] = action.copy()
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params
        
        return actions


# 为了向后兼容，创建别名
EnhancedTD3Environment = EnhancedTD3Wrapper
EnhancedCAMTD3Environment = EnhancedTD3Wrapper  # CAM_TD3增强版使用相同的wrapper
