"""
通用状态空间和动作空间定义
用于确保所有单智能体算法的一致性
"""
import numpy as np
from typing import Dict


class UnifiedStateActionSpace:
    """统一的状态和动作空间定义"""
    
    @staticmethod
    def calculate_state_dim(num_vehicles: int, num_rsus: int, num_uavs: int) -> tuple:
        """
        计算状态维度
        
        返回:
            (local_state_dim, global_state_dim, total_state_dim)
        """
        base_global_dim = 8
        task_type_feature_dim = 8  # 4个任务类型队列占比 + 4个归一化截止期裕度
        local_state_dim = num_vehicles * 5 + num_rsus * 5 + num_uavs * 5
        global_state_dim = base_global_dim + task_type_feature_dim
        total_state_dim = local_state_dim + global_state_dim
        return local_state_dim, global_state_dim, total_state_dim
    
    @staticmethod
    def calculate_action_dim(num_rsus: int, num_uavs: int) -> int:
        """
        计算连续动作维度
        
        返回:
            total_action_dim
        """
        # 3(任务分配) + num_rsus(RSU选择) + num_uavs(UAV选择) + 10(缓存/迁移/联动控制参数)
        return 3 + num_rsus + num_uavs + 10
    
    @staticmethod
    def build_global_state(node_states: Dict, system_metrics: Dict, 
                          num_vehicles: int, num_rsus: int) -> np.ndarray:
        """
        构建全局系统状态（16维：基础8维 + 任务类型8维）
        
        参数:
            node_states: 节点状态字典
            system_metrics: 系统指标字典
            num_vehicles: 车辆数量
            num_rsus: RSU数量
            
        返回:
            global_state: 8维全局状态向量
        """
        # 收集所有节点的队列信息
        all_queues = []
        for i in range(num_vehicles):
            v_state = node_states.get(f'vehicle_{i}')
            if v_state is not None and len(v_state) > 3:
                all_queues.append(v_state[3])  # 队列维度
        for i in range(num_rsus):
            r_state = node_states.get(f'rsu_{i}')
            if r_state is not None and len(r_state) > 3:
                all_queues.append(r_state[3])
        
        # 计算全局指标
        avg_queue = np.mean(all_queues) if all_queues else 0.0
        congestion_ratio = len([q for q in all_queues if q > 0.5]) / max(1, len(all_queues))
        
        # 从system_metrics获取系统级指标
        completion_rate = system_metrics.get('task_completion_rate', 0.5)
        avg_energy = system_metrics.get('total_energy_consumption', 0.0) / max(1, num_vehicles + num_rsus + 2)
        cache_hit_rate = system_metrics.get('cache_hit_rate', 0.0)
        
        # 构建全局状态基础向量
        base_features = [
            np.clip(avg_queue, 0.0, 1.0),           # 平均队列占用率
            np.clip(congestion_ratio, 0.0, 1.0),    # 拥塞节点比例
            np.clip(completion_rate, 0.0, 1.0),     # 任务完成率
            np.clip(avg_energy / 1000.0, 0.0, 1.0), # 平均能耗
            np.clip(cache_hit_rate, 0.0, 1.0),      # 缓存命中率
            0.0,  # episode进度（保留位）
            np.clip(len([q for q in all_queues if q > 0]) / max(1, len(all_queues)), 0.0, 1.0),  # 活跃节点比例
            np.clip(sum(all_queues) / max(1, len(all_queues)), 0.0, 1.0)  # 网络总负载
        ]
        
        def _to_fixed_length(values, length=4):
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif not isinstance(values, (list, tuple)):
                values = []
            values = [float(v) for v in values[:length]]
            if len(values) < length:
                values.extend([0.0] * (length - len(values)))
            return [float(np.clip(v, 0.0, 1.0)) for v in values]
        
        queue_distribution = _to_fixed_length(system_metrics.get('task_type_queue_distribution'))
        deadline_remaining = _to_fixed_length(system_metrics.get('task_type_deadline_remaining'))
        
        global_state = np.array(base_features + queue_distribution + deadline_remaining, dtype=np.float32)
        
        return global_state
    
    @staticmethod
    def build_state_vector(node_states: Dict, system_metrics: Dict,
                          num_vehicles: int, num_rsus: int, num_uavs: int,
                          state_dim: int) -> np.ndarray:
        """
        构建完整状态向量
        
        参数:
            node_states: 节点状态字典
            system_metrics: 系统指标字典
            num_vehicles, num_rsus, num_uavs: 网络拓扑参数
            state_dim: 期望的状态维度
            
        返回:
            state_vector: 完整状态向量
        """
        state_components = []
        
        # ========== 1. 局部节点状态 ==========
        
        # 车辆状态 (N×5维)
        for i in range(num_vehicles):
            vehicle_key = f'vehicle_{i}'
            if vehicle_key in node_states:
                vehicle_state = node_states[vehicle_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in vehicle_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        # RSU状态 (M×5维)
        for i in range(num_rsus):
            rsu_key = f'rsu_{i}'
            if rsu_key in node_states:
                rsu_state = node_states[rsu_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in rsu_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.0, 0.0, 0.0])
        
        # UAV状态 (K×5维)
        for i in range(num_uavs):
            uav_key = f'uav_{i}'
            if uav_key in node_states:
                uav_state = node_states[uav_key][:5]
                valid_state = [float(v) if np.isfinite(v) else 0.5 for v in uav_state]
                state_components.extend(valid_state)
            else:
                state_components.extend([0.5, 0.5, 0.5, 0.0, 0.0])
        
        # ========== 2. 全局系统状态 (8维) ==========
        global_state = UnifiedStateActionSpace.build_global_state(
            node_states, system_metrics, num_vehicles, num_rsus
        )
        state_components.extend(global_state)
        
        # ========== 3. 最终处理 ==========
        state_vector = np.array(state_components[:state_dim], dtype=np.float32)
        
        # 维度不足时补齐
        if len(state_vector) < state_dim:
            padding_needed = state_dim - len(state_vector)
            state_vector = np.pad(state_vector, (0, padding_needed), mode='constant', constant_values=0.5)
        
        # 数值安全检查
        state_vector = np.nan_to_num(state_vector, nan=0.5, posinf=1.0, neginf=0.0)
        state_vector = np.clip(state_vector, 0.0, 1.0)
        
        return state_vector
    
    @staticmethod
    def decompose_action(action: np.ndarray, num_rsus: int, num_uavs: int, action_dim: int) -> Dict[str, np.ndarray]:
        """
        将全局动作分解为各节点动作
        
        参数:
            action: 全局动作向量
            num_rsus, num_uavs: 网络拓扑参数
            action_dim: 动作维度
            
        返回:
            actions: 动作字典
        """
        actions = {}
        
        # 确保action长度足够
        if len(action) < action_dim:
            action = np.pad(action, (0, action_dim - len(action)), mode='constant')
        
        # 动态分解动作
        idx = 0
        
        # 1. 任务分配偏好（3维）
        task_allocation = action[idx:idx+3]
        idx += 3
        
        # 2. RSU选择权重（num_rsus维）
        rsu_selection = action[idx:idx+num_rsus]
        idx += num_rsus
        
        # 3. UAV选择权重（num_uavs维）
        uav_selection = action[idx:idx+num_uavs]
        idx += num_uavs
        
        # 4. 控制参数（动态维度）
        control_param_dim = max(0, action_dim - (3 + num_rsus + num_uavs))
        control_params = action[idx:idx+control_param_dim]
        
        # 构建vehicle_agent的完整动作
        actions['vehicle_agent'] = np.concatenate([
            task_allocation,
            rsu_selection,
            uav_selection,
            control_params
        ])
        
        # RSU和UAV agent的动作
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params
        
        return actions

