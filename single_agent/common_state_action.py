"""
通用状态空间和动作空间定义
用于确保所有单智能体算法的一致性

=== 动作空间结构 (OPTIMIZED_TD3, 26维) ===

基础动作 (19维):
  [0:3]   卸载偏好 (3维): [local_pref, rsu_pref, uav_pref]
          - 通过softmax转换为概率分布，决定任务分配到本地/RSU/UAV的倾向
  
  [3:7]   RSU选择权重 (4维): 每个RSU的选择分数
          - 通过softmax转换为概率，决定卸载到哪个RSU
  
  [7:9]   UAV选择权重 (2维): 每个UAV的选择分数
          - 通过softmax转换为概率，决定卸载到哪个UAV
  
  [9:19]  联动控制参数 (10维):
          [9:13]  缓存控制参数 (4维):
                  - [9]  cache_aggressiveness: 缓存激进度 [-1,1] -> 控制缓存预取的积极程度
                  - [10] cache_eviction_threshold: 驱逐阈值 [-1,1] -> 转换为[0.3,0.9]
                  - [11] cache_priority_local: 本地缓存优先级 [-1,1] -> 转换为[0,1]
                  - [12] cache_collaborative_weight: 协作缓存权重 [-1,1] -> 转换为[0,1]
          
          [13:16] 迁移控制参数 (3维):
                  - [13] migration_threshold: 迁移触发阈值 [-1,1] -> 转换为[0.4,0.8]
                  - [14] migration_cost_weight: 迁移成本权重 [-1,1] -> 转换为[0.1,0.9]
                  - [15] migration_urgency_factor: 迁移紧迫因子 [-1,1] -> 转换为[0,1]
          
          [16:19] 联合策略参数 (3维):
                  - [16] load_balance_weight: 负载均衡权重 [-1,1] -> 转换为[0,1]
                  - [17] queue_aware_factor: 队列感知因子 [-1,1] -> 转换为[0,1]
                  - [18] energy_efficiency_weight: 能效权重 [-1,1] -> 转换为[0,1]

中央资源分配动作 (7维, AGGREGATED_CENTRAL=1 时启用):
  [19:23] 车辆分组资源 (4维): 将12辆车分为4组，每组共享资源分配
  [23:25] RSU资源聚合 (2维): RSU资源分配聚合指标
  [25:26] UAV资源聚合 (1维): UAV资源分配聚合指标

=== 状态空间结构 (114维, use_central_resource=True) ===

节点状态 (90维):
  车辆: 12 × 5维 = 60维 [pos_x, pos_y, velocity, queue_util, energy]
  RSU:   4 × 5维 = 20维 [pos_x, pos_y, cache_util, queue_util, energy]
  UAV:   2 × 5维 = 10维 [pos_x, pos_y, queue_util, cache_util, energy]

全局状态 (8维):
  [0] 平均队列占用率
  [1] 拥塞节点比例
  [2] 任务完成率
  [3] 平均能耗
  [4] 缓存命中率
  [5] episode进度
  [6] 活跃节点比例
  [7] 网络总负载

中央资源状态 (16维):
  [0:4]   带宽分配统计 (mean, max, min, std)
  [4:8]   车辆计算资源统计 (mean, max, min, std)
  [8:12]  RSU计算资源 (4个RSU各一个值)
  [12:14] UAV计算资源 (2个UAV各一个值)
  [14:16] 保留位
"""
import numpy as np
from typing import Dict, Tuple, List


# =============================================================================
# 动作空间常量定义
# =============================================================================

# 基础动作维度
ACTION_DIM_OFFLOAD_PREF = 3      # 卸载偏好 [local, rsu, uav]
ACTION_DIM_CONTROL_PARAMS = 10   # 联动控制参数

# 控制参数索引 (相对于控制参数段的开始位置)
CTRL_CACHE_AGGRESSIVENESS = 0    # 缓存激进度
CTRL_CACHE_EVICTION_THRESH = 1   # 驱逐阈值
CTRL_CACHE_PRIORITY_LOCAL = 2    # 本地缓存优先级
CTRL_CACHE_COLLAB_WEIGHT = 3     # 协作缓存权重
CTRL_MIG_THRESHOLD = 4           # 迁移阈值
CTRL_MIG_COST_WEIGHT = 5         # 迁移成本权重
CTRL_MIG_URGENCY = 6             # 迁移紧迫因子
CTRL_LOAD_BALANCE = 7            # 负载均衡权重
CTRL_QUEUE_AWARE = 8             # 队列感知因子
CTRL_ENERGY_EFFICIENCY = 9       # 能效权重

# 中央资源动作维度 (聚合模式)
CENTRAL_VEHICLE_GROUPS = 4       # 车辆分组数
CENTRAL_RSU_AGGREGATE = 2        # RSU聚合维度
CENTRAL_UAV_AGGREGATE = 1        # UAV聚合维度


# =============================================================================
# 状态空间常量定义
# =============================================================================

STATE_DIM_PER_VEHICLE = 5        # 每辆车的状态维度
STATE_DIM_PER_RSU = 5            # 每RSU的状态维度
STATE_DIM_PER_UAV = 5            # 每UAV的状态维度
STATE_DIM_GLOBAL = 8             # 全局状态维度
STATE_DIM_CENTRAL = 16           # 中央资源状态维度


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
    def calculate_action_dim(num_rsus: int, num_uavs: int, include_central: bool = False) -> int:
        """
        计算连续动作维度
        
        参数:
            num_rsus: RSU数量
            num_uavs: UAV数量
            include_central: 是否包含中央资源分配动作
            
        返回:
            total_action_dim: 总动作维度
            
        动作空间结构:
            - 卸载偏好 (3维): [local, rsu, uav]
            - RSU选择权重 (num_rsus维)
            - UAV选择权重 (num_uavs维)
            - 联动控制参数 (10维): 缓存(4) + 迁移(3) + 联合(3)
            - [可选] 中央资源分配 (7维): 车辆分组(4) + RSU聚合(2) + UAV聚合(1)
        """
        base_dim = ACTION_DIM_OFFLOAD_PREF + num_rsus + num_uavs + ACTION_DIM_CONTROL_PARAMS
        
        if include_central:
            central_dim = CENTRAL_VEHICLE_GROUPS + CENTRAL_RSU_AGGREGATE + CENTRAL_UAV_AGGREGATE
            return base_dim + central_dim
        
        return base_dim
    
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
            num_rsus: RSU数量
            num_uavs: UAV数量
            action_dim: 动作维度
            
        返回:
            actions: 动作字典
                - vehicle_agent: 完整动作向量
                - rsu_agent: RSU选择权重
                - uav_agent: UAV选择权重
                - control_params: 联动控制参数
                - offload_preference: 卸载偏好 [local, rsu, uav]
                - cache_params: 缓存控制参数字典
                - migration_params: 迁移控制参数字典
                - joint_params: 联合策略参数字典
        """
        actions = {}
        
        # 确保action长度足够
        if len(action) < action_dim:
            action = np.pad(action, (0, action_dim - len(action)), mode='constant')
        
        # 动态分解动作
        idx = 0
        
        # 1. 卸载偏好（3维）
        offload_preference = action[idx:idx+ACTION_DIM_OFFLOAD_PREF]
        idx += ACTION_DIM_OFFLOAD_PREF
        
        # 2. RSU选择权重（num_rsus维）
        rsu_selection = action[idx:idx+num_rsus]
        idx += num_rsus
        
        # 3. UAV选择权重（num_uavs维）
        uav_selection = action[idx:idx+num_uavs]
        idx += num_uavs
        
        # 4. 控制参数（10维）
        control_params = action[idx:idx+ACTION_DIM_CONTROL_PARAMS]
        if len(control_params) < ACTION_DIM_CONTROL_PARAMS:
            control_params = np.pad(control_params, (0, ACTION_DIM_CONTROL_PARAMS - len(control_params)))
        
        # 构建vehicle_agent的完整动作
        actions['vehicle_agent'] = np.concatenate([
            offload_preference,
            rsu_selection,
            uav_selection,
            control_params
        ])
        
        # RSU和UAV agent的动作
        actions['rsu_agent'] = rsu_selection
        actions['uav_agent'] = uav_selection
        actions['control_params'] = control_params
        actions['offload_preference'] = offload_preference
        
        # 解析控制参数为语义化字典
        actions['cache_params'] = {
            'aggressiveness': float(control_params[CTRL_CACHE_AGGRESSIVENESS]),
            'eviction_threshold': float(control_params[CTRL_CACHE_EVICTION_THRESH]),
            'priority_local': float(control_params[CTRL_CACHE_PRIORITY_LOCAL]),
            'collaborative_weight': float(control_params[CTRL_CACHE_COLLAB_WEIGHT]),
        }
        
        actions['migration_params'] = {
            'threshold': float(control_params[CTRL_MIG_THRESHOLD]),
            'cost_weight': float(control_params[CTRL_MIG_COST_WEIGHT]),
            'urgency_factor': float(control_params[CTRL_MIG_URGENCY]),
        }
        
        actions['joint_params'] = {
            'load_balance_weight': float(control_params[CTRL_LOAD_BALANCE]),
            'queue_aware_factor': float(control_params[CTRL_QUEUE_AWARE]),
            'energy_efficiency_weight': float(control_params[CTRL_ENERGY_EFFICIENCY]),
        }
        
        return actions
    
    @staticmethod
    def convert_control_param(value: float, target_range: Tuple[float, float] = (0.0, 1.0)) -> float:
        """
        将[-1, 1]范围的控制参数转换到目标范围
        
        参数:
            value: 原始值 [-1, 1]
            target_range: 目标范围 (min, max)
            
        返回:
            转换后的值
        """
        normalized = (value + 1.0) / 2.0  # [-1,1] -> [0,1]
        low, high = target_range
        return low + normalized * (high - low)
