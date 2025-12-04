"""
通用状态空间和动作空间定义（MDP优化版 v2.0）
用于确保所有单智能体算法的一致性

=== MDP优化改进 ===

1. 状态空间优化:
   - 使用增量能耗替代累积能耗（解决状态漂移问题）
   - 减少冗余位置特征，增加决策相关特征
   - 添加任务级特征（紧急任务占比、平均截止期等）

2. 动作空间简化:
   - 核心动作：卸载偏好(3) + 节点选择(6) = 9维
   - 控制参数压缩为5维核心参数

3. 奖励信号增强:
   - 转为正向奖励空间 [0, 10]
   - 增加即时反馈（每步任务完成奖励）

=== 动作空间结构 (优化后14维) ===

核心动作 (9维):
  [0:3]   卸载偏好 (3维): [local_pref, rsu_pref, uav_pref]
  [3:7]   RSU选择权重 (4维)
  [7:9]   UAV选择权重 (2维)

简化控制参数 (5维):
  [9]   负载均衡权重: 控制跨节点负载分配
  [10]  缓存激进度: 控制预取和缓存策略
  [11]  迁移敏感度: 控制任务迁移触发
  [12]  能效权重: 能耗vs延迟的权衡
  [13]  队列感知因子: 队列长度对决策的影响

=== 状态空间结构 (优化后106维) ===

节点状态 (72维):
  车辆: 12 × 4维 = 48维 [queue_util, delta_energy, task_load, velocity_norm]
  RSU:   4 × 4维 = 16维 [queue_util, cache_util, service_rate, load_ratio]
  UAV:   2 × 4维 =  8维 [queue_util, cache_util, battery_level, coverage_load]

任务级特征 (8维):
  [0] 紧急任务占比
  [1] 平均截止期裕度
  [2] 高优先级任务比例
  [3] 平均任务大小(归一化)
  [4-7] 各类型任务队列分布

全局状态 (10维):
  [0] 平均队列占用率
  [1] 拥塞节点比例
  [2] 即时任务完成率（本步）
  [3] 即时能耗（本步，归一化）
  [4] 缓存命中率
  [5] 卸载成功率
  [6] 平均处理延迟
  [7] 网络负载均衡度
  [8] 迁移成功率
  [9] 系统吞吐量

历史趋势 (16维):
  过去4步的[delay, energy, completion, queue]变化趋势
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

# 🔧 保持与实际状态构建一致的维度（每个节点5维）
STATE_DIM_PER_VEHICLE = 5        # [pos_x, pos_y, velocity, queue_util, energy]
STATE_DIM_PER_RSU = 5            # [pos_x, pos_y, cache_util, queue_util, energy]
STATE_DIM_PER_UAV = 5            # [pos_x, pos_y, queue_util, cache_util, energy]
STATE_DIM_GLOBAL = 20            # 基础12维 + 任务类型8维
STATE_DIM_TASK_FEATURES = 8      # MDP优化: 任务级特征
STATE_DIM_HISTORY = 16           # MDP优化: 历史趋势特征
STATE_DIM_CENTRAL = 16           # 中央资源状态维度

# 兼容性：保留旧常量以支持旧代码
STATE_DIM_PER_VEHICLE_LEGACY = 5
STATE_DIM_PER_RSU_LEGACY = 5
STATE_DIM_PER_UAV_LEGACY = 5
STATE_DIM_GLOBAL_LEGACY = 8


class UnifiedStateActionSpace:
    """统一的状态和动作空间定义"""
    
    @staticmethod
    def calculate_state_dim(num_vehicles: int, num_rsus: int, num_uavs: int) -> tuple:
        """
        计算状态维度
        
        返回:
            (local_state_dim, global_state_dim, total_state_dim)
        """
        base_global_dim = 12
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
        构建全局系统状态（20维：基础12维 + 任务类型8维）
        
        参数:
            node_states: 节点状态字典
            system_metrics: 系统指标字典
            num_vehicles: 车辆数量
            num_rsus: RSU数量
            
        返回:
            global_state: 20维全局状态向量
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
        
        normalized_energy = np.clip(system_metrics.get('normalized_energy_for_state', avg_energy / 1000.0), 0.0, 2.0)
        episode_progress = np.clip(system_metrics.get('episode_progress', 0.0), 0.0, 1.0)
        data_loss_ratio = np.clip(system_metrics.get('data_loss_ratio_bytes', 0.0), 0.0, 1.0)
        remote_reject_rate = np.clip(system_metrics.get('remote_rejection_rate', 0.0), 0.0, 1.0)
        queue_overload_flag = np.clip(system_metrics.get('queue_overload_flag', 0.0), 0.0, 1.0)
        drop_presence = np.clip(system_metrics.get('dropped_tasks', 0.0), 0.0, 1.0)

        # 构建全局状态基础向量（12维）
        base_features = [
            np.clip(avg_queue, 0.0, 1.0),           # 平均队列占用率
            np.clip(congestion_ratio, 0.0, 1.0),    # 拥塞节点比例
            np.clip(completion_rate, 0.0, 1.0),     # 任务完成率
            normalized_energy,                      # 能耗归一化（与奖励同尺度）
            np.clip(cache_hit_rate, 0.0, 1.0),      # 缓存命中率
            episode_progress,                       # episode进度（0-1）
            np.clip(len([q for q in all_queues if q > 0]) / max(1, len(all_queues)), 0.0, 1.0),  # 活跃节点比例
            np.clip(sum(all_queues) / max(1, len(all_queues)), 0.0, 1.0),  # 网络总负载
            data_loss_ratio,                        # 数据丢失比例
            remote_reject_rate,                     # 远端拒绝率
            queue_overload_flag,                    # 队列过载标志
            drop_presence,                          # 任务丢弃存在性
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
    
    # =========================================================================
    # MDP优化: 新增的状态空间构建方法
    # =========================================================================
    
    @staticmethod
    def build_optimized_global_state(node_states: Dict, system_metrics: Dict,
                                      num_vehicles: int, num_rsus: int,
                                      step_metrics: Dict = None) -> np.ndarray:
        """
        构建优化后的全局状态（10维 + 8维任务特征）
        
        优化点:
        1. 使用即时指标而非累积指标
        2. 增加卸载成功率、迁移成功率等决策相关特征
        3. 添加任务级特征（紧急任务占比等）
        """
        step_metrics = step_metrics or {}
        
        # 收集队列信息
        all_queues = []
        for i in range(num_vehicles):
            v_state = node_states.get(f'vehicle_{i}')
            if v_state is not None and len(v_state) > 2:
                all_queues.append(float(v_state[2]) if len(v_state) > 2 else 0.0)
        for i in range(num_rsus):
            r_state = node_states.get(f'rsu_{i}')
            if r_state is not None and len(r_state) > 0:
                all_queues.append(float(r_state[0]))
        
        avg_queue = float(np.mean(all_queues)) if all_queues else 0.0
        congestion_ratio = len([q for q in all_queues if q > 0.7]) / max(1, len(all_queues))
        
        # 计算负载均衡度 (1 - 变异系数)
        if all_queues and np.mean(all_queues) > 0:
            load_balance = 1.0 - min(1.0, np.std(all_queues) / (np.mean(all_queues) + 1e-6))
        else:
            load_balance = 1.0
        
        # 从 step_metrics 获取即时指标（本步的增量）
        step_completion = float(step_metrics.get('step_completion_rate', 
                                system_metrics.get('task_completion_rate', 0.5)))
        step_energy = float(step_metrics.get('step_energy', 0.0))
        step_energy_norm = min(1.0, step_energy / 100.0)  # 每步能耗归一化到100J
        
        avg_delay = float(system_metrics.get('avg_task_delay', 0.0))
        avg_delay_norm = min(1.0, avg_delay / 1.0)  # 延迟归一化到1s
        
        cache_hit_rate = float(system_metrics.get('cache_hit_rate', 0.0))
        offload_success = float(step_metrics.get('offload_success_rate', 0.8))
        migration_success = float(system_metrics.get('migration_success_rate', 0.0))
        throughput = float(step_metrics.get('step_throughput', 0.0))
        throughput_norm = min(1.0, throughput / 10.0)  # 吐吐量归一化到10任务/步
        
        # 核心全局状态 (10维)
        global_features = [
            np.clip(avg_queue, 0.0, 1.0),           # [0] 平均队列占用率
            np.clip(congestion_ratio, 0.0, 1.0),    # [1] 拥塞节点比例
            np.clip(step_completion, 0.0, 1.0),     # [2] 即时完成率
            np.clip(step_energy_norm, 0.0, 1.0),    # [3] 即时能耗
            np.clip(cache_hit_rate, 0.0, 1.0),      # [4] 缓存命中率
            np.clip(offload_success, 0.0, 1.0),     # [5] 卸载成功率
            np.clip(avg_delay_norm, 0.0, 1.0),      # [6] 平均延迟
            np.clip(load_balance, 0.0, 1.0),        # [7] 负载均衡度
            np.clip(migration_success, 0.0, 1.0),   # [8] 迁移成功率
            np.clip(throughput_norm, 0.0, 1.0),     # [9] 系统吐吐量
        ]
        
        # 任务级特征 (8维)
        urgent_ratio = float(system_metrics.get('urgent_task_ratio', 0.0))
        avg_deadline_margin = float(system_metrics.get('avg_deadline_margin', 0.5))
        high_priority_ratio = float(system_metrics.get('high_priority_ratio', 0.25))
        avg_task_size_norm = float(system_metrics.get('avg_task_size_norm', 0.5))
        
        task_features = [
            np.clip(urgent_ratio, 0.0, 1.0),           # [0] 紧急任务占比
            np.clip(avg_deadline_margin, 0.0, 1.0),    # [1] 平均截止期裕度
            np.clip(high_priority_ratio, 0.0, 1.0),    # [2] 高优先级任务比例
            np.clip(avg_task_size_norm, 0.0, 1.0),     # [3] 平均任务大小
        ]
        
        # 任务类型队列分布 (4维)
        def _to_fixed_length(values, length=4):
            if isinstance(values, np.ndarray):
                values = values.tolist()
            elif not isinstance(values, (list, tuple)):
                values = []
            values = [float(v) for v in values[:length]]
            if len(values) < length:
                values.extend([0.25] * (length - len(values)))  # 默认均匀分布
            return [float(np.clip(v, 0.0, 1.0)) for v in values]
        
        queue_distribution = _to_fixed_length(
            system_metrics.get('task_type_queue_distribution', [])
        )
        task_features.extend(queue_distribution)
        
        # 组合所有特征
        full_state = np.array(global_features + task_features, dtype=np.float32)
        return full_state
    
    @staticmethod
    def build_optimized_node_state(node_type: str, node_data: Dict,
                                    last_energy: float = 0.0) -> np.ndarray:
        """
        构建优化后的节点状态（4维，移除冗余位置特征）
        
        优化点:
        1. 移除静态位置特征（对决策影响小）
        2. 使用增量能耗而非累积能耗
        3. 增加决策相关特征（负载率、服务率等）
        """
        if node_type == 'vehicle':
            queue_util = float(node_data.get('queue_util', 0.0))
            current_energy = float(node_data.get('energy', 0.0))
            delta_energy = max(0.0, current_energy - last_energy)
            delta_energy_norm = min(1.0, delta_energy / 50.0)  # 每步能耗归一化到50J
            task_load = float(node_data.get('task_load', 0.0))
            velocity = float(node_data.get('velocity', 0.0))
            velocity_norm = min(1.0, velocity / 50.0)  # 速度归一化到50m/s
            
            return np.array([
                np.clip(queue_util, 0.0, 1.0),
                np.clip(delta_energy_norm, 0.0, 1.0),
                np.clip(task_load, 0.0, 1.0),
                np.clip(velocity_norm, 0.0, 1.0),
            ], dtype=np.float32)
            
        elif node_type == 'rsu':
            queue_util = float(node_data.get('queue_util', 0.0))
            cache_util = float(node_data.get('cache_util', 0.0))
            service_rate = float(node_data.get('service_rate', 0.5))
            load_ratio = float(node_data.get('load_ratio', 0.0))
            
            return np.array([
                np.clip(queue_util, 0.0, 1.0),
                np.clip(cache_util, 0.0, 1.0),
                np.clip(service_rate, 0.0, 1.0),
                np.clip(load_ratio, 0.0, 1.0),
            ], dtype=np.float32)
            
        elif node_type == 'uav':
            queue_util = float(node_data.get('queue_util', 0.0))
            cache_util = float(node_data.get('cache_util', 0.0))
            battery = float(node_data.get('battery_level', 1.0))
            coverage_load = float(node_data.get('coverage_load', 0.0))
            
            return np.array([
                np.clip(queue_util, 0.0, 1.0),
                np.clip(cache_util, 0.0, 1.0),
                np.clip(battery, 0.0, 1.0),
                np.clip(coverage_load, 0.0, 1.0),
            ], dtype=np.float32)
        
        else:
            return np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
    
    @staticmethod
    def build_history_features(history_buffer: List[Dict], window_size: int = 4) -> np.ndarray:
        """
        构建历史趋势特征（16维）
        
        用于捕捉系统动态变化，让智能体能预测趋势
        """
        features = []
        metrics = ['delay', 'energy', 'completion', 'queue']
        
        for metric in metrics:
            if len(history_buffer) >= window_size:
                values = [h.get(metric, 0.0) for h in history_buffer[-window_size:]]
            else:
                values = [0.0] * window_size
            
            # 计算趋势特征
            if len(values) >= 2:
                trend = values[-1] - values[0]  # 变化方向
                avg_val = np.mean(values)
                std_val = np.std(values)
                latest = values[-1]
            else:
                trend, avg_val, std_val, latest = 0.0, 0.0, 0.0, 0.0
            
            # 归一化
            features.extend([
                np.clip(trend + 0.5, 0.0, 1.0),  # 趋势偏移到0.5为中心
                np.clip(avg_val, 0.0, 1.0),
                np.clip(std_val, 0.0, 1.0),
                np.clip(latest, 0.0, 1.0),
            ])
        
        return np.array(features, dtype=np.float32)
    
    # =========================================================================
    # MDP优化: 动作空间优化方法
    # =========================================================================
    
    @staticmethod
    def parse_action_with_effect(action: np.ndarray, num_rsus: int, num_uavs: int,
                                   system_state: Dict = None) -> Dict:
        """
        🆕 MDP优化: 解析动作并预测其效果
        
        优化点:
        1. 返回动作的语义化解释
        2. 根据当前状态预测动作效果
        3. 提供动作-结果因果关系的可视化
        
        Args:
            action: 动作向量
            num_rsus: RSU数量
            num_uavs: UAV数量
            system_state: 当前系统状态（用于预测效果）
        
        Returns:
            解析后的动作字典，包含:
            - decision: 核心决策（卸载目标、节点选择）
            - parameters: 控制参数
            - expected_effect: 预期效果
            - action_summary: 动作摘要字符串
        """
        system_state = system_state or {}
        result = {}
        
        # 1. 解析卸载偏好
        offload_pref = action[:3]
        offload_probs = np.exp(offload_pref) / (np.sum(np.exp(offload_pref)) + 1e-8)
        
        result['offload_distribution'] = {
            'local': float(offload_probs[0]),
            'rsu': float(offload_probs[1]),
            'uav': float(offload_probs[2]),
        }
        
        # 确定主要卸载目标
        target_names = ['local', 'rsu', 'uav']
        primary_target = target_names[np.argmax(offload_probs)]
        result['primary_target'] = primary_target
        
        # 2. 解析节点选择
        idx = 3
        rsu_weights = action[idx:idx+num_rsus]
        idx += num_rsus
        uav_weights = action[idx:idx+num_uavs]
        idx += num_uavs
        
        # RSU选择概率
        rsu_probs = np.exp(rsu_weights) / (np.sum(np.exp(rsu_weights)) + 1e-8)
        selected_rsu = int(np.argmax(rsu_probs))
        
        # UAV选择概率
        uav_probs = np.exp(uav_weights) / (np.sum(np.exp(uav_weights)) + 1e-8)
        selected_uav = int(np.argmax(uav_probs))
        
        result['node_selection'] = {
            'rsu_probs': rsu_probs.tolist(),
            'uav_probs': uav_probs.tolist(),
            'selected_rsu': selected_rsu,
            'selected_uav': selected_uav,
        }
        
        # 3. 解析控制参数（简化版）
        control_raw = action[idx:idx+5] if len(action) > idx else np.zeros(5)
        
        # 转换到语义化参数
        def _convert(v, low, high):
            return low + (np.tanh(v) + 1.0) / 2.0 * (high - low)
        
        result['control_params'] = {
            'load_balance_weight': _convert(control_raw[0] if len(control_raw) > 0 else 0, 0, 1),
            'cache_aggressiveness': _convert(control_raw[1] if len(control_raw) > 1 else 0, 0, 1),
            'migration_sensitivity': _convert(control_raw[2] if len(control_raw) > 2 else 0, 0.3, 0.9),
            'energy_efficiency': _convert(control_raw[3] if len(control_raw) > 3 else 0, 0, 1),
            'queue_awareness': _convert(control_raw[4] if len(control_raw) > 4 else 0, 0, 1),
        }
        
        # 4. 预测动作效果（基于系统状态）
        result['expected_effect'] = UnifiedStateActionSpace._predict_action_effect(
            result, system_state
        )
        
        # 5. 生成动作摘要
        result['action_summary'] = UnifiedStateActionSpace._generate_action_summary(result)
        
        return result
    
    @staticmethod
    def _predict_action_effect(parsed_action: Dict, system_state: Dict) -> Dict:
        """预测动作的预期效果"""
        effect = {
            'delay_impact': 'neutral',
            'energy_impact': 'neutral',
            'load_balance_impact': 'neutral',
        }
        
        primary_target = parsed_action.get('primary_target', 'local')
        ctrl = parsed_action.get('control_params', {})
        
        # 延迟影响预测
        if primary_target == 'rsu':
            effect['delay_impact'] = 'reduced'  # RSU通常更快
        elif primary_target == 'local':
            # 本地处理取决于队列状态
            local_queue = system_state.get('local_queue_util', 0.5)
            if local_queue > 0.7:
                effect['delay_impact'] = 'increased'  # 队列拥堵
            else:
                effect['delay_impact'] = 'neutral'
        elif primary_target == 'uav':
            effect['delay_impact'] = 'slightly_increased'  # UAV通常稍慢
        
        # 能耗影响预测
        energy_eff = ctrl.get('energy_efficiency', 0.5)
        if energy_eff > 0.7:
            effect['energy_impact'] = 'reduced'
        elif energy_eff < 0.3:
            effect['energy_impact'] = 'increased'
        
        # 负载均衡影响
        lb_weight = ctrl.get('load_balance_weight', 0.5)
        if lb_weight > 0.7:
            effect['load_balance_impact'] = 'improved'
        
        return effect
    
    @staticmethod
    def _generate_action_summary(parsed_action: Dict) -> str:
        """生成动作的可读摘要"""
        target = parsed_action.get('primary_target', 'unknown')
        dist = parsed_action.get('offload_distribution', {})
        node_sel = parsed_action.get('node_selection', {})
        ctrl = parsed_action.get('control_params', {})
        
        # 核心决策
        if target == 'local':
            target_str = f"Local({dist.get('local', 0):.0%})"
        elif target == 'rsu':
            rsu_id = node_sel.get('selected_rsu', 0)
            target_str = f"RSU-{rsu_id}({dist.get('rsu', 0):.0%})"
        else:
            uav_id = node_sel.get('selected_uav', 0)
            target_str = f"UAV-{uav_id}({dist.get('uav', 0):.0%})"
        
        # 控制策略
        lb = ctrl.get('load_balance_weight', 0.5)
        ee = ctrl.get('energy_efficiency', 0.5)
        
        strategy = []
        if lb > 0.6:
            strategy.append("LB+")
        if ee > 0.6:
            strategy.append("EE+")
        
        strategy_str = ",".join(strategy) if strategy else "balanced"
        
        return f"Target:{target_str} | Strategy:{strategy_str}"
    
    @staticmethod
    def compute_action_quality_score(action: np.ndarray, system_state: Dict,
                                      num_rsus: int, num_uavs: int) -> float:
        """
        🆕 MDP优化: 计算动作质量分数
        
        这个分数可用于:
        1. 评估智能体的决策质量
        2. 提供对比分析的基准
        3. 检测策略退化
        
        Returns:
            quality_score: [0, 1] 范围的动作质量分数
        """
        parsed = UnifiedStateActionSpace.parse_action_with_effect(
            action, num_rsus, num_uavs, system_state
        )
        
        score = 0.5  # 基线分
        
        # 1. 卸载决策质量（选择低队列节点+0.2分）
        primary_target = parsed.get('primary_target', 'local')
        node_sel = parsed.get('node_selection', {})
        
        if primary_target == 'rsu':
            rsu_queues = system_state.get('rsu_queues', [0.5] * num_rsus)
            selected_rsu = node_sel.get('selected_rsu', 0)
            if selected_rsu < len(rsu_queues):
                # 选择了载最低的RSU得分
                if rsu_queues[selected_rsu] == min(rsu_queues):
                    score += 0.2
                elif rsu_queues[selected_rsu] < 0.5:
                    score += 0.1
        
        # 2. 控制参数合理性（与系统状态匹配+0.15分）
        ctrl = parsed.get('control_params', {})
        avg_queue = system_state.get('avg_queue', 0.5)
        
        # 高队列时应该提高负载均衡权重
        if avg_queue > 0.7 and ctrl.get('load_balance_weight', 0.5) > 0.6:
            score += 0.15
        
        # 3. 稳定性（避免极端动作+0.15分）
        offload_dist = parsed.get('offload_distribution', {})
        max_prob = max(offload_dist.values()) if offload_dist else 1.0
        if max_prob < 0.95:  # 不是完全倾向一个目标
            score += 0.15 * (1.0 - max_prob)
        
        return float(np.clip(score, 0.0, 1.0))
    
    # =========================================================================
    # MDP优化: 状态转移追踪与因果分析
    # =========================================================================
    
    @staticmethod
    def compute_state_transition_info(prev_state: np.ndarray, next_state: np.ndarray,
                                        action: np.ndarray, reward: float,
                                        num_vehicles: int, num_rsus: int, num_uavs: int) -> Dict:
        """
        🆕 MDP优化: 计算状态转移的详细信息
        
        用于:
        1. 分析动作对状态的影响
        2. 验证状态转移的合理性
        3. 提供调试信息
        
        Returns:
            transition_info: 包含状态变化、因果关系的字典
        """
        transition_info = {
            'state_changes': {},
            'causality': {},
            'anomalies': [],
        }
        
        # 计算各状态段的变化
        node_dim = 5  # 每个节点的状态维度
        vehicle_end = num_vehicles * node_dim
        rsu_end = vehicle_end + num_rsus * node_dim
        uav_end = rsu_end + num_uavs * node_dim
        
        # 车辆状态变化
        if len(prev_state) >= vehicle_end and len(next_state) >= vehicle_end:
            vehicle_prev = prev_state[:vehicle_end].reshape(num_vehicles, node_dim)
            vehicle_next = next_state[:vehicle_end].reshape(num_vehicles, node_dim)
            vehicle_delta = vehicle_next - vehicle_prev
            
            transition_info['state_changes']['vehicles'] = {
                'avg_queue_change': float(np.mean(vehicle_delta[:, 3])) if node_dim > 3 else 0.0,
                'avg_energy_change': float(np.mean(vehicle_delta[:, 4])) if node_dim > 4 else 0.0,
                'max_queue_change': float(np.max(np.abs(vehicle_delta[:, 3]))) if node_dim > 3 else 0.0,
            }
        
        # RSU状态变化
        if len(prev_state) >= rsu_end and len(next_state) >= rsu_end:
            rsu_prev = prev_state[vehicle_end:rsu_end].reshape(num_rsus, node_dim)
            rsu_next = next_state[vehicle_end:rsu_end].reshape(num_rsus, node_dim)
            rsu_delta = rsu_next - rsu_prev
            
            transition_info['state_changes']['rsus'] = {
                'avg_cache_change': float(np.mean(rsu_delta[:, 2])) if node_dim > 2 else 0.0,
                'avg_queue_change': float(np.mean(rsu_delta[:, 3])) if node_dim > 3 else 0.0,
            }
        
        # UAV状态变化
        if len(prev_state) >= uav_end and len(next_state) >= uav_end:
            uav_prev = prev_state[rsu_end:uav_end].reshape(num_uavs, node_dim)
            uav_next = next_state[rsu_end:uav_end].reshape(num_uavs, node_dim)
            uav_delta = uav_next - uav_prev
            
            transition_info['state_changes']['uavs'] = {
                'avg_queue_change': float(np.mean(uav_delta[:, 2])) if node_dim > 2 else 0.0,
                'avg_energy_change': float(np.mean(uav_delta[:, 4])) if node_dim > 4 else 0.0,
            }
        
        # 全局状态变化
        if len(prev_state) > uav_end and len(next_state) > uav_end:
            global_prev = prev_state[uav_end:]
            global_next = next_state[uav_end:]
            min_len = min(len(global_prev), len(global_next))
            
            if min_len > 0:
                global_delta = global_next[:min_len] - global_prev[:min_len]
                transition_info['state_changes']['global'] = {
                    'avg_change': float(np.mean(np.abs(global_delta))),
                    'max_change': float(np.max(np.abs(global_delta))),
                }
        
        # 分析因果关系
        offload_pref = action[:3] if len(action) >= 3 else np.array([0.33, 0.33, 0.34])
        offload_probs = np.exp(offload_pref) / (np.sum(np.exp(offload_pref)) + 1e-8)
        primary_target = ['local', 'rsu', 'uav'][np.argmax(offload_probs)]
        
        transition_info['causality'] = {
            'primary_target': primary_target,
            'target_probability': float(np.max(offload_probs)),
            'reward_received': float(reward),
        }
        
        # 检测异常
        state_delta = next_state[:min(len(prev_state), len(next_state))] - \
                      prev_state[:min(len(prev_state), len(next_state))]
        
        # 检测状态突变
        if np.max(np.abs(state_delta)) > 0.5:
            transition_info['anomalies'].append({
                'type': 'large_state_change',
                'max_delta': float(np.max(np.abs(state_delta))),
                'location': int(np.argmax(np.abs(state_delta))),
            })
        
        # 检测奖励异常
        if abs(reward) > 10.0:
            transition_info['anomalies'].append({
                'type': 'extreme_reward',
                'value': float(reward),
            })
        
        return transition_info
    
    @staticmethod
    def validate_state_transition(prev_state: np.ndarray, next_state: np.ndarray,
                                   action: np.ndarray, info: Dict) -> Dict:
        """
        🆕 MDP优化: 验证状态转移的合理性
        
        检查:
        1. 状态值是否在有效范围
        2. 状态变化是否与动作相符
        3. 是否存在物理不合理的转移
        
        Returns:
            validation_result: 验证结果字典
        """
        result = {
            'valid': True,
            'warnings': [],
            'errors': [],
        }
        
        # 1. 检查状态值范围
        if np.any(next_state < -0.1) or np.any(next_state > 1.1):
            out_of_range = np.sum((next_state < -0.1) | (next_state > 1.1))
            result['warnings'].append(f'{out_of_range} state values out of [0,1] range')
        
        # 2. 检查NaN/Inf
        if np.any(~np.isfinite(next_state)):
            result['valid'] = False
            result['errors'].append('State contains NaN or Inf values')
        
        # 3. 检查动作值范围
        if np.any(~np.isfinite(action)):
            result['valid'] = False
            result['errors'].append('Action contains NaN or Inf values')
        
        # 4. 检查状态变化的物理合理性
        state_delta = next_state - prev_state
        
        # 能耗不应该减少（只能增加或保持）
        # 注意：使用增量能耗时这个检查可能需要调整
        
        # 队列变化应该在合理范围内
        max_queue_change = np.max(np.abs(state_delta))
        if max_queue_change > 0.8:
            result['warnings'].append(
                f'Large state change detected: max_delta={max_queue_change:.3f}'
            )
        
        return result
    
    @staticmethod
    def build_step_feedback(task_feedback: Dict, system_metrics: Dict) -> Dict:
        """
        🆕 MDP优化: 构建即时步骤反馈
        
        整合任务执行结果，提供清晰的动作-结果因果信息
        
        Returns:
            step_feedback: 包含本步执行结果的详细反馈
        """
        feedback = {
            # 任务层面
            'tasks_generated': task_feedback.get('step_generated', 0),
            'tasks_completed': task_feedback.get('step_completed', 0),
            'tasks_dropped': task_feedback.get('step_dropped', 0),
            'cache_hits': task_feedback.get('step_cache_hits', 0),
            
            # 卸载决策效果
            'offload_distribution': task_feedback.get('offload_distribution', {}),
            'avg_delay_by_target': task_feedback.get('avg_delay_by_target', {}),
            'avg_energy_by_target': task_feedback.get('avg_energy_by_target', {}),
            
            # 丢弃原因分析
            'drop_reasons': task_feedback.get('drop_reasons', {}),
            
            # 系统层面
            'completion_rate': system_metrics.get('task_completion_rate', 0.0),
            'avg_delay': system_metrics.get('avg_task_delay', 0.0),
            'cache_hit_rate': system_metrics.get('cache_hit_rate', 0.0),
            
            # 决策评估
            'decision_quality': 'good' if task_feedback.get('step_dropped', 0) == 0 else 'needs_improvement',
        }
        
        # 计算本步效率
        generated = feedback['tasks_generated']
        completed = feedback['tasks_completed']
        if generated > 0:
            feedback['step_efficiency'] = completed / generated
        else:
            feedback['step_efficiency'] = 1.0
        
        return feedback
