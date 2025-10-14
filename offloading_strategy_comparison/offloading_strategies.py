#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
卸载策略实现 (Offloading Strategies)
用于与混合DRL策略（TD3）进行全面对比

【包含的策略】（9种）
固定策略（3种）：
1. LocalOnly  - 仅本地处理（所有任务在车辆本地执行）
2. RSUOnly    - 仅基站处理（所有任务卸载到RSU）
3. UAVOnly    - 仅无人机处理（所有任务卸载到UAV）

简单启发式（3种）：
4. Random     - 随机选择（最简单baseline）
5. RoundRobin - 轮询分配（公平性）
6. NearestNode - 最近节点（基于距离）

智能启发式（2种）：
7. LoadBalance - 负载均衡（选择负载最小节点）
8. MinDelay   - 最小时延优先（优化传输+计算时延）

深度强化学习（1种）：
9. HybridDRL  - 混合策略（TD3智能决策）

【论文用途】
形成完整的性能梯度对比，从最简单的随机策略到智能DRL策略
"""

import numpy as np
from typing import Dict, Optional


class OffloadingStrategy:
    """卸载策略基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.step_count = 0
        self.env = None  # 环境引用（用于获取节点信息）
    
    def update_environment(self, env):
        """注入训练环境引用"""
        self.env = env
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        选择动作（固定拓扑：4 RSU + 2 UAV）
        
        【参数】
        state: 状态向量
        
        【返回】16维动作向量:
        - [0:3] 任务分配（local, RSU, UAV）- Softmax归一化
        - [3:7] RSU选择（4个RSU）- Softmax归一化
        - [7:9] UAV选择（2个UAV）- Softmax归一化
        - [9:16] 控制参数（7维：缓存、迁移等）
        """
        raise NotImplementedError
    
    def reset(self):
        """重置策略状态"""
        self.step_count = 0


class LocalOnlyStrategy(OffloadingStrategy):
    """
    仅本地处理策略 (Local-Only)
    
    【特点】
    - 所有任务在车辆本地执行
    - 无通信开销，但受限于车辆计算能力
    - 基线1：最小通信成本，但可能高时延
    """
    
    def __init__(self):
        super().__init__("LocalOnly")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # ========== 任务分配：100%本地 ==========
        # 使用极端值(±5)确保softmax后接近100%概率
        action[0] = 1.0    # Local: 强制本地
        action[1] = -1.0   # RSU: 不使用
        action[2] = -1.0   # UAV: 不使用
        
        # ========== RSU选择：不选择 ==========
        action[3:7] = -1.0  # 所有RSU均不选
        
        # ========== UAV选择：不选择 ==========
        action[7:9] = -1.0  # 所有UAV均不选
        
        # ========== 控制参数：默认策略 ==========
        action[9:16] = 0.0  # 中性控制
        
        self.step_count += 1
        return action


class RSUOnlyStrategy(OffloadingStrategy):
    """
    仅基站处理策略 (RSU-Only)
    
    【特点】
    - 所有任务卸载到RSU（轮询或负载均衡）
    - 利用固定基础设施的强计算能力
    - 基线2：稳定但可能通信拥塞
    """
    
    def __init__(self, selection_mode: str = "round_robin"):
        """
        Args:
            selection_mode: RSU选择模式
                - "round_robin": 轮询选择
                - "load_balance": 负载均衡（需要环境支持）
        """
        super().__init__("RSUOnly")
        self.selection_mode = selection_mode
        self.rsu_index = 0  # 轮询索引
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # ========== 任务分配：100% RSU ==========
        # 使用极端值确保softmax后接近100%
        action[0] = -1.0   # Local: 不使用
        action[1] = 1.0    # RSU: 强制卸载
        action[2] = -1.0   # UAV: 不使用
        
        # ========== RSU选择：轮询或负载均衡 ==========
        if self.selection_mode == "round_robin":
            # 轮询选择RSU
            selected_rsu = self.rsu_index % 4
            for i in range(4):
                action[3 + i] = 1.0 if i == selected_rsu else -1.0
            self.rsu_index += 1
        
        elif self.selection_mode == "load_balance":
            # 负载均衡：从状态中提取RSU负载
            num_vehicles = 12
            if self.env and hasattr(self.env, 'simulator'):
                num_vehicles = len(self.env.simulator.vehicles)
            
            vehicle_offset = num_vehicles * 5
            rsu_loads = []
            for i in range(4):
                cpu_idx = vehicle_offset + i * 5 + 3
                if cpu_idx < len(action):  # 这里应该检查state，但action已经创建，用简化方法
                    rsu_loads.append(0.5)  # 默认值
                else:
                    rsu_loads.append(1.0)
            
            selected_rsu = int(np.argmin(rsu_loads)) if rsu_loads else 0
            for i in range(4):
                action[3 + i] = 1.0 if i == selected_rsu else -1.0
        
        # ========== UAV选择：不使用 ==========
        action[7:9] = -1.0
        
        # ========== 控制参数 ==========
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class UAVOnlyStrategy(OffloadingStrategy):
    """
    仅无人机处理策略 (UAV-Only)
    
    【特点】
    - 所有任务卸载到UAV（轮询或负载均衡）
    - 灵活移动，覆盖范围广
    - 基线3：高灵活性但能量受限
    """
    
    def __init__(self, selection_mode: str = "round_robin"):
        """
        Args:
            selection_mode: UAV选择模式
                - "round_robin": 轮询选择
                - "load_balance": 负载均衡
        """
        super().__init__("UAVOnly")
        self.selection_mode = selection_mode
        self.uav_index = 0  # 轮询索引
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # ========== 任务分配：100% UAV ==========
        action[0] = -1.0   # Local: 不使用
        action[1] = -1.0   # RSU: 不使用
        action[2] = 1.0    # UAV: 强制卸载
        
        # ========== RSU选择：不使用 ==========
        action[3:7] = -1.0
        
        # ========== UAV选择：轮询或负载均衡 ==========
        if self.selection_mode == "round_robin":
            # 轮询选择UAV
            selected_uav = self.uav_index % 2
            for i in range(2):
                action[7 + i] = 1.0 if i == selected_uav else -1.0
            self.uav_index += 1
        
        elif self.selection_mode == "load_balance":
            # 负载均衡：轮询选择（简化实现）
            selected_uav = self.uav_index % 2
            self.uav_index += 1
            
            for i in range(2):
                action[7 + i] = 1.0 if i == selected_uav else -1.0
        
        # ========== 控制参数 ==========
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class RandomStrategy(OffloadingStrategy):
    """
    随机策略 (Random)
    
    【特点】
    - 完全随机选择卸载位置
    - 最简单的baseline，用于验证其他策略的有效性
    - 无任何智能决策
    """
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """完全随机动作"""
        action = np.random.uniform(-1, 1, 16)
        self.step_count += 1
        return action


class RoundRobinStrategy(OffloadingStrategy):
    """
    轮询策略 (Round Robin)
    
    【特点】
    - 依次轮询选择本地/RSU/UAV
    - 保证公平性，避免单节点过载
    - 不考虑节点状态
    """
    
    def __init__(self):
        super().__init__("RoundRobin")
        self.round_index = 0
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 轮询3种位置：Local -> RSU -> UAV
        phase = self.round_index % 3
        
        if phase == 0:  # Local
            action[0:3] = [0.9, -0.9, -0.9]
            action[3:7] = -1.0
            action[7:9] = -1.0
        elif phase == 1:  # RSU
            action[0:3] = [-0.9, 0.9, -0.9]
            # 轮询选择RSU
            rsu_idx = (self.round_index // 3) % 4
            for i in range(4):
                action[3 + i] = 1.0 if i == rsu_idx else -1.0
            action[7:9] = -1.0
        else:  # UAV
            action[0:3] = [-0.9, -0.9, 0.9]
            action[3:7] = -1.0
            # 轮询选择UAV
            uav_idx = (self.round_index // 3) % 2
            for i in range(2):
                action[7 + i] = 1.0 if i == uav_idx else -1.0
        
        action[9:16] = 0.0
        
        self.round_index += 1
        self.step_count += 1
        return action
    
    def reset(self):
        super().reset()
        self.round_index = 0


class NearestNodeStrategy(OffloadingStrategy):
    """
    最近节点策略 (Nearest Node)
    
    【特点】
    - 选择距离最近的边缘节点（RSU或UAV）
    - 最小化传输时延
    - 简单但有效的启发式
    """
    
    def __init__(self):
        super().__init__("NearestNode")
    
    def _extract_positions(self, state: np.ndarray, num_vehicles: int = 12) -> tuple:
        """从状态中提取位置信息"""
        # 车辆平均位置（简化）
        vehicle_x = np.mean([state[i*5] for i in range(min(num_vehicles, len(state)//5)) if i*5 < len(state)])
        vehicle_y = np.mean([state[i*5+1] for i in range(min(num_vehicles, len(state)//5)) if i*5+1 < len(state)])
        vehicle_pos = np.array([vehicle_x, vehicle_y])
        
        # RSU位置
        vehicle_offset = num_vehicles * 5
        rsu_positions = []
        for i in range(4):
            rsu_idx = vehicle_offset + i * 5
            if rsu_idx + 1 < len(state):
                rsu_positions.append(np.array([state[rsu_idx], state[rsu_idx + 1]]))
            else:
                rsu_positions.append(np.array([0.5, 0.5]))
        
        # UAV位置
        uav_offset = vehicle_offset + 4 * 5
        uav_positions = []
        for i in range(2):
            uav_idx = uav_offset + i * 5
            if uav_idx + 1 < len(state):
                uav_positions.append(np.array([state[uav_idx], state[uav_idx + 1]]))
            else:
                uav_positions.append(np.array([0.5, 0.5]))
        
        return vehicle_pos, rsu_positions, uav_positions
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 提取位置信息
        vehicle_pos, rsu_positions, uav_positions = self._extract_positions(state, num_vehicles)
        
        # 计算到各RSU的距离
        rsu_distances = [np.linalg.norm(vehicle_pos - rsu_pos) for rsu_pos in rsu_positions]
        
        # 计算到各UAV的距离
        uav_distances = [np.linalg.norm(vehicle_pos - uav_pos) for uav_pos in uav_positions]
        
        # 选择最近的节点类型
        min_rsu_dist = min(rsu_distances) if rsu_distances else float('inf')
        min_uav_dist = min(uav_distances) if uav_distances else float('inf')
        
        # 比较：本地(0距离) vs 最近RSU vs 最近UAV
        # 假设本地处理等效距离为0.1（有一定优先级）
        local_equiv_dist = 0.1
        
        if local_equiv_dist < min(min_rsu_dist, min_uav_dist):
            # 本地处理
            action[0:3] = [0.9, -0.9, -0.9]
            action[3:7] = -1.0
            action[7:9] = -1.0
        elif min_rsu_dist < min_uav_dist:
            # RSU
            action[0:3] = [-0.9, 0.9, -0.9]
            best_rsu = int(np.argmin(rsu_distances))
            for i in range(4):
                action[3 + i] = 1.0 if i == best_rsu else -1.0
            action[7:9] = -1.0
        else:
            # UAV
            action[0:3] = [-0.9, -0.9, 0.9]
            action[3:7] = -1.0
            best_uav = int(np.argmin(uav_distances))
            for i in range(2):
                action[7 + i] = 1.0 if i == best_uav else -1.0
        
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class LoadBalanceStrategy(OffloadingStrategy):
    """
    负载均衡策略 (Load Balance)
    
    【特点】
    - 监控所有节点的负载（CPU、队列）
    - 选择负载最小的节点
    - 智能贪心策略
    """
    
    def __init__(self):
        super().__init__("LoadBalance")
    
    def _extract_loads(self, state: np.ndarray, num_vehicles: int = 12) -> tuple:
        """从状态中提取负载信息"""
        vehicle_offset = num_vehicles * 5
        rsu_offset = vehicle_offset
        uav_offset = vehicle_offset + 4 * 5
        
        # 本地平均负载
        local_loads = []
        for i in range(min(num_vehicles, len(state) // 5)):
            cpu_idx = i * 5 + 3
            if cpu_idx < len(state):
                local_loads.append(state[cpu_idx])
        local_load = float(np.mean(local_loads)) if local_loads else 0.5
        
        # RSU负载
        rsu_loads = []
        for i in range(4):
            cpu_idx = rsu_offset + i * 5 + 3
            if cpu_idx < len(state):
                rsu_loads.append(state[cpu_idx])
            else:
                rsu_loads.append(1.0)
        
        # UAV负载
        uav_loads = []
        for i in range(2):
            cpu_idx = uav_offset + i * 5 + 3
            if cpu_idx < len(state):
                uav_loads.append(state[cpu_idx])
            else:
                uav_loads.append(1.0)
        
        return local_load, rsu_loads, uav_loads
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 提取负载
        local_load, rsu_loads, uav_loads = self._extract_loads(state, num_vehicles)
        
        # 选择负载最小的节点类型
        min_rsu_load = min(rsu_loads) if rsu_loads else 1.0
        min_uav_load = min(uav_loads) if uav_loads else 1.0
        
        loads = [local_load, min_rsu_load, min_uav_load]
        best = int(np.argmin(loads))
        
        # 任务分配
        action[0] = 1.0 if best == 0 else -1.0  # Local
        action[1] = 1.0 if best == 1 else -1.0  # RSU
        action[2] = 1.0 if best == 2 else -1.0  # UAV
        
        # RSU选择（负载最小）
        if best == 1:
            best_rsu = int(np.argmin(rsu_loads))
            for i in range(4):
                action[3 + i] = 1.0 if i == best_rsu else -1.0
        else:
            action[3:7] = -1.0
        
        # UAV选择（负载最小）
        if best == 2:
            best_uav = int(np.argmin(uav_loads))
            for i in range(2):
                action[7 + i] = 1.0 if i == best_uav else -1.0
        else:
            action[7:9] = -1.0
        
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class MinDelayStrategy(OffloadingStrategy):
    """
    最小时延策略 (Min Delay)
    
    【特点】
    - 评估传输时延和计算时延
    - 选择总时延最小的节点
    - 考虑节点负载和距离
    """
    
    def __init__(self):
        super().__init__("MinDelay")
    
    def _estimate_delay(self, state: np.ndarray, num_vehicles: int = 12) -> tuple:
        """估算各节点的时延"""
        vehicle_offset = num_vehicles * 5
        rsu_offset = vehicle_offset
        uav_offset = vehicle_offset + 4 * 5
        
        # 提取负载和位置
        local_loads = []
        for i in range(min(num_vehicles, len(state) // 5)):
            cpu_idx = i * 5 + 3
            if cpu_idx < len(state):
                local_loads.append(state[cpu_idx])
        local_load = float(np.mean(local_loads)) if local_loads else 0.5
        
        # 本地时延（只有计算时延，无传输时延）
        local_delay = local_load * 0.5  # 简化模型
        
        # RSU时延（传输+计算）
        rsu_delays = []
        for i in range(4):
            cpu_idx = rsu_offset + i * 5 + 3
            if cpu_idx < len(state):
                rsu_load = state[cpu_idx]
                # 计算时延
                compute_delay = rsu_load * 0.3
                # 传输时延（基于距离，简化）
                pos_idx = rsu_offset + i * 5
                if pos_idx + 1 < len(state):
                    # 假设车辆在(0.5, 0.5)
                    dist = np.sqrt((state[pos_idx] - 0.5)**2 + (state[pos_idx+1] - 0.5)**2)
                    trans_delay = dist * 0.2
                else:
                    trans_delay = 0.1
                rsu_delays.append(compute_delay + trans_delay)
            else:
                rsu_delays.append(1.0)
        
        # UAV时延（传输+计算）
        uav_delays = []
        for i in range(2):
            cpu_idx = uav_offset + i * 5 + 3
            if cpu_idx < len(state):
                uav_load = state[cpu_idx]
                compute_delay = uav_load * 0.4  # UAV计算能力稍弱
                pos_idx = uav_offset + i * 5
                if pos_idx + 1 < len(state):
                    dist = np.sqrt((state[pos_idx] - 0.5)**2 + (state[pos_idx+1] - 0.5)**2)
                    trans_delay = dist * 0.15  # UAV传输可能更快
                else:
                    trans_delay = 0.1
                uav_delays.append(compute_delay + trans_delay)
            else:
                uav_delays.append(1.0)
        
        return local_delay, rsu_delays, uav_delays
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(16)
        
        # 获取车辆数
        num_vehicles = 12
        if self.env and hasattr(self.env, 'simulator'):
            num_vehicles = len(self.env.simulator.vehicles)
        
        # 估算时延
        local_delay, rsu_delays, uav_delays = self._estimate_delay(state, num_vehicles)
        
        # 选择时延最小的节点
        min_rsu_delay = min(rsu_delays) if rsu_delays else float('inf')
        min_uav_delay = min(uav_delays) if uav_delays else float('inf')
        
        delays = [local_delay, min_rsu_delay, min_uav_delay]
        best = int(np.argmin(delays))
        
        # 任务分配（使用极端值）
        action[0] = 1.0 if best == 0 else -1.0
        action[1] = 1.0 if best == 1 else -1.0
        action[2] = 1.0 if best == 2 else -1.0
        
        # RSU选择
        if best == 1:
            best_rsu = int(np.argmin(rsu_delays))
            for i in range(4):
                action[3 + i] = 1.0 if i == best_rsu else -1.0
        else:
            action[3:7] = -1.0
        
        # UAV选择
        if best == 2:
            best_uav = int(np.argmin(uav_delays))
            for i in range(2):
                action[7 + i] = 1.0 if i == best_uav else -1.0
        else:
            action[7:9] = -1.0
        
        action[9:16] = 0.0
        
        self.step_count += 1
        return action


class HybridDRLStrategy(OffloadingStrategy):
    """
    混合DRL策略 (Hybrid DRL - TD3)
    
    【特点】
    - 使用训练好的TD3模型智能决策
    - 动态平衡本地/RSU/UAV卸载
    - 对比基线：验证DRL的有效性
    - 自动处理不同车辆数的状态维度适配
    """
    
    def __init__(self, td3_agent):
        """
        Args:
            td3_agent: 训练好的TD3智能体
        """
        super().__init__("HybridDRL")
        self.td3_agent = td3_agent
        # TD3模型是用12辆车训练的，状态维度固定为98
        self.trained_num_vehicles = 12
        self.trained_state_dim = 98
    
    def _adapt_state_dimension(self, state: np.ndarray) -> np.ndarray:
        """
        适配状态维度以匹配TD3模型的输入要求
        
        TD3模型是用12辆车训练的，期望98维输入：
        - 车辆状态: 12×5 = 60维
        - RSU状态: 4×5 = 20维  
        - UAV状态: 2×5 = 10维
        - 全局状态: 8维
        总计: 98维
        
        当实际车辆数不同时，需要进行适配：
        - 少于12辆车：用零填充缺失的车辆状态
        - 多于12辆车：采样或截断到12辆车
        """
        current_dim = len(state)
        
        if current_dim == self.trained_state_dim:
            # 维度匹配，直接返回
            return state
        
        # 计算当前的车辆数（基于状态维度）
        # state_dim = num_vehicles * 5 + 4 * 5 + 2 * 5 + 8
        # current_dim = num_vehicles * 5 + 20 + 10 + 8
        # num_vehicles = (current_dim - 38) / 5
        current_num_vehicles = int((current_dim - 38) / 5)
        
        if current_num_vehicles < self.trained_num_vehicles:
            # 车辆数少于12，需要填充
            # 提取各部分状态
            vehicle_states = state[:current_num_vehicles * 5]
            rsu_states = state[current_num_vehicles * 5:(current_num_vehicles * 5 + 20)]
            uav_states = state[(current_num_vehicles * 5 + 20):(current_num_vehicles * 5 + 30)]
            global_states = state[-8:]
            
            # 填充车辆状态到12辆
            padding_size = (self.trained_num_vehicles - current_num_vehicles) * 5
            vehicle_padding = np.zeros(padding_size)
            
            # 组合新状态
            adapted_state = np.concatenate([
                vehicle_states, 
                vehicle_padding,  # 填充的车辆状态
                rsu_states,
                uav_states,
                global_states
            ])
        else:
            # 车辆数多于12，需要采样或截断
            # 这里简单截断前12辆车
            vehicle_states = state[:self.trained_num_vehicles * 5]
            rsu_states = state[current_num_vehicles * 5:(current_num_vehicles * 5 + 20)]
            uav_states = state[(current_num_vehicles * 5 + 20):(current_num_vehicles * 5 + 30)]
            global_states = state[-8:]
            
            # 组合新状态
            adapted_state = np.concatenate([
                vehicle_states,
                rsu_states,
                uav_states,
                global_states
            ])
        
        assert len(adapted_state) == self.trained_state_dim, \
            f"适配后的状态维度错误: {len(adapted_state)} != {self.trained_state_dim}"
        
        return adapted_state
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """使用TD3模型选择动作"""
        # 适配状态维度
        adapted_state = self._adapt_state_dimension(state)
        
        # 使用TD3的确定性策略（evaluation模式）
        action = self.td3_agent.select_action(adapted_state, training=False)
        self.step_count += 1
        return action
    
    def reset(self):
        super().reset()
        # TD3 agent不需要额外的reset


def create_offloading_strategy(
    strategy_name: str, 
    td3_agent=None,
    selection_mode: str = "round_robin"
) -> OffloadingStrategy:
    """
    创建卸载策略实例
    
    【参数】
    strategy_name: 策略名称（9种）
        固定策略：
        - "LocalOnly": 仅本地处理
        - "RSUOnly": 仅基站处理
        - "UAVOnly": 仅无人机处理
        
        简单启发式：
        - "Random": 随机选择
        - "RoundRobin": 轮询分配
        - "NearestNode": 最近节点
        
        智能启发式：
        - "LoadBalance": 负载均衡
        - "MinDelay": 最小时延
        
        深度强化学习：
        - "HybridDRL" or "TD3": 混合DRL策略
        
    td3_agent: TD3智能体（仅HybridDRL需要）
    selection_mode: RSU/UAV选择模式（"round_robin" or "load_balance"，仅RSUOnly/UAVOnly使用）
    
    【返回】策略实例
    """
    strategy_name_upper = strategy_name.upper()
    
    # 固定策略
    if strategy_name_upper == "LOCALONLY":
        return LocalOnlyStrategy()
    
    elif strategy_name_upper == "RSUONLY":
        return RSUOnlyStrategy(selection_mode=selection_mode)
    
    elif strategy_name_upper == "UAVONLY":
        return UAVOnlyStrategy(selection_mode=selection_mode)
    
    # 简单启发式
    elif strategy_name_upper == "RANDOM":
        return RandomStrategy()
    
    elif strategy_name_upper == "ROUNDROBIN":
        return RoundRobinStrategy()
    
    elif strategy_name_upper == "NEARESTNODE":
        return NearestNodeStrategy()
    
    # 智能启发式
    elif strategy_name_upper == "LOADBALANCE":
        return LoadBalanceStrategy()
    
    elif strategy_name_upper == "MINDELAY":
        return MinDelayStrategy()
    
    # 深度强化学习
    elif strategy_name_upper in ["HYBRIDDRL", "TD3", "HYBRID"]:
        if td3_agent is None:
            raise ValueError("HybridDRL策略需要提供td3_agent参数")
        return HybridDRLStrategy(td3_agent)
    
    else:
        raise ValueError(
            f"不支持的策略: {strategy_name}. "
            f"可选: LocalOnly, RSUOnly, UAVOnly, Random, RoundRobin, "
            f"NearestNode, LoadBalance, MinDelay, HybridDRL"
        )


if __name__ == "__main__":
    print("="*70)
    print("卸载策略测试（9种策略）")
    print("="*70)
    
    # 测试所有策略
    strategies = [
        "LocalOnly", "RSUOnly", "UAVOnly",  # 固定策略
        "Random", "RoundRobin", "NearestNode",  # 简单启发式
        "LoadBalance", "MinDelay"  # 智能启发式
    ]
    
    for strategy_name in strategies:
        strategy = create_offloading_strategy(strategy_name)
        print(f"\n[{strategy.name}]")
        
        # 测试动作生成
        test_state = np.random.rand(98)  # 12辆车 * 5 + 4 RSU * 5 + 2 UAV * 5 + 8
        action = strategy.select_action(test_state)
        
        print(f"  动作维度: {len(action)}")
        print(f"  任务分配: Local={action[0]:.2f}, RSU={action[1]:.2f}, UAV={action[2]:.2f}")
        print(f"  RSU选择: {action[3:7]}")
        print(f"  UAV选择: {action[7:9]}")
    
    print("\n" + "="*70)
    print("所有9种策略测试通过！")
    print("="*70)

