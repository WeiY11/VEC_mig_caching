#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
真实Baseline算法实现
所有算法都真实执行，不使用模拟数据

【包含的Baseline】
1. Random - 随机选择策略
2. Greedy - 贪心最小负载
3. RoundRobin - 轮询分配
4. LocalFirst - 本地优先
5. NearestNode - 最近节点优先
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

# 添加父目录到路径
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))


class BaselineAlgorithm:
    """Baseline算法基类"""
    
    def __init__(self, name: str):
        self.name = name
        self.step_count = 0
        self.rr_index = 0  # 用于RoundRobin
        self.env = None  # 将在运行时注入环境
    
    def update_environment(self, env):
        """注入训练环境，便于读取仿真器状态"""
        self.env = env
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        """
        根据状态选择动作

        【参数】
        - state: 130维系统状态向量

        【返回】18维连续动作向量
        """
        raise NotImplementedError
    
    def reset(self):
        """重置算法状态"""
        self.step_count = 0
        self.rr_index = 0


class RandomBaseline(BaselineAlgorithm):
    """随机策略"""
    
    def __init__(self):
        super().__init__("Random")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        self.step_count += 1
        return np.random.uniform(-1, 1, 18)


class GreedyBaseline(BaselineAlgorithm):
    """贪心策略"""
    
    def __init__(self):
        super().__init__("Greedy")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(18)

        # Vehicle local queue (归一化)
        local_load = state[3] if len(state) > 3 else 0.5

        # RSU loads (最多4个)
        rsu_loads = []
        for i in range(4):
            rsu_start = 60 + i * 9
            if rsu_start + 3 < len(state):
                rsu_loads.append(state[rsu_start + 3])
            else:
                rsu_loads.append(1.0)

        # UAV loads (最多2个)
        uav_loads = []
        for i in range(2):
            uav_start = 114 + i * 8
            if uav_start + 4 < len(state):
                uav_loads.append(state[uav_start + 4])
            else:
                uav_loads.append(1.0)

        loads = [local_load, min(rsu_loads), min(uav_loads)]
        best = int(np.argmin(loads))

        action[0] = 0.9 if best == 0 else -0.9
        action[1] = 0.9 if best == 1 else -0.9
        action[2] = 0.9 if best == 2 else -0.9

        # RSU selection (one-hot over first 4)
        best_rsu = int(np.argmin(rsu_loads)) if rsu_loads else 0
        for i in range(6):
            action[3 + i] = 1.0 if (i == best_rsu and best == 1) else -1.0

        # UAV selection (two positions)
        best_uav = int(np.argmin(uav_loads)) if uav_loads else 0
        action[9] = 1.0 if (best == 2 and best_uav == 0) else -1.0
        action[10] = 1.0 if (best == 2 and best_uav == 1) else -1.0

        # Neutral cache/migration controls
        action[11:] = 0.0

        self.step_count += 1
        return action


class RoundRobinBaseline(BaselineAlgorithm):
    """轮询策略"""
    
    def __init__(self):
        super().__init__("RoundRobin")
    
    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(18)
        phase = self.rr_index % 3

        if phase == 0:
            action[0], action[1], action[2] = 0.9, -0.9, -0.9
        elif phase == 1:
            action[0], action[1], action[2] = -0.9, 0.9, -0.9
            rsu_idx = (self.rr_index // 3) % max(1, len(self.env.simulator.rsus)) if self.env else 0
            for i in range(6):
                action[3 + i] = 1.0 if i == rsu_idx else -1.0
        else:
            action[0], action[1], action[2] = -0.9, -0.9, 0.9
            uav_idx = (self.rr_index // 3) % max(1, len(self.env.simulator.uavs)) if self.env else 0
            action[9] = 1.0 if uav_idx == 0 else -1.0
            action[10] = 1.0 if uav_idx == 1 else -1.0

        self.step_count += 1
        self.rr_index += 1
        return action


class LocalFirstBaseline(BaselineAlgorithm):
    """本地优先策略"""

    def __init__(self, threshold: float = 0.7):
        super().__init__("LocalFirst")
        self.threshold = threshold

    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(18)
        local_load = state[3] if len(state) > 3 else 0.5

        if local_load < self.threshold:
            action[0], action[1], action[2] = 0.9, -0.9, -0.9
        else:
            action[0], action[1], action[2] = -0.9, 0.9, -0.9

            rsu_loads = []
            for i in range(4):
                idx = 60 + i * 9
                rsu_loads.append(state[idx + 3] if idx + 3 < len(state) else 1.0)
            best = int(np.argmin(rsu_loads)) if rsu_loads else 0
            for i in range(6):
                action[3 + i] = 1.0 if i == best else -1.0

        self.step_count += 1
        return action


class NearestNodeBaseline(BaselineAlgorithm):
    """最近节点策略"""

    def __init__(self):
        super().__init__("NearestNode")

    def select_action(self, state: np.ndarray) -> np.ndarray:
        action = np.zeros(18)
        vehicle_pos = np.array([state[0], state[1]]) if len(state) >= 2 else np.array([0.5, 0.5])

        rsu_distances = []
        for i in range(4):
            idx = 60 + i * 9
            if idx + 1 < len(state):
                pos = np.array([state[idx], state[idx + 1]])
            else:
                pos = np.array([1.0, 1.0])
            rsu_distances.append(np.linalg.norm(vehicle_pos - pos))

        uav_distances = []
        for i in range(2):
            idx = 114 + i * 8
            if idx + 1 < len(state):
                pos = np.array([state[idx], state[idx + 1]])
            else:
                pos = np.array([1.0, 1.0])
            uav_distances.append(np.linalg.norm(vehicle_pos - pos))

        candidates = [0.1, min(rsu_distances), min(uav_distances)]
        best = int(np.argmin(candidates))

        action[0] = 0.9 if best == 0 else -0.9
        action[1] = 0.9 if best == 1 else -0.9
        action[2] = 0.9 if best == 2 else -0.9

        best_rsu = int(np.argmin(rsu_distances)) if rsu_distances else 0
        for i in range(6):
            action[3 + i] = 1.0 if (best == 1 and i == best_rsu) else -1.0

        best_uav = int(np.argmin(uav_distances)) if uav_distances else 0
        action[9] = 1.0 if (best == 2 and best_uav == 0) else -1.0
        action[10] = 1.0 if (best == 2 and best_uav == 1) else -1.0

        self.step_count += 1
        return action


def create_baseline_algorithm(name: str) -> BaselineAlgorithm:
    """
    创建Baseline算法实例
    
    【参数】
    - name: 算法名称（Random, Greedy, RoundRobin, LocalFirst, NearestNode）
    
    【返回】算法实例
    """
    algorithms = {
        'Random': RandomBaseline,
        'Greedy': GreedyBaseline,
        'RoundRobin': RoundRobinBaseline,
        'LocalFirst': LocalFirstBaseline,
        'NearestNode': NearestNodeBaseline
    }
    
    if name not in algorithms:
        raise ValueError(f"未知的Baseline算法: {name}. 可选: {list(algorithms.keys())}")
    
    return algorithms[name]()


if __name__ == "__main__":
    # 测试：创建所有Baseline算法
    print("="*60)
    print("Baseline算法测试")
    print("="*60)
    
    baselines = ['Random', 'Greedy', 'RoundRobin', 'LocalFirst', 'NearestNode']
    
    for name in baselines:
        algo = create_baseline_algorithm(name)
        print(f"\n{name}:")
        print(f"  实例: {algo}")
        
        # 测试动作生成
        test_state = np.random.rand(130)
        test_node_states = {}
        actions = algo.select_action(test_state, test_node_states)
        print(f"  动作维度: vehicle={len(actions['vehicle_agent'])}, "
              f"rsu={len(actions['rsu_agent'])}, uav={len(actions['uav_agent'])}")
    
    print("\n" + "="*60)
    print(f"所有 {len(baselines)} 个Baseline算法测试通过！")
    print("="*60)

