#!/usr/bin/env python3
"""
Dual-stage controller environment wrapper.

Stage 1: Offloading head (algorithm A) produces offloading preferences
         - 3 logits for [local, rsu, uav]
         - K logits for RSU selection
         - M logits for UAV selection
Stage 2: Base RL env (algorithm B) produces the full action vector; we keep
         the cache/migration control (last 8 dims) and optionally other parts.

This wrapper combines both into a single action dict compatible with the
existing SingleAgentTrainingEnvironment._build_simulator_actions.

Notes:
- Stage 1 here is a heuristic policy (several variants). RL stage 1 can be
  added later by swapping the policy implementation.
- Training is applied to Stage 2 RL env only. Stage 1 is stateless.
"""

from typing import Any, Dict, Optional, Tuple
import numpy as np


class _HeuristicOffloadPolicy:
    """🧠 智能多因素启发式卸载策略
    
    策略类型:
      - 'smart': 智能综合评分（距离+队列+缓存+能耗）⭐推荐
      - 'delay_optimal': 延迟优先（最小化传输+队列+计算时延）
      - 'energy_optimal': 能耗优先（最小化传输能耗+计算能耗）
      - 'cache_aware': 缓存感知（强烈偏好缓存命中）
      - 'load_balance': 负载均衡（避免热点，均匀分配）
      - 'heuristic': 经典启发式（队列优先）[旧版]
      - 'greedy': 贪婪最近节点 [旧版]
    
    可配置权重（仅'smart'策略）:
      - weight_delay: 延迟权重（默认2.0）
      - weight_energy: 能耗权重（默认1.0）
      - weight_cache: 缓存权重（默认3.0）
      - weight_queue: 队列权重（默认1.5）
    """

    def __init__(self, strategy: str = 'smart', **kwargs):
        self.strategy = (strategy or 'smart').lower()
        
        # 智能策略的可配置权重
        self.weight_delay = kwargs.get('weight_delay', 2.0)
        self.weight_energy = kwargs.get('weight_energy', 1.0)
        self.weight_cache = kwargs.get('weight_cache', 3.0)
        self.weight_queue = kwargs.get('weight_queue', 1.5)
        
        # 距离阈值（米）
        self.near_distance = 200.0
        self.far_distance = 400.0

    @staticmethod
    def _logit_from_probs(probs: np.ndarray) -> np.ndarray:
        eps = 1e-6
        p = np.clip(probs.astype(np.float32), eps, 1.0 - eps)
        return np.log(p)
    
    def _get_vehicle_position(self, simulator, vehicle_idx: int):
        """获取车辆位置"""
        if hasattr(simulator, 'vehicles') and vehicle_idx < len(simulator.vehicles):
            return simulator.vehicles[vehicle_idx].get('position', (0, 0))
        return (0, 0)
    
    def _calculate_distance(self, simulator, pos1, pos2):
        """计算距离"""
        if hasattr(simulator, 'calculate_distance'):
            return simulator.calculate_distance(pos1, pos2)
        # 简单欧氏距离
        import math
        return math.sqrt((pos1[0]-pos2[0])**2 + (pos1[1]-pos2[1])**2)
    
    def _estimate_delay(self, distance: float, queue_len: int, has_cache: bool, node_type: str) -> float:
        """估算延迟（秒）"""
        # 传输延迟（基于距离）
        base_rate = 80e6 if node_type == 'rsu' else 45e6  # bps
        attenuation = 1.0 + distance / 800.0
        tx_delay = (1e6 * 8) / (base_rate / attenuation)  # 假设1MB数据
        
        # 队列等待
        wait_delay = queue_len * 0.15 if node_type == 'rsu' else queue_len * 0.22
        
        # 计算延迟
        comp_delay = 0.0 if has_cache else (0.2 if node_type == 'rsu' else 0.3)
        
        return tx_delay + wait_delay + comp_delay
    
    def _estimate_energy(self, distance: float, node_type: str) -> float:
        """估算能耗（焦耳）"""
        tx_power = 0.18 if node_type == 'rsu' else 0.12  # W
        base_rate = 80e6 if node_type == 'rsu' else 45e6
        attenuation = 1.0 + distance / 800.0
        tx_time = (1e6 * 8) / (base_rate / attenuation)
        return tx_power * tx_time

    def decide(self, simulator, vehicle_idx: int) -> Dict[str, np.ndarray]:
        """🧠 智能决策：选择最优卸载目标"""
        num_rsus = len(simulator.rsus)
        num_uavs = len(simulator.uavs)
        
        # 获取车辆位置
        vehicle_pos = self._get_vehicle_position(simulator, vehicle_idx)
        
        # ========== 策略路由 ==========
        if self.strategy == 'smart':
            return self._decide_smart(simulator, vehicle_pos, num_rsus, num_uavs)
        elif self.strategy == 'delay_optimal':
            return self._decide_delay_optimal(simulator, vehicle_pos, num_rsus, num_uavs)
        elif self.strategy == 'energy_optimal':
            return self._decide_energy_optimal(simulator, vehicle_pos, num_rsus, num_uavs)
        elif self.strategy == 'cache_aware':
            return self._decide_cache_aware(simulator, vehicle_pos, num_rsus, num_uavs)
        elif self.strategy == 'load_balance':
            return self._decide_load_balance(simulator, vehicle_pos, num_rsus, num_uavs)
        else:
            # 旧版启发式
            return self._decide_legacy(simulator, num_rsus, num_uavs)
    
    def _decide_smart(self, simulator, vehicle_pos, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """⭐ 智能综合评分策略"""
        rsu_scores = []
        uav_scores = []
        
        # 评估所有RSU
        for i, rsu in enumerate(simulator.rsus):
            rsu_pos = rsu.get('position', (0, 0))
            dist = self._calculate_distance(simulator, vehicle_pos, rsu_pos)
            queue_len = len(rsu.get('computation_queue', []))
            cache_size = len(rsu.get('cache', {}))
            has_cache = cache_size > 0
            
            # 多因素评分（越小越好）
            delay_score = self._estimate_delay(dist, queue_len, has_cache, 'rsu')
            energy_score = self._estimate_energy(dist, 'rsu')
            cache_bonus = -self.weight_cache if has_cache else 0.0
            queue_penalty = self.weight_queue * queue_len * 0.1
            
            total_score = (self.weight_delay * delay_score + 
                          self.weight_energy * energy_score + 
                          cache_bonus + queue_penalty)
            
            rsu_scores.append((total_score, i, dist))
        
        # 评估所有UAV
        for j, uav in enumerate(simulator.uavs):
            uav_pos = uav.get('position', (0, 0))
            dist = self._calculate_distance(simulator, vehicle_pos, uav_pos)
            queue_len = len(uav.get('computation_queue', []))
            
            delay_score = self._estimate_delay(dist, queue_len, False, 'uav')
            energy_score = self._estimate_energy(dist, 'uav')
            queue_penalty = self.weight_queue * queue_len * 0.1
            
            total_score = (self.weight_delay * delay_score + 
                          self.weight_energy * energy_score + 
                          queue_penalty)
            
            uav_scores.append((total_score, j, dist))
        
        # 生成logits
        rsu_logits = self._scores_to_logits(rsu_scores, num_rsus)
        uav_logits = self._scores_to_logits(uav_scores, num_uavs)
        
        # 三路选择logits (local/RSU/UAV)
        local_score = 5.0  # 本地基准
        rsu_best = min(rsu_scores, key=lambda x: x[0])[0] if rsu_scores else 999
        uav_best = min(uav_scores, key=lambda x: x[0])[0] if uav_scores else 999
        
        # 归一化为概率（越小越好→越大概率）
        scores = np.array([local_score, rsu_best, uav_best])
        probs = np.exp(-scores / 2.0)  # 温度参数=2.0
        probs = probs / probs.sum()
        three_logits = np.log(probs + 1e-6)
        
        return {
            'head3_logits': three_logits.astype(np.float32),
            'rsu_logits': rsu_logits,
            'uav_logits': uav_logits,
        }
    
    def _decide_delay_optimal(self, simulator, vehicle_pos, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """⏱️ 延迟优先策略"""
        rsu_scores = []
        for i, rsu in enumerate(simulator.rsus):
            dist = self._calculate_distance(simulator, vehicle_pos, rsu.get('position', (0, 0)))
            queue = len(rsu.get('computation_queue', []))
            cache = len(rsu.get('cache', {})) > 0
            delay = self._estimate_delay(dist, queue, cache, 'rsu')
            rsu_scores.append((delay, i, dist))
        
        uav_scores = []
        for j, uav in enumerate(simulator.uavs):
            dist = self._calculate_distance(simulator, vehicle_pos, uav.get('position', (0, 0)))
            queue = len(uav.get('computation_queue', []))
            delay = self._estimate_delay(dist, queue, False, 'uav')
            uav_scores.append((delay, j, dist))
        
        return self._build_logits_from_scores(rsu_scores, uav_scores, num_rsus, num_uavs)
    
    def _decide_energy_optimal(self, simulator, vehicle_pos, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """⚡ 能耗优先策略"""
        rsu_scores = []
        for i, rsu in enumerate(simulator.rsus):
            dist = self._calculate_distance(simulator, vehicle_pos, rsu.get('position', (0, 0)))
            energy = self._estimate_energy(dist, 'rsu')
            rsu_scores.append((energy, i, dist))
        
        uav_scores = []
        for j, uav in enumerate(simulator.uavs):
            dist = self._calculate_distance(simulator, vehicle_pos, uav.get('position', (0, 0)))
            energy = self._estimate_energy(dist, 'uav')
            uav_scores.append((energy, j, dist))
        
        return self._build_logits_from_scores(rsu_scores, uav_scores, num_rsus, num_uavs)
    
    def _decide_cache_aware(self, simulator, vehicle_pos, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """📦 缓存感知策略"""
        rsu_scores = []
        for i, rsu in enumerate(simulator.rsus):
            dist = self._calculate_distance(simulator, vehicle_pos, rsu.get('position', (0, 0)))
            cache_size = len(rsu.get('cache', {}))
            # 缓存越多分数越低（越好）
            score = -cache_size * 10.0 + dist * 0.01
            rsu_scores.append((score, i, dist))
        
        uav_scores = []
        for j, uav in enumerate(simulator.uavs):
            dist = self._calculate_distance(simulator, vehicle_pos, uav.get('position', (0, 0)))
            score = dist * 0.01  # UAV无缓存
            uav_scores.append((score, j, dist))
        
        return self._build_logits_from_scores(rsu_scores, uav_scores, num_rsus, num_uavs)
    
    def _decide_load_balance(self, simulator, vehicle_pos, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """⚖️ 负载均衡策略"""
        rsu_scores = []
        for i, rsu in enumerate(simulator.rsus):
            dist = self._calculate_distance(simulator, vehicle_pos, rsu.get('position', (0, 0)))
            queue = len(rsu.get('computation_queue', []))
            # 队列越长惩罚越重
            score = queue * 5.0 + dist * 0.001
            rsu_scores.append((score, i, dist))
        
        uav_scores = []
        for j, uav in enumerate(simulator.uavs):
            dist = self._calculate_distance(simulator, vehicle_pos, uav.get('position', (0, 0)))
            queue = len(uav.get('computation_queue', []))
            score = queue * 5.0 + dist * 0.001
            uav_scores.append((score, j, dist))
        
        return self._build_logits_from_scores(rsu_scores, uav_scores, num_rsus, num_uavs)
    
    def _decide_legacy(self, simulator, num_rsus: int, num_uavs: int) -> Dict[str, np.ndarray]:
        """旧版简单启发式"""
        rsu_logits = np.full((num_rsus,), -6.0, dtype=np.float32)
        if num_rsus > 0:
            chosen = None
            best = None
            for i, rsu in enumerate(simulator.rsus):
                q = len(rsu.get('computation_queue', []))
                if best is None or q < best:
                    best = q
                    chosen = i
            if chosen is not None:
                rsu_logits[chosen] = 6.0
        
        uav_logits = np.full((num_uavs,), -6.0, dtype=np.float32)
        if num_uavs > 0:
            chosen = None
            best = None
            for j, uav in enumerate(simulator.uavs):
                q = len(uav.get('computation_queue', []))
                if best is None or q < best:
                    best = q
                    chosen = j
            if chosen is not None:
                uav_logits[chosen] = 6.0
        
        three_logits = np.array([0.0, 1.0 if num_rsus > 0 else -2.0, 
                                 0.5 if num_uavs > 0 else -2.0], dtype=np.float32)
        
        return {
            'head3_logits': three_logits,
            'rsu_logits': rsu_logits,
            'uav_logits': uav_logits,
        }
    
    def _scores_to_logits(self, scores: list, num_nodes: int) -> np.ndarray:
        """将评分转换为logits（分数越低越好）"""
        logits = np.full((num_nodes,), -6.0, dtype=np.float32)
        if scores:
            # 找到最佳节点
            best_idx = min(scores, key=lambda x: x[0])[1]
            logits[best_idx] = 6.0
            
            # 给次优节点一些概率
            sorted_scores = sorted(scores, key=lambda x: x[0])
            if len(sorted_scores) > 1:
                second_best = sorted_scores[1][1]
                logits[second_best] = 2.0
        
        return logits
    
    def _build_logits_from_scores(self, rsu_scores, uav_scores, num_rsus, num_uavs):
        """通用的logits构建"""
        rsu_logits = self._scores_to_logits(rsu_scores, num_rsus)
        uav_logits = self._scores_to_logits(uav_scores, num_uavs)
        
        local_score = 5.0
        rsu_best = min(rsu_scores, key=lambda x: x[0])[0] if rsu_scores else 999
        uav_best = min(uav_scores, key=lambda x: x[0])[0] if uav_scores else 999
        
        scores = np.array([local_score, rsu_best, uav_best])
        probs = np.exp(-scores / 2.0)
        probs = probs / probs.sum()
        three_logits = np.log(probs + 1e-6)
        
        return {
            'head3_logits': three_logits.astype(np.float32),
            'rsu_logits': rsu_logits,
            'uav_logits': uav_logits,
        }


class DualStageControllerEnv:
    """🔧 改进的两阶段控制器：分离动作空间
    
    Stage 1（启发式）：控制卸载决策（前10维）
    Stage 2（RL）：只学习缓存/迁移控制（8维）
    
    关键改进：
    1. base_env 只输出8维动作（缓存/迁移）
    2. Stage 1 启发式独立生成卸载决策
    3. 避免信用分配混乱：RL只为它控制的部分负责
    """

    def __init__(self, base_env: Any, simulator: Any, stage1_strategy: str = 'heuristic'):
        self.base = base_env  # Stage-2 RL env (TD3/SAC/...) implementing same interface
        self.simulator = simulator
        self.stage1 = _HeuristicOffloadPolicy(stage1_strategy)
        # 动作维度保持与base一致
        self.action_dim = getattr(self.base, 'action_dim', 18)
        self.config = getattr(self.base, 'config', None)
        # 保存覆盖后的动作用于训练（确保训练-执行一致性）
        self._last_covered_action = None
        
        print(f"🧠 DualStageControllerEnv初始化:")
        print(f"   Stage 1 (启发式): 卸载决策 [{stage1_strategy}]")
        print(f"   Stage 2 (RL): 缓存/迁移控制")
        print(f"   ⚠️  训练策略: 网络学习完整动作，但前10维会被覆盖")

    # ---- Policy interface passthrough with patching ----
    def get_actions(self, state: np.ndarray, training: bool = True):
        act = self.base.get_actions(state, training=training)
        # Normalize to a dict with 'vehicle_agent'
        if isinstance(act, dict):
            actions_dict = dict(act)
            vehicle_vec = actions_dict.get('vehicle_agent')
            if vehicle_vec is None:
                # fallback: construct from base vector if exists
                base_vec = None
            else:
                base_vec = np.array(vehicle_vec, dtype=np.float32)
        else:
            # tuple or ndarray; try first element or itself
            base_arr = act[0] if isinstance(act, (tuple, list)) else act
            base_vec = np.array(base_arr, dtype=np.float32)
            actions_dict = {
                'vehicle_agent': base_vec,
            }

        num_rsus = len(getattr(self.simulator, 'rsus', []))
        num_uavs = len(getattr(self.simulator, 'uavs', []))
        # Ensure base_vec sized
        if base_vec is None:
            base_vec = np.zeros(max(3 + num_rsus + num_uavs + 8, getattr(self.base, 'action_dim', 18)), dtype=np.float32)

        vec = base_vec.copy()

        # Stage-1 decision -> overwrite offloading segments
        policy = self.stage1.decide(self.simulator, 0)  # vehicle_idx not used for head3
        # For RSU/UAV logits, we need vehicle index; approximate with 0. OK for many vehicles since per-step repeats
        head3 = policy['head3_logits']
        rsu_logits = policy['rsu_logits']
        uav_logits = policy['uav_logits']

        # Place logits into vector segments
        rsu_start = 3
        rsu_end = rsu_start + num_rsus
        uav_end = rsu_end + num_uavs
        if vec.size < uav_end + 8:
            padded = np.zeros(uav_end + 8, dtype=np.float32)
            padded[:vec.size] = vec
            vec = padded

        vec[:3] = head3
        if num_rsus > 0:
            vec[rsu_start:rsu_end] = rsu_logits
        if num_uavs > 0:
            vec[rsu_end:uav_end] = uav_logits

        # 🔧 保存覆盖后的动作用于训练（确保训练-执行一致）
        self._last_covered_action = vec.copy()

        actions_dict['vehicle_agent'] = vec
        actions_dict['rsu_agent'] = vec[rsu_start:rsu_end]
        actions_dict['uav_agent'] = vec[rsu_end:uav_end]
        return actions_dict

    # ---- Training methods proxy to Stage-2 RL env ----
    def train_step(self, state, action, reward, next_state, done):
        """🔧 训练修复：确保训练-执行一致性
        
        核心问题：
        - 执行时：使用覆盖后的动作（Stage1前10维 + RL后8维）
        - 奖励：基于覆盖后动作的执行结果
        - 训练：应该用覆盖后的动作，让网络学习正确的因果关系
        
        解决方案：
        - 使用self._last_covered_action（覆盖后的动作）进行训练
        - 网络会学习：输出接近覆盖后动作的值
        - 虽然前10维会被再次覆盖，但后8维能正确学习
        - 网络会自然地发现：前10维的变化不影响奖励
        """
        # 使用覆盖后的动作进行训练（与实际执行一致）
        training_action = self._last_covered_action if self._last_covered_action is not None else action
        return self.base.train_step(state, training_action, reward, next_state, done)

    def store_experience(self, **kwargs):
        # PPO special path; forward to base env
        if hasattr(self.base, 'store_experience'):
            return self.base.store_experience(**kwargs)
        return {}

    def update(self, *args, **kwargs):
        if hasattr(self.base, 'update'):
            return self.base.update(*args, **kwargs)
        return {}

    # ---- State and reward methods proxy to base env ----
    def get_state_vector(self, node_states: Dict, system_metrics: Dict) -> np.ndarray:
        """获取状态向量 - 代理到基础环境"""
        return self.base.get_state_vector(node_states, system_metrics)

    def calculate_reward(self, system_metrics: Dict, 
                        cache_metrics: Optional[Dict] = None,
                        migration_metrics: Optional[Dict] = None) -> float:
        """计算奖励 - 代理到基础环境"""
        return self.base.calculate_reward(system_metrics, cache_metrics, migration_metrics)

    def decompose_action(self, action: np.ndarray) -> Dict[str, np.ndarray]:
        """分解动作 - 代理到基础环境"""
        if hasattr(self.base, 'decompose_action'):
            return self.base.decompose_action(action)
        # Fallback: simple decomposition
        num_rsus = len(getattr(self.simulator, 'rsus', []))
        num_uavs = len(getattr(self.simulator, 'uavs', []))
        rsu_start = 3
        rsu_end = rsu_start + num_rsus
        uav_end = rsu_end + num_uavs
        return {
            'vehicle_agent': action,
            'rsu_agent': action[rsu_start:rsu_end] if len(action) > rsu_start else np.array([]),
            'uav_agent': action[rsu_end:uav_end] if len(action) > rsu_end else np.array([])
        }

    # ---- Model save/load proxy to base env ----
    def save_models(self, filepath: str):
        """保存模型 - 代理到基础环境"""
        if hasattr(self.base, 'save_models'):
            return self.base.save_models(filepath)

    def load_models(self, filepath: str):
        """加载模型 - 代理到基础环境"""
        if hasattr(self.base, 'load_models'):
            return self.base.load_models(filepath)
