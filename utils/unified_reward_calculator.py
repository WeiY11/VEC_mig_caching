#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缁熶竴濂栧姳璁＄畻鍣?(Unified Reward Calculator)
閫傜敤浜庢墍鏈夊崟鏅鸿兘浣揇RL绠楁硶锛圖DPG, TD3, DQN, PPO, SAC锛?

璁捐鍘熷垯锛?
1. 鏍稿績浼樺寲鐩爣锛氭椂寤?+ 鑳借€楀弻鐩爣鍔犳潈鍜?
2. 杈呭姪绾︽潫锛氶€氳繃涓㈠純浠诲姟鎯╃綒淇濊瘉瀹屾垚鐜?
3. 鎴愭湰鏈€灏忓寲锛氬鍔变弗鏍间负璐熷€硷紙鎴愭湰锛?
4. 绠楁硶閫傞厤锛歋AC淇濈暀杞诲井璋冩暣浠ラ€傚簲鏈€澶х喌妗嗘灦
"""

import numpy as np
from typing import Dict, Optional, List
from config import config


class UnifiedRewardCalculator:
    """
    缁熶竴濂栧姳璁＄畻鍣?- 鎵€鏈夌畻娉曞叡浜牳蹇冮€昏緫
    """

    def __init__(self, algorithm: str = "general"):
        """
        鍒濆鍖栫粺涓€濂栧姳璁＄畻鍣?
        
        Args:
            algorithm: 绠楁硶绫诲瀷 ("general", "sac")
                - "general": 閫氱敤鐗堟湰锛圖DPG, TD3, DQN, PPO锛?
                - "sac": SAC涓撶敤鐗堟湰锛堣€冭檻鏈€澶х喌鐗规€э級
        """
        self.algorithm = algorithm.upper()
        
        # 浠庨厤缃姞杞芥牳蹇冩潈閲?
        self.weight_delay = config.rl.reward_weight_delay      # 榛樿 2.0
        self.weight_energy = config.rl.reward_weight_energy    # 榛樿 1.2
        self.penalty_dropped = config.rl.reward_penalty_dropped # 榛樿 0.02
        self.weight_cache = getattr(config.rl, 'reward_weight_cache', 0.0)
        self.weight_migration = getattr(config.rl, 'reward_weight_migration', 0.0)
        priority_weights = getattr(config, 'task', None)
        if priority_weights is not None:
            priority_weights = getattr(priority_weights, 'type_priority_weights', None)
        if isinstance(priority_weights, dict) and priority_weights:
            total_priority = sum(float(v) for v in priority_weights.values()) or 1.0
            self.task_priority_weights = {
                int(task_type): float(value) / total_priority
                for task_type, value in priority_weights.items()
            }
        else:
            # 鍧囧寑鏉冮噸浣滀负鍥為€€
            self.task_priority_weights = {1: 0.25, 2: 0.25, 3: 0.25, 4: 0.25}
        
        # 馃幆 鏍稿績璁捐锛氬綊涓€鍖栧洜瀛愶紙纭繚鏃跺欢鍜岃兘鑰楀湪鐩稿悓鏁伴噺绾э級
        # 鐩爣锛歞elay=0.2s 鍜?energy=600J 褰掍竴鍖栧悗璐＄尞鐩稿綋
        self.delay_normalizer = 0.2      # 0.2s 鈫?0.2
        self.energy_normalizer = 1000.0   # 馃敡 璋冩暣锛氱獊鍑鸿兘鑰楀弽棣?
        
        # 馃敡 SAC涓撶敤璋冩暣锛氭洿婵€杩涚殑褰掍竴鍖栦互骞宠　鎺㈢储
        if self.algorithm == "SAC":
            self.delay_normalizer = 0.25      # 0.2s 鈫?0.67锛堟洿鏁忔劅锛?
            self.energy_normalizer = 1200.0  # 1000J 鈫?0.67锛堟洿鏁忔劅锛?
        
        # 濂栧姳鑼冨洿闄愬埗
        if self.algorithm == "SAC":
            # SAC鍏佽灏忓箙姝ｅ€煎鍔憋紙鏈€澶х喌闇€瑕佹槑纭縺鍔憋級
            self.reward_clip_range = (-15.0, 3.0)
        else:
            # 閫氱敤鐗堟湰锛氱函鎴愭湰鏈€灏忓寲
            self.reward_clip_range = (-80.0, -0.005)
        
        print(f"[OK] 缁熶竴濂栧姳璁＄畻鍣ㄥ垵濮嬪寲 ({self.algorithm})")
        print(f"   鏍稿績鏉冮噸: Delay={self.weight_delay:.2f}, Energy={self.weight_energy:.2f}")
        print(f"   褰掍竴鍖? Delay/{self.delay_normalizer:.2f}, Energy/{self.energy_normalizer:.0f}")
        print(f"   濂栧姳鑼冨洿: {self.reward_clip_range}")
        print(f"   浼樺寲鐩爣: 鏈€灏忓寲 {self.weight_delay}*Delay + {self.weight_energy}*Energy")
        if self.weight_cache > 0 or self.weight_migration > 0:
            print(f"   鎷撳睍鏉冮噸: Cache={self.weight_cache:.2f}, Migration={self.weight_migration:.2f}")

    def calculate_reward(self, 
                        system_metrics: Dict,
                        cache_metrics: Optional[Dict] = None,
                        migration_metrics: Optional[Dict] = None) -> float:
        """
        璁＄畻缁熶竴濂栧姳锛堟敮鎸佺紦瀛樺拰杩佺Щ鎸囨爣锛屼絾涓嶅奖鍝嶆牳蹇冨鍔憋級
        
        Args:
            system_metrics: 绯荤粺鎬ц兘鎸囨爣
            cache_metrics: 缂撳瓨鎸囨爣锛堝彲閫夛紝鐢ㄤ簬鏈潵鎵╁睍锛?
            migration_metrics: 杩佺Щ鎸囨爣锛堝彲閫夛紝鐢ㄤ簬鏈潵鎵╁睍锛?
        
        Returns:
            reward: 鏍囬噺濂栧姳鍊?
        """
        # 1锔忊儯 鎻愬彇鏍稿績鎸囨爣锛堝畨鍏ㄥ鐞哊one鍊硷級
        def safe_float(value, default=0.0):
            """瀹夊叏杞崲涓篺loat锛屽鐞哊one鍜屽紓甯稿€?""
            if value is None:
                return default
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return default
        
        def safe_int(value, default=0):
            """瀹夊叏杞崲涓篿nt锛屽鐞哊one鍜屽紓甯稿€?""
            if value is None:
                return default
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default

        def to_float_list(source) -> List[float]:
            """灏嗙郴缁熸寚鏍囦腑鐨勫垪琛?鏁扮粍瀹夊叏杞崲涓烘诞鐐瑰垪琛ㄣ€?""
            if isinstance(source, np.ndarray):
                values = source.tolist()
            elif isinstance(source, (list, tuple)):
                values = list(source)
            else:
                return []
            cleaned = []
            for item in values:
                try:
                    cleaned.append(float(item))
                except (TypeError, ValueError):
                    cleaned.append(0.0)
            return cleaned
        
        avg_delay = safe_float(system_metrics.get('avg_task_delay'), 0.0)
        total_energy = safe_float(system_metrics.get('total_energy_consumption'), 0.0)
        dropped_tasks = safe_int(system_metrics.get('dropped_tasks'), 0)
        
        # 2锔忊儯 褰掍竴鍖?
        norm_delay = avg_delay / self.delay_normalizer
        norm_energy = total_energy / self.energy_normalizer
        
        # 3锔忊儯 璁＄畻鍩虹鎴愭湰锛堝弻鐩爣鍔犳潈鍜岋級
        base_cost = (self.weight_delay * norm_delay + 
                     self.weight_energy * norm_energy)
        
        # 4锔忊儯 涓㈠純浠诲姟鎯╃綒锛堜繚璇佸畬鎴愮巼绾︽潫锛?
        dropped_penalty = self.penalty_dropped * dropped_tasks
        
        # 5锔忊儯 鑷€傚簲闃堝€兼儵缃氾紙闃叉鏋佺鎯呭喌锛?
        delay_threshold_penalty = 0.0
        energy_threshold_penalty = 0.0
        
        if self.algorithm == "SAC":
            # SAC锛氭洿婵€杩涚殑闃堝€兼儵缃?
            if avg_delay > 0.25:
                delay_threshold_penalty = (avg_delay - 0.25) * 8.0
            if total_energy > 2000:
                energy_threshold_penalty = (total_energy - 2000) / 1000.0
        else:
            # 閫氱敤绠楁硶锛氭俯鍜岀殑闃堝€兼儵缃?
            if avg_delay > 0.30:
                delay_threshold_penalty = (avg_delay - 0.30) * 5.0
            if total_energy > 3000:
                energy_threshold_penalty = (total_energy - 3000) / 1500.0
        
        # 6锔忊儯 鎬绘垚鏈?
        cache_penalty = 0.0
        if self.weight_cache > 0:
            cache_hit_rate = safe_float(system_metrics.get('cache_hit_rate'), 0.0)
            cache_requests = safe_float(system_metrics.get('cache_requests', 0.0), 0.0)
            cache_evictions = safe_float(system_metrics.get('cache_evictions', 0.0), 0.0)
            cache_eviction_rate = system_metrics.get('cache_eviction_rate')
            if cache_eviction_rate is None:
                if cache_requests > 0:
                    cache_eviction_rate = min(1.0, cache_evictions / max(1.0, cache_requests))
                else:
                    cache_eviction_rate = 0.0
            else:
                try:
                    cache_eviction_rate = max(0.0, min(1.0, float(cache_eviction_rate)))
                except (TypeError, ValueError):
                    cache_eviction_rate = 0.0

            cache_penalty = max(0.0, 1.0 - min(1.0, cache_hit_rate))
            cache_penalty += min(0.6, cache_eviction_rate)

            mitigation = 0.0
            if cache_requests > 0:
                collaborative_ratio = safe_float(system_metrics.get('cache_collaborative_writes', 0.0)) / max(1.0, cache_requests)
                local_hit_ratio = safe_float(system_metrics.get('local_cache_hits', 0.0)) / max(1.0, cache_requests)
                mitigation = min(0.4, 0.25 * collaborative_ratio + 0.5 * local_hit_ratio)

            cache_penalty = min(1.4, max(-0.4, cache_penalty - mitigation))

        migration_penalty = 0.0
        if self.weight_migration > 0:
            migration_success = safe_float(system_metrics.get('migration_success_rate'), 0.0)
            migration_avg_cost = safe_float(system_metrics.get('migration_avg_cost', 0.0))
            migration_delay_saved = safe_float(system_metrics.get('migration_avg_delay_saved', 0.0))
            migration_penalty = max(0.0, 1.0 - min(1.0, migration_success))
            migration_penalty += 0.0001 * migration_avg_cost
            migration_penalty -= min(0.2, migration_delay_saved * 0.05)
            migration_penalty = max(-0.3, migration_penalty)

        drop_rates = [np.clip(v, 0.0, 1.0) for v in to_float_list(system_metrics.get('task_type_drop_rate'))]
        queue_distribution = [np.clip(v, 0.0, 1.0) for v in to_float_list(system_metrics.get('task_type_queue_distribution'))]
        generated_share = [np.clip(v, 0.0, 1.0) for v in to_float_list(system_metrics.get('task_type_generated_share'))]

        total_priority = sum(self.task_priority_weights.values()) or 1.0
        weighted_drop = 0.0
        weighted_queue = 0.0
        weighted_presence = 0.0
        for idx in range(4):
            weight = self.task_priority_weights.get(idx + 1, 1.0)
            if idx < len(drop_rates):
                weighted_drop += weight * drop_rates[idx]
            if idx < len(queue_distribution):
                weighted_queue += weight * queue_distribution[idx]
            if idx < len(generated_share):
                weighted_presence += weight * generated_share[idx]

        weighted_drop /= total_priority
        weighted_queue /= total_priority
        weighted_presence /= total_priority

        # 鏍规嵁楂樹紭鍏堢骇浠诲姟鐨勫帇鍔涜皟鏁寸紦瀛?杩佺Щ鎯╃綒
        cache_penalty *= (1.0 + np.clip(weighted_queue + 0.5 * weighted_presence, 0.0, 2.0))
        migration_penalty *= (1.0 + np.clip(weighted_drop + 0.5 * weighted_presence, 0.0, 2.0))

        priority_pressure = np.clip(weighted_drop * 1.2 + weighted_queue * 0.6 + weighted_presence * 0.3, 0.0, 1.8)

        total_cost = (base_cost +
                     dropped_penalty +
                     delay_threshold_penalty +
                     energy_threshold_penalty +
                     self.weight_cache * cache_penalty +
                     self.weight_migration * migration_penalty)
        total_cost *= (1.0 + priority_pressure)
        
        # 7锔忊儯 SAC涓撶敤锛氭鍚戞縺鍔辨満鍒讹紙鏈€澶х喌妗嗘灦闇€瑕佹槑纭?濂?鐨勪俊鍙凤級
        bonus = 0.0
        if self.algorithm == "SAC":
            completion_rate = safe_float(system_metrics.get('task_completion_rate'), 0.0)
            
            # 寤惰繜浼樼濂栧姳
            if avg_delay < 0.20:
                bonus += (0.20 - avg_delay) * 3.0
            
            # 瀹屾垚鐜囦紭绉€濂栧姳
            if completion_rate > 0.95:
                bonus += (completion_rate - 0.95) * 15.0
        
        # 8锔忊儯 鏈€缁堝鍔?
        if self.algorithm == "SAC":
            reward = bonus - total_cost  # SAC: bonus鍙兘涓烘
        else:
            reward = -total_cost  # 閫氱敤: 绾礋鍊兼垚鏈?
        
        # 9锔忊儯 瑁佸壀鍒板悎鐞嗚寖鍥?
        clipped_reward = np.clip(reward, *self.reward_clip_range)
        
        return clipped_reward
    
    def get_reward_breakdown(self, system_metrics: Dict) -> str:
        """鑾峰彇濂栧姳鍒嗚В鐨勫彲璇绘姤鍛?""
        def safe_float(value, default=0.0):
            if value is None:
                return default
            try:
                return max(0.0, float(value))
            except (TypeError, ValueError):
                return default
        
        def safe_int(value, default=0):
            if value is None:
                return default
            try:
                return max(0, int(value))
            except (TypeError, ValueError):
                return default
        
        avg_delay = safe_float(system_metrics.get('avg_task_delay'), 0.0)
        total_energy = safe_float(system_metrics.get('total_energy_consumption'), 0.0)
        dropped_tasks = safe_int(system_metrics.get('dropped_tasks'), 0)
        completion_rate = safe_float(system_metrics.get('task_completion_rate'), 0.0)
        
        reward = self.calculate_reward(system_metrics)
        
        breakdown = f"""
濂栧姳鍒嗚В鎶ュ憡 ({self.algorithm}):
鈹溾攢鈹€ 鎬诲鍔? {reward:.3f}
鈹溾攢鈹€ 鏍稿績鎸囨爣:
鈹?  鈹溾攢鈹€ 鏃跺欢: {avg_delay:.3f}s (褰掍竴鍖? {avg_delay/self.delay_normalizer:.3f})
鈹?  鈹溾攢鈹€ 鑳借€? {total_energy:.2f}J (褰掍竴鍖? {total_energy/self.energy_normalizer:.3f})
鈹?  鈹斺攢鈹€ 瀹屾垚鐜? {completion_rate:.1%}
鈹溾攢鈹€ 鎴愭湰璐＄尞:
鈹?  鈹溾攢鈹€ 鏃跺欢鎴愭湰: {self.weight_delay * avg_delay/self.delay_normalizer:.3f}
鈹?  鈹溾攢鈹€ 鑳借€楁垚鏈? {self.weight_energy * total_energy/self.energy_normalizer:.3f}
鈹?  鈹斺攢鈹€ 涓㈠純鎯╃綒: {self.penalty_dropped * dropped_tasks:.3f} ({dropped_tasks}涓换鍔?
鈹斺攢鈹€ 浼樺寲鏂瑰悜: {'鏈€澶у寲濂栧姳锛堝惈bonus锛? if self.algorithm == 'SAC' else '鏈€灏忓寲鎴愭湰'}
        """
        
        return breakdown.strip()


# ==================== 鍏ㄥ眬瀹炰緥鍜屼究鎹锋帴鍙?====================

# 閫氱敤鐗堟湰锛圖DPG, TD3, DQN, PPO锛?
_general_reward_calculator = UnifiedRewardCalculator(algorithm="general")

# SAC涓撶敤鐗堟湰
_sac_reward_calculator = UnifiedRewardCalculator(algorithm="sac")


def calculate_unified_reward(system_metrics: Dict,
                             cache_metrics: Optional[Dict] = None,
                             migration_metrics: Optional[Dict] = None,
                             algorithm: str = "general") -> float:
    """
    缁熶竴濂栧姳璁＄畻鎺ュ彛锛堟墍鏈夌畻娉曡皟鐢級
    
    Args:
        system_metrics: 绯荤粺鎬ц兘鎸囨爣
        cache_metrics: 缂撳瓨鎸囨爣锛堝彲閫夛級
        migration_metrics: 杩佺Щ鎸囨爣锛堝彲閫夛級
        algorithm: 绠楁硶绫诲瀷 ("general" 鎴?"sac")
    
    Returns:
        reward: 鏍囬噺濂栧姳鍊?
    """
    if algorithm.upper() == "SAC":
        calculator = _sac_reward_calculator
    else:
        calculator = _general_reward_calculator
    
    return calculator.calculate_reward(system_metrics, cache_metrics, migration_metrics)


def get_reward_breakdown(system_metrics: Dict, algorithm: str = "general") -> str:
    """鑾峰彇濂栧姳鍒嗚В鎶ュ憡"""
    if algorithm.upper() == "SAC":
        calculator = _sac_reward_calculator
    else:
        calculator = _general_reward_calculator
    
    return calculator.get_reward_breakdown(system_metrics)


# ==================== 鍚戝悗鍏煎鎺ュ彛 ====================

def calculate_enhanced_reward(system_metrics: Dict,
                             cache_metrics: Optional[Dict] = None,
                             migration_metrics: Optional[Dict] = None) -> float:
    """鍚戝悗鍏煎鎺ュ彛锛堜緵鐜版湁浠ｇ爜璋冪敤锛?""
    return calculate_unified_reward(system_metrics, cache_metrics, migration_metrics, "general")


def calculate_sac_reward(system_metrics: Dict) -> float:
    """SAC涓撶敤鎺ュ彛锛堝悜鍚庡吋瀹癸級"""
    return calculate_unified_reward(system_metrics, algorithm="sac")


def calculate_simple_reward(system_metrics: Dict) -> float:
    """绠€鍖栨帴鍙ｏ紙鍚戝悗鍏煎锛?""
    return calculate_unified_reward(system_metrics, algorithm="general")



