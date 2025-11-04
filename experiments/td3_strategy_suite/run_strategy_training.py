#!/usr/bin/env python3
"""
TD3 Strategy Training Runner
--------------------------------

銆愬姛鑳姐€?TD3娑堣瀺瀹為獙璁粌杩愯鍣?鐢ㄤ簬绯荤粺鍦拌瘎浼板悇鍐崇瓥妯″潡鐨勭嫭绔嬭础鐚€?閫氳繃绂佺敤/鍚敤涓嶅悓鐨勭郴缁熺粍浠讹紙鍗歌浇銆佽祫婧愬垎閰嶃€佽縼绉荤瓑锛夛紝閲忓寲姣忎釜妯″潡瀵规暣浣撴€ц兘鐨勫奖鍝嶃€?
銆愯鏂囧搴斻€?- 娑堣瀺瀹為獙锛圓blation Study锛夛細璇勪及绯荤粺鍚勬ā鍧楃殑蹇呰鎬?- 瀵规瘮浠ヤ笅6绉嶇瓥鐣ラ厤缃細
  1. local-only: 浠呮湰鍦版墽琛岋紙鏃犲嵏杞斤級
  2. remote-only锛堝崟RSU杩滅▼鎵ц锛?  3. offloading-only: 鍗歌浇鍐崇瓥锛堟湰鍦皏s鍗昍SU锛?  4. resource-only: 澶氳妭鐐硅祫婧愬垎閰嶏紙鏃犺縼绉伙級
  5. comprehensive-no-migration: 瀹屾暣绯荤粺锛堟棤杩佺Щ锛?  6. comprehensive-migration: 瀹屾暣TD3绯荤粺

銆愬伐浣滄祦绋嬨€?1. 姣忔璋冪敤杩愯鍗曚釜绛栫暐閰嶇疆
2. 璁粌缁撴灉锛圝SON/鍥捐〃/鎶ュ憡锛夎澶嶅埗鍒扮瓥鐣ヤ笓灞炴枃浠跺す
3. 缁存姢婊氬姩鏇存柊鐨剆ummary.json锛岃褰曟墍鏈夌瓥鐣ョ殑鎬ц兘鎸囨爣
4. 鍚庣画鍙娇鐢╯ummary.json鐢熸垚绛栫暐瀵规瘮鍥捐〃

銆愪娇鐢ㄧず渚嬨€?```bash
# 杩愯鍗曚釜绛栫暐
python experiments/td3_strategy_suite/run_strategy_training.py \\
    --strategy local-only --episodes 800 --seed 42

# 鎵归噺杩愯鎵€鏈夌瓥鐣ワ紙闇€澶栭儴鑴氭湰锛?for strategy in local-only remote-only offloading-only resource-only \\
                comprehensive-no-migration comprehensive-migration; do
    python experiments/td3_strategy_suite/run_strategy_training.py \\
        --strategy $strategy --suite-id ablation_20231029 --episodes 800
done
```
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

# 娣诲姞椤圭洰鏍圭洰褰曞埌Python璺緞
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config import config
from train_single_agent import (
    _apply_global_seed_from_env,
    _build_scenario_config,
    train_single_algorithm,
)
from utils.unified_reward_calculator import UnifiedRewardCalculator

StrategyPreset = Dict[str, Any]  # 绛栫暐棰勮閰嶇疆绫诲瀷

# ========== 鍒濆鍖栫粺涓€濂栧姳璁＄畻鍣?==========
# 浣跨敤缁熶竴濂栧姳璁＄畻鍣ㄧ‘淇濅笌璁粌鏃剁殑濂栧姳鍑芥暟涓€鑷?_reward_calculator = None


def _get_reward_calculator() -> UnifiedRewardCalculator:
    """鑾峰彇鍏ㄥ眬濂栧姳璁＄畻鍣ㄥ疄渚嬶紙寤惰繜鍒濆鍖栵級"""
    global _reward_calculator
    if _reward_calculator is None:
        _reward_calculator = UnifiedRewardCalculator(algorithm="general")
    return _reward_calculator

# ========== 榛樿瀹為獙鍙傛暟 ==========
DEFAULT_EPISODES = 800   # 榛樿璁粌杞暟锛堝钩琛℃敹鏁涜川閲忎笌鏃堕棿鎴愭湰锛?DEFAULT_SEED = 42        # 榛樿闅忔満绉嶅瓙锛堜繚璇佸疄楠屽彲閲嶅鎬э級

# ========== 绛栫暐鎵ц椤哄簭 ==========
# 鎸夌収澶嶆潅搴﹂€掑鎺掑垪锛氫粠鍗曚竴鍔熻兘鍒板畬鏁寸郴缁?# 杩欎釜椤哄簭涔熺敤浜庣敓鎴愬姣斿浘琛ㄦ椂鐨勫睍绀洪『搴?STRATEGY_ORDER = [
    "local-only",                    # 鍩哄噯1锛氱函鏈湴璁＄畻
    "remote-only",                   # 鍩哄噯2锛氬己鍒惰繙绋嬪嵏杞?    "offloading-only",               # 妯″潡1锛氬嵏杞藉喅绛?    "resource-only",                 # 妯″潡2锛氳祫婧愬垎閰?    "comprehensive-no-migration",    # 妯″潡3锛氬畬鏁寸郴缁燂紙鏃犺縼绉伙級
    "comprehensive-migration",       # 瀹屾暣绯荤粺锛氭墍鏈夋ā鍧楀惎鐢?]



def _build_override(
    num_rsus: Optional[int],
    num_uavs: Optional[int],
    allow_local: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    构建策略专用的场景覆盖配置（在默认配置基础上做最小修改）

    【功能】避免与默认命令 `python train_single_agent.py --algorithm TD3`
    出现配置漂移，仅调整与策略严格相关的参数，确保对比公平。

    【参数】
    num_rsus: Optional[int] - RSU 数量（None 表示沿用默认配置）
    num_uavs: Optional[int] - UAV 数量（None 表示沿用默认配置）
    allow_local: Optional[bool] - 是否允许本地处理（None 表示沿用默认值）

    【返回值】Dict[str, Any] - 覆盖字典，仅包含被修改的键

    【设计原则】
    - 继承默认的车辆规模、覆盖半径等基础参数
    - 仅调整 RSU/UAV 数量或本地执行开关
    - 固定拓扑，减少跨策略比较的随机性
    """
    _ = _build_scenario_config()  # 调用以确保配置加载，与默认训练保持同步
    override: Dict[str, Any] = {}

    if num_rsus is not None:
        override["num_rsus"] = num_rsus
    if num_uavs is not None:
        override["num_uavs"] = num_uavs
    if allow_local is not None:
        override["allow_local_processing"] = allow_local

    override["override_topology"] = True
    return override

STRATEGY_PRESETS: "OrderedDict[str, StrategyPreset]" = OrderedDict(
    [
        # ===== 绛栫暐1: 绾湰鍦版墽琛屽熀鍑?=====
        (
            "local-only",
            {
                "description": "All tasks execute locally; edge nodes and migration are disabled.",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 鏃犵紦瀛樺崗浣?                "disable_migration": True,        # 绂佺敤杩佺Щ
                "enforce_offload_mode": "local_only",  # 寮哄埗鏈湴鎵ц
                "override_scenario": _build_override(num_rsus=0, num_uavs=0, allow_local=True),  # 鏃犺竟缂樿妭鐐?            },
        ),
        # ===== 绛栫暐2: 寮哄埗杩滅▼鍗歌浇鍩哄噯 =====
        (
            "remote-only",
            {
                "description": "Edge-Only (single RSU offload)",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 鏃犵紦瀛樺崗浣?                "disable_migration": True,        # 绂佺敤杩佺Щ
                "enforce_offload_mode": "remote_only",  # 寮哄埗杩滅▼鍗歌浇
                "override_scenario": _build_override(num_rsus=1, num_uavs=0, allow_local=False),  # 鍗昍SU锛岀鐢ㄦ湰鍦?            },
        ),
        # ===== 绛栫暐3: 鍗歌浇鍐崇瓥妯″潡 =====
        (
            "offloading-only",
            {
                "description": "Agent decides between local execution and a single RSU; migration and multi-node balancing are disabled.",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": False,      # 鏃犵紦瀛樺崗浣?                "disable_migration": True,        # 绂佺敤杩佺Щ
                "enforce_offload_mode": None,     # 鍏佽鏅鸿兘鍗歌浇鍐崇瓥
                "override_scenario": _build_override(num_rsus=1, num_uavs=0, allow_local=True),  # 鍗昍SU锛屽彲閫夋湰鍦?            },
        ),
        # ===== 绛栫暐4: 璧勬簮鍒嗛厤妯″潡 =====
        (
            "resource-only",
            {
                "description": "All tasks must offload; the agent only balances load across RSUs/UAVs (no migration).",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 鍚敤缂撳瓨鍗忎綔
                "disable_migration": True,        # 绂佺敤杩佺Щ
                "enforce_offload_mode": "remote_only",  # 寮哄埗鍗歌浇
                "override_scenario": _build_override(num_rsus=4, num_uavs=2, allow_local=False),  # 澶氳妭鐐癸紝绂佺敤鏈湴
            },
        ),
        # ===== 绛栫暐5: 瀹屾暣绯荤粺锛堟棤杩佺Щ锛?=====
        (
            "comprehensive-no-migration",
            {
                "description": "Full offloading and resource allocation with migration disabled.",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 鍚敤缂撳瓨鍗忎綔
                "disable_migration": True,        # 绂佺敤杩佺Щ
                "enforce_offload_mode": None,     # 鍏佽鏅鸿兘鍐崇瓥
                "override_scenario": _build_override(num_rsus=4, num_uavs=2, allow_local=True),  # 瀹屾暣閰嶇疆
            },
        ),
        # ===== 绛栫暐6: 瀹屾暣TD3绯荤粺 =====
        (
            "comprehensive-migration",
            {
                "description": "Complete TD3 strategy: offloading, resource allocation, and migration all enabled.",
                "algorithm": "TD3",
                "episodes": DEFAULT_EPISODES,
                "use_enhanced_cache": True,       # 鍚敤缂撳瓨鍗忎綔
                "disable_migration": False,       # 鍚敤杩佺Щ锛堝叧閿樊寮傦級
                "enforce_offload_mode": None,     # 鍏佽鏅鸿兘鍐崇瓥
                "override_scenario": None,  # 默认配置（与 python train_single_agent.py --algorithm TD3 对齐）
            },
        ),
    ]
)


def tail_mean(values: Any) -> float:
    """
    璁＄畻搴忓垪鍚庡崐閮ㄥ垎鐨勭ǔ瀹氬潎鍊?    
    銆愬姛鑳姐€?    浣跨敤璁粌鍚庢湡鏁版嵁璁＄畻鎬ц兘鎸囨爣鐨勭ǔ瀹氬潎鍊硷紝閬垮厤鍓嶆湡鎺㈢储闃舵鐨勯珮鏂瑰樊骞叉壈銆?    杩欐槸璇勪及鏀舵暃鍚庢€ц兘鐨勬爣鍑嗘柟娉曘€?    
    銆愬弬鏁般€?    values: Any - 鎬ц兘鎸囨爣搴忓垪锛堝姣忚疆鐨勬椂寤躲€佽兘鑰楃瓑锛?    
    銆愯繑鍥炲€笺€?    float - 鍚庢湡绋冲畾闃舵鐨勫潎鍊?    
    銆愯绠楃瓥鐣ャ€?    - 搴忓垪闀垮害 >= 100: 浣跨敤鍚?0%鏁版嵁锛堝厖鍒嗘敹鏁涳級
    - 搴忓垪闀垮害 >= 50: 浣跨敤鏈€鍚?0杞暟鎹?    - 搴忓垪闀垮害 < 50: 浣跨敤鍏ㄩ儴鏁版嵁锛堝揩閫熸祴璇曟ā寮忥級
    
    銆愯鏂囧搴斻€?    璇勪及鏀舵暃鎬ц兘鏃讹紝閫氬父浣跨敤璁粌鍚庢湡鐨勫钩鍧囧€间綔涓烘渶缁堟€ц兘鎸囨爣
    """
    if not values:
        return 0.0
    seq = list(map(float, values))
    length = len(seq)
    if length >= 100:
        subset = seq[length // 2 :]  # 鍚?0%
    elif length >= 50:
        subset = seq[-30:]           # 鏈€鍚?0杞?    else:
        subset = seq                 # 鍏ㄩ儴鏁版嵁
    return float(sum(subset) / max(1, len(subset)))


def compute_raw_cost(delay_mean: float, energy_mean: float) -> float:
    """
    璁＄畻缁熶竴浠ｄ环鍑芥暟鐨勫師濮嬪€?    
    銆愬姛鑳姐€?    浣跨敤缁熶竴濂栧姳璁＄畻鍣ㄨ绠椾唬浠凤紝纭繚涓庤缁冩椂浣跨敤鐨勫鍔卞嚱鏁板畬鍏ㄤ竴鑷淬€?    璇ュ嚱鏁扮敤浜庣瓥鐣ラ棿鐨勫叕骞冲姣斻€?    
    銆愬弬鏁般€?    delay_mean: float - 骞冲潎鏃跺欢锛堢锛?    energy_mean: float - 骞冲潎鑳借€楋紙鐒﹁€筹級
    
    銆愯繑鍥炲€笺€?    float - 褰掍竴鍖栧悗鐨勫姞鏉冧唬浠?    
    銆愯绠楀叕寮忋€?    Raw Cost = 蠅_T 路 (T / T_target) + 蠅_E 路 (E / E_target)
    鍏朵腑锛?    - 蠅_T = 2.0锛堟椂寤舵潈閲嶏級
    - 蠅_E = 1.2锛堣兘鑰楁潈閲嶏級
    - T_target = 0.4s锛堟椂寤剁洰鏍囧€硷紝鐢ㄤ簬褰掍竴鍖栵級
    - E_target = 1200J锛堣兘鑰楃洰鏍囧€硷紝鐢ㄤ簬褰掍竴鍖栵級
    
    銆愯鏂囧搴斻€?    浼樺寲鐩爣锛歮inimize 蠅_T路鏃跺欢 + 蠅_E路鑳借€?    璇ユ寚鏍囪秺灏忥紝绯荤粺鎬ц兘瓒婂ソ
    
    銆愪慨澶嶈鏄庛€?    鉁?淇鍚庯細浣跨敤latency_target鍜宔nergy_target锛屼笌璁粌鏃剁殑濂栧姳璁＄畻瀹屽叏涓€鑷?    鉁?淇鍓嶏細閿欒浣跨敤浜哾elay_normalizer(0.2)鍜宔nergy_normalizer(1000)
    鉁?澶嶇敤缁熶竴妯″潡锛岄伒寰狣RY鍘熷垯
    """
    weight_delay = float(config.rl.reward_weight_delay)      # 蠅_T = 2.0
    weight_energy = float(config.rl.reward_weight_energy)    # 蠅_E = 1.2
    
    # 鉁?淇锛氫娇鐢ㄤ笌璁粌鏃跺畬鍏ㄤ竴鑷寸殑褰掍竴鍖栧洜瀛?    calc = _get_reward_calculator()
    delay_normalizer = calc.latency_target  # 0.4锛堜笌璁粌涓€鑷达級
    energy_normalizer = calc.energy_target  # 1200.0锛堜笌璁粌涓€鑷达級
    
    return (
        weight_delay * (delay_mean / max(delay_normalizer, 1e-6))
        + weight_energy * (energy_mean / max(energy_normalizer, 1e-6))
    )


def update_summary(
    suite_path: Path,
    strategy: str,
    preset: StrategyPreset,
    result: Dict[str, Any],
    metrics: Dict[str, float],
    artifacts: Dict[str, str],
    episodes: int,
    seed: int,
) -> None:
    """
    鏇存柊绛栫暐瀹為獙鎽樿JSON鏂囦欢
    
    銆愬姛鑳姐€?    灏嗗崟涓瓥鐣ョ殑璁粌缁撴灉杩藉姞鍒皊uite绾у埆鐨剆ummary.json涓€?    璇ユ枃浠舵眹鎬绘墍鏈夌瓥鐣ョ殑鎬ц兘鎸囨爣锛岀敤浜庡悗缁殑瀵规瘮鍒嗘瀽鍜屽彲瑙嗗寲銆?    
    銆愬弬鏁般€?    suite_path: Path - Suite鏍圭洰褰曡矾寰?    strategy: str - 绛栫暐鍚嶇О锛堝"local-only"锛?    preset: StrategyPreset - 绛栫暐棰勮閰嶇疆
    result: Dict[str, Any] - 璁粌杩斿洖鐨勫畬鏁寸粨鏋?    metrics: Dict[str, float] - 璁＄畻鍚庣殑鎬ц兘鎸囨爣
    artifacts: Dict[str, str] - 鐢熸垚鐨勬枃浠惰矾寰?    episodes: int - 瀹為檯璁粌杞暟
    seed: int - 浣跨敤鐨勯殢鏈虹瀛?    
    銆愯繑鍥炲€笺€?    None锛堢洿鎺ュ啓鍏ユ枃浠讹級
    
    銆恠ummary.json缁撴瀯銆?    {
      "suite_id": "20231029_123456",
      "created_at": "2023-10-29T12:34:56",
      "updated_at": "2023-10-29T13:45:00",
      "strategies": {
        "local-only": {
          "description": "...",
          "metrics": {"delay_mean": 0.15, ...},
          "controls": {...},
          "artifacts": {...}
        },
        ...
      }
    }
    
    銆愪娇鐢ㄥ満鏅€?    - 姣忎釜绛栫暐璁粌瀹屾垚鍚庤皟鐢ㄤ竴娆?    - 鏀寔澧為噺鏇存柊锛堝彲澶氭杩愯涓嶅悓绛栫暐锛?    - 鍚庣画鍙敤浜庣敓鎴愬姣斿浘琛?    """
    summary_path = suite_path / "summary.json"
    
    # ========== 鍔犺浇鎴栧垱寤簊ummary ==========
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {
            "suite_id": suite_path.name,
            "created_at": datetime.now().isoformat(),
            "strategies": {},
        }
    
    # ========== 鏇存柊绛栫暐淇℃伅 ==========
    summary["updated_at"] = datetime.now().isoformat()
    summary["strategies"][strategy] = {
        "description": preset["description"],
        "timestamp": result.get("timestamp"),
        "algorithm": result.get("algorithm"),
        "episodes": episodes,
        "seed": seed,
        "controls": {
            "use_enhanced_cache": preset["use_enhanced_cache"],
            "disable_migration": preset["disable_migration"],
            "enforce_offload_mode": preset["enforce_offload_mode"],
        },
        "metrics": metrics,
        "artifacts": artifacts,
    }
    
    # ========== 鎸佷箙鍖栦繚瀛?==========
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def copy_artifacts(
    result: Dict[str, Any],
    strategy_dir: Path,
) -> Dict[str, str]:
    """
    澶嶅埗璁粌浜х敓鐨勬牳蹇冩枃浠跺埌绛栫暐涓撳睘鐩綍
    
    銆愬姛鑳姐€?    灏唗rain_single_agent.py鐢熸垚鐨勭粨鏋滄枃浠讹紙JSON/鍥捐〃/鎶ュ憡锛夊鍒跺埌
    绛栫暐涓撳睘鏂囦欢澶癸紝渚夸簬鍚庣画鍒嗘瀽鍜屽綊妗ｃ€?    
    銆愬弬鏁般€?    result: Dict[str, Any] - 璁粌缁撴灉瀛楀吀锛堝寘鍚玜lgorithm銆乼imestamp绛夛級
    strategy_dir: Path - 绛栫暐涓撳睘鐩綍锛堝results/td3_strategy_suite/suite_id/local-only/锛?    
    銆愯繑鍥炲€笺€?    Dict[str, str] - 澶嶅埗鍚庣殑鏂囦欢璺緞瀛楀吀
        {
          "training_json": "path/to/training_results.json",
          "training_chart": "path/to/training_overview.png",
          "training_report": "path/to/training_report.html"
        }
    
    銆愬鍒剁殑鏂囦欢銆?    1. training_results_{timestamp}.json - 瀹屾暣璁粌鏁版嵁
    2. training_overview.png - 璁粌鏇茬嚎鍥捐〃
    3. training_report_{timestamp}.html - 璁粌鎶ュ憡
    
    銆愭簮鏂囦欢浣嶇疆銆?    results/single_agent/{algorithm}/
    
    銆愮洰鏍囦綅缃€?    results/td3_strategy_suite/{suite_id}/{strategy}/
    """
    algorithm = str(result.get("algorithm", "")).lower()
    timestamp = result.get("timestamp")
    artifacts: Dict[str, str] = {}

    # ========== 纭畾婧愭枃浠惰矾寰?==========
    src_root = Path("results") / "single_agent" / algorithm
    if timestamp:
        json_name = f"training_results_{timestamp}.json"
        report_name = f"training_report_{timestamp}.html"
    else:
        json_name = "training_results.json"
        report_name = "training_report.html"
    chart_name = "training_overview.png"

    # ========== 瀹氫箟澶嶅埗娓呭崟 ==========
    copies = [
        ("training_json", src_root / json_name),
        ("training_chart", src_root / chart_name),
        ("training_report", src_root / report_name),
    ]
    
    # ========== 鎵ц澶嶅埗 ==========
    strategy_dir.mkdir(parents=True, exist_ok=True)
    for key, src in copies:
        if src.exists():
            dst = strategy_dir / src.name
            shutil.copy2(src, dst)  # copy2淇濈暀鍏冩暟鎹?            artifacts[key] = str(dst)
    
    return artifacts


def run_strategy(strategy: str, args: argparse.Namespace) -> None:
    """
    鎵ц鍗曚釜绛栫暐鐨勫畬鏁磋缁冩祦绋?    
    銆愬姛鑳姐€?    杩欐槸涓绘墽琛屽嚱鏁帮紝瀹屾垚浠ヤ笅浠诲姟锛?    1. 鍔犺浇绛栫暐閰嶇疆
    2. 璁剧疆闅忔満绉嶅瓙
    3. 璋冪敤train_single_algorithm杩涜璁粌
    4. 璁＄畻绋冲畾鎬ц兘鎸囨爣
    5. 澶嶅埗缁撴灉鏂囦欢
    6. 鏇存柊summary.json
    7. 鎵撳嵃缁撴灉鎽樿
    
    銆愬弬鏁般€?    strategy: str - 绛栫暐鍚嶇О锛堝繀椤诲湪STRATEGY_PRESETS涓畾涔夛級
    args: argparse.Namespace - 鍛戒护琛屽弬鏁?    
    銆愬伐浣滄祦绋嬨€?    姝ラ1: 楠岃瘉绛栫暐鍚嶇О
    姝ラ2: 璁剧疆闅忔満绉嶅瓙锛堜繚璇佸彲閲嶅鎬э級
    姝ラ3: 璋冪敤璁粌鍑芥暟锛堜娇鐢ㄧ瓥鐣ヤ笓灞為厤缃級
    姝ラ4: 浠庤缁冪粨鏋滀腑鎻愬彇鎬ц兘鎸囨爣
    姝ラ5: 璁＄畻绋冲畾鍧囧€硷紙浣跨敤tail_mean锛?    姝ラ6: 澶嶅埗鐢熸垚鐨勬枃浠跺埌绛栫暐鐩綍
    姝ラ7: 鏇存柊姹囨€籎SON
    姝ラ8: 鎵撳嵃缁撴灉
    
    銆愯緭鍑烘枃浠剁粨鏋勩€?    results/td3_strategy_suite/{suite_id}/
    鈹溾攢鈹€ summary.json                    # 姹囨€绘枃浠讹紙鎵€鏈夌瓥鐣ワ級
    鈹溾攢鈹€ local-only/
    鈹?  鈹溾攢鈹€ training_results_*.json
    鈹?  鈹溾攢鈹€ training_overview.png
    鈹?  鈹斺攢鈹€ training_report_*.html
    鈹溾攢鈹€ remote-only/
    鈹?  鈹斺攢鈹€ ...
    鈹斺攢鈹€ ...
    
    銆愭€ц兘鎸囨爣銆?    - delay_mean: 骞冲潎浠诲姟鏃跺欢锛堢锛?    - energy_mean: 骞冲潎鎬昏兘鑰楋紙鐒﹁€筹級
    - completion_mean: 浠诲姟瀹屾垚鐜囷紙0-1锛?    - raw_cost: 缁熶竴浠ｄ环鍑芥暟锛堣秺灏忚秺濂斤級
    """
    # ========== 姝ラ1: 鍔犺浇绛栫暐閰嶇疆 ==========
    if strategy not in STRATEGY_PRESETS:
        raise ValueError(f"Unknown strategy: {strategy}")
    preset = STRATEGY_PRESETS[strategy]

    # ========== 姝ラ2: 纭畾璁粌鍙傛暟 ==========
    episodes = args.episodes or preset["episodes"]
    seed = args.seed if args.seed is not None else DEFAULT_SEED

    # ========== 姝ラ3: 璁剧疆闅忔満绉嶅瓙 ==========
    os.environ["RANDOM_SEED"] = str(seed)
    _apply_global_seed_from_env()

    # ========== 姝ラ4: 鎵ц璁粌 ==========
    # 浣跨敤args.silent鍙傛暟鎺у埗闈欓粯妯″紡锛堟壒閲忓疄楠屾帹鑽愬紑鍚級
    silent = getattr(args, 'silent', True)  # 榛樿闈欓粯妯″紡锛岄伩鍏嶄氦浜掑崱浣?    results = train_single_algorithm(
        preset["algorithm"],
        num_episodes=episodes,
        silent_mode=silent,
        override_scenario=preset["override_scenario"],
        use_enhanced_cache=preset["use_enhanced_cache"],
        disable_migration=preset["disable_migration"],
        enforce_offload_mode=preset["enforce_offload_mode"],
    )

    # ========== 姝ラ5: 鎻愬彇鎬ц兘鎸囨爣 ==========
    episode_metrics: Dict[str, Any] = results.get("episode_metrics", {})
    delay_mean = tail_mean(episode_metrics.get("avg_delay", []))
    energy_mean = tail_mean(episode_metrics.get("total_energy", []))
    completion_mean = tail_mean(episode_metrics.get("task_completion_rate", []))
    raw_cost = compute_raw_cost(delay_mean, energy_mean)

    # ========== 姝ラ6: 鍑嗗杈撳嚭鐩綍 ==========
    suite_id = args.suite_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_path = Path(args.output_root) / suite_id
    strategy_dir = suite_path / strategy
    suite_path.mkdir(parents=True, exist_ok=True)

    # ========== 姝ラ7: 澶嶅埗缁撴灉鏂囦欢 ==========
    artifacts = copy_artifacts(results, strategy_dir)

    # ========== 姝ラ8: 姹囨€绘€ц兘鎸囨爣 ==========
    metrics = {
        "delay_mean": delay_mean,
        "energy_mean": energy_mean,
        "completion_mean": completion_mean,
        "raw_cost": raw_cost,
    }
    
    # ========== 姝ラ9: 鏇存柊summary.json ==========
    update_summary(suite_path, strategy, preset, results, metrics, artifacts, episodes, seed)

    # ========== 姝ラ10: 鎵撳嵃缁撴灉鎽樿 ==========
    print("\n=== Strategy Run Completed ===")
    print(f"Suite ID        : {suite_id}")
    print(f"Strategy        : {strategy}")
    print(f"Episodes        : {episodes}")
    print(f"Seed            : {seed}")
    print(f"Average Delay   : {delay_mean:.4f} s")
    print(f"Average Energy  : {energy_mean:.2f} J")
    print(f"Completion Rate : {completion_mean:.3f}")
    print(f"Raw Cost        : {raw_cost:.4f}")
    if artifacts:
        print("Artifacts:")
        for key, path in artifacts.items():
            print(f"  - {key}: {path}")
    summary_path = suite_path / "summary.json"
    print(f"Summary updated : {summary_path}")


def build_argument_parser() -> argparse.ArgumentParser:
    """
    鏋勫缓鍛戒护琛屽弬鏁拌В鏋愬櫒
    
    銆愬姛鑳姐€?    瀹氫箟鑴氭湰鐨勫懡浠よ鎺ュ彛锛屾敮鎸佺伒娲婚厤缃缁冨弬鏁般€?    
    銆愯繑鍥炲€笺€?    argparse.ArgumentParser - 閰嶇疆濂界殑鍙傛暟瑙ｆ瀽鍣?    
    銆愬懡浠よ鍙傛暟銆?    --strategy: str (蹇呴渶)
        - 绛栫暐鍚嶇О锛屽彲閫夊€? local-only, remote-only, offloading-only, 
          resource-only, comprehensive-no-migration, comprehensive-migration
    
    --episodes: int (鍙€?
        - 璁粌杞暟锛岄粯璁?00
        - 蹇€熸祴璇曞彲鐢?0-100锛屽畬鏁村疄楠屽缓璁?00-1000
    
    --seed: int (鍙€?
        - 闅忔満绉嶅瓙锛岄粯璁?2
        - 鐢ㄤ簬淇濊瘉瀹為獙鍙噸澶嶆€?    
    --suite-id: str (鍙€?
        - Suite鏍囪瘑绗︼紝鐢ㄤ簬灏嗗涓瓥鐣ュ綊涓哄悓涓€缁勫疄楠?        - 鏈寚瀹氭椂鑷姩鐢熸垚鏃堕棿鎴筹紙YYYYMMDD_HHMMSS锛?    
    --output-root: str (鍙€?
        - 杈撳嚭鏍圭洰褰曪紝榛樿"results/td3_strategy_suite"
    
    --silent: bool (鍙€?
        - 闈欓粯妯″紡锛屽噺灏戣缁冭繃绋嬬殑杈撳嚭
        - 鉁?娉ㄦ剰锛氭壒閲忓疄楠岃剼鏈粯璁ゅ凡鍚敤闈欓粯妯″紡锛屾棤闇€鎵嬪姩浜や簰
    
    銆愪娇鐢ㄧず渚嬨€?    # 鉁?榛樿闈欓粯杩愯锛堟棤闇€鎵嬪姩浜や簰锛屾帹鑽愶級
    # 鍩烘湰鐢ㄦ硶
    python run_strategy_training.py --strategy local-only
    
    # 鎸囧畾鍙傛暟 - 鑷姩淇濆瓨鎶ュ憡锛屾棤浜哄€煎畧杩愯
    python run_strategy_training.py --strategy comprehensive-migration \\
        --episodes 1000 --seed 123 --suite-id exp_ablation_v1
    
    # 蹇€熸祴璇曪紙宸查粯璁ら潤榛橈級
    python run_strategy_training.py --strategy offloading-only \\
        --episodes 50
    
    # 馃挕 濡傞渶浜や簰寮忕‘璁や繚瀛樻姤鍛婏紝娣诲姞 --interactive 鍙傛暟
    python run_strategy_training.py --strategy td3-full \\
        --episodes 500 --interactive
    """
    parser = argparse.ArgumentParser(
        description="Run TD3 under a specific strategy baseline and collect results."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        required=True,
        choices=list(STRATEGY_PRESETS.keys()),
        help="Select which strategy preset to train.",
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        help="Override number of training episodes (default 800)."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        help="Random seed (default 42)."
    )
    parser.add_argument(
        "--suite-id", 
        type=str, 
        help="Suite identifier to group multiple runs."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/td3_strategy_suite",
        help="Root folder where per-strategy results will be stored.",
    )
    parser.add_argument(
        "--silent", 
        action="store_true", 
        help="Run training in silent mode."
    )
    return parser


def main() -> None:
    """
    鑴氭湰涓诲叆鍙ｅ嚱鏁?    
    銆愬姛鑳姐€?    瑙ｆ瀽鍛戒护琛屽弬鏁板苟鍚姩绛栫暐璁粌娴佺▼銆?    
    銆愭墽琛屾祦绋嬨€?    1. 鏋勫缓鍙傛暟瑙ｆ瀽鍣?    2. 瑙ｆ瀽鍛戒护琛屽弬鏁?    3. 璋冪敤run_strategy鎵ц璁粌
    
    銆愰敊璇鐞嗐€?    - 鏈煡绛栫暐鍚嶇О锛歏alueError
    - 鍙傛暟缂哄け锛歛rgparse鑷姩鎻愮ず
    - 璁粌杩囩▼閿欒锛氱敱train_single_algorithm澶勭悊
    """
    parser = build_argument_parser()
    args = parser.parse_args()
    run_strategy(args.strategy, args)


if __name__ == "__main__":
    main()

