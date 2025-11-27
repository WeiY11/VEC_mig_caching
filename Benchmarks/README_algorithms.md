# Benchmarks-aligned standalone algorithms

位置：`Benchmarks/`

每个文件都是**独立实现**，不依赖仓库里已有的 TD3/DDPG/SAC 等代码。默认支持任意符合 Gym 接口的环境（`reset()` / `step(action)`），动作空间为连续型。

## 文件列表
- `lillicrap_ddpg_vanilla.py`：纯 DDPG（Lillicrap et al.）。含 replay buffer、actor/critic、软更新。
- `cam_td3_uav_mec.py`：带双 Q + 策略延迟的 TD3 变体，用于 UAV + 缓存/迁移场景（Wang et al.）。
- `zhang_robust_sac.py`：鲁棒 SAC，训练时加入观测扰动/动作噪声 + L2 正则（Zhang et al.，RoNet 思路）。
- `liu_online_sa.py`：在线模拟退火（Liu & Cao），针对卸载/缓存参数的快速自适应，无需重训 RL。
- `nath_dynamic_offload_heuristic.py`：动态卸载启发式（Nath & Wu），考虑信道、队列、负载，输出连续偏好向量。

## 运行方式示例
以 TD3 为例（其它类似）：
```bash
python - <<'PY'
import gym
from Benchmarks.cam_td3_uav_mec import CAMTD3Config, train_cam_td3

env = gym.make("Pendulum-v1")  # 替换为你的环境
cfg = CAMTD3Config()
out = train_cam_td3(env, cfg, max_steps=50_000, seed=42)
print("Episodes:", out["episodes"], "AvgR(last10):", sum(out["episode_rewards"][-10:])/10)
PY
```

在线模拟退火示例：
```bash
import numpy as np
from Benchmarks.liu_online_sa import SAConfig, OnlineSimulatedAnnealing

def eval_fn(params: np.ndarray) -> float:
    # 自定义评估函数，返回越大越好
    return -np.sum((params - 0.5)**2)

cfg = SAConfig(max_iters=200)
sa = OnlineSimulatedAnnealing(dim=3, bounds=[(0,1)]*3, cfg=cfg)
print(sa.search(eval_fn))
```

动态卸载启发式示例：
```bash
import numpy as np
from Benchmarks.nath_dynamic_offload_heuristic import DynamicOffloadHeuristic

heur = DynamicOffloadHeuristic(num_rsus=2, num_uavs=1)
state = np.random.rand(5*2 + 5*2 + 5*1)  # 车辆/RSU/UAV 状态拼接
print(heur.select_action(state))
```

## 集成提示
- 如果要接入你自己的仿真环境，确保动作维度与环境匹配，`env.action_space.high` 给出上界。
- 想要自定义超参，修改对应 `Config` dataclass 即可。
- 文件彼此独立，可按需拷贝或裁剪。保持 ASCII 文本，便于移植。
