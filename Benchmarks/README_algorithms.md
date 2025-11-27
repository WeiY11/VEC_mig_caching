# Benchmarks-aligned standalone algorithms

位置：`Benchmarks/`

每个文件都是**独立实现**，不依赖仓库里已有的 TD3/DDPG/SAC 等代码。默认支持任意符合 Gym 接口的环境（`reset()` / `step(action)`），动作空间为连续型。

## 文件列表
- `lillicrap_ddpg_vanilla.py`：Lillicrap et al. DDPG（400/300 MLP，tanh 输出，OU 噪声，τ=0.001，Critic L2）。
- `cam_td3_uav_mec.py`：TD3 变体（Wang et al. 场景）：400/300 MLP，policy delay=2，target smoothing，结构化动作（offload softmax + RSU/UAV 选择 + 缓存/迁移控制）。
- `zhang_robust_sac.py`：RoNet 风格鲁棒 SAC：观测/对抗扰动，QoS 惩罚，可选自动温度，256/256 MLP。
- `liu_online_sa.py`：在线模拟退火（Liu & Cao），改进了温度衰减（0.9，min 1e-3，iters 800）。
- `nath_dynamic_offload_heuristic.py`：Nath & Wu 启发式，按延迟/能耗加权的信道/队列/负载评分。
- `local_only_policy.py`：纯本地处理基线。
- `vec_env_adapter.py`：将上述算法适配到仓库的 VEC 单智能体仿真器，动作/状态与 OPTIMIZED_TD3 一致。
- `reward_adapter.py`：复用 VEC 的统一奖励计算。
- `run_compare_with_optimized_td3.py`：运行 OPTIMIZED_TD3 参考。
- `run_benchmarks_vs_optimized_td3.py`：在 VEC 仿真器里跑基线并可选跑 OPTIMIZED_TD3 做对比；内置 Local/Heuristic/SA 的环境评估。

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
