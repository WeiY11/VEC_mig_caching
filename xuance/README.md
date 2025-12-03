# XuanCe 训练模块 - VEC边缘计算系统

本目录包含基于 XuanCe 框架风格的统一训练接口，提供 YAML 配置管理、命令行参数覆盖和多种算法支持。

## 目录结构

```
xuance/
├── train.py           # 主训练脚本
├── vec_env.py         # VEC环境Gymnasium封装
├── __init__.py        # 模块初始化
├── configs/           # YAML配置文件目录
│   ├── td3_vec.yaml   # TD3算法配置
│   ├── sac_vec.yaml   # SAC算法配置
│   ├── ppo_vec.yaml   # PPO算法配置
│   └── ddpg_vec.yaml  # DDPG算法配置
└── README.md          # 本文件
```

## 快速开始

### 基本训练

```bash
# 使用默认算法(OPTIMIZED_TD3)训练
python xuance/train.py

# 指定算法
python xuance/train.py --method td3 --episodes 200

# 使用YAML配置文件
python xuance/train.py --config xuance/configs/td3_vec.yaml
```

### 命令行参数覆盖

```bash
# 覆盖环境参数
python xuance/train.py --method sac --num-vehicles 20 --num-rsus 6 --num-uavs 3

# 覆盖训练参数
python xuance/train.py --episodes 500 --seed 123 --device cuda:0
```

## 支持的算法

### DRL 算法

| 算法 | 参数 | 说明 |
|------|------|------|
| OPTIMIZED_TD3 | `--method optimized_td3` | **默认**，项目主算法 |
| TD3 | `--method td3` | Twin Delayed DDPG |
| SAC | `--method sac` | Soft Actor-Critic |
| PPO | `--method ppo` | Proximal Policy Optimization |
| DDPG | `--method ddpg` | Deep Deterministic PG |
| DQN | `--method dqn` | Deep Q-Network |
| CAM_TD3 | `--method cam_td3` | 缓存感知迁移TD3 |
| TD3-LE | `--method td3_le` | 延迟能耗优化TD3 |

### 对比方案 (Baselines)

| 算法 | 参数 | 说明 |
|------|------|------|
| Local-Only | `--method local` | 本地处理策略 |
| Heuristic | `--method heuristic` | 动态卸载启发式 (Nath) |
| SA | `--method sa` | 模拟退火 (Liu) |
| Benchmark-TD3 | `--method benchmark_td3` | 论文版TD3实现 |
| Benchmark-DDPG | `--method benchmark_ddpg` | Lillicrap DDPG |
| Benchmark-SAC | `--method benchmark_sac` | Zhang RobustSAC |

## 完整命令行参数

```
python xuance/train.py --help

参数说明:
  --method, -m      训练算法 (默认: optimized_td3)
  --config, -c      YAML配置文件路径
  --episodes        训练轮次
  --max-steps       每轮最大步数 (默认: 200)
  --seed            随机种子 (默认: 42)
  --device          计算设备 (默认: cuda:0)
  --num-vehicles    车辆数量
  --num-rsus        RSU数量
  --num-uavs        UAV数量
  --arrival-rate    任务到达率
  --log-dir         日志目录 (默认: ./logs/)
  --model-dir       模型保存目录 (默认: ./models/)
  --verbose, -v     详细输出
```

## 配置优先级

配置参数按以下优先级生效（高优先级覆盖低优先级）：

```
命令行参数 > YAML配置文件 > Python默认值
```

## YAML 配置示例

```yaml
# xuance/configs/td3_vec.yaml
dl_toolbox: "torch"
project_name: "VEC_TD3"
device: "cuda:0"

# TD3算法参数
actor_learning_rate: 0.00009
critic_learning_rate: 0.00009
batch_size: 384
buffer_size: 100000
gamma: 0.99

# VEC环境参数
vec_config:
  num_vehicles: 12
  num_rsus: 4
  num_uavs: 2
  arrival_rate: 3.5
  max_episode_steps: 200
```

## 使用示例

### 1. 训练主算法

```bash
python xuance/train.py --method optimized_td3 --episodes 200
```

### 2. 对比实验

```bash
# 运行所有对比方案
python xuance/train.py --method local --episodes 50
python xuance/train.py --method heuristic --episodes 50
python xuance/train.py --method benchmark_ddpg --episodes 100
```

### 3. 不同环境规模

```bash
# 小规模
python xuance/train.py --num-vehicles 8 --num-rsus 2 --episodes 200

# 大规模
python xuance/train.py --num-vehicles 24 --num-rsus 8 --num-uavs 4 --episodes 300
```

## Python API

```python
from xuance.vec_env import VECEnv, register_vec_env

# 注册环境
register_vec_env()

# 创建环境
env = VECEnv(config={
    'vec_config': {
        'num_vehicles': 12,
        'num_rsus': 4,
        'num_uavs': 2
    }
})

# 标准Gym接口
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

## 注意事项

1. 首次运行会自动注册VEC环境到Gymnasium
2. 训练结果保存在 `results/single_agent/<算法名>/` 目录下
3. 模型检查点保存在 `--model-dir` 指定的目录
4. 日志可通过TensorBoard查看：`tensorboard --logdir=./logs/`
