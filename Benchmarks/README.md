# Benchmarks 说明

本目录提供与仓库 OPTIMIZED_TD3 环境一致的对比基线及运行脚本，默认每个对比实验跑 5 组种子（`--groups 5`）。

## 算法实现

- `lillicrap_ddpg_vanilla.py`：原始 DDPG（400/300 MLP、tanh 输出、OU 噪声、软更新）。
- `wang_ippo_uav_mec.py`：IPPO（Independent PPO）用于 UAV-MEC 联合任务卸载和迁移优化（Wang et al. 2025）。
- `zhang_robust_sac.py`：鲁棒 SAC（对抗扰动 + QoS 惩罚，可选自动温度）。
- `liu_online_sa.py`：在线模拟退火基线（温度衰减 0.9，最小 1e-3，默认 800 次）。
- `nath_dynamic_offload_heuristic.py`：动态卸载启发式。
- `local_only_policy.py`：纯本地处理基线。
- `vec_env_adapter.py`：适配 OPTIMIZED_TD3 的环境封装。
- `reward_adapter.py`：统一奖励计算。

## 运行脚本（每个实验 5 组对比）

- 带宽对比（MHz）  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg ippo ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --bandwidths 12 18 24`
- 边缘总计算资源对比（Hz，总量按 5:1 分配 RSU:UAV）  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --edge-compute 3e10 6e10`
- 车辆数对比  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --vehicles-list 8 12 16`
- 数据大小对比（KB，固定为 min/max）  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --data-sizes 256 512`
- 任务到达率对比  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --arrival-rates 1.5 2.0 2.5`
- 全量扫参与对比示例  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --bandwidths 12 18 24 --edge-compute 3e10 6e10 --vehicles-list 8 12 16 --data-sizes 256 512 --arrival-rates 1.5 2.0 2.5`
- 仅参考跑法（无基线）  
  `python Benchmarks/run_compare_with_optimized_td3.py --episodes 400 --seed 42`

## 如何训练这些算法

虽然这些脚本主要用于“基准对比”，但您也可以单独训练某个算法。

### 1. 训练单个基准算法 (如 SAC)

**注意**：`Benchmarks/` 目录下的 `zhang_robust_sac.py` 等文件是算法实现模块，**不能直接运行**（没有 `main` 函数）。
必须通过统一的运行脚本 `run_benchmarks_vs_optimized_td3.py` 来调用它们。

**正确命令**：

```bash
# 训练 SAC 算法 500 轮
python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg sac --episodes 500 --groups 1 --seed 42
```

支持的算法代码（`--alg` 参数）：

- `ippo`: Independent PPO for UAV-MEC (`wang_ippo_uav_mec.py`)
- `ddpg`: Vanilla DDPG (`lillicrap_ddpg_vanilla.py`)
- `sac`: Robust SAC (`zhang_robust_sac.py`)
- `local`: Local Only Policy (`local_only_policy.py`)
- `heuristic`: Dynamic Offload Heuristic (`nath_dynamic_offload_heuristic.py`)
- `sa`: Simulated Annealing (`liu_online_sa.py`)

### 2. 训练 OPTIMIZED_TD3 (本仓库核心算法)

虽然可以通过 `--run-ref` 在基准脚本中运行它，但推荐直接使用主训练脚本以获得更详细的日志和检查点：

```bash
# 标准训练
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 800
```

### 3. 批量训练所有基线

如果您想一次性训练所有基线算法以获取数据：

```bash
python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --episodes 200 --groups 1
```

## 提示

- 结果自动保存到 `results/benchmarks_sweeps/`，包含每组种子和汇总均值/方差。
- 可按需调整 `--episodes`、`--alg` 列表或扫参取值，保持 `--groups 5` 即可满足“五组数据”要求。\*\*\*
