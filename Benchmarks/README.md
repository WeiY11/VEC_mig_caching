# Benchmarks 说明

本目录提供与仓库 OPTIMIZED_TD3 环境一致的对比基线及运行脚本，默认每个对比实验跑 5 组种子（`--groups 5`）。

## 算法实现
- `lillicrap_ddpg_vanilla.py`：原始 DDPG（400/300 MLP、tanh 输出、OU 噪声、软更新）。
- `cam_td3_uav_mec.py`：TD3 变体（双 Q、延迟策略更新、结构化动作）。
- `zhang_robust_sac.py`：鲁棒 SAC（对抗扰动 + QoS 惩罚，可选自动温度）。
- `liu_online_sa.py`：在线模拟退火基线（温度衰减 0.9，最小 1e-3，默认 800 次）。
- `nath_dynamic_offload_heuristic.py`：动态卸载启发式。
- `local_only_policy.py`：纯本地处理基线。
- `vec_env_adapter.py`：适配 OPTIMIZED_TD3 的环境封装。
- `reward_adapter.py`：统一奖励计算。

## 运行脚本（每个实验 5 组对比）
- 带宽对比（MHz）  
  `python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg td3 ddpg sac local heuristic sa --groups 5 --run-ref --episodes 200 --bandwidths 12 18 24`
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

## 提示
- 结果自动保存到 `results/benchmarks_sweeps/`，包含每组种子和汇总均值/方差。
- 可按需调整 `--episodes`、`--alg` 列表或扫参取值，保持 `--groups 5` 即可满足“五组数据”要求。***
