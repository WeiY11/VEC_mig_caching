# TD3 及基线算法对比实验方案（论文级标准）

## 1. 实验目标、假设与变量控制
- **实验目的**：比较深度强化学习算法 TD3 及其改进变体与多种非学习型策略（启发式、元启发式）在车辆边缘计算任务迁移场景中的性能差异，验证 DRL 算法在复杂业务负载与动态网络环境下的优势与局限。
- **研究假设**：
  1. 相较于静态启发式（LocalOnly、RSUOnly、RoundRobin 等），TD3 系列算法在任务完成率与平均时延上具有显著优势。
  2. CAM-TD3（缓存感知改进版）在能耗效率与数据丢失率上优于基础 TD3。
  3. 元启发式（SimulatedAnnealing）在小规模车辆数量下可接近 DRL 算法的表现，但在大规模场景中表现退化。
- **自变量**：调度算法类型、车辆数量（8/12/16/20）、随机种子（控制训练/模拟随机性）。
- **控制变量**：RSU/车载设备算力与缓存配置、通信带宽、调度周期、任务到达过程等环境参数；训练轮数、探索策略等训练参数（详见 3.3 节）。
- **因变量**：任务完成率、平均时延、能耗效率、数据丢失率及补充指标（详见第 2 节）。

## 2. 指标体系与计算定义

| 指标名称 | 物理意义 | 计算方式 | 说明 |
| --- | --- | --- | --- |
| 任务完成率 (Task Completion Rate, %) | 完成迁移且满足时限的任务占比 | `completion_rate = (Σ completed_tasks) / (Σ arrived_tasks)`；从 `episode_metrics.task_completion_rate` 取后段均值或 `final_performance.avg_completion` | 作为主要可靠性指标，输出时乘以 100% |
| 平均时延 (Average Latency, s) | 单任务从生成到完成的平均时间 | `avg_latency = mean(episode_metrics.avg_delay)` 的后段均值 | 对低值敏感，建议给出均值±标准差 |
| 能耗效率 (Energy Efficiency, completed tasks/J) | 单位能耗完成任务数量 | `energy_eff = completed_tasks / total_energy_consumption`；其中 `total_energy_consumption = mean(episode_metrics.total_energy)`，`completed_tasks = completion_rate × total_tasks` | 若仅提供 `total_energy`，按剧本总任务数 200 × episodes 估算 |
| 数据丢失率 (Data Loss Ratio, %) | 因缓存/通信失败导致的数据丢失占比 | `loss_ratio = mean(episode_metrics.data_loss_ratio_bytes)` × 100% | 作为补充鲁棒性指标 |
| 训练收敛速率 (Training Convergence Rate) | 收敛到稳定性能的速度 | 统计达到稳定阈值的 episode 编号；需要对 DRL 算法记录奖励趋势 | 仅适用于需要训练的算法 |
| 训练耗时 (Training Time, h) | 完成既定 episode 的训练时间 | 直接读取 `training_config.training_time_hours` | 对可复现性与资源评估有帮助 |

> **尾段统计说明**：对 DRL 算法取最后 `tail_fraction = 0.2` 的 episodes（建议 160/800）计算均值与标准差，以降低早期探索噪声。对无训练的启发式/元启发式算法，可直接使用全程平均。

## 3. 实验设置

### 3.1 对比算法集合

| 分类 | 算法 | 说明 | 关键超参数 |
| --- | --- | --- | --- |
| 深度强化学习 | TD3（baseline） | 现有单智能体 TD3 实现（命令见 3.2） | `episodes=800`, `γ=0.99`, `τ=0.005`, `policy_delay=2`, `buffer_size=2e6` |
| 深度强化学习 | CAM-TD3 | 引入缓存命中奖励的改进版 | 在 TD3 基础上设定缓存感知系数 `β_cache=0.3`，其余沿用默认 |
| 深度强化学习 | TD3_Xuance | 结合 Xuance 框架超参数的 TD3 实现 | `episodes=600`, `actor_lr=1e-3`, `critic_lr=1e-3`, `exploration_noise=0.1` |
| 启发式 | LocalOnly / RSUOnly | 全本地执行 / 全 RSU 执行 | 无训练；统一运行 `episodes=150` 以进行统计采样 |
| 启发式 | RoundRobin / Random | 按车辆/RSU轮询或随机指派 | 与上同 |
| 元启发式 | SimulatedAnnealing | 退火搜索迁移组合 | 初温 `T0=10`, 冷却率 `α=0.92`, `episodes=180` |
| 元启发式（扩展） | GreedyEnergy (可选) | 以能耗最小优先 | 贪心搜索阈值 `λ_energy=0.6` |

### 3.2 运行命令与脚本化流程

1. **训练命令模板**  
   - DRL 类算法（TD3/CAM-TD3/TD3_Xuance）：  
     `python train_single_agent.py --algorithm {ALG} --episodes {EPISODES} --num-vehicles {N_VEH} --seed {SEED} --output-suffix paper`
   - 启发式/元启发式算法（基于 `run_algorithm_comparison.py` 调度）：  
     `python run_algorithm_comparison.py --include {ALG_NAMES} --episodes {EPISODES} --seeds {SEED_LIST} --output-dir results/paper_comparison`

2. **车辆规模多场景批量脚本**  
   - 推荐在 `experiments/` 目录新增批处理脚本（见交付代码示例），自动遍历 `vehicle_counts = [8, 12, 16, 20]` 与 `seeds = [42, 2025, 3407]`，并写出 `metadata.json` 记录运行时间与超参数。

3. **随机种子与复现性**  
   - Python、NumPy、PyTorch、环境模拟器统一设置：`seed`、`torch.manual_seed`、`np.random.seed`、`random.seed` 。
   - 记录 Git 提交版本号、依赖库版本（见句 `pip freeze > results/paper_comparison/requirements_snapshot.txt`）。

### 3.3 环境参数表

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| RSU 数量 | 4 | 与现有 TD3 结果保持一致，16/20 车辆场景可扩展至 5 作为对照 |
| UAV 数量 | 2 | 若进行敏感性分析，可为 20 车辆场景额外记录 `num_uavs=3` |
| 通信带宽 | 20 MHz | 与 `network_config.bandwidth` 对应；额外 sweep：12/16/20 MHz |
| 任务到达率 | 2.5 req/s | 与 `task_migration_config.task_arrival_rate` 一致 |
| 每回合时隙数 | 200 | `training_config.max_steps_per_episode` |
| 模拟时长 | 1000 s | `system_config.simulation_time` |

> 所有参数写入 `config/paper_experiment_config.json` 并附在数据包中，保证同行复现。

## 4. 多场景实验矩阵与数据记录

- **固定因素**：带宽、RSU/UAV 数、任务分布。
- **变量因素**：车辆数量、算法类别、随机种子。
- **矩阵设计**：

- **极端工况补充实验**：针对“车辆高负载、带宽紧张”两类硬场景，单独开组实验以验证 CAM-TD3 的鲁棒性和性能优势：

| 场景 | 描述 | 关键调整参数 | 车辆数 | 算法集 | 种子 | 预期输出 |
| --- | --- | --- | --- | --- | --- | --- |
| S1 | 基准 | `task_arrival_rate=2.5`，`bandwidth=20 MHz` | 8 | 所有算法 | 42 / 2025 / 3407 | `training_results_*.json` + 指标汇总 CSV |
| S2 | 基准（含现有 TD3 数据） | 同 S1 | 12 | 所有算法 | 同上 | 同上 |
| S3 | 基准 | 同 S1 | 16 | 所有算法 | 同上 | 同上 |
| S4 | 基准 | 同 S1 | 20 | 所有算法 | 同上 | 同上 |
| S5 | 高负载 | `task_arrival_rate=3.5`，任务规模均值 +15% (`task_size_mean=1.12e6`) | 8 / 12 / 16 / 20 | TJCTM（原 CAM-TD3）、TD3、TD3-NoMig、TD3_xuance、RoundRobin、RSUOnly、SA | 种子：42 | 单独目录 `results/paper_comparison/high_load/...`，输出 JSON + CSV |
| S6 | 带宽紧张 | `bandwidth=12 MHz`，`coverage_radius=280 m` | 12 / 20 | CAM-TD3、TD3、RoundRobin、RSUOnly | 同上 | 单独目录 `results/paper_comparison/low_bw/...` |

- **运行建议**：
  1. 为 S5/S6 复制 `config/paper_experiment_config.json`，分别调整参数后保存为 `config/paper_extreme_high_load.json` 与 `config/paper_extreme_low_bw.json`。
  2. 运行命令示例：  
     `python run_algorithm_comparison.py --config config/paper_extreme_high_load.json --include CAM-TD3 TD3 TD3_NoMig TD3_Xuance RoundRobin RSUOnly SimulatedAnnealing --output-dir results/paper_comparison/high_load`  
     `python run_algorithm_comparison.py --config config/paper_extreme_low_bw.json --include CAM-TD3 TD3 RoundRobin RSUOnly --output-dir results/paper_comparison/low_bw`
  3. 亦可直接使用批处理脚本：  
     `python tools/run_extreme_conditions.py --scenario all`（`--dry-run` 可仅打印命令）  
     `python tools/run_extreme_conditions.py --scenario high_load`
  4. 生成指标与图表时分别调用：  
     `python visualization/generate_paper_comparison.py --results-root results/paper_comparison/high_load --output-dir academic_figures/paper_comparison/high_load`  
     `python visualization/generate_paper_comparison.py --results-root results/paper_comparison/low_bw --output-dir academic_figures/paper_comparison/low_bw`

- **数据记录规范**：
  1. 每次运行后写入 `results/paper_comparison/{algorithm}/{vehicles}/{seed}/metadata.json`（算法配置、时间戳、硬件信息）。
  2. 指标脚本汇总输出 `metrics_vehicle_{vehicles}.csv`，包含 `mean`、`std`、`ci95` 字段。
  3. 生成图表以 `PDF+PNG (300 DPI)` 格式存放于 `academic_figures/paper_comparison/`。

## 5. 图表规范与论文撰写要点

- 使用 Matplotlib 或 Seaborn，自定义风格满足 IEEE/Springer 要求：
  - 字体：Times New Roman，字号 10–12；
  - 线宽 ≥ 1.5，主要算法（如 TD3）突出显示；
  - 坐标轴标注含单位（例如 `Average Latency (s)`）；
  - 网格线轻度虚线，图例置于上方或右上角；
  - 输出 `*.pdf` 以便矢量化编辑。
- 图表类型：
  1. **车辆数量 vs 任务完成率**（折线图）；
  2. **车辆数量 vs 平均时延**（折线图）；
  3. **车辆数量 vs 能耗效率**（折线图，Y 轴可采用对数刻度）；
  4. **车辆数量 vs 数据丢失率**（可与能耗效率并列次要轴）；必要时拆分为子图。
- 图表配套文字（Figure caption）需包含：
  - 场景假设简述（车辆数、RSU 数、任务负载）；
  - 重点观察现象（例如 “CAM-TD3 在 20 辆车辆场景下的能耗效率提升 12%”）。
- 数据分析建议：
  - 对 DRL 算法给出标准差或 95% 置信区间；
  - 对启发式算法说明其无方差（或基于 Monte Carlo 运行提供标准差）。

## 6. 预期结果与后续工作

- **预期趋势**：
  1. TD3/CAM-TD3 在 12+ 车辆场景下任务完成率保持 ≥ 95%，平均时延低于 0.6 s；
  2. CAM-TD3 的能耗效率预计相较 TD3 提升 8–12%；
  3. SimulatedAnnealing 在 8 车辆场景可能接近 TD3，但在 20 车辆时延显著劣化；
  4. Random/LocalOnly 等基线在数据丢失率上表现最差。
- **扩展实验**（若时间允许）：
  - 各算法在不同带宽、RSU 密度下的敏感性分析；
  - 针对 TD3 的消融实验（关闭缓存奖励、减少 critic 数量）；
  - 迁移学习/在线更新能力测试。

## 7. 交付与存档

- 提交物包含：原始 JSON、指标汇总 CSV、图表（PDF/PNG）、本实验方案文档。
- 所有脚本与生成文件在 Git 版本库中跟踪，利用 `Makefile` 或 `invoke` 提供一键再现命令。
- 建议附录中列出硬件环境（CPU/GPU/内存）与软件版本（Python、PyTorch、Gym、Matplotlib）。

> 本方案可直接支持学术论文撰写：第 3 节对应 Methodology，第 4 节对应 Experiment Setup，第 5 节对应 Results & Discussion 所需图表规范，第 6 节给出预期分析要点。实验完成后只需将实际指标与图表替换预期描述即可。
