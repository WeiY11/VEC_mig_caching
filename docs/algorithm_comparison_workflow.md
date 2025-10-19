# VEC多算法对比实验工作流

本文档介绍新增的统一对比实验框架，帮助在定制化的 VEC（Vehicular Edge Computing）场景中系统比较不同类型算法（DRL、启发式、元启发式）的性能表现。

## 1. 工作流概览

新框架由以下组件组成：

- `config/algorithm_comparison_config.json`：定义公共场景、默认超参数以及需要比较的算法列表。
- `experiments/algorithm_comparison.py`：核心执行器，负责统一调度不同类别的算法、收集指标并汇总统计。
- `visualization/algorithm_comparison.py`：生成对比指标的可视化图表。
- `run_algorithm_comparison.py`：命令行入口脚本，读取配置、运行实验并输出结果。

所有结果默认保存在 `results/algorithm_comparison/<timestamp>` 目录下，包括：

- `aggregated_results.json`：按算法聚合后的性能指标（均值、标准差、最大值、最小值等）。
- `per_seed_runs.json`：每个随机种子的详细记录，便于复现实验。
- `summary.csv`：便于导入其它分析工具的压缩版摘要。
- `metric_overview.png`：核心指标（平均奖励、时延、能耗、完成率等）的柱状对比图。

## 2. 配置文件说明

`config/algorithm_comparison_config.json` 由三个部分组成：

```json
{
  "scenario": {
    "num_vehicles": 12,
    "num_rsus": 4,
    "num_uavs": 2,
    "max_steps_per_episode": 200,
    "use_enhanced_cache": true,
    "bandwidth": 18,
    "computation_capacity": 900
  },
  "defaults": {
    "episodes": {
      "drl": 200,
      "heuristic": 150,
      "meta": 200
    },
    "seeds": [42, 2025, 3407],
    "metrics": [
      "avg_reward",
      "avg_delay",
      "avg_energy",
      "avg_completion_rate",
      "cache_hit_rate",
      "migration_success_rate",
      "training_time_hours"
    ]
  },
  "algorithms": [
    {"name": "TD3", "label": "TD3", "category": "drl", "episodes": 200},
    {"name": "Greedy", "label": "贪心", "category": "heuristic"},
    {"name": "LocalOnly", "label": "纯本地", "category": "heuristic"},
    {"name": "RSUOnly", "label": "仅基站", "category": "heuristic"},
    {"name": "GA", "label": "遗传算法", "category": "meta", "params": {"population_size": 24}}
  ],
  "sweeps": [
    {"name": "vehicle_scaling", "parameter": "num_vehicles", "values": [8, 12, 16, 20, 24]},
    {"name": "bandwidth_levels", "parameter": "bandwidth", "values": [10, 15, 20, 25]},
    {"name": "compute_capacity", "parameter": "computation_capacity", "values": [600, 800, 1000, 1200]}
  ]
}
```

- `scenario`：统一的拓扑和仿真设置，会通过覆盖参数传递给 `SingleAgentTrainingEnvironment` 与 `CompleteSystemSimulator`。
- `defaults.episodes`：不同类别算法的默认训练轮次（可在算法条目中单独覆盖）。
- `defaults.seeds`：默认的随机种子列表，框架会对每个种子独立运行并统计均值/方差。
- `defaults.metrics`：希望在结果中汇总和绘图的指标。
- 默认覆盖算法：CAM-TD3、TD3_xuance、Random、RoundRobin、LocalOnly、RSUOnly、SimulatedAnnealing。
- `algorithms`：待比较的算法列表。支持字段：
  - `name`：算法名称（DRL 部分与 `train_single_agent` 中的别名一致，启发式/元启发式与 baseline 工具保持同名）。
  - `category`：`drl`、`heuristic`、`meta` 三类。
  - `episodes`：可选，覆盖默认训练轮次。
  - `seeds`：可选，覆盖默认种子。
  - `params`：可选，传递给算法构造函数的额外参数（目前主要用于 GA/PSO 等元启发式）。

## 3. 运行实验

基础命令：

```bash
python run_algorithm_comparison.py
```

常用参数：

- `--config PATH`：指定自定义配置文件。
- `--output-dir PATH`：修改结果输出目录。
- `--include TD3 Greedy`：仅运行指定算法（名字不区分大小写）。
- `--metrics avg_reward avg_delay`：临时重写需要汇总的指标。
- `--seeds 42 123 456`：覆盖默认随机种子集合。
- `--episodes 100`：统一覆盖未显式指定 `episodes` 的算法轮次。

示例：

```bash
# 仅比较 TD3 与 GA，统一采用 150 轮，输出到自定义目录
python run_algorithm_comparison.py \
  --include TD3 GA \
  --episodes 150 \
  --output-dir results/custom_comparison
```

## 4. 输出解读

`aggregated_results.json` 采用如下结构：

```json
{
  "timestamp": "20251019_193349",
  "results": {
    "TD3": {
      "algorithm": "TD3",
      "category": "drl",
      "episodes": 200,
      "seeds": [42, 2025, 3407],
      "summary": {
        "avg_reward": {"mean": 0.823, "std": 0.057, "min": 0.741, "max": 0.882, "count": 3},
        "avg_delay": {"mean": 0.612, "std": 0.024, ...}
      }
    }
  }
}
```

- `summary` 中的 `mean/std/min/max/count` 分别对应多个随机种子的统计指标。
- `window_size` 表示对每个算法取最近若干轮的稳定窗口（默认 20%）来计算均值。
- 若某项指标在某类算法中不可用，会在 JSON 中填入 `null` 并在 CSV 中留空。

`metric_overview.png` 会为每个指标绘制柱状图（附带标准差误差条），便于直观比较各算法的优劣。

## 5. 与现有代码的集成关系

- DRL 类算法直接复用 `train_single_agent.train_single_algorithm`，并开启 `silent_mode` 避免交互提示。
- 启发式与改进型 baseline 复用了 `baseline_comparison` 中的实现，通过统一的环境封装采集指标。
- 元启发式算法（GA/PSO）加载 `baseline_comparison.individual_runners.metaheuristic` 中的实现，可通过 `params` 注入人口规模、变异率等超参数。
- 统一的随机种子管理和拓扑覆盖保证每次对比实验在同一场景下运行。

## 6. 后续拓展建议

1. **增加新算法**：只需在配置文件中新增条目，并确保在 `experiments/algorithm_comparison.py` 中能找到相应运行入口。
2. **扩展指标**：在 `defaults.metrics` 中加入新指标，同时在 `_summarise_training_output` 或 `_run_baseline_family` 中补充对应的计算逻辑即可。
3. **联动多智能体训练**：可在 `AlgorithmSpec` 中增加 `mode` 字段，并在执行器内分派到 `train_multi_agent.py`，得到一致的统计结果。
4. **批量生成论文图表**：利用导出的 CSV/JSON，可在现有的可视化脚本中进一步生成箱线图、折线图等高阶可视化。

通过该统一框架，可以快速获得多算法在一致 VEC 场景下的定量比较结果，并自动生成可视化图表，支撑性能评估与论文撰写。

## 7. 场景扫描与折线图

- 在配置文件的 `sweeps` 字段中指定参数与取值列表（如车辆数、带宽、覆盖半径、噪声密度、RSU 数量），脚本会自动迭代运行并在 `results/algorithm_comparison/<timestamp>/sweeps/` 下生成对应的 JSON、CSV 与折线图。
- CSV 文件默认提供长表结构（参数值、算法、指标、均值、标准差），方便导入 Origin/Excel 等绘图工具；折线图带有标准差阴影，可直接用于论文插图。
- 每个扫描点的原始汇总结果仍会保存在独立子目录中，可与柱状概览图配合使用；如需扩展更多指标，可在 `metrics` 中添加，或在 `_summarise_training_output` / `_run_baseline_family` 中补充计算逻辑。
