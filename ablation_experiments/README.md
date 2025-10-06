# 🔬 TD3算法消融实验

## 📁 目录结构

```
ablation_experiments/
├── README.md                    # 本文件
├── run_ablation_td3.py         # 主实验脚本
├── ablation_configs.py         # 消融实验配置
├── results/                    # 实验结果
│   ├── Full-System/           # 完整系统
│   ├── No-Cache/              # 无缓存
│   ├── No-Migration/          # 无迁移
│   ├── No-Priority/           # 无优先级
│   ├── No-Adaptive/           # 无自适应
│   ├── No-Collaboration/      # 无协作
│   └── Minimal-System/        # 最小系统
└── analysis/                   # 分析结果和图表

```

## 🎯 实验目的

通过消融实验验证VEC边缘计算系统中各个模块的有效性：
1. **边缘缓存模块** - 减少数据传输时延
2. **任务迁移模块** - 负载均衡和任务完成率
3. **任务优先级队列** - 紧急任务优先处理
4. **自适应控制** - 动态调整缓存和迁移策略
5. **RSU协作缓存** - 多RSU间数据共享

## 📊 实验配置

### 完整系统配置 (Full-System)
- ✅ 边缘缓存
- ✅ 任务迁移
- ✅ 任务优先级
- ✅ 自适应控制
- ✅ RSU协作

### 消融配置
1. **No-Cache**: 禁用边缘缓存模块
2. **No-Migration**: 禁用任务迁移模块
3. **No-Priority**: 禁用任务优先级队列
4. **No-Adaptive**: 禁用自适应控制
5. **No-Collaboration**: 禁用RSU间协作缓存
6. **Minimal-System**: 最小系统（仅基础功能）

## 🚀 使用方法

### 快速测试（30轮）
```bash
cd ablation_experiments
python run_ablation_td3.py --episodes 30 --quick
```

### 标准实验（200轮）
```bash
python run_ablation_td3.py --episodes 200
```

### 完整实验（500轮，论文质量）
```bash
python run_ablation_td3.py --episodes 500 --full
```

### 单独运行某个配置
```bash
python run_ablation_td3.py --config No-Cache --episodes 100
```

## 📈 评估指标

### 主要指标
- **平均时延** (avg_delay): 任务平均完成时延
- **总能耗** (total_energy): 系统总能量消耗
- **任务完成率** (task_completion_rate): 成功完成任务比例

### 辅助指标
- **缓存命中率** (cache_hit_rate): 数据缓存命中比例
- **迁移成功率** (migration_success_rate): 任务迁移成功比例
- **数据丢失率** (data_loss_ratio): 任务丢弃比例

## 📊 结果分析

实验完成后，会自动生成：
1. **JSON结果文件**: `results/ablation_results_<timestamp>.json`
2. **对比图表**: `analysis/ablation_comparison.png`
3. **雷达图**: `analysis/module_impact_radar.png`
4. **分析报告**: `analysis/ablation_analysis_<timestamp>.json`
5. **HTML报告**: `analysis/ablation_report_<timestamp>.html`

## 🎓 论文使用建议

### 消融实验表格
```latex
\begin{table}[h]
\centering
\caption{消融实验结果对比}
\begin{tabular}{lccc}
\hline
配置 & 平均时延(s) & 总能耗(J) & 完成率(\%) \\
\hline
Full-System & 0.182 & 956.3 & 98.7 \\
No-Cache & 0.245 & 1023.5 & 96.2 \\
No-Migration & 0.198 & 982.1 & 95.8 \\
\ldots & \ldots & \ldots & \ldots \\
\hline
\end{tabular}
\end{table}
```

### 影响力分析
实验会自动计算各模块对系统性能的影响百分比，按重要性排序。

## ⚙️ 技术细节

- **算法**: TD3 (Twin Delayed DDPG)
- **状态空间**: 130维 (车辆60 + RSU54 + UAV16)
- **动作空间**: 18维 (任务分配 + 缓存迁移控制)
- **奖励函数**: 统一奖励 = -(2.0×时延 + 1.2×能耗) - 0.02×丢弃任务数
- **网络结构**: Actor(400-400) + Twin Critic(400-400)

## 📝 注意事项

1. **独立环境**: 此文件夹完全独立，不影响主项目
2. **结果保存**: 所有结果保存在 `ablation_experiments/results/` 
3. **随机种子**: 固定随机种子确保可重复性
4. **资源需求**: 完整实验需要3-4小时（7个配置 × 200轮）

