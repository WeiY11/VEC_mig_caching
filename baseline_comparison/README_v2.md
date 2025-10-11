# 🏆 Baseline对比实验系统 v2.0（完善版）

## 📋 概述

完整的算法对比实验框架，用于论文中证明TD3算法的优越性。

### 对比算法（共10种）

#### DRL算法（5种，需训练）
1. **TD3** - Twin Delayed DDPG ⭐（您的方法）
2. **DDPG** - Deep Deterministic Policy Gradient
3. **SAC** - Soft Actor-Critic
4. **PPO** - Proximal Policy Optimization
5. **DQN** - Deep Q-Network

#### 启发式算法（5种，策略执行）
6. **Random** - 随机选择
7. **Greedy** - 贪心最小负载
8. **RoundRobin** - 轮询分配
9. **LocalFirst** - 本地优先
10. **NearestNode** - 最近节点

---

## 🆕 v2.0新增功能

### 1. 固定拓扑支持 ✅
- 统一使用 **4 RSU + 2 UAV**
- 支持可变车辆数（默认12辆）
- 保证公平对比

### 2. 改进的Baseline算法 ✅
- **正确动作维度**: 16维（旧版为18维）
- **智能状态解析**: 自动适配不同车辆数
- **健壮性增强**: 边界条件处理

### 3. 复合指标分析 ✅
- **目标函数**: J = 2.0×Delay + 1.2×Energy
- **自动排名**: 按复合指标排序
- **相对性能**: 计算vs TD3的提升百分比

### 4. 统计显著性检验 ✅
- **t-test**: 独立双样本检验
- **p-value**: 自动计算并显示
- **显著性标记**: ***/\*\*/\*/n.s.

### 5. 增强的可视化 ✅
- **3张高清图表** (300 DPI)
- **目标函数对比图** (新增)
- **TD3突出显示** (橙色标记)

### 6. 灵活运行模式 ✅
- **完整模式**: 所有10个算法
- **DRL模式**: 只运行5个DRL算法
- **Baseline模式**: 只运行5个启发式算法
- **单算法模式**: 测试特定算法
- **多seed模式**: 增强可靠性

---

## 🚀 快速开始

### 方式1: 使用批处理脚本（推荐）

#### Windows
```bash
# 快速测试（10分钟）
run_quick_comparison.bat

# 论文级实验（40分钟）
run_paper_experiment.bat

# 只测试DRL算法（25分钟）
run_drl_only.bat
```

### 方式2: 命令行

#### 快速测试（50轮，验证功能）
```bash
python run_baseline_comparison.py --quick
```

#### 标准对比（200轮，论文数据）
```bash
python run_baseline_comparison.py --episodes 200
```

#### 完整实验（500轮，高质量数据）
```bash
python run_baseline_comparison.py --full
```

---

## 📊 高级用法

### 指定车辆数
```bash
# 16辆车配置
python run_baseline_comparison.py --episodes 200 --num-vehicles 16

# 20辆车配置
python run_baseline_comparison.py --episodes 200 --num-vehicles 20
```

### 分类运行

#### 只运行DRL算法
```bash
python run_baseline_comparison.py --episodes 200 --only-drl
# 运行: TD3, DDPG, SAC, PPO, DQN
# 跳过: Random, Greedy, RoundRobin, LocalFirst, NearestNode
```

#### 只运行启发式算法
```bash
python run_baseline_comparison.py --episodes 100 --only-baseline
# 运行: Random, Greedy, RoundRobin, LocalFirst, NearestNode
# 跳过: TD3, DDPG, SAC, PPO, DQN
```

### 单独测试算法
```bash
# 测试TD3
python run_baseline_comparison.py --algorithm TD3 --episodes 100

# 测试Greedy
python run_baseline_comparison.py --algorithm Greedy --episodes 100

# 快速测试Random
python run_baseline_comparison.py --algorithm Random --episodes 50
```

### 多seed运行（增强可靠性）
```bash
# 3个不同随机种子
python run_baseline_comparison.py --episodes 200 --multi-seed 3
# 种子: 42, 142, 242
```

---

## 📈 输出结果

### 文件结构
```
baseline_comparison/
├── results/                    # 实验数据
│   ├── TD3/
│   │   └── result_TD3.json
│   ├── DDPG/
│   ├── ...
│   └── comparison_summary_YYYYMMDD_HHMMSS.json
│
└── analysis/                   # 分析结果
    ├── performance_comparison.png      # 三指标对比（柱状图）
    ├── objective_comparison.png        # 目标函数对比⭐（柱状图）
    ├── convergence_curves.png          # 收敛曲线（平滑）
    └── discrete_line_plots.png         # 离散折线图⭐（新增）
```

### 数据内容
每个算法的结果包含：
- 平均性能指标（时延、能耗、完成率）
- 标准差（评估稳定性）
- 收敛性指标（初始/最终性能、改善程度）
- 完整episode历史数据

---

## 📊 可视化图表

### 1. performance_comparison.png
**内容**: 三指标柱状图对比
- 平均任务时延
- 系统总能耗  
- 任务完成率

**特点**:
- DRL算法和Baseline算法分组显示
- 蓝色（DRL）vs 红色（Baseline）
- 竖线分隔两组

### 2. objective_comparison.png ⭐（新增）
**内容**: 目标函数值对比
- J = 2.0×Delay + 1.2×Energy
- TD3用橙色突出显示
- 数值标签直接显示在柱状图上

**用途**: 论文主图，展示TD3的综合优势

### 3. convergence_curves.png
**内容**: DRL算法训练过程（4个子图）
- 时延收敛曲线
- 能耗收敛曲线
- 完成率收敛曲线
- 奖励收敛曲线

**特点**:
- 滑动平均（窗口=20）
- 多算法对比
- 网格辅助线

---

## 📊 终端输出

### 1. 性能对比表
```
算法            类型        时延(s)      时延提升     能耗(J)      能耗提升     完成率
TD3            DRL         0.235        +0.0%       512.3        +0.0%       96.5%     ⭐
DDPG           DRL         0.248        +5.5%       543.1        +6.0%       95.2%
Greedy         Baseline    0.312        +32.8%      689.4        +34.6%      92.1%
Random         Baseline    0.456        +94.0%      823.7        +60.8%      85.3%
```

### 2. 复合指标排名
```
算法            类型        目标函数值      相对TD3
TD3            DRL         1.4982         +0.0%        ⭐
DDPG           DRL         1.5843         +5.7%
SAC            DRL         1.6234         +8.4%
Greedy         Baseline    2.0124         +34.3%
Random         Baseline    2.5678         +71.4%
```

### 3. 统计显著性检验
```
算法            时延p-value     能耗p-value     显著性
DDPG            0.012340        0.034560        */*
SAC             0.023000        0.041000        */n.s.
Greedy          0.000012        0.000234        ***/***
Random          0.000001        0.000001        ***/***
```

---

## ⏱️ 时间估算

### 快速模式（--quick, 50轮）
- DRL算法: 5 × 50 × 1.5秒 ≈ 6分钟
- Baseline算法: 5 × 50 × 0.5秒 ≈ 2分钟
- **总计**: ~10分钟

### 标准模式（200轮）
- DRL算法: 5 × 200 × 1.5秒 ≈ 25分钟
- Baseline算法: 5 × 200 × 0.5秒 ≈ 8分钟
- **总计**: ~35-40分钟

### 完整模式（--full, 500轮）
- DRL算法: 5 × 500 × 1.5秒 ≈ 62分钟
- Baseline算法: 5 × 500 × 0.5秒 ≈ 21分钟
- **总计**: ~1.5小时

### 多seed模式（--multi-seed 3, 200轮）
- 基础时间 × 3
- **总计**: ~2小时

---

## 🎓 论文使用建议

### 图表选择
| 图表 | 用途 | 说明 |
|-----|------|------|
| objective_comparison.png | **主图** | 展示TD3综合优势 |
| performance_comparison.png | 辅助图 | 详细指标对比 |
| convergence_curves.png | 补充图 | 训练过程分析 |

### 表格数据
从终端输出复制：
1. **表1**: 性能对比表（均值±标准差）
2. **表2**: 复合指标排名
3. **表3**: 统计显著性检验

### 文字描述模板
```latex
实验在固定拓扑（4 RSU + 2 UAV）、12辆车配置下，
对比了5种DRL算法和5种启发式算法。结果表明，
TD3在复合目标函数J=2.0×Delay+1.2×Energy上显著
优于所有对比算法（p<0.001），相比最佳启发式算法
Greedy，综合性能提升34.3%。
```

---

## 🔧 技术改进详情

### 动作维度修复
```
旧版: 18维（错误）
      3(分配) + 6(RSU?) + 2(UAV) + 7(控制) = 18
      
新版: 16维（正确）
      3(分配) + 4(RSU) + 2(UAV) + 7(控制) = 16
      ↑适配固定拓扑4 RSU + 2 UAV
```

### 状态解析优化
```python
# 旧版: 硬编码状态索引
rsu_start = 60 + i * 9  # 假设12辆车

# 新版: 动态计算
vehicle_offset = num_vehicles * 5
rsu_offset = vehicle_offset
rsu_start = rsu_offset + i * 5  # 适配任意车辆数
```

### 环境变量支持
```python
# 设置车辆数
if num_vehicles != 12:
    overrides = {"num_vehicles": num_vehicles}
    os.environ['TRAINING_SCENARIO_OVERRIDES'] = json.dumps(overrides)
```

---

## ✅ 验证清单

运行完整实验前，建议检查：

- [x] 改进的baseline算法测试通过（已验证）
- [x] 动作维度正确（16维）
- [x] 状态维度适配正确（8/12/16/20/24辆车）
- [ ] scipy已安装（统计检验需要）
  ```bash
  pip install scipy
  ```
- [ ] 至少有2GB可用磁盘空间
- [ ] Python 3.7+

---

## 💡 使用建议

### 初次使用
1. **快速测试**: 先运行 `--quick` 模式，验证所有功能
2. **查看结果**: 检查生成的3张图表
3. **确认无误**: 查看终端输出的统计表格
4. **运行完整**: 再运行200轮标准实验

### 论文准备
1. **标准实验**: 200轮，种子42
2. **多seed验证**: 3个seed，增强可靠性
3. **图表导出**: 使用300 DPI高清图表
4. **数据记录**: 保存JSON结果文件

### 调试问题
1. **单独测试**: 使用 `--algorithm` 测试单个算法
2. **快速迭代**: 使用 `--quick` 减少等待
3. **查看日志**: 终端输出包含详细进度

---

## 🎯 实验流程

### 标准流程（推荐）
```bash
# 步骤1: 快速验证
python run_baseline_comparison.py --quick
# 预计: 10分钟

# 步骤2: 完整实验
python run_baseline_comparison.py --episodes 200
# 预计: 40分钟

# 步骤3: 查看结果
# - analysis/objective_comparison.png（论文主图）
# - analysis/performance_comparison.png（详细对比）
# - analysis/convergence_curves.png（训练过程）
```

### 快速流程（时间有限）
```bash
# 只测试关键算法
python run_baseline_comparison.py --algorithm TD3 --episodes 200
python run_baseline_comparison.py --algorithm DDPG --episodes 200
python run_baseline_comparison.py --algorithm Greedy --episodes 200
python run_baseline_comparison.py --algorithm Random --episodes 200
```

---

## 📝 结果解读

### 性能指标
- **时延**: 越低越好，目标<0.25s
- **能耗**: 越低越好，典型500-800J
- **完成率**: 越高越好，目标>95%

### 复合指标
- **目标函数**: 越低越好
- **TD3基准**: 通常最优或接近最优
- **提升百分比**: 负值表示优于TD3，正值表示劣于TD3

### 统计检验
- **p < 0.05**: 差异显著，有统计意义
- **p ≥ 0.05**: 差异不显著，可能是随机波动
- **用于论文**: 声称"显著优于"需要p < 0.05

---

## 🔬 学术价值

### 对比维度
1. **vs DRL算法**: 证明TD3在深度强化学习方法中的优势
2. **vs 启发式算法**: 证明学习方法优于规则方法
3. **综合对比**: 目标函数全面评估

### 论文贡献点
- TD3综合性能最优
- 统计显著性验证
- 多指标平衡优化
- 收敛速度分析

---

## 🆚 v1.0 vs v2.0

| 特性 | v1.0 | v2.0 |
|-----|------|------|
| 动作维度 | 18维（错误） | 16维（正确）✅ |
| 拓扑配置 | 未明确 | 固定4 RSU+2 UAV✅ |
| 复合指标 | ❌ | ✅ |
| 统计检验 | ❌ | ✅ |
| 可视化图表 | 2张 | 3张✅ |
| 多seed支持 | ❌ | ✅ |
| 灵活运行 | 有限 | 多模式✅ |

---

## 🐛 已知问题

### scipy未安装
**症状**: 统计检验跳过  
**解决**: `pip install scipy`  
**影响**: 不影响其他功能

### 多seed聚合未实现
**症状**: 多seed结果未合并  
**状态**: TODO  
**影响**: 可手动分析多个结果文件

---

## 📞 支持

### 查看帮助
```bash
python run_baseline_comparison.py --help
```

### 查看改进功能
```bash
# 阅读本文件
IMPROVED_FEATURES.md
```

### 旧版兼容
```bash
# 如需使用旧版baseline算法
# 在improved_baseline_algorithms.py中导入
```

---

## 🎉 总结

**v2.0完善版特点**:
- ✅ 完全修复动作维度问题
- ✅ 支持固定拓扑配置
- ✅ 添加学术规范的统计检验
- ✅ 提供论文级图表
- ✅ 灵活多样的运行模式

**推荐使用**: 论文Baseline对比实验  
**预计时间**: 标准模式~40分钟，完整模式~1.5小时  
**输出质量**: 高分辨率（300 DPI）+ 统计检验

