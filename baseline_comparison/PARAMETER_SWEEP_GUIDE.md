# 📊 参数对比折线图使用指南

## 🎯 功能说明

**参数对比折线图**用于展示算法在不同参数配置下的性能变化，是论文中展示**扩展性分析**和**参数敏感性**的重要工具。

### 典型应用场景

#### 1. **车辆数扫描** ⭐（最常用）
```
X轴: 车辆数（8, 12, 16, 20, 24）
Y轴: 性能指标（时延、能耗、完成率、目标函数）
多条线: 不同算法（TD3, DDPG, Greedy, Random等）

用途: 证明算法在不同负载下的表现
论文价值: 展示TD3的扩展性和鲁棒性
```

#### 2. **负载强度扫描**
```
X轴: 任务到达率（1.0, 1.5, 2.0, 2.5, 3.0 tasks/s）
Y轴: 性能指标
用途: 分析算法在不同负载压力下的性能
```

#### 3. **网络条件扫描**
```
X轴: 带宽（10, 20, 30, 40, 50 MHz）
Y轴: 性能指标
用途: 分析网络条件对算法性能的影响
```

---

## 🚀 快速开始

### 最简单的使用（推荐）

#### 测试TD3的车辆数扩展性
```bash
cd baseline_comparison

# 快速测试（3个车辆数，30轮，约3分钟）
python run_parameter_sweep.py --param vehicles --values 8 12 16 --algorithms TD3 --episodes 30

# 标准测试（5个车辆数，200轮，约25分钟）
python run_parameter_sweep.py --param vehicles --values 8 12 16 20 24 --algorithms TD3 --episodes 200
```

### 对比多个算法
```bash
# TD3 vs DDPG vs Greedy（3个算法）
python run_parameter_sweep.py --param vehicles --values 8 12 16 20 24 --algorithms TD3 DDPG Greedy --episodes 200

# 预计时间: 5个车辆数 × 3个算法 × 200轮 × 1.5秒 ≈ 75分钟
```

### 使用批处理脚本（Windows）
```bash
# 快速测试
test_parameter_sweep.bat

# 查看生成的图表
parameter_sweep_results/analysis/parameter_comparison_lines.png
```

---

## 📊 输出图表

### 主图表: parameter_comparison_lines.png

**布局**: 2×2四个子图

```
┌─────────────────────┬─────────────────────┐
│  时延 vs 车辆数     │  能耗 vs 车辆数     │
│  (Delay)            │  (Energy)           │
├─────────────────────┼─────────────────────┤
│  完成率 vs 车辆数   │  目标函数 vs 车辆数 │
│  (Completion)       │  (Objective)        │
└─────────────────────┴─────────────────────┘
```

**特点**:
- ✅ 每个数据点用**空心marker**标记
- ✅ 不同算法用**不同形状和颜色**
- ✅ 线宽2.5px，清晰易读
- ✅ 300 DPI高分辨率
- ✅ 网格辅助线

### 数据点marker样式
```
TD3:    ○ 圆形   (蓝色 #2E86AB)
DDPG:   □ 方形   (紫红 #A23B72)
SAC:    △ 三角   (橙色 #F18F01)
PPO:    ◇ 菱形   (红色 #C73E1D)
DQN:    ▽ 倒三角 (灰色 #6C757D)
Greedy: ⬟ 五边形 (青色 #17BEBB)
Random: ★ 星形   (紫色 #9B59B6)
```

---

## 📋 详细用法

### 基本语法
```bash
python run_parameter_sweep.py \
    --param <参数类型> \
    --values <参数值列表> \
    --algorithms <算法列表> \
    --episodes <轮次>
```

### 参数说明

#### --param（扫描参数类型）
```bash
--param vehicles     # 车辆数扫描（默认，当前唯一支持）
--param load         # 负载扫描（TODO）
--param bandwidth    # 带宽扫描（TODO）
```

#### --values（参数值列表）
```bash
--values 8 12 16 20 24              # 5个车辆数配置
--values 8 12 16                    # 3个配置（快速测试）
--values 4 8 12 16 20 24 28         # 7个配置（详细分析）
```

#### --algorithms（算法列表）
```bash
# DRL算法
--algorithms TD3                     # 单个算法
--algorithms TD3 DDPG SAC            # 多个DRL算法
--algorithms TD3 DDPG PPO SAC DQN    # 所有DRL算法

# 启发式算法
--algorithms Greedy Random RoundRobin LocalFirst NearestNode

# 混合对比
--algorithms TD3 DDPG Greedy Random  # DRL + Baseline
```

#### --episodes（每配置轮次）
```bash
--episodes 50      # 快速测试
--episodes 200     # 标准实验（推荐）
--episodes 500     # 高质量数据
```

#### --quick（快速模式）
```bash
--quick            # 等价于 --episodes 50
```

---

## 🎓 论文实验示例

### 实验1: TD3车辆数扩展性（核心）
```bash
# 展示TD3在不同负载下的性能
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 20 24 \
    --algorithms TD3 \
    --episodes 200 \
    --seed 42

# 时间: 5 × 200 × 1.5秒 ≈ 25分钟
# 用途: 论文核心图表，展示算法扩展性
```

### 实验2: 多算法车辆数对比
```bash
# 对比所有DRL算法的扩展性
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 20 24 \
    --algorithms TD3 DDPG SAC PPO DQN \
    --episodes 200

# 时间: 5 × 5 × 200 × 1.5秒 ≈ 2小时
# 用途: 展示TD3相对其他DRL算法的优势
```

### 实验3: DRL vs Baseline扩展性
```bash
# 对比学习方法vs启发式方法
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 20 24 \
    --algorithms TD3 Greedy Random \
    --episodes 200

# 时间: 5 × 3 × 200 × 1.5秒 ≈ 50分钟
# 用途: 展示学习方法的显著优势
```

---

## 📈 生成的图表解读

### 子图1: 时延 vs 车辆数
**含义**: 车辆数增加时任务时延的变化
- **理想曲线**: 缓慢上升或基本持平
- **差的曲线**: 快速上升（负载敏感）

**TD3期望**: 在固定拓扑下保持相对稳定

### 子图2: 能耗 vs 车辆数
**含义**: 车辆数增加时系统能耗的变化
- **理想曲线**: 线性或次线性增长
- **差的曲线**: 超线性增长（效率降低）

**TD3期望**: 能耗增长可控

### 子图3: 完成率 vs 车辆数
**含义**: 车辆数增加时任务完成率的变化
- **理想曲线**: 始终保持高完成率（>95%）
- **差的曲线**: 车辆数多时完成率下降

**TD3期望**: 始终保持高完成率

### 子图4: 目标函数 vs 车辆数 ⭐
**含义**: 车辆数增加时综合性能的变化
- **J = 2.0×Delay + 1.2×Energy/600**
- **理想曲线**: 最低，增长最慢
- **差的曲线**: 高值，快速增长

**TD3期望**: 所有车辆数配置下都是最优

---

## 🎨 图表特点

### 数据点标记
- **大小**: 8px（清晰可见）
- **样式**: 空心（白色填充）
- **边框**: 2.5px（醒目）
- **颜色**: 与曲线颜色一致

### 线条样式
- **线宽**: 2.5px（粗实线）
- **透明度**: 0.85（略透明以便重叠时可见）
- **平滑**: 自然连接数据点

### 视觉效果
```
      ●────●────●────●────●  TD3 (蓝色，圆形)
     /
    ■────■────■────■────■    DDPG (紫红，方形)
   /
  ▲────▲────▲────▲────▲      SAC (橙色，三角)

8    12    16    20    24    车辆数
```

---

## ⏱️ 时间估算

### 单算法扫描
| 车辆数配置 | 轮次 | 预计时间 |
|-----------|------|---------|
| 3个 (8,12,16) | 50 | ~4分钟 |
| 3个 (8,12,16) | 200 | ~15分钟 |
| 5个 (8,12,16,20,24) | 200 | ~25分钟 |
| 5个 (8,12,16,20,24) | 500 | ~60分钟 |

### 多算法对比
| 算法数 | 车辆数配置 | 轮次 | 预计时间 |
|-------|-----------|------|---------|
| 2算法 | 5个 | 200 | ~50分钟 |
| 3算法 | 5个 | 200 | ~75分钟 |
| 5算法 | 5个 | 200 | ~125分钟 |

---

## 📂 输出文件结构

```
baseline_comparison/parameter_sweep_results/
├── analysis/
│   └── parameter_comparison_lines.png   # 参数对比折线图⭐
├── sweep_summary_YYYYMMDD_HHMMSS.json   # 汇总结果
├── TD3_8v.json                           # TD3 8辆车结果
├── TD3_12v.json                          # TD3 12辆车结果
├── TD3_16v.json                          # ...
├── DDPG_8v.json                          # DDPG 8辆车结果
└── ...
```

---

## 🎓 论文使用建议

### 图表标题建议
```latex
图X: 不同车辆数下的算法性能对比
(Network topology: 4 RSU + 2 UAV, Fixed)

子图说明:
(a) 平均任务时延随车辆数的变化
(b) 系统能耗随车辆数的变化
(c) 任务完成率随车辆数的变化
(d) 复合目标函数随车辆数的变化
```

### 文字描述模板
```
图X展示了在固定网络拓扑（4 RSU + 2 UAV）下，不同车辆数
配置的性能对比。可以观察到：

(1) 时延方面：随着车辆数从8增加到24，TD3的时延增长仅为
    X%，显著低于DDPG的Y%和Greedy的Z%，表明TD3在高负载
    下具有更好的性能稳定性。

(2) 能耗方面：TD3在所有配置下都保持最低能耗，相比次优算法
    平均降低A%。

(3) 目标函数：TD3在复合指标J=2.0×Delay+1.2×Energy上始终
    最优，且在车辆数增加时保持相对稳定，证明了算法的鲁棒性。
```

### 关键观察点
✅ **TD3曲线最平缓** → 扩展性好  
✅ **TD3值最低** → 性能最优  
✅ **TD3波动小** → 鲁棒性强  
✅ **TD3全局最优** → 各配置下都优于对比算法

---

## 💡 实验设计建议

### 方案A: 快速验证（推荐先做）
```bash
# 3个车辆数，2个算法，30轮
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 \
    --algorithms TD3 Greedy \
    --episodes 30

# 时间: 3 × 2 × 30 × 1.5秒 ≈ 4分钟
# 用途: 验证功能，查看图表效果
```

### 方案B: TD3扩展性分析（核心）
```bash
# 5个车辆数，TD3单算法，200轮
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 20 24 \
    --algorithms TD3 \
    --episodes 200

# 时间: 5 × 200 × 1.5秒 ≈ 25分钟
# 用途: 论文核心图表，展示TD3扩展性
```

### 方案C: 完整算法对比（全面）
```bash
# 5个车辆数，3个关键算法，200轮
python run_parameter_sweep.py \
    --param vehicles \
    --values 8 12 16 20 24 \
    --algorithms TD3 DDPG Greedy \
    --episodes 200

# 时间: 5 × 3 × 200 × 1.5秒 ≈ 75分钟
# 用途: 全面对比，论文补充图表
```

---

## 📊 图表示例解读

### 示例数据
假设实验结果如下：

| 车辆数 | TD3时延 | DDPG时延 | Greedy时延 |
|-------|---------|----------|-----------|
| 8 | 0.18s | 0.19s | 0.25s |
| 12 | 0.22s | 0.24s | 0.35s |
| 16 | 0.24s | 0.28s | 0.48s |
| 20 | 0.26s | 0.34s | 0.65s |
| 24 | 0.28s | 0.42s | 0.85s |

### 图表解读
```
时延 (s)
  1.0 |                           
      |                         ◆ Greedy (陡峭上升)
  0.8 |                       ◆
      |                     ◆
  0.6 |                   ◆
      |                 ◆
  0.4 |               □ DDPG (中等上升)
      |             □
  0.2 | ○────○────○────○ TD3 (平缓，最优)
      |
  0.0 └────┴────┴────┴────┴────
      8    12   16   20   24    车辆数

关键发现:
1. TD3增长最平缓（扩展性最好）
2. TD3始终最低（性能最优）
3. Greedy增长陡峭（负载敏感）
```

---

## 🔬 实验配置建议

### 车辆数选择
```bash
# 稀疏扫描（快速）
--values 8 16 24          # 3个配置

# 标准扫描（推荐）
--values 8 12 16 20 24    # 5个配置

# 密集扫描（详细）
--values 6 8 10 12 14 16 18 20 22 24    # 10个配置
```

### 轮次选择
```bash
--episodes 30      # 快速测试，趋势观察
--episodes 100     # 快速实验，基本可靠
--episodes 200     # 标准实验，论文级（推荐）
--episodes 500     # 高质量，统计可靠
```

### 算法组合建议

#### 组合1: TD3单算法（扩展性分析）
```bash
--algorithms TD3
# 用途: 展示您的算法在不同负载下的表现
# 时间: 最短
```

#### 组合2: TD3 vs 最佳Baseline（优势对比）
```bash
--algorithms TD3 Greedy
# 用途: 展示学习方法vs启发式方法的差距
# 时间: 较短
```

#### 组合3: TD3 vs 其他DRL（技术对比）
```bash
--algorithms TD3 DDPG SAC
# 用途: 展示TD3在DRL方法中的优势
# 时间: 中等
```

#### 组合4: 全面对比（完整分析）
```bash
--algorithms TD3 DDPG SAC Greedy Random
# 用途: 论文完整实验，全面展示优势
# 时间: 较长
```

---

## 📊 输出数据

### JSON结果文件
每个配置保存一个文件：
```json
{
  "algorithm": "TD3",
  "num_episodes": 200,
  "avg_delay": 0.226,
  "std_delay": 0.003,
  "avg_energy": 679.9,
  "std_energy": 45.2,
  "avg_completion_rate": 0.9716,
  ...
}
```

### 汇总文件（sweep_summary_*.json）
```json
{
  "8": {
    "TD3": {...},
    "DDPG": {...},
    "Greedy": {...}
  },
  "12": {
    "TD3": {...},
    ...
  },
  ...
}
```

---

## 🎯 最佳实践

### 1. 实验顺序
```
步骤1: 快速测试（3个车辆数，TD3，30轮）
       ↓ 验证功能正常
步骤2: 标准实验（5个车辆数，TD3，200轮）
       ↓ 获取核心数据
步骤3: 扩展对比（5个车辆数，多算法，200轮）
       ↓ 获取完整对比
步骤4: 分析图表，撰写论文
```

### 2. 参数值选择
```bash
# 包含端点
最小值: 8辆车（轻负载）
最大值: 24辆车（重负载）

# 包含默认值
中间值: 12辆车（标准配置）

# 均匀分布
间隔: 4辆车
```

### 3. 数据可靠性
```bash
# 提高可靠性的方法：
1. 增加训练轮次（200 → 500）
2. 使用多个随机种子
3. 取稳定期数据（后50%）
```

---

## 💻 高级用法

### 使用不同随机种子
```bash
# Seed 42
python run_parameter_sweep.py --values 8 12 16 --algorithms TD3 --episodes 200 --seed 42

# Seed 2025
python run_parameter_sweep.py --values 8 12 16 --algorithms TD3 --episodes 200 --seed 2025

# Seed 3407
python run_parameter_sweep.py --values 8 12 16 --algorithms TD3 --episodes 200 --seed 3407

# 然后手动合并结果计算均值和置信区间
```

### 只扫描部分车辆数
```bash
# 只测试低负载
python run_parameter_sweep.py --values 8 12 --algorithms TD3 DDPG --episodes 200

# 只测试高负载
python run_parameter_sweep.py --values 20 24 --algorithms TD3 DDPG --episodes 200
```

### 单点补充实验
```bash
# 如果某个配置需要重新运行
python run_parameter_sweep.py --values 16 --algorithms TD3 --episodes 200
```

---

## 🔧 图表定制

### 修改marker样式
编辑 `run_parameter_sweep.py` 中的 `generate_parameter_comparison_plots` 方法：

```python
# 修改marker形状
markers = ['o', 's', '^', 'D', 'v', 'p', '*']
# 'o': 圆  's': 方  '^': 三角  'D': 菱形
# 'v': 倒三角  'p': 五边形  '*': 星形

# 修改颜色
colors = ['#2E86AB', '#A23B72', '#F18F01', ...]

# 修改marker大小
markersize=8  # 默认8，可调整为6-12
```

### 修改线条样式
```python
# 在plot调用中
linewidth=2.5      # 线宽（默认2.5）
alpha=0.85         # 透明度（默认0.85）
linestyle='-'      # 线型（'-' 实线，'--' 虚线，'-.' 点划线）
```

### 添加趋势线
```python
# 可选：添加线性拟合趋势线
from numpy.polynomial import polynomial as P
coefs = P.polyfit(param_values, data, 1)
trend = P.polyval(param_values, coefs)
ax.plot(param_values, trend, '--', alpha=0.3, color=color)
```

---

## 📝 实验记录建议

创建实验日志记录每次扫描：

```
实验日期: 2025-10-10
实验配置:
  - 车辆数: 8, 12, 16, 20, 24
  - 算法: TD3, DDPG, Greedy
  - 轮次: 200
  - 种子: 42

结果观察:
  - TD3在所有配置下性能最优
  - 车辆数从8到24，TD3时延增长18%
  - DDPG时延增长35%，Greedy增长140%
  - 结论: TD3扩展性显著优于对比算法

图表保存:
  - parameter_comparison_lines.png
```

---

## ✅ 验证清单

运行参数扫描前：
- [ ] 确认固定拓扑优化器工作正常
- [ ] 确认TD3环境变量读取正常
- [ ] 至少5GB可用磁盘空间
- [ ] 足够的运行时间

运行后检查：
- [ ] 所有配置都成功完成
- [ ] 图表清晰，数据点可见
- [ ] 趋势符合预期
- [ ] JSON文件保存完整

---

## 🚀 快速开始（3步）

### 步骤1: 快速测试
```bash
cd baseline_comparison
test_parameter_sweep.bat
# 或
python run_parameter_sweep.py --values 8 12 16 --algorithms TD3 --episodes 30
```

### 步骤2: 查看图表
```bash
# 打开
parameter_sweep_results/analysis/parameter_comparison_lines.png
```

### 步骤3: 运行完整实验
```bash
# 满意后运行完整版
python run_parameter_sweep.py --values 8 12 16 20 24 --algorithms TD3 --episodes 200
```

---

## 🎯 推荐实验流程

### 论文核心实验
```bash
# 第1步: TD3扩展性（必做）
python run_parameter_sweep.py \
    --param vehicles --values 8 12 16 20 24 \
    --algorithms TD3 --episodes 200

# 第2步: TD3 vs DDPG对比（推荐）
python run_parameter_sweep.py \
    --param vehicles --values 8 12 16 20 24 \
    --algorithms TD3 DDPG --episodes 200

# 第3步: TD3 vs Baseline（可选）
python run_parameter_sweep.py \
    --param vehicles --values 8 12 16 20 24 \
    --algorithms TD3 Greedy Random --episodes 200
```

---

## 📞 常见问题

**Q: 为什么需要固定拓扑（4 RSU + 2 UAV）？**  
A: 固定基础设施，公平对比算法在不同负载（车辆数）下的适应能力。

**Q: 能否同时扫描车辆数和RSU数？**  
A: 当前不支持。建议分别进行单参数扫描。

**Q: 如何解释"扩展性好"？**  
A: 性能曲线平缓，车辆数增加时性能降低幅度小。

**Q: 某个配置失败了怎么办？**  
A: 可以单独重新运行该配置（--values 16）。

**Q: 如何合并多个seed的结果？**  
A: 需要手动处理JSON文件，计算均值和置信区间。

---

## 📖 相关文档

- `IMPROVED_FEATURES.md` - 新功能总览
- `README_v2.md` - 完整使用指南  
- `DISCRETE_PLOTS_GUIDE.md` - 训练过程可视化（不同于参数扫描）

---

**更新日期**: 2025-10-10  
**功能**: 参数对比折线图生成  
**状态**: 已实现，待测试


