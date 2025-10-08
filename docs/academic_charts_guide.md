# 📊 学术论文图表生成指南

**版本**: v1.0  
**日期**: 2025-10-08

---

## 🎯 概述

这个工具可以从训练结果自动生成**9种专业学术图表**，完全符合IEEE/ACM/Springer论文标准，300 DPI分辨率，色盲友好配色。

###已生成的6种图表类型（单个算法）

1. **学习曲线（带置信区间）** - Learning Curve with Confidence Intervals
2. **累积分布函数（CDF）** - Cumulative Distribution Function  
3. **箱线图（训练阶段对比）** - Boxplot by Training Phase
4. **指标相关性热力图** - Metric Correlation Heatmap
5. **时延-能耗散点图（含回归线）** - Delay-Energy Scatter Plot
6. **多维性能雷达图** - Multi-dimensional Performance Radar

### 额外支持的5种图表（多算法对比）

7. **收敛性对比** - Convergence Comparison
8. **性能分布箱线图对比** - Boxplot Comparison
9. **CDF对比** - CDF Comparison
10. **小提琴图对比** - Violin Plot Comparison
11. **柱状图对比（含误差棒）** - Bar Chart with Error Bars

---

## 🚀 快速开始

### 单个算法图表

```bash
# 生成单个算法的6种图表
python generate_academic_charts.py results/single_agent/td3/training_results_xxx.json

# 指定输出目录
python generate_academic_charts.py input.json -o my_figures/

# 自定义分辨率
python generate_academic_charts.py input.json --dpi 600
```

### 多算法对比图表

```bash
# 生成多算法对比图表（5种）
python generate_academic_charts.py \\
    results/single_agent/td3/training_results_xxx.json \\
    results/single_agent/ddpg/training_results_xxx.json \\
    results/single_agent/sac/training_results_xxx.json \\
    --compare
```

---

## 📊 图表详解

### 1️⃣ 学习曲线（带置信区间）

**文件名**: `{Algorithm}_learning_curve_variance.png`

**特点**：
- 显示平均奖励的移动平均线
- ±1σ置信区间（约68%置信度）
- ±2σ置信区间（约95%置信度）
- 300 DPI分辨率

**适用场景**：
- ✅ 展示算法收敛过程
- ✅ 说明训练稳定性
- ✅ 论文Section: Convergence Analysis

**LaTeX使用**：
```latex
\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.48\\textwidth]{TD3_learning_curve_variance.png}
  \\caption{Training convergence of TD3 algorithm with confidence intervals.}
  \\label{fig:td3_convergence}
\\end{figure}
```

---

### 2️⃣ 累积分布函数（CDF）

**文件名**: `{Algorithm}_reward_cdf.png`

**特点**：
- 展示奖励值的累积概率分布
- 可直观对比不同算法的分布差异

**适用场景**：
- ✅ 统计分析
- ✅ 性能分布对比
- ✅ 论文Section: Statistical Analysis

**解读方法**：
- 曲线越靠右，性能越好
- 曲线越陡峭，性能越稳定

---

### 3️⃣ 箱线图（训练阶段对比）

**文件名**: `{Algorithm}_reward_boxplot_phases.png`

**特点**：
- 分为前期（1-33%）、中期（34-66%）、后期（67-100%）
- 显示中位数、四分位数、异常值
- 自动标注均值

**适用场景**：
- ✅ 展示训练过程性能变化
- ✅ 识别异常Episode
- ✅ 论文Section: Performance Evolution

**统计信息**：
- 盒子：25%-75%分位数（IQR）
- 线：中位数（红色）、均值（蓝色虚线）
- 须：1.5×IQR范围
- 点：异常值

---

### 4️⃣ 指标相关性热力图

**文件名**: `{Algorithm}_metric_correlation.png`

**特点**：
- 显示所有指标之间的相关系数
- 颜色编码：红色（正相关）、蓝色（负相关）
- 数值标注（-1到+1）

**适用场景**：
- ✅ 分析指标之间的关系
- ✅ 发现权衡（trade-offs）
- ✅ 论文Section: Correlation Analysis

**解读方法**：
- |r| > 0.7：强相关
- 0.4 < |r| < 0.7：中等相关
- |r| < 0.4：弱相关

---

### 5️⃣ 时延-能耗散点图（含回归线）

**文件名**: `{Algorithm}_delay_energy_scatter.png`

**特点**：
- X轴：平均时延，Y轴：总能耗
- 红色虚线：线性回归线
- R²值标注

**适用场景**：
- ✅ 展示时延与能耗的权衡关系
- ✅ 论文核心优化目标分析
- ✅ 论文Section: Optimization Trade-offs

**解读方法**：
- R² > 0.7：强线性关系
- 正相关：时延↑能耗↑（需优化）
- 负相关：时延↑能耗↓（有权衡）

---

### 6️⃣ 多维性能雷达图

**文件名**: `{Algorithm}_performance_radar.png`

**特点**：
- 归一化到[0, 1]
- 包含5个维度：
  - Task Completion（任务完成率）
  - Cache Hit Rate（缓存命中率）
  - Reward（奖励）
  - Low Delay（低时延）
  - Stability（稳定性）

**适用场景**：
- ✅ 多维度综合性能展示
- ✅ 一图概览系统表现
- ✅ 论文Section: Performance Overview

**解读方法**：
- 面积越大，综合性能越好
- 形状越对称，各维度越均衡

---

### 7️⃣ 收敛性对比（多算法）

**文件名**: `algorithms_convergence_comparison.png`

**特点**：
- 多条曲线（不同颜色）
- 移动平均平滑
- 可选置信区间

**适用场景**：
- ✅ 多算法性能对比
- ✅ Baseline比较
- ✅ 论文Section: Algorithm Comparison

---

### 8️⃣ 性能分布箱线图对比（多算法）

**文件名**: `algorithms_boxplot_comparison.png`

**特点**：
- 并排箱线图
- 均值标注
- 自动配色

**适用场景**：
- ✅ 统计显著性展示
- ✅ 性能分布对比
- ✅ 论文Section: Statistical Comparison

---

### 9️⃣ CDF对比（多算法）

**文件名**: `algorithms_cdf_comparison.png`

**特点**：
- 多条CDF曲线
- 易于对比分布差异

**适用场景**：
- ✅ 概率分布对比
- ✅ 论文Section: Distribution Analysis

---

### 🔟 小提琴图对比（多算法）

**文件名**: `algorithms_violin_comparison.png`

**特点**：
- 结合箱线图和核密度估计
- 展示完整分布形态

**适用场景**：
- ✅ 详细分布对比
- ✅ 发现多峰分布
- ✅ 论文Section: Distribution Comparison

---

### 1️⃣1️⃣ 柱状图对比（含误差棒）

**文件名**: `algorithms_bar_comparison.png`

**特点**：
- 误差棒（标准差）
- 数值标注
- 直观对比

**适用场景**：
- ✅ 简洁的性能对比
- ✅ 论文摘要/结论图表
- ✅ 论文Section: Performance Summary

---

## 🎨 图表特性

### IEEE标准配色

所有图表使用色盲友好配色方案：

| 颜色 | 十六进制 | 用途 |
|------|---------|------|
| 蓝色 | #0173B2 | 主要曲线/算法1 |
| 橙色 | #DE8F05 | 次要曲线/算法2 |
| 绿色 | #029E73 | 算法3 |
| 红色 | #D55E00 | 警告/算法4 |
| 紫色 | #CC78BC | 算法5 |
| 棕色 | #CA9161 | 算法6 |

### 技术规格

| 参数 | 值 | 说明 |
|------|-----|------|
| 分辨率 | 300 DPI | 适合论文投稿 |
| 图表大小 | 8×5 英寸 | 双栏适配 |
| 字体 | Serif | 学术标准 |
| 格式 | PNG | 无损压缩 |
| 文件大小 | 200-500 KB | 平衡质量和大小 |

---

## 📝 LaTeX集成

### 基本模板

```latex
\\documentclass{IEEEtran}
\\usepackage{graphicx}

\\begin{document}

\\section{Experimental Results}

\\subsection{Convergence Analysis}
Figure~\\ref{fig:convergence} shows the learning curve...

\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.48\\textwidth]{TD3_learning_curve_variance.png}
  \\caption{Training convergence with confidence intervals.}
  \\label{fig:convergence}
\\end{figure}

\\subsection{Algorithm Comparison}
As shown in Figure~\\ref{fig:comparison}...

\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{algorithms_convergence_comparison.png}
  \\caption{Performance comparison of different algorithms.}
  \\label{fig:comparison}
\\end{figure*}

\\end{document}
```

### 双栏布局

```latex
% 单图
\\begin{figure}[htbp]
  \\centering
  \\includegraphics[width=0.48\\textwidth]{chart.png}
  \\caption{Caption text.}
  \\label{fig:single}
\\end{figure}

% 并排双图
\\begin{figure}[htbp]
  \\centering
  \\subfloat[Convergence]{
    \\includegraphics[width=0.45\\textwidth]{chart1.png}
    \\label{fig:sub1}
  }
  \\hfill
  \\subfloat[Distribution]{
    \\includegraphics[width=0.45\\textwidth]{chart2.png}
    \\label{fig:sub2}
  }
  \\caption{Combined results.}
  \\label{fig:combined}
\\end{figure}

% 跨栏图
\\begin{figure*}[t]
  \\centering
  \\includegraphics[width=0.95\\textwidth]{large_chart.png}
  \\caption{Full-width chart.}
  \\label{fig:full}
\\end{figure*}
```

---

## 🔬 论文使用建议

### 推荐图表组合

#### 方案1：完整性能展示
1. 学习曲线（展示收敛）
2. 箱线图（展示分布）
3. 相关性热力图（展示关系）
4. 雷达图（展示综合）

#### 方案2：算法对比
1. 收敛性对比（主图）
2. 箱线图对比（统计）
3. 柱状图对比（摘要）

#### 方案3：最小集合（空间有限）
1. 收敛性对比（必需）
2. 柱状图对比（必需）

### 章节对应

| 论文章节 | 推荐图表 |
|---------|---------|
| Introduction | 柱状图对比（简洁） |
| Methodology | 无（文字说明） |
| Convergence Analysis | 学习曲线、CDF |
| Performance Evaluation | 收敛对比、箱线图对比 |
| Trade-off Analysis | 散点图、相关性热力图 |
| Conclusion | 雷达图、柱状图对比 |

---

## 💡 高级用法

### 批量生成

```bash
# 为所有算法生成图表
for algo in td3 ddpg sac ppo; do
  python generate_academic_charts.py \\
    results/single_agent/$algo/training_results_*.json \\
    -o figures/$algo
done
```

### 自定义参数

修改`utils/academic_chart_generator.py`：

```python
# 自定义DPI
generator = AcademicChartGenerator(dpi=600)

# 自定义图表大小
plt.subplots(figsize=(10, 6))

# 自定义配色
ACADEMIC_COLORS = {
    'blue': '#YOUR_COLOR',
    # ...
}
```

---

## 📊 数据要求

### 输入格式

训练结果JSON需包含：

```json
{
  "algorithm": "TD3",
  "episode_rewards": [100.0, 150.0, ...],
  "episode_metrics": {
    "avg_delay": [1.2, 1.1, ...],
    "total_energy": [500, 480, ...],
    "task_completion_rate": [0.95, 0.96, ...],
    "cache_hit_rate": [0.75, 0.78, ...]
  }
}
```

### 最低数据要求

| 图表类型 | 最少Episodes | 推荐Episodes |
|---------|-------------|-------------|
| 学习曲线 | 20 | 100+ |
| 箱线图 | 30 | 100+ |
| CDF | 50 | 200+ |
| 相关性热力图 | 50 | 200+ |
| 散点图 | 50 | 200+ |
| 雷达图 | 20 | 100+ |

---

## 🐛 故障排查

### 问题1：图表模糊
**原因**：DPI太低  
**解决**：使用`--dpi 300`或更高

### 问题2：中文显示为方框
**原因**：字体不支持中文  
**解决**：本工具使用英文标签，无此问题

### 问题3：内存不足
**原因**：数据量过大  
**解决**：减少Episodes或分批生成

### 问题4：颜色不明显
**原因**：显示器色彩配置  
**解决**：图表已优化，实际论文打印效果佳

---

## 📞 支持

### 相关文档
- 📄 **本指南**: `docs/academic_charts_guide.md`
- 📄 **代码文档**: `utils/academic_chart_generator.py`
- 📄 **生成脚本**: `generate_academic_charts.py`

### 示例输出
- 📂 **示例图表**: `academic_figures/td3/`

---

## 🎓 引用建议

如果这些图表对您的研究有帮助，欢迎在论文致谢中提及本工具。

---

**更新**: 2025-10-08  
**维护**: VEC边缘计算团队  
**版本**: v1.0
