# 🚀 开始消融实验

## ⚡ 快速开始（3步）

### 步骤1️⃣: 测试环境

```bash
python test_env.py
```

如果看到 "所有测试通过"，继续下一步。

### 步骤2️⃣: 快速实验（10分钟）

```bash
python run_ablation_td3.py --episodes 10 --quick
```

这将运行一个超快速的实验（每个配置10轮），验证流程是否正常。

### 步骤3️⃣: 标准实验（2-3小时）

```bash
python run_ablation_td3.py --episodes 200
```

这将运行完整的消融实验（每个配置200轮），生成论文数据。

## 📁 结果位置

- **实验数据**: `results/` 文件夹
- **图表分析**: `analysis/` 文件夹

## 🔍 查看结果

实验完成后，查看：
- `analysis/ablation_comparison.png` - 对比图表
- `analysis/training_curves.png` - 训练曲线
- `analysis/ablation_table.tex` - LaTeX表格
- `analysis/comparison_report.md` - 分析报告

## 📚 更多信息

详见 `使用指南.md`

---

**注意**: 这个文件夹完全独立，不会影响原始项目！

