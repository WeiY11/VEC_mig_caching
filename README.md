# VEC边缘计算迁移与缓存系统

## 🎯 项目简介

车联网边缘计算(VEC)系统的任务迁移与缓存优化研究项目，使用深度强化学习(DRL)算法（TD3、DDPG、SAC、PPO等）进行智能决策。

**核心优化目标**:
```
minimize: ω_T × 时延 + ω_E × 能耗
```

---

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行训练

**单智能体训练**（推荐）:
```bash
python train_single_agent.py --algorithm TD3 --episodes 200
```

**多智能体训练**:
```bash
python train_multi_agent.py --algorithm MADDPG --episodes 200
```

**分层智能体训练**:
```bash
python train_hierarchical_agent.py --episodes 200
```

### 运行学术实验

```bash
# Baseline对比实验
python scripts/run/run_full_experiment.py --mode baseline

# 完整实验套件
python experiments/run_complete_experiments.py
```

---

## 📁 项目结构

```
VEC_mig_caching/
├── 🎯 核心训练脚本
│   ├── train_single_agent.py       # 单智能体训练
│   ├── train_multi_agent.py        # 多智能体训练
│   ├── train_hierarchical_agent.py # 分层智能体训练
│   └── main.py                     # 主入口
│
├── 🧠 核心算法模块
│   ├── single_agent/               # 单智能体算法（TD3/DDPG/SAC/PPO/DQN）
│   ├── algorithms/                 # 多智能体算法（MADDPG/MAPPO/MATD3）
│   └── hierarchical_learning/      # 分层强化学习
│
├── 🏗️ 系统核心模块
│   ├── models/                     # 数据模型（车辆、RSU、UAV、任务）
│   ├── config/                     # 配置系统
│   ├── evaluation/                 # 系统仿真器
│   ├── decision/                   # 决策模块（卸载、规划）
│   ├── caching/                    # 协作缓存
│   ├── migration/                  # 任务迁移
│   ├── communication/              # 3GPP通信模型
│   ├── core/                       # 队列管理
│   └── utils/                      # 工具函数
│
├── 🔬 实验脚本（新整理）
│   ├── experiments/                # 学术实验套件
│   │   ├── baseline_algorithms.py
│   │   ├── ablation_study.py
│   │   └── camtd3_strategy_suite/
│   │
│   └── scripts/                    # ⭐ 运行脚本（新整理）
│       ├── run/                    # 实验运行
│       ├── compare/                # 算法对比
│       ├── analyze/                # 结果分析
│       └── visualize/              # 数据可视化
│
├── 📊 结果与输出
│   ├── results/                    # 训练结果
│   ├── test_results/               # 测试结果
│   └── figures/                    # ⭐ 图表统一存放
│       ├── academic/               # 学术图表
│       └── reports/                # 报告图表
│
├── 📚 文档
│   ├── docs/
│   │   ├── paper_ending.tex        # 理论模型（论文标准）
│   │   ├── VEC系统模型代码质量综合分析.pdf
│   │   └── analysis/               # ⭐ 分析报告
│   │       └── md/                 # 系统分析文档
│   │
│   └── 文件整理方案.md              # 本次整理说明
│
├── 🚀 部署相关（新整理）
│   └── deployment/                 # ⭐ 部署统一管理
│       ├── server/                 # 服务器部署
│       ├── kaggle/                 # Kaggle部署
│       ├── quick_deploy.py
│       ├── deploy_to_server.sh
│       └── deploy_manual.md
│
├── 📦 归档
│   └── archives/                   # ⭐ 压缩包归档
│       ├── models.tar.gz
│       └── vec_project.tar.gz
│
└── 🧪 测试与可视化
    ├── tests/                      # 单元测试
    ├── visualization/              # 可视化工具
    └── tools/                      # 其他工具脚本
```

---

## 🎓 核心算法

### 单智能体算法
- **TD3** (Twin Delayed DDPG) - 推荐，最稳定
- **DDPG** (Deep Deterministic Policy Gradient)
- **SAC** (Soft Actor-Critic)
- **PPO** (Proximal Policy Optimization)
- **DQN** (Deep Q-Network)

### 多智能体算法
- **MADDPG** (Multi-Agent DDPG)
- **MAPPO** (Multi-Agent PPO)
- **MATD3** (Multi-Agent TD3)
- **QMIX** (Q-Mixing)

---

## 📊 实验功能

### 运行脚本 (`scripts/run/`)

```bash
# 完整实验
python scripts/run/run_full_experiment.py

# 算法对比
python scripts/run/run_algorithm_comparison.py

# TD3专项实验
python scripts/run/run_td3_comparison.py
python scripts/run/run_td3_focused.py
python scripts/run/run_td3_realistic.py
```

### 对比分析 (`scripts/compare/`)

```bash
# 多算法对比
python scripts/compare/compare_config.py

# SAC vs TD3对比
python scripts/compare/compare_sac_td3.py
python scripts/compare/compare_sac_td3_simple.py
```

### 结果分析 (`scripts/analyze/`)

```bash
# 分析最新结果
python scripts/analyze/analyze_latest.py

# 多种子结果分析
python scripts/analyze/analyze_multi_seed_results.py
```

### 可视化 (`scripts/visualize/`)

```bash
# 生成学术图表
python scripts/visualize/generate_academic_charts.py

# 生成HTML报告
python scripts/visualize/generate_html_report.py

# 可视化结果
python scripts/visualize/visualize_results.py

# 实时可视化
python scripts/visualize/realtime_visualization.py
```

---

## ⚙️ 配置说明

### 核心配置文件

- `config/system_config.py` - 系统核心配置
  - 网络拓扑：12车辆、4 RSU、2 UAV
  - 任务生成：泊松到达、8种场景
  - 奖励权重：时延2.0、能耗1.8

- `config/algorithm_config.py` - 算法超参数
- `config/network_config.py` - 网络参数

### 实验配置

- `config/algorithm_comparison_config.json` - 算法对比配置
- `config/td3_experiment_config.json` - TD3实验配置
- `config/paper_extreme_*.json` - 极端场景配置

---

## 🔧 工具脚本 (`tools/`)

- `fixed_topology_optimizer.py` - 拓扑优化
- `td3_unified_metrics.py` - 统一度量计算
- 其他10个工具脚本

---

## 📈 结果输出

### 训练结果 (`results/`)

```
results/
├── single_agent/           # 单智能体结果
├── multi_agent/            # 多智能体结果
└── hierarchical/           # 分层智能体结果
```

### 图表输出 (`figures/`)

```
figures/
├── academic/               # 学术论文图表
│   ├── paper_comparison/
│   ├── paper_style/
│   └── td3_*/
└── reports/                # 其他报告图表
```

---

## 🚀 部署指南

### 服务器部署

```bash
cd deployment
bash deploy_to_server.sh
```

详见 `deployment/deploy_manual.md`

### Kaggle部署

```bash
cd deployment/kaggle
bash kaggle_setup.sh
```

---

## 📚 文档资源

### 学术文档
- `docs/paper_ending.tex` - 系统理论模型（论文标准）
- `docs/VEC系统模型代码质量综合分析.pdf` - 系统分析报告

### 分析报告 (`docs/analysis/md/`)
- `00_START_HERE_分析报告导航.md` - 报告索引
- `VEC_System_Analysis_*.md` - 系统分析系列
- `VEC_Critical_Issues_and_Solutions.md` - 问题与解决方案

---

## 🎯 最近更新

### v2.0 优化（2025-10-30）

**任务生成与分类优化**:
- ✅ 场景化任务生成（8种应用场景）
- ✅ 重尾数据大小分布（Pareto分布）
- ✅ Zipf内容热度分布（协作缓存）
- ✅ 多维特征任务分类

**文件结构整理**:
- ✅ 根目录简化（30+ → 核心4个）
- ✅ 脚本按功能分类（scripts/）
- ✅ 部署统一管理（deployment/）
- ✅ 图表集中存放（figures/）

详见：`文件整理方案.md`

---

## 📖 使用建议

### 新手入门
1. 阅读 `docs/analysis/md/00_START_HERE_分析报告导航.md`
2. 运行快速测试：`python train_single_agent.py --algorithm TD3 --episodes 50`
3. 查看结果：`python scripts/visualize/visualize_results.py`

### 论文实验
1. 完整Baseline对比：`python experiments/run_complete_experiments.py --mode baseline`
2. 消融实验：`python experiments/ablation_study.py`
3. 生成论文图表：`python scripts/visualize/generate_academic_charts.py`

### 自定义实验
1. 修改配置：`config/system_config.py`
2. 运行训练：`train_single_agent.py`
3. 分析结果：`scripts/analyze/analyze_latest.py`

---

## 🤝 贡献指南

### 代码规范
- Python 3.8+
- 遵循PEP 8
- 详细注释（中英文）

### 提交规范
- 功能开发：`feature/功能名称`
- Bug修复：`fix/问题描述`
- 文档更新：`docs/文档类型`

---

## 📄 许可证

本项目用于学术研究，适用于投稿至：
- IEEE INFOCOM
- ACM MobiCom
- IEEE TMC/TVT

---

## 📧 联系方式

如有问题，请查看文档或提交Issue。

---

**最后更新**: 2025-10-30  
**版本**: v2.0  
**状态**: ✅ 生产就绪

