# MATD3-MIG 车联网边缘缓存系统

## 🎯 项目简介

基于多智能体深度强化学习的车联网边缘缓存系统，采用MATD3算法优化缓存决策和任务卸载策略。

## 🏗️ 项目结构

```
VEC_mig_caching/
├── algorithms/              # 强化学习算法
│   ├── matd3.py            # MATD3算法实现
│   ├── maddpg.py           # MADDPG算法实现
│   └── base_algorithm.py   # 算法基类
├── models/                 # 系统模型
│   ├── vehicle_node.py     # 车辆节点
│   ├── rsu_node.py         # RSU节点
│   └── uav_node.py         # UAV节点
├── environment/            # 仿真环境
│   ├── vec_env.py          # 车联网环境
│   └── cache_manager.py    # 缓存管理
├── decision/               # 决策模块
│   ├── offloading_manager.py # 卸载决策
│   └── cache_policy.py     # 缓存策略
├── training/               # 训练模块
│   └── train_multi_agent.py # 多智能体训练
├── single_agent/           # 单智能体算法
│   ├── ddpg.py            # DDPG算法
│   └── ppo.py             # PPO算法
├── experiments/            # 实验评估
│   └── evaluation.py       # 性能评估
├── results/                # 实验结果
└── docs/                   # 项目文档
```

## 🚀 快速开始

### 环境配置
```bash
conda create -n MATD3 python=3.8
conda activate MATD3
pip install torch numpy matplotlib pandas seaborn
```

### 运行实验
```bash
# 完整实验
python run_full_experiment.py --episodes 10 --runs 3

# 多智能体训练
python train_multi_agent.py

# 单智能体训练
python train_single_agent.py

# 结果可视化
python visualize_results.py

# 高级分析
python advanced_analysis.py

# 系统演示
python demo.py
```

## 📊 性能指标

| 场景 | 算法 | 时延(s) | 完成率(%) | 能耗(MJ) | 缓存命中率(%) |
|------|------|---------|-----------|----------|---------------|
| 标准 | MATD3-MIG | **1.001** | **85.0** | **25.5** | **100.0** |
| 高负载 | MATD3-MIG | **0.999** | **84.9** | **9.2** | **100.0** |
| 大规模 | MATD3-MIG | **0.999** | **85.0** | **6.8** | **100.0** |

## 🎯 核心特性

- ✅ 多智能体协作决策
- ✅ 智能缓存管理
- ✅ 动态任务卸载
- ✅ 能耗优化
- ✅ 实时性能监控

## 📈 算法优势

- 🚀 时延改进: 最高41.2%
- ⚡ 能耗改进: 最高7.3%
- ✅ 完成率改进: 最高30.7%
- 💾 缓存命中率: 100%

## 📚 文档

- [项目完成报告](PROJECT_COMPLETION_REPORT.md)
- [能耗计算修复指南](docs/ENERGY_CALCULATION_FIX_GUIDE.md)
- [高级分析报告](results/advanced_analysis_report.md)

## 🤝 贡献

欢迎提交Issue和Pull Request来改进项目。

## 📄 许可证

MIT License
