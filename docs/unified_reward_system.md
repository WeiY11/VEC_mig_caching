# 统一奖励函数系统设计文档

## 📋 概述

本文档描述了 VEC 系统中**统一奖励计算器**的设计与实现，解决了之前多个奖励计算器并存导致的不一致问题。

## 🎯 核心设计原则

### 1. **简化优化目标**
- **主目标**：最小化时延和能耗的加权和
- **数学表达**：`minimize α·Delay + β·Energy`
- **默认权重**：`α=2.0, β=1.2`

### 2. **移除数据丢失量指标**
**原因分析：**
- 数据丢失本质上是**时延的衍生指标**（任务超时→被丢弃→数据丢失）
- 之前权重配置：delay(2.0) >> energy(1.2) >> loss(0.1)
- 过多指标可能导致优化目标冲突

**替代方案：**
- 通过 `dropped_tasks` 惩罚直接保证完成率
- 优化时延自然会减少数据丢失

### 3. **算法一致性与适配性**
- **通用版本**（DDPG, TD3, PPO, DQN）：纯成本最小化
- **SAC专用版本**：考虑最大熵框架，保留正向激励机制

## 📊 奖励函数设计

### 核心公式

```python
# 1. 归一化
norm_delay = avg_delay / delay_normalizer
norm_energy = total_energy / energy_normalizer

# 2. 基础成本（双目标加权和）
base_cost = weight_delay * norm_delay + weight_energy * norm_energy

# 3. 丢弃任务惩罚（保证完成率）
dropped_penalty = penalty_weight * dropped_tasks

# 4. 自适应阈值惩罚（防止极端情况）
threshold_penalty = delay_threshold_penalty + energy_threshold_penalty

# 5. 总成本
total_cost = base_cost + dropped_penalty + threshold_penalty

# 6. 最终奖励
reward = -total_cost  # 通用版本
reward = bonus - total_cost  # SAC版本（bonus可能为正）
```

### 参数配置

| 参数 | 通用算法 | SAC算法 | 说明 |
|------|---------|---------|------|
| **时延归一化** | 1.0 | 0.3 | SAC更敏感 |
| **能耗归一化** | 1000.0 | 1500.0 | SAC更敏感 |
| **奖励范围** | [-20.0, -0.01] | [-15.0, 3.0] | SAC允许正值 |
| **时延权重** | 2.0 | 2.0 | 一致 |
| **能耗权重** | 1.2 | 1.2 | 一致 |
| **丢弃惩罚** | 0.02 | 0.02 | 一致 |

## 🔄 算法迁移对照表

### 之前的奖励计算器

| 算法 | 旧奖励计算器 | 特点 |
|------|-------------|------|
| TD3 | `enhanced_reward_calculator` | 复杂的子系统奖励 |
| DDPG | `enhanced_reward_calculator` | 复杂的子系统奖励 |
| SAC | `sac_reward_calculator` | SAC专用版本 |
| PPO | `simple_reward_calculator` | 简化版本 |
| DQN | `simple_reward_calculator` | 简化版本 |

### 现在的统一系统

| 算法 | 新奖励计算器 | 调用方式 |
|------|-------------|---------|
| **所有算法** | `unified_reward_calculator` | 统一接口 |
| TD3/DDPG/PPO/DQN | `algorithm="general"` | 通用版本 |
| SAC | `algorithm="sac"` | SAC专用版本 |

## 💻 使用示例

### 基本使用

```python
from utils.unified_reward_calculator import calculate_unified_reward

# 系统性能指标
system_metrics = {
    'avg_task_delay': 0.2,              # 平均时延（秒）
    'total_energy_consumption': 1000.0,  # 总能耗（焦耳）
    'dropped_tasks': 0,                  # 丢弃任务数
    'task_completion_rate': 0.98         # 任务完成率
}

# 通用算法（TD3, DDPG, PPO, DQN）
reward_general = calculate_unified_reward(system_metrics, algorithm="general")

# SAC算法
reward_sac = calculate_unified_reward(system_metrics, algorithm="sac")
```

### 向后兼容接口

```python
# 这些旧接口仍然可用（内部调用统一奖励计算器）
from utils.unified_reward_calculator import (
    calculate_enhanced_reward,  # 替代旧的enhanced_reward_calculator
    calculate_sac_reward,        # 替代旧的sac_reward_calculator
    calculate_simple_reward      # 替代旧的simple_reward_calculator
)

reward = calculate_enhanced_reward(system_metrics)  # 等价于 algorithm="general"
```

### 获取奖励分解报告

```python
from utils.unified_reward_calculator import get_reward_breakdown

print(get_reward_breakdown(system_metrics, algorithm="general"))
```

**输出示例：**
```
奖励分解报告 (GENERAL):
├── 总奖励: -1.600
├── 核心指标:
│   ├── 时延: 0.200s (归一化: 0.200)
│   ├── 能耗: 1000.0J (归一化: 1.000)
│   └── 完成率: 98.0%
├── 成本贡献:
│   ├── 时延成本: 0.400
│   ├── 能耗成本: 1.200
│   └── 丢弃惩罚: 0.000 (0个任务)
└── 优化方向: 最小化成本
```

## 📈 测试结果

### 场景1: 正常性能
- **指标**：时延0.2s, 能耗1000J
- **通用算法奖励**：-1.600
- **SAC算法奖励**：-1.683

### 场景2: 优秀性能
- **指标**：时延0.15s, 能耗800J
- **通用算法奖励**：-1.260
- **SAC算法奖励**：-0.890 ✨（包含bonus）

### 场景3: 较差性能
- **指标**：时延0.35s, 能耗3500J, 5个丢弃任务
- **通用算法奖励**：-5.583
- **SAC算法奖励**：-7.533

### 奖励趋势验证
```
优秀 > 正常 > 较差: 0.15s < 0.20s < 0.35s ✓
```

## 🔧 算法代码修改

所有算法的 `calculate_reward` 方法已统一为：

```python
def calculate_reward(self, system_metrics: Dict, 
                   cache_metrics: Optional[Dict] = None,
                   migration_metrics: Optional[Dict] = None) -> float:
    """使用统一奖励计算器"""
    from utils.unified_reward_calculator import calculate_unified_reward
    
    # 通用算法用 "general"，SAC用 "sac"
    return calculate_unified_reward(
        system_metrics, 
        cache_metrics, 
        migration_metrics, 
        algorithm="general"  # 或 "sac"
    )
```

## 📁 文件清理

### 已备份的旧文件
- `utils/enhanced_reward_calculator.py.backup`
- `utils/sac_reward_calculator.py.backup`
- `utils/simple_reward_calculator.py.backup`

### 新增文件
- `utils/unified_reward_calculator.py` - **统一奖励计算器**
- `test_unified_reward.py` - 测试脚本

## ✅ 优势总结

### 1. **简化与聚焦**
- 从3个指标（时延+能耗+数据丢失）简化为2个核心目标
- 移除冗余的子系统奖励（缓存、迁移），聚焦主要优化目标

### 2. **一致性保证**
- 所有算法共享核心奖励逻辑
- 减少维护成本和潜在的不一致问题

### 3. **算法适配性**
- SAC保留专门调整以适应最大熵框架
- 其他算法使用统一的成本最小化逻辑

### 4. **向后兼容**
- 保留所有旧接口，现有代码无需修改
- 平滑过渡，无破坏性变更

## 🚀 快速开始

### 运行测试
```bash
python test_unified_reward.py
```

### 训练算法（使用新奖励函数）
```bash
# TD3算法
python train_single_agent.py --algorithm TD3 --episodes 200

# SAC算法
python train_single_agent.py --algorithm SAC --episodes 200

# 所有算法对比
python train_single_agent.py --compare --episodes 200
```

## 📚 相关文档
- 配置文件：`config/system_config.py` (奖励权重定义)
- 算法实现：`single_agent/*.py` (各算法使用统一奖励)
- 训练脚本：`train_single_agent.py` (主训练逻辑)

---

**最后更新**：2025-10-02  
**维护者**：VEC系统开发团队

