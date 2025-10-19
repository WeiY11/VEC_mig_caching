# 元启发式算法实现

## 概述

本模块实现了两种经典的元启发式优化算法，用于车联网边缘计算场景中的任务迁移和缓存决策优化。

## 算法介绍

### 1. GA (Genetic Algorithm) - 遗传算法

**原理：**
- 模拟生物进化过程，通过选择、交叉、变异不断优化
- 维护一个种群，每个个体代表一个动作向量（16维）
- 通过评估适应度函数来指导进化方向

**特点：**
- ✅ 全局搜索能力强，可以跳出局部最优
- ✅ 适合处理复杂的非线性优化问题
- ✅ 支持离散和连续变量混合优化
- ⚠️ 计算开销较大，收敛速度相对较慢

**关键参数：**
- `population_size`: 种群大小（默认20）
- `elite_ratio`: 精英比例（默认0.2）
- `mutation_rate`: 变异率（默认0.1）
- `crossover_rate`: 交叉率（默认0.8）

### 2. PSO (Particle Swarm Optimization) - 粒子群优化

**原理：**
- 模拟鸟群觅食行为的群体智能算法
- 每个粒子具有位置和速度，通过个体经验和群体经验更新
- 粒子向个体最优和全局最优位置移动

**特点：**
- ✅ 收敛速度快，参数较少
- ✅ 算法简单，易于实现
- ✅ 适合连续优化问题
- ⚠️ 容易陷入局部最优（通过惯性权重衰减缓解）

**关键参数：**
- `swarm_size`: 粒子群大小（默认20）
- `w`: 惯性权重（默认0.7）
- `c1`: 个体学习因子（默认1.5）
- `c2`: 社会学习因子（默认1.5）
- `w_decay`: 惯性权重衰减率（默认0.99）

## 使用方法

### 快速开始

```bash
# 运行GA算法（100轮）
python baseline_comparison/individual_runners/metaheuristic/run_ga.py --episodes 100 --seed 42

# 运行PSO算法（100轮）
python baseline_comparison/individual_runners/metaheuristic/run_pso.py --episodes 100 --seed 42
```

### 批量运行

```bash
# Windows
run_all_metaheuristic.bat 100 42

# Linux/Mac
./run_all_metaheuristic.sh 100 42
```

### 参数调优

#### GA参数调优示例
```bash
# 增大种群规模，提高探索能力
python baseline_comparison/individual_runners/metaheuristic/run_ga.py \
    --episodes 100 \
    --population_size 50 \
    --elite_ratio 0.1 \
    --mutation_rate 0.2
```

#### PSO参数调优示例
```bash
# 使用更大的惯性权重，增强全局搜索
python baseline_comparison/individual_runners/metaheuristic/run_pso.py \
    --episodes 100 \
    --swarm_size 40 \
    --w 0.9 \
    --w_decay 0.95
```

## 算法比较

| 特性 | GA | PSO |
|------|-----|-----|
| 收敛速度 | 中等 | 快 |
| 全局搜索能力 | 强 | 中等 |
| 参数敏感性 | 中等 | 低 |
| 计算复杂度 | O(n²) | O(n) |
| 适用场景 | 复杂非线性问题 | 连续优化问题 |

## 实验结果示例

根据快速测试（10轮）结果：

### GA性能
- 平均时延: 0.250s
- 平均能耗: 4280.1J
- 任务完成率: 97.82%
- 缓存命中率: 3.7%

### PSO性能
- 平均时延: 0.282s
- 平均能耗: 4329.6J
- 任务完成率: 98.34%
- 缓存命中率: 9.5%

## 与其他算法对比

元启发式算法在启发式和深度强化学习之间提供了一个平衡点：
- 比启发式算法更智能，能找到更优解
- 比深度强化学习更简单，不需要训练神经网络
- 适合中等规模的优化问题

## 扩展和改进

### 可能的改进方向
1. **混合算法**：结合GA和PSO的优点
2. **自适应参数**：根据收敛情况动态调整参数
3. **并行化**：利用多核CPU加速评估
4. **问题特定优化**：针对VEC场景定制操作算子

### 添加新的元启发式算法
1. 继承`BaselineAlgorithm`基类
2. 实现`select_action`方法
3. 添加算法特定的优化逻辑
4. 创建运行脚本

## 注意事项

1. **评估函数**：当前使用简化的启发式评估，实际应用中可以考虑更精确的仿真
2. **实时性**：元启发式算法需要多次迭代，可能不适合严格实时场景
3. **参数调优**：算法性能对参数敏感，建议根据具体场景调优

## 相关文献

1. GA: Holland, J. H. (1992). "Adaptation in natural and artificial systems"
2. PSO: Kennedy, J., & Eberhart, R. (1995). "Particle swarm optimization"

---
更新日期：2024-10-13










