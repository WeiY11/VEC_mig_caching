# SAC 奖励统一化 - 变更日志

## 变更时间

2025-11-29 11:59

## 变更类型

**破坏性变更** - 移除 SAC 算法的特殊奖励处理

## 变更内容

### 修改的文件

- `utils/unified_reward_calculator.py`

### 具体变更

#### 1. 移除 SAC 特殊归一化参数（第 145-148 行）

**之前**:

```python
if self.algorithm == "SAC":
    self.delay_normalizer = self.latency_target
    self.energy_normalizer = self.energy_target
```

**之后**:

```python
# 已移除SAC的特殊归一化参数，所有算法现在使用统一的归一化逻辑
```

#### 2. 统一奖励裁剪范围（第 169-174 行）

**之前**:

```python
if self.algorithm == "SAC":
    self.reward_clip_range = (-8.0, 6.0)
else:
    self.reward_clip_range = (-10.0, 10.0)
```

**之后**:

```python
# 所有算法统一使用负奖励范围
self.reward_clip_range = (-10.0, 0.0)
```

#### 3. 统一奖励计算公式（第 418-426 行）

**之前**:

```python
if self.algorithm == "SAC":
    base_reward = 5.0
    completion_bonus = (completion_rate - 0.95) * 10.0 if completion_rate > 0.95 else 0.0
    reward_raw = base_reward + completion_bonus - components.total_cost
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], self.reward_clip_range[1]))
else:
    reward_raw = -abs(components.total_cost)
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], 0.0))
```

**之后**:

```python
# 所有算法统一使用成本最小化奖励
reward_raw = -abs(components.total_cost)
reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], self.reward_clip_range[1]))
```

## 影响范围

### 受影响的组件

1. **SAC 训练**:

   - 所有未来的 SAC 训练将使用新的奖励范围 `[-10, 0]`
   - 奖励不再有正值，完全基于成本最小化

2. **Benchmarks 对比实验**:

   - SAC、TD3、DDPG、DDPG_Vanilla 的奖励值现在可以直接比较
   - 所有算法的奖励曲线将在相同的数值范围内

3. **已保存的 SAC 模型**:
   - ⚠️ **不兼容**: 旧模型的价值函数是基于正奖励空间训练的
   - **建议**: 不要加载旧模型继续训练，需要从头重新训练

### 不受影响的组件

- TD3/DDPG/其他算法：没有任何变化
- 环境仿真器：没有变化
- 成本指标计算：没有变化

## 迁移指南

### 对于新实验

直接使用修改后的代码即可，无需任何额外操作。

### 对于已有 SAC 训练结果

1. **重新训练**: 使用新的奖励函数从头开始训练 SAC
2. **对比分析**: 旧结果（正奖励）和新结果（负奖励）不可直接比较
3. **文档更新**: 在论文/报告中说明 SAC 的奖励计算已更新

### 对于代码维护者

- 已移除 `_sac_reward_calculator` 的特殊逻辑，但单例对象仍然保留
- 可以考虑在未来版本中完全移除 `_sac_reward_calculator`，统一使用 `_general_reward_calculator`

## 验证检查点

运行以下命令验证变更：

```bash
# 1. 检查奖励范围
python -c "from utils.unified_reward_calculator import UnifiedRewardCalculator; calc = UnifiedRewardCalculator('SAC'); print(f'SAC reward range: {calc.reward_clip_range}')"

# 2. 运行小规模实验
python Benchmarks/run_benchmarks_vs_optimized_td3.py --alg sac td3 --episodes 50 --groups 2

# 3. 检查奖励是否在 [-10, 0] 范围内
```

预期输出：

- SAC reward range: `(-10.0, 0.0)`
- 实验结果中所有奖励值都应为负数

## 理论依据

**为什么 SAC 不需要正奖励？**

SAC 的优化目标是：

```
maximize E[Σ(r_t + α·H(π(·|s_t)))]
```

其中：

- `r_t` 是奖励（可正可负）
- `α` 是温度参数
- `H(π)` 是策略熵（始终为正）

奖励的绝对值不影响算法的工作原理，只要奖励信号能够区分好坏状态即可。在成本最小化框架下：

- 低成本 → 高奖励（接近 0）
- 高成本 → 低奖励（接近-10）

这与使用正奖励（如 `5.0 - cost`）在优化方向上完全等价。

## 相关资源

- 实施计划: `implementation_plan.md`
- 审计报告: `.gemini/Benchmarks_Comparison_Audit.md`
- 任务清单: `task.md`

---

**变更批准**: 用户已确认同意移除 SAC 特殊处理  
**变更执行**: AI Assistant  
**审核状态**: ✅ 已完成代码修改，等待验证测试
