# TD3策略对比实验 - 环境变量配置文档

## 概述

本文档详细说明TD3策略对比实验套件中使用的所有环境变量，确保实验可重复性和配置透明性。

---

## 核心环境变量

### 1. CENTRAL_RESOURCE

**用途**: 控制是否启用中央资源分配架构

**可选值**:
- `"1"` 或 `"true"`: 启用分层模式（Phase 1决策 + Phase 2执行）
- `"0"` 或 `"false"` 或 未设置: 标准模式（固定资源分配）

**影响范围**:
- 资源分配策略
- 智能体决策流程
- 实验对比模式

**使用示例**:
```bash
# 启用中央资源分配
export CENTRAL_RESOURCE=1
python experiments/td3_strategy_suite/run_four_key_experiments.py --episodes 1500

# 标准模式
export CENTRAL_RESOURCE=0
python experiments/td3_strategy_suite/run_four_key_experiments.py --episodes 1500
```

**相关文件**:
- `comparison_suite.py::ModeSpec.env_overrides()`
- `strategy_runner.py::_run_strategy_suite_internal()`

---

### 2. RESOURCE_ALLOCATION_MODE

**用途**: 指定资源初始化模式

**可选值**:
- `"learned"`: 使用学习到的资源分配策略（默认）
- `"heuristic"`: 使用启发式资源分配策略
- `"random"`: 随机资源分配
- `"fixed"`: 固定资源分配

**影响范围**:
- 资源分配初始化
- 智能体训练起点
- 策略收敛速度

**使用示例**:
```bash
# 使用学习策略
export RESOURCE_ALLOCATION_MODE=learned
python train_single_agent.py --algorithm TD3

# 使用启发式策略
export RESOURCE_ALLOCATION_MODE=heuristic
python train_single_agent.py --algorithm TD3
```

**相关文件**:
- `comparison_suite.py::ModeSpec`
- `strategy_runner.py`

---

### 3. RANDOM_SEED

**用途**: 设置全局随机种子，确保实验可重复性

**可选值**: 任意整数（推荐使用42）

**影响范围**:
- NumPy随机数生成器
- PyTorch随机数生成器
- Python内置random模块
- 环境初始化

**使用示例**:
```bash
# 设置随机种子
export RANDOM_SEED=42
python experiments/td3_strategy_suite/run_strategy_training.py --strategy comprehensive-migration
```

**相关文件**:
- `strategy_runner.py::_run_strategy_suite_internal()`
- `train_single_agent.py::_apply_global_seed_from_env()`

---

## 策略特定环境变量

### 4. DISABLE_MIGRATION

**用途**: 禁用任务迁移功能

**可选值**:
- `"1"` 或 `"true"`: 禁用迁移
- `"0"` 或 未设置: 启用迁移

**影响范围**:
- 任务迁移决策
- 策略对比实验（comprehensive-no-migration vs comprehensive-migration）

**使用示例**:
```bash
# 禁用迁移
export DISABLE_MIGRATION=1
python train_single_agent.py --algorithm TD3
```

---

### 5. ENFORCE_OFFLOAD_MODE

**用途**: 强制指定卸载模式（用于消融实验）

**可选值**:
- `"local"`: 强制本地执行
- `"rsu"`: 强制卸载到RSU
- `"uav"`: 强制卸载到UAV
- 未设置: 由智能体学习决策

**影响范围**:
- 卸载决策
- 策略对比实验（local-only, remote-only等）

**使用示例**:
```bash
# 强制本地执行
export ENFORCE_OFFLOAD_MODE=local
python train_single_agent.py --algorithm TD3

# 强制卸载到RSU
export ENFORCE_OFFLOAD_MODE=rsu
python train_single_agent.py --algorithm TD3
```

---

## 实验控制变量

### 6. SILENT_MODE

**用途**: 控制训练过程的输出详细程度

**可选值**:
- `"1"` 或 `"true"`: 静默模式（减少输出）
- `"0"` 或 未设置: 详细输出

**影响范围**:
- 训练日志输出
- 进度显示
- 调试信息

**使用示例**:
```bash
# 静默模式
export SILENT_MODE=1
python experiments/td3_strategy_suite/run_batch_experiments.py --mode full --all
```

---

### 7. ENABLE_ENHANCED_CACHE

**用途**: 启用增强型缓存管理

**可选值**:
- `"1"` 或 `"true"`: 启用增强缓存
- `"0"` 或 未设置: 使用标准缓存

**影响范围**:
- 缓存策略
- 缓存命中率
- 系统性能

---

## 调试与验证变量

### 8. DEBUG_MODE

**用途**: 启用调试模式（详细日志、断言检查）

**可选值**:
- `"1"` 或 `"true"`: 启用调试
- `"0"` 或 未设置: 正常模式

**使用示例**:
```bash
export DEBUG_MODE=1
python experiments/td3_strategy_suite/run_strategy_training.py --strategy comprehensive-migration
```

---

### 9. DRY_RUN

**用途**: 仅验证配置，不实际执行训练

**可选值**:
- `"1"` 或 `"true"`: 干运行模式
- `"0"` 或 未设置: 正常执行

**使用示例**:
```bash
# 仅验证配置
export DRY_RUN=1
python experiments/td3_strategy_suite/run_four_key_experiments.py --episodes 1500
```

---

## 环境变量使用最佳实践

### 1. 环境隔离

使用 `_temporary_environ` 上下文管理器确保环境变量不污染全局状态：

```python
from comparison_suite import _temporary_environ

with _temporary_environ({"CENTRAL_RESOURCE": "1", "RANDOM_SEED": "42"}):
    # 在此处执行实验
    result = train_single_algorithm(...)
# 环境变量自动恢复
```

### 2. 配置验证

在实验开始前打印当前环境变量状态：

```python
import os

def print_env_config():
    env_vars = [
        "CENTRAL_RESOURCE",
        "RESOURCE_ALLOCATION_MODE", 
        "RANDOM_SEED",
        "DISABLE_MIGRATION",
        "ENFORCE_OFFLOAD_MODE",
    ]
    
    print("=" * 70)
    print("环境变量配置")
    print("=" * 70)
    for var in env_vars:
        value = os.environ.get(var, "<未设置>")
        print(f"  {var}: {value}")
    print("=" * 70)
```

### 3. 批量实验建议

在批量运行实验时，为每个实验明确设置环境变量：

```python
# 推荐做法
experiments = [
    {"name": "标准模式", "env": {"CENTRAL_RESOURCE": "0"}},
    {"name": "分层模式", "env": {"CENTRAL_RESOURCE": "1"}},
]

for exp in experiments:
    with _temporary_environ(exp["env"]):
        run_experiment(...)
```

### 4. 日志记录

在实验结果中记录环境变量配置：

```python
metadata = {
    "experiment_name": "arrival_rate_comparison",
    "timestamp": "2024-11-16T10:30:00",
    "env_config": {
        "CENTRAL_RESOURCE": os.environ.get("CENTRAL_RESOURCE", "0"),
        "RANDOM_SEED": os.environ.get("RANDOM_SEED", "42"),
    }
}
```

---

## 常见问题排查

### Q1: 实验结果不一致？

**检查项**:
1. 确认 `RANDOM_SEED` 是否一致
2. 验证 `CENTRAL_RESOURCE` 设置
3. 检查是否有残留的环境变量

**解决方案**:
```bash
# 清理所有相关环境变量
unset CENTRAL_RESOURCE
unset RESOURCE_ALLOCATION_MODE
unset DISABLE_MIGRATION
unset ENFORCE_OFFLOAD_MODE

# 重新设置需要的变量
export RANDOM_SEED=42
python run_experiment.py
```

### Q2: 策略模式不符合预期？

**检查项**:
1. 验证 `CENTRAL_RESOURCE` 值（应为 "0" 或 "1"）
2. 确认 `RESOURCE_ALLOCATION_MODE` 设置

**解决方案**:
```python
# 在代码中验证
import os
assert os.environ.get("CENTRAL_RESOURCE") in ["0", "1"], "Invalid CENTRAL_RESOURCE"
```

### Q3: 并行实验出现环境变量污染？

**问题原因**: 多个实验同时修改全局环境变量

**解决方案**: 使用进程隔离而非线程隔离
```python
# 不推荐：线程共享环境变量
import threading

# 推荐：进程隔离
import subprocess
subprocess.run([sys.executable, "run_experiment.py"], env=custom_env)
```

---

## 环境变量检查清单

在运行重要实验前，使用此清单验证配置：

- [ ] `RANDOM_SEED` 已设置且一致
- [ ] `CENTRAL_RESOURCE` 符合预期模式
- [ ] `RESOURCE_ALLOCATION_MODE` 正确配置
- [ ] 清理了上次实验的残留环境变量
- [ ] 在实验日志中记录了环境变量配置
- [ ] 使用了环境变量隔离机制（如有并行）

---

## 参考资料

### 相关文件
- `experiments/td3_strategy_suite/comparison_suite.py`
- `experiments/td3_strategy_suite/strategy_runner.py`
- `train_single_agent.py`

### 相关文档
- [TD3策略对比实验全面审查报告](../../.qoder/quests/unnamed-task-1763294023.md)
- [VEC系统参数配置报告](../../VEC系统参数配置报告.md)

---

**最后更新**: 2025-11-16  
**维护人员**: AI Assistant  
**版本**: v1.0.0
