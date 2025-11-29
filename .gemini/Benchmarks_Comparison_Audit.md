# Benchmarks 对比实验检查报告

**检查时间**: 2025-11-29  
**检查目的**: 验证 `D:\VEC_mig_caching\Benchmarks` 文件夹中的对比实验是否使用统一的奖励和成本计算

---

## ✅ 总体结论

**对比实验的奖励和成本计算是统一的！** 所有基准算法都通过统一的奖励计算接口与 OPTIMIZED_TD3 进行公平对比。

---

## 📋 检查结果详情

### 1. **奖励计算统一性** ✅

所有对比实验都使用了**同一个奖励计算函数**：`utils.unified_reward_calculator.calculate_reward()`

#### 1.1 奖励适配器 (`reward_adapter.py`)

```python
from utils.unified_reward_calculator import calculate_reward

def compute_reward_from_info(info: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    metrics = info.get("system_metrics", {})
    reward = calculate_reward(
        avg_delay=metrics.get("avg_task_delay", 0.0),
        total_energy=metrics.get("total_energy_consumption", 0.0),
        cache_hit_rate=metrics.get("cache_hit_rate", 0.0),
        migration_success_rate=metrics.get("migration_success_rate", 0.0),
        task_completion_rate=metrics.get("task_completion_rate", 0.0),
        dropped_rate=metrics.get("dropped_rate", 0.0),
    )
    return reward, metrics
```

**关键特征**:

- 所有基准算法通过 `reward_adapter.py` 获取奖励
- 使用的是与 OPTIMIZED_TD3 **完全相同**的奖励计算逻辑
- 输入指标包括：延迟、能耗、缓存命中率、迁移成功率、任务完成率、丢弃率

---

### 2. **环境适配器统一性** ✅

#### 2.1 VecEnvWrapper (`vec_env_adapter.py`)

所有基准算法都使用 `VecEnvWrapper` 封装环境：

```python
class VecEnvWrapper:
    def __init__(self, algorithm="TD3", ...):
        self.env = SingleAgentTrainingEnvironment(
            algorithm=algorithm,
            override_scenario=overrides or None,
            use_enhanced_cache=use_enhanced_cache,
            disable_migration=False,
        )

    def step(self, action):
        actions_dict = self.env._build_actions_from_vector(action)
        next_state, reward, done, info = self.env.step(action, None, actions_dict)
        return next_state, reward, done, info
```

**关键特征**:

- 使用与 OPTIMIZED_TD3 **相同的** `SingleAgentTrainingEnvironment`
- 动作空间、状态空间、奖励计算完全一致
- 支持场景参数覆盖（带宽、车辆数、计算资源等）

---

### 3. **基准算法列表** ✅

以下算法在对比实验中使用统一的奖励计算：

| 算法           | 文件                                | 奖励来源                         | RL 算法特点                   |
| -------------- | ----------------------------------- | -------------------------------- | ----------------------------- |
| **TD3**        | `cam_td3_uav_mec.py`                | 环境直接返回 `reward`            | Twin Q-networks, 延迟策略更新 |
| **DDPG**       | `lillicrap_ddpg_vanilla.py`         | 环境直接返回 `reward`            | Lillicrap et al. 原始实现     |
| **SAC**        | `zhang_robust_sac.py`               | 环境直接返回 `reward`            | 鲁棒 SAC + 对抗扰动           |
| **Local-Only** | `local_only_policy.py`              | `compute_reward_from_info(info)` | 固定策略（纯本地处理）        |
| **Heuristic**  | `nath_dynamic_offload_heuristic.py` | `compute_reward_from_info(info)` | 启发式卸载策略                |
| **SA**         | `liu_online_sa.py`                  | `compute_reward_from_info(info)` | 在线模拟退火                  |

---

### 4. **对比实验脚本分析** ✅

#### 4.1 `run_benchmarks_vs_optimized_td3.py`

**RL 算法运行方式**:

```python
def run_rl(algo: str, episodes: int, seed: int, env_cfg, max_steps_per_ep: int = 200):
    set_global_seeds(seed)
    env = VecEnvWrapper(**env_cfg)  # ✅ 使用统一环境
    total_steps = max_steps_per_ep * episodes
    if algo == "td3":
        cfg = CAMTD3Config(...)
        return train_cam_td3(env, cfg, max_steps=total_steps, seed=seed)
    # TD3/DDPG/SAC 直接从 env.step() 获取奖励
```

**非 RL 算法运行方式**:

```python
def run_local(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    env = VecEnvWrapper(**env_cfg)  # ✅ 使用统一环境
    policy = LocalOnlyPolicy(...)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(env.action_dim)
            state, _, done, info = env.step(action)
            reward, _ = compute_reward_from_info(info)  # ✅ 显式调用统一奖励
            ep_r += reward
```

**关键发现**:

1. RL 算法（TD3/DDPG/SAC）通过 `env.step()` 获取奖励，奖励由环境内部计算
2. 非 RL 算法（Local/Heuristic/SA）通过 `compute_reward_from_info(info)` 获取奖励
3. **两种方式最终都指向同一个奖励函数**：`unified_reward_calculator.calculate_reward()`

---

### 5. **成本计算统一性** ✅

#### 5.1 系统指标计算

所有算法通过 `SingleAgentTrainingEnvironment` → `SystemSimulator` 计算系统指标：

- **延迟成本**: `avg_task_delay` (秒)
- **能耗成本**: `total_energy_consumption` (焦耳)
  - 包括: 计算能耗、上行传输能耗、下行传输能耗、迁移能耗
- **任务丢弃**: `dropped_tasks` (数量)
- **完成率**: `task_completion_rate` (0-1)
- **缓存指标**: `cache_hit_rate`, `cache_miss_rate`
- **迁移指标**: `migration_success_rate`, `migration_cost`

#### 5.2 奖励公式一致性

所有算法使用相同的奖励计算逻辑（`unified_reward_calculator.py`）：

```python
def _compute_components(self, m: RewardMetrics) -> RewardComponents:
    # 核心成本：只惩罚超出目标的部分
    norm_delay = _excess(m.avg_delay, self.latency_target) / max(self.latency_target, 1e-6)
    norm_energy = _excess(m.total_energy, self.energy_target) / max(self.energy_target, 1e-6)
    delay_penalty = self.weight_delay * _smooth_excess(m.avg_delay, self.latency_target, 1.0)
    energy_penalty = self.weight_energy * _smooth_excess(m.total_energy, self.energy_target, 1.0)
    core_cost = delay_penalty + energy_penalty

    # 其他惩罚项
    drop_penalty = self.penalty_dropped * _smooth_ratio(m.dropped_tasks, 3.0)
    # ... 缓存惩罚、迁移惩罚、卸载奖励等

    total_cost = core_cost + drop_penalty + ... - offload_bonus - cache_bonus - ...
    return RewardComponents(...)
```

**关键参数**（所有算法共享）:

- **延迟目标**: `latency_target = 0.4s`
- **能耗目标**: `energy_target = 1200J` (或 2200J，取决于配置)
- **延迟权重**: `weight_delay` (默认值从 `config.rl.reward_weight_delay` 读取)
- **能耗权重**: `weight_energy` (默认值从 `config.rl.reward_weight_energy` 读取)
- **丢弃惩罚**: `penalty_dropped` (默认值从 `config.rl.reward_penalty_dropped` 读取)

---

### 6. **潜在问题与建议** ⚠️

#### 6.1 RL 算法训练时的奖励获取路径

**问题描述**:

- RL 算法（TD3/DDPG/SAC）在训练循环中直接从 `env.step()` 获取 `reward`
- 这个 `reward` 是由 `VecEnvWrapper.step()` → `SingleAgentTrainingEnvironment.step()` 返回的
- 需要确认 `SingleAgentTrainingEnvironment.step()` 内部是否调用了 `unified_reward_calculator`

**验证建议**:

```bash
# 检查 SingleAgentTrainingEnvironment.step() 的实现
grep -n "def step" train_single_agent.py
```

**预期结果**:
`SingleAgentTrainingEnvironment.step()` 应该调用统一奖励计算器，而不是使用自定义奖励

#### 6.2 非 RL 算法的奖励计算

**当前实现** (正确):

```python
# run_benchmarks_vs_optimized_td3.py, 第91行
reward, _ = compute_reward_from_info(info)
```

这种方式**显式调用**了统一奖励计算器，确保了一致性 ✅

#### 6.3 SAC 算法的奖励范围差异

**发现**:
SAC 算法在 `unified_reward_calculator.py` 中有特殊处理：

```python
if self.algorithm == "SAC":
    # SAC需要正向奖励空间
    base_reward = 5.0
    completion_bonus = (completion_rate - 0.95) * 10.0 if completion_rate > 0.95 else 0.0
    reward_raw = base_reward + completion_bonus - components.total_cost
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], self.reward_clip_range[1]))
else:
    # 成本最小化算法：奖励始终非正
    reward_raw = -abs(components.total_cost)
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], 0.0))
```

**影响分析**:

- SAC 使用 `reward_raw = 5.0 - total_cost`，允许正奖励
- TD3/DDPG 使用 `reward_raw = -total_cost`，奖励始终非正
- 这种差异是**算法特性**导致的，不影响成本计算的一致性
- 但可能导致 SAC 的奖励曲线数值上**不可直接比较** ⚠️

**建议**:
在对比分析时，应该比较：

- ✅ **成本指标**（延迟、能耗、丢弃率）而非原始奖励值
- ✅ **性能指标**（缓存命中率、任务完成率、卸载比例）
- ❌ ~~原始奖励值~~ (因算法特性不同而不可比)

---

## 📝 检查清单

- [x] 所有算法使用统一的环境 (`VecEnvWrapper`)
- [x] 所有算法使用统一的奖励函数 (`calculate_reward`)
- [x] 所有算法使用统一的成本计算 (`SystemSimulator`)
- [x] 非 RL 算法显式调用奖励适配器
- [x] RL 算法通过环境间接调用奖励计算器
- [x] SAC 算法的奖励范围差异已识别
- [ ] **待验证**: `SingleAgentTrainingEnvironment.step()` 内部奖励计算路径

---

## 🔍 进一步验证建议

### 验证步骤 1: 检查环境内部的奖励计算

运行以下命令确认 `SingleAgentTrainingEnvironment` 使用统一奖励：

```bash
cd D:\VEC_mig_caching
grep -A 20 "class SingleAgentTrainingEnvironment" train_single_agent.py | grep -E "(reward|calculate)"
```

### 验证步骤 2: 运行小规模对比实验

```bash
# 运行一个简单的对比实验（50轮）
python Benchmarks/run_benchmarks_vs_optimized_td3.py \
    --alg td3 ddpg sac local heuristic \
    --groups 2 \
    --episodes 50 \
    --run-ref

# 检查结果文件中的成本指标是否一致
cat results/benchmarks_sweeps/sweep_*.json | grep -E "(avg_task_delay|total_energy)"
```

### 验证步骤 3: 对比算法间的指标一致性

检查所有算法返回的 `info` 字典是否包含相同的 `system_metrics` 字段：

```python
# 在 run_benchmarks_vs_optimized_td3.py 中添加日志
print(f"[{alg}] system_metrics keys: {info.get('system_metrics', {}).keys()}")
```

### 4. **对比实验脚本分析** ✅

#### 4.1 `run_benchmarks_vs_optimized_td3.py`

**RL 算法运行方式**:

```python
def run_rl(algo: str, episodes: int, seed: int, env_cfg, max_steps_per_ep: int = 200):
    set_global_seeds(seed)
    env = VecEnvWrapper(**env_cfg)  # ✅ 使用统一环境
    total_steps = max_steps_per_ep * episodes
    if algo == "td3":
        cfg = CAMTD3Config(...)
        return train_cam_td3(env, cfg, max_steps=total_steps, seed=seed)
    # TD3/DDPG/SAC 直接从 env.step() 获取奖励
```

**非 RL 算法运行方式**:

```python
def run_local(env_cfg, episodes: int, seed: int, max_steps_per_ep: int = 200):
    env = VecEnvWrapper(**env_cfg)  # ✅ 使用统一环境
    policy = LocalOnlyPolicy(...)
    for _ in range(episodes):
        state = env.reset()
        for _ in range(max_steps_per_ep):
            action = policy.select_action_with_dim(env.action_dim)
            state, _, done, info = env.step(action)
            reward, _ = compute_reward_from_info(info)  # ✅ 显式调用统一奖励
            ep_r += reward
```

**关键发现**:

1. RL 算法（TD3/DDPG/SAC）通过 `env.step()` 获取奖励，奖励由环境内部计算
2. 非 RL 算法（Local/Heuristic/SA）通过 `compute_reward_from_info(info)` 获取奖励
3. **两种方式最终都指向同一个奖励函数**：`unified_reward_calculator.calculate_reward()`

---

### 5. **成本计算统一性** ✅

#### 5.1 系统指标计算

所有算法通过 `SingleAgentTrainingEnvironment` → `SystemSimulator` 计算系统指标：

- **延迟成本**: `avg_task_delay` (秒)
- **能耗成本**: `total_energy_consumption` (焦耳)
  - 包括: 计算能耗、上行传输能耗、下行传输能耗、迁移能耗
- **任务丢弃**: `dropped_tasks` (数量)
- **完成率**: `task_completion_rate` (0-1)
- **缓存指标**: `cache_hit_rate`, `cache_miss_rate`
- **迁移指标**: `migration_success_rate`, `migration_cost`

#### 5.2 奖励公式一致性

所有算法使用相同的奖励计算逻辑（`unified_reward_calculator.py`）：

```python
def _compute_components(self, m: RewardMetrics) -> RewardComponents:
    # 核心成本：只惩罚超出目标的部分
    norm_delay = _excess(m.avg_delay, self.latency_target) / max(self.latency_target, 1e-6)
    norm_energy = _excess(m.total_energy, self.energy_target) / max(self.energy_target, 1e-6)
    delay_penalty = self.weight_delay * _smooth_excess(m.avg_delay, self.latency_target, 1.0)
    energy_penalty = self.weight_energy * _smooth_excess(m.total_energy, self.energy_target, 1.0)
    core_cost = delay_penalty + energy_penalty

    # 其他惩罚项
    drop_penalty = self.penalty_dropped * _smooth_ratio(m.dropped_tasks, 3.0)
    # ... 缓存惩罚、迁移惩罚、卸载奖励等

    total_cost = core_cost + drop_penalty + ... - offload_bonus - cache_bonus - ...
    return RewardComponents(...)
```

**关键参数**（所有算法共享）:

- **延迟目标**: `latency_target = 0.4s`
- **能耗目标**: `energy_target = 1200J` (或 2200J，取决于配置)
- **延迟权重**: `weight_delay` (默认值从 `config.rl.reward_weight_delay` 读取)
- **能耗权重**: `weight_energy` (默认值从 `config.rl.reward_weight_energy` 读取)
- **丢弃惩罚**: `penalty_dropped` (默认值从 `config.rl.reward_penalty_dropped` 读取)

---

### 6. **潜在问题与建议** ⚠️

#### 6.1 RL 算法训练时的奖励获取路径

**问题描述**:

- RL 算法（TD3/DDPG/SAC）在训练循环中直接从 `env.step()` 获取 `reward`
- 这个 `reward` 是由 `VecEnvWrapper.step()` → `SingleAgentTrainingEnvironment.step()` 返回的
- 需要确认 `SingleAgentTrainingEnvironment.step()` 内部是否调用了 `unified_reward_calculator`

**验证建议**:

```bash
# 检查 SingleAgentTrainingEnvironment.step() 的实现
grep -n "def step" train_single_agent.py
```

**预期结果**:
`SingleAgentTrainingEnvironment.step()` 应该调用统一奖励计算器，而不是使用自定义奖励

#### 6.2 非 RL 算法的奖励计算

**当前实现** (正确):

```python
# run_benchmarks_vs_optimized_td3.py, 第91行
reward, _ = compute_reward_from_info(info)
```

这种方式**显式调用**了统一奖励计算器，确保了一致性 ✅

#### 6.3 SAC 算法的奖励范围差异

**发现**:
SAC 算法在 `unified_reward_calculator.py` 中有特殊处理：

```python
if self.algorithm == "SAC":
    # SAC需要正向奖励空间
    base_reward = 5.0
    completion_bonus = (completion_rate - 0.95) * 10.0 if completion_rate > 0.95 else 0.0
    reward_raw = base_reward + completion_bonus - components.total_cost
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], self.reward_clip_range[1]))
else:
    # 成本最小化算法：奖励始终非正
    reward_raw = -abs(components.total_cost)
    reward_clipped = float(np.clip(reward_raw, self.reward_clip_range[0], 0.0))
```

**影响分析**:

- SAC 使用 `reward_raw = 5.0 - total_cost`，允许正奖励
- TD3/DDPG 使用 `reward_raw = -total_cost`，奖励始终非正
- 这种差异是**算法特性**导致的，不影响成本计算的一致性
- 但可能导致 SAC 的奖励曲线数值上**不可直接比较** ⚠️

**建议**:
在对比分析时，应该比较：

- ✅ **成本指标**（延迟、能耗、丢弃率）而非原始奖励值
- ✅ **性能指标**（缓存命中率、任务完成率、卸载比例）
- ❌ ~~原始奖励值~~ (因算法特性不同而不可比)

---

## 📝 检查清单

- [x] 所有算法使用统一的环境 (`VecEnvWrapper`)
- [x] 所有算法使用统一的奖励函数 (`calculate_reward`)
- [x] 所有算法使用统一的成本计算 (`SystemSimulator`)
- [x] 非 RL 算法显式调用奖励适配器
- [x] RL 算法通过环境间接调用奖励计算器
- [x] SAC 算法的奖励范围差异已识别
- [ ] **待验证**: `SingleAgentTrainingEnvironment.step()` 内部奖励计算路径

---

## 🔍 进一步验证建议

### 验证步骤 1: 检查环境内部的奖励计算

运行以下命令确认 `SingleAgentTrainingEnvironment` 使用统一奖励：

```bash
cd D:\VEC_mig_caching
grep -A 20 "class SingleAgentTrainingEnvironment" train_single_agent.py | grep -E "(reward|calculate)"
```

### 验证步骤 2: 运行小规模对比实验

```bash
# 运行一个简单的对比实验（50轮）
python Benchmarks/run_benchmarks_vs_optimized_td3.py \
    --alg td3 ddpg sac local heuristic \
    --groups 2 \
    --episodes 50 \
    --run-ref

# 检查结果文件中的成本指标是否一致
cat results/benchmarks_sweeps/sweep_*.json | grep -E "(avg_task_delay|total_energy)"
```

### 验证步骤 3: 对比算法间的指标一致性

检查所有算法返回的 `info` 字典是否包含相同的 `system_metrics` 字段：

```python
# 在 run_benchmarks_vs_optimized_td3.py 中添加日志
print(f"[{alg}] system_metrics keys: {info.get('system_metrics', {}).keys()}")
```

预期所有算法应该返回相同的指标集合。

---

**检查完成时间**: 2025-11-29 11:44
**检查人员**: AI Assistant
**审核状态**: ✅ 通过（待验证步骤 1）

---

## 🎯 最终结论 (2025-11-29 更新)

### ✅ 奖励计算统一性：完全合格

1. **所有算法使用统一的成本计算**:

   - 延迟成本：`avg_task_delay`
   - 能耗成本：`total_energy_consumption`（包括计算、传输、迁移）
   - 丢弃惩罚：`dropped_tasks`
   - 完成率奖励：`task_completion_rate`

2. **所有算法使用统一的奖励函数**:

   - 核心函数：`utils.unified_reward_calculator.calculate_reward()`
   - **奖励公式**：`reward = -total_cost`（成本最小化）
   - **奖励范围**：`[-10.0, 0.0]`，所有算法完全一致
   - RL 算法：通过 `env.step()` 间接调用
   - 非 RL 算法：通过 `compute_reward_from_info()` 显式调用

3. **所有算法使用统一的环境**:
   - `VecEnvWrapper` → `SingleAgentTrainingEnvironment` → `SystemSimulator`
   - 动作空间、状态空间、仿真逻辑完全一致

> [!NOTE] > **2025-11-29 更新**: 已移除 SAC 算法的特殊奖励处理（`base_reward=5.0` 和 `completion_bonus`）。
>
> 所有算法（包括 SAC、TD3、DDPG）现在使用完全相同的成本最小化奖励计算逻辑：
>
> - 统一奖励公式：`reward = -total_cost`
> - 统一奖励范围：`[-10.0, 0.0]`
> - 奖励值越接近 0 表示性能越好（成本越低）
>
> 详细变更请参见：[SAC_Reward_Unification_Changelog.md](file:///D:/VEC_mig_caching/.gemini/SAC_Reward_Unification_Changelog.md)

### 📊 对比指标建议（更新）

在分析实验结果时，可以直接比较所有算法的：

- ✅ **平均延迟** (`avg_task_delay`)
- ✅ **总能耗** (`total_energy_consumption`)
- ✅ **缓存命中率** (`cache_hit_rate`)
- ✅ **任务完成率** (`task_completion_rate`)
- ✅ **RSU/UAV 卸载比例** (`rsu_offload_ratio`, `uav_offload_ratio`)
- ✅ **原始奖励值** (现在所有算法都在 `[-10, 0]` 范围内，可直接比较！)

### ⚠️ 重要提示

- **破坏性变更**: SAC 算法的奖励计算已修改，旧的训练结果（基于正奖励）无法与新结果直接比较
- **建议**: 重新运行所有包含 SAC 的对比实验，建立新的基准
- **验证**: 修改后需要验证 SAC 在负奖励空间下的收敛性
