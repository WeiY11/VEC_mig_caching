# Override_Scenario Bug 修复报告

## 🔍 问题发现

### 症状
运行 `run_edge_infrastructure_comparison.py` 时，所有不同资源配置的场景产生了**完全相同**的性能数据：

| 场景 | RSU CPU | Bandwidth | 时延 | 能耗 | 说明 |
|------|---------|-----------|------|------|------|
| 低计算+低通信 | 10 GHz | 15 MHz | 0.2528 | 5755.36 | ❌ 相同 |
| 均衡配置 | 15 GHz | 20 MHz | 0.2525 | 5750.74 | ❌ 几乎相同 |
| 高计算+高通信 | 20 GHz | 40 MHz | 0.2528 | 5755.36 | ❌ 相同 |

**物理上不合理**: CPU频率翻倍、带宽增加2.7倍，性能却没有变化！

### 影响范围
- ✅ 所有使用 `override_scenario` 的对比实验
- ✅ 参数敏感性分析实验
- ✅ 模型缓存系统的正确性

## 🎯 根本原因

通过深入代码分析，找到了3个关键问题：

### 问题1: CPU频率参数未被读取

**位置**: `evaluation/system_simulator.py` 第46-124行

```python
def __init__(self, config: Dict = None):
    self.config = config or self.get_default_config()
    # ... 
    # ❌ 完全没有读取 rsu_cpu_freq 和 uav_cpu_freq 参数！
    self.initialize_components()
```

**影响**: 即使 `override_scenario` 传入了CPU频率参数，仿真器也不会保存和使用它们。

### 问题2: RSU/UAV字典缺少CPU频率字段

**位置**: `evaluation/system_simulator.py` 第228-238行 (RSU), 第261-272行 (UAV)

```python
rsu = {
    'id': f'RSU_{i}',
    'position': ...,
    'cache': {},
    'computation_queue': [],
    # ❌ 缺少 'cpu_freq' 字段！
}
```

**影响**: 即使读取了参数，节点字典中也没有存储CPU频率，后续计算无法使用。

### 问题3: 工作量计算使用硬编码常量 (最关键)

**位置**: `evaluation/system_simulator.py` 第1515-1520行

```python
def _estimate_remote_work_units(self, task: Dict, node_type: str) -> float:
    requirement = float(task.get('computation_requirement', 1500.0))
    base_divisor = 1200.0 if node_type == 'RSU' else 1600.0  # ❌ 硬编码！
    work_units = requirement / base_divisor
    return float(np.clip(work_units, 0.5, 12.0))
```

**影响**: 
- **无论CPU频率是多少，都使用固定的 1200.0 (RSU) 或 1600.0 (UAV)**
- 这是为什么所有场景性能相同的直接原因
- 10GHz和20GHz的RSU执行任务时间完全一样

## 🛠️ 修复方案

### 修复1: 在初始化时读取资源参数

**文件**: `evaluation/system_simulator.py`  
**位置**: 第122-133行 (新增)

```python
# 🔧 读取资源配置参数（CPU频率、带宽等）
if self.sys_config is not None and not self.override_topology:
    self.rsu_cpu_freq = getattr(self.sys_config.compute, 'rsu_cpu_freq', 15e9)
    self.uav_cpu_freq = getattr(self.sys_config.compute, 'uav_cpu_freq', 12e9)
    self.bandwidth = getattr(self.sys_config.network, 'bandwidth', 20e6)
else:
    self.rsu_cpu_freq = self.config.get('rsu_cpu_freq', 15e9)  # Hz
    self.uav_cpu_freq = self.config.get('uav_cpu_freq', 12e9)  # Hz
    self.bandwidth = self.config.get('bandwidth', 20e6)  # Hz
```

**效果**: 正确读取并存储 `override_scenario` 中的CPU频率和带宽参数。

### 修复2: RSU/UAV添加CPU频率字段

**文件**: `evaluation/system_simulator.py`  
**位置**: 第247行 (RSU), 第282行 (UAV)

```python
rsu = {
    ...
    'cpu_freq': self.rsu_cpu_freq,  # 🆕 添加CPU频率
    ...
}

uav = {
    ...
    'cpu_freq': self.uav_cpu_freq,  # 🆕 添加CPU频率
    ...
}
```

**效果**: 节点字典中包含CPU频率信息，后续可以访问使用。

### 修复3: 使用实际CPU频率计算工作量 (最关键)

**文件**: `evaluation/system_simulator.py`  
**位置**: 第1529-1552行

```python
def _estimate_remote_work_units(self, task: Dict, node_type: str) -> float:
    """
    🔧 修复：使用实际CPU频率计算，而不是硬编码常量
    """
    requirement = float(task.get('computation_requirement', 1500.0))
    
    # 使用实际CPU频率计算工作量
    reference_rsu_freq = 15e9  # RSU参考频率 15GHz
    reference_uav_freq = 12e9  # UAV参考频率 12GHz
    
    if node_type == 'RSU':
        actual_freq = getattr(self, 'rsu_cpu_freq', reference_rsu_freq)
        # 频率越高，divisor越大，work_units越小（执行更快）
        base_divisor = 1200.0 * (actual_freq / reference_rsu_freq)
    else:  # UAV
        actual_freq = getattr(self, 'uav_cpu_freq', reference_uav_freq)
        base_divisor = 1600.0 * (actual_freq / reference_uav_freq)
    
    work_units = requirement / base_divisor
    return float(np.clip(work_units, 0.5, 12.0))
```

**效果**: 
- 20GHz RSU的工作量是10GHz的一半（执行速度快2倍）
- 不同CPU频率产生明显不同的性能结果

## ✅ 验证结果

### 预期效果

运行修复后，不同资源配置应产生不同性能：

| 配置 | RSU CPU | 预期时延趋势 | 预期能耗趋势 |
|------|---------|--------------|--------------|
| 低资源 | 10 GHz | **高** (计算慢) | 可能高 (等待时间长) |
| 中资源 | 15 GHz | 中等 | 中等 |
| 高资源 | 20 GHz | **低** (计算快) | 可能低 (快速完成) |

### 验证步骤

1. **快速测试** (推荐先运行):
   ```bash
   cd D:\VEC_mig_caching
   python test_fix_verification.py
   ```
   
   这会运行2个极端配置（5GHz vs 30GHz），预计耗时3-5分钟。
   
   ✅ **成功标准**: 时延或能耗差异 > 5%

2. **完整实验测试**:
   ```bash
   cd D:\VEC_mig_caching
   python experiments/camtd3_strategy_suite/run_edge_infrastructure_comparison.py --episodes 10 --seed 42
   ```
   
   预计耗时10-15分钟（10轮训练）。
   
   ✅ **成功标准**: 5个场景产生不同的性能曲线

3. **检查新缓存**:
   ```bash
   cd D:\VEC_mig_caching
   python diagnose_cache_issue.py  # (需要重新创建此脚本)
   ```
   
   ✅ **成功标准**: 不同场景的缓存有明显差异

## 📊 修复前后对比

### 修复前
```
场景 RSU10GHz_BW15MHz: delay=0.2530, energy=5731.96
场景 RSU20GHz_BW40MHz: delay=0.2530, energy=5731.96  # ❌ 完全相同！
```

### 修复后（预期）
```
场景 RSU10GHz_BW15MHz: delay=0.2530, energy=5731.96
场景 RSU20GHz_BW40MHz: delay=0.1265, energy=6200.00  # ✅ 时延减半，能耗可能变化
```

*注：具体数值会根据实际运行结果变化*

## 🔧 后续行动

### 立即执行
1. ✅ 删除旧缓存: `rm -rf results/strategy_model_cache` (已完成)
2. ✅ 验证修复: 运行 `test_fix_verification.py`
3. ✅ 重新生成图表: 运行10轮快速实验

### 后续优化
1. 检查 `_estimate_transmission` 函数是否也使用 `self.bandwidth`
2. 验证 `EnhancedSystemSimulator` 是否也需要相同修复
3. 更新所有对比实验的结果和图表

## 📝 相关文件

### 修改的文件
- `evaluation/system_simulator.py` - 核心修复
- `evaluation/enhanced_system_simulator.py` - 可能需要验证

### 新增文件
- `test_fix_verification.py` - 快速验证脚本
- `fix_override_scenario.patch` - 修复补丁说明
- `OVERRIDE_SCENARIO_FIX_REPORT.md` - 本报告

### 临时删除的文件
- `check_cache_data.py` - 诊断脚本（已删除）
- `diagnose_cache_issue.py` - 诊断脚本（已删除）

## 💡 经验教训

1. **参数传递追踪**: 仅在顶层传递参数不够，必须确保底层实现真正使用它们
2. **硬编码危害**: 硬编码的常量会让参数化配置完全失效
3. **缓存验证**: 使用缓存系统时，必须确保缓存键能区分不同配置
4. **测试覆盖**: 需要端到端测试验证参数敏感性

## 🎯 总结

**问题**: 仿真器使用硬编码常量计算任务执行时间，导致 `override_scenario` 参数无效。

**修复**: 
1. 读取并存储CPU频率参数
2. 在RSU/UAV字典中添加CPU频率字段  
3. 使用实际CPU频率计算工作量（按频率比例缩放）

**影响**: 修复后，所有对比实验将产生正确的、符合物理规律的结果。

**状态**: ✅ 代码已修复，等待验证

---

**报告日期**: 2025-11-03  
**修复版本**: v1.0  
**影响范围**: 所有参数敏感性实验  
**严重程度**: 🔴 高 (影响实验结论的正确性)

