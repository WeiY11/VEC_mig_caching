# TD3策略对比实验归档文件说明

## 归档原因

本目录包含7个已弃用的实验脚本，这些实验在2024年被合并优化以提升实验效率。

---

## 归档时间

**归档日期**: 2024年（具体日期未知）  
**记录日期**: 2025-11-16  
**文档版本**: v1.0.0

---

## 归档文件清单

### 1. run_bandwidth_cost_comparison.py (6.7KB)
- **原用途**: 带宽成本对比实验（旧版本）
- **替代文件**: `../run_bandwidth_cost_comparison.py` (新版，30.6KB)
- **弃用原因**: 功能扩展，新版包含带宽/RSU计算资源/UAV计算资源三类实验
- **关键差异**: 
  - 旧版: 仅带宽对比
  - 新版: 整合3类资源对比实验

### 2. run_channel_quality_comparison.py (10.1KB)
- **原用途**: 信道质量对比实验
- **替代文件**: `../run_network_topology_comparison.py` (16.6KB)
- **弃用原因**: 合并到网络与拓扑综合对比实验
- **合并优化**: 与带宽、拓扑密度合并，节省配置数和训练时间

### 3. run_edge_communication_capacity_comparison.py (17.0KB)
- **原用途**: 边缘通信容量对比实验
- **替代文件**: `../run_edge_infrastructure_comparison.py` (23.5KB)
- **弃用原因**: 合并到边缘基础设施综合对比实验
- **合并优化**: 与边缘计算能力合并，统一边缘资源评估

### 4. run_edge_compute_capacity_comparison.py (13.3KB)
- **原用途**: 边缘计算能力对比实验
- **替代文件**: `../run_edge_infrastructure_comparison.py` (23.5KB)
- **弃用原因**: 合并到边缘基础设施综合对比实验
- **合并优化**: 统一评估边缘计算和通信资源

### 5. run_local_resource_cost_comparison.py (6.9KB)
- **原用途**: 本地资源成本对比实验
- **替代文件**: `../run_local_compute_resource_comparison.py` (14.0KB)
- **弃用原因**: 合并到本地计算资源综合对比实验
- **合并优化**: 与本地资源卸载对比合并

### 6. run_local_resource_offload_comparison.py (6.7KB)
- **原用途**: 本地资源卸载对比实验
- **替代文件**: `../run_local_compute_resource_comparison.py` (14.0KB)
- **弃用原因**: 合并到本地计算资源综合对比实验
- **合并优化**: 统一评估本地资源的成本和卸载效果

### 7. run_topology_density_comparison.py (8.4KB)
- **原用途**: 拓扑密度对比实验
- **替代文件**: `../run_network_topology_comparison.py` (16.6KB)
- **弃用原因**: 合并到网络与拓扑综合对比实验
- **合并优化**: 与带宽、信道质量合并，全面评估网络特性

---

## 实验整合优化效果

### 优化前（18个实验）
- **带宽/信道/拓扑**: 3个独立实验
- **边缘计算/通信**: 2个独立实验  
- **本地成本/卸载**: 2个独立实验
- **总配置数**: 95个
- **预估训练时间**: 285小时（@100轮/配置）

### 优化后（14个实验）
- **网络与拓扑综合**: 1个合并实验（6个场景）
- **边缘基础设施综合**: 1个合并实验（5个场景）
- **本地计算资源综合**: 1个合并实验（3个场景）
- **总配置数**: 37个
- **预估训练时间**: 111小时（节省61%）

### 关键改进
1. ✅ **配置数减少**: 95 → 37（节省61%）
2. ✅ **训练时间缩短**: 285h → 111h（节省61%）
3. ✅ **实验逻辑更清晰**: 按资源类型分组
4. ✅ **代码维护性提升**: 减少重复代码

---

## 合并映射关系

| 原实验类别 | 原实验数 | 新实验名称 | 新配置数 |
|-----------|---------|-----------|---------|
| 带宽 + 信道质量 + 拓扑密度 | 3 | 网络与拓扑综合对比 | 6 |
| 边缘计算能力 + 边缘通信资源 | 2 | 边缘基础设施综合对比 | 5 |
| 本地资源成本 + 本地资源卸载 | 2 | 本地计算资源综合对比 | 3 |

---

## 迁移指南

### 如何使用新版实验？

#### 1. 运行网络与拓扑综合对比
```bash
# 替代旧版的带宽/信道/拓扑实验
python experiments/td3_strategy_suite/run_network_topology_comparison.py --episodes 1500
```

#### 2. 运行边缘基础设施综合对比
```bash
# 替代旧版的边缘计算/通信实验
python experiments/td3_strategy_suite/run_edge_infrastructure_comparison.py --episodes 1500
```

#### 3. 运行本地计算资源综合对比
```bash
# 替代旧版的本地成本/卸载实验
python experiments/td3_strategy_suite/run_local_compute_resource_comparison.py --episodes 1500
```

### 结果对比说明

新版实验的结果目录结构：
```
results/parameter_sensitivity/
├── network_topology_<timestamp>/
│   ├── bandwidth_20mhz/
│   ├── bandwidth_30mhz/
│   ├── channel_good/
│   ├── channel_poor/
│   ├── topology_dense/
│   └── topology_sparse/
├── edge_infrastructure_<timestamp>/
│   ├── compute_high/
│   ├── compute_low/
│   ├── communication_high/
│   └── communication_low/
└── local_compute_<timestamp>/
    ├── low_1.2ghz/
    ├── medium_2.0ghz/
    └── high_2.8ghz/
```

---

## 恢复使用旧版实验（不推荐）

如需运行旧版实验（例如用于复现历史结果），可以：

1. **复制归档文件到主目录**:
   ```bash
   cp archived_experiments/run_bandwidth_cost_comparison.py ../run_bandwidth_cost_comparison_legacy.py
   ```

2. **重命名避免冲突**:
   ```bash
   # 旧版文件重命名
   mv run_bandwidth_cost_comparison_legacy.py run_bandwidth_legacy.py
   ```

3. **运行旧版脚本**:
   ```bash
   python experiments/td3_strategy_suite/run_bandwidth_legacy.py --episodes 1500
   ```

**注意**: 
- 旧版脚本可能缺少新版的优化和Bug修复
- 不保证旧版脚本与当前系统完全兼容
- 建议仅用于历史结果复现，不用于新实验

---

## 版本控制建议

### Git管理
如果需要追踪归档文件的历史版本：
```bash
# 查看文件历史
git log -- archived_experiments/run_bandwidth_cost_comparison.py

# 恢复特定版本
git checkout <commit-hash> -- archived_experiments/run_bandwidth_cost_comparison.py
```

### 分支管理
建议在Git中创建归档分支：
```bash
# 创建归档分支（如果尚未创建）
git checkout -b archive/legacy-experiments-2024

# 切换回主分支
git checkout main
```

---

## 清理计划

### 短期（当前）
- ✅ 保留归档文件（用于历史参考）
- ✅ 添加此说明文档
- ✅ 文件重命名添加日期前缀（可选）

### 中期（3-6个月后）
- 确认新版实验稳定运行
- 所有团队成员已完成迁移
- 考虑压缩归档文件或移至版本控制的归档分支

### 长期（1年后）
- 如无历史复现需求，可考虑完全删除
- 保留Git历史记录即可

---

## 联系方式

如有疑问或需要恢复旧版实验，请联系：
- **维护团队**: VEC项目组
- **文档更新**: 2025-11-16
- **问题反馈**: 通过项目Issue tracker

---

## 附录：文件大小对比

| 文件名 | 旧版大小 | 新版大小 | 增量 |
|--------|---------|---------|------|
| bandwidth_cost_comparison | 6.7KB | 30.6KB | +357% |
| network_topology_comparison | - | 16.6KB | 新增 |
| edge_infrastructure_comparison | - | 23.5KB | 新增 |
| local_compute_resource_comparison | - | 14.0KB | 新增 |

**说明**: 新版文件大小增加主要因为：
1. 整合多个实验逻辑
2. 增强的错误处理
3. 更详细的文档字符串
4. 新增的指标收集和可视化功能

---

**最后更新**: 2025-11-16  
**维护人员**: AI Assistant  
**版本**: v1.0.0
