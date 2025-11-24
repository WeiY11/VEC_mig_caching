# TD3 系列算法对比实验指南

## 快速开始

### 1. 快速测试（推荐先运行）

```bash
# 50轮训练，约5-10分钟
python compare_enhanced_td3.py --mode quick
```

### 2. 标准实验

```bash
# 500轮训练，约1-2小时
python compare_enhanced_td3.py --mode standard
```

### 3. 完整实验

```bash
# 1500轮训练，约3-5小时
python compare_enhanced_td3.py --mode full
```

### 4. 自定义配置

```bash
# 指定轮次、车辆数和随机种子
python compare_enhanced_td3.py --episodes 800 --num-vehicles 16 --seed 123
```

## 对比的算法

脚本会自动依次运行以下 4 个算法：

1. **TD3** - 标准 Twin Delayed DDPG

   - 基线算法
   - 无特殊优化

2. **CAM_TD3** - Cache-Aware Migration TD3

   - 缓存感知
   - 任务迁移优化

3. **ENHANCED_TD3** - 增强型 TD3（全优化）

   - ✅ 分布式 Critic（51 分位数）
   - ✅ 熵正则化（自适应温度）
   - ✅ 模型化队列预测（5 步 rollout）
   - ✅ 队列感知回放（智能采样）
   - ✅ GAT 路由器（协同缓存）

4. **ENHANCED_CAM_TD3** - 增强型 CAM_TD3（队列焦点）
   - ✅ 分布式 Critic（尾部惩罚 0.7）
   - ✅ 队列感知回放（权重 0.4）
   - ✅ 模型化队列预测
   - 专注高拥塞场景

## 生成的报告

实验完成后，会在 `results/td3_comparison/run_YYYYMMDD_HHMMSS/` 目录下生成：

### 1. 对比表格 (`comparison_table.csv`)

- 最终奖励
- 任务完成率
- 平均时延
- 总能耗
- 缓存命中率
- 数据丢失率
- 训练用时

### 2. 训练曲线图 (`training_curves.png`)

包含 4 个子图：

- 训练奖励曲线
- 平均时延曲线
- 缓存命中率曲线
- 任务完成率曲线

### 3. 性能雷达图 (`performance_radar.png`)

5 个维度的雷达对比：

- 任务完成率
- 缓存命中率
- 时延（越低越好）
- 能耗（越低越好）
- 数据丢失（越低越好）

### 4. 文本摘要 (`summary.txt`)

- 算法性能排名
- 详细指标对比
- 训练用时统计

### 5. 实验配置 (`experiment_config.json`)

- 实验参数记录
- 可重现性保证

## 查看结果

```bash
# 查看对比表格
cat results/td3_comparison/run_*/comparison_table.csv

# 查看文本摘要
cat results/td3_comparison/run_*/summary.txt

# 打开图表（Windows）
explorer results\td3_comparison\run_*\training_curves.png
explorer results\td3_comparison\run_*\performance_radar.png
```

## 注意事项

### 时间预估

- **Quick 模式（50 轮）**: 5-10 分钟
- **Standard 模式（500 轮）**: 1-2 小时
- **Full 模式（1500 轮）**: 3-5 小时

### 资源需求

- **内存**: 建议 >= 8GB
- **GPU**: 可选，有 GPU 会更快（自动检测）
- **磁盘**: 每次实验约 100-500MB

### 中断恢复

如果实验中断：

- 已完成的算法结果会保存
- 可以手动运行未完成的算法：
  ```bash
  python train_single_agent.py --algorithm ENHANCED_TD3 --episodes 500 --num-vehicles 12
  ```

## 高级用法

### 只运行部分算法

编辑 `compare_enhanced_td3.py` 中的 `ALGORITHMS` 列表：

```python
ALGORITHMS = ['TD3', 'ENHANCED_TD3']  # 只对比这两个
```

### 修改车辆数进行实验

```bash
# 测试不同车辆数的性能
python compare_enhanced_td3.py --mode standard --num-vehicles 8
python compare_enhanced_td3.py --mode standard --num-vehicles 12
python compare_enhanced_td3.py --mode standard --num-vehicles 16
python compare_enhanced_td3.py --mode standard --num-vehicles 20
```

### 多次实验（不同随机种子）

```bash
python compare_enhanced_td3.py --mode standard --seed 42
python compare_enhanced_td3.py --mode standard --seed 123
python compare_enhanced_td3.py --mode standard --seed 456
```

## 预期结果

根据优化设计，预期性能排名：

### 常规场景（正常负载）

1. **ENHANCED_TD3** - 全面优化，平衡最好
2. **ENHANCED_CAM_TD3** - 队列优化强，但可能探索不足
3. **CAM_TD3** - 缓存感知有优势
4. **TD3** - 基线性能

### 高拥塞场景（>3.0 tasks/s 或 >16 车辆）

1. **ENHANCED_CAM_TD3** - 队列焦点优化发挥优势
2. **ENHANCED_TD3** - 全优化依然强劲
3. **CAM_TD3** - 基础缓存优化
4. **TD3** - 基线

## 故障排除

### 问题：训练失败

- 检查是否有足够内存
- 查看错误日志
- 尝试减少 `--num-vehicles` 或 `--episodes`

### 问题：结果文件未找到

- 等待 2-3 秒后再查看
- 检查 `results/single_agent/` 目录
- 确认训练是否真的完成

### 问题：图表无法生成

- 确保安装了 `matplotlib` 和 `pandas`：
  ```bash
  pip install matplotlib pandas
  ```

## 贡献者

VEC_mig_caching Team
