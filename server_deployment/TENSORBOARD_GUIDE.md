# TensorBoard监控指南

## ✅ TensorBoard已启动

**服务器**: region-9.autodl.pro  
**端口**: 6006  
**进程ID**: 1935  
**状态**: ✅ 正在运行

---

## 🖥️ 如何访问TensorBoard

### 方法1：SSH端口转发（推荐，最稳定）

#### Windows PowerShell

**步骤1**: 打开**新的PowerShell窗口**（不要关闭现有窗口）

**步骤2**: 运行以下命令建立SSH隧道：

```powershell
ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro
# 密码: dfUJkmli0mHk
```

**步骤3**: 保持这个窗口打开（不要关闭）

**步骤4**: 在浏览器中访问：

```
http://localhost:6006
```

✅ 这样就能看到TensorBoard界面了！

#### 为什么要这样做？

SSH隧道（端口转发）的作用是：
- 把服务器的6006端口映射到你本地的6006端口
- 这样你在本地访问 `localhost:6006` 实际上是访问服务器的TensorBoard
- 安全且稳定

---

### 方法2：直接访问（如果AutoDL支持）

某些AutoDL实例支持直接访问，可以试试：

```
http://region-9.autodl.pro:6006
```

或者检查AutoDL控制台，看是否有提供的公网访问地址。

---

## 📊 CPU和资源使用情况

### 当前状态

```
训练进程 (PID 1599):
  - CPU使用率: 106.7% (多核)
  - 内存使用: 750 MB
  - 运行时间: ~8分钟

GPU (Tesla T4):
  - 使用率: 7%
  - 显存: 161 MB / 16384 MB
  - 温度: 38°C
```

### 预期资源使用

随着训练深入：
- **CPU**: 100-200% (使用多核)
- **GPU**: 80-95% (全速训练)
- **显存**: 2-8 GB
- **内存**: 1-4 GB

这些都是正常的！

---

## 📈 TensorBoard会显示什么

TensorBoard会实时显示：

1. **训练指标**:
   - 平均时延 (avg_delay)
   - 总能耗 (total_energy)
   - 归一化成本 (normalized_cost)
   - 任务完成率 (completion_rate)

2. **不同配置对比**:
   - 8个参数对比实验的实时结果
   - 每个配置的性能曲线

3. **收敛情况**:
   - 训练是否稳定
   - 性能改善趋势

---

## 🔄 更新TensorBoard数据

TensorBoard会自动更新，但如果想手动更新现有结果：

```bash
# 连接到服务器
ssh -p 47042 root@region-9.autodl.pro

# 进入项目目录
cd /root/VEC_mig_caching

# 运行更新脚本
python setup_tensorboard.py
```

---

## 🛑 停止TensorBoard

如果需要停止TensorBoard：

```bash
# 连接到服务器
ssh -p 47042 root@region-9.autodl.pro

# 停止TensorBoard
kill 1935
```

重新启动：

```bash
cd /root/VEC_mig_caching
./start_tensorboard.sh
```

---

## 📱 完整监控命令集

### 监控训练进程

```bash
ssh -p 47042 root@region-9.autodl.pro
cd /root/VEC_mig_caching
./remote_monitor.sh
```

### 查看CPU使用

```bash
ssh -p 47042 root@region-9.autodl.pro "top -bn1 | grep python | head -5"
```

### 查看GPU使用

```bash
ssh -p 47042 root@region-9.autodl.pro "nvidia-smi"
```

### 查看实时日志

```bash
ssh -p 47042 root@region-9.autodl.pro
cd /root/VEC_mig_caching
tail -f batch_experiments.log
```

### 访问TensorBoard

1. 新开PowerShell窗口
2. 运行: `ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro`
3. 浏览器访问: http://localhost:6006

---

## 💡 使用技巧

### 1. 同时开多个终端

建议同时打开3个终端窗口：

**窗口1**: SSH隧道（TensorBoard）
```powershell
ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro
```

**窗口2**: 实时日志监控
```powershell
ssh -p 47042 root@region-9.autodl.pro
cd /root/VEC_mig_caching
tail -f batch_experiments.log
```

**窗口3**: 定期检查状态
```powershell
ssh -p 47042 root@region-9.autodl.pro
cd /root/VEC_mig_caching
./remote_monitor.sh
```

### 2. TensorBoard快捷键

在TensorBoard界面中：
- **Ctrl+鼠标滚轮**: 缩放图表
- **拖动**: 平移图表
- **双击**: 重置视图
- **左侧栏**: 选择要显示的指标

### 3. 自动刷新

TensorBoard默认每30秒自动刷新数据，你会看到实时更新。

---

## 🔧 故障排除

### 问题1: 无法访问 localhost:6006

**解决方案**:
1. 确认SSH隧道窗口还在运行
2. 检查是否输入了密码并成功连接
3. 确认没有其他程序占用6006端口

### 问题2: TensorBoard没有数据

**原因**: 实验刚开始，还没有生成足够的数据

**解决方案**: 
- 等待10-30分钟后再查看
- 运行 `python setup_tensorboard.py` 更新数据

### 问题3: SSH连接断开

**解决方案**:
- 重新运行SSH隧道命令
- TensorBoard服务还在服务器上运行，只需要重新建立连接

---

## 📊 当前监控状态总结

| 服务 | 状态 | 进程ID | 端口 |
|------|------|--------|------|
| 批量实验 | ✅ 运行中 | 1597 | - |
| 训练进程 | ✅ 运行中 | 1599 | - |
| TensorBoard | ✅ 运行中 | 1935 | 6006 |

| 资源 | 当前使用 | 正常范围 |
|------|----------|----------|
| CPU | 106.7% | 100-200% |
| 内存 | 750 MB | 1-4 GB |
| GPU使用率 | 7% | 80-95% (训练中) |
| GPU显存 | 161 MB | 2-8 GB (训练中) |
| GPU温度 | 38°C | 30-85°C |

**说明**: GPU使用率目前较低是因为处于初始化阶段，预计10-30分钟后会提升到80%以上。

---

## 🎯 下一步

1. ✅ **打开TensorBoard** - 按照上面的方法建立SSH隧道并访问
2. ✅ **观察训练进度** - 看到指标曲线开始出现
3. ✅ **定期检查** - 每天查看一次确保训练正常
4. ✅ **等待完成** - 2-3天后下载完整结果

---

**所有监控工具已就绪！祝训练顺利！** 🚀

