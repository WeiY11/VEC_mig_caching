# 立即停止训练并保存结果

## ⚠️ 立即执行的步骤

---

## 步骤1：停止训练（30秒）

### 在PowerShell中运行：

```powershell
ssh -p 21960 root@connect.westc.gpuhub.com
```
**密码**: `B9iXNm5Ee0l4`

### 连接后，复制粘贴以下命令：

```bash
# 停止训练进程
pkill -f run_batch_experiments.py

# 取消自动关机
shutdown -c

# 确认已停止
ps aux | grep python | grep -v grep

echo ""
echo "=========================================="
echo "✅ 训练已停止"
echo "✅ 自动关机已取消"
echo "=========================================="
```

---

## 步骤2：检查已保存的结果（1分钟）

### 在服务器上运行：

```bash
cd /root/VEC_mig_caching

# 查看结果目录结构
echo "【已完成的实验】"
ls -lh results/camtd3_strategy_suite/

echo ""
echo "【结果文件统计】"
find results/camtd3_strategy_suite -type f | wc -l

echo ""
echo "【各类型文件数量】"
echo "PNG图片: $(find results/camtd3_strategy_suite -name '*.png' | wc -l)"
echo "JSON数据: $(find results/camtd3_strategy_suite -name '*.json' | wc -l)"
echo "CSV数据: $(find results/camtd3_strategy_suite -name '*.csv' | wc -l)"

echo ""
echo "【结果目录大小】"
du -sh results/camtd3_strategy_suite/
```

---

## 步骤3：下载结果到本地（5-10分钟）

### 断开SSH（输入 `exit` 或按 Ctrl+D）

### 在本地PowerShell运行：

```powershell
cd D:\VEC_mig_caching

# 创建结果保存目录
New-Item -ItemType Directory -Path "results_from_server" -Force

# 下载所有结果
scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/results/camtd3_strategy_suite ./results_from_server/

# 下载日志
scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/logs ./results_from_server/logs

echo "✅ 结果下载完成！"
```

**密码**: `B9iXNm5Ee0l4`

---

## 步骤4：验证本地结果（1分钟）

```powershell
# 查看下载的结果
Get-ChildItem -Recurse results_from_server | Measure-Object | Select-Object Count

# 查看结果目录
explorer results_from_server
```

---

## 步骤5：关闭服务器（可选，节省费用）

### 重新SSH连接：

```powershell
ssh -p 21960 root@connect.westc.gpuhub.com
```

### 确认结果已下载后，关闭服务器：

```bash
# 立即关机
shutdown -h now
```

或者保留服务器（继续计费）：
```bash
# 不关机，退出SSH
exit
```

---

## 📊 预期下载内容

您将获得：

### 结果文件
```
results_from_server/
├── camtd3_strategy_suite/
│   ├── data_size/              # 实验1结果
│   │   ├── *.png              # 图表
│   │   ├── *.json             # 数据
│   │   └── *.csv              # 原始数据
│   ├── vehicle_count/          # 实验2结果
│   ├── local_resource_offload/ # 实验3结果
│   ├── local_resource_cost/    # 实验4结果（可能未完成）
│   └── ...
└── logs/
    └── training_*.log          # 训练日志
```

### 预计文件数量
- 图片: 20-30个 PNG
- 数据: 20-30个 JSON/CSV
- 日志: 1-2个 LOG

---

## 🆘 故障排查

### 问题1：显示"进程不存在"
→ 训练可能已经停止，继续执行步骤2

### 问题2：下载速度很慢
→ 正常，结果文件较大，耐心等待

### 问题3：提示"目录不存在"
→ 检查路径：
```bash
ls -la /root/VEC_mig_caching/results/
```

### 问题4：本地没有results_from_server目录
→ 手动创建：
```powershell
mkdir results_from_server
```

---

## ✅ 完成检查清单

- [ ] 训练进程已停止
- [ ] 自动关机已取消
- [ ] 已查看服务器结果
- [ ] 结果已下载到本地
- [ ] 本地结果可以打开
- [ ] 服务器已关机（可选）

---

## 💾 结果备份建议

下载完成后建议：

1. **打包保存**
```powershell
Compress-Archive -Path results_from_server -DestinationPath results_backup_$(Get-Date -Format 'yyyyMMdd').zip
```

2. **检查重要文件**
- 打开几个PNG图片确认完整性
- 检查JSON文件是否有数据
- 查看LOG文件最后几行

---

**现在开始执行步骤1吧！** 🚀




















