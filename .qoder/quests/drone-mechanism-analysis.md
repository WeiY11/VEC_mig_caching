# 无人机(UAV)机制详细分析

## 一、UAV基础定位与角色

### 1.1 系统角色定位

UAV在VEC系统中作为**空中移动边缘计算节点**，主要职能包括:

- **空中计算卸载平台**: 为地面车辆提供计算资源，特别是在RSU覆盖不足的区域
- **动态覆盖补充**: 补充RSU静态部署的覆盖盲区，提供灵活的服务覆盖
- **应急响应节点**: 在高负载或紧急情况下提供快速资源调度
- **协同计算参与者**: 与车辆、RSU形成三层异构计算架构

### 1.2 工作模式

系统中UAV采用**固定悬停模式**(Fixed Hover Mode):

- **位置策略**: 在预定义位置悬停，不进行主动移动(velocity=0.0)
- **服务范围**: 通过固定覆盖半径(350米)提供服务
- **部署方式**: 根据网络拓扑需求静态部署在战略位置

## 二、UAV网络拓扑与部署

### 2.1 拓扑配置

| 参数 | 默认值 | 可配置范围 | 说明 |
|------|--------|-----------|------|
| UAV数量(num_uavs) | 2 | ≥0 | 系统部署的UAV节点总数 |
| 覆盖半径 | 350米 | - | 单个UAV的服务覆盖范围 |
| 飞行高度 | 120米 | - | UAV悬停的固定高度(z轴坐标) |
| 区域大小 | 2500m×2500m | - | 仿真区域总范围 |

### 2.2 空间部署策略

#### 2.2.1 默认双UAV部署方案

当系统配置num_uavs=2时(默认配置):

```
UAV部署位置(x, y, z):
- UAV_0: (300.0, 500.0, 120.0)
- UAV_1: (700.0, 500.0, 120.0)

部署逻辑:
- 沿道路中线(y=500)均匀分布
- 与路口位置对齐(x=300和x=700)
- 固定飞行高度120米
```

#### 2.2.2 动态多UAV部署算法

当num_uavs > 2时,系统采用自动均匀分布策略:

```
计算逻辑:
spacing = 600.0 / (num_uavs - 1)

for i in range(num_uavs):
    x_pos = 200.0 + i * spacing
    position = (x_pos, 500.0, 120.0)
    
覆盖优化:
- 保持沿主干道均匀间隔
- 最大化整体覆盖范围
- 避免覆盖重叠浪费
```

#### 2.2.3 与RSU的协同部署

UAV部署与RSU形成互补关系:

- RSU固定部署在路口和关键节点
- UAV填补RSU间的覆盖空白
- 共同形成无缝覆盖网络

## 三、UAV计算资源配置

### 3.1 计算能力参数

| 资源类型 | 配置值 | 单位 | 说明 |
|---------|--------|------|------|
| CPU总容量 | 8 GHz | GHz | 2个UAV共享的总计算资源池 |
| 单UAV初始分配 | 4.0 GHz | GHz | 均分后每个UAV的计算频率 |
| CPU频率范围 | 7-9 GHz | GHz | 可动态调整的频率范围 |
| 内存容量 | 4 GB | GB | 单个UAV的内存大小 |
| 缓存容量 | 4 GB | GB | 单个UAV的缓存空间 |

### 3.2 资源管理机制

#### 3.2.1 静态资源初始化

在系统启动时,每个UAV被分配:

```
初始化参数:
- cpu_freq: 4.0 GHz (固定均分)
- allocated_compute: 4.0 GHz (可调度资源)
- compute_usage: 0.0 (初始利用率)
- cache_capacity_bytes: 4e9 (4GB缓存)
- computation_queue: [] (空任务队列)
```

#### 3.2.2 动态资源调度(中央资源池模式)

在启用中央资源池模式时:

- **Phase 1 - 中央决策**: 智能体决定各UAV的资源分配比例
- **Phase 2 - 本地执行**: 各UAV根据分配资源执行任务调度
- **资源边界**: allocated_compute可在频率范围内动态调整

## 四、UAV动作空间设计

### 4.1 专用8维连续动作空间

UAV采用专门设计的动作空间(`algorithms/uav_action_space.py`),定义8个连续决策维度:

| 维度 | 动作类型 | 取值范围 | 映射参数范围 | 功能说明 |
|------|---------|---------|-------------|---------|
| 0 | 电池功率管理 | [-1,1] | [30%, 100%] | 控制计算功率级别,影响性能与能耗平衡 |
| 1 | 服务优先级权重 | [-1,1] | [10%, 100%] | 调整不同优先级任务的服务权重 |
| 2 | 覆盖区域调整 | [-1,1] | [0.5×, 1.5×] | 覆盖半径倍数调整(基准350米) |
| 3 | 应急响应敏感度 | [-1,1] | [0.2, 0.8] | 应急模式触发阈值 |
| 4 | 协调积极性 | [-1,1] | [0, 1.0] | 与其他节点(RSU/车辆)协调的权重 |
| 5 | 悬停效率优化 | [-1,1] | [0.6, 1.0] | 悬停稳定性因子 |
| 6 | 带宽分配策略 | [-1,1] | [30%, 100%] | 带宽分配的激进程度 |
| 7 | 缓存策略 | [-1,1] | [10%, 90%] | 主动缓存的积极性 |

### 4.2 动作映射机制

#### 4.2.1 动作解释流程

```
输入: 8维动作向量 action ∈ [-1,1]^8

步骤1 - 归一化映射:
normalized = (action + 1.0) / 2.0  → [0,1]

步骤2 - 参数范围映射:
parameter = min_val + normalized × (max_val - min_val)

输出: 具体执行参数字典
{
  'power_level': float ∈ [0.3, 1.0],
  'service_priority': float ∈ [0.1, 1.0],
  'coverage_radius': float ∈ [0.5, 1.5],
  ...
}
```

#### 4.2.2 动作分解器(UAVActionDecomposer)

动作分解器将抽象动作转换为5类具体执行参数:

**1. 电池管理参数(battery_management)**
```
- power_level: 计算功率级别 [0.3-1.0]
- energy_saving_mode: 节能模式开关 (power_level < 0.5时启用)
```

**2. 服务策略参数(service_strategy)**
```
- priority_weight: 优先级任务权重 [0.1-1.0]
- coverage_radius_multiplier: 覆盖半径倍数 [0.5-1.5]
- emergency_response_enabled: 应急响应启用 (threshold > 0.5时)
```

**3. 协调参数(coordination)**
```
- cooperation_weight: 节点协作权重 [0-1.0]
- bandwidth_allocation_ratio: 带宽分配比例 [0.3-1.0]
```

**4. 悬停控制参数(hover_control)**
```
- stability_factor: 稳定性因子 [0.6-1.0]
- position_adjustment_enabled: 位置微调开关 (stability < 0.8时启用)
```

**5. 缓存策略参数(cache_strategy)**
```
- aggressiveness: 缓存积极性 [0.1-0.9]
- proactive_caching: 主动预缓存开关 (aggressiveness > 0.5时启用)
```

### 4.3 预定义动作模式

系统提供3种预定义动作模板:

#### 4.3.1 默认动作(Default Action)

```
动作向量: [0, 0, 0, 0, 0, 0, 0, 0]

执行效果:
- 中性功率: 65% 功率级别
- 均衡优先级: 55% 权重
- 标准覆盖: 1.0× 半径
- 标准应急: 50% 阈值
- 适度协调: 50% 权重
- 平衡悬停: 80% 稳定性
- 中等带宽: 65% 分配
- 温和缓存: 50% 积极性
```

#### 4.3.2 应急动作(Emergency Action)

```
动作向量: [0.5, 1.0, 0.8, 1.0, 0.9, 0.9, 0.8, 0.6]

执行效果:
- 中等功率: 75% (平衡响应速度与续航)
- 最高优先级: 100% (优先处理紧急任务)
- 扩大覆盖: 1.4× 半径 (扩展服务范围)
- 最高应急: 100% 敏感度
- 积极协调: 95% (主动协同其他节点)
- 高稳定性: 95% (确保服务质量)
- 较多带宽: 90% (快速传输)
- 中等缓存: 68% (平衡缓存效率)

适用场景: 
- 系统高负载
- 紧急任务处理
- RSU过载时补充
```

#### 4.3.3 节能动作(Energy Saving Action)

```
动作向量: [-0.8, -0.2, -0.3, -0.5, 0.2, 0.8, -0.4, -0.3]

执行效果:
- 低功率: 36% (最小化能耗)
- 较低优先级: 42% (选择性服务)
- 缩小覆盖: 0.65× 半径 (减少服务负担)
- 标准应急: 25% 阈值
- 适度协调: 60% 
- 高稳定性: 90% (节能悬停)
- 保守带宽: 48% (降低通信功耗)
- 被动缓存: 28% (减少缓存更新)

适用场景:
- 电池电量低
- 系统低负载
- 延长续航需求
```

### 4.4 动作影响预测模型

系统提供动作影响评估机制,预测8类性能指标:

| 指标 | 计算公式 | 取值范围 | 说明 |
|------|---------|---------|------|
| battery_consumption_rate | 0.2 + 0.6×power_level | [0.2, 0.8] | 电池消耗速率 |
| service_quality | 0.3 + 0.7×service_priority | [0.3, 1.0] | 服务质量水平 |
| coverage_efficiency | 0.4 + 0.6×coverage_radius | [0.4, 1.0] | 覆盖效率 |
| emergency_readiness | emergency_threshold | [0.2, 0.8] | 应急准备度 |
| coordination_benefit | 0.2 + 0.8×coordination_weight | [0.2, 1.0] | 协调收益 |
| hover_stability | hover_stability | [0.6, 1.0] | 悬停稳定性 |
| bandwidth_efficiency | 0.5 + 0.5×bandwidth_ratio | [0.5, 1.0] | 带宽效率 |
| cache_hit_rate | 0.3 + 0.4×cache_aggressiveness | [0.3, 0.7] | 预期缓存命中率 |

## 五、UAV能耗模型

### 5.1 能耗组成结构

UAV总能耗由4个主要部分组成:

```
E_total = E_compute + E_comm + E_hover + E_movement
```

### 5.2 计算能耗(E_compute)

#### 5.2.1 能耗公式(论文式25-28)

```
E_compute = E_dynamic + E_static + E_memory

其中:
E_dynamic = κ₃ × f³ × t_active
E_static = P_static × t_active  
E_memory = P_dram × t_active × memory_ratio
```

#### 5.2.2 关键参数

| 参数 | 符号 | 数值 | 单位 | 说明 |
|------|------|------|------|------|
| 动态功耗系数 | κ₃ | 8.89e-31 | W/(Hz)³ | 功耗受限UAV芯片的CMOS系数 |
| 静态功率 | P_static | 2.5 | W | 轻量化设计的静态功耗 |
| DRAM功率 | P_dram | 2.0 | W | 低功耗内存设计 |
| 内存访问比例 | memory_ratio | 0.35 | - | 内存访问时间占比35% |
| 并行效率因子 | parallel_eff | 0.8 | - | 并行处理效率 |

#### 5.2.3 电池影响因子

```
考虑电池电量对性能的影响:

battery_factor = max(0.5, battery_level)
effective_frequency = cpu_frequency × battery_factor

说明:
- battery_level < 0.5时,性能降至50%
- 模拟电池电量下降对计算能力的约束
```

### 5.3 悬停能耗(E_hover)

#### 5.3.1 悬停功率模型

```
E_hover = P_hover × t_duration

参数:
- P_hover = 25.0 W (基于四旋翼UAV实测数据)
- t_duration: 悬停持续时间(秒)

数据来源:
基于DJI Phantom类似型号的实测悬停功耗
```

#### 5.3.2 时隙能耗计算

```
每个时隙(0.2秒)的悬停能耗:
E_slot = 25.0 W × 0.2 s = 5.0 J

100步仿真的总悬停能耗:
E_100_steps = 5.0 J × 100 × num_uavs
            = 1000 J (当num_uavs=2时)
```

### 5.4 移动能耗(E_movement)

```
E_movement = P_movement × t_movement

参数:
P_movement = P_hover × 1.8  (移动功率为悬停功率的1.8倍)
t_movement = distance / speed

说明:
- 1.8倍系数基于克服空气阻力和加速度的实验数据
- 默认移动速度: 10 m/s
- 当前固定悬停模式下,该项为0
```

### 5.5 通信能耗(E_comm)

#### 5.5.1 上行传输能耗

```
E_tx = P_tx × t_tx

参数:
- P_tx: 发射功率(基于3GPP标准)
- t_tx: 传输时长(基于数据大小和信道速率)
```

#### 5.5.2 下行接收能耗

```
E_rx = (P_rx + P_circuit) × t_rx

参数:
- P_rx = 2.8 W (UAV接收功率,3GPP TS 38.306标准)
- P_circuit: 电路功耗
- t_rx: 接收时长
```

### 5.6 电池管理

#### 5.6.1 电池配置

| 参数 | 数值 | 单位 | 说明 |
|------|------|------|------|
| 电池容量 | 5000 | J | 单个UAV的电池总容量 |
| 初始电量 | 1.0 | - | 初始电池电量水平(100%) |
| 最低电量阈值 | 0.2 | - | 低于20%时限制服务 |

#### 5.6.2 电量动态更新

```
每个时隙后更新:

battery_level_new = battery_level_old - (E_consumed / battery_capacity)

约束条件:
- battery_level ∈ [0, 1.0]
- 当battery_level < 0.2时,UAV不被考虑用于新任务卸载
```

## 六、UAV状态空间

### 6.1 本地状态(5维)

UAV的本地观测状态包含5个维度:

| 维度 | 状态变量 | 归一化方式 | 取值范围 | 说明 |
|------|---------|-----------|---------|------|
| 0 | position_x | x / 1000.0 | [0, 1] | UAV位置x坐标归一化 |
| 1 | position_y | y / 1000.0 | [0, 1] | UAV位置y坐标归一化 |
| 2 | position_z (altitude) | z / 200.0 | [0, 1] | UAV高度归一化(基准200米) |
| 3 | cache_utilization | used / capacity | [0, 1] | 缓存利用率 |
| 4 | energy_consumed | consumed / 1000.0 | [0, ∞) | 累计能耗归一化 |

### 6.2 扩展状态(智能体专用)

在中央资源分配模式下,UAV扩展状态包含:

```
UAV扩展状态(4维):
- load_rate: 计算负载率 compute_usage ∈ [0, 1]
- battery_level: 电池电量水平 ∈ [0, 1]
- queue_length: 任务队列长度 / 10.0 (归一化)
- available_resource: allocated_compute / 4e9 (归一化)
```

### 6.3 全局状态感知

UAV还能访问全局系统状态(16维):

```
全局状态组成:
- 系统级指标: 平均时延、总能耗、丢包率等(8维)
- 任务类型分布: 4类任务的队列分布(4维)
- 任务紧迫度: 4类任务的截止期裕度(4维)

说明:
全局状态用于集中式训练,帮助UAV理解整体系统负载
```

## 七、UAV通信机制

### 7.1 通信参数配置

| 参数类型 | 参数名 | 数值 | 单位 | 说明 |
|---------|--------|------|------|------|
| 发射功率 | uav_tx_power | 25 | dBm | 等效1W发射功率 |
| 接收功率 | uav_rx_power | 2.8 | W | 3GPP标准接收功率 |
| 天线增益 | antenna_gain_uav | 5.0 | dBi | UAV天线增益 |
| 覆盖半径 | coverage_radius | 350 | m | 服务覆盖范围 |
| 载波频率 | carrier_frequency | 3.5 | GHz | 3GPP NR n78频段 |

### 7.2 信道模型

#### 7.2.1 空对地(Air-to-Ground)传播模型

采用3GPP TR 38.901标准的UMi-Street Canyon场景模型:

```
视距(LoS)概率计算:
P_LoS = 1 / (1 + C × exp(-D × (θ - C)))

其中:
- θ: 仰角(elevation angle)
- C, D: 场景相关参数
- 距离: d_2d (水平距离) 和 d_3d (三维距离)
```

#### 7.2.2 路径损耗模型

```
LoS路径损耗:
PL_LoS = 32.4 + 20×log₁₀(f) + 20×log₁₀(d_3d)

NLoS路径损耗:
PL_NLoS = 32.4 + 20×log₁₀(f) + 30×log₁₀(d_3d)

阴影衰落:
- σ_LoS = 4.0 dB
- σ_NLoS = 7.82 dB
```

### 7.3 车辆-UAV连接判定

#### 7.3.1 覆盖检测算法

```
判定流程:
1. 计算车辆到UAV的三维距离:
   d_3d = √[(x_v - x_u)² + (y_v - y_u)² + z_u²]

2. 判断是否在覆盖范围内:
   is_covered = (d_3d ≤ coverage_radius)

3. 计算仰角:
   elevation = arctan(z_u / d_2d) × 180/π

4. 更新车辆状态:
   vehicle['in_uav_coverage'] = is_covered
   vehicle['distance_to_uav'] = d_3d
   vehicle['elevation_angle'] = elevation
```

#### 7.3.2 UAV选择机制

当车辆需要选择UAV进行卸载时:

```
选择流程:
1. 过滤可用UAV:
   - 在覆盖范围内
   - 电池电量 ≥ 0.2
   - 计算负载 < 过载阈值(0.8)

2. 评估候选UAV:
   - 计算传输时延
   - 计算处理时延
   - 计算能耗代价
   - 综合评分

3. 选择最优UAV:
   best_uav = argmin(total_cost)
```

## 八、UAV缓存机制

### 8.1 缓存配置

| 参数 | 数值 | 单位 | 说明 |
|------|------|------|------|
| 缓存容量 | 4 GB | Bytes | 单个UAV的缓存空间 |
| 缓存数据结构 | Dict | - | {content_id: content_data} |
| 缓存替换策略 | LRU/LFU | - | 可配置的替换算法 |

### 8.2 缓存决策维度

通过动作空间第7维(cache_strategy)控制:

```
缓存积极性映射:
action[7] ∈ [-1, 1] → aggressiveness ∈ [0.1, 0.9]

缓存模式:
- aggressiveness < 0.5: 被动缓存(仅缓存请求过的内容)
- aggressiveness ≥ 0.5: 主动预缓存(预测并缓存热点内容)
```

### 8.3 缓存协同机制

UAV与RSU的协同缓存策略:

```
协同逻辑:
1. UAV缓存轻量级、移动性强的内容
2. RSU缓存大容量、稳定的热点内容
3. 通过coordination_weight控制协同程度
4. 避免冗余缓存,优化整体命中率
```

## 九、UAV任务处理流程

### 9.1 任务接收

```
任务来源:
1. 车辆直接卸载
   - 车辆判断UAV在覆盖范围内
   - 车辆选择UAV作为卸载目标
   - 任务通过无线信道上传

2. RSU迁移
   - RSU过载时触发任务迁移
   - 迁移管理器选择UAV作为目标
   - 任务通过有线回传+无线信道迁移
```

### 9.2 任务处理

```
处理流程:

1. 任务入队:
   uav['computation_queue'].append(task)

2. 优先级排序:
   根据service_priority和任务优先级重新排序队列

3. 资源分配:
   allocated_freq = uav['allocated_compute']
   processing_time = task.compute_cycles / allocated_freq

4. 能耗计算:
   energy_info = calculate_uav_compute_energy(
       task, allocated_freq, processing_time, battery_level
   )

5. 任务执行:
   模拟处理时延和能耗消耗

6. 结果返回:
   通过下行链路返回结果给车辆
```

### 9.3 任务队列管理

```
队列管理策略:

1. 队列长度监控:
   queue_len = len(uav['computation_queue'])
   queue_utilization = queue_len / max_queue_capacity

2. 过载检测:
   is_overloaded = (compute_usage > 0.8) or (queue_len > threshold)

3. 任务拒绝/迁移:
   if is_overloaded:
       if emergency_response_enabled:
           trigger_task_migration()
       else:
           reject_new_tasks()

4. 队列优先级调度:
   根据service_priority动作调整任务执行顺序
```

## 十、UAV智能体训练

### 10.1 强化学习集成

#### 10.1.1 多智能体框架中的UAV

在MATD3/MADDPG等多智能体算法中:

```
智能体角色:
- 智能体ID: 'uav_agent'
- 智能体数量: num_uavs (默认2个UAV共享一个策略)
- 训练模式: 参数共享(Parameter Sharing)

状态空间:
- 本地观测: 5维UAV状态
- 全局观测: 16维系统状态(集中式训练)

动作空间:
- 8维连续动作(UAV专用动作空间)

奖励函数:
- 本地奖励: 基于UAV自身性能(能耗、时延、服务质量)
- 全局奖励: 系统整体性能(与车辆、RSU共享)
```

#### 10.1.2 单智能体框架中的UAV

在TD3/SAC等单智能体算法中:

```
动作集成:
全局动作向量包含UAV子动作:

action_dim = 3 + num_rsus + num_uavs + control_params

UAV动作提取:
idx = 3 + num_rsus
uav_selection = action[idx : idx+num_uavs]

说明:
- uav_selection: UAV选择权重 ∈ [0, 1]^num_uavs
- 通过softmax归一化得到选择概率
```

### 10.2 训练目标与奖励设计

#### 10.2.1 奖励函数组成

```
UAV相关奖励项:

R_total = w_delay × R_delay + w_energy × R_energy + w_qos × R_qos

其中:
R_delay = -avg_task_delay (UAV处理的任务平均时延)
R_energy = -uav_energy_consumed (UAV能耗)
R_qos = task_completion_rate (任务完成率)

权重配置:
w_delay = 0.15
w_energy = 0.7  (高权重,强调节能)
w_qos = 0.15
```

#### 10.2.2 特殊奖励机制

```
电池约束奖励:
if battery_level < 0.2:
    R_penalty = -10.0  (强制惩罚低电量服务)

协同奖励:
R_coop = coordination_weight × (system_performance_improvement)
说明: 鼓励UAV与其他节点协同

应急响应奖励:
if emergency_mode and task_urgent:
    R_bonus = +5.0  (奖励及时响应紧急任务)
```

### 10.3 探索策略

```
探索噪声配置:

noise_type: Ornstein-Uhlenbeck过程
initial_noise: 0.1  (低初始探索)
noise_decay: 0.9999 (缓慢衰减)
min_noise: 0.005    (保持最小探索)

说明:
较低的探索噪声适配UAV的连续精细控制需求
```

## 十一、UAV与其他节点的协同

### 11.1 UAV-车辆协同

#### 11.1.1 卸载决策协同

```
车辆卸载决策考虑UAV因素:

1. 距离因素:
   distance_score = 1.0 - (distance_to_uav / max_distance)

2. 负载因素:
   load_score = 1.0 - uav_compute_usage

3. 电量因素:
   battery_score = uav_battery_level

4. 综合评分:
   uav_score = w1×distance + w2×load + w3×battery

5. 与RSU/本地比较:
   best_option = argmax(local_score, rsu_score, uav_score)
```

#### 11.1.2 移动性协同

```
车辆移动对UAV服务的影响:

1. 预测车辆轨迹:
   predicted_position = current_pos + velocity × Δt

2. 覆盖持续时间估计:
   stay_time = estimate_coverage_duration(vehicle, uav)

3. 切换决策:
   if stay_time < task_duration:
       trigger_handover()  (切换到其他UAV或RSU)
```

### 11.2 UAV-RSU协同

#### 11.2.1 负载均衡协同

```
协同负载均衡机制:

1. RSU过载检测:
   if rsu_load > threshold:
       request_uav_assistance()

2. UAV接管任务:
   uav_takeover_tasks = select_tasks_from_rsu(
       criteria: distance, urgency, compute_requirement
   )

3. 反向卸载:
   if uav_battery_low and rsu_available:
       offload_tasks_to_rsu()
```

#### 11.2.2 缓存协同

```
协同缓存策略:

1. 内容分层:
   RSU缓存: 大文件、高频访问内容
   UAV缓存: 小文件、移动性强的内容

2. 缓存同步:
   if coordination_weight > 0.5:
       sync_popular_content(rsu, uav)

3. 缓存替换协调:
   避免RSU和UAV缓存相同内容
   通过中央调度器协调缓存决策
```

### 11.3 UAV-UAV协同

#### 11.3.1 覆盖协同

```
多UAV覆盖优化:

1. 覆盖重叠检测:
   overlap_area = calculate_coverage_overlap(uav_i, uav_j)

2. 动态调整覆盖:
   if overlap_area > threshold:
       adjust_coverage_radius(action[2])  (通过动作2调整)

3. 负载均衡:
   if load_imbalance > threshold:
       redistribute_tasks_between_uavs()
```

#### 11.3.2 资源协同

```
UAV间资源共享:

1. 能量协同:
   if uav_i.battery_low and uav_j.battery_high:
       transfer_tasks(uav_i → uav_j)

2. 计算协同:
   if uav_i.overloaded and uav_j.idle:
       offload_tasks(uav_i → uav_j)

说明:
通过coordination_weight(动作4)控制协同积极性
```

## 十二、UAV性能指标

### 12.1 关键性能指标(KPIs)

| 指标类别 | 指标名称 | 计算方式 | 目标 |
|---------|---------|---------|------|
| 服务性能 | UAV任务完成率 | 完成任务数/总任务数 | 最大化 |
| | UAV平均时延 | Σ(task_delay)/task_count | 最小化 |
| | UAV服务车辆数 | len(connected_vehicles) | - |
| 能耗指标 | UAV总能耗 | E_compute + E_hover + E_comm | 最小化 |
| | 单任务能耗 | E_total / task_count | 最小化 |
| | 电池消耗率 | ΔE / Δt | 监控 |
| 资源利用 | UAV计算利用率 | compute_usage | 优化 |
| | UAV缓存命中率 | cache_hits / total_requests | 最大化 |
| | UAV队列长度 | len(computation_queue) | 控制 |
| 协同效果 | 协同收益 | system_perf_with_coop - baseline | 最大化 |
| | 任务迁移成功率 | migration_success / migration_total | 最大化 |

### 12.2 UAV特有指标

```
1. 电池健康度:
   battery_health = current_capacity / initial_capacity

2. 悬停效率:
   hover_efficiency = useful_service_time / total_hover_time

3. 覆盖有效性:
   coverage_effectiveness = served_vehicles / vehicles_in_range

4. 应急响应率:
   emergency_response_rate = urgent_tasks_handled / urgent_tasks_total

5. 协调有效性:
   coordination_effectiveness = cooperative_gain / coordination_cost
```

### 12.3 性能监控

```
实时监控机制:

1. 每时隙更新:
   - compute_usage
   - battery_level
   - queue_length
   - energy_consumed

2. 周期性统计:
   - 每100步统计平均性能
   - 记录峰值负载
   - 分析能耗分布

3. 异常检测:
   - battery_level < 0.2 → 触发告警
   - compute_usage > 0.9 → 过载告警
   - queue_length > max_capacity → 队列溢出告警
```

## 十三、UAV机制总结

### 13.1 核心特性概览

```
UAV在VEC系统中的核心定位:

✓ 空中移动边缘计算节点
✓ 固定悬停服务模式
✓ 动态资源调度能力
✓ 智能协同决策能力
✓ 能耗感知服务策略
```

### 13.2 关键技术要点

```
1. 动作空间设计:
   - 8维专用连续动作空间
   - 覆盖功率、服务、覆盖、应急、协调、悬停、带宽、缓存
   - 灵活的参数映射机制

2. 能耗模型:
   - 计算能耗: κ₃×f³ 模型
   - 悬停能耗: 25W 实测数据
   - 电池约束: 20%最低阈值

3. 协同机制:
   - 与车辆的卸载协同
   - 与RSU的负载均衡
   - UAV间的资源共享

4. 智能决策:
   - 强化学习驱动的动作选择
   - 多目标优化(时延-能耗-QoS)
   - 自适应服务策略
```

### 13.3 系统优势

```
UAV机制的核心优势:

1. 灵活性:
   - 动态覆盖补充
   - 快速部署响应
   - 自适应策略调整

2. 协同性:
   - 多层次协同架构
   - 智能负载均衡
   - 资源优化共享

3. 智能性:
   - AI驱动决策
   - 自主学习优化
   - 预测性服务

4. 能效性:
   - 能耗感知控制
   - 节能模式支持
   - 电池管理优化
```

### 13.4 应用场景

```
UAV典型应用场景:

1. 覆盖补充:
   - RSU间隙覆盖
   - 临时热点区域
   - 偏远地区服务

2. 负载均衡:
   - RSU过载时分担
   - 高峰时段增援
   - 应急事件响应

3. 移动服务:
   - 跟随移动热点
   - 事件现场支持
   - 临时网络部署

4. 协同优化:
   - 多节点资源池化
   - 全局负载优化
   - 网络弹性增强
```

### 13.5 未来扩展方向

```
潜在增强方向:

1. 移动性增强:
   - 支持动态飞行轨迹优化
   - 实时位置调整决策
   - 预测性移动部署

2. 协同深化:
   - 更复杂的多UAV协同策略
   - 分层协同决策架构
   - 联邦学习集成

3. 能效优化:
   - 更精细的能耗预测模型
   - 动态节能策略
   - 能量收集集成(太阳能等)

4. 智能增强:
   - 深度强化学习算法
   - 迁移学习支持
   - 元学习快速适应
```
