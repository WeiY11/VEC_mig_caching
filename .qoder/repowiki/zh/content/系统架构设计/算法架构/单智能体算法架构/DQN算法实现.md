# DQN算法实现

<cite>
**本文档引用的文件**   
- [dqn.py](file://single_agent/dqn.py)
- [data_structures.py](file://models/data_structures.py)
- [standardized_reward.py](file://utils/standardized_reward.py)
- [system_config.py](file://config/system_config.py)
</cite>

## 目录
1. [引言](#引言)
2. [DQN算法核心组件](#dqn算法核心组件)
3. [任务卸载决策机制](#任务卸载决策机制)
4. [经验回放机制](#经验回放机制)
5. [目标网络与Q值更新](#目标网络与q值更新)
6. [ε-greedy探索策略](#ε-greedy探索策略)
7. [Q值函数设计与收敛特性](#q值函数设计与收敛特性)
8. [训练效率与维度灾难](#训练效率与维度灾难)
9. [训练监控与优化策略](#训练监控与优化策略)
10. [Double DQN与Dueling DQN扩展](#double-dqn与dueling-dqn扩展)

## 引言
本文档全面解析DQN（Deep Q-Network）算法在`single_agent/dqn.py`中的实现，重点介绍其在离散动作空间下的任务卸载决策机制。文档详细说明经验回放的存储结构与采样策略优化，目标网络的更新机制，以及ε-greedy探索策略的调度方式。结合`data_structures.py`中定义的任务类型与动作选项，阐述Q值函数的设计与收敛特性。分析DQN在低复杂度场景下的训练效率优势，及其在高维状态空间中面临的维度灾难问题。提供训练过程监控指标、学习率调整策略，并讨论Double DQN和Dueling DQN的潜在扩展方向。

## DQN算法核心组件

DQN算法实现包含四个核心组件：DQNConfig配置类、DQNNetwork神经网络、DQNReplayBuffer经验回放缓冲区和DQNAgent智能体。这些组件协同工作，实现强化学习中的Q-learning算法。

```mermaid
classDiagram
class DQNConfig {
+hidden_dim : int
+lr : float
+batch_size : int
+buffer_size : int
+target_update_freq : int
+gamma : float
+epsilon : float
+epsilon_decay : float
+min_epsilon : float
+update_freq : int
+warmup_steps : int
+double_dqn : bool
+dueling_dqn : bool
}
class DQNNetwork {
-state_dim : int
-action_dim : int
-dueling : bool
-feature_layers : Sequential
-value_stream : Sequential
-advantage_stream : Sequential
-q_network : Sequential
+forward(state : Tensor) : Tensor
+_init_weights() : void
}
class DQNReplayBuffer {
-capacity : int
-ptr : int
-size : int
-states : ndarray
-actions : ndarray
-rewards : ndarray
-next_states : ndarray
-dones : ndarray
+push(state, action, reward, next_state, done) : void
+sample(batch_size) : Tuple[Tensor, ...]
+__len__() : int
}
class DQNAgent {
-state_dim : int
-action_dim : int
-config : DQNConfig
-device : Device
-q_network : DQNNetwork
-target_q_network : DQNNetwork
-optimizer : Optimizer
-replay_buffer : DQNReplayBuffer
-epsilon : float
-step_count : int
-update_count : int
-losses : List[float]
-q_values : List[float]
+select_action(state, training) : int
+store_experience(state, action, reward, next_state, done) : void
+update() : Dict[str, float]
+_compute_loss(states, actions, rewards, next_states, dones) : Tensor
+hard_update(target, source) : void
+save_model(filepath) : void
+load_model(filepath) : void
}
DQNAgent --> DQNConfig : "使用"
DQNAgent --> DQNNetwork : "创建"
DQNAgent --> DQNReplayBuffer : "使用"
DQNNetwork --> DQNConfig : "参数"
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L32-L338)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L20-L338)

## 任务卸载决策机制

DQN算法在VEC系统中实现任务卸载决策，通过离散动作空间对车辆、RSU和UAV节点进行协同控制。动作空间设计为125个离散动作组合，对应5³的笛卡尔积空间，每个节点有5个可选动作。

```mermaid
graph TD
A[全局状态向量] --> B[DQNNetwork]
B --> C[Q值预测]
C --> D[ε-greedy策略]
D --> E[动作选择]
E --> F[动作分解]
F --> G[车辆动作]
F --> H[RSU动作]
F --> I[UAV动作]
G --> J[任务卸载决策]
H --> J
I --> J
J --> K[系统执行]
K --> L[奖励计算]
L --> M[经验存储]
M --> N[DQNReplayBuffer]
N --> O[网络更新]
O --> B
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L341-L481)
- [data_structures.py](file://models/data_structures.py#L12-L96)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L367-L383)
- [data_structures.py](file://models/data_structures.py#L12-L96)

## 经验回放机制

DQN实现中采用经验回放缓冲区（DQNReplayBuffer）来存储智能体与环境交互的经验。缓冲区采用循环队列设计，预分配内存以提高性能，支持高效的经验采样。

```mermaid
classDiagram
class DQNReplayBuffer {
-capacity : int
-ptr : int
-size : int
-states : ndarray
-actions : ndarray
-rewards : ndarray
-next_states : ndarray
-dones : ndarray
+push(state, action, reward, next_state, done) : void
+sample(batch_size) : Tuple[Tensor, ...]
+__len__() : int
}
class DQNAgent {
-replay_buffer : DQNReplayBuffer
+store_experience(state, action, reward, next_state, done) : void
+update() : Dict[str, float]
}
DQNAgent --> DQNReplayBuffer : "包含"
DQNReplayBuffer ..> DQNAgent : "提供经验"
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L134-L174)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L134-L174)

## 目标网络与Q值更新

DQN算法采用目标网络（target network）来稳定训练过程。目标网络定期从主网络进行硬更新，用于计算目标Q值，避免训练过程中的振荡。

```mermaid
sequenceDiagram
participant Agent as DQNAgent
participant QNet as Q Network
participant TargetNet as Target Network
participant Buffer as Replay Buffer
Agent->>Buffer : 采样经验批次
Buffer-->>Agent : states, actions, rewards, next_states, dones
Agent->>QNet : 计算当前Q值
QNet-->>Agent : current_q_values
Agent->>QNet : 预测下一状态动作
QNet-->>Agent : next_actions
Agent->>TargetNet : 计算目标Q值
TargetNet-->>Agent : next_q_values
Agent->>Agent : 计算目标Q值 target_q_values
Agent->>Agent : 计算损失 loss
Agent->>QNet : 反向传播更新
QNet-->>Agent : 更新后的网络
Agent->>Agent : 检查是否更新目标网络
alt 达到更新频率
Agent->>TargetNet : 硬更新
TargetNet-->>Agent : 更新完成
end
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L177-L338)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L313-L316)
- [dqn.py](file://single_agent/dqn.py#L289-L311)

## ε-greedy探索策略

DQN智能体采用ε-greedy探索策略，在训练过程中平衡探索与利用。初始探索率设置为1.0，随着训练进行按指数衰减，最终收敛到最小探索率0.05。

```mermaid
flowchart TD
Start([开始选择动作]) --> CheckTraining{"训练模式?"}
CheckTraining --> |是| CheckEpsilon{"随机数 < ε?"}
CheckTraining --> |否| Greedy[贪婪选择]
CheckEpsilon --> |是| Random[随机探索]
CheckEpsilon --> |否| Greedy
Random --> ReturnAction
Greedy --> ForwardPass["前向传播计算Q值"]
ForwardPass --> GetMax["获取最大Q值动作"]
GetMax --> ReturnAction
ReturnAction([返回动作]) --> End([结束])
UpdateEpsilon([更新探索率]) --> Decay["ε = ε * decay_rate"]
Decay --> MinCheck{"ε < min_epsilon?"}
MinCheck --> |是| SetMin["ε = min_epsilon"]
MinCheck --> |否| KeepCurrent["保持当前ε"]
SetMin --> EndUpdate
KeepCurrent --> EndUpdate
EndUpdate --> EndProcess([探索率更新完成])
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L218-L230)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L218-L230)

## Q值函数设计与收敛特性

Q值函数采用Dueling DQN架构设计，将Q值分解为状态价值函数V(s)和优势函数A(s,a)。这种设计有助于智能体更好地区分不同动作的相对优势，提高学习效率和收敛速度。

```mermaid
classDiagram
class DQNNetwork {
-feature_layers : Sequential
-value_stream : Sequential
-advantage_stream : Sequential
+forward(state) : Tensor
}
class FeatureExtractor {
+Linear(state_dim, hidden_dim)
+ReLU()
+Linear(hidden_dim, hidden_dim)
+ReLU()
}
class ValueStream {
+Linear(hidden_dim, hidden_dim//2)
+ReLU()
+Linear(hidden_dim//2, 1)
}
class AdvantageStream {
+Linear(hidden_dim, hidden_dim//2)
+ReLU()
+Linear(hidden_dim//2, action_dim)
}
DQNNetwork --> FeatureExtractor : "特征提取"
DQNNetwork --> ValueStream : "价值流"
DQNNetwork --> AdvantageStream : "优势流"
note right of DQNNetwork
Q(s,a) = V(s) + A(s,a) - mean(A(s, : ))
这种设计确保了Q值的可识别性
end note
```

**图示来源**
- [dqn.py](file://single_agent/dqn.py#L58-L131)

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L116-L131)

## 训练效率与维度灾难

DQN在低复杂度场景下表现出良好的训练效率，但在高维状态空间中面临维度灾难问题。算法通过经验回放和目标网络机制缓解这一问题，但仍受限于离散动作空间的指数增长。

| 场景复杂度 | 状态维度 | 动作维度 | 训练效率 | 维度灾难风险 |
|-----------|---------|---------|---------|------------|
| 低复杂度 | 20-40 | 25-64 | 高 | 低 |
| 中复杂度 | 40-80 | 64-256 | 中 | 中 |
| 高复杂度 | >80 | >256 | 低 | 高 |

当动作空间超过一定规模时，DQN的训练效率显著下降，因为需要探索的动作组合呈指数增长。在当前实现中，动作维度为125（5³），处于中等复杂度范围。

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L341-L345)
- [data_structures.py](file://models/data_structures.py#L12-L96)

## 训练监控与优化策略

DQN实现提供全面的训练监控指标，包括损失值、平均Q值、探索率和缓冲区大小等。这些指标有助于评估训练过程的稳定性和收敛性。

```mermaid
graph TD
A[训练步骤] --> B{缓冲区大小 >= 批次大小?}
B --> |否| C[返回空字典]
B --> |是| D{步数 >= 预热期?}
D --> |否| C
D --> |是| E{步数 % 更新频率 == 0?}
E --> |否| C
E --> |是| F[采样经验批次]
F --> G[计算损失]
G --> H[反向传播]
H --> I[更新网络]
I --> J{更新次数 % 目标网络更新频率 == 0?}
J --> |是| K[硬更新目标网络]
J --> |否| L[跳过]
K --> M[衰减探索率]
L --> M
M --> N[记录损失和Q值]
N --> O[返回训练信息]
```

学习率设置为1e-4，采用Adam优化器。通过梯度裁剪（clip_grad_norm_）防止梯度爆炸，确保训练稳定性。

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L237-L287)
- [dqn.py](file://single_agent/dqn.py#L32-L55)

## Double DQN与Dueling DQN扩展

DQN实现支持Double DQN和Dueling DQN两种重要扩展。Double DQN通过分离动作选择和Q值评估来减少过估计问题，Dueling DQN通过网络架构改进提高学习效率。

```mermaid
graph TD
A[标准DQN] --> B[目标Q值计算]
B --> C{"使用目标网络<br>max(Q(s',a'))"}
C --> D[过估计风险]
A --> E[Double DQN]
E --> F[动作选择]
F --> G{"使用主网络<br>argmax(Q(s',a'))"}
G --> H[Q值评估]
H --> I{"使用目标网络<br>Q(s',argmax(Q(s',a')))"}
I --> J[减少过估计]
A --> K[Dueling DQN]
K --> L[共享特征层]
L --> M[价值流 V(s)]
L --> N[优势流 A(s,a)]
M --> O[Q(s,a) = V(s) + A(s,a) - mean(A)]
N --> O
O --> P[更好区分动作优势]
```

在配置中，`double_dqn`和`dueling_dqn`标志位均默认启用，充分利用了这两种改进技术的优势。

**本节来源**
- [dqn.py](file://single_agent/dqn.py#L32-L55)
- [dqn.py](file://single_agent/dqn.py#L289-L311)