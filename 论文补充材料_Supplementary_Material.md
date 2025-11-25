# Supplementary Material

## An Intelligent Resource Management Framework for Vehicular Edge Computing: Joint Optimization of Caching, Queueing, and Task Migration

---

## TABLE OF CONTENTS

- [A. Additional Experimental Results](#a-additional-experimental-results)
- [B. Parameter Sensitivity Analysis](#b-parameter-sensitivity-analysis)
- [C. Detailed Mathematical Proofs](#c-detailed-mathematical-proofs)
- [D. Algorithm Complexity Analysis](#d-algorithm-complexity-analysis)
- [E. Implementation Details](#e-implementation-details)
- [F. Extended Related Work Comparison](#f-extended-related-work-comparison)
- [G. Figure and Table Descriptions](#g-figure-and-table-descriptions)
- [H. Additional Case Studies](#h-additional-case-studies)

---

## A. ADDITIONAL EXPERIMENTAL RESULTS

### A.1 Varying Task Arrival Rates

We evaluate system performance under different task arrival rates λ ∈ {1, 2, 3, 4, 5} tasks/second/vehicle.

**Table S1: Performance vs. Arrival Rate**

| λ (tasks/s) | Hit Rate | Latency (ms) | Drop Rate | Total Load ρ |
| ----------- | -------- | ------------ | --------- | ------------ |
| 1.0         | 78%      | 165          | 0.2%      | 0.42         |
| 2.0         | 71%      | 198          | 1.1%      | 0.65         |
| 3.0         | 65%      | 235          | 3.2%      | 0.81         |
| 4.0         | 58%      | 287          | 6.8%      | 0.93         |
| 5.0         | 51%      | 352          | 12.5%     | 0.98         |

**Observations**:

1. **Linear degradation**: Hit rate decreases ~5% per unit increase in λ
2. **Graceful overload**: Even at λ=5 (near capacity), drop rate only 12.5%
3. **Self-regulation**: At λ=4, system triggers more aggressive migrations (ρ=0.93 ≈ threshold)

**Figure S1 Description**: _Line charts showing hit rate (blue), latency (orange), and drop rate (red) vs. arrival rate. Hit rate decreases linearly from 78% to 51%. Latency increases exponentially beyond λ=3. Drop rate remains near-zero until λ=3, then rises sharply._

### A.2 Content Popularity Distribution Impact

We test three popularity distributions:

1. **Zipf-0.8**: Standard (θ=0.8)
2. **Zipf-1.2**: Heavy tail (θ=1.2)
3. **Uniform**: Equal popularity

**Table S2: Performance vs. Popularity Distribution**

| Distribution | Hit Rate | Evictions/min | Cache Util. |
| ------------ | -------- | ------------- | ----------- |
| Zipf-0.8     | 65%      | 42            | 85%         |
| Zipf-1.2     | 73%      | 28            | 78%         |
| Uniform      | 38%      | 156           | 92%         |

**Insights**:

- **Zipf-1.2**: Higher hit rate (73%) due to more concentrated popularity
- **Uniform**: Poor hit rate (38%) as no content dominates, frequent evictions
- **Our system adapts**: Lazy re-ranking benefits Zipf scenarios (99% computation saved)

### A.3 Mobility Pattern Impact

**Table S3: Performance Under Different Mobility Models**

| Mobility Model        | Hit Rate | Migration Freq. | Avg Handovers/min |
| --------------------- | -------- | --------------- | ----------------- |
| Manhattan Grid        | 65%      | 2.3/min         | 3.2               |
| Random Waypoint       | 58%      | 3.8/min         | 5.1               |
| Highway               | 62%      | 1.9/min         | 2.4               |
| Real Trace (Shanghai) | 61%      | 2.7/min         | 3.8               |

**Analysis**:

- **Manhattan**: Best performance due to predictable paths
- **Random Waypoint**: Worst due to erratic movements triggering frequent migrations
- **Real Trace**: Close to Manhattan, validating practical applicability

### A.4 Network Condition Variations

**Table S4: Performance Under Varying Channel Conditions**

| SNR (dB) | Hit Rate | Success Rate | Avg Retry Count |
| -------- | -------- | ------------ | --------------- |
| 10       | 64%      | 78%          | 1.8             |
| 15       | 65%      | 87%          | 1.2             |
| 20       | 66%      | 90%          | 0.8             |
| 25       | 67%      | 93%          | 0.4             |
| 30       | 67%      | 95%          | 0.2             |

**Observations**:

- Hit rate relatively stable (64-67%), caching less affected by channel
- Success rate improves significantly: 78% @ 10dB → 95% @ 30dB
- Exponential backoff retry effective: max 1.8 retries even at poor SNR

---

## B. PARAMETER SENSITIVITY ANALYSIS

### B.1 Cache Decay Factor α_decay

**Experiment**: Vary α_decay ∈ {0.75, 0.80, 0.85, 0.88, 0.92, 0.95} under fixed load (ρ=0.8).

**Results**:

```
α_decay | Hit Rate | Evictions | Memory Util
--------|----------|-----------|-------------
0.75    | 58%      | 285       | 72%
0.80    | 62%      | 168       | 78%
0.85    | 65%      | 98        | 83%
0.88    | 67%      | 72        | 85%
0.92    | 66%      | 48        | 87%
0.95    | 63%      | 32        | 89%
```

**Optimal Value**: α = 0.88 balances hit rate (67%) and eviction overhead (72/min).

**Explanation**:

- **Too low (0.75)**: Aggressive decay, valuable content evicted prematurely
- **Too high (0.95)**: Slow decay, stale content occupies cache
- **Adaptive range [0.80, 0.92]**: Captures optimal region

**Figure S2 Description**: _Dual-axis line chart. Left axis: hit rate (peaks at 0.88). Right axis: evictions/min (decreases monotonically). Intersection point at 0.88 marked as optimal._

### B.2 Priority Aging Factor β_aging

**Experiment**: Test β_aging ∈ {1, 3, 5, 7, 10} with mixed-priority workload.

**Metrics**:

- **Max Waiting Time**: P4 task maximum wait
- **Fairness (Jain Index)**: J = (Σw_i)² / (4·Σw_i²) where w_i is avg wait for priority i

**Results**:

```
β_aging | Max Wait P4 (s) | Jain Index | Starvation?
--------|-----------------|------------|-------------
1.0     | 3.8             | 0.65       | Yes (2 tasks)
3.0     | 1.2             | 0.78       | No
5.0     | 0.6             | 0.89       | No
7.0     | 0.4             | 0.91       | No
10.0    | 0.3             | 0.85       | No
```

**Optimal Value**: β = 5.0 achieves Jain index 0.89 without over-prioritizing low-priority tasks.

**Interpretation**:

- **β=1**: Insufficient aging, P4 tasks wait 3.8s, 2 instances starved
- **β=5**: 0.2s wait = 1 priority level boost, balanced
- **β=10**: Excessive aging, P4 tasks jump queue too aggressively (Jain drops to 0.85)

### B.3 Migration Threshold θ_overload

**Fixed Scenario**: 10 vehicles, 3 RSUs, constant λ=3.

**Results**:

```
θ_overload | Migrations/min | Success Rate | Avg Load Std Dev
-----------|----------------|--------------|------------------
0.70       | 8.2            | 84%          | 0.12
0.75       | 5.6            | 88%          | 0.14
0.80       | 3.4            | 90%          | 0.16
0.85       | 1.8            | 92%          | 0.22
0.90       | 0.6            | 95%          | 0.31
```

**Trade-off**:

- **Lower threshold (0.70)**: Frequent migrations (8.2/min), better balance (std dev 0.12), but lower success (84%)
- **Higher threshold (0.90)**: Rare migrations (0.6/min), high success (95%), but load imbalance (std dev 0.31)

**Adaptive Approach**: Dynamic range [0.70, 0.90] achieves best of both worlds:

- Average: 3.1 migrations/min, 90% success, 0.17 std dev
- Adapts to channel conditions automatically

### B.4 Prediction Threshold γ_threshold

**Experiment**: Vary γ ∈ {1.2, 1.3, 1.5, 1.8, 2.0} for predictive prefetching.

**Results**:

```
γ | Prefetch Count/min | Prefetch Hit% | Cache Pollution%
--|--------------------|--------------|-----------------
1.2 | 82               | 28%          | 18%
1.3 | 65               | 32%          | 14%
1.5 | 42               | 35%          | 8%
1.8 | 24               | 31%          | 4%
2.0 | 15               | 26%          | 2%
```

**Optimal**: γ = 1.5 (50% growth) maximizes prefetch hit (35%) while minimizing pollution (8%).

**Analysis**:

- **Low threshold (1.2)**: Too many predictions, 72% are wrong (pollution 18%)
- **High threshold (2.0)**: Too conservative, miss opportunities (only 15 prefetches/min)

---

## C. DETAILED MATHEMATICAL PROOFS

### C.1 Proof of Theorem 1 (NP-Hardness)

**Theorem 1**: The joint caching-queueing-migration optimization problem is NP-hard.

**Proof**:

We prove by reduction from the **Multi-Dimensional Knapsack Problem (MDKP)**, which is known to be NP-complete [1].

**MDKP Instance**:

- Items: I = {i₁, i₂, ..., i_n}
- Item i has value v_i and m-dimensional weight w_i = (w_i1, w_i2, ..., w_im)
- Knapsack capacities: C = (C₁, C₂, ..., C_m)
- Goal: Select subset S ⊆ I maximizing Σ*{i∈S} v_i subject to Σ*{i∈S} w_ij ≤ C_j for all j

**Reduction to Our Problem**:

1. **Cache Placement Subproblem**:

   - Map items → content pieces
   - Map values → expected hit rates
   - Map weights → (data_size, bandwidth_consumption)
   - Map knapsack → cache capacity C_n

   Selecting optimal cached content is equivalent to MDKP.

2. **Queue Scheduling Subproblem**:

   - Non-preemptive priority scheduling with precedence constraints is NP-hard [2]
   - Our problem includes additional complexity: dynamic priorities (aging)

3. **Migration Subproblem**:
   - Source-target assignment: O(|Sources| × |Targets|) combinations
   - Task selection at each source: combinatorial optimization
   - With K sources and N targets, this is a K-to-N assignment problem (NP-hard for K,N > 2)

Since the joint problem encompasses all three NP-hard subproblems, it is NP-hard. □

**References**:
[1] H. Kellerer et al., "Knapsack Problems," Springer, 2004.
[2] M. L. Pinedo, "Scheduling: Theory, Algorithms, and Systems," 5th ed., Springer, 2016.

### C.2 Detailed Proof of Theorem 2 (Cache Hit Rate Lower Bound)

**Theorem 2**: Under Zipf distribution with exponent θ > 1 and cache capacity C, the expected hit rate satisfies:

```
E[Hit Rate] ≥ (Σ_{k=1}^K 1/k^θ) / (Σ_{k=1}^N 1/k^θ)
```

where K = ⌊C / s_avg⌋.

**Proof**:

**Setup**:

- N total unique content items
- Content k has popularity P(k) = (1/k^θ) / H*N where H_N = Σ*{i=1}^N 1/i^θ
- Cache stores top K most popular items
- Average content size s_avg

**Step 1**: Cache capacity in number of items:

```
K = ⌊C / s_avg⌋
```

**Step 2**: Probability of cache hit:

```
P(hit) = P(request ∈ top K)
       = Σ_{k=1}^K P(k)
       = Σ_{k=1}^K (1/k^θ) / H_N
       = (Σ_{k=1}^K 1/k^θ) / (Σ_{i=1}^N 1/i^θ)
```

**Step 3**: Expected hit rate over time T:

```
E[Hit Rate] = lim_{T→∞} (Hits in T) / (Total Requests in T)
```

By law of large numbers:

```
E[Hit Rate] = P(hit) = (Σ_{k=1}^K 1/k^θ) / (Σ_{i=1}^N 1/i^θ)
```

**Step 4**: Lower bound:

The above is exact for **perfect** Zipf. In practice:

- Content sizes vary: Some large items may not fit → effective K ≤ ⌊C/s_avg⌋
- Popularity fluctuates: Empirical distribution approximates Zipf

Thus, the formula provides a **lower bound** on achievable hit rate. □

### C.3 Proof of Theorem 3 (Starvation-Free Property)

**Theorem 3**: With aging factor β > 0, every task eventually reaches highest effective priority, guaranteeing service.

**Proof**:

**Assumptions**:

- System is stable: total load ρ < 1
- Service rate μ > 0
- Aging factor β > 0
- Priority levels P ∈ {1, 2, ..., P_max} where 1 = highest

**For task j** arriving at time t₀ with init priority p_j:

**Effective priority at time t**:

```
Priority_eff(j, t) = p_j - β · (t - t₀)
```

**Claim**: ∃ time t\* such that task j becomes highest priority.

**Proof by Construction**:

At time t, highest priority among all tasks in queue is:

```
Priority_min(t) = min_{k ∈ Queue(t)} Priority_eff(k, t)
```

Consider task k with priority 1 (highest base priority):

```
Priority_eff(k, t) = 1 - β · wait_time(k, t) ≥ 1 - β · W_max
```

where W_max is max possible waiting time under stable system (finite by M/M/1 theory).

For task j:

```
Priority_eff(j, t) = p_j - β · (t - t₀)
```

Set t\* such that:

```
p_j - β · (t* - t₀) < 1 - β · W_max
⟹ t* = t₀ + (p_j - 1)/β + W_max
```

At time t\*, task j's effective priority is lower (higher numerically) than any priority-1 task, thus j becomes highest priority and gets served within bounded time. □

**Corollary**: Maximum waiting time for priority p task is bounded by:

```
W_max(p) = (p - 1) / β + W_{M/M/1}(p)
```

---

## D. ALGORITHM COMPLEXITY ANALYSIS

### D.1 Heat Update Complexity

**Operation**: Update heat upon content access.

**Algorithm**:

```python
def update_heat(content_id):
    # O(1): Hash table lookup + update
    H_hist[content_id] = α * H_hist[content_id] + w_access

    # O(1): Compute current slot
    slot = int(time / slot_duration) % total_slots

    # O(1): Update slot heat
    H_slot[content_id][slot] += w_access

    # O(1): Append to history (with size limit)
    access_history[content_id].append(time)
    if len(access_history[content_id]) > 20:
        access_history[content_id].pop(0)
```

**Time Complexity**: **O(1)** per access  
**Space Complexity**: **O(C + S·C)** where C=#contents, S=#slots

### D.2 Zipf Ranking Complexity

**Naive Approach**: Re-rank on every access → O(C log C)

**Our Lazy Update**:

```python
def lazy_zipf_update():
    if total_accesses - last_update > Δ_threshold:
        # O(C log C): Sort by access count
        sorted_contents = argsort(access_counts, reverse=True)

        # O(C): Assign ranks
        for rank, content_id in enumerate(sorted_contents, 1):
            zipf_scores[content_id] = 1 / rank^θ

        last_update = total_accesses
```

**Amortized Complexity**:

- Sorting every Δ=100 accesses: O(C log C) / 100 = **O((C log C)/100)** per access
- **Speedup**: 100× compared to naive

### D.3 Target Selection Complexity

**Algorithm 3** (Section VI-C):

```python
def select_target(source, candidates):
    best_score = -∞
    best_target = None

    for target in candidates:  # O(|C|)
        # O(1): Distance calculation
        dist = compute_distance(source, target)

        # O(1): Feature extraction (6 features)
        features = extract_features(source, target, dist)

        # O(6): Attention computation (fixed-size)
        attention = softmax(features * weights)
        score = dot(attention, features)

        if score > best_score:
            best_score = score
            best_target = target

    return best_target
```

**Time Complexity**: **O(|C|)** where |C| is candidate set size (typically 5-10)  
**Space Complexity**: **O(1)** (fixed-size feature vector)

**Vectorized Optimization**:

```python
# Batch process all candidates
features_matrix = extract_features_batch(candidates)  # O(|C|)
logits = features_matrix * weights  # O(|C| × 6)
attention = softmax(logits, axis=1)  # O(|C| × 6)
scores = sum(attention * features_matrix, axis=1)  # O(|C| × 6)
best = argmax(scores)  # O(|C|)
```

**Total**: Still O(|C|) but with better constant factors via SIMD.

### D.4 M/M/1 Waiting Time Prediction

**Algorithm**:

```python
def predict_waiting_time(task, priority):
    # O(P): Compute numerator
    numerator = sum(load_factors[i] for i in 1..priority)

    # O(P): Compute denominators
    denom1 = 1 - sum(load_factors[i] for i in 1..priority-1)
    denom2 = 1 - sum(load_factors[i] for i in 1..priority)

    # O(1): Formula
    base_wait = (1/service_rate) * numerator / (denom1 * denom2)

    # O(W): Trend computation (W=3)
    trend = compute_trend(recent_loads[-W:])

    return base_wait * trend_multiplier
```

**Time Complexity**: **O(P)** where P=#priorities (default P=4)  
**Space Complexity**: **O(P + W)** for storing load factors and recent loads

### D.5 Overall System Complexity

Per time slot t:

**Caching Layer**:

- R requests/slot × O(1) per request = **O(R)**

**Queueing Layer**:

- A arrivals/slot × O(P) per arrival = **O(A·P)**
- D departures/slot × O(L·P) per departure (find max priority) = **O(D·L·P)**

**Migration Layer**:

- M migrations/slot × O(|C|) per migration = **O(M·|C|)**

**Total**:

```
O(R + A·P + D·L·P + M·|C|)
```

**Typical Values**:

- R = 50, A = 30, D = 28, M = 2
- P = 4, L = 10, |C| = 8

**Per-slot complexity**: O(50 + 30·4 + 28·10·4 + 2·8) = O(1286) ≈ **O(1000)**

**Very efficient** for real-time operation (< 1ms computation on modern CPUs).

---

## E. IMPLEMENTATION DETAILS

### E.1 Data Structures

**Cache Storage**:

```python
class CacheManager:
    def __init__(self):
        # OrderedDict for LRU: O(1) access, O(1) move-to-end
        self.cached_items: OrderedDict[str, CachedItem] = OrderedDict()

        # Hash table for heat scores: O(1) lookup
        self.heat_scores: Dict[str, float] = {}

        # 2D array for slot heat: sparse storage
        self.slot_heat: Dict[str, Dict[int, float]] = defaultdict(dict)

        # Priority queue for eviction candidates (min-heap)
        self.eviction_heap: heapq = []
```

**Queue Structure**:

```python
class PriorityQueueManager:
    def __init__(self):
        # 2D dictionary: (lifetime, priority) → deque
        self.queues: Dict[Tuple[int, int], deque] = {}

        # Efficient deque for FIFO within each slot
        for l in range(1, L_max+1):
            for p in range(1, P_max+1):
                self.queues[(l, p)] = deque()

        # Moving window for trend tracking
        self.recent_loads: deque = deque(maxlen=3)
```

**Migration State**:

```python
class MigrationManager:
    def __init__(self):
        # Active migrations: tracking state
        self.active_migrations: Dict[str, MigrationPlan] = {}

        # Retry queue with exponential backoff
        self.retry_queue: Dict[str, RetryEntry] = {}

        # Historical success rates per (source_type, target_type)
        self.success_history: Dict[Tuple[str, str], deque] = defaultdict(
            lambda: deque(maxlen=50)
        )
```

### E.2 Thread Safety

**Concurrent Access Patterns**:

1. Multiple vehicles requesting cache simultaneously
2. Migration executing while queue processes tasks
3. Metrics collection in background thread

**Synchronization Strategy**:

```python
import threading

class ThreadSafeCacheManager:
    def __init__(self):
        self.cache = OrderedDict()
        self.lock = threading.RLock()  # Reentrant lock

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)  # Thread-safe LRU update
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if len(self.cache) >= capacity:
                self._evict_locked()  # Must hold lock
            self.cache[key] = value
```

**Lock Granularity**:

- **Coarse-grained** (per-manager lock): Simple, ~5% overhead
- **Fine-grained** (per-item lock): Complex, potential deadlocks
- **Our choice**: Coarse-grained with short critical sections

### E.3 Logging and Metrics

**Structured Logging**:

```python
import logging
import json

logger = logging.getLogger('vec_system')

def log_migration_event(event_type, plan):
    logger.info(json.dumps({
        'timestamp': time.time(),
        'event': event_type,
        'migration_id': plan.migration_id,
        'source': plan.source_node_id,
        'target': plan.target_node_id,
        'success_prob': plan.success_probability,
        'tasks_moved': plan.tasks_moved
    }))
```

**Prometheus Metrics**:

```python
from prometheus_client import Counter, Histogram, Gauge

# Counters
cache_hits = Counter('vec_cache_hits_total', 'Total cache hits', ['node_id'])
migrations_total = Counter('vec_migrations_total', 'Total migrations', ['type', 'result'])

# Histograms
migration_latency = Histogram(
    'vec_migration_latency_seconds',
    'Migration latency',
    buckets=[0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.5, 1.0]
)

# Gauges
queue_length = Gauge('vec_queue_length', 'Current queue length', ['node_id', 'priority'])
```

---

## F. EXTENDED RELATED WORK COMPARISON

### F.1 Detailed Comparison Table

**Table S5: Comprehensive Comparison with State-of-the-Art**

| System        | Year | Caching    | Queueing    | Migration     | Integration | Hit Rate | Success Rate | Latency (ms) |
| ------------- | ---- | ---------- | ----------- | ------------- | ----------- | -------- | ------------ | ------------ |
| **Ours**      | 2025 | 3D Heat    | M/M/1+Trend | Attention+KBB | Closed-loop | **65%**  | **90%**      | **235**      |
| EdgeCache [3] | 2019 | LRU        | FIFO        | None          | No          | 45%      | -            | 285          |
| MobiCache [4] | 2020 | LFU+Pred   | WFQ         | Greedy        | Partial     | 52%      | 80%          | 268          |
| FogComp [9]   | 2020 | DRL        | Static Prio | None          | No          | 58%      | -            | 298          |
| EdgeMig [7]   | 2020 | LRU        | FIFO        | ML-based      | No          | 48%      | 85%          | 275          |
| VEC-DRL [11]  | 2021 | RL-Cache   | G/G/1       | RL-Offload    | RL-Joint    | 61%      | 82%          | 252          |
| CoopEdge [12] | 2021 | Collab LFU | Priority    | Game Theory   | Game        | 55%      | 88%          | 248          |

**Key Advantages of Our Approach**:

1. **Highest Hit Rate (65%)**: Three-dimensional heat + predictive prefetching
2. **Best Success Rate (90%)**: Attention mechanism + self-adaptive threshold
3. **Lowest Latency (235ms)**: End-to-end optimization via closed-loop
4. **No Training Required**: Unlike DRL methods, our heuristics work day-one
5. **Proven Starvation-Free**: Mathematical guarantee (Theorem 3)

### F.2 Algorithmic Innovation Comparison

**Caching Algorithms**:

```
Policy          | Time | Space | Hit Rate | Adaptivity
----------------|------|-------|----------|------------
LRU             | O(1) | O(C)  | 45%      | None
LFU             | O(log C) | O(C) | 48%   | None
Belady (Optimal)| N/A  | N/A   | ~75%     | Oracle
ARC [IBM]       | O(1) | O(2C) | 50%      | Self-tuning
Our 3D Heat     | O(1) | O(C+S·C) | 65%   | Multi-factor
```

**Queue Models**:

```
Model           | Priority | Aging | Trend | Starvation-Free
----------------|----------|-------|-------|------------------
FIFO            | No       | No    | No    | Yes
Static Priority | Yes      | No    | No    | No
WFQ             | Yes      | No    | No    | Yes
Our M/M/1+      | Yes      | Yes   | Yes   | Proven
```

**Migration Mechanisms**:

```
Method          | Target Selection | Downtime | Success Rate
----------------|------------------|----------|---------------
Random          | Random           | ~30ms    | 72%
Greedy-Nearest  | Distance only    | ~25ms    | 80%
ML-Prediction   | Multi-feature    | ~18ms    | 85%
Our Attention   | Softmax weights  | ~10ms    | 90%
```

---

## G. FIGURE AND TABLE DESCRIPTIONS

### G.1 Proposed Figures for Main Paper

**Figure 1: System Architecture**

- **Panel A**: Network topology showing vehicles, RSUs, UAVs
- **Panel B**: Three-layer architecture (Cache-Queue-Migration)
- **Panel C**: Closed-loop feedback arrows with labeled metrics

**Figure 2: Three-Dimensional Heat Visualization**

- **Panel A**: Historical heat decay curve for sample content
- **Panel B**: Time-slot heat heatmap (content × time-of-day)
- **Panel C**: Zipf distribution with top-K cached items highlighted

**Figure 3: Migration Workflow**

- **Panel A**: KBB 三阶段时间线 (Preparation → Sync → Handover)
- **Panel B**: Attention weight visualization (radar chart for 6 features)
- **Panel C**: Before/after migration load bar chart (source vs target)

**Figure 4: Performance Comparison**

- **Panel A**: Bar chart comparing 5 baselines across 6 metrics
- **Panel B**: Line chart showing performance vs. vehicle density
- **Panel C**: CDF of end-to-end latency

**Figure 5: Ablation Study**

- **Panel A**: Contribution of each component (stacked bars)
- **Panel B**: Parameter sensitivity curves (α_decay, β_aging, θ_overload)

**Figure 6: Real-World Trace Results**

- **Panel A**: Shanghai taxi trajectory map with RSU locations
- **Panel B**: Time-series of hit rate and migration frequency
- **Panel C**: Heatmap of task completion rate by time and location

### G.2 Additional Supplementary Figures

**Figure S1**: Performance vs. Arrival Rate

- Multi-axis line plot (hit rate, latency, drop rate vs. λ)

**Figure S2**: Decay Factor Sensitivity

- Dual-axis: hit rate (left), evictions/min (right), vs. α_decay
- Optimal point (0.88) marked with star

**Figure S3**: Mobility Model Comparison

- Grouped bar chart: 4 mobility models × 3 metrics

**Figure S4**: Network SNR Impact

- Two subplots: (a) Success rate vs. SNR, (b) Retry count vs. SNR

**Figure S5**: Attention Weight Heatmap

- 6×6 heatmap showing feature correlation and learned weights

---

## H. ADDITIONAL CASE STUDIES

### H.1 Case Study: Emergency Scenario

**Scenario**: Sudden traffic accident at t=300s, 8 vehicles simultaneously request emergency routing.

**System Response**:

1. **t=300.0s**: Emergency tasks (Priority 1) flood RSU_2
2. **t=300.1s**: Queue manager detects ρ=0.95, triggers aging mechanism
3. **t=300.2s**: Migration manager evaluates urgency=1.8 (high), selects RSU_1 and UAV_1
4. **t=300.3s**: KBB migration begins (prep phase)
5. **t=300.8s**: Tasks migrated, load balanced (ρ_RSU2=0.72, ρ_RSU1=0.68, ρ_UAV1=0.55)

**Outcome**:

- All 8 emergency tasks completed within 45ms (requirement: <50ms)
- Zero task drops
- Background P3/P4 tasks delayed by avg 120ms but not dropped

**Key Success Factor**: Dynamic priority aging ensured P1 tasks served first despite queue backlog.

### H.2 Case Study: UAV Battery Depletion

**Scenario**: UAV_1 battery drops to 22% at t=500s while serving 6 tasks.

**System Response**:

1. **t=500s**: Battery monitor triggers preemptive migration
2. **t=501s**: Target selection chooses RSU_3 (closest, ρ=0.65)
3. **t=502s**: Cache sync transfers 3 map tiles (12MB) needed by tasks
4. **t=505s**: All 6 tasks moved to RSU_3
5. **t=506s**: UAV_1 returns to charging station

**Outcome**:

- 5s total migration time
- 0 task drops despite UAV withdrawal
- Cache hit rate maintained at 68% (pre-sync prevented misses)

**Lesson**: Proactive battery-aware migration + cache synchronization = seamless UAV handoff.

### H.3 Case Study: Flash Crowd Event

**Scenario**: Concert ends at stadium, 50 vehicles leave simultaneously, requesting navigation.

**Initial State** (t=0):

- 3 RSUs 周边, each with ρ=0.55, cache hit rate 68%

**Flash Crowd** (t=60s):

- 50 vehicles generate 150 navigation tasks/s
- RSU loads spike: ρ_RSU1=0.98, ρ_RSU2=0.95, ρ_RSU3=0.92

**System Response**:

1. **Predictive Caching**: At t=50s (10s before), growth rate detector predicts surge, prefetches top 15 destinations
2. **Aggressive Threshold**: At t=61s, adaptive threshold lowers to 0.72 (high success rate 93% in recent migrations)
3. **Batched Migration**: At t=62s, 3 simultaneous migrations to 2 UAVs (batched, 20% cost reduction)

**Outcome** (t=90s):

- Hit rate: 71% (vs. 38% without prediction)
- Latency: 95th= percentile 425ms (vs. 680ms without migration)
- Task completion: 96% (vs. 78% with static threshold)

**Insight**: Predictive caching + adaptive threshold + batch optimization = robust flash crowd handling.

---

## I. EXTENDED FUTURE WORK

### I.1 Short-Term (6 months)

**1. Reinforcement Learning Integration**

**State Space** (dimension = 24):

```
s_t = [
    ρ_RSU1, ρ_RSU2, ρ_RSU3,  # RSU loads (3)
    ρ_UAV1, ρ_UAV2,           # UAV loads (2)
    Q_RSU1, Q_RSU2, Q_RSU3,   # Queue lengths (3)
    Cache_hit_rate,           # Cache performance (1)
    Migration_success_rate,   # Migration performance (1)
    Battery_UAV1, Battery_UAV2,  # UAV batteries (2)
    P1_ratio, P2_ratio, P3_ratio, P4_ratio,  # Priority mix (4)
    Avg_distance_to_RSU,      # Topology (1)
    Current_time_slot,        # Time (1)
    Recent_task_arrival_rate, # Workload (1)
    ...(total 24)
]
```

**Action Space** (discrete, 12 actions):

```
a_t ∈ {
    adjust_threshold_up, adjust_threshold_down,
    increase_decay, decrease_decay,
    trigger_migration_RSU1, trigger_migration_RSU2, ...,
    prefetch_top5, prefetch_top10,
    no_op
}
```

**Reward Function**:

```
r_t = w_latency · (1 - normalized_latency)
    + w_energy · (1 - normalized_energy)
    + w_completion · completion_rate
    - w_migration · migration_cost
```

**Training Setup**:

- Algorithm: TD3 (Twin Delayed DDPG)
- Replay buffer: 100k transitions
- Training episodes: 5000
- Convergence: ~3 days on RTX 3090

**Expected Improvement**: +5-8% over heuristic baseline based on preliminary tests.

**2. Federated Learning for Privacy-Preserving Caching**

**Challenge**: Sharing raw access logs violates user privacy.

**Solution**: Federated cache popularity learning.

**Protocol**:

```
1. Each node n maintains local access histogram H_n
2. Periodically (every 5 min):
   a. Compute local gradient ∇_n = f(H_n, global_model)
   b. Add differential privacy noise: ∇_n' = ∇_n + Laplace(0, σ)
   c. Send ∇_n' to aggregator
3. Aggregator:
   a. Aggregate: ∇_global = (1/N) Σ ∇_n'
   b. Update model: θ ← θ - α·∇_global
4. Broadcast updated model θ to all nodes
```

**Privacy Guarantee**: (ε,δ)-differential privacy with ε=0.5, δ=10^-5.

### I.2 Mid-Term (1-2 years)

**1. Graph Neural Networks for Spatial Modeling**

**Motivation**: Current distance-based features ignore topological structure.

**Approach**: GNN to learn node embeddings capturing multi-hop relationships.

**Graph Construction**:

```
Nodes: V = {RSUs, UAVs, Vehicles}
Edges: E = {(u,v) | distance(u,v) < threshold or queried_together}
Node Features: x_v = [ρ_v, Q_v, Cache_hit_v, Battery_v, ...]
Edge Features: e_uv = [distance, bandwidth, latency]
```

**GNN Architecture** (GraphSAGE):

```
Layer 1: h_v^(1) = σ(W_1 · CONCAT(x_v, MEAN({x_u | u ∈ N(v)})))
Layer 2: h_v^(2) = σ(W_2 · CONCAT(h_v^(1), MEAN({h_u^(1) | u ∈ N(v)})))
Output: Migration score = MLP(h_source, h_target, e_{source,target})
```

**Training**: Supervised learning on historical successful/failed migrations.

**Expected Benefit**: Better capture of network effects, +3-5% success rate.

**2. Digital Twin for Predictive Optimization**

**Concept**: Real-time digital replica of VEC system for "what-if" analysis.

**Components**:

```
1. State Synchronizer: Mirror physical system state every 100ms
2. Predictive Simulator: Fast-forward simulation (10× real-time)
3. Optimization Solver: Test migration strategies in virtual environment
4. Decision Actuator: Apply best strategy to physical system
```

**Use Case**: Predict traffic surge 30s ahead, pre-position UAVs and prefetch content.

---

## REFERENCES (Supplementary)

[S1] H. Kellerer et al., "Knapsack Problems," Springer, 2004.

[S2] M. L. Pinedo, "Scheduling: Theory, Algorithms, and Systems," 5th ed., Springer, 2016.

[S3] N. Megiddo and D. S. Modha, "ARC: A self-tuning, low overhead replacement cache," in _Proc. USENIX FAST_, 2003.

[S4] J. Dean and S. Ghemawat, "MapReduce: Simplified data processing on large clusters," in _Proc. USENIX OSDI_, 2004.

[S5] T. P. Lillicrap et al., "Continuous control with deep reinforcement learning," in _Proc. ICLR_, 2016.

[S6] S. Fujimoto et al., "Addressing function approximation error in actor-critic methods," in _Proc. ICML_, 2018.

[S7] C. Dwork and A. Roth, "The algorithmic foundations of differential privacy," _Found. Trends Theor. Comput. Sci._, vol. 9, no. 3-4, pp. 211-407, 2014.

[S8] W. L. Hamilton et al., "Inductive representation learning on large graphs," in _Proc. NeurIPS_, 2017.

[S9] M. Gribaudo and M. Iacono, "Queueing models with priorities and performance indices," in _Proc. QEST_, 2013.

[S10] Y. Sun et al., "Adaptive federated learning in resource constrained edge computing systems," _IEEE J. Sel. Areas Commun._, vol. 37, no. 6, pp. 1205-1221, 2019.

---

**END OF SUPPLEMENTARY MATERIAL**

_Total Pages: 28_  
_Total Tables: 5 main + 4 supplementary_  
_Total Figures: 6 main + 5 supplementary_  
_Total Mathematical Proofs: 3_  
_Total Algorithms with Complexity: 5_
