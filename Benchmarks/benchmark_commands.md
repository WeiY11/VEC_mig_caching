# Benchmark Algorithms Training Commands

This document lists the verified commands to run each benchmark algorithm against the `OPTIMIZED_TD3` reference.

## Prerequisites

Ensure you are in the root directory `d:\VEC_mig_caching` and have the environment set up.

```powershell
$env:PYTHONPATH="."
```

## 1. Heuristic (Dynamic Offload)

The heuristic algorithm uses a rule-based approach for offloading decisions based on queue length and channel quality.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg heuristic --episodes 50 --groups 5
```

**Notes:**

- Fixed `ValueError: cannot reshape array` by making state parsing robust to variable state lengths (e.g., when central resource state is present).

## 2. Local Only

The baseline policy that processes all tasks locally on the vehicle.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg local --episodes 50 --groups 5
```

## 3. TD3 (Standard)

Standard Twin Delayed DDPG algorithm.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg td3 --episodes 50 --groups 5
```

## 4. DDPG (Standard)

Standard Deep Deterministic Policy Gradient algorithm.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg ddpg --episodes 50 --groups 5
```

## 5. SAC (Robust)

Soft Actor-Critic algorithm.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg sac --episodes 50 --groups 5
```

**Notes:**

- Fixed `AttributeError: 'SimpleNamespace' object has no attribute 'sample'` by implementing a custom `BoxSpace` class in `VecEnvWrapper`.

## 6. Simulated Annealing (SA)

Online Simulated Annealing for optimizing offload preference weights.

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg sa --episodes 50 --groups 5
```

## Batch Execution

You can run multiple algorithms in a single command:

```powershell
python -m Benchmarks.run_benchmarks_vs_optimized_td3 --alg heuristic local td3 ddpg sac sa --episodes 50 --groups 5
```

## Configuration

- `--episodes`: Number of episodes to run per group.
- `--groups`: Number of independent runs (seeds) to average results.
- `--vehicles`, `--rsus`, `--uavs`: Override topology (defaults: 12, 4, 2).
