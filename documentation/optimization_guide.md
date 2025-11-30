# OPTIMIZED_TD3 Optimization Guide

## Overview

This document tracks the optimization steps and configurations for the `OPTIMIZED_TD3` algorithm in the VEC system.

## Key Changes (2025-11-30)

### 1. Baseline Verification

- Added `--quick-test` flag to `train_single_agent.py` to run a 5-episode baseline test with a fixed local-only policy.
- Purpose: Verify environment solvability and reward scale.

### 2. Hyperparameter Tuning

- **Learning Rate**: Reduced to stabilize training.
  - Actor LR: `5e-5` (was `1e-4`)
  - Critic LR: `8e-5` (was `2e-4`)
- **Batch Size**: Reduced to improve update frequency relative to data collection.
  - Batch Size: `512` (was `1024`)
- **Reward Smoothing**: Increased smoothing factor.
  - `reward_smooth_alpha`: `0.3` (was `0.25`)

### 3. Reward Function Adjustments

- **Targets**: Relaxed targets to match early training reality.
  - `latency_target`: `1.5s` (was `0.4s`)
  - `energy_target`: `9000J` (was `2200J`)
- **Debug**: Added debug prints to `UnifiedRewardCalculator` to monitor raw metrics input.

## How to Verify

### Run Quick Baseline Test

```bash
python scripts/quick_test.py
```

Expected Output:

- 5 episodes run.
- Rewards should be negative but stable (e.g., -50 to -10).
- No crashes.

### Run Full Training

```bash
python train_single_agent.py --algorithm OPTIMIZED_TD3 --episodes 400
```

Expected Outcome:

- Reward curve should show improvement over time.
- Convergence within 400 episodes.
- Cache hit rate should increase.

## Troubleshooting

- If rewards remain flat at -100 (clipped), check if `latency_target` is still too aggressive.
- If training is unstable, try reducing learning rate further (e.g., `1e-5`).
