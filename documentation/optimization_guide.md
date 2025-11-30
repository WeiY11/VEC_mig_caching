- Critic LR: `8e-5` (was `2e-4`)
- **Batch Size**: Reduced to improve update frequency relative to data collection.
  - Batch Size: `512` (was `1024`)
- **Reward Smoothing**: Increased smoothing factor.
- Cache hit rate should increase.

## Troubleshooting

- If rewards remain flat at -100 (clipped), check if `latency_target` is still too aggressive.
- If training is unstable, try reducing learning rate further (e.g., `1e-5`).
