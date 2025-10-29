import json

latest = json.load(open('results/single_agent/td3/training_results_20251029_183047.json'))
original = json.load(open('results/single_agent/td3/12/5/training_results_20251028_200556.json'))

print("Configuration Comparison:")
print(f"  Latest seed:     {latest['system_config'].get('random_seed', 'N/A')}")
print(f"  Original seed:   {orig['system_config'].get('random_seed', 'N/A')}")
print(f"  Latest episodes: {len(latest['episode_rewards'])}")
print(f"  Original episodes: {len(original['episode_rewards'])}")


