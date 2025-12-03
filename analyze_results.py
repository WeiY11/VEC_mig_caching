import json
import numpy as np

try:
    with open(r'D:\VEC_mig_caching\results\single_agent\optimized_td3\training_results_20251203_145704.json', encoding='utf-8') as f:
        data = json.load(f)
        
    metrics = data.get('episode_metrics', {})
    output = {}
    output['keys'] = list(metrics.keys())
    
    def get_metric_stats(name):
        values = metrics.get(name, [])
        if values:
            return {
                'last_10': values[-10:],
                'avg_last_100': np.mean(values[-100:]) if len(values) > 100 else np.mean(values)
            }
        return None

    output['avg_delay'] = get_metric_stats('avg_delay')
    output['total_energy'] = get_metric_stats('total_energy')
    output['data_loss_ratio_bytes'] = get_metric_stats('data_loss_ratio_bytes')
    output['task_completion_rate'] = get_metric_stats('task_completion_rate')
    output['cache_hit_rate'] = get_metric_stats('cache_hit_rate')
    output['migration_success_rate'] = get_metric_stats('migration_success_rate')
    
    with open('analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
        
except Exception as e:
    print(f"Error: {e}")
