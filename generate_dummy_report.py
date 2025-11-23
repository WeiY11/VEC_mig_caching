
import sys
import os
import random
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.getcwd())

from utils.html_report_generator import HTMLReportGenerator

# Dummy classes to mimic the environment and results
class DummyEnv:
    def __init__(self):
        self.episode_rewards = [random.uniform(-100, 100) for _ in range(50)]
        self.episode_metrics = {
            'avg_delay': [random.uniform(0.1, 2.0) for _ in range(50)],
            'total_energy': [random.uniform(100, 500) for _ in range(50)],
            'task_completion_rate': [random.uniform(0.8, 1.0) for _ in range(50)],
            'cache_hit_rate': [random.uniform(0.2, 0.8) for _ in range(50)],
            'rsu_utilization': [random.uniform(0.4, 0.9) for _ in range(50)],
            'offload_ratio': [random.uniform(0.3, 0.7) for _ in range(50)],
            # New Metrics
            'local_offload_ratio': [random.uniform(0.1, 0.3) for _ in range(50)],
            'rsu_offload_ratio': [random.uniform(0.4, 0.6) for _ in range(50)],
            'uav_offload_ratio': [random.uniform(0.1, 0.3) for _ in range(50)],
            'migration_success_rate': [random.uniform(0.8, 1.0) for _ in range(50)],
            'migration_avg_cost': [random.uniform(10, 50) for _ in range(50)],
            'rsu_hotspot_peak': [random.uniform(0.8, 1.0) for _ in range(50)],
            'queue_overload_events': [random.randint(0, 5) for _ in range(50)],
        }
        self.num_vehicles = 12
        self.num_rsus = 4
        self.num_uavs = 2

def generate_dummy():
    generator = HTMLReportGenerator()
    env = DummyEnv()
    
    results = {
        'final_performance': {
            'avg_reward': 50.5,
            'avg_delay': 0.5,
            'avg_completion': 0.98,
            'avg_episode_reward': 1200
        },
        'training_config': {
            'num_episodes': 50,
            'max_steps_per_episode': 100,
            'training_time_hours': 1.5
        },
        'training_start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'agent_type': 'single_agent'
    }
    
    # Generate report
    report_html = generator.generate_full_report(
        algorithm="TD3",
        training_env=env,
        training_time=5400,
        results=results,
        simulator_stats={}
    )
    
    with open("dummy_report_old.html", "w", encoding="utf-8") as f:
        f.write(report_html)
    
    print("Generated dummy_report_old.html")

if __name__ == "__main__":
    generate_dummy()
