"""
简化的实时训练可视化（完全自包含，无外部依赖）
"""
import os
import json
import threading
from datetime import datetime
from typing import Dict, List, Optional
from flask import Flask, render_template_string
from flask_socketio import SocketIO
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, 'static')

app = Flask(__name__, static_folder=static_folder)
app.config['SECRET_KEY'] = 'vec-training-monitor'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading', ping_timeout=60, ping_interval=25)

# 全局训练数据存储
class TrainingDataStore:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.algorithm = "Unknown"
        self.episode_rewards = []
        self.episode_metrics = {
            'avg_delay': [],
            'total_energy': [],
            'task_completion_rate': [],
            'cache_hit_rate': [],
        }
        self.current_episode = 0
        self.total_episodes = 0
        self.training_start_time = None
    
    def update_episode(self, episode: int, reward: float, metrics: Dict):
        self.current_episode = episode
        self.episode_rewards.append(float(reward))
        
        for key in self.episode_metrics:
            if key in metrics:
                self.episode_metrics[key].append(float(metrics[key]))
    
    def get_latest_stats(self) -> Dict:
        if not self.episode_rewards:
            return {}
        
        window = min(20, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window:]
        
        stats = {
            'current_episode': int(self.current_episode),
            'total_episodes': int(self.total_episodes),
            'latest_reward': float(self.episode_rewards[-1]) if self.episode_rewards else 0,
            'avg_reward': float(np.mean(recent_rewards)),
            'best_reward': float(np.max(self.episode_rewards)),
            'progress': (self.current_episode / self.total_episodes * 100) if self.total_episodes > 0 else 0
        }
        
        for key, values in self.episode_metrics.items():
            if values:
                stats[f'latest_{key}'] = float(values[-1])
                stats[f'avg_{key}'] = float(np.mean(values[-window:]))
        
        return stats
    
    def get_chart_data(self) -> Dict:
        return {
            'episodes': list(range(1, len(self.episode_rewards) + 1)),
            'rewards': [float(x) for x in self.episode_rewards],
            'metrics': {k: [float(x) for x in v] for k, v in self.episode_metrics.items()}
        }

data_store = TrainingDataStore()

# 极简HTML模板（使用Chart.js CDN）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VEC Training Monitor - {{ algorithm }}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/socket.io-client@4.5.4/dist/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Arial', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            min-height: 100vh;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center;
            margin-bottom: 30px;
            font-size: 2.5rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            transition: transform 0.3s;
        }
        .metric-card:hover { transform: translateY(-5px); }
        .metric-label {
            font-size: 0.9rem;
            opacity: 0.8;
            margin-bottom: 10px;
        }
        .metric-value {
            font-size: 2rem;
            font-weight: bold;
        }
        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .progress-bar {
            width: 100%;
            height: 10px;
            background: rgba(255,255,255,0.2);
            border-radius: 5px;
            overflow: hidden;
            margin-bottom: 30px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #4ade80, #3b82f6);
            width: 0%;
            transition: width 0.5s;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 VEC Training Monitor - {{ algorithm }}</h1>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress"></div>
        </div>
        
        <div class="metrics">
            <div class="metric-card">
                <div class="metric-label">Episode</div>
                <div class="metric-value" id="episode">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Reward</div>
                <div class="metric-value" id="reward">0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Completion</div>
                <div class="metric-value" id="completion">0%</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Delay (s)</div>
                <div class="metric-value" id="delay">0.00</div>
            </div>
        </div>
        
        <div class="chart-container">
            <canvas id="rewardChart"></canvas>
        </div>
        
        <div class="chart-container">
            <canvas id="metricsChart"></canvas>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // Reward Chart
        const rewardCtx = document.getElementById('rewardChart').getContext('2d');
        const rewardChart = new Chart(rewardCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Reward',
                    data: [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Reward Evolution', color: '#1f2937', font: { size: 16 } }
                },
                scales: {
                    y: { beginAtZero: false }
                }
            }
        });
        
        // Metrics Chart
        const metricsCtx = document.getElementById('metricsChart').getContext('2d');
        const metricsChart = new Chart(metricsCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Completion Rate (%)',
                        data: [],
                        borderColor: '#10b981',
                        yAxisID: 'y',
                        tension: 0.4
                    },
                    {
                        label: 'Avg Delay (s)',
                        data: [],
                        borderColor: '#f59e0b',
                        yAxisID: 'y1',
                        tension: 0.4
                    }
                ]
            },
            options: {
                responsive: true,
                plugins: {
                    title: { display: true, text: 'Key Metrics', color: '#1f2937', font: { size: 16 } }
                },
                scales: {
                    y: { type: 'linear', position: 'left', title: { display: true, text: 'Completion %' } },
                    y1: { type: 'linear', position: 'right', title: { display: true, text: 'Delay (s)' }, grid: { drawOnChartArea: false } }
                }
            }
        });
        
        socket.on('training_update', function(data) {
            document.getElementById('episode').textContent = data.current_episode;
            document.getElementById('reward').textContent = data.latest_reward.toFixed(2);
            if (data.latest_task_completion_rate !== undefined) {
                document.getElementById('completion').textContent = (data.latest_task_completion_rate * 100).toFixed(1) + '%';
            }
            if (data.latest_avg_delay !== undefined) {
                document.getElementById('delay').textContent = data.latest_avg_delay.toFixed(3);
            }
            document.getElementById('progress').style.width = data.progress + '%';
        });
        
        socket.on('chart_update', function(data) {
            rewardChart.data.labels.push(data.episode);
            rewardChart.data.datasets[0].data.push(data.reward);
            rewardChart.update('none');
            
            if (data.metrics.task_completion_rate !== undefined) {
                metricsChart.data.labels.push(data.episode);
                metricsChart.data.datasets[0].data.push(data.metrics.task_completion_rate * 100);
                metricsChart.data.datasets[1].data.push(data.metrics.avg_delay);
                metricsChart.update('none');
            }
        });
        
        console.log('Visualization initialized successfully!');
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, algorithm=data_store.algorithm)

@socketio.on('connect')
def handle_connect():
    print('客户端已连接')

@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开')

class RealtimeVisualizer:
    def __init__(self, algorithm: str = "Unknown", total_episodes: int = 100, port: int = 5000, auto_open: bool = True):
        self.algorithm = algorithm
        self.total_episodes = total_episodes
        self.port = port
        self.auto_open = auto_open
        self.server_thread = None
        
        data_store.reset()
        data_store.algorithm = algorithm
        data_store.total_episodes = total_episodes
        data_store.training_start_time = datetime.now()
    
    def start(self):
        print(f"🌐 启动实时可视化服务器在 http://localhost:{self.port}")
        
        self.server_thread = threading.Thread(
            target=lambda: socketio.run(app, host='0.0.0.0', port=self.port, debug=False, use_reloader=False, allow_unsafe_werkzeug=True)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        if self.auto_open:
            import time
            import webbrowser
            time.sleep(1)
            webbrowser.open(f'http://localhost:{self.port}')
            print(f"✅ 浏览器已打开")
    
    def update(self, episode: int, reward: float, metrics: Dict):
        data_store.update_episode(episode, reward, metrics)
        stats = data_store.get_latest_stats()
        
        socketio.emit('training_update', stats)
        socketio.emit('chart_update', {
            'episode': int(episode),
            'reward': float(reward),
            'metrics': {k: float(v) for k, v in metrics.items()}
        })
    
    def complete(self):
        print("✅ 训练完成")

def create_visualizer(algorithm: str = "Unknown", total_episodes: int = 100, 
                     port: int = 5000, auto_open: bool = True) -> RealtimeVisualizer:
    visualizer = RealtimeVisualizer(algorithm, total_episodes, port, auto_open)
    visualizer.start()
    return visualizer
