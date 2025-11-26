"""
实时训练可视化系统
使用Flask + Socket.IO实现训练过程的实时监控和可视化
"""
import os
import json
import threading
import webbrowser
from datetime import datetime
from typing import Dict, List, Optional, Any
from flask import Flask, render_template_string, jsonify
from flask_socketio import SocketIO, emit
import numpy as np

app = Flask(__name__)
app.config['SECRET_KEY'] = 'vec-training-monitor'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
            'data_loss_ratio_bytes': [],
            'migration_success_rate': []
        }
        self.training_config = {}
        self.training_start_time = None
        self.current_episode = 0
        self.total_episodes = 0
        self.performance_stats = {}
    
    def update_episode(self, episode: int, reward: float, metrics: Dict):
        """更新单个episode的数据"""
        self.current_episode = episode
        self.episode_rewards.append(reward)
        
        for key in self.episode_metrics:
            if key in metrics:
                self.episode_metrics[key].append(metrics[key])
    
    def get_latest_stats(self) -> Dict:
        """获取最新统计信息"""
        if not self.episode_rewards:
            return {}
        
        # 计算移动平均
        window = min(20, len(self.episode_rewards))
        recent_rewards = self.episode_rewards[-window:]
        
        stats = {
            'current_episode': self.current_episode,
            'total_episodes': self.total_episodes,
            'latest_reward': self.episode_rewards[-1] if self.episode_rewards else 0,
            'avg_reward': float(np.mean(recent_rewards)),
            'best_reward': float(np.max(self.episode_rewards)),
            'worst_reward': float(np.min(self.episode_rewards)),
            'progress': (self.current_episode / self.total_episodes * 100) if self.total_episodes > 0 else 0
        }
        
        # 添加最新指标
        for key, values in self.episode_metrics.items():
            if values:
                stats[f'latest_{key}'] = values[-1]
                stats[f'avg_{key}'] = float(np.mean(values[-window:]))
        
        return stats
    
    def get_chart_data(self) -> Dict:
        """获取图表数据"""
        return {
            'episodes': list(range(1, len(self.episode_rewards) + 1)),
            'rewards': self.episode_rewards,
            'metrics': self.episode_metrics
        }

# 全局数据存储实例
data_store = TrainingDataStore()

# HTML模板（实时更新版本）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VEC AI Training Monitor - {{ algorithm }}</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: 1px solid rgba(255, 255, 255, 0.1);
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-color: #38bdf8;
            --success-color: #4ade80;
            --warning-color: #facc15;
            --danger-color: #f87171;
            --glass-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }

        .container {
            max-width: 1800px;
            margin: 0 auto;
        }

        /* Header */
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 20px 30px;
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border-radius: 16px;
            border: var(--card-border);
            box-shadow: var(--glass-shadow);
            margin-bottom: 24px;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .header h1 {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .status-badge {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 12px;
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid rgba(74, 222, 128, 0.2);
            border-radius: 20px;
            font-size: 0.85rem;
            color: var(--success-color);
        }

        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success-color);
            border-radius: 50%;
            box-shadow: 0 0 8px var(--success-color);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
            70% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
            100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
        }

        /* Progress Bar */
        .progress-container {
            margin-bottom: 24px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            height: 6px;
            overflow: hidden;
        }

        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #38bdf8, #818cf8);
            width: 0%;
            transition: width 0.5s ease;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px;
            margin-bottom: 24px;
        }

        .metric-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: var(--card-border);
            border-radius: 16px;
            padding: 20px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-4px);
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(56, 189, 248, 0.3);
        }

        .metric-label {
            font-size: 0.85rem;
            color: var(--text-secondary);
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: var(--text-primary);
            display: flex;
            align-items: baseline;
            gap: 4px;
        }

        .metric-unit {
            font-size: 0.9rem;
            color: var(--text-secondary);
            font-weight: 400;
        }

        /* Charts Layout */
        .main-grid {
            display: grid;
            grid-template-columns: 2fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }

        .chart-card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: var(--card-border);
            border-radius: 16px;
            padding: 20px;
            height: 400px;
            position: relative;
        }
        
        .chart-card.full-width {
            grid-column: 1 / -1;
        }

        .chart-title {
            font-size: 1rem;
            font-weight: 600;
            margin-bottom: 15px;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .chart-title::before {
            content: '';
            display: block;
            width: 4px;
            height: 16px;
            background: var(--accent-color);
            border-radius: 2px;
        }

        /* Responsive */
        @media (max-width: 1200px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header class="header">
            <div class="header-title">
                <h1>VEC AI Training Monitor</h1>
                <span style="color: var(--text-secondary); font-size: 1.2rem;">/</span>
                <span style="color: var(--accent-color); font-weight: 600;">{{ algorithm }}</span>
            </div>
            <div class="status-badge">
                <div class="status-dot"></div>
                <span id="status-text">Training Active</span>
            </div>
        </header>

        <!-- Progress -->
        <div class="progress-container">
            <div class="progress-bar" id="progress-fill"></div>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">Episode</div>
                <div class="metric-value" id="current-episode">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Latest Reward</div>
                <div class="metric-value" style="color: var(--accent-color);" id="latest-reward">0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Reward (100ep)</div>
                <div class="metric-value" id="avg-reward">0.00</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Completion Rate</div>
                <div class="metric-value" style="color: var(--success-color);" id="completion-rate">0.0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Avg Delay</div>
                <div class="metric-value" style="color: var(--warning-color);" id="avg-delay">0.00<span class="metric-unit">s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Cache Hit Rate</div>
                <div class="metric-value" style="color: #c084fc;" id="cache-hit-rate">0.0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Energy</div>
                <div class="metric-value" style="color: var(--danger-color);" id="total-energy">0<span class="metric-unit">J</span></div>
            </div>
        </div>

        <!-- Main Charts Area -->
        <div class="main-grid">
            <!-- Reward Chart (Large) -->
            <div class="chart-card full-width">
                <div class="chart-title">Reward Evolution</div>
                <div id="reward-chart" style="width: 100%; height: 340px;"></div>
            </div>

            <!-- Task Assignment -->
            <div class="chart-card">
                <div class="chart-title">Task Assignment Distribution</div>
                <div id="assignment-chart" style="width: 100%; height: 340px;"></div>
            </div>

            <!-- Delay -->
            <div class="chart-card">
                <div class="chart-title">Average Latency</div>
                <div id="delay-chart" style="width: 100%; height: 340px;"></div>
            </div>

            <!-- Cache & Completion -->
            <div class="chart-card">
                <div class="chart-title">Cache Hit Rate</div>
                <div id="cache-chart" style="width: 100%; height: 340px;"></div>
            </div>

            <div class="chart-card">
                <div class="chart-title">Completion Rate</div>
                <div id="completion-chart" style="width: 100%; height: 340px;"></div>
            </div>
            
            <div class="chart-card full-width">
                 <div class="chart-title">System Energy Consumption</div>
                 <div id="energy-chart" style="width: 100%; height: 340px;"></div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Common Plotly Layout for Dark Theme
        const commonLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'Inter' },
            margin: { t: 10, b: 40, l: 50, r: 20 },
            xaxis: { 
                gridcolor: 'rgba(255,255,255,0.05)',
                zerolinecolor: 'rgba(255,255,255,0.1)'
            },
            yaxis: { 
                gridcolor: 'rgba(255,255,255,0.05)',
                zerolinecolor: 'rgba(255,255,255,0.1)'
            },
            hovermode: 'x unified',
            showlegend: true,
            legend: { orientation: 'h', y: 1.1 }
        };

        // Initialize Charts
        const rewardTrace = {
            x: [], y: [], type: 'scatter', mode: 'lines',
            name: 'Reward', line: { color: '#38bdf8', width: 2 },
            fill: 'tozeroy', fillcolor: 'rgba(56, 189, 248, 0.1)'
        };
        Plotly.newPlot('reward-chart', [rewardTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        const delayTrace = {
            x: [], y: [], type: 'scatter', mode: 'lines',
            name: 'Delay', line: { color: '#facc15', width: 2 }
        };
        Plotly.newPlot('delay-chart', [delayTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        const completionTrace = {
            x: [], y: [], type: 'scatter', mode: 'lines',
            name: 'Rate', line: { color: '#4ade80', width: 2 }
        };
        Plotly.newPlot('completion-chart', [completionTrace], { ...commonLayout, yaxis: { ...commonLayout.yaxis, range: [0, 105] } }, {responsive: true, displayModeBar: false});

        const cacheTrace = {
            x: [], y: [], type: 'scatter', mode: 'lines',
            name: 'Hit Rate', line: { color: '#c084fc', width: 2 }
        };
        Plotly.newPlot('cache-chart', [cacheTrace], { ...commonLayout, yaxis: { ...commonLayout.yaxis, range: [0, 105] } }, {responsive: true, displayModeBar: false});

        const energyTrace = {
            x: [], y: [], type: 'scatter', mode: 'lines',
            name: 'Energy', line: { color: '#f87171', width: 2 }
        };
        Plotly.newPlot('energy-chart', [energyTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        // Assignment Chart (Stacked Bar)
        const assignLocal = { x: [], y: [], type: 'bar', name: 'Local', marker: { color: '#94a3b8' } };
        const assignRSU = { x: [], y: [], type: 'bar', name: 'RSU', marker: { color: '#38bdf8' } };
        const assignUAV = { x: [], y: [], type: 'bar', name: 'UAV', marker: { color: '#4ade80' } };
        
        const assignmentLayout = { 
            ...commonLayout, 
            barmode: 'stack',
            xaxis: { ...commonLayout.xaxis, title: 'Episode' },
            yaxis: { ...commonLayout.yaxis, title: 'Task Count' }
        };
        Plotly.newPlot('assignment-chart', [assignLocal, assignRSU, assignUAV], assignmentLayout, {responsive: true, displayModeBar: false});

        // Data Buffers
        let rewardHistory = [];

        // Socket Events
        socket.on('training_update', function(data) {
            // Update Metrics with animation
            updateMetric('current-episode', data.current_episode);
            updateMetric('latest-reward', data.latest_reward.toFixed(2));
            updateMetric('avg-reward', data.avg_reward.toFixed(2));
            
            if (data.latest_task_completion_rate !== undefined) {
                document.getElementById('completion-rate').innerHTML = 
                    (data.latest_task_completion_rate * 100).toFixed(1) + '<span class="metric-unit">%</span>';
            }
            if (data.latest_avg_delay !== undefined) {
                document.getElementById('avg-delay').innerHTML = 
                    data.latest_avg_delay.toFixed(3) + '<span class="metric-unit">s</span>';
            }
            if (data.latest_cache_hit_rate !== undefined) {
                document.getElementById('cache-hit-rate').innerHTML = 
                    (data.latest_cache_hit_rate * 100).toFixed(1) + '<span class="metric-unit">%</span>';
            }
            if (data.latest_total_energy !== undefined) {
                document.getElementById('total-energy').innerHTML = 
                    data.latest_total_energy.toFixed(0) + '<span class="metric-unit">J</span>';
            }

            // Update Progress
            const progress = data.progress;
            document.getElementById('progress-fill').style.width = progress + '%';
        });

        function updateMetric(id, value) {
            const el = document.getElementById(id);
            if (el.innerText !== value) {
                el.style.opacity = 0;
                setTimeout(() => {
                    el.innerText = value;
                    el.style.opacity = 1;
                }, 200);
            }
        }

        socket.on('chart_update', function(data) {
            rewardHistory.push(data.reward);
            
            // Calculate Confidence Interval
            let upper = null, lower = null;
            if (rewardHistory.length >= 5) {
                const window = Math.min(20, rewardHistory.length);
                const recent = rewardHistory.slice(-window);
                const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
                const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
                const std = Math.sqrt(variance);
                upper = data.reward + std;
                lower = data.reward - std;
            }

            // Update Reward Chart
            const rewardUpdate = { x: [[data.episode]], y: [[data.reward]] };
            if (upper !== null) {
                // Add confidence traces if needed (simplified here to just main line for performance)
            }
            Plotly.extendTraces('reward-chart', rewardUpdate, [0]);

            // Update Other Charts
            if (data.metrics.avg_delay !== undefined) Plotly.extendTraces('delay-chart', { x: [[data.episode]], y: [[data.metrics.avg_delay]] }, [0]);
            if (data.metrics.task_completion_rate !== undefined) Plotly.extendTraces('completion-chart', { x: [[data.episode]], y: [[data.metrics.task_completion_rate * 100]] }, [0]);
            if (data.metrics.total_energy !== undefined) Plotly.extendTraces('energy-chart', { x: [[data.episode]], y: [[data.metrics.total_energy]] }, [0]);
            if (data.metrics.cache_hit_rate !== undefined) Plotly.extendTraces('cache-chart', { x: [[data.episode]], y: [[data.metrics.cache_hit_rate * 100]] }, [0]);
            
            if (data.metrics.local_tasks_count !== undefined) {
                Plotly.extendTraces('assignment-chart', {
                    x: [[data.episode], [data.episode], [data.episode]],
                    y: [[data.metrics.local_tasks_count], [data.metrics.rsu_tasks_count], [data.metrics.uav_tasks_count]]
                }, [0, 1, 2]);
            }
        });

        socket.on('training_complete', function() {
            const statusText = document.getElementById('status-text');
            statusText.innerText = 'Training Complete';
            statusText.style.color = 'var(--success-color)';
            document.querySelector('.status-dot').style.animation = 'none';
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页面"""
    return render_template_string(HTML_TEMPLATE, algorithm=data_store.algorithm)

@app.route('/api/stats')
def get_stats():
    """获取统计数据API"""
    return jsonify(data_store.get_latest_stats())

@app.route('/api/charts')
def get_chart_data():
    """获取图表数据API"""
    return jsonify(data_store.get_chart_data())

@socketio.on('connect')
def handle_connect():
    """客户端连接"""
    print('客户端已连接')
    emit('status', {'status': 'connected'})

@socketio.on('disconnect')
def handle_disconnect():
    """客户端断开"""
    print('客户端已断开')

class RealtimeVisualizer:
    """实时可视化管理器"""
    
    def __init__(self, algorithm: str = "Unknown", total_episodes: int = 100, port: int = 5000, auto_open: bool = True):
        self.algorithm = algorithm
        self.total_episodes = total_episodes
        self.port = port
        self.auto_open = auto_open
        self.server_thread = None
        
        # 重置数据存储
        data_store.reset()
        data_store.algorithm = algorithm
        data_store.total_episodes = total_episodes
        data_store.training_start_time = datetime.now()
    
    def start(self):
        """启动可视化服务器"""
        print(f"🌐 启动实时可视化服务器在 http://localhost:{self.port}")
        
        # 在新线程中启动Flask服务器
        self.server_thread = threading.Thread(
            target=lambda: socketio.run(app, host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
        )
        self.server_thread.daemon = True
        self.server_thread.start()
        
        # 自动打开浏览器
        if self.auto_open:
            import time
            time.sleep(1)  # 等待服务器启动
            webbrowser.open(f'http://localhost:{self.port}')
            print(f"✅ 浏览器已打开，访问 http://localhost:{self.port} 查看实时可视化")
    
    def update(self, episode: int, reward: float, metrics: Dict):
        """更新训练数据"""
        # 更新数据存储
        data_store.update_episode(episode, reward, metrics)
        
        # 获取最新统计
        stats = data_store.get_latest_stats()
        
        # 通过WebSocket发送更新（兼容新版本flask-socketio）
        socketio.emit('training_update', stats)
        socketio.emit('chart_update', {
            'episode': episode,
            'reward': reward,
            'metrics': metrics
        })
    
    def complete(self):
        """标记训练完成"""
        socketio.emit('training_complete', {
            'total_episodes': data_store.current_episode,
            'final_reward': data_store.episode_rewards[-1] if data_store.episode_rewards else 0
        })
        print("✅ 训练完成，可视化数据已更新")

# 便捷函数
def create_visualizer(algorithm: str = "Unknown", total_episodes: int = 100, 
                     port: int = 5000, auto_open: bool = True) -> RealtimeVisualizer:
    """创建实时可视化器"""
    visualizer = RealtimeVisualizer(algorithm, total_episodes, port, auto_open)
    visualizer.start()
    return visualizer

if __name__ == "__main__":
    # 测试模式：模拟训练数据
    import time
    
    visualizer = create_visualizer(algorithm="TD3", total_episodes=100, port=5000)
    
    print("开始模拟训练...")
    for episode in range(1, 101):
        # 模拟训练数据
        reward = -1000 + episode * 15 + np.random.randn() * 50
        metrics = {
            'avg_delay': 0.5 - episode * 0.003 + np.random.randn() * 0.05,
            'total_energy': 800 - episode * 3 + np.random.randn() * 20,
            'task_completion_rate': 0.7 + episode * 0.002 + np.random.randn() * 0.02,
            'cache_hit_rate': 0.5 + episode * 0.004 + np.random.randn() * 0.03,
            'data_loss_ratio_bytes': 0.2 - episode * 0.001,
            'migration_success_rate': 0.6 + episode * 0.003
        }
        
        visualizer.update(episode, reward, metrics)
        print(f"Episode {episode}/100, Reward: {reward:.2f}")
        time.sleep(0.5)  # 模拟训练时间
    
    visualizer.complete()
    print("模拟训练完成！可视化界面将保持打开。按 Ctrl+C 退出。")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n退出可视化服务器")

