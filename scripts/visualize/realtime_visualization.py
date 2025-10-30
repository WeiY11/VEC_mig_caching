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
    <title>实时训练监控 - {{ algorithm }}</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333;
            padding: 20px;
        }
        
        .container {
            max-width: 1600px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        .status-running {
            background: #28a745;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            padding: 20px;
            background: #f8f9fa;
        }
        
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: #666;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        
        .metric-unit {
            font-size: 0.5em;
            color: #999;
        }
        
        .chart-container {
            padding: 20px;
            margin: 20px;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        
        .progress-bar {
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 20px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            transition: width 0.5s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
        }
        
        .charts-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            padding: 20px;
        }
        
        @media (max-width: 1200px) {
            .charts-row {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 实时训练监控</h1>
            <div style="margin-top: 10px;">
                <span class="status-indicator status-running"></span>
                <span id="status-text">训练进行中...</span>
            </div>
            <div style="margin-top: 10px; font-size: 1.1em;">
                算法: <strong id="algorithm-name">{{ algorithm }}</strong>
            </div>
        </div>
        
        <div class="progress-bar">
            <div class="progress-fill" id="progress-fill" style="width: 0%;">0%</div>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-label">当前轮次</div>
                <div class="metric-value" id="current-episode">0</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最新奖励</div>
                <div class="metric-value" id="latest-reward">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">平均奖励</div>
                <div class="metric-value" id="avg-reward">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">最佳奖励</div>
                <div class="metric-value" id="best-reward">0.000</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">任务完成率</div>
                <div class="metric-value" id="completion-rate">0.0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">平均时延</div>
                <div class="metric-value" id="avg-delay">0.000<span class="metric-unit">s</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">缓存命中率</div>
                <div class="metric-value" id="cache-hit-rate">0.0<span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">总能耗</div>
                <div class="metric-value" id="total-energy">0<span class="metric-unit">J</span></div>
            </div>
        </div>
        
        <div class="chart-container">
            <div id="reward-chart" style="width: 100%; height: 400px;"></div>
        </div>
        
        <div class="charts-row">
            <div class="chart-container">
                <div id="delay-chart" style="width: 100%; height: 300px;"></div>
            </div>
            <div class="chart-container">
                <div id="completion-chart" style="width: 100%; height: 300px;"></div>
            </div>
        </div>
        
        <div class="charts-row">
            <div class="chart-container">
                <div id="energy-chart" style="width: 100%; height: 300px;"></div>
            </div>
            <div class="chart-container">
                <div id="cache-chart" style="width: 100%; height: 300px;"></div>
            </div>
        </div>
    </div>
    
    <script>
        const socket = io();
        
        // 初始化图表
        const rewardTrace = {
            x: [],
            y: [],
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Episode Reward',
            line: { color: '#667eea', width: 2 }
        };
        
        const rewardLayout = {
            title: '奖励演化曲线',
            xaxis: { title: 'Episode' },
            yaxis: { title: 'Reward' },
            hovermode: 'closest'
        };
        
        Plotly.newPlot('reward-chart', [rewardTrace], rewardLayout, {responsive: true});
        
        // 其他图表初始化
        const delayTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Avg Delay', line: { color: '#dc3545' } };
        const delayLayout = { title: '平均时延', xaxis: { title: 'Episode' }, yaxis: { title: 'Delay (s)' } };
        Plotly.newPlot('delay-chart', [delayTrace], delayLayout, {responsive: true});
        
        const completionTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Completion Rate', line: { color: '#28a745' } };
        const completionLayout = { title: '任务完成率', xaxis: { title: 'Episode' }, yaxis: { title: 'Rate (%)' } };
        Plotly.newPlot('completion-chart', [completionTrace], completionLayout, {responsive: true});
        
        const energyTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Total Energy', line: { color: '#ff6b6b' } };
        const energyLayout = { title: '总能耗', xaxis: { title: 'Episode' }, yaxis: { title: 'Energy (J)' } };
        Plotly.newPlot('energy-chart', [energyTrace], energyLayout, {responsive: true});
        
        const cacheTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Cache Hit Rate', line: { color: '#17a2b8' } };
        const cacheLayout = { title: '缓存命中率', xaxis: { title: 'Episode' }, yaxis: { title: 'Hit Rate (%)' } };
        Plotly.newPlot('cache-chart', [cacheTrace], cacheLayout, {responsive: true});
        
        // 置信区间数据存储
        let rewardHistory = [];
        let delayHistory = [];
        let completionHistory = [];
        
        // 监听数据更新
        socket.on('training_update', function(data) {
            // 更新指标卡片
            document.getElementById('current-episode').textContent = data.current_episode;
            document.getElementById('latest-reward').textContent = data.latest_reward.toFixed(3);
            document.getElementById('avg-reward').textContent = data.avg_reward.toFixed(3);
            document.getElementById('best-reward').textContent = data.best_reward.toFixed(3);
            
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
            
            // 更新进度条
            const progress = data.progress;
            document.getElementById('progress-fill').style.width = progress + '%';
            document.getElementById('progress-fill').textContent = progress.toFixed(1) + '%';
        });
        
        socket.on('chart_update', function(data) {
            // 存储历史数据用于计算置信区间
            rewardHistory.push(data.reward);
            if (data.metrics.avg_delay !== undefined) delayHistory.push(data.metrics.avg_delay);
            if (data.metrics.task_completion_rate !== undefined) completionHistory.push(data.metrics.task_completion_rate * 100);
            
            // 🎯 计算置信区间（使用最近20个数据点）
            function calculateConfidence(history, currentValue) {
                if (history.length < 5) return null;
                const window = Math.min(20, history.length);
                const recent = history.slice(-window);
                const mean = recent.reduce((a, b) => a + b, 0) / recent.length;
                const variance = recent.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / recent.length;
                const std = Math.sqrt(variance);
                return { upper: currentValue + std, lower: currentValue - std };
            }
            
            // 更新奖励图表（带置信区间）
            const rewardConfidence = calculateConfidence(rewardHistory, data.reward);
            if (rewardConfidence && rewardHistory.length >= 20) {
                // 如果还没有置信区间轨迹，添加它们
                if (Plotly.data && document.getElementById('reward-chart').data.length === 1) {
                    Plotly.addTraces('reward-chart', [
                        {
                            x: [], y: [], type: 'scatter', mode: 'lines',
                            line: { width: 0 }, showlegend: false, hoverinfo: 'skip'
                        },
                        {
                            x: [], y: [], type: 'scatter', mode: 'lines',
                            fill: 'tonexty', fillcolor: 'rgba(102, 126, 234, 0.2)',
                            line: { width: 0 }, name: '±1σ Confidence', showlegend: true
                        }
                    ]);
                }
                // 更新所有轨迹
                Plotly.extendTraces('reward-chart', {
                    x: [[data.episode], [data.episode], [data.episode]],
                    y: [[data.reward], [rewardConfidence.lower], [rewardConfidence.upper]]
                }, [0, 1, 2]);
            } else {
                Plotly.extendTraces('reward-chart', {
                    x: [[data.episode]],
                    y: [[data.reward]]
                }, [0]);
            }
            
            // 更新其他图表
            if (data.metrics.avg_delay !== undefined) {
                Plotly.extendTraces('delay-chart', {
                    x: [[data.episode]],
                    y: [[data.metrics.avg_delay]]
                }, [0]);
            }
            
            if (data.metrics.task_completion_rate !== undefined) {
                Plotly.extendTraces('completion-chart', {
                    x: [[data.episode]],
                    y: [[data.metrics.task_completion_rate * 100]]
                }, [0]);
            }
            
            if (data.metrics.total_energy !== undefined) {
                Plotly.extendTraces('energy-chart', {
                    x: [[data.episode]],
                    y: [[data.metrics.total_energy]]
                }, [0]);
            }
            
            if (data.metrics.cache_hit_rate !== undefined) {
                Plotly.extendTraces('cache-chart', {
                    x: [[data.episode]],
                    y: [[data.metrics.cache_hit_rate * 100]]
                }, [0]);
            }
        });
        
        socket.on('training_complete', function(data) {
            document.getElementById('status-text').textContent = '训练已完成 ✓';
            document.querySelector('.status-indicator').classList.remove('status-running');
            document.querySelector('.status-indicator').style.background = '#28a745';
            document.querySelector('.status-indicator').style.animation = 'none';
        });
        
        // 连接状态监控
        socket.on('connect', function() {
            console.log('Connected to server');
        });
        
        socket.on('disconnect', function() {
            console.log('Disconnected from server');
            document.getElementById('status-text').textContent = '连接断开';
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

