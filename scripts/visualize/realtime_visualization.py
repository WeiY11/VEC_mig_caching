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

# 获取当前脚本的目录，用于配置static文件夹
current_dir = os.path.dirname(os.path.abspath(__file__))
static_folder = os.path.join(current_dir, 'static')

app = Flask(__name__, static_folder=static_folder, static_url_path='/static')
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
# HTML模板（实时更新版本 - 带动画与CDN回退）
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>VEC AI Training Monitor - {{ algorithm }}</title>
    
    <!-- Socket.IO with CDN Fallback -->
    <script src="/static/js/socket.io.min.js" onerror="this.onerror=null;this.src='https://cdn.socket.io/4.5.4/socket.io.min.js';"></script>
    
    <!-- Plotly.js with CDN Fallback -->
    <script src="/static/js/plotly.min.js" onerror="this.onerror=null;this.src='https://cdn.plot.ly/plotly-2.24.1.min.js';"></script>
    
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

        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', sans-serif;
            background: var(--bg-gradient);
            color: var(--text-primary);
            min-height: 100vh;
            padding: 20px;
            overflow-x: hidden;
        }

        .container { max-width: 1800px; margin: 0 auto; }

        /* Loading Overlay */
        #loading-overlay {
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(15, 23, 42, 0.9);
            z-index: 9999;
            display: flex; justify-content: center; align-items: center; flex-direction: column;
            transition: opacity 0.5s ease;
        }
        .loader {
            width: 48px; height: 48px;
            border: 5px solid #FFF;
            border-bottom-color: var(--accent-color);
            border-radius: 50%;
            animation: rotation 1s linear infinite;
            margin-bottom: 20px;
        }
        @keyframes rotation { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

        /* Header */
        .header {
            display: flex; justify-content: space-between; align-items: center;
            padding: 20px 30px;
            background: var(--card-bg); backdrop-filter: blur(10px);
            border-radius: 16px; border: var(--card-border);
            box-shadow: var(--glass-shadow); margin-bottom: 24px;
        }
        .header h1 {
            font-size: 1.5rem; font-weight: 700;
            background: linear-gradient(to right, #38bdf8, #818cf8);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }
        .status-badge {
            display: flex; align-items: center; gap: 8px;
            padding: 6px 12px;
            background: rgba(74, 222, 128, 0.1);
            border: 1px solid rgba(74, 222, 128, 0.2);
            border-radius: 20px;
            font-size: 0.85rem; color: var(--success-color);
        }
        .status-dot {
            width: 8px; height: 8px; background: var(--success-color);
            border-radius: 50%; box-shadow: 0 0 8px var(--success-color);
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0.7); }
            70% { opacity: 0.7; box-shadow: 0 0 0 6px rgba(74, 222, 128, 0); }
            100% { opacity: 1; box-shadow: 0 0 0 0 rgba(74, 222, 128, 0); }
        }

        /* Topology Animation Section */
        .topology-container {
            background: var(--card-bg); backdrop-filter: blur(10px);
            border: var(--card-border); border-radius: 16px;
            padding: 20px; margin-bottom: 24px;
            height: 300px; position: relative; overflow: hidden;
        }
        .topology-title {
            position: absolute; top: 20px; left: 20px;
            font-size: 1rem; font-weight: 600; color: var(--text-primary);
            z-index: 10;
        }
        canvas#topology-canvas {
            width: 100%; height: 100%;
            display: block;
        }

        /* Progress Bar */
        .progress-container {
            margin-bottom: 24px; background: rgba(255, 255, 255, 0.05);
            border-radius: 8px; height: 6px; overflow: hidden;
        }
        .progress-bar {
            height: 100%; background: linear-gradient(90deg, #38bdf8, #818cf8);
            width: 0%; transition: width 0.5s ease;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.5);
        }

        /* Metrics Grid */
        .metrics-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 20px; margin-bottom: 24px;
        }
        .metric-card {
            background: var(--card-bg); backdrop-filter: blur(10px);
            border: var(--card-border); border-radius: 16px;
            padding: 20px; transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
            border-color: rgba(56, 189, 248, 0.3);
        }
        .metric-label {
            font-size: 0.85rem; color: var(--text-secondary);
            margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.8rem; font-weight: 700; color: var(--text-primary);
            display: flex; align-items: baseline; gap: 4px;
        }
        .metric-unit { font-size: 0.9rem; color: var(--text-secondary); font-weight: 400; }

        /* Charts Layout */
        .main-grid {
            display: grid; grid-template-columns: 1fr 1fr;
            gap: 24px; margin-bottom: 24px;
        }
        .chart-card {
            background: var(--card-bg); backdrop-filter: blur(10px);
            border: var(--card-border); border-radius: 16px;
            padding: 20px; height: 400px; position: relative;
        }
        .chart-card.full-width { grid-column: 1 / -1; }
        .chart-title {
            font-size: 1rem; font-weight: 600; margin-bottom: 15px;
            color: var(--text-primary); display: flex; align-items: center; gap: 8px;
        }
        .chart-title::before {
            content: ''; display: block; width: 4px; height: 16px;
            background: var(--accent-color); border-radius: 2px;
        }

        @media (max-width: 1200px) { .main-grid { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div id="loading-overlay">
        <div class="loader"></div>
        <div style="color: #fff; font-size: 1.2rem;">Connecting to Training Server...</div>
    </div>

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

        <!-- Topology Animation -->
        <div class="topology-container">
            <div class="topology-title">Real-time Task Offloading</div>
            <canvas id="topology-canvas"></canvas>
        </div>

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
            <div class="chart-card full-width">
                <div class="chart-title">Reward Evolution</div>
                <div id="reward-chart" style="width: 100%; height: 340px;"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Task Assignment Distribution</div>
                <div id="assignment-chart" style="width: 100%; height: 340px;"></div>
            </div>
            <div class="chart-card">
                <div class="chart-title">Average Latency</div>
                <div id="delay-chart" style="width: 100%; height: 340px;"></div>
            </div>
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
        // --- Socket Connection ---
        const socket = io({
            reconnection: true,
            reconnectionAttempts: Infinity,
            reconnectionDelay: 1000
        });

        socket.on('connect', () => {
            console.log("Connected to server");
            document.getElementById('loading-overlay').style.opacity = '0';
            setTimeout(() => {
                document.getElementById('loading-overlay').style.display = 'none';
            }, 500);
        });

        socket.on('connect_error', (err) => {
            console.error("Connection error:", err);
            // Don't show overlay on reconnect attempt to avoid flickering, just log
        });

        // --- Topology Animation Logic ---
        const canvas = document.getElementById('topology-canvas');
        const ctx = canvas.getContext('2d');
        
        // Resize canvas
        function resizeCanvas() {
            canvas.width = canvas.offsetWidth;
            canvas.height = canvas.offsetHeight;
        }
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();

        // System Nodes
        const nodes = {
            uavs: [],
            rsus: [],
            vehicles: []
        };
        const particles = []; // Task particles

        // Coordinate Mapping
        const SIM_W = 1030;
        const SIM_H = 2060;
        
        function toCanvasX(simX) {
            return (simX / SIM_W) * canvas.width;
        }
        
        function toCanvasY(simY) {
            // Invert Y because canvas 0 is top, sim 0 is bottom (or we map 2060 to top)
            // Let's map Sim Y=0 to Canvas Bottom, Sim Y=2060 to Canvas Top
            // Actually, usually map view is top-down. Let's assume Sim Y=0 is bottom.
            return canvas.height - (simY / SIM_H) * canvas.height;
        }

        // Initialize static nodes (RSUs)
        function initNodes() {
            const w = canvas.width;
            const h = canvas.height;
            
            // RSUs (Fixed positions from simulator)
            // RSU_0: (615, 1610), RSU_1: (450, 1395), RSU_2: (615, 795), RSU_3: (450, 395)
            const rsuData = [
                {x: 615, y: 1610}, {x: 450, y: 1395}, {x: 615, y: 795}, {x: 450, y: 395}
            ];
            
            nodes.rsus = rsuData.map((pos, i) => ({
                x: toCanvasX(pos.x),
                y: toCanvasY(pos.y),
                type: 'RSU',
                color: '#38bdf8',
                id: i
            }));

            // UAVs (Fixed positions from simulator)
            // UAV_0: (515, 1545), UAV_1: (515, 515)
            const uavData = [
                {x: 515, y: 1545}, {x: 515, y: 515}
            ];
            
            nodes.uavs = uavData.map((pos, i) => ({
                x: toCanvasX(pos.x),
                y: toCanvasY(pos.y),
                type: 'UAV',
                color: '#c084fc', // Purple
                angle: i * Math.PI,
                id: i
            }));

            // Vehicles (Dynamic, updated via socket)
            nodes.vehicles = [];
            // Initial placeholders if needed, or wait for first update
            for (let i=0; i<12; i++) {
                 nodes.vehicles.push({
                    x: -100, y: -100, // Off-screen initially
                    type: 'Vehicle',
                    color: '#facc15',
                    id: i
                });
            }
        }
        initNodes(); // Initial setup
        // Re-init on resize
        window.addEventListener('resize', initNodes);

        // Listen for topology updates
        socket.on('topology_update', (data) => {
            if (data.vehicles) {
                // Update or create vehicles
                data.vehicles.forEach(v => {
                    let existing = nodes.vehicles.find(n => n.id === v.id);
                    if (!existing) {
                        existing = {
                            id: v.id,
                            type: 'Vehicle',
                            color: '#facc15'
                        };
                        nodes.vehicles.push(existing);
                    }
                    existing.simX = v.x;
                    existing.simY = v.y;
                    existing.dir = v.dir;
                    // Interpolation target could be added here for smoother animation
                    existing.x = toCanvasX(v.x);
                    existing.y = toCanvasY(v.y);
                });
            }
        });

        function drawRoads() {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.1)';
            
            // Main Road (Vertical at X=515, Width=30)
            const mainX = toCanvasX(515 - 15);
            const mainW = toCanvasX(515 + 15) - mainX;
            ctx.fillRect(mainX, 0, mainW, canvas.height);
            
            // Upper Intersection (Horizontal at Y=1545, Width=30)
            const upperY = toCanvasY(1545 + 15); // Top edge in canvas
            const upperH = toCanvasY(1545 - 15) - upperY;
            ctx.fillRect(0, upperY, canvas.width, upperH);
            
            // Lower Intersection (Horizontal at Y=515, Width=30)
            const lowerY = toCanvasY(515 + 15);
            const lowerH = toCanvasY(515 - 15) - lowerY;
            ctx.fillRect(0, lowerY, canvas.width, lowerH);
            
            // Dashed Lines
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.setLineDash([5, 5]);
            ctx.lineWidth = 1;
            
            // Main Center
            ctx.beginPath();
            ctx.moveTo(toCanvasX(515), 0);
            ctx.lineTo(toCanvasX(515), canvas.height);
            ctx.stroke();
            
            // Upper Center
            ctx.beginPath();
            ctx.moveTo(0, toCanvasY(1545));
            ctx.lineTo(canvas.width, toCanvasY(1545));
            ctx.stroke();
            
            // Lower Center
            ctx.beginPath();
            ctx.moveTo(0, toCanvasY(515));
            ctx.lineTo(canvas.width, toCanvasY(515));
            ctx.stroke();
            
            ctx.setLineDash([]);
        }

        // Animation Loop
        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const w = canvas.width;
            const h = canvas.height;

            drawRoads();

            // Draw Connections (Backhaul) - Optional, maybe distracting with roads
            // ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
            // ctx.lineWidth = 1;
            // ctx.beginPath();
            // ... (Skip RSU-RSU lines for cleaner look)

            // Update & Draw UAVs
            nodes.uavs.forEach(uav => {
                uav.angle += 0.02;
                // uav.x += Math.sin(uav.angle) * 0.5; // Hover effect
                const hoverY = uav.y + Math.sin(uav.angle) * 5;
                
                ctx.fillStyle = uav.color;
                ctx.beginPath();
                ctx.arc(uav.x, hoverY, 8, 0, Math.PI*2);
                ctx.fill();
                // Label
                ctx.fillStyle = '#fff';
                ctx.font = '10px Inter';
                ctx.fillText(`UAV_${uav.id}`, uav.x - 15, hoverY - 12);
                
                // Range Circle
                ctx.strokeStyle = 'rgba(74, 222, 128, 0.1)';
                ctx.beginPath();
                ctx.arc(uav.x, hoverY, toCanvasX(350) - toCanvasX(0), 0, Math.PI*2); // 350m radius
                ctx.stroke();
            });

            // Update & Draw RSUs
            nodes.rsus.forEach(rsu => {
                ctx.fillStyle = rsu.color;
                ctx.beginPath();
                ctx.rect(rsu.x - 6, rsu.y - 6, 12, 12);
                ctx.fill();
                ctx.fillStyle = '#fff';
                ctx.font = '10px Inter';
                ctx.fillText(`RSU_${rsu.id}`, rsu.x - 15, rsu.y - 12);
            });

            // Update & Draw Vehicles
            nodes.vehicles.forEach(veh => {
                // Position is updated via socket, just draw
                if (veh.x === undefined) return;

                ctx.fillStyle = veh.color;
                ctx.beginPath();
                // Draw as triangle pointing direction
                ctx.save();
                ctx.translate(veh.x, veh.y);
                // veh.dir is in radians (0 = East, PI/2 = North)
                // Canvas Y is inverted, so North is -Y.
                // If sim dir 0 is East (X+), PI/2 is North (Y+).
                // In canvas: 0 is Right (X+), -PI/2 is Up (Y-).
                // So we need to negate the angle.
                const angle = -(veh.dir || 0); 
                ctx.rotate(angle);
                ctx.moveTo(6, 0);
                ctx.lineTo(-4, 4);
                ctx.lineTo(-4, -4);
                ctx.fill();
                ctx.restore();
                
                // Label
                // ctx.fillStyle = '#fff';
                // ctx.font = '8px Inter';
                // ctx.fillText(veh.id, veh.x, veh.y - 8);
            });

            // Update & Draw Particles (Tasks)
            for(let i=particles.length-1; i>=0; i--) {
                const p = particles[i];
                const dx = p.targetX - p.x;
                const dy = p.targetY - p.y;
                const dist = Math.sqrt(dx*dx + dy*dy);
                
                if(dist < 5) {
                    particles.splice(i, 1); // Arrived
                    continue;
                }
                
                p.x += (dx / dist) * p.speed;
                p.y += (dy / dist) * p.speed;
                
                ctx.fillStyle = p.color;
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3, 0, Math.PI*2);
                ctx.fill();
            }

            requestAnimationFrame(animate);
        }
        animate();

        // Handle Task Event
        socket.on('task_event', (data) => {
            // data: { type: 'local'|'rsu'|'uav', vehicle_id: 0-11, target_id: 0-N }
            const veh = nodes.vehicles[data.vehicle_id % nodes.vehicles.length];
            if(!veh) return;

            let target = null;
            let color = '#fff';

            if(data.type === 'local') {
                // Local processing: particle goes up slightly and fades (simulated by short target)
                target = { x: veh.x, y: veh.y - 20 };
                color = '#94a3b8'; // Grey
            } else if(data.type === 'rsu') {
                target = nodes.rsus[data.target_id % nodes.rsus.length];
                color = '#38bdf8'; // Blue
            } else if(data.type === 'uav') {
                target = nodes.uavs[data.target_id % nodes.uavs.length];
                color = '#4ade80'; // Green
            }

            if(target) {
                particles.push({
                    x: veh.x,
                    y: veh.y,
                    targetX: target.x,
                    targetY: target.y,
                    speed: 4 + Math.random() * 2,
                    color: color
                });
            }
        });

        // --- Chart Logic (Existing) ---
        const commonLayout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            font: { color: '#94a3b8', family: 'Inter' },
            margin: { t: 10, b: 40, l: 50, r: 20 },
            xaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            yaxis: { gridcolor: 'rgba(255,255,255,0.05)', zerolinecolor: 'rgba(255,255,255,0.1)' },
            hovermode: 'x unified',
            showlegend: true,
            legend: { orientation: 'h', y: 1.1 }
        };

        const rewardTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Reward', line: { color: '#38bdf8', width: 2 }, fill: 'tozeroy', fillcolor: 'rgba(56, 189, 248, 0.1)' };
        Plotly.newPlot('reward-chart', [rewardTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        const delayTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Delay', line: { color: '#facc15', width: 2 } };
        Plotly.newPlot('delay-chart', [delayTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        const completionTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Rate', line: { color: '#4ade80', width: 2 } };
        Plotly.newPlot('completion-chart', [completionTrace], { ...commonLayout, yaxis: { ...commonLayout.yaxis, range: [0, 105] } }, {responsive: true, displayModeBar: false});

        const cacheTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Hit Rate', line: { color: '#c084fc', width: 2 } };
        Plotly.newPlot('cache-chart', [cacheTrace], { ...commonLayout, yaxis: { ...commonLayout.yaxis, range: [0, 105] } }, {responsive: true, displayModeBar: false});

        const energyTrace = { x: [], y: [], type: 'scatter', mode: 'lines', name: 'Energy', line: { color: '#f87171', width: 2 } };
        Plotly.newPlot('energy-chart', [energyTrace], { ...commonLayout }, {responsive: true, displayModeBar: false});

        const assignLocal = { x: [], y: [], type: 'bar', name: 'Local', marker: { color: '#94a3b8' } };
        const assignRSU = { x: [], y: [], type: 'bar', name: 'RSU', marker: { color: '#38bdf8' } };
        const assignUAV = { x: [], y: [], type: 'bar', name: 'UAV', marker: { color: '#4ade80' } };
        Plotly.newPlot('assignment-chart', [assignLocal, assignRSU, assignUAV], { ...commonLayout, barmode: 'stack', xaxis: { ...commonLayout.xaxis, title: 'Episode' }, yaxis: { ...commonLayout.yaxis, title: 'Task Count' } }, {responsive: true, displayModeBar: false});

        // Socket Events for Metrics
        socket.on('training_update', function(data) {
            updateMetric('current-episode', data.current_episode);
            updateMetric('latest-reward', data.latest_reward.toFixed(2));
            updateMetric('avg-reward', data.avg_reward.toFixed(2));
            if (data.latest_task_completion_rate !== undefined) document.getElementById('completion-rate').innerHTML = (data.latest_task_completion_rate * 100).toFixed(1) + '<span class="metric-unit">%</span>';
            if (data.latest_avg_delay !== undefined) document.getElementById('avg-delay').innerHTML = data.latest_avg_delay.toFixed(3) + '<span class="metric-unit">s</span>';
            if (data.latest_cache_hit_rate !== undefined) document.getElementById('cache-hit-rate').innerHTML = (data.latest_cache_hit_rate * 100).toFixed(1) + '<span class="metric-unit">%</span>';
            if (data.latest_total_energy !== undefined) document.getElementById('total-energy').innerHTML = data.latest_total_energy.toFixed(0) + '<span class="metric-unit">J</span>';
            document.getElementById('progress-fill').style.width = data.progress + '%';
        });

        function updateMetric(id, value) {
            const el = document.getElementById(id);
            if (el.innerText !== value) {
                el.innerText = value;
            }
        }

        socket.on('chart_update', function(data) {
            Plotly.extendTraces('reward-chart', { x: [[data.episode]], y: [[data.reward]] }, [0]);
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
        print("✅ 训练完成")
    
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

