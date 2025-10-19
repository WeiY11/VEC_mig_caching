#!/usr/bin/env python3
"""
交互式算法结果可视化工具

提供Web界面实时查看和比较算法性能
"""

import json
from pathlib import Path
from flask import Flask, render_template_string, jsonify
import plotly.graph_objs as go
import plotly.utils
import numpy as np
from datetime import datetime

app = Flask(__name__)

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VEC算法对比可视化</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 30px;
        }
        .controls {
            margin-bottom: 20px;
            text-align: center;
        }
        .controls button {
            margin: 5px;
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }
        .controls button:hover {
            background-color: #45a049;
        }
        .chart-container {
            margin-bottom: 30px;
            border: 1px solid #ddd;
            border-radius: 5px;
            padding: 10px;
        }
        .info-box {
            background-color: #f0f0f0;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .algorithm-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        .algo-card {
            padding: 10px;
            border-radius: 5px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .algo-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .heuristic { background-color: #FFE5CC; }
        .metaheuristic { background-color: #D4F1D4; }
        .drl { background-color: #D4E6F1; }
        .selected { border: 3px solid #333; }
    </style>
</head>
<body>
    <div class="container">
        <h1>VEC边缘计算系统 - 算法性能对比可视化</h1>
        
        <div class="info-box">
            <p><strong>最后更新时间：</strong> <span id="update-time">-</span></p>
            <p><strong>已加载算法数：</strong> <span id="algo-count">0</span></p>
        </div>
        
        <div class="algorithm-list" id="algorithm-list"></div>
        
        <div class="controls">
            <button onclick="refreshData()">刷新数据</button>
            <button onclick="selectAll()">全选</button>
            <button onclick="selectNone()">全不选</button>
            <button onclick="selectType('heuristic')">只看启发式</button>
            <button onclick="selectType('metaheuristic')">只看元启发式</button>
            <button onclick="selectType('drl')">只看DRL</button>
        </div>
        
        <div class="chart-container">
            <div id="performance-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="learning-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="radar-chart"></div>
        </div>
        
        <div class="chart-container">
            <div id="box-chart"></div>
        </div>
    </div>
    
    <script>
        let allData = {};
        let selectedAlgorithms = new Set();
        
        // 算法信息
        const algorithmInfo = {
            'Random': {type: 'heuristic', color: '#FF6B6B'},
            'Greedy': {type: 'heuristic', color: '#4ECDC4'},
            'RoundRobin': {type: 'heuristic', color: '#45B7D1'},
            'LocalFirst': {type: 'heuristic', color: '#96CEB4'},
            'NearestNode': {type: 'heuristic', color: '#FECA57'},
            'MinDelay': {type: 'heuristic', color: '#DDA0DD'},
            'MinEnergy': {type: 'heuristic', color: '#98D8C8'},
            'LoadBalance': {type: 'heuristic', color: '#F7DC6F'},
            'HybridGreedy': {type: 'heuristic', color: '#BB8FCE'},
            'GA': {type: 'metaheuristic', color: '#FF7F50'},
            'PSO': {type: 'metaheuristic', color: '#32CD32'},
            'DQN': {type: 'drl', color: '#FF1493'},
            'DDPG': {type: 'drl', color: '#00CED1'},
            'TD3': {type: 'drl', color: '#FFD700'},
            'SAC': {type: 'drl', color: '#9370DB'},
            'PPO': {type: 'drl', color: '#20B2AA'}
        };
        
        function refreshData() {
            fetch('/api/results')
                .then(response => response.json())
                .then(data => {
                    allData = data;
                    updateAlgorithmList();
                    updateCharts();
                    document.getElementById('update-time').textContent = new Date().toLocaleString();
                    document.getElementById('algo-count').textContent = Object.keys(data).length;
                });
        }
        
        function updateAlgorithmList() {
            const container = document.getElementById('algorithm-list');
            container.innerHTML = '';
            
            for (const [algo, data] of Object.entries(allData)) {
                const info = algorithmInfo[algo] || {type: 'unknown', color: '#999'};
                const card = document.createElement('div');
                card.className = `algo-card ${info.type} ${selectedAlgorithms.has(algo) ? 'selected' : ''}`;
                card.innerHTML = `
                    <h4>${algo}</h4>
                    <p>时延: ${data.avg_delay ? data.avg_delay.toFixed(3) : '-'}s</p>
                    <p>完成率: ${data.avg_completion_rate ? (data.avg_completion_rate * 100).toFixed(1) : '-'}%</p>
                `;
                card.onclick = () => toggleAlgorithm(algo);
                container.appendChild(card);
            }
        }
        
        function toggleAlgorithm(algo) {
            if (selectedAlgorithms.has(algo)) {
                selectedAlgorithms.delete(algo);
            } else {
                selectedAlgorithms.add(algo);
            }
            updateAlgorithmList();
            updateCharts();
        }
        
        function selectAll() {
            selectedAlgorithms = new Set(Object.keys(allData));
            updateAlgorithmList();
            updateCharts();
        }
        
        function selectNone() {
            selectedAlgorithms.clear();
            updateAlgorithmList();
            updateCharts();
        }
        
        function selectType(type) {
            selectedAlgorithms.clear();
            for (const [algo, info] of Object.entries(algorithmInfo)) {
                if (info.type === type && allData[algo]) {
                    selectedAlgorithms.add(algo);
                }
            }
            updateAlgorithmList();
            updateCharts();
        }
        
        function updateCharts() {
            const selected = Array.from(selectedAlgorithms);
            
            // 1. 性能对比图
            updatePerformanceChart(selected);
            
            // 2. 学习曲线
            updateLearningChart(selected);
            
            // 3. 雷达图
            updateRadarChart(selected);
            
            // 4. 箱线图
            updateBoxChart(selected);
        }
        
        function updatePerformanceChart(algorithms) {
            const traces = [
                {
                    x: algorithms,
                    y: algorithms.map(a => allData[a]?.avg_delay || 0),
                    name: '平均时延',
                    type: 'bar',
                    yaxis: 'y',
                    marker: {color: algorithms.map(a => algorithmInfo[a]?.color || '#999')}
                },
                {
                    x: algorithms,
                    y: algorithms.map(a => allData[a]?.avg_energy || 0),
                    name: '平均能耗',
                    type: 'bar',
                    yaxis: 'y2',
                    marker: {color: algorithms.map(a => algorithmInfo[a]?.color || '#999'), opacity: 0.7}
                }
            ];
            
            const layout = {
                title: '算法性能对比',
                barmode: 'group',
                yaxis: {title: '时延 (秒)', side: 'left'},
                yaxis2: {title: '能耗 (焦耳)', side: 'right', overlaying: 'y'},
                height: 400
            };
            
            Plotly.newPlot('performance-chart', traces, layout);
        }
        
        function updateLearningChart(algorithms) {
            const traces = [];
            
            for (const algo of algorithms) {
                if (allData[algo]?.episode_rewards) {
                    traces.push({
                        x: Array.from({length: allData[algo].episode_rewards.length}, (_, i) => i + 1),
                        y: allData[algo].episode_rewards,
                        name: algo,
                        type: 'scatter',
                        mode: 'lines',
                        line: {color: algorithmInfo[algo]?.color || '#999'}
                    });
                }
            }
            
            const layout = {
                title: '学习曲线',
                xaxis: {title: '训练轮次'},
                yaxis: {title: '奖励值'},
                height: 400
            };
            
            Plotly.newPlot('learning-chart', traces, layout);
        }
        
        function updateRadarChart(algorithms) {
            const traces = [];
            const categories = ['时延↓', '能耗↓', '完成率↑', '奖励↑', '稳定性↑'];
            
            for (const algo of algorithms.slice(0, 5)) {  // 最多显示5个
                if (allData[algo]) {
                    const data = allData[algo];
                    const values = [
                        1 - (data.avg_delay || 1) / 2,
                        1 - (data.avg_energy || 5000) / 10000,
                        data.avg_completion_rate || 0,
                        ((data.final_reward || -20) + 20) / 20,
                        1 - (data.std_delay || 0.5) / 1
                    ].map(v => Math.max(0, Math.min(1, v)));
                    
                    traces.push({
                        type: 'scatterpolar',
                        r: values,
                        theta: categories,
                        fill: 'toself',
                        name: algo,
                        line: {color: algorithmInfo[algo]?.color || '#999'}
                    });
                }
            }
            
            const layout = {
                title: '多维度性能雷达图',
                polar: {
                    radialaxis: {
                        visible: true,
                        range: [0, 1]
                    }
                },
                height: 500
            };
            
            Plotly.newPlot('radar-chart', traces, layout);
        }
        
        function updateBoxChart(algorithms) {
            const types = ['heuristic', 'metaheuristic', 'drl'];
            const typeNames = {'heuristic': '启发式', 'metaheuristic': '元启发式', 'drl': 'DRL'};
            const traces = [];
            
            for (const type of types) {
                const typeAlgos = algorithms.filter(a => algorithmInfo[a]?.type === type);
                if (typeAlgos.length > 0) {
                    traces.push({
                        y: typeAlgos.map(a => allData[a]?.avg_delay || 0),
                        name: typeNames[type],
                        type: 'box',
                        boxpoints: 'all',
                        jitter: 0.3,
                        pointpos: -1.8
                    });
                }
            }
            
            const layout = {
                title: '算法类型时延分布',
                yaxis: {title: '时延 (秒)'},
                height: 400
            };
            
            Plotly.newPlot('box-chart', traces, layout);
        }
        
        // 初始化
        window.onload = () => {
            refreshData();
            setInterval(refreshData, 30000);  // 每30秒自动刷新
        };
    </script>
</body>
</html>
"""


class ResultsLoader:
    """结果加载器"""
    
    def __init__(self, results_dir: str = None):
        if results_dir is None:
            self.results_dir = Path(__file__).parent.parent / "results"
        else:
            self.results_dir = Path(results_dir)
            
    def load_all_results(self):
        """加载所有算法的最新结果"""
        results = {}
        algorithms = [
            'Random', 'Greedy', 'RoundRobin', 'LocalFirst', 'NearestNode',
            'MinDelay', 'MinEnergy', 'LoadBalance', 'HybridGreedy',
            'GA', 'PSO',
            'DQN', 'DDPG', 'TD3', 'SAC', 'PPO'
        ]
        
        for algo in algorithms:
            algo_dir = self.results_dir / algo.lower()
            latest_file = algo_dir / f"{algo.lower()}_latest.json"
            
            if latest_file.exists():
                try:
                    with open(latest_file, 'r', encoding='utf-8') as f:
                        results[algo] = json.load(f)
                except:
                    pass
                    
        return results


# 全局结果加载器
loader = ResultsLoader()


@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/results')
def get_results():
    """API: 获取所有结果"""
    results = loader.load_all_results()
    return jsonify(results)


def main():
    print("启动交互式可视化服务器...")
    print("访问 http://localhost:5000 查看可视化界面")
    print("按 Ctrl+C 停止服务器")
    app.run(debug=True, port=5000)










