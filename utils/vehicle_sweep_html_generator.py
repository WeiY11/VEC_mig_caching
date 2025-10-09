"""
车辆数扫描实验 - 增强版HTML报告生成器
包含导航、深色模式、交互图表、智能分析等所有高级功能
"""
import base64
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import json

class VehicleSweepHTMLGenerator:
    """车辆数扫描实验HTML报告生成器（增强版）"""
    
    def generate_enhanced_report(self, summaries: List[Dict], timestamp: str,
                                chart1_path: Path, chart2_path: Path) -> str:
        """
        生成增强版HTML报告
        
        Args:
            summaries: 实验结果汇总列表
            timestamp: 时间戳
            chart1_path: 综合对比图路径
            chart2_path: 柱状图对比路径
            
        Returns:
            HTML内容字符串
        """
        # 读取图表
        chart1_base64 = self._img_to_base64(chart1_path)
        chart2_base64 = self._img_to_base64(chart2_path)
        
        # 提取基础数据
        vehicles = [s['num_vehicles'] for s in summaries]
        
        # 构建HTML
        html_parts = []
        html_parts.append(self._generate_header())
        html_parts.append(self._generate_navigation())
        html_parts.append(self._generate_back_to_top())
        html_parts.append(self._generate_container_start())
        html_parts.append(self._generate_page_header(summaries, timestamp))
        html_parts.append(self._generate_executive_summary(summaries))
        html_parts.append(self._generate_smart_insights(summaries))
        html_parts.append(self._generate_detailed_results_table(summaries))
        html_parts.append(self._generate_statistical_analysis(summaries))
        html_parts.append(self._generate_visualization_charts(chart1_base64, chart2_base64))
        html_parts.append(self._generate_interactive_charts(summaries))
        html_parts.append(self._generate_raw_data_export(summaries, timestamp))
        html_parts.append(self._generate_footer())
        html_parts.append(self._generate_javascript())
        
        return '\n'.join(html_parts)
    
    def _img_to_base64(self, img_path: Path) -> str:
        """图片转base64"""
        if img_path.exists():
            with open(img_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        return ""
    
    def _generate_header(self) -> str:
        """生成HTML头部和CSS"""
        return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TD3 Vehicle Sweep Experiment Report (Enhanced)</title>
    
    <!-- Plotly.js for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    
    <style>
        /* ==================== CSS Variables ==================== */
        :root {
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --text-color: #333;
            --bg-color: #ffffff;
            --section-bg: #f8f9fa;
            --border-color: #dee2e6;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
            --shadow-hover: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        [data-theme="dark"] {
            --text-color: #e0e0e0;
            --bg-color: #1a1a1a;
            --section-bg: #2d2d2d;
            --border-color: #404040;
        }
        
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            transition: all 0.3s ease;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto 0 250px;
            background: var(--bg-color);
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .container.nav-collapsed {
            margin-left: 20px;
        }
        
        /* Floating Navigation */
        .floating-nav {
            position: fixed;
            left: 20px;
            top: 20px;
            width: 220px;
            background: var(--bg-color);
            border-radius: 15px;
            box-shadow: var(--shadow-hover);
            padding: 20px;
            max-height: calc(100vh - 40px);
            overflow-y: auto;
            z-index: 1000;
            transition: all 0.3s ease;
        }
        
        .floating-nav.collapsed {
            width: 60px;
            padding: 15px;
        }
        
        .nav-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }
        
        .nav-title {
            font-weight: 700;
            color: var(--primary-color);
        }
        
        .nav-toggle {
            background: none;
            border: none;
            font-size: 1.3em;
            cursor: pointer;
            color: var(--text-color);
        }
        
        .nav-links {
            list-style: none;
        }
        
        .nav-link {
            display: block;
            padding: 10px 15px;
            margin: 5px 0;
            color: var(--text-color);
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s ease;
            border-left: 3px solid transparent;
        }
        
        .nav-link:hover {
            background: var(--section-bg);
            border-left-color: var(--primary-color);
            padding-left: 20px;
        }
        
        .nav-link.active {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
        }
        
        /* Back to Top Button */
        .back-to-top {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 50%;
            font-size: 1.5em;
            cursor: pointer;
            box-shadow: var(--shadow-hover);
            opacity: 0;
            visibility: hidden;
            transition: all 0.3s ease;
            z-index: 999;
        }
        
        .back-to-top.visible {
            opacity: 1;
            visibility: visible;
        }
        
        .back-to-top:hover {
            transform: translateY(-5px) scale(1.1);
        }
        
        /* Header and Toolbar */
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header .subtitle {
            font-size: 1.2em;
            opacity: 0.9;
        }
        
        .toolbar {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
        }
        
        .toolbar-btn {
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.4);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .toolbar-btn:hover {
            background: rgba(255,255,255,0.3);
            transform: translateY(-2px);
        }
        
        /* Content Sections */
        .content {
            padding: 40px;
        }
        
        .section {
            margin-bottom: 40px;
            padding: 30px;
            background: var(--section-bg);
            border-radius: 10px;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }
        
        .section-title {
            font-size: 1.8em;
            color: var(--primary-color);
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid var(--primary-color);
            cursor: pointer;
            user-select: none;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .section-title:hover {
            color: var(--secondary-color);
        }
        
        .section-content {
            transition: max-height 0.3s ease, opacity 0.3s ease;
            overflow: hidden;
        }
        
        .section-content.collapsed {
            max-height: 0 !important;
            opacity: 0;
        }
        
        /* Metrics Grid */
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        
        .metric-card {
            background: var(--bg-color);
            padding: 20px;
            border-radius: 8px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--primary-color);
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: var(--shadow-hover);
        }
        
        .metric-label {
            font-size: 0.9em;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 5px;
        }
        
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }
        
        .metric-unit {
            font-size: 0.5em;
            opacity: 0.6;
        }
        
        /* Tables */
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-color);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }
        
        th {
            background: var(--primary-color);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }
        
        td {
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-color);
        }
        
        tr:hover {
            background: var(--section-bg);
        }
        
        /* Charts */
        .chart-container {
            margin: 30px 0;
            text-align: center;
            position: relative;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: var(--shadow);
        }
        
        .chart-title {
            font-size: 1.2em;
            margin-bottom: 15px;
            color: var(--text-color);
            font-weight: 600;
        }
        
        /* Insight Cards */
        .insight-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid var(--primary-color);
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
        }
        
        .insight-card.success {
            border-left-color: var(--success-color);
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(76, 175, 80, 0.1) 100%);
        }
        
        .insight-card.warning {
            border-left-color: var(--warning-color);
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
        }
        
        .insight-title {
            font-size: 1.1em;
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--text-color);
        }
        
        .insight-content {
            font-size: 0.95em;
            color: var(--text-color);
            line-height: 1.8;
        }
        
        /* Export Buttons */
        .export-buttons {
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }
        
        .export-btn {
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .export-btn:hover {
            background: var(--secondary-color);
            transform: translateY(-2px);
        }
        
        /* Footer */
        .footer {
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        /* Responsive */
        @media (max-width: 1200px) {
            .container {
                margin-left: 20px;
            }
            .floating-nav {
                transform: translateX(-100%);
            }
            .floating-nav:hover {
                transform: translateX(0);
            }
        }
        
        @media print {
            .floating-nav, .back-to-top, .toolbar {
                display: none;
            }
            body {
                background: white;
            }
            .container {
                margin-left: 0;
                box-shadow: none;
            }
        }
    </style>
</head>
"""
    
    def _generate_navigation(self) -> str:
        """生成浮动导航"""
        return """<body>
    <nav class="floating-nav" id="floatingNav">
        <div class="nav-header">
            <span class="nav-title">📑 目录</span>
            <button class="nav-toggle" id="navToggle">☰</button>
        </div>
        <ul class="nav-links" id="navLinks">
            <!-- 由JavaScript动态生成 -->
        </ul>
    </nav>
"""
    
    def _generate_back_to_top(self) -> str:
        """生成返回顶部按钮"""
        return """    <button class="back-to-top" id="backToTop">↑</button>
"""
    
    def _generate_container_start(self) -> str:
        """生成容器开始标签"""
        return """    <div class="container">
"""
    
    def _generate_page_header(self, summaries: List[Dict], timestamp: str) -> str:
        """生成页面头部"""
        return f"""        <div class="header">
            <div class="toolbar">
                <button class="toolbar-btn" id="darkModeToggle">🌙 深色</button>
                <button class="toolbar-btn" onclick="exportTableCSV()">📊 导出CSV</button>
                <button class="toolbar-btn" onclick="exportJSON()">📥 导出JSON</button>
                <button class="toolbar-btn" onclick="window.print()">🖨️ 打印</button>
            </div>
            <h1>🚗 TD3 Vehicle Sweep Experiment (Enhanced)</h1>
            <div class="subtitle">Scalability Analysis | {len(summaries)} Configurations | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
        </div>
        
        <div class="content">
"""
    
    def _generate_executive_summary(self, summaries: List[Dict]) -> str:
        """生成执行摘要"""
        vehicles = [s['num_vehicles'] for s in summaries]
        total_time = sum(s['training_time_hours'] for s in summaries)
        
        return f"""            <div class="section" id="section-0">
                <h2 class="section-title">📊 Executive Summary</h2>
                <div class="section-content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Configurations Tested</div>
                            <div class="metric-value">{len(summaries)}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Vehicle Range</div>
                            <div class="metric-value">{min(vehicles)}-{max(vehicles)}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Episodes per Config</div>
                            <div class="metric-value">{summaries[0]['episodes']}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Total Training Time</div>
                            <div class="metric-value">{total_time:.2f} <span class="metric-unit">hours</span></div>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    def _generate_smart_insights(self, summaries: List[Dict]) -> str:
        """生成智能分析洞察"""
        # 分析数据
        best_delay_idx = min(range(len(summaries)), key=lambda i: summaries[i]['avg_delay'])
        best_completion_idx = max(range(len(summaries)), key=lambda i: summaries[i]['avg_completion'])
        best_reward_idx = max(range(len(summaries)), key=lambda i: summaries[i]['avg_step_reward'])
        
        delay_increase = ((summaries[-1]['avg_delay'] - summaries[0]['avg_delay']) / summaries[0]['avg_delay'] * 100) if summaries[0]['avg_delay'] > 0 else 0
        completion_change = ((summaries[-1]['avg_completion'] - summaries[0]['avg_completion']) * 100) if len(summaries) > 1 else 0
        time_scaling = summaries[-1]['training_time_hours'] / summaries[0]['training_time_hours'] if summaries[0]['training_time_hours'] > 0 else 1
        vehicle_scaling = summaries[-1]['num_vehicles'] / summaries[0]['num_vehicles']
        
        # 性能评估
        all_high_completion = all(s['avg_completion'] > 0.90 for s in summaries)
        manageable_delay = delay_increase < 50
        
        insight_level = "success" if (all_high_completion and manageable_delay) else "warning"
        
        return f"""            <div class="section" id="section-1">
                <h2 class="section-title">🤖 Smart Insights</h2>
                <div class="section-content">
                    <div class="insight-card success">
                        <div class="insight-title">🏆 Best Configurations</div>
                        <div class="insight-content">
                            <ul style="margin-left: 20px; line-height: 2;">
                                <li><strong>Lowest Delay:</strong> {summaries[best_delay_idx]['num_vehicles']} vehicles ({summaries[best_delay_idx]['avg_delay']:.4f}s)</li>
                                <li><strong>Highest Completion Rate:</strong> {summaries[best_completion_idx]['num_vehicles']} vehicles ({summaries[best_completion_idx]['avg_completion']:.2%})</li>
                                <li><strong>Best Reward:</strong> {summaries[best_reward_idx]['num_vehicles']} vehicles ({summaries[best_reward_idx]['avg_step_reward']:.4f})</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="insight-card {insight_level}">
                        <div class="insight-title">📈 Scalability Analysis</div>
                        <div class="insight-content">
                            <ul style="margin-left: 20px; line-height: 2;">
                                <li><strong>Delay Growth:</strong> From {summaries[0]['avg_delay']:.4f}s to {summaries[-1]['avg_delay']:.4f}s ({delay_increase:+.1f}% change)</li>
                                <li><strong>Completion Rate Change:</strong> {completion_change:+.2f} percentage points</li>
                                <li><strong>Training Time Scaling:</strong> {time_scaling:.2f}x increase for {vehicle_scaling:.2f}x vehicles</li>
                                <li><strong>Computational Complexity:</strong> O(n^{{{time_scaling/vehicle_scaling:.2f}}}) scaling observed</li>
                            </ul>
                        </div>
                    </div>
                    
                    <div class="insight-card">
                        <div class="insight-title">💡 Recommendations</div>
                        <div class="insight-content">
                            <ul style="margin-left: 20px; line-height: 2;">
"""
        
        # 智能建议
        recommendations_html = ""
        if all_high_completion:
            recommendations_html += "                                <li>✅ <strong>Excellent scalability</strong>: System maintains >90% completion rate across all scales.</li>\n"
        else:
            recommendations_html += "                                <li>⚠️ <strong>Scalability concern</strong>: Completion rate drops below 90% at higher scales.</li>\n"
        
        if manageable_delay:
            recommendations_html += "                                <li>✅ <strong>Manageable delay growth</strong>: <50% increase shows good scalability.</li>\n"
        else:
            recommendations_html += "                                <li>⚠️ <strong>Significant delay increase</strong>: >50% growth may indicate capacity limits.</li>\n"
        
        recommendations_html += f"                                <li>📊 <strong>Optimal configuration</strong>: {summaries[best_completion_idx]['num_vehicles']} vehicles for best balance.</li>\n"
        recommendations_html += "                                <li>🔬 <strong>Further experiments</strong>: Consider testing edge cases and different RSU/UAV configurations.</li>\n"
        
        return recommendations_html + """                            </ul>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    def _generate_detailed_results_table(self, summaries: List[Dict]) -> str:
        """生成详细结果表格"""
        table_html = """            <div class="section" id="section-2">
                <h2 class="section-title">📋 Detailed Results</h2>
                <div class="section-content">
                    <div class="export-buttons">
                        <button class="export-btn" onclick="exportTableCSV()">📊 Export as CSV</button>
                        <button class="export-btn" onclick="copyTableData()">📋 Copy Data</button>
                    </div>
                    
                    <table id="resultsTable">
                        <thead>
                            <tr>
                                <th>Vehicles</th>
                                <th>State Dim</th>
                                <th>Episodes</th>
                                <th>Training Time</th>
                                <th>Avg Step Reward</th>
                                <th>Avg Delay (s)</th>
                                <th>Completion Rate</th>
                            </tr>
                        </thead>
                        <tbody>
"""
        
        for summary in summaries:
            table_html += f"""                            <tr>
                                <td><strong>{summary['num_vehicles']}</strong></td>
                                <td>{summary['state_dim']}</td>
                                <td>{summary['episodes']}</td>
                                <td>{summary['training_time_hours']:.3f}h</td>
                                <td>{summary['avg_step_reward']:.4f}</td>
                                <td>{summary['avg_delay']:.4f}</td>
                                <td>{summary['avg_completion']:.2%}</td>
                            </tr>
"""
        
        table_html += """                        </tbody>
                    </table>
                </div>
            </div>
"""
        return table_html
    
    def _generate_statistical_analysis(self, summaries: List[Dict]) -> str:
        """生成统计分析"""
        import numpy as np
        
        delays = [s['avg_delay'] for s in summaries]
        completions = [s['avg_completion'] for s in summaries]
        rewards = [s['avg_step_reward'] for s in summaries]
        
        return f"""            <div class="section" id="section-3">
                <h2 class="section-title">📊 Statistical Analysis</h2>
                <div class="section-content">
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-label">Delay: Mean ± Std</div>
                            <div class="metric-value">{np.mean(delays):.3f} ± {np.std(delays):.3f}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Completion: Mean ± Std</div>
                            <div class="metric-value">{np.mean(completions)*100:.1f}% ± {np.std(completions)*100:.1f}%</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Reward: Mean ± Std</div>
                            <div class="metric-value">{np.mean(rewards):.3f} ± {np.std(rewards):.3f}</div>
                        </div>
                        
                        <div class="metric-card">
                            <div class="metric-label">Coefficient of Variation</div>
                            <div class="metric-value">{np.std(delays)/np.mean(delays)*100:.1f}%</div>
                        </div>
                    </div>
                </div>
            </div>
"""
    
    def _generate_visualization_charts(self, chart1_base64: str, chart2_base64: str) -> str:
        """生成可视化图表章节"""
        charts_html = """            <div class="section" id="section-4">
                <h2 class="section-title">📈 Visualization Charts</h2>
                <div class="section-content">
"""
        
        if chart1_base64:
            charts_html += f"""                    <div class="chart-container">
                        <div class="chart-title">Comprehensive Comparison (6 Subplots)</div>
                        <img src="data:image/png;base64,{chart1_base64}" alt="Comprehensive Comparison">
                    </div>
"""
        
        if chart2_base64:
            charts_html += f"""                    <div class="chart-container">
                        <div class="chart-title">Bar Chart Comparison</div>
                        <img src="data:image/png;base64,{chart2_base64}" alt="Bar Chart Comparison">
                    </div>
"""
        
        charts_html += """                </div>
            </div>
"""
        return charts_html
    
    def _generate_interactive_charts(self, summaries: List[Dict]) -> str:
        """生成交互式图表（Plotly）"""
        vehicles = [s['num_vehicles'] for s in summaries]
        delays = [s['avg_delay'] for s in summaries]
        completions = [s['avg_completion'] * 100 for s in summaries]
        rewards = [s['avg_step_reward'] for s in summaries]
        
        # 生成JSON数据供JavaScript使用
        chart_data = {
            'vehicles': vehicles,
            'delays': delays,
            'completions': completions,
            'rewards': rewards
        }
        
        return f"""            <div class="section" id="section-5">
                <h2 class="section-title">🎯 Interactive Analysis</h2>
                <div class="section-content">
                    <p style="margin-bottom: 20px; color: var(--text-color);">
                        交互式图表：鼠标悬停查看数值，双击重置视图，拖拽缩放区域
                    </p>
                    
                    <div id="interactiveChart1" style="width:100%; height:500px;"></div>
                    <div id="interactiveChart2" style="width:100%; height:500px; margin-top: 30px;"></div>
                    
                    <script>
                        var chartData = {json.dumps(chart_data)};
                        
                        // Chart 1: Delay and Completion Rate
                        var trace1 = {{
                            x: chartData.vehicles,
                            y: chartData.delays,
                            name: 'Avg Delay (s)',
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: {{color: '#D55E00', size: 10}},
                            line: {{width: 3}}
                        }};
                        
                        var trace2 = {{
                            x: chartData.vehicles,
                            y: chartData.completions,
                            name: 'Completion Rate (%)',
                            type: 'scatter',
                            mode: 'lines+markers',
                            yaxis: 'y2',
                            marker: {{color: '#029E73', size: 10}},
                            line: {{width: 3}}
                        }};
                        
                        var layout1 = {{
                            title: 'Delay and Completion Rate vs Vehicle Count',
                            xaxis: {{title: 'Number of Vehicles'}},
                            yaxis: {{title: 'Average Delay (s)', titlefont: {{color: '#D55E00'}}}},
                            yaxis2: {{
                                title: 'Completion Rate (%)',
                                titlefont: {{color: '#029E73'}},
                                overlaying: 'y',
                                side: 'right'
                            }},
                            hovermode: 'x unified',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)'
                        }};
                        
                        Plotly.newPlot('interactiveChart1', [trace1, trace2], layout1, {{responsive: true}});
                        
                        // Chart 2: Reward Trend
                        var trace3 = {{
                            x: chartData.vehicles,
                            y: chartData.rewards,
                            name: 'Avg Step Reward',
                            type: 'scatter',
                            mode: 'lines+markers',
                            marker: {{color: '#0173B2', size: 10}},
                            line: {{width: 3}},
                            fill: 'tozeroy',
                            fillcolor: 'rgba(1, 115, 178, 0.2)'
                        }};
                        
                        var layout2 = {{
                            title: 'Average Step Reward vs Vehicle Count',
                            xaxis: {{title: 'Number of Vehicles'}},
                            yaxis: {{title: 'Average Step Reward'}},
                            hovermode: 'closest',
                            plot_bgcolor: 'rgba(0,0,0,0)',
                            paper_bgcolor: 'rgba(0,0,0,0)'
                        }};
                        
                        Plotly.newPlot('interactiveChart2', [trace3], layout2, {{responsive: true}});
                    </script>
                </div>
            </div>
"""
    
    def _generate_raw_data_export(self, summaries: List[Dict], timestamp: str) -> str:
        """生成原始数据导出章节"""
        return f"""            <div class="section" id="section-6">
                <h2 class="section-title">💾 Raw Data Export</h2>
                <div class="section-content">
                    <p style="margin-bottom: 15px; color: var(--text-color);">
                        下载完整的实验数据用于进一步分析
                    </p>
                    
                    <div class="export-buttons">
                        <button class="export-btn" onclick="downloadJSON()">📥 Download JSON</button>
                        <button class="export-btn" onclick="downloadCSV()">📊 Download CSV</button>
                        <button class="export-btn" onclick="copyRawData()">📋 Copy to Clipboard</button>
                    </div>
                    
                    <textarea id="rawData" style="width:100%; height:200px; margin-top:15px; padding:10px; border-radius:8px; border:1px solid var(--border-color); font-family:monospace; font-size:0.9em; background:var(--bg-color); color:var(--text-color);" readonly>{json.dumps(summaries, indent=2)}</textarea>
                </div>
            </div>
"""
    
    def _generate_footer(self) -> str:
        """生成页脚"""
        return """        </div>
        
        <div class="footer">
            <p>VEC Migration Caching System - Vehicle Sweep Experiment Report (Enhanced)</p>
            <p>Generated by TD3 Scalability Analysis Tool v2.0</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                © 2025 All Rights Reserved
            </p>
        </div>
    </div>
"""
    
    def _generate_javascript(self) -> str:
        """生成JavaScript功能"""
        return """    <script>
        // Initialize on page load
        document.addEventListener('DOMContentLoaded', function() {
            initDarkMode();
            initNavigation();
            initBackToTop();
            initSectionToggle();
            initExportFunctions();
        });
        
        // Dark Mode
        function initDarkMode() {
            const btn = document.getElementById('darkModeToggle');
            const html = document.documentElement;
            
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {
                html.setAttribute('data-theme', savedTheme);
                updateDarkModeIcon(savedTheme === 'dark');
            }
            
            if (btn) {
                btn.addEventListener('click', function() {
                    const isDark = html.getAttribute('data-theme') === 'dark';
                    const newTheme = isDark ? 'light' : 'dark';
                    html.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    updateDarkModeIcon(!isDark);
                });
            }
        }
        
        function updateDarkModeIcon(isDark) {
            const btn = document.getElementById('darkModeToggle');
            if (btn) btn.textContent = isDark ? '☀️ 浅色' : '🌙 深色';
        }
        
        // Navigation
        function initNavigation() {
            const sections = document.querySelectorAll('.section');
            const navLinks = document.getElementById('navLinks');
            
            if (navLinks) {
                sections.forEach((section, index) => {
                    const title = section.querySelector('.section-title');
                    if (title) {
                        const titleText = title.textContent.replace(/[▼▶]/g, '').trim();
                        const sectionId = section.id || `section-${index}`;
                        section.id = sectionId;
                        
                        const li = document.createElement('li');
                        const a = document.createElement('a');
                        a.href = `#${sectionId}`;
                        a.className = 'nav-link';
                        a.textContent = titleText;
                        a.addEventListener('click', function(e) {
                            e.preventDefault();
                            section.scrollIntoView({ behavior: 'smooth' });
                        });
                        li.appendChild(a);
                        navLinks.appendChild(li);
                    }
                });
            }
            
            const navToggle = document.getElementById('navToggle');
            const nav = document.querySelector('.floating-nav');
            const container = document.querySelector('.container');
            
            if (navToggle) {
                navToggle.addEventListener('click', function() {
                    nav.classList.toggle('collapsed');
                    container.classList.toggle('nav-collapsed');
                });
            }
        }
        
        // Back to Top
        function initBackToTop() {
            const btn = document.getElementById('backToTop');
            
            window.addEventListener('scroll', function() {
                if (window.pageYOffset > 300) {
                    btn.classList.add('visible');
                } else {
                    btn.classList.remove('visible');
                }
            });
            
            if (btn) {
                btn.addEventListener('click', function() {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                });
            }
        }
        
        // Section Toggle
        function initSectionToggle() {
            const sectionTitles = document.querySelectorAll('.section-title');
            
            sectionTitles.forEach(title => {
                const icon = document.createElement('span');
                icon.textContent = '▼';
                icon.style.fontSize = '0.7em';
                title.appendChild(icon);
                
                const content = title.nextElementSibling;
                if (content && content.classList.contains('section-content')) {
                    title.addEventListener('click', function() {
                        content.classList.toggle('collapsed');
                        icon.textContent = content.classList.contains('collapsed') ? '▶' : '▼';
                    });
                }
            });
        }
        
        // Export Functions
        function initExportFunctions() {
            window.exportTableCSV = function() {
                const table = document.getElementById('resultsTable');
                if (!table) return;
                
                const csv = [];
                const rows = table.querySelectorAll('tr');
                
                rows.forEach(row => {
                    const cols = row.querySelectorAll('td, th');
                    const csvRow = [];
                    cols.forEach(col => csvRow.push(col.textContent));
                    csv.push(csvRow.join(','));
                });
                
                downloadFile(csv.join('\\n'), 'vehicle_sweep_results.csv', 'text/csv');
            };
            
            window.downloadJSON = function() {
                const data = document.getElementById('rawData').value;
                downloadFile(data, 'vehicle_sweep_data.json', 'application/json');
            };
            
            window.downloadCSV = function() {
                exportTableCSV();
            };
            
            window.exportJSON = function() {
                const data = {
                    experiment: 'TD3 Vehicle Sweep',
                    timestamp: new Date().toISOString(),
                    note: 'Exported from HTML report'
                };
                downloadFile(JSON.stringify(data, null, 2), 'experiment_info.json', 'application/json');
            };
            
            window.copyTableData = function() {
                const table = document.getElementById('resultsTable');
                const text = Array.from(table.querySelectorAll('tr'))
                    .map(row => Array.from(row.querySelectorAll('td, th'))
                        .map(cell => cell.textContent).join('\\t'))
                    .join('\\n');
                navigator.clipboard.writeText(text).then(() => {
                    alert('Table data copied to clipboard!');
                });
            };
            
            window.copyRawData = function() {
                const textarea = document.getElementById('rawData');
                textarea.select();
                document.execCommand('copy');
                alert('Raw data copied to clipboard!');
            };
        }
        
        function downloadFile(content, filename, mimeType) {
            const blob = new Blob([content], { type: mimeType });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    </script>
</body>
</html>
"""
    
    def save_report(self, html_content: str, filepath: Path) -> bool:
        """保存HTML报告"""
        try:
            with filepath.open('w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Error saving HTML report: {e}")
            return False
