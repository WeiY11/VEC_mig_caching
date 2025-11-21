"""
HTML训练报告生成器
生成全面详细的训练结果HTML报告，包含可视化图表、性能指标和系统统计
"""
import os
import json
import base64
import io
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False


class HTMLReportGenerator:
    """HTML训练报告生成器"""
    
    def __init__(self):
        self.report_sections = []
        
    def generate_full_report(self, 
                           algorithm: str,
                           training_env: Any,
                           training_time: float,
                           results: Dict,
                           simulator_stats: Optional[Dict] = None) -> str:
        """
        生成完整的HTML报告
        
        Args:
            algorithm: 算法名称
            training_env: 训练环境对象
            training_time: 训练总时间（秒）
            results: 训练结果字典
            simulator_stats: 仿真器统计信息
            
        Returns:
            HTML报告内容字符串
        """
        html_parts = []
        
        # 添加HTML头部和样式
        html_parts.append(self._generate_html_header(algorithm))
        
        # 1. 执行摘要
        html_parts.append(self._generate_executive_summary(algorithm, training_env, training_time, results))
        
        # 🆕 2. 智能分析洞察（提前）
        html_parts.append(self._generate_smart_insights(algorithm, training_env, results))
        
        # 🆕 3. 训练曲线可视化（移到前面⭐）
        html_parts.append(self._generate_training_charts(algorithm, training_env))

        # 🆕 4. 交互式图表分析（移到前面⭐）
        html_parts.append(self._generate_interactive_charts(algorithm, training_env))
        
        # 🆕 5. 阶段性能对比（移到前面⭐）
        html_parts.append(self._generate_phase_comparison(training_env))
        
        # 🆕 6. 统计分析详情（移到前面⭐）
        html_parts.append(self._generate_statistical_details(training_env))
        
        # 7. 性能指标总览
        html_parts.append(self._generate_performance_overview(training_env, results))
        
        # 8. 训练配置
        html_parts.append(self._generate_training_config(results))

        # 9. 系统参数总览
        html_parts.append(self._generate_system_parameters(results))

        # 10. 网络配置参数
        html_parts.append(self._generate_network_parameters(results))

        # 11. 计算能力参数
        html_parts.append(self._generate_compute_parameters(results))

        # 12. 任务和迁移参数
        html_parts.append(self._generate_task_migration_parameters(results))

        # 13. 奖励函数参数
        html_parts.append(self._generate_reward_parameters(results))

        # 14. 算法配置参数
        html_parts.append(self._generate_algorithm_parameters(results))

        # 15. 详细指标分析
        html_parts.append(self._generate_detailed_metrics(training_env))

        # 12. 算法超参数和网络架构
        html_parts.append(self._generate_algorithm_details(algorithm, training_env))

        # 13. 训练过程深度分析
        html_parts.append(self._generate_training_analysis(training_env, results))

        # 14. 每轮详细数据表格
        html_parts.append(self._generate_episode_data_table(training_env, results))
        
        # 15. 系统统计信息
        if simulator_stats:
            html_parts.append(self._generate_system_statistics(simulator_stats))

        # 16. 自适应控制器统计
        html_parts.append(self._generate_adaptive_controller_stats(training_env))

        # 17. 收敛性分析
        html_parts.append(self._generate_convergence_analysis(training_env))

        # 18. 指标相关性分析（新增）
        html_parts.append(self._generate_correlation_analysis(training_env))

        # 19. 逐指标趋势分析（新增）
        html_parts.append(self._generate_per_metric_analysis(training_env))

        # 20. 性能雷达图和对比（新增）
        html_parts.append(self._generate_radar_chart_analysis(training_env, results))

        # 21. 完整数据导出表格（新增）
        html_parts.append(self._generate_complete_data_table(training_env))

        # 22. 峰值和异常分析（新增）
        html_parts.append(self._generate_peak_anomaly_analysis(training_env))

        # 23. 学习曲线平滑度分析（新增）
        html_parts.append(self._generate_smoothness_analysis(training_env))

        # 24. 建议和结论
        html_parts.append(self._generate_recommendations(training_env, results))
        
        # 添加HTML尾部
        html_parts.append(self._generate_html_footer())
        
        return '\n'.join(html_parts)
    
    def _generate_html_header(self, algorithm: str) -> str:
        """生成HTML头部和CSS样式（增强版 - 包含导航、深色模式、交互功能）"""
        return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{algorithm} 训练报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    
    <!-- 🆕 Plotly.js for interactive charts -->
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js" charset="utf-8"></script>
    
    <style>
        /* ==================== 基础样式 ==================== */
        :root {{
            --primary-color: #667eea;
            --secondary-color: #764ba2;
            --success-color: #28a745;
            --warning-color: #ffc107;
            --danger-color: #dc3545;
            --bg-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --text-color: #333;
            --bg-color: #ffffff;
            --section-bg: #f8f9fa;
            --border-color: #dee2e6;
            --shadow: 0 2px 10px rgba(0,0,0,0.1);
            --shadow-hover: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        /* 🌙 深色模式变量 */
        [data-theme="dark"] {{
            --text-color: #e0e0e0;
            --bg-color: #1a1a1a;
            --section-bg: #2d2d2d;
            --border-color: #404040;
            --shadow: 0 2px 10px rgba(0,0,0,0.3);
            --shadow-hover: 0 5px 15px rgba(0,0,0,0.5);
        }}
        
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background: var(--bg-gradient);
            padding: 20px;
            transition: background-color 0.3s ease, color 0.3s ease;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto 0 250px;  /* 🆕 为左侧导航留空间 */
            background: var(--bg-color);
            border-radius: 15px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
            transition: margin-left 0.3s ease, background-color 0.3s ease;
        }}
        
        /* 🆕 导航栏收起时的样式 */
        .container.nav-collapsed {{
            margin-left: 20px;
        }}
        
        .header {{
            background: var(--bg-gradient);
            color: white;
            padding: 40px;
            text-align: center;
            position: relative;
        }}
        
        .header h1 {{
            font-size: 2.5em;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
        }}
        
        /* 🆕 工具栏（深色模式、导出等按钮） */
        .toolbar {{
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            gap: 10px;
            z-index: 100;
        }}
        
        .toolbar-btn {{
            background: rgba(255,255,255,0.2);
            border: 2px solid rgba(255,255,255,0.4);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s ease;
            backdrop-filter: blur(10px);
        }}
        
        .toolbar-btn:hover {{
            background: rgba(255,255,255,0.3);
            border-color: rgba(255,255,255,0.6);
            transform: translateY(-2px);
        }}
        
        .toolbar-btn i {{
            margin-right: 5px;
        }}
        
        .content {{
            padding: 40px;
        }}
        
        .section {{
            margin-bottom: 40px;
            padding: 30px;
            background: var(--section-bg);
            border-radius: 10px;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }}
        
        /* 🆕 章节折叠功能 */
        .section-title {{
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
            transition: all 0.3s ease;
        }}
        
        .section-title:hover {{
            color: var(--secondary-color);
            border-bottom-color: var(--secondary-color);
        }}
        
        .section-title .toggle-icon {{
            font-size: 0.7em;
            transition: transform 0.3s ease;
        }}
        
        .section-title.collapsed .toggle-icon {{
            transform: rotate(-90deg);
        }}
        
        .section-content {{
            transition: max-height 0.3s ease, opacity 0.3s ease;
            overflow: hidden;
        }}
        
        .section-content.collapsed {{
            max-height: 0 !important;
            opacity: 0;
            margin: 0;
            padding: 0;
        }}
        
        .section-subtitle {{
            font-size: 1.3em;
            color: var(--secondary-color);
            margin: 25px 0 15px 0;
        }}
        
        /* 🆕 浮动导航栏 */
        .floating-nav {{
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
        }}
        
        .floating-nav.collapsed {{
            width: 60px;
            padding: 15px;
        }}
        
        .floating-nav.collapsed .nav-title,
        .floating-nav.collapsed .nav-links {{
            display: none;
        }}
        
        .nav-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 2px solid var(--border-color);
        }}
        
        .nav-title {{
            font-weight: 700;
            color: var(--primary-color);
            font-size: 1.1em;
        }}
        
        .nav-toggle {{
            background: none;
            border: none;
            font-size: 1.3em;
            cursor: pointer;
            color: var(--text-color);
            padding: 5px;
            transition: transform 0.3s ease;
        }}
        
        .nav-toggle:hover {{
            transform: scale(1.1);
        }}
        
        .nav-links {{
            list-style: none;
        }}
        
        .nav-link {{
            display: block;
            padding: 10px 15px;
            margin: 5px 0;
            color: var(--text-color);
            text-decoration: none;
            border-radius: 8px;
            transition: all 0.2s ease;
            font-size: 0.95em;
            border-left: 3px solid transparent;
        }}
        
        .nav-link:hover {{
            background: var(--section-bg);
            border-left-color: var(--primary-color);
            padding-left: 20px;
        }}
        
        .nav-link.active {{
            background: var(--bg-gradient);
            color: white;
            font-weight: 600;
            border-left-color: white;
        }}
        
        /* 🆕 返回顶部按钮 */
        .back-to-top {{
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 50px;
            height: 50px;
            background: var(--bg-gradient);
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
        }}
        
        .back-to-top.visible {{
            opacity: 1;
            visibility: visible;
        }}
        
        .back-to-top:hover {{
            transform: translateY(-5px) scale(1.1);
        }}
        
        /* 🆕 数据导出按钮组 */
        .export-buttons {{
            display: flex;
            gap: 10px;
            margin: 15px 0;
            flex-wrap: wrap;
        }}
        
        .export-btn {{
            background: var(--primary-color);
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 0.9em;
            font-weight: 600;
            transition: all 0.3s ease;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }}
        
        .export-btn:hover {{
            background: var(--secondary-color);
            transform: translateY(-2px);
            box-shadow: var(--shadow-hover);
        }}
        
        .export-btn:active {{
            transform: translateY(0);
        }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        
        .metric-card {{
            background: var(--bg-color);
            padding: 20px;
            border-radius: 8px;
            box-shadow: var(--shadow);
            border-left: 4px solid var(--primary-color);
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            transform: translateY(-5px);
            box-shadow: var(--shadow-hover);
        }}
        
        .metric-label {{
            font-size: 0.9em;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 5px;
        }}
        
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: var(--primary-color);
        }}
        
        .metric-unit {{
            font-size: 0.5em;
            color: var(--text-color);
            opacity: 0.6;
        }}
        
        .chart-container {{
            margin: 30px 0;
            text-align: center;
            position: relative;
        }}
        
        /* 🆕 图表下载按钮 */
        .chart-download {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(102, 126, 234, 0.9);
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 0.85em;
            opacity: 0;
            transition: opacity 0.3s ease;
        }}
        
        .chart-container:hover .chart-download {{
            opacity: 1;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: var(--shadow);
            transition: all 0.3s ease;
        }}
        
        .chart-container img:hover {{
            box-shadow: var(--shadow-hover);
        }}
        
        .chart-title {{
            font-size: 1.2em;
            margin-bottom: 15px;
            color: var(--text-color);
            font-weight: 600;
        }}
        
        /* 🆕 交互式图表容器 */
        .plotly-chart {{
            margin: 30px 0;
            background: var(--bg-color);
            border-radius: 8px;
            padding: 15px;
            box-shadow: var(--shadow);
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            background: var(--bg-color);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: var(--shadow);
        }}
        
        th {{
            background: var(--primary-color);
            color: white;
            padding: 15px;
            text-align: left;
            font-weight: 600;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid var(--border-color);
            color: var(--text-color);
        }}
        
        tr:hover {{
            background: var(--section-bg);
        }}
        
        .status-badge {{
            display: inline-block;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .status-success {{
            background: #d4edda;
            color: #155724;
        }}
        
        .status-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .status-info {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .progress-bar {{
            width: 100%;
            height: 30px;
            background: #e9ecef;
            border-radius: 15px;
            overflow: hidden;
            margin: 10px 0;
        }}
        
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: 600;
            transition: width 0.3s ease;
        }}
        
        .recommendation {{
            background: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 15px;
            margin: 15px 0;
            border-radius: 5px;
        }}
        
        .recommendation-title {{
            font-weight: 600;
            color: #856404;
            margin-bottom: 5px;
        }}
        
        .footer {{
            background: #2c3e50;
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .footer a {{
            color: #667eea;
            text-decoration: none;
        }}
        
        .data-table {{
            overflow-x: auto;
        }}
        
        .highlight {{
            background: linear-gradient(120deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2px 8px;
            border-radius: 4px;
        }}
        
        /* 🆕 智能分析卡片 */
        .insight-card {{
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border-left: 4px solid var(--primary-color);
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: var(--shadow);
        }}
        
        .insight-card.warning {{
            background: linear-gradient(135deg, rgba(255, 193, 7, 0.1) 0%, rgba(255, 152, 0, 0.1) 100%);
            border-left-color: var(--warning-color);
        }}
        
        .insight-card.success {{
            background: linear-gradient(135deg, rgba(40, 167, 69, 0.1) 0%, rgba(76, 175, 80, 0.1) 100%);
            border-left-color: var(--success-color);
        }}
        
        .insight-card.danger {{
            background: linear-gradient(135deg, rgba(220, 53, 69, 0.1) 0%, rgba(255, 87, 34, 0.1) 100%);
            border-left-color: var(--danger-color);
        }}
        
        .insight-title {{
            font-size: 1.1em;
            font-weight: 700;
            margin-bottom: 10px;
            color: var(--text-color);
        }}
        
        .insight-content {{
            font-size: 0.95em;
            color: var(--text-color);
            line-height: 1.8;
        }}
        
        /* 🆕 评级指示器 */
        .rating {{
            display: inline-flex;
            align-items: center;
            gap: 5px;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            font-weight: 600;
        }}
        
        .rating.excellent {{
            background: #d4edda;
            color: #155724;
        }}
        
        .rating.good {{
            background: #d1ecf1;
            color: #0c5460;
        }}
        
        .rating.fair {{
            background: #fff3cd;
            color: #856404;
        }}
        
        .rating.poor {{
            background: #f8d7da;
            color: #721c24;
        }}
        
        /* 🆕 异常标记 */
        .anomaly-marker {{
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: var(--danger-color);
            animation: pulse 2s infinite;
        }}
        
        @keyframes pulse {{
            0%, 100% {{
                opacity: 1;
            }}
            50% {{
                opacity: 0.3;
            }}
        }}
        
        /* 🆕 性能对比表 */
        .comparison-table {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 15px;
            margin: 20px 0;
        }}
        
        .comparison-item {{
            background: var(--bg-color);
            padding: 15px;
            border-radius: 8px;
            box-shadow: var(--shadow);
            text-align: center;
        }}
        
        .comparison-label {{
            font-size: 0.85em;
            color: var(--text-color);
            opacity: 0.7;
            margin-bottom: 8px;
        }}
        
        .comparison-value {{
            font-size: 1.5em;
            font-weight: 700;
            color: var(--primary-color);
        }}
        
        /* 🆕 加载动画 */
        .loading-spinner {{
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(102, 126, 234, 0.3);
            border-top-color: var(--primary-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        /* 🆕 响应式设计 */
        @media (max-width: 1200px) {{
            .container {{
                margin-left: 20px;
            }}
            
            .floating-nav {{
                transform: translateX(-100%);
            }}
            
            .floating-nav:hover {{
                transform: translateX(0);
            }}
        }}
        
        @media (max-width: 768px) {{
            .toolbar {{
                flex-direction: column;
                gap: 5px;
            }}
            
            .metrics-grid {{
                grid-template-columns: 1fr;
            }}
            
            .comparison-table {{
                grid-template-columns: 1fr;
            }}
        }}
        
        @media print {{
            body {{
                background: white;
            }}
            .container {{
                box-shadow: none;
            }}
            .metric-card {{
                break-inside: avoid;
            }}
        }}
    </style>
    
    <script>
        /* ==================== JavaScript功能 ==================== */
        
        // 页面加载完成后初始化
        document.addEventListener('DOMContentLoaded', function() {{
            initDarkMode();
            initNavigation();
            initBackToTop();
            initSectionToggle();
            initExportFunctions();
            initLazyLoading();
            initSmartAnalysis();
        }});
        
        // 🌙 深色模式
        function initDarkMode() {{
            const darkModeBtn = document.getElementById('darkModeToggle');
            const html = document.documentElement;
            
            // 检查本地存储的主题设置
            const savedTheme = localStorage.getItem('theme');
            if (savedTheme) {{
                html.setAttribute('data-theme', savedTheme);
                updateDarkModeIcon(savedTheme === 'dark');
            }}
            
            if (darkModeBtn) {{
                darkModeBtn.addEventListener('click', function() {{
                    const isDark = html.getAttribute('data-theme') === 'dark';
                    const newTheme = isDark ? 'light' : 'dark';
                    html.setAttribute('data-theme', newTheme);
                    localStorage.setItem('theme', newTheme);
                    updateDarkModeIcon(!isDark);
                }});
            }}
        }}
        
        function updateDarkModeIcon(isDark) {{
            const btn = document.getElementById('darkModeToggle');
            if (btn) {{
                btn.textContent = isDark ? '☀️ 浅色' : '🌙 深色';
            }}
        }}
        
        // 🧭 导航功能
        function initNavigation() {{
            // 生成导航链接
            const sections = document.querySelectorAll('.section');
            const navLinks = document.getElementById('navLinks');
            
            if (navLinks) {{
                sections.forEach((section, index) => {{
                    const title = section.querySelector('.section-title');
                    if (title) {{
                        const titleText = title.textContent.replace(/[▼▶]/g, '').trim();
                        const sectionId = `section-${{index}}`;
                        section.id = sectionId;
                        
                        const li = document.createElement('li');
                        const a = document.createElement('a');
                        a.href = `#${{sectionId}}`;
                        a.className = 'nav-link';
                        a.textContent = titleText;
                        a.addEventListener('click', function(e) {{
                            e.preventDefault();
                            section.scrollIntoView({{ behavior: 'smooth' }});
                            updateActiveNav();
                        }});
                        li.appendChild(a);
                        navLinks.appendChild(li);
                    }}
                }});
            }}
            
            // 导航栏折叠/展开
            const navToggle = document.getElementById('navToggle');
            const floatingNav = document.querySelector('.floating-nav');
            const container = document.querySelector('.container');
            
            if (navToggle && floatingNav) {{
                navToggle.addEventListener('click', function() {{
                    floatingNav.classList.toggle('collapsed');
                    container.classList.toggle('nav-collapsed');
                }});
            }}
            
            // 滚动时更新导航高亮
            window.addEventListener('scroll', updateActiveNav);
        }}
        
        function updateActiveNav() {{
            const sections = document.querySelectorAll('.section');
            const navLinks = document.querySelectorAll('.nav-link');
            
            let currentSection = '';
            sections.forEach(section => {{
                const sectionTop = section.offsetTop;
                const sectionHeight = section.clientHeight;
                if (window.pageYOffset >= sectionTop - 100) {{
                    currentSection = section.getAttribute('id');
                }}
            }});
            
            navLinks.forEach(link => {{
                link.classList.remove('active');
                if (link.getAttribute('href') === '#' + currentSection) {{
                    link.classList.add('active');
                }}
            }});
        }}
        
        // ⬆️ 返回顶部
        function initBackToTop() {{
            const backToTopBtn = document.getElementById('backToTop');
            
            window.addEventListener('scroll', function() {{
                if (window.pageYOffset > 300) {{
                    backToTopBtn.classList.add('visible');
                }} else {{
                    backToTopBtn.classList.remove('visible');
                }}
            }});
            
            if (backToTopBtn) {{
                backToTopBtn.addEventListener('click', function() {{
                    window.scrollTo({{ top: 0, behavior: 'smooth' }});
                }});
            }}
        }}
        
        // 📁 章节折叠/展开
        function initSectionToggle() {{
            const sectionTitles = document.querySelectorAll('.section-title');
            
            sectionTitles.forEach(title => {{
                // 添加折叠图标
                const icon = document.createElement('span');
                icon.className = 'toggle-icon';
                icon.textContent = '▼';
                title.appendChild(icon);
                
                // 获取章节内容
                const section = title.parentElement;
                const content = Array.from(section.children).filter(el => el !== title);
                
                // 创建内容包装器
                const contentWrapper = document.createElement('div');
                contentWrapper.className = 'section-content';
                content.forEach(el => contentWrapper.appendChild(el));
                section.appendChild(contentWrapper);
                
                // 点击标题折叠/展开
                title.addEventListener('click', function() {{
                    title.classList.toggle('collapsed');
                    contentWrapper.classList.toggle('collapsed');
                }});
            }});
        }}
        
        // 📤 导出功能
        function initExportFunctions() {{
            // CSV导出
            window.exportTableToCSV = function(tableId, filename) {{
                const table = document.getElementById(tableId);
                if (!table) return;
                
                const csv = [];
                const rows = table.querySelectorAll('tr');
                
                rows.forEach(row => {{
                    const cols = row.querySelectorAll('td, th');
                    const csvRow = [];
                    cols.forEach(col => csvRow.push(col.textContent));
                    csv.push(csvRow.join(','));
                }});
                
                downloadFile(csv.join('\\n'), filename, 'text/csv');
            }};
            
            // JSON导出
            window.exportJSON = function() {{
                const data = {{
                    algorithm: '{algorithm}',
                    generatedAt: new Date().toISOString(),
                    // 这里可以添加更多数据
                }};
                downloadFile(JSON.stringify(data, null, 2), 'training_report.json', 'application/json');
            }};
            
            // 图表下载
            window.downloadChart = function(imgElement) {{
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                const img = new Image();
                img.src = imgElement.src;
                img.onload = function() {{
                    canvas.width = img.width;
                    canvas.height = img.height;
                    ctx.drawImage(img, 0, 0);
                    canvas.toBlob(function(blob) {{
                        const url = URL.createObjectURL(blob);
                        const a = document.createElement('a');
                        a.href = url;
                        a.download = 'chart_{{Date.now()}}.png';
                        a.click();
                        URL.revokeObjectURL(url);
                    }});
                }};
            }};
            
            // 打印优化
            window.optimizedPrint = function() {{
                window.print();
            }};
        }}
        
        function downloadFile(content, filename, mimeType) {{
            const blob = new Blob([content], {{ type: mimeType }});
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }}
        
        // 🖼️ 图片懒加载
        function initLazyLoading() {{
            const images = document.querySelectorAll('img[data-src]');
            
            const imageObserver = new IntersectionObserver((entries, observer) => {{
                entries.forEach(entry => {{
                    if (entry.isIntersecting) {{
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.removeAttribute('data-src');
                        observer.unobserve(img);
                    }}
                }});
            }});
            
            images.forEach(img => imageObserver.observe(img));
        }}
        
        // 🤖 智能分析（简化版 - 基于规则）
        function initSmartAnalysis() {{
            // 这个函数会在报告生成时由Python代码填充实际的分析逻辑
            console.log('Smart analysis initialized');
        }}
        
        // 🎨 动态生成Plotly图表的辅助函数
        window.createInteractiveChart = function(divId, data, layout, config) {{
            if (typeof Plotly !== 'undefined') {{
                Plotly.newPlot(divId, data, layout, config);
            }} else {{
                console.warn('Plotly is not loaded');
            }}
        }};
    </script>
</head>
<body>
    <!-- 🆕 浮动导航栏 -->
    <nav class="floating-nav" id="floatingNav">
        <div class="nav-header">
            <span class="nav-title">📑 目录</span>
            <button class="nav-toggle" id="navToggle">☰</button>
        </div>
        <ul class="nav-links" id="navLinks">
            <!-- 导航链接将由JavaScript动态生成 -->
        </ul>
    </nav>
    
    <!-- 🆕 返回顶部按钮 -->
    <button class="back-to-top" id="backToTop">↑</button>
    
    <div class="container">
        <div class="header">
            <!-- 🆕 工具栏 -->
            <div class="toolbar">
                <button class="toolbar-btn" id="darkModeToggle">🌙 深色</button>
                <button class="toolbar-btn" onclick="optimizedPrint()">🖨️ 打印</button>
                <button class="toolbar-btn" onclick="exportJSON()">📥 导出JSON</button>
            </div>
            <h1>🚀 {algorithm} 训练报告（增强版）</h1>
            <div class="subtitle">生成时间: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}</div>
        </div>
        <div class="content">
"""
    
    def _generate_executive_summary(self, algorithm: str, training_env: Any, 
                                    training_time: float, results: Dict) -> str:
        """生成执行摘要"""
        final_perf = results.get('final_performance', {})
        avg_reward = final_perf.get('avg_reward', 0)
        avg_delay = final_perf.get('avg_delay', 0)
        avg_completion = final_perf.get('avg_completion', 0)
        
        num_episodes = len(training_env.episode_rewards)
        training_hours = training_time / 3600
        
        # 计算改进幅度（基于Episode总奖励）
        initial_reward = training_env.episode_rewards[0] if training_env.episode_rewards else 0
        final_reward = training_env.episode_rewards[-1] if training_env.episode_rewards else 0
        # 注意：负值奖励，越大越好（-100改进到-50是100%改进）
        improvement = ((final_reward - initial_reward) / abs(initial_reward) * 100) if initial_reward != 0 else 0
        
        return f"""
        <div class="section">
            <h2 class="section-title">📊 执行摘要</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">算法类型</div>
                    <div class="metric-value">{algorithm}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">训练轮次</div>
                    <div class="metric-value">{num_episodes}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">训练时长</div>
                    <div class="metric-value">{training_hours:.2f} <span class="metric-unit">小时</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">平均每轮时间</div>
                    <div class="metric-value">{training_time/num_episodes if num_episodes > 0 else 0:.2f} <span class="metric-unit">秒</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">🎯 最终性能指标</h3>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Episode总奖励</div>
                    <div class="metric-value">{final_perf.get('avg_episode_reward', avg_reward * 100):.2f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">训练优化目标</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">每步平均奖励</div>
                    <div class="metric-value">{avg_reward:.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">便于对比评估</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">平均时延</div>
                    <div class="metric-value">{avg_delay:.3f} <span class="metric-unit">秒</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">任务完成率</div>
                    <div class="metric-value">{avg_completion*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">性能改进</div>
                    <div class="metric-value" style="color: {'#28a745' if improvement > 0 else '#dc3545'}">
                        {improvement:+.1f} <span class="metric-unit">%</span>
                    </div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">基于Episode总奖励</div>
                </div>
            </div>
            
            <div style="margin-top: 30px;">
                <div class="metric-label">训练完成度</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: 100%;">100%</div>
                </div>
            </div>
        </div>
"""
    
    def _generate_smart_insights(self, algorithm: str, training_env: Any, results: Dict) -> str:
        """
        🤖 生成智能分析洞察
        基于训练数据自动生成性能评语、异常检测、收敛评级和优化建议
        """
        insights_html = []
        insights_html.append("""
        <div class="section">
            <h2 class="section-title">🤖 智能分析洞察</h2>
            <p class="metric-description">基于训练数据的自动化分析和建议</p>
""")
        
        # 分析训练数据
        rewards = training_env.episode_rewards
        if not rewards:
            return ""
        
        # 1. 收敛性分析
        convergence_analysis = self._analyze_convergence(rewards)
        insights_html.append(f"""
            <div class="insight-card {convergence_analysis['level']}">
                <div class="insight-title">📈 收敛性评估: <span class="rating {convergence_analysis['rating']}">{convergence_analysis['rating_text']}</span></div>
                <div class="insight-content">
                    {convergence_analysis['description']}
                </div>
            </div>
""")
        
        # 2. 性能评级
        performance_rating = self._evaluate_performance(training_env, results)
        insights_html.append(f"""
            <div class="insight-card {performance_rating['level']}">
                <div class="insight-title">⭐ 性能评级: <span class="rating {performance_rating['rating']}">{performance_rating['rating_text']}</span></div>
                <div class="insight-content">
                    {performance_rating['description']}
                </div>
            </div>
""")
        
        # 3. 异常检测
        anomalies = self._detect_anomalies(rewards)
        if anomalies['count'] > 0:
            insights_html.append(f"""
            <div class="insight-card warning">
                <div class="insight-title">⚠️ 异常检测: 发现 {anomalies['count']} 个异常Episode</div>
                <div class="insight-content">
                    {anomalies['description']}
                </div>
            </div>
""")
        
        # 4. 优化建议
        recommendations = self._generate_smart_recommendations(algorithm, training_env, results)
        insights_html.append(f"""
            <div class="insight-card">
                <div class="insight-title">💡 优化建议</div>
                <div class="insight-content">
                    <ul style="margin-left: 20px; line-height: 2;">
""")
        for rec in recommendations:
            insights_html.append(f"                        <li>{rec}</li>\n")
        
        insights_html.append("""
                    </ul>
                </div>
            </div>
        </div>
""")
        
        return '\n'.join(insights_html)
    
    def _analyze_convergence(self, rewards: List[float]) -> Dict:
        """分析收敛性"""
        if len(rewards) < 20:
            return {
                'rating': 'fair',
                'rating_text': '数据不足',
                'level': 'warning',
                'description': '训练轮次较少，无法准确评估收敛性。建议至少训练100轮以上。'
            }
        
        # 计算后期稳定性（最后20%的方差）
        last_20_percent = rewards[-len(rewards)//5:]
        variance = np.var(last_20_percent)
        mean_reward = np.mean(last_20_percent)
        cv = np.sqrt(variance) / abs(mean_reward) if mean_reward != 0 else float('inf')
        
        # 计算改进趋势
        first_half = np.mean(rewards[:len(rewards)//2])
        second_half = np.mean(rewards[len(rewards)//2:])
        improvement = ((second_half - first_half) / abs(first_half) * 100) if first_half != 0 else 0
        
        # 评级
        if cv < 0.1 and improvement > 10:
            return {
                'rating': 'excellent',
                'rating_text': '优秀',
                'level': 'success',
                'description': f'✅ 算法收敛良好，后期稳定性高（变异系数: {cv:.3f}）。性能提升显著（{improvement:.1f}%），建议保存当前模型。'
            }
        elif cv < 0.2 and improvement > 5:
            return {
                'rating': 'good',
                'rating_text': '良好',
                'level': 'success',
                'description': f'✅ 算法基本收敛（变异系数: {cv:.3f}），性能有所提升（{improvement:.1f}%）。可以继续训练或进行超参数微调。'
            }
        elif cv < 0.3:
            return {
                'rating': 'fair',
                'rating_text': '一般',
                'level': 'warning',
                'description': f'⚠️ 算法收敛缓慢（变异系数: {cv:.3f}），性能提升有限（{improvement:.1f}%）。建议检查学习率、奖励函数设计或增加训练轮次。'
            }
        else:
            return {
                'rating': 'poor',
                'rating_text': '较差',
                'level': 'danger',
                'description': f'❌ 算法未收敛（变异系数: {cv:.3f}），性能波动较大。建议降低学习率、检查环境稳定性或更换算法。'
            }
    
    def _evaluate_performance(self, training_env: Any, results: Dict) -> Dict:
        """评估整体性能"""
        final_perf = results.get('final_performance', {})
        completion_rate = final_perf.get('avg_completion', 0)
        avg_delay = final_perf.get('avg_delay', float('inf'))
        
        # 综合评分
        score = 0
        details = []
        
        if completion_rate > 0.95:
            score += 40
            details.append(f'任务完成率优秀（{completion_rate*100:.1f}%）')
        elif completion_rate > 0.9:
            score += 30
            details.append(f'任务完成率良好（{completion_rate*100:.1f}%）')
        else:
            score += 20
            details.append(f'任务完成率需提升（{completion_rate*100:.1f}%）')
        
        if avg_delay < 2.0:
            score += 30
            details.append(f'平均时延优秀（{avg_delay:.2f}s）')
        elif avg_delay < 5.0:
            score += 20
            details.append(f'平均时延良好（{avg_delay:.2f}s）')
        else:
            score += 10
            details.append(f'平均时延较高（{avg_delay:.2f}s）')
        
        # 根据分数评级
        if score >= 60:
            return {
                'rating': 'excellent',
                'rating_text': f'优秀（{score}/100分）',
                'level': 'success',
                'description': '🎉 ' + '；'.join(details) + '。系统性能表现优异！'
            }
        elif score >= 45:
            return {
                'rating': 'good',
                'rating_text': f'良好（{score}/100分）',
                'level': 'success',
                'description': '👍 ' + '；'.join(details) + '。系统性能达到预期目标。'
            }
        elif score >= 30:
            return {
                'rating': 'fair',
                'rating_text': f'一般（{score}/100分）',
                'level': 'warning',
                'description': '⚠️ ' + '；'.join(details) + '。系统性能有待提升。'
            }
        else:
            return {
                'rating': 'poor',
                'rating_text': f'较差（{score}/100分）',
                'level': 'danger',
                'description': '❌ ' + '；'.join(details) + '。系统性能需要优化。'
            }
    
    def _detect_anomalies(self, rewards: List[float]) -> Dict:
        """检测异常Episode"""
        if len(rewards) < 10:
            return {'count': 0, 'description': '数据不足，无法检测异常。'}
        
        mean = np.mean(rewards)
        std = np.std(rewards)
        
        # 异常定义：超过3个标准差
        anomalies = []
        for i, reward in enumerate(rewards):
            if abs(reward - mean) > 3 * std:
                anomalies.append((i+1, reward))
        
        if len(anomalies) == 0:
            return {'count': 0, 'description': '未检测到显著异常。'}
        
        anomaly_list = ', '.join([f'Episode {ep}' for ep, _ in anomalies[:5]])
        if len(anomalies) > 5:
            anomaly_list += f' 等{len(anomalies)}个'
        
        return {
            'count': len(anomalies),
            'description': f'在 {anomaly_list} 检测到异常表现（偏离均值超过3σ）。这可能是由于：<br>' +
                          '• 环境随机性导致的极端情况<br>' +
                          '• 探索策略产生的随机动作<br>' +
                          '• 系统状态的罕见配置<br>' +
                          '建议检查这些Episode的详细日志以确定原因。'
        }
    
    def _generate_smart_recommendations(self, algorithm: str, training_env: Any, results: Dict) -> List[str]:
        """生成智能优化建议（用于智能分析洞察章节）"""
        recommendations = []
        
        rewards = training_env.episode_rewards
        if not rewards:
            return ['训练数据不足，无法生成建议。']
        
        # 基于收敛性的建议
        last_episodes = rewards  # 使用全量奖励序列以生成完整曲线
        variance = np.var(last_episodes)
        mean_reward = np.mean(last_episodes)
        
        if variance / (mean_reward ** 2) > 0.1:
            recommendations.append('🔧 <strong>减小学习率</strong>：后期训练波动较大，建议将学习率降低至当前的50%以提高稳定性。')
        
        # 基于性能的建议
        final_perf = results.get('final_performance', {})
        completion_rate = final_perf.get('avg_completion', 0)
        
        if completion_rate < 0.9:
            recommendations.append(f'⚠️ <strong>提升任务完成率</strong>：当前完成率{completion_rate * 100:.1f}%，建议增加dropped_tasks的惩罚权重或优化资源分配策略。')
        
        # 基于算法的建议
        if algorithm in ['TD3', 'DDPG']:
            recommendations.append('🎯 <strong>探索策略优化</strong>：考虑调整噪声参数（policy_noise、noise_clip）以平衡探索与利用。')
        elif algorithm == 'SAC':
            recommendations.append('🌡️ <strong>温度参数调节</strong>：SAC算法的熵温度系数影响探索程度，建议根据收敛情况调整alpha值。')
        elif algorithm == 'PPO':
            recommendations.append('📊 <strong>批次大小优化</strong>：PPO对批次大小敏感，当前batch_size可能需要调整以提高样本效率。')
        
        # 通用建议
        if len(rewards) < 200:
            recommendations.append(f'⏱️ <strong>增加训练轮次</strong>：当前训练{len(rewards)}轮，建议至少训练200-500轮以充分收敛。')
        
        recommendations.append('💾 <strong>保存检查点</strong>：定期保存训练检查点，以便在性能下降时回滚到最佳模型。')
        recommendations.append('📈 <strong>对比实验</strong>：与其他算法（DDPG、SAC、PPO等）进行对比实验，验证当前算法的优势。')
        
        return recommendations
    
    def _generate_training_config(self, results: Dict) -> str:
        """生成训练配置信息"""
        config = results.get('training_config', {})
        
        return f"""
        <div class="section">
            <h2 class="section-title">⚙️ 训练配置</h2>
            
            <table>
                <thead>
                    <tr>
                        <th>配置项</th>
                        <th>值</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>训练轮次</td>
                        <td><span class="highlight">{config.get('num_episodes', 'N/A')}</span></td>
                    </tr>
                    <tr>
                        <td>每轮最大步数</td>
                        <td>{config.get('max_steps_per_episode', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>训练总时长</td>
                        <td>{config.get('training_time_hours', 0):.3f} 小时</td>
                    </tr>
                    <tr>
                        <td>开始时间</td>
                        <td>{results.get('training_start_time', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>智能体类型</td>
                        <td><span class="status-badge status-info">{results.get('agent_type', 'single_agent')}</span></td>
                    </tr>
                </tbody>
            </table>
        </div>
"""
    
    def _generate_performance_overview(self, training_env: Any, results: Dict) -> str:
        """生成性能总览"""
        metrics = training_env.episode_metrics
        
        # 计算统计信息
        def calc_stats(data_list):
            if not data_list:
                return {'mean': 0, 'std': 0, 'min': 0, 'max': 0}
            arr = np.array(data_list)
            return {
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr))
            }
        
        delay_stats = calc_stats(metrics.get('avg_delay', []))
        energy_stats = calc_stats(metrics.get('total_energy', []))
        completion_stats = calc_stats(metrics.get('task_completion_rate', []))
        cache_hit_stats = calc_stats(metrics.get('cache_hit_rate', []))
        
        return f"""
        <div class="section">
            <h2 class="section-title">📈 性能总览</h2>
            
            <h3 class="section-subtitle">平均任务时延</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{delay_stats['mean']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{delay_stats['std']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{delay_stats['min']:.3f} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{delay_stats['max']:.3f} <span class="metric-unit">秒</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">总能耗</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{energy_stats['mean']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{energy_stats['std']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{energy_stats['min']:.1f} <span class="metric-unit">J</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{energy_stats['max']:.1f} <span class="metric-unit">J</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">任务完成率</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{completion_stats['mean']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{completion_stats['std']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{completion_stats['min']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{completion_stats['max']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">缓存命中率</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平均值</div>
                    <div class="metric-value">{cache_hit_stats['mean']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">标准差</div>
                    <div class="metric-value">{cache_hit_stats['std']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{cache_hit_stats['min']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{cache_hit_stats['max']*100:.2f} <span class="metric-unit">%</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_training_charts(self, algorithm: str, training_env: Any) -> str:
        """生成训练曲线图表"""
        charts_html = []
        
        charts_html.append(f"""
        <div class="section">
            <h2 class="section-title">📊 训练曲线可视化</h2>
            <p class="metric-description">包含Per-Step级别的详细训练曲线和Episode级别的汇总指标</p>
""")
        
        # 🆕 0. 检查并嵌入已生成的训练总览图（Per-Step版本）
        external_charts = self._embed_external_charts(algorithm)
        if external_charts:
            charts_html.append(external_charts)
        
        # 1. 奖励曲线
        if training_env.episode_rewards:
            reward_chart = self._create_reward_chart(training_env.episode_rewards)
            charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">奖励演化曲线 (Episode级别)</div>
                <img src="data:image/png;base64,{reward_chart}" alt="奖励曲线">
            </div>
""")
        
        # 2. 多指标对比图
        multi_metric_chart = self._create_multi_metric_chart(training_env.episode_metrics)
        charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">关键性能指标演化 (Episode级别)</div>
                <img src="data:image/png;base64,{multi_metric_chart}" alt="多指标对比">
            </div>
""")
        
        # 3. 能耗和时延对比
        energy_delay_chart = self._create_energy_delay_chart(training_env.episode_metrics)
        charts_html.append(f"""
            <div class="chart-container">
                <div class="chart-title">能耗与时延权衡分析 (Episode级别)</div>
                <img src="data:image/png;base64,{energy_delay_chart}" alt="能耗时延">
            </div>
        </div>
""")
        
        return '\n'.join(charts_html)
    
    def _embed_external_charts(self, algorithm: str) -> str:
        """
        嵌入已生成的训练图表（training_overview.png 和 objective_analysis.png）
        
        Args:
            algorithm: 算法名称（如TD3, DDPG等）
            
        Returns:
            包含嵌入图表的HTML字符串，如果图表不存在则返回空字符串
        """
        charts_html = []
        algorithm_lower = algorithm.lower()
        
        # 查找图表文件的可能位置
        possible_paths = [
            f"results/single_agent/{algorithm_lower}",
            f"results/multi_agent/{algorithm_lower}",
            f"results/{algorithm_lower}",
        ]
        
        chart_files = {
            'training_overview.png': '训练总览 - Per-Step详细分析',
            'objective_analysis.png': '优化目标分析 - 时延与能耗'
        }
        
        found_charts = {}
        
        # 搜索图表文件
        for chart_file, chart_title in chart_files.items():
            for base_path in possible_paths:
                chart_path = os.path.join(base_path, chart_file)
                if os.path.exists(chart_path):
                    # 读取图片并转换为base64
                    try:
                        with open(chart_path, 'rb') as f:
                            img_data = base64.b64encode(f.read()).decode('utf-8')
                            found_charts[chart_file] = {
                                'title': chart_title,
                                'data': img_data,
                                'path': chart_path
                            }
                        break
                    except Exception as e:
                        print(f"⚠️  无法读取图表 {chart_path}: {e}")
                        continue
        
        # 如果找到了图表，生成HTML
        if found_charts:
            charts_html.append("""
            <div class="subsection">
                <h3 class="section-subtitle">🎯 Per-Step级别训练曲线</h3>
                <p class="metric-description">
                    以下图表展示了每个训练步骤(step)的平均性能指标，相比Episode级别的聚合数据，
                    Per-Step分析能够更细致地揭示算法的学习动态和收敛特性。
                </p>
""")
            
            # 嵌入找到的图表
            for chart_file, chart_info in found_charts.items():
                charts_html.append(f"""
                <div class="chart-container" style="margin-top: 20px;">
                    <div class="chart-title" style="font-size: 1.1em; color: #764ba2;">
                        {chart_info['title']}
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-bottom: 10px;">
                        📂 来源: {chart_info['path']}
                    </div>
                    <img src="data:image/png;base64,{chart_info['data']}" 
                         alt="{chart_info['title']}" 
                         style="max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                </div>
""")
            
            charts_html.append("""
            </div>
            <hr style="margin: 30px 0; border: none; border-top: 2px solid #eee;">
""")
        
        return '\n'.join(charts_html)
    
    def _generate_interactive_charts(self, algorithm: str, training_env: Any) -> str:
        """
        🆕 生成交互式图表（Plotly.js）
        提供可缩放、悬停显示数值的动态图表
        """
        import json
        
        # 检查是否有足够的数据
        if not training_env.episode_rewards or len(training_env.episode_rewards) < 5:
            return ""  # 数据不足，跳过
        
        # 准备数据
        episodes = list(range(1, len(training_env.episode_rewards) + 1))
        rewards = training_env.episode_rewards
        
        # 提取指标数据
        delays = training_env.episode_metrics.get('avg_delay', [])
        energies = training_env.episode_metrics.get('total_energy', [])
        completions = training_env.episode_metrics.get('task_completion_rate', [])
        cache_hits = training_env.episode_metrics.get('cache_hit_rate', [])
        
        # 构建JSON数据
        chart_data = {
            'episodes': episodes,
            'rewards': rewards,
            'delays': delays[:len(episodes)],
            'energies': energies[:len(episodes)],
            'completions': [c * 100 for c in completions[:len(episodes)]],  # 转为百分比
            'cache_hits': [c * 100 for c in cache_hits[:len(episodes)]]  # 转为百分比
        }
        
        html = f"""
        <div class="section">
            <h2 class="section-title">🎯 Interactive Analysis (Plotly)</h2>
            <p class="metric-description">
                交互式图表：鼠标悬停查看精确数值，双击重置视图，拖拽选择区域缩放
            </p>
            
            <div class="plotly-chart" id="interactiveRewardChart"></div>
            <div class="plotly-chart" id="interactiveMetricsChart"></div>
            
            <script>
                (function() {{
                    var chartData = {json.dumps(chart_data)};
                    
                    // Chart 1: Reward Evolution with Smoothing
                    var rawTrace = {{
                        x: chartData.episodes,
                        y: chartData.rewards,
                        name: 'Raw Reward',
                        type: 'scatter',
                        mode: 'lines',
                        line: {{color: 'rgba(102, 126, 234, 0.3)', width: 1}},
                        hovertemplate: 'Episode %{{x}}<br>Reward: %{{y:.3f}}<extra></extra>'
                    }};
                    
                    // 计算移动平均
                    var window = Math.max(5, Math.floor(chartData.rewards.length / 20));
                    var smoothed = [];
                    for (var i = window - 1; i < chartData.rewards.length; i++) {{
                        var sum = 0;
                        for (var j = 0; j < window; j++) {{
                            sum += chartData.rewards[i - j];
                        }}
                        smoothed.push(sum / window);
                    }}
                    
                    var smoothTrace = {{
                        x: chartData.episodes.slice(window - 1),
                        y: smoothed,
                        name: 'Smoothed (MA-' + window + ')',
                        type: 'scatter',
                        mode: 'lines',
                        line: {{color: '#667eea', width: 3}},
                        hovertemplate: 'Episode %{{x}}<br>Avg Reward: %{{y:.3f}}<extra></extra>'
                    }};
                    
                    var layout1 = {{
                        title: '{algorithm} Reward Evolution (Interactive)',
                        xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                        yaxis: {{title: 'Average Reward', gridcolor: '#e0e0e0'}},
                        hovermode: 'x unified',
                        plot_bgcolor: 'rgba(248, 249, 250, 0.5)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: {{family: 'Segoe UI, sans-serif'}},
                        showlegend: true,
                        legend: {{x: 0.02, y: 0.98}}
                    }};
                    
                    Plotly.newPlot('interactiveRewardChart', [rawTrace, smoothTrace], layout1, {{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                        toImageButtonOptions: {{
                            format: 'png',
                            filename: '{algorithm.lower()}_reward_interactive',
                            height: 600,
                            width: 1200,
                            scale: 2
                        }}
                    }});
                    
                    // Chart 2: Multi-Metric Comparison
                    var delayTrace = {{
                        x: chartData.episodes.slice(0, chartData.delays.length),
                        y: chartData.delays,
                        name: 'Avg Delay (s)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {{color: '#D55E00', size: 4}},
                        line: {{width: 2}},
                        yaxis: 'y1',
                        hovertemplate: 'Delay: %{{y:.4f}}s<extra></extra>'
                    }};
                    
                    var completionTrace = {{
                        x: chartData.episodes.slice(0, chartData.completions.length),
                        y: chartData.completions,
                        name: 'Completion Rate (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {{color: '#029E73', size: 4}},
                        line: {{width: 2}},
                        yaxis: 'y2',
                        hovertemplate: 'Completion: %{{y:.1f}}%<extra></extra>'
                    }};
                    
                    var cacheTrace = {{
                        x: chartData.episodes.slice(0, chartData.cache_hits.length),
                        y: chartData.cache_hits,
                        name: 'Cache Hit Rate (%)',
                        type: 'scatter',
                        mode: 'lines+markers',
                        marker: {{color: '#0173B2', size: 4}},
                        line: {{width: 2}},
                        yaxis: 'y2',
                        hovertemplate: 'Cache Hit: %{{y:.1f}}%<extra></extra>'
                    }};
                    
                    var layout2 = {{
                        title: 'Multi-Metric Evolution (Interactive)',
                        xaxis: {{title: 'Episode', gridcolor: '#e0e0e0'}},
                        yaxis: {{
                            title: 'Delay (s)',
                            titlefont: {{color: '#D55E00'}},
                            tickfont: {{color: '#D55E00'}},
                            gridcolor: '#e0e0e0'
                        }},
                        yaxis2: {{
                            title: 'Rate (%)',
                            titlefont: {{color: '#029E73'}},
                            tickfont: {{color: '#029E73'}},
                            overlaying: 'y',
                            side: 'right'
                        }},
                        hovermode: 'x unified',
                        plot_bgcolor: 'rgba(248, 249, 250, 0.5)',
                        paper_bgcolor: 'rgba(0,0,0,0)',
                        font: {{family: 'Segoe UI, sans-serif'}},
                        showlegend: true,
                        legend: {{x: 0.02, y: 0.98}}
                    }};
                    
                    Plotly.newPlot('interactiveMetricsChart', [delayTrace, completionTrace, cacheTrace], layout2, {{
                        responsive: true,
                        displayModeBar: true,
                        modeBarButtonsToRemove: ['lasso2d', 'select2d'],
                        toImageButtonOptions: {{
                            format: 'png',
                            filename: '{algorithm.lower()}_metrics_interactive',
                            height: 600,
                            width: 1200,
                            scale: 2
                        }}
                    }});
                }})();
            </script>
        </div>
"""
        return html
    
    def _generate_phase_comparison(self, training_env: Any) -> str:
        """
        🆕 生成训练阶段对比分析
        比较训练前期、中期、后期的性能差异
        """
        rewards = training_env.episode_rewards
        if not rewards or len(rewards) < 30:
            return ""  # 数据不足
        
        # 分为三个阶段
        n = len(rewards)
        early = rewards[:n//3]
        middle = rewards[n//3:2*n//3]
        late = rewards[2*n//3:]
        
        # 同样分析指标
        delays = training_env.episode_metrics.get('avg_delay', [])
        completions = training_env.episode_metrics.get('task_completion_rate', [])
        
        early_delay = delays[:n//3] if len(delays) >= n//3 else []
        middle_delay = delays[n//3:2*n//3] if len(delays) >= 2*n//3 else []
        late_delay = delays[2*n//3:] if len(delays) > 2*n//3 else []
        
        early_comp = completions[:n//3] if len(completions) >= n//3 else []
        middle_comp = completions[n//3:2*n//3] if len(completions) >= 2*n//3 else []
        late_comp = completions[2*n//3:] if len(completions) > 2*n//3 else []
        
        # 计算统计量
        def safe_mean(data):
            return np.mean(data) if len(data) > 0 else 0.0
        
        def safe_std(data):
            return np.std(data) if len(data) > 0 else 0.0
        
        def safe_improvement(early_data, late_data):
            early_mean = safe_mean(early_data)
            late_mean = safe_mean(late_data)
            if early_mean != 0:
                return ((late_mean - early_mean) / abs(early_mean) * 100)
            return 0.0
        
        # 计算改进幅度
        reward_improvement = safe_improvement(early, late)
        delay_improvement = -safe_improvement(early_delay, late_delay)  # 时延减少是改进
        completion_improvement = safe_improvement(early_comp, late_comp) * 100  # 百分点
        
        # 评估训练效果
        if reward_improvement > 15:
            training_effectiveness = "excellent"
            effectiveness_text = "优秀"
            effectiveness_desc = "训练效果显著，性能大幅提升"
        elif reward_improvement > 8:
            training_effectiveness = "good"
            effectiveness_text = "良好"
            effectiveness_desc = "训练效果良好，性能稳步提升"
        elif reward_improvement > 3:
            training_effectiveness = "fair"
            effectiveness_text = "一般"
            effectiveness_desc = "训练有效果，但提升有限"
        else:
            training_effectiveness = "poor"
            effectiveness_text = "较差"
            effectiveness_desc = "训练效果不明显，需检查配置"
        
        html = f"""
        <div class="section">
            <h2 class="section-title">📊 Training Phase Comparison</h2>
            <p class="metric-description">
                对比训练前期、中期、后期的性能变化，评估训练效果
            </p>
            
            <div class="insight-card {training_effectiveness}">
                <div class="insight-title">🎯 训练效果评估: <span class="rating {training_effectiveness}">{effectiveness_text}</span></div>
                <div class="insight-content">
                    {effectiveness_desc} - 奖励提升{reward_improvement:+.1f}%
                </div>
            </div>
            
            <div class="comparison-table">
                <div class="comparison-item">
                    <div class="comparison-label">前期 (1-33%)</div>
                    <div class="comparison-value">{safe_mean(early):.3f}</div>
                    <div style="font-size: 0.8em; color: #666;">Reward ± {safe_std(early):.3f}</div>
                </div>
                
                <div class="comparison-item">
                    <div class="comparison-label">中期 (34-66%)</div>
                    <div class="comparison-value">{safe_mean(middle):.3f}</div>
                    <div style="font-size: 0.8em; color: #666;">Reward ± {safe_std(middle):.3f}</div>
                </div>
                
                <div class="comparison-item">
                    <div class="comparison-label">后期 (67-100%)</div>
                    <div class="comparison-value">{safe_mean(late):.3f}</div>
                    <div style="font-size: 0.8em; color: #666;">Reward ± {safe_std(late):.3f}</div>
                </div>
            </div>
            
            <h3 style="margin-top: 30px; color: var(--primary-color);">📈 关键指标改进</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">奖励提升</div>
                    <div class="metric-value" style="color: {'var(--success-color)' if reward_improvement > 0 else 'var(--danger-color)'};">
                        {reward_improvement:+.1f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">时延改进</div>
                    <div class="metric-value" style="color: {'var(--success-color)' if delay_improvement > 0 else 'var(--danger-color)'};">
                        {delay_improvement:+.1f}%
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">完成率变化</div>
                    <div class="metric-value" style="color: {'var(--success-color)' if completion_improvement > 0 else 'var(--danger-color)'};">
                        {completion_improvement:+.2f} <span class="metric-unit">pp</span>
                    </div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _generate_statistical_details(self, training_env: Any) -> str:
        """
        🆕 生成详细的统计分析
        包括分布分析、趋势检验等
        """
        rewards = training_env.episode_rewards
        if not rewards or len(rewards) < 10:
            return ""
        
        from scipy import stats as scipy_stats
        
        # 基础统计
        mean_reward = np.mean(rewards)
        median_reward = np.median(rewards)
        std_reward = np.std(rewards)
        min_reward = np.min(rewards)
        max_reward = np.max(rewards)
        q25 = np.percentile(rewards, 25)
        q75 = np.percentile(rewards, 75)
        
        # 趋势分析（线性回归）
        x = np.arange(len(rewards))
        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(x, rewards)
        
        # 正态性检验
        _, normality_p = scipy_stats.shapiro(rewards[:min(5000, len(rewards))])  # Shapiro-Wilk test
        
        # 趋势评估
        if p_value < 0.05 and slope > 0:
            trend_assessment = "显著上升趋势 ✅"
            trend_color = "success"
        elif p_value < 0.05 and slope < 0:
            trend_assessment = "显著下降趋势 ⚠️"
            trend_color = "warning"
        else:
            trend_assessment = "无显著趋势"
            trend_color = ""
        
        html = f"""
        <div class="section">
            <h2 class="section-title">📊 Statistical Analysis Details</h2>
            
            <h3 style="color: var(--primary-color); margin-bottom: 15px;">描述性统计</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">均值 (Mean)</div>
                    <div class="metric-value">{mean_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">中位数 (Median)</div>
                    <div class="metric-value">{median_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">标准差 (Std)</div>
                    <div class="metric-value">{std_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">变异系数 (CV)</div>
                    <div class="metric-value">{(std_reward/abs(mean_reward)*100 if mean_reward != 0 else 0):.1f}%</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">范围 (Range)</div>
                    <div class="metric-value">{max_reward - min_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">四分位距 (IQR)</div>
                    <div class="metric-value">{q75 - q25:.4f}</div>
                </div>
            </div>
            
            <h3 style="color: var(--primary-color); margin: 30px 0 15px 0;">趋势分析</h3>
            <div class="insight-card {trend_color}">
                <div class="insight-title">📈 线性趋势检验</div>
                <div class="insight-content">
                    <ul style="margin-left: 20px; line-height: 2;">
                        <li><strong>趋势评估:</strong> {trend_assessment}</li>
                        <li><strong>回归斜率:</strong> {slope:.6f} (每episode变化)</li>
                        <li><strong>R² 值:</strong> {r_value**2:.4f} (拟合优度)</li>
                        <li><strong>P-value:</strong> {p_value:.4e} (显著性水平)</li>
                        <li><strong>回归方程:</strong> y = {slope:.4f}x + {intercept:.4f}</li>
                    </ul>
                </div>
            </div>
            
            <h3 style="color: var(--primary-color); margin: 30px 0 15px 0;">分布特征</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">最小值</div>
                    <div class="metric-value">{min_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">25% 分位数</div>
                    <div class="metric-value">{q25:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">50% 分位数</div>
                    <div class="metric-value">{median_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">75% 分位数</div>
                    <div class="metric-value">{q75:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">最大值</div>
                    <div class="metric-value">{max_reward:.4f}</div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-label">正态性检验</div>
                    <div class="metric-value" style="font-size: 1.2em;">
                        {'✅ 正态' if normality_p > 0.05 else '⚠️ 非正态'}
                    </div>
                    <div style="font-size: 0.8em; color: #666;">p = {normality_p:.4f}</div>
                </div>
            </div>
        </div>
"""
        return html
    
    def _create_reward_chart(self, rewards: List[float]) -> str:
        """创建奖励曲线图并返回base64编码"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = range(1, len(rewards) + 1)
        ax.plot(episodes, rewards, label='Episode Reward', color='#667eea', linewidth=2)
        
        # 添加移动平均
        window = min(20, len(rewards) // 5)
        if window > 1:
            moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(rewards) + 1), moving_avg, 
                   label=f'Moving Average ({window})', color='#764ba2', linewidth=2.5)
        
        ax.set_xlabel('Episode', fontsize=12)
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Reward Over Episodes', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return self._fig_to_base64(fig)
    
    def _create_multi_metric_chart(self, metrics: Dict) -> str:
        """创建多指标对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Key Performance Metrics Evolution', fontsize=16, fontweight='bold')
        
        # 1. 任务完成率
        if 'task_completion_rate' in metrics and metrics['task_completion_rate']:
            ax = axes[0, 0]
            data = [x * 100 for x in metrics['task_completion_rate']]
            ax.plot(data, color='#28a745', linewidth=2)
            ax.set_title('Task Completion Rate (%)')
            ax.set_ylabel('Completion Rate (%)')
            ax.grid(True, alpha=0.3)
        
        # 2. 平均时延
        if 'avg_delay' in metrics and metrics['avg_delay']:
            ax = axes[0, 1]
            ax.plot(metrics['avg_delay'], color='#dc3545', linewidth=2)
            ax.set_title('Average Task Delay (s)')
            ax.set_ylabel('Delay (s)')
            ax.grid(True, alpha=0.3)
        
        # 3. 缓存命中率
        if 'cache_hit_rate' in metrics and metrics['cache_hit_rate']:
            ax = axes[1, 0]
            data = [x * 100 for x in metrics['cache_hit_rate']]
            ax.plot(data, color='#17a2b8', linewidth=2)
            ax.set_title('Cache Hit Rate (%)')
            ax.set_ylabel('Hit Rate (%)')
            ax.set_xlabel('Episode')
            ax.grid(True, alpha=0.3)
        
        # 4. 数据丢失率
        if 'data_loss_ratio_bytes' in metrics and metrics['data_loss_ratio_bytes']:
            ax = axes[1, 1]
            data = [x * 100 for x in metrics['data_loss_ratio_bytes']]
            ax.plot(data, color='#ffc107', linewidth=2)
            ax.set_title('Data Loss Ratio (%)')
            ax.set_ylabel('Loss Ratio (%)')
            ax.set_xlabel('Episode')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_energy_delay_chart(self, metrics: Dict) -> str:
        """创建能耗和时延对比图"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 能耗曲线
        if 'total_energy' in metrics and metrics['total_energy']:
            ax1.plot(metrics['total_energy'], color='#ff6b6b', linewidth=2, label='Total Energy')
            ax1.set_ylabel('Energy (J)', fontsize=12)
            ax1.set_title('Energy Consumption Over Episodes', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 时延曲线
        if 'avg_delay' in metrics and metrics['avg_delay']:
            ax2.plot(metrics['avg_delay'], color='#4ecdc4', linewidth=2, label='Avg Delay')
            ax2.set_ylabel('Delay (s)', fontsize=12)
            ax2.set_xlabel('Episode', fontsize=12)
            ax2.set_title('Average Task Delay Over Episodes', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _fig_to_base64(self, fig) -> str:
        """将matplotlib图形转换为base64编码字符串"""
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def _generate_detailed_metrics(self, training_env: Any) -> str:
        """生成详细指标分析"""
        metrics = training_env.episode_metrics
        
        # 计算最近表现 vs 初始表现
        def compare_performance(data_list, window=20):
            if not data_list or len(data_list) < window:
                return "N/A", "N/A", "N/A"
            initial = np.mean(data_list[:window])
            final = np.mean(data_list[-window:])
            improvement = ((final - initial) / abs(initial) * 100) if initial != 0 else 0
            return f"{initial:.4f}", f"{final:.4f}", f"{improvement:+.2f}%"
        
        # 分析各项指标
        delay_initial, delay_final, delay_improve = compare_performance(metrics.get('avg_delay', []))
        energy_initial, energy_final, energy_improve = compare_performance(metrics.get('total_energy', []))
        completion_initial, completion_final, completion_improve = compare_performance(
            metrics.get('task_completion_rate', [])
        )
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔍 详细指标分析</h2>
            
            <h3 class="section-subtitle">性能改进对比（首20轮 vs 末20轮）</h3>
            
            <table>
                <thead>
                    <tr>
                        <th>指标</th>
                        <th>初始表现</th>
                        <th>最终表现</th>
                        <th>改进幅度</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>平均时延</td>
                        <td>{delay_initial} 秒</td>
                        <td>{delay_final} 秒</td>
                        <td><span class="status-badge status-success">{delay_improve}</span></td>
                    </tr>
                    <tr>
                        <td>总能耗</td>
                        <td>{energy_initial} J</td>
                        <td>{energy_final} J</td>
                        <td><span class="status-badge status-success">{energy_improve}</span></td>
                    </tr>
                    <tr>
                        <td>任务完成率</td>
                        <td>{completion_initial}</td>
                        <td>{completion_final}</td>
                        <td><span class="status-badge status-success">{completion_improve}</span></td>
                    </tr>
                </tbody>
            </table>
            
            <h3 class="section-subtitle">训练稳定性分析</h3>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">奖励方差</div>
                    <div class="metric-value">{np.var(training_env.episode_rewards):.2f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">奖励范围</div>
                    <div class="metric-value">
                        {np.max(training_env.episode_rewards) - np.min(training_env.episode_rewards):.2f}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最佳奖励</div>
                    <div class="metric-value">{np.max(training_env.episode_rewards):.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">最差奖励</div>
                    <div class="metric-value">{np.min(training_env.episode_rewards):.3f}</div>
                </div>
            </div>
        </div>
"""
    
    def _generate_system_statistics(self, simulator_stats: Dict) -> str:
        """生成系统统计信息"""
        return f"""
        <div class="section">
            <h2 class="section-title">🖥️ 系统统计信息</h2>
            
            <h3 class="section-subtitle">中央RSU调度器</h3>
            <table>
                <thead>
                    <tr>
                        <th>统计项</th>
                        <th>值</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>调度调用次数</td>
                        <td>{simulator_stats.get('scheduling_calls', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>负载均衡指数</td>
                        <td>{simulator_stats.get('load_balance_index', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>系统健康状态</td>
                        <td><span class="status-badge status-success">{simulator_stats.get('system_health', 'N/A')}</span></td>
                    </tr>
                </tbody>
            </table>
            
            <h3 class="section-subtitle">迁移统计</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">RSU迁移数据量</div>
                    <div class="metric-value">{simulator_stats.get('rsu_migration_data', 0):.1f} <span class="metric-unit">MB</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">车辆跟随迁移</div>
                    <div class="metric-value">{simulator_stats.get('handover_migrations', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">UAV迁移次数</div>
                    <div class="metric-value">{simulator_stats.get('uav_migration_count', 0)}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">回传网络能耗</div>
                    <div class="metric-value">{simulator_stats.get('backhaul_total_energy', 0):.2f} <span class="metric-unit">J</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_algorithm_details(self, algorithm: str, training_env: Any) -> str:
        """生成算法超参数和网络架构详情"""
        # 获取算法特定配置
        algo_params = {}
        
        # 检查是否有agent对象（TD3/DDPG/SAC等）
        agent = None
        if hasattr(training_env, 'agent_env') and hasattr(training_env.agent_env, 'agent'):
            agent = training_env.agent_env.agent
        
        if agent:
            # 网络结构信息
            if hasattr(agent, 'actor') and hasattr(agent.actor, 'fc1'):
                actor = agent.actor
                if hasattr(actor, 'fc1'):
                    algo_params['actor_layer1'] = actor.fc1.out_features if hasattr(actor.fc1, 'out_features') else 'N/A'
                if hasattr(actor, 'fc2'):
                    algo_params['actor_layer2'] = actor.fc2.out_features if hasattr(actor.fc2, 'out_features') else 'N/A'
            
            # 获取学习率等超参数
            if hasattr(agent, 'actor_optimizer'):
                algo_params['actor_lr'] = agent.actor_optimizer.param_groups[0]['lr']
            if hasattr(agent, 'critic_optimizer'):
                algo_params['critic_lr'] = agent.critic_optimizer.param_groups[0]['lr']
            if hasattr(agent, 'config'):
                config = agent.config
                if hasattr(config, 'gamma'):
                    algo_params['gamma'] = config.gamma
                if hasattr(config, 'tau'):
                    algo_params['tau'] = config.tau
                if hasattr(config, 'policy_noise'):
                    algo_params['policy_noise'] = config.policy_noise
                if hasattr(config, 'noise_clip'):
                    algo_params['noise_clip'] = config.noise_clip
                if hasattr(config, 'policy_delay'):
                    algo_params['policy_delay'] = config.policy_delay
        
        params_html = ""
        if algo_params:
            for key, value in algo_params.items():
                params_html += f"""
                <tr>
                    <td>{key.replace('_', ' ').title()}</td>
                    <td><code>{value}</code></td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔧 算法配置详情</h2>
            
            <h3 class="section-subtitle">算法类型</h3>
            <p style="font-size: 1.1em; margin: 15px 0;">
                <span class="highlight">{algorithm}</span> - 
                {'Twin Delayed Deep Deterministic Policy Gradient' if algorithm == 'TD3' else
                 'Deep Deterministic Policy Gradient' if algorithm == 'DDPG' else
                 'Soft Actor-Critic' if algorithm == 'SAC' else
                 'Proximal Policy Optimization' if algorithm == 'PPO' else
                 'Deep Q-Network' if algorithm == 'DQN' else algorithm}
            </p>
            
            <h3 class="section-subtitle">网络架构</h3>
            <table>
                <thead>
                    <tr>
                        <th>组件</th>
                        <th>配置</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>状态维度</td>
                        <td><code>{getattr(training_env.agent_env, 'state_dim', 'N/A')}</code></td>
                    </tr>
                    <tr>
                        <td>动作维度</td>
                        <td><code>{getattr(training_env.agent_env, 'action_dim', 'N/A')}</code></td>
                    </tr>
                    <tr>
                        <td>动作空间类型</td>
                        <td><code>{'连续' if algorithm in ['DDPG', 'TD3', 'SAC', 'PPO'] else '离散'}</code></td>
                    </tr>
                    {params_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">训练技巧</h3>
            <div style="margin: 20px 0; line-height: 2;">
                {'• <strong>目标网络</strong>: 使用软更新 (τ=' + str(algo_params.get('tau', 'N/A')) + ')<br>' if 'tau' in algo_params else ''}
                {'• <strong>延迟策略更新</strong>: 每' + str(algo_params.get('policy_delay', 'N/A')) + '步更新一次Actor<br>' if algorithm == 'TD3' else ''}
                {'• <strong>目标策略平滑</strong>: 噪声=' + str(algo_params.get('policy_noise', 'N/A')) + ', 裁剪=' + str(algo_params.get('noise_clip', 'N/A')) + '<br>' if algorithm == 'TD3' else ''}
                • <strong>经验回放</strong>: 使用Replay Buffer存储经验<br>
                • <strong>批量训练</strong>: 从Replay Buffer采样进行训练
            </div>
        </div>
"""
    
    def _generate_training_analysis(self, training_env: Any, results: Dict) -> str:
        """生成训练过程深度分析"""
        rewards = training_env.episode_rewards
        
        # 分段分析：前25%，中间50%，后25%
        n = len(rewards)
        if n < 4:
            return ""
        
        quarter = n // 4
        early_rewards = rewards[:quarter]
        mid_rewards = rewards[quarter:3*quarter]
        late_rewards = rewards[3*quarter:]
        
        early_avg = np.mean(early_rewards)
        mid_avg = np.mean(mid_rewards)
        late_avg = np.mean(late_rewards)
        
        early_std = np.std(early_rewards)
        mid_std = np.std(mid_rewards)
        late_std = np.std(late_rewards)
        
        # 计算趋势
        improvement_early_to_mid = ((mid_avg - early_avg) / abs(early_avg) * 100) if early_avg != 0 else 0
        improvement_mid_to_late = ((late_avg - mid_avg) / abs(mid_avg) * 100) if mid_avg != 0 else 0
        
        # 检测异常值
        all_rewards = np.array(rewards)
        q1 = np.percentile(all_rewards, 25)
        q3 = np.percentile(all_rewards, 75)
        iqr = q3 - q1
        outliers = np.sum((all_rewards < q1 - 1.5*iqr) | (all_rewards > q3 + 1.5*iqr))
        
        # 生成更多可视化
        phase_chart = self._create_phase_analysis_chart(early_rewards, mid_rewards, late_rewards)
        distribution_chart = self._create_distribution_chart(rewards)
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔬 训练过程深度分析</h2>
            
            <h3 class="section-subtitle">分阶段性能分析</h3>
            <table>
                <thead>
                    <tr>
                        <th>阶段</th>
                        <th>轮次范围</th>
                        <th>平均奖励</th>
                        <th>标准差</th>
                        <th>相对改进</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><span class="status-badge status-info">探索期 (前25%)</span></td>
                        <td>1 - {quarter}</td>
                        <td>{early_avg:.3f}</td>
                        <td>{early_std:.3f}</td>
                        <td>-</td>
                    </tr>
                    <tr>
                        <td><span class="status-badge status-warning">学习期 (中50%)</span></td>
                        <td>{quarter+1} - {3*quarter}</td>
                        <td>{mid_avg:.3f}</td>
                        <td>{mid_std:.3f}</td>
                        <td style="color: {'#28a745' if improvement_early_to_mid > 0 else '#dc3545'}">
                            {improvement_early_to_mid:+.1f}%
                        </td>
                    </tr>
                    <tr>
                        <td><span class="status-badge status-success">收敛期 (后25%)</span></td>
                        <td>{3*quarter+1} - {n}</td>
                        <td>{late_avg:.3f}</td>
                        <td>{late_std:.3f}</td>
                        <td style="color: {'#28a745' if improvement_mid_to_late > 0 else '#dc3545'}">
                            {improvement_mid_to_late:+.1f}%
                        </td>
                    </tr>
                </tbody>
            </table>
            
            <div class="chart-container">
                <div class="chart-title">三阶段奖励分布对比</div>
                <img src="data:image/png;base64,{phase_chart}" alt="阶段分析">
            </div>
            
            <div class="chart-container">
                <div class="chart-title">奖励分布直方图与核密度估计</div>
                <img src="data:image/png;base64,{distribution_chart}" alt="分布图">
            </div>
            
            <h3 class="section-subtitle">统计特征</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">偏度 (Skewness)</div>
                    <div class="metric-value">{self._calculate_skewness(rewards):.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        {self._interpret_skewness(self._calculate_skewness(rewards))}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">峰度 (Kurtosis)</div>
                    <div class="metric-value">{self._calculate_kurtosis(rewards):.3f}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        {self._interpret_kurtosis(self._calculate_kurtosis(rewards))}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">变异系数 (CV)</div>
                    <div class="metric-value">{(np.std(rewards) / abs(np.mean(rewards)) * 100):.2f}<span class="metric-unit">%</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        相对变异程度
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">异常值数量</div>
                    <div class="metric-value">{outliers}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">
                        基于IQR方法检测
                    </div>
                </div>
            </div>
        </div>
"""
    
    def _generate_episode_data_table(self, training_env: Any, results: Dict) -> str:
        """生成每轮详细数据表格"""
        rewards = training_env.episode_rewards
        metrics = training_env.episode_metrics
        
        # 只显示前10轮、中间10轮、最后10轮
        n = len(rewards)
        if n <= 30:
            display_indices = list(range(n))
        else:
            display_indices = (list(range(10)) + 
                             ['...'] + 
                             list(range(n//2 - 5, n//2 + 5)) + 
                             ['...'] + 
                             list(range(n-10, n)))
        
        rows_html = ""
        for idx in display_indices:
            if idx == '...':
                rows_html += """
                <tr style="text-align: center; background: #f0f0f0;">
                    <td colspan="8">...</td>
                </tr>
"""
            else:
                episode_num = idx + 1
                reward = rewards[idx]
                delay = metrics.get('avg_delay', [0])[idx] if idx < len(metrics.get('avg_delay', [])) else 0
                energy = metrics.get('total_energy', [0])[idx] if idx < len(metrics.get('total_energy', [])) else 0
                completion = metrics.get('task_completion_rate', [0])[idx] if idx < len(metrics.get('task_completion_rate', [])) else 0
                cache_hit = metrics.get('cache_hit_rate', [0])[idx] if idx < len(metrics.get('cache_hit_rate', [])) else 0
                data_loss = metrics.get('data_loss_ratio_bytes', [0])[idx] if idx < len(metrics.get('data_loss_ratio_bytes', [])) else 0
                migration_success = metrics.get('migration_success_rate', [0])[idx] if idx < len(metrics.get('migration_success_rate', [])) else 0
                
                # 根据阶段着色
                if idx < n // 4:
                    phase_color = '#e3f2fd'  # 蓝色 - 探索期
                elif idx < 3 * n // 4:
                    phase_color = '#fff3e0'  # 橙色 - 学习期
                else:
                    phase_color = '#e8f5e9'  # 绿色 - 收敛期
                
                rows_html += f"""
                <tr style="background: {phase_color}">
                    <td><strong>{episode_num}</strong></td>
                    <td>{reward:.3f}</td>
                    <td>{delay:.4f}s</td>
                    <td>{energy:.1f}J</td>
                    <td>{completion*100:.2f}%</td>
                    <td>{cache_hit*100:.1f}%</td>
                    <td>{data_loss*100:.2f}%</td>
                    <td>{migration_success*100:.1f}%</td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">📋 每轮详细数据表</h2>
            
            <p style="margin-bottom: 20px; color: #666;">
                完整训练数据记录（显示前10轮、中间10轮、最后10轮）
            </p>
            
            <div class="data-table" style="max-height: 600px; overflow-y: auto;">
                <table>
                    <thead style="position: sticky; top: 0; z-index: 10;">
                        <tr>
                            <th>轮次</th>
                            <th>奖励</th>
                            <th>平均时延</th>
                            <th>总能耗</th>
                            <th>完成率</th>
                            <th>缓存命中率</th>
                            <th>数据丢失率</th>
                            <th>迁移成功率</th>
                        </tr>
                    </thead>
                    <tbody>
                        {rows_html}
                    </tbody>
                </table>
            </div>
            
            <div style="margin-top: 20px; display: flex; gap: 20px; flex-wrap: wrap;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #e3f2fd; border: 1px solid #90caf9;"></div>
                    <span>探索期 (前25%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #fff3e0; border: 1px solid #ffb74d;"></div>
                    <span>学习期 (中50%)</span>
                </div>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 20px; height: 20px; background: #e8f5e9; border: 1px solid #81c784;"></div>
                    <span>收敛期 (后25%)</span>
                </div>
            </div>
        </div>
"""
    
    def _generate_convergence_analysis(self, training_env: Any) -> str:
        """生成收敛性分析"""
        rewards = training_env.episode_rewards
        
        if len(rewards) < 20:
            return ""
        
        # 计算移动平均的方差（衡量收敛性）
        window = 20
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        recent_variance = np.var(moving_avg[-window:]) if len(moving_avg) >= window else np.var(moving_avg)
        
        # 判断是否收敛
        convergence_threshold = np.var(rewards) * 0.1  # 10%的总方差
        is_converged = recent_variance < convergence_threshold
        
        # 估算收敛轮次
        convergence_episode = 0
        if is_converged:
            for i in range(len(moving_avg) - window, -1, -1):
                if np.var(moving_avg[i:i+window]) > convergence_threshold:
                    convergence_episode = i + window
                    break
        
        # 创建收敛图表
        convergence_chart = self._create_convergence_chart(rewards, moving_avg, convergence_episode)
        
        return f"""
        <div class="section">
            <h2 class="section-title">📉 收敛性分析</h2>
            
            <div class="metrics-grid">
                <div class="metric-card" style="border-left-color: {'#28a745' if is_converged else '#ffc107'}">
                    <div class="metric-label">收敛状态</div>
                    <div class="metric-value" style="color: {'#28a745' if is_converged else '#ffc107'}">
                        {'已收敛 ✓' if is_converged else '收敛中 ○'}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">预估收敛轮次</div>
                    <div class="metric-value">{convergence_episode if convergence_episode > 0 else 'N/A'}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">近期方差</div>
                    <div class="metric-value">{recent_variance:.3f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">收敛阈值</div>
                    <div class="metric-value">{convergence_threshold:.3f}</div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">收敛过程可视化</div>
                <img src="data:image/png;base64,{convergence_chart}" alt="收敛分析">
            </div>
            
            <h3 class="section-subtitle">收敛性评价</h3>
            <p style="line-height: 2; font-size: 1.05em; padding: 15px; background: white; border-radius: 8px;">
                {self._generate_convergence_comment(is_converged, convergence_episode, len(rewards))}
            </p>
        </div>
"""
    
    def _calculate_skewness(self, data):
        """计算偏度"""
        if len(data) == 0:
            return 0
        from scipy import stats
        return float(stats.skew(data))
    
    def _calculate_kurtosis(self, data):
        """计算峰度"""
        if len(data) == 0:
            return 0
        from scipy import stats
        return float(stats.kurtosis(data))
    
    def _interpret_skewness(self, skewness):
        """解释偏度"""
        if abs(skewness) < 0.5:
            return "接近对称分布"
        elif skewness > 0:
            return "右偏分布（正偏）"
        else:
            return "左偏分布（负偏）"
    
    def _interpret_kurtosis(self, kurtosis):
        """解释峰度"""
        if abs(kurtosis) < 0.5:
            return "接近正态分布"
        elif kurtosis > 0:
            return "尖峰分布"
        else:
            return "平峰分布"
    
    def _generate_convergence_comment(self, is_converged, convergence_episode, total_episodes):
        """生成收敛性评论"""
        if not is_converged:
            return "⚠️ 训练尚未完全收敛，建议增加训练轮次或调整超参数（如学习率）以达到更稳定的性能。"
        elif convergence_episode < total_episodes * 0.5:
            return f"✅ 训练在第{convergence_episode}轮左右达到收敛，收敛速度较快，表明算法和超参数配置良好。"
        elif convergence_episode < total_episodes * 0.75:
            return f"✓ 训练在第{convergence_episode}轮左右达到收敛，收敛速度适中，性能稳定。"
        else:
            return f"○ 训练在第{convergence_episode}轮才达到收敛，收敛较慢，可能需要调整学习率或网络架构。"
    
    def _create_phase_analysis_chart(self, early, mid, late):
        """创建三阶段分析箱线图"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data = [early, mid, late]
        positions = [1, 2, 3]
        labels = ['Exploration\n(First 25%)', 'Learning\n(Middle 50%)', 'Convergence\n(Last 25%)']
        
        bp = ax.boxplot(data, positions=positions, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True)
        
        colors = ['#90caf9', '#ffb74d', '#81c784']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Reward', fontsize=12)
        ax.set_title('Training Phase Analysis (Boxplot)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        return self._fig_to_base64(fig)
    
    def _create_distribution_chart(self, rewards):
        """创建奖励分布图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1.hist(rewards, bins=30, color='#667eea', alpha=0.7, edgecolor='black')
        ax1.axvline(np.mean(rewards), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rewards):.2f}')
        ax1.axvline(np.median(rewards), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rewards):.2f}')
        ax1.set_xlabel('Reward', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Reward Distribution Histogram', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q图
        from scipy import stats
        stats.probplot(rewards, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot (Normality Test)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_convergence_chart(self, rewards, moving_avg, convergence_point):
        """创建收敛图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励和移动平均
        ax1.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
        ax1.plot(range(len(moving_avg)), moving_avg, color='#667eea', linewidth=2, label='Moving Average')
        if convergence_point > 0:
            ax1.axvline(convergence_point, color='red', linestyle='--', linewidth=2, label=f'Convergence Point: {convergence_point}')
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward Convergence', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 移动方差
        window = 20
        rolling_var = []
        for i in range(len(rewards) - window + 1):
            rolling_var.append(np.var(rewards[i:i+window]))
        
        ax2.plot(rolling_var, color='#ff6b6b', linewidth=2)
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Rolling Variance', fontsize=12)
        ax2.set_title('Rolling Variance (Stability Indicator)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_adaptive_controller_stats(self, training_env: Any) -> str:
        """生成自适应控制器统计"""
        cache_metrics = training_env.adaptive_cache_controller.get_cache_metrics()
        migration_metrics = training_env.adaptive_migration_controller.get_migration_metrics()
        
        return f"""
        <div class="section">
            <h2 class="section-title">🤖 自适应控制器统计</h2>
            
            <h3 class="section-subtitle">缓存控制器</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">有效性</div>
                    <div class="metric-value">{cache_metrics.get('effectiveness', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">缓存利用率</div>
                    <div class="metric-value">{cache_metrics.get('utilization', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
            </div>
            
            <h3 class="section-subtitle">迁移控制器</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">有效性</div>
                    <div class="metric-value">{migration_metrics.get('effectiveness', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">决策准确度</div>
                    <div class="metric-value">{migration_metrics.get('decision_quality', 0)*100:.1f} <span class="metric-unit">%</span></div>
                </div>
            </div>
        </div>
"""
    
    def _generate_correlation_analysis(self, training_env: Any) -> str:
        """生成指标相关性分析"""
        metrics = training_env.episode_metrics
        
        # 提取关键指标
        metric_names = ['avg_delay', 'total_energy', 'task_completion_rate', 
                       'cache_hit_rate', 'data_loss_ratio_bytes', 'migration_success_rate']
        
        available_metrics = {}
        for name in metric_names:
            if name in metrics and metrics[name]:
                available_metrics[name] = metrics[name]
        
        if len(available_metrics) < 2:
            return ""
        
        # 计算相关性矩阵
        correlation_chart = self._create_correlation_heatmap(available_metrics)
        
        # 计算强相关对
        strong_correlations = self._find_strong_correlations(available_metrics)
        
        corr_text = ""
        for corr in strong_correlations[:5]:  # 显示前5个
            corr_text += f"""
            <div class="recommendation" style="border-left-color: {'#28a745' if corr['value'] > 0 else '#dc3545'}">
                <div class="recommendation-title">
                    {corr['metric1']} ↔️ {corr['metric2']}: 
                    <strong>{'正相关' if corr['value'] > 0 else '负相关'}</strong> 
                    (r={corr['value']:.3f})
                </div>
                <div>{corr['interpretation']}</div>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">🔗 指标相关性分析</h2>
            
            <p style="margin-bottom: 20px; line-height: 1.8;">
                通过分析不同性能指标之间的相关关系，可以发现系统行为的内在联系和优化方向。
            </p>
            
            <div class="chart-container">
                <div class="chart-title">指标相关性热力图</div>
                <img src="data:image/png;base64,{correlation_chart}" alt="相关性热力图">
            </div>
            
            <h3 class="section-subtitle">强相关关系解读</h3>
            {corr_text if corr_text else '<p>未发现显著的强相关关系</p>'}
            
            <h3 class="section-subtitle">相关性解释</h3>
            <div style="padding: 15px; background: white; border-radius: 8px; line-height: 1.8;">
                • <strong>正相关 (r > 0.5)</strong>: 两个指标趋向于同时增大或减小<br>
                • <strong>负相关 (r < -0.5)</strong>: 一个指标增大时另一个趋向于减小<br>
                • <strong>弱相关 (|r| < 0.5)</strong>: 两个指标之间关系不明显<br>
                • <strong>相关系数范围</strong>: -1 (完全负相关) 到 +1 (完全正相关)
            </div>
        </div>
"""
    
    def _generate_per_metric_analysis(self, training_env: Any) -> str:
        """生成逐指标详细趋势分析"""
        metrics = training_env.episode_metrics
        
        # 为每个关键指标生成独立的详细图表
        metrics_charts = self._create_all_metrics_charts(metrics)
        
        charts_html = ""
        for metric_info in metrics_charts:
            charts_html += f"""
            <div class="chart-container">
                <div class="chart-title">{metric_info['title']}</div>
                <img src="data:image/png;base64,{metric_info['chart']}" alt="{metric_info['name']}">
                <p style="margin-top: 10px; color: #666; font-size: 0.95em;">
                    {metric_info['description']}
                </p>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">📊 逐指标详细趋势分析</h2>
            
            <p style="margin-bottom: 20px; line-height: 1.8;">
                每个性能指标的完整演化过程，包含原始数据、移动平均、趋势线和置信区间。
            </p>
            
            {charts_html}
        </div>
"""
    
    def _generate_radar_chart_analysis(self, training_env: Any, results: Dict) -> str:
        """生成性能雷达图分析"""
        metrics = training_env.episode_metrics
        
        # 计算不同阶段的归一化性能
        radar_chart = self._create_radar_chart(training_env, metrics)
        
        # 计算综合评分
        n = len(training_env.episode_rewards)
        if n < 4:
            return ""
        
        quarter = n // 4
        
        # 归一化各项指标并计算综合分数
        def normalize_metric(values, inverse=False):
            """归一化到0-100分"""
            if not values:
                return 0
            arr = np.array(values)
            min_val, max_val = np.min(arr), np.max(arr)
            if max_val == min_val:
                return 50
            normalized = (arr - min_val) / (max_val - min_val)
            if inverse:  # 对于越小越好的指标
                normalized = 1 - normalized
            return float(np.mean(normalized[-quarter:]) * 100)
        
        scores = {
            '任务完成率': normalize_metric(metrics.get('task_completion_rate', []), inverse=False),
            '缓存命中率': normalize_metric(metrics.get('cache_hit_rate', []), inverse=False),
            '时延性能': normalize_metric(metrics.get('avg_delay', []), inverse=True),
            '能耗效率': normalize_metric(metrics.get('total_energy', []), inverse=True),
            '数据可靠性': normalize_metric(metrics.get('data_loss_ratio_bytes', []), inverse=True),
            '迁移成功率': normalize_metric(metrics.get('migration_success_rate', []), inverse=False)
        }
        
        overall_score = np.mean(list(scores.values()))
        
        scores_html = ""
        for metric_name, score in scores.items():
            color = '#28a745' if score >= 70 else '#ffc107' if score >= 50 else '#dc3545'
            scores_html += f"""
            <div class="metric-card">
                <div class="metric-label">{metric_name}</div>
                <div class="metric-value" style="color: {color}">{score:.1f}</div>
                <div class="progress-bar" style="height: 10px; margin-top: 10px;">
                    <div class="progress-fill" style="width: {score}%; font-size: 0;"></div>
                </div>
            </div>
"""
        
        grade = 'A+' if overall_score >= 90 else 'A' if overall_score >= 80 else 'B' if overall_score >= 70 else 'C' if overall_score >= 60 else 'D'
        grade_color = '#28a745' if overall_score >= 70 else '#ffc107' if overall_score >= 60 else '#dc3545'
        
        return f"""
        <div class="section">
            <h2 class="section-title">🎯 综合性能雷达图</h2>
            
            <div class="chart-container">
                <div class="chart-title">多维性能雷达图（三阶段对比）</div>
                <img src="data:image/png;base64,{radar_chart}" alt="性能雷达图">
            </div>
            
            <h3 class="section-subtitle">各维度性能评分（后25%轮次）</h3>
            <div class="metrics-grid">
                {scores_html}
            </div>
            
            <h3 class="section-subtitle">综合性能评级</h3>
            <div style="text-align: center; padding: 30px; background: white; border-radius: 10px;">
                <div style="font-size: 4em; font-weight: bold; color: {grade_color}; margin-bottom: 10px;">
                    {grade}
                </div>
                <div style="font-size: 1.5em; color: #666;">
                    综合得分: {overall_score:.1f}/100
                </div>
                <div style="margin-top: 15px; color: #999;">
                    {'优秀' if overall_score >= 80 else '良好' if overall_score >= 70 else '及格' if overall_score >= 60 else '需改进'}
                </div>
            </div>
        </div>
"""
    
    def _generate_complete_data_table(self, training_env: Any) -> str:
        """生成完整的可导出数据表格"""
        rewards = training_env.episode_rewards
        metrics = training_env.episode_metrics
        
        # 生成CSV格式的数据
        csv_data = "Episode,Reward,Avg_Delay,Total_Energy,Completion_Rate,Cache_Hit_Rate,Data_Loss_Ratio,Migration_Success_Rate\n"
        
        for i in range(len(rewards)):
            csv_data += f"{i+1},{rewards[i]:.6f}"
            for metric_name in ['avg_delay', 'total_energy', 'task_completion_rate', 
                               'cache_hit_rate', 'data_loss_ratio_bytes', 'migration_success_rate']:
                if metric_name in metrics and i < len(metrics[metric_name]):
                    csv_data += f",{metrics[metric_name][i]:.6f}"
                else:
                    csv_data += ",0"
            csv_data += "\n"
        
        # Base64编码CSV数据供下载
        csv_b64 = base64.b64encode(csv_data.encode()).decode()
        
        # 生成统计摘要
        summary_rows = ""
        for metric_name, display_name in [
            ('avg_delay', '平均时延'),
            ('total_energy', '总能耗'),
            ('task_completion_rate', '任务完成率'),
            ('cache_hit_rate', '缓存命中率'),
            ('data_loss_ratio_bytes', '数据丢失率'),
            ('migration_success_rate', '迁移成功率')
        ]:
            if metric_name in metrics and metrics[metric_name]:
                data = np.array(metrics[metric_name])
                summary_rows += f"""
                <tr>
                    <td><strong>{display_name}</strong></td>
                    <td>{np.mean(data):.6f}</td>
                    <td>{np.std(data):.6f}</td>
                    <td>{np.min(data):.6f}</td>
                    <td>{np.percentile(data, 25):.6f}</td>
                    <td>{np.median(data):.6f}</td>
                    <td>{np.percentile(data, 75):.6f}</td>
                    <td>{np.max(data):.6f}</td>
                </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">💾 完整数据导出</h2>
            
            <p style="margin-bottom: 20px;">
                以下是所有训练轮次的完整数据统计，可以下载CSV文件进行进一步分析。
            </p>
            
            <div style="text-align: center; margin: 30px 0;">
                <a href="data:text/csv;base64,{csv_b64}" 
                   download="training_data.csv" 
                   style="display: inline-block; padding: 15px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                          color: white; text-decoration: none; border-radius: 8px; font-size: 1.1em; 
                          box-shadow: 0 4px 12px rgba(0,0,0,0.2); transition: transform 0.2s;"
                   onmouseover="this.style.transform='translateY(-3px)'"
                   onmouseout="this.style.transform='translateY(0)'">
                    📥 下载完整CSV数据
                </a>
                <div style="margin-top: 10px; color: #666; font-size: 0.9em;">
                    包含 {len(rewards)} 轮训练数据，8列指标
                </div>
            </div>
            
            <h3 class="section-subtitle">数据统计摘要（所有指标）</h3>
            <div class="data-table" style="overflow-x: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>指标</th>
                            <th>均值</th>
                            <th>标准差</th>
                            <th>最小值</th>
                            <th>Q1 (25%)</th>
                            <th>中位数</th>
                            <th>Q3 (75%)</th>
                            <th>最大值</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td><strong>奖励</strong></td>
                            <td>{np.mean(rewards):.6f}</td>
                            <td>{np.std(rewards):.6f}</td>
                            <td>{np.min(rewards):.6f}</td>
                            <td>{np.percentile(rewards, 25):.6f}</td>
                            <td>{np.median(rewards):.6f}</td>
                            <td>{np.percentile(rewards, 75):.6f}</td>
                            <td>{np.max(rewards):.6f}</td>
                        </tr>
                        {summary_rows}
                    </tbody>
                </table>
            </div>
        </div>
"""
    
    def _generate_peak_anomaly_analysis(self, training_env: Any) -> str:
        """生成峰值和异常分析"""
        rewards = training_env.episode_rewards
        
        # 找出最佳和最差的episodes
        rewards_arr = np.array(rewards)
        top_5_idx = np.argsort(rewards_arr)[-5:][::-1]
        bottom_5_idx = np.argsort(rewards_arr)[:5]
        
        # 找出异常波动
        if len(rewards) > 1:
            reward_changes = np.diff(rewards)
            large_jumps_idx = np.where(np.abs(reward_changes) > np.std(reward_changes) * 2)[0]
        else:
            large_jumps_idx = []
        
        top_html = ""
        for rank, idx in enumerate(top_5_idx, 1):
            top_html += f"""
            <tr style="background: #e8f5e9;">
                <td>{rank}</td>
                <td><strong>Episode {idx + 1}</strong></td>
                <td style="color: #28a745; font-weight: bold;">{rewards[idx]:.3f}</td>
                <td>{self._get_episode_description(training_env, idx)}</td>
            </tr>
"""
        
        bottom_html = ""
        for rank, idx in enumerate(bottom_5_idx, 1):
            bottom_html += f"""
            <tr style="background: #ffebee;">
                <td>{rank}</td>
                <td><strong>Episode {idx + 1}</strong></td>
                <td style="color: #dc3545; font-weight: bold;">{rewards[idx]:.3f}</td>
                <td>{self._get_episode_description(training_env, idx)}</td>
            </tr>
"""
        
        jumps_html = ""
        for idx in large_jumps_idx[:10]:  # 只显示前10个
            change = reward_changes[idx]
            jumps_html += f"""
            <tr>
                <td>Episode {idx + 1} → {idx + 2}</td>
                <td style="color: {'#28a745' if change > 0 else '#dc3545'}; font-weight: bold;">
                    {change:+.3f}
                </td>
                <td style="color: {'#28a745' if change > 0 else '#dc3545'}">
                    {abs(change / rewards[idx]) * 100:.1f}%
                </td>
                <td>{'显著提升 ↗' if change > 0 else '显著下降 ↘'}</td>
            </tr>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">⚡ 峰值与异常分析</h2>
            
            <h3 class="section-subtitle">🏆 最佳表现 Top 5</h3>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>轮次</th>
                        <th>奖励</th>
                        <th>特征</th>
                    </tr>
                </thead>
                <tbody>
                    {top_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">📉 最差表现 Bottom 5</h3>
            <table>
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>轮次</th>
                        <th>奖励</th>
                        <th>特征</th>
                    </tr>
                </thead>
                <tbody>
                    {bottom_html}
                </tbody>
            </table>
            
            <h3 class="section-subtitle">🔄 显著波动事件</h3>
            {'<table><thead><tr><th>轮次变化</th><th>奖励变化</th><th>变化率</th><th>趋势</th></tr></thead><tbody>' + jumps_html + '</tbody></table>' if jumps_html else '<p style="color: #666;">未检测到显著的奖励波动，训练过程相对平稳。</p>'}
            
            <div style="margin-top: 20px; padding: 15px; background: #fff3cd; border-left: 4px solid #ffc107; border-radius: 5px;">
                <strong>💡 提示：</strong> 显著波动通常由探索策略、学习率或环境随机性引起。
                如果波动过大，考虑调整探索噪声或学习率。
            </div>
        </div>
"""
    
    def _generate_smoothness_analysis(self, training_env: Any) -> str:
        """生成学习曲线平滑度分析"""
        rewards = training_env.episode_rewards
        
        if len(rewards) < 10:
            return ""
        
        # 计算平滑度指标
        # 1. 一阶差分的标准差（波动性）
        first_diff = np.diff(rewards)
        volatility = np.std(first_diff)
        
        # 2. 自相关性
        autocorr = np.corrcoef(rewards[:-1], rewards[1:])[0, 1] if len(rewards) > 1 else 0
        
        # 3. 趋势强度
        x = np.arange(len(rewards))
        trend_coef = np.polyfit(x, rewards, 1)[0]
        
        # 4. 平滑指数（基于移动平均的偏离）
        window = min(20, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        deviations = []
        for i in range(len(moving_avg)):
            deviations.append(abs(rewards[i + window//2] - moving_avg[i]))
        smoothness_score = 100 - min(100, np.mean(deviations) * 10)
        
        # 创建平滑度分析图表
        smoothness_chart = self._create_smoothness_chart(rewards, first_diff)
        
        # 评价
        smoothness_grade = '优秀' if smoothness_score >= 70 else '良好' if smoothness_score >= 50 else '一般' if smoothness_score >= 30 else '较差'
        volatility_grade = '低' if volatility < np.std(rewards) * 0.3 else '中' if volatility < np.std(rewards) * 0.6 else '高'
        
        return f"""
        <div class="section">
            <h2 class="section-title">📈 学习曲线平滑度分析</h2>
            
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">平滑度评分</div>
                    <div class="metric-value" style="color: {'#28a745' if smoothness_score >= 70 else '#ffc107' if smoothness_score >= 50 else '#dc3545'}">
                        {smoothness_score:.1f}
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {smoothness_grade}
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">波动性</div>
                    <div class="metric-value">{volatility:.3f}</div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {volatility_grade}波动
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">自相关系数</div>
                    <div class="metric-value">{autocorr:.3f}</div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {'强' if abs(autocorr) > 0.7 else '中' if abs(autocorr) > 0.4 else '弱'}相关
                    </div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">总体趋势</div>
                    <div class="metric-value" style="color: {'#28a745' if trend_coef > 0 else '#dc3545'}">
                        {trend_coef:+.3f}
                    </div>
                    <div style="font-size: 0.85em; color: #666; margin-top: 5px;">
                        {'上升' if trend_coef > 0 else '下降'}趋势
                    </div>
                </div>
            </div>
            
            <div class="chart-container">
                <div class="chart-title">奖励变化率分析</div>
                <img src="data:image/png;base64,{smoothness_chart}" alt="平滑度分析">
            </div>
            
            <h3 class="section-subtitle">平滑度解读</h3>
            <div style="padding: 15px; background: white; border-radius: 8px; line-height: 2;">
                • <strong>平滑度评分</strong>: {smoothness_score:.1f}/100 - 表示学习曲线的稳定程度<br>
                • <strong>波动性</strong>: {volatility:.3f} - 相邻轮次奖励变化的标准差<br>
                • <strong>自相关</strong>: {autocorr:.3f} - 反映连续轮次之间的相似性<br>
                • <strong>趋势系数</strong>: {trend_coef:+.3f} - {'正值表示整体进步，数值越大进步越快' if trend_coef > 0 else '负值表示性能下降，需要关注'}<br>
                <br>
                <strong>💡 建议</strong>: 
                {'学习曲线平滑，训练稳定，可以考虑加快学习速度。' if smoothness_score >= 70 else 
                 '学习曲线波动适中，训练正常进行。' if smoothness_score >= 50 else
                 '学习曲线波动较大，建议降低学习率或增加批量大小以提高稳定性。'}
            </div>
        </div>
"""
    
    # 辅助方法
    def _create_correlation_heatmap(self, metrics_dict):
        """创建相关性热力图"""
        import seaborn as sns
        
        # 准备数据
        data = []
        labels = []
        for name, values in metrics_dict.items():
            data.append(values)
            labels.append(name.replace('_', ' ').title())
        
        # 计算相关性矩阵
        corr_matrix = np.corrcoef(data)
        
        # 绘图
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(corr_matrix, cmap='RdYlGn', vmin=-1, vmax=1, aspect='auto')
        
        # 设置标签
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # 添加数值
        for i in range(len(labels)):
            for j in range(len(labels)):
                text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=9)
        
        ax.set_title('Metrics Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
        fig.colorbar(im, ax=ax, label='Correlation Coefficient')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _find_strong_correlations(self, metrics_dict):
        """找出强相关关系"""
        names = list(metrics_dict.keys())
        correlations = []
        
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                corr = np.corrcoef(metrics_dict[names[i]], metrics_dict[names[j]])[0, 1]
                if abs(corr) > 0.5:  # 强相关阈值
                    interpretation = ""
                    if abs(corr) > 0.8:
                        interpretation = "非常强的相关性，这两个指标几乎同步变化"
                    elif abs(corr) > 0.6:
                        interpretation = "强相关性，这两个指标有明显的关联"
                    else:
                        interpretation = "中等相关性，这两个指标有一定关联"
                    
                    correlations.append({
                        'metric1': names[i].replace('_', ' ').title(),
                        'metric2': names[j].replace('_', ' ').title(),
                        'value': corr,
                        'interpretation': interpretation
                    })
        
        # 按绝对值排序
        correlations.sort(key=lambda x: abs(x['value']), reverse=True)
        return correlations
    
    def _create_all_metrics_charts(self, metrics):
        """为所有指标创建详细图表"""
        metric_configs = [
            {'name': 'avg_delay', 'title': 'Average Task Delay Evolution', 'unit': 'seconds', 
             'description': 'Reflects the average delay in processing tasks, including transmission, queuing, and computation delays'},
            {'name': 'total_energy', 'title': 'Total Energy Consumption Evolution', 'unit': 'Joules',
             'description': 'Total system energy consumption, including computation, transmission, and migration energy'},
            {'name': 'task_completion_rate', 'title': 'Task Completion Rate Evolution', 'unit': '%',
             'description': 'Ratio of successfully completed tasks to total tasks, measuring system reliability'},
            {'name': 'cache_hit_rate', 'title': 'Cache Hit Rate Evolution', 'unit': '%',
             'description': 'Ratio of requests directly served from cache, reflecting cache policy effectiveness'},
            {'name': 'data_loss_ratio_bytes', 'title': 'Data Loss Ratio Evolution', 'unit': '%',
             'description': 'Ratio of data lost due to timeout or insufficient resources'},
            {'name': 'migration_success_rate', 'title': 'Migration Success Rate Evolution', 'unit': '%',
             'description': 'Ratio of successfully executed migrations to total migration operations'}
        ]
        
        charts = []
        for config in metric_configs:
            if config['name'] in metrics and metrics[config['name']]:
                chart = self._create_detailed_metric_chart(
                    metrics[config['name']], 
                    config['title'],
                    config['unit']
                )
                charts.append({
                    'name': config['name'],
                    'title': config['title'],
                    'chart': chart,
                    'description': config['description']
                })
        
        return charts
    
    def _create_detailed_metric_chart(self, data, title, unit):
        """创建单个指标的详细图表"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        episodes = np.arange(1, len(data) + 1)
        
        # 原始数据
        ax.plot(episodes, data, alpha=0.3, color='gray', label='Raw Data', linewidth=1)
        
        # 移动平均
        window = min(20, len(data) // 5)
        if window > 1:
            moving_avg = np.convolve(data, np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(data) + 1), moving_avg, 
                   color='#667eea', linewidth=2.5, label=f'Moving Average ({window})')
            
            # 趋势线
            x = np.arange(len(moving_avg))
            z = np.polyfit(x, moving_avg, 2)
            p = np.poly1d(z)
            ax.plot(range(window, len(data) + 1), p(x), 
                   '--', color='#dc3545', linewidth=2, label='Trend (Polynomial)')
        
        ax.set_xlabel('Episode', fontsize=12)
        # 移除中文标题中的"演化"部分，直接使用英文标题
        ax.set_ylabel(f'Value ({unit})', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _create_radar_chart(self, training_env, metrics):
        """创建性能雷达图"""
        n = len(training_env.episode_rewards)
        if n < 4:
            return self._fig_to_base64(plt.figure())
        
        quarter = n // 4
        
        # 提取三个阶段的数据
        categories = ['Task\nCompletion', 'Cache Hit\nRate', 'Delay\nPerformance', 'Energy\nEfficiency', 'Data\nReliability', 'Migration\nSuccess']
        
        def get_stage_score(metric_name, stage_slice, inverse=False):
            if metric_name not in metrics or not metrics[metric_name]:
                return 0
            values = metrics[metric_name][stage_slice]
            if not values:
                return 0
            score = np.mean(values)
            # 归一化到0-1
            all_values = metrics[metric_name]
            min_val, max_val = np.min(all_values), np.max(all_values)
            if max_val == min_val:
                return 0.5
            normalized = (score - min_val) / (max_val - min_val)
            return 1 - normalized if inverse else normalized
        
        early_scores = [
            get_stage_score('task_completion_rate', slice(0, quarter)),
            get_stage_score('cache_hit_rate', slice(0, quarter)),
            get_stage_score('avg_delay', slice(0, quarter), inverse=True),
            get_stage_score('total_energy', slice(0, quarter), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(0, quarter), inverse=True),
            get_stage_score('migration_success_rate', slice(0, quarter))
        ]
        
        mid_scores = [
            get_stage_score('task_completion_rate', slice(quarter, 3*quarter)),
            get_stage_score('cache_hit_rate', slice(quarter, 3*quarter)),
            get_stage_score('avg_delay', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('total_energy', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(quarter, 3*quarter), inverse=True),
            get_stage_score('migration_success_rate', slice(quarter, 3*quarter))
        ]
        
        late_scores = [
            get_stage_score('task_completion_rate', slice(3*quarter, n)),
            get_stage_score('cache_hit_rate', slice(3*quarter, n)),
            get_stage_score('avg_delay', slice(3*quarter, n), inverse=True),
            get_stage_score('total_energy', slice(3*quarter, n), inverse=True),
            get_stage_score('data_loss_ratio_bytes', slice(3*quarter, n), inverse=True),
            get_stage_score('migration_success_rate', slice(3*quarter, n))
        ]
        
        # 闭合雷达图
        early_scores += early_scores[:1]
        mid_scores += mid_scores[:1]
        late_scores += late_scores[:1]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        ax.plot(angles, early_scores, 'o-', linewidth=2, label='Exploration Phase', color='#90caf9')
        ax.fill(angles, early_scores, alpha=0.15, color='#90caf9')
        
        ax.plot(angles, mid_scores, 'o-', linewidth=2, label='Learning Phase', color='#ffb74d')
        ax.fill(angles, mid_scores, alpha=0.15, color='#ffb74d')
        
        ax.plot(angles, late_scores, 'o-', linewidth=2, label='Convergence Phase', color='#81c784')
        ax.fill(angles, late_scores, alpha=0.15, color='#81c784')
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, size=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'])
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.set_title('Performance Radar Chart by Training Phase', size=14, fontweight='bold', pad=20)
        ax.grid(True)
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _get_episode_description(self, training_env, idx):
        """获取episode的简要描述"""
        metrics = training_env.episode_metrics
        
        # 提取该episode的关键特征
        features = []
        
        if 'task_completion_rate' in metrics and idx < len(metrics['task_completion_rate']):
            rate = metrics['task_completion_rate'][idx]
            if rate >= 0.95:
                features.append("极高完成率")
            elif rate <= 0.85:
                features.append("较低完成率")
        
        if 'cache_hit_rate' in metrics and idx < len(metrics['cache_hit_rate']):
            rate = metrics['cache_hit_rate'][idx]
            if rate >= 0.8:
                features.append("高缓存命中")
            elif rate <= 0.4:
                features.append("低缓存命中")
        
        if 'avg_delay' in metrics and idx < len(metrics['avg_delay']):
            delay = metrics['avg_delay'][idx]
            avg_delay = np.mean(metrics['avg_delay'])
            if delay < avg_delay * 0.8:
                features.append("低延迟")
            elif delay > avg_delay * 1.2:
                features.append("高延迟")
        
        return ", ".join(features) if features else "正常表现"
    
    def _create_smoothness_chart(self, rewards, first_diff):
        """创建平滑度分析图表"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 奖励及其移动平均
        window = min(20, len(rewards) // 5)
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        
        ax1.plot(rewards, alpha=0.5, label='Raw Reward', color='gray')
        ax1.plot(range(window//2, window//2 + len(moving_avg)), moving_avg, 
                linewidth=2.5, label=f'MA({window})', color='#667eea')
        ax1.set_ylabel('Reward', fontsize=12)
        ax1.set_title('Reward Smoothness', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 变化率（一阶差分）
        ax2.bar(range(len(first_diff)), first_diff, color=['#28a745' if x > 0 else '#dc3545' for x in first_diff], alpha=0.7)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax2.axhline(y=np.std(first_diff), color='red', linestyle='--', linewidth=1, label=f'±1 Std ({np.std(first_diff):.2f})')
        ax2.axhline(y=-np.std(first_diff), color='red', linestyle='--', linewidth=1)
        ax2.set_xlabel('Episode Transition', fontsize=12)
        ax2.set_ylabel('Reward Change', fontsize=12)
        ax2.set_title('Episode-to-Episode Change Rate', fontsize=13, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return self._fig_to_base64(fig)
    
    def _generate_recommendations(self, training_env: Any, results: Dict) -> str:
        """生成建议和结论"""
        recommendations = []
        
        # 基于性能指标生成建议
        final_perf = results.get('final_performance', {})
        avg_completion = final_perf.get('avg_completion', 0)
        
        if avg_completion < 0.9:
            recommendations.append({
                'title': '任务完成率偏低',
                'content': '建议增加计算资源或优化任务调度策略，以提高任务完成率至90%以上。'
            })
        
        if len(training_env.episode_rewards) > 20:
            recent_var = np.var(training_env.episode_rewards[-20:])
            if recent_var > 1000:
                recommendations.append({
                    'title': '训练不够稳定',
                    'content': '最近轮次的奖励方差较大，建议调整学习率或增加训练轮次以提高稳定性。'
                })
        
        # 能耗建议
        avg_energy = np.mean(training_env.episode_metrics.get('total_energy', [0]))
        if avg_energy > 500:
            recommendations.append({
                'title': '能耗较高',
                'content': f'平均能耗为{avg_energy:.1f}J，建议优化计算卸载策略，增加本地处理比例。'
            })
        
        recommendations_html = ""
        for rec in recommendations:
            recommendations_html += f"""
            <div class="recommendation">
                <div class="recommendation-title">💡 {rec['title']}</div>
                <div>{rec['content']}</div>
            </div>
"""
        
        return f"""
        <div class="section">
            <h2 class="section-title">💡 建议与结论</h2>
            
            {recommendations_html if recommendations_html else '<p>训练表现良好，各项指标均在正常范围内。</p>'}
            
            <h3 class="section-subtitle">总体评价</h3>
            <p style="line-height: 1.8; font-size: 1.1em;">
                本次训练成功完成 <span class="highlight">{len(training_env.episode_rewards)}</span> 个轮次，
                Episode总奖励为 <span class="highlight">{final_perf.get('avg_episode_reward', final_perf.get('avg_reward', 0) * 100):.2f}</span>
                （每步平均 <span class="highlight">{final_perf.get('avg_reward', 0):.3f}</span>），
                任务完成率达到 <span class="highlight">{avg_completion*100:.1f}%</span>。
                {'训练过程稳定，模型收敛良好。' if np.var(training_env.episode_rewards[-20:]) < 1000 else '建议继续优化以提高稳定性。'}
            </p>
        </div>
"""
    
    def _generate_system_parameters(self, results: Dict) -> str:
        """生成系统参数总览"""
        system_config = results.get('system_config', {})

        return f"""
        <div class="section">
            <h2 class="section-title">🏗️ 系统拓扑参数</h2>

            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">车辆数量</div>
                    <div class="metric-value">{system_config.get('num_vehicles', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">移动计算节点</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">RSU数量</div>
                    <div class="metric-value">{system_config.get('num_rsus', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">边缘计算节点</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">UAV数量</div>
                    <div class="metric-value">{system_config.get('num_uavs', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">空中计算节点</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">仿真时长</div>
                    <div class="metric-value">{system_config.get('simulation_time', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">时隙数</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">时隙长度</div>
                    <div class="metric-value">{system_config.get('time_slot', 'N/A')} <span class="metric-unit">秒</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">决策周期</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">计算设备</div>
                    <div class="metric-value">{system_config.get('device', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">硬件加速</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">随机种子</div>
                    <div class="metric-value">{system_config.get('random_seed', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">可重复性保证</div>
                </div>
            </div>

            <h3 class="section-subtitle">📊 网络拓扑信息</h3>
            <table>
                <thead>
                    <tr>
                        <th>拓扑参数</th>
                        <th>值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>车辆数量</td>
                        <td><span class="highlight">{system_config.get('num_vehicles', 'N/A')}</span></td>
                        <td>移动车辆节点，提供分布式计算能力</td>
                    </tr>
                    <tr>
                        <td>RSU数量</td>
                        <td>{system_config.get('num_rsus', 'N/A')}</td>
                        <td>路边单元，提供边缘计算服务</td>
                    </tr>
                    <tr>
                        <td>UAV数量</td>
                        <td>{system_config.get('num_uavs', 'N/A')}</td>
                        <td>无人机，提供空中计算支持</td>
                    </tr>
                    <tr>
                        <td>仿真区域</td>
                        <td>{results.get('network_config', {}).get('area_width', 'N/A')} × {results.get('network_config', {}).get('area_height', 'N/A')} m</td>
                        <td>仿真场景的地理范围</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    def _generate_network_parameters(self, results: Dict) -> str:
        """生成网络配置参数"""
        network_config = results.get('network_config', {})

        return f"""
        <div class="section">
            <h2 class="section-title">📡 网络配置参数</h2>

            <h3 class="section-subtitle">无线通信参数</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">系统带宽</div>
                    <div class="metric-value">{network_config.get('bandwidth', 0)/1e6:.1f} <span class="metric-unit">MHz</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">总可用带宽</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">载波频率</div>
                    <div class="metric-value">{network_config.get('carrier_frequency', 0)/1e9:.1f} <span class="metric-unit">GHz</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">工作频段</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">覆盖半径</div>
                    <div class="metric-value">{network_config.get('coverage_radius', 'N/A')} <span class="metric-unit">米</span></div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">节点覆盖范围</div>
                </div>
            </div>

            <h3 class="section-subtitle">3GPP标准通信参数</h3>
            <table>
                <thead>
                    <tr>
                        <th>通信参数</th>
                        <th>车辆</th>
                        <th>RSU</th>
                        <th>UAV</th>
                        <th>标准依据</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>发射功率 (dBm)</td>
                        <td>{results.get('communication_config', {}).get('vehicle_tx_power', 'N/A')}</td>
                        <td>{results.get('communication_config', {}).get('rsu_tx_power', 'N/A')}</td>
                        <td>{results.get('communication_config', {}).get('uav_tx_power', 'N/A')}</td>
                        <td>3GPP TS 38.101</td>
                    </tr>
                    <tr>
                        <td>天线增益 (dBi)</td>
                        <td>{results.get('communication_config', {}).get('antenna_gain_vehicle', 'N/A')}</td>
                        <td>{results.get('communication_config', {}).get('antenna_gain_rsu', 'N/A')}</td>
                        <td>{results.get('communication_config', {}).get('antenna_gain_uav', 'N/A')}</td>
                        <td>3GPP TR 38.901</td>
                    </tr>
                    <tr>
                        <td>总带宽 (MHz)</td>
                        <td colspan="3">{results.get('communication_config', {}).get('total_bandwidth', 0)/1e6:.1f}</td>
                        <td>3GPP标准配置</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    def _generate_compute_parameters(self, results: Dict) -> str:
        """生成计算能力参数"""
        compute_config = results.get('compute_config', {})

        return f"""
        <div class="section">
            <h2 class="section-title">💻 计算能力参数</h2>

            <h3 class="section-subtitle">节点计算能力</h3>
            <table>
                <thead>
                    <tr>
                        <th>节点类型</th>
                        <th>CPU频率 (GHz)</th>
                        <th>内存容量 (GB)</th>
                        <th>典型应用场景</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>车辆节点</strong></td>
                        <td>{compute_config.get('vehicle_cpu_freq', 0)/1e9:.1f}</td>
                        <td>{compute_config.get('vehicle_memory', 0)/1e9:.1f}</td>
                        <td>轻量级任务处理，移动计算</td>
                    </tr>
                    <tr>
                        <td><strong>RSU节点</strong></td>
                        <td>{compute_config.get('rsu_cpu_freq', 0)/1e9:.1f}</td>
                        <td>{compute_config.get('rsu_memory', 0)/1e9:.1f}</td>
                        <td>高性能边缘计算，大任务处理</td>
                    </tr>
                    <tr>
                        <td><strong>UAV节点</strong></td>
                        <td>{compute_config.get('uav_cpu_freq', 0)/1e9:.1f}</td>
                        <td>{compute_config.get('uav_memory', 0)/1e9:.1f}</td>
                        <td>中等计算能力，移动覆盖</td>
                    </tr>
                </tbody>
            </table>

            <h3 class="section-subtitle">能耗模型参数</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">车辆静态功耗</div>
                    <div class="metric-value">{compute_config.get('vehicle_static_power', 'N/A')} <span class="metric-unit">W</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RSU静态功耗</div>
                    <div class="metric-value">{compute_config.get('rsu_static_power', 'N/A')} <span class="metric-unit">W</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">UAV悬停功耗</div>
                    <div class="metric-value">{compute_config.get('uav_hover_power', 'N/A')} <span class="metric-unit">W</span></div>
                </div>
            </div>
        </div>
"""

    def _generate_task_migration_parameters(self, results: Dict) -> str:
        """生成任务和迁移参数"""
        task_config = results.get('task_config', {})
        migration_config = results.get('migration_config', {})
        cache_config = results.get('cache_config', {})
        
        # 处理可能为None的值
        rsu_threshold = migration_config.get('rsu_overload_threshold')
        rsu_threshold_str = f"{rsu_threshold*100:.1f}" if rsu_threshold is not None else "N/A"
        
        uav_threshold = migration_config.get('uav_overload_threshold')
        uav_threshold_str = f"{uav_threshold*100:.1f}" if uav_threshold is not None else "N/A"

        return f"""
        <div class="section">
            <h2 class="section-title">📋 任务与迁移参数</h2>

            <h3 class="section-subtitle">任务生成参数</h3>
            <table>
                <thead>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>任务到达率</td>
                        <td>{task_config.get('arrival_rate', 'N/A')} <span class="metric-unit">个/秒</span></td>
                        <td>泊松过程生成任务频率</td>
                    </tr>
                    <tr>
                        <td>数据大小范围</td>
                        <td>{task_config.get('data_size_range', [0, 0])[0]/8/1e6:.2f} - {task_config.get('data_size_range', [0, 0])[1]/8/1e6:.2f} <span class="metric-unit">MB</span></td>
                        <td>任务输入数据大小范围</td>
                    </tr>
                    <tr>
                        <td>计算量范围</td>
                        <td>{task_config.get('compute_cycles_range', [0, 0])[0]/1e9:.1f} - {task_config.get('compute_cycles_range', [0, 0])[1]/1e9:.1f} <span class="metric-unit">Gcycles</span></td>
                        <td>任务计算复杂度范围</td>
                    </tr>
                    <tr>
                        <td>截止时间范围</td>
                        <td>{task_config.get('deadline_range', [0, 0])[0]:.1f} - {task_config.get('deadline_range', [0, 0])[1]:.1f} <span class="metric-unit">秒</span></td>
                        <td>任务最大容忍延迟</td>
                    </tr>
                    <tr>
                        <td>优先级等级</td>
                        <td>{task_config.get('priority_levels', 'N/A')}</td>
                        <td>任务调度优先级划分</td>
                    </tr>
                </tbody>
            </table>

            <h3 class="section-subtitle">迁移策略参数</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">迁移带宽</div>
                    <div class="metric-value">{migration_config.get('migration_bandwidth', 0)/1e6:.1f} <span class="metric-unit">Mbps</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">迁移阈值</div>
                    <div class="metric-value">{migration_config.get('migration_threshold', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">冷却周期</div>
                    <div class="metric-value">{migration_config.get('cooldown_period', 'N/A')} <span class="metric-unit">秒</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">RSU过载阈值</div>
                    <div class="metric-value">{rsu_threshold_str}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">UAV过载阈值</div>
                    <div class="metric-value">{uav_threshold_str}<span class="metric-unit">%</span></div>
                </div>
            </div>

            <h3 class="section-subtitle">缓存配置参数</h3>
            <table>
                <thead>
                    <tr>
                        <th>节点类型</th>
                        <th>缓存容量 (GB)</th>
                        <th>替换策略</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>车辆缓存</td>
                        <td>{cache_config.get('vehicle_cache_capacity', 0)/1e9:.1f}</td>
                        <td rowspan="3">{cache_config.get('cache_policy', 'N/A')}</td>
                    </tr>
                    <tr>
                        <td>RSU缓存</td>
                        <td>{cache_config.get('rsu_cache_capacity', 0)/1e9:.1f}</td>
                    </tr>
                    <tr>
                        <td>UAV缓存</td>
                        <td>{cache_config.get('uav_cache_capacity', 0)/1e9:.1f}</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    def _generate_reward_parameters(self, results: Dict) -> str:
        """生成奖励函数参数"""
        reward_config = results.get('reward_config', {})

        return f"""
        <div class="section">
            <h2 class="section-title">🎯 奖励函数参数</h2>

            <h3 class="section-subtitle">优化目标权重</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">时延权重 (ω_T)</div>
                    <div class="metric-value">{reward_config.get('reward_weight_delay', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">核心优化目标：最小化任务时延</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">能耗权重 (ω_E)</div>
                    <div class="metric-value">{reward_config.get('reward_weight_energy', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">核心优化目标：最小化系统能耗</div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">丢弃惩罚 (ω_D)</div>
                    <div class="metric-value">{reward_config.get('reward_penalty_dropped', 'N/A')}</div>
                    <div style="font-size: 0.8em; color: #666; margin-top: 5px;">约束条件：保证任务完成率</div>
                </div>
            </div>

            <h3 class="section-subtitle">奖励函数公式</h3>
            <div style="padding: 20px; background: white; border-radius: 8px; border: 2px solid #667eea; margin: 20px 0;">
                <div style="font-family: 'Courier New', monospace; font-size: 1.1em; text-align: center;">
                    <strong>Reward = -(ω_T × 时延 + ω_E × 能耗) - ω_D × dropped_tasks</strong>
                </div>
                <div style="margin-top: 15px; line-height: 1.8;">
                    • <strong>主优化目标</strong>: ω_T × 时延 + ω_E × 能耗（权重分别为{reward_config.get('reward_weight_delay', 'N/A')}和{reward_config.get('reward_weight_energy', 'N/A')}）<br>
                    • <strong>约束条件</strong>: ω_D × dropped_tasks（权重为{reward_config.get('reward_penalty_dropped', 'N/A')}，轻微惩罚保证完成率）<br>
                    • <strong>设计理念</strong>: 聚焦于时延和能耗双目标优化，缓存和迁移成功率作为手段而非目标
                </div>
            </div>

            <h3 class="section-subtitle">权重配置说明</h3>
            <table>
                <thead>
                    <tr>
                        <th>权重类型</th>
                        <th>值</th>
                        <th>优化目标</th>
                        <th>论文依据</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>时延权重</td>
                        <td><span class="highlight">{reward_config.get('reward_weight_delay', 'N/A')}</span></td>
                        <td>最小化任务处理时延</td>
                        <td>核心QoS指标，车联网首要目标</td>
                    </tr>
                    <tr>
                        <td>能耗权重</td>
                        <td>{reward_config.get('reward_weight_energy', 'N/A')}</td>
                        <td>最小化系统总能耗</td>
                        <td>绿色计算，资源效率优化</td>
                    </tr>
                    <tr>
                        <td>丢弃惩罚</td>
                        <td>{reward_config.get('reward_penalty_dropped', 'N/A')}</td>
                        <td>保证任务完成率</td>
                        <td>系统可靠性约束，轻微惩罚</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    def _generate_algorithm_parameters(self, results: Dict) -> str:
        """生成算法配置参数"""
        algorithm_config = results.get('algorithm_config', {})

        return f"""
        <div class="section">
            <h2 class="section-title">⚙️ 算法配置参数</h2>

            <h3 class="section-subtitle">神经网络架构</h3>
            <table>
                <thead>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>隐藏层维度</td>
                        <td>{algorithm_config.get('hidden_dim', 'N/A')}</td>
                        <td>神经网络隐藏层神经元数量</td>
                    </tr>
                    <tr>
                        <td>批次大小</td>
                        <td>{algorithm_config.get('batch_size', 'N/A')}</td>
                        <td>每次训练使用的样本数量</td>
                    </tr>
                    <tr>
                        <td>经验池大小</td>
                        <td>{algorithm_config.get('memory_size', 'N/A')}</td>
                        <td>存储历史经验的最大容量</td>
                    </tr>
                </tbody>
            </table>

            <h3 class="section-subtitle">学习率配置</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Actor学习率</div>
                    <div class="metric-value">{algorithm_config.get('actor_lr', 'N/A')}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Critic学习率</div>
                    <div class="metric-value">{algorithm_config.get('critic_lr', 'N/A')}</div>
                </div>
            </div>

            <h3 class="section-subtitle">探索与稳定参数</h3>
            <table>
                <thead>
                    <tr>
                        <th>参数</th>
                        <th>值</th>
                        <th>说明</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>折扣因子 (γ)</td>
                        <td>{algorithm_config.get('gamma', 'N/A')}</td>
                        <td>未来奖励的衰减因子</td>
                    </tr>
                    <tr>
                        <td>软更新参数 (τ)</td>
                        <td>{algorithm_config.get('tau', 'N/A')}</td>
                        <td>目标网络更新的平滑程度</td>
                    </tr>
                    <tr>
                        <td>噪声标准差</td>
                        <td>{algorithm_config.get('noise_std', 'N/A')}</td>
                        <td>动作探索的噪声幅度</td>
                    </tr>
                    <tr>
                        <td>策略延迟更新</td>
                        <td>{algorithm_config.get('policy_delay', 'N/A')}</td>
                        <td>Actor网络更新频率控制</td>
                    </tr>
                </tbody>
            </table>
        </div>
"""

    def _generate_html_footer(self) -> str:
        """生成HTML尾部"""
        return f"""
        </div>
        <div class="footer">
            <p>VEC Migration Caching System - Training Report</p>
            <p>Generated by HTML Report Generator v2.0 (Enhanced Parameters)</p>
            <p style="margin-top: 10px; font-size: 0.9em;">
                © 2025 All Rights Reserved | <a href="#">Documentation</a> | <a href="#">GitHub</a>
            </p>
        </div>
    </div>
</body>
</html>
"""
    
    def save_report(self, html_content: str, filepath: str) -> bool:
        """
        保存HTML报告到文件
        
        Args:
            html_content: HTML内容
            filepath: 保存路径
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            return True
        except Exception as e:
            print(f"保存报告失败: {e}")
            return False
