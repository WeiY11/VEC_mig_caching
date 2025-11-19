@echo off
REM AutoDL服务器部署脚本 - 带宽成本对比实验 (Windows版本)
REM 用途：自动化部署并在AutoDL服务器上运行RSU计算资源对比实验

setlocal enabledelayedexpansion

REM ========== 服务器配置 ==========
set SERVER_HOST=region-41.seetacloud.com
set SERVER_PORT=38597
set SERVER_USER=root
set SERVER_PASSWORD=dXI7ldI+vPec
set REMOTE_DIR=/root/VEC_mig_caching

REM ========== 实验配置 ==========
set EXPERIMENT_SCRIPT=experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py
set EXPERIMENT_TYPES=rsu_compute
set RSU_COMPUTE_LEVELS=default
set EPISODES=1200
set SEED=42

echo ==========================================
echo AutoDL VEC项目部署 - 带宽成本对比实验
echo 目标服务器: %SERVER_USER%@%SERVER_HOST%:%SERVER_PORT%
echo ==========================================
echo.
echo 实验配置:
echo   实验类型: %EXPERIMENT_TYPES%
echo   RSU计算资源档位: %RSU_COMPUTE_LEVELS%
echo   训练轮次: %EPISODES%
echo   随机种子: %SEED%
echo   优化启发式: 是
echo.

REM ========== 检查依赖 ==========
echo [检查] 检查必要工具...

where plink >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到plink！请安装PuTTY工具套件
    echo.
    echo 下载地址: https://www.putty.org/
    echo 或使用Git Bash运行Linux版本脚本
    pause
    exit /b 1
)

where pscp >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ 未找到pscp！请安装PuTTY工具套件
    echo.
    echo 下载地址: https://www.putty.org/
    echo 或使用Git Bash运行Linux版本脚本
    pause
    exit /b 1
)

echo ✅ 工具检查完成
echo.

REM ========== 步骤1：测试连接 ==========
echo [1/6] 测试服务器连接...
echo echo 连接成功! | plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST% 2>nul
if %errorlevel% neq 0 (
    echo ❌ 连接失败！请检查服务器信息
    pause
    exit /b 1
)
echo ✅ 连接成功
echo.

REM ========== 步骤2：创建远程目录 ==========
echo [2/6] 创建远程项目目录...
echo mkdir -p %REMOTE_DIR% | plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST%
echo.

REM ========== 步骤3：创建同步脚本 ==========
echo [3/6] 准备文件同步...

REM 创建临时上传列表
echo 正在生成上传文件列表...
dir /b /s *.py > upload_list.tmp
echo.

echo 使用pscp上传项目文件（这可能需要几分钟）...
echo 提示: 如需更快的上传，建议使用Git Bash运行Linux版本脚本
echo.

REM 使用pscp上传整个项目目录
pscp -P %SERVER_PORT% -pw %SERVER_PASSWORD% -r ^
    -batch ^
    algorithms caching communication config core decision evaluation experiments ^
    hierarchical_learning migration scripts single_agent tests tools utils visualization ^
    requirements.txt train_single_agent.py train_multi_agent.py ^
    %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/

if %errorlevel% neq 0 (
    echo ❌ 文件上传失败！
    echo.
    echo 建议使用以下替代方案:
    echo   1. 使用Git Bash运行Linux版本脚本
    echo   2. 使用WinSCP等GUI工具手动上传
    echo   3. 使用Git将代码推送到远程仓库，然后在服务器上git clone
    pause
    exit /b 1
)

del upload_list.tmp 2>nul
echo.

REM ========== 步骤4：配置服务器环境 ==========
echo [4/6] 配置服务器环境...

REM 创建临时配置脚本
echo cd /root/VEC_mig_caching > setup_env.sh
echo echo "检查Python和CUDA环境..." >> setup_env.sh
echo python --version >> setup_env.sh
echo nvcc --version 2^>^/dev/null ^|^| echo "CUDA未安装" >> setup_env.sh
echo echo "" >> setup_env.sh
echo echo "检查GPU..." >> setup_env.sh
echo nvidia-smi ^|^| echo "无法检测GPU" >> setup_env.sh
echo echo "" >> setup_env.sh
echo echo "安装Python依赖..." >> setup_env.sh
echo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple >> setup_env.sh
echo echo "" >> setup_env.sh
echo echo "验证PyTorch和CUDA..." >> setup_env.sh
echo python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}')" >> setup_env.sh

REM 上传并执行配置脚本
pscp -P %SERVER_PORT% -pw %SERVER_PASSWORD% setup_env.sh %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/
plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST% "cd %REMOTE_DIR% && bash setup_env.sh"

del setup_env.sh 2>nul
echo.

REM ========== 步骤5：创建远程训练脚本 ==========
echo [5/6] 创建远程训练脚本...

REM 创建启动脚本
(
echo #!/bin/bash
echo # 带宽成本对比实验启动脚本
echo.
echo echo "=========================================="
echo echo "VEC项目 - RSU计算资源对比实验"
echo echo "时间: $(date)"
echo echo "=========================================="
echo.
echo LOG_FILE="bandwidth_experiment_$(date +%%Y%%m%%d_%%H%%M%%S).log"
echo.
echo echo "启动实验（后台运行）..."
echo nohup python %EXPERIMENT_SCRIPT% \
echo     --experiment-types %EXPERIMENT_TYPES% \
echo     --rsu-compute-levels %RSU_COMPUTE_LEVELS% \
echo     --episodes %EPISODES% \
echo     --seed %SEED% \
echo     --optimize-heuristic \
echo     ^> ${LOG_FILE} 2^>^&1 ^&
echo.
echo PID=$!
echo echo $PID ^> bandwidth_experiment.pid
echo.
echo echo ""
echo echo "✅ 实验已在后台启动！"
echo echo "   进程ID: ${PID}"
echo echo "   日志文件: ${LOG_FILE}"
echo echo ""
) > start_bandwidth_experiment.sh

REM 创建监控脚本
(
echo #!/bin/bash
echo echo "=========================================="
echo echo "VEC实验监控 - $(date)"
echo echo "=========================================="
echo echo ""
echo echo "GPU使用情况:"
echo nvidia-smi
echo echo ""
echo if [ -f bandwidth_experiment.pid ]; then
echo     PID=$(cat bandwidth_experiment.pid)
echo     echo "实验进程ID: ${PID}"
echo     if ps -p ${PID} ^> /dev/null 2^>^&1; then
echo         echo "状态: ✅ 正在运行"
echo     else
echo         echo "状态: ❌ 已停止"
echo     fi
echo fi
echo echo ""
echo echo "最新日志（最后30行）:"
echo LATEST_LOG=$(ls -t bandwidth_experiment_*.log 2^>/dev/null ^| head -1)
echo if [ -n "${LATEST_LOG}" ]; then
echo     tail -30 ${LATEST_LOG}
echo fi
) > monitor_experiment.sh

REM 上传脚本
pscp -P %SERVER_PORT% -pw %SERVER_PASSWORD% start_bandwidth_experiment.sh monitor_experiment.sh %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/
plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST% "cd %REMOTE_DIR% && chmod +x start_bandwidth_experiment.sh monitor_experiment.sh"

del start_bandwidth_experiment.sh monitor_experiment.sh 2>nul
echo.

REM ========== 步骤6：启动实验 ==========
echo [6/6] 启动实验...
plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST% "cd %REMOTE_DIR% && ./start_bandwidth_experiment.sh"

echo.
echo 等待3秒确认启动...
timeout /t 3 /nobreak >nul

echo.
echo 检查实验状态...
plink -P %SERVER_PORT% -pw %SERVER_PASSWORD% %SERVER_USER%@%SERVER_HOST% "cd %REMOTE_DIR% && ./monitor_experiment.sh"

echo.
echo ==========================================
echo ✅ 部署并启动完成！
echo ==========================================
echo.
echo 📝 监控和管理命令：
echo.
echo 1️⃣  连接到服务器（使用PuTTY或命令行）:
echo    plink -P %SERVER_PORT% %SERVER_USER%@%SERVER_HOST%
echo    密码: %SERVER_PASSWORD%
echo.
echo 2️⃣  监控实验（连接后在服务器上执行）:
echo    cd %REMOTE_DIR%
echo    ./monitor_experiment.sh
echo    tail -f bandwidth_experiment_*.log
echo.
echo 3️⃣  下载结果（在本地Windows执行）:
echo    pscp -P %SERVER_PORT% -pw %SERVER_PASSWORD% -r %SERVER_USER%@%SERVER_HOST%:%REMOTE_DIR%/results ./results_from_autodl
echo.
echo ==========================================
echo.
echo 💡 预计实验时间: 30-38小时 (使用GPU加速)
echo ⚠️  确保AutoDL实例有足够的运行时长（建议至少40小时）
echo.

pause
