@echo off
chcp 65001 >nul
echo ==========================================
echo   AutoDL VEC 实验快速部署指南
echo ==========================================
echo.
echo 服务器信息:
echo   主机: region-41.seetacloud.com
echo   端口: 38597
echo   用户: root
echo   密码: dXI7ldI+vPec
echo.
echo ==========================================
echo.

:MENU
echo 请选择操作:
echo.
echo   1. 连接到AutoDL服务器（需要手动输入密码）
echo   2. 查看完整部署命令（复制粘贴执行）
echo   3. 退出
echo.
set /p choice=请输入选择 (1/2/3): 

if "%choice%"=="1" goto CONNECT
if "%choice%"=="2" goto SHOW_COMMANDS
if "%choice%"=="3" goto END
echo 无效选择，请重新输入
echo.
goto MENU

:CONNECT
echo.
echo ==========================================
echo 正在连接到AutoDL服务器...
echo 密码: dXI7ldI+vPec
echo ==========================================
echo.
ssh -p 38597 root@region-41.seetacloud.com
goto END

:SHOW_COMMANDS
echo.
echo ==========================================
echo 完整部署命令（在服务器上执行）
echo ==========================================
echo.
echo 1. 首先连接到服务器（在本地执行）:
echo    ssh -p 38597 root@region-41.seetacloud.com
echo    密码: dXI7ldI+vPec
echo.
echo ----------------------------------------
echo 2. 上传项目文件（在本地执行，需要Git Bash）:
echo.
echo cd d:/VEC_mig_caching
echo rsync -avz --progress -e "ssh -p 38597" \
echo     --exclude '__pycache__' --exclude 'results/' --exclude '.git' \
echo     ./ root@region-41.seetacloud.com:/root/VEC_mig_caching/
echo.
echo ----------------------------------------
echo 3. 配置环境（连接到服务器后执行）:
echo.
echo cd /root/VEC_mig_caching
echo pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA可用:', torch.cuda.is_available())"
echo.
echo ----------------------------------------
echo 4. 启动实验（在服务器上执行）:
echo.
echo nohup python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
echo     --experiment-types rsu_compute \
echo     --rsu-compute-levels default \
echo     --episodes 1200 \
echo     --seed 42 \
echo     --optimize-heuristic \
echo     ^> bandwidth_experiment.log 2^>^&1 ^&
echo.
echo echo $! ^> bandwidth_experiment.pid
echo.
echo ----------------------------------------
echo 5. 监控实验（在服务器上执行）:
echo.
echo # 查看日志
echo tail -f bandwidth_experiment.log
echo.
echo # 监控GPU
echo watch -n 5 nvidia-smi
echo.
echo # 查看进程
echo ps aux ^| grep python
echo.
echo ----------------------------------------
echo 6. 下载结果（在本地执行）:
echo.
echo scp -P 38597 -r root@region-41.seetacloud.com:/root/VEC_mig_caching/results ./results_from_autodl
echo.
echo ==========================================
echo.
pause
goto MENU

:END
echo.
echo 谢谢使用！
echo.
pause
