@echo off
REM 在本地Windows电脑上查看服务器训练进度
chcp 65001 >nul
echo ==========================================
echo VEC服务器训练进度查看
echo ==========================================
echo.

REM 使用SSH执行远程命令
ssh -p 21960 root@connect.westc.gpuhub.com "cd /root/VEC_mig_caching && bash deployment/quick_progress.sh"

echo.
echo ==========================================
echo 详细信息请运行:
echo ssh -p 21960 root@connect.westc.gpuhub.com
echo cd /root/VEC_mig_caching
echo bash deployment/check_experiment_progress.sh
echo ==========================================
pause

