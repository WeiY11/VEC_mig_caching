@echo off
chcp 65001 >nul
echo ==========================================
echo 上传项目到AutoDL服务器
echo ==========================================
echo.
echo 服务器: region-41.seetacloud.com:38597
echo 密码: dXI7ldI+vPec
echo.
echo 提示: 需要手动输入密码
echo ==========================================
echo.

cd /d d:\VEC_mig_caching

echo 开始上传...
echo.

rsync -avz --progress -e "ssh -p 38597" --exclude "__pycache__" --exclude ".git" --exclude "results/" --exclude "deployment/" ./ root@region-41.seetacloud.com:/root/VEC_mig_caching/

echo.
echo ==========================================
echo 上传完成！
echo ==========================================
pause
