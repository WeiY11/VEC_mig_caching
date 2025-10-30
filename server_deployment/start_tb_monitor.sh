#!/bin/bash
cd /root/VEC_mig_caching
nohup python monitor_to_tensorboard.py > tensorboard_monitor.log 2>&1 &
echo $! > tensorboard_monitor.pid
echo "TensorBoard监控已启动，PID: $(cat tensorboard_monitor.pid)"

