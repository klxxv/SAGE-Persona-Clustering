#!/bin/bash

# 2. 监控脚本：monitor_gpu.sh
# 实时显示仪表盘和显存占用

while true; do
    clear
    echo "=== SAGE GPU Experiment Status ($(date +'%H:%M:%S')) ==="
    
    echo -e "\n--- Live Dashboard ---"
    if [ -f "checkpoints/live_status.txt" ]; then
        cat checkpoints/live_status.txt
    else
        echo "Dashboard not yet available..."
    fi

    echo -e "\n--- GPU Status ---"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | awk -F', ' '{print "GPU "$1": "$2" | Util: "$3"% | Mem: "$4"/"$5" MB"}'

    echo -e "\n--- Latest Checkpoints ---"
    find checkpoints -name "*.pt" -printf "%T@ %p\n" | sort -n | tail -n 5 | awk '{print $2}'

    echo -e "\nPress Ctrl+C to stop monitoring."
    sleep 5
done
