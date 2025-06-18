#!/bin/bash

TILE_SIZE=16
pwd
python3 shaders/glsl/compileshaders.py --glslang /usr/bin/
ninja -C build

> vulkan_result
> gpu_monitor.log

sudo ./gpu_monitor.sh &
monitor_pid=$!

sleep 1

for N in 2048 2048 2048 2048
# 16 32 64 128 256 512 1024
do 
    LOG_FILE="$LOG_DIR/gpu_N${N}.log"
    # sudo ./gpu_monitor.sh &
    echo "N = ${N}" >> vulkan_result
    output=$(echo "" | ./build/bin/matmul $N $TILE_SIZE)
    echo "$output" | grep 'time =' >> vulkan_result
    echo "" >> vulkan_result
done

# sleep 5
# sudo kill "$monitor_pid"