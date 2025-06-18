#!/bin/bash

TILE_SIZE=128

for N in 16 32 64 128 256 512 1024
do 
    echo "N = ${N}" >> vulkan_result
    output=$(echo "" | ./build/bin/matmul $N $TILE_SIZE)
    echo "$output" | grep 'time =' >> vulkan_result
    echo "" >> vulkan_result
done

