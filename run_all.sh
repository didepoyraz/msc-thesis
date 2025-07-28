#!/bin/bash
cd vulkan/build/bin
# List of (MATRIX_SIZE, BLOCK_SIZE) pairs
pairs=(
  "16 8 4"
  "32 16 8"
  "64 16 8"
  "128 16 8"
  "256 16 8"
  "512 16 8"
  "1024 16 8"
  "2048 16 8"
  "4096 16 8"
)

# Loop through each pair and call the program
for pair in "${pairs[@]}"; do
  read -r MATRIX_SIZE BLOCK_SIZE RES_P_THREAD<<< "$pair"
  echo "Running ./matmul $MATRIX_SIZE $BLOCK_SIZE $RES_P_THREAD"
  ./matmul "$MATRIX_SIZE" "$BLOCK_SIZE"  "$RES_P_THREAD"
done