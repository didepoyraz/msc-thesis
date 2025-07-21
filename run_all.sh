#!/bin/bash
cd vulkan/build/bin
# List of (MATRIX_SIZE, BLOCK_SIZE) pairs
pairs=(
  "16 8"
  "32 16"
  "64 32"
  "128 64"
  "256 128"
  "512 256"
  "1024 512"
  "2048 1024"
  "4096 1024"
)

# Loop through each pair and call the program
for pair in "${pairs[@]}"; do
  read -r MATRIX_SIZE BLOCK_SIZE <<< "$pair"
  echo "Running ./matmul $MATRIX_SIZE $BLOCK_SIZE"
  ./matmul "$MATRIX_SIZE" "$BLOCK_SIZE"
done