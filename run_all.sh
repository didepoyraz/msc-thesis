#!/bin/bash
cd vulkan/build/bin
# List of (MATRIX_SIZE, BLOCK_SIZE) pairs
pairs=(
  # "16 8"
  "32 16"
  "64 16"
  "128 16"
  "256 16"
  "512 16"
  "1024 16"
  "2048 16"
  "4096 16"
)

# Loop through each pair and call the program
for pair in "${pairs[@]}"; do
  read -r MATRIX_SIZE BLOCK_SIZE <<< "$pair"
  echo "Running ./matmul $MATRIX_SIZE $BLOCK_SIZE"
  ./matmul "$MATRIX_SIZE" "$BLOCK_SIZE"
done