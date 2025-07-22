#!/bin/bash
cd vulkan/build/bin
# List of (MATRIX_SIZE, BLOCK_SIZE) pairs
pairs=(
  "16 4"
  "32 4"
  "64 4"
  # "128 4"
  # "256 4"
  # "512 4"
  # "1024 4"
  # "2048 4"
  # "4096 4"
)

# Loop through each pair and call the program
for pair in "${pairs[@]}"; do
  read -r MATRIX_SIZE BLOCK_SIZE <<< "$pair"
  echo "Running ./matmul $MATRIX_SIZE $BLOCK_SIZE"
  ./matmul "$MATRIX_SIZE" "$BLOCK_SIZE"
done