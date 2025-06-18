#!/bin/bash

LOG_FILE="gpu_monitor.log"
DURATION=20       
INTERVAL=0.000001  

> "$LOG_FILE"
echo "Logging started..." | tee "$LOG_FILE"
start_time=$(date +%s)

while true; do
  now=$(date +%s)
  elapsed=$((now - start_time))
  if (( elapsed >= DURATION )); then
    break
  fi

  echo "--- $(date '+%Y-%m-%d %H:%M:%S.%3N') ---" >> "$LOG_FILE"
  cat /sys/kernel/debug/dri/0/measure_clock >> "$LOG_FILE"
  cat /sys/kernel/debug/dri/0/v3d_mm >> "$LOG_FILE"
  cat /sys/kernel/debug/dri/0/bo_stats >> "$LOG_FILE" 2>/dev/null
  echo "" >> "$LOG_FILE"

  # sleep "$INTERVAL"
done

echo "GPU logging finished after $DURATION seconds."
