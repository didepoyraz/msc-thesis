import subprocess
import csv
import statistics
import math
import random

configs = [
    {"size": 16, "tile": 8},
    {"size": 32, "tile": 16},
    {"size": 64, "tile": 16},
    {"size": 128, "tile": 16},
    {"size": 256, "tile": 16},
    {"size": 512, "tile": 16},
    {"size": 1024, "tile": 16},
    {"size": 2048, "tile": 16},
    {"size": 4096, "tile": 16},
]

N_RUNS = 30   # Number of runs per configuration
WARMUP_RUNS = 5  # Warm-up iterations 

def run_command(size, tile):
    """Run matrix multiplication program once and return compute time in nanoseconds."""

    result = subprocess.run(
    ["./vulkan/build/bin/matmul", str(size), str(tile)],
    cwd="/home/pi/Desktop/msc-thesis",
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
    )
    
    return float(result.stdout.strip())

def mean_ci(data, confidence=0.95):
    """Return mean and half-width of the 95% confidence interval."""
    mean_val = statistics.mean(data)
    stdev = statistics.stdev(data)
    n = len(data)
    h = stdev * 1.96 / math.sqrt(n) 
    return mean_val, h

results = []

for cfg in configs:
    size = cfg["size"]
    tile = cfg["tile"]
    print("running config, Matrix Size:", size, "Tile Size:",tile)
    
    compute_times = []
    
    # Warm-up runs
    for _ in range(WARMUP_RUNS):
        run_command(size, tile)

    # Measured runs
    for _ in range(N_RUNS):
        compute_time = run_command(size, tile)
        compute_times.append(compute_time)

    median_t = statistics.median(compute_times)
    mean_t, ci = mean_ci(compute_times)
    
    min_t = min(compute_times)
    max_t = max(compute_times)
    gflops = (2 * size * size * size) / (median_t * 1e9)
    
    results.append({
        "size": size,
        "tile": tile,
        "median_time_s": median_t,
        "mean_time_s": mean_t,
        "ci_95_s": ci,
        "min_time_s": min_t,
        "max_time_s": max_t,
        "gflops": gflops
    })

# Save results to CSV
with open("matmul_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print("Results saved to matmul_results.csv")
 