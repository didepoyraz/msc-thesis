#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"
#include "tile_queue.h"
#include <pthread.h>
#include <assert.h>

int cpu_counter = 0;
int gpu_counter = 0;

#define DEBUG_MODE 0


#if DEBUG_MODE 
    #define DEBUG_PRINT(...) printf(__VA_ARGS__)
#else 
    # define DEBUG_PRINT(...) do {} while (0)
#endif

int main(int argc, char* argv[]) {

    int M = 8192;
    int N = 8192;
    int K = 8192;
    int mode = 0;

    int num_iter;
    int num_warmups = 5;
    double total_compute_time = 0;

    if (argc > 2) {
        M = atoi(argv[1]);
        N = atoi(argv[1]);
        K = atoi(argv[1]);

        num_iter = atoi(argv[2]);
    }

    DEBUG_PRINT("num tiles: %i\n", num_tiles);

    // initialise the gpu float vectors
    float* A = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    // fill the vectors 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i;
    }

    vulkan_init(A, C, N);

    // warm up runs
    for(int i = 0; i <num_warmups; i ++){
        vulkan_submit_tile(M, N,mode);
    }

    for(int i = 0; i <num_iter; i ++){
        total_compute_time += vulkan_submit_tile(M, N,mode);
    }

    vulkan_print_total_time();
    printf("total compute time of runs: %f\n", total_compute_time);
    // printf("%f", elapsed_compute);
    
    // //----------------Calculate Bandwidth
    // double average_compute_time_s = (total_compute_time / num_iter) / 1000000000;
    // float bytes_read = M * N * 4;
    // float mean_bandwidth_gbs = (bytes_read / average_compute_time_s) / 1000000000;
    // double gibs = mean_bandwidth_gbs / (1024.0*1024.0*1024.0);

    // printf("Mean Bandwidth: %f, and %f\n", mean_bandwidth_gbs, gibs);
    const double ns_per_tick = 1.0; // Pi reports 1 ns/tick

    // totals you already accumulated in TICKS:
    uint64_t total_compute_ticks = total_compute_time ;   // across K runs
    int L = num_iter;

    // 1) mean time per run (seconds)
 
    double avg_compute_time   = (total_compute_time / num_iter) / 1000000000;

    // 2) bytes read once (use 64-bit or double!)
    double bytes = (double) M * (double) N * 4.0 * 64;
    double gb = bytes / 1e9;

    // 3) bandwidths
    double bandwidth_gbs  = gb / avg_compute_time;               // bytes/second
    double bandwidth_Bps  = bytes / avg_compute_time;               // bytes/second
 
    double bandwidth_GBs  = bandwidth_Bps / 1e9;                      // decimal GB/s
    double bandwidth_GiBs = bandwidth_Bps / (1024.0*1024.0*1024.0);   // binary GiB/s

    printf("Mean Bandwidth: %.2f GiB/s (%.2f GB/s), gbs: %.2f\n", bandwidth_GiBs, bandwidth_GBs, bandwidth_gbs);

    free(A); free(C);
    vulkan_cleanup();

    return 0;
}