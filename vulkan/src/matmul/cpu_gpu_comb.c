#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"
#include "tile_queue.h"
#include <pthread.h>

#define offsetA(i, p, ldN) (i * ldN + p)
#define offsetB(p, j, ldN) (p * ldN + j)
#define offsetC(i, j, ldN) (i * ldN + j)
#define NUM_THREADS 3

typedef struct {
    int i;
    int j;
    int p;
} TileConfig;

void blis_matmul(double* A, double* B, double* C, uint32_t N, uint32_t TILE){
    bli_init();

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    obj_t A_blis, B_blis, C_blis;

    uint32_t M = N;
    uint32_t K = N;

	bli_obj_create_with_attached_buffer(BLIS_DOUBLE, M, K, A, 1, K, &A_blis);
	bli_obj_create_with_attached_buffer(BLIS_DOUBLE, K, N, B, 1, N, &B_blis);
	bli_obj_create_with_attached_buffer(BLIS_DOUBLE, M, N, C, 1, N, &C_blis);

	
    bli_gemm(&BLIS_ONE, &A_blis, &B_blis, &BLIS_ZERO, &C_blis);

    bli_finalize();
}

TileConfig* generate_all_tiles(int N, int TILE, int* num_tiles){
    int max_tiles = (N / TILE) * (N / TILE) * (N / TILE);
    TileConfig* tiles = malloc(max_tiles * sizeof(TileConfig));
    int count = 0;

    for (int i = 0; i < N; i += TILE) {
        for (int j = 0; j < N; j += TILE) {
            for (int k = 0; k < N; k += TILE) {
                if(idx % 2 == 0){

                }
                else{

                }
                tiles[count++] = (TileConfig){i, j, k};
            }
        }
    }

    *num_tiles = count;
    return tiles;
}

void cpu_worker(TileQueue* q){

}

int main(int argc, char* argv[]) {

    int M = 1024;
    int N = 1024;
    int K = 1024;
    int BLOCK_SIZE = 512;
    int GPU_TILE_SIZE = 16;
   
    if (argc > 2) {
        M = atoi(argv[1]);
        N = atoi(argv[1]);
        K = atoi(argv[1]);
        
        BLOCK_SIZE = atoi(argv[2]);
    }
    // int ldN = N;
    
    // initialise the gpu float vectors
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    // fill the vectors 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i+1;
        B[i] = (float)i+1;
    }

    double elapsed_full_execution = 0;
    struct timespec start, end;

    struct timespec tic, toc;
    double elapsed;

    int num_tiles = 0;

    TileQueue queue = { .head = 0, .tail = 0, .size = 0, .done = false };
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.not_empty, NULL);

    clock_gettime(CLOCK_MONOTONIC, &start);

    // points to an array of tiles
    // TileConfig* tiles = generate_all_tiles(N, BLOCK_SIZE, &num_tiles);

    pthread_t threads[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, cpu_worker, &queue);
    }

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                TileConfig tile = {i, j, k};
                if((i + j+ k) % 2 == 0){
                    enqueue_tile(&queue, tile);
                }
                else{
                    vulkan_submit_tile(tile.i, tile.p, tile.p, tile.j, tile.i, tile.j);
                }
            }
        }
    }
    vulkan_init(A, B, C, N, BLOCK_SIZE);
    bli_init(); 
    // printf("BLIS default threads: %d\n", bli_thread_get_num_threads());



    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_full_execution = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    printf("Total Compute Time: %f ns \n", elapsed_full_execution);
    // printf("Resulting Matrix: \n");
    // print_first_row_matrix_float(C, N);

    printf("BLIS GEMM Computation Time: %f ns\n----------------\n", elapsed);
    vulkan_print_total_time();

    // free everything
    free(A); free(B); free(C);

    vulkan_cleanup();
    bli_finalize();

    return 0;
}