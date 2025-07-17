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

void* cpu_worker(void* arg){
    TileQueue* q = (TileQueue*) arg;
    TileConfig tile;

    while(dequeue_tile(q, &tile)) {
         obj_t A_blis, B_blis, C_blis;

        bli_obj_create_with_attached_buffer(BLIS_FLOAT, q->matrix.BLOCK_SIZE, q->matrix.BLOCK_SIZE, q->matrix.A + offsetA(tile.i, tile.p, q->matrix.N), q->matrix.N, 1, &A_blis);
        bli_obj_create_with_attached_buffer(BLIS_FLOAT, q->matrix.BLOCK_SIZE, q->matrix.BLOCK_SIZE, q->matrix.B + offsetB(tile.p, tile.j, q->matrix.N), q->matrix.N, 1, &B_blis);
        bli_obj_create_with_attached_buffer(BLIS_FLOAT, q->matrix.BLOCK_SIZE, q->matrix.BLOCK_SIZE, q->matrix.C + offsetC(tile.i, tile.j, q->matrix.N), q->matrix.N, 1, &C_blis);
        
        bli_gemm(&BLIS_ONE, &A_blis, &B_blis, &BLIS_ONE, &C_blis);
    }
    return NULL;
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
 
    double elapsed_full_execution = 0;
    struct timespec start, end;

    struct timespec tic, toc;
    double elapsed;
    
    // initialise the gpu float vectors
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    // fill the vectors 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i+1;
        B[i] = (float)i+1;
    }

    Mult matrix = {.A = A, .B = B, .C = C, .N = N, .BLOCK_SIZE = BLOCK_SIZE};
    TileQueue queue = { .head = 0, .tail = 0, .size = 0, .done = false, .matrix = matrix};
    
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.not_empty, NULL);

    clock_gettime(CLOCK_MONOTONIC, &start);

    vulkan_init(A, B, C, N, BLOCK_SIZE);
    bli_init(); 

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

    pthread_mutex_lock(&queue.lock);
    queue.done = true;
    pthread_cond_broadcast(&queue.not_empty);
    pthread_mutex_unlock(&queue.lock);
    // printf("BLIS default threads: %d\n", bli_thread_get_num_threads());

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

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