#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"
#include "tile_queue.h"
#include <pthread.h>

#define DEBUG_MODE 0

#if DEBUG_MODE 
    #define DEBUG_PRINT(...) prinft(__VA_ARGS__)
#else 
    # define DEBUG_PRINT(...) do {} while (0)
#endif

#define offsetA(i, p, ldN) (i * ldN + p)
#define offsetB(p, j, ldN) (p * ldN + j)
#define offsetC(i, j, ldN) (i * ldN + j)
#define NUM_THREADS 4

void* cpu_worker(void* arg){
    TileQueue* q = (TileQueue*) arg;
    TileConfig tile;
    DEBUG_PRINT("\nCPU threads starting up!");

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
        A[i] = (float)i;
        B[i] = (float)i+ N*N;
    }

    Mult matrix = {.A = A, .B = B, .C = C, .N = N, .BLOCK_SIZE = BLOCK_SIZE};
    TileQueue queue = { .head = 0, .tail = 0, .size = 0, .done = false, .matrix = matrix};
    
    pthread_mutex_init(&queue.lock, NULL);
    pthread_cond_init(&queue.not_empty, NULL);

    clock_gettime(CLOCK_MONOTONIC, &start);

    vulkan_init(A, B, C, N, BLOCK_SIZE);
    bli_init(); 

    DEBUG_PRINT("initialising threads!\n");

    pthread_t threads[NUM_THREADS];
    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], NULL, cpu_worker, &queue);
    }
    int cpu_gpu_ratio = 6;  
    int count = 0;  

    DEBUG_PRINT("\nstarting to submit tiles to the tile queue!\n");

    for (int i = 0; i < N; i += BLOCK_SIZE) {
        for (int j = 0; j < N; j += BLOCK_SIZE) {
            for (int k = 0; k < N; k += BLOCK_SIZE) {
                TileConfig tile = {i, j, k};
                // DEBUG_PRINT("\n\n-----submitting tile: (%d, %d, %d)-------\n", tile.i, tile.j, tile.p);
                DEBUG_PRINT("\n======= submitting tile: A(%i, %i), B(%i, %i), C(%i, %i) =======\n", tile.i, tile.p, tile.p, tile.j, tile.i, tile.j);

                if((count % (cpu_gpu_ratio + 1)) < cpu_gpu_ratio){
                    //TODO: add error checking here
                    enqueue_tile(&queue, tile);
                }
                else{
                    DEBUG_PRINT("\nsubmitting to vulkan!\n");
                    vulkan_submit_tile(tile.i, tile.p, tile.p, tile.j, tile.i, tile.j);
                }
                count++;
            }
        }
    }

    pthread_mutex_lock(&queue.lock);
    queue.done = true;
    pthread_cond_broadcast(&queue.not_empty);
    pthread_mutex_unlock(&queue.lock);
    // DEBUG_PRINT("BLIS default threads: %d\n", bli_thread_get_num_threads());

    for (int i = 0; i < NUM_THREADS; i++){
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    elapsed_full_execution = (end.tv_sec - start.tv_sec) * 1000000000LL + (end.tv_nsec - start.tv_nsec);
    printf("Total Compute Time: %f ns \n", elapsed_full_execution);
    // printf("Resulting Matrix: \n");
    // print_first_row_matrix_float(C, N);
    printf("matrix c first element %f, last element %f", C[0], C[(N*N)-1]);
    // printf("BLIS GEMM Computation Time: %f ns\n----------------\n", elapsed);
    vulkan_print_total_time();


    char filename[128];
    char* s = "/home/pi/Desktop/msc-thesis/vulkan/results/output_matrix";
    snprintf(filename, sizeof(filename), "%s_%d_%d.csv ", s, N, BLOCK_SIZE);
    
    printf("filename: %s", filename );
    save_matrix_to_file(filename, C, N);

    // free everything
    free(A); free(B); free(C);

    vulkan_cleanup();
    bli_finalize();

    return 0;
}