#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"
#define ldN 4

// #define offsetA(i, p) A[(i) + (p)*ldN]
// #define offsetB(p, j) B[(p) + (j)*ldN]
// #define offsetC(i, j) C[(i) + (j)*ldN]

#define offsetA(i, p) (i * ldN + p)
#define offsetB(p, j) (p * ldN + j)
#define offsetC(i, j) (i * ldN + j)

void blis_matmul(float* A, float* B, float* C, uint32_t N){
    bli_init();

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    obj_t A_blis, B_blis, C_blis;

    uint32_t M = N;
    uint32_t K = N;

	bli_obj_create_with_attached_buffer(BLIS_FLOAT, M, K, A, K, 1, &A_blis);
	bli_obj_create_with_attached_buffer(BLIS_FLOAT, K, N, B, N, 1, &B_blis);
	bli_obj_create_with_attached_buffer(BLIS_FLOAT, M, N, C, N, 1, &C_blis);

	
    bli_gemm(&BLIS_ONE, &A_blis, &B_blis, &BLIS_ZERO, &C_blis);

    bli_finalize();
}

int main(int argc, char* argv[]) {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int GPU_TILE_SIZE = 16;
   
    if (argc > 2) {
        M = atoi(argv[1]);
        N = atoi(argv[1]);
        K = atoi(argv[1]);
    }

    // initialise the float vectors
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    // fill the vectors 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i;
        B[i] = (float)i+(N*N);
    }

    // print_matrix_float(A, N);
    // printf("----------------------\n\n");
    // print_matrix_float(B, N);

    struct timespec tic, toc;
    double elapsed;
    // bli_thread_set_num_threads(3);
    printf("Calling BLIS GEMM with N = %d\n----------------\n", N);

    clock_gettime(CLOCK_MONOTONIC, &tic);
    blis_matmul(A, B, C, N);
    clock_gettime(CLOCK_MONOTONIC, &toc);

    elapsed = (toc.tv_sec - tic.tv_sec) * 1000000000LL + (toc.tv_nsec - tic.tv_nsec);
    // printf("Resulting Matrix: \n");
    // print_matrix_float(C, N);
    printf("BLIS GEMM Computation Time: %f ns\n----------------\n", elapsed);

    // free everything
    free(A); free(B); free(C);
    return 0;
}