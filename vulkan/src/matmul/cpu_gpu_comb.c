#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"

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

int main(int argc, char* argv[]) {
    int N = 1024;
    int TILE = 16;

    if (argc > 2) {
        N = atoi(argv[1]);
        TILE = atoi(argv[2]);
    }

    // initialise the gpu float vectors
    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    // initialise the blis double vectors
    double* A_b = (double*)malloc(N * N * sizeof(double));
	double* B_b = (double*)malloc(N * N * sizeof(double));
	double* C_b = (double*)calloc(N * N,  sizeof(double));

    // fill the vectors 
    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i;
        B[i] = (float)i;
        A_b[i] = (double)i;
        B_b[i] = (double)i;    
    }

    struct timespec tic, toc;
    double elapsed;

    printf("Calling BLIS GEMM with N = %d\n----------------\n", N);

    clock_gettime(CLOCK_MONOTONIC, &tic);
    blis_matmul(A_b, B_b, C_b, N, TILE);
    clock_gettime(CLOCK_MONOTONIC, &toc);

    elapsed = (toc.tv_sec - tic.tv_sec) * 1000000000LL + (toc.tv_nsec - tic.tv_nsec);
    
    printf("BLIS GEMM Computation Time: %f ns\n----------------\n", elapsed);

    vulkan_matmul(A, B, C, N, TILE);

    // print_matrix_double(C_b, N);

    // free everything
    free(A); free(B); free(C);
    free(A_b); free(B_b); free(C_b); 
    return 0;
}