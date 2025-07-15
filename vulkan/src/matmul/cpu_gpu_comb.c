#include <stdio.h>
#include <stdlib.h>
#include<blis/blis.h>
#include "vulkan_matmul.h"
#include <time.h>
#include "utils.h"
#define ldN 1024

// #define offsetA(i, p) A[(i) + (p)*ldN]
// #define offsetB(p, j) B[(p) + (j)*ldN]
// #define offsetC(i, j) C[(i) + (j)*ldN]

#define offsetA(i, p) (i * ldN + p)
#define offsetB(p, j) (p * ldN + j)
#define offsetC(i, j) (i * ldN + j)

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
    blis_matmul(A_b, B_b, C_b, N, BLOCK_SIZE);
    clock_gettime(CLOCK_MONOTONIC, &toc);

    elapsed = (toc.tv_sec - tic.tv_sec) * 1000000000LL + (toc.tv_nsec - tic.tv_nsec);
    printf("Resulting Matrix: \n");
    print_matrix_double(C_b, N);
    printf("BLIS GEMM Computation Time: %f ns\n----------------\n", elapsed);

    // print_matrix_float(A, N);
    // print_matrix_float(B, N);

    vulkan_init(A, B, C, N, BLOCK_SIZE);
    vulkan_submit_tile(0, 0, 0);


    // for (int i = 0; i < M; i += BLOCK_SIZE) {
    //     for (int j = 0; j < N; j +=  BLOCK_SIZE) {
    //         for (int p = 0; p < K; p +=  BLOCK_SIZE) {
    //             printf("ith: %i, jth: %i, pth: %i loop\n", i , j, p);
    //             // TODO add the offsets as input to the vulkan multiplication
    //             vulkan_submit_tile(offsetA(i, p), offsetB(p, j), offsetC(i, j));
    //         }
    //     }
    // }
    printf("\noutput matrix after vulkan: ");
    print_matrix_float(C, N);
    // free everything
    free(A); free(B); free(C);
    free(A_b); free(B_b); free(C_b); 
    return 0;
}