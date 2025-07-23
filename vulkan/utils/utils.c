#include <stdio.h>
#include "utils.h"
#include<blis/blis.h>
#include <math.h>
#include <stdbool.h>

void print_matrix_float(float* C, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            printf("%8.2f ", C[i * N + j]);
        }
        printf("\n");
    }
}

void print_first_row_matrix_float(float* C, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        printf("%8.2f ", C[i]);
        printf("\n");
    }
}

void print_matrix_double(double* C, uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            printf("%8.2f ", C[i * N + j]);
        }
        printf("\n");
    }
}

float* blis_matmul(float* A, float*B, uint32_t N){

    // initialise the float vectors
    float* C = malloc(N * N * sizeof(float));

    // fill the vectors 
    // for (int i = 0; i < N * N; i++) {
    //     A[i] = (float)i;
    //     B[i] = (float)i + N*N;
    // }

    bli_init();

    printf("BLIS using %d threads\n", bli_thread_get_num_threads());

    if (!A || !B || !C) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(EXIT_FAILURE);
    }

    obj_t A_blis, B_blis, C_blis;

    uint32_t M = N;
    uint32_t K = N;

    bli_obj_create_with_attached_buffer(BLIS_FLOAT, M, K, A, 1, K, &A_blis);
    bli_obj_create_with_attached_buffer(BLIS_FLOAT, K, N, B, 1, N, &B_blis);
    bli_obj_create_with_attached_buffer(BLIS_FLOAT, M, N, C, 1, N, &C_blis);
        
    bli_gemm(&BLIS_ONE, &A_blis, &B_blis, &BLIS_ZERO, &C_blis);

    bli_finalize();

    return C;
}

bool compare_matrices(float* A, float* B, int N, float epsilon){
    int diff_counter = 0;
    float diff_accumulator = 0;
    for (int i = 0; i < N*N; i++){
        float diff = fabsf(A[i] - B[i]);
        if (diff > epsilon) {
            // printf("Mismatch at index %d: A = %f, B = %f, Diff = %f\n", i, A[i], B[i], diff);
            diff_counter++;
            diff_accumulator += diff;
        }
    }
    printf("total of %i different values bigger than %f and average difference is %f\n", diff_counter, epsilon, (diff_accumulator/diff_counter));
    return true;
}

void save_matrix_to_file(const char* filename, float* matrix, int N) {
    FILE* fp = fopen(filename, "w");
    if (!fp) {
        perror("Error opening file for writing");
        return;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fp, "%.6f", matrix[i * N + j]);  // 6-digit precision
            if (j < N - 1) {
                fprintf(fp, ",");
            }
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Matrix saved to %s\n", filename);
}