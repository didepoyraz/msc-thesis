#include <stdio.h>
#include "utils.h"

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