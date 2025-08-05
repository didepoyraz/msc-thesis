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

void save_value_to_file(const char* filename, float value) {
    FILE* fp = fopen(filename, "a");
    if (!fp) {
        perror("Error opening file for writing");
        return;
    }

    if (fp) {
        fprintf(fp, "%.6f, ", value);
        fclose(fp);

    } else {
        perror("Error opening timings.csv");
    }
}