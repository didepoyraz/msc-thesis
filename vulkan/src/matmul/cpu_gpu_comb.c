#include <stdio.h>
#include <stdlib.h>
#include "vulkan_matmul.h"

int main() {
    int N = 1024;
    int TILE = 16;

    float* A = malloc(N * N * sizeof(float));
    float* B = malloc(N * N * sizeof(float));
    float* C = malloc(N * N * sizeof(float));

    for (int i = 0; i < N * N; i++) {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    vulkan_matmul(A, B, C, N, TILE);

    printf("C[0] = %f\n", C[0]);

    free(A); free(B); free(C);
    return 0;
}