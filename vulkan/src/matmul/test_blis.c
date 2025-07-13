#include <stdio.h>
#include <stdlib.h>
#include <blis/blis.h>

int main() {
    const int N = 4;

    double* A = malloc(N * N * sizeof(double));
    double* B = malloc(N * N * sizeof(double));
    double* C = calloc(N * N, sizeof(double)); 

    for (int i = 0; i < N * N; i++) {
        A[i] = (double)i;
        B[i] = (double)i;
    }

    bli_init();

    obj_t A_blis, B_blis, C_blis;
    // ROW-MAJOR layout
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, N, N, A, N, 1, &A_blis);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, N, N, B, N, 1, &B_blis);
    bli_obj_create_with_attached_buffer(BLIS_DOUBLE, N, N, C, N, 1, &C_blis);

    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &A_blis);
    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &B_blis);
    bli_obj_set_conjtrans(BLIS_NO_TRANSPOSE, &C_blis);

    bli_gemm(&BLIS_ONE, &A_blis, &B_blis, &BLIS_ZERO, &C_blis);

    printf("C[0] = %.2f\n", C[0]);

    printf("C matrix:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f ", C[i * N + j]);
        }
        printf("\n");
    }

    bli_finalize();
    free(A); free(B); free(C);
    return 0;
}
