#include <stdint.h>
#include<blis/blis.h>

#ifndef UTILS_H
#define UTILS_H

void print_matrix_float(float* C, uint32_t N);
void print_matrix_double(double* C, uint32_t N);
void print_first_row_matrix_float(float* C, uint32_t N);
float* blis_matmul(float* A, float*B, uint32_t N);
bool compare_matrices(float* A, float* B, int N, float epsilon);
void save_matrix_to_file(const char* filename, float* matrix, int N);
#endif