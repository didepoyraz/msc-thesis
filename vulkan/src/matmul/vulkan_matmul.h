#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_matmul(float* A, float* B, float* C, unsigned int N, unsigned int TILE);

#ifdef __cplusplus
}
#endif

#endif