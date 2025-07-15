
#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#include <stdint.h> 

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_init(float* A, float* B, float* C, unsigned int ldN, unsigned int N);
void vulkan_submit_tile(uint32_t offsetRowA, uint32_t offsetColA, uint32_t offsetRowB, uint32_t offsetColB, uint32_t offsetRowC, uint32_t offsetColC);

#ifdef __cplusplus
}
#endif

#endif