
#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#include <stdint.h> 

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_init(float* A, float* B, float* C, unsigned int ldN, unsigned int N);
void vulkan_submit_tile(uint32_t offsetA, uint32_t offsetB, uint32_t offsetC);

#ifdef __cplusplus
}
#endif

#endif