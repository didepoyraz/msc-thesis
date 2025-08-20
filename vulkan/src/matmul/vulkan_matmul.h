
#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#include <stdint.h> 

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_init(float* A, float* C, unsigned int ldN);
uint64_t vulkan_submit_tile(uint32_t M, uint32_t N);
void vulkan_cleanup();
void vulkan_print_total_time();
#ifdef __cplusplus
}
#endif

#endif