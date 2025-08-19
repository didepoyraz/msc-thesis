
#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#include <stdint.h> 

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_init();
void vulkan_submit_tile(uint32_t offsetRowA, uint32_t offsetColA, uint32_t offsetRowB, uint32_t offsetColB, uint32_t offsetRowC, uint32_t offsetColC);
void vulkan_cleanup();
void vulkan_print_total_time();
#ifdef __cplusplus
}
#endif

#endif