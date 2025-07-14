#ifndef VULKAN_MATMUL_H
#define VULKAN_MATMUL_H

#ifdef __cplusplus
extern "C" {
#endif


void vulkan_matmul(float* A, float* B, float* C, unsigned int N, unsigned int TILE);

typedef struct VulkanExample VulkanExample;

VulkanExample* vulkan_create(unsigned int N, unsigned int TILE);
void vulkan_multiply(VulkanExample* instance, float* A, float* B, float* C);
void vulkan_destroy(VulkanExample* instance);

#ifdef __cplusplus
}
#endif

#endif