#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>
#include "vulkan_matmul.h"
#include <vulkan/vulkan.h>
#include "VulkanTools.h"
#include <pthread.h>
#include "CommandLineParser.hpp"


#define LOG(...) printf(__VA_ARGS__)

// uint32_t N = 10; // matrix size, default
// uint32_t TILE = 1;

CommandLineParser commandLineParser;

struct PushConstants {
    uint32_t offsetRowA;
	uint32_t offsetColA;
	uint32_t offsetRowB;
	uint32_t offsetColB;
	uint32_t offsetRowC;
	uint32_t offsetColC;
};

class VulkanExample
{
private: 
	float* inA;
	float* inB;
	float* outC;
public:
	VkBuffer deviceBufferA, deviceBufferB, deviceBufferC;
	VkBuffer hostBufferA, hostBufferB, hostBufferC;

	VkDeviceMemory deviceMemoryA, deviceMemoryB, deviceMemoryC;
	VkDeviceMemory hostMemoryA, hostMemoryB, hostMemoryC;
	VkDeviceSize bufferSize;
	uint32_t N;
	uint32_t ldN;
	uint32_t TILE;
	pthread_mutex_t* C_locks = nullptr;
	
	float* mappedHostC = nullptr;
	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	VkDevice device;
	uint32_t queueFamilyIndex;
	
	VkQueue queue;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	
	VkPipelineCache pipelineCache;
	VkFence fence;
	VkDescriptorPool descriptorPool;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorSet descriptorSet;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkShaderModule shaderModule;

	// stats
	VkQueryPool queryPool;
	// VkQueryPool queryPool_mem;

	uint64_t totalSetupTime = 0;
	uint64_t totalComputeTime = 0;
	uint64_t totalTransferTime = 0;
	uint64_t totalTime = 0;


	/*
		get timestamps from the query pool
	*/
	void queryTimestamps() {
		uint64_t timestamp1, timestampStart, timestampExeEnd, timestampEnd;
		vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(timestamp1), &timestamp1, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 1, 1, sizeof(timestampStart), &timestampStart, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 2, 1, sizeof(timestampExeEnd), &timestampExeEnd, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 3, 1, sizeof(timestampEnd), &timestampEnd, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

		// std::cout << "Timestamp Pipeline Start: " << timestamp1 << std::endl;
		// std::cout << "Timestamp Exe Start: " << timestampStart << std::endl;
		// std::cout << "Timestamp Exe End: " << timestampExeEnd << std::endl;
		// std::cout << "Timestamp All End: " << timestampEnd << std::endl;

		uint64_t setupTime = timestampStart - timestamp1;
		uint64_t computeTime = timestampExeEnd - timestampStart;
		uint64_t transferTime = timestampEnd - timestampExeEnd;
		uint64_t total = timestampEnd - timestamp1;

		totalSetupTime += setupTime;
		totalComputeTime += computeTime;
		totalTransferTime += transferTime;
		totalTime += total;
			
		// std::cout << "Bufffer setup time = " << setupTime << " ns" << std::endl;
		// std::cout << "Computation time = " << computeTime << " ns" << std::endl;
		// std::cout << "Buffer write + GPU->host transfer time = " << transferTime << " ns" << std::endl;
		// std::cout << "Total Execution time = " << total << " ns" << std::endl;
		// std::cout << "******************************************" << std::endl;
	}

	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr, void** persistentMapping = nullptr)
	{
		// Create the buffer handle
		VkBufferCreateInfo bufferCreateInfo = vks::initializers::bufferCreateInfo(usageFlags, size);
		bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		VK_CHECK_RESULT(vkCreateBuffer(device, &bufferCreateInfo, nullptr, buffer));

		// Create the memory backing up the buffer handle
		VkPhysicalDeviceMemoryProperties deviceMemoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &deviceMemoryProperties);
		VkMemoryRequirements memReqs;
		VkMemoryAllocateInfo memAlloc = vks::initializers::memoryAllocateInfo();
		vkGetBufferMemoryRequirements(device, *buffer, &memReqs); // querying vulkan to find out how much memory we need for this buffer
		
		// std::cout << "Buffer Size: " << memReqs.size / (1024 * 1024) << " MB" << std::endl;

		memAlloc.allocationSize = memReqs.size;
		// Find a memory type index that fits the properties of the buffer
		bool memTypeFound = false;
		for (uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++) {
			if ((memReqs.memoryTypeBits & 1) == 1) {
				if ((deviceMemoryProperties.memoryTypes[i].propertyFlags & memoryPropertyFlags) == memoryPropertyFlags) {
					memAlloc.memoryTypeIndex = i;
					memTypeFound = true;
					break;
				}
			}
			memReqs.memoryTypeBits >>= 1;
		}
		assert(memTypeFound);
		VK_CHECK_RESULT(vkAllocateMemory(device, &memAlloc, nullptr, memory));

		if (memoryPropertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
			void* mapped;
			VK_CHECK_RESULT(vkMapMemory(device, *memory, 0, size, 0, &mapped));

			if (data != nullptr) {
				memcpy(mapped, data, size);
			}

			if (persistentMapping != nullptr) {
				*persistentMapping = mapped;  // keep it mapped
			} else {
				vkUnmapMemory(device, *memory);
			}
		}

		VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *memory, 0));

		return VK_SUCCESS;
	}

	void vkInit(float* inputA, float* inputB, float* outputC, uint32_t ldN, uint32_t N, pthread_mutex_t* locks){

		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan matrix multiplication";
		appInfo.pEngineName = "VulkanExample";
		appInfo.apiVersion = VK_API_VERSION_1_0;

		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pApplicationInfo = &appInfo;

		VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

		uint32_t deviceCount = 0;
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
		physicalDevice = physicalDevices[0];
		
		 uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data());

		for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				queueFamilyIndex = i;
				break;
			}
		}

		const float defaultQueuePriority(1.0f);
		VkDeviceQueueCreateInfo queueCreateInfo = {};

		queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = 1;
		queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
	
		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		
		VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));
    	vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));

		VkQueryPoolCreateInfo queryPoolCreateInfo = {};
		queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
		queryPoolCreateInfo.queryCount = 4;
		queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		VK_CHECK_RESULT(vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &queryPool));

	
		inA = inputA;
		inB = inputB;
		outC = outputC;
		this->ldN = ldN;
		this->N = N;
		this->C_locks = locks;
		
		bufferSize = ldN * ldN * sizeof(float);
		size_t count = ldN * ldN;
		size_t bytes = count * sizeof(float);
		// matrix A
		// std::vector<float> Input_MatrixA(inA, inA + ldN * ldN);
		// matrix B
		// std::vector<float> Input_MatrixB(inB, inB + ldN * ldN);

		float* Input_MatrixA = static_cast<float*>(std::aligned_alloc(16, bytes));
		float* Input_MatrixB = static_cast<float*>(std::aligned_alloc(16, bytes));

		assert(reinterpret_cast<uintptr_t>(Input_MatrixA) % 16 == 0);
		assert(reinterpret_cast<uintptr_t>(Input_MatrixB) % 16 == 0);


		// float* A_data = static_cast<float*>(std::aligned_alloc(16, ldN *  ldN * sizeof(float)));
		// float* B_data = static_cast<float*>(std::aligned_alloc(16, ldN *  ldN* sizeof(float)));
		// float* C_data = static_cast<float*>(std::aligned_alloc(16, matrix_size * matrix_size * sizeof(float)));

		
		std::memcpy(Input_MatrixA, inA, bytes);
		std::memcpy(Input_MatrixB, inB, bytes);
		// Copy input data to GPU mem using staging buffer 
		
			// for matrix A
		createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // memory accessible from the CPU to be able to transfer to and fro the CPU.
			&hostBufferA,
			&hostMemoryA,
			bufferSize,
			Input_MatrixA);

		// for matrix B
		createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			&hostBufferB,
			&hostMemoryB,
			bufferSize,
			Input_MatrixB);
		
		// for matrix C, no data initialized, or can be initialized with 0...
		createBuffer(
			VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
			&hostBufferC,
			&hostMemoryC,
			bufferSize,
			NULL); 

		// Flush writes to host visible buffer
		void* mapped;
		vkMapMemory(device, hostMemoryA, 0, VK_WHOLE_SIZE, 0, &mapped);
		VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
		mappedRange.memory = hostMemoryA;
		mappedRange.offset = 0;
		mappedRange.size = VK_WHOLE_SIZE;
		vkFlushMappedMemoryRanges(device, 1, &mappedRange);
		vkUnmapMemory(device, hostMemoryA);
		mapped = NULL;

		vkMapMemory(device, hostMemoryB, 0, VK_WHOLE_SIZE, 0, &mapped);
		mappedRange = vks::initializers::mappedMemoryRange();
		mappedRange.memory = hostMemoryB;
		mappedRange.offset = 0;
		mappedRange.size = VK_WHOLE_SIZE;
		vkFlushMappedMemoryRanges(device, 1, &mappedRange);
		vkUnmapMemory(device, hostMemoryB);

		// device-local buffer for matrix A
		createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, // First bit makes the buffer bindable as a storage buffer in the compute shader, so it can be read/written from the shader (layout(binding = ...) buffer { ... };). Second bit allows copying data into it from the host-visible staging buffer
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, // not accesible from the CPU
			&deviceBufferA, &deviceMemoryA, bufferSize);

		// Create device-local buffer for matrix B
		createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&deviceBufferB, &deviceMemoryB, bufferSize);
		
		// device-local buffer for matrix C
		createBuffer(
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
			&deviceBufferC, &deviceMemoryC, bufferSize);
		
						// Copy to staging buffer
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VkCommandBuffer copyCmd;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &copyCmd));
		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(copyCmd, &cmdBufInfo));

		VkBufferCopy copyRegionA = {};
		VkBufferCopy copyRegionB = {};
		copyRegionA.size = bufferSize;
		copyRegionB.size = bufferSize;
		vkCmdCopyBuffer(copyCmd, hostBufferA, deviceBufferA, 1, &copyRegionA);
		vkCmdCopyBuffer(copyCmd, hostBufferB, deviceBufferB, 1, &copyRegionB);
		
		VK_CHECK_RESULT(vkEndCommandBuffer(copyCmd));

		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &copyCmd;
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo(VK_FLAGS_NONE);
		VkFence fence;
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &fence)); 

		// Submit to the queue
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, fence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

		vkDestroyFence(device, fence, nullptr);
		vkFreeCommandBuffers(device, commandPool, 1, &copyCmd);

		// 2 storage buffers for input + 1 for output
		std::vector<VkDescriptorPoolSize> poolSizes = {
			vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3),
		};

		VkDescriptorPoolCreateInfo descriptorPoolInfo =
			vks::initializers::descriptorPoolCreateInfo(static_cast<uint32_t>(poolSizes.size()), poolSizes.data(), 1);
		VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

		std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 0), // binding 0
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 1), // binding 1
			vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_COMPUTE_BIT, 2), // binding 2 for Matrix C
		};
		VkDescriptorSetLayoutCreateInfo descriptorLayout =
			vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
		VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

		VkPushConstantRange pushConstantRange{};
		pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		pushConstantRange.offset = 0;
		pushConstantRange.size = sizeof(PushConstants);

		VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
			vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
			pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
			pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;

		VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout));
		VkDescriptorSetAllocateInfo allocInfo =
				vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
			VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));


		//VkDescriptorBufferInfo bufferDescriptor = { deviceBuffer, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo bufferDescriptorA = { deviceBufferA, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo bufferDescriptorB = { deviceBufferB, 0, VK_WHOLE_SIZE };
		VkDescriptorBufferInfo bufferDescriptorC = { deviceBufferC, 0, VK_WHOLE_SIZE };  // for output matrix

		// std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
		// 	vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bufferDescriptor),
		// };

		std::vector<VkWriteDescriptorSet> computeWriteDescriptorSets = {
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 0, &bufferDescriptorA), // binding 0
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, &bufferDescriptorB), // binding 1
			vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2, &bufferDescriptorC), // binding 2
		};
		vkUpdateDescriptorSets(device, static_cast<uint32_t>(computeWriteDescriptorSets.size()), computeWriteDescriptorSets.data(), 0, NULL);

		VkPipelineCacheCreateInfo pipelineCacheCreateInfo = {};
		pipelineCacheCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
		VK_CHECK_RESULT(vkCreatePipelineCache(device, &pipelineCacheCreateInfo, nullptr, &pipelineCache));

		// Create pipeline
		VkComputePipelineCreateInfo computePipelineCreateInfo = vks::initializers::computePipelineCreateInfo(pipelineLayout, 0);

		// compute tile size according to the size of N
		TILE = (N >= 16) ? 16 : 1;
		// Pass SSBO size via specialization constant
		struct SpecializationData {
			uint32_t MATRIX_SIZE;
			uint32_t TILE;
			uint32_t ldN;
		};
		
		SpecializationData specializationData = {
			N,     // MATRIX_SIZE
			TILE,   // TILE_Y
			ldN
		};
		std::vector<VkSpecializationMapEntry> specializationMapEntries = {
			{vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, MATRIX_SIZE), sizeof(uint32_t))},
			{vks::initializers::specializationMapEntry(1,offsetof(SpecializationData, TILE), sizeof(uint32_t))},
			{vks::initializers::specializationMapEntry(2, offsetof(SpecializationData, ldN), sizeof(uint32_t))}
		};
			VkSpecializationInfo specializationInfo = vks::initializers::specializationInfo(
			3, specializationMapEntries.data(), sizeof(SpecializationData), &specializationData);

		std::string shaderDir = "glsl";
		if (commandLineParser.isSet("shaders")) {
			shaderDir = commandLineParser.getValueAsString("shaders", "glsl");
		}
		const std::string shadersPath = getShaderBasePath() + shaderDir + "/matmul/";

		VkPipelineShaderStageCreateInfo shaderStage = {};
		shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		shaderStage.stage = VK_SHADER_STAGE_COMPUTE_BIT;

		shaderStage.module = vks::tools::loadShader((shadersPath + "matmul.comp.spv").c_str(), device);

		shaderStage.pName = "main";
		shaderStage.pSpecializationInfo = &specializationInfo;
		shaderModule = shaderStage.module;

		assert(shaderStage.module != VK_NULL_HANDLE);
		computePipelineCreateInfo.stage = shaderStage;
		VK_CHECK_RESULT(vkCreateComputePipelines(device, pipelineCache, 1, &computePipelineCreateInfo, nullptr, &pipeline));

		// // Create a command buffer for compute operations
		// VkCommandBufferAllocateInfo cmdBufAllocateInfo =
		// 	vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		// VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

		// // Fence for compute CB sync
		// VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
		// VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
	}

	void submitComputeWork(const PushConstants& pc) {
		// Allocate new command buffer
		VkCommandBufferAllocateInfo cmdBufAllocateInfo = 
			vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
		VkCommandBuffer cmdBuf;
		VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &cmdBuf));

		VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();
		VK_CHECK_RESULT(vkBeginCommandBuffer(cmdBuf, &cmdBufInfo));

		vkCmdResetQueryPool(cmdBuf, queryPool, 0, 4);
		vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

		// Bind compute pipeline and descriptor set
		vkCmdBindPipeline(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdBindDescriptorSets(cmdBuf, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

		// Push constants
		vkCmdPushConstants(cmdBuf, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PushConstants), &pc);

		vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queryPool, 1);
		vkCmdDispatch(cmdBuf, N / TILE, N / TILE, 1);
		vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_TRANSFER_BIT, queryPool, 2);

		// Copy back to host
		VkBufferCopy copyRegion = {};
		copyRegion.size = bufferSize;
		vkCmdCopyBuffer(cmdBuf, deviceBufferC, hostBufferC, 1, &copyRegion);

		// Final timestamp
		vkCmdWriteTimestamp(cmdBuf, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 3);

		VK_CHECK_RESULT(vkEndCommandBuffer(cmdBuf));

		// Submit
		VkFence computeFence;
		VkFenceCreateInfo fenceInfo = vks::initializers::fenceCreateInfo();
		VK_CHECK_RESULT(vkCreateFence(device, &fenceInfo, nullptr, &computeFence));

		VkSubmitInfo submitInfo = vks::initializers::submitInfo();
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &cmdBuf;
		VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, computeFence));
		VK_CHECK_RESULT(vkWaitForFences(device, 1, &computeFence, VK_TRUE, UINT64_MAX));
		vkDestroyFence(device, computeFence, nullptr);

		// Copy from mapped to C matrix
		void* mapped;
		vkMapMemory(device, hostMemoryC, 0, VK_WHOLE_SIZE, 0, &mapped);
		VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
		mappedRange.memory = hostMemoryC;
		mappedRange.offset = 0;
		mappedRange.size = VK_WHOLE_SIZE;
		vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);
		float* fdata = static_cast<float*>(mapped);

		// LOG("\nmapped: ");
		// for (int i = 0; i < 16; ++i) {
		// 	printf(": %f\t", fdata[i]);
		// }
		// LOG("\n--------------\nMatrix C before adding mapped: \n");
		// for (int i = 0; i < 16; ++i) {
		// 	printf(": %f\t", outC[i]);
		// }
		// LOG("\n\n");
		int tile_row = pc.offsetRowC / N;
		int tile_col = pc.offsetColC / N;
		int tile_index = tile_row * (ldN / N) + tile_col; // (ldN/N) is to find the number of blocks

		pthread_mutex_lock(&C_locks[tile_index]);
		for (int r = 0; r < N; ++r) {
			for (int c = 0; c < N; ++c) {
				int idx = (pc.offsetRowC + r) * ldN + (pc.offsetColC + c);
				outC[idx] += ((float*)mapped)[idx];
			}
		}
		pthread_mutex_unlock(&C_locks[tile_index]);

		//TODO: you need to flush it to the GPU otherwise this is not going to be
		// set to 0 when you only do memset.
		memset(mapped, 0, ldN * ldN * sizeof(float));

		vkUnmapMemory(device, hostMemoryC);

		// Cleanup
		vkFreeCommandBuffers(device, commandPool, 1, &cmdBuf);
		queryTimestamps();
	}


	void printTotalExecutionTime() const {
	std::cout << "\n===== Vulkan Timing Summary Across All Tiles =====\n";
	std::cout << "Total Buffer Setup Time:        " << totalSetupTime     << " ns\n";
	std::cout << "Total Compute Shader Time:      " << totalComputeTime   << " ns\n";
	std::cout << "Total GPU->Host Transfer Time:  " << totalTransferTime  << " ns\n";
	std::cout << "Total End-to-End GPU Time:      " << totalTime          << " ns\n";
	std::cout << "==================================================\n";
}

	void cleanupAllBuffers() {
		vkDestroyBuffer(device, deviceBufferA, nullptr);
		vkFreeMemory(device, deviceMemoryA, nullptr);
		vkDestroyBuffer(device, hostBufferA, nullptr);
		vkFreeMemory(device, hostMemoryA, nullptr);
        vkDestroyBuffer(device, deviceBufferB, nullptr);
		vkFreeMemory(device, deviceMemoryB, nullptr);
		vkDestroyBuffer(device, hostBufferB, nullptr);
		vkFreeMemory(device, hostMemoryB, nullptr);
	}

	~VulkanExample()
	{
		cleanupAllBuffers();
		vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
		vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
		vkDestroyDescriptorPool(device, descriptorPool, nullptr);
		vkDestroyPipeline(device, pipeline, nullptr);
		vkDestroyPipelineCache(device, pipelineCache, nullptr);
		vkDestroyFence(device, fence, nullptr);
		vkDestroyCommandPool(device, commandPool, nullptr);
		vkDestroyShaderModule(device, shaderModule, nullptr);
		vkDestroyDevice(device, nullptr);
#if DEBUG
		if (debugReportCallback) {
			PFN_vkDestroyDebugReportCallbackEXT vkDestroyDebugReportCallback = reinterpret_cast<PFN_vkDestroyDebugReportCallbackEXT>(vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT"));
			assert(vkDestroyDebugReportCallback);
			vkDestroyDebugReportCallback(instance, debugReportCallback, nullptr);
		}
#endif
		vkDestroyInstance(instance, nullptr);
	}
};

static VulkanExample* vkInstance = nullptr;

extern "C" void vulkan_init(float* A, float* B, float* C, uint32_t ldN, uint32_t N, pthread_mutex_t* locks) {
    
	if (vkInstance) {
		delete vkInstance;
	}

	LOG("Initialising Vulkan\n");
	// LOG("First row of matrix A:\n");
	// for (int i = 0; i < 4; ++i) {
	// 	LOG("%f \t", B[i]);
	// }

	vkInstance = new VulkanExample();
	
	vkInstance->vkInit(A, B, C, ldN, N, locks);                          // Vulkan instance
    // vkInstance->pickPhysicalDevAndQueue();         // Pick GPU and queue family
    // vkInstance->createLogicalDevAndQueue();        // Create logical device and queue
    // vkInstance->createCommandAndQueryPool();       // Command pool and timestamp pool

	// //Create buffers and upload input matrices
    // vkInstance->generateMatrixBuffersAndCopyToDev(A, B, C, ldN, N);

    // //Create compute pipeline and descriptor sets
    // vkInstance->createComputePipeline();
	LOG("Finished Initialisation\n");
}

extern "C" void vulkan_submit_tile(uint32_t offsetRowA, uint32_t offsetColA, uint32_t offsetRowB, uint32_t offsetColB, uint32_t offsetRowC, uint32_t offsetColC){
	if (!vkInstance) {
		std::cerr << "Error: Vulkan has not been initialized!" << std::endl;
	}

	// LOG("Submitting task to Compute Pipeline\n");

	PushConstants pc = {
        offsetRowA,
		offsetColA,
		offsetRowB,
		offsetColB,
        offsetRowC,
		offsetColC
    };
	// printf("\nSubmitting tile A(%d,%d) B(%d,%d) C(%d,%d)\n", pc.offsetRowA, pc.offsetColA, pc.offsetRowB, pc.offsetColB, pc.offsetRowC, pc.offsetColC);
	vkInstance->submitComputeWork(pc);
}

extern "C" void vulkan_cleanup() {
    if (vkInstance) {
        delete vkInstance;
        vkInstance = nullptr;
    }
}

extern "C" void vulkan_print_total_time() {
	if (vkInstance) {
			vkInstance->printTotalExecutionTime();
	}
}