#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <vector>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <vulkan/vulkan.h>
#include "VulkanTools.h"
#include "CommandLineParser.hpp"


#define LOG(...) printf(__VA_ARGS__)

uint32_t N = 10; // matrix size, default
uint32_t TILE = 1;

CommandLineParser commandLineParser;

class VulkanExample
{
public:
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


	/*
		get timestamps from the query pool
	*/
	void queryTimestamps() {
		uint64_t timestamp1, timestampStart, timestampExeEnd, timestampEnd;
		vkGetQueryPoolResults(device, queryPool, 0, 1, sizeof(timestamp1), &timestamp1, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 1, 1, sizeof(timestampStart), &timestampStart, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 2, 1, sizeof(timestampExeEnd), &timestampExeEnd, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
		vkGetQueryPoolResults(device, queryPool, 3, 1, sizeof(timestampEnd), &timestampEnd, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);

		std::cout << "Timestamp Pipeline Start: " << timestamp1 << std::endl;
		std::cout << "Timestamp Exe Start: " << timestampStart << std::endl;
		std::cout << "Timestamp Exe End: " << timestampExeEnd << std::endl;
		std::cout << "Timestamp All End: " << timestampEnd << std::endl;

		uint64_t elapsedTime1 = timestampStart - timestamp1;
		uint64_t elapsedTime2 = timestampExeEnd - timestampStart;
		uint64_t elapsedTime3 = timestampEnd - timestampExeEnd;
		uint64_t totalExecutionTime = timestampEnd - timestamp1;
		
		std::cout << "Bufffer setup time = " << elapsedTime1 << " ns" << std::endl;
		std::cout << "Computation time = " << elapsedTime2 << " ns" << std::endl;
		std::cout << "Buffer write + GPU->host transfer time = " << elapsedTime3 << " ns" << std::endl;
		std::cout << "Total Execution time = " << totalExecutionTime << " ns" << std::endl;
	}

	VkResult createBuffer(VkBufferUsageFlags usageFlags, VkMemoryPropertyFlags memoryPropertyFlags, VkBuffer *buffer, VkDeviceMemory *memory, VkDeviceSize size, void *data = nullptr)
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
		
		std::cout << "Buffer Size: " << memReqs.size / (1024 * 1024) << " MB" << std::endl;

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

		if (data != nullptr) {
			void *mapped;
			VK_CHECK_RESULT(vkMapMemory(device, *memory, 0, size, 0, &mapped));
			memcpy(mapped, data, size);
			vkUnmapMemory(device, *memory);
		}

		VK_CHECK_RESULT(vkBindBufferMemory(device, *buffer, *memory, 0));

		return VK_SUCCESS;
	}

	VulkanExample()
	{
		LOG("Running matrix multiplication!\n");
		VkApplicationInfo appInfo = {};
		appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		appInfo.pApplicationName = "Vulkan matrix multiplication";
		appInfo.pEngineName = "VulkanExample";
		appInfo.apiVersion = VK_API_VERSION_1_0;

		/*
			Vulkan instance creation (without surface extensions)
		*/
		VkInstanceCreateInfo instanceCreateInfo = {};
		instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		instanceCreateInfo.pApplicationInfo = &appInfo;

		uint32_t layerCount = 1;
		const char* validationLayers[] = { "VK_LAYER_KHRONOS_validation" };

		std::vector<const char*> instanceExtensions = {};

		instanceCreateInfo.enabledExtensionCount = (uint32_t)instanceExtensions.size();
		instanceCreateInfo.ppEnabledExtensionNames = instanceExtensions.data();
		VK_CHECK_RESULT(vkCreateInstance(&instanceCreateInfo, nullptr, &instance));

		/*
			Vulkan device creation
		*/
		// Physical device (always use first)
		uint32_t deviceCount = 0;
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr));
		std::vector<VkPhysicalDevice> physicalDevices(deviceCount);
		VK_CHECK_RESULT(vkEnumeratePhysicalDevices(instance, &deviceCount, physicalDevices.data()));
		physicalDevice = physicalDevices[0];

		VkPhysicalDeviceFeatures features;
		vkGetPhysicalDeviceFeatures(physicalDevice, &features);
		std::cout << "shaderInt64: " << features.shaderInt64 << std::endl;

		VkPhysicalDeviceProperties deviceProperties;
		vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
		LOG("GPU: %s\n", deviceProperties.deviceName);

		// Request a single compute queue
		const float defaultQueuePriority(0.0f);
		VkDeviceQueueCreateInfo queueCreateInfo = {};
		uint32_t queueFamilyCount;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
		std::vector<VkQueueFamilyProperties> queueFamilyProperties(queueFamilyCount);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilyProperties.data()); // Fills in the queueFamilyProperties vector with actual data. Each item describes what capabilities the queue family supports (graphics, compute, transfer).
	
		for (uint32_t i = 0; i < static_cast<uint32_t>(queueFamilyProperties.size()); i++) {

			if (queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { // Checks if this queue family supports compute operations (for running compute shaders).
				queueFamilyIndex = i;
				queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
				queueCreateInfo.queueFamilyIndex = i;
				queueCreateInfo.queueCount = 1;
				queueCreateInfo.pQueuePriorities = &defaultQueuePriority;
				break;
			}
		}

		// Create logical device
		VkDeviceCreateInfo deviceCreateInfo = {};
		deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
		//update: enable the performance query features
		//deviceCreateInfo.pNext = &performanceQueryFeatures;

		std::vector<const char*> deviceExtensions = {};
        
		deviceCreateInfo.enabledExtensionCount = (uint32_t)deviceExtensions.size();
		deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
		VK_CHECK_RESULT(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

		// Get a compute queue
		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);

		// Compute command pool
		VkCommandPoolCreateInfo cmdPoolInfo = {};
		cmdPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		cmdPoolInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
		VK_CHECK_RESULT(vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &commandPool));


		/*
			query pool
		*/
		VkQueryPoolCreateInfo queryPoolCreateInfo = {};
		queryPoolCreateInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    	queryPoolCreateInfo.queryCount = 4; // 4 timestamps
    	queryPoolCreateInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
		vkCreateQueryPool(device, &queryPoolCreateInfo, nullptr, &queryPool);

		/*
			Prepare storage buffers
		*/
		// matrix A
		std::vector<float> Input_MatrixA(N * N);
		// matrix B
		std::vector<float> Input_MatrixB(N * N);
		// output matrix
		std::vector<float> Output_Matrix(N * N);

		// fill input data
		uint32_t n = 0;
		std::generate(Input_MatrixA.begin(), Input_MatrixA.end(), [&n] { return n++; });
        std::generate(Input_MatrixB.begin(), Input_MatrixB.end(), [&n] { return n++; });

		const VkDeviceSize bufferSize = N * N * sizeof(float);

		VkBuffer deviceBufferA, hostBufferA, deviceBufferB, hostBufferB, deviceBufferC, hostBufferC;
		VkDeviceMemory deviceMemoryA, hostMemoryA, deviceMemoryB, hostMemoryB, deviceMemoryC, hostMemoryC;

		// Copy input data to GPU mem using staging buffer 
		{
			// for matrix A
            createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, // memory accessible from the CPU to be able to transfer to and fro the CPU.
				&hostBufferA,
				&hostMemoryA,
				bufferSize,
				Input_MatrixA.data());

            // for matrix B
            createBuffer(
				VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
				VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
				&hostBufferB,
				&hostMemoryB,
				bufferSize,
				Input_MatrixB.data());
			
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
		}

		/*
			Prepare compute pipeline
		*/
		{
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

			VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo =
				vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
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

			// Pass SSBO size via specialization constant
			struct SpecializationData {
				uint32_t MATRIX_SIZE = N;
				uint32_t TILE_X = TILE;
				uint32_t TILE_Y = TILE;
			} specializationData;

			std::vector<VkSpecializationMapEntry> specializationMapEntries = {
				{vks::initializers::specializationMapEntry(0, offsetof(SpecializationData, MATRIX_SIZE), sizeof(uint32_t))},
				{vks::initializers::specializationMapEntry(1,offsetof(SpecializationData, TILE_X), sizeof(uint32_t))},
				{vks::initializers::specializationMapEntry(2, offsetof(SpecializationData, TILE_Y), sizeof(uint32_t))},
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

			// Create a command buffer for compute operations
			VkCommandBufferAllocateInfo cmdBufAllocateInfo =
				vks::initializers::commandBufferAllocateInfo(commandPool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1);
			VK_CHECK_RESULT(vkAllocateCommandBuffers(device, &cmdBufAllocateInfo, &commandBuffer));

			// Fence for compute CB sync
			VkFenceCreateInfo fenceCreateInfo = vks::initializers::fenceCreateInfo(VK_FENCE_CREATE_SIGNALED_BIT);
			VK_CHECK_RESULT(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));
		}

		/*
			Command buffer creation (for compute work submission)
		*/
		{
			VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

		
			VK_CHECK_RESULT(vkBeginCommandBuffer(commandBuffer, &cmdBufInfo));

			vkCmdResetQueryPool(commandBuffer, queryPool, 0, 4);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 0);

			// Barrier to ensure that input buffer transfer is finished before compute shader reads from it
			// VkBufferMemoryBarrier bufferBarrier = vks::initializers::bufferMemoryBarrier();
			// Barrier to ensure that input buffers' transfer is finished before compute shader reads from them
            // update to 2 barriers for matrix A & matrix B
            VkBufferMemoryBarrier bufferBarriers[2] = {
                vks::initializers::bufferMemoryBarrier(),
                vks::initializers::bufferMemoryBarrier(),
            };

            // 1st barrier for deviceBufferA
            bufferBarriers[0].buffer = deviceBufferA;
            bufferBarriers[0].size = VK_WHOLE_SIZE;
            bufferBarriers[0].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bufferBarriers[0].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bufferBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            // 2nd for deviceBufferB
            bufferBarriers[1].buffer = deviceBufferB;
            bufferBarriers[1].size = VK_WHOLE_SIZE;
            bufferBarriers[1].srcAccessMask = VK_ACCESS_HOST_WRITE_BIT;
            bufferBarriers[1].dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
            bufferBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufferBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				2, bufferBarriers,
				0, nullptr);

            
			// bind piplines and descriptor sets
			vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
			vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, 0);

			// vkCmdBeginQuery(commandBuffer, queryPool_mem, 0, 0); //another query pool for perf @ begin query

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queryPool, 1);
            // dispatch shader (2D, x=y=N, z=1)
			vkCmdDispatch(commandBuffer, N/TILE, N/TILE, 1);

			// Barrier to ensure that shader writes are finished before buffer is read back from GPU
			VkBufferMemoryBarrier outputBarrier = vks::initializers::bufferMemoryBarrier();
            outputBarrier.buffer = deviceBufferC; // use BufferC for writing back
            outputBarrier.size = VK_WHOLE_SIZE;
            outputBarrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
            outputBarrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
            outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &outputBarrier,
				0, nullptr);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 2);

			// vkCmdEndQuery(commandBuffer, queryPool_mem, 0); // perf @ end query

			// Read back to host visible buffer
			VkBufferCopy copyRegion = {};
			copyRegion.size = bufferSize;
			vkCmdCopyBuffer(commandBuffer, deviceBufferC, hostBufferC, 1, &copyRegion);

			// Barrier to ensure that buffer copy is finished before host reading from it
			// reuse and reset the barrier
            outputBarrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			outputBarrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
			outputBarrier.buffer = hostBufferC;
			outputBarrier.size = VK_WHOLE_SIZE;
			outputBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
			outputBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;

			vkCmdPipelineBarrier(
				commandBuffer,
				VK_PIPELINE_STAGE_TRANSFER_BIT,
				VK_PIPELINE_STAGE_HOST_BIT,
				VK_FLAGS_NONE,
				0, nullptr,
				1, &outputBarrier,
				0, nullptr);

			vkCmdWriteTimestamp(commandBuffer, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queryPool, 3);

			VK_CHECK_RESULT(vkEndCommandBuffer(commandBuffer));

			// Submit compute work
			vkResetFences(device, 1, &fence);
			// wait for at least buffer transfer happened
            const VkPipelineStageFlags waitStageMask = VK_PIPELINE_STAGE_TRANSFER_BIT;
			VkSubmitInfo computeSubmitInfo = vks::initializers::submitInfo();
			computeSubmitInfo.pWaitDstStageMask = &waitStageMask;
			computeSubmitInfo.commandBufferCount = 1;
			computeSubmitInfo.pCommandBuffers = &commandBuffer;
			VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &computeSubmitInfo, fence));
			VK_CHECK_RESULT(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

			// Make device writes visible to the host
			void *mapped;
			vkMapMemory(device, hostMemoryC, 0, VK_WHOLE_SIZE, 0, &mapped);
			VkMappedMemoryRange mappedRange = vks::initializers::mappedMemoryRange();
			mappedRange.memory = hostMemoryC;
			mappedRange.offset = 0;
			mappedRange.size = VK_WHOLE_SIZE;
			vkInvalidateMappedMemoryRanges(device, 1, &mappedRange);

			// Copy to output
			memcpy(Output_Matrix.data(), mapped, bufferSize);
			vkUnmapMemory(device, hostMemoryC);
		}

		vkQueueWaitIdle(queue);


		queryTimestamps();

	   // Output buffer contents
		int cols = 1024;  

		// LOG("First row of matrix A:\n");
		// for (int i = 0; i < cols; ++i) {
		// 	LOG("%f \t", Input_MatrixA[i]);
		// }
		// LOG("First row of matrix B:\n");
		// for (int i = 0; i < cols; ++i) {
		// 	LOG("%f \t", Input_MatrixB[i]);
		// }
		LOG("%f \t", Output_Matrix[0]);
		LOG("First row of output matrix:\n");
		for (int i = 0; i < cols; ++i) {
			LOG("%f \t", Output_Matrix[i]);
		}

		std::cout << std::endl;

		// Clean up
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

int main(int argc, char* argv[]) {

	if (argc > 2) {
        N = std::atoi(argv[1]);
		TILE = std::atoi(argv[2]);
    }

	std::cout << "Using N = " << N << std::endl;

	// commandLineParser.add("shaders", { "-s", "--shaders" }, 1, "Select shader type to use (glsl or hlsl)");
	// commandLineParser.parse(argc, argv);

	int threadsPerGroup = 1; //current thread group's local size is x=y=1
	int totalThreads = (N * N);
	std::cout << "Threads per group is " << threadsPerGroup << "; Total thread groups is " << totalThreads << std::endl;

	VulkanExample *vulkanExample = new VulkanExample();
	std::cout << "Finished. Press enter to terminate...";
	std::cin.get();
	delete(vulkanExample);
	return 0;
}