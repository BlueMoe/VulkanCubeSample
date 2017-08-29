#pragma once


#if defined(__ANDROID__)
#define VK_USE_PLATFORM_ANDROID_KHR
#elif defined(__linux__)
#define VK_USE_PLATFORM_XLIB_KHR
#elif defined(_WIN32)
#define VK_USE_PLATFORM_WIN32_KHR
#endif

#include <vulkan/vulkan.hpp>
#include "SDL2/SDL_video.h"
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/vec4.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 uv;
	glm::vec3 normal;
};

struct uniformStruct
{
	glm::mat4 mvp;
};

class VulkanDemo
{
public:
	bool initInstance(int width, int height, std::string windowName);
	bool prepare();
	void run();
	void stop();
	virtual ~VulkanDemo();

private:
	bool createWindow(int width, int height, std::string windowName);
	bool initInstance();

	void createLogicDevice();
	void setDebugExtension();
	void createCommandPool();
	void createCommandBuffer();
	void createSwapChain();
	void createRenderPass();
	void createImageView();
	void createFrameBuffer();
	void createShaderModule();
	void createDescriptorPool();
	void createDescriptorSetLayout();
	void createDescriptorSet();
	void createPipelineLayout();
	void createSemaphore();
	void createFence();
	void createGraphicPipeline();
	void createBuffer();
	void createMatrixBuffer();
	void createTextureImage();
	void prepareDrawCommand();
	void prepareMatrix();

	void updateMatrix(int index);
	void submitDraw(uint32_t index);
	uint32_t chooseHeapFromFlags(
		vk::PhysicalDevice device,
		const vk::MemoryRequirements& memoryRequirements,
		vk::MemoryPropertyFlags requiredFlags,
		vk::MemoryPropertyFlags preferredFlags);

	std::vector<const char*> getAvailableWSIExtensions();
	vk::SurfaceKHR createVulkanSurface(const vk::Instance& instance, SDL_Window* window);
	char* VulkanDemo::readSpv(const char *filename, size_t *psize);
private:
	glm::mat4 projection_;
	glm::mat4 model_;
	glm::mat4 view_;
	glm::mat4 rotate_;
	vk::Instance instance_;
	vk::SurfaceKHR surface_;
	vk::Device device_;
	vk::PhysicalDevice physicalDevice_;
	vk::ShaderModule vertexShaderModule_;
	vk::ShaderModule fragmentShaderModule_;
	vk::PipelineCache pipelineCache_;
	vk::PipelineLayout pipelineLayout_;
	vk::RenderPass renderPass_;
	vk::Pipeline graphicsPipeline_;
	vk::CommandPool commandPool_;
	vk::DescriptorPool descriptorPool_;
	vk::DescriptorSetLayout descriptorSetLayout_;
	std::vector<vk::Sampler> sampler_;
	std::vector<vk::ImageView> textureImageView_;
	std::vector<vk::DescriptorSet> descriptorSets_;
	std::vector<vk::CommandBuffer> commandBuffers_;
	std::vector<vk::Framebuffer> frameBuffers_;
	std::vector<vk::ImageView> imageViews_;
	std::vector<vk::DeviceMemory> imageMemory_;
	std::vector<vk::DeviceMemory> bufferMemory_;
	std::vector<vk::DeviceMemory> matrixMemory_;
	std::vector<vk::DeviceMemory> textureMemory_;
	std::vector<vk::Image> texture_;
	std::vector<vk::Image> images_;
	std::vector<vk::Buffer> buffers_;
	std::vector<vk::Buffer> matrixs_;
	std::vector<vk::Fence> fences_;
	vk::SwapchainKHR swapChain_;
	vk::Queue queue_;
	std::vector<vk::Semaphore> imageReady_;
	std::vector<vk::Semaphore> drawComplete_;
	SDL_Window* window_;
	bool isSeparatePresentQueue_ = false;
	int graphicsQueueFamilyIndex_ = -1;
	int presentQueueFamilyIndex_ = -1;
	bool isPrepared = false;
	int frameIndex_ = 0;
	float theta_[2] = { 0,0 };
};