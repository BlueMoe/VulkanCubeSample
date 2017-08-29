#include "VulkanDemo.h"
#include <glm/glm.hpp>
#include "SDL2\SDL.h"
#include "SDL2\SDL_syswm.h"
#include <iostream>
#include <vulkan/vulkan.hpp>
#include "GlobalDefine.h"
#define STB_IMAGE_IMPLEMENTATION
#include <stb/stb_image.h>

bool VulkanDemo::initInstance(int width, int height, std::string windowName)
{
	if (!createWindow(width, height, windowName))
		return false;
	if (!initInstance())
		return false;
	surface_ = createVulkanSurface(instance_, window_);
	if (surface_ == nullptr)
		return false;
	return true;
}

bool VulkanDemo::prepare()
{
	createLogicDevice();
	setDebugExtension();
	createCommandPool();
	createCommandBuffer();
	createSwapChain();
	prepareMatrix();
	createTextureImage();
	createMatrixBuffer();
	createDescriptorSetLayout();
	createDescriptorPool();
	createDescriptorSet();
	createPipelineLayout();
	createRenderPass();
	createImageView();
	createFrameBuffer();
	createShaderModule();
	createSemaphore();
	createFence();
	createGraphicPipeline();
	createBuffer();
	prepareDrawCommand();

	isPrepared = true;

	return true;
}

bool VulkanDemo::createWindow(int width, int height, std::string windowName)
{
	// Create an SDL window that supports Vulkan and OpenGL rendering.
	if (SDL_Init(SDL_INIT_VIDEO) != 0) {
		std::cout << "Could not initialize SDL." << std::endl;
		return false;
	}
	window_ = SDL_CreateWindow(windowName.c_str(), SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED, width, height, SDL_WINDOW_OPENGL);
	if (window_ == NULL) {
		std::cout << "Could not create SDL window." << std::endl;
		return false;
	}
	return true;
}

bool VulkanDemo::initInstance()
{
	auto lay = vk::enumerateInstanceLayerProperties();

	std::vector<const char*> layer;
	std::vector<const char*> extension = getAvailableWSIExtensions();
	layer.push_back("VK_LAYER_LUNARG_standard_validation");
	layer.push_back("VK_LAYER_RENDERDOC_Capture");

	auto appInfo = vk::ApplicationInfo();
	appInfo.setApiVersion(VK_API_VERSION_1_0);
	appInfo.setApplicationVersion(1);
	appInfo.setEngineVersion(1);
	appInfo.setPApplicationName("Vulkan Demo by Amoe");
	appInfo.setPEngineName("LunarG SDK");
	appInfo.setPNext(nullptr);

	auto ext = vk::enumerateInstanceExtensionProperties();

	auto instanceInfo = vk::InstanceCreateInfo();
	instanceInfo.setEnabledExtensionCount(static_cast<uint32_t>(extension.size()));
	instanceInfo.setEnabledLayerCount(static_cast<uint32_t>(layer.size()));
	instanceInfo.setFlags(vk::InstanceCreateFlags());
	instanceInfo.setPApplicationInfo(&appInfo);
	instanceInfo.setPNext(nullptr);
	instanceInfo.setPpEnabledExtensionNames(extension.data());
	instanceInfo.setPpEnabledLayerNames(layer.data());
	try
	{
		instance_ = vk::createInstance(instanceInfo);
	}
	catch (const std::exception& e)
	{
		std::cout << "Could not create a Vulkan instance: " << e.what() << std::endl;
		return false;
	}

	return true;
}

void VulkanDemo::createLogicDevice()
{
	//just need first GPU
	auto physicalDevices = instance_.enumeratePhysicalDevices();
	physicalDevice_ = physicalDevices[0];

	auto queueProp = physicalDevice_.getQueueFamilyProperties();

	std::vector<VkBool32> supportsPresent = std::vector<VkBool32>(queueProp.size());

	for (int i = 0; i < queueProp.size(); i++)
	{
		supportsPresent[i] = physicalDevice_.getSurfaceSupportKHR(i, surface_);
	}

	float queuePriority = 0.0f;
	int graphicsFamilyIndex = -1;
	int presentFamilyIndex = -1;
	int i = 0;

	for (const auto& prop : queueProp)
	{
		//only support graphics
		if (graphicsFamilyIndex == -1 && prop.queueFlags & vk::QueueFlagBits::eGraphics)
		{
			graphicsFamilyIndex = i;
		}

		//support both graphics and present family index
		if (supportsPresent[i] == VK_TRUE)
		{
			graphicsFamilyIndex = i;
			presentFamilyIndex = i;
			break;
		}
		i++;
	}

	//if no support both index , find an index support present;
	if (presentFamilyIndex == -1)
	{
		for (int i = 0; i < supportsPresent.size(); i++)
		{
			if (supportsPresent[i] == VK_TRUE)
			{
				presentFamilyIndex = i;
				break;
			}
		}
	}

	assert(graphicsFamilyIndex != -1 && presentFamilyIndex != -1);

	graphicsQueueFamilyIndex_ = graphicsFamilyIndex;
	presentQueueFamilyIndex_ = presentFamilyIndex;
	isSeparatePresentQueue_ = (graphicsFamilyIndex != presentFamilyIndex);

	std::vector<const char*> extensions;
	extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
	extensions.push_back(VK_EXT_DEBUG_MARKER_EXTENSION_NAME);

	vk::DeviceQueueCreateInfo queueCreateInfo[2];
	queueCreateInfo[0].setQueueFamilyIndex(graphicsFamilyIndex);
	queueCreateInfo[0].setQueueCount(1);
	queueCreateInfo[0].setPQueuePriorities(&queuePriority);

	auto deviceInfo = vk::DeviceCreateInfo();
	deviceInfo.setEnabledLayerCount(0);
	deviceInfo.setPEnabledFeatures(&vk::PhysicalDeviceFeatures());
	deviceInfo.setEnabledExtensionCount(extensions.size());
	deviceInfo.setPpEnabledExtensionNames(extensions.data());
	deviceInfo.setPpEnabledLayerNames(nullptr);
	deviceInfo.setPQueueCreateInfos(queueCreateInfo);
	deviceInfo.setQueueCreateInfoCount(1);

	if (isSeparatePresentQueue_)
	{
		queueCreateInfo[1].setQueueFamilyIndex(presentFamilyIndex);
		queueCreateInfo[1].setQueueCount(1);
		queueCreateInfo[1].setPQueuePriorities(&queuePriority);

		deviceInfo.setQueueCreateInfoCount(2);
	}
	device_ = physicalDevice_.createDevice(deviceInfo);

	queue_ = device_.getQueue(graphicsFamilyIndex, 0);

}

void VulkanDemo::setDebugExtension()
{

}

void VulkanDemo::createCommandPool()
{
	auto commandCreateInfo = vk::CommandPoolCreateInfo();
	commandCreateInfo.setQueueFamilyIndex(graphicsQueueFamilyIndex_);
	commandPool_ = device_.createCommandPool(commandCreateInfo);
}

void VulkanDemo::createCommandBuffer()
{
	auto allocateCommandBufferInfo = vk::CommandBufferAllocateInfo();
	allocateCommandBufferInfo.setCommandPool(commandPool_);
	allocateCommandBufferInfo.setCommandBufferCount(GlobalDefine::swapChainImageCount);
	allocateCommandBufferInfo.setLevel(vk::CommandBufferLevel::ePrimary);

	auto commandBuffers = device_.allocateCommandBuffers(allocateCommandBufferInfo);
	commandBuffers_ = commandBuffers;

	auto inheritanceInfo = vk::CommandBufferInheritanceInfo();
}

void VulkanDemo::createImageView()
{
	for (const auto& image : images_)
	{
		auto imageViewCreateInfo = vk::ImageViewCreateInfo()
			.setImage(image)
			.setViewType(vk::ImageViewType::e2D)
			.setFormat(vk::Format::eB8G8R8A8Unorm)
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor,0,~0u,0,~0u));

		auto imageView = device_.createImageView(imageViewCreateInfo);
		imageViews_.push_back(imageView);
	}
}

void VulkanDemo::createFrameBuffer()
{
	for (const auto& imageView : imageViews_)
	{
		auto framebufferCreateInfo = vk::FramebufferCreateInfo();
		framebufferCreateInfo.setRenderPass(renderPass_);
		framebufferCreateInfo.setAttachmentCount(1);
		framebufferCreateInfo.setPAttachments(&imageView);
		framebufferCreateInfo.setWidth(800);
		framebufferCreateInfo.setHeight(800);
		framebufferCreateInfo.setLayers(1);
		auto frameBuffer = device_.createFramebuffer(framebufferCreateInfo);
		frameBuffers_.push_back(frameBuffer);
	}
}

void VulkanDemo::createSwapChain()
{
	auto swapChainSupportInfo = physicalDevice_.getSurfaceCapabilitiesKHR(surface_);

	vk::CompositeAlphaFlagBitsKHR alphaFlag[4] = {
		vk::CompositeAlphaFlagBitsKHR::eInherit,
		vk::CompositeAlphaFlagBitsKHR::eOpaque,
		vk::CompositeAlphaFlagBitsKHR::ePostMultiplied,
		vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
	};
	vk::CompositeAlphaFlagBitsKHR alphaFlagBit;
	for (int i = 0; i < 4; ++i)
	{
		if (swapChainSupportInfo.supportedCompositeAlpha & alphaFlag[i])
		{
			alphaFlagBit = alphaFlag[i];
			break;
		}
	}

	auto surfaceFormat = physicalDevice_.getSurfaceFormatsKHR(surface_);
	vk::Format format;
	vk::ColorSpaceKHR colorSpace;
	if (surfaceFormat[0].format == vk::Format::eUndefined)
	{
		format = vk::Format::eR8G8B8A8Srgb;
	}
	else
	{
		format = surfaceFormat[0].format;
	}
	colorSpace = surfaceFormat[0].colorSpace;

	auto presentMode = physicalDevice_.getSurfacePresentModesKHR(surface_);

	auto swapChainPresentMode = vk::PresentModeKHR::eFifo;

	for (const auto& mode : presentMode)
	{
		if (swapChainPresentMode != mode)
		{
			swapChainPresentMode = mode;
			break;
		}
	}

	vk::SurfaceTransformFlagBitsKHR transform;
	if (swapChainSupportInfo.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
	{
		transform = vk::SurfaceTransformFlagBitsKHR::eIdentity;
	}
	else
	{
		transform = swapChainSupportInfo.currentTransform;
	}


	auto swapChainCreateInfo = vk::SwapchainCreateInfoKHR();
	swapChainCreateInfo.setMinImageCount(GlobalDefine::swapChainImageCount);
	swapChainCreateInfo.setImageFormat(format);
	swapChainCreateInfo.setImageColorSpace(colorSpace);
	swapChainCreateInfo.setImageExtent(swapChainSupportInfo.currentExtent);
	swapChainCreateInfo.setImageArrayLayers(1);
	swapChainCreateInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
	swapChainCreateInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	swapChainCreateInfo.setPreTransform(transform);
	swapChainCreateInfo.setCompositeAlpha(alphaFlagBit);
	swapChainCreateInfo.setPresentMode(swapChainPresentMode);
	swapChainCreateInfo.setSurface(surface_);

	swapChain_ = device_.createSwapchainKHR(swapChainCreateInfo);
	images_ = device_.getSwapchainImagesKHR(swapChain_);
}

void VulkanDemo::createSemaphore()
{
	auto semaphoreCreateInfo = vk::SemaphoreCreateInfo();
	for (const auto& image : images_)
	{
		imageReady_.push_back(device_.createSemaphore(semaphoreCreateInfo));
		drawComplete_.push_back(device_.createSemaphore(semaphoreCreateInfo));
	}
}

void VulkanDemo::createFence()
{
	auto fenceCreateInfo = vk::FenceCreateInfo()
		.setFlags(vk::FenceCreateFlagBits::eSignaled);
	for (const auto& image : images_)
	{
		fences_.push_back(device_.createFence(fenceCreateInfo));
	}
}

void VulkanDemo::createShaderModule()
{
	uint32_t codeSize = 0;
	void* vertexCode = readSpv("simple-vert.spv", &codeSize);

	auto shaderModuleCreateInfo = vk::ShaderModuleCreateInfo();
	shaderModuleCreateInfo.setCodeSize(codeSize);
	shaderModuleCreateInfo.setPCode(static_cast<uint32_t*>(vertexCode));

	vertexShaderModule_ = device_.createShaderModule(shaderModuleCreateInfo);

	delete vertexCode;

	void* fragmentCode = readSpv("simple-frag.spv", &codeSize);

	shaderModuleCreateInfo = vk::ShaderModuleCreateInfo();
	shaderModuleCreateInfo.setCodeSize(codeSize);
	shaderModuleCreateInfo.setPCode(static_cast<uint32_t*>(fragmentCode));

	fragmentShaderModule_ = device_.createShaderModule(shaderModuleCreateInfo);

	delete fragmentCode;

}

void VulkanDemo::createDescriptorPool()
{
	vk::DescriptorPoolSize descriptorPoolSize[2];
	descriptorPoolSize[0]
		.setDescriptorCount(GlobalDefine::swapChainImageCount)
		.setType(vk::DescriptorType::eUniformBuffer);
	descriptorPoolSize[1]
		.setDescriptorCount(GlobalDefine::swapChainImageCount)
		.setType(vk::DescriptorType::eCombinedImageSampler);


	auto decriptorPoolCreateInfo = vk::DescriptorPoolCreateInfo()
		.setMaxSets(GlobalDefine::swapChainImageCount)
		.setPoolSizeCount(2)
		.setPPoolSizes(descriptorPoolSize)
		.setFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet);

	descriptorPool_ = device_.createDescriptorPool(decriptorPoolCreateInfo);
}

void VulkanDemo::createDescriptorSetLayout()
{
	vk::DescriptorSetLayoutBinding bindings[2];
	bindings[0].setBinding(0);
	bindings[0].setDescriptorCount(1);
	bindings[0].setDescriptorType(vk::DescriptorType::eUniformBuffer);
	bindings[0].setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eVertex));
	bindings[0].setPImmutableSamplers(nullptr);

 	bindings[1].setBinding(1);
 	bindings[1].setDescriptorCount(1);
 	bindings[1].setDescriptorType(vk::DescriptorType::eCombinedImageSampler);
 	bindings[1].setStageFlags(vk::ShaderStageFlags(vk::ShaderStageFlagBits::eFragment));
 	bindings[1].setPImmutableSamplers(nullptr);

	auto descriptorSetLayoutCreateInfo = vk::DescriptorSetLayoutCreateInfo();
	descriptorSetLayoutCreateInfo.setBindingCount(2);
	descriptorSetLayoutCreateInfo.setPBindings(bindings);

	descriptorSetLayout_ = device_.createDescriptorSetLayout(descriptorSetLayoutCreateInfo);
}

void VulkanDemo::createDescriptorSet()
{
	uint32_t swapChainImageCount = images_.size();

	for (int i = 0; i < swapChainImageCount; i++)
	{
		auto descriptorSetAllocateInfo = vk::DescriptorSetAllocateInfo()
			.setDescriptorPool(descriptorPool_)
			.setDescriptorSetCount(1)
			.setPSetLayouts(&descriptorSetLayout_);

		auto set = device_.allocateDescriptorSets(descriptorSetAllocateInfo);

		vk::DescriptorBufferInfo bufferInfo;
		bufferInfo.setBuffer(matrixs_[i])
			.setOffset(0)
			.setRange(sizeof(uniformStruct));

		vk::DescriptorImageInfo imageInfo;
		imageInfo.setImageLayout(vk::ImageLayout::eGeneral)
			.setImageView(textureImageView_[i])
			.setSampler(sampler_[i]);

		vk::WriteDescriptorSet write[2];
		write[0].setDstSet(set[0])
			.setDstBinding(0)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eUniformBuffer)
			.setPBufferInfo(&bufferInfo);

		write[1].setDstSet(set[0])
			.setDstBinding(1)
			.setDescriptorCount(1)
			.setDescriptorType(vk::DescriptorType::eCombinedImageSampler)
			.setPImageInfo(&imageInfo);

		device_.updateDescriptorSets(2, write, 0, nullptr);
		descriptorSets_.push_back(set[0]);
	}
}

void VulkanDemo::updateMatrix(int index)
{
	theta_[index] += GlobalDefine::rotateSpeed;

	model_ *= rotate_;

	auto MVP = projection_ * view_ * model_;

	auto ptr = device_.mapMemory(matrixMemory_[index],0, VK_WHOLE_SIZE, vk::MemoryMapFlags());

	memcpy(ptr, &MVP, sizeof(MVP));

	device_.unmapMemory(matrixMemory_[index]);

}

void VulkanDemo::createPipelineLayout()
{
	auto pipeLineCacheCreateInfo = vk::PipelineCacheCreateInfo();
	pipelineCache_ = device_.createPipelineCache(pipeLineCacheCreateInfo);

	auto pipelineLayoutCreateInfo = vk::PipelineLayoutCreateInfo();
	pipelineLayoutCreateInfo.setSetLayoutCount(1);
	pipelineLayoutCreateInfo.setPSetLayouts(&descriptorSetLayout_);

	pipelineLayout_ = device_.createPipelineLayout(pipelineLayoutCreateInfo);
}

void VulkanDemo::createRenderPass()
{
	auto format = physicalDevice_.getSurfaceFormatsKHR(surface_);

	auto attachmentDescription = vk::AttachmentDescription();
	attachmentDescription.setFormat(vk::Format::eB8G8R8A8Unorm);
	attachmentDescription.setSamples(vk::SampleCountFlagBits::e1);
	attachmentDescription.setLoadOp(vk::AttachmentLoadOp::eClear);
	attachmentDescription.setStoreOp(vk::AttachmentStoreOp::eStore);
	attachmentDescription.setStencilLoadOp(vk::AttachmentLoadOp::eDontCare);
	attachmentDescription.setStencilStoreOp(vk::AttachmentStoreOp::eDontCare);
	attachmentDescription.setInitialLayout(vk::ImageLayout::eUndefined);
	attachmentDescription.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);

	auto attachmentReference = vk::AttachmentReference();
	attachmentReference.setAttachment(0);
	attachmentReference.setLayout(vk::ImageLayout::eColorAttachmentOptimal);

	auto subpassDescription = vk::SubpassDescription();
	subpassDescription.setPipelineBindPoint(vk::PipelineBindPoint::eGraphics);
	subpassDescription.setColorAttachmentCount(1);
	subpassDescription.setPColorAttachments(&attachmentReference);

	auto renderPassCreateInfo = vk::RenderPassCreateInfo();
	renderPassCreateInfo.setAttachmentCount(1);
	renderPassCreateInfo.setPAttachments(&attachmentDescription);
	renderPassCreateInfo.setDependencyCount(0);
	renderPassCreateInfo.setPDependencies(nullptr);
	renderPassCreateInfo.setPSubpasses(&subpassDescription);
	renderPassCreateInfo.setSubpassCount(1);

	renderPass_ = device_.createRenderPass(renderPassCreateInfo);
}

void VulkanDemo::createGraphicPipeline()
{
	auto prop = physicalDevice_.getProperties();

	vk::PipelineShaderStageCreateInfo shaderStageCreateInfo[2];
	shaderStageCreateInfo[0]
		.setStage(vk::ShaderStageFlagBits::eVertex)
		.setModule(vertexShaderModule_)
		.setPName("main");

	shaderStageCreateInfo[1]
		.setStage(vk::ShaderStageFlagBits::eFragment)
		.setModule(fragmentShaderModule_)
		.setPName("main");

	auto vertexDescription = vk::VertexInputBindingDescription();
	vertexDescription.setBinding(0)
		.setStride(sizeof(vertex))
		.setInputRate(vk::VertexInputRate::eVertex);
	

	vk::VertexInputAttributeDescription vertexAttributeDescription[4];
	vertexAttributeDescription[0]
		.setLocation(0)
		.setBinding(0)
		.setFormat(vk::Format::eR32G32B32Sfloat)
		.setOffset(offsetof(vertex, pos));
	vertexAttributeDescription[1].
		setLocation(1)
		.setBinding(0)
		.setFormat(vk::Format::eR32G32B32Sfloat)
		.setOffset(offsetof(vertex, color));
	vertexAttributeDescription[2].
		setLocation(2)
		.setBinding(0)
		.setFormat(vk::Format::eR32G32Sfloat)
		.setOffset(offsetof(vertex, uv));
	vertexAttributeDescription[3].
		setLocation(3)
		.setBinding(0)
		.setFormat(vk::Format::eR32G32B32Sfloat)
		.setOffset(offsetof(vertex, normal));


	auto vertexInputStateCreateInfo = vk::PipelineVertexInputStateCreateInfo()
		.setVertexAttributeDescriptionCount(4)
		.setPVertexAttributeDescriptions(vertexAttributeDescription)
		.setVertexBindingDescriptionCount(1)
		.setPVertexBindingDescriptions(&vertexDescription);

	auto inputAssemblyCreateInfo = vk::PipelineInputAssemblyStateCreateInfo()
		.setPrimitiveRestartEnable(VK_FALSE)
		.setTopology(vk::PrimitiveTopology::eTriangleList);

	auto viewport = vk::Viewport()
		.setWidth(800)
		.setHeight(800)
		.setX(0)
		.setY(0)
		.setMinDepth(0)
		.setMaxDepth(1);

	auto scissor = vk::Rect2D()
		.setOffset({ 0,0 })
		.setExtent({ 800,800 });

	auto viewportStateCreateInfo = vk::PipelineViewportStateCreateInfo()
		.setPScissors(&scissor)
		.setPViewports(&viewport)
		.setScissorCount(1)
		.setViewportCount(1);

	auto rasterizationStateCreateInfo = vk::PipelineRasterizationStateCreateInfo()
		.setDepthClampEnable(VK_FALSE)
		.setRasterizerDiscardEnable(VK_FALSE)
		.setPolygonMode(vk::PolygonMode::eFill)
		.setCullMode(vk::CullModeFlagBits::eBack)
		.setFrontFace(vk::FrontFace::eCounterClockwise)
		.setDepthBiasEnable(VK_FALSE)
		.setDepthBiasConstantFactor(0)
		.setDepthBiasClamp(0)
		.setDepthBiasSlopeFactor(0)
		.setLineWidth(1);

	auto multiSampleStateCreateInfo = vk::PipelineMultisampleStateCreateInfo()
		.setRasterizationSamples(vk::SampleCountFlagBits::e1);

	auto colorBlendAttachment = vk::PipelineColorBlendAttachmentState()
		.setColorWriteMask(vk::ColorComponentFlags(
			vk::ColorComponentFlagBits::eR |
			vk::ColorComponentFlagBits::eG |
			vk::ColorComponentFlagBits::eB |
			vk::ColorComponentFlagBits::eA))
		.setAlphaBlendOp(vk::BlendOp::eAdd)
		.setColorBlendOp(vk::BlendOp::eAdd)
		.setBlendEnable(VK_TRUE)
		.setSrcColorBlendFactor(vk::BlendFactor::eSrcAlpha)
		.setDstColorBlendFactor(vk::BlendFactor::eOneMinusSrcAlpha)
		.setSrcAlphaBlendFactor(vk::BlendFactor::eOne)
		.setDstAlphaBlendFactor(vk::BlendFactor::eZero);

	auto colorBlendStateCreateInfo = vk::PipelineColorBlendStateCreateInfo()
		.setAttachmentCount(1)
		.setPAttachments(&colorBlendAttachment);

	auto depthStencilStateCreateInfo = vk::PipelineDepthStencilStateCreateInfo()
		.setDepthTestEnable(VK_TRUE)
		.setDepthWriteEnable(VK_FALSE)
		.setDepthCompareOp(vk::CompareOp::eLessOrEqual)
		.setDepthBoundsTestEnable(VK_FALSE)
		.setStencilTestEnable(VK_FALSE)
		.setMinDepthBounds(0)
		.setMaxDepthBounds(0);

	auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo()
		.setStageCount(2)
		.setPStages(shaderStageCreateInfo)
		.setPVertexInputState(&vertexInputStateCreateInfo)
		.setPInputAssemblyState(&inputAssemblyCreateInfo)
		.setPViewportState(&viewportStateCreateInfo)
		.setPTessellationState(nullptr)
		.setPRasterizationState(&rasterizationStateCreateInfo)
		.setPMultisampleState(&multiSampleStateCreateInfo)
		.setPDepthStencilState(&depthStencilStateCreateInfo)
		.setPColorBlendState(&colorBlendStateCreateInfo)
		.setPDynamicState(nullptr)
		.setLayout(pipelineLayout_)
		.setRenderPass(renderPass_)
		.setSubpass(0)
		.setBasePipelineHandle(nullptr)
		.setBasePipelineIndex(-1);

	graphicsPipeline_ = device_.createGraphicsPipeline(nullptr, pipelineCreateInfo);
}

void VulkanDemo::createBuffer()
{
	for (const auto& image : images_)
	{
		auto bufferCreateInfo = vk::BufferCreateInfo()
			.setSize(sizeof(GlobalDefine::triangleVertex))
			.setUsage(vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst);

		auto buffer = device_.createBuffer(bufferCreateInfo);
		buffers_.push_back(buffer);

		auto memoryRequire = device_.getBufferMemoryRequirements(buffer);

		auto memoryType = chooseHeapFromFlags(
			physicalDevice_,
			memoryRequire,
			vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent),
			vk::MemoryPropertyFlags(vk::MemoryPropertyFlagBits::eHostVisible));

		auto allocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memoryRequire.size)
			.setMemoryTypeIndex(memoryType);

		auto bufferMemory = device_.allocateMemory(allocateInfo);
		bufferMemory_.push_back(bufferMemory);
		auto ptr = device_.mapMemory(bufferMemory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());

		memcpy(ptr, GlobalDefine::triangleVertex, sizeof(GlobalDefine::triangleVertex));

		device_.unmapMemory(bufferMemory);

		device_.bindBufferMemory(buffer, bufferMemory, 0);

	}
}

//transformation matrix buffer
void VulkanDemo::createMatrixBuffer()
{
	int i = 0;
	for (const auto& image : images_)
	{
		auto bufferCreateInfo = vk::BufferCreateInfo()
			.setSize(sizeof(uniformStruct))
			.setUsage(vk::BufferUsageFlagBits::eUniformBuffer);

		auto buffer = device_.createBuffer(bufferCreateInfo);
		matrixs_.push_back(buffer);

		auto memoryRequire = device_.getBufferMemoryRequirements(buffer);

		auto memoryType = chooseHeapFromFlags(
			physicalDevice_,
			memoryRequire,
			vk::MemoryPropertyFlagBits::eHostVisible,
			vk::MemoryPropertyFlagBits::eHostVisible);

		auto allocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memoryRequire.size)
			.setMemoryTypeIndex(memoryType);

		auto memory = device_.allocateMemory(allocateInfo);
		matrixMemory_.push_back(memory);
		
		device_.bindBufferMemory(buffer, memory, 0);

		updateMatrix(i++);
	}
}

void VulkanDemo::createTextureImage()
{
	int width, height, channel;
	stbi_uc* pixels = stbi_load("cube.png", &width, &height, &channel, STBI_rgb_alpha);
	assert(pixels != nullptr);

	for (int i = 0; i < GlobalDefine::swapChainImageCount; ++i)
	{
		auto imageCreateInfo = vk::ImageCreateInfo()
			.setImageType(vk::ImageType::e2D)
			.setFormat(vk::Format::eR8G8B8A8Unorm)
			.setExtent({ static_cast<uint32_t>(width),static_cast<uint32_t>(height),1 })
			.setMipLevels(1)
			.setArrayLayers(1)
			.setSamples(vk::SampleCountFlagBits::e1)
			.setTiling(vk::ImageTiling::eLinear)
			.setUsage(vk::ImageUsageFlagBits::eSampled)
			.setSharingMode(vk::SharingMode::eExclusive)
			.setQueueFamilyIndexCount(0)
			.setPQueueFamilyIndices(nullptr)
			.setInitialLayout(vk::ImageLayout::ePreinitialized);

		auto image = device_.createImage(imageCreateInfo);
		texture_.push_back(image);

		auto memoryRequire = device_.getImageMemoryRequirements(image);

		auto memoryType = chooseHeapFromFlags(
			physicalDevice_,
			memoryRequire,
			vk::MemoryPropertyFlagBits::eHostVisible,
			vk::MemoryPropertyFlagBits::eHostVisible
		);

		auto memoryAllocateInfo = vk::MemoryAllocateInfo()
			.setAllocationSize(memoryRequire.size)
			.setMemoryTypeIndex(memoryType);

		auto memory = device_.allocateMemory(memoryAllocateInfo);
		textureMemory_.push_back(memory);

		auto ptr = device_.mapMemory(memory, 0, VK_WHOLE_SIZE, vk::MemoryMapFlags());
		memcpy(ptr, pixels, width * height * 4);

		device_.unmapMemory(memory);

		device_.bindImageMemory(image, memory, 0);

		auto imageViewCreateInfo = vk::ImageViewCreateInfo()
			.setViewType(vk::ImageViewType::e2D)
			.setImage(image)
			.setFormat(vk::Format::eR8G8B8A8Unorm)
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		auto imageView = device_.createImageView(imageViewCreateInfo);
		textureImageView_.push_back(imageView);

		auto samplerCreateInfo = vk::SamplerCreateInfo()
			.setMagFilter(vk::Filter::eNearest)
			.setMinFilter(vk::Filter::eNearest)
			.setMipmapMode(vk::SamplerMipmapMode::eNearest)
			.setAddressModeU(vk::SamplerAddressMode::eClampToEdge)
			.setAddressModeV(vk::SamplerAddressMode::eClampToEdge)
			.setAddressModeW(vk::SamplerAddressMode::eClampToEdge)
			.setMipLodBias(0.0f)
			.setAnisotropyEnable(VK_FALSE)
			.setMaxAnisotropy(1)
			.setCompareEnable(VK_FALSE)
			.setCompareOp(vk::CompareOp::eNever)
			.setMaxLod(0)
			.setMinLod(0)
			.setBorderColor(vk::BorderColor::eFloatTransparentBlack)
			.setUnnormalizedCoordinates(VK_FALSE);
		auto sampler = device_.createSampler(samplerCreateInfo);
		sampler_.push_back(sampler);
	}
}

void VulkanDemo::prepareDrawCommand()
{
	int index = 0;
	for (const auto& image : images_)
	{

		auto commandBufferBeginInfo = vk::CommandBufferBeginInfo();
		commandBufferBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eSimultaneousUse);

		auto clearValue = vk::ClearValue()
			.setColor(vk::ClearColorValue().setFloat32({ 0.1f, 0.1f, 0.1f, 0.0f }))
			.setDepthStencil(vk::ClearDepthStencilValue().setDepth(1).setStencil(0u));

		auto renderPassBeginInfo = vk::RenderPassBeginInfo()
			.setFramebuffer(frameBuffers_[index])
			.setRenderArea(vk::Rect2D({ 0, 0 }, { 800, 800 }))
			.setRenderPass(renderPass_)
			.setClearValueCount(1)
			.setPClearValues(&clearValue);

		auto barrier = vk::ImageMemoryBarrier()
			.setSrcAccessMask(vk::AccessFlagBits::eHostWrite)
			.setDstAccessMask(vk::AccessFlagBits::eShaderRead | vk::AccessFlagBits::eColorAttachmentRead)
			.setOldLayout(vk::ImageLayout::ePreinitialized)
			.setNewLayout(vk::ImageLayout::eShaderReadOnlyOptimal)
			.setSrcQueueFamilyIndex(0)
			.setDstQueueFamilyIndex(0)
			.setImage(texture_[index])
			.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

		vk::DeviceSize offset = 0;

		commandBuffers_[index].begin(commandBufferBeginInfo);
		//commandBuffers_[index].pipelineBarrier(
		//	vk::PipelineStageFlagBits::eTopOfPipe,
		//	vk::PipelineStageFlagBits::eFragmentShader,
		//	vk::DependencyFlags(),
		//	0, nullptr,
		//	0, nullptr,
		//	1, &barrier);
		commandBuffers_[index].bindVertexBuffers(0, 1, &buffers_[index], &offset);
		commandBuffers_[index].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout_, 0, 1, &descriptorSets_[0], 0, nullptr);
		commandBuffers_[index].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline_);
		commandBuffers_[index].beginRenderPass(&renderPassBeginInfo, vk::SubpassContents::eInline);
		commandBuffers_[index].draw(sizeof(GlobalDefine::triangleVertex) / sizeof(vertex), 1, 0, 0);
		commandBuffers_[index].endRenderPass();
		commandBuffers_[index].end();
		index++;
	}
}

void VulkanDemo::prepareMatrix()
{
	projection_ = glm::perspective(45.0f * glm::pi<float>() / 180.0f, 1.0f, 0.0f, 100.0f);

	projection_[1][1] *= -1;

	view_ = glm::lookAt(
		glm::vec3(0, 3, 5),
		glm::vec3(0, 0, 0),
		glm::vec3(0, 1, 0)
	);

	model_ = glm::mat4(
		1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1
	);

	rotate_ = glm::rotate(model_, GlobalDefine::rotateSpeed * glm::pi<float>() / 180, glm::vec3(0, 1, 0));
}

void VulkanDemo::run()
{
	if (!isPrepared)
	{
		return;
	}
	updateMatrix(frameIndex_);
	submitDraw(frameIndex_);
	frameIndex_++;
	frameIndex_ %=images_.size();
}

void VulkanDemo::stop()
{
	isPrepared = false;

	device_.waitIdle();

	for (const auto & fence : fences_)
	{
		device_.waitForFences(1, &fence, VK_TRUE, ~0U);
		device_.destroyFence(fence);
	}
	for (const auto & semaphore : imageReady_)
	{
		device_.destroySemaphore(semaphore);
	}
	for (const auto & semaphore : drawComplete_)
	{
		device_.destroySemaphore(semaphore);
	}
	for (const auto & frameBuffer : frameBuffers_)
	{
		device_.destroyFramebuffer(frameBuffer);
	}
	device_.destroyPipeline(graphicsPipeline_);
	device_.destroyPipelineCache(pipelineCache_);
	device_.destroyRenderPass(renderPass_);
	device_.destroyPipelineLayout(pipelineLayout_);
	for (const auto & view : imageViews_)
	{
		device_.destroyImageView(view);
	}
	for (const auto & buffer : buffers_)
	{
		device_.destroyBuffer(buffer);
	}

	for (const auto & buffer : matrixs_)
	{
		device_.destroyBuffer(buffer);
	}

	for (const auto & image : texture_)
	{
		device_.destroyImage(image);
	}

	for (const auto & memory : imageMemory_)
	{
		device_.freeMemory(memory);
	}
	for (const auto & memory : bufferMemory_)
	{
		device_.freeMemory(memory);
	}
	for (const auto & memory : matrixMemory_)
	{
		device_.freeMemory(memory);
	}
	for (const auto & memory : textureMemory_)
	{
		device_.freeMemory(memory);
	}
	for (const auto& sampler : sampler_)
	{
		device_.destroySampler(sampler);
	}	
	for (const auto& textureImageView : textureImageView_)
	{
		device_.destroyImageView(textureImageView);
	}

	device_.freeCommandBuffers(commandPool_, commandBuffers_.size(), commandBuffers_.data());
	device_.destroyCommandPool(commandPool_);
	device_.destroyShaderModule(vertexShaderModule_);
	device_.destroyShaderModule(fragmentShaderModule_);
	device_.destroySwapchainKHR(swapChain_);

	for (const auto& set : descriptorSets_)
	{
		device_.freeDescriptorSets(descriptorPool_, set);
	}
	device_.destroyDescriptorSetLayout(descriptorSetLayout_);
	device_.destroyDescriptorPool(descriptorPool_);
	device_.waitIdle();
	device_.destroy();
	instance_.destroySurfaceKHR(surface_);
	SDL_DestroyWindow(window_);
	SDL_Quit();
	instance_.destroy();
}


void VulkanDemo::submitDraw(uint32_t index)
{
	device_.waitForFences(1, &fences_[index], VK_TRUE, ~0U);
	device_.resetFences(1, &fences_[index]);

	device_.acquireNextImageKHR(swapChain_, ~0U, imageReady_[index], nullptr, &index);

	auto submitInfo = vk::SubmitInfo();
	submitInfo.setCommandBufferCount(1)
		.setPCommandBuffers(&commandBuffers_[index])
		.setSignalSemaphoreCount(1)
		.setPSignalSemaphores(&drawComplete_[index])
		.setWaitSemaphoreCount(1)
		.setPWaitSemaphores(&imageReady_[index])
		.setPWaitDstStageMask(&vk::PipelineStageFlags(vk::PipelineStageFlagBits::eColorAttachmentOutput));

	queue_.submit(1, &submitInfo, fences_[index]);

	auto presentInfo = vk::PresentInfoKHR();
	presentInfo.setPImageIndices(&index)
		.setSwapchainCount(1)
		.setPSwapchains(&swapChain_)
		.setWaitSemaphoreCount(1)
		.setPWaitSemaphores(&drawComplete_[index]);

	queue_.presentKHR(presentInfo);
}

uint32_t VulkanDemo::chooseHeapFromFlags(
	vk::PhysicalDevice device,
	const vk::MemoryRequirements& memoryRequirements,
	vk::MemoryPropertyFlags requiredFlags,
	vk::MemoryPropertyFlags preferredFlags)

{
	uint32_t selectedType = ~0u;

	auto deviceMemoryProperties = device.getMemoryProperties();

	//find preferred
	for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
	{
		if (memoryRequirements.memoryTypeBits & (1 << i))
		{
			const vk::MemoryType& memoryType = deviceMemoryProperties.memoryTypes[i];

			if ((memoryType.propertyFlags & preferredFlags) == preferredFlags)
			{
				selectedType = i;
				break;
			}
		}
	}

	//if no preferred memory type, iterates through again, find hard require
	if (selectedType == ~0u)
	{
		for (uint32_t i = 0; i < VK_MAX_MEMORY_TYPES; ++i)
		{
			if (memoryRequirements.memoryTypeBits & (1 << i))
			{
				const vk::MemoryType& memoryType = deviceMemoryProperties.memoryTypes[i];

				if ((memoryType.propertyFlags & requiredFlags) == requiredFlags)
				{
					selectedType = i;
					break;
				}
			}
		}
	}
	return selectedType;
}


VulkanDemo::~VulkanDemo()
{
	
}

vk::SurfaceKHR VulkanDemo::createVulkanSurface(const vk::Instance& instance, SDL_Window* window)
{
	SDL_SysWMinfo windowInfo;
	SDL_VERSION(&windowInfo.version);
	if (!SDL_GetWindowWMInfo(window, &windowInfo)) {
		throw std::system_error(std::error_code(), "SDK window manager info is not available.");
	}

	switch (windowInfo.subsystem) {

#if defined(SDL_VIDEO_DRIVER_ANDROID) && defined(VK_USE_PLATFORM_ANDROID_KHR)
	case SDL_SYSWM_ANDROID: {
		vk::AndroidSurfaceCreateInfoKHR surfaceInfo = vk::AndroidSurfaceCreateInfoKHR()
			.setWindow(windowInfo.info.android.window);
		return instance.createAndroidSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_MIR) && defined(VK_USE_PLATFORM_MIR_KHR)
	case SDL_SYSWM_MIR: {
		vk::MirSurfaceCreateInfoKHR surfaceInfo = vk::MirSurfaceCreateInfoKHR()
			.setConnection(windowInfo.info.mir.connection)
			.setMirSurface(windowInfo.info.mir.surface);
		return instance.createMirSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_WAYLAND) && defined(VK_USE_PLATFORM_WAYLAND_KHR)
	case SDL_SYSWM_WAYLAND: {
		vk::WaylandSurfaceCreateInfoKHR surfaceInfo = vk::WaylandSurfaceCreateInfoKHR()
			.setDisplay(windowInfo.info.wl.display)
			.setSurface(windowInfo.info.wl.surface);
		return instance.createWaylandSurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_WINDOWS) && defined(VK_USE_PLATFORM_WIN32_KHR)
	case SDL_SYSWM_WINDOWS: {
		vk::Win32SurfaceCreateInfoKHR surfaceInfo = vk::Win32SurfaceCreateInfoKHR()
			.setHinstance(GetModuleHandle(NULL))
			.setHwnd(windowInfo.info.win.window);
		return instance.createWin32SurfaceKHR(surfaceInfo);
	}
#endif

#if defined(SDL_VIDEO_DRIVER_X11) && defined(VK_USE_PLATFORM_XLIB_KHR)
	case SDL_SYSWM_X11: {
		vk::XlibSurfaceCreateInfoKHR surfaceInfo = vk::XlibSurfaceCreateInfoKHR()
			.setDpy(windowInfo.info.x11.display)
			.setWindow(windowInfo.info.x11.window);
		return instance.createXlibSurfaceKHR(surfaceInfo);
	}
#endif

	default:
		throw std::system_error(std::error_code(), "Unsupported window manager is in use.");
	}
}

std::vector<const char*> VulkanDemo::getAvailableWSIExtensions()
{
	std::vector<const char*> extensions;
	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
	extensions.push_back(VK_KHR_ANDROID_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_MIR_KHR)
	extensions.push_back(VK_KHR_MIR_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_WAYLAND_KHR)
	extensions.push_back(VK_KHR_WAYLAND_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_WIN32_KHR)
	extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);
#endif
#if defined(VK_USE_PLATFORM_XLIB_KHR)
	extensions.push_back(VK_KHR_XLIB_SURFACE_EXTENSION_NAME);
#endif

	return extensions;
}

char* VulkanDemo::readSpv(const char *filename, size_t *psize)
{
	FILE *fp = fopen(filename, "rb");
	if (!fp)
	{
		return nullptr;
	}

	fseek(fp, 0L, SEEK_END);
	long int size = ftell(fp);

	fseek(fp, 0L, SEEK_SET);

	void *shader_code = malloc(size);
	size_t retval = fread(shader_code, size, 1, fp);
	assert(retval == 1);

	*psize = size;

	fclose(fp);

	return (char *)shader_code;
}