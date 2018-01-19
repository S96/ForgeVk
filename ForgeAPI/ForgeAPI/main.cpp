#define GLFW_INCLUDE_VULKAN
#include <glfw/glfw3.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <chrono>
#include <fstream>
#include <algorithm>
#include <vector>
#include <array>
#include <set>
#include <iostream>
#include <stdexcept>

// debug extension functions
#ifndef NDEBUG
VkResult CreateDebugReportCallbackEXT(VkInstance instance, const VkDebugReportCallbackCreateInfoEXT* p_create_info, const VkAllocationCallbacks* p_allocator, VkDebugReportCallbackEXT* p_callback)
{
	auto function = (PFN_vkCreateDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugReportCallbackEXT");
	if (function != nullptr)
	{
		return function(instance, p_create_info, p_allocator, p_callback);
	}
	else
	{
		return VK_ERROR_EXTENSION_NOT_PRESENT;
	}
}

void DestroyDebugReportCallbackEXT(VkInstance instance, VkDebugReportCallbackEXT callback, const VkAllocationCallbacks* pAllocator)
{
	auto func = (PFN_vkDestroyDebugReportCallbackEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugReportCallbackEXT");
	if (func != nullptr)
	{
		func(instance, callback, pAllocator);
	}
}
#endif // !NDEBUG

const std::vector<const char*> VALIDATION_LAYERS = { "VK_LAYER_LUNARG_standard_validation" };
const std::vector<const char*> DEVICE_EXTENSIONS = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

struct Vertex
{
	glm::vec3 pos;
	glm::vec3 color;
	glm::vec2 uv;

	static VkVertexInputBindingDescription GetBindingDescription()
	{
		VkVertexInputBindingDescription bind_description = {};

		bind_description.binding = 0;
		bind_description.stride = sizeof(Vertex);
		bind_description.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

		return bind_description;
	}

	static std::array<VkVertexInputAttributeDescription, 3> GetAttributeDescriptions()
	{
		std::array<VkVertexInputAttributeDescription, 3> attribute_descriptions = {};

		attribute_descriptions[0].binding = 0;
		attribute_descriptions[0].location = 0;
		attribute_descriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[0].offset = offsetof(Vertex, pos);

		attribute_descriptions[1].binding = 0;
		attribute_descriptions[1].location = 1;
		attribute_descriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
		attribute_descriptions[1].offset = offsetof(Vertex, color);

		attribute_descriptions[2].binding = 0;
		attribute_descriptions[2].location = 2;
		attribute_descriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
		attribute_descriptions[2].offset = offsetof(Vertex, uv);

		return attribute_descriptions;
	}
};

struct UniformBufferObject
{
	glm::mat4 model;
	glm::mat4 view;
	glm::mat4 proj;
};

struct QueueFamilies
{
	int _graphics_family = -1;
	int _present_family = -1;

	bool QueuesAquired()
	{
		return _graphics_family >= 0 && _present_family >= 0;
	}
};

struct SwapChainSupport
{
	VkSurfaceCapabilitiesKHR _capabilities;
	std::vector<VkSurfaceFormatKHR> _formats;
	std::vector<VkPresentModeKHR> _present_modes;
};

class HelloTriangleApplication
{
public:
	HelloTriangleApplication()
	{
		_window_height = 600;
		_window_width = 800;
		_window_name = "ForgeVK";
	}

	void Run()
	{
		InitializeWindow();
		InitializeVulkan();
		MainLoop();
		EndProgram();
	}

private:
	void InitializeWindow()
	{
		glfwInit();
		glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
		_p_glfw_window = glfwCreateWindow(_window_width, _window_height, _window_name, nullptr, nullptr);
		glfwSetWindowUserPointer(_p_glfw_window, this);
		glfwSetWindowSizeCallback(_p_glfw_window, HelloTriangleApplication::OnWindowResize);
	}

	void InitializeVulkan()
	{
		CreateVkInstance();

		// create debug callback
#ifndef NDEBUG
		VkDebugReportCallbackCreateInfoEXT create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DEBUG_REPORT_CALLBACK_CREATE_INFO_EXT;
		create_info.flags = VK_DEBUG_REPORT_ERROR_BIT_EXT | VK_DEBUG_REPORT_WARNING_BIT_EXT;
		create_info.pfnCallback = DebugCallback;

		if (CreateDebugReportCallbackEXT(_vk_instance, &create_info, nullptr, &_vk_callback) != VK_SUCCESS)
		{
			throw std::runtime_error("Debug Callback Initialization Failed!");
		}
#endif

		CreateSurface();
		SelectPhysicalDevice();
		CreateLogicalDevice(_available_queue_families);
		CreateSwapChain(_available_queue_families);
		CreateImageViews();
		CreateRenderPass();
		CreateDescriptorSetLayout();
		CreateGraphicsPipeline();
		CreateCommandPool(_available_queue_families);
		CreateDepthResources();
		CreateFrameBuffers();
		CreateTextureImage();
		CreateTextureImageView();
		CreateTextureSampler();
		LoadModel();
		CreateVertexBuffer();
		CreateIndexBuffer();
		CreateUniformBuffer();
		CreateDescriptorPool();
		CreateDescriptorSet();
		CreateCommandBuffers();
		CreateSemaphores();
	}

	void MainLoop()
	{
		while (!glfwWindowShouldClose(_p_glfw_window))
		{
			glfwPollEvents();
			UpdateUniformBuffer();
			Draw();
		}

		vkDeviceWaitIdle(_vk_logical_device);
	}

	void EndProgram()
	{
		ReleaseSwapchain();
		vkDestroySampler(_vk_logical_device, _vk_texture_sampler, nullptr);
		vkDestroyImageView(_vk_logical_device, _vk_texture_image_view, nullptr);
		vkDestroyImage(_vk_logical_device, _vk_texture_image, nullptr);
		vkFreeMemory(_vk_logical_device, _vk_texture_image_memory, nullptr);
		vkDestroyDescriptorPool(_vk_logical_device, _vk_descriptor_pool, nullptr);
		vkDestroyDescriptorSetLayout(_vk_logical_device, _vk_descriptor_set_layout, nullptr);
		vkDestroyBuffer(_vk_logical_device, _vk_uniform_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, _vk_uniform_buffer_memory, nullptr);
		vkDestroyBuffer(_vk_logical_device, _vk_index_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, _vk_index_buffer_memory, nullptr);
		vkDestroyBuffer(_vk_logical_device, _vk_vertex_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, _vk_vertex_buffer_memory, nullptr);
		vkDestroySwapchainKHR(_vk_logical_device, _vk_swapchain, nullptr);
		vkDestroySemaphore(_vk_logical_device, _vk_frame_complete_semaphore, nullptr);
		vkDestroySemaphore(_vk_logical_device, _vk_image_available_semaphore, nullptr);
		vkDestroyCommandPool(_vk_logical_device, _vk_command_pool, nullptr);
		vkDestroyDevice(_vk_logical_device, nullptr);

#ifndef NDEBUG
		DestroyDebugReportCallbackEXT(_vk_instance, _vk_callback, nullptr);
#endif // !NDEBUG

		vkDestroySurfaceKHR(_vk_instance, _vk_surface, nullptr);
		vkDestroyInstance(_vk_instance, nullptr);
		glfwDestroyWindow(_p_glfw_window);
		glfwTerminate();
	}

	static void OnWindowResize(GLFWwindow* window, int width, int height)
	{
		if (width != 0 && height != 0)
		{
			HelloTriangleApplication* application = reinterpret_cast<HelloTriangleApplication*>(glfwGetWindowUserPointer(window));
			application->RecreateSwapChain();
		}
	}

	void CreateVkInstance()
	{
#ifndef NDEBUG
		if(!CheckValidationLayers())
		{
			throw std::runtime_error("Validation Layers Unnavailable!");
		}
#endif

		// info to init the vulkan instance
		VkApplicationInfo app_info = {};
		app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
		app_info.pApplicationName = "ForgeVK Test";
		app_info.applicationVersion = VK_MAKE_VERSION(0, 0, 0);
		app_info.pEngineName = "ForgeVK";
		app_info.engineVersion = VK_MAKE_VERSION(0, 0, 0);
		app_info.apiVersion = VK_API_VERSION_1_0;
		
		std::vector<const char*> required_extensions = RequiredExtensions();

		VkInstanceCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
		create_info.pApplicationInfo = &app_info;
		create_info.enabledExtensionCount = static_cast<uint32_t>(required_extensions.size());
		create_info.ppEnabledExtensionNames = required_extensions.data();

		// add validation layers
#ifndef NDEBUG
		create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
		create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
#else
		create_info.enabledLayerCount = 0;
#endif
		
		// init vulknan instance
		if (vkCreateInstance(&create_info, nullptr, &_vk_instance) != VK_SUCCESS)
		{
			throw std::runtime_error("Vk Instance Creation Failed!");
		}
	}
	
	void CreateSurface()
	{
		if (glfwCreateWindowSurface(_vk_instance, _p_glfw_window, nullptr, &_vk_surface) != VK_SUCCESS)
		{
			throw std::runtime_error("Window Surface Creation Failed!");
		}
	}

	void SelectPhysicalDevice()
	{
		// find all devices
		uint32_t device_num = 0;
		vkEnumeratePhysicalDevices(_vk_instance, &device_num, nullptr);
		if (device_num == 0)
		{
			throw std::runtime_error("No Vulkan GPUs Available!");
		}
		std::vector<VkPhysicalDevice> devices(device_num);
		vkEnumeratePhysicalDevices(_vk_instance, &device_num, devices.data());

		std::vector<VkPhysicalDevice> sufficient_devices;

		// test devices
		for (const auto& device : devices)
		{
			if (TestPhysicalDevice(device))
			{
				sufficient_devices.push_back(device);
			}
		}

		if (sufficient_devices.size() > 0)
		{
			_vk_physical_device = sufficient_devices[0];
		}
		else
		{
			throw std::runtime_error("Failed To Find Suitable Device!");
		}
	}

	void CreateLogicalDevice(QueueFamilies& queue_family_data)
	{
		queue_family_data = CheckQueueFamilies(_vk_physical_device);

		// generate queue create structs
		std::vector<VkDeviceQueueCreateInfo> queue_create_infos;
		std::set<int> unique_queue_families = { queue_family_data._graphics_family, queue_family_data._present_family };

		float queue_priority = 1.0f;

		for (int queue_index : unique_queue_families)
		{
			VkDeviceQueueCreateInfo queue_create_info = {};
			queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
			queue_create_info.queueFamilyIndex = queue_index;
			queue_create_info.queueCount = 1;
			queue_create_info.pQueuePriorities = &queue_priority;
			
			queue_create_infos.push_back(queue_create_info);
		}
		
		// device features
		VkPhysicalDeviceFeatures device_features = {};
		device_features.samplerAnisotropy = VK_TRUE;

		// logical device info
		VkDeviceCreateInfo device_create_info = {};
		device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
		device_create_info.pQueueCreateInfos = queue_create_infos.data();
		device_create_info.queueCreateInfoCount = static_cast<uint32_t>(queue_create_infos.size());
		device_create_info.pEnabledFeatures = &device_features;
		device_create_info.enabledExtensionCount = static_cast<uint32_t>(DEVICE_EXTENSIONS.size());
		device_create_info.ppEnabledExtensionNames = DEVICE_EXTENSIONS.data();
#ifndef DEBUG
		device_create_info.enabledLayerCount = static_cast<uint32_t>(VALIDATION_LAYERS.size());
		device_create_info.ppEnabledLayerNames = VALIDATION_LAYERS.data();
#else
		device_create_info.enabledLayerCount = 0;
#endif // !DEBUG

		if (vkCreateDevice(_vk_physical_device, &device_create_info, nullptr, &_vk_logical_device) != VK_SUCCESS)
		{
			throw std::runtime_error("Logical Device Creation Failed!");
		}

		// get queue handles - no new creation occurs here
		vkGetDeviceQueue(_vk_logical_device, queue_family_data._graphics_family, 0, &_vk_graphics_queue);
		vkGetDeviceQueue(_vk_logical_device, queue_family_data._present_family, 0, &_vk_present_queue);
	}
	
	void ReleaseSwapchain()
	{
		vkDestroyImageView(_vk_logical_device, _vk_depth_image_view, nullptr);
		vkDestroyImage(_vk_logical_device, _vk_depth_image, nullptr);
		vkFreeMemory(_vk_logical_device, _vk_depth_image_memory, nullptr);

		for (auto frame_buffer : _vk_swapchain_frame_buffers)
		{
			vkDestroyFramebuffer(_vk_logical_device, frame_buffer, nullptr);
		}
		
		vkFreeCommandBuffers(_vk_logical_device, _vk_command_pool, static_cast<uint32_t>(_vk_command_buffers.size()), _vk_command_buffers.data());
		
		vkDestroyPipeline(_vk_logical_device, _vk_pipeline, nullptr);
		vkDestroyPipelineLayout(_vk_logical_device, _vk_pipeline_layout, nullptr);
		
		vkDestroyRenderPass(_vk_logical_device, _vk_render_pass, nullptr);
		
		for (auto view : _vk_swapchain_image_views)
		{
			vkDestroyImageView(_vk_logical_device, view, nullptr);
		}
	}

	void RecreateSwapChain()
	{
		VkSwapchainKHR old = _vk_swapchain;
		CreateSwapChain(_available_queue_families, old);
		vkDeviceWaitIdle(_vk_logical_device);
		ReleaseSwapchain();
		vkDestroySwapchainKHR(_vk_logical_device, old, nullptr);
		CreateImageViews();
		CreateRenderPass();
		CreateGraphicsPipeline();
		CreateFrameBuffers();
		CreateCommandBuffers();
	}

	void CreateSwapChain(const QueueFamilies& queue_families, VkSwapchainKHR old = VK_NULL_HANDLE)
	{
		// get swap chain details
		SwapChainSupport swap_details = CheckSwapChainSupport(_vk_physical_device);
		VkSurfaceFormatKHR surface_format = ChooseSwapSurfaceFormat(swap_details._formats);
		VkPresentModeKHR present_mode = ChooseSwapPresentMode(swap_details._present_modes);
		VkExtent2D extent = ChooseSwapExtent(swap_details._capabilities);

		// save for later swap chain adjustments
		_vk_swapchain_format = surface_format.format;
		_vk_swapchain_extent = extent;

		// check queue length
		uint32_t image_count = swap_details._capabilities.minImageCount + 1;
		if (swap_details._capabilities.maxImageCount > 0 && image_count > swap_details._capabilities.maxImageCount)
		{
			image_count = swap_details._capabilities.maxImageCount;
		}

		// create swap chain
		VkSwapchainCreateInfoKHR create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
		create_info.surface = _vk_surface;
		create_info.minImageCount = image_count;
		create_info.imageFormat = surface_format.format;
		create_info.imageColorSpace = surface_format.colorSpace;
		create_info.imageExtent = extent;
		create_info.imageArrayLayers = 1;
		// bit field for flags indicating usage - includes depth buffer etc - target for future custom options
		create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

		// multiple queues need access, use concurrent mode
		if (queue_families._graphics_family != queue_families._present_family)
		{
			create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
			create_info.queueFamilyIndexCount = 2;
			uint32_t indices[] = { static_cast<uint32_t>(queue_families._graphics_family), static_cast<uint32_t>(queue_families._present_family) };
			create_info.pQueueFamilyIndices = indices;
		}
		else
		{
			create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
		}

		// more fields available for optional settings
		create_info.preTransform = swap_details._capabilities.currentTransform;
		create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;

		create_info.presentMode = present_mode;
		create_info.clipped = VK_TRUE;
		
		create_info.oldSwapchain = old;

		if (vkCreateSwapchainKHR(_vk_logical_device, &create_info, nullptr, &_vk_swapchain) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Swapchain!");
		}

		// get swapchain images
		vkGetSwapchainImagesKHR(_vk_logical_device, _vk_swapchain, &image_count, nullptr);
		_vk_swapchain_images.resize(image_count);
		vkGetSwapchainImagesKHR(_vk_logical_device, _vk_swapchain, &image_count, _vk_swapchain_images.data());
	}

	void CreateImageViews()
	{
		_vk_swapchain_image_views.resize(_vk_swapchain_images.size());

		for (uint32_t i = 0; i < _vk_swapchain_image_views.size(); ++i)
		{
			_vk_swapchain_image_views[i] = CreateImageView(_vk_swapchain_images[i], _vk_swapchain_format, VK_IMAGE_ASPECT_COLOR_BIT);
		}
	}

	void CreateRenderPass()
	{
		VkAttachmentDescription depth_attachment = {};
		depth_attachment.format = FindDepthFormat();
		depth_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		depth_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		depth_attachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		depth_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		depth_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		depth_attachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		VkAttachmentReference depth_attachment_reference = {};
		depth_attachment_reference.attachment = 1;
		depth_attachment_reference.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

		// render pass attachment
		VkAttachmentDescription color_attachment = {};
		color_attachment.format = _vk_swapchain_format;
		color_attachment.samples = VK_SAMPLE_COUNT_1_BIT;
		color_attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
		color_attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
		color_attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
		color_attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
		color_attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		color_attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

		// subpasses
		VkAttachmentReference attachment_reference = {};
		attachment_reference.attachment = 0;
		attachment_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

		VkSubpassDescription subpass = {};
		subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
		subpass.colorAttachmentCount = 1;
		subpass.pColorAttachments = &attachment_reference;
		subpass.pDepthStencilAttachment = &depth_attachment_reference;

		VkSubpassDependency subpass_dependency = {};
		subpass_dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
		subpass_dependency.dstSubpass = 0;
		subpass_dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dependency.srcAccessMask = 0;
		subpass_dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
		subpass_dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

		std::array<VkAttachmentDescription, 2> attachments = { color_attachment, depth_attachment };

		// create render pass
		VkRenderPassCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
		create_info.attachmentCount = attachments.size();
		create_info.pAttachments = attachments.data();
		create_info.subpassCount = 1;
		create_info.pSubpasses = &subpass;
		create_info.dependencyCount = 1;
		create_info.pDependencies = &subpass_dependency;

		if (vkCreateRenderPass(_vk_logical_device, &create_info, nullptr, &_vk_render_pass) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Render Pass!");
		}
	}

	void CreateDescriptorSetLayout()
	{
		VkDescriptorSetLayoutBinding ubo_layout_binding = {};
		ubo_layout_binding.binding = 0;
		ubo_layout_binding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		ubo_layout_binding.descriptorCount = 1;
		ubo_layout_binding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

		VkDescriptorSetLayoutBinding combined_image_sampler_binding = {};
		combined_image_sampler_binding.binding = 1;
		combined_image_sampler_binding.descriptorCount = 1;
		combined_image_sampler_binding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		combined_image_sampler_binding.pImmutableSamplers = nullptr;
		combined_image_sampler_binding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

		std::array<VkDescriptorSetLayoutBinding, 2> bindings = { ubo_layout_binding, combined_image_sampler_binding };

		VkDescriptorSetLayoutCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		create_info.bindingCount = static_cast<uint32_t>(bindings.size());
		create_info.pBindings = bindings.data();

		if (vkCreateDescriptorSetLayout(_vk_logical_device, &create_info, nullptr, &_vk_descriptor_set_layout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Descriptor Set Layout!");
		}
	}

	void CreateGraphicsPipeline()
	{
		// shader blob acquisition
		auto vert_blob = ReadFile("Shaders/vert.spv");
		auto frag_blob = ReadFile("Shaders/frag.spv");

		VkShaderModule vertex_shader_module = CreateShaderModule(vert_blob);
		VkShaderModule fragment_shader_module = CreateShaderModule(frag_blob);

		// create vertex shader info
		VkPipelineShaderStageCreateInfo vertex_stage_create_info = {};
		vertex_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		vertex_stage_create_info.stage = VK_SHADER_STAGE_VERTEX_BIT;
		vertex_stage_create_info.module = vertex_shader_module;
		vertex_stage_create_info.pName = "main";
		// optional ability to add shader constants to stage
		//vertex_stage_create_info.pSpecializationInfo = nullptr;

		// fragment shader info
		VkPipelineShaderStageCreateInfo fragment_stage_create_info = {};
		fragment_stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		fragment_stage_create_info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
		fragment_stage_create_info.module = fragment_shader_module;
		fragment_stage_create_info.pName = "main";

		VkPipelineShaderStageCreateInfo shader_stages[] = { vertex_stage_create_info, fragment_stage_create_info };

		// vertex binding data
		auto vertex_binding_description = Vertex::GetBindingDescription();
		auto vertex_attribute_descriptions = Vertex::GetAttributeDescriptions();

		// vertex stage
		VkPipelineVertexInputStateCreateInfo vertex_input_create_info = {};
		vertex_input_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
		vertex_input_create_info.vertexBindingDescriptionCount = 1;
		vertex_input_create_info.pVertexBindingDescriptions = &vertex_binding_description;
		vertex_input_create_info.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_attribute_descriptions.size());
		vertex_input_create_info.pVertexAttributeDescriptions = vertex_attribute_descriptions.data();

		// Input assembly
		VkPipelineInputAssemblyStateCreateInfo input_assembly_create_info = {};
		input_assembly_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
		input_assembly_create_info.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
		input_assembly_create_info.primitiveRestartEnable = VK_FALSE;
		
		// pipeline viewport with vieport and scissor rectangle
		VkViewport viewport = {};
		viewport.x = 0.0f;
		viewport.y = 0.0f;
		viewport.width = static_cast<float>(_vk_swapchain_extent.width);
		viewport.height = static_cast<float>(_vk_swapchain_extent.height);
		viewport.minDepth = 0.0f;
		viewport.maxDepth = 1.0f;

		VkRect2D scissor_rect = {};
		scissor_rect.offset = { 0, 0 };
		scissor_rect.extent = _vk_swapchain_extent;

		VkPipelineViewportStateCreateInfo viewport_state_create_info = {};
		viewport_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
		viewport_state_create_info.viewportCount = 1;
		viewport_state_create_info.pViewports = &viewport;
		viewport_state_create_info.scissorCount = 1;
		viewport_state_create_info.pScissors = &scissor_rect;

		// rasterizer
		VkPipelineRasterizationStateCreateInfo rasterizer_create_info = {};
		rasterizer_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
		rasterizer_create_info.depthClampEnable = VK_FALSE;
		rasterizer_create_info.rasterizerDiscardEnable = VK_FALSE;
		rasterizer_create_info.polygonMode = VK_POLYGON_MODE_FILL;
		rasterizer_create_info.lineWidth = 1.0f;
		rasterizer_create_info.cullMode = VK_CULL_MODE_BACK_BIT;
		rasterizer_create_info.frontFace = VK_FRONT_FACE_CLOCKWISE;
		rasterizer_create_info.depthBiasEnable = VK_FALSE;

		// multisampling
		VkPipelineMultisampleStateCreateInfo multisampling = {};
		multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
		multisampling.sampleShadingEnable = VK_FALSE;
		multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
		
		// depth stencil test
		
		// color blending
		VkPipelineColorBlendAttachmentState color_blend_attachment = {};
		color_blend_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_R_BIT;
		color_blend_attachment.blendEnable = VK_FALSE;

		VkPipelineColorBlendStateCreateInfo color_blend_create_info = {};
		color_blend_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
		color_blend_create_info.logicOpEnable = VK_FALSE;
		color_blend_create_info.attachmentCount = 1;
		color_blend_create_info.pAttachments = &color_blend_attachment;

		VkPipelineDepthStencilStateCreateInfo depth_stencil_create_info = {};
		depth_stencil_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
		depth_stencil_create_info.depthTestEnable = VK_TRUE;
		depth_stencil_create_info.depthWriteEnable = VK_TRUE;
		depth_stencil_create_info.depthCompareOp = VK_COMPARE_OP_LESS;
		depth_stencil_create_info.depthBoundsTestEnable = VK_FALSE;
		depth_stencil_create_info.stencilTestEnable = VK_FALSE;
		
		// dynamic state
		VkDynamicState dynamic_states[] = { VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR, VK_DYNAMIC_STATE_LINE_WIDTH };
		
		VkPipelineDynamicStateCreateInfo dynamic_state_create_info = {};
		dynamic_state_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
		dynamic_state_create_info.dynamicStateCount = 2;
		dynamic_state_create_info.pDynamicStates = dynamic_states;

		// pipeline layout
		VkPipelineLayoutCreateInfo pipeline_layout_create_info = {};
		pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		pipeline_layout_create_info.setLayoutCount = 1;
		pipeline_layout_create_info.pSetLayouts = &_vk_descriptor_set_layout;
		
		if (vkCreatePipelineLayout(_vk_logical_device, &pipeline_layout_create_info, nullptr, &_vk_pipeline_layout) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Pipeline Layout!");
		}

		// create graphics pipeline
		VkGraphicsPipelineCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
		create_info.stageCount = 2;
		create_info.pStages = shader_stages;
		create_info.pVertexInputState = &vertex_input_create_info;
		create_info.pInputAssemblyState = &input_assembly_create_info;
		create_info.pViewportState = &viewport_state_create_info;
		create_info.pRasterizationState = &rasterizer_create_info;
		create_info.pMultisampleState = &multisampling;
		create_info.pDepthStencilState = &depth_stencil_create_info;
		create_info.pColorBlendState = &color_blend_create_info;
		create_info.pDynamicState = nullptr;
		create_info.layout = _vk_pipeline_layout;
		create_info.renderPass = _vk_render_pass;
		create_info.subpass = 0;

		// pipeline cache - data saved for fast pipeline creation later and from file

		if (vkCreateGraphicsPipelines(_vk_logical_device, VK_NULL_HANDLE, 1, &create_info, nullptr, &_vk_pipeline) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Graphics Pipeline!");
		}

		// clean up VK resources
		vkDestroyShaderModule(_vk_logical_device, vertex_shader_module, nullptr);
		vkDestroyShaderModule(_vk_logical_device, fragment_shader_module, nullptr);
	}

	VkFormat FindDepthFormat()
	{
		return FindSupportedFormat({ VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT }, VK_IMAGE_TILING_OPTIMAL, VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT);
	}

	bool HasStencilComponent(VkFormat format)
	{
		return format == VK_FORMAT_D32_SFLOAT_S8_UINT || format == VK_FORMAT_D24_UNORM_S8_UINT;
	}

	void CreateDepthResources()
	{
		VkFormat depth_format = FindDepthFormat();
		CreateImage(_vk_swapchain_extent.width, _vk_swapchain_extent.height, depth_format, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vk_depth_image, _vk_depth_image_memory);
		_vk_depth_image_view = CreateImageView(_vk_depth_image, depth_format, VK_IMAGE_ASPECT_DEPTH_BIT);
		TransitionImageLayout(_vk_depth_image, depth_format, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
	}

	void CreateFrameBuffers()
	{
		_vk_swapchain_frame_buffers.resize(_vk_swapchain_image_views.size());

		std::array<VkImageView, 2> attachments = { _vk_swapchain_image_views[0], _vk_depth_image_view };

		for (size_t i = 0; i < _vk_swapchain_image_views.size(); ++i)
		{
			attachments[0] = _vk_swapchain_image_views[i];

			VkFramebufferCreateInfo create_info = {};
			create_info.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
			create_info.renderPass = _vk_render_pass;
			create_info.attachmentCount = static_cast<uint32_t>(attachments.size());
			create_info.pAttachments = attachments.data();
			create_info.width = _vk_swapchain_extent.width;
			create_info.height = _vk_swapchain_extent.height;
			create_info.layers = 1;

			if (vkCreateFramebuffer(_vk_logical_device, &create_info, nullptr, &_vk_swapchain_frame_buffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed To Create Framebuffer!");
			}
		}
	}

	void CreateCommandPool(QueueFamilies queue_families)
	{
		VkCommandPoolCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		create_info.queueFamilyIndex = queue_families._graphics_family;

		if (vkCreateCommandPool(_vk_logical_device, &create_info, nullptr, &_vk_command_pool) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Command Pool!");
		}
	}
	
	void CreateTextureImage()
	{
		int texture_width, texture_height, texture_channels;

		stbi_uc* pixels = stbi_load("Textures/body.tga", &texture_width, &texture_height, &texture_channels, STBI_rgb_alpha);

		VkDeviceSize image_size = texture_width * texture_height * 4;

		if (!pixels)
		{
			throw std::runtime_error("Failed To Load Image File!");
		}

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		CreateBuffer(image_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, staging_buffer, staging_buffer_memory);

		void* data;
		vkMapMemory(_vk_logical_device, staging_buffer_memory, 0, image_size, 0, &data);
		memcpy(data, pixels, static_cast<size_t>(image_size));
		vkUnmapMemory(_vk_logical_device, staging_buffer_memory);

		stbi_image_free(pixels);

		CreateImage(texture_width, texture_height, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vk_texture_image, _vk_texture_image_memory);

		TransitionImageLayout(_vk_texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);

		CopyBufferToImage(staging_buffer, _vk_texture_image, static_cast<uint32_t>(texture_width), static_cast<uint32_t>(texture_height));

		TransitionImageLayout(_vk_texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

		vkDestroyBuffer(_vk_logical_device, staging_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, staging_buffer_memory, nullptr);
	}

	void CreateTextureImageView()
	{
		_vk_texture_image_view = CreateImageView(_vk_texture_image, VK_FORMAT_R8G8B8A8_UNORM, VK_IMAGE_ASPECT_COLOR_BIT);
	}

	void CreateTextureSampler()
	{
		VkSamplerCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
		create_info.magFilter = VK_FILTER_LINEAR;
		create_info.minFilter = VK_FILTER_LINEAR;
		create_info.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		create_info.addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		create_info.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
		create_info.anisotropyEnable = VK_TRUE;
		create_info.maxAnisotropy = 16;
		create_info.borderColor = VK_BORDER_COLOR_INT_OPAQUE_BLACK;
		create_info.unnormalizedCoordinates = VK_FALSE;
		create_info.compareEnable = VK_FALSE;
		create_info.compareOp = VK_COMPARE_OP_ALWAYS;
		create_info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
		create_info.mipLodBias = 0.0f;
		create_info.minLod = 0.0f;
		create_info.maxLod = 0.0f;

		if (vkCreateSampler(_vk_logical_device, &create_info, nullptr, &_vk_texture_sampler) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Texture Sampler!");
		}
	}

	void CreateVertexBuffer()
	{
		VkDeviceSize buffer_size = sizeof(_vertices[0]) * _vertices.size();

		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;

		CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);
		
		void* data;
		vkMapMemory(_vk_logical_device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, _vertices.data(), static_cast<size_t>(buffer_size));
		vkUnmapMemory(_vk_logical_device, staging_buffer_memory);

		CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vk_vertex_buffer, _vk_vertex_buffer_memory);

		CopyBuffer(staging_buffer, _vk_vertex_buffer, buffer_size);

		vkDestroyBuffer(_vk_logical_device, staging_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, staging_buffer_memory, nullptr);
	}

	void CreateIndexBuffer()
	{
		VkDeviceSize buffer_size = sizeof(_indices[0]) * _indices.size();
		VkBuffer staging_buffer;
		VkDeviceMemory staging_buffer_memory;
		CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, staging_buffer, staging_buffer_memory);

		void* data;
		vkMapMemory(_vk_logical_device, staging_buffer_memory, 0, buffer_size, 0, &data);
		memcpy(data, _indices.data(), static_cast<size_t>(buffer_size));
		vkUnmapMemory(_vk_logical_device, staging_buffer_memory);

		CreateBuffer(buffer_size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, _vk_index_buffer, _vk_index_buffer_memory);

		CopyBuffer(staging_buffer, _vk_index_buffer, buffer_size);

		vkDestroyBuffer(_vk_logical_device, staging_buffer, nullptr);
		vkFreeMemory(_vk_logical_device, staging_buffer_memory, nullptr);
	}

	void CreateUniformBuffer()
	{
		VkDeviceSize buffer_size = sizeof(UniformBufferObject);
		CreateBuffer(buffer_size, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, _vk_uniform_buffer, _vk_uniform_buffer_memory);

	}

	void CreateDescriptorPool()
	{
		std::array<VkDescriptorPoolSize, 2> pool_sizes = {};

		pool_sizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		pool_sizes[0].descriptorCount = 1;

		pool_sizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		pool_sizes[1].descriptorCount = 1;
		
		VkDescriptorPoolCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		create_info.poolSizeCount = static_cast<uint32_t>(pool_sizes.size());
		create_info.pPoolSizes = pool_sizes.data();
		create_info.maxSets = 1;

		if (vkCreateDescriptorPool(_vk_logical_device, &create_info, nullptr, &_vk_descriptor_pool) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Descriptor Pool!");
		}
	}

	void CreateDescriptorSet()
	{
		VkDescriptorSetLayout layouts[] = { _vk_descriptor_set_layout };
		VkDescriptorSetAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		alloc_info.descriptorPool = _vk_descriptor_pool;
		alloc_info.descriptorSetCount = 1;
		alloc_info.pSetLayouts = layouts;

		if (vkAllocateDescriptorSets(_vk_logical_device, &alloc_info, &_vk_descriptor_set) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Allocate Descriptor Set!");
		}

		VkDescriptorBufferInfo buffer_info = {};
		buffer_info.buffer = _vk_uniform_buffer;
		buffer_info.offset = 0;
		buffer_info.range = sizeof(UniformBufferObject);

		VkDescriptorImageInfo image_info = {};
		image_info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
		image_info.imageView = _vk_texture_image_view;
		image_info.sampler = _vk_texture_sampler;

		std::array<VkWriteDescriptorSet, 2> write_descriptors = {};

		write_descriptors[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptors[0].dstSet = _vk_descriptor_set;
		write_descriptors[0].dstBinding = 0;
		write_descriptors[0].dstArrayElement = 0;
		write_descriptors[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		write_descriptors[0].descriptorCount = 1;
		write_descriptors[0].pBufferInfo = &buffer_info;

		write_descriptors[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write_descriptors[1].dstSet = _vk_descriptor_set;
		write_descriptors[1].dstBinding = 1;
		write_descriptors[1].dstArrayElement = 0;
		write_descriptors[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
		write_descriptors[1].descriptorCount = 1;
		write_descriptors[1].pImageInfo = &image_info;

		vkUpdateDescriptorSets(_vk_logical_device, static_cast<uint32_t>(write_descriptors.size()), write_descriptors.data(), 0, nullptr);
	}

	void CreateCommandBuffers()
	{
		_vk_command_buffers.resize(_vk_swapchain_frame_buffers.size());

		VkCommandBufferAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.commandPool = _vk_command_pool;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandBufferCount = static_cast<uint32_t>(_vk_command_buffers.size());

		if (vkAllocateCommandBuffers(_vk_logical_device, &alloc_info, _vk_command_buffers.data()) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Allocate Command Buffers!");
		}

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_SIMULTANEOUS_USE_BIT;

		VkRenderPassBeginInfo render_pass_begin_info = {};
		render_pass_begin_info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
		render_pass_begin_info.renderPass = _vk_render_pass;
		render_pass_begin_info.renderArea.offset = { 0,0 };
		render_pass_begin_info.renderArea.extent = _vk_swapchain_extent;

		std::array<VkClearValue, 2> clear_values = {};
		clear_values[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
		clear_values[1].depthStencil = { 1.0f, 0 };

		render_pass_begin_info.clearValueCount = static_cast<uint32_t>(clear_values.size());
		render_pass_begin_info.pClearValues = clear_values.data();
		
		VkBuffer vertex_buffers[] = { _vk_vertex_buffer };
		VkDeviceSize offsets[] = {0};

		for (size_t i = 0; i < _vk_command_buffers.size(); ++i)
		{
			vkBeginCommandBuffer(_vk_command_buffers[i], &begin_info);
			render_pass_begin_info.framebuffer = _vk_swapchain_frame_buffers[i];
			vkCmdBeginRenderPass(_vk_command_buffers[i], &render_pass_begin_info, VK_SUBPASS_CONTENTS_INLINE);
			vkCmdBindPipeline(_vk_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _vk_pipeline);
			vkCmdBindVertexBuffers(_vk_command_buffers[i], 0, 1, vertex_buffers, offsets);
			vkCmdBindIndexBuffer(_vk_command_buffers[i], _vk_index_buffer, 0, VK_INDEX_TYPE_UINT32);
			vkCmdBindDescriptorSets(_vk_command_buffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, _vk_pipeline_layout, 0, 1, &_vk_descriptor_set, 0, nullptr);
			vkCmdDrawIndexed(_vk_command_buffers[i], static_cast<uint32_t>(_indices.size()), 1, 0, 0, 0);
			vkCmdEndRenderPass(_vk_command_buffers[i]);
			if (vkEndCommandBuffer(_vk_command_buffers[i]) != VK_SUCCESS)
			{
				throw std::runtime_error("Failed To Record Command Buffer!");
			}
		}
	}

	void CreateSemaphores()
	{
		VkSemaphoreCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
		if (vkCreateSemaphore(_vk_logical_device, &create_info, nullptr, &_vk_image_available_semaphore) != VK_SUCCESS || 
			vkCreateSemaphore(_vk_logical_device, &create_info, nullptr, &_vk_frame_complete_semaphore) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Semaphores!");
		}
	}

	void UpdateUniformBuffer()
	{
		static auto start_time = std::chrono::high_resolution_clock::now();

		auto current_time = std::chrono::high_resolution_clock::now();
		float time = std::chrono::duration<float, std::chrono::seconds::period>(current_time - start_time).count();

		UniformBufferObject ubo = {};
		ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
		ubo.proj = glm::perspective(glm::radians(45.0f), _vk_swapchain_extent.width / static_cast<float>(_vk_swapchain_extent.height), 0.1f, 10.0f);
		ubo.proj[1][1] *= -1;

		void* data;
		vkMapMemory(_vk_logical_device, _vk_uniform_buffer_memory, 0, sizeof(ubo), 0,&data);
		memcpy(data, &ubo, sizeof(ubo));
		vkUnmapMemory(_vk_logical_device, _vk_uniform_buffer_memory);
	}

	void Draw()
	{
		// wait for previous frame
		vkDeviceWaitIdle(_vk_logical_device);
		
		uint32_t image_index;
		VkResult result = vkAcquireNextImageKHR(_vk_logical_device, _vk_swapchain, std::numeric_limits<uint64_t>::max(), _vk_image_available_semaphore, VK_NULL_HANDLE, &image_index);
		
		// break and update swapchain
		if (result == VK_ERROR_OUT_OF_DATE_KHR)
		{
			RecreateSwapChain();
			return;
		}
		else if (result != VK_SUCCESS && result != VK_SUBOPTIMAL_KHR)
		{
			throw std::runtime_error("Failed To Acquire Swapchain Image!");
		}

		VkSemaphore wait_semaphores[] = { _vk_image_available_semaphore };
		VkPipelineStageFlags wait_stages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
		VkSemaphore signal_semaphores[] = { _vk_frame_complete_semaphore };

		VkSubmitInfo submit_info= {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitSemaphores = wait_semaphores;
		submit_info.pWaitDstStageMask = wait_stages;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &_vk_command_buffers[image_index];
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = signal_semaphores;

		if (vkQueueSubmit(_vk_graphics_queue, 1, &submit_info, VK_NULL_HANDLE) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Submit Draw Command Buffer!");
		}

		VkSwapchainKHR swapchains[] = { _vk_swapchain };

		VkPresentInfoKHR present_info = {};
		present_info.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
		present_info.waitSemaphoreCount = 1;
		present_info.pWaitSemaphores = signal_semaphores;
		present_info.swapchainCount = 1;
		present_info.pSwapchains = swapchains;
		present_info.pImageIndices = &image_index;

		result = vkQueuePresentKHR(_vk_present_queue, &present_info);

		if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
		{
			RecreateSwapChain();
		}
		else if (result != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Present Swapchain Image!");
		}
	}

	int TestPhysicalDevice(VkPhysicalDevice device)
	{
		bool swap_chain_supported = false;

		if (CheckExtensions(device))
		{
			VkPhysicalDeviceProperties device_properties;
			vkGetPhysicalDeviceProperties(device, &device_properties);

			VkPhysicalDeviceFeatures device_features;
			vkGetPhysicalDeviceFeatures(device, &device_features);
			
			SwapChainSupport swap_chain_support = CheckSwapChainSupport(device);

			swap_chain_supported = !swap_chain_support._formats.empty() && !swap_chain_support._present_modes.empty() && device_features.samplerAnisotropy;
		}

		return swap_chain_supported;
	}

	bool CheckExtensions(VkPhysicalDevice device)
	{
		bool ret_val = true;
		
		uint32_t extension_num;
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_num, nullptr);
		std::vector<VkExtensionProperties> extensions(extension_num);
		vkEnumerateDeviceExtensionProperties(device, nullptr, &extension_num, extensions.data());
		
		for (const char* name : DEVICE_EXTENSIONS)
		{
			bool found = false;

			for (VkExtensionProperties properties : extensions)
			{
				if (strncmp(properties.extensionName, name, strlen(properties.extensionName)) == 0)
				{
					found = true;
					break;
				}
			}

			if (!found)
			{
				ret_val = false;
				break;
			}
		}

		return ret_val;
	}

	bool CheckValidationLayers()
	{
		bool ret_val = true;
		uint32_t layer_num;
		vkEnumerateInstanceLayerProperties(&layer_num, nullptr);
		std::vector<VkLayerProperties> layers(layer_num);
		vkEnumerateInstanceLayerProperties(&layer_num, layers.data());

		for (const char* layer_name : VALIDATION_LAYERS)
		{
			bool layer_found = false;

			for (const auto& layer : layers)
			{
				if (strncmp(layer_name, layer.layerName, strlen(layer_name)) == 0)
				{
					layer_found = true;
					break;
				}
			}

			if (!layer_found)
			{
				ret_val = false;
				break;
			}
		}

		return ret_val;
	}

	QueueFamilies CheckQueueFamilies(VkPhysicalDevice device)
	{
		QueueFamilies queue_family_data;

		// get physical device queue families to use
		uint32_t queue_family_num = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(_vk_physical_device, &queue_family_num, nullptr);

		std::vector<VkQueueFamilyProperties> queue_families(queue_family_num);
		vkGetPhysicalDeviceQueueFamilyProperties(_vk_physical_device, &queue_family_num, queue_families.data());

		for (uint32_t i = 0; i < queue_families.size(); ++i)
		{
			if (queue_families[i].queueCount > 0 && queue_families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)
			{
				queue_family_data._graphics_family = i;
			}

			VkBool32 present_support = false;
			vkGetPhysicalDeviceSurfaceSupportKHR(_vk_physical_device, i, _vk_surface, &present_support);

			if (queue_families[i].queueCount > 0 && present_support)
			{
				queue_family_data._present_family = i;
			}
		}

		if (!queue_family_data.QueuesAquired())
		{
			throw std::runtime_error("Required Queues Unavailable!");
		}

		return queue_family_data;
	}

	SwapChainSupport CheckSwapChainSupport(VkPhysicalDevice device)
	{
		SwapChainSupport support;

		vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, _vk_surface, &support._capabilities);
		
		uint32_t format_num;
		vkGetPhysicalDeviceSurfaceFormatsKHR(device, _vk_surface, &format_num, nullptr);
		if (format_num != 0)
		{
			support._formats.resize(format_num);
			vkGetPhysicalDeviceSurfaceFormatsKHR(device, _vk_surface, &format_num, support._formats.data());
		}

		uint32_t present_mode_num;
		vkGetPhysicalDeviceSurfacePresentModesKHR(device, _vk_surface, &present_mode_num, nullptr);
		if (present_mode_num != 0)
		{
			support._present_modes.resize(present_mode_num);
			vkGetPhysicalDeviceSurfacePresentModesKHR(device, _vk_surface, &present_mode_num, support._present_modes.data());
		}

		return support;
	}

	VkSurfaceFormatKHR ChooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR> & available_formats)
	{
		VkSurfaceFormatKHR ret_val;

		// check for no preferred format
		if (available_formats.size() == 1 && available_formats[0].format == VK_FORMAT_UNDEFINED)
		{
			ret_val = { VK_FORMAT_B8G8R8A8_UNORM, VK_COLOR_SPACE_SRGB_NONLINEAR_KHR };
		}
		else
		{
			// default to first option
			ret_val = available_formats[0];

			// try to find ideal option
			for (const auto & format : available_formats)
			{
				if (format.format == VK_FORMAT_B8G8R8A8_UNORM && format.colorSpace == VK_COLORSPACE_SRGB_NONLINEAR_KHR)
				{
					ret_val = format;
				}
			}
		}

		return ret_val;
	}

	VkPresentModeKHR ChooseSwapPresentMode(const std::vector<VkPresentModeKHR> available_modes)
	{
		// defualt to single guaranteed mode
		VkPresentModeKHR ret_val = VK_PRESENT_MODE_FIFO_KHR;

		// test for other modes
		for (const auto & mode : available_modes)
		{
			// best mode
			if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
			{
				ret_val = mode;
				break;
			}
			else if(mode == VK_PRESENT_MODE_IMMEDIATE_KHR)
			{
				ret_val = mode;
			}
		}

		return ret_val;
	}
	
	VkExtent2D ChooseSwapExtent(const VkSurfaceCapabilitiesKHR & capabilities)
	{
		VkExtent2D ret_val = capabilities.currentExtent;

		if (capabilities.currentExtent.width == std::numeric_limits<uint32_t>::max())
		{
			int width, height;
			glfwGetWindowSize(_p_glfw_window, &width, &height);
			ret_val = { static_cast<uint32_t>(width), static_cast<uint32_t>(height) };

			if (ret_val.width < capabilities.minImageExtent.width)
			{
				ret_val.width = capabilities.minImageExtent.width;
			}
			else if (ret_val.width > capabilities.maxImageExtent.width)
			{
				ret_val.width = capabilities.maxImageExtent.width;
			}

			if (ret_val.height < capabilities.minImageExtent.height)
			{
				ret_val.height = capabilities.minImageExtent.height;
			}
			else if (ret_val.height > capabilities.maxImageExtent.height)
			{
				ret_val.height = capabilities.maxImageExtent.height;
			}
		}

		return ret_val;
	}

	void TransitionImageLayout(VkImage image, VkFormat format, VkImageLayout old_layout, VkImageLayout new_layout)
	{
		VkCommandBuffer command_buffer = StartSingleTimeCommands();

		VkImageMemoryBarrier barrier = {};
		barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
		barrier.oldLayout = old_layout;
		barrier.newLayout = new_layout;
		barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
		barrier.image = image;

		if (new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_DEPTH_BIT;

			if (HasStencilComponent(format))
			{
				barrier.subresourceRange.aspectMask |= VK_IMAGE_ASPECT_STENCIL_BIT;
			}
		}
		else
		{
			barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		}

		barrier.subresourceRange.baseMipLevel = 0;
		barrier.subresourceRange.levelCount = 1;
		barrier.subresourceRange.baseArrayLayer = 0;
		barrier.subresourceRange.layerCount = 1;
		barrier.srcAccessMask = 0;
		barrier.dstAccessMask = 0;

		VkPipelineStageFlags source_stage;
		VkPipelineStageFlags dest_stage;

		if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dest_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
		}
		else if (old_layout == VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL && new_layout == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
		{
			barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
			barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
			source_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;
			dest_stage = VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
		}
		else if (old_layout == VK_IMAGE_LAYOUT_UNDEFINED && new_layout == VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
		{
			barrier.srcAccessMask = 0;
			barrier.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
			source_stage = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT;
			dest_stage = VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
		}
		else
		{
			throw std::runtime_error("Unsupported Layout Transition!");
		}

		vkCmdPipelineBarrier(command_buffer, source_stage, dest_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);

		EndSingleTimeCommands(command_buffer);
	}

	void CreateImage(uint32_t width, uint32_t height, VkFormat format, VkImageTiling tiling, VkImageUsageFlags usage, VkMemoryPropertyFlags mem_properties, VkImage& image, VkDeviceMemory& image_memory)
	{
		VkImageCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
		create_info.imageType = VK_IMAGE_TYPE_2D;
		create_info.extent.width = width;
		create_info.extent.height = height;
		create_info.extent.depth = 1;
		create_info.mipLevels = 1;
		create_info.arrayLayers = 1;
		create_info.format = format;
		create_info.tiling = tiling;
		create_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
		create_info.usage = usage;
		create_info.samples = VK_SAMPLE_COUNT_1_BIT;
		create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateImage(_vk_logical_device, &create_info, nullptr, &image) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Image!");
		}

		VkMemoryRequirements mem_requirements;
		vkGetImageMemoryRequirements(_vk_logical_device, image, &mem_requirements);

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = mem_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(mem_requirements.memoryTypeBits, mem_properties);

		if (vkAllocateMemory(_vk_logical_device, &alloc_info, nullptr, &image_memory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Allocate Image Memory!");
		}

		vkBindImageMemory(_vk_logical_device, image, image_memory, 0);
	}

	VkImageView CreateImageView(VkImage image, VkFormat format, VkImageAspectFlags aspect_flags)
	{
		VkImageViewCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
		create_info.image = image;
		create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
		create_info.format = format;
		create_info.subresourceRange.aspectMask = aspect_flags;
		create_info.subresourceRange.baseMipLevel = 0;
		create_info.subresourceRange.levelCount = 1;
		create_info.subresourceRange.baseArrayLayer = 0;
		create_info.subresourceRange.layerCount = 1;

		VkImageView image_view;
		if (vkCreateImageView(_vk_logical_device, &create_info, nullptr, &image_view) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Texture Image View!");
		}

		return image_view;
	}

	void CopyBufferToImage(VkBuffer buffer, VkImage image, uint32_t width, uint32_t height)
	{
		VkCommandBuffer command_buffer = StartSingleTimeCommands();

		VkBufferImageCopy image_copy = {};
		image_copy.bufferOffset = 0;
		image_copy.bufferRowLength = 0;
		image_copy.bufferImageHeight = 0;
		image_copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
		image_copy.imageSubresource.mipLevel = 0;
		image_copy.imageSubresource.baseArrayLayer = 0;
		image_copy.imageSubresource.layerCount = 1;
		image_copy.imageOffset = { 0, 0, 0 };
		image_copy.imageExtent = { width, height, 1 };

		vkCmdCopyBufferToImage(command_buffer, buffer, image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &image_copy);

		EndSingleTimeCommands(command_buffer);
	}

	// 
	

	// CHANGE COMMANDS SHIT TO ASYNC
	// setupcommandbuffer, helpers with buffer comms add and then flush any added comms


	VkCommandBuffer StartSingleTimeCommands()
	{
		VkCommandBufferAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		alloc_info.commandPool = _vk_command_pool;
		alloc_info.commandBufferCount = 1;

		VkCommandBuffer command_buffer;
		vkAllocateCommandBuffers(_vk_logical_device, &alloc_info, &command_buffer);

		VkCommandBufferBeginInfo begin_info = {};
		begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
		begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		vkBeginCommandBuffer(command_buffer, &begin_info);

		return command_buffer;
	}

	void EndSingleTimeCommands(VkCommandBuffer command_buffer)
	{
		vkEndCommandBuffer(command_buffer);

		VkSubmitInfo submit_info = {};
		submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &command_buffer;

		vkQueueSubmit(_vk_graphics_queue, 1, &submit_info, VK_NULL_HANDLE);
		vkQueueWaitIdle(_vk_graphics_queue);

		vkFreeCommandBuffers(_vk_logical_device, _vk_command_pool, 1, &command_buffer);
	}






	void CopyBuffer(VkBuffer src, VkBuffer dst, VkDeviceSize size)
	{
		VkCommandBuffer command_buffer = StartSingleTimeCommands();
		
		VkBufferCopy copy = {};
		copy.size = size;
		vkCmdCopyBuffer(command_buffer, src, dst, 1, &copy);
		
		EndSingleTimeCommands(command_buffer);
	}

	void CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& buffer_memory)
	{
		VkBufferCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		create_info.size = size;
		create_info.usage = usage;
		create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		if (vkCreateBuffer(_vk_logical_device, &create_info, nullptr, &buffer) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Buffer!");
		}

		VkMemoryRequirements memory_requirements;
		vkGetBufferMemoryRequirements(_vk_logical_device, buffer, &memory_requirements);

		VkMemoryAllocateInfo alloc_info = {};
		alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		alloc_info.allocationSize = memory_requirements.size;
		alloc_info.memoryTypeIndex = FindMemoryType(memory_requirements.memoryTypeBits, properties);

		if (vkAllocateMemory(_vk_logical_device, &alloc_info, nullptr, &buffer_memory) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Allocate Buffer Memory!");
		}

		vkBindBufferMemory(_vk_logical_device, buffer, buffer_memory, 0);
	}

	uint32_t FindMemoryType(uint32_t type_filter, VkMemoryPropertyFlags properties)
	{
		VkPhysicalDeviceMemoryProperties memory_properties;
		vkGetPhysicalDeviceMemoryProperties(_vk_physical_device, &memory_properties);

		for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i)
		{
			if (type_filter & (1 << i) && (memory_properties.memoryTypes[i].propertyFlags & properties) == properties)
			{
				return i;
			}
		}

		throw std::runtime_error("Failed To Find Suitable Memory Type!");
	}

	VkFormat FindSupportedFormat(const std::vector<VkFormat>& candidates, VkImageTiling tiling, VkFormatFeatureFlags features)
	{
		for (VkFormat format : candidates)
		{
			VkFormatProperties props;
			vkGetPhysicalDeviceFormatProperties(_vk_physical_device, format, &props);

			if (tiling == VK_IMAGE_TILING_LINEAR && (props.linearTilingFeatures & features) == features)
			{
				return format;
			}
			else if (tiling == VK_IMAGE_TILING_OPTIMAL && (props.optimalTilingFeatures & features) == features)
			{
				return format;
			}

			throw std::runtime_error("Failed To Find Supported Format!");
		}
	}

	std::vector<const char*> RequiredExtensions()
	{
		std::vector<const char*> ret_val;

		// get required extensions
		uint32_t glfw_ret_val_num = 0;
		const char** pp_glfw_ret_val_names;
		pp_glfw_ret_val_names = glfwGetRequiredInstanceExtensions(&glfw_ret_val_num);

		if (pp_glfw_ret_val_names == nullptr)
		{
			// vulkan not supported or no ret_vals to draw to screen exist
			throw std::runtime_error("Vulkan Graphics Unavailable!");
		}

		// add layer extensions
		for (uint32_t i = 0; i < glfw_ret_val_num; ++i)
		{
			ret_val.push_back(pp_glfw_ret_val_names[i]);
		}

		// add validation layer extensions
#ifndef NDEBUG
		ret_val.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
#endif

		// get available extensions
		uint32_t extension_num = 0;
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_num, nullptr);
		std::vector<VkExtensionProperties> extensions(extension_num);
		vkEnumerateInstanceExtensionProperties(nullptr, &extension_num, extensions.data());

		// validate that all required extensions are present
		for (uint32_t i = 0; i < ret_val.size(); ++i)
		{
			VkResult vk_result = VkResult::VK_ERROR_EXTENSION_NOT_PRESENT;
			for each (VkExtensionProperties vk_ext in extensions)
			{
				if (strncmp(ret_val[i], vk_ext.extensionName, strlen(vk_ext.extensionName)) == 0)
				{
					vk_result = VkResult::VK_SUCCESS;
					break;
				}
			}

			if (vk_result == VkResult::VK_ERROR_EXTENSION_NOT_PRESENT)
			{
				throw std::runtime_error("Missing Required Extensions!");
			}
		}

		return ret_val;
	}

	VkShaderModule CreateShaderModule(const std::vector<char> & blob)
	{
		VkShaderModule shader_module;

		VkShaderModuleCreateInfo create_info = {};
		create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		create_info.codeSize = blob.size();
		create_info.pCode = reinterpret_cast<const uint32_t*>(blob.data());

		if (vkCreateShaderModule(_vk_logical_device, &create_info, nullptr, &shader_module) != VK_SUCCESS)
		{
			throw std::runtime_error("Failed To Create Shader Module!");
		}

		return shader_module;
	}

	void LoadModel()
	{

	}

	static std::vector<char> ReadFile(const std::string & filename)
	{
		std::ifstream file(filename, std::ios::ate | std::ios::binary);

		if (!file.is_open())
		{
			throw std::runtime_error("Failed To Open File!");
		}

		size_t file_size = static_cast<size_t>(file.tellg());
		std::vector<char> buffer(file_size);

		file.seekg(0);
		file.read(buffer.data(), file_size);

		file.close();

		return buffer;
	}

	static VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallback(
		VkDebugReportFlagsEXT flags,
		VkDebugReportObjectTypeEXT obj_type,
		uint64_t obj,
		uint32_t location,
		int32_t code,
		const char* layer_prefix,
		const char* msg,
		void* user_data
	)
	{
		std::cerr << "Validation Layer: " << msg << std::endl;

		return VK_FALSE;
	}
	
	// BEGIN PRIVATE MEMBERS
	VkInstance _vk_instance;
	VkDebugReportCallbackEXT _vk_callback;
	VkSurfaceKHR _vk_surface;
	VkPhysicalDevice _vk_physical_device = VK_NULL_HANDLE;	///< The vk device - IMPLICITLY DESTROYED WITH INSTANCE
	VkDevice _vk_logical_device;

	VkSwapchainKHR _vk_swapchain = VK_NULL_HANDLE;
	std::vector<VkImage> _vk_swapchain_images;  ///< The swapchain images - OWNED BY SWAPCHAIN, DO NOT DESTROY
	std::vector<VkImageView> _vk_swapchain_image_views;
	std::vector<VkFramebuffer> _vk_swapchain_frame_buffers;
	VkFormat _vk_swapchain_format;
	VkExtent2D _vk_swapchain_extent;

	QueueFamilies _available_queue_families;
	VkQueue _vk_graphics_queue; ///< Queue of vk graphics - IMPLICITLY DESTROYED WITH DEVICE
	VkQueue _vk_present_queue;

	VkRenderPass _vk_render_pass;

	VkDescriptorSetLayout _vk_descriptor_set_layout;
	VkDescriptorPool _vk_descriptor_pool;
	VkDescriptorSet _vk_descriptor_set; ///< IMPLICITLY DESTROYED BY POOL

	VkPipelineLayout _vk_pipeline_layout;
	VkPipeline _vk_pipeline;

	VkCommandPool _vk_command_pool;

	VkBuffer _vk_uniform_buffer;
	VkDeviceMemory _vk_uniform_buffer_memory;

	VkImage _vk_texture_image;
	VkDeviceMemory _vk_texture_image_memory;
	VkImageView _vk_texture_image_view;
	VkSampler _vk_texture_sampler;

	std::vector<Vertex> _vertices;
	std::vector<uint32_t> _indices;
	VkBuffer _vk_vertex_buffer;
	VkDeviceMemory _vk_vertex_buffer_memory;
	VkBuffer _vk_index_buffer;
	VkDeviceMemory _vk_index_buffer_memory;

	VkImage _vk_depth_image;
	VkDeviceMemory _vk_depth_image_memory;
	VkImageView _vk_depth_image_view;

	std::vector<VkCommandBuffer> _vk_command_buffers;

	VkSemaphore _vk_image_available_semaphore;
	VkSemaphore _vk_frame_complete_semaphore;

	GLFWwindow* _p_glfw_window;
	uint32_t _window_width;
	uint32_t _window_height;
	char* _window_name;
	// END PRIVATE MEMBERS
};

int main()
{
	HelloTriangleApplication* p_app = new HelloTriangleApplication();

	try
	{
		p_app->Run();
	}
	catch (const std::runtime_error& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return EXIT_SUCCESS;
}