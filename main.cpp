/*
	Shows the bug reported
	here
	https://devtalk.nvidia.com/default/topic/1033055/vulkan/vulkan-fragment-geometry-shader-issues-since-the-397-xx-drivers-beta-official-/
	and
	https://devtalk.nvidia.com/default/topic/790452/reporting-graphics-driver-bugs-/?offset=15

	The example is very simplified and does not do what described there besides creating bounding boxes in the geom shader and shows them, while if node id is
	valid or red if not. (Red Color should never appear ==> it is the bug)

	If geometry shader provides a uint output to the fragment shader and fragment shader takes the value with flat modifier, the input becomes invalid!
	It causes crashes on official vk 1.1 drivers and instability on all vulkan 1.1  beta drivers starting with first 39X.XX the 389.20 (10) are working
	without issues.

	Please take a look at the geomertry shader and fragment shader, especially at the layout (location = 0) in flat uint node_id; and
	the if in the fragment shader.

	Thanks in advance!
*/
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>

#define VK_USE_PLATFORM_WIN32_KHR
#include <vulkan/vulkan.hpp>
#include <glfw/glfw3.h>
#include <iostream>

#include <glm/glm.hpp>
#include <glm/vec3.hpp>
#include <glm/fwd.hpp>
#include <glm/mat4x4.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#define GLFW_EXPOSE_NATIVE_WIN32
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#include "vertex.vert.h"
#include "fragment.frag.h"

////////////////////////////////////////////////////////////////////////////////
// some utility code
///////////////////////////////////////////////////////////////////////////////
bool isInstanceExtensionAvailable(std::string const& ext)
{
	std::vector<std::string> avail_exts;
	for (auto const& e : vk::enumerateInstanceExtensionProperties())
		avail_exts.push_back(e.extensionName);
	return std::find(avail_exts.begin(), avail_exts.end(), ext) != avail_exts.end();
}

bool isInstanceLayerAvailable(std::string const& layer)
{
	std::vector<std::string> avail_layers;
	for (auto const& l : vk::enumerateInstanceLayerProperties())
		avail_layers.push_back(l.layerName);
	return std::find(avail_layers.begin(), avail_layers.end(), layer) != avail_layers.end();
}

////////////////////////////////////////////////////////////////////////////////
template<typename T, std::size_t N>
std::vector<T> toVector(T const (&a)[N])
{
	return std::vector<T>(std::begin(a), std::end(a));
}

////////////////////////////////////////////////////////////////////////////////
vk::UniqueShaderModule createShader(vk::Device dev, std::vector<std::uint32_t> const& spv)
{
	auto const shader_info{ vk::ShaderModuleCreateInfo{}
		.setCodeSize(spv.size() * sizeof(std::uint32_t))
		.setPCode(spv.data()) };
	return dev.createShaderModuleUnique(shader_info);
}

// chooses a memory type index based on memory requirements as well as preferred and required memory property flags

////////////////////////////////////////////////////////////////////////////////
uint32_t selectMemoryTypeIndex(
	vk::PhysicalDevice phys_dev,
	vk::MemoryRequirements mem_req,
	vk::MemoryPropertyFlags preferred,
	vk::MemoryPropertyFlags required)
{
	auto const mem_props{ phys_dev.getMemoryProperties() };
	for (unsigned i{ 0 }; i < VK_MAX_MEMORY_TYPES; ++i)
	{
		if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & preferred) == preferred)
			return i;
	}
	if (required != preferred)
	{
		for (unsigned i{ 0 }; i < VK_MAX_MEMORY_TYPES; ++i)
		{
			if ((mem_req.memoryTypeBits & (1 << i)) && (mem_props.memoryTypes[i].propertyFlags & required) == required)
				return i;
		}
	}

	throw std::runtime_error{ "required memory type not available" };
}

//////////////////////////////////////////////////////////////////////////////////

struct Image
{
	vk::UniqueImage image;
	vk::UniqueDeviceMemory memory;

	Image(vk::UniqueImage&& _image,vk::UniqueDeviceMemory&& _memory) :
		image{ std::move(_image) },
		memory(std::move(_memory))
	{}

	Image()
	{}
};

struct Buffer
{
	vk::UniqueBuffer buffer;
	vk::UniqueDeviceMemory memory;

	Buffer(vk::UniqueBuffer&& _buffer, vk::UniqueDeviceMemory&& _memory) :
		buffer{ std::move(_buffer) },
		memory(std::move(_memory))
	{}

	Buffer()
	{}
};

class Scene
{

public:
	using Window = GLFWwindow;
	Scene();
	~Scene();
public:
	void initialize();
	void run();
	void shutdown();

private:
	void createWindowAndSurface();
	void initializeVKInstance();
	void selectQueueFamilyAndPhysicalDevice();
	void initializeDevice();

	///////////////////////////////////////////////////
	void createSurface();
	void createSwapChainAndImages();
	void createSwapChainAndDepthImageViews();

	////////////////////////////////////////////////////////////////////
	// Renderer
	void createPass();

	void createFramebuffer(); // one frame buffer is enough for both render passes (they are compatible)
	void allocateCommandBuffer();

	// descriptors and pipeline layouts
	void createShaderInterface();

	void createPipeline();
	void initSyncEntities();

	void buildCommandBuffer(uint32_t image_index);

private:
	// helper
	Image allocateImage(const vk::ImageCreateInfo& img_ci, vk::MemoryPropertyFlags required);

private:
	Window* m_window;
	uint32_t m_width = 1280;
	uint32_t m_height = 720;

	// Tweaker
	const vk::Format m_swapchain_format = vk::Format::eB8G8R8A8Unorm;
	const vk::Format m_depth_image_format = vk::Format::eD32Sfloat;

	const vk::PresentModeKHR m_present_mode = vk::PresentModeKHR::eFifo;
	uint32_t m_sw_num_images = 3; // determines how many images swap chain will have (and how many cmd buffers the renderer will use)
	////////////////////////////////////////////////////////////////////////////////////////////////

	vk::UniqueInstance m_instance;
	vk::PhysicalDevice m_phys_dev;
	uint32_t m_gq_fam_idx = std::numeric_limits<uint32_t>::max(); // graphics queue fam index
	vk::UniqueDevice m_device;
	vk::Queue m_gr_queue;

	////////////////////////////////////////////////////
	// surface and swap chain

	vk::UniqueSurfaceKHR m_surface;
	vk::UniqueSwapchainKHR m_swapchain;
	Image m_depth_buffer_image;
	std::vector<vk::Image> m_swapchain_imgs;
	std::vector<vk::UniqueImageView> m_swapchain_img_views;
	vk::UniqueImageView m_depth_image_view;

	/////////////////////////////////////////////////////

	//////////////////////////////////////////////////////////////////////////////////////////////////////
	// Renderer
	vk::UniqueRenderPass m_render_pass;

	std::vector<vk::UniqueFramebuffer> m_framebuffers;

	vk::UniqueCommandPool m_cmd_b_pool;

	// not really good way to do it but it is ok for bug reporting
	std::vector<vk::UniqueCommandBuffer> m_command_buffers;

	vk::UniqueShaderModule m_vert_shader;
	vk::UniqueShaderModule m_frag_shader;

	vk::UniquePipelineLayout m_pipeline_layout;


	vk::UniquePipeline m_pipeline;

	std::vector<vk::UniqueFence> m_fences;
	vk::UniqueSemaphore m_present_semaphore;
	vk::UniqueSemaphore m_draw_semaphore;

	struct AABB
	{
		float center[4];
		float radius[4];
	};

	std::vector<AABB> m_bboxes;
};

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


Scene::Scene()
{

}

Scene::~Scene()
{
	if (m_window)
		glfwDestroyWindow(m_window);

	glfwTerminate();
}


void Scene::initialize()
{
	// Initializing window instance devices and swapchain

	try
	{
		createWindowAndSurface();
		initializeVKInstance();
		selectQueueFamilyAndPhysicalDevice();
		initializeDevice();
		createSurface();
		createSwapChainAndImages();
		createSwapChainAndDepthImageViews();

		////////////////////////////////////////////////////////////

		// Initializing Renderer
		createPass();
		createFramebuffer();
		allocateCommandBuffer();
		createShaderInterface();
		createPipeline();
		initSyncEntities();
	}
	catch (vk::SystemError& e)
	{
		throw std::runtime_error(e.what());
	}
}

void Scene::run()
{
	while (1)
	{
		glfwPollEvents();
		if (glfwWindowShouldClose(m_window))
			break;

		const auto result = m_device->acquireNextImageKHR(*m_swapchain, UINT64_MAX, *m_draw_semaphore, {});

		if (result.result != vk::Result::eSuccess && result.result != vk::Result::eSuboptimalKHR)
		{
			throw std::runtime_error("Error getting next swapchain image!");
		}


		const uint32_t& image_index = result.value;

		m_device->waitForFences(*m_fences[image_index], true, UINT64_MAX);
		m_device->resetFences(*m_fences[image_index]);

		buildCommandBuffer(image_index);

		const vk::PipelineStageFlags wait_mask = vk::PipelineStageFlagBits::eColorAttachmentOutput;

		vk::SubmitInfo submit_info{};
		submit_info.commandBufferCount = 1;
		submit_info.pCommandBuffers = &*m_command_buffers[image_index];
		submit_info.pSignalSemaphores = &*m_present_semaphore;
		submit_info.waitSemaphoreCount = 1;
		submit_info.pWaitDstStageMask = &wait_mask;
		submit_info.pWaitSemaphores = &*m_draw_semaphore;
		submit_info.signalSemaphoreCount = 1;
		submit_info.pSignalSemaphores = &*m_present_semaphore;

		m_gr_queue.submit(submit_info, *m_fences[image_index]);

		vk::PresentInfoKHR present_info{};
		present_info.pImageIndices = &image_index;
		present_info.pSwapchains = &*m_swapchain;
		present_info.pWaitSemaphores = &*m_present_semaphore;
		present_info.swapchainCount = 1;
		present_info.waitSemaphoreCount = 1;

		try
		{
			const auto result = m_gr_queue.presentKHR(present_info);
		}
		catch (vk::SystemError& e)
		{
			if (e.code().value() != VK_ERROR_OUT_OF_DATE_KHR)
				throw std::runtime_error(e.what());
		}

	}
}

void Scene::shutdown()
{

}

void glfwerror(int ec, const char* emsg)
{
	std::cout << "Error Code: " << ec << " Error Msg: " << emsg << std::endl;
}

void Scene::createWindowAndSurface()
{
	glfwInit();
	glfwSetErrorCallback(glfwerror);

	glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

	m_window = glfwCreateWindow(m_width,m_height,"",nullptr,nullptr);

	if (m_window == nullptr)
	{
		throw std::runtime_error("Window Creation failed!");
		glfwTerminate();
	}
}

void Scene::initializeVKInstance()
{
	std::vector<const char*> extensions;
	std::vector < const char*> layers;

	if (!isInstanceExtensionAvailable(VK_KHR_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");

	if (!isInstanceExtensionAvailable(VK_KHR_WIN32_SURFACE_EXTENSION_NAME))
		throw std::runtime_error(std::string(VK_KHR_SURFACE_EXTENSION_NAME) + " is not available!");

	extensions.push_back(VK_KHR_SURFACE_EXTENSION_NAME);
	extensions.push_back(VK_KHR_WIN32_SURFACE_EXTENSION_NAME);

	if (isInstanceLayerAvailable("VK_LAYER_LUNARG_standard_validation"))
		layers.push_back("VK_LAYER_LUNARG_standard_validation");

	vk::InstanceCreateInfo inst_ci{};

	inst_ci.enabledLayerCount = (uint32_t)layers.size();
	inst_ci.ppEnabledLayerNames = layers.data();
	inst_ci.enabledExtensionCount = (uint32_t)extensions.size();
	inst_ci.ppEnabledExtensionNames = extensions.data();

	vk::ApplicationInfo app_info{};
	app_info.applicationVersion = VK_MAKE_VERSION(1,0,0);
	app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	app_info.pEngineName = "Test Engine";
	app_info.apiVersion = VK_MAKE_VERSION(1, 1, 0);

	inst_ci.pApplicationInfo = &app_info;

	m_instance = vk::createInstanceUnique(inst_ci);
}

void Scene::selectQueueFamilyAndPhysicalDevice()
{
	// simplified version

	// change id num to select required device
	const uint32_t required_phys_idx = 0;
	const auto phys_devs = m_instance->enumeratePhysicalDevices();

	if (phys_devs.size() <= required_phys_idx)
	{
		throw std::runtime_error("Invalid Physical Device Index provided!");
	}

	m_phys_dev = phys_devs[required_phys_idx];


	// I need only one graphics queue to show the bug ==> simply find first graphics queue family index with one queue

	const auto queue_fam_props = m_phys_dev.getQueueFamilyProperties();

	for (std::size_t i = 0;queue_fam_props.size();++i)
	{
		const auto prop = queue_fam_props[i];

		if (prop.queueFlags & vk::QueueFlagBits::eGraphics && prop.queueCount > 0)
		{
			m_gq_fam_idx = (uint32_t)i;
			break;
		}
	}

	if (m_gq_fam_idx == std::numeric_limits<uint32_t>::max())
	{
		// something very terrible is happened (device has no graphics queue)
		throw std::runtime_error("Can not find graphics family index!");
	}
}

void Scene::initializeDevice()
{
	vk::DeviceCreateInfo dev_ci{};

	vk::PhysicalDeviceFeatures dev_features{};
	dev_features.geometryShader                 = true;
	dev_features.robustBufferAccess             = true;
	dev_features.fragmentStoresAndAtomics       = true;
	dev_features.vertexPipelineStoresAndAtomics = true;

	vk::DeviceQueueCreateInfo dev_q_ci{};

	float queue_prio = 1.0f;

	dev_q_ci.queueCount = 1;
	dev_q_ci.pQueuePriorities = &queue_prio;
	dev_q_ci.queueFamilyIndex = m_gq_fam_idx;

	dev_ci.pEnabledFeatures = &dev_features;
	dev_ci.queueCreateInfoCount = 1;
	dev_ci.pQueueCreateInfos = &dev_q_ci;

	std::vector<const char*> extensions;

	extensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);

	dev_ci.enabledExtensionCount = (uint32_t)extensions.size();
	dev_ci.ppEnabledExtensionNames = extensions.data();

	m_device = m_phys_dev.createDeviceUnique(dev_ci);
	m_gr_queue = m_device->getQueue(m_gq_fam_idx, 0);
}

void Scene::createSurface()
{
	VkSurfaceKHR surface;
	void* hwnd = glfwGetWin32Window(m_window);

	auto const create_info = vk::Win32SurfaceCreateInfoKHR{}
		.setHinstance( GetModuleHandle(nullptr))
		.setHwnd(static_cast<HWND>(hwnd));
	vk::UniqueSurfaceKHR surf_tmp = m_instance->createWin32SurfaceKHRUnique(create_info, nullptr);

	if (surf_tmp.get() == vk::SurfaceKHR() || !m_phys_dev.getSurfaceSupportKHR(m_gq_fam_idx,(*surf_tmp)))
	{
		throw std::runtime_error("Can not create Surface!");
	}

	m_surface = std::move(surf_tmp);
}

void Scene::createSwapChainAndImages()
{
	// Keep it as simple as possible!!!

	auto const caps{ m_phys_dev.getSurfaceCapabilitiesKHR(*m_surface) };
	if (m_width != caps.currentExtent.width || m_height != caps.currentExtent.height)
		throw std::runtime_error{ "chosen image size not supported by window surface" };
	if (m_sw_num_images < caps.minImageCount)
		throw std::runtime_error{ "chosen image count is too small and not supported by the window surface" };
	if ((caps.maxImageCount != 0 && m_sw_num_images > caps.maxImageCount))
		throw std::runtime_error{ "chosen image count is too large and not supported by the window surface" };
	if (!(caps.supportedUsageFlags & vk::ImageUsageFlagBits::eColorAttachment))
		throw std::runtime_error{ "window surface cannot be used as color attachment" };

	bool format_found{ false };
	for (auto const& surf_format : m_phys_dev.getSurfaceFormatsKHR(*m_surface))
	{
		if (surf_format.format == vk::Format::eUndefined || surf_format.format == m_swapchain_format)
		{
			format_found = true;
			break;
		}
	}
	if (!format_found)
		throw std::runtime_error{ "window surface not compatible with chosen color format" };

	bool present_mode_found = false;

	for (const auto& surf_mode : m_phys_dev.getSurfacePresentModesKHR(*m_surface))
	{
		if (m_present_mode == surf_mode)
		{
			present_mode_found = true;
		}
	}

	if (!present_mode_found)
		throw std::runtime_error("Chosen Present Mode is not supported!");

	vk::SwapchainCreateInfoKHR sw_ci{};
	sw_ci.setSurface(*m_surface);
	sw_ci.setMinImageCount(m_sw_num_images);
	sw_ci.setImageFormat(m_swapchain_format);
	sw_ci.setImageExtent(vk::Extent2D{ m_width, m_height });
	sw_ci.setImageArrayLayers(1);
	sw_ci.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
	sw_ci.setPreTransform(vk::SurfaceTransformFlagBitsKHR::eIdentity);
	sw_ci.setCompositeAlpha(vk::CompositeAlphaFlagBitsKHR::eOpaque);
	sw_ci.setPresentMode(m_present_mode);
	sw_ci.setClipped(true);

	m_swapchain = m_device->createSwapchainKHRUnique(sw_ci);
	m_swapchain_imgs = m_device->getSwapchainImagesKHR(*m_swapchain);

	// now create DepthBuffer Image

	vk::ImageCreateInfo im_ci{};
	im_ci.arrayLayers  = 1;
	im_ci.extent = vk::Extent3D{ m_width,m_height,1 };
	im_ci.format = m_depth_image_format;
	im_ci.imageType = vk::ImageType::e2D;
	im_ci.initialLayout = vk::ImageLayout::eUndefined;
	im_ci.mipLevels = 1;
	im_ci.samples = vk::SampleCountFlagBits::e1;
	im_ci.tiling = vk::ImageTiling::eOptimal;
	im_ci.usage = vk::ImageUsageFlagBits::eDepthStencilAttachment;

	m_depth_buffer_image = allocateImage(im_ci, vk::MemoryPropertyFlagBits::eDeviceLocal);

}

Image Scene::allocateImage(const vk::ImageCreateInfo& img_ci, vk::MemoryPropertyFlags required)
{

	vk::UniqueImage img = m_device->createImageUnique(img_ci);
	const auto mem_req = m_device->getImageMemoryRequirements(*img);

	uint32_t index = selectMemoryTypeIndex(m_phys_dev, mem_req, required, required);

	vk::MemoryAllocateInfo ma_i{};
	ma_i.allocationSize = mem_req.size;
	ma_i.memoryTypeIndex = index;

	vk::UniqueDeviceMemory dev_mem = m_device->allocateMemoryUnique(ma_i);

	// Bind image to memory;
	m_device->bindImageMemory(*img, *dev_mem, 0);

	return{ std::move(img),std::move(dev_mem) };

}

void Scene::createSwapChainAndDepthImageViews()
{
	vk::ImageSubresourceRange img_sb_range{};
	img_sb_range.aspectMask = vk::ImageAspectFlagBits::eColor;
	img_sb_range.levelCount = 1;
	img_sb_range.layerCount = 1;

	vk::ImageViewCreateInfo sw_imgv_ci{};
	sw_imgv_ci.subresourceRange = img_sb_range;
	sw_imgv_ci.format = m_swapchain_format;
	sw_imgv_ci.viewType = vk::ImageViewType::e2D;

	for (uint32_t i = 0;i < m_sw_num_images;++i)
	{
		sw_imgv_ci.image = m_swapchain_imgs[i];
		m_swapchain_img_views.push_back(m_device->createImageViewUnique(sw_imgv_ci));
	}

	// depth image view
	img_sb_range.aspectMask = vk::ImageAspectFlagBits::eDepth;
	sw_imgv_ci.subresourceRange = img_sb_range;
	sw_imgv_ci.format = m_depth_image_format;
	sw_imgv_ci.image = *m_depth_buffer_image.image;

	m_depth_image_view = m_device->createImageViewUnique(sw_imgv_ci);

}

void Scene::createPass()
{
	vk::AttachmentDescription color_att_desc{};
	color_att_desc.format      = m_swapchain_format;
	color_att_desc.samples     = vk::SampleCountFlagBits::e1;
	color_att_desc.loadOp      = vk::AttachmentLoadOp::eClear;
	color_att_desc.storeOp     = vk::AttachmentStoreOp::eStore;
	color_att_desc.finalLayout = vk::ImageLayout::ePresentSrcKHR;


	vk::AttachmentDescription depth_att_desc{};
	depth_att_desc.format       = m_depth_image_format;
	depth_att_desc.samples      = vk::SampleCountFlagBits::e1;
	depth_att_desc.loadOp       = vk::AttachmentLoadOp::eClear;
	depth_att_desc.storeOp      = vk::AttachmentStoreOp::eDontCare;
	depth_att_desc.finalLayout  = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	const std::array<vk::AttachmentDescription, 2> att_descs
	{
		color_att_desc,
		depth_att_desc
	};

	vk::AttachmentReference color_att_ref{};
	color_att_ref.attachment = 0;
	color_att_ref.layout = vk::ImageLayout::eColorAttachmentOptimal;

	vk::AttachmentReference depth_att_ref{};

	depth_att_ref.attachment = 1;
	depth_att_ref.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

	vk::SubpassDescription subpass_desc{};
	subpass_desc.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
	subpass_desc.colorAttachmentCount = 1;
	subpass_desc.pColorAttachments = &color_att_ref;
	subpass_desc.pDepthStencilAttachment = &depth_att_ref;

	vk::RenderPassCreateInfo rp_ci{};

	rp_ci.attachmentCount = (uint32_t)att_descs.size();
	rp_ci.pAttachments = att_descs.data();
	rp_ci.subpassCount = 1;
	rp_ci.pSubpasses = &subpass_desc;

	m_render_pass = m_device->createRenderPassUnique(rp_ci);

}


void Scene::createFramebuffer()
{
	std::array<vk::ImageView,2> attachments;
	attachments[0] = nullptr;
	attachments[1] = *m_depth_image_view;

	vk::FramebufferCreateInfo fb_ci{};
	fb_ci.renderPass = *m_render_pass;
	fb_ci.attachmentCount = (uint32_t)attachments.size();
	fb_ci.pAttachments = attachments.data();
	fb_ci.width = m_width;
	fb_ci.height = m_height;
	fb_ci.layers = 1;

	for (uint32_t i = 0;i < m_sw_num_images;++i)
	{
		attachments[0] = *m_swapchain_img_views[i];
		m_framebuffers.push_back(m_device->createFramebufferUnique(fb_ci));
	}
}

void Scene::allocateCommandBuffer()
{
	vk::CommandPoolCreateInfo cmd_pool_ci{};
	cmd_pool_ci.queueFamilyIndex = m_gq_fam_idx;
	cmd_pool_ci.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;

	m_cmd_b_pool = m_device->createCommandPoolUnique(cmd_pool_ci);

	vk::CommandBufferAllocateInfo cmd_b_ai{};
	cmd_b_ai.commandBufferCount = 3;
	cmd_b_ai.commandPool = *m_cmd_b_pool;
	cmd_b_ai.level = vk::CommandBufferLevel::ePrimary;

	m_command_buffers = m_device->allocateCommandBuffersUnique(cmd_b_ai);
}


void Scene::createShaderInterface()
{
	vk::PipelineLayoutCreateInfo pl_ci{};

	std::vector<vk::PushConstantRange> ranges
	{
		{vk::ShaderStageFlagBits::eVertex,0, sizeof(uint32_t) * 16 }, // just to generate a bit bigger array
	};

	pl_ci.pushConstantRangeCount = (uint32_t)ranges.size();
	pl_ci.pPushConstantRanges = ranges.data();

	m_pipeline_layout = m_device->createPipelineLayoutUnique(pl_ci);
}

void Scene::createPipeline()
{
	//////////////////////////////////////////////////////////////////////////
	// Vertex Format
	vk::PipelineVertexInputStateCreateInfo vt_inp_ci{};
	vt_inp_ci.pVertexAttributeDescriptions = nullptr;
	vt_inp_ci.pVertexBindingDescriptions = nullptr;
	vt_inp_ci.vertexAttributeDescriptionCount = 0;
	vt_inp_ci.vertexBindingDescriptionCount = 0;

	vk::PipelineColorBlendAttachmentState cbas_ci{};
	cbas_ci.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

	vk::PipelineColorBlendStateCreateInfo cbs_ci{};
	cbs_ci.attachmentCount = 1;
	cbs_ci.pAttachments = &cbas_ci;

	vk::PipelineDepthStencilStateCreateInfo dss_ci{};
	dss_ci.depthWriteEnable = VK_FALSE;
	dss_ci.depthTestEnable = VK_FALSE;

	vk::PipelineInputAssemblyStateCreateInfo as_ci{};
	as_ci.topology = vk::PrimitiveTopology::eTriangleList;

	vk::PipelineMultisampleStateCreateInfo mss_ci{};
	mss_ci.rasterizationSamples = vk::SampleCountFlagBits::e1;

	vk::PipelineRasterizationStateCreateInfo rss_ci{};
	rss_ci.cullMode = vk::CullModeFlagBits::eNone;
	rss_ci.polygonMode = vk::PolygonMode::eFill;
	rss_ci.lineWidth = 1.0f;

	m_vert_shader = createShader(*m_device, toVector(::Vertex_vert));
	m_frag_shader = createShader(*m_device, toVector(::Fragment_frag));

	std::vector <vk::PipelineShaderStageCreateInfo> sh_stages;

	vk::PipelineShaderStageCreateInfo ss_ci{};
	ss_ci.pName = "main";

	ss_ci.module = *m_vert_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eVertex;

	sh_stages.push_back(ss_ci);

	ss_ci.module = *m_frag_shader;
	ss_ci.stage = vk::ShaderStageFlagBits::eFragment;

	sh_stages.push_back(ss_ci);

	vk::PipelineViewportStateCreateInfo vps_ci{};

	std::array<vk::Viewport,2> viewport{};
	viewport.at(0).width = (float) m_width / 2;
	viewport.at(0).height = (float) m_height;
	viewport.at(0).minDepth = 0.0f;
	viewport.at(0).maxDepth = 1.0f;

	viewport.at(1).x = (float)m_width / 2;
	viewport.at(1).width = (float)m_width / 2;
	viewport.at(1).height = (float)m_height;
	viewport.at(1).minDepth = 0.0f;
	viewport.at(1).maxDepth = 1.0f;

	vps_ci.viewportCount = 2;
	vps_ci.pViewports = viewport.data();

	std::array<vk::Rect2D,2> scissor{};
	scissor.at(0).extent.width = m_width / 2;
	scissor.at(0).extent.height = m_height;

	scissor.at(1).offset.x = m_width / 2;
	scissor.at(1).extent.width = m_width / 2;
	scissor.at(1).extent.height = m_height;

	vps_ci.scissorCount = 1;
	vps_ci.pScissors = scissor.data();

	///////////////////////////////////////////////////////////////////////////
	vk::GraphicsPipelineCreateInfo gp_ci{};

	gp_ci.pVertexInputState = &vt_inp_ci;
	gp_ci.layout = *m_pipeline_layout;
	gp_ci.pColorBlendState = &cbs_ci;
	gp_ci.pDepthStencilState = &dss_ci;
	gp_ci.pInputAssemblyState = &as_ci;
	gp_ci.pMultisampleState = &mss_ci;
	gp_ci.pRasterizationState = &rss_ci;
	gp_ci.stageCount = (uint32_t)sh_stages.size();
	gp_ci.pStages = sh_stages.data();
	gp_ci.renderPass = *m_render_pass;
	gp_ci.subpass = 0;
	gp_ci.pViewportState = &vps_ci;

	m_pipeline = m_device->createGraphicsPipelineUnique({}, gp_ci);
}

void Scene::initSyncEntities()
{
	vk::FenceCreateInfo f_ci{};
	f_ci.flags = vk::FenceCreateFlagBits::eSignaled;

	for (uint32_t i = 0;i < m_sw_num_images;++i)
	{
		m_fences.push_back(m_device->createFenceUnique(f_ci));
	}

	m_draw_semaphore = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
	m_present_semaphore = m_device->createSemaphoreUnique(vk::SemaphoreCreateInfo());
}

void Scene::buildCommandBuffer(uint32_t image_index)
{
	vk::CommandBufferBeginInfo cmd_begin_info{};
	auto& cmd = *m_command_buffers[image_index];

	cmd.begin(cmd_begin_info);
	const std::array<float,4> clear_color{0,0,1,1};

	const std::array<vk::ClearValue, 2> clear_values
	{
		vk::ClearColorValue(clear_color),
		vk::ClearDepthStencilValue(1,0)
	};

	vk::RenderPassBeginInfo rp_begin_info{};
	rp_begin_info.framebuffer = *m_framebuffers[image_index];
	rp_begin_info.renderArea = vk::Rect2D({ 0,0 }, { m_width,m_height });
	rp_begin_info.renderPass = *m_render_pass;
	rp_begin_info.clearValueCount = (uint32_t)clear_values.size();
	rp_begin_info.pClearValues = clear_values.data();

	cmd.beginRenderPass(rp_begin_info, vk::SubpassContents::eInline);

	cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline);
	std::array<uint32_t, 16> active_view_data;
	active_view_data.at(0) = 0;
	active_view_data.at(1) = 1;

	cmd.pushConstants<uint32_t>(*m_pipeline_layout, vk::ShaderStageFlagBits::eVertex,0,active_view_data);

	cmd.draw(3, 2, 0, 0);

	cmd.endRenderPass();
	cmd.end();
}

int main()
{
	SetProcessDPIAware();

	Scene scene;
	try
	{
		scene.initialize();
		scene.run();

	}
	catch (std::exception& e)
	{
		scene.shutdown();
		std::cout << "Error Occurred: " << e.what() << std::endl;
	}

	return 0;
}
