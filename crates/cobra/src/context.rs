use crate::{BindlessShader, FragmentShader, VertexShader};
use crate::{CobraInitError, ComputeQueue, GraphicsQueue, ImageFormat, Queue, TransferQueue};
use ash::vk;
use raw_window_handle::RawDisplayHandle;
use std::prelude::v1::Vec;
use std::sync::atomic::AtomicU32;
use std::{
    mem::ManuallyDrop,
    sync::{
        atomic::{AtomicU64, Ordering}, Arc,
        Mutex,
    },
};

#[derive(Debug, Default)]
pub(crate) struct BindlessHandle {
    counter: AtomicU32,
    recycle_list: Mutex<Vec<u32>>,
}

impl BindlessHandle {
    pub(crate) fn acquire(&self) -> u32 {
        if let Some(handle) = self.recycle_list.lock().unwrap().pop() {
            handle
        } else {
            self.counter.fetch_add(1, Ordering::SeqCst) + 1
        }
    }

    pub(crate) fn recycle(&self, handle: u32) {
        self.recycle_list.lock().unwrap().push(handle);
    }
}

pub struct Context {
    pub(crate) entry: ash::Entry,
    pub(crate) instance: ash::Instance,
    pub(crate) gpu: vk::PhysicalDevice,
    pub(crate) device: ash::Device,
    pub(crate) surface_khr: ash::khr::surface::Instance,
    pub(crate) swapchain_khr: ash::khr::swapchain::Device,

    pub(crate) allocator: ManuallyDrop<vk_mem::Allocator>,
    pub(crate) timeline_value: AtomicU64,
    pub(crate) deletion_queue: Mutex<DeletionQueue>,
    graphics_queue: GraphicsQueue,
    transfer_queue: Option<TransferQueue>,
    compute_queue: Option<ComputeQueue>,

    descriptor_pool: vk::DescriptorPool,
    descriptor_set_layout: vk::DescriptorSetLayout,
    pub(crate) descriptor_set: vk::DescriptorSet,
    pub(crate) pipeline_layout: vk::PipelineLayout,

    pub(crate) storage_handle: BindlessHandle,
    pub(crate) sampled_image_handle: BindlessHandle,
    pub(crate) storage_image_handle: BindlessHandle,
    pub(crate) sampler_handle: BindlessHandle,
}

impl Context {
    /// Initialize the context, using the display handle to grab platform specific WSI extensions
    /// # Errors
    /// * [`CobraInitError::InstanceInitialization`]: The [`VK_EXT_surface_maintenance1`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_surface_maintenance1.html)
    ///   or [`VK_KHR_get_surface_capabilities2`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_get_surface_capabilities2.html) extensions aren't present
    /// * [`CobraInitError::NoAvailableGpus`]: There's no device that supports Vulkan
    /// * [`CobraInitError::NoGraphicsQueueAvailable`]: Device doesn't have a queue that supports graphics available
    /// * [`CobraInitError::DeviceCreation`]: Device doesn't support Vulkan 1.3 or the required
    ///   [`VK_KHR_maintenance5`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_KHR_maintenance5.html) and
    ///   [`VK_EXT_swapchain_maintenance1`](https://registry.khronos.org/vulkan/specs/latest/man/html/VK_EXT_swapchain_maintenance1.html) extensions
    /// # Panics
    /// Will panic if the host or device run out of memory
    pub fn new(display_handle: RawDisplayHandle) -> Result<Arc<Context>, CobraInitError> {
        unsafe {
            let entry = ash::Entry::load().expect("Failed to load vulkan DLL");

            let mut instance_extensions = ash_window::enumerate_required_extensions(display_handle)
                .unwrap()
                .to_vec();
            instance_extensions.extend_from_slice(&[
                vk::EXT_SURFACE_MAINTENANCE1_NAME.as_ptr(),
                vk::KHR_GET_SURFACE_CAPABILITIES2_NAME.as_ptr(),
            ]);

            let Ok(instance) = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(
                        &vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3),
                    )
                    .enabled_extension_names(instance_extensions.as_slice()),
                None,
            ) else {
                return Err(CobraInitError::InstanceInitialization);
            };

            let physical_devices = instance.enumerate_physical_devices().unwrap();
            let gpu = physical_devices
                .iter()
                .copied()
                .find(|&device| {
                    let properties = instance.get_physical_device_properties(device);
                    properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
                })
                .unwrap_or(match physical_devices.first() {
                    Some(&v) => v,
                    None => return Err(CobraInitError::NoAvailableGpus),
                });

            let families = instance.get_physical_device_queue_family_properties(gpu);
            let families = [
                (vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, None),
                (
                    vk::QueueFlags::TRANSFER,
                    Some(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
                ),
                (
                    vk::QueueFlags::COMPUTE,
                    Some(vk::QueueFlags::GRAPHICS | vk::QueueFlags::TRANSFER),
                ),
            ]
            .into_iter()
            .map(|(required, does_not_contain)| {
                families
                    .iter()
                    .position(|q| {
                        q.queue_flags.contains(required)
                            && does_not_contain.is_none_or(|f| !q.queue_flags.contains(f))
                    })
                    .map(|f| u32::try_from(f).unwrap())
            })
            .collect::<Vec<_>>();
            if families[0].is_none() {
                return Err(CobraInitError::NoGraphicsQueueAvailable);
            }

            let Ok(device) = instance.create_device(
                gpu,
                &vk::DeviceCreateInfo::default()
                    .queue_create_infos(
                        families
                            .iter()
                            .copied()
                            .flatten()
                            .map(|f| {
                                vk::DeviceQueueCreateInfo::default()
                                    .queue_family_index(f)
                                    .queue_priorities(&[1.0])
                            })
                            .collect::<Vec<_>>()
                            .as_slice(),
                    )
                    .enabled_extension_names(&[
                        vk::KHR_SWAPCHAIN_NAME.as_ptr(),
                        vk::KHR_MAINTENANCE5_NAME.as_ptr(),
                        vk::EXT_SWAPCHAIN_MAINTENANCE1_NAME.as_ptr(),
                    ])
                    .push_next(
                        &mut vk::PhysicalDeviceFeatures2::default()
                            .features(
                                vk::PhysicalDeviceFeatures::default()
                                    .shader_sampled_image_array_dynamic_indexing(true)
                                    .shader_storage_image_array_dynamic_indexing(true)
                                    .shader_storage_buffer_array_dynamic_indexing(true),
                            )
                            .push_next(
                                &mut vk::PhysicalDeviceVulkan12Features::default()
                                    .timeline_semaphore(true)
                                    .scalar_block_layout(true)
                                    .vulkan_memory_model(true)
                                    .descriptor_indexing(true)
                                    .shader_storage_buffer_array_non_uniform_indexing(true)
                                    .shader_sampled_image_array_non_uniform_indexing(true)
                                    .shader_storage_image_array_non_uniform_indexing(true)
                                    .runtime_descriptor_array(true)
                                    .descriptor_binding_partially_bound(true)
                                    .descriptor_binding_storage_buffer_update_after_bind(true)
                                    .descriptor_binding_sampled_image_update_after_bind(true)
                                    .descriptor_binding_storage_image_update_after_bind(true),
                            )
                            .push_next(
                                &mut vk::PhysicalDeviceVulkan13Features::default()
                                    .synchronization2(true)
                                    .dynamic_rendering(true),
                            )
                            .push_next(
                                &mut vk::PhysicalDeviceMaintenance5FeaturesKHR::default()
                                    .maintenance5(true),
                            )
                            .push_next(
                                &mut vk::PhysicalDeviceSwapchainMaintenance1FeaturesEXT::default()
                                    .swapchain_maintenance1(true),
                            ),
                    ),
                None,
            ) else {
                return Err(CobraInitError::DeviceCreation);
            };

            let allocator = ManuallyDrop::new(
                vk_mem::Allocator::new(vk_mem::AllocatorCreateInfo::new(&instance, &device, gpu))
                    .unwrap(),
            );
            let graphics_queue = Queue::new(
                &device,
                device.get_device_queue(families[0].unwrap(), 0),
                families[0].unwrap(),
            );
            let transfer_queue =
                families[1].map(|f| Queue::new(&device, device.get_device_queue(f, 0), f));
            let compute_queue =
                families[2].map(|f| Queue::new(&device, device.get_device_queue(f, 0), f));

            let (descriptor_pool, descriptor_set_layout, descriptor_set, pipeline_layout) =
                Self::create_bindless_objects(&device);
            let surface_khr = ash::khr::surface::Instance::new(&entry, &instance);
            let swapchain_khr = ash::khr::swapchain::Device::new(&instance, &device);

            Ok(Arc::new(Context {
                entry,
                instance,
                gpu,
                device,
                surface_khr,
                swapchain_khr,
                allocator,
                graphics_queue,
                transfer_queue,
                compute_queue,
                descriptor_pool,
                descriptor_set_layout,
                descriptor_set,
                pipeline_layout,

                timeline_value: AtomicU64::default(),
                deletion_queue: Mutex::default(),
                storage_handle: BindlessHandle::default(),
                sampled_image_handle: BindlessHandle::default(),
                storage_image_handle: BindlessHandle::default(),
                sampler_handle: BindlessHandle::default(),
            }))
        }
    }

    pub fn graphics_queue(&self) -> &GraphicsQueue {
        &self.graphics_queue
    }
    pub fn transfer_queue(&self) -> Option<&TransferQueue> {
        self.transfer_queue.as_ref()
    }
    pub fn compute_queue(&self) -> Option<&ComputeQueue> {
        self.compute_queue.as_ref()
    }

    pub(crate) fn advance_timeline(&self) -> u64 {
        self.timeline_value.fetch_add(1, Ordering::SeqCst) + 1
    }

    fn create_bindless_objects(
        device: &ash::Device,
    ) -> (
        vk::DescriptorPool,
        vk::DescriptorSetLayout,
        vk::DescriptorSet,
        vk::PipelineLayout,
    ) {
        unsafe {
            let (pool_sizes, bindings): (Vec<_>, Vec<_>) = [
                (
                    vk::DescriptorType::STORAGE_BUFFER,
                    100_000,
                    crate::STORAGE_BINDING,
                ),
                (
                    vk::DescriptorType::SAMPLED_IMAGE,
                    1 << 20,
                    crate::SAMPLED_IMAGE_BINDING,
                ),
                (
                    vk::DescriptorType::STORAGE_IMAGE,
                    1 << 20,
                    crate::STORAGE_IMAGE_BINDING,
                ),
                (vk::DescriptorType::SAMPLER, 1 << 12, crate::SAMPLER_BINDING),
            ]
            .map(|(ty, size, binding)| {
                let pool_size = vk::DescriptorPoolSize::default()
                    .ty(ty)
                    .descriptor_count(size);
                let binding = vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(ty)
                    .descriptor_count(size)
                    .stage_flags(vk::ShaderStageFlags::ALL);
                (pool_size, binding)
            })
            .into_iter()
            .unzip();

            let descriptor_pool = device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(1)
                        .pool_sizes(pool_sizes.as_slice())
                        .flags(vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND),
                    None,
                )
                .unwrap();
            let descriptor_set_layout = device
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default()
                        .bindings(bindings.as_slice())
                        .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                        .push_next(
                            &mut vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
                                .binding_flags(
                                    bindings
                                        .iter()
                                        .map(|_| {
                                            vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                                                | vk::DescriptorBindingFlags::PARTIALLY_BOUND
                                        })
                                        .collect::<Vec<_>>()
                                        .as_slice(),
                                ),
                        ),
                    None,
                )
                .unwrap();
            let descriptor_set = device
                .allocate_descriptor_sets(
                    &vk::DescriptorSetAllocateInfo::default()
                        .descriptor_pool(descriptor_pool)
                        .set_layouts(&[descriptor_set_layout]),
                )
                .unwrap()[0];

            let pipeline_layout = device
                .create_pipeline_layout(
                    &vk::PipelineLayoutCreateInfo::default()
                        .set_layouts(&[descriptor_set_layout])
                        .push_constant_ranges(&[vk::PushConstantRange::default()
                            .stage_flags(vk::ShaderStageFlags::ALL)
                            .size(128)]),
                    None,
                )
                .unwrap();
            (
                descriptor_pool,
                descriptor_set_layout,
                descriptor_set,
                pipeline_layout,
            )
        }
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            self.deletion_queue.lock().unwrap().flush(self);
            ManuallyDrop::drop(&mut self.allocator);

            self.graphics_queue.destroy(&self.device);
            if let Some(q) = self.transfer_queue() {
                q.destroy(&self.device);
            }
            if let Some(q) = self.compute_queue() {
                q.destroy(&self.device);
            }

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

#[doc(hidden)]
pub struct Pipeline<const CAPABILITIES: u32> {
    pub context: Arc<Context>,
    pub pipeline: vk::Pipeline,
}
pub type GraphicsPipeline = Pipeline<{ crate::GRAPHICS_CAPABILITY }>;
pub type ComputePipeline = Pipeline<{ crate::GRAPHICS_CAPABILITY }>;

impl GraphicsPipeline {
    pub fn new(
        context: Arc<Context>,
        vs: &impl BindlessShader<ShaderType = VertexShader>,
        fs: &impl BindlessShader<ShaderType = FragmentShader>,
        color_formats: &[ImageFormat],
    ) -> Self {
        unsafe {
            let vs = vs.spirv_binary();
            let fs = fs.spirv_binary();

            let pipeline = context
                .device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[vk::GraphicsPipelineCreateInfo::default()
                        .stages(&[
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::VERTEX)
                                .name(vs.entry_point_name)
                                .push_next(
                                    &mut vk::ShaderModuleCreateInfo::default().code(vs.binary),
                                ),
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(vk::ShaderStageFlags::FRAGMENT)
                                .name(fs.entry_point_name)
                                .push_next(
                                    &mut vk::ShaderModuleCreateInfo::default().code(fs.binary),
                                ),
                        ])
                        .vertex_input_state(&vk::PipelineVertexInputStateCreateInfo::default())
                        .input_assembly_state(
                            &vk::PipelineInputAssemblyStateCreateInfo::default()
                                .topology(vk::PrimitiveTopology::TRIANGLE_LIST),
                        )
                        .tessellation_state(&vk::PipelineTessellationStateCreateInfo::default())
                        .viewport_state(
                            &vk::PipelineViewportStateCreateInfo::default()
                                .viewport_count(1)
                                .scissor_count(1),
                        )
                        .rasterization_state(
                            &vk::PipelineRasterizationStateCreateInfo::default()
                                .polygon_mode(vk::PolygonMode::FILL)
                                .cull_mode(vk::CullModeFlags::NONE)
                                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                                .line_width(1.0),
                        )
                        .multisample_state(
                            &vk::PipelineMultisampleStateCreateInfo::default()
                                .rasterization_samples(vk::SampleCountFlags::TYPE_1),
                        )
                        .depth_stencil_state(&vk::PipelineDepthStencilStateCreateInfo::default())
                        .color_blend_state(
                            &vk::PipelineColorBlendStateCreateInfo::default().attachments(&[
                                vk::PipelineColorBlendAttachmentState::default()
                                    .color_write_mask(vk::ColorComponentFlags::RGBA)
                                    .blend_enable(true)
                                    .src_color_blend_factor(vk::BlendFactor::ONE)
                                    .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA)
                                    .color_blend_op(vk::BlendOp::ADD)
                                    .src_alpha_blend_factor(vk::BlendFactor::ONE_MINUS_DST_ALPHA)
                                    .dst_alpha_blend_factor(vk::BlendFactor::ONE)
                                    .alpha_blend_op(vk::BlendOp::ADD),
                            ]),
                        )
                        .dynamic_state(
                            &vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&[
                                vk::DynamicState::VIEWPORT,
                                vk::DynamicState::SCISSOR,
                            ]),
                        )
                        .layout(context.pipeline_layout)
                        .push_next(
                            &mut vk::PipelineRenderingCreateInfo::default()
                                .color_attachment_formats(
                                    &color_formats
                                        .iter()
                                        .copied()
                                        .map(std::convert::Into::into)
                                        .collect::<Vec<_>>(),
                                ),
                        )],
                    None,
                )
                .unwrap()[0];
            Pipeline { context, pipeline }
        }
    }
}

impl<const C: u32> Drop for Pipeline<C> {
    fn drop(&mut self) {
        self.context
            .deletion_queue
            .lock()
            .unwrap()
            .push_pipeline(self.context.as_ref(), self.pipeline);
    }
}

#[derive(Default)]
pub struct DeletionQueue {
    pipelines: Vec<(vk::Pipeline, u64)>,
    buffers: Vec<((vk::Buffer, vk_mem::Allocation), u64)>,
    images: Vec<((vk::Image, vk_mem::Allocation), u64)>,
    image_views: Vec<(vk::ImageView, u64)>,
    samplers: Vec<(vk::Sampler, u64)>,
    swapchains: Vec<(vk::SwapchainKHR, Vec<vk::Semaphore>, vk::Fence)>, // TODO: maybe its worth trying to recycle the swapchains?
}

impl DeletionQueue {
    pub fn push_pipeline(&mut self, context: &Context, pipeline: vk::Pipeline) {
        self.pipelines
            .push((pipeline, context.timeline_value.load(Ordering::SeqCst)));
    }
    pub fn push_buffer(&mut self, context: &Context, buffer: (vk::Buffer, vk_mem::Allocation)) {
        self.buffers
            .push((buffer, context.timeline_value.load(Ordering::SeqCst)));
    }
    pub fn push_image(&mut self, context: &Context, image: (vk::Image, vk_mem::Allocation)) {
        self.images
            .push((image, context.timeline_value.load(Ordering::SeqCst)));
    }
    pub fn push_image_view(&mut self, context: &Context, image_view: vk::ImageView) {
        self.image_views
            .push((image_view, context.timeline_value.load(Ordering::SeqCst)));
    }
    pub fn push_sampler(&mut self, context: &Context, sampler: vk::Sampler) {
        self.samplers
            .push((sampler, context.timeline_value.load(Ordering::SeqCst)));
    }
    pub fn push_swapchain(&mut self, swapchain: (vk::SwapchainKHR, Vec<vk::Semaphore>, vk::Fence)) {
        self.swapchains.push(swapchain);
    }

    pub fn flush(&mut self, context: &Context) {
        Self::clear_queue(context, &mut self.pipelines, |p| unsafe {
            context.device.destroy_pipeline(*p, None);
        });
        Self::clear_queue(context, &mut self.buffers, |(b, a)| unsafe {
            context.allocator.destroy_buffer(*b, a);
        });
        Self::clear_queue(context, &mut self.images, |(m, a)| unsafe {
            context.allocator.destroy_image(*m, a);
        });
        Self::clear_queue(context, &mut self.image_views, |m| unsafe {
            context.device.destroy_image_view(*m, None);
        });
        Self::clear_queue(context, &mut self.samplers, |s| unsafe {
            context.device.destroy_sampler(*s, None);
        });

        self.swapchains
            .retain(|(swapchain, semaphores, fence)| unsafe {
                let signaled = context.device.get_fence_status(*fence).unwrap();
                if signaled {
                    context.swapchain_khr.destroy_swapchain(*swapchain, None);
                    for semaphore in semaphores {
                        context.device.destroy_semaphore(*semaphore, None);
                    }
                }
                !signaled
            });
    }

    fn clear_queue<T, F>(context: &Context, queue: &mut Vec<(T, u64)>, func: F)
    where
        F: Fn(&mut T),
    {
        queue.retain_mut(|(v, value)| {
            if Self::queue_should_be_cleared(context, *value, Some(context.graphics_queue()))
                && Self::queue_should_be_cleared(context, *value, context.transfer_queue())
                && Self::queue_should_be_cleared(context, *value, context.compute_queue())
            {
                func(v);
                return false;
            }

            true
        });
    }

    fn queue_should_be_cleared<const C: u32>(
        context: &Context,
        value: u64,
        queue: Option<&Queue<C>>,
    ) -> bool {
        match queue {
            Some(queue) => {
                value <= queue.fence.current_value(context)
                    || !queue.submitted_since_clear.swap(false, Ordering::SeqCst)
            }
            None => true,
        }
    }
}

pub(crate) fn subresource_range(aspect_mask: vk::ImageAspectFlags) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange::default()
        .aspect_mask(aspect_mask)
        .level_count(1)
        .layer_count(1)
}
