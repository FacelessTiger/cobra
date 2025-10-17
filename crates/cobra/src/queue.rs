use std::prelude::v1::Vec;
use std::sync::{
    atomic::{AtomicBool, AtomicU64, Ordering},
    Mutex,
};

use crate::image::Image;
use crate::swapchain::Swapchain;
use crate::Context;
use crate::{filling_span, ImageLayout};
use ash::vk;
use log::info;

pub struct Fence {
    timeline_semaphore: vk::Semaphore,
    last_seen_value: AtomicU64,
}

impl Fence {
    pub fn new(device: &ash::Device) -> Fence {
        unsafe {
            let timeline_semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(
                        &mut vk::SemaphoreTypeCreateInfo::default()
                            .semaphore_type(vk::SemaphoreType::TIMELINE)
                            .initial_value(0),
                    ),
                    None,
                )
                .unwrap();
            Fence {
                timeline_semaphore,
                last_seen_value: AtomicU64::new(0),
            }
        }
    }

    pub fn current_value(&self, context: &Context) -> u64 {
        let last_seen = self.last_seen_value.load(Ordering::SeqCst);
        if last_seen >= context.timeline_value.load(Ordering::SeqCst) {
            return last_seen;
        }

        unsafe {
            let last_seen = context
                .device
                .get_semaphore_counter_value(self.timeline_semaphore)
                .unwrap();
            self.last_seen_value.store(last_seen, Ordering::SeqCst);
            last_seen
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct CommandPool {
    pool: vk::CommandPool,
    buffer: vk::CommandBuffer,
}

pub struct Queue<const CAPABILITIES: u32> {
    queue: vk::Queue,
    family: u32,
    pub(crate) submitted_since_clear: AtomicBool,
    image_acquired: Mutex<(bool, vk::Semaphore, vk::SwapchainKHR, u32, vk::Fence)>,

    pub(crate) fence: Fence,
    // TODO: do queues "own" command pools, as in does it make more sense for queues to hold this
    // or for the context to hold this
    available_pools: Mutex<Vec<CommandPool>>,
    pending_pools: Mutex<Vec<(CommandPool, u64)>>,
}

pub type GraphicsQueue = Queue<{ crate::GRAPHICS_CAPABILITY }>;
pub type ComputeQueue = Queue<{ crate::COMPUTE_CAPABILITY }>;
pub type TransferQueue = Queue<{ crate::TRANSFER_CAPABILITY }>;

#[derive(Debug, Default, Clone, Copy)]
pub struct Timestamp {
    value: u64,
    ty: Option<u32>,
}

impl Timestamp {
    /// # Panics
    /// Will panic if run out of device or host memory
    pub fn wait(&self, context: &Context) {
        if self.ty.is_none() {
            return;
        }

        unsafe {
            context
                .device
                .wait_semaphores(
                    &vk::SemaphoreWaitInfo::default()
                        .semaphores(&[self.get_fence(context).timeline_semaphore])
                        .values(&[self.value]),
                    u64::MAX,
                )
                .unwrap();
        }
    }

    fn get_fence<'a>(&self, context: &'a Context) -> &'a Fence {
        fn get_fence_internal<const C: u32>(queue: &Queue<C>) -> &Fence {
            &queue.fence
        }

        match self.ty.unwrap() {
            crate::GRAPHICS_CAPABILITY => get_fence_internal(context.graphics_queue()),
            crate::TRANSFER_CAPABILITY => get_fence_internal(context.transfer_queue().unwrap()),
            crate::COMPUTE_CAPABILITY => get_fence_internal(context.compute_queue().unwrap()),
            _ => unimplemented!("Unknown queue capability"),
        }
    }
}

impl<const CAPABILITIES: u32> Queue<CAPABILITIES> {
    pub fn new(device: &ash::Device, queue: vk::Queue, family: u32) -> Self {
        Queue {
            queue,
            family,
            submitted_since_clear: AtomicBool::new(false),
            image_acquired: Mutex::new((
                false,
                vk::Semaphore::null(),
                vk::SwapchainKHR::null(),
                0,
                vk::Fence::null(),
            )),
            fence: Fence::new(device),
            available_pools: Mutex::new(Vec::new()),
            pending_pools: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_semaphore(self.fence.timeline_semaphore, None);

            self.available_pools
                .lock()
                .unwrap()
                .iter()
                .for_each(|pool| {
                    device.destroy_command_pool(pool.pool, None);
                });
            self.pending_pools
                .lock()
                .unwrap()
                .iter()
                .for_each(|(pool, _)| {
                    device.destroy_command_pool(pool.pool, None);
                });
        }
    }

    /// # Panics
    /// Will panic if ran out of host or device memory, or if the surface is lost
    pub fn acquire<const R: bool>(
        &self,
        context: &Context,
        window: &mut Swapchain,
        cmd: &mut filling_span::GraphicsOnly<R>,
        func: impl FnOnce(&Image, &mut filling_span::GraphicsOnly<R>),
    ) {
        unsafe {
            // TODO: once vulkan allows timeline semaphores directly with WSI in forever these empty submits and binary semaphores wont be necessary
            window.increment_semaphore_index();
            let (index, _) = context
                .swapchain_khr
                .acquire_next_image(
                    window.swapchain,
                    u64::MAX,
                    window.current_semaphore(),
                    vk::Fence::null(),
                )
                .unwrap();
            window.image_index = index;

            // convert acquire binary value to timeline
            context
                .device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default()
                        .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(window.current_semaphore())
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.fence.timeline_semaphore)
                            .value(context.advance_timeline())
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                    vk::Fence::null(),
                )
                .unwrap();
            window.increment_semaphore_index();

            func(&window.images[index as usize], cmd);
            cmd.transition_image(&window.images[index as usize], ImageLayout::PresentSrc);

            context
                .device
                .reset_fences(&[window.fences[index as usize]])
                .unwrap();
            *self.image_acquired.lock().unwrap() = (
                true,
                window.current_semaphore(),
                window.swapchain,
                index,
                window.fences[index as usize],
            );
        }
    }

    /// # Panics
    /// Will panic if ran out of host or device memory
    pub fn wait_idle(&self, context: &Context) {
        unsafe {
            context.device.queue_wait_idle(self.queue).unwrap();
        }
    }

    fn submit_private(
        &self,
        context: &Context,
        waits: &[Timestamp],
        func: impl FnOnce(filling_span::AnyCapabilityOutsidePass<CAPABILITIES>),
    ) -> Timestamp {
        unsafe {
            context.deletion_queue.lock().unwrap().flush(context);
            self.make_finished_pools_available(context);

            let command_pool = self.acquire_pool(context);
            context
                .device
                .begin_command_buffer(
                    command_pool.buffer,
                    &vk::CommandBufferBeginInfo::default()
                        .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                )
                .unwrap();
            if CAPABILITIES == crate::GRAPHICS_CAPABILITY {
                context.device.cmd_bind_descriptor_sets(
                    command_pool.buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    context.pipeline_layout,
                    0,
                    &[context.descriptor_set],
                    &[],
                );
            }
            if CAPABILITIES != crate::TRANSFER_CAPABILITY {
                context.device.cmd_bind_descriptor_sets(
                    command_pool.buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    context.pipeline_layout,
                    0,
                    &[context.descriptor_set],
                    &[],
                );
            }
            func(filling_span::AnyCapabilityOutsidePass {
                context,
                cmd: command_pool.buffer,
            });
            context
                .device
                .end_command_buffer(command_pool.buffer)
                .unwrap();

            context
                .device
                .queue_submit2(
                    self.queue,
                    &[vk::SubmitInfo2::default()
                        .wait_semaphore_infos(
                            waits
                                .iter()
                                .map(|w| {
                                    vk::SemaphoreSubmitInfo::default()
                                        .semaphore(w.get_fence(context).timeline_semaphore)
                                        .value(w.value)
                                        .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                                })
                                .collect::<Vec<_>>()
                                .as_slice(),
                        )
                        .command_buffer_infos(&[vk::CommandBufferSubmitInfo::default()
                            .command_buffer(command_pool.buffer)])
                        .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                            .semaphore(self.fence.timeline_semaphore)
                            .value(context.advance_timeline())
                            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                    vk::Fence::null(),
                )
                .unwrap();

            let (image_acquired, semaphore, swapchain, index, fence) =
                *self.image_acquired.lock().unwrap();
            if image_acquired {
                // convert timeline value to binary
                context
                    .device
                    .queue_submit2(
                        self.queue,
                        &[vk::SubmitInfo2::default()
                            .wait_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                                .semaphore(self.fence.timeline_semaphore)
                                .value(context.timeline_value.load(Ordering::SeqCst))
                                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])
                            .signal_semaphore_infos(&[vk::SemaphoreSubmitInfo::default()
                                .semaphore(semaphore)
                                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)])],
                        vk::Fence::null(),
                    )
                    .unwrap();
                context
                    .swapchain_khr
                    .queue_present(
                        self.queue,
                        &vk::PresentInfoKHR::default()
                            .wait_semaphores(&[semaphore])
                            .swapchains(&[swapchain])
                            .image_indices(&[index])
                            .push_next(
                                &mut vk::SwapchainPresentFenceInfoEXT::default().fences(&[fence]),
                            ),
                    )
                    .unwrap();
                self.image_acquired.lock().unwrap().0 = false;
            }

            // Add pool to pending
            let timeline_value = context.timeline_value.load(Ordering::SeqCst);
            self.pending_pools
                .lock()
                .unwrap()
                .push((command_pool, timeline_value));
            self.submitted_since_clear.store(true, Ordering::SeqCst);
            Timestamp {
                value: timeline_value,
                ty: Some(CAPABILITIES),
            }
        }
    }

    fn make_finished_pools_available(&self, context: &Context) {
        self.pending_pools
            .lock()
            .unwrap()
            .retain(|(pool, pending_value)| {
                let available = *pending_value <= self.fence.current_value(context);
                if available {
                    self.available_pools.lock().unwrap().push(*pool);
                }
                !available
            });
    }

    fn acquire_pool(&self, context: &Context) -> CommandPool {
        unsafe {
            if let Some(pool) = self.available_pools.lock().unwrap().pop() {
                context
                    .device
                    .reset_command_pool(pool.pool, vk::CommandPoolResetFlags::empty())
                    .unwrap();
                pool
            } else {
                let pool = context
                    .device
                    .create_command_pool(
                        &vk::CommandPoolCreateInfo::default().queue_family_index(self.family),
                        None,
                    )
                    .unwrap();
                let buffer = context
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )
                    .unwrap()[0];

                info!(
                    "cobra::queue: Created command pool in queue with capability {}",
                    CAPABILITIES
                );
                CommandPool { pool, buffer }
            }
        }
    }
}

impl GraphicsQueue {
    pub fn submit(
        &self,
        context: &Context,
        waits: &[Timestamp],
        func: impl FnOnce(filling_span::GraphicsOnlyOutsidePass),
    ) -> Timestamp {
        self.submit_private(context, waits, func)
    }
}
