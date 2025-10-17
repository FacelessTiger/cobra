use std::marker::PhantomData;
use std::prelude::v1::Vec;
use std::sync::Arc;

use crate::image::Image;
use crate::Context;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub trait WindowHandle: HasWindowHandle + HasDisplayHandle {}
impl<T: HasWindowHandle + HasDisplayHandle> WindowHandle for T {}
impl<'a, T: WindowHandle + 'a> From<T> for SwapchainTarget<'a> {
    fn from(value: T) -> Self {
        SwapchainTarget::Window(Box::new(value))
    }
}

pub enum SwapchainTarget<'a> {
    Window(Box<dyn WindowHandle + 'a>),
}

pub struct Swapchain<'a> {
    context: Arc<Context>,
    pub(crate) swapchain: vk::SwapchainKHR,
    surface: vk::SurfaceKHR,

    pub(crate) images: Vec<Image>,
    pub(crate) fences: Vec<vk::Fence>, // Size equal to images.len()
    semaphores: Vec<vk::Semaphore>,    // Size equal to 2 * images.len()
    semaphore_index: usize,
    pub(crate) image_index: u32,

    _phantom: PhantomData<&'a ()>,
}

impl<'a> Swapchain<'a> {
    /// # Panics
    /// Will panic if device or host run out of memory
    pub fn new(
        context: Arc<Context>,
        window: impl Into<SwapchainTarget<'a>>,
        size: impl Into<[u32; 2]>,
    ) -> Self {
        unsafe {
            let size = size.into();
            let SwapchainTarget::Window(window) = window.into();

            let surface = ash_window::create_surface(
                &context.entry,
                &context.instance,
                window.display_handle().unwrap().into(),
                window.window_handle().unwrap().into(),
                None,
            )
            .unwrap();
            let (swapchain, images, semaphores) =
                Self::create_swapchain(&context, surface, size, None);
            let fences = images
                .iter()
                .map(|_| {
                    context
                        .device
                        .create_fence(&vk::FenceCreateInfo::default(), None)
                        .unwrap()
                })
                .collect::<Vec<_>>();

            Swapchain {
                context,
                swapchain,
                surface,
                images,
                fences,
                semaphores,
                semaphore_index: 0,
                image_index: 0,
                _phantom: PhantomData,
            }
        }
    }

    pub fn resize(&mut self, size: impl Into<[u32; 2]>) {
        let size = size.into();
        if size == self.size() {
            return;
        }

        let (swapchain, images, mut semaphores) =
            Self::create_swapchain(&self.context, self.surface, size, Some(self.swapchain));
        std::mem::swap(&mut self.semaphores, &mut semaphores);

        #[expect(clippy::missing_panics_doc, reason = "infallible")]
        self.context.deletion_queue.lock().unwrap().push_swapchain((
            self.swapchain,
            semaphores,
            self.fences[self.image_index as usize],
        ));
        self.swapchain = swapchain;
        self.images = images;
    }

    pub fn size(&self) -> [u32; 2] {
        self.images[0].size()
    }

    pub fn current_semaphore(&self) -> vk::Semaphore {
        self.semaphores[self.semaphore_index]
    }
    pub fn increment_semaphore_index(&mut self) {
        self.semaphore_index = (self.semaphore_index + 1) % self.semaphores.len();
    }

    fn create_swapchain(
        context: &Arc<Context>,
        surface: vk::SurfaceKHR,
        size: [u32; 2],
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> (vk::SwapchainKHR, Vec<Image>, Vec<vk::Semaphore>) {
        unsafe {
            let capabilities = context
                .surface_khr
                .get_physical_device_surface_capabilities(context.gpu, surface)
                .unwrap();
            let swapchain = context
                .swapchain_khr
                .create_swapchain(
                    &vk::SwapchainCreateInfoKHR::default()
                        .surface(surface)
                        .min_image_count(capabilities.min_image_count + 1)
                        .image_format(vk::Format::R8G8B8A8_UNORM)
                        .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
                        .image_extent(vk::Extent2D {
                            width: size[0],
                            height: size[1],
                        })
                        .image_array_layers(1)
                        .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                        .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                        .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                        .present_mode(vk::PresentModeKHR::MAILBOX)
                        .old_swapchain(old_swapchain.unwrap_or(vk::SwapchainKHR::null()))
                        .push_next(
                            &mut vk::SwapchainPresentScalingCreateInfoEXT::default()
                                .scaling_behavior(vk::PresentScalingFlagsEXT::STRETCH)
                                .present_gravity_x(vk::PresentGravityFlagsEXT::CENTERED)
                                .present_gravity_y(vk::PresentGravityFlagsEXT::CENTERED),
                        ),
                    None,
                )
                .unwrap();

            let images = context
                .swapchain_khr
                .get_swapchain_images(swapchain)
                .unwrap()
                .iter()
                .map(|image| {
                    Image::new_swapchain(context.clone(), size, vk::Format::R8G8B8A8_UNORM, *image)
                })
                .collect::<Vec<_>>();
            let semaphores = images
                .iter()
                .flat_map(|_| {
                    [(); 2].map(|()| {
                        context
                            .device
                            .create_semaphore(&vk::SemaphoreCreateInfo::default(), None)
                            .unwrap()
                    })
                })
                .collect::<Vec<_>>();
            (swapchain, images, semaphores)
        }
    }
}

impl Drop for Swapchain<'_> {
    fn drop(&mut self) {
        unsafe {
            // TODO: maybe try to do this without wait idle, main issue is not allowed to destroy surface til swapchain is destroyed
            self.context.device.device_wait_idle().unwrap(); // This does cover present because we're using swapchain maintenance
            self.context
                .swapchain_khr
                .destroy_swapchain(self.swapchain, None);
            self.context.surface_khr.destroy_surface(self.surface, None);

            for fence in &self.fences {
                self.context.device.destroy_fence(*fence, None);
            }

            for semaphore in &self.semaphores {
                self.context.device.destroy_semaphore(*semaphore, None);
            }
        }
    }
}
