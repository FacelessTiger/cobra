use std::sync::{Arc, Mutex};

use ash::vk;
use vk_mem::Alloc;

use crate::context::subresource_range;
use crate::{Context, ImageFormat, ImageHandle, ImageLayout, ImageUsage, Sampler};
use cobra_bindless::image_handle::SampleType;

pub struct Image {
    context: Arc<Context>,

    pub(crate) image: vk::Image,
    allocation: Option<vk_mem::Allocation>,
    pub(crate) image_view: vk::ImageView,
    pub(crate) size: [u32; 2],

    format: ImageFormat,
    pub(crate) layout: Mutex<ImageLayout>, // TODO: prefer Atomic

    sampled_image_handle: Option<u32>,
    storage_image_handle: Option<u32>,
}

impl Image {
    /// # Panics
    /// Will panic if host or device run out of memory
    pub fn new(
        context: Arc<Context>,
        size: impl Into<[u32; 2]>,
        format: ImageFormat,
        usages: ImageUsage,
    ) -> Image {
        unsafe {
            let size = size.into();
            let vulkan_format = format.into();

            let (image, allocation) = context
                .allocator
                .create_image(
                    &vk::ImageCreateInfo::default()
                        .image_type(vk::ImageType::TYPE_2D)
                        .format(vulkan_format)
                        .extent(vk::Extent3D {
                            width: size[0],
                            height: size[1],
                            depth: 1,
                        })
                        .mip_levels(1)
                        .array_layers(1)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .usage(usages.into())
                        .initial_layout(vk::ImageLayout::UNDEFINED),
                    &vk_mem::AllocationCreateInfo {
                        usage: vk_mem::MemoryUsage::AutoPreferDevice,
                        ..Default::default()
                    },
                )
                .unwrap();
            let image_view = Self::create_image_view(&context, image, vulkan_format);

            let sampled_image_handle = if usages.contains(ImageUsage::Sampled) {
                Some(Self::update_descriptor_set(&context, image_view, true))
            } else {
                None
            };
            let storage_image_handle = if usages.contains(ImageUsage::Storage) {
                Some(Self::update_descriptor_set(&context, image_view, false))
            } else {
                None
            };
            Image {
                context,
                image,
                allocation: Some(allocation),
                image_view,
                size,
                format,
                layout: Mutex::new(ImageLayout::Undefined),
                sampled_image_handle,
                storage_image_handle,
            }
        }
    }

    pub(crate) fn new_swapchain(
        context: Arc<Context>,
        size: impl Into<[u32; 2]>,
        format: vk::Format,
        image: vk::Image,
    ) -> Image {
        let size = size.into();
        let image_view = Self::create_image_view(&context, image, format);

        Image {
            context,
            image,
            allocation: None,
            image_view,
            size,
            format: format.into(),
            layout: Mutex::new(ImageLayout::Undefined),
            sampled_image_handle: None,
            storage_image_handle: None,
        }
    }

    pub fn size(&self) -> [u32; 2] {
        self.size
    }
    pub fn width(&self) -> u32 {
        self.size[0]
    }
    pub fn height(&self) -> u32 {
        self.size[1]
    }
    pub fn format(&self) -> ImageFormat {
        self.format
    }

    pub fn block_size(format: vk::Format) -> u32 {
        match format {
            vk::Format::UNDEFINED => 0,
            vk::Format::R8G8B8A8_UNORM => 4,
            _ => unimplemented!(),
        }
    }

    pub fn byte_size(&self) -> usize {
        (self.width() * self.height() * self.format.block_size()) as usize
    }

    /// # Panics
    /// Will panic if used on an image made without SAMPLED usage
    pub fn sampled_handle<T: SampleType>(&self, sampler: &Sampler) -> ImageHandle<T> {
        ImageHandle::new_sampled(self.sampled_image_handle.unwrap(), sampler.handle)
    }

    /// # Panics
    /// Will panic if used on an image made without STORAGE usage
    pub fn storage_handle<T: SampleType>(&self) -> ImageHandle<T> {
        ImageHandle::new_storage(self.storage_image_handle.unwrap())
    }

    fn create_image_view(context: &Context, image: vk::Image, format: vk::Format) -> vk::ImageView {
        unsafe {
            context
                .device
                .create_image_view(
                    &vk::ImageViewCreateInfo::default()
                        .image(image)
                        .format(format)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR)),
                    None,
                )
                .unwrap()
        }
    }

    fn update_descriptor_set(context: &Context, view: vk::ImageView, sampled: bool) -> u32 {
        unsafe {
            let (binding, handle, ty) = if sampled {
                (
                    crate::SAMPLED_IMAGE_BINDING,
                    context.sampled_image_handle.acquire(),
                    vk::DescriptorType::SAMPLED_IMAGE,
                )
            } else {
                (
                    crate::STORAGE_IMAGE_BINDING,
                    context.storage_image_handle.acquire(),
                    vk::DescriptorType::STORAGE_IMAGE,
                )
            };

            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(context.descriptor_set)
                    .dst_binding(binding)
                    .dst_array_element(handle)
                    .descriptor_type(ty)
                    .image_info(&[vk::DescriptorImageInfo::default()
                        .image_view(view)
                        .image_layout(vk::ImageLayout::READ_ONLY_OPTIMAL)])],
                &[],
            );
            handle
        }
    }
}

impl Drop for Image {
    fn drop(&mut self) {
        let mut deletion_queue = self.context.deletion_queue.lock().unwrap();
        if let Some(allocation) = self.allocation {
            deletion_queue.push_image(self.context.as_ref(), (self.image, allocation));
        }
        if let Some(handle) = self.sampled_image_handle {
            self.context.sampled_image_handle.recycle(handle);
        }
        if let Some(handle) = self.storage_image_handle {
            self.context.storage_image_handle.recycle(handle);
        }

        deletion_queue.push_image_view(self.context.as_ref(), self.image_view);
    }
}
