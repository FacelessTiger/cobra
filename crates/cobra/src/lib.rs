use ash::vk;
use thiserror::Error;

pub(crate) mod bindless_shader;
pub(crate) mod buffer;
pub(crate) mod context;
/// Acts like the command buffer, using separate types to statically disallow invalid use
/// (e.g. trying to start a render pass in a compute queue)
pub mod filling_span;
pub(crate) mod image;
pub(crate) mod queue;
pub(crate) mod sampler;
pub(crate) mod swapchain;

pub use bindless_shader::*;
pub use buffer::Buffer;
pub use cobra_bindless::*;
pub use context::{ComputePipeline, Context, GraphicsPipeline};
pub use image::Image;
pub use queue::{ComputeQueue, GraphicsQueue, Queue, Timestamp, TransferQueue};
pub use sampler::Sampler;
pub use swapchain::Swapchain;

pub(crate) const GRAPHICS_CAPABILITY: u32 = 0;
pub(crate) const TRANSFER_CAPABILITY: u32 = 1;
pub(crate) const COMPUTE_CAPABILITY: u32 = 2;

pub(crate) const STORAGE_BINDING: u32 = 0;
pub(crate) const SAMPLED_IMAGE_BINDING: u32 = 1;
pub(crate) const STORAGE_IMAGE_BINDING: u32 = 2;
pub(crate) const SAMPLER_BINDING: u32 = 3;

#[derive(Debug, Error)]
pub enum CobraInitError {
    #[error("Creating Vulkan instance failed")]
    InstanceInitialization,
    #[error("No available GPUs that support Vulkan")]
    NoAvailableGpus,
    #[error("No graphics queue available")]
    NoGraphicsQueueAvailable,
    #[error("Creating vulkan device failed")]
    DeviceCreation,
}

#[derive(Debug, Clone, Copy)]
pub enum ClearValue {
    Vec4([f32; 4]),
    IVec4([i32; 4]),
    UVec4([u32; 4]),
}

impl From<[f32; 4]> for ClearValue {
    fn from(value: [f32; 4]) -> Self {
        ClearValue::Vec4(value)
    }
}

impl From<[i32; 4]> for ClearValue {
    fn from(value: [i32; 4]) -> Self {
        ClearValue::IVec4(value)
    }
}

impl From<[u32; 4]> for ClearValue {
    fn from(value: [u32; 4]) -> Self {
        ClearValue::UVec4(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HeapType {
    /// Device local heap, not able to access buffer directly from host
    Default,
    /// Host visible and host cached, good for readback from device and uploads. Shouldn't be used
    /// directly in shaders but rather staged into [`HeapType::Default`]
    HostVisible,
    /// Device local but host mapped, reading from this heap type is
    /// extremely slow, so the only operation enabled is [`Buffer::write`]. Unlike
    /// [`HeapType::HostVisible`], using a buffer with this heap type directly in shaders is optimal
    DeviceUpload,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormat {
    Undefined = 0,
    Rgba8Unorm,
}

impl ImageFormat {
    pub fn block_size(&self) -> u32 {
        match self {
            ImageFormat::Undefined => 0,
            ImageFormat::Rgba8Unorm => 4,
        }
    }
}

impl From<vk::Format> for ImageFormat {
    fn from(format: vk::Format) -> Self {
        match format {
            vk::Format::UNDEFINED => ImageFormat::Undefined,
            vk::Format::R8G8B8A8_UNORM => ImageFormat::Rgba8Unorm,
            _ => todo!(),
        }
    }
}

impl From<ImageFormat> for vk::Format {
    fn from(format: ImageFormat) -> Self {
        match format {
            ImageFormat::Undefined => vk::Format::UNDEFINED,
            ImageFormat::Rgba8Unorm => vk::Format::R8G8B8A8_UNORM,
        }
    }
}

// TODO: get rid of this from the public API and build off of filling span to make a render graph
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageLayout {
    Undefined = 0,
    ReadOnly,
    Attachment,
    TransferSrc,
    TransferDst,
    PresentSrc,
}

impl From<ImageLayout> for vk::ImageLayout {
    fn from(layout: ImageLayout) -> Self {
        match layout {
            ImageLayout::Undefined => vk::ImageLayout::UNDEFINED,
            ImageLayout::ReadOnly => vk::ImageLayout::READ_ONLY_OPTIMAL,
            ImageLayout::Attachment => vk::ImageLayout::ATTACHMENT_OPTIMAL,
            ImageLayout::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            ImageLayout::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            ImageLayout::PresentSrc => vk::ImageLayout::PRESENT_SRC_KHR,
        }
    }
}
bitflags::bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct ImageUsage: u32 {
        const Sampled = vk::ImageUsageFlags::SAMPLED.as_raw();
        const Storage = vk::ImageUsageFlags::STORAGE.as_raw();
        const TransferSrc = vk::ImageUsageFlags::TRANSFER_SRC.as_raw();
        const TransferDst = vk::ImageUsageFlags::TRANSFER_DST.as_raw();
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        vk::ImageUsageFlags::from_raw(usage.bits())
    }
}
