#![no_std]

use spirv_std::image::{Image2d, Image2dU, StorageImage2d};
use spirv_std::{RuntimeArray, TypedBuffer};

pub mod device_pointer;
pub mod image_handle;

pub use cobra_bindless_macro::bindless;
pub use device_pointer::{DevicePointer, PhysicalPointer, PointerType};
pub use image_handle::{Image, ImageHandle, StorageImage};

pub struct Descriptors<'a> {
    pub buffers: &'a RuntimeArray<TypedBuffer<[u32]>>,
    pub images_f32: &'a RuntimeArray<Image2d>,
    pub images_u32: &'a RuntimeArray<Image2dU>,
    pub storage_images_f32: &'a RuntimeArray<StorageImage2d>,
    pub samplers: &'a RuntimeArray<spirv_std::Sampler>,
}
