use crate::Descriptors;
use core::marker::PhantomData;
use spirv_std::float::Float;
use spirv_std::image::{
    Arrayed, Dimensionality, ImageCoordinate, ImageDepth, ImageFormat, Multisampled, Sampled,
};
use spirv_std::integer::Integer;
use spirv_std::{RuntimeArray, Sampler};

const FORMAT: u32 = ImageFormat::Unknown as u32;
const DIM: u32 = Dimensionality::TwoD as u32;
const ARRAYED: u32 = Arrayed::False as u32;
const COMPONENTS: u32 = 4;

pub trait SampleType: spirv_std::image::SampleType<FORMAT, COMPONENTS> {}
impl<T: spirv_std::image::SampleType<FORMAT, COMPONENTS>> SampleType for T {}

pub struct ImageHandle<T: SampleType> {
    handle: u32,
    _phantom: PhantomData<T>,
}

impl<T: SampleType> ImageHandle<T> {
    pub fn new_storage(handle: u32) -> Self {
        ImageHandle {
            handle,
            _phantom: PhantomData,
        }
    }

    pub fn new_sampled(image_handle: u32, sampler_handle: u32) -> Self {
        ImageHandle {
            handle: image_handle | (sampler_handle << 20),
            _phantom: PhantomData,
        }
    }

    pub fn as_raw(&self) -> u32 {
        self.handle
    }

    fn validate_helper<'a>(
        &self,
        descriptors: &'a Descriptors,
        images: &'a RuntimeArray<ImageReference<T, { Sampled::Yes as u32 }>>,
    ) -> Image<'a, T> {
        unsafe {
            Image {
                image: images.index((self.handle & 0xfffff) as usize),
                sampler: *descriptors.samplers.index((self.handle << 20) as usize),
            }
        }
    }

    fn validate_helper_storage<'a>(
        &self,
        images: &'a RuntimeArray<ImageReference<T, { Sampled::No as u32 }>>,
    ) -> StorageImage<'a, T> {
        unsafe {
            StorageImage {
                image: images.index((self.handle & 0xfffff) as usize),
            }
        }
    }
}

impl ImageHandle<f32> {
    pub fn validate<'a>(&self, descriptors: &'a Descriptors) -> Image<'a, f32> {
        self.validate_helper(descriptors, descriptors.images_f32)
    }
    pub fn validate_storage<'a>(&self, descriptors: &'a Descriptors) -> StorageImage<'a, f32> {
        self.validate_helper_storage(descriptors.storage_images_f32)
    }
}
impl ImageHandle<u32> {
    pub fn validate<'a>(&self, descriptors: &'a Descriptors) -> Image<'a, u32> {
        self.validate_helper(descriptors, descriptors.images_u32)
    }
}

type ImageReference<T, const SAMPLED: u32> = spirv_std::image::Image<
    T,
    DIM,
    { ImageDepth::Unknown as u32 },
    ARRAYED,
    { Multisampled::False as u32 },
    SAMPLED,
    FORMAT,
    COMPONENTS,
>;

pub struct Image<'a, T: SampleType> {
    image: &'a ImageReference<T, { Sampled::Yes as u32 }>,
    sampler: Sampler,
}

impl<T: SampleType> Image<'_, T> {
    pub fn sample<F: Float>(
        &self,
        coords: impl ImageCoordinate<F, DIM, ARRAYED>,
    ) -> T::SampleResult {
        self.image.sample(self.sampler, coords)
    }

    pub fn sample_by_lod<F: Float>(
        &self,
        coords: impl ImageCoordinate<F, DIM, ARRAYED>,
        lod: f32,
    ) -> T::SampleResult {
        self.image.sample_by_lod(self.sampler, coords, lod)
    }
}

pub struct StorageImage<'a, T: SampleType> {
    image: &'a ImageReference<T, { Sampled::No as u32 }>,
}

impl<T: SampleType> StorageImage<'_, T> {
    pub fn read<I: Integer>(
        &self,
        coords: impl ImageCoordinate<I, DIM, ARRAYED>,
    ) -> T::SampleResult {
        self.image.read(coords)
    }
}
