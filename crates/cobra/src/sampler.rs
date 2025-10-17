use crate::Context;
use ash::vk;
use std::sync::Arc;

pub struct Sampler {
    context: Arc<Context>,
    sampler: vk::Sampler,

    pub(crate) handle: u32,
}

impl Sampler {
    /// # Panics
    /// If out of device or host memory
    pub fn new(context: Arc<Context>) -> Self {
        unsafe {
            let sampler = context
                .device
                .create_sampler(
                    &vk::SamplerCreateInfo::default()
                        .mag_filter(vk::Filter::LINEAR)
                        .min_filter(vk::Filter::LINEAR),
                    None,
                )
                .unwrap();

            let handle = context.sampler_handle.acquire();
            context.device.update_descriptor_sets(
                &[vk::WriteDescriptorSet::default()
                    .dst_set(context.descriptor_set)
                    .dst_binding(crate::SAMPLER_BINDING)
                    .dst_array_element(handle)
                    .descriptor_type(vk::DescriptorType::SAMPLER)
                    .image_info(&[vk::DescriptorImageInfo::default().sampler(sampler)])],
                &[],
            );

            Sampler {
                context,
                sampler,
                handle,
            }
        }
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.context
            .deletion_queue
            .lock()
            .unwrap()
            .push_sampler(&self.context, self.sampler);
        self.context.sampler_handle.recycle(self.handle);
    }
}
