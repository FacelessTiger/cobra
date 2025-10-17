use crate::buffer::Buffer;
use crate::context::{GraphicsPipeline, subresource_range};
use crate::image::Image;
use crate::{ClearValue, ComputePipeline, Context, ImageLayout};
use ash::vk;
use std::prelude::v1::Vec;

// Acts as the command buffer, using constants to statically disallow
// improper use (such as trying to begin a render pass in a compute queue)
#[doc(hidden)]
pub struct FillingSpan<'a, const RENDERING: bool, const CAPABILITIES: u32> {
    pub(crate) context: &'a Context,
    pub(crate) cmd: vk::CommandBuffer,
}

/// Commands that can only be used on a graphics queue, whether inside
/// a render pass or not
pub type GraphicsOnly<'a, const R: bool> = FillingSpan<'a, R, { crate::GRAPHICS_CAPABILITY }>;
/// Commands that can only be used on a compute *or* graphics queue, whether inside
/// a render pass or not
pub type ComputeOnly<'a, const R: bool> = FillingSpan<'a, R, { crate::COMPUTE_CAPABILITY }>;
/// Commands that can be used on a queue of any type, whether inside a render pass
/// or not
pub type AnyCapability<'a, const R: bool, const C: u32> = FillingSpan<'a, R, C>;
/// Same as [`GraphicsOnly`], but covers commands that only work *outside* a render pass
pub type GraphicsOnlyOutsidePass<'a> = GraphicsOnly<'a, false>;
/// Same as [`GraphicsOnly`], but covers commands that only work *inside* a render pass
pub type GraphicsOnlyInsidePass<'a> = GraphicsOnly<'a, true>;
/// Same as [`AnyCapability`], but covers commands that only work *outside* a render pass.
/// Note that you can only start a render pass on a graphics queue, so this only really restricts
/// those type of queues
pub type AnyCapabilityOutsidePass<'a, const C: u32> = FillingSpan<'a, false, C>;

#[doc(hidden)]
trait SpanGet<'a> {
    fn context(&self) -> &'a Context;
    fn cmd(&self) -> vk::CommandBuffer;
}

impl<'a, const R: bool, const C: u32> SpanGet<'a> for FillingSpan<'a, R, C> {
    fn context(&self) -> &'a Context {
        self.context
    }
    fn cmd(&self) -> vk::CommandBuffer {
        self.cmd
    }
}

impl<const R: bool> GraphicsOnly<'_, R> {
    pub fn set_viewport(&self, size: impl Into<[f32; 2]>, min_depth: f32, max_depth: f32) {
        unsafe {
            let size = size.into();

            self.context.device.cmd_set_viewport(
                self.cmd,
                0,
                &[vk::Viewport::default()
                    .width(size[0])
                    .height(size[1])
                    .min_depth(min_depth)
                    .max_depth(max_depth)],
            );
        }
    }

    pub fn set_scissor(&self, size: impl Into<[u32; 2]>, offset: impl Into<[i32; 2]>) {
        unsafe {
            let size = size.into();
            let offset = offset.into();

            self.context.device.cmd_set_scissor(
                self.cmd,
                0,
                &[vk::Rect2D::default()
                    .extent(vk::Extent2D {
                        width: size[0],
                        height: size[1],
                    })
                    .offset(vk::Offset2D {
                        x: offset[0],
                        y: offset[1],
                    })],
            );
        }
    }

    pub fn bind_graphics_pipeline(&self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.context.device.cmd_bind_pipeline(
                self.cmd,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
        }
    }

    pub fn bind_index_buffer_u32(&self, buffer: &Buffer<u32>, offset: u64) {
        self.bind_index_buffer(buffer.buffer, offset, vk::IndexType::UINT32);
    }
    pub fn bind_index_buffer_u16(&self, buffer: &Buffer<u16>, offset: u64) {
        self.bind_index_buffer(buffer.buffer, offset, vk::IndexType::UINT16);
    }
    fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64, index_type: vk::IndexType) {
        unsafe {
            self.context
                .device
                .cmd_bind_index_buffer(self.cmd, buffer, offset, index_type);
        }
    }
}

#[doc(hidden)]
struct ColorAttachment<'a> {
    image: &'a Image,
    clear: Option<ClearValue>,
}

#[doc(hidden)]
enum DepthAttachment<'a> {
    Some(&'a Image, Option<f32>),
    None,
}

#[allow(clippy::from_over_into)]
impl<'a> Into<ColorAttachment<'a>> for &'a Image {
    fn into(self) -> ColorAttachment<'a> {
        ColorAttachment {
            image: self,
            clear: None,
        }
    }
}

#[allow(clippy::from_over_into)]
impl<'a, C: Into<[f32; 4]>> Into<ColorAttachment<'a>> for (&'a Image, C) {
    fn into(self) -> ColorAttachment<'a> {
        ColorAttachment {
            image: self.0,
            clear: Some(ClearValue::from(self.1.into())),
        }
    }
}

#[allow(clippy::from_over_into)]
impl<'a> Into<DepthAttachment<'a>> for (&'a Image, f32) {
    fn into(self) -> DepthAttachment<'a> {
        DepthAttachment::Some(self.0, Some(self.1))
    }
}

#[allow(clippy::from_over_into)]
impl<'a> Into<DepthAttachment<'a>> for &'a Image {
    fn into(self) -> DepthAttachment<'a> {
        DepthAttachment::Some(self, None)
    }
}

#[allow(clippy::from_over_into)]
impl<'a> Into<DepthAttachment<'a>> for Option<()> {
    fn into(self) -> DepthAttachment<'a> {
        DepthAttachment::None
    }
}

impl GraphicsOnlyOutsidePass<'_> {
    #[allow(private_bounds)]
    pub fn render<'a>(
        &mut self,
        extent: impl Into<[u32; 2]>,
        color_attachments: &[impl Into<ColorAttachment<'a>> + Clone],
        depth_attachment: impl Into<DepthAttachment<'a>>,
        func: impl FnOnce(GraphicsOnlyInsidePass),
    ) {
        fn render_attachment<T: Copy>(
            image: &'_ Image,
            clear_value: Option<T>,
            func: impl FnOnce(T) -> vk::ClearValue,
        ) -> vk::RenderingAttachmentInfo<'_> {
            vk::RenderingAttachmentInfo::default()
                .image_view(image.image_view)
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .load_op(
                    clear_value.map_or(vk::AttachmentLoadOp::LOAD, |_| vk::AttachmentLoadOp::CLEAR),
                )
                .store_op(vk::AttachmentStoreOp::STORE)
                .clear_value(clear_value.map_or(vk::ClearValue::default(), func))
        }

        let color_attachments = color_attachments
            .iter()
            .map(|attachment| (*attachment).clone().into());
        let depth_attachment = depth_attachment.into();

        unsafe {
            let extent = extent.into();
            let color_attachments = color_attachments
                .map(|attachment| {
                    self.transition_image(attachment.image, ImageLayout::Attachment);
                    render_attachment(attachment.image, attachment.clear, clear_value_to_vulkan)
                })
                .collect::<Vec<_>>();
            let depth_attachment = match depth_attachment {
                DepthAttachment::Some(image, depth) => {
                    self.transition_image(image, ImageLayout::Attachment);
                    render_attachment(image, depth, depth_clear_value_to_vulkan)
                }
                DepthAttachment::None => vk::RenderingAttachmentInfo::default(),
            };

            self.context.device.cmd_begin_rendering(
                self.cmd,
                &vk::RenderingInfo::default()
                    .render_area(vk::Rect2D {
                        offset: vk::Offset2D::default(),
                        extent: vk::Extent2D {
                            width: extent[0],
                            height: extent[1],
                        },
                    })
                    .layer_count(1)
                    .color_attachments(color_attachments.as_slice())
                    .depth_attachment(&depth_attachment),
            );
            func(FillingSpan {
                context: self.context,
                cmd: self.cmd,
            });
            self.context.device.cmd_end_rendering(self.cmd);
        }
    }
}

impl GraphicsOnlyInsidePass<'_> {
    pub fn clear_color_attachment(
        &self,
        index: u32,
        clear_value: impl Into<ClearValue>,
        size: impl Into<[u32; 2]>,
    ) {
        unsafe {
            let size = size.into();
            self.context.device.cmd_clear_attachments(
                self.cmd,
                &[vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .color_attachment(index)
                    .clear_value(clear_value_to_vulkan(clear_value.into()))],
                &[vk::ClearRect::default()
                    .rect(vk::Rect2D::default().extent(vk::Extent2D {
                        width: size[0],
                        height: size[1],
                    }))
                    .layer_count(1)],
            );
        }
    }

    pub fn clear_depth_attachment(&self, depth: f32, size: impl Into<[u32; 2]>) {
        unsafe {
            let size = size.into();

            self.context.device.cmd_clear_attachments(
                self.cmd,
                &[vk::ClearAttachment::default()
                    .aspect_mask(vk::ImageAspectFlags::DEPTH)
                    .clear_value(depth_clear_value_to_vulkan(depth))],
                &[vk::ClearRect::default()
                    .rect(vk::Rect2D::default().extent(vk::Extent2D {
                        width: size[0],
                        height: size[1],
                    }))
                    .layer_count(1)],
            );
        }
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw(
                self.cmd,
                vertex_count,
                instance_count,
                first_vertex,
                first_instance,
            );
        }
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        unsafe {
            self.context.device.cmd_draw_indexed(
                self.cmd,
                index_count,
                instance_count,
                first_index,
                vertex_offset,
                first_instance,
            );
        }
    }
}

impl<const C: u32> AnyCapabilityOutsidePass<'_, C> {
    pub fn copy_image_to_buffer<T>(&self, src: &Image, dst: &Buffer<T>) {
        unsafe {
            self.transition_image(src, ImageLayout::TransferSrc);
            self.context.device.cmd_copy_image_to_buffer2(
                self.cmd,
                &vk::CopyImageToBufferInfo2::default()
                    .src_image(src.image)
                    .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
                    .dst_buffer(dst.buffer)
                    .regions(&[whole_image_region(src.size, [0, 0])]),
            );
        }
    }

    pub fn copy_buffer_to_image<T>(
        &self,
        src: &Buffer<T>,
        dst: &Image,
        size: impl Into<[u32; 2]>,
        image_offset: impl Into<[i32; 2]>,
    ) {
        unsafe {
            self.transition_image(dst, ImageLayout::TransferDst);
            self.context.device.cmd_copy_buffer_to_image2(
                self.cmd,
                &vk::CopyBufferToImageInfo2::default()
                    .src_buffer(src.buffer)
                    .dst_image(dst.image)
                    .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .regions(&[whole_image_region(size.into(), image_offset.into())]),
            );
        }
    }

    /// Update's the given buffer in place, required that T has an alignment of 4
    /// and that the update data size is less than 65,536 bytes
    /// # Panics
    /// Under any of the following conditions
    /// * The data would be written out of bounds
    /// * Data's size in bytes is greater than 65,536
    /// * The alignment of the updated type isn't 4
    pub fn update_buffer<T>(&self, dst: &Buffer<T>, dst_offset: usize, data: &[T]) {
        assert!(dst_offset < dst.len());
        assert!(data.len() <= (dst.len() - dst_offset));
        assert!(size_of_val(data) < 65_536);
        assert_eq!(4, align_of::<T>());
        let dst_offset = dst_offset * size_of::<T>();

        unsafe {
            self.context.device.cmd_update_buffer(
                self.cmd,
                dst.buffer,
                dst_offset as u64,
                data.align_to().1,
            );
        }
    }
}

impl<const R: bool, const C: u32> AnyCapability<'_, R, C> {
    pub fn transition_image(&self, image: &Image, new_layout: ImageLayout) {
        unsafe {
            #[expect(clippy::missing_panics_doc, reason = "infallible")]
            let mut old_layout = image.layout.lock().unwrap();
            if new_layout == *old_layout {
                return;
            }

            // TODO: this is a global barrier, very bad
            self.context.device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().image_memory_barriers(&[
                    vk::ImageMemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .src_access_mask(
                            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                        )
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(
                            vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                        )
                        .old_layout((*old_layout).into())
                        .new_layout(new_layout.into())
                        .image(image.image)
                        .subresource_range(subresource_range(vk::ImageAspectFlags::COLOR)),
                ]),
            );
            *old_layout = new_layout;
        }
    }

    pub fn global_barrier(&self) {
        unsafe {
            self.context.device.cmd_pipeline_barrier2(
                self.cmd,
                &vk::DependencyInfo::default().memory_barriers(&[vk::MemoryBarrier2::default()
                    .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                    .dst_access_mask(
                        vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                    )]),
            );
        }
    }
}

// Really both compute only and compute + graphics only, since our graphics queues imply compute
#[allow(private_bounds)]
pub trait ComputeOnlyMethods<'a>: SpanGet<'a> {
    fn dispatch(&self, x: u32, y: u32, z: u32) {
        unsafe {
            self.context().device.cmd_dispatch(self.cmd(), x, y, z);
        }
    }

    fn bind_compute_pipeline(&self, pipeline: &ComputePipeline) {
        unsafe {
            self.context().device.cmd_bind_pipeline(
                self.cmd(),
                vk::PipelineBindPoint::COMPUTE,
                pipeline.pipeline,
            );
        }
    }

    fn push_constant<T>(&self, constant: &T) {
        unsafe {
            assert!(
                size_of::<T>() <= 128,
                "Push constants can only be up to 128 bytes"
            );

            self.context().device.cmd_push_constants(
                self.cmd(),
                self.context().pipeline_layout,
                vk::ShaderStageFlags::ALL,
                0,
                std::slice::from_raw_parts(std::ptr::from_ref(constant).cast(), size_of::<T>()),
            );
        }
    }
}

impl<'a, const R: bool> ComputeOnlyMethods<'a> for GraphicsOnly<'a, R> {}
impl<'a, const R: bool> ComputeOnlyMethods<'a> for ComputeOnly<'a, R> {}

#[doc(hidden)]
fn clear_value_to_vulkan(value: ClearValue) -> vk::ClearValue {
    vk::ClearValue {
        color: match value {
            ClearValue::Vec4(value) => vk::ClearColorValue { float32: value },
            ClearValue::IVec4(value) => vk::ClearColorValue { int32: value },
            ClearValue::UVec4(value) => vk::ClearColorValue { uint32: value },
        },
    }
}

#[doc(hidden)]
fn depth_clear_value_to_vulkan(depth: f32) -> vk::ClearValue {
    vk::ClearValue {
        depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
    }
}

#[doc(hidden)]
fn whole_image_region(size: [u32; 2], offset: [i32; 2]) -> vk::BufferImageCopy2<'static> {
    vk::BufferImageCopy2::default()
        .image_subresource(
            vk::ImageSubresourceLayers::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .layer_count(1),
        )
        .image_offset(vk::Offset3D {
            x: offset[0],
            y: offset[1],
            z: 0,
        })
        .image_extent(vk::Extent3D {
            width: size[0],
            height: size[1],
            depth: 1,
        })
}
