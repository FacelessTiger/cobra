mod shader;

use cobra::filling_span::ComputeOnlyMethods;
use cobra::{HeapType, ImageFormat, ImageHandle, ImageLayout, ImageUsage, PointerType};
use egui::epaint;
use egui_cobra_shaders::{PushConstant, Vertex};
use epaint::emath::NumExt;
use epaint::TextureId;
use epaint::{ImageDelta, Primitive};
use std::num::NonZeroUsize;
use std::sync::Arc;

pub struct Renderer {
    vertex_buffers: Vec<cobra::Buffer<Vertex>>,
    index_buffers: Vec<cobra::Buffer<u32>>,
    pipeline: cobra::GraphicsPipeline,
    fif_count: usize,
    fif: usize,

    egui_texture: Option<cobra::Image>,
    egui_sampler: cobra::Sampler,
}

impl Renderer {
    pub fn new(context: &Arc<cobra::Context>, fif_count: usize) -> Self {
        const START_VERTEX_LEN: NonZeroUsize = NonZeroUsize::new(1024).unwrap();

        let mut vertex_buffers = Vec::with_capacity(fif_count);
        let mut index_buffers = Vec::with_capacity(fif_count);
        for _ in 0..fif_count {
            vertex_buffers.push(cobra::Buffer::new(
                context.clone(),
                START_VERTEX_LEN,
                HeapType::DeviceUpload,
            ));
            index_buffers.push(cobra::Buffer::new(
                context.clone(),
                START_VERTEX_LEN
                    .checked_mul(NonZeroUsize::new(3).unwrap())
                    .unwrap(),
                HeapType::DeviceUpload,
            ));
        }

        let pipeline = cobra::GraphicsPipeline::new(
            context.clone(),
            shader::main_vs::new(),
            shader::main_fs::new(),
            &[ImageFormat::Rgba8Unorm],
        );
        let egui_sampler = cobra::Sampler::new(context.clone());

        Renderer {
            vertex_buffers,
            index_buffers,
            pipeline,
            egui_sampler,
            fif_count,
            fif: 0,
            egui_texture: None,
        }
    }

    pub fn update_texture(
        &mut self,
        context: &Arc<cobra::Context>,
        queue: &cobra::GraphicsQueue,
        id: TextureId,
        image_delta: &ImageDelta,
    ) {
        assert_eq!(
            id,
            TextureId::Managed(0),
            "Trying to update non built in texture, invalid usage of backend"
        );

        let width = u32::try_from(image_delta.image.width()).unwrap();
        let height = u32::try_from(image_delta.image.height()).unwrap();
        let data = match &image_delta.image {
            egui::ImageData::Color(image) => image.pixels.as_slice(),
        };
        let buffer = cobra::Buffer::from_slice(context.clone(), false, data);

        let write_data_to_texture = |image, offset| {
            queue
                .submit(context, &[], |cmd| {
                    cmd.copy_buffer_to_image(&buffer, image, [width, height], offset);
                    cmd.transition_image(image, ImageLayout::ReadOnly);
                })
                .wait(context);
        };

        if let Some(texture) = &self.egui_texture {
            let pos = image_delta.pos.unwrap();
            write_data_to_texture(
                texture,
                [
                    i32::try_from(pos[0]).unwrap(),
                    i32::try_from(pos[1]).unwrap(),
                ],
            );
        } else {
            let texture = cobra::Image::new(
                context.clone(),
                [width, height],
                ImageFormat::Rgba8Unorm,
                ImageUsage::Sampled | ImageUsage::TransferDst,
            );
            write_data_to_texture(&texture, [0, 0]);
            self.egui_texture = Some(texture);
        }
    }

    pub fn update_buffers(
        &mut self,
        context: &Arc<cobra::Context>,
        paint_jobs: &[egui::ClippedPrimitive],
    ) {
        self.fif = (self.fif + 1) % self.fif_count;

        let (vertex_count, index_count) =
            paint_jobs.iter().fold((0, 0), |acc, clipped_primitive| {
                match &clipped_primitive.primitive {
                    Primitive::Mesh(mesh) => {
                        (acc.0 + mesh.vertices.len(), acc.1 + mesh.indices.len())
                    }
                    Primitive::Callback(_) => todo!(),
                }
            });

        let vertex_buffer = &mut self.vertex_buffers[self.fif];
        let index_buffer = &mut self.index_buffers[self.fif];

        if vertex_count > 0 {
            if vertex_buffer.len() < vertex_count {
                let len = (vertex_buffer.len() * 2).at_least(vertex_count);
                *vertex_buffer = cobra::Buffer::new(
                    context.clone(),
                    NonZeroUsize::new(len).unwrap(),
                    HeapType::DeviceUpload,
                );
            }

            // TODO: do staging once we have GPU only buffers
            let mut vertex_offset = 0;
            for egui::ClippedPrimitive { primitive, .. } in paint_jobs {
                match primitive {
                    Primitive::Mesh(mesh) => {
                        vertex_buffer.write(
                            vertex_offset,
                            &mesh
                                .vertices
                                .iter()
                                .map(|v| Vertex {
                                    position: [v.pos.x, v.pos.y],
                                    uv: [v.uv.x, v.uv.y],
                                    color: u32::from_ne_bytes(v.color.to_array()),
                                })
                                .collect::<Vec<_>>(),
                        );
                        vertex_offset += mesh.vertices.len();
                    }
                    Primitive::Callback(_) => (),
                }
            }
        }

        if index_count > 0 {
            if index_buffer.len() < index_count {
                let len = (index_buffer.len() * 2).at_least(index_count);
                *index_buffer = cobra::Buffer::new(
                    context.clone(),
                    NonZeroUsize::new(len).unwrap(),
                    HeapType::DeviceUpload,
                );
            }

            // TODO: do staging once we have GPU only buffers
            let mut index_offset = 0;
            for epaint::ClippedPrimitive { primitive, .. } in paint_jobs {
                match primitive {
                    Primitive::Mesh(mesh) => {
                        index_buffer.write(index_offset, &mesh.indices);
                        index_offset += mesh.indices.len();
                    }
                    Primitive::Callback(_) => (),
                }
            }
        }
    }

    pub fn render(
        &self,
        cmd: &cobra::filling_span::GraphicsOnlyInsidePass,
        paint_jobs: &[epaint::ClippedPrimitive],
        target_size: impl Into<[u32; 2]>,
    ) {
        let target_size = target_size.into();
        let target_size_float = [target_size[0] as f32, target_size[1] as f32];

        let mut index_buffer_offset = 0;
        let mut vertex_buffer_offset = 0;

        cmd.set_viewport(target_size_float, 0.0, 1.0);
        cmd.bind_graphics_pipeline(&self.pipeline);

        for egui::ClippedPrimitive {
            clip_rect,
            primitive,
        } in paint_jobs
        {
            let mesh = match primitive {
                Primitive::Mesh(mesh) => mesh,
                Primitive::Callback(_) => todo!(),
            };
            let scissor_rect = ScissorRect::new(clip_rect, target_size);
            let image = match mesh.texture_id {
                TextureId::Managed(0) => self
                    .egui_texture
                    .as_ref()
                    .unwrap()
                    .sampled_handle(&self.egui_sampler),
                TextureId::User(id) => unsafe {
                    std::mem::transmute::<u32, ImageHandle<f32>>(u32::try_from(id).unwrap())
                },
                TextureId::Managed(_) => {
                    panic!("Trying to update managed texture, invalid texture ID")
                }
            };

            cmd.set_scissor(scissor_rect.size, scissor_rect.offset);
            cmd.bind_index_buffer_u32(
                &self.index_buffers[self.fif],
                index_buffer_offset * (size_of::<u32>() as u64),
            );
            cmd.push_constant(&PushConstant {
                vertices: self.vertex_buffers[self.fif]
                    .device_address()
                    .offset(vertex_buffer_offset),
                image,
                target_size: target_size_float,
            });
            cmd.draw_indexed(u32::try_from(mesh.indices.len()).unwrap(), 1, 0, 0, 0);

            index_buffer_offset += mesh.indices.len() as u64;
            vertex_buffer_offset += isize::try_from(mesh.vertices.len()).unwrap();
        }
    }
}

struct ScissorRect {
    offset: [i32; 2],
    size: [u32; 2],
}

impl ScissorRect {
    #[allow(clippy::cast_sign_loss)]
    #[allow(clippy::cast_possible_truncation)]
    fn new(clip_rect: &epaint::Rect, target_size: [u32; 2]) -> Self {
        let clip_min = clip_rect.min.round();
        let clip_min = [clip_min.x as u32, clip_min.y as u32];

        let clip_max = clip_rect.max.round();
        let clip_max = [clip_max.x as u32, clip_max.y as u32];

        let clip_min = clip_min.clamp([0, 0], target_size);
        let clip_max = clip_max.clamp(clip_min, target_size);
        Self {
            offset: [clip_min[0] as i32, clip_min[1] as i32],
            size: [clip_max[0] - clip_min[0], clip_max[1] - clip_min[1]],
        }
    }
}

pub fn image_to_sized_texture(
    image: &cobra::Image,
    sampler: &cobra::Sampler,
    size: [f32; 2],
) -> egui::load::SizedTexture {
    egui::load::SizedTexture {
        id: TextureId::User(u64::from(image.sampled_handle::<f32>(sampler).as_raw())),
        size: size.into(),
    }
}
