#![no_std]

use cobra_bindless::{bindless, DevicePointer, ImageHandle, PointerType};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::spirv;

pub struct Vertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: u32,
}

pub struct PushConstant {
    pub vertices: DevicePointer<Vertex>,
    pub image: ImageHandle<f32>,
    pub target_size: [f32; 2],
}

pub fn linear_from_srgb(srgb: glam::Vec3) -> glam::Vec3 {
    srgb.map(|v| {
        if v <= 0.04045 {
            v / 12.92
        } else {
            ((v + 0.055) / 1.055).powf(2.4)
        }
    })
}

#[bindless(descriptors)]
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(push_constant)] push: &PushConstant,
    #[spirv(vertex_index)] vertex_id: usize,

    color: &mut glam::Vec4,
    uv: &mut glam::Vec2,

    #[spirv(position)] pos: &mut glam::Vec4,
) {
    let vertex = unsafe { push.vertices.validate(&descriptors).add(vertex_id).deref() };
    *color = spirv_std::float::u8x4_to_vec4_unorm(vertex.color);
    *uv = vertex.uv.into();

    // We have to normalize the coordinates because they're in screen coords for w/e reason
    *pos = glam::vec4(
        ((vertex.position[0] / push.target_size[0]) - 0.5) * 2.0,
        ((vertex.position[1] / push.target_size[1]) - 0.5) * 2.0,
        0.0,
        1.0,
    );
}

#[bindless(descriptors)]
#[inline(never)]
#[spirv(fragment)]
pub fn main_fs(
    #[spirv(push_constant)] push: &PushConstant,

    color: glam::Vec4,
    uv: glam::Vec2,

    output: &mut glam::Vec4,
) {
    let image = push.image.validate(&descriptors);
    *output = color * image.sample(uv);
    //*output = linear_from_gamma_rgb(color.xyz()).extend(color.w);
}
