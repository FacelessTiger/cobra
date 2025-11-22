# `cobra-bindless-macro`

Contains a `bindless` procedural macro, meant to be used in shaders to put all the necessary bindings in and
contain it in a `Descriptors` variable

# Usage

Just use shaders like usual and put the procedural macro before the `spirv` macro like so:

```rust
#[bindless(descriptors)]
#[spirv(vertex)]
pub fn main_vs(
    #[spirv(push_constant)] push: &DevicePointer<Vertex>,
    #[spirv(vertex_index)] vertex_id: usize,

    color: &mut glam::Vec4,
    uv: &mut glam::Vec2,

    #[spirv(position)] pos: &mut glam::Vec4,
) {
    let vertex = unsafe { push.vertices.validate(&descriptors).add(vertex_id).deref() };

    // ...
}
```

`descriptors` is just the name for the variable that holds the bindings, can be named whatever you wish.