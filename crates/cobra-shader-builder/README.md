# `cobra-shader-builder`

This crate utilizes [cargo-gpu](https://github.com/Rust-GPU/cargo-gpu) to enable you to build rust crates as spir-v and
utilize them in `cobra`. Namely, it encourages having one crate for host code, and another for all of your shader-side
code.

# Usage

This is intended to be used in a build script. For example if you have a crate named `host`, and another crate right
next it named `shader` with all the shader code, then in `host`'s `build.rs` you can insert

```rust
use cobra_shader_builder::ShaderBuilder;
use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    ShaderBuilder::new("shader")?.build()?;

    Ok(())
}
```

This will compile the shader code in `shader` to spir-v then make a corresponding ZST struct for each entry point that
implements the trait `BindlessShader` with the appropriate `ShaderType`, `spirv_binary` with static lifetime, and a
`new` method that returns a reference with static lifetime to the struct. What this means is that due to [rvalue
static promotion](https://rust-lang.github.io/rfcs/1414-rvalue_static_promotion.html) the spir-v binary for each
entry point will be in static memory, likely putting it in the data section of the executable itself. With `new`
being able to be called as much as you want with no performance penalty since it just gives a read-only reference
to that memory.

All of that generated code will be put into `%OUT_DIR%/shader.rs`, so generally host-side you'll use `include!`
to copy that content to a module you can use, like so

```rust
#![allow(non_camel_case_types)]
include!(concat!(env!("OUT_DIR"), "/shader.rs"));
```

Now since it implements those traits you can use it wherever it's accepted, like for example if the above `shader`
crate contains the entry points `main_vs` and `main_fs` then you can use it to make a graphics pipeline in `cobra`
like so

```rust
let pipeline = cobra::GraphicsPipeline::new(
    context.clone(),
    shader::main_vs::new(),
    shader::main_fs::new(),
    &[ImageFormat::Rgba8Unorm],
);
```

Since `GraphicsPipeline::new` requires the correct `ShaderType` that means you also can't accidentally pass the
incorrect entry point either, if you swapped `shader::main_vs::new()` and `shader::main_fs::new()` above it'd be a
compile error. Another nice thing is the above codegen replicates the module layout, so for example if the `shader`
crate has an entry point at `engine::pbr_simple::main_fs` then you could use it host side as
`shader::engine::pbr_simple::main_fs::new()`.