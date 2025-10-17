use cobra_shader_builder::ShaderBuilder;
use std::error::Error;

pub fn main() -> Result<(), Box<dyn Error>> {
    ShaderBuilder::new("egui-cobra-shaders")?.build()?;

    Ok(())
}
