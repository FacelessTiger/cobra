use std::ffi::CStr;

pub trait BindlessShader {
    type ShaderType: ShaderType;

    fn spirv_binary(&self) -> &SpirvBinary<'_>;
}

pub enum Shader {
    Vertex,
    Fragment,
    Compute,
}

pub trait ShaderType {
    const SHADER: Shader;
}

pub struct VertexShader;
impl ShaderType for VertexShader {
    const SHADER: Shader = Shader::Vertex;
}

pub struct FragmentShader;
impl ShaderType for FragmentShader {
    const SHADER: Shader = Shader::Fragment;
}

pub struct ComputeShader;
impl ShaderType for ComputeShader {
    const SHADER: Shader = Shader::Compute;
}

pub struct SpirvBinary<'a> {
    pub binary: &'a [u32],
    pub entry_point_name: &'a CStr,
}
