use crate::mod_node::ModNode;
use cargo_gpu::spirv_builder::{
    Capability, MetadataPrintout, ModuleResult, ShaderPanicStrategy, SpirvBuilder, SpirvMetadata,
};
use proc_macro2::TokenStream;
use quote::quote;
use spirq::prelude::ExecutionModel;
use spirq::ReflectConfig;
use std::borrow::Cow;
use std::ffi::CString;
use std::fs;
use std::fs::File;
use std::path::PathBuf;

pub struct ShaderBuilder {
    spirv_builder: SpirvBuilder,
}

impl ShaderBuilder {
    /// Build the shader crate named `crate_name` at path `../{crate_name}` relative to your `Cargo.toml`.
    /// # Errors
    /// If the crate path is invalid
    pub fn new(crate_name: &str) -> anyhow::Result<Self> {
        ShaderBuilder::new_relative_path(crate_name)
    }

    /// Build the shader crate at path `../{relative_crate_path}` relative to your `Cargo.toml`
    /// and assume the crate is accessible with the ident `crate_ident`.
    /// # Errors
    /// If the crate path is invalid
    /// # Panics
    /// If ran outside a build script
    pub fn new_relative_path(relative_crate_path: &str) -> anyhow::Result<Self> {
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR")
            .expect("missing env var, must be called within a build script");
        let crate_path = [&manifest_dir, "..", relative_crate_path]
            .iter()
            .copied()
            .collect::<PathBuf>();

        ShaderBuilder::new_absolute_path(crate_path)
    }

    /// Build the shader crate at path `absolute_crate_path`
    /// # Errors
    /// If the crate path is invalid
    pub fn new_absolute_path(absolute_crate_path: PathBuf) -> anyhow::Result<Self> {
        let install = cargo_gpu::Install::from_shader_crate(absolute_crate_path.clone()).run()?;

        Ok(Self {
            spirv_builder: install
                .to_spirv_builder(absolute_crate_path, "spirv-unknown-vulkan1.2")
                .multimodule(true)
                // required capabilities
                .capability(Capability::RuntimeDescriptorArray)
                .capability(Capability::ShaderNonUniform)
                .capability(Capability::StorageBufferArrayDynamicIndexing)
                .capability(Capability::StorageImageArrayDynamicIndexing)
                .capability(Capability::SampledImageArrayDynamicIndexing)
                .capability(Capability::StorageBufferArrayNonUniformIndexing)
                .capability(Capability::StorageImageArrayNonUniformIndexing)
                .capability(Capability::SampledImageArrayNonUniformIndexing)
                .scalar_block_layout(true)
                // metadata
                .print_metadata(MetadataPrintout::DependencyOnly)
                .spirv_metadata(SpirvMetadata::Full)
                // print panics
                .shader_panic_strategy(ShaderPanicStrategy::DebugPrintfThenExit {
                    print_inputs: true,
                    print_backtrace: true,
                }),
        })
    }

    /// # Errors
    /// If there is a spirv build error
    pub fn build(self) -> anyhow::Result<()> {
        let spirv_result = self.spirv_builder.build()?;
        let out_path = std::path::Path::new(&std::env::var("OUT_DIR")?).join("shader.rs");

        match &spirv_result.module {
            ModuleResult::SingleModule(path) => codegen_shaders(
                spirv_result
                    .entry_points
                    .iter()
                    .map(|name| (name.as_str(), path)),
                &out_path,
            )?,
            ModuleResult::MultiModule(m) => codegen_shaders(
                m.iter().map(|(name, path)| (name.as_str(), path)),
                &out_path,
            )?,
        }

        Ok(())
    }
}

fn codegen_try_pretty_print(tokens: &TokenStream) -> (String, Option<syn::Error>) {
    match syn::parse2(tokens.clone()) {
        Ok(parse) => (prettyplease::unparse(&parse), None),
        Err(e) => (tokens.to_string(), Some(e)),
    }
}

fn codegen_shaders<'a>(
    shaders: impl Iterator<Item = (&'a str, &'a PathBuf)>,
    out_path: &PathBuf,
) -> anyhow::Result<()> {
    let mut root = ModNode::root();
    for shader in shaders {
        root.insert(shader.0.split("::").map(Cow::Borrowed), shader)?;
    }
    let tokens = root.to_tokens(|ident, (entry_point_name, spv_path)| {
        let entry_point_name = CString::new(*entry_point_name).unwrap();

        let mut spv_file = File::open(spv_path).unwrap();
        let spv_binary = ash::util::read_spv(&mut spv_file).unwrap();

        let entry_points = ReflectConfig::new()
            .spv(spv_binary.clone())
            .reflect()
            .unwrap();
        let shader_type: syn::TypePath = match entry_points[0].exec_model {
            ExecutionModel::Vertex => syn::parse_quote!(cobra::VertexShader),
            ExecutionModel::Fragment => syn::parse_quote!(cobra::FragmentShader),
            _ => todo!(),
        };

        quote! {
            pub struct #ident;

            impl cobra::BindlessShader for #ident {
                type ShaderType = #shader_type;

                #[allow(clippy::unreadable_literal)]
                #[allow(clippy::too_many_lines)]
                #[allow(clippy::large_stack_arrays)]
                fn spirv_binary(&self) -> &cobra::SpirvBinary<'static> {
                    &cobra::SpirvBinary {
                        binary: &[#(#spv_binary),*],
                        entry_point_name: #entry_point_name,
                    }
                }
            }

            impl #ident {
                pub fn new() -> &'static #ident {
                    &#ident {}
                }
            }
        }
    });

    let (content, error) = codegen_try_pretty_print(&tokens);
    fs::write(out_path, content)?;
    eprintln!("Shader file written to {}", out_path.display());
    if let Some(error) = error {
        Err(error)?
    } else {
        Ok(())
    }
}
