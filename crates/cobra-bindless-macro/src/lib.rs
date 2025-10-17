use proc_macro::TokenStream;
use quote::ToTokens;
use syn::punctuated::Punctuated;
use syn::{Attribute, Ident, ItemFn, PatType, Token, parse_macro_input, parse_quote};

#[proc_macro_attribute]
pub fn bindless(attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as ItemFn);
    let attr = parse_macro_input!(attr as Ident);
    input
        .attrs
        .push(parse_quote! { #[allow(clippy::too_many_arguments)] });

    // TODO: for some reason quote isn't letting me insert the attributes directly
    let attributes: Vec<Attribute> = parse_quote! {
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)]
        #[spirv(descriptor_set = 0, binding = 1)]
        #[spirv(descriptor_set = 0, binding = 1)]
        #[spirv(descriptor_set = 0, binding = 2)]
        #[spirv(descriptor_set = 0, binding = 3)]
    };
    let buffers: Punctuated<PatType, Token![,]> = parse_quote! {
        buffers: &::spirv_std::RuntimeArray<::spirv_std::TypedBuffer<[u32]>>,
        images_f32: &::spirv_std::RuntimeArray<::spirv_std::image::Image2d>,
        images_u32: &::spirv_std::RuntimeArray<::spirv_std::image::Image2dU>,
        storage_images_f32: &::spirv_std::RuntimeArray<::spirv_std::image::StorageImage2d>,
        samplers: &::spirv_std::RuntimeArray<::spirv_std::Sampler>
    };
    for (i, mut arg) in buffers.into_iter().enumerate() {
        arg.attrs.push(attributes[i].clone());
        input.sig.inputs.push(arg.into());
    }

    input.block.stmts.insert(0, parse_quote! {
        let mut #attr = ::cobra_bindless::Descriptors { buffers, images_f32, images_u32, storage_images_f32, samplers };
    });
    input.to_token_stream().into()
}
