# `cobra-bindless`

Contains `DevicePointer` and `ImageHandle` to use bindless resources in `cobra`. Also re-exports `cobra-bindless-macro`
and is `no_std` so it's usable in shaders. Note that `cobra` also re-exports this crates handles, so in practice
it's only used shader-side.