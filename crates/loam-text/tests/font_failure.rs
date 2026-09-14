use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::pass::{FrameFormat, FramePass};
use loam_text::TextPass;
use wgpu::{
    BackendOptions, Backends, Instance, InstanceDescriptor, NoopBackendOptions, TextureFormat,
};

#[test]
fn a_font_that_does_not_parse_leaves_the_text_pass_drawing_nothing() {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });
    let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("noop context");
    let mut pass = TextPass::new("hud", vec![0, 1, 2, 3], 16.0);
    pass.attach(
        &gpu,
        FrameFormat {
            color: TextureFormat::Rgba8Unorm,
            depth: TextureFormat::Depth32Float,
        },
    )
    .expect("a font that does not parse is not fatal to the host");
    assert!(!pass.ready(), "the pass claims to be ready with no font");
    assert!(pass.failure().is_some(), "the parse failure was not kept");
}
