use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::pass::{
    FrameFormat, FramePass, FrameTarget, PassFault, PassStage, PublishedFrame,
};
use loam_render::present::Presenter;
use wgpu::*;

const SIZE: (u32, u32) = (64, 64);
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

fn frame_at(gpu: &GpuContext, presenter: &mut Presenter, view: &TextureView, size: (u32, u32)) {
    presenter.upload(
        &gpu.device,
        &gpu.queue,
        &PublishedFrame {
            eye: Default::default(),
            viewport_px: glam::Vec2::splat(64.0),
            pixels_per_point: 1.0,
            text_origin_px: [0.0; 2],
            views: &[],
            overlay: &Default::default(),
            backdrop: None,
        },
    );
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    presenter
        .record_scene(&gpu.device, &mut encoder, view, size, Color::BLACK)
        .expect("recorded the scene");
    presenter
        .record_overlays(&mut encoder, view, size)
        .expect("recorded the overlays");
    gpu.queue.submit(Some(encoder.finish()));
    presenter.after_submit();
}

const PAINT_WGSL: &str = r#"
@vertex
fn vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var corners = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(corners[index], 0.0, 1.0);
}

@fragment
fn fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

struct Paint {
    pipeline: Option<RenderPipeline>,
    attaches: Arc<AtomicU32>,
}

impl FramePass for Paint {
    fn name(&self) -> &'static str {
        "paint"
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> Result<(), PassFault> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            return Err(PassFault::Unattached);
        };
        let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("paint"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view: target.color,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Load,
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        pass.set_pipeline(pipeline);
        pass.draw(0..3, 0..1);
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> Result<(), PassFault> {
        self.attaches.fetch_add(1, Ordering::Relaxed);
        let shader = gpu.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("paint"),
            source: ShaderSource::Wgsl(PAINT_WGSL.into()),
        });
        self.pipeline = Some(
            gpu.device
                .create_render_pipeline(&RenderPipelineDescriptor {
                    label: Some("paint"),
                    layout: None,
                    vertex: VertexState {
                        module: &shader,
                        entry_point: Some("vertex"),
                        compilation_options: Default::default(),
                        buffers: &[],
                    },
                    primitive: PrimitiveState::default(),
                    depth_stencil: None,
                    multisample: MultisampleState::default(),
                    fragment: Some(FragmentState {
                        module: &shader,
                        entry_point: Some("fragment"),
                        compilation_options: Default::default(),
                        targets: &[Some(ColorTargetState {
                            format: frame.color,
                            blend: None,
                            write_mask: ColorWrites::ALL,
                        })],
                    }),
                    multiview: None,
                    cache: None,
                }),
        );
        Ok(())
    }
}

fn paintable(device: &Device, size: (u32, u32)) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some("paint target"),
        size: Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn pixel(gpu: &GpuContext, texture: &Texture, size: (u32, u32), at: (u32, u32)) -> [u8; 4] {
    let bytes_per_row = (size.0 * 4).next_multiple_of(COPY_BYTES_PER_ROW_ALIGNMENT);
    let readback = gpu.device.create_buffer(&BufferDescriptor {
        label: Some("paint readback"),
        size: u64::from(bytes_per_row) * u64::from(size.1),
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = gpu
        .device
        .create_command_encoder(&CommandEncoderDescriptor { label: None });
    encoder.copy_texture_to_buffer(
        texture.as_image_copy(),
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: size.0,
            height: size.1,
            depth_or_array_layers: 1,
        },
    );
    gpu.queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| {});
    gpu.device
        .poll(PollType::wait_indefinitely())
        .expect("readback poll");
    let mapped = readback.slice(..).get_mapped_range();
    let start = (at.1 * bytes_per_row + at.0 * 4) as usize;
    let mut out = [0; 4];
    out.copy_from_slice(&mapped[start..start + 4]);
    out
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn resize_and_device_recreation_gpu_probe() {
    const GREEN: [u8; 4] = [0, 255, 0, 255];
    const RESIZED: (u32, u32) = (96, 48);
    let mut gpu = pollster::block_on(GpuContext::new(
        Instance::default(),
        FeatureRequest::default(),
        None,
    ))
    .expect("gpu context");
    let attaches = Arc::new(AtomicU32::new(0));
    let mut presenter = Presenter::new(COLOR_FORMAT).expect("presenter");
    presenter
        .register_pass(Box::new(Paint {
            pipeline: None,
            attaches: attaches.clone(),
        }))
        .expect("registered");
    presenter.attach(&gpu).expect("attach");

    let texture = paintable(&gpu.device, SIZE);
    frame_at(
        &gpu,
        &mut presenter,
        &texture.create_view(&TextureViewDescriptor::default()),
        SIZE,
    );
    assert_eq!(
        pixel(&gpu, &texture, SIZE, (SIZE.0 / 2, SIZE.1 / 2)),
        GREEN,
        "the custom pass did not reach the first frame"
    );

    let texture = paintable(&gpu.device, RESIZED);
    frame_at(
        &gpu,
        &mut presenter,
        &texture.create_view(&TextureViewDescriptor::default()),
        RESIZED,
    );
    assert_eq!(
        pixel(&gpu, &texture, RESIZED, (RESIZED.0 - 1, RESIZED.1 - 1)),
        GREEN,
        "the custom pass did not reach the far corner of the resized frame"
    );

    let lost = gpu.device.clone();
    lost.destroy();
    pollster::block_on(gpu.recover()).expect("recover");
    presenter.attach(&gpu).expect("reattach");
    assert_eq!(attaches.load(Ordering::Relaxed), 2);

    gpu.device.push_error_scope(ErrorFilter::Validation);
    let texture = paintable(&gpu.device, RESIZED);
    frame_at(
        &gpu,
        &mut presenter,
        &texture.create_view(&TextureViewDescriptor::default()),
        RESIZED,
    );
    let error = pollster::block_on(gpu.device.pop_error_scope());
    assert!(
        error.is_none(),
        "a pass resource stayed on the lost device: {error:?}"
    );
    assert_eq!(
        pixel(&gpu, &texture, RESIZED, (0, 0)),
        GREEN,
        "the custom pass did not reach the frame on the recreated device"
    );
    assert!(presenter
        .sections()
        .iter()
        .any(|section| section.name == "paint"));
}
