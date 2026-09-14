use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::gpu_timer::SectionTimer;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage};
use loam_render::present::Presenter;
use loam_render::{DepthConvention, GpuTime};
use wgpu::*;

const SIZE: (u32, u32) = (64, 64);
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

struct Recorder {
    name: &'static str,
    convention: Option<DepthConvention>,
    log: Arc<AtomicU32>,
    tag: u32,
}

impl FramePass for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        self.convention
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        self.log.store(
            self.log.load(Ordering::Relaxed) * 10 + self.tag,
            Ordering::Relaxed,
        );
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some(self.name),
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
        Ok(())
    }

    fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
        Ok(())
    }
}

const COPY_BYTES: u64 = 64 << 20;

#[derive(Default)]
struct Blit {
    buffers: Option<(Buffer, Buffer)>,
}

impl FramePass for Blit {
    fn name(&self) -> &'static str {
        "blit"
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn record(
        &mut self,
        encoder: &mut CommandEncoder,
        _target: &FrameTarget<'_>,
    ) -> anyhow::Result<()> {
        if let Some((source, sink)) = self.buffers.as_ref() {
            encoder.copy_buffer_to_buffer(source, 0, sink, 0, COPY_BYTES);
        }
        Ok(())
    }

    fn attach(&mut self, gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
        let held = |usage| {
            gpu.device.create_buffer(&BufferDescriptor {
                label: Some("blit"),
                size: COPY_BYTES,
                usage,
                mapped_at_creation: false,
            })
        };
        self.buffers = Some((held(BufferUsages::COPY_SRC), held(BufferUsages::COPY_DST)));
        Ok(())
    }
}

fn noop_context() -> GpuContext {
    let instance = Instance::new(&InstanceDescriptor {
        backends: Backends::NOOP,
        backend_options: BackendOptions {
            noop: NoopBackendOptions { enable: true },
            ..Default::default()
        },
        ..Default::default()
    });
    pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
        .expect("noop context")
}

fn target(device: &Device) -> TextureView {
    device
        .create_texture(&TextureDescriptor {
            label: Some("custom pass target"),
            size: Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: COLOR_FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        })
        .create_view(&TextureViewDescriptor::default())
}

fn frame(gpu: &GpuContext, presenter: &mut Presenter, view: &TextureView) {
    frame_at(gpu, presenter, view, SIZE);
}

fn frame_at(gpu: &GpuContext, presenter: &mut Presenter, view: &TextureView, size: (u32, u32)) {
    presenter.upload(
        &gpu.device,
        &gpu.queue,
        &Default::default(),
        glam::Vec2::splat(64.0),
        &[],
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

#[test]
fn without_a_timer_a_recorded_pass_reports_its_gpu_time_unavailable() {
    let gpu = noop_context();
    let view = target(&gpu.device);
    let mut presenter = Presenter::new(COLOR_FORMAT).expect("presenter");
    presenter
        .register_pass(Box::new(Recorder {
            name: "overlay",
            convention: None,
            log: Arc::new(AtomicU32::new(0)),
            tag: 1,
        }))
        .expect("registered");

    frame(&gpu, &mut presenter, &view);
    let overlay = presenter
        .sections()
        .iter()
        .find(|section| section.name == "overlay")
        .expect("the overlay is a recorded section");
    assert_eq!(overlay.gpu, GpuTime::Unavailable);
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_timed_pass_reports_a_positive_gpu_duration_gpu_probe() {
    let gpu = pollster::block_on(GpuContext::new(
        Instance::default(),
        FeatureRequest::default(),
        None,
    ))
    .expect("gpu context");
    let timestamps = gpu
        .device
        .features()
        .contains(loam_render::device::GPU_TIMER_FEATURES);
    let view = target(&gpu.device);
    let mut presenter = Presenter::new(COLOR_FORMAT).expect("presenter");
    presenter
        .register_pass(Box::new(Blit::default()))
        .expect("registered");
    presenter.attach(&gpu).expect("attach");

    let mut empty = GpuTime::Unavailable;
    let mut loaded = GpuTime::Unavailable;
    for _ in 0..4 {
        frame(&gpu, &mut presenter, &view);
        gpu.device
            .poll(PollType::wait_indefinitely())
            .expect("poll");
        for section in presenter.sections() {
            match section.name {
                "present-draw" => empty = section.gpu,
                "blit" => loaded = section.gpu,
                _ => {}
            }
        }
    }
    match (timestamps, empty, loaded) {
        (true, GpuTime::Measured(_), GpuTime::Measured(elapsed)) => assert!(
            elapsed > std::time::Duration::ZERO,
            "a {COPY_BYTES} byte copy reported {elapsed:?}"
        ),
        (true, empty, loaded) => {
            panic!("the timer left a section unmeasured: {empty:?} and {loaded:?}")
        }
        (false, empty, loaded) => {
            assert_eq!(
                (empty, loaded),
                (GpuTime::Unavailable, GpuTime::Unavailable)
            )
        }
    }
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn an_unresolved_frame_maps_nothing_while_the_previous_map_is_pending_gpu_probe() {
    let gpu = pollster::block_on(GpuContext::new(
        Instance::default(),
        FeatureRequest::default(),
        None,
    ))
    .expect("gpu context");
    let Some(mut timer) = SectionTimer::new(&gpu.device, &gpu.queue) else {
        assert!(
            !gpu.device
                .features()
                .contains(loam_render::device::GPU_TIMER_FEATURES),
            "a device with timestamps must yield a section timer"
        );
        return;
    };

    let mut frame = || {
        timer.begin_frame();
        let mut encoder = gpu
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        if let Some(slot) = timer.open(&mut encoder, "probe") {
            timer.close(&mut encoder, slot);
        }
        timer.resolve(&mut encoder);
        drop(encoder);
        timer.after_submit()
    };
    assert!(frame(), "the first frame resolved and must map its results");
    assert!(
        !frame(),
        "the second frame resolved nothing and must not map over a pending map"
    );
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
    ) -> anyhow::Result<()> {
        let Some(pipeline) = self.pipeline.as_ref() else {
            anyhow::bail!("paint recorded before attach");
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

    fn attach(&mut self, gpu: &GpuContext, frame: FrameFormat) -> anyhow::Result<()> {
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
fn a_custom_pass_paints_the_frame_after_a_resize_and_a_device_recreation_gpu_probe() {
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
