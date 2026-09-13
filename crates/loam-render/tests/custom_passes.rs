use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::gpu_timer::SectionTimer;
use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_COLOR};
use loam_render::present::Presenter;
use loam_render::{DepthConvention, GpuTime};
use wgpu::*;

const SIZE: (u32, u32) = (64, 64);
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const GLOW: ResourceId = "glow";

struct Recorder {
    name: &'static str,
    reads: &'static [ResourceId],
    writes: &'static [ResourceId],
    convention: Option<DepthConvention>,
    log: Arc<AtomicU32>,
    tag: u32,
}

impl FramePass for Recorder {
    fn name(&self) -> &'static str {
        self.name
    }

    fn reads(&self) -> &[ResourceId] {
        self.reads
    }

    fn writes(&self) -> &[ResourceId] {
        self.writes
    }

    fn stage(&self) -> PassStage {
        PassStage::Scene
    }

    fn depth_convention(&self) -> Option<DepthConvention> {
        self.convention
    }

    fn record(&self, encoder: &mut CommandEncoder, target: &FrameTarget<'_>) -> anyhow::Result<()> {
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
        &self,
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

fn presenter_on(gpu: &GpuContext) -> Presenter {
    let mut presenter = Presenter::new(COLOR_FORMAT, 1).expect("presenter");
    presenter.attach(gpu).expect("attach");
    presenter
}

fn frame(gpu: &GpuContext, presenter: &mut Presenter, view: &TextureView) {
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
        .record(&gpu.device, &mut encoder, view, SIZE, Color::BLACK)
        .expect("recorded");
    gpu.queue.submit(Some(encoder.finish()));
    presenter.after_submit();
}

#[test]
fn a_consumer_registered_first_is_recorded_after_the_pass_that_writes_its_input() {
    let gpu = noop_context();
    let view = target(&gpu.device);
    let mut presenter = presenter_on(&gpu);
    let log = Arc::new(AtomicU32::new(0));
    for (name, reads, writes, tag) in [
        ("consumer", [GLOW].as_slice(), [].as_slice(), 2u32),
        ("producer", [].as_slice(), [GLOW].as_slice(), 1),
    ] {
        presenter
            .register_pass(Box::new(Recorder {
                name,
                reads,
                writes,
                convention: None,
                log: log.clone(),
                tag,
            }))
            .expect("registered");
    }

    frame(&gpu, &mut presenter, &view);
    assert_eq!(
        log.load(Ordering::Relaxed),
        12,
        "the producer must record before the consumer that reads its output"
    );
}

#[test]
fn without_a_timer_a_recorded_pass_reports_its_gpu_time_unavailable() {
    let gpu = noop_context();
    let view = target(&gpu.device);
    let mut presenter = Presenter::new(COLOR_FORMAT, 1).expect("presenter");
    presenter
        .register_pass(Box::new(Recorder {
            name: "overlay",
            reads: &[SCENE_COLOR],
            writes: &[],
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
    let mut presenter = Presenter::new(COLOR_FORMAT, 1).expect("presenter");
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
