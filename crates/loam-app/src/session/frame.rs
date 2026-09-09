use std::sync::Arc;

use glam::Vec2;
use web_time::Instant;
use wgpu::{CommandBuffer, CommandEncoder, Texture, TextureFormat, TextureView};
use winit::window::Window;

use loam_render::device::GpuContext;
use loam_render::present::Presenter;
use loam_render::work::{BulkBuffers, Readbacks};
#[cfg(not(target_arch = "wasm32"))]
use loam_runtime::host::HostConfig;
use loam_runtime::host::HostError;
use loam_runtime::{Landing, Records, RequestId, Session, SnapshotPolicy, Stores};
use loam_time::{frame_trace, FixedTimestep};

use super::app::{FrameHook, SessionApp};
use super::debug_layer::DebugLayer;
use super::input::InputMap;
use super::WorkContext;

const BACKGROUND: wgpu::Color = wgpu::Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};

pub struct Target<'a> {
    pub view: &'a TextureView,
    pub texture: &'a Texture,
    pub format: TextureFormat,
    pub size: (u32, u32),
}

pub(crate) fn failed(error: impl std::fmt::Display) -> HostError {
    HostError::Host(error.to_string())
}

struct Inner<A: Stores> {
    session: Session<A>,
    records: Records<A>,
    timestep: FixedTimestep,
    buffers: BulkBuffers,
    readbacks: Readbacks,
    issued: Vec<RequestId>,
    callbacks: Vec<CommandBuffer>,
    input: InputMap,
    app: SessionApp<A>,
    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    capture: crate::capture::Capture,
}

pub(crate) struct Frame<A: Stores> {
    presenter: Option<Presenter>,
    layer: Option<DebugLayer>,
    inner: Inner<A>,
}

impl<A: Stores> Frame<A> {
    pub(crate) fn new(mut session: Session<A>, app: SessionApp<A>) -> Self {
        let sim = session.config();
        app.console.install(&mut session);
        Self {
            presenter: None,
            layer: None,
            inner: Inner {
                session,
                records: Records::default(),
                timestep: FixedTimestep::new(sim.fixed_hz)
                    .with_max_catch_up(sim.max_ticks_per_frame),
                buffers: BulkBuffers::default(),
                readbacks: Readbacks::default(),
                issued: Vec::new(),
                callbacks: Vec::new(),
                input: InputMap::default(),
                app,
                #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
                capture: crate::capture::Capture::new(),
            },
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn config(&self) -> &HostConfig {
        &self.inner.app.config
    }

    pub(crate) fn app_mut(&mut self) -> &mut SessionApp<A> {
        &mut self.inner.app
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn input_mut(&mut self) -> &mut InputMap {
        &mut self.inner.input
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn apply_message(&mut self, message: &crate::wasm::input_queue::InputMessage) {
        super::input::apply(
            &mut self.inner.input,
            &self.inner.app.config.bindings,
            message,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn action(&mut self, key: loam_runtime::Key, pressed: bool) {
        self.inner
            .input
            .action(&self.inner.app.config.bindings, key, pressed);
    }

    pub(crate) fn layer(&self) -> Option<&DebugLayer> {
        self.layer.as_ref()
    }

    pub(crate) fn reset_clock(&mut self, now: Instant) {
        self.inner.timestep.reset_clock(now);
        self.inner.app.pacer.reset();
    }

    pub(crate) fn attach(
        &mut self,
        gpu: &GpuContext,
        format: TextureFormat,
        sample_count: u32,
        window: Option<Arc<Window>>,
        size: (u32, u32),
        scale: f32,
    ) -> Result<(), HostError> {
        let mut presenter = Presenter::new(format, sample_count);
        for pass in self.inner.app.passes.drain(..) {
            presenter.register_pass(pass).map_err(failed)?;
        }
        if self.inner.app.debug_layer {
            let layer = match self.layer.clone() {
                Some(layer) => layer,
                None => match window {
                    Some(window) => DebugLayer::on_window(gpu, format, sample_count, window, size),
                    None => DebugLayer::offscreen(gpu, format, sample_count, size, scale),
                },
            };
            presenter.register_pass(layer.pass()).map_err(failed)?;
            self.layer = Some(layer);
        }
        presenter.attach(gpu).map_err(failed)?;
        self.presenter = Some(presenter);
        self.resize(size.0, size.1, scale);
        Ok(())
    }

    pub(crate) fn recover(&mut self, gpu: &GpuContext) -> Result<(), HostError> {
        if let Some(presenter) = self.presenter.as_mut() {
            presenter.attach(gpu).map_err(failed)?;
        }
        self.inner.recover_work(gpu)
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.inner.input.resize(width, height, scale);
        if let Some(layer) = self.layer.as_ref() {
            layer.resize(width, height, scale);
        }
    }

    pub(crate) fn step(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        finish: impl FnOnce(&mut CommandEncoder),
    ) -> Result<(), HostError> {
        let Some(presenter) = self.presenter.as_mut() else {
            return Ok(());
        };
        if target.size.0 == 0 || target.size.1 == 0 {
            return Ok(());
        }
        frame_trace::begin_frame();
        let outcome = {
            let _frame = frame_trace::scope("frame");
            self.inner
                .drive(gpu, target, now, finish, presenter, self.layer.as_ref())
        };
        frame_trace::end_frame();
        outcome
    }
}

impl<A: Stores> Inner<A> {
    fn recover_work(&mut self, gpu: &GpuContext) -> Result<(), HostError> {
        self.session.cancel_work();
        self.readbacks.cancel();
        for (id, spec) in self.session.bulk().iter() {
            self.buffers.remove(id);
            self.buffers.ensure(&gpu.device, id, spec);
            match spec.snapshot {
                SnapshotPolicy::Authoritative => match self.session.checkpoint_rows(id) {
                    Some(rows) => self.buffers.write(&gpu.queue, id, rows),
                    None => {
                        return Err(HostError::Host(format!(
                            "{} is authoritative and has no checkpoint to recover",
                            spec.name
                        )))
                    }
                },
                SnapshotPolicy::Reinitializable => self.buffers.reinitialize(&gpu.queue, id),
                SnapshotPolicy::Derived => {}
            }
        }
        Ok(())
    }

    fn advance(&mut self, now: Instant, gpu: &GpuContext) -> Result<(), HostError> {
        let controls = self.app.console.take_controls();
        if let Some(fps) = controls.target_fps {
            self.app.pacer.set_target_fps(fps);
        }
        if controls.vsync.is_some() {
            self.app.vsync = controls.vsync;
        }
        let Inner {
            session,
            readbacks,
            input,
            timestep,
            app,
            ..
        } = self;
        let mut broken = None;
        readbacks.poll(&gpu.device, |request, rows| {
            if session.land_readback(request, rows) == Landing::Failed {
                broken = Some(request);
            }
        });
        if let Some(request) = broken {
            return Err(HostError::Host(format!(
                "a GPU readback failed for {request:?}"
            )));
        }
        let resuming = match session.waiting() {
            None => false,
            Some(wait) => {
                if !session
                    .readbacks()
                    .any(|landed| landed.request == wait.request)
                {
                    return Ok(());
                }
                true
            }
        };
        {
            let _dispatch = frame_trace::scope("dispatch");
            let gathered = if resuming {
                loam_runtime::Input::default()
            } else {
                input.take()
            };
            session.boundary(gathered)?;
        }
        {
            let _simulation = frame_trace::scope("simulation");
            for _ in timestep.advance(now) {
                session.tick()?;
            }
        }
        if session.waiting().is_none() {
            input.reclaim(session.take_input());
        }
        app.console.collect(session);
        Ok(())
    }

    fn drive(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        finish: impl FnOnce(&mut CommandEncoder),
        presenter: &mut Presenter,
        layer: Option<&DebugLayer>,
    ) -> Result<(), HostError> {
        self.advance(now, gpu)?;

        let (width, height) = target.size;
        let eye = {
            let root = self.session.views().root();
            let Some(image) = self.session.views_mut().get_mut(root) else {
                return Ok(());
            };
            image.eye.aspect = width as f32 / height as f32;
            image.eye
        };
        let published = {
            let _publication = frame_trace::scope("publication");
            self.records
                .publish(&mut self.session)
                .map_err(|error| HostError::Host(format!("{error:?}")))?;
            self.records.lend()
        };
        let Some(published) = published else {
            return Ok(());
        };

        let context = layer.map(DebugLayer::begin);
        if let Some(hook) = self.app.frame.as_mut() {
            hook(&mut FrameHook {
                session: &mut self.session,
                sections: presenter.sections(),
                ui: context.as_ref(),
                size: target.size,
            });
        }
        if let Some(context) = context.as_ref() {
            loam_egui::ConsoleUi::ui(self.app.console.ui_mut(), context);
        }
        self.app.console.dispatch_pending();
        if let Some(layer) = layer {
            layer.finish();
        }

        let _presentation = frame_trace::scope("presentation");
        presenter.upload(
            &gpu.device,
            &gpu.queue,
            &eye,
            Vec2::new(width as f32, height as f32),
            &published.views,
        );
        self.records.release(published);

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loam-app::session"),
            });
        for (id, spec) in self.session.bulk().iter() {
            self.buffers.ensure(&gpu.device, id, spec);
        }
        self.issued.clear();
        let Inner {
            session,
            buffers,
            readbacks,
            issued,
            app,
            ..
        } = self;
        let record_work = &mut app.work;
        session.issue_work(|order| {
            issued.push(order.request);
            record_work(WorkContext {
                gpu,
                encoder: &mut encoder,
                order,
                buffers,
                readbacks,
            });
        });
        presenter.record(
            &gpu.device,
            &mut encoder,
            target.view,
            target.size,
            BACKGROUND,
        );
        finish(&mut encoder);
        if let Some(layer) = layer {
            layer.take_callbacks(&mut self.callbacks);
        }
        gpu.queue
            .submit(self.callbacks.drain(..).chain(Some(encoder.finish())));
        presenter.after_submit();
        self.readbacks.after_submit();
        while let Some(request) = self.issued.pop() {
            self.session.submitted(request);
        }
        self.tap_capture(gpu, target);
        Ok(())
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    fn tap_capture(&mut self, gpu: &GpuContext, target: &Target<'_>) {
        if !self.app.captures.is_empty() {
            let requests = std::mem::take(&mut self.app.captures);
            for line in self.capture.apply_requests(requests) {
                tracing::info!("{line}");
            }
        }
        let now = Instant::now();
        if !self.capture.should_capture(now) {
            return;
        }
        let stages = [
            (true, self.capture.wants_pre()),
            (false, self.capture.wants_post()),
        ];
        if stages.iter().any(|(_, wanted)| *wanted) {
            match crate::capture::read_texture_rgba(
                &gpu.device,
                &gpu.queue,
                target.texture,
                target.size.0,
                target.size.1,
                target.format,
            ) {
                Ok(image) => {
                    for (is_pre, wanted) in stages {
                        if !wanted {
                            continue;
                        }
                        if let Err(error) = self.capture.consume_frame(
                            is_pre,
                            image.rgba.clone(),
                            image.width,
                            image.height,
                            now,
                        ) {
                            tracing::error!("capture: write failed: {error:#}");
                        }
                    }
                }
                Err(error) => tracing::error!("capture: readback failed: {error:#}"),
            }
        }
        self.capture.advance_frame(now);
    }

    #[cfg(not(all(feature = "capture", not(target_arch = "wasm32"))))]
    fn tap_capture(&mut self, _gpu: &GpuContext, _target: &Target<'_>) {}
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};

    use loam_render::device::{FeatureRequest, MissingGpuCapability};
    use loam_render::pass::{FramePass, FrameTarget, PassOrder};
    use loam_runtime::{
        Access, ActionId, Bindings, BulkSpec, Commands, Ctx, HostConfig, Input, Key, Phase,
        Readback, RequestId, Schedule, SimConfig, SnapshotPolicy, WorkItem,
    };
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };

    use super::*;
    use crate::args::Args;

    const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
    const SIZE: (u32, u32) = (64, 48);

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Bare {}
    }

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Watched {
            walked: Value<bool>,
        }
    }

    struct Probe {
        recorded: Arc<AtomicU32>,
        rebuilt: Arc<AtomicU32>,
    }

    impl FramePass for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn order(&self) -> PassOrder {
            PassOrder::AfterScene
        }

        fn record(&self, _encoder: &mut CommandEncoder, _target: &FrameTarget<'_>) {
            self.recorded.fetch_add(1, Ordering::Relaxed);
        }

        fn rebuild(&mut self, _gpu: &GpuContext) -> Result<(), MissingGpuCapability> {
            self.rebuilt.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    fn noop_gpu() -> GpuContext {
        let instance = Instance::new(&InstanceDescriptor {
            backends: Backends::NOOP,
            backend_options: BackendOptions {
                noop: NoopBackendOptions { enable: true },
                ..Default::default()
            },
            ..Default::default()
        });
        pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("the noop backend always yields a context")
    }

    fn offscreen(gpu: &GpuContext) -> Texture {
        gpu.device.create_texture(&TextureDescriptor {
            label: Some("session frame probe"),
            size: Extent3d {
                width: SIZE.0,
                height: SIZE.1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: FORMAT,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn bare(app: SessionApp<Bare>) -> Frame<Bare> {
        Frame::new(Session::new(Bare::default(), SimConfig::default()), app)
    }

    fn host(name: &'static str) -> SessionApp<Bare> {
        SessionApp::with_args(HostConfig::new(name, Bindings::new()), Args::default())
            .debug_layer(false)
    }

    fn run_one<A: Stores>(frame: &mut Frame<A>, gpu: &GpuContext, texture: &Texture) {
        let view = texture.create_view(&TextureViewDescriptor::default());
        let target = Target {
            view: &view,
            texture,
            format: FORMAT,
            size: SIZE,
        };
        frame
            .step(gpu, &target, Instant::now(), |_| {})
            .expect("the frame stepped");
    }

    #[test]
    fn a_registered_pass_records_again_after_the_device_is_recovered() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let recorded = Arc::new(AtomicU32::new(0));
        let rebuilt = Arc::new(AtomicU32::new(0));
        let app = host("passes").pass(Box::new(Probe {
            recorded: recorded.clone(),
            rebuilt: rebuilt.clone(),
        }));
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        run_one(&mut frame, &gpu, &texture);
        assert_eq!(recorded.load(Ordering::Relaxed), 1);
        assert_eq!(rebuilt.load(Ordering::Relaxed), 1);

        frame.recover(&gpu).expect("recovered");
        run_one(&mut frame, &gpu, &texture);
        assert_eq!(
            rebuilt.load(Ordering::Relaxed),
            2,
            "recovery never rebuilt the application's pass on the new device"
        );
        assert_eq!(
            recorded.load(Ordering::Relaxed),
            2,
            "the recovered presenter no longer records the application's pass"
        );
    }

    #[test]
    fn the_frame_callback_reads_the_sections_the_previous_frame_recorded() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let seen: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
        let reported = seen.clone();
        let app = host("sections")
            .pass(Box::new(Probe {
                recorded: Arc::new(AtomicU32::new(0)),
                rebuilt: Arc::new(AtomicU32::new(0)),
            }))
            .on_frame(move |hook| {
                let mut names = reported.lock().unwrap_or_else(|error| error.into_inner());
                names.clear();
                names.extend(hook.sections.iter().map(|section| section.name));
            });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");

        run_one(&mut frame, &gpu, &texture);
        run_one(&mut frame, &gpu, &texture);

        let names = seen.lock().unwrap_or_else(|error| error.into_inner());
        assert!(
            names.contains(&"probe") && names.contains(&"present-draw"),
            "the callback ran after the presenter cleared its sections: {names:?}"
        );
    }

    #[test]
    fn input_gathered_while_the_session_waits_survives_to_the_boundary_that_reads_it() {
        const WALK: ActionId = ActionId(0);
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let runs = Arc::new(AtomicU32::new(0));
        let counted = runs.clone();

        let mut session = Session::new(Watched::default(), SimConfig::default());
        let grid = session.register_bulk(BulkSpec {
            name: "grid",
            element_size: 4,
            count: 4,
            readback: Readback::Required,
            snapshot: SnapshotPolicy::Derived,
            schedule: Schedule::InStep,
        });
        session.work(
            Phase::Simulation,
            WorkItem::new("reduce", Schedule::InStep, Readback::Required).writes(grid),
        );
        session.system(
            Phase::Dispatch,
            "observe",
            Access::new(),
            |ctx: Ctx<'_, Watched>| {
                if ctx.input.pressed(WALK) {
                    ctx.app.walked.set(true);
                }
            },
        );
        session.system(
            Phase::Dispatch,
            "consume",
            Access::new().awaits("reduce"),
            move |_input: &Input, _commands: &mut Commands<Watched>| {
                counted.fetch_add(1, Ordering::Relaxed);
            },
        );
        session.boundary(Input::default()).expect("first boundary");
        session.tick().expect("tick");
        let mut issued: Option<RequestId> = None;
        session.issue_work(|order| issued = Some(order.request));
        let request = issued.expect("the tick ordered the work item");
        session.submitted(request);
        session.boundary(Input::default()).expect("second boundary");
        assert_eq!(runs.load(Ordering::Relaxed), 1);
        assert!(session.waiting().is_some());

        let bindings = Bindings::new().key(Key::Letter('w'), WALK);
        let app = SessionApp::with_args(HostConfig::new("waiting", bindings), Args::default())
            .debug_layer(false);
        let mut frame = Frame::new(session, app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        frame.action(Key::Letter('w'), true);

        run_one(&mut frame, &gpu, &texture);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            1,
            "the host ran a boundary while the session was waiting on its readback"
        );

        frame.inner.session.land_readback(request, Some(&[0u8; 16]));
        run_one(&mut frame, &gpu, &texture);
        assert_eq!(
            runs.load(Ordering::Relaxed),
            2,
            "the landed readback never resumed the suspended entry"
        );

        run_one(&mut frame, &gpu, &texture);
        assert!(
            *frame.inner.session.app.walked.get(),
            "a boundary that only resumed a suspended entry swallowed the gathered input"
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn a_capture_from_the_session_host_writes_the_targets_pixels_gpu_probe() {
        let instance = Instance::default();
        let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("a wgpu adapter");
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let app = host("capture").capture(crate::capture::CaptureRequest::OneShot {
            stage: crate::capture::CaptureStage::Post,
            dir: Some(directory.path().to_path_buf()),
            name: Some("session".into()),
        });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        run_one(&mut frame, &gpu, &texture);
        drop(frame);

        let written = directory.path().join("session_post.png");
        let decoded = ::image::open(&written).expect("the capture wrote a png");
        assert_eq!(
            (decoded.width(), decoded.height()),
            SIZE,
            "the capture read a different rectangle than the presenter's target"
        );
    }
}
