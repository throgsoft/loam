use std::sync::Arc;

use glam::Vec2;
use web_time::Instant;
use wgpu::{CommandBuffer, CommandEncoder, Texture, TextureFormat, TextureView};
use winit::window::Window;

use loam_render::device::GpuContext;
use loam_render::present::Presenter;
#[cfg(not(target_arch = "wasm32"))]
use loam_runtime::host::HostConfig;
use loam_runtime::host::HostError;
use loam_runtime::{Eye, Publication, PublishError, Records, Session, Stores};
use loam_time::{frame_trace, FixedTimestep};

use super::app::{CaptureControl, FrameHook, InputHook, SessionApp};
use super::cursor::CursorCapture;
use super::debug_layer::DebugLayer;
use super::input::InputMap;

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
    callbacks: Vec<CommandBuffer>,
    input: InputMap,
    cursor: CursorCapture,
    app: SessionApp<A>,
    #[cfg(test)]
    presented: Option<Eye>,
    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    capture: crate::capture::Capture,
}

pub(crate) struct Frame<A: Stores> {
    presenter: Option<Presenter>,
    layer: Option<DebugLayer>,
    inner: Inner<A>,
}

impl<A: Stores> Frame<A> {
    pub(crate) fn new(session: Session<A>, app: SessionApp<A>) -> Self {
        let sim = session.config();
        Self {
            presenter: None,
            layer: None,
            inner: Inner {
                session,
                records: Records::default(),
                timestep: FixedTimestep::new(sim.fixed_hz)
                    .with_max_catch_up(sim.max_ticks_per_frame),
                callbacks: Vec::new(),
                input: InputMap::default(),
                cursor: CursorCapture::new(),
                app,
                #[cfg(test)]
                presented: None,
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

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn phase_error(&self) -> Option<loam_runtime::PhaseError> {
        self.inner.session.phase_error()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn input_mut(&mut self) -> &mut InputMap {
        &mut self.inner.input
    }

    pub(crate) fn alt(&mut self, index: usize, pressed: bool) {
        self.inner.cursor.alt(index, pressed);
    }

    pub(crate) fn focus(&mut self, focused: bool) {
        self.inner.cursor.focus(focused);
        if !focused {
            self.inner.input.release_all();
        }
    }

    pub(crate) fn cursor_applied(&mut self, locked: bool) {
        self.inner.cursor.applied(locked);
        self.inner.input.set_cursor_locked(locked);
    }

    pub(crate) fn cursor_locked(&self) -> bool {
        self.inner.cursor.locked()
    }

    pub(crate) fn take_cursor_request(&mut self) -> Option<bool> {
        self.inner.cursor.take_request()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn apply_message(
        &mut self,
        message: &crate::wasm::input_queue::InputMessage,
        consumed: bool,
    ) {
        super::input::apply(
            &mut self.inner.input,
            &self.inner.app.config.bindings,
            message,
            consumed,
        );
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn observe_message(&mut self, message: &crate::wasm::input_queue::InputMessage) {
        use crate::wasm::input_queue::InputMessage;

        match message {
            InputMessage::Key {
                code,
                pressed,
                repeat,
                ..
            } if !repeat || !pressed => match code.as_str() {
                "AltLeft" => self.alt(0, *pressed),
                "AltRight" => self.alt(1, *pressed),
                _ => {}
            },
            InputMessage::Focus(focused) => self.focus(*focused),
            InputMessage::PointerLockChanged { locked, released } => {
                self.cursor_applied(*locked);
                if *released {
                    self.inner.cursor.released();
                }
            }
            _ => {}
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn action(&mut self, key: loam_runtime::Key, pressed: bool, consumed: bool) {
        self.inner
            .input
            .host_action(&self.inner.app.config.bindings, key, pressed, consumed);
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
        let mut presenter = Presenter::new(format, sample_count).map_err(failed)?;
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
        Ok(())
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
        finish: impl FnMut(&mut CommandEncoder),
    ) -> Result<(), HostError> {
        gpu.device.poll(wgpu::PollType::Poll).map_err(failed)?;
        if let Some(error) = gpu.take_uncaptured_error() {
            return Err(failed(error));
        }
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
    fn advance(
        &mut self,
        now: Instant,
        size: (u32, u32),
        ui: Option<&loam_egui::egui::Context>,
    ) -> Result<(), HostError> {
        let controls = self.app.console.take_controls();
        if let Some(fps) = controls.target_fps {
            self.app.pacer.set_target_fps(fps);
        }
        if controls.vsync.is_some() {
            self.app.vsync = controls.vsync;
        }
        let Inner {
            session,
            input,
            timestep,
            app,
            ..
        } = self;
        let aspect = size.0 as f32 / size.1 as f32;
        session.views_mut().root_mut().eye.aspect = aspect;
        let mut gathered = input.take();
        let sender = app.commands.sender();
        let was_faulted = session.faulted_phase().is_some();
        let recovery = app
            .fault_recovery
            .filter(|action| was_faulted && gathered.pressed(*action));
        if let Some(action) = recovery {
            gathered.actions.retain(|event| event.action != action);
            sender.reset();
        }
        if let Some(hook) = app.input.as_mut() {
            hook(&InputHook {
                session,
                input: &gathered,
                ui,
                size,
                sender: &sender,
            });
        }
        let result = (|| {
            {
                let _dispatch = frame_trace::scope("dispatch");
                app.boundary(session, gathered)?;
            }
            session.views_mut().root_mut().eye.aspect = aspect;
            if was_faulted && session.faulted_phase().is_none() {
                timestep.reset_clock(now);
            }
            {
                let _simulation = frame_trace::scope("simulation");
                for _ in timestep.advance(now) {
                    session.tick()?;
                }
            }
            Ok(())
        })();
        input.reclaim(session.take_input());
        result
    }

    fn prepare(
        &mut self,
        now: Instant,
        size: (u32, u32),
        ui: Option<&loam_egui::egui::Context>,
    ) -> Result<(Publication<A>, Eye), HostError> {
        self.advance(now, size, ui)?;
        {
            let _publication = frame_trace::scope("publication");
            self.records
                .publish(&mut self.session)
                .map_err(|error| match error {
                    PublishError::Borrowed => {
                        HostError::Host("publication buffer is borrowed".into())
                    }
                    PublishError::Phase(error) => HostError::Phase(error),
                })?;
        }
        let root = self.session.views().root();
        let eye = self
            .session
            .views()
            .get(root)
            .map(|view| view.eye)
            .ok_or_else(|| HostError::Host("the root view is missing".into()))?;
        let records = self
            .records
            .lend()
            .ok_or_else(|| HostError::Host("the publication buffer is unavailable".into()))?;
        Ok((records, eye))
    }

    fn drive(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        mut finish: impl FnMut(&mut CommandEncoder),
        presenter: &mut Presenter,
        layer: Option<&DebugLayer>,
    ) -> Result<(), HostError> {
        let context = layer.map(DebugLayer::begin);
        let faulted_before = self.session.faulted_phase();
        let scene_before = self.session.scene();
        let terminal = match self.prepare(now, target.size, context.as_ref()) {
            Ok((records, eye)) => {
                {
                    let sender = self.app.commands.sender();
                    let (hook, captures, cursor) = (
                        self.app.frame.as_mut(),
                        &mut self.app.captures,
                        &mut self.cursor,
                    );
                    if let Some(hook) = hook {
                        hook(&mut FrameHook {
                            session: &self.session,
                            published: &records,
                            sections: presenter.sections(),
                            ui: context.as_ref(),
                            size: target.size,
                            sender: &sender,
                            capture: CaptureControl::new(captures),
                            cursor,
                        });
                    }
                }
                let _presentation = frame_trace::scope("presentation");
                #[cfg(test)]
                {
                    self.presented = Some(eye);
                }
                presenter.upload(
                    &gpu.device,
                    &gpu.queue,
                    &eye,
                    Vec2::new(target.size.0 as f32, target.size.1 as f32),
                    &records.views,
                );
                self.records.release(records);
                None
            }
            Err(error) if self.session.faulted_phase().is_some() => {
                if faulted_before.is_none() || self.session.scene() != scene_before {
                    tracing::error!("session frame failed: {error:?}");
                    self.cursor.suspend();
                }
                self.timestep.reset_clock(now);
                None
            }
            Err(error) => Some(error),
        };
        if let Some(context) = context.as_ref() {
            loam_egui::ConsoleUi::ui(self.app.console.ui_mut(), context);
        }
        if let Some(driver) = self.app.script.as_mut() {
            driver.advance_console(&mut self.app.console);
        }
        self.app.console.dispatch_pending();
        if let Some(layer) = layer {
            layer.finish();
        }
        if let Some(error) = terminal {
            return Err(error);
        }

        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let (wants_pre, wants_post) = self.capture_intent(now);
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("loam-app::session"),
            });
        presenter
            .record_scene(
                &gpu.device,
                &mut encoder,
                target.view,
                target.size,
                BACKGROUND,
            )
            .map_err(failed)?;
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let pre = if wants_pre {
            finish(&mut encoder);
            self.record_capture(&gpu.device, &mut encoder, target)
        } else {
            None
        };
        presenter
            .record_overlays(&mut encoder, target.view, target.size)
            .map_err(failed)?;
        finish(&mut encoder);
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        let post = wants_post
            .then(|| self.record_capture(&gpu.device, &mut encoder, target))
            .flatten();
        if let Some(layer) = layer {
            layer.take_callbacks(&mut self.callbacks);
        }
        gpu.queue
            .submit(self.callbacks.drain(..).chain(Some(encoder.finish())));
        presenter.after_submit();
        #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
        self.consume_capture(&gpu.device, now, pre, post);
        Ok(())
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    fn capture_intent(&mut self, now: Instant) -> (bool, bool) {
        if !self.app.captures.is_empty() {
            let requests = std::mem::take(&mut self.app.captures);
            for line in self.capture.apply_requests(requests) {
                tracing::info!("{line}");
            }
        }
        if !self.capture.should_capture(now) {
            return (false, false);
        }
        (self.capture.wants_pre(), self.capture.wants_post())
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    fn record_capture(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        target: &Target<'_>,
    ) -> Option<crate::capture::TextureReadback> {
        match crate::capture::record_texture_rgba(
            device,
            encoder,
            target.texture,
            target.size.0,
            target.size.1,
            target.format,
        ) {
            Ok(readback) => Some(readback),
            Err(error) => {
                tracing::error!("capture: copy failed: {error:#}");
                None
            }
        }
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    fn consume_capture(
        &mut self,
        device: &wgpu::Device,
        now: Instant,
        pre: Option<crate::capture::TextureReadback>,
        post: Option<crate::capture::TextureReadback>,
    ) {
        for (is_pre, readback) in [(true, pre), (false, post)] {
            let Some(readback) = readback else {
                continue;
            };
            match readback.read(device) {
                Ok(image) => {
                    if let Err(error) = self.capture.consume_frame(
                        is_pre,
                        image.rgba,
                        image.width,
                        image.height,
                        now,
                    ) {
                        tracing::error!("capture: write failed: {error:#}");
                    }
                }
                Err(error) => tracing::error!("capture: readback failed: {error:#}"),
            }
        }
        self.capture.advance_frame(now);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use glam::Vec3;
    use loam_math::EuclideanR3;
    use loam_render::device::FeatureRequest;
    use loam_render::pass::{
        FrameFormat, FramePass, FrameTarget, PassStage, ResourceId, SCENE_COLOR,
    };
    use loam_runtime::{
        ActionId, Bindings, Command, DomainBuilder, HostConfig, Identity3, Key, LogCapacity, Phase,
        Pose, SimConfig, SpawnBundle, Tick, ViewSpec,
    };
    use wgpu::{
        BackendOptions, Backends, Color, Extent3d, Instance, InstanceDescriptor, LoadOp,
        NoopBackendOptions, Operations, RenderPassColorAttachment, RenderPassDescriptor, StoreOp,
        TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };

    use super::*;
    use crate::args::Args;
    use crate::session::CursorPolicy;

    const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
    const SIZE: (u32, u32) = (64, 48);
    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Bare {}
    }

    const CAPTURE_FPS: u16 = 60;
    const SCENE: [ResourceId; 1] = [SCENE_COLOR];

    struct Probe {
        recorded: Arc<AtomicU32>,
        rebuilt: Arc<AtomicU32>,
    }

    impl FramePass for Probe {
        fn name(&self) -> &'static str {
            "probe"
        }

        fn stage(&self) -> PassStage {
            PassStage::Scene
        }

        fn record(
            &self,
            _encoder: &mut CommandEncoder,
            _target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            self.recorded.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }

        fn attach(&mut self, _gpu: &GpuContext, _frame: FrameFormat) -> anyhow::Result<()> {
            self.rebuilt.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    struct Paint {
        name: &'static str,
        color: Color,
        overlay: bool,
    }

    impl FramePass for Paint {
        fn name(&self) -> &'static str {
            self.name
        }

        fn writes(&self) -> &[ResourceId] {
            &SCENE
        }

        fn stage(&self) -> PassStage {
            if self.overlay {
                PassStage::Overlay
            } else {
                PassStage::Scene
            }
        }

        fn record(
            &self,
            encoder: &mut CommandEncoder,
            target: &FrameTarget<'_>,
        ) -> anyhow::Result<()> {
            let _pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some(self.name),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: target.color,
                    depth_slice: None,
                    resolve_target: None,
                    ops: Operations {
                        load: LoadOp::Clear(self.color),
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

    fn host<A: Stores>(name: &'static str) -> SessionApp<A> {
        SessionApp::with_args(HostConfig::new(name, Bindings::new()), Args::default())
            .debug_layer(false)
    }

    fn run_one<A: Stores>(frame: &mut Frame<A>, gpu: &GpuContext, texture: &Texture) {
        run_at(frame, gpu, texture, Instant::now());
    }

    fn run_at<A: Stores>(frame: &mut Frame<A>, gpu: &GpuContext, texture: &Texture, now: Instant) {
        let view = texture.create_view(&TextureViewDescriptor::default());
        let target = Target {
            view: &view,
            texture,
            format: FORMAT,
            size: SIZE,
        };
        frame
            .step(gpu, &target, now, |_| {})
            .expect("the frame stepped");
    }

    #[test]
    fn incompatible_dynamic_shader_reaches_the_host_with_pass_and_cause() {
        let source = r#"
struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
}
@group(0) @binding(0) var<storage, read> wrong: array<u32>;
@group(0) @binding(1) var<storage, read> bodies: array<u32>;
@vertex fn vs_fullscreen(@builtin(vertex_index) vertex: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vertex), 0.0, 0.0, 1.0);
}
@fragment fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(0.0);
}
@fragment fn fs_depth() -> Fragment {
    return Fragment(vec4<f32>(0.0), 0.5);
}
"#;
        let gpu = noop_gpu();
        let app =
            host("shader error").pass(Box::new(loam_render::HyperslicePass::new(source.into())));
        let mut frame = bare(app);
        let error = frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect_err("incompatible shader");
        let message = format!("{error:?}");
        assert!(message.contains("pass `hyperslice` failed during attach"));
        assert!(message.contains("WGSL binding 0:0 must use Uniform"));
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
    fn an_input_command_updates_the_eye_before_publication() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let published: Arc<Mutex<Option<Eye>>> = Arc::new(Mutex::new(None));
        let recorded = published.clone();
        let mut step = 0.0_f32;
        let app = host::<Bare>("camera")
            .on_input(move |hook| {
                step += 1.0;
                let eye = Eye::looking_at([step, 2.0, 3.0], [0.0; 3], [0.0, 1.0, 0.0]);
                hook.sender.app_fn("camera", move |dispatch| {
                    let aspect = dispatch.views.root_mut().eye.aspect;
                    dispatch.views.root_mut().eye = Eye { aspect, ..eye };
                });
            })
            .on_frame(move |hook| {
                let root = hook.session.views().root();
                let eye = hook.session.views().get(root).map(|view| view.eye);
                *recorded.lock().unwrap_or_else(|error| error.into_inner()) = eye;
            });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");

        run_one(&mut frame, &gpu, &texture);
        run_one(&mut frame, &gpu, &texture);

        let published = published
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .expect("the frame hook read the published eye");
        assert_eq!(
            frame.inner.presented,
            Some(published),
            "the presenter did not use the eye committed before publication"
        );
    }

    #[test]
    fn installed_reset_key_recovers_the_fault_and_restores_cursor_capture_once() {
        const RESET: ActionId = ActionId(0);

        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let recorded = Arc::new(AtomicU32::new(0));
        let filled = Arc::new(AtomicU32::new(0));
        let fills = filled.clone();
        let app = SessionApp::<Bare>::with_args(
            HostConfig::new(
                "fault recovery",
                Bindings::new().key(Key::Letter('r'), RESET),
            ),
            Args::default(),
        )
        .debug_layer(false)
        .recover_on_fault(RESET)
        .pass(Box::new(Probe {
            recorded: recorded.clone(),
            rebuilt: Arc::new(AtomicU32::new(0)),
        }))
        .on_frame(move |hook| {
            fills.fetch_add(1, Ordering::Relaxed);
            hook.capture_cursor(true, CursorPolicy::Toggle);
        });
        let mut session = Session::new(Bare::default(), SimConfig::default());
        let r3 = session
            .register_domain(DomainBuilder::new("r3", EuclideanR3).tracked(LogCapacity::default()));
        let root = session.views().root();
        let eye = session
            .dispatch(|dispatch| {
                let eye = dispatch.spawn(SpawnBundle::new().at(r3, Pose::at(Vec3::ZERO)))?;
                dispatch
                    .domains
                    .typed(r3)?
                    .add_view(ViewSpec::new(root, eye, Identity3));
                Ok::<_, loam_runtime::Rejection>(eye)
            })
            .expect("view");
        session.system(
            Phase::Dispatch,
            "authored reset",
            |ctx: loam_runtime::Ctx<'_, Bare>| {
                if ctx.input.pressed(RESET) {
                    ctx.commands.submit(Command::Reset);
                }
                Ok(())
            },
        );
        session.set_initial().expect("initial state");
        let epoch = session.scene().epoch();
        let mut frame = Frame::new(session, app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        let start = Instant::now();
        let expected_sections = ["present-clear", "triangles", "present-draw", "probe"];

        run_at(&mut frame, &gpu, &texture, start);
        assert_eq!(
            frame
                .presenter
                .as_ref()
                .expect("the presenter attached")
                .sections()
                .iter()
                .map(|section| section.name)
                .collect::<Vec<_>>(),
            expected_sections
        );
        assert_eq!(frame.take_cursor_request(), Some(true));
        frame.cursor_applied(true);
        frame.inner.app.sender().submit(Command::Despawn(eye));
        run_at(&mut frame, &gpu, &texture, start + Duration::from_millis(1));
        assert_eq!(
            frame
                .presenter
                .as_ref()
                .expect("the presenter attached")
                .sections()
                .iter()
                .map(|section| section.name)
                .collect::<Vec<_>>(),
            expected_sections
        );
        assert_eq!(
            frame.phase_error().map(|error| error.phase),
            Some(Phase::Publication)
        );
        assert_eq!(filled.load(Ordering::Relaxed), 1);
        assert_eq!(recorded.load(Ordering::Relaxed), 2);
        assert_eq!(frame.take_cursor_request(), Some(false));
        frame.cursor_applied(false);
        for millis in 2_u64..=4 {
            run_at(
                &mut frame,
                &gpu,
                &texture,
                start + Duration::from_millis(millis),
            );
            assert_eq!(
                frame
                    .presenter
                    .as_ref()
                    .expect("the presenter attached")
                    .sections()
                    .iter()
                    .map(|section| section.name)
                    .collect::<Vec<_>>(),
                expected_sections
            );
        }
        assert_eq!(recorded.load(Ordering::Relaxed), 5);

        frame.action(Key::Letter('r'), true, false);
        run_at(&mut frame, &gpu, &texture, start + Duration::from_secs(60));
        assert_eq!(recorded.load(Ordering::Relaxed), 6);
        assert_eq!(filled.load(Ordering::Relaxed), 2);

        assert_eq!(frame.phase_error(), None);
        assert_eq!(frame.inner.session.scene().epoch(), epoch.advance());
        assert_eq!(frame.inner.session.current_tick(), Tick(0));
        assert_eq!(filled.load(Ordering::Relaxed), 2);
        assert_eq!(frame.take_cursor_request(), Some(true));
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_secs(60) + Duration::from_millis(1),
        );
        assert_eq!(frame.inner.session.scene().epoch(), epoch.advance());
    }

    #[test]
    fn a_scripted_line_reaches_its_verb_on_the_frame_the_script_names() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("demo.script");
        std::fs::write(&path, "0 mark first\n2 mark second\n").expect("wrote the script");

        let marked: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = marked.clone();
        let args = Args::from_argv([format!("--script={}", path.display())]);
        let app = SessionApp::<Bare>::with_args(HostConfig::new("script", Bindings::new()), args)
            .debug_layer(false)
            .command("mark", "record a marker", move |args, _submit, _out| {
                recorded
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push(args.join(" "));
                Ok(())
            });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");

        let mut after_each = Vec::new();
        for _ in 0..3 {
            run_one(&mut frame, &gpu, &texture);
            after_each.push(
                marked
                    .lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .len(),
            );
        }

        assert_eq!(
            after_each,
            [1, 1, 2],
            "the host drove the script off its own frame index"
        );
        assert_eq!(
            *marked.lock().unwrap_or_else(|error| error.into_inner()),
            ["first", "second"]
        );
    }

    #[test]
    fn a_host_without_a_script_argument_holds_no_driver() {
        let app = SessionApp::<Bare>::with_args(
            HostConfig::new("no script", Bindings::new()),
            Args::from_argv(["--fps=30"]),
        );
        assert!(app.script.is_none());
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn a_capture_from_the_session_host_writes_the_targets_pixels_gpu_probe() {
        let instance = Instance::default();
        let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("a wgpu adapter");
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let app = host("capture")
            .capture(crate::capture::CaptureRequest::OneShot {
                stage: crate::capture::CaptureStage::Post,
                dir: Some(directory.path().to_path_buf()),
                name: Some("session".into()),
            })
            .expect("capture supported");
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

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn pre_capture_excludes_the_overlay_that_post_capture_includes_gpu_probe() {
        let instance = Instance::default();
        let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("a wgpu adapter");
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let app = host("capture stages")
            .pass(Box::new(Paint {
                name: "red scene",
                color: Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
                overlay: false,
            }))
            .pass(Box::new(Paint {
                name: "green overlay",
                color: Color {
                    r: 0.0,
                    g: 1.0,
                    b: 0.0,
                    a: 1.0,
                },
                overlay: true,
            }))
            .capture(crate::capture::CaptureRequest::OneShot {
                stage: crate::capture::CaptureStage::Both,
                dir: Some(directory.path().to_path_buf()),
                name: Some("stages".into()),
            })
            .expect("capture supported");
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        run_one(&mut frame, &gpu, &texture);

        let pre = ::image::open(directory.path().join("stages_pre.png"))
            .expect("the pre capture")
            .to_rgba8();
        let post = ::image::open(directory.path().join("stages_post.png"))
            .expect("the post capture")
            .to_rgba8();
        assert_eq!(pre.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(post.get_pixel(0, 0).0, [0, 255, 0, 255]);
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn a_hook_that_stops_its_capture_finishes_the_file_gpu_probe() {
        let instance = Instance::default();
        let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("a wgpu adapter");
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let into = directory.path().to_path_buf();
        let mut frames = 0_u32;
        let app = host::<Bare>("capture").on_frame(move |hook| {
            frames += 1;
            if frames == 1 {
                hook.capture
                    .start(crate::capture::CaptureRequest::StartSequence {
                        format: crate::capture::CaptureFormat::Apng,
                        stage: crate::capture::CaptureStage::Post,
                        dir: Some(into.clone()),
                        name: Some("session".into()),
                        fps: Some(CAPTURE_FPS),
                        scale: None,
                        palette: crate::capture::PaletteMode::default(),
                    })
                    .expect("capture supported");
            }
            if frames == 3 {
                hook.capture.stop().expect("capture supported");
            }
        });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, 1, None, SIZE, 1.0)
            .expect("attached");
        for _ in 0..4 {
            run_one(&mut frame, &gpu, &texture);
            std::thread::sleep(Duration::from_millis(1000 / u64::from(CAPTURE_FPS) + 4));
        }
        drop(frame);

        let written = directory.path().join("session.apng");
        let bytes = std::fs::read(&written).expect("the stopped capture finished its file");
        assert_eq!(
            animation_frames(&bytes),
            Some(2),
            "the file holds the frames drawn between the start and the stop"
        );
    }

    fn animation_frames(png: &[u8]) -> Option<u32> {
        let mut at = 8;
        while at + 12 <= png.len() {
            let length = u32::from_be_bytes(png[at..at + 4].try_into().ok()?) as usize;
            if &png[at + 4..at + 8] == b"acTL" {
                return Some(u32::from_be_bytes(png[at + 8..at + 12].try_into().ok()?));
            }
            at += 12 + length;
        }
        None
    }
}
