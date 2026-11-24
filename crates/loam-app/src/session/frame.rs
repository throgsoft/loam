#[cfg(not(target_arch = "wasm32"))]
use std::sync::Arc;

use glam::Vec2;
use web_time::Instant;
use wgpu::{CommandBuffer, CommandEncoder, Texture, TextureFormat, TextureView};
#[cfg(not(target_arch = "wasm32"))]
use winit::window::Window;

use loam_render::device::GpuContext;
use loam_render::pass::PublishedFrame;
use loam_render::present::Presenter;
use loam_runtime::{Eye, PhaseError, Publication, Session, Stores};
use loam_time::par::Executor;
use loam_time::{frame_trace, FixedTimestep};

use super::app::{CaptureControl, InputHook, Posts, PublishedHook, Redraw, SessionApp};
use super::console::SessionConsole;
use super::cursor::CursorCapture;
#[cfg(feature = "egui")]
use super::debug_layer::DebugLayer;
use super::debug_layer::LayerSlot;
use super::input::InputMap;
#[cfg(feature = "text")]
use super::text::ScreenText;
use crate::capture::Capture;
use crate::host::HostError;
#[cfg(not(target_arch = "wasm32"))]
use crate::host::{HostConfig, Key};
use crate::script::{driver_from_args, ScriptRun};

pub struct Target<'a> {
    pub view: &'a TextureView,
    pub texture: &'a Texture,
    pub format: TextureFormat,
    pub size: (u32, u32),
}

pub(crate) fn failed(error: impl std::fmt::Display) -> HostError {
    HostError::Host(error.to_string())
}

#[derive(Clone, Copy, Default, PartialEq)]
struct Shown {
    content: u64,
    eye: Eye,
    size: (u32, u32),
    background: wgpu::Color,
}

fn redraws(policy: Redraw, drawn: Option<&Shown>, next: &Shown, forced: bool) -> bool {
    policy == Redraw::EveryFrame || forced || drawn != Some(next)
}

fn warn_once<A: Stores>(reported: &mut bool, console: &mut SessionConsole<A>, note: &str) {
    if !std::mem::replace(reported, true) {
        tracing::warn!("{note}");
        console.note(note.to_string());
    }
}

struct LoopState<A: Stores> {
    session: Session<A>,
    executor: Box<dyn Executor>,
    publication: Publication<A::Published>,
    timestep: Option<FixedTimestep>,
    callbacks: Vec<CommandBuffer>,
    input: InputMap,
    cursor: CursorCapture,
    app: SessionApp<A>,
    posts: Posts,
    layer: LayerSlot,
    no_views_reported: bool,
    dropped_text_reported: bool,
    #[cfg(feature = "text")]
    text: ScreenText,
    shown: Shown,
    drawn: Option<Shown>,
    capture: Capture,
    script: Option<ScriptRun>,
}

pub(crate) struct Frame<A: Stores> {
    presenter: Option<Presenter>,
    state: LoopState<A>,
    trace: Option<frame_trace::Scope>,
}

impl<A: Stores> Frame<A> {
    pub(crate) fn new(
        session: Session<A>,
        app: SessionApp<A>,
        executor: Box<dyn Executor>,
    ) -> Result<Self, HostError> {
        #[cfg(feature = "text")]
        let (font, app) = {
            let mut app = app;
            (app.font.take(), app)
        };
        let sim = session.config();
        sim.check().map_err(failed)?;
        let timestep = sim
            .dt()
            .map(|_| FixedTimestep::new(sim.fixed_hz).with_max_catch_up(sim.max_ticks_per_frame));
        let script = driver_from_args(&app.args)
            .map_err(|error| failed(format!("{error:#}")))
            .and_then(|driver| {
                driver
                    .map(|driver| ScriptRun::new(driver, &app.console))
                    .transpose()
            })
            .unwrap_or_else(|error| {
                tracing::error!("--script ignored: {error}");
                None
            });
        Ok(Self {
            presenter: None,
            state: LoopState {
                session,
                executor,
                publication: Publication::default(),
                timestep,
                callbacks: Vec::new(),
                input: InputMap::default(),
                cursor: CursorCapture::new(),
                app,
                posts: Posts::default(),
                layer: LayerSlot::default(),
                no_views_reported: false,
                dropped_text_reported: false,
                #[cfg(feature = "text")]
                text: ScreenText::new(font),
                shown: Shown::default(),
                drawn: None,
                capture: Capture::new(),
                script,
            },
            trace: None,
        })
    }

    pub(crate) fn redraw(&self) -> Redraw {
        self.state.app.redraw
    }

    pub(crate) fn invalidate(&mut self) {
        self.state.drawn = None;
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn config(&self) -> &HostConfig {
        &self.state.app.config
    }

    pub(crate) fn app_mut(&mut self) -> &mut SessionApp<A> {
        &mut self.state.app
    }

    #[cfg(test)]
    pub(crate) fn phase_error(&self) -> Option<loam_runtime::PhaseError> {
        self.state.session.phase_error()
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn posts(&self) -> impl Iterator<Item = (&'static str, &[f32])> {
        self.state.posts.iter()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn input_mut(&mut self) -> &mut InputMap {
        &mut self.state.input
    }

    pub(crate) fn alt(&mut self, index: usize, pressed: bool) {
        self.state.cursor.alt(index, pressed);
    }

    pub(crate) fn focus(&mut self, focused: bool) {
        self.state.cursor.focus(focused);
        if !focused {
            self.state.input.release_all();
        }
    }

    pub(crate) fn cursor_applied(&mut self, locked: bool) {
        self.state.cursor.applied(locked);
        self.state.input.set_cursor_locked(locked);
    }

    pub(crate) fn cursor_locked(&self) -> bool {
        self.state.cursor.locked()
    }

    pub(crate) fn take_cursor_request(&mut self) -> Option<bool> {
        self.state.cursor.take_request()
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn apply_message(
        &mut self,
        message: &super::input::message::InputMessage,
        consumed: bool,
    ) {
        super::input::apply(
            &mut self.state.input,
            &self.state.app.config.bindings,
            message,
            consumed,
        );
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn observe_message(&mut self, message: &super::input::message::InputMessage) {
        use super::input::message::InputMessage;

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
                    self.state.cursor.released();
                }
            }
            _ => {}
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn action(&mut self, key: Key, pressed: bool, consumed: bool) {
        self.state
            .input
            .host_action(&self.state.app.config.bindings, key, pressed, consumed);
    }

    #[cfg(feature = "egui")]
    pub(crate) fn layer(&self) -> Option<&DebugLayer> {
        self.state.layer.get()
    }

    pub(crate) fn reset_clock(&mut self, now: Instant) {
        if let Some(timestep) = self.state.timestep.as_mut() {
            timestep.reset_clock(now);
        }
        self.state.app.pacer.reset();
    }

    pub(crate) fn attach(
        &mut self,
        gpu: &GpuContext,
        format: TextureFormat,
        #[cfg(not(target_arch = "wasm32"))] window: Option<Arc<Window>>,
        size: (u32, u32),
        scale: f32,
    ) -> Result<(), HostError> {
        let mut presenter = Presenter::new(format).map_err(failed)?;
        for pass in self.state.app.passes.drain(..) {
            presenter.register_pass(pass).map_err(failed)?;
        }
        #[cfg(feature = "text")]
        if let Some(text) = self.state.text.pass() {
            presenter.register_pass(Box::new(text)).map_err(failed)?;
        }
        if self.state.app.debug_layer {
            if let Some(pass) = self.state.layer.attach(
                gpu,
                format,
                #[cfg(not(target_arch = "wasm32"))]
                window,
                size,
                scale,
            ) {
                presenter.register_pass(pass).map_err(failed)?;
            }
        }
        presenter.attach(gpu).map_err(failed)?;
        self.presenter = Some(presenter);
        self.resize(size.0, size.1, scale);
        Ok(())
    }

    pub(crate) fn recover(&mut self, gpu: &GpuContext) -> Result<(), HostError> {
        self.invalidate();
        if let Some(presenter) = self.presenter.as_mut() {
            presenter.attach(gpu).map_err(failed)?;
        }
        Ok(())
    }

    pub(crate) fn resize(&mut self, width: u32, height: u32, scale: f32) {
        self.invalidate();
        self.state.input.resize(width, height, scale);
        self.state.layer.resize(width, height, scale);
    }

    pub(crate) fn step(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        finish: impl FnMut(&mut CommandEncoder),
    ) -> Result<(), HostError> {
        if self.update(gpu, target.size, now)? {
            self.draw(gpu, target, now, finish)?;
        }
        Ok(())
    }

    pub(crate) fn update(
        &mut self,
        gpu: &GpuContext,
        size: (u32, u32),
        now: Instant,
    ) -> Result<bool, HostError> {
        self.state.posts.clear();
        self.end_trace();
        let polled = gpu
            .device
            .poll(wgpu::PollType::Poll)
            .map_err(failed)
            .and_then(|_| {
                gpu.take_uncaptured_error()
                    .map_or(Ok(()), |error| Err(failed(error)))
            });
        let outcome = match (polled, self.presenter.as_mut()) {
            (Err(error), _) => Err(error),
            (Ok(()), Some(presenter)) if size.0 > 0 && size.1 > 0 => {
                frame_trace::begin_frame();
                self.trace = Some(frame_trace::scope("frame"));
                self.state.present(gpu, size, now, presenter)
            }
            (Ok(()), _) => Ok(false),
        };
        if !matches!(outcome, Ok(true)) {
            self.end_trace();
        }
        if let Err(error) = &outcome {
            tracing::error!("frame failed: {error}");
        }
        outcome
    }

    pub(crate) fn draw(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        finish: impl FnMut(&mut CommandEncoder),
    ) -> Result<(), HostError> {
        let outcome = match self.presenter.as_mut() {
            Some(presenter) => self.state.draw(gpu, target, now, finish, presenter),
            None => Ok(()),
        };
        self.end_trace();
        if let Err(error) = &outcome {
            tracing::error!("frame failed: {error}");
        }
        outcome
    }

    fn end_trace(&mut self) {
        if self.trace.is_some() {
            self.trace = None;
            frame_trace::end_frame();
        }
    }
}

// `simulate` runs the ticks and learns whether this boundary recovered a fault.
pub(crate) fn advance_session<A: Stores>(
    session: &mut Session<A>,
    app: &mut SessionApp<A>,
    input: &mut InputMap,
    cursor: &mut CursorCapture,
    size: (u32, u32),
    layer: &LayerSlot,
    simulate: impl FnOnce(&mut Session<A>, bool) -> Result<(), PhaseError>,
) -> Result<(), HostError> {
    let aspect = input
        .display_aspect()
        .unwrap_or(size.0 as f32 / size.1 as f32);
    session.views_mut().root_mut().eye.aspect = aspect;
    let gathered = input.take();
    let sender = app.commands.sender();
    let was_faulted = session.faulted_phase().is_some();
    if app
        .fault_recovery
        .is_some_and(|action| was_faulted && gathered.pressed(action))
    {
        sender.reset();
    }
    if let Some(hook) = app.input.as_mut() {
        hook(&InputHook {
            session,
            input: &gathered,
            #[cfg(feature = "egui")]
            ui: layer.ui(),
            size,
            sender: &sender,
        });
    }
    let result = (|| -> Result<(), HostError> {
        {
            let _dispatch = frame_trace::scope("dispatch");
            app.boundary(session, gathered)?;
        }
        session.views_mut().root_mut().eye.aspect = aspect;
        {
            let _simulation = frame_trace::scope("simulation");
            let recovered = was_faulted && session.faulted_phase().is_none();
            simulate(session, recovered)?;
        }
        session.views_mut().root_mut().eye.aspect = aspect;
        Ok(())
    })();
    input.reclaim(session.take_input());
    result?;
    let views = session.views();
    let wanted = views.root().rig.cursor();
    cursor.follow_rig(wanted.filter(|_| !layer.typing()));
    Ok(())
}

impl<A: Stores> LoopState<A> {
    fn advance(&mut self, now: Instant, size: (u32, u32)) -> Result<(), HostError> {
        let controls = self.app.console.take_controls();
        if let Some(fps) = controls.target_fps {
            self.app.pacer.set_target_fps(fps);
        }
        if controls.vsync.is_some() {
            self.app.vsync = controls.vsync;
        }
        let LoopState {
            session,
            executor,
            input,
            cursor,
            timestep,
            app,
            layer,
            ..
        } = self;
        advance_session(
            session,
            app,
            input,
            cursor,
            size,
            layer,
            |session, recovered| {
                if let Some(timestep) = timestep.as_mut() {
                    if recovered {
                        timestep.reset_clock(now);
                    }
                    for _ in timestep.advance(now) {
                        session.tick(executor.as_ref())?;
                    }
                }
                Ok::<_, PhaseError>(())
            },
        )
    }

    fn publish(&mut self, size: (u32, u32)) -> Result<Eye, HostError> {
        {
            let _publication = frame_trace::scope("publication");
            let alpha = self.timestep.as_ref().map_or(1.0, FixedTimestep::alpha);
            self.session.publish(&mut self.publication, alpha)?;
        }
        self.session.views_mut().root_mut().eye.aspect = self
            .input
            .display_aspect()
            .unwrap_or(size.0 as f32 / size.1 as f32);
        let eye = self.session.views().root().eye;
        if self.publication.views.is_empty() {
            warn_once(
                &mut self.no_views_reported,
                &mut self.app.console,
                "the session published no views, so nothing is drawn",
            );
        }
        Ok(eye)
    }

    fn present(
        &mut self,
        gpu: &GpuContext,
        size: (u32, u32),
        now: Instant,
        presenter: &mut Presenter,
    ) -> Result<bool, HostError> {
        self.layer.begin();
        let faulted_before = self.session.faulted_phase();
        let scene_before = self.session.scene();
        let published = self.advance(now, size).and_then(|()| self.publish(size));
        let mut requested = false;
        let terminal = match published {
            Ok(eye) => {
                let records = &self.publication;
                {
                    let sender = self.app.commands.sender();
                    if let Some(hook) = self.app.published.as_mut() {
                        hook(&mut PublishedHook {
                            published: records,
                            eye,
                            sections: presenter.sections(),
                            #[cfg(feature = "egui")]
                            ui: self.layer.ui(),
                            size,
                            sender: &sender,
                            capture: CaptureControl::new(&mut self.app.captures),
                            cursor: &mut self.cursor,
                            background: &mut self.app.background,
                            posts: &mut self.posts,
                            redraw: &mut requested,
                        });
                    }
                }
                let _presentation = frame_trace::scope("presentation");
                let (text_origin_px, pixels_per_point) = self.layer.seat(self.input.scale());
                presenter.upload(
                    &gpu.device,
                    &gpu.queue,
                    &PublishedFrame {
                        eye,
                        viewport_px: Vec2::new(size.0 as f32, size.1 as f32),
                        pixels_per_point,
                        text_origin_px,
                        views: &records.views,
                        overlay: &records.overlay,
                        backdrop: records.backdrop,
                    },
                );
                #[cfg(feature = "text")]
                let dropped = self.text.drops(&records.overlay);
                #[cfg(not(feature = "text"))]
                let dropped = records.overlay.text().next().is_some();
                if dropped {
                    warn_once(
                        &mut self.dropped_text_reported,
                        &mut self.app.console,
                        "the session published text, but this build has no font or no text feature, so none is drawn",
                    );
                }
                self.shown.content = records.content();
                self.shown.eye = eye;
                None
            }
            Err(error) if self.session.faulted_phase().is_some() => {
                if faulted_before.is_none() || self.session.scene() != scene_before {
                    tracing::error!("session frame failed: {error}");
                    self.cursor.suspend();
                }
                if let Some(timestep) = self.timestep.as_mut() {
                    timestep.reset_clock(now);
                }
                None
            }
            Err(error) => Some(error),
        };
        let layered = self.layer.finish(&mut self.app.console);
        let scripted = self.script.as_mut().is_some_and(|script| {
            let app = &mut self.app;
            match script.advance(&mut self.input, &app.config.bindings, &mut app.console) {
                Ok((_, ran)) => ran,
                Err(error) => {
                    tracing::warn!("{error}");
                    true
                }
            }
        });
        let dispatched = self.app.console.dispatch_pending() || scripted;
        if let Some(error) = terminal {
            return Err(error);
        }
        let capturing = self.capture.pending(&self.app.captures, now);
        self.shown.size = size;
        self.shown.background = self.app.background;
        let draw = redraws(
            self.app.redraw,
            self.drawn.as_ref(),
            &self.shown,
            requested || dispatched || layered || capturing,
        );
        // Set again only after submit, so a failed acquire keeps the draw pending.
        if draw {
            self.drawn = None;
        }
        Ok(draw)
    }

    fn draw(
        &mut self,
        gpu: &GpuContext,
        target: &Target<'_>,
        now: Instant,
        mut finish: impl FnMut(&mut CommandEncoder),
        presenter: &mut Presenter,
    ) -> Result<(), HostError> {
        let mut shot = self.capture.begin(&mut self.app.captures, now);
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
                self.app.background,
            )
            .map_err(failed)?;
        if shot.wants(true) {
            finish(&mut encoder);
            self.capture
                .record(&mut shot, true, &gpu.device, &mut encoder, target);
        }
        presenter
            .record_overlays(&mut encoder, target.view, target.size)
            .map_err(failed)?;
        finish(&mut encoder);
        self.capture
            .record(&mut shot, false, &gpu.device, &mut encoder, target);
        self.layer.take_callbacks(&mut self.callbacks);
        gpu.queue
            .submit(self.callbacks.drain(..).chain(Some(encoder.finish())));
        presenter.after_submit();
        crate::trace::record_presentation(presenter.sections(), presenter.uploads());
        self.capture.finish(shot, &gpu.device, now);
        self.drawn = Some(self.shown);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use glam::Vec4;
    use loam_math::EuclideanR4;
    use loam_render::device::FeatureRequest;
    use loam_render::pass::{FrameFormat, FramePass, FrameTarget, PassStage};
    use loam_runtime::{
        ActionId, Command, DomainBuilder, FreeCamera, LogCapacity, Material, Phase, PointerPhase,
        Pose, PreparedGeometry, Rig, Section4, SimConfig, SpawnBundle, Tick, ViewSpec,
    };
    use wgpu::{
        BackendOptions, Backends, Extent3d, Instance, InstanceDescriptor, NoopBackendOptions,
        TextureDescriptor, TextureDimension, TextureUsages, TextureViewDescriptor,
    };

    use loam_time::par::Sequential;

    use super::*;
    use crate::args::Args;
    use crate::host::Bindings;
    use loam_runtime::CursorPolicy;

    const FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
    const SIZE: (u32, u32) = (64, 48);
    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Bare {}
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    const CAPTURE_FPS: u16 = 60;

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
            &mut self,
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
        Frame::new(
            Session::new(Bare::default(), SimConfig::default()),
            app,
            Box::new(Sequential),
        )
        .expect("the frame accepted the simulation config")
    }

    fn host<A: Stores>(name: &'static str) -> SessionApp<A> {
        quiet(SessionApp::with_args(
            HostConfig::new(name, Bindings::new()),
            Args::default(),
        ))
    }

    fn quiet<A: Stores>(app: SessionApp<A>) -> SessionApp<A> {
        app.debug_layer(false)
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

    fn world(moves: bool) -> Session<Bare> {
        let mut session = Session::new(Bare::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.0; 4], [1.0, 0.0, 0.0, 0.0]]],
        });
        let material = session.add_material(Material::lines([1.0; 4], 1.0));
        let root = session.views().root_id();
        let body = session
            .dispatch(|dispatch| {
                let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::Z * 4.0)))?;
                dispatch.domains.typed(r4)?.add_view(ViewSpec::new(
                    root,
                    eye,
                    Section4 { w: 0.0 },
                ))?;
                dispatch.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::ZERO))
                        .instance(loam_runtime::Instance::new(geometry, material)),
                )
            })
            .expect("the world spawned");
        session.system(
            Phase::Simulation,
            "drift",
            move |ctx: loam_runtime::Ctx<'_, Bare>| -> Result<(), loam_runtime::DomainError> {
                if moves {
                    let x = (ctx.step.tick.0 + 1) as f32;
                    ctx.domains
                        .typed(r4)?
                        .set_pose(body, Pose::at(Vec4::X * x))?;
                }
                Ok(())
            },
        );
        session
    }

    fn on_change(
        session: Session<Bare>,
        app: SessionApp<Bare>,
        gpu: &GpuContext,
    ) -> (Frame<Bare>, Arc<AtomicU32>) {
        let recorded = Arc::new(AtomicU32::new(0));
        let app = app.redraw(Redraw::OnChange).pass(Box::new(Probe {
            recorded: recorded.clone(),
            rebuilt: Arc::new(AtomicU32::new(0)),
        }));
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        (frame, recorded)
    }

    #[test]
    fn on_change_content_or_scale() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let hooked = Arc::new(AtomicU32::new(0));
        let hooks = hooked.clone();
        let app = host("still").on_published(move |_| {
            hooks.fetch_add(1, Ordering::Relaxed);
        });
        let (mut frame, recorded) = on_change(world(false), app, &gpu);
        let start = Instant::now();
        for millis in [0, 20, 25] {
            run_at(
                &mut frame,
                &gpu,
                &texture,
                start + Duration::from_millis(millis),
            );
        }
        assert_eq!(recorded.load(Ordering::Relaxed), 1);
        assert_eq!(hooked.load(Ordering::Relaxed), 3);
        frame.resize(SIZE.0, SIZE.1, 2.0);
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_millis(30),
        );
        assert_eq!(recorded.load(Ordering::Relaxed), 2);

        let (mut frame, recorded) = on_change(world(true), host("moving"), &gpu);
        for millis in [0, 20, 25] {
            run_at(
                &mut frame,
                &gpu,
                &texture,
                start + Duration::from_millis(millis),
            );
        }
        assert_eq!(recorded.load(Ordering::Relaxed), 3);
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    fn on_change_capture_request() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let mut frames = 0_u32;
        let app = host("still").on_published(move |hook| {
            frames += 1;
            if frames == 3 {
                hook.capture.stop().expect("capture supported");
            }
        });
        let (mut frame, recorded) = on_change(world(false), app, &gpu);
        let start = Instant::now();
        for (millis, drawn) in [(0, 1), (20, 1), (25, 2), (30, 2)] {
            run_at(
                &mut frame,
                &gpu,
                &texture,
                start + Duration::from_millis(millis),
            );
            assert_eq!(
                recorded.load(Ordering::Relaxed),
                drawn,
                "draws after the frame at {millis} ms"
            );
        }
    }

    #[test]
    fn pass_after_device_recovery() {
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
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
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
    fn post_across_frames() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let mut frames = 0_u32;
        let app = host::<Bare>("posts").on_published(move |hook| {
            frames += 1;
            if frames == 1 {
                hook.post("rect", &[1.0, 2.0, 3.0, 4.0]);
                hook.post("ready", &[]);
            }
        });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");

        run_one(&mut frame, &gpu, &texture);
        let posted: Vec<(&str, Vec<f32>)> = frame
            .posts()
            .map(|(topic, values)| (topic, values.to_vec()))
            .collect();
        assert_eq!(
            posted,
            [("rect", vec![1.0, 2.0, 3.0, 4.0]), ("ready", Vec::new())]
        );

        run_one(&mut frame, &gpu, &texture);
        assert_eq!(frame.posts().count(), 0);
    }

    #[test]
    fn frame_between_ticks() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let mut session = Session::new(Bare::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let geometry = session.prepare(PreparedGeometry::Lines4 {
            segments: vec![[[0.1, 0.0, 0.0, 0.0], [-0.1, 0.0, 0.0, 0.0]]],
        });
        let material = session.add_material(Material::flat([1.0; 4]));
        let root = session.views().root_id();
        let body = session
            .dispatch(|dispatch| {
                let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::Z * 4.0)))?;
                dispatch.domains.typed(r4)?.add_view(ViewSpec::new(
                    root,
                    eye,
                    Section4 { w: 0.0 },
                ))?;
                dispatch.spawn(
                    SpawnBundle::new()
                        .at(r4, Pose::at(Vec4::ZERO))
                        .instance(loam_runtime::Instance::new(geometry, material)),
                )
            })
            .expect("the body spawned");
        session.system(
            Phase::Simulation,
            "move",
            move |ctx: loam_runtime::Ctx<'_, Bare>| {
                if ctx.step.tick == Tick(0) {
                    ctx.domains.typed(r4)?.set_pose(body, Pose::at(Vec4::X))?;
                }
                Ok(())
            },
        );
        let mut frame = Frame::new(session, host("alpha"), Box::new(Sequential))
            .expect("the frame accepted the config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        let start = Instant::now();
        let dt = Duration::from_nanos(1_000_000_000 / u64::from(SimConfig::default().fixed_hz));
        run_at(&mut frame, &gpu, &texture, start);
        run_at(&mut frame, &gpu, &texture, start + dt * 3 / 2);

        let publication = &frame.state.publication;
        let x = publication.views[0].records.instances.rows()[0]
            .pose
            .coordinates[0];
        assert!((x - 0.5).abs() < 1e-3, "the body was drawn at x = {x}");
    }

    #[test]
    fn publication_eye_write() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let app = host::<Bare>("aspect");
        let mut session = Session::new(Bare::default(), SimConfig::default());
        session.register_domain(DomainBuilder::new("r4", EuclideanR4).untracked());
        session.system(
            Phase::Publication,
            "camera",
            |ctx: loam_runtime::Ctx<'_, Bare>| {
                ctx.views.root_mut().eye =
                    Eye::looking_at([0.0, 2.0, 5.0], [0.0; 3], [0.0, 1.0, 0.0]);
                Ok(())
            },
        );
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        run_at(&mut frame, &gpu, &texture, Instant::now());

        let aspect = frame.state.session.views().root().eye.aspect;
        assert!(
            (aspect - SIZE.0 as f32 / SIZE.1 as f32).abs() <= 1e-6,
            "a Publication-phase eye write published aspect {aspect}"
        );
    }

    #[test]
    fn reset_key_after_fault() {
        const RESET: ActionId = ActionId("reset");

        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let recorded = Arc::new(AtomicU32::new(0));
        let filled = Arc::new(AtomicU32::new(0));
        let fills = filled.clone();
        let app = quiet(SessionApp::<Bare>::with_args(
            HostConfig::new(
                "fault recovery",
                Bindings::new().key(Key::Letter('r'), RESET),
            ),
            Args::default(),
        ))
        .recover_on_fault(RESET)
        .pass(Box::new(Probe {
            recorded: recorded.clone(),
            rebuilt: Arc::new(AtomicU32::new(0)),
        }))
        .on_published(move |hook| {
            fills.fetch_add(1, Ordering::Relaxed);
            hook.capture_cursor(true, CursorPolicy::Toggle);
        });
        let mut session = Session::new(Bare::default(), SimConfig::default());
        let r4 = session
            .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
        let root = session.views().root_id();
        session
            .dispatch(|dispatch| {
                let eye = dispatch.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
                dispatch.domains.typed(r4)?.add_view(ViewSpec::new(
                    root,
                    eye,
                    Section4 { w: 0.0 },
                ))?;
                Ok::<_, loam_runtime::Rejection>(())
            })
            .expect("view");
        let armed = Arc::new(AtomicBool::new(false));
        let trip = armed.clone();
        session.system(
            Phase::Publication,
            "trip",
            move |_ctx: loam_runtime::Ctx<'_, Bare>| {
                if trip.swap(false, Ordering::Relaxed) {
                    return Err(loam_runtime::DomainError::Unsupported("injected"));
                }
                Ok(())
            },
        );
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
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        let start = Instant::now();

        run_at(&mut frame, &gpu, &texture, start);
        assert_eq!(frame.take_cursor_request(), Some(true));
        frame.cursor_applied(true);
        armed.store(true, Ordering::Relaxed);
        run_at(&mut frame, &gpu, &texture, start + Duration::from_millis(1));
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
        }
        assert_eq!(recorded.load(Ordering::Relaxed), 5);

        frame.action(Key::Letter('r'), true, false);
        run_at(&mut frame, &gpu, &texture, start + Duration::from_secs(60));
        assert_eq!(recorded.load(Ordering::Relaxed), 6);
        assert_eq!(filled.load(Ordering::Relaxed), 2);

        assert_eq!(frame.phase_error(), None);
        assert_eq!(frame.state.session.scene().epoch(), epoch.advance());
        assert_eq!(frame.state.session.current_tick(), Tick(0));
        assert_eq!(filled.load(Ordering::Relaxed), 2);
        assert_eq!(frame.take_cursor_request(), Some(true));
        run_at(
            &mut frame,
            &gpu,
            &texture,
            start + Duration::from_secs(60) + Duration::from_millis(1),
        );
        assert_eq!(frame.state.session.scene().epoch(), epoch.advance());
    }

    #[test]
    fn cursor_request_in_free_rig() {
        const RESET: ActionId = ActionId("reset");
        const LAND: ActionId = ActionId("land");

        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let own = Arc::new(AtomicBool::new(false));
        let owns = own.clone();
        let app = quiet(SessionApp::<Bare>::with_args(
            HostConfig::new(
                "rig cursor",
                Bindings::new()
                    .key(Key::Letter('r'), RESET)
                    .key(Key::Letter('l'), LAND),
            ),
            Args::default(),
        ))
        .recover_on_fault(RESET)
        .on_published(move |hook| {
            if owns.load(Ordering::Relaxed) {
                hook.capture_cursor(true, CursorPolicy::Toggle);
            }
        });
        let mut session = Session::new(Bare::default(), SimConfig::default());
        session.set_rig(Rig::Free(FreeCamera::default()));
        session.on_action(LAND, |dispatch| {
            dispatch.views.root_mut().rig = Rig::Manual;
            Ok(())
        });
        let armed = Arc::new(AtomicBool::new(false));
        let trip = armed.clone();
        session.system(
            Phase::Publication,
            "trip",
            move |_ctx: loam_runtime::Ctx<'_, Bare>| {
                if trip.swap(false, Ordering::Relaxed) {
                    return Err(loam_runtime::DomainError::Unsupported("injected"));
                }
                Ok(())
            },
        );
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        let start = Instant::now();
        let mut millis = 0;
        let mut run = |frame: &mut Frame<Bare>| {
            millis += 1;
            run_at(frame, &gpu, &texture, start + Duration::from_millis(millis));
            frame.take_cursor_request()
        };

        assert_eq!(
            run(&mut frame),
            Some(true),
            "a free rig set before launch never asked for the cursor"
        );
        frame.cursor_applied(true);
        assert_eq!(run(&mut frame), None);
        frame.alt(0, true);
        assert_eq!(frame.take_cursor_request(), Some(false));
        frame.cursor_applied(false);
        frame.alt(0, false);
        assert_eq!(
            run(&mut frame),
            None,
            "the rig took back a cursor the user freed with Alt"
        );

        armed.store(true, Ordering::Relaxed);
        assert_eq!(run(&mut frame), Some(false));
        assert_eq!(
            frame.phase_error().map(|error| error.phase),
            Some(Phase::Publication)
        );
        frame.action(Key::Letter('r'), true, false);
        assert_eq!(
            run(&mut frame),
            Some(true),
            "the recovered free rig did not ask for the cursor again"
        );
        assert_eq!(frame.phase_error(), None);
        frame.cursor_applied(true);

        frame.action(Key::Letter('l'), true, false);
        assert_eq!(
            run(&mut frame),
            Some(false),
            "the cursor stayed captured after the program left the free rig"
        );
        frame.cursor_applied(false);
        assert_eq!(run(&mut frame), None);

        own.store(true, Ordering::Relaxed);
        assert_eq!(run(&mut frame), Some(true));
        frame.cursor_applied(true);
        for _ in 0..2 {
            assert_eq!(
                run(&mut frame),
                None,
                "the shell re-issued or released a capture the frame hook holds under a manual rig"
            );
        }
    }

    #[test]
    fn scripted_line() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("demo.script");
        std::fs::write(&path, "0 mark first\n2 mark second\n").expect("wrote the script");

        let marked: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded = marked.clone();
        let args = Args::from_argv([format!("--script={}", path.display())]);
        let app = quiet(SessionApp::<Bare>::with_args(
            HostConfig::new("script", Bindings::new()),
            args,
        ))
        .command("mark", "record a marker", move |args, _submit, _out| {
            recorded
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(args.join(" "));
            Ok(())
        });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
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
    fn scripted_input_lines() {
        const MARK: ActionId = ActionId("mark");

        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("input.script");
        std::fs::write(&path, "0 key KeyM down\n0 pointer 0.25 0.5 down\n")
            .expect("wrote the script");

        let args = Args::from_argv([format!("--script={}", path.display())]);
        let app = quiet(SessionApp::<Bare>::with_args(
            HostConfig::new("script", Bindings::new().key(Key::Letter('m'), MARK)),
            args,
        ));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let saw = seen.clone();
        let mut session = Session::new(Bare::default(), SimConfig::default());
        session.system(
            Phase::Dispatch,
            "watch",
            move |ctx: loam_runtime::Ctx<'_, Bare>| {
                let began = ctx
                    .input
                    .pointers
                    .iter()
                    .find(|pointer| pointer.phase == PointerPhase::Began)
                    .map(|pointer| pointer.ndc);
                saw.lock()
                    .unwrap_or_else(|error| error.into_inner())
                    .push((ctx.input.pressed(MARK), began));
                Ok(())
            },
        );
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");

        for _ in 0..2 {
            run_one(&mut frame, &gpu, &texture);
        }

        assert_eq!(
            *seen.lock().unwrap_or_else(|error| error.into_inner()),
            [(false, None), (true, Some([0.25, 0.5]))],
            "the boundary after the lines' frame did not get their key and pointer"
        );
    }

    struct Refuse;

    impl loam_runtime::AppCommand<Bare> for Refuse {
        fn name(&self) -> &'static str {
            "reject"
        }

        fn apply(
            &mut self,
            _dispatch: &mut loam_runtime::Dispatch<'_, Bare>,
        ) -> Result<loam_runtime::Outcome, loam_runtime::Rejection> {
            Err(loam_runtime::Rejection::Unsupported("refused"))
        }
    }

    #[test]
    fn scripted_refusals() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let path = directory.path().join("refused.script");
        std::fs::write(&path, "0 strict\n0 mark first\n0 reject\n2 mark second\n")
            .expect("wrote the script");

        let marked = Arc::new(AtomicU32::new(0));
        let marks = marked.clone();
        let args = Args::from_argv([format!("--script={}", path.display())]);
        let app = quiet(SessionApp::<Bare>::with_args(
            HostConfig::new("script", Bindings::new()),
            args,
        ))
        .command("mark", "count a marker", move |_args, _submit, _out| {
            marks.fetch_add(1, Ordering::Relaxed);
            Ok(())
        })
        .command("strict", "refuse at dispatch", |_args, _submit, _out| {
            anyhow::bail!("usage: strict <n>")
        })
        .command("reject", "refuse at the boundary", |_args, submit, _out| {
            submit.app(Refuse);
            Ok(())
        });
        let mut frame = bare(app);
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");

        let mut after_each = Vec::new();
        for _ in 0..3 {
            run_one(&mut frame, &gpu, &texture);
            after_each.push(marked.load(Ordering::Relaxed));
        }

        assert_eq!(
            after_each,
            [1, 1, 2],
            "a refused line stopped the lines after it or moved them off their frames"
        );
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
    #[test]
    #[ignore = "requires a working wgpu adapter; run with --include-ignored"]
    fn hook_stopped_capture_gpu_probe() {
        let instance = Instance::default();
        let gpu = pollster::block_on(GpuContext::new(instance, FeatureRequest::default(), None))
            .expect("a wgpu adapter");
        let texture = offscreen(&gpu);
        let directory = tempfile::tempdir().expect("temp dir");
        let into = directory.path().to_path_buf();
        let mut frames = 0_u32;
        let app = host::<Bare>("capture").on_published(move |hook| {
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
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
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

    loam_runtime::stores! {
        #[derive(Default)]
        pub struct Valued {}
        publishes u32;
    }

    #[test]
    fn on_published_hook() {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let mut session = Session::new(Valued::default(), SimConfig::default());
        let mut publishes = 0_u32;
        session.publish_value("publishes", move |_, value: &mut u32| {
            publishes += 1;
            *value = publishes;
            Ok(())
        });
        let seen: Arc<Mutex<Vec<u32>>> = Arc::default();
        let read = seen.clone();
        let app = host::<Valued>("value").on_published(move |hook| {
            read.lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(hook.published.value);
        });
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");

        run_one(&mut frame, &gpu, &texture);
        run_one(&mut frame, &gpu, &texture);

        assert_eq!(
            *seen.lock().unwrap_or_else(|error| error.into_inner()),
            [1, 2],
            "the hook did not read each frame's published value"
        );
    }

    #[cfg(not(feature = "egui"))]
    fn notes_after_three_frames(labelled: bool, app: SessionApp<Bare>) -> usize {
        let gpu = noop_gpu();
        let texture = offscreen(&gpu);
        let mut session = Session::new(Bare::default(), SimConfig::default());
        session.system(Phase::Publication, "label", move |ctx| {
            if labelled {
                ctx.overlay.push_text(
                    "score",
                    loam_runtime::TextRecord {
                        origin_pt: [8.0, 16.0],
                        size_pt: 16.0,
                        color: [1.0; 4],
                    },
                );
            }
            Ok(())
        });
        let mut frame = Frame::new(session, app, Box::new(Sequential))
            .expect("the frame accepted the simulation config");
        frame
            .attach(&gpu, FORMAT, None, SIZE, 1.0)
            .expect("attached");
        for _ in 0..3 {
            run_one(&mut frame, &gpu, &texture);
        }
        frame.state.app.console.ui().history().len()
    }

    #[cfg(not(feature = "egui"))]
    #[test]
    fn text_without_font() {
        let silent = notes_after_three_frames(false, host("plain"));
        assert_eq!(
            notes_after_three_frames(true, host("fontless")),
            silent + 1,
            "three frames of dropped text did not leave exactly one note"
        );
        #[cfg(feature = "text")]
        {
            let font = include_bytes!("../../../examples/fonts/lmroman10-bold.otf").to_vec();
            assert_eq!(
                notes_after_three_frames(true, host("font").font(font)),
                silent,
                "text drawn with the app's font was reported as dropped"
            );
        }
    }

    #[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
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
