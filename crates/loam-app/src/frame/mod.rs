pub(crate) mod cursor;
#[cfg(feature = "egui")]
pub mod debug_layer;
#[cfg(not(feature = "egui"))]
#[path = "debug_layer_off.rs"]
pub(crate) mod debug_layer;
pub mod pacing;
pub(crate) mod surface;
#[cfg(feature = "text")]
mod text;

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

use crate::app::{CaptureControl, InputHook, Posts, PublishedHook, Redraw, SessionApp};
use crate::capture::Capture;
use crate::console::SessionConsole;
use crate::headless::script::{driver_from_args, ScriptRun};
use crate::host::HostError;
#[cfg(not(target_arch = "wasm32"))]
use crate::host::{HostConfig, Key};
use crate::input::InputMap;
use cursor::CursorCapture;
#[cfg(feature = "egui")]
use debug_layer::DebugLayer;
use debug_layer::LayerSlot;
#[cfg(feature = "text")]
use text::ScreenText;

pub struct Target<'a> {
    pub view: &'a TextureView,
    pub texture: &'a Texture,
    pub format: TextureFormat,
    pub size: (u32, u32),
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
        sim.check()?;
        let timestep = sim
            .dt()
            .map(|_| FixedTimestep::new(sim.fixed_hz).with_max_catch_up(sim.max_ticks_per_frame));
        let script = driver_from_args(&app.args)
            .and_then(|driver| {
                driver
                    .map(|driver| ScriptRun::new(driver, &app.console))
                    .transpose()
            })
            .unwrap_or_else(|error| {
                tracing::error!("--script ignored: {error}");
                None
            });
        // A program that draws through its own pass publishes no views on purpose.
        let draws_itself = !app.passes.is_empty();
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
                no_views_reported: draws_itself,
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
        message: &crate::input::message::InputMessage,
        consumed: bool,
    ) {
        crate::input::apply(
            &mut self.state.input,
            &self.state.app.config.bindings,
            message,
            consumed,
        );
    }

    #[cfg(target_arch = "wasm32")]
    pub(crate) fn observe_message(&mut self, message: &crate::input::message::InputMessage) {
        use crate::input::message::InputMessage;

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
        view_format: Option<TextureFormat>,
        #[cfg(not(target_arch = "wasm32"))] window: Option<Arc<Window>>,
        size: (u32, u32),
        scale: f32,
    ) -> Result<(), HostError> {
        let mut presenter = Presenter::new(format)?;
        for pass in self.state.app.passes.drain(..) {
            presenter.register_pass(pass)?;
        }
        #[cfg(feature = "text")]
        if let Some(text) = self.state.text.pass() {
            presenter.register_pass(Box::new(text))?;
        }
        if self.state.app.debug_layer {
            if let Some(pass) = self.state.layer.attach(
                gpu,
                format,
                view_format,
                #[cfg(not(target_arch = "wasm32"))]
                window,
                size,
                scale,
            ) {
                presenter.register_pass(pass)?;
            }
        }
        presenter.attach(gpu)?;
        self.presenter = Some(presenter);
        self.resize(size.0, size.1, scale);
        Ok(())
    }

    pub(crate) fn recover(&mut self, gpu: &GpuContext) -> Result<(), HostError> {
        self.invalidate();
        if let Some(presenter) = self.presenter.as_mut() {
            presenter.attach(gpu)?;
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
            .map_err(HostError::from)
            .and_then(|_| {
                gpu.take_uncaptured_error()
                    .map_or(Ok(()), |error| Err(HostError::from(error)))
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
                label: Some("loam-app::frame"),
            });
        presenter.record_scene(
            &gpu.device,
            &mut encoder,
            target.view,
            target.size,
            self.app.background,
        )?;
        if shot.wants(true) {
            finish(&mut encoder);
            self.capture
                .record(&mut shot, true, &gpu.device, &mut encoder, target);
        }
        self.layer.aim(target.texture);
        presenter.record_overlays(&mut encoder, target.view, target.size)?;
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
mod tests;
