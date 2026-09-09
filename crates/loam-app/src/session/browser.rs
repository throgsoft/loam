use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, OffscreenCanvas};

use loam_egui::egui;
use loam_render::device::{FeatureRequest, GpuContext, RenderDevice};
use loam_runtime::host::{HostConfig, HostError};
use loam_runtime::{Session, Stores};
use web_time::Instant;

use super::animation::{self, Next};
use super::app::SessionApp;
use super::frame::{failed, Frame, Target};
use super::pacing::Pace;
use super::WorkContext;
use crate::wasm::input_queue::{self, InputMessage};
use crate::wasm::messages;
use crate::wasm::{install_logging_idempotent, post_failure, worker_scope};

/// [`launch`] with a default [`SessionApp`]; on the page it returns at once and the worker owns the loop.
pub fn run<A: Stores>(session: Session<A>, config: HostConfig) -> Result<(), HostError> {
    launch(session, SessionApp::new(config))
}

/// `record` runs once per issued order inside the frame's encoder before the presenter draws.
pub fn run_with_work<A: Stores>(
    session: Session<A>,
    config: HostConfig,
    record: impl FnMut(WorkContext<'_>) + 'static,
) -> Result<(), HostError> {
    launch(session, SessionApp::new(config).work(record))
}

/// In the worker, listens for `init` and drives frames from the offscreen canvas; on the page, starts the worker and returns at once.
pub fn launch<A: Stores>(session: Session<A>, app: SessionApp<A>) -> Result<(), HostError> {
    install_logging_idempotent();
    if crate::wasm::is_worker_context() {
        return listen(session, app).map_err(|error| failed(format!("{error:#}")));
    }
    let sim = crate::SimConfig {
        fixed_hz: session.config().fixed_hz,
        catch_up: crate::CatchUp::Cap(session.config().max_ticks_per_frame),
        seed: session.config().seed,
        overlap: session.config().overlap,
    };
    let wasm = app.wasm.clone();
    drop(session);
    drop(app);
    crate::wasm::launch_on_click(&wasm.host_id, &wasm.button_id, &wasm.canvas_id, sim)
        .map_err(|error| failed(format!("{error:#}")))
}

type Pending<A> = Rc<RefCell<Option<(Session<A>, SessionApp<A>)>>>;

thread_local! {
    static RAF_KICKOFF: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    static RAF_RESTART: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    static RAF_PENDING: Cell<bool> = const { Cell::new(false) };
    static START_REQUESTED: Cell<bool> = const { Cell::new(false) };
    static PAUSED: Cell<bool> = const { Cell::new(false) };
    static LOOP_STARTED: Cell<bool> = const { Cell::new(false) };
}

fn listen<A: Stores>(session: Session<A>, app: SessionApp<A>) -> Result<()> {
    let scope = worker_scope()?;
    let pending: Pending<A> = Rc::new(RefCell::new(Some((session, app))));
    let handler_scope = scope.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        if let Err(error) = on_message(&handler_scope, event, &pending) {
            tracing::error!("loam-app::session::browser: message handler failed: {error:#}");
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    scope
        .add_event_listener_with_callback("message", on_message.as_ref().unchecked_ref())
        .map_err(|error| anyhow!("addEventListener('message'): {error:?}"))?;
    on_message.forget();
    post(&scope, "ready");
    Ok(())
}

fn post(scope: &DedicatedWorkerGlobalScope, kind: &str) {
    let message = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("kind"),
        &JsValue::from_str(kind),
    );
    if let Err(error) = scope.post_message(&message) {
        tracing::warn!("loam-app::session::browser: post {kind} failed: {error:?}");
    }
}

fn on_message<A: Stores>(
    scope: &DedicatedWorkerGlobalScope,
    event: MessageEvent,
    pending: &Pending<A>,
) -> Result<()> {
    let data: JsValue = event.data();
    let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
        .ok()
        .and_then(|value| value.as_string());
    match kind.as_deref() {
        Some("start") => {
            match RAF_KICKOFF.with(|kickoff| kickoff.borrow_mut().take()) {
                Some(kickoff) => kickoff(),
                None => START_REQUESTED.with(|started| started.set(true)),
            }
            return Ok(());
        }
        Some("pause") => {
            if !PAUSED.with(|paused| paused.replace(true)) {
                input_queue::enqueue(InputMessage::Focus(false));
            }
            return Ok(());
        }
        Some("resume") => {
            let was_paused = PAUSED.with(|paused| paused.replace(false));
            if was_paused && LOOP_STARTED.with(|started| started.get()) {
                RAF_RESTART.with(|restart| {
                    if let Some(restart) = restart.borrow().as_ref() {
                        restart();
                    }
                });
            }
            return Ok(());
        }
        Some("init") => {
            let Some((session, app)) = pending.borrow_mut().take() else {
                return Err(anyhow!("a second init reached the session worker"));
            };
            let canvas = js_sys::Reflect::get(&data, &JsValue::from_str("canvas"))
                .map_err(|error| anyhow!("init missing 'canvas': {error:?}"))?
                .dyn_into::<OffscreenCanvas>()
                .map_err(|error| anyhow!("init 'canvas' is not an OffscreenCanvas: {error:?}"))?;
            let read_u32 = |key: &str| {
                js_sys::Reflect::get(&data, &JsValue::from_str(key))
                    .ok()
                    .and_then(|value| value.as_f64())
                    .map(|value| value as u32)
            };
            let read_str = |key: &str| {
                js_sys::Reflect::get(&data, &JsValue::from_str(key))
                    .ok()
                    .and_then(|value| value.as_string())
                    .unwrap_or_default()
            };
            crate::args::set_query_override(read_str("search"), read_str("hash"));
            let width = read_u32("width").unwrap_or(800);
            let height = read_u32("height").unwrap_or(600);
            let dpr = messages::read_device_pixel_ratio(&data);
            let scope = scope.clone();
            let failure_scope = scope.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let started = start(scope, session, app, canvas, width, height, dpr).await;
                if let Err(error) = started {
                    let message = format!("initialization failed: {error:#}");
                    tracing::error!("loam-app::session::browser: {message}");
                    post_failure(&failure_scope, &message);
                }
            });
            return Ok(());
        }
        _ => {}
    }
    if PAUSED.with(|paused| paused.get())
        && matches!(
            kind.as_deref(),
            Some("mouse_move" | "mouse_button" | "mouse_wheel" | "key" | "pointer")
        )
    {
        return Ok(());
    }
    if let Some(message) = messages::parse_non_init(&data)? {
        input_queue::enqueue(message);
    }
    Ok(())
}

async fn start<A: Stores>(
    scope: DedicatedWorkerGlobalScope,
    session: Session<A>,
    mut app: SessionApp<A>,
    canvas: OffscreenCanvas,
    width: u32,
    height: u32,
    dpr: f32,
) -> Result<()> {
    app.args = crate::args::Args::current();
    app.apply_args();
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let surface = instance
        .create_surface(wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone()))
        .context("create_surface from OffscreenCanvas")?;
    let request = FeatureRequest {
        optional_features: wgpu::Features::empty(),
        ..FeatureRequest::default()
    };
    let context = GpuContext::new(instance, request, Some(&surface))
        .await
        .context("GpuContext::new")?;
    let size = winit::dpi::PhysicalSize::new(width, height);
    let rd = RenderDevice::attach(context, surface, size, 1).context("RenderDevice::attach")?;

    let mut frame = Frame::new(session, app);
    frame
        .attach(
            &rd.context,
            rd.target_format(),
            rd.sample_count(),
            None,
            (width, height),
            dpr,
        )
        .map_err(|error| anyhow!("{error:?}"))?;
    let worker = Rc::new(RefCell::new(Worker {
        frame,
        rd,
        canvas,
        messages: VecDeque::new(),
        dpr,
    }));
    post(&scope, "preview_ready");
    install_animation_frame(scope, worker);
    Ok(())
}

fn install_animation_frame<A: Stores>(
    scope: DedicatedWorkerGlobalScope,
    worker: Rc<RefCell<Worker<A>>>,
) {
    let callback: Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>> = Rc::new(RefCell::new(None));
    let callback_for_closure = callback.clone();
    let scope_for_closure = scope.clone();
    let worker_for_closure = worker.clone();
    *callback.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
        RAF_PENDING.with(|pending| pending.set(false));
        let paused = PAUSED.with(|paused| paused.get());
        let loss = match paused {
            true => None,
            false => worker_for_closure.borrow().rd.take_device_loss(),
        };
        match animation::frame(paused, loss.is_some(), || {
            worker_for_closure.borrow_mut().animate()
        }) {
            Next::Idle => {}
            Next::Frame => request_frame(&scope_for_closure, &callback_for_closure),
            Next::Failed(message) => {
                tracing::error!("loam-app::session::browser: {message}");
                post_failure(&scope_for_closure, &message);
            }
            Next::Recover => {
                let worker = worker_for_closure.clone();
                let scope = scope_for_closure.clone();
                let callback = callback_for_closure.clone();
                wasm_bindgen_futures::spawn_local(async move {
                    let outcome = match loss {
                        Some(loss) => recover(&worker, &loss).await,
                        None => Ok(()),
                    };
                    let paused = PAUSED.with(|paused| paused.get());
                    let pending = RAF_PENDING.with(|pending| pending.get());
                    match animation::recovered(paused, pending, outcome) {
                        Next::Frame => {
                            worker.borrow_mut().frame.reset_clock(Instant::now());
                            request_frame(&scope, &callback);
                        }
                        Next::Failed(message) => {
                            tracing::error!("loam-app::session::browser: {message}");
                            post_failure(&scope, &message);
                        }
                        Next::Idle | Next::Recover => {}
                    }
                });
            }
        }
    }) as Box<dyn FnMut(f64)>));

    let scope_for_kickoff = scope.clone();
    let callback_for_kickoff = callback.clone();
    let worker_for_kickoff = worker.clone();
    RAF_KICKOFF.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(move || {
            worker_for_kickoff
                .borrow_mut()
                .frame
                .reset_clock(Instant::now());
            LOOP_STARTED.with(|started| started.set(true));
            request_frame(&scope_for_kickoff, &callback_for_kickoff);
        }));
    });

    let scope_for_restart = scope.clone();
    let callback_for_restart = callback.clone();
    RAF_RESTART.with(|slot| {
        *slot.borrow_mut() = Some(Box::new(move || {
            if RAF_PENDING.with(|pending| pending.get()) {
                return;
            }
            worker.borrow_mut().frame.reset_clock(Instant::now());
            request_frame(&scope_for_restart, &callback_for_restart);
        }));
    });

    if START_REQUESTED.with(|started| started.replace(false)) {
        if let Some(kickoff) = RAF_KICKOFF.with(|slot| slot.borrow_mut().take()) {
            kickoff();
        }
    }
}

fn request_frame(
    scope: &DedicatedWorkerGlobalScope,
    callback: &Rc<RefCell<Option<Closure<dyn FnMut(f64)>>>>,
) {
    let held = callback.borrow();
    let Some(callback) = held.as_ref() else {
        return;
    };
    match scope.request_animation_frame(callback.as_ref().unchecked_ref()) {
        Ok(_) => RAF_PENDING.with(|pending| pending.set(true)),
        Err(error) => tracing::error!("loam-app::session::browser: RAF failed: {error:?}"),
    }
}

async fn recover<A: Stores>(
    worker: &Rc<RefCell<Worker<A>>>,
    loss: &loam_render::device::DeviceLoss,
) -> std::result::Result<(), String> {
    let rebuilt = {
        let mut held = worker.borrow_mut();
        held.rd.recover().await
    };
    if let Err(error) = rebuilt {
        return Err(format!(
            "GPU device lost ({:?}: {}); the device was not rebuilt: {error:#}",
            loss.reason, loss.message
        ));
    }
    let mut held = worker.borrow_mut();
    let Worker { frame, rd, .. } = &mut *held;
    frame
        .recover(&rd.context)
        .map_err(|error| format!("presentation recovery failed: {error:?}"))
}

fn feed_layer(layer: &super::DebugLayer, message: &InputMessage) -> bool {
    let context = layer.context();
    match message {
        InputMessage::MouseMove { x, y, .. } => {
            layer.push(egui::Event::PointerMoved(
                egui::pos2(*x, *y) / context.zoom_factor(),
            ));
            context.wants_pointer_input()
        }
        InputMessage::MouseButton {
            x,
            y,
            button,
            pressed,
        } => {
            if let Some(egui_button) = crate::keymap::mouse_button_egui(*button) {
                layer.push(egui::Event::PointerButton {
                    pos: egui::pos2(*x, *y) / context.zoom_factor(),
                    button: egui_button,
                    pressed: *pressed,
                    modifiers: layer.modifiers(),
                });
            }
            context.wants_pointer_input()
        }
        InputMessage::MouseWheel { dx, dy } => {
            layer.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Line,
                delta: egui::vec2(-*dx, -*dy),
                modifiers: layer.modifiers(),
            });
            context.wants_pointer_input()
        }
        InputMessage::Key {
            code,
            key,
            pressed,
            repeat,
            ctrl,
            shift,
            alt,
            meta,
        } => {
            let modifiers = egui::Modifiers {
                alt: *alt,
                ctrl: *ctrl,
                shift: *shift,
                mac_cmd: *meta,
                command: *ctrl || *meta,
            };
            layer.set_modifiers(modifiers);
            if let Some(egui_key) = crate::keymap::keycode_egui(code) {
                layer.push(egui::Event::Key {
                    key: egui_key,
                    physical_key: Some(egui_key),
                    pressed: *pressed,
                    repeat: *repeat,
                    modifiers,
                });
            }
            if *pressed
                && !*ctrl
                && !*alt
                && !*meta
                && key.chars().count() == 1
                && !key.starts_with(char::is_control)
            {
                layer.push(egui::Event::Text(key.clone()));
            }
            context.wants_keyboard_input()
        }
        InputMessage::Focus(focused) => {
            layer.push(egui::Event::WindowFocused(*focused));
            false
        }
        InputMessage::Resize { .. }
        | InputMessage::Visibility(_)
        | InputMessage::Start
        | InputMessage::PointerLockChanged(_)
        | InputMessage::Pointer { .. } => false,
    }
}

struct Worker<A: Stores> {
    frame: Frame<A>,
    rd: RenderDevice,
    canvas: OffscreenCanvas,
    messages: VecDeque<InputMessage>,
    dpr: f32,
}

impl<A: Stores> Worker<A> {
    fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        if width == 0 || height == 0 {
            return;
        }
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        self.rd.resize(winit::dpi::PhysicalSize::new(width, height));
        self.dpr = dpr;
        self.frame.resize(width, height, dpr);
    }

    fn apply(&mut self, message: InputMessage) {
        let layer = self.frame.layer().cloned();
        let consumed = layer
            .as_ref()
            .is_some_and(|layer| feed_layer(layer, &message));
        match &message {
            InputMessage::Resize { width, height, dpr } => self.resize(*width, *height, *dpr),
            _ if consumed => {}
            _ => self.frame.apply_message(&message),
        }
    }

    fn animate(&mut self) -> Result<(), HostError> {
        input_queue::drain_messages_into(&mut self.messages);
        while let Some(message) = self.messages.pop_front() {
            self.apply(message);
        }
        if let Some(enabled) = self.frame.app_mut().vsync.take() {
            crate::frame_pacing::apply_present_mode(&mut self.rd, enabled);
        }
        let now = Instant::now();
        if let Pace::Wait(_) = self.frame.app_mut().pacer.decide(now) {
            return Ok(());
        }
        let size = self.rd.surface_bundle.size;
        let Ok((surface, swap_view)) = self.rd.begin_frame() else {
            return Ok(());
        };
        let Worker { frame, rd, .. } = self;
        {
            let view = rd.msaa_view().or(rd.scene_view()).unwrap_or(&swap_view);
            let target = Target {
                view,
                texture: &surface.texture,
                format: rd.surface_bundle.config.format,
                size: (size.width, size.height),
            };
            frame.step(&rd.context, &target, now, |encoder| {
                if rd.sample_count() > 1 {
                    rd.resolve_scene_to_swap(encoder, &swap_view);
                }
                if rd.scene_view().is_some() {
                    rd.composite_to_swap(encoder, &swap_view);
                }
            })?;
        }
        surface.present();
        Ok(())
    }
}
