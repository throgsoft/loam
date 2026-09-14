use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use anyhow::{anyhow, Context, Result};
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, MessageEvent, OffscreenCanvas};

use loam_egui::egui;
use loam_render::device::{FeatureRequest, RenderDevice};
use loam_runtime::host::HostError;
use loam_runtime::{Session, Stores};
use web_time::Instant;

use super::animation::{self, Lifecycle, Next};
use super::app::SessionApp;
use super::frame::{failed, Frame};
use super::input::TouchCapture;
use super::pacing::Pace;
use super::surface::{Attempt, SurfaceHost};
use crate::wasm::input_queue::{self, InputMessage};
use crate::wasm::messages;
use crate::wasm::{install_logging_idempotent, post_failure, worker_scope};
use crate::{args::Args, WasmConfig};

#[cfg(feature = "measure")]
#[path = "browser_measurement.rs"]
mod measurement;
#[cfg(feature = "measure")]
use measurement::Probe;

pub fn launch<A: Stores>(
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
) -> Result<(), HostError> {
    launch_with(WasmConfig::default(), factory)
}

pub fn launch_with<A: Stores>(
    wasm: WasmConfig,
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
) -> Result<(), HostError> {
    install_logging_idempotent();
    if crate::wasm::is_worker_context() {
        return listen(factory).map_err(|error| failed(format!("{error:#}")));
    }
    drop(factory);
    crate::wasm::launch_on_click(&wasm.host_id, &wasm.button_id, &wasm.canvas_id)
        .map_err(|error| failed(format!("{error:#}")))
}

pub fn launch_or_headless<A: Stores>(
    factory: impl FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
    headless: impl FnOnce(Args) -> Result<(), HostError>,
) -> Result<(), HostError> {
    drop(headless);
    launch(factory)
}

type Pending<F> = Rc<RefCell<Option<F>>>;

thread_local! {
    static RAF_KICKOFF: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    static RAF_RESTART: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    static RAF_PENDING: Cell<bool> = const { Cell::new(false) };
    static START_REQUESTED: Cell<bool> = const { Cell::new(false) };
    static PAUSED: Cell<bool> = const { Cell::new(false) };
    static LOOP_STARTED: Cell<bool> = const { Cell::new(false) };
    static LIFECYCLE: Cell<Lifecycle> = const { Cell::new(Lifecycle::Ready) };
}

fn listen<A: Stores, F>(factory: F) -> Result<()>
where
    F: FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
{
    let scope = worker_scope()?;
    let pending: Pending<F> = Rc::new(RefCell::new(Some(factory)));
    let handler_scope = scope.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        if let Err(error) = on_message(&handler_scope, event, &pending) {
            let message = format!("message handler failed: {error:#}");
            tracing::error!("loam-app::session::browser: {message}");
            post_failure(&handler_scope, &message);
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

fn post_cursor_request(scope: &DedicatedWorkerGlobalScope, locked: bool) {
    let message = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("kind"),
        &JsValue::from_str("cursor_lock"),
    );
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("locked"),
        &JsValue::from_bool(locked),
    );
    if let Err(error) = scope.post_message(&message) {
        tracing::warn!("loam-app::session::browser: cursor request failed: {error:?}");
    }
}

#[cfg(feature = "measure")]
fn post_measurement(scope: &DedicatedWorkerGlobalScope, result: &str) {
    let message = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("kind"),
        &JsValue::from_str("measurement"),
    );
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("result"),
        &JsValue::from_str(result),
    );
    if let Err(error) = scope.post_message(&message) {
        tracing::warn!("loam-app::session::browser: measurement post failed: {error:?}");
    }
}

#[cfg(feature = "measure")]
fn committed_wasm_memory_bytes() -> u64 {
    let memory = wasm_bindgen::memory().unchecked_into::<js_sys::WebAssembly::Memory>();
    let buffer = memory.buffer().unchecked_into::<js_sys::ArrayBuffer>();
    u64::from(buffer.byte_length())
}

fn on_message<A: Stores, F>(
    scope: &DedicatedWorkerGlobalScope,
    event: MessageEvent,
    pending: &Pending<F>,
) -> Result<()>
where
    F: FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
{
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
                post_cursor_request(scope, false);
            }
            return Ok(());
        }
        Some("resume") => {
            let was_paused = PAUSED.with(|paused| paused.replace(false));
            if was_paused {
                input_queue::enqueue(InputMessage::Focus(true));
            }
            let started = LOOP_STARTED.with(|started| started.get());
            let pending = RAF_PENDING.with(|pending| pending.get());
            let lifecycle = LIFECYCLE.with(|lifecycle| lifecycle.get());
            if animation::resumed(was_paused, started, pending, lifecycle) == Next::Frame {
                RAF_RESTART.with(|restart| {
                    if let Some(restart) = restart.borrow().as_ref() {
                        restart();
                    }
                });
            }
            return Ok(());
        }
        Some("init") => {
            let Some(factory) = pending.borrow_mut().take() else {
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
            let args = Args::current();
            let (session, app) = factory(args.clone())
                .map_err(|error| anyhow!("session factory failed: {error:?}"))?;
            let width = read_u32("width").unwrap_or(800);
            let height = read_u32("height").unwrap_or(600);
            let dpr = messages::read_device_pixel_ratio(&data);
            #[cfg(feature = "measure")]
            let measurement = Probe::from_args(&args, width, height, dpr);
            let scope = scope.clone();
            let failure_scope = scope.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let started = start(
                    scope,
                    session,
                    app,
                    canvas,
                    (width, height),
                    dpr,
                    #[cfg(feature = "measure")]
                    measurement,
                )
                .await;
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
    app: SessionApp<A>,
    canvas: OffscreenCanvas,
    size: (u32, u32),
    dpr: f32,
    #[cfg(feature = "measure")] measurement: Option<Probe>,
) -> Result<()> {
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
    let (surface, rd) = SurfaceHost::new(instance, surface, size, request)
        .await
        .context("SurfaceHost::new")?;

    let mut frame = Frame::new(session, app);
    frame
        .attach(&rd.context, rd.target_format(), None, size, dpr)
        .map_err(|error| anyhow!("{error:?}"))?;
    let worker = Rc::new(RefCell::new(Some(Worker {
        frame,
        surface,
        rd,
        canvas,
        scope: scope.clone(),
        messages: VecDeque::new(),
        touches: TouchCapture::default(),
        dpr,
        #[cfg(feature = "measure")]
        measurement,
    })));
    post(&scope, "preview_ready");
    install_animation_frame(scope, worker);
    Ok(())
}

#[cfg(feature = "measure")]
fn measure<A: Stores>(worker: &mut Worker<A>) -> std::result::Result<(), HostError> {
    let started = worker.measurement.as_ref().map(|_| Instant::now());
    let attempt = worker.animate()?;
    if let Some(started) = started {
        let cpu_ms = started.elapsed().as_secs_f64() * 1000.0;
        let report = match worker.frame.phase_error() {
            Some(error) => worker.measurement.take().map(|probe| probe.invalid(error)),
            None => worker
                .measurement
                .as_mut()
                .and_then(|probe| probe.observe(attempt, started, cpu_ms)),
        };
        if let Some(mut report) = report {
            report.push_str(&format!(
                "\nWASM linear memory committed at report: {} bytes\nGPU memory: unavailable\nGPU duration: unavailable; no timestamp query",
                committed_wasm_memory_bytes()
            ));
            post_measurement(&worker.scope, &report);
            worker.measurement = None;
        }
    }
    Ok(())
}

fn install_animation_frame<A: Stores>(
    scope: DedicatedWorkerGlobalScope,
    worker: Rc<RefCell<Option<Worker<A>>>>,
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
            false => worker_for_closure
                .borrow()
                .as_ref()
                .and_then(|worker| worker.rd.take_device_loss()),
        };
        let next = LIFECYCLE.with(|lifecycle| {
            let mut state = lifecycle.get();
            let next = animation::frame(paused, &mut state, loss.is_some(), || {
                let mut held = worker_for_closure.borrow_mut();
                let Some(worker) = held.as_mut() else {
                    return Err(failed("the session worker is unavailable"));
                };
                #[cfg(feature = "measure")]
                measure(worker)?;
                #[cfg(not(feature = "measure"))]
                worker.animate()?;
                Ok(())
            });
            lifecycle.set(state);
            next
        });
        match next {
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
                    let next = LIFECYCLE.with(|lifecycle| {
                        let mut state = lifecycle.get();
                        let next = animation::recovered(&mut state, paused, pending, outcome);
                        lifecycle.set(state);
                        next
                    });
                    match next {
                        Next::Frame => {
                            let mut held = worker.borrow_mut();
                            let Some(worker) = held.as_mut() else {
                                return;
                            };
                            worker.frame.reset_clock(Instant::now());
                            drop(held);
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
            let mut held = worker_for_kickoff.borrow_mut();
            let Some(worker) = held.as_mut() else {
                return;
            };
            worker.frame.reset_clock(Instant::now());
            drop(held);
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
            let mut held = worker.borrow_mut();
            let Some(worker) = held.as_mut() else {
                return;
            };
            worker.frame.reset_clock(Instant::now());
            drop(held);
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
    worker: &Rc<RefCell<Option<Worker<A>>>>,
    loss: &loam_render::device::DeviceLoss,
) -> std::result::Result<(), String> {
    let active = worker.borrow_mut().take();
    let Some(mut active) = active else {
        return Err("the session worker is unavailable during recovery".to_owned());
    };
    #[cfg(feature = "measure")]
    if let Some(probe) = active.measurement.as_mut() {
        probe.recover();
    }

    let outcome = if let Err(error) = active.rd.recover().await {
        Err(format!(
            "GPU device lost ({:?}: {}); the device was not rebuilt: {error:#}",
            loss.reason, loss.message
        ))
    } else {
        active.surface.reconfigure(&active.rd.context.device);
        active
            .frame
            .recover(&active.rd.context)
            .map_err(|error| format!("presentation recovery failed: {error:?}"))
    };
    *worker.borrow_mut() = Some(active);
    outcome
}

fn feed_layer(
    layer: &super::DebugLayer,
    touches: &mut TouchCapture,
    message: &InputMessage,
) -> bool {
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
            ..
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
            if !focused {
                layer.cancel_touches(touches);
            }
            layer.push(egui::Event::WindowFocused(*focused));
            false
        }
        InputMessage::Pointer {
            id, x, y, phase, ..
        } => {
            let pos = egui::pos2(*x, *y) / context.zoom_factor();
            let pointer_before = touches.is_pointer(*id);
            let over_ui = *phase == crate::wasm::input_queue::PointerPhase::Down
                && context.layer_id_at(pos).is_some_and(|layer| {
                    layer.order != egui::Order::Background
                        || !context.available_rect().contains(pos)
                });
            let consumed = touches.route(*id, *phase, over_ui, [pos.x, pos.y]);
            let pointer = pointer_before || touches.is_pointer(*id);
            if !consumed {
                return false;
            }
            let touch_phase = match phase {
                crate::wasm::input_queue::PointerPhase::Down => egui::TouchPhase::Start,
                crate::wasm::input_queue::PointerPhase::Move => egui::TouchPhase::Move,
                crate::wasm::input_queue::PointerPhase::Up => egui::TouchPhase::End,
                crate::wasm::input_queue::PointerPhase::Cancel => egui::TouchPhase::Cancel,
            };
            if *phase == crate::wasm::input_queue::PointerPhase::Cancel {
                layer.cancel_touch(*id, pos, pointer);
            } else {
                layer.push(egui::Event::Touch {
                    device_id: egui::TouchDeviceId(0),
                    id: egui::TouchId::from(*id),
                    phase: touch_phase,
                    pos,
                    force: None,
                });
            }
            if pointer {
                match phase {
                    crate::wasm::input_queue::PointerPhase::Down => {
                        layer.push(egui::Event::PointerMoved(pos));
                        layer.push(egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed: true,
                            modifiers: layer.modifiers(),
                        });
                    }
                    crate::wasm::input_queue::PointerPhase::Move => {
                        layer.push(egui::Event::PointerMoved(pos));
                    }
                    crate::wasm::input_queue::PointerPhase::Up => {
                        layer.push(egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed: false,
                            modifiers: layer.modifiers(),
                        });
                        layer.push(egui::Event::PointerGone);
                    }
                    crate::wasm::input_queue::PointerPhase::Cancel => {}
                }
            }
            consumed
        }
        InputMessage::Resize { .. }
        | InputMessage::Visibility(_)
        | InputMessage::Start
        | InputMessage::PointerLockChanged { .. } => false,
    }
}

struct Worker<A: Stores> {
    frame: Frame<A>,
    surface: SurfaceHost,
    rd: RenderDevice,
    canvas: OffscreenCanvas,
    scope: DedicatedWorkerGlobalScope,
    messages: VecDeque<InputMessage>,
    touches: TouchCapture,
    dpr: f32,
    #[cfg(feature = "measure")]
    measurement: Option<Probe>,
}

impl<A: Stores> Worker<A> {
    fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        self.canvas.set_width(width);
        self.canvas.set_height(height);
        let size = (width, height);
        self.surface.resize(&self.rd.context.device, size);
        self.rd.resize(size);
        self.dpr = dpr;
        self.frame.resize(width, height, dpr);
    }

    fn apply(&mut self, message: InputMessage) {
        #[cfg(feature = "measure")]
        if let Some(probe) = self.measurement.as_mut() {
            probe.observe_message(&message);
        }
        self.frame.observe_message(&message);
        if self.frame.cursor_locked()
            && matches!(
                &message,
                InputMessage::MouseMove { .. }
                    | InputMessage::MouseButton { .. }
                    | InputMessage::MouseWheel { .. }
            )
        {
            if matches!(&message, InputMessage::MouseMove { .. }) {
                self.frame.apply_message(&message, false);
            }
            return;
        }
        let layer = self.frame.layer().cloned();
        let consumed = layer
            .as_ref()
            .is_some_and(|layer| feed_layer(layer, &mut self.touches, &message));
        match &message {
            InputMessage::Resize { width, height, dpr } => self.resize(*width, *height, *dpr),
            _ => self.frame.apply_message(&message, consumed),
        }
    }

    fn animate(&mut self) -> Result<Attempt, HostError> {
        input_queue::drain_messages_into(&mut self.messages);
        while let Some(message) = self.messages.pop_front() {
            self.apply(message);
        }
        self.flush_cursor_request();
        if let Some(enabled) = self.frame.app_mut().vsync.take() {
            self.surface.set_vsync(&self.rd.context.device, enabled);
        }
        let now = Instant::now();
        if let Pace::Wait(_) = self.frame.app_mut().pacer.decide(now) {
            return Ok(Attempt::Paced);
        }
        let size = self.surface.size();
        if size.0 == 0 || size.1 == 0 {
            return Ok(Attempt::EmptySurface);
        }
        let Worker {
            frame,
            surface,
            rd,
            scope,
            ..
        } = self;
        surface.present(rd, frame, now, |frame, _| {
            if let Some(locked) = frame.take_cursor_request() {
                post_cursor_request(scope, locked);
            }
        })
    }

    fn flush_cursor_request(&mut self) {
        if let Some(locked) = self.frame.take_cursor_request() {
            post_cursor_request(&self.scope, locked);
        }
    }
}
