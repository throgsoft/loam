use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;

use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{HtmlCanvasElement, MessageEvent, OffscreenCanvas};

#[cfg(feature = "egui")]
use loam_egui::egui;
use loam_render::device::{FeatureRequest, RenderDevice};
use loam_runtime::{Session, Stores};
use loam_time::par::Sequential;
use web_time::Instant;

use super::animation::{self, Lifecycle, Next};
use super::error::{js, listener, WebError};
use super::messages::{self, Kind};
use super::{install_logging_idempotent, post_failure, worker_scope, Endpoint};
use crate::host::HostError;
use crate::session::app::SessionApp;
use crate::session::frame::Frame;
use crate::session::input::message::{InputMessage, PointerPhase};
use crate::session::input::queue;
#[cfg(feature = "egui")]
use crate::session::input::TouchCapture;
use crate::session::surface::{Attempt, SurfaceHost};
use crate::{args::Args, WasmConfig};

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
    if super::is_worker_context() {
        let scope = worker_scope()?;
        return listen(Endpoint::Worker(scope), None, factory);
    }
    let in_page: super::InPage = Box::new(move |port, window, canvas| {
        listen(Endpoint::Page(port, window), Some(canvas), factory)
    });
    super::launch_page(&wasm, in_page)
}

type Pending<F> = Rc<RefCell<Option<F>>>;

thread_local! {
    static RAF_KICKOFF: RefCell<Option<Box<dyn FnOnce()>>> = const { RefCell::new(None) };
    static RAF_RESTART: RefCell<Option<Box<dyn Fn()>>> = const { RefCell::new(None) };
    static RAF_PENDING: Cell<bool> = const { Cell::new(false) };
    static START_REQUESTED: Cell<bool> = const { Cell::new(false) };
    static PAUSED: Cell<bool> = const { Cell::new(false) };
    static HOST_PAUSED: Cell<bool> = const { Cell::new(false) };
    static HIDDEN: Cell<bool> = const { Cell::new(false) };
    static LOOP_STARTED: Cell<bool> = const { Cell::new(false) };
    static LIFECYCLE: Cell<Lifecycle> = const { Cell::new(Lifecycle::Ready) };
}

fn listen<A: Stores, F>(
    endpoint: Endpoint,
    page_canvas: Option<HtmlCanvasElement>,
    factory: F,
) -> Result<(), HostError>
where
    F: FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
{
    let pending: Pending<F> = Rc::new(RefCell::new(Some(factory)));
    let page_canvas = RefCell::new(page_canvas);
    let handler_endpoint = endpoint.clone();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        if let Err(error) = on_message(&handler_endpoint, event, &pending, &page_canvas) {
            let message = format!("message handler failed: {error}");
            tracing::error!("loam_app::web::session: {message}");
            post_failure(&handler_endpoint, &message);
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    endpoint
        .listen(on_message.as_ref().unchecked_ref())
        .map_err(listener("message"))?;
    on_message.forget();
    post(&endpoint, Kind::Ready);
    Ok(())
}

fn input_paused() -> bool {
    PAUSED.with(Cell::get) || HOST_PAUSED.with(Cell::get)
}

fn halted() -> bool {
    input_paused() || HIDDEN.with(Cell::get)
}

fn set_paused(endpoint: &Endpoint, flag: &'static std::thread::LocalKey<Cell<bool>>, paused: bool) {
    let was_halted = halted();
    let was_input_paused = input_paused();
    flag.with(|flag| flag.set(paused));
    match (was_input_paused, input_paused()) {
        (false, true) => {
            queue::enqueue(InputMessage::Focus(false));
            post_cursor_request(endpoint, false);
        }
        (true, false) => queue::enqueue(InputMessage::Focus(true)),
        _ => {}
    }
    if !paused {
        restart_if_unhalted(was_halted);
    }
}

fn restart_if_unhalted(was_halted: bool) {
    let started = LOOP_STARTED.with(|started| started.get());
    let pending = RAF_PENDING.with(|pending| pending.get());
    let lifecycle = LIFECYCLE.with(|lifecycle| lifecycle.get());
    if animation::resumed(was_halted, halted(), started, pending, lifecycle) == Next::Frame {
        RAF_RESTART.with(|restart| {
            if let Some(restart) = restart.borrow().as_ref() {
                restart();
            }
        });
    }
}

fn post(endpoint: &Endpoint, kind: Kind) {
    if let Err(error) = endpoint.post(&kind.message()) {
        tracing::warn!("loam_app::web::session: post {kind:?} failed: {error:?}");
    }
}

fn post_cursor_request(endpoint: &Endpoint, locked: bool) {
    let message = Kind::CursorLock.message();
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("locked"),
        &JsValue::from_bool(locked),
    );
    if let Err(error) = endpoint.post(&message) {
        tracing::warn!("loam_app::web::session: cursor request failed: {error:?}");
    }
}

fn post_to_page(endpoint: &Endpoint, topic: &str, values: &[f32]) {
    let message = Kind::HostPost.message();
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("topic"),
        &JsValue::from_str(topic),
    );
    let _ = js_sys::Reflect::set(
        &message,
        &JsValue::from_str("values"),
        &js_sys::Float32Array::from(values),
    );
    if let Err(error) = endpoint.post(&message) {
        tracing::warn!("loam_app::web::session: post {topic} failed: {error:?}");
    }
}

fn on_message<A: Stores, F>(
    endpoint: &Endpoint,
    event: MessageEvent,
    pending: &Pending<F>,
    page_canvas: &RefCell<Option<HtmlCanvasElement>>,
) -> Result<(), HostError>
where
    F: FnOnce(Args) -> Result<(Session<A>, SessionApp<A>), HostError> + 'static,
{
    let data: JsValue = event.data();
    let kind = Kind::of(&data);
    match kind {
        Some(Kind::Start) => {
            match RAF_KICKOFF.with(|kickoff| kickoff.borrow_mut().take()) {
                Some(kickoff) => kickoff(),
                None => START_REQUESTED.with(|started| started.set(true)),
            }
            return Ok(());
        }
        Some(Kind::Pause) => {
            set_paused(endpoint, &PAUSED, true);
            return Ok(());
        }
        Some(Kind::Resume) => {
            set_paused(endpoint, &PAUSED, false);
            return Ok(());
        }
        Some(Kind::HostPause) => {
            set_paused(endpoint, &HOST_PAUSED, true);
            return Ok(());
        }
        Some(Kind::HostResume) => {
            set_paused(endpoint, &HOST_PAUSED, false);
            return Ok(());
        }
        Some(Kind::Visibility) => {
            let was_halted = halted();
            let visible = js_sys::Reflect::get(&data, &JsValue::from_str("visible"))
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            HIDDEN.with(|hidden| hidden.set(!visible));
            restart_if_unhalted(was_halted);
        }
        Some(Kind::Init) => {
            let Some(factory) = pending.borrow_mut().take() else {
                return Err(WebError::SecondInit.into());
            };
            let canvas = match page_canvas.borrow_mut().take() {
                Some(canvas) => Canvas::Page(canvas),
                None => Canvas::Offscreen(
                    js_sys::Reflect::get(&data, &JsValue::from_str("canvas"))
                        .map_err(js("init missing 'canvas'"))?
                        .dyn_into::<OffscreenCanvas>()
                        .map_err(js("init 'canvas' is not an OffscreenCanvas"))?,
                ),
            };
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
            let visible = js_sys::Reflect::get(&data, &JsValue::from_str("visible"))
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(true);
            HIDDEN.with(|hidden| hidden.set(!visible));
            crate::args::set_query_override(read_str("search"), read_str("hash"));
            let args = Args::current();
            let (mut session, app) = factory(args.clone())?;
            session.capture_initial();
            let width = read_u32("width").unwrap_or(800);
            let height = read_u32("height").unwrap_or(600);
            let dpr = messages::read_device_pixel_ratio(&data);
            let endpoint = endpoint.clone();
            let failure_endpoint = endpoint.clone();
            wasm_bindgen_futures::spawn_local(async move {
                let started = start(endpoint, session, app, canvas, (width, height), dpr).await;
                if let Err(error) = started {
                    let message = format!("initialization failed: {error}");
                    tracing::error!("loam_app::web::session: {message}");
                    post_failure(&failure_endpoint, &message);
                }
            });
            return Ok(());
        }
        _ => {}
    }
    if input_paused()
        && matches!(
            kind,
            Some(
                Kind::MouseMove | Kind::MouseButton | Kind::MouseWheel | Kind::Key | Kind::Pointer
            )
        )
    {
        return Ok(());
    }
    if let Some(message) = messages::parse_non_init(&data)? {
        queue::enqueue(message);
    }
    Ok(())
}

async fn start<A: Stores>(
    endpoint: Endpoint,
    session: Session<A>,
    app: SessionApp<A>,
    canvas: Canvas,
    size: (u32, u32),
    dpr: f32,
) -> Result<(), HostError> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });
    let surface = instance.create_surface(canvas.target())?;
    let request = FeatureRequest {
        optional_features: wgpu::Features::empty(),
        ..FeatureRequest::default()
    };
    let (surface, rd) = SurfaceHost::new(instance, surface, size, request).await?;

    let mut frame = Frame::new(session, app, Box::new(Sequential))?;
    frame.attach(
        &rd.context,
        rd.target_format(),
        surface.view_format(),
        size,
        dpr,
    )?;
    let worker = Rc::new(RefCell::new(Some(Worker {
        frame,
        surface,
        rd,
        canvas,
        endpoint: endpoint.clone(),
        messages: VecDeque::new(),
        #[cfg(feature = "egui")]
        touches: TouchCapture::default(),
        dpr,
        presented: false,
    })));
    post(&endpoint, Kind::PreviewReady);
    install_animation_frame(endpoint, worker);
    Ok(())
}

fn install_animation_frame<A: Stores>(endpoint: Endpoint, worker: Rc<RefCell<Option<Worker<A>>>>) {
    let callback: Pending<Closure<dyn FnMut(f64)>> = Rc::new(RefCell::new(None));
    let callback_for_closure = callback.clone();
    let scope_for_closure = endpoint.clone();
    let worker_for_closure = worker.clone();
    *callback.borrow_mut() = Some(Closure::wrap(Box::new(move |_timestamp: f64| {
        RAF_PENDING.with(|pending| pending.set(false));
        let idle = halted();
        let loss = match idle {
            true => None,
            false => worker_for_closure
                .borrow()
                .as_ref()
                .and_then(|worker| worker.rd.take_device_loss()),
        };
        let next = LIFECYCLE.with(|lifecycle| {
            let mut state = lifecycle.get();
            let next = animation::frame(idle, &mut state, loss.is_some(), || {
                let mut held = worker_for_closure.borrow_mut();
                let Some(worker) = held.as_mut() else {
                    return Err(WebError::WorkerUnavailable.into());
                };
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
                tracing::error!("loam_app::web::session: {message}");
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
                    let pending = RAF_PENDING.with(|pending| pending.get());
                    let next = LIFECYCLE.with(|lifecycle| {
                        let mut state = lifecycle.get();
                        let next = animation::recovered(&mut state, halted(), pending, outcome);
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
                            tracing::error!("loam_app::web::session: {message}");
                            post_failure(&scope, &message);
                        }
                        Next::Idle | Next::Recover => {}
                    }
                });
            }
        }
    }) as Box<dyn FnMut(f64)>));

    let scope_for_kickoff = endpoint.clone();
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

    let scope_for_restart = endpoint.clone();
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
            worker.frame.invalidate();
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

fn request_frame(endpoint: &Endpoint, callback: &Pending<Closure<dyn FnMut(f64)>>) {
    let held = callback.borrow();
    let Some(callback) = held.as_ref() else {
        return;
    };
    match endpoint.request_animation_frame(callback.as_ref().unchecked_ref()) {
        Ok(_) => RAF_PENDING.with(|pending| pending.set(true)),
        Err(error) => tracing::error!("loam_app::web::session: RAF failed: {error:?}"),
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
    let outcome = if let Err(error) = active.rd.recover().await {
        Err(format!(
            "GPU device lost ({:?}: {}); the device was not rebuilt: {error}",
            loss.reason, loss.message
        ))
    } else {
        active.surface.reconfigure(&active.rd.context.device);
        active
            .frame
            .recover(&active.rd.context)
            .map_err(|error| format!("presentation recovery failed: {error}"))
    };
    *worker.borrow_mut() = Some(active);
    outcome
}

#[cfg(feature = "egui")]
fn feed_layer(
    layer: &crate::session::DebugLayer,
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
            if let Some(egui_button) = super::keymap::mouse_button_egui(*button) {
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
            if let Some(egui_key) = super::keymap::keycode_egui(code) {
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
            let over_ui = *phase == PointerPhase::Down
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
                PointerPhase::Down => egui::TouchPhase::Start,
                PointerPhase::Move => egui::TouchPhase::Move,
                PointerPhase::Up => egui::TouchPhase::End,
                PointerPhase::Cancel => egui::TouchPhase::Cancel,
            };
            if *phase == PointerPhase::Cancel {
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
                    PointerPhase::Down => {
                        layer.push(egui::Event::PointerMoved(pos));
                        layer.push(egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed: true,
                            modifiers: layer.modifiers(),
                        });
                    }
                    PointerPhase::Move => {
                        layer.push(egui::Event::PointerMoved(pos));
                    }
                    PointerPhase::Up => {
                        layer.push(egui::Event::PointerButton {
                            pos,
                            button: egui::PointerButton::Primary,
                            pressed: false,
                            modifiers: layer.modifiers(),
                        });
                        layer.push(egui::Event::PointerGone);
                    }
                    PointerPhase::Cancel => {}
                }
            }
            consumed
        }
        InputMessage::Resize { .. }
        | InputMessage::Viewport { .. }
        | InputMessage::Visibility(_)
        | InputMessage::Start
        | InputMessage::PointerLockChanged { .. }
        | InputMessage::Host { .. } => false,
    }
}

enum Canvas {
    Offscreen(OffscreenCanvas),
    Page(HtmlCanvasElement),
}

impl Canvas {
    fn set_size(&self, width: u32, height: u32) {
        match self {
            Self::Offscreen(canvas) => {
                canvas.set_width(width);
                canvas.set_height(height);
            }
            Self::Page(canvas) => {
                canvas.set_width(width);
                canvas.set_height(height);
            }
        }
    }

    fn target(&self) -> wgpu::SurfaceTarget<'static> {
        match self {
            Self::Offscreen(canvas) => wgpu::SurfaceTarget::OffscreenCanvas(canvas.clone()),
            Self::Page(canvas) => wgpu::SurfaceTarget::Canvas(canvas.clone()),
        }
    }
}

struct Worker<A: Stores> {
    frame: Frame<A>,
    surface: SurfaceHost,
    rd: RenderDevice,
    canvas: Canvas,
    endpoint: Endpoint,
    messages: VecDeque<InputMessage>,
    #[cfg(feature = "egui")]
    touches: TouchCapture,
    dpr: f32,
    presented: bool,
}

impl<A: Stores> Worker<A> {
    fn resize(&mut self, width: u32, height: u32, dpr: f32) {
        self.canvas.set_size(width, height);
        let size = (width, height);
        self.surface.resize(&self.rd.context.device, size);
        self.rd.resize(size);
        self.dpr = dpr;
        self.frame.resize(width, height, dpr);
    }

    fn apply(&mut self, message: InputMessage) {
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
        #[cfg(feature = "egui")]
        let consumed = self
            .frame
            .layer()
            .cloned()
            .is_some_and(|layer| feed_layer(&layer, &mut self.touches, &message));
        #[cfg(not(feature = "egui"))]
        let consumed = false;
        match &message {
            InputMessage::Resize { width, height, dpr } => self.resize(*width, *height, *dpr),
            _ => self.frame.apply_message(&message, consumed),
        }
    }

    fn animate(&mut self) -> Result<Attempt, HostError> {
        queue::drain_messages_into(&mut self.messages);
        while let Some(message) = self.messages.pop_front() {
            self.apply(message);
        }
        self.flush_cursor_request();
        let now = Instant::now();
        let Worker {
            frame,
            surface,
            rd,
            endpoint,
            presented,
            ..
        } = self;
        let attempt = surface.present(rd, frame, now, |frame, _| {
            if let Some(locked) = frame.take_cursor_request() {
                post_cursor_request(endpoint, locked);
            }
            for (topic, values) in frame.posts() {
                post_to_page(endpoint, topic, values);
            }
        })?;
        if matches!(attempt, Attempt::Presented) && !*presented {
            *presented = true;
            post(endpoint, Kind::Presented);
        }
        Ok(attempt)
    }

    fn flush_cursor_request(&mut self) {
        if let Some(locked) = self.frame.take_cursor_request() {
            post_cursor_request(&self.endpoint, locked);
        }
    }
}
