use anyhow::{anyhow, Context, Result};
use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    Element, HtmlCanvasElement, MessageChannel, MessageEvent, MessagePort, Window, Worker,
    WorkerOptions, WorkerType,
};

use super::input_queue::MESSAGE_QUEUE_CAPACITY;
use crate::{LaunchMode, WasmConfig};

#[derive(Clone, Copy, PartialEq)]
struct CanvasMetrics {
    width: u32,
    height: u32,
    scale: f32,
}

type Pending<T> = Rc<RefCell<Option<T>>>;

pub(crate) type InPage = Box<dyn FnOnce(MessagePort, Window, HtmlCanvasElement) -> Result<()>>;

#[derive(Clone)]
enum Link {
    Worker(Worker),
    Page(MessagePort),
}

impl Link {
    fn post_message(&self, message: &JsValue) -> Result<(), JsValue> {
        match self {
            Self::Worker(worker) => worker.post_message(message),
            Self::Page(port) => port.post_message(message),
        }
    }

    fn post_message_with_transfer(
        &self,
        message: &JsValue,
        transfer: &JsValue,
    ) -> Result<(), JsValue> {
        match self {
            Self::Worker(worker) => worker.post_message_with_transfer(message, transfer),
            Self::Page(port) => port.post_message_with_transferable(message, transfer),
        }
    }

    fn add_event_listener_with_callback(
        &self,
        kind: &str,
        callback: &js_sys::Function,
    ) -> Result<(), JsValue> {
        match self {
            Self::Worker(worker) => worker.add_event_listener_with_callback(kind, callback),
            Self::Page(port) => port.add_event_listener_with_callback(kind, callback),
        }
    }
}

thread_local! {
    static PAGE_HOST: RefCell<Option<(Element, String)>> = const { RefCell::new(None) };
}

pub(crate) fn fail_page(message: &str) {
    PAGE_HOST.with(|slot| {
        if let Some((host, button_id)) = slot.borrow().as_ref() {
            show_worker_failure(message, host, button_id);
        }
    });
}

// Gecko copies a worker's WebGPU canvas back on every page repaint; only Gecko has navigator.buildID.
fn runs_in_page(window: &Window) -> bool {
    match crate::args::Args::current().get("loam-session") {
        Some("page") => true,
        Some("worker") => false,
        _ => js_sys::Reflect::has(&window.navigator(), &JsValue::from_str("buildID"))
            .unwrap_or(false),
    }
}

const HOST_STATE_ATTR: &str = "data-state";
const HOST_STATE_EVENT: &str = "loam-state";
const HOST_CONTROL_EVENT: &str = "loam-control";
const HOST_MESSAGE_EVENT: &str = "loam-message";
const HOST_POST_EVENT: &str = "loam-post";
// A page iterable can be unbounded; this caps the copy per message.
const MAX_HOST_VALUES: usize = 64;

struct HostMessage {
    topic: String,
    values: Vec<f32>,
}

#[derive(Default)]
struct HostChannel {
    ready: bool,
    control: Option<&'static str>,
    pending: VecDeque<HostMessage>,
}

impl HostChannel {
    fn push(&mut self, topic: &str, values: Vec<f32>) {
        if !values.is_empty() {
            let earlier = self
                .pending
                .iter_mut()
                .rev()
                .take_while(|message| !message.values.is_empty())
                .find(|message| message.topic == topic);
            if let Some(earlier) = earlier {
                earlier.values = values;
                return;
            }
        }
        if self.pending.len() >= MESSAGE_QUEUE_CAPACITY {
            self.pending.pop_front();
        }
        self.pending.push_back(HostMessage {
            topic: topic.to_owned(),
            values,
        });
    }

    fn flush(&mut self, worker: &Link) {
        if !self.ready {
            return;
        }
        for message in self.pending.drain(..) {
            let msg = build_msg("host_message");
            set_msg_string(&msg, "topic", &message.topic);
            let _ = js_sys::Reflect::set(
                &msg,
                &JsValue::from_str("values"),
                &js_sys::Float32Array::from(message.values.as_slice()),
            );
            if let Err(error) = worker.post_message(&msg) {
                tracing::error!("loam_app::wasm: postMessage host_message failed: {error:?}");
            }
        }
    }
}

fn read_host_values(values: &JsValue) -> Vec<f32> {
    if let Some(number) = values.as_f64() {
        return vec![number as f32];
    }
    match js_sys::try_iter(values) {
        Ok(Some(items)) => items
            .take(MAX_HOST_VALUES)
            .map(|item| item.ok().and_then(|item| item.as_f64()).unwrap_or(f64::NAN) as f32)
            .collect(),
        _ => Vec::new(),
    }
}

fn install_host_channel(
    worker: &Link,
    host: &Element,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<()> {
    {
        let channel = channel.clone();
        let callback = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
            let detail = event.detail();
            let topic = js_sys::Reflect::get(&detail, &JsValue::from_str("topic"))
                .ok()
                .and_then(|value| value.as_string());
            let Some(topic) = topic else {
                tracing::warn!(
                    "loam_app::wasm: {HOST_MESSAGE_EVENT} without a string topic ignored"
                );
                return;
            };
            let values = js_sys::Reflect::get(&detail, &JsValue::from_str("values"))
                .map(|values| read_host_values(&values))
                .unwrap_or_default();
            channel.borrow_mut().push(&topic, values);
        }) as Box<dyn FnMut(web_sys::CustomEvent)>);
        host.add_event_listener_with_callback(
            HOST_MESSAGE_EVENT,
            callback.as_ref().unchecked_ref(),
        )
        .map_err(|error| anyhow!("addEventListener('{HOST_MESSAGE_EVENT}'): {error:?}"))?;
        callback.forget();
    }

    let host = host.clone();
    let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|value| value.as_string());
        if kind.as_deref() != Some("host_post") {
            return;
        }
        let detail = js_sys::Object::new();
        for key in ["topic", "values"] {
            let key = JsValue::from_str(key);
            if let Ok(value) = js_sys::Reflect::get(&data, &key) {
                let _ = js_sys::Reflect::set(&detail, &key, &value);
            }
        }
        let init = web_sys::CustomEventInit::new();
        init.set_detail(&detail);
        match web_sys::CustomEvent::new_with_event_init_dict(HOST_POST_EVENT, &init) {
            Ok(event) => {
                let _ = host.dispatch_event(&event);
            }
            Err(error) => {
                tracing::error!("loam_app::wasm: create {HOST_POST_EVENT} event: {error:?}")
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
        .map_err(|error| anyhow!("worker.addEventListener('message') for posts: {error:?}"))?;
    callback.forget();
    Ok(())
}

pub(crate) fn launch_page(config: &WasmConfig, in_page: InPage) -> Result<()> {
    super::install_logging_idempotent();
    if let Some(host) = host_element(&config.host_id) {
        if host.has_attribute(HOST_STATE_ATTR) {
            tracing::warn!(
                "loam_app::wasm: host '{}' already carries {HOST_STATE_ATTR}; second launch ignored",
                config.host_id
            );
            return Ok(());
        }
    }
    spawn_session(config, in_page).inspect_err(|error| {
        tracing::error!("loam_app::wasm: launch failed: {error:#}");
        if let Some(host) = host_element(&config.host_id) {
            set_host_state(&host, "failed");
        }
    })
}

fn host_element(host_id: &str) -> Option<Element> {
    web_sys::window()
        .and_then(|window| window.document())
        .and_then(|document| document.get_element_by_id(host_id))
}

fn set_host_state(host: &Element, state: &str) {
    let current = host.get_attribute(HOST_STATE_ATTR);
    if matches!(current.as_deref(), Some(s) if s == state || s == "failed") {
        return;
    }
    let _ = host.set_attribute(HOST_STATE_ATTR, state);
    let detail = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &detail,
        &JsValue::from_str("state"),
        &JsValue::from_str(state),
    );
    let init = web_sys::CustomEventInit::new();
    init.set_detail(&detail);
    match web_sys::CustomEvent::new_with_event_init_dict(HOST_STATE_EVENT, &init) {
        Ok(event) => {
            let _ = host.dispatch_event(&event);
        }
        Err(error) => tracing::error!("loam_app::wasm: create {HOST_STATE_EVENT} event: {error:?}"),
    }
}

// Set by the demo's inline script (`window.__loam_wasm_url = import.meta.url`).
fn read_wasm_bundle_url() -> Result<String> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let val = js_sys::Reflect::get(&window, &JsValue::from_str("__loam_wasm_url"))
        .map_err(|e| anyhow!("read __loam_wasm_url: {e:?}"))?;
    val.as_string()
        .ok_or_else(|| anyhow!("__loam_wasm_url is not a string; demo's index.html must set it"))
}

fn spawn_session(config: &WasmConfig, in_page: InPage) -> Result<()> {
    let host_id = config.host_id.as_str();
    let button_id = config.button_id.as_str();
    let canvas_id = config.canvas_id.as_str();
    let background = config.mode == LaunchMode::Background;
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let host = document
        .get_element_by_id(host_id)
        .ok_or_else(|| anyhow!("no host element with id '{host_id}'"))?;
    let canvas = document
        .get_element_by_id(canvas_id)
        .ok_or_else(|| anyhow!("no element with id '{canvas_id}'"))?
        .dyn_into::<HtmlCanvasElement>()
        .map_err(|_| anyhow!("element '{canvas_id}' is not a canvas"))?;
    let launch_overlay = if background {
        None
    } else {
        Some(super::launch::inject_launch_overlay(host_id, button_id)?)
    };

    let window = web_sys::window().ok_or_else(|| anyhow!("no global window"))?;
    let dpr = window.device_pixel_ratio() as f32;
    let max_pixels = crate::args::Args::current()
        .parse::<u32>("max-pixels")
        .or(config.max_pixels)
        .filter(|pixels| *pixels > 0);
    let css = (canvas.client_width(), canvas.client_height());
    let probe = canvas.width();
    canvas.set_width(probe + 1);
    let follows_buffer = canvas.client_width() != css.0;
    canvas.set_width(probe);
    if follows_buffer {
        let style = canvas.style();
        let _ = style.set_property("width", &format!("{}px", css.0.max(1)));
        let _ = style.set_property("height", &format!("{}px", css.1.max(1)));
        tracing::warn!(
            "loam_app::wasm: canvas '{canvas_id}' has no CSS size; pinned to {}x{} CSS px",
            css.0,
            css.1
        );
    }
    let metrics = canvas_metrics(&canvas, dpr, max_pixels);
    let width = metrics.width;
    let height = metrics.height;
    canvas.set_width(width);
    canvas.set_height(height);
    tracing::info!(
        "loam_app::wasm::worker: canvas sized to {width}x{height} (scale {})",
        metrics.scale
    );
    PAGE_HOST.with(|slot| *slot.borrow_mut() = Some((host.clone(), button_id.to_string())));

    let (worker, offscreen, bootstrap_url) = if runs_in_page(&window) {
        tracing::info!("loam_app::wasm: running the session on the page");
        let channel = MessageChannel::new().map_err(|e| anyhow!("MessageChannel::new: {e:?}"))?;
        in_page(channel.port2(), window.clone(), canvas.clone())?;
        (Link::Page(channel.port1()), None, None)
    } else {
        let (worker, offscreen, blob_url) = spawn_worker(&canvas)?;
        (Link::Worker(worker), Some(offscreen), Some(blob_url))
    };
    let channel: Rc<RefCell<HostChannel>> = Rc::new(RefCell::new(HostChannel::default()));
    let worker_for_ready = worker.clone();
    let offscreen_for_ready = offscreen;
    let host_for_ready = host.clone();
    let button_for_ready = button_id.to_string();
    let document_for_ready = document.clone();
    let channel_for_ready = channel.clone();
    let on_ready = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("ready") {
            return;
        }
        if let Some(url) = bootstrap_url.as_deref() {
            let _ = web_sys::Url::revoke_object_url(url);
        }

        let msg = build_msg("init");
        set_msg_u32(&msg, "width", width);
        set_msg_u32(&msg, "height", height);
        set_msg_f32(&msg, "dpr", metrics.scale);
        let (search, hash) = match (background, web_sys::window()) {
            (false, Some(window)) => {
                let loc = window.location();
                (
                    loc.search().unwrap_or_default(),
                    loc.hash().unwrap_or_default(),
                )
            }
            _ => (String::new(), String::new()),
        };
        set_msg_string(&msg, "search", &search);
        set_msg_string(&msg, "hash", &hash);
        set_msg_bool(&msg, "visible", is_visible(&document_for_ready));
        if let Some(max_pixels) = max_pixels {
            set_msg_u32(&msg, "max_pixels", max_pixels);
        }
        let posted = match offscreen_for_ready.as_ref() {
            Some(offscreen) => {
                let _ = js_sys::Reflect::set(&msg, &JsValue::from_str("canvas"), offscreen);
                let transfer = js_sys::Array::new();
                transfer.push(offscreen);
                worker_for_ready.post_message_with_transfer(&msg, &transfer)
            }
            None => worker_for_ready.post_message(&msg),
        };
        if let Err(e) = posted {
            show_worker_failure(
                &format!("worker initialization message failed: {e:?}"),
                &host_for_ready,
                &button_for_ready,
            );
            return;
        }
        if background {
            if let Err(e) = worker_for_ready.post_message(&build_msg("start")) {
                show_worker_failure(
                    &format!("worker start message failed: {e:?}"),
                    &host_for_ready,
                    &button_for_ready,
                );
            }
        }
        let control = {
            let mut channel = channel_for_ready.borrow_mut();
            channel.ready = true;
            channel.control.take()
        };
        if let Some(kind) = control {
            if let Err(error) = worker_for_ready.post_message(&build_msg(kind)) {
                tracing::error!("loam_app::wasm: postMessage {kind} failed: {error:?}");
            }
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", on_ready.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message'): {e:?}"))?;
    on_ready.forget();

    install_cursor_control(&worker, &canvas, background).context("install_cursor_control")?;
    install_dom_input_forwarders(&worker, &canvas, metrics, max_pixels, background, &channel)
        .context("install_dom_input_forwarders")?;
    install_host_channel(&worker, &host, &channel).context("install_host_channel")?;

    install_worker_state_handler(&worker, &host, button_id)?;
    install_worker_failure_handler(&worker, &host, button_id)?;
    install_page_control(&worker, &host, &channel)?;
    set_host_state(&host, "loading");
    #[cfg(feature = "measure")]
    install_measurement_result_handler(&worker, host_id)?;
    if let Link::Page(port) = &worker {
        port.start();
    }

    let Some(launch_overlay) = launch_overlay else {
        return Ok(());
    };
    install_embed_lifecycle(&worker, host_id, button_id).context("install_embed_lifecycle")?;

    {
        let worker_for_click = worker.clone();
        let overlay_for_click = launch_overlay.clone();
        let host_for_click = host_id.to_string();
        let fired: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));
        let on_click = Closure::wrap(Box::new(move || {
            if fired.get() {
                tracing::debug!("loam_app::wasm::worker: launch click ignored (already fired)");
                return;
            }
            if !overlay_for_click.class_name().contains("ready") {
                tracing::debug!(
                    "loam_app::wasm::worker: launch click ignored (not yet ready, overlay state = {})",
                    overlay_for_click.class_name()
                );
                return;
            }
            fired.set(true);

            let msg = build_msg("start");
            if let Err(e) = worker_for_click.post_message(&msg) {
                fired.set(false);
                tracing::error!("post start failed: {e:?}");
                return;
            }

            overlay_for_click.remove();
            dispatch_embed_activated(&host_for_click);
        }) as Box<dyn FnMut()>);
        launch_overlay
            .add_event_listener_with_callback("click", on_click.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("launch overlay click listener: {e:?}"))?;
        on_click.forget();
    }

    Ok(())
}

fn spawn_worker(canvas: &HtmlCanvasElement) -> Result<(Worker, web_sys::OffscreenCanvas, String)> {
    let offscreen = canvas
        .transfer_control_to_offscreen()
        .map_err(|e| anyhow!("transfer_control_to_offscreen: {e:?}"))?;

    let js_url = read_wasm_bundle_url()?;
    let (js_path, query) = js_url.split_once('?').unwrap_or((js_url.as_str(), ""));
    let wasm_url = format!(
        "{}_bg.wasm{}{query}",
        js_path.strip_suffix(".js").unwrap_or(js_path),
        if query.is_empty() { "" } else { "?" }
    );
    tracing::info!("loam_app::wasm::worker: spawning worker (js={js_url}, wasm={wasm_url})");

    let bootstrap_js =
        format!("import init from {js_url:?};\nawait init({{ module_or_path: {wasm_url:?} }});\n");
    let blob_parts = js_sys::Array::new();
    blob_parts.push(&JsValue::from_str(&bootstrap_js));
    let blob_options = web_sys::BlobPropertyBag::new();
    blob_options.set_type("application/javascript");
    let blob = web_sys::Blob::new_with_str_sequence_and_options(&blob_parts, &blob_options)
        .map_err(|e| anyhow!("Blob::new: {e:?}"))?;
    let blob_url = web_sys::Url::create_object_url_with_blob(&blob)
        .map_err(|e| anyhow!("createObjectURL: {e:?}"))?;

    let opts = WorkerOptions::new();
    opts.set_type(WorkerType::Module);
    let worker =
        Worker::new_with_options(&blob_url, &opts).map_err(|e| anyhow!("Worker::new: {e:?}"))?;
    Ok((worker, offscreen, blob_url))
}

fn is_visible(document: &web_sys::Document) -> bool {
    document.visibility_state() != web_sys::VisibilityState::Hidden
}

fn canvas_metrics(
    canvas: &HtmlCanvasElement,
    device_scale: f32,
    max_pixels: Option<u32>,
) -> CanvasMetrics {
    let (width, height, scale) = super::metrics::capped(
        canvas.client_width().max(1) as u32,
        canvas.client_height().max(1) as u32,
        device_scale,
        max_pixels,
    );
    CanvasMetrics {
        width,
        height,
        scale,
    }
}

fn install_page_control(
    worker: &Link,
    host: &Element,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<()> {
    let worker = worker.clone();
    let channel = channel.clone();
    let callback = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
        let detail = event.detail().as_string();
        let kind = match detail.as_deref() {
            Some("pause") => "host_pause",
            Some("resume") => "host_resume",
            other => {
                tracing::warn!("loam_app::wasm: {HOST_CONTROL_EVENT} ignored: {other:?}");
                return;
            }
        };
        let mut channel = channel.borrow_mut();
        if !channel.ready {
            channel.control = Some(kind);
            return;
        }
        if let Err(error) = worker.post_message(&build_msg(kind)) {
            tracing::error!("loam_app::wasm: postMessage {kind} failed: {error:?}");
        }
    }) as Box<dyn FnMut(web_sys::CustomEvent)>);
    host.add_event_listener_with_callback(HOST_CONTROL_EVENT, callback.as_ref().unchecked_ref())
        .map_err(|error| anyhow!("addEventListener('{HOST_CONTROL_EVENT}'): {error:?}"))?;
    callback.forget();
    Ok(())
}

// Dispatched on `document` when an embed activates; detail = host id.
const EMBED_ACTIVATED_EVENT: &str = "loam-embed-activated";

fn dispatch_embed_activated(host_id: &str) {
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let init = web_sys::CustomEventInit::new();
    init.set_detail(&JsValue::from_str(host_id));
    match web_sys::CustomEvent::new_with_event_init_dict(EMBED_ACTIVATED_EVENT, &init) {
        Ok(event) => {
            let _ = document.dispatch_event(&event);
        }
        Err(e) => tracing::error!("loam_app::wasm::worker: create activated event: {e:?}"),
    }
}

fn install_embed_lifecycle(worker: &Link, host_id: &str, button_id: &str) -> Result<()> {
    let document = web_sys::window()
        .and_then(|w| w.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let host_el = document
        .get_element_by_id(host_id)
        .ok_or_else(|| anyhow!("no host element with id '{host_id}'"))?;
    let active: Rc<std::cell::Cell<bool>> = Rc::new(std::cell::Cell::new(false));

    let worker_for_resume = worker.clone();
    let host_for_resume = host_id.to_string();
    let button_for_resume = button_id.to_string();
    let on_resume_click: Rc<Closure<dyn FnMut()>> = Rc::new(Closure::wrap(Box::new(move || {
        let msg = build_msg("resume");
        if let Err(e) = worker_for_resume.post_message(&msg) {
            tracing::error!(
                "loam_app::wasm::worker: postMessage resume failed: {e:?}; \
                 overlay retained for retry"
            );
            return;
        }
        if let Some(doc) = web_sys::window().and_then(|w| w.document()) {
            if let Some(button) = doc.get_element_by_id(&button_for_resume) {
                button.remove();
            }
        }
        dispatch_embed_activated(&host_for_resume);
    })
        as Box<dyn FnMut()>));

    let worker_for_deact = worker.clone();
    let host_for_deact = host_id.to_string();
    let button_for_deact = button_id.to_string();
    let active_for_deact = active.clone();
    let resume_cb = on_resume_click.clone();
    let deactivate: Rc<dyn Fn()> = Rc::new(move || {
        if !active_for_deact.replace(false) {
            return;
        }
        let msg = build_msg("pause");
        if let Err(e) = worker_for_deact.post_message(&msg) {
            tracing::error!("loam_app::wasm::worker: postMessage pause failed: {e:?}");
        }
        match super::launch::show_resume_overlay(&host_for_deact, &button_for_deact) {
            Ok(button) => {
                if let Err(e) = button.add_event_listener_with_callback(
                    "click",
                    (*resume_cb).as_ref().unchecked_ref(),
                ) {
                    tracing::error!("loam_app::wasm::worker: resume click listener: {e:?}");
                }
            }
            Err(e) => tracing::error!("loam_app::wasm::worker: show_resume_overlay: {e:#}"),
        }
    });

    let active_for_evt = active.clone();
    let host_for_evt = host_id.to_string();
    let deactivate_for_evt = deactivate.clone();
    let on_activated = Closure::wrap(Box::new(move |event: web_sys::CustomEvent| {
        if event.detail().as_string().as_deref() == Some(host_for_evt.as_str()) {
            active_for_evt.set(true);
        } else {
            deactivate_for_evt();
        }
    }) as Box<dyn FnMut(web_sys::CustomEvent)>);
    document
        .add_event_listener_with_callback(
            EMBED_ACTIVATED_EVENT,
            on_activated.as_ref().unchecked_ref(),
        )
        .map_err(|e| anyhow!("addEventListener('{EMBED_ACTIVATED_EVENT}'): {e:?}"))?;
    on_activated.forget();

    let active_for_ptr = active.clone();
    let deactivate_for_ptr = deactivate.clone();
    let on_pointerdown = Closure::wrap(Box::new(move |event: web_sys::Event| {
        if !active_for_ptr.get() {
            return;
        }
        let inside = event
            .target()
            .and_then(|t| t.dyn_into::<web_sys::Node>().ok())
            .map(|node| host_el.contains(Some(&node)))
            .unwrap_or(false);
        if !inside {
            deactivate_for_ptr();
        }
    }) as Box<dyn FnMut(web_sys::Event)>);
    document
        .add_event_listener_with_callback_and_bool(
            "pointerdown",
            on_pointerdown.as_ref().unchecked_ref(),
            true,
        )
        .map_err(|e| anyhow!("addEventListener('pointerdown'): {e:?}"))?;
    on_pointerdown.forget();

    Ok(())
}

fn install_cursor_control(
    worker: &Link,
    canvas: &HtmlCanvasElement,
    background: bool,
) -> Result<()> {
    let document = web_sys::window()
        .and_then(|window| window.document())
        .ok_or_else(|| anyhow!("no document on global window"))?;
    let wanted = Rc::new(Cell::new(false));

    {
        let worker = worker.clone();
        let worker_for_callback = worker.clone();
        let canvas = canvas.clone();
        let document = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
            let data: JsValue = event.data();
            let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
                .ok()
                .and_then(|value| value.as_string());
            if kind.as_deref() != Some("cursor_lock") {
                return;
            }
            let locked = js_sys::Reflect::get(&data, &JsValue::from_str("locked"))
                .ok()
                .and_then(|value| value.as_bool())
                .unwrap_or(false);
            wanted.set(locked);
            if locked {
                request_pointer_lock(&canvas);
            } else if document
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref())
            {
                document.exit_pointer_lock();
            } else {
                let message = build_msg("pointer_lock_changed");
                set_msg_bool(&message, "locked", false);
                set_msg_bool(&message, "released", false);
                let _ = worker_for_callback.post_message(&message);
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        worker
            .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
            .map_err(|error| anyhow!("cursor lock message listener: {error:?}"))?;
        callback.forget();
    }

    for event_name in ["pointerlockchange", "pointerlockerror"] {
        let release_event = event_name == "pointerlockchange";
        let worker = worker.clone();
        let canvas = canvas.clone();
        let document_for_event = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document_for_event
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref());
            let released = release_event && !locked && wanted.replace(false);
            if locked && !wanted.get() {
                document_for_event.exit_pointer_lock();
            }
            let message = build_msg("pointer_lock_changed");
            set_msg_bool(&message, "locked", locked);
            set_msg_bool(&message, "released", released);
            let _ = worker.post_message(&message);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback(event_name, callback.as_ref().unchecked_ref())
            .map_err(|error| anyhow!("{event_name} listener: {error:?}"))?;
        callback.forget();
    }

    {
        let canvas = canvas.clone();
        let canvas_for_callback = canvas.clone();
        let document = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas_for_callback.as_ref());
            if wanted.get() && !locked {
                request_pointer_lock(&canvas_for_callback);
            }
        }) as Box<dyn FnMut()>);
        canvas
            .add_event_listener_with_callback("mousedown", callback.as_ref().unchecked_ref())
            .map_err(|error| anyhow!("cursor lock mousedown listener: {error:?}"))?;
        callback.forget();
    }

    if !background {
        let canvas = canvas.clone();
        let document_for_event = document.clone();
        let wanted = wanted.clone();
        let callback = Closure::wrap(Box::new(move || {
            let locked = document_for_event
                .pointer_lock_element()
                .as_ref()
                .is_some_and(|element| element == canvas.as_ref());
            if wanted.get() && !locked {
                request_pointer_lock(&canvas);
            }
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("keydown", callback.as_ref().unchecked_ref())
            .map_err(|error| anyhow!("cursor lock keydown listener: {error:?}"))?;
        callback.forget();
    }

    Ok(())
}

fn request_pointer_lock(canvas: &HtmlCanvasElement) {
    let Ok(method) =
        js_sys::Reflect::get(canvas.as_ref(), &JsValue::from_str("requestPointerLock"))
            .and_then(|value| value.dyn_into::<js_sys::Function>())
    else {
        tracing::warn!("loam_app::wasm::worker: requestPointerLock is unavailable");
        return;
    };
    let Ok(result) = method.call0(canvas.as_ref()) else {
        tracing::warn!("loam_app::wasm::worker: requestPointerLock failed");
        return;
    };
    let Ok(promise) = result.dyn_into::<js_sys::Promise>() else {
        return;
    };
    wasm_bindgen_futures::spawn_local(async move {
        if let Err(error) = wasm_bindgen_futures::JsFuture::from(promise).await {
            tracing::warn!("loam_app::wasm::worker: requestPointerLock rejected: {error:?}");
        }
    });
}

fn install_dom_input_forwarders(
    worker: &Link,
    canvas: &HtmlCanvasElement,
    launched: CanvasMetrics,
    max_pixels: Option<u32>,
    background: bool,
    channel: &Rc<RefCell<HostChannel>>,
) -> Result<()> {
    let window = web_sys::window().ok_or_else(|| anyhow!("no window"))?;
    let document = window
        .document()
        .ok_or_else(|| anyhow!("no document on window"))?;

    const RESIZE_DEBOUNCE_FRAMES: u32 = 6;
    {
        let pending: Pending<(CanvasMetrics, u32)> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let viewport: Pending<(f32, f32)> = Rc::new(RefCell::new(None));
        let viewport_for_listener = viewport.clone();
        let canvas_for_listener = canvas.clone();
        let window_for_listener = window.clone();
        let cb = Closure::wrap(Box::new(move || {
            let metrics = canvas_metrics(
                &canvas_for_listener,
                window_for_listener.device_pixel_ratio() as f32,
                max_pixels,
            );
            *pending_for_listener.borrow_mut() = Some((metrics, 0));
            *viewport_for_listener.borrow_mut() = Some((
                canvas_for_listener.client_width().max(1) as f32,
                canvas_for_listener.client_height().max(1) as f32,
            ));
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback("resize", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("window resize listener: {e:?}"))?;
        let observer = web_sys::ResizeObserver::new(cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("ResizeObserver::new: {e:?}"))?;
        observer.observe(canvas);
        cb.forget();

        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let channel_for_resize = channel.clone();
        let committed = Cell::new(launched);
        let raf_cb: Pending<Closure<dyn FnMut()>> = Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            if channel_for_resize.borrow().ready {
                if let Some((width, height)) = viewport.borrow_mut().take() {
                    let msg = build_msg("viewport");
                    set_msg_f32(&msg, "width", width);
                    set_msg_f32(&msg, "height", height);
                    let _ = worker_for_raf.post_message(&msg);
                }
            }
            let commit = {
                let mut p = pending_for_raf.borrow_mut();
                match p.as_mut() {
                    Some((_, frames)) => {
                        *frames += 1;
                        if *frames >= RESIZE_DEBOUNCE_FRAMES && channel_for_resize.borrow().ready {
                            p.take().map(|(metrics, _)| metrics)
                        } else {
                            None
                        }
                    }
                    None => None,
                }
            };
            if let Some(metrics) = commit.filter(|metrics| *metrics != committed.get()) {
                committed.set(metrics);
                let msg = build_msg("resize");
                set_msg_u32(&msg, "width", metrics.width);
                set_msg_u32(&msg, "height", metrics.height);
                set_msg_f32(&msg, "dpr", metrics.scale);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(|e| anyhow!("resize rAF init: {e:?}"))?;
            }
        }
    }

    {
        let pending: Pending<(f32, f32, u32, f32, f32, f64)> = Rc::new(RefCell::new(None));
        let pending_for_listener = pending.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let mut p = pending_for_listener.borrow_mut();
            let (sum_dx, sum_dy) = match *p {
                Some((_, _, _, dx, dy, _)) => (dx, dy),
                None => (0.0, 0.0),
            };
            *p = Some((
                ev.offset_x() as f32,
                ev.offset_y() as f32,
                ev.buttons() as u32,
                sum_dx + ev.movement_x() as f32,
                sum_dy + ev.movement_y() as f32,
                ev.time_stamp(),
            ));
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback("mousemove", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("mousemove listener: {e:?}"))?;
        cb.forget();

        let worker_for_raf = worker.clone();
        let pending_for_raf = pending.clone();
        let window_for_raf = window.clone();
        let channel_for_raf = channel.clone();
        let raf_cb: Pending<Closure<dyn FnMut()>> = Rc::new(RefCell::new(None));
        let raf_cb_for_closure = raf_cb.clone();
        *raf_cb.borrow_mut() = Some(Closure::wrap(Box::new(move || {
            channel_for_raf.borrow_mut().flush(&worker_for_raf);
            if let Some((x, y, buttons, dx, dy, time)) = pending_for_raf.borrow_mut().take() {
                let msg = build_msg("mouse_move");
                set_msg_f32(&msg, "x", x);
                set_msg_f32(&msg, "y", y);
                set_msg_u32(&msg, "buttons", buttons);
                set_msg_f32(&msg, "dx", dx);
                set_msg_f32(&msg, "dy", dy);
                set_msg_f64(&msg, "time", time);
                let _ = worker_for_raf.post_message(&msg);
            }
            let cb_ref = raf_cb_for_closure.borrow();
            if let Some(cb) = cb_ref.as_ref() {
                let _ = window_for_raf.request_animation_frame(cb.as_ref().unchecked_ref());
            }
        }) as Box<dyn FnMut()>));
        {
            let first_cb = raf_cb.borrow();
            if let Some(first) = first_cb.as_ref() {
                window
                    .request_animation_frame(first.as_ref().unchecked_ref())
                    .map_err(|e| anyhow!("mousemove rAF init: {e:?}"))?;
            }
        }
    }

    for (event_name, pressed) in [("mousedown", true), ("mouseup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::MouseEvent| {
            let prevent = if background {
                pressed && ev.button() == 0
            } else {
                ev.button() != 0
            };
            if prevent {
                ev.prevent_default();
            }
            let msg = build_msg("mouse_button");
            set_msg_f32(&msg, "x", ev.offset_x() as f32);
            set_msg_f32(&msg, "y", ev.offset_y() as f32);
            set_msg_u32(&msg, "button", ev.button() as u32);
            set_msg_bool(&msg, "pressed", pressed);
            set_msg_f64(&msg, "time", ev.time_stamp());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::MouseEvent)>);
        canvas
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    if !background {
        let _ = canvas.style().set_property("touch-action", "none");
    }
    for event_name in ["pointerdown", "pointermove", "pointerup", "pointercancel"] {
        let worker = worker.clone();
        let canvas_for_capture = canvas.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::PointerEvent| {
            if event_name == "pointerdown" {
                let _ = canvas_for_capture.set_pointer_capture(ev.pointer_id());
            }
            if ev.pointer_type() == "mouse" {
                return;
            }
            let msg = build_msg("pointer");
            set_msg_f64(&msg, "id", f64::from(ev.pointer_id()));
            set_msg_f32(&msg, "x", ev.offset_x() as f32);
            set_msg_f32(&msg, "y", ev.offset_y() as f32);
            set_msg_string(&msg, "phase", event_name);
            set_msg_f64(&msg, "time", ev.time_stamp());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::PointerEvent)>);
        canvas
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    if background {
        install_focus_and_visibility(worker, &window, &document)?;
        return Ok(());
    }

    {
        let cb = Closure::wrap(Box::new(move |ev: web_sys::Event| {
            ev.prevent_default();
        }) as Box<dyn FnMut(web_sys::Event)>);
        canvas
            .add_event_listener_with_callback("contextmenu", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("contextmenu listener: {e:?}"))?;
        cb.forget();
    }

    {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::WheelEvent| {
            let (dx, dy) = match ev.delta_mode() {
                1 => (ev.delta_x() as f32, ev.delta_y() as f32),
                _ => (ev.delta_x() as f32 / 100.0, ev.delta_y() as f32 / 100.0),
            };
            ev.prevent_default();
            let msg = build_msg("mouse_wheel");
            set_msg_f32(&msg, "dx", dx);
            set_msg_f32(&msg, "dy", dy);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::WheelEvent)>);
        canvas
            .add_event_listener_with_callback("wheel", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("wheel listener: {e:?}"))?;
        cb.forget();
    }

    for (event_name, pressed) in [("keydown", true), ("keyup", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move |ev: web_sys::KeyboardEvent| {
            let code = ev.code();
            let no_modifier = !ev.ctrl_key() && !ev.alt_key() && !ev.meta_key();
            let is_alt_self = matches!(code.as_str(), "AltLeft" | "AltRight");
            let suppress_alt = is_alt_self && !ev.ctrl_key() && !ev.meta_key();
            let owned_unmodified = matches!(
                code.as_str(),
                "Tab"
                    | "Space"
                    | "ArrowLeft"
                    | "ArrowRight"
                    | "ArrowUp"
                    | "ArrowDown"
                    | "Slash"
                    | "Quote",
            );
            if suppress_alt || (owned_unmodified && no_modifier) {
                ev.prevent_default();
            }
            let msg = build_msg("key");
            set_msg_string(&msg, "code", &code);
            set_msg_string(&msg, "key", &ev.key());
            set_msg_bool(&msg, "pressed", pressed);
            set_msg_bool(&msg, "repeat", ev.repeat());
            set_msg_bool(&msg, "ctrl", ev.ctrl_key());
            set_msg_bool(&msg, "shift", ev.shift_key());
            set_msg_bool(&msg, "alt", ev.alt_key());
            set_msg_bool(&msg, "meta", ev.meta_key());
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut(web_sys::KeyboardEvent)>);
        document
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    install_focus_and_visibility(worker, &window, &document)
}

fn install_focus_and_visibility(
    worker: &Link,
    window: &web_sys::Window,
    document: &web_sys::Document,
) -> Result<()> {
    for (event_name, focused) in [("focus", true), ("blur", false)] {
        let worker = worker.clone();
        let cb = Closure::wrap(Box::new(move || {
            let msg = build_msg("focus");
            set_msg_bool(&msg, "focused", focused);
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        window
            .add_event_listener_with_callback(event_name, cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("{event_name} listener: {e:?}"))?;
        cb.forget();
    }

    {
        let worker = worker.clone();
        let document_for_query = document.clone();
        let cb = Closure::wrap(Box::new(move || {
            let msg = build_msg("visibility");
            set_msg_bool(&msg, "visible", is_visible(&document_for_query));
            let _ = worker.post_message(&msg);
        }) as Box<dyn FnMut()>);
        document
            .add_event_listener_with_callback("visibilitychange", cb.as_ref().unchecked_ref())
            .map_err(|e| anyhow!("visibilitychange listener: {e:?}"))?;
        cb.forget();
    }

    Ok(())
}

fn build_msg(kind: &str) -> js_sys::Object {
    let obj = js_sys::Object::new();
    let _ = js_sys::Reflect::set(&obj, &JsValue::from_str("kind"), &JsValue::from_str(kind));
    obj
}

fn set_msg_u32(obj: &js_sys::Object, key: &str, v: u32) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(v as f64));
}

fn set_msg_f32(obj: &js_sys::Object, key: &str, v: f32) {
    set_msg_f64(obj, key, f64::from(v));
}

fn set_msg_f64(obj: &js_sys::Object, key: &str, v: f64) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_f64(v));
}

fn set_msg_bool(obj: &js_sys::Object, key: &str, v: bool) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_bool(v));
}

fn set_msg_string(obj: &js_sys::Object, key: &str, v: &str) {
    let _ = js_sys::Reflect::set(obj, &JsValue::from_str(key), &JsValue::from_str(v));
}

fn install_worker_state_handler(worker: &Link, host: &Element, button_id: &str) -> Result<()> {
    let host = host.clone();
    let button_id_owned: String = button_id.to_string();
    let cb = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        match kind.as_deref() {
            Some("presented") => set_host_state(&host, "ready"),
            Some("preview_ready") => {
                let Some(document) = web_sys::window().and_then(|w| w.document()) else {
                    return;
                };
                if let Some(loader) = document.get_element_by_id("loam-page-loader") {
                    let _ = loader.set_attribute("hidden", "");
                }
                if let Some(overlay) = document.get_element_by_id(&button_id_owned) {
                    overlay.set_class_name("loam-demo-launch ready");
                }
            }
            _ => {}
        }
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", cb.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message') for state: {e:?}"))?;
    cb.forget();
    Ok(())
}

#[cfg(feature = "measure")]
fn install_measurement_result_handler(worker: &Link, host_id: &str) -> Result<()> {
    let host_id = host_id.to_owned();
    let callback = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|value| value.as_string());
        if kind.as_deref() != Some("measurement") {
            return;
        }
        let Some(result) = js_sys::Reflect::get(&data, &JsValue::from_str("result"))
            .ok()
            .and_then(|value| value.as_string())
        else {
            return;
        };
        let Some(document) = web_sys::window().and_then(|window| window.document()) else {
            return;
        };
        let Some(host) = document.get_element_by_id(&host_id) else {
            return;
        };
        let Ok(panel) = document.create_element("pre") else {
            return;
        };
        panel.set_text_content(Some(&result));
        let _ = panel.set_attribute("tabindex", "0");
        let _ = panel.set_attribute("role", "status");
        let _ = panel.set_attribute(
            "style",
            "position:fixed;inset:1rem;z-index:2147483647;margin:0;padding:1rem;overflow:auto;white-space:pre-wrap;background:#111;color:#eee;font:14px/1.45 monospace;user-select:text;-webkit-user-select:text;touch-action:auto",
        );
        let _ = host.append_child(&panel);
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
        .map_err(|error| anyhow!("measurement result listener: {error:?}"))?;
    callback.forget();
    Ok(())
}

fn install_worker_failure_handler(worker: &Link, host: &Element, button_id: &str) -> Result<()> {
    let host_for_message = host.clone();
    let button_for_message = button_id.to_string();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        let kind = js_sys::Reflect::get(&data, &JsValue::from_str("kind"))
            .ok()
            .and_then(|v| v.as_string());
        if kind.as_deref() != Some("error") {
            return;
        }
        let message = js_sys::Reflect::get(&data, &JsValue::from_str("message"))
            .ok()
            .and_then(|v| v.as_string())
            .unwrap_or_else(|| "worker error".to_owned());
        show_worker_failure(&message, &host_for_message, &button_for_message);
    }) as Box<dyn FnMut(MessageEvent)>);
    worker
        .add_event_listener_with_callback("message", on_message.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('message') for error: {e:?}"))?;
    on_message.forget();

    let Link::Worker(worker) = worker else {
        return Ok(());
    };
    // A panic traps the worker; the trap arrives here without the panic text.
    let host_for_error = host.clone();
    let button_for_error = button_id.to_string();
    let on_error = Closure::wrap(Box::new(move |event: web_sys::Event| {
        let message = js_sys::Reflect::get(event.as_ref(), &JsValue::from_str("message"))
            .ok()
            .and_then(|value| value.as_string())
            .filter(|message| !message.is_empty())
            .unwrap_or_else(|| "worker stopped without an error message".to_owned());
        show_worker_failure(
            &format!("worker error: {message}"),
            &host_for_error,
            &button_for_error,
        );
    }) as Box<dyn FnMut(web_sys::Event)>);
    worker
        .add_event_listener_with_callback("error", on_error.as_ref().unchecked_ref())
        .map_err(|e| anyhow!("worker.addEventListener('error'): {e:?}"))?;
    on_error.forget();
    Ok(())
}

fn show_worker_failure(message: &str, host: &Element, button_id: &str) {
    tracing::error!("loam_app::wasm: {message}");
    set_host_state(host, "failed");
    let Some(document) = web_sys::window().and_then(|w| w.document()) else {
        return;
    };
    let Some(text) = document.get_element_by_id("loam-page-loader-message") else {
        return;
    };
    if !text.has_attribute("hidden") {
        return;
    }
    text.set_text_content(Some(message));
    let _ = text.remove_attribute("hidden");
    if let Some(track) = document
        .query_selector("#loam-page-loader .loam-progress-track")
        .ok()
        .flatten()
    {
        let _ = track.set_attribute("hidden", "");
    }
    if let Some(loader) = document.get_element_by_id("loam-page-loader") {
        let _ = loader.remove_attribute("hidden");
        let _ = loader.set_attribute("aria-busy", "false");
    }
    if let Some(overlay) = document.get_element_by_id(button_id) {
        overlay.remove();
    }
}
