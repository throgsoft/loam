use anyhow::{anyhow, Context, Result};
use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::Closure;
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{
    Element, HtmlCanvasElement, MessageChannel, MessageEvent, MessagePort, Window, Worker,
    WorkerOptions, WorkerType,
};

mod canvas;
mod embed;
mod forward;
mod host_api;
mod pointer_lock;

use crate::web::messages::Kind;
use crate::{LaunchMode, WasmConfig};
use canvas::canvas_metrics;
use embed::{dispatch_embed_activated, install_embed_lifecycle};
use forward::install_dom_input_forwarders;
use host_api::{install_host_channel, install_page_control, HostChannel};
use pointer_lock::install_cursor_control;

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

// Gecko and WebKit stall on a worker's WebGPU canvas; only Chromium has navigator.userAgentData.
fn runs_in_page(window: &Window) -> bool {
    match crate::args::Args::current().get("loam-session") {
        Some("page") => true,
        Some("worker") => false,
        _ => !js_sys::Reflect::has(&window.navigator(), &JsValue::from_str("userAgentData"))
            .unwrap_or(false),
    }
}

const HOST_STATE_ATTR: &str = "data-state";
const HOST_STATE_EVENT: &str = "loam-state";

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

// The page's inline script sets this to the wasm-bindgen JS bundle's URL.
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
        if Kind::of(&data) != Some(Kind::Ready) {
            return;
        }
        if let Some(url) = bootstrap_url.as_deref() {
            let _ = web_sys::Url::revoke_object_url(url);
        }

        let msg = Kind::Init.message();
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
            if let Err(e) = worker_for_ready.post_message(&Kind::Start.message()) {
                show_worker_failure(
                    &format!("worker start message failed: {e:?}"),
                    &host_for_ready,
                    &button_for_ready,
                );
            }
        }
        let control = channel_for_ready.borrow_mut().open();
        if let Some(kind) = control {
            if let Err(error) = worker_for_ready.post_message(&kind.message()) {
                tracing::error!("loam_app::wasm: postMessage {kind:?} failed: {error:?}");
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

            let msg = Kind::Start.message();
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
        match Kind::of(&data) {
            Some(Kind::Presented) => set_host_state(&host, "ready"),
            Some(Kind::PreviewReady) => {
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

fn install_worker_failure_handler(worker: &Link, host: &Element, button_id: &str) -> Result<()> {
    let host_for_message = host.clone();
    let button_for_message = button_id.to_string();
    let on_message = Closure::wrap(Box::new(move |event: MessageEvent| {
        let data: JsValue = event.data();
        if Kind::of(&data) != Some(Kind::Error) {
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
