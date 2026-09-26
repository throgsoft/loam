//! Requires WebGPU. The session runs in a module worker on a transferred OffscreenCanvas, or on
//! the page itself in Firefox, which copies a worker's canvas back on every page repaint.

pub mod input_queue;
pub mod launch;
pub mod main_launcher;
pub mod messages;
mod metrics;

pub(crate) use main_launcher::{launch_page, InPage};

use anyhow::{anyhow, Result};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::{DedicatedWorkerGlobalScope, MessagePort, Window};

pub fn is_worker_context() -> bool {
    js_sys::global()
        .dyn_into::<DedicatedWorkerGlobalScope>()
        .is_ok()
}

pub(crate) fn worker_scope() -> Result<DedicatedWorkerGlobalScope> {
    js_sys::global()
        .dyn_into::<DedicatedWorkerGlobalScope>()
        .map_err(|_| anyhow!("not running in a DedicatedWorkerGlobalScope"))
}

#[derive(Clone)]
pub(crate) enum Endpoint {
    Worker(DedicatedWorkerGlobalScope),
    Page(MessagePort, Window),
}

impl Endpoint {
    pub(crate) fn post(&self, message: &JsValue) -> Result<(), JsValue> {
        match self {
            Self::Worker(scope) => scope.post_message(message),
            Self::Page(port, _) => port.post_message(message),
        }
    }

    pub(crate) fn listen(&self, callback: &js_sys::Function) -> Result<(), JsValue> {
        match self {
            Self::Worker(scope) => scope.add_event_listener_with_callback("message", callback),
            Self::Page(port, _) => {
                port.add_event_listener_with_callback("message", callback)?;
                port.start();
                Ok(())
            }
        }
    }

    pub(crate) fn request_animation_frame(
        &self,
        callback: &js_sys::Function,
    ) -> Result<i32, JsValue> {
        match self {
            Self::Worker(scope) => scope.request_animation_frame(callback),
            Self::Page(_, window) => window.request_animation_frame(callback),
        }
    }
}

// Main ends the loader state on this message.
pub(crate) fn post_failure(endpoint: &Endpoint, message: &str) {
    let msg = js_sys::Object::new();
    let _ = js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("kind"),
        &JsValue::from_str("error"),
    );
    let _ = js_sys::Reflect::set(
        &msg,
        &JsValue::from_str("message"),
        &JsValue::from_str(message),
    );
    if let Err(e) = endpoint.post(&msg) {
        tracing::error!("loam_app::wasm: post error failed: {e:?}");
    }
}

pub(crate) fn install_logging_idempotent() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            console_error_panic_hook::hook(info);
            match worker_scope() {
                Ok(scope) => {
                    post_failure(&Endpoint::Worker(scope), &format!("worker panic: {info}"))
                }
                // A panic traps the page's only instance, so no message listener would report it.
                Err(_) => main_launcher::fail_page(&format!("panic: {info}")),
            }
        }));
        #[cfg(debug_assertions)]
        tracing_wasm::set_as_global_default();
        #[cfg(not(debug_assertions))]
        tracing_wasm::set_as_global_default_with_config(
            tracing_wasm::WASMLayerConfigBuilder::new()
                .set_max_level(tracing::Level::WARN)
                .build(),
        );
    });
}
