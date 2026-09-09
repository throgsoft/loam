//! Requires WebGPU, transferable OffscreenCanvas, and module workers.

pub mod input_queue;
pub mod launch;
pub mod main_launcher;
pub mod messages;

pub use main_launcher::launch_on_click;

use anyhow::{anyhow, Result};
use wasm_bindgen::{JsCast, JsValue};
use web_sys::DedicatedWorkerGlobalScope;

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

// Main ends the loader state on this message.
pub(crate) fn post_failure(scope: &DedicatedWorkerGlobalScope, message: &str) {
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
    if let Err(e) = scope.post_message(&msg) {
        tracing::error!("loam_app::wasm: post error failed: {e:?}");
    }
}

pub(crate) fn install_logging_idempotent() {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        std::panic::set_hook(Box::new(|info| {
            console_error_panic_hook::hook(info);
            if let Ok(scope) = worker_scope() {
                post_failure(&scope, &format!("worker panic: {info}"));
            }
        }));
        tracing_wasm::set_as_global_default();
    });
}
