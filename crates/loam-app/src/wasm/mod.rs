//! Requires WebGPU, transferable OffscreenCanvas, and module workers.

pub mod host_action;
pub mod input_queue;
pub mod launch;
pub mod main_launcher;
pub mod messages;
pub mod modifier_sync;
pub mod worker;
pub mod worker_ui;

pub use main_launcher::launch_on_click;

pub use launch::{is_manual_mode, js_heap_sampler};

pub fn is_worker_context() -> bool {
    use wasm_bindgen::JsCast;
    js_sys::global()
        .dyn_into::<web_sys::DedicatedWorkerGlobalScope>()
        .is_ok()
}
