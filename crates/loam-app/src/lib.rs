mod sim_config;
pub use sim_config::{CatchUp, SimConfig, DEFAULT_MAX_TICKS_PER_FRAME};

#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

pub mod args;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
pub mod capture {
    pub use crate::capture_types::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};
}
mod capture_types;

pub mod command;
pub mod environment;
pub mod frame_pacing;
pub mod keymap;
#[cfg(not(target_arch = "wasm32"))]
pub mod par_native;
pub mod script;
pub mod session;
pub mod trace;
#[cfg(target_arch = "wasm32")]
pub mod wasm;
#[cfg(not(target_arch = "wasm32"))]
pub mod wasm {
    #[path = "input_queue.rs"]
    pub mod input_queue;
}

pub use loam_egui::egui;

#[derive(Clone)]
pub struct WasmConfig {
    pub host_id: String,
    pub button_id: String,
    pub canvas_id: String,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            host_id: "loam-canvas-host".into(),
            button_id: "loam-launch".into(),
            canvas_id: "loam-canvas".into(),
        }
    }
}

// `crate::trace` subtracts these from `frame` to report `unscoped`.
pub(crate) const FRAME_LOOP_SECTIONS: &[&str] =
    &["dispatch", "simulation", "publication", "presentation"];
