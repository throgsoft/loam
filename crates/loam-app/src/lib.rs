#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
pub mod capture;

pub mod args;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
pub mod capture {
    pub use crate::capture_types::{
        CaptureFormat, CaptureRequest, CaptureStage, CaptureUnavailable, PaletteMode,
    };
}
mod capture_types;

pub mod environment;
#[cfg(feature = "egui")]
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

#[cfg(feature = "egui")]
pub use loam_egui::egui;
pub use session::CursorPolicy;

/// `Background` starts at load with no overlay and leaves keys, wheel, and page scrolling to the page.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaunchMode {
    #[default]
    Click,
    Background,
}

/// The page sizes the canvas with CSS; otherwise the launcher pins it at its launch size.
#[derive(Clone, Debug)]
pub struct WasmConfig {
    pub host_id: String,
    pub button_id: String,
    pub canvas_id: String,
    pub mode: LaunchMode,
    /// Caps the drawing buffer's width times height; `None` or `Some(0)` is no cap.
    pub max_pixels: Option<u32>,
}

impl Default for WasmConfig {
    fn default() -> Self {
        Self {
            host_id: "loam-canvas-host".into(),
            button_id: "loam-launch".into(),
            canvas_id: "loam-canvas".into(),
            mode: LaunchMode::Click,
            max_pixels: None,
        }
    }
}

pub(crate) const FRAME_LOOP_SECTIONS: &[&str] =
    &["dispatch", "simulation", "publication", "presentation"];
