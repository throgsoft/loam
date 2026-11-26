pub mod app;
pub mod args;
pub mod capture;
mod commands;
pub mod console;
pub mod environment;
pub mod frame;
pub mod headless;
pub mod host;
pub mod input;
#[cfg(not(target_arch = "wasm32"))]
pub mod native;
pub mod trace;
#[cfg(target_arch = "wasm32")]
pub mod web;

#[cfg(feature = "egui")]
pub use loam_egui::egui;

pub use app::{CaptureControl, InputHook, PublishedHook, Redraw, SessionApp};
pub use args::Args;
pub use capture::{CaptureFormat, CaptureRequest, CaptureStage, CaptureUnavailable, PaletteMode};
pub use commands::CommandSender;
pub use console::{SessionConsole, Submit};
pub use environment::Environment;
pub use host::{Bindings, HostConfig, HostError, Key};
pub use loam_console::CommandError;
#[cfg(not(target_arch = "wasm32"))]
pub use native::{launch, launch_with};
#[cfg(target_arch = "wasm32")]
pub use web::{launch, launch_with};

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
