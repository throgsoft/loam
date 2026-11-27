#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod apng;
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod error;
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod gif;
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod readback;
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod scale;
#[cfg(all(feature = "capture", not(target_arch = "wasm32")))]
mod state;
#[cfg(any(not(feature = "capture"), target_arch = "wasm32"))]
#[path = "state_off.rs"]
mod state;
mod types;

pub(crate) use state::Capture;
pub use types::{CaptureFormat, CaptureRequest, CaptureStage, CaptureUnavailable, PaletteMode};
