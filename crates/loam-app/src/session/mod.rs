pub mod app;
pub mod console;
#[cfg(feature = "egui")]
pub mod debug_layer;
#[cfg(not(feature = "egui"))]
#[path = "debug_layer_off.rs"]
mod debug_layer;
pub mod pacing;

pub mod input;

mod commands;
mod cursor;
pub(crate) mod frame;
mod headless;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod records;
pub(crate) mod surface;
#[cfg(feature = "text")]
mod text;

pub use app::{CaptureControl, InputHook, PublishedHook, Redraw, SessionApp};
pub use commands::CommandSender;
pub use console::{SessionConsole, Submit};
#[cfg(feature = "egui")]
pub use debug_layer::{DebugLayer, DEBUG_LAYER};
pub use frame::Target;
pub use headless::run_script;
pub use pacing::{Pace, Pacer};

#[cfg(not(target_arch = "wasm32"))]
pub use crate::native::{launch, launch_with};
#[cfg(target_arch = "wasm32")]
pub use crate::web::{launch, launch_with};
