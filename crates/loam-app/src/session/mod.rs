pub mod app;
pub mod camera;
pub mod console;
pub mod debug_layer;
pub mod pacing;

pub mod input;

mod commands;
mod cursor;
mod frame;
mod surface;

#[cfg(target_arch = "wasm32")]
mod animation;
#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(not(target_arch = "wasm32"))]
mod native;

pub use app::{CaptureControl, FrameHook, InputHook, SessionApp};
pub use camera::{look, orbit, FreeCamera, Orbit};
pub use commands::CommandSender;
pub use console::{SessionConsole, Submit};
pub use cursor::CursorPolicy;
pub use debug_layer::{DebugLayer, DEBUG_LAYER};
pub use frame::Target;
pub use pacing::{Pace, Pacer};

#[cfg(target_arch = "wasm32")]
pub use browser::{launch, launch_or_headless, launch_with};
#[cfg(not(target_arch = "wasm32"))]
pub use native::{launch, launch_or_headless, launch_with};
