pub mod app;
pub mod console;
pub mod debug_layer;
pub mod pacing;

pub mod input;

mod frame;

#[cfg(target_arch = "wasm32")]
mod browser;
#[cfg(not(target_arch = "wasm32"))]
mod native;

use loam_render::device::GpuContext;
use loam_render::work::{BulkBuffers, Readbacks};
use loam_runtime::WorkOrder;

pub use app::{FrameHook, SessionApp};
pub use console::{SessionConsole, Submit};
pub use debug_layer::{DebugLayer, DEBUG_LAYER};
pub use frame::Target;
pub use pacing::{Pace, Pacer};

#[cfg(target_arch = "wasm32")]
pub use browser::{launch, run, run_with_work};
#[cfg(not(target_arch = "wasm32"))]
pub use native::{launch, run, run_with_work};

pub struct WorkContext<'a> {
    pub gpu: &'a GpuContext,
    pub encoder: &'a mut wgpu::CommandEncoder,
    pub order: &'a WorkOrder,
    pub buffers: &'a BulkBuffers,
    pub readbacks: &'a mut Readbacks,
}
