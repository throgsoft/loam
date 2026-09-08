use loam_egui::Console;

pub use crate::capture_types::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};

pub fn register_commands<Ctx: 'static>(console: &mut Console<Ctx>, _runtime: &crate::Runtime) {
    console.register(loam_egui::cmd(
        "capture",
        "frame capture (unavailable in this build)",
        |_args, _ctx, out| {
            out.line(
                "Frame capture is unavailable in this build. Run a native desktop \
                 build to enable PNG / GIF / APNG capture.",
            );
            Ok(())
        },
    ));
}

pub fn bind_default_hotkeys<Ctx: 'static>(_console: &mut Console<Ctx>) {}

#[derive(Default)]
pub struct CapturePanel {
    _private: (),
}

impl CapturePanel {
    pub fn new() -> Self {
        Self { _private: () }
    }

    pub fn show(&mut self, _ctx: &loam_egui::egui::Context, _runtime: &crate::Runtime) {}
}
