use winit::event::WindowEvent;
use winit::window::Window;

/// egui's winit state behind loam-egui, so a host feeds window events without depending on egui_winit.
pub struct WinitInput {
    state: egui_winit::State,
}

impl WinitInput {
    pub fn new(ctx: &egui::Context, window: &Window) -> Self {
        Self {
            state: egui_winit::State::new(
                ctx.clone(),
                egui::ViewportId::ROOT,
                window,
                Some(window.scale_factor() as f32),
                window.theme(),
                None,
            ),
        }
    }

    pub fn on_event(&mut self, window: &Window, event: &WindowEvent) -> egui_winit::EventResponse {
        self.state.on_window_event(window, event)
    }

    pub fn take(&mut self, window: &Window) -> egui::RawInput {
        self.state.take_egui_input(window)
    }

    pub fn handle_output(&mut self, window: &Window, output: egui::PlatformOutput) {
        self.state.handle_platform_output(window, output);
    }
}
