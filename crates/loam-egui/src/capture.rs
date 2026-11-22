/// `pointer` and `keyboard` are read at different points in egui's pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiCapture {
    pub pointer: bool,
    pub keyboard: bool,
}

impl UiCapture {
    /// Read after `begin_pass` and before building widgets.
    pub fn read(ctx: &egui::Context) -> Self {
        Self {
            pointer: ctx.is_using_pointer()
                || ctx.interaction_snapshot(|i| !i.contains_pointer.is_empty()),
            keyboard: ctx.wants_keyboard_input(),
        }
    }
}
