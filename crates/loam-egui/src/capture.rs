/// `pointer` and `keyboard` are read at different points in egui's pass.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UiCapture {
    /// This frame's pointer against the previous build's layout.
    pub pointer: bool,
    /// Focus as of the previous build, minus what this frame's Escape dropped.
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

#[cfg(test)]
mod tests {
    use super::UiCapture;

    const SCREEN: egui::Vec2 = egui::vec2(800.0, 600.0);
    const PANEL_BLANK: egui::Pos2 = egui::pos2(400.0, 8.0);
    const FIELD: egui::Pos2 = egui::pos2(120.0, 10.0);
    const OPEN_SCENE: egui::Pos2 = egui::pos2(200.0, 400.0);
    const WINDOW_POS: egui::Pos2 = egui::pos2(400.0, 300.0);
    const IN_WINDOW: egui::Pos2 = egui::pos2(430.0, 320.0);

    fn build(ctx: &egui::Context, text: &mut String) {
        egui::TopBottomPanel::top("bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let _ = ui.button("File");
                ui.add(egui::TextEdit::singleline(text).id(egui::Id::new("edit")));
            });
        });
        egui::Window::new("win")
            .fixed_pos(WINDOW_POS)
            .fixed_size(egui::vec2(100.0, 80.0))
            .show(ctx, |ui| {
                let _ = ui.button("press");
            });
    }

    struct Host {
        ctx: egui::Context,
        text: String,
    }

    impl Host {
        fn new() -> Self {
            let mut host = Self {
                ctx: egui::Context::default(),
                text: String::new(),
            };
            host.frame(vec![]);
            host.frame(vec![]);
            host
        }

        fn frame(&mut self, events: Vec<egui::Event>) -> UiCapture {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, SCREEN)),
                events,
                ..Default::default()
            };
            let mut capture = UiCapture::default();
            let text = &mut self.text;
            let _ = self.ctx.run(input, |ctx| {
                capture = UiCapture::read(ctx);
                build(ctx, text);
            });
            capture
        }

        fn hover(&mut self, pos: egui::Pos2) -> UiCapture {
            self.frame(vec![egui::Event::PointerMoved(pos)])
        }

        fn press(&mut self, pos: egui::Pos2) -> UiCapture {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
            ])
        }

        fn click(&mut self, pos: egui::Pos2) -> UiCapture {
            self.frame(vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: true,
                    modifiers: egui::Modifiers::default(),
                },
                egui::Event::PointerButton {
                    pos,
                    button: egui::PointerButton::Primary,
                    pressed: false,
                    modifiers: egui::Modifiers::default(),
                },
            ])
        }

        fn key(&mut self, key: egui::Key) -> UiCapture {
            self.frame(vec![egui::Event::Key {
                key,
                physical_key: None,
                pressed: true,
                repeat: false,
                modifiers: egui::Modifiers::default(),
            }])
        }
    }

    #[test]
    fn pointer_capture_reports_a_press_on_the_frame_the_pointer_arrives() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let capture = host.press(IN_WINDOW);
        assert!(capture.pointer);
    }

    #[test]
    fn pointer_capture_covers_a_hovered_panel_without_claiming_the_keyboard() {
        let mut host = Host::new();
        host.hover(OPEN_SCENE);
        let capture = host.hover(PANEL_BLANK);
        assert!(capture.pointer);
        assert!(!capture.keyboard);
    }

    #[test]
    fn pointer_capture_follows_a_drag_off_the_widget_it_started_on() {
        let mut host = Host::new();
        host.hover(FIELD);
        host.press(FIELD);
        let capture = host.hover(OPEN_SCENE);
        assert!(capture.pointer);
    }

    #[test]
    fn capture_is_clear_over_open_scene() {
        let mut host = Host::new();
        host.hover(PANEL_BLANK);
        let capture = host.hover(OPEN_SCENE);
        assert_eq!(capture, UiCapture::default());
    }

    #[test]
    fn a_focused_field_claims_the_keyboard_without_claiming_the_pointer() {
        let mut host = Host::new();
        host.click(egui::pos2(120.0, 10.0));
        let capture = host.hover(OPEN_SCENE);
        assert!(capture.keyboard);
        assert!(!capture.pointer);
    }

    #[test]
    fn keyboard_capture_trails_the_click_that_focuses_a_field_by_one_build() {
        let mut host = Host::new();
        let on_click = host.click(egui::pos2(120.0, 10.0));
        assert!(!on_click.keyboard);
        let next = host.frame(vec![]);
        assert!(next.keyboard);
    }

    #[test]
    fn escape_clears_keyboard_capture_on_the_frame_it_is_pressed() {
        let mut host = Host::new();
        host.click(egui::pos2(120.0, 10.0));
        host.frame(vec![]);
        let capture = host.key(egui::Key::Escape);
        assert!(!capture.keyboard);
    }
}
