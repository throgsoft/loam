use loam_egui::console::{Console, ConsoleUi, Key};

#[test]
fn printable_toggle_does_not_enter_the_prompt() {
    for (toggle, key, text) in [
        (Key::Backtick, egui::Key::Backtick, "`"),
        (Key::A, egui::Key::A, "a"),
        (Key::Num5, egui::Key::Num5, "5"),
        (Key::Minus, egui::Key::Minus, "-"),
    ] {
        let ctx = egui::Context::default();
        let mut console = Console::<()>::new().with_toggle_key(toggle);
        let _ = ctx.run(
            egui::RawInput {
                events: vec![
                    egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed: true,
                        repeat: false,
                        modifiers: egui::Modifiers::NONE,
                    },
                    egui::Event::Text(text.into()),
                ],
                ..Default::default()
            },
            |ctx| {
                console.ui(ctx);
                assert!(!ctx.input(|i| i
                    .events
                    .iter()
                    .any(|event| matches!(event, egui::Event::Text(value) if value == text))));
            },
        );
        assert!(console.is_open());
        assert!(console.input().is_empty());
    }
}
