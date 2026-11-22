use loam_egui::console::{Console, ConsoleUi};

#[test]
fn printable_toggle_does_not_enter_the_prompt() {
    let text = "`";
    let ctx = egui::Context::default();
    let mut console = Console::<()>::new();
    let _ = ctx.run(
        egui::RawInput {
            events: vec![
                egui::Event::Key {
                    key: egui::Key::Backtick,
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
