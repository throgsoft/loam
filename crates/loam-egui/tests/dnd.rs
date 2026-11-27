use egui::{vec2, DragAndDrop, Id, Rect, Sense};
use loam_egui::{dnd::drag_source_collapsing, egui};

fn screen() -> Rect {
    Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(800.0, 600.0))
}

// egui requires increasing frame times for pointer gestures.
fn warmup_input(time: f64) -> egui::RawInput {
    egui::RawInput {
        screen_rect: Some(screen()),
        time: Some(time),
        ..Default::default()
    }
}

fn pointer_press(time: f64, pos: egui::Pos2) -> egui::RawInput {
    let mut input = warmup_input(time);
    input.events.push(egui::Event::PointerMoved(pos));
    input.events.push(egui::Event::PointerButton {
        pos,
        button: egui::PointerButton::Primary,
        pressed: true,
        modifiers: Default::default(),
    });
    input
}

fn pointer_move(time: f64, pos: egui::Pos2) -> egui::RawInput {
    let mut input = warmup_input(time);
    input.events.push(egui::Event::PointerMoved(pos));
    input
}

#[test]
fn framed_card_drag() {
    let ctx = egui::Context::default();
    let id = Id::new("dnd-test-card");
    let card_pos = egui::pos2(60.0, 30.0);
    let render = |ctx: &egui::Context| {
        egui::CentralPanel::default().show(ctx, |ui| {
            let _ = drag_source_collapsing(ui, id, 42_usize, |ui| {
                egui::Frame::default()
                    .fill(egui::Color32::DARK_GRAY)
                    .inner_margin(egui::Margin::symmetric(4, 6))
                    .show(ui, |ui| {
                        ui.allocate_exact_size(vec2(80.0, 18.0), Sense::hover());
                    });
            });
        });
    };
    let _ = ctx.run(warmup_input(0.0), render);
    let _ = ctx.run(pointer_press(0.05, card_pos), render);
    let _ = ctx.run(pointer_move(0.10, card_pos + vec2(20.0, 0.0)), render);
    let _ = ctx.run(pointer_move(0.15, card_pos + vec2(40.0, 0.0)), render);
    assert!(
        ctx.is_being_dragged(id),
        "drag should be active after press + move past threshold"
    );
    assert_eq!(DragAndDrop::payload::<usize>(&ctx).as_deref(), Some(&42));
}
