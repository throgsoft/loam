use egui::{vec2, Button, CursorIcon, DragValue, RichText, Slider, SliderClamping, Ui};

#[derive(Copy, Clone, Debug, Default)]
pub struct SliderInteraction {
    pub changed: bool,
    pub dragged: bool,
}

/// Right-click on the value cell opens a [`DragValue`] popup.
pub fn slider_with_edit(
    ui: &mut Ui,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    formatted: &str,
    edit_suffix: &str,
    edit_decimals: usize,
    value_cell_w: f32,
) -> SliderInteraction {
    let slider_resp = ui.add(
        Slider::new(value, range.clone())
            .show_value(false)
            .smart_aim(false)
            .clamping(SliderClamping::Always),
    );
    let mut popup_changed = false;
    let label_resp = ui.add_sized(
        vec2(value_cell_w, 14.0),
        Button::new(RichText::new(formatted).monospace())
            .frame(false)
            .small()
            .truncate(),
    );
    label_resp
        .on_hover_cursor(CursorIcon::ContextMenu)
        .on_hover_text("Right-click to edit value")
        .context_menu(|ui| {
            let drag_resp = ui.add(
                DragValue::new(value)
                    .range(range)
                    .suffix(edit_suffix)
                    .fixed_decimals(edit_decimals),
            );
            if drag_resp.changed() {
                popup_changed = true;
            }
        });
    SliderInteraction {
        changed: slider_resp.changed() || popup_changed,
        dragged: slider_resp.dragged(),
    }
}
