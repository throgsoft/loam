use egui::{
    self, emath::TSTransform, vec2, DragAndDrop, Id, LayerId, Order, Rect, Response, Sense, Ui,
    UiBuilder,
};

pub fn drag_source_collapsing<P>(
    ui: &mut Ui,
    id: Id,
    payload: P,
    body: impl FnOnce(&mut Ui),
) -> Response
where
    P: 'static + Send + Sync,
{
    let ctx = ui.ctx().clone();
    let is_dragged = ctx.is_being_dragged(id);
    if !is_dragged {
        return ui.dnd_drag_source(id, payload, body).response;
    }
    DragAndDrop::set_payload(&ctx, payload);
    let layer_id = LayerId::new(Order::Tooltip, id);
    let mut child = ui.new_child(UiBuilder::new().layer_id(layer_id));
    body(&mut child);
    let body_rect = child.min_rect();
    if let Some(pos) = ctx.pointer_interact_pos() {
        let delta = pos - body_rect.center();
        ctx.transform_layer_shapes(layer_id, TSTransform::from_translation(delta));
    }
    ui.interact(body_rect, id, Sense::hover())
}

/// `true` on a pointer release over the targeted gap.
pub fn make_room_gap(
    ui: &mut Ui,
    is_target: bool,
    slot_id: Id,
    height: f32,
    open_width: f32,
) -> bool {
    let target_w = if is_target { open_width } else { 0.0 };
    let smooth_w = ui.ctx().animate_value_with_time(slot_id, target_w, 0.12);
    if smooth_w >= 0.5 {
        let _ = ui.allocate_exact_size(vec2(smooth_w, height), Sense::hover());
    }
    let dropped = is_target && ui.ctx().input(|i| i.pointer.any_released());
    if dropped {
        let _ = ui.ctx().animate_value_with_time(slot_id, 0.0, 0.0);
    }
    dropped
}

/// `None` when not dragging or more than 40 pt off the row.
pub fn drop_target_idx(
    ctx: &egui::Context,
    is_dragging: bool,
    row_rect: Rect,
    item_count: usize,
) -> Option<usize> {
    if !is_dragging {
        return None;
    }
    let cursor = ctx.input(|i| i.pointer.hover_pos())?;
    let band = row_rect.expand2(vec2(0.0, 40.0));
    if !band.x_range().contains(cursor.x) || !band.y_range().contains(cursor.y) {
        return None;
    }
    let n_slots = item_count + 1;
    let slot_w = (row_rect.width() / n_slots as f32).max(1.0);
    let rel = (cursor.x - row_rect.left()).max(0.0);
    Some(((rel / slot_w) as usize).min(item_count))
}

/// The Tooltip layer never registers hover, so the dragged body dims without this.
pub fn force_opaque_active(ui: &mut Ui) {
    let active = ui.visuals().widgets.active;
    let v = ui.visuals_mut();
    v.widgets.inactive.bg_fill = active.bg_fill;
    v.widgets.inactive.weak_bg_fill = active.weak_bg_fill;
    v.widgets.inactive.fg_stroke = active.fg_stroke;
    v.widgets.inactive.bg_stroke = active.bg_stroke;
    v.widgets.noninteractive.bg_fill = active.bg_fill;
    v.widgets.noninteractive.weak_bg_fill = active.weak_bg_fill;
}

/// In `[0.0, 1.0]`, animated over 120 ms at drag start and drag end.
pub fn pickup_t(ctx: &egui::Context, drag_id: Id) -> f32 {
    let target = if ctx.is_being_dragged(drag_id) {
        1.0
    } else {
        0.0
    };
    ctx.animate_value_with_time(drag_id.with("pickup"), target, 0.12)
}
