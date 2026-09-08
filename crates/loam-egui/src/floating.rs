use egui::{Context, Id, Painter, Pos2, Rect, Stroke, Ui, Window};

pub fn floating_panel<R>(
    ctx: &Context,
    id: &str,
    title: &str,
    open: &mut bool,
    content: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    Window::new(title)
        .id(Id::new(id))
        .open(open)
        .resizable(false)
        .default_width(260.0)
        .show(ctx, content)
        .and_then(|response| response.inner)
}

/// Closes only on click-outside or Esc, not on a click inside.
pub fn sticky_menu<R>(
    ui: &mut Ui,
    label: &str,
    add_contents: impl FnOnce(&mut Ui) -> R,
) -> Option<R> {
    let response = ui.button(label);
    egui::Popup::menu(&response)
        .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
        .show(add_contents)
        .map(|r| r.inner)
}

#[derive(Clone, Debug)]
pub struct CalloutState {
    pub window_pos: Pos2,
    pub open: bool,
}

impl CalloutState {
    pub fn open_at(window_pos: Pos2) -> Self {
        Self {
            window_pos,
            open: true,
        }
    }
}

/// No-op while `state.open` is false.
pub fn callout(
    ctx: &Context,
    id: &str,
    anchor_screen_pos: Pos2,
    state: &mut CalloutState,
    title: &str,
    content: impl FnOnce(&mut Ui),
) {
    if !state.open {
        return;
    }

    const ANCHOR_RADIUS: f32 = 4.0;
    const LEADER_STROKE: f32 = 1.5;
    const PANEL_DEFAULT_WIDTH: f32 = 220.0;
    let leader_color = ctx.style().visuals.window_fill;
    let anchor_outline = ctx.style().visuals.window_stroke.color;

    // Window first so the leader line can attach to its captured frame rect.
    let window_response = Window::new(title)
        .id(Id::new(id))
        .open(&mut state.open)
        .collapsible(true)
        .resizable(false)
        .default_width(PANEL_DEFAULT_WIDTH)
        .current_pos(state.window_pos)
        .show(ctx, content);

    let window_rect: Option<Rect> = window_response.as_ref().map(|r| r.response.rect);
    if let Some(rect) = window_rect {
        state.window_pos = rect.min;
    }

    // Background order: under the window, over the scene.
    let painter_layer =
        egui::LayerId::new(egui::Order::Background, Id::new(id).with("callout-overlay"));
    let painter = Painter::new(ctx.clone(), painter_layer, ctx.content_rect());
    if let Some(rect) = window_rect {
        painter.line_segment(
            [rect.center(), anchor_screen_pos],
            Stroke::new(LEADER_STROKE, leader_color),
        );
    }
    painter.circle_filled(anchor_screen_pos, ANCHOR_RADIUS, leader_color);
    painter.circle_stroke(
        anchor_screen_pos,
        ANCHOR_RADIUS + 1.0,
        Stroke::new(1.0, anchor_outline),
    );
}
