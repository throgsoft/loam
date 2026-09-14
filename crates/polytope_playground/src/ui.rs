use loam::app::egui;
use loam::app::session::CommandSender;
use loam::math::{Rotor, Rotor4};
use loam::runtime::{Entity, Session};
use loam_egui::{
    dnd::{drag_source_collapsing, drop_target_idx, force_opaque_active, make_room_gap, pickup_t},
    media::{add_button, chevron_button, play_pause_button, rate_toggle, refresh_button},
    slider_with_edit,
};

use crate::color::ColorMode;
use crate::consts::{BASE_ROTATION_RATE, W_RANGE};
use crate::display::Surface;
use crate::mode::Mode;
use crate::projection::Family;
use crate::strip::{Cell, Strip, MAX_CELLS, MAX_T_EXTENT, MIN_CELLS, MIN_T_EXTENT};
use crate::toy::DepthBand;
use crate::{catalog, Action, Playground};

const PLANE_LABELS: [&str; 6] = ["xy", "xz", "xw", "yz", "yw", "zw"];
const OVERLAY_PAD: f32 = 16.0;
const CONTROL_H: f32 = 28.0;
const CONTROL_W: f32 = 28.0;
const PLAY_W: f32 = 36.0;
const CARD_WIDTH: f32 = 74.0;
const CARD_HEIGHT: f32 = CONTROL_H + 2.0;

#[derive(Default)]
pub(crate) struct Panel {
    expanded: bool,
    show_render: bool,
    show_about: bool,
    pub(crate) show_callouts: bool,
}

pub(crate) fn draw(
    context: &egui::Context,
    session: &Session<Playground>,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
    rotor: Rotor4,
    depths: &[DepthBand],
) {
    let app = &session.app;
    menu_bar(context, app, panel, sender);
    rotor_formula(context, app, rotor);
    render_settings(context, app, panel, sender);
    about(context, panel);
    if *app.controls.get() {
        overlay(context, app, panel, sender, depths);
    }
}

fn menu_bar(
    context: &egui::Context,
    app: &Playground,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
) {
    egui::TopBottomPanel::top("polytope-playground-menu").show(context, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("Demo", |ui| {
                for (mode, label) in [(Mode::Rotate, "Rotate"), (Mode::Toybox, "Toybox")] {
                    if ui
                        .selectable_label(*app.mode.get() == mode, label)
                        .clicked()
                    {
                        sender.app(Action::Mode(mode));
                        ui.close_kind(egui::UiKind::Menu);
                    }
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui.button("Reset poses, rotation, and slice").clicked() {
                    sender.app(Action::Reset);
                    ui.close_kind(egui::UiKind::Menu);
                }
            });
            loam_egui::sticky_menu(ui, "View", |ui| {
                let mut controls = *app.controls.get();
                if ui.checkbox(&mut controls, "Controls").changed() {
                    sender.app(Action::Controls(None));
                }
                let mut formula = *app.formula.get();
                if ui.checkbox(&mut formula, "Rotation formula").changed() {
                    sender.app(Action::Formula(None));
                }
                ui.checkbox(&mut panel.show_render, "Render settings");
                ui.checkbox(&mut panel.show_callouts, "Shape labels");
                ui.separator();
                if ui.button("About this program").clicked() {
                    panel.show_about = true;
                    egui::Popup::close_all(ui.ctx());
                }
            });
        });
    });
}

fn overlay(
    context: &egui::Context,
    app: &Playground,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
    depths: &[DepthBand],
) {
    let screen = context.content_rect();
    let width = (screen.width() - 2.0 * (OVERLAY_PAD + 10.0)).clamp(280.0, 760.0);
    let visuals = &context.style().visuals;
    let frame = egui::Frame::default()
        .fill(visuals.window_fill)
        .stroke(visuals.window_stroke)
        .corner_radius(visuals.window_corner_radius)
        .inner_margin(10.0);
    egui::Window::new("polytope-playground-controls")
        .id(egui::Id::new("polytope-playground-controls"))
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .movable(true)
        .auto_sized()
        .pivot(egui::Align2::CENTER_BOTTOM)
        .default_pos(egui::pos2(screen.center().x, screen.bottom() - OVERLAY_PAD))
        .default_width(width)
        .frame(frame)
        .show(context, |ui| {
            ui.set_width(width);
            if *app.mode.get() == Mode::Toybox {
                toybox_depth(ui, app, panel, sender, depths);
            } else {
                if panel.expanded {
                    egui::ScrollArea::vertical()
                        .max_height((screen.height() - 190.0).max(80.0))
                        .show(ui, |ui| expanded(ui, app, sender));
                    ui.separator();
                }
                sliders(ui, app, sender);
                transport(ui, app, panel, sender);
            }
        });
}

fn expanded(ui: &mut egui::Ui, app: &Playground, sender: &CommandSender<Playground>) {
    view_tabs(ui, app, sender);
    if app.strip.get().on {
        filmstrip(ui, *app.strip.get(), sender);
    } else {
        shapes(ui, app, sender);
    }
    ui.separator();
    active_set(ui, app, sender);
}

fn view_tabs(ui: &mut egui::Ui, app: &Playground, sender: &CommandSender<Playground>) {
    let display = *app.display.get();
    let strip = *app.strip.get();
    ui.horizontal_wrapped(|ui| {
        if ui
            .selectable_label(!display.single && !strip.on, "Shapes")
            .clicked()
        {
            let mut next = display;
            next.single = false;
            sender.app(Action::Display(next));
            if strip.on {
                sender.app(Action::Strip(Strip { on: false, ..strip }));
            }
        }
        if ui
            .selectable_label(display.single && !strip.on, "Single")
            .clicked()
        {
            let mut next = display;
            next.single = true;
            sender.app(Action::Display(next));
            if strip.on {
                sender.app(Action::Strip(Strip { on: false, ..strip }));
            }
        }
        if ui.selectable_label(strip.on, "Filmstrip").clicked() {
            sender.app(Action::Strip(Strip { on: true, ..strip }));
        }
        let mut formula = *app.formula.get();
        if ui.checkbox(&mut formula, "Formula").changed() {
            sender.app(Action::Formula(None));
        }
    });
}

fn active_set(ui: &mut egui::Ui, app: &Playground, sender: &CommandSender<Playground>) {
    const PLANES: [usize; 6] = [0, 1, 3, 2, 4, 5];
    const CELL_SPACING: f32 = 4.0;
    const CHECKBOX_W: f32 = 18.0;
    const LABEL_W: f32 = 22.0;
    const VALUE_W: f32 = 56.0;
    const ROW_GAP: f32 = 6.0;

    let time = *app.time.get();
    let spin = *app.spin.get();
    let total = ui.available_width();
    let cell_width = ((total - 2.0 * ROW_GAP) / 3.0).floor();
    let slider_width = (cell_width - CHECKBOX_W - LABEL_W - VALUE_W - 3.0 * CELL_SPACING).max(40.0);
    for row in PLANES.chunks(3) {
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = ROW_GAP;
            for &index in row {
                ui.allocate_ui_with_layout(
                    egui::vec2(cell_width, CONTROL_H),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        ui.spacing_mut().item_spacing.x = CELL_SPACING;
                        ui.spacing_mut().slider_width = slider_width;
                        let mut active = spin.planes[index];
                        if ui
                            .add_sized([CHECKBOX_W, 18.0], egui::Checkbox::new(&mut active, ""))
                            .changed()
                        {
                            sender.app(Action::Plane(index));
                        }
                        ui.add_sized(
                            [LABEL_W, 18.0],
                            egui::Label::new(egui::RichText::new(PLANE_LABELS[index]).monospace()),
                        );
                        let angle = app.angles.get()[index]
                            + if spin.planes[index] {
                                time * BASE_ROTATION_RATE
                            } else {
                                0.0
                            };
                        let mut degrees = wrap_degrees(angle.to_degrees());
                        let slider = egui::Slider::new(&mut degrees, -720.0..=720.0)
                            .show_value(false)
                            .smart_aim(false)
                            .clamping(egui::SliderClamping::Always);
                        if ui.add_sized([slider_width, 18.0], slider).changed() {
                            sender.app(Action::PlaneAngle(index, degrees.to_radians()));
                        }
                        ui.add_sized(
                            [VALUE_W, 18.0],
                            egui::Label::new(
                                egui::RichText::new(format!("{degrees:>+6.1}°")).monospace(),
                            ),
                        );
                    },
                );
            }
        });
    }
}

fn wrap_degrees(degrees: f32) -> f32 {
    let wrapped = degrees.rem_euclid(1440.0);
    if wrapped > 720.0 {
        wrapped - 1440.0
    } else {
        wrapped
    }
}

fn shapes(ui: &mut egui::Ui, app: &Playground, sender: &CommandSender<Playground>) {
    let active = *app.active.get();
    let row_len = app.slots.len();
    let row_rect_id = ui.make_persistent_id("shape-row-rect");
    let last_row_rect: Option<egui::Rect> =
        ui.ctx().memory(|memory| memory.data.get_temp(row_rect_id));
    let dragging = egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
    let drop_index =
        last_row_rect.and_then(|rect| drop_target_idx(ui.ctx(), dragging, rect, row_len));
    let released = ui.ctx().input(|input| input.pointer.any_released());
    let drop_id = ui.make_persistent_id("shape-row-drop");
    // Keep the preview until the reorder command commits.
    let mut dropped = ui.ctx().data_mut(|data| {
        let (entity, from, destination) = data.get_temp::<(Entity, usize, usize)>(drop_id)?;
        if app.slots.get(entity).is_some_and(|slot| slot.index == from) {
            Some((from, destination))
        } else {
            data.remove::<(Entity, usize, usize)>(drop_id);
            None
        }
    });
    if released {
        if let (Some(to), Some(from)) = (
            drop_index,
            egui::DragAndDrop::payload::<usize>(ui.ctx()).map(|payload| *payload),
        ) {
            let _ = egui::DragAndDrop::take_payload::<usize>(ui.ctx());
            let destination = if to > from { to - 1 } else { to };
            if let Some((entity, _)) = app
                .slots
                .iter()
                .find(|(_, slot)| slot.index == from && to <= row_len && destination != from)
            {
                sender.app(Action::ReorderShape { from, to });
                dropped = Some((from, destination));
                ui.ctx()
                    .data_mut(|data| data.insert_temp(drop_id, (entity, from, destination)));
            }
        }
    }
    ui.separator();
    let row_rect = egui::ScrollArea::horizontal()
        .auto_shrink([false, true])
        .id_salt("polytope-playground-shapes")
        .show(ui, |ui| {
            ui.with_layout(egui::Layout::left_to_right(egui::Align::Min), |ui| {
                ui.spacing_mut().item_spacing.x = 4.0;
                ui.set_min_height(CARD_HEIGHT);
                if released {
                    for index in 0..=row_len {
                        let gap = ui.make_persistent_id(("shape-gap", index));
                        let card = ui.make_persistent_id(("shape-card", index));
                        let _ = ui.ctx().animate_value_with_time(gap, 0.0, 0.0);
                        let _ = ui
                            .ctx()
                            .animate_value_with_time(card.with("pickup"), 0.0, 0.0);
                    }
                }
                let still_dragging = egui::DragAndDrop::payload::<usize>(ui.ctx()).is_some();
                let render_drop_index = if still_dragging { drop_index } else { None };
                for index in 0..row_len {
                    let gap_id = ui.make_persistent_id(("shape-gap", index));
                    let _ = make_room_gap(
                        ui,
                        render_drop_index == Some(index),
                        gap_id,
                        CARD_HEIGHT,
                        CARD_WIDTH,
                    );
                    let source = dropped.map_or(index, |(from, destination)| {
                        crate::row::reordered_index(index, destination, from)
                    });
                    let Some(slot) = app
                        .slots
                        .iter()
                        .find(|(_, slot)| slot.index == source)
                        .map(|(_, slot)| *slot)
                    else {
                        continue;
                    };
                    let card_id = ui.make_persistent_id(("shape-card", index));
                    let raised = pickup_t(ui.ctx(), card_id);
                    let selected = slot.index == active;
                    let visuals = ui.visuals();
                    let fill = if selected {
                        visuals.selection.bg_fill
                    } else {
                        visuals.widgets.noninteractive.bg_fill
                    };
                    let stroke = if raised > 0.0 {
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 200, 60))
                    } else if selected {
                        visuals.selection.stroke
                    } else {
                        visuals.widgets.noninteractive.bg_stroke
                    };
                    let response = drag_source_collapsing(ui, card_id, slot.index, |ui| {
                        if ui.ctx().is_being_dragged(card_id) {
                            force_opaque_active(ui);
                        }
                        egui::Frame::default()
                            .fill(fill)
                            .stroke(stroke)
                            .inner_margin(egui::Margin::symmetric(4, 6))
                            .corner_radius(3)
                            .show(ui, |ui| {
                                ui.add_sized(
                                    [64.0, CONTROL_H - 12.0],
                                    egui::Label::new(
                                        egui::RichText::new(slot.entry.label).strong(),
                                    )
                                    .selectable(false),
                                );
                            });
                    })
                    .on_hover_cursor(egui::CursorIcon::Grab)
                    .on_hover_ui(|ui| {
                        ui.label(slot.entry.long_name);
                        ui.label("Drag to reorder. Right-click to replace or remove.");
                    })
                    .interact(egui::Sense::click());
                    if response.clicked() {
                        sender.app(Action::Active(slot.index));
                    }
                    response.context_menu(|ui| {
                        ui.set_min_width(160.0);
                        ui.label("Replace with");
                        catalog::shape_catalog_menu(ui, |card| {
                            sender.app(Action::Shape(slot.index, card));
                        });
                        ui.separator();
                        if ui.button("Remove from row").clicked() {
                            sender.app(Action::RemoveShape(slot.index));
                            ui.close_kind(egui::UiKind::Menu);
                        }
                    });
                }
                let trailing_id = ui.make_persistent_id(("shape-gap", row_len));
                let _ = make_room_gap(
                    ui,
                    render_drop_index == Some(row_len),
                    trailing_id,
                    CARD_HEIGHT,
                    CARD_WIDTH,
                );
                let add = add_button(ui, egui::vec2(CONTROL_W, CONTROL_H))
                    .on_hover_text("Add a shape to the row.");
                egui::Popup::menu(&add).show(|ui| {
                    ui.set_min_width(160.0);
                    catalog::shape_catalog_menu(ui, |card| {
                        sender.app(Action::AddShape(card));
                    });
                });
            })
            .response
            .rect
        })
        .inner;
    if egui::DragAndDrop::payload::<usize>(ui.ctx()).is_none() {
        ui.ctx()
            .memory_mut(|memory| memory.data.insert_temp(row_rect_id, row_rect));
    }
}

fn toybox_depth(
    ui: &mut egui::Ui,
    app: &Playground,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
    depths: &[DepthBand],
) {
    let slice = *app.slice.get();
    let visible = depths
        .iter()
        .filter(|band| slice >= band.min && slice <= band.max)
        .count();
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new("w depth").strong());
        ui.monospace(format!("{slice:+.2}"));
        ui.label(format!("{visible}/{} in slice", depths.len()));
    });
    if let Some(next) = depth_ruler(ui, slice, depths) {
        sender.app(Action::Slice(next));
    }
    if panel.expanded {
        ui.label("Primary-drag a shape to carry it. Release it to throw.");
        ui.label("Click or drag the depth ruler to move the w slice.");
    }
    transport(ui, app, panel, sender);
}

fn depth_ruler(ui: &mut egui::Ui, slice: f32, depths: &[DepthBand]) -> Option<f32> {
    const LABEL_WIDTH: f32 = 76.0;
    const ROW_HEIGHT: f32 = 18.0;
    let height = ROW_HEIGHT * depths.len().max(1) as f32;
    let (rect, response) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), height),
        egui::Sense::click_and_drag(),
    );
    let chart = egui::Rect::from_min_max(
        egui::pos2((rect.left() + LABEL_WIDTH).min(rect.right()), rect.top()),
        rect.max,
    );
    let reach = depths.iter().fold(W_RANGE, |held, band| {
        held.max(band.min.abs()).max(band.max.abs())
    });
    let x_for = |w: f32| {
        let position = (w / reach).clamp(-1.0, 1.0) * 0.5 + 0.5;
        chart.left() + position * chart.width()
    };
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, egui::Color32::from_rgb(20, 22, 28));
    for (index, band) in depths.iter().enumerate() {
        let top = rect.top() + index as f32 * ROW_HEIGHT;
        let row = egui::Rect::from_min_max(
            egui::pos2(rect.left(), top),
            egui::pos2(rect.right(), top + ROW_HEIGHT),
        );
        let dim = if band.asleep { 0.5 } else { 1.0 };
        let channel = |value: f32| (value * dim * 255.0).clamp(0.0, 255.0) as u8;
        let color = egui::Color32::from_rgb(
            channel(band.color[0]),
            channel(band.color[1]),
            channel(band.color[2]),
        );
        painter.text(
            egui::pos2(rect.left() + 5.0, row.center().y),
            egui::Align2::LEFT_CENTER,
            band.label,
            egui::FontId::monospace(11.0),
            color,
        );
        let band_rect = egui::Rect::from_min_max(
            egui::pos2(x_for(band.min), row.top() + 3.0),
            egui::pos2(x_for(band.max), row.bottom() - 3.0),
        );
        painter.rect_filled(band_rect, 2.0, color.gamma_multiply(0.55));
        painter.line_segment(
            [
                egui::pos2(x_for(band.center), row.top() + 2.0),
                egui::pos2(x_for(band.center), row.bottom() - 2.0),
            ],
            egui::Stroke::new(1.5, color),
        );
        if slice >= band.min && slice <= band.max {
            painter.rect_stroke(
                band_rect,
                2.0,
                egui::Stroke::new(1.0, egui::Color32::WHITE),
                egui::StrokeKind::Inside,
            );
        }
    }
    let slice_x = x_for(slice);
    painter.line_segment(
        [
            egui::pos2(slice_x, rect.top()),
            egui::pos2(slice_x, rect.bottom()),
        ],
        egui::Stroke::new(2.0, egui::Color32::WHITE),
    );
    let pointer = response.interact_pointer_pos()?;
    if !(response.clicked() || response.dragged()) || !chart.contains(pointer) {
        return None;
    }
    let position =
        ((pointer.x - chart.left()) / chart.width().max(f32::MIN_POSITIVE)).clamp(0.0, 1.0);
    Some((position * 2.0 - 1.0) * reach)
}

fn sliders(ui: &mut egui::Ui, app: &Playground, sender: &CommandSender<Playground>) {
    const VALUE_WIDTH: f32 = 72.0;
    let width = ui.available_width();
    let spacing = ui.spacing().item_spacing.x;
    ui.spacing_mut().slider_width = (width - VALUE_WIDTH - spacing).max(140.0);
    let row = egui::vec2(width, CONTROL_H);
    let layout = egui::Layout::left_to_right(egui::Align::Center);
    let mut slice = *app.slice.get();
    ui.allocate_ui_with_layout(row, layout, |ui| {
        let label = format!("w {:>+.3}", slice);
        let change = slider_with_edit(
            ui,
            &mut slice,
            -W_RANGE..=W_RANGE,
            &label,
            "",
            3,
            VALUE_WIDTH,
        );
        if change.changed {
            sender.app(Action::Slice(slice));
        }
    });
    if *app.mode.get() == Mode::Toybox {
        return;
    }
    let mut time = *app.time.get();
    let limit = MAX_T_EXTENT.max(time);
    ui.allocate_ui_with_layout(row, layout, |ui| {
        let label = format!("t {:>5.2}s", time);
        let change = slider_with_edit(ui, &mut time, 0.0..=limit, &label, "s", 2, VALUE_WIDTH);
        if change.changed {
            sender.app(Action::Time(time));
        }
    });
}

fn transport(
    ui: &mut egui::Ui,
    app: &Playground,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
) {
    let mut rate = app.spin.get().rate;
    let toybox = *app.mode.get() == Mode::Toybox;
    ui.horizontal(|ui| {
        let group_width = if toybox { CONTROL_W } else { 215.0 };
        let leading = ((ui.available_width() - group_width) / 2.0).max(8.0);
        ui.add_space(leading);
        let control = egui::vec2(CONTROL_W, CONTROL_H);
        if !toybox {
            rate_toggle(ui, control, &mut rate, 0.25, true, false);
            rate_toggle(ui, control, &mut rate, 0.5, false, false);
            if play_pause_button(ui, egui::vec2(PLAY_W, CONTROL_H), app.spin.get().running)
                .on_hover_text("Play or pause rotation.")
                .clicked()
            {
                sender.app(Action::Running(Some(!app.spin.get().running)));
            }
            rate_toggle(ui, control, &mut rate, 2.0, false, true);
            rate_toggle(ui, control, &mut rate, 4.0, true, true);
        }
        if refresh_button(ui, control)
            .on_hover_text("Reset poses, rotation, and slice.")
            .clicked()
        {
            sender.app(Action::Reset);
        }
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if chevron_button(
                ui,
                control,
                !panel.expanded,
                if panel.expanded {
                    "Collapse controls."
                } else {
                    "Expand controls."
                },
            )
            .clicked()
            {
                panel.expanded = !panel.expanded;
            }
            let utility = egui::vec2(CONTROL_W, CONTROL_H);
            if ui
                .add(egui::Button::new(egui::RichText::new("⚙").strong()).min_size(utility))
                .on_hover_text("Render settings.")
                .clicked()
            {
                panel.show_render = !panel.show_render;
            }
            if ui
                .add(egui::Button::new(egui::RichText::new("?").strong()).min_size(utility))
                .on_hover_text("Help.")
                .clicked()
            {
                panel.show_about = true;
            }
        });
    });
    if !toybox && rate != app.spin.get().rate {
        sender.app(Action::Rate(rate));
    }
}

fn render_settings(
    context: &egui::Context,
    app: &Playground,
    panel: &mut Panel,
    sender: &CommandSender<Playground>,
) {
    let before = *app.display.get();
    let mut display = before;
    loam_egui::floating_panel(
        context,
        "polytope-playground-render",
        "Render",
        &mut panel.show_render,
        |ui| {
            ui.label(egui::RichText::new("Surface").strong());
            ui.radio_value(&mut display.surface, Surface::Raster, "Raster");
            ui.radio_value(&mut display.surface, Surface::Sdf, "SDF");
            ui.radio_value(&mut display.surface, Surface::Off, "Off");
            ui.separator();
            ui.checkbox(&mut display.wireframe, "Wireframe");
            ui.checkbox(&mut display.section_perimeter, "Section perimeter");
            ui.add_enabled_ui(display.wireframe, |ui| {
                ui.add(
                    egui::Slider::new(&mut display.wireframe_width_px, 0.5..=6.0)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Line width")
                        .suffix(" px"),
                );
                ui.add(
                    egui::Slider::new(&mut display.wireframe_opacity, 0.05..=1.0)
                        .clamping(egui::SliderClamping::Edits)
                        .text("Line opacity")
                        .fixed_decimals(2),
                );
            });
            ui.separator();
            ui.label(egui::RichText::new("Color").strong());
            for mode in ColorMode::ALL {
                if ui.radio(*app.color.get() == mode, mode.name()).clicked() {
                    sender.app(Action::Color(mode));
                }
            }
            ui.separator();
            ui.label(egui::RichText::new("Projection").strong());
            for family in Family::ALL {
                if ui
                    .radio(*app.projection.get() == family, family.name())
                    .clicked()
                {
                    sender.app(Action::Projection(family));
                }
            }
        },
    );
    if display != before {
        sender.app(Action::Display(display));
    }
}

fn about(context: &egui::Context, panel: &mut Panel) {
    egui::Window::new("About Polytope Playground")
        .id(egui::Id::new("polytope-playground-about"))
        .open(&mut panel.show_about)
        .resizable(true)
        .collapsible(false)
        .default_width(420.0)
        .show(context, |ui| {
            ui.heading("Polytope Playground");
            ui.label("The w control moves a 3D slice through each 4D shape.");
            ui.label("Rotation in xy, xz, or yz turns the shape within visible space. Rotation in xw, yw, or zw carries it through the slice.");
            ui.label("Shadow projection drops w. Perspective uses a 4D focal point. Stereographic projection maps directions on S³ into 3D.");
            ui.separator();
            ui.label("In Toybox, primary-drag a shape to carry it, then release it to throw. Use the depth ruler to move the w slice.");
            ui.label("Right-drag to orbit the camera. Scroll to zoom.");
            ui.label("In Rotate, Space plays or pauses rotation. Q and E move the w slice in orbit mode.");
        });
}

fn rotor_formula(context: &egui::Context, app: &Playground, rotor: Rotor4) {
    if !*app.formula.get() {
        return;
    }
    let available = context.available_rect();
    egui::Window::new("Rotation formula")
        .id(egui::Id::new("polytope-playground-formula"))
        .title_bar(false)
        .resizable(false)
        .collapsible(false)
        .default_pos(egui::pos2(
            available.right() - 280.0,
            available.top() + 16.0,
        ))
        .default_width(320.0)
        .show(context, |ui| {
            ui.monospace("R = exp(B)");
            ui.label(egui::RichText::new("B = log(R), in degrees").small().weak());
            loam_egui::bivector_matrix(ui, &rotor.log());
        });
}

const CALLOUT_INSET_PT: f32 = 10.0;

pub(crate) fn callouts(
    context: &egui::Context,
    session: &Session<Playground>,
    anchors: &[(usize, &'static str, glam::Vec3)],
) {
    let screen = context.viewport_rect();
    let half = screen.size() * 0.5;
    for (index, label, point) in anchors {
        let Some([x, y]) = session.views().ndc(point.to_array()) else {
            continue;
        };
        let at = egui::pos2(
            screen.center().x + x * half.x,
            screen.center().y - y * half.y - CALLOUT_INSET_PT,
        );
        if !screen.contains(at) {
            continue;
        }
        egui::Area::new(egui::Id::new(("callout", index)))
            .fixed_pos(at)
            .pivot(egui::Align2::CENTER_BOTTOM)
            .order(egui::Order::Foreground)
            .show(context, |ui| {
                egui::Frame::default()
                    .fill(egui::Color32::from_black_alpha(160))
                    .inner_margin(egui::Margin::symmetric(6, 2))
                    .corner_radius(3)
                    .show(ui, |ui| {
                        ui.monospace(*label);
                    });
            });
    }
}

fn filmstrip(ui: &mut egui::Ui, strip: Strip, sender: &CommandSender<Playground>) {
    let mut held = strip;
    ui.horizontal_wrapped(|ui| {
        ui.label("filmstrip");
        if ui.selectable_label(strip.w, "w cells").clicked() {
            held.w = !held.w;
        }
        if ui.selectable_label(strip.t, "t cells").clicked() {
            held.t = !held.t;
        }
        if strip.w && strip.t && ui.selectable_label(strip.swap_axes, "swap").clicked() {
            held.swap_axes = !held.swap_axes;
        }
    });
    ui.horizontal_wrapped(|ui| {
        if strip.w {
            ui.add(
                egui::DragValue::new(&mut held.count_w)
                    .range(MIN_CELLS..=MAX_CELLS)
                    .speed(0.2)
                    .prefix("w: "),
            );
        }
        if strip.t {
            ui.add(
                egui::DragValue::new(&mut held.count_t)
                    .range(MIN_CELLS..=MAX_CELLS)
                    .speed(0.2)
                    .prefix("t: "),
            );
            ui.add(
                egui::DragValue::new(&mut held.t_extent)
                    .range(MIN_T_EXTENT..=MAX_T_EXTENT)
                    .speed(0.02)
                    .fixed_decimals(2)
                    .suffix("s")
                    .prefix("dt: "),
            );
        }
        let subject = ui.button(format!(
            "subject: {}",
            catalog::SHAPE_CATALOG[strip.subject()].label
        ));
        egui::Popup::menu(&subject).show(|ui| {
            ui.set_min_width(140.0);
            for (card, entry) in catalog::SHAPE_CATALOG.iter().enumerate() {
                if ui.button(entry.label).clicked() {
                    held.subject = card;
                    ui.close();
                }
            }
        });
    });
    if held != strip {
        sender.app(Action::Strip(held));
    }
}

pub(crate) fn strip_labels(context: &egui::Context, session: &Session<Playground>, cells: &[Cell]) {
    let strip = *session.app.strip.get();
    if !strip.on {
        return;
    }
    let per_point = context.pixels_per_point().max(1e-3);
    let label_top = context.available_rect().top() + 8.0;
    for (index, cell) in cells.iter().enumerate() {
        let at = egui::pos2(
            (cell.viewport.x as f32 + cell.viewport.width as f32 * 0.5) / per_point,
            ((cell.viewport.y as f32 + 8.0) / per_point).max(label_top),
        );
        let text = if strip.t {
            format!("w={:+.3}  t={:.2}s", cell.w, cell.t)
        } else {
            format!("w={:+.3}", cell.w)
        };
        egui::Area::new(egui::Id::new(("strip-cell", index)))
            .fixed_pos(at)
            .pivot(egui::Align2::CENTER_TOP)
            .order(egui::Order::Foreground)
            .show(context, |ui| {
                egui::Frame::default()
                    .fill(egui::Color32::from_black_alpha(160))
                    .inner_margin(egui::Margin::symmetric(6, 2))
                    .corner_radius(3)
                    .show(ui, |ui| {
                        ui.monospace(text);
                    });
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text_position(output: &egui::FullOutput, label: &str) -> egui::Pos2 {
        output
            .shapes
            .iter()
            .find_map(|shape| match &shape.shape {
                egui::Shape::Text(text) if text.galley.text() == label => Some(text.pos),
                _ => None,
            })
            .expect("visible label")
    }

    #[test]
    fn dropping_a_shape_keeps_neighbor_positions_and_row_height() {
        for multipass in [false, true] {
            let context = egui::Context::default();
            let mut app = loam::app::session::SessionApp::<Playground>::new(
                loam::runtime::HostConfig::new("test", crate::bindings()),
            );
            let sender = app.sender();
            let mut booted = crate::boot(catalog::DEFAULT_ROW).expect("playground");
            let render = |app: &mut loam::app::session::SessionApp<Playground>,
                          booted: &mut crate::Boot,
                          time,
                          events| {
                app.boundary(&mut booted.session, loam::runtime::Input::default())
                    .expect("boundary");
                let input = egui::RawInput {
                    screen_rect: Some(egui::Rect::from_min_size(
                        egui::Pos2::ZERO,
                        egui::vec2(800.0, 600.0),
                    )),
                    time: Some(time),
                    events,
                    ..Default::default()
                };
                let draw = |context: &egui::Context| {
                    egui::CentralPanel::default().show(context, |ui| {
                        shapes(ui, &booted.session.app, &sender);
                        ui.label("after row");
                    });
                    if multipass && context.input(|input| input.pointer.any_released()) {
                        context.request_discard("reordered row layout");
                    }
                };
                if multipass {
                    context.run(input, draw)
                } else {
                    context.begin_pass(input);
                    draw(&context);
                    context.end_pass()
                }
            };
            let initial = render(&mut app, &mut booted, 0.0, Vec::new());
            let start = text_position(&initial, "24-cell") + egui::vec2(16.0, 6.0);
            let target = text_position(&initial, "8-cell") + egui::vec2(60.0, 6.0);
            let button = |pos, pressed| egui::Event::PointerButton {
                pos,
                pressed,
                button: egui::PointerButton::Primary,
                modifiers: Default::default(),
            };
            render(
                &mut app,
                &mut booted,
                0.01,
                vec![egui::Event::PointerMoved(start), button(start, true)],
            );
            render(
                &mut app,
                &mut booted,
                0.03,
                vec![egui::Event::PointerMoved(target)],
            );
            for frame in 2..20 {
                render(
                    &mut app,
                    &mut booted,
                    frame as f64 * 0.02,
                    vec![egui::Event::PointerMoved(target)],
                );
            }
            let dragging = render(
                &mut app,
                &mut booted,
                0.42,
                vec![egui::Event::PointerMoved(target)],
            );
            let released = render(&mut app, &mut booted, 0.44, vec![button(target, false)]);
            let committed = render(&mut app, &mut booted, 0.46, Vec::new());
            for label in ["5-cell", "after row"] {
                let before = text_position(&dragging, label);
                assert!(
                    before.distance(text_position(&released, label)) < 0.5,
                    "{label} shifted on release: {:?} to {:?}",
                    before,
                    text_position(&released, label)
                );
                assert!(
                    before.distance(text_position(&committed, label)) < 0.5,
                    "{label} shifted on commit: {:?} to {:?}",
                    before,
                    text_position(&committed, label)
                );
            }
            assert!(booted
                .session
                .app
                .slots
                .iter()
                .any(|(_, slot)| slot.entry.label == "24-cell" && slot.index == 3));
        }
    }
}
