use loam_app::egui;
use loam_runtime::Session;

use crate::color::ColorMode;
use crate::composer::{parse_term, Composer};
use crate::consts::{MAX_RATE, W_RANGE};
use crate::mode::Mode;
use crate::projection::Family;
use crate::strip::{Cell, Strip, MAX_CELLS, MAX_T_EXTENT, MIN_CELLS, MIN_T_EXTENT};
use crate::{catalog, push, Intent, Intents, Playground};

const PLANE_LABELS: [&str; 6] = ["xy", "xz", "xw", "yz", "yw", "zw"];

/// Largest magnitude `Rotor4::log` returns, in degrees: pi times the square root of two.
const SCRUB_LIMIT_DEG: f32 = 254.558_44;

#[derive(Default)]
pub(crate) struct Panel {
    formula: String,
    error: Option<String>,
    label: String,
}

pub(crate) fn draw(
    context: &egui::Context,
    session: &Session<Playground>,
    panel: &mut Panel,
    intents: &Intents,
) {
    let app = &session.app;
    egui::SidePanel::left("polytope-playground-controls")
        .default_width(240.0)
        .show(context, |ui| {
            ui.heading("polytope playground");
            ui.separator();

            ui.horizontal(|ui| {
                for mode in Mode::ALL {
                    if ui
                        .selectable_label(*app.mode.get() == mode, mode.name())
                        .clicked()
                    {
                        push(intents, Intent::Mode(mode));
                    }
                }
            });

            ui.separator();
            ui.label("rotation planes");
            ui.horizontal_wrapped(|ui| {
                for (index, label) in PLANE_LABELS.into_iter().enumerate() {
                    if ui
                        .selectable_label(app.spin.get().planes[index], label)
                        .clicked()
                    {
                        push(intents, Intent::Plane(index));
                    }
                }
            });
            playback(ui, app.spin.get().running, app.spin.get().rate, intents);
            if ui
                .selectable_label(*app.gimbal.get(), "gimbal")
                .on_hover_text("drag a ring to turn the whole row")
                .clicked()
            {
                push(intents, Intent::Gimbal);
            }

            if *app.mode.get() == Mode::Compose {
                ui.separator();
                composer(ui, app.composer.get(), panel, intents);
            }

            ui.separator();
            let mut slice = *app.slice.get();
            if ui
                .add(egui::Slider::new(&mut slice, -W_RANGE..=W_RANGE).text("w"))
                .changed()
            {
                push(intents, Intent::Slice(slice));
            }

            ui.separator();
            ui.horizontal_wrapped(|ui| {
                ui.label("colour");
                for mode in ColorMode::ALL {
                    if ui
                        .selectable_label(*app.color.get() == mode, mode.name())
                        .clicked()
                    {
                        push(intents, Intent::Color(mode));
                    }
                }
                if ui
                    .selectable_label(*app.points.get(), "points")
                    .on_hover_text("vertices and cell centres")
                    .clicked()
                {
                    push(intents, Intent::Points);
                }
            });

            ui.separator();
            ui.label("projection");
            ui.horizontal_wrapped(|ui| {
                for family in Family::ALL {
                    if ui
                        .selectable_label(*app.projection.get() == family, family.name())
                        .clicked()
                    {
                        push(intents, Intent::Projection(family));
                    }
                }
            });

            ui.separator();
            filmstrip(ui, *app.strip.get(), intents);

            ui.separator();
            ui.label("row");
            let active = *app.active.get();
            let mut slots: Vec<_> = app.slots.iter().map(|(_, slot)| *slot).collect();
            slots.sort_by_key(|slot| slot.index);
            for slot in slots {
                ui.horizontal(|ui| {
                    if ui
                        .selectable_label(slot.index == active, slot.entry.label)
                        .on_hover_text(slot.entry.long_name)
                        .clicked()
                    {
                        push(intents, Intent::Active(slot.index));
                    }
                    let swap = ui.small_button("swap");
                    egui::Popup::menu(&swap).show(|ui| {
                        ui.set_min_width(140.0);
                        for (card, entry) in catalog::SHAPE_CATALOG.iter().enumerate() {
                            if ui
                                .button(entry.label)
                                .on_hover_text(entry.long_name)
                                .clicked()
                            {
                                push(intents, Intent::Shape(slot.index, card));
                                ui.close();
                            }
                        }
                    });
                });
            }
        });
}

fn composer(ui: &mut egui::Ui, composer: &Composer, panel: &mut Panel, intents: &Intents) {
    ui.label("formula");
    ui.horizontal_wrapped(|ui| {
        let entry = ui.add(
            egui::TextEdit::singleline(&mut panel.formula)
                .hint_text("90deg (xy + zw)")
                .desired_width(150.0),
        );
        let entered = entry.lost_focus() && ui.input(|state| state.key_pressed(egui::Key::Enter));
        if entered || ui.small_button("add").clicked() {
            match parse_term(&panel.formula) {
                Ok(term) => {
                    push(intents, Intent::Term(term));
                    panel.formula.clear();
                    panel.error = None;
                }
                Err(error) => panel.error = Some(error),
            }
        } else if panel.formula.is_empty() {
            panel.error = None;
        }
    });
    if let Some(error) = &panel.error {
        ui.colored_label(egui::Color32::from_rgb(255, 120, 120), error.as_str());
    }

    ui.horizontal_wrapped(|ui| {
        for (index, label) in PLANE_LABELS.into_iter().enumerate() {
            if ui.small_button(format!("+{label}")).clicked() {
                push(intents, Intent::Draft(index));
            }
        }
    });

    let drafted: u32 = composer.draft.iter().map(|count| u32::from(*count)).sum();
    if drafted > 0 {
        panel.label.clear();
        crate::composer::Term {
            planes: composer.draft,
            scalar: None,
        }
        .write(&mut panel.label);
        ui.horizontal_wrapped(|ui| {
            ui.monospace(panel.label.as_str());
            if ui.small_button("commit").clicked() {
                push(intents, Intent::CommitDraft);
            }
            if ui.small_button("discard").clicked() {
                push(intents, Intent::ClearDraft);
            }
        });
    }

    for (index, term) in composer.terms().iter().enumerate() {
        panel.label.clear();
        term.write(&mut panel.label);
        ui.horizontal_wrapped(|ui| {
            ui.monospace(panel.label.as_str());
            if ui.small_button("x").clicked() {
                push(intents, Intent::DropTerm(index));
            }
        });
    }
    if !composer.terms().is_empty() && ui.button("clear sequence").clicked() {
        push(intents, Intent::ClearTerms);
    }

    if composer.axis().is_some() {
        let mut degrees = composer.scrub.to_degrees();
        if ui
            .add(egui::Slider::new(&mut degrees, -SCRUB_LIMIT_DEG..=SCRUB_LIMIT_DEG).text("f"))
            .changed()
        {
            push(intents, Intent::Scrub(degrees.to_radians()));
        }
    }
}

const CALLOUT_INSET_PT: f32 = 10.0;

/// Anchors one label per slot at the body's own image point, through the root view's NDC.
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

/// Play, pause, and the rotation rate the row and the strip's t fan share.
fn playback(ui: &mut egui::Ui, running: bool, rate: f32, intents: &Intents) {
    ui.horizontal(|ui| {
        if ui.button(if running { "pause" } else { "play" }).clicked() {
            push(intents, Intent::Running(!running));
        }
        let mut held = rate;
        if ui
            .add(egui::Slider::new(&mut held, 0.0..=MAX_RATE).text("rate"))
            .changed()
        {
            push(intents, Intent::Rate(held));
        }
    });
}

fn filmstrip(ui: &mut egui::Ui, strip: Strip, intents: &Intents) {
    let mut held = strip;
    ui.horizontal_wrapped(|ui| {
        ui.label("filmstrip");
        if ui.selectable_label(strip.on, "on").clicked() {
            held.on = !held.on;
        }
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
        push(intents, Intent::Strip(held));
    }
}

/// One w and t readout over each cell of the grid, in the cell's own rectangle.
pub(crate) fn strip_labels(context: &egui::Context, session: &Session<Playground>, cells: &[Cell]) {
    let strip = *session.app.strip.get();
    if !strip.on {
        return;
    }
    let per_point = context.pixels_per_point().max(1e-3);
    for (index, cell) in cells.iter().enumerate() {
        let at = egui::pos2(
            (cell.viewport.x as f32 + cell.viewport.width as f32 * 0.5) / per_point,
            (cell.viewport.y as f32 + 8.0) / per_point,
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
