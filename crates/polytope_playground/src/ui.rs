use loam_app::egui;
use loam_runtime::Session;

use crate::color::ColorMode;
use crate::composer::{parse_term, Composer};
use crate::consts::W_RANGE;
use crate::mode::Mode;
use crate::projection::Family;
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
            ui.horizontal(|ui| {
                let running = app.spin.get().running;
                if ui.button(if running { "pause" } else { "spin" }).clicked() {
                    push(intents, Intent::Running(!running));
                }
                if ui
                    .selectable_label(*app.gimbal.get(), "gimbal")
                    .on_hover_text("drag a ring to turn the whole row")
                    .clicked()
                {
                    push(intents, Intent::Gimbal);
                }
            });

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
