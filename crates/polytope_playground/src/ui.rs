use loam_app::egui;
use loam_runtime::Session;

use crate::consts::W_RANGE;
use crate::mode::Mode;
use crate::{push, Intent, Intents, Playground};

const PLANE_LABELS: [&str; 6] = ["xy", "xz", "xw", "yz", "yw", "zw"];

pub(crate) fn draw(context: &egui::Context, session: &Session<Playground>, intents: &Intents) {
    let app = &session.app;
    egui::SidePanel::left("polytope-playground-controls")
        .default_width(220.0)
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
            let running = app.spin.get().running;
            if ui.button(if running { "pause" } else { "spin" }).clicked() {
                push(intents, Intent::Running(!running));
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
            ui.label("row");
            let active = *app.active.get();
            let mut slots: Vec<_> = app.slots.iter().map(|(_, slot)| *slot).collect();
            slots.sort_by_key(|slot| slot.index);
            for slot in slots {
                if ui
                    .selectable_label(slot.index == active, slot.entry.label)
                    .on_hover_text(slot.entry.long_name)
                    .clicked()
                {
                    push(intents, Intent::Active(slot.index));
                }
            }
        });
}
