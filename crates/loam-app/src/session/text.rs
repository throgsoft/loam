use loam_runtime::Overlay;
use loam_text::{TextDraw, TextPass};

// loam-text has no mip chain, so bake at 4x a 16 pt readout and only ever minify.
const BAKE_PX: f32 = 64.0;

pub(crate) struct ScreenText {
    pass: Option<TextPass>,
}

#[derive(Clone, Copy)]
pub(crate) struct Seat {
    pub(crate) origin_px: [f32; 2],
    pub(crate) pixels_per_point: f32,
}

impl ScreenText {
    pub(crate) fn new() -> Self {
        Self {
            pass: font().map(|font| TextPass::new("text", font, BAKE_PX)),
        }
    }

    pub(crate) fn pass(&self) -> Option<TextPass> {
        self.pass.clone()
    }

    pub(crate) fn publish(&self, overlay: &Overlay, seat: Seat) {
        let Some(pass) = self.pass.as_ref() else {
            return;
        };
        let scale = seat.pixels_per_point;
        pass.publish(overlay.text().map(|(text, record)| {
            (
                text,
                TextDraw {
                    origin_px: [
                        seat.origin_px[0] + record.origin_pt[0] * scale,
                        seat.origin_px[1] + record.origin_pt[1] * scale,
                    ],
                    size_px: record.size_pt * scale,
                    color: record.color,
                },
            )
        }));
    }
}

#[cfg(feature = "egui")]
fn font() -> Option<Vec<u8>> {
    let fonts = loam_egui::egui::FontDefinitions::default();
    let chosen = fonts
        .font_data
        .iter()
        .find(|(name, _)| name.starts_with("Hack"))
        .or_else(|| fonts.font_data.iter().next())?;
    Some(chosen.1.font.to_vec())
}

#[cfg(not(feature = "egui"))]
fn font() -> Option<Vec<u8>> {
    None
}
