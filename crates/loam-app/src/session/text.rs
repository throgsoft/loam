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
    pub(crate) fn new(font: Option<Vec<u8>>) -> Self {
        #[cfg(feature = "egui")]
        let font = font.or_else(bundled_font);
        Self {
            pass: font.map(|font| TextPass::new("text", font, BAKE_PX)),
        }
    }

    pub(crate) fn pass(&self) -> Option<TextPass> {
        self.pass.clone()
    }

    /// Returns true when published text is dropped for want of a font.
    pub(crate) fn publish(&self, overlay: &Overlay, seat: Seat) -> bool {
        let Some(pass) = self.pass.as_ref() else {
            return overlay.text().next().is_some();
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
        false
    }
}

#[cfg(feature = "egui")]
fn bundled_font() -> Option<Vec<u8>> {
    let fonts = loam_egui::egui::FontDefinitions::default();
    let chosen = fonts
        .font_data
        .iter()
        .find(|(name, _)| name.starts_with("Hack"))
        .or_else(|| fonts.font_data.iter().next())?;
    Some(chosen.1.font.to_vec())
}
