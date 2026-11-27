use loam_runtime::Overlay;
use loam_text::TextPass;

// loam-text has no mip chain, so bake at 4x a 16 pt readout and only ever minify.
const BAKE_PX: f32 = 64.0;

pub(crate) struct ScreenText {
    pass: Option<TextPass>,
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

    /// True when published text is dropped for want of a font.
    pub(crate) fn drops(&self, overlay: &Overlay) -> bool {
        self.pass.is_none() && overlay.text().next().is_some()
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
