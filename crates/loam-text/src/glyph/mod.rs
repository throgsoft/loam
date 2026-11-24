mod field;
mod hull;
mod outline;
mod solid;

pub use field::DistanceField2D;
pub use solid::{append_field_prism, GlyphSolid};

use ab_glyph::{Font, FontRef, GlyphId};
use glam::Vec2;

/// Finite so callers can still combine it with `min`/`max` without producing NaN.
pub const BLANK_DISTANCE: f32 = 1.0e9;

/// Below this many cells per em the render bake cannot resolve a stem.
pub const MIN_RESOLUTION: u32 = 4;

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum GlyphError {
    #[error("font has no glyph for {ch:?}")]
    NoGlyph { ch: char },

    #[error("{ch:?} is a control character; the glyph pipeline takes printable text only")]
    ControlCharacter { ch: char },

    #[error("outline for {ch:?} encloses no area")]
    DegenerateOutline { ch: char },

    #[error("font declares no units_per_em")]
    NoUnitsPerEm,

    #[error("GlyphParams::{field} must be finite and positive, got {value}")]
    NonPositive { field: &'static str, value: f32 },

    #[error("GlyphParams::{field} must be at least {MIN_RESOLUTION}, got {resolution}")]
    Resolution {
        field: &'static str,
        resolution: u32,
    },
}

/// All lengths are world units except [`Self::flatten_tolerance_em`].
#[derive(Clone, Debug, PartialEq)]
pub struct GlyphParams {
    pub em_size: f32,
    /// Extent along `z`, centred on `z = 0`.
    pub depth: f32,
    /// `(min, max)` `w` extent of the slab the letter occupies.
    pub slab: (f32, f32),
    /// Grid cells per em for the bake and the render decomposition.
    pub resolution: u32,
    /// Grid cells per em for the collider cover.
    pub collider_resolution: u32,
    /// Requested chord deviation in em, subject to a 64-subdivision budget per Bezier segment.
    pub flatten_tolerance_em: f32,
    /// RGBA linear.
    pub color: [f32; 4],
}

impl Default for GlyphParams {
    fn default() -> Self {
        Self {
            em_size: 1.0,
            depth: 0.15,
            slab: (-0.075, 0.075),
            resolution: 48,
            collider_resolution: 48,
            flatten_tolerance_em: 0.002,
            color: [1.0, 1.0, 1.0, 1.0],
        }
    }
}

impl GlyphParams {
    fn validate(&self) -> Result<(), GlyphError> {
        for (field, value) in [
            ("em_size", self.em_size),
            ("depth", self.depth),
            ("slab", self.slab.1 - self.slab.0),
            ("flatten_tolerance_em", self.flatten_tolerance_em),
        ] {
            if value <= 0.0 || !value.is_finite() {
                return Err(GlyphError::NonPositive { field, value });
            }
        }
        for (field, resolution) in [
            ("resolution", self.resolution),
            ("collider_resolution", self.collider_resolution),
        ] {
            if resolution < MIN_RESOLUTION {
                return Err(GlyphError::Resolution { field, resolution });
            }
        }
        Ok(())
    }
}

/// Places glyphs in a shared word frame with baseline `y = 0` and first pen origin `x = 0`.
pub fn layout_word(
    font: &FontRef<'_>,
    text: &str,
    params: &GlyphParams,
) -> Result<Vec<GlyphSolid>, GlyphError> {
    params.validate()?;
    let units_per_em = font.units_per_em().ok_or(GlyphError::NoUnitsPerEm)?;
    let units_to_world = params.em_size / units_per_em;
    let tolerance_world = params.flatten_tolerance_em * params.em_size;
    let cell = params.em_size / params.resolution as f32;

    let mut solids = Vec::with_capacity(text.chars().count());
    let mut pen_x = 0.0_f32;
    let mut previous: Option<GlyphId> = None;

    for ch in text.chars() {
        if ch.is_control() {
            return Err(GlyphError::ControlCharacter { ch });
        }
        let id = font.glyph_id(ch);
        // Glyph 0 is `.notdef` by the OpenType specification.
        if id.0 == 0 {
            return Err(GlyphError::NoGlyph { ch });
        }

        if let Some(previous) = previous {
            pen_x += font.kern_unscaled(previous, id) * units_to_world;
        }
        let advance = font.h_advance_unscaled(id) * units_to_world;
        let pen_origin = Vec2::new(pen_x, 0.0);

        let field = match font.outline(id) {
            None => None,
            Some(raw) => {
                let mut contours =
                    outline::contours_from_curves(&raw.curves, units_to_world, tolerance_world);
                for contour in &mut contours {
                    for point in &mut contour.points {
                        *point += pen_origin;
                    }
                }
                Some(
                    field::DistanceField2D::bake(&contours, cell)
                        .ok_or(GlyphError::DegenerateOutline { ch })?,
                )
            }
        };

        solids.push(GlyphSolid::new(ch, pen_origin, advance, params, field));
        pen_x += advance;
        previous = Some(id);
    }

    Ok(solids)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> GlyphParams {
        GlyphParams {
            resolution: 32,
            ..GlyphParams::default()
        }
    }

    #[test]
    fn letter_cover() {
        let bytes = include_bytes!("../../../examples/fonts/lmroman10-bold.otf");
        let font = FontRef::try_from_slice(bytes).expect("parse font");
        let letters = layout_word(&font, "LOAM", &params()).expect("layout");

        for letter in &letters {
            let cover = letter.collider_cover().expect("cover");
            assert!(!cover.clipped(), "{:?} ran off its domain", letter.ch());
            let margin = letter.collider_margin();

            const PROBES: usize = 161;
            let x0 = letter.pen_origin().x - 0.25 * letter.advance();
            let width = 1.5 * letter.advance();
            let (mut ink, mut clear) = (0, 0);
            for j in 0..PROBES {
                for i in 0..PROBES {
                    let p = Vec2::new(
                        x0 + width * (i as f32 + 0.317) / PROBES as f32,
                        -0.25 + 1.25 * (j as f32 + 0.211) / PROBES as f32,
                    );
                    let d = letter.distance_2d(p);
                    if d <= 0.0 {
                        ink += 1;
                        assert!(
                            cover.contains(p.to_array()),
                            "{:?} leaves ink at {p} uncovered",
                            letter.ch()
                        );
                    } else if d > margin {
                        clear += 1;
                        assert!(
                            !cover.contains(p.to_array()),
                            "{:?} covers {p}, which is {d} clear of the ink",
                            letter.ch()
                        );
                    }
                }
            }
            assert!(ink > 1_000, "{:?}: only {ink} ink probes", letter.ch());
            assert!(
                clear > 1_000,
                "{:?}: only {clear} clear probes",
                letter.ch()
            );
        }
    }
}
