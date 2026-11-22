use std::fmt::Write as _;

use loam::app::egui;
use loam::math::Plane4;
use loam::text::{TextDraw, TextPass};

const SIZE_PT: f32 = 16.0;

// loam-text has no mip chain, so bake at 4x and only ever minify.
const BAKE_PX: f32 = 4.0 * SIZE_PT;

const INSET_PT: f32 = 16.0;

const COLOR: [f32; 4] = [0.92, 0.96, 1.0, 1.0];

const SHADOW_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.7];

const SHADOW_OFFSET_PT: f32 = 1.0;

const PLANE_OFF: &str = "..";

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Readout {
    pub(crate) slice: f32,
    pub(crate) rate: f32,
    pub(crate) bodies: usize,
    pub(crate) planes: [bool; 6],
    pub(crate) toybox: bool,
}

// loam-text lays out on advance widths only, so columns need padded formatting.
pub(crate) fn write_readout(out: &mut String, readout: &Readout) {
    out.clear();
    let _ = writeln!(out, "{:<6} {:>+8.3}", "w", readout.slice);
    if readout.toybox {
        let _ = write!(out, "{:<6} {:>7}", "bodies", readout.bodies);
        return;
    }
    let _ = writeln!(out, "{:<6} {:>7.2}x", "rate", readout.rate);
    let _ = writeln!(out, "{:<6} {:>7}", "bodies", readout.bodies);
    let _ = write!(out, "{:<6} ", "planes");
    for (index, plane) in Plane4::ALL.into_iter().enumerate() {
        if index > 0 {
            out.push(' ');
        }
        out.push_str(if readout.planes[index] {
            plane.label()
        } else {
            PLANE_OFF
        });
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Seat {
    pub(crate) origin_px: [f32; 2],
    pub(crate) pixels_per_point: f32,
}

impl Default for Seat {
    fn default() -> Self {
        Self {
            origin_px: [INSET_PT, INSET_PT],
            pixels_per_point: 1.0,
        }
    }
}

impl Seat {
    pub(crate) fn in_panel(free: egui::Rect, pixels_per_point: f32) -> Self {
        Self {
            origin_px: [
                (free.left() + INSET_PT) * pixels_per_point,
                (free.top() + INSET_PT) * pixels_per_point,
            ],
            pixels_per_point,
        }
    }

    fn size_px(&self) -> f32 {
        SIZE_PT * self.pixels_per_point
    }

    // Shadow first so the body paints over it.
    fn draws(&self) -> [TextDraw; 2] {
        let offset = SHADOW_OFFSET_PT * self.pixels_per_point;
        [
            TextDraw {
                origin_px: [self.origin_px[0] + offset, self.origin_px[1] + offset],
                size_px: self.size_px(),
                color: SHADOW_COLOR,
            },
            TextDraw {
                origin_px: self.origin_px,
                size_px: self.size_px(),
                color: COLOR,
            },
        ]
    }
}

pub(crate) fn pass() -> TextPass {
    TextPass::new("hud", font_bytes().unwrap_or_default(), BAKE_PX)
}

pub(crate) fn publish(pass: &TextPass, readout: Option<&Readout>, seat: Seat, lines: &mut String) {
    match readout {
        Some(readout) => {
            write_readout(lines, readout);
            pass.publish(lines, &seat.draws());
        }
        None => pass.publish("", &[]),
    }
}

fn font_bytes() -> Option<Vec<u8>> {
    let fonts = egui::FontDefinitions::default();
    let chosen = fonts
        .font_data
        .iter()
        .find(|(name, _)| name.starts_with("Hack"))
        .or_else(|| fonts.font_data.iter().next())?;
    Some(chosen.1.font.to_vec())
}
