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

#[cfg(test)]
mod tests {
    use super::*;

    fn readout(value: f32, planes: [bool; 6]) -> Readout {
        Readout {
            slice: value,
            rate: value,
            bodies: 8,
            planes,
            toybox: false,
        }
    }

    #[test]
    fn every_extreme_float_stays_inside_the_fonts_own_glyphs() {
        let mut out = String::new();
        for value in [
            0.0_f32,
            -0.0,
            f32::NAN,
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::MIN,
            f32::MAX,
            f32::MIN_POSITIVE,
            -1.0e-30,
        ] {
            write_readout(&mut out, &readout(value, [true; 6]));
            assert!(
                loam::text::is_renderable(&out),
                "the readout for {value} carries characters loam-text drops: {out:?}"
            );
        }
    }

    #[test]
    fn the_plane_strip_names_exactly_the_active_planes_in_place() {
        let mut out = String::new();
        let mut planes = [false; 6];
        planes[2] = true;
        planes[3] = true;
        write_readout(&mut out, &readout(0.0, planes));
        let strip = out.lines().last().expect("the planes line").to_string();
        assert!(
            strip.ends_with(".. .. xw yz .. .."),
            "the strip was {strip:?}"
        );

        write_readout(&mut out, &readout(0.0, [false; 6]));
        let off = out.lines().last().expect("the planes line");
        assert_eq!(
            off.chars().count(),
            strip.chars().count(),
            "the strip changes width with the active planes, so the column below it moves"
        );
    }

    #[test]
    fn the_seat_scales_its_whole_placement_by_pixels_per_point() {
        let free = egui::Rect::from_min_max(egui::pos2(0.0, 24.0), egui::pos2(1280.0, 720.0));
        let unit = Seat::in_panel(free, 1.0);
        for ratio in [1.25_f32, 1.5, 2.0, 3.0] {
            let scaled = Seat::in_panel(free, ratio);
            for (at, held) in scaled.draws().iter().zip(unit.draws().iter()) {
                assert!(
                    (at.origin_px[0] - held.origin_px[0] * ratio).abs() < 1e-3
                        && (at.origin_px[1] - held.origin_px[1] * ratio).abs() < 1e-3,
                    "at {ratio}x the origin is {:?}, not {:?} scaled",
                    at.origin_px,
                    held.origin_px
                );
                assert!((at.size_px - held.size_px * ratio).abs() < 1e-3);
                assert_eq!(at.color, held.color, "the scale changed a color");
            }
        }
    }
}
