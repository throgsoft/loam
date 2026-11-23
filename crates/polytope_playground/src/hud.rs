use std::fmt::Write as _;

use loam::math::Plane4;
use loam::runtime::{Overlay, TextRecord};

const SIZE_PT: f32 = 16.0;

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

pub(crate) fn draw(overlay: &mut Overlay, readout: &Readout, lines: &mut String) {
    write_readout(lines, readout);
    let shadow = INSET_PT + SHADOW_OFFSET_PT;
    // Shadow first so the body paints over it.
    overlay.push_text(
        lines,
        TextRecord {
            origin_pt: [shadow, shadow],
            size_pt: SIZE_PT,
            color: SHADOW_COLOR,
        },
    );
    overlay.push_text(
        lines,
        TextRecord {
            origin_pt: [INSET_PT, INSET_PT],
            size_pt: SIZE_PT,
            color: COLOR,
        },
    );
}
