use std::fmt::Write as _;

use anyhow::Result;
use loam_app::{egui, RenderCtx};
use loam_text::TextRenderer;

use crate::state::Demo;

const HUD_SIZE_PT: f32 = 16.0;
// loam-text has no mip chain, so bake at 4x and only ever minify.
const HUD_BAKE_PX: f32 = 4.0 * HUD_SIZE_PT;
const HUD_INSET_PT: f32 = 16.0;
const HUD_COLOR: [f32; 4] = [0.92, 0.96, 1.0, 1.0];
const HUD_SHADOW_COLOR: [f32; 4] = [0.0, 0.0, 0.0, 0.7];
const HUD_SHADOW_OFFSET_PT: f32 = 1.0;

// Order matches `Plane4::ALL`.
const PLANE_NAMES: [&str; 6] = ["xy", "xz", "xw", "yz", "yw", "zw"];
const PLANE_OFF: &str = "..";

struct Readout {
    w_slice: f32,
    rot_time: f32,
    rate_scale: f32,
    slots: usize,
    active: [bool; 6],
}

impl Readout {
    fn from_demo(demo: &Demo) -> Self {
        Self {
            w_slice: demo.w_slice,
            rot_time: demo.rot_time,
            rate_scale: demo.rate_scale,
            slots: demo.render_row().len(),
            active: demo.spins.spin().active,
        }
    }
}

// loam-text lays out on advance widths only, so columns need padded formatting.
fn write_readout(out: &mut String, r: &Readout) {
    out.clear();
    let _ = writeln!(out, "{:<6} {:>+8.3}", "w", r.w_slice);
    let _ = writeln!(out, "{:<6} {:>7.2}s", "t", r.rot_time);
    let _ = writeln!(out, "{:<6} {:>7.2}x", "rate", r.rate_scale);
    let _ = writeln!(out, "{:<6} {:>7}", "bodies", r.slots);
    let _ = write!(out, "{:<6} ", "planes");
    for (i, name) in PLANE_NAMES.iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        out.push_str(if r.active[i] { name } else { PLANE_OFF });
    }
}

// loam-text positions in physical pixels and has no scale-factor notion.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HudSeat {
    origin_px: [f32; 2],
    pixels_per_point: f32,
}

impl Default for HudSeat {
    fn default() -> Self {
        Self {
            origin_px: [HUD_INSET_PT, HUD_INSET_PT],
            pixels_per_point: 1.0,
        }
    }
}

impl HudSeat {
    fn size_px(&self) -> f32 {
        HUD_SIZE_PT * self.pixels_per_point
    }
}

fn hud_origin(free: egui::Rect) -> egui::Pos2 {
    free.left_top() + egui::vec2(HUD_INSET_PT, HUD_INSET_PT)
}

pub(crate) fn hud_seat(free: egui::Rect, pixels_per_point: f32) -> HudSeat {
    let origin = hud_origin(free);
    HudSeat {
        origin_px: [origin.x * pixels_per_point, origin.y * pixels_per_point],
        pixels_per_point,
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct HudDraw {
    origin_px: [f32; 2],
    size_px: f32,
    color: [f32; 4],
}

// Shadow first so the body paints over it.
fn draw_list(seat: HudSeat) -> [HudDraw; 2] {
    let offset = HUD_SHADOW_OFFSET_PT * seat.pixels_per_point;
    [
        HudDraw {
            origin_px: [seat.origin_px[0] + offset, seat.origin_px[1] + offset],
            size_px: seat.size_px(),
            color: HUD_SHADOW_COLOR,
        },
        HudDraw {
            origin_px: seat.origin_px,
            size_px: seat.size_px(),
            color: HUD_COLOR,
        },
    ]
}

pub(crate) struct TextHud {
    text: TextRenderer,
    line_buf: String,
}

impl TextHud {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        samples: u32,
    ) -> Result<Self> {
        let text = TextRenderer::new(
            device,
            queue,
            format,
            hud_font_bytes(),
            HUD_BAKE_PX,
            samples,
        )?;
        Ok(Self {
            text,
            line_buf: String::new(),
        })
    }

    // Recorded, not submitted: a nested submit would land under the scene passes.
    pub(crate) fn record(&mut self, ctx: &mut RenderCtx<'_>, demo: &Demo, seat: HudSeat) {
        if !demo.show_text_hud {
            return;
        }
        let Self { text, line_buf } = self;
        write_readout(line_buf, &Readout::from_demo(demo));
        for draw in draw_list(seat) {
            text.queue(line_buf, draw.origin_px, draw.size_px, draw.color);
        }
        let cfg = &ctx.rd.surface_bundle.config;
        text.record(
            &ctx.rd.device,
            &ctx.rd.queue,
            ctx.encoder,
            ctx.view,
            [cfg.width as f32, cfg.height as f32],
        );
    }
}

fn hud_font_bytes() -> &'static [u8] {
    epaint_default_fonts::HACK_REGULAR
}

#[cfg(test)]
mod tests {
    use super::*;

    fn readout(w_slice: f32, rot_time: f32, rate_scale: f32, active: [bool; 6]) -> Readout {
        Readout {
            w_slice,
            rot_time,
            rate_scale,
            slots: 8,
            active,
        }
    }

    #[test]
    fn readout_is_renderable_for_every_extreme_float() {
        let mut out = String::new();
        for &value in &[
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
            write_readout(&mut out, &readout(value, value, value, [true; 6]));
            assert!(
                loam_text::is_renderable(&out),
                "readout for {value} contains characters loam-text would drop: {out:?}"
            );
        }
    }

    #[test]
    fn plane_strip_names_exactly_the_active_planes() {
        let mut out = String::new();
        let mut active = [false; 6];
        active[2] = true;
        active[3] = true;
        write_readout(&mut out, &readout(0.0, 0.0, 1.0, active));
        let strip = out.lines().last().expect("planes line").to_string();
        assert!(strip.ends_with(".. .. xw yz .. .."), "strip was {strip:?}");

        write_readout(&mut out, &readout(0.0, 0.0, 1.0, [false; 6]));
        let off = out.lines().last().expect("planes line");
        assert_eq!(off.chars().count(), strip.chars().count());
    }

    #[test]
    fn draw_list_scales_the_whole_placement_by_pixels_per_point() {
        let free = egui::Rect::from_min_max(egui::pos2(0.0, 24.0), egui::pos2(1280.0, 720.0));
        let unit = draw_list(hud_seat(free, 1.0));
        for ppp in [1.25_f32, 1.5, 2.0, 3.0] {
            let scaled = draw_list(hud_seat(free, ppp));
            for (u, s) in unit.iter().zip(scaled.iter()) {
                assert!(
                    (s.origin_px[0] - u.origin_px[0] * ppp).abs() < 1e-3
                        && (s.origin_px[1] - u.origin_px[1] * ppp).abs() < 1e-3,
                    "at {ppp}x the origin was {:?}, expected {:?} scaled",
                    s.origin_px,
                    u.origin_px
                );
                assert!((s.size_px - u.size_px * ppp).abs() < 1e-3, "size at {ppp}x");
                assert_eq!(s.color, u.color, "scale must not touch color");
            }
        }
    }
}
