//! Edge tessellation and radial clipping in the projected body frame.

use crate::LineMesh;
use glam::{Vec3, Vec4};

const MIN_EDGE_RADIUS: f32 = 1e-6;
const STEREOGRAPHIC_POLE_FAR_CAP: f32 = 1.0e4;

pub fn stereographic_clip_radius(
    projection: &loam_math::Projection<4>,
    view_radius: f32,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Stereographic { .. } => Some(view_radius),
        loam_math::Projection::Identity
        | loam_math::Projection::Orthographic { .. }
        | loam_math::Projection::Perspective4D { .. }
        | loam_math::Projection::Schlegel { .. } => None,
    }
}

/// Samples in the body frame, then clips before adding the world translation.
#[allow(clippy::too_many_arguments)]
pub fn push_projected_chord(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    view_radius: f32,
    samples: usize,
) {
    let samples = samples.max(1);
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    let sample_at = |p4: Vec4| {
        let projected = stereographic_view_point(p4, projection);
        (projected, (projected + body_pos_r3).to_array())
    };
    let (proj0, world0) = sample_at(a);
    let mut prev_world = world0;
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    for k in 1..=samples {
        let s = k as f32 / samples as f32;
        let (proj, world) = sample_at(a.lerp(b, s));
        let c = [
            color_a[0] + (color_b[0] - color_a[0]) * s,
            color_a[1] + (color_b[1] - color_a[1]) * s,
            color_a[2] + (color_b[2] - color_a[2]) * s,
            color_a[3] + (color_b[3] - color_a[3]) * s,
        ];
        let cur_in = sample_in_radius(proj, clip_radius);
        if prev_in && cur_in {
            mesh.segments.push((prev_world, world));
            mesh.colors.push((prev_c, c));
            mesh.widths.push(width);
        }
        prev_world = world;
        prev_c = c;
        prev_in = cur_in;
    }
}

#[inline]
pub fn sample_in_radius(projected: Vec3, radius: Option<f32>) -> bool {
    projected.is_finite() && radius.is_none_or(|r| projected.length_squared() <= r * r)
}

// do Carmo, Differential Geometry of Curves and Surfaces, 1976, §1.5.
fn radius_crossing_t(p_in: Vec3, p_out: Vec3, r: f32) -> f32 {
    let d = p_out - p_in;
    let a = d.length_squared();
    if a <= f32::MIN_POSITIVE {
        return 1.0;
    }
    let b = p_in.dot(d);
    let c = p_in.length_squared() - r * r;
    let disc = (b * b - a * c).max(0.0);
    ((-b + disc.sqrt()) / a).clamp(0.0, 1.0)
}

fn clip_point(
    p_in: Vec3,
    p_out: Vec3,
    c_in: [f32; 4],
    c_out: [f32; 4],
    t: f32,
    body_pos: Vec3,
) -> ([f32; 3], [f32; 4]) {
    let boundary = (p_in.lerp(p_out, t) + body_pos).to_array();
    let color = [
        c_in[0] + (c_out[0] - c_in[0]) * t,
        c_in[1] + (c_out[1] - c_in[1]) * t,
        c_in[2] + (c_out[2] - c_in[2]) * t,
        c_in[3] + (c_out[3] - c_in[3]) * t,
    ];
    (boundary, color)
}

fn push_clipped_subsegment(
    mesh: &mut LineMesh<3>,
    clip_radius: Option<f32>,
    width: f32,
    body_pos_r3: Vec3,
    prev: (Vec3, [f32; 3], [f32; 4], bool),
    cur: (Vec3, [f32; 3], [f32; 4], bool),
) {
    let (prev_proj, prev_world, prev_c, prev_in) = prev;
    let (cur_proj, cur_world, cur_c, cur_in) = cur;
    if !prev_proj.is_finite() || !cur_proj.is_finite() {
        return;
    }
    let mut push = |a_world, b_world, a_c, b_c| {
        mesh.segments.push((a_world, b_world));
        mesh.colors.push((a_c, b_c));
        mesh.widths.push(width);
    };
    match (clip_radius, prev_in, cur_in) {
        (None, _, _) | (Some(_), true, true) => push(prev_world, cur_world, prev_c, cur_c),
        (Some(_), false, false) => {}
        (Some(r), true, false) => {
            let t = radius_crossing_t(prev_proj, cur_proj, r);
            let (bw, bc) = clip_point(prev_proj, cur_proj, prev_c, cur_c, t, body_pos_r3);
            push(prev_world, bw, prev_c, bc);
        }
        (Some(r), false, true) => {
            let t = radius_crossing_t(cur_proj, prev_proj, r);
            let (bw, bc) = clip_point(cur_proj, prev_proj, cur_c, prev_c, t, body_pos_r3);
            push(bw, cur_world, bc, cur_c);
        }
    }
}

/// Restores near-pole magnitude; the exact pole produces a non-finite sample for culling.
// Coxeter, Introduction to Geometry, 1969, §6.9.
pub fn stereographic_view_point(p: Vec4, projection: &loam_math::Projection<4>) -> Vec3 {
    let proj =
        <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection);
    let loam_math::Projection::Stereographic { pole } = projection else {
        return proj;
    };
    let dot = p.normalize().dot(*pole).clamp(-1.0, 1.0);
    let raw = 1.0 - dot;
    if raw < loam_math::STEREOGRAPHIC_POLE_EPSILON && proj.length() <= MIN_EDGE_RADIUS {
        return Vec3::NAN;
    }
    if raw < loam_math::STEREOGRAPHIC_POLE_EPSILON && proj.length() > MIN_EDGE_RADIUS {
        let true_mag = ((1.0 + dot) / raw.max(f32::MIN_POSITIVE))
            .sqrt()
            .min(STEREOGRAPHIC_POLE_FAR_CAP);
        proj.normalize() * true_mag
    } else {
        proj
    }
}

/// Removes appended triangles with any vertex outside the projected radius.
pub fn retain_in_radius_triangles(
    indices: &mut Vec<[u32; 3]>,
    start_i: usize,
    start_v: usize,
    projected: &[Vec3],
    radius: Option<f32>,
) {
    if radius.is_none() {
        return;
    }
    let appended = &mut indices[start_i..];
    let mut write = 0usize;
    for read in 0..appended.len() {
        let tri = appended[read];
        let in_radius = tri
            .iter()
            .all(|&i| sample_in_radius(projected[i as usize - start_v], radius));
        if in_radius {
            appended[write] = tri;
            write += 1;
        }
    }
    indices.truncate(start_i + write);
}

/// Blends body-frame chords and arcs with `blend` in `[0, 1]`; arc radii interpolate about `arc_center`.
#[allow(clippy::too_many_arguments)]
pub fn push_blended_edge(
    mesh: &mut LineMesh<3>,
    a: Vec4,
    b: Vec4,
    arc_center: Vec4,
    color_a: [f32; 4],
    color_b: [f32; 4],
    width: f32,
    blend: f32,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
    samples: usize,
    view_radius: f32,
) {
    let offset_a = a - arc_center;
    let offset_b = b - arc_center;
    let (radius_a, radius_b) = if blend <= 0.0 {
        (0.0, 0.0)
    } else {
        (offset_a.length(), offset_b.length())
    };
    if blend <= 0.0 || radius_a < MIN_EDGE_RADIUS || radius_b < MIN_EDGE_RADIUS {
        let clip_radius = stereographic_clip_radius(projection, view_radius);
        let a3 = stereographic_view_point(a, projection);
        let b3 = stereographic_view_point(b, projection);
        if sample_in_radius(a3, clip_radius) && sample_in_radius(b3, clip_radius) {
            mesh.segments
                .push(((a3 + body_pos_r3).to_array(), (b3 + body_pos_r3).to_array()));
            mesh.colors.push((color_a, color_b));
            mesh.widths.push(width);
        }
        return;
    }

    let samples = samples.max(1);
    let clip_radius = stereographic_clip_radius(projection, view_radius);
    let p0u = offset_a / radius_a;
    let p1u = offset_b / radius_b;
    let proj0 = stereographic_view_point(a, projection);
    let mut prev_proj = proj0;
    let mut prev_world = (proj0 + body_pos_r3).to_array();
    let mut prev_c = color_a;
    let mut prev_in = sample_in_radius(proj0, clip_radius);
    let mut k = 0;
    <loam_math::SphericalS3Embedded as loam_math::RasterizableSpace<4>>::tessellate_segment(
        p0u,
        p1u,
        samples,
        |arc_pt| {
            let index = k;
            k += 1;
            if index == 0 {
                return;
            }
            let k = index;
            let s = k as f32 / samples as f32;
            let flat = a.lerp(b, s);
            let radius = radius_a + (radius_b - radius_a) * s;
            let sphere = arc_center + radius * arc_pt;
            let proj = stereographic_view_point(flat.lerp(sphere, blend), projection);
            let world = (proj + body_pos_r3).to_array();
            let c = [
                color_a[0] + (color_b[0] - color_a[0]) * s,
                color_a[1] + (color_b[1] - color_a[1]) * s,
                color_a[2] + (color_b[2] - color_a[2]) * s,
                color_a[3] + (color_b[3] - color_a[3]) * s,
            ];
            let cur_in = sample_in_radius(proj, clip_radius);
            push_clipped_subsegment(
                mesh,
                clip_radius,
                width,
                body_pos_r3,
                (prev_proj, prev_world, prev_c, prev_in),
                (proj, world, c, cur_in),
            );
            prev_proj = proj;
            prev_world = world;
            prev_c = c;
            prev_in = cur_in;
        },
    );
}
