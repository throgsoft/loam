use glam::{Vec3, Vec4};

use crate::consts::HYPERSLICE_MIN_THICKNESS;
pub(crate) use loam_shape::projected_edges::{
    push_blended_edge, retain_in_radius_triangles, sample_in_radius, stereographic_clip_radius,
    stereographic_view_point,
};

pub(crate) const PERSPECTIVE_SCALE_DENOM_EPSILON: f32 = 1e-4;

pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FRACTION: f32 = 0.75;

pub(crate) const STEREOGRAPHIC_VIEW_RADIUS_FLOOR: f32 = 2.5;

pub(crate) const STEREOGRAPHIC_RADIUS_MAX: f32 = 10.0;

#[cfg(test)]
pub(crate) const STEREOGRAPHIC_VIEW_RADIUS: f32 = 6.0;

pub(crate) fn stereographic_view_radius(camera_distance: f32) -> f32 {
    (camera_distance * STEREOGRAPHIC_VIEW_RADIUS_FRACTION)
        .clamp(STEREOGRAPHIC_VIEW_RADIUS_FLOOR, STEREOGRAPHIC_RADIUS_MAX)
}

pub(crate) fn perspective_scale_at_w(
    w_slice: f32,
    projection: &loam_math::Projection<4>,
) -> Option<f32> {
    match *projection {
        loam_math::Projection::Identity | loam_math::Projection::Orthographic { .. } => Some(1.0),
        loam_math::Projection::Perspective4D { focal_distance } => {
            Some(focal_distance / (focal_distance - w_slice).max(PERSPECTIVE_SCALE_DENOM_EPSILON))
        }
        loam_math::Projection::Schlegel { .. } | loam_math::Projection::Stereographic { .. } => {
            None
        }
    }
}

pub(crate) fn local_r3_to_world(p: [f32; 3], section_scale: f32, body_pos_r3: Vec3) -> [f32; 3] {
    let scaled = Vec3::from_array(p) * section_scale;
    (scaled + body_pos_r3).to_array()
}

pub(crate) fn cap_vertex_projected_and_world(
    p_r3: [f32; 3],
    w_slice: f32,
    section_scale: Option<f32>,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> (Vec3, [f32; 3]) {
    match section_scale {
        Some(scale) => {
            let projected = Vec3::from_array(p_r3) * scale;
            (projected, local_r3_to_world(p_r3, scale, body_pos_r3))
        }
        None => {
            let p4 = Vec4::new(p_r3[0], p_r3[1], p_r3[2], w_slice);
            let projected = stereographic_view_point(p4, projection);
            (projected, (projected + body_pos_r3).to_array())
        }
    }
}

#[cfg(test)]
pub(crate) fn project_to_world(
    p: Vec4,
    projection: &loam_math::Projection<4>,
    body_pos_r3: Vec3,
) -> Vec3 {
    <loam_math::EuclideanR4 as loam_math::RasterizableSpace<4>>::project_point(p, projection)
        + body_pos_r3
}

pub(crate) fn slab_overlaps(
    interval_min: f32,
    interval_max: f32,
    w_slice: f32,
    thickness: f32,
) -> bool {
    let half = thickness.max(HYPERSLICE_MIN_THICKNESS) * 0.5;
    let slab_min = w_slice - half;
    let slab_max = w_slice + half;
    interval_min <= slab_max && interval_max >= slab_min
}

pub(crate) fn cell_w_range(cell: &[u32], local_vertices: &[Vec4]) -> (f32, f32) {
    cell.iter()
        .map(|&i| local_vertices[i as usize].w)
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), w| {
            (lo.min(w), hi.max(w))
        })
}
