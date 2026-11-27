use glam::{Vec3, Vec4};

use crate::space::Space;
use crate::{EuclideanR3, EuclideanR4};

const PROJECTION_DENOM_EPSILON: f32 = 1e-4;

/// Floor on `1 - dot(p, pole)` at the projection singularity.
pub const STEREOGRAPHIC_POLE_EPSILON: f32 = 1e-4;

// do Carmo, Differential Geometry of Curves and Surfaces, 1976, sec. 1.4.
fn perp_frame(n: Vec4) -> (Vec4, Vec4, Vec4) {
    let ax = n.x.abs();
    let ay = n.y.abs();
    let az = n.z.abs();
    let aw = n.w.abs();
    let drop = ax.max(ay).max(az).max(aw);
    let mut seeds = [Vec4::X, Vec4::Y, Vec4::Z, Vec4::W];
    let drop_idx = if drop == ax {
        0
    } else if drop == ay {
        1
    } else if drop == az {
        2
    } else {
        3
    };
    seeds[drop_idx] = Vec4::ZERO;

    let mut basis = [Vec4::ZERO; 3];
    let mut count = 0usize;
    for s in seeds {
        if s == Vec4::ZERO {
            continue;
        }
        let mut v = s - s.dot(n) * n;
        for b in basis.iter().take(count) {
            v -= v.dot(*b) * *b;
        }
        basis[count] = v.normalize();
        count += 1;
    }
    (basis[0], basis[1], basis[2])
}

// Wikipedia, Stereographic projection.
pub(crate) fn stereographic_to_r3(p: Vec4, pole: Vec4) -> Vec3 {
    let dot = p.dot(pole).clamp(-1.0, 1.0);
    let denom = (1.0 - dot).max(STEREOGRAPHIC_POLE_EPSILON);
    if pole == Vec4::W {
        return Vec3::new(p.x, p.y, p.z) / denom;
    }
    let perp = p - dot * pole;
    let scaled = perp / denom;
    let (e1, e2, e3) = perp_frame(pole);
    Vec3::new(scaled.dot(e1), scaled.dot(e2), scaled.dot(e3))
}

/// Unsupported dimension/variant combinations return `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Projection<const N: usize> {
    /// Keeps the first three components, padding with zero if needed.
    #[default]
    Identity,

    /// Drops the indexed axis; an out-of-range index returns `None`.
    Orthographic { drop_axis: usize },

    /// Pinhole at `(0, 0, 0, focal_distance)`; a vertex at or beyond the eye in w is refused with `None`.
    Perspective4D { focal_distance: f32 },

    // Wikipedia, Stereographic projection.
    /// Projects unit S3 from a unit pole into its perpendicular 3-flat.
    Stereographic {
        /// Unit pole; `Vec4::W` gives the usual xyz readout.
        pole: Vec4,
    },
}

pub trait RasterizableSpace<const N: usize>: Space {
    fn array_to_point(arr: [f32; N]) -> Self::Point;

    fn project_point(point: Self::Point, projection: &Projection<N>) -> Option<Vec3>;

    /// Emits both endpoints and `samples - 1` interior points when `samples > 1`.
    fn tessellate_segment(
        p0: Self::Point,
        p1: Self::Point,
        samples: usize,
        emit: impl FnMut(Self::Point),
    );
}

impl RasterizableSpace<3> for EuclideanR3 {
    fn array_to_point(arr: [f32; 3]) -> Vec3 {
        Vec3::from_array(arr)
    }

    fn project_point(point: Vec3, projection: &Projection<3>) -> Option<Vec3> {
        match projection {
            Projection::Identity => Some(point),
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Some(Vec3::new(point.y, point.z, 0.0)),
                1 => Some(Vec3::new(point.x, point.z, 0.0)),
                2 => Some(Vec3::new(point.x, point.y, 0.0)),
                _ => None,
            },
            Projection::Perspective4D { .. } | Projection::Stereographic { .. } => None,
        }
    }

    fn tessellate_segment(p0: Vec3, p1: Vec3, samples: usize, mut emit: impl FnMut(Vec3)) {
        emit(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            emit(p0.lerp(p1, t));
        }
        emit(p1);
    }
}

impl RasterizableSpace<4> for EuclideanR4 {
    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        Vec4::from_array(arr)
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> Option<Vec3> {
        match projection {
            Projection::Identity => Some(Vec3::new(point.x, point.y, point.z)),
            Projection::Orthographic { drop_axis } => match *drop_axis {
                0 => Some(Vec3::new(point.y, point.z, point.w)),
                1 => Some(Vec3::new(point.x, point.z, point.w)),
                2 => Some(Vec3::new(point.x, point.y, point.w)),
                3 => Some(Vec3::new(point.x, point.y, point.z)),
                _ => None,
            },
            Projection::Perspective4D { focal_distance } => {
                let denom = focal_distance - point.w;
                if denom < PROJECTION_DENOM_EPSILON {
                    return None;
                }
                let scale = focal_distance / denom;
                Some(Vec3::new(point.x, point.y, point.z) * scale)
            }
            Projection::Stereographic { pole } => {
                Some(stereographic_to_r3(point.normalize(), *pole))
            }
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, mut emit: impl FnMut(Vec4)) {
        emit(p0);
        for i in 1..samples {
            let t = i as f32 / samples as f32;
            emit(p0.lerp(p1, t));
        }
        emit(p1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn project4(p: Vec4, projection: &Projection<4>) -> Vec3 {
        <EuclideanR4 as RasterizableSpace<4>>::project_point(p, projection)
            .expect("projection should accept this vertex")
    }

    #[test]
    fn flat_tessellation() {
        let (p0, p1) = (Vec3::ZERO, Vec3::new(2.0, 4.0, -6.0));
        let mut out = Vec::new();
        <EuclideanR3 as RasterizableSpace<3>>::tessellate_segment(p0, p1, 1, |p| out.push(p));
        assert_eq!(out, [p0, p1]);

        let (p0, p1) = (Vec4::ZERO, Vec4::new(4.0, 8.0, 12.0, 16.0));
        let mut out = Vec::new();
        <EuclideanR4 as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, |p| out.push(p));
        assert_eq!(out, [p0, Vec4::new(2.0, 4.0, 6.0, 8.0), p1]);
    }

    #[test]
    fn perspective4d() {
        let focal = 2.0;
        let proj = Projection::Perspective4D {
            focal_distance: focal,
        };
        let near = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let far = Vec4::new(0.5, 0.5, 0.5, -0.5);
        let pn = project4(near, &proj);
        let pf = project4(far, &proj);

        let r_near = (pn.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        let r_far = (pf.length() / 0.5_f32.mul_add(3.0_f32.sqrt(), 0.0)).abs();
        assert!((r_near - 4.0 / 3.0).abs() < 1e-5, "near scale {r_near}");
        assert!((r_far - 4.0 / 5.0).abs() < 1e-5, "far scale {r_far}");
        assert!(pn.length() > pf.length(), "near={pn:?} far={pf:?}");

        let in_eye_plane = Vec4::new(0.1, 0.2, 0.3, 2.0);
        let got = <EuclideanR4 as RasterizableSpace<4>>::project_point(in_eye_plane, &proj);
        assert_eq!(got, None, "{proj:?} placed {in_eye_plane:?}");
    }

    // Wikipedia, Stereographic projection.
    fn stereo_inverse(q: Vec3, pole: Vec4) -> Vec4 {
        let perp = if pole == Vec4::W {
            q.extend(0.0)
        } else {
            let (e1, e2, e3) = perp_frame(pole);
            q.x * e1 + q.y * e2 + q.z * e3
        };
        let s = perp.length_squared();
        let dot = (s - 1.0) / (s + 1.0);
        dot * pole + (1.0 - dot) * perp
    }

    #[test]
    fn stereographic() {
        let oblique = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let cases = [
            (Vec4::W, Vec4::new(0.2, 0.1, -0.3, 0.4).normalize()),
            (Vec4::W, Vec4::new(-0.5, 0.5, 0.5, -0.5)),
            (Vec4::W, Vec4::new(0.7, -0.2, 0.1, 0.1).normalize()),
            (oblique, Vec4::new(0.6, 0.5, -0.2, 0.0).normalize()),
            (oblique, Vec4::new(-0.3, 0.4, 0.5, -0.2).normalize()),
        ];
        for (pole, p) in cases {
            let proj = Projection::Stereographic { pole };
            let back = stereo_inverse(project4(p, &proj), pole);
            assert_relative_eq!(back.x, p.x, epsilon = 1e-5);
            assert_relative_eq!(back.y, p.y, epsilon = 1e-5);
            assert_relative_eq!(back.z, p.z, epsilon = 1e-5);
            assert_relative_eq!(back.w, p.w, epsilon = 1e-5);
        }

        let proj = Projection::Stereographic { pole: Vec4::W };
        for p in [
            Vec4::new(0.3, -0.1, 0.2, -0.5).normalize(),
            Vec4::new(-0.4, 0.6, 0.1, 0.3).normalize(),
            Vec4::new(0.0, 0.0, 0.0, -1.0),
        ] {
            let unit = project4(p, &proj);
            for k in [0.25_f32, 1.5, 3.0] {
                let scaled = project4(k * p, &proj);
                assert!(
                    scaled.abs_diff_eq(unit, 1e-5),
                    "scale {k}: {scaled:?} vs {unit:?}"
                );
            }
        }

        for pole in [Vec4::W, Vec4::new(0.5, 0.5, 0.5, 0.5), oblique] {
            let proj = Projection::Stereographic { pole };
            let origin = project4(-pole, &proj);
            assert!(
                origin.abs().max_element() <= 1e-6,
                "antipode of {pole:?} landed at {origin:?}"
            );

            let at_pole = project4(pole, &proj);
            assert!(
                at_pole.is_finite(),
                "pole input must clamp finite, got {at_pole:?}"
            );

            let (e1, _, _) = perp_frame(pole);
            let near = (pole * 0.9999 + e1 * 0.01).normalize();
            let near_img = project4(near, &proj);
            assert!(
                near_img.is_finite(),
                "near-pole input must stay finite, got {near_img:?}"
            );
        }
    }
}
