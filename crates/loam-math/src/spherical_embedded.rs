use glam::Vec4;

use crate::rasterizable::{Projection, RasterizableSpace};
use crate::space::{IsometryGroup, Space};
use crate::spherical::Iso4;
use crate::EuclideanR4;

const GEODESIC_DIRECTION_MIN: f32 = 1e-7;

const TRANSPORT_DENOM_MIN: f32 = 1e-7;

const UNIT_POINT_TOLERANCE: f32 = 1e-4;

/// S³ with unit Vec4 points and ambient tangents; methods assume unit points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3Embedded;

impl Space for SphericalS3Embedded {
    type Point = Vec4;

    type Vector = Vec4;

    type Frame = Iso4;

    fn frame_at(&self, at: Vec4) -> Iso4 {
        Iso4::left_translation(at)
    }

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        let half_chord = (a - b).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        let v_tan = v - v.dot(at) * at;
        let theta = v_tan.length();
        if theta < GEODESIC_DIRECTION_MIN {
            return at;
        }
        at * theta.cos() + v_tan * (theta.sin() / theta)
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        let d = self.distance(from, to);

        let dot = from.dot(to).clamp(-1.0, 1.0);
        let perp = to - dot * from;
        let n = perp.length();
        if n < GEODESIC_DIRECTION_MIN {
            return Vec4::ZERO;
        }
        perp * (d / n)
    }

    fn parallel_transport(&self, from: Vec4, to: Vec4, v: Vec4) -> Vec4 {
        // do Carmo, Riemannian Geometry, ch. 2.
        let sum = from + to;
        let denom = (sum.length_squared() * 0.5).max(TRANSPORT_DENOM_MIN);
        v - (v.dot(to) / denom) * sum
    }

    fn chart_envelope(&self) -> f32 {
        std::f32::consts::PI
    }

    fn valid_point(&self, p: Vec4) -> bool {
        p.is_finite() && (p.length_squared() - 1.0).abs() <= UNIT_POINT_TOLERANCE
    }
}

impl IsometryGroup for SphericalS3Embedded {
    type Iso = Iso4;

    fn iso_identity(&self) -> Iso4 {
        Iso4::IDENTITY
    }

    fn iso_compose(&self, a: Iso4, b: Iso4) -> Iso4 {
        Iso4 {
            matrix: a.matrix * b.matrix,
        }
    }

    fn iso_inverse(&self, a: Iso4) -> Iso4 {
        Iso4 {
            matrix: a.matrix.transpose(),
        }
    }

    fn iso_apply(&self, iso: Iso4, p: Vec4) -> Vec4 {
        (iso.matrix * p).normalize()
    }

    fn iso_transport(&self, iso: Iso4, _at: Vec4, v: Vec4) -> Vec4 {
        iso.matrix * v
    }
}

impl RasterizableSpace<4> for SphericalS3Embedded {
    fn point_to_array(p: Vec4) -> [f32; 4] {
        p.to_array()
    }

    fn array_to_point(arr: [f32; 4]) -> Vec4 {
        Vec4::from_array(arr).normalize()
    }

    fn project_point(point: Vec4, projection: &Projection<4>) -> Option<glam::Vec3> {
        match projection {
            Projection::Stereographic { pole } => {
                Some(crate::rasterizable::stereographic_to_r3(point, *pole))
            }

            Projection::Identity
            | Projection::Orthographic { .. }
            | Projection::Perspective4D { .. } => {
                <EuclideanR4 as RasterizableSpace<4>>::project_point(point, projection)
            }
        }
    }

    fn tessellate_segment(p0: Vec4, p1: Vec4, samples: usize, mut emit: impl FnMut(Vec4)) {
        // Absil, Mahony and Sepulchre, Optimization Algorithms on Matrix Manifolds, 2008, §3.6.
        // Shoemake, Animating Rotation with Quaternion Curves, 1985.
        let dot = p0.dot(p1).clamp(-1.0, 1.0);
        let half_chord = (p0 - p1).length() * 0.5;
        let omega = 2.0 * half_chord.clamp(0.0, 1.0).asin();

        let perp = p1 - dot * p0;
        let n = perp.length();
        let dir = if n > GEODESIC_DIRECTION_MIN {
            perp / n
        } else {
            deterministic_perp(p0)
        };
        emit(p0);
        for i in 1..samples {
            let ang = i as f32 / samples as f32 * omega;
            emit(ang.cos() * p0 + ang.sin() * dir);
        }
        emit(p1);
    }
}

// do Carmo, Differential Geometry of Curves and Surfaces, 1976, §1.4.
fn deterministic_perp(p0: Vec4) -> Vec4 {
    let a = p0.abs();

    let mut min_idx = 0usize;
    let mut min_v = a.x;
    if a.y < min_v {
        min_v = a.y;
        min_idx = 1;
    }
    if a.z < min_v {
        min_v = a.z;
        min_idx = 2;
    }
    if a.w < min_v {
        min_idx = 3;
    }
    let axis = match min_idx {
        0 => Vec4::X,
        1 => Vec4::Y,
        2 => Vec4::Z,
        _ => Vec4::W,
    };
    (axis - axis.dot(p0) * p0).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use std::f32::consts::PI;

    fn tessellate(p0: Vec4, p1: Vec4, samples: usize) -> Vec<Vec4> {
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, samples, |p| {
            out.push(p)
        });
        out
    }

    #[test]
    fn tessellation_walks_the_great_circle_and_stays_on_the_sphere_for_degenerate_arcs() {
        let s = SphericalS3Embedded;

        let out = tessellate(Vec4::X, Vec4::Y, 4);
        assert_eq!(out.len(), 5);
        assert_relative_eq!(out[0].x, 1.0, epsilon = 1e-6);
        assert_relative_eq!(out[4].y, 1.0, epsilon = 1e-6);

        let mid = tessellate(Vec4::X, Vec4::W, 2)[1];
        assert_relative_eq!(
            s.distance(mid, Vec4::X),
            s.distance(mid, Vec4::W),
            epsilon = 1e-6
        );
        let c = (PI / 4.0).cos();
        assert_relative_eq!(mid.x, c, epsilon = 1e-5);
        assert_relative_eq!(mid.w, c, epsilon = 1e-5);

        let p0 = Vec4::new(0.2, 0.1, -0.3, 0.9).normalize();
        let p1 = Vec4::new(-0.1, 0.4, 0.2, 0.8).normalize();
        let total = s.distance(p0, p1);
        for samples in [2usize, 3, 8, 17] {
            let out = tessellate(p0, p1, samples);
            let arc_sum: f32 = out.windows(2).map(|w| s.distance(w[0], w[1])).sum();
            assert_relative_eq!(arc_sum, total, epsilon = 1e-6);
        }

        let out = tessellate(Vec4::X, -Vec4::X, 16);
        assert_eq!(out.len(), 17);
        for p in &out {
            assert!(
                p.is_finite(),
                "antipodal slerp sample must be finite: {p:?}"
            );
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
        }

        for delta in [1e-3_f32, 1e-5, GEODESIC_DIRECTION_MIN * 2.0] {
            let omega = PI - delta;
            let p1 = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            let out = tessellate(Vec4::X, p1, 16);
            for p in &out[1..out.len() - 1] {
                assert!(p.is_finite(), "near-antipode sample must be finite: {p:?}");
                assert!(
                    (p.length() - 1.0).abs() < 1e-4,
                    "near-antipode (omega = PI - {delta:e}) sample off-sphere: |p| = {}",
                    p.length()
                );
            }
        }

        let p0 = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let nudge = GEODESIC_DIRECTION_MIN * 0.01;
        let p1 = (p0 + Vec4::new(nudge, 0.0, -nudge, 0.0)).normalize();
        for p in tessellate(p0, p1, 8) {
            assert!(p.is_finite(), "coincident sample must be finite: {p:?}");
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
            let drift = SphericalS3Embedded.distance(p, p0);
            assert!(
                drift < 1e-4,
                "coincident-arc sample should stay at p0, dist {drift}"
            );
        }
    }
}
