//! S³ uses unit ambient Vec4 points and tangent vectors perpendicular to their base points.

use glam::Vec4;

use crate::rasterizable::{Projection, RasterizableSpace};
use crate::space::{IsometryGroup, Space};
use crate::spherical::Iso4;
use crate::EuclideanR4;

const GEODESIC_DIRECTION_MIN: f32 = 1e-7;

const TRANSPORT_DENOM_MIN: f32 = 1e-7;

/// S³ with unit Vec4 points and ambient tangents; methods assume unit points.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3Embedded;

impl Space for SphericalS3Embedded {
    type Point = Vec4;

    type Vector = Vec4;

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

    fn project_point(point: Vec4, projection: &Projection<4>) -> glam::Vec3 {
        match projection {
            Projection::Stereographic { pole } => {
                crate::rasterizable::stereographic_to_r3(point, *pole)
            }

            Projection::Identity
            | Projection::Orthographic { .. }
            | Projection::Perspective4D { .. }
            | Projection::Schlegel { .. } => {
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

    fn s3() -> SphericalS3Embedded {
        SphericalS3Embedded
    }

    #[test]
    fn distance_orthonormal_is_quarter_circle() {
        let s = s3();
        assert_relative_eq!(s.distance(Vec4::X, Vec4::Y), PI / 2.0, epsilon = 1e-6);
        assert_relative_eq!(s.distance(Vec4::X, Vec4::W), PI / 2.0, epsilon = 1e-6);
    }

    #[test]
    fn distance_at_antipode_is_pi() {
        let s = s3();
        let a = Vec4::new(0.5, 0.5, 0.5, 0.5);
        assert_relative_eq!(s.distance(a, -a), PI, epsilon = 1e-5);
    }

    #[test]
    fn log_magnitude_is_distance_and_is_tangent() {
        let s = s3();
        let from = Vec4::new(0.3, 0.2, 0.1, 0.9).normalize();
        let to = Vec4::new(-0.2, 0.5, 0.0, 0.8).normalize();
        let v = s.log(from, to);
        assert_relative_eq!(v.length(), s.distance(from, to), epsilon = 1e-5);
        assert_relative_eq!(v.dot(from), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn exp_stays_on_sphere_with_non_tangent_input() {
        let s = s3();
        let at = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let v = Vec4::new(0.4, 0.2, 0.0, 0.7);
        let moved = s.exp(at, v);
        assert_relative_eq!(moved.length(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn parallel_transport_preserves_norm_and_tangency() {
        let s = s3();
        let from = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let to = Vec4::new(0.6, 0.0, 0.0, 0.8);
        let v = Vec4::new(0.5, 0.0, 0.0, 0.0);
        let vt = s.parallel_transport(from, to, v);
        assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-5);
        assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-5);
        assert!((vt - v).length() > 1e-3, "in-plane vector should rotate");
    }

    #[test]
    fn parallel_transport_preserves_norm_near_antipode() {
        let s = s3();
        let from = Vec4::X;
        for delta in [3e-3_f32, 2e-3, 1e-3] {
            let omega = PI - delta;
            let to = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();

            let v = Vec4::Y;
            let vt = s.parallel_transport(from, to, v);
            assert_relative_eq!(vt.length(), v.length(), epsilon = 1e-3);
            assert_relative_eq!(vt.dot(to), 0.0, epsilon = 1e-3);
        }
    }

    #[test]
    fn iso_transport_keeps_tangency_and_norm() {
        let s = s3();
        let iso = Iso4::from_translation(glam::Vec3::new(0.2, -0.1, 0.15));
        let at = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let v = Vec4::new(0.3, 0.2, 0.0, 0.0);
        let moved_at = s.iso_apply(iso, at);
        let moved_v = s.iso_transport(iso, at, v);
        assert_relative_eq!(moved_v.length(), v.length(), epsilon = 1e-5);
        assert_relative_eq!(moved_v.dot(moved_at), 0.0, epsilon = 1e-5);
    }

    #[test]
    fn slerp_endpoints_exact_and_count() {
        let p0 = Vec4::X;
        let p1 = Vec4::Y;
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 4, |p| {
            out.push(p)
        });
        assert_eq!(out.len(), 5);
        assert_relative_eq!(out[0].x, p0.x, epsilon = 1e-6);
        assert_relative_eq!(out[4].y, p1.y, epsilon = 1e-6);
    }

    #[test]
    fn slerp_samples_stay_on_sphere() {
        let p0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 8, |p| {
            out.push(p)
        });
        for p in &out {
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn slerp_midpoint_is_on_great_circle_not_chord() {
        let s = s3();
        let p0 = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let p1 = Vec4::new(0.0, 0.0, 0.0, 1.0);
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 2, |p| {
            out.push(p)
        });
        let mid = out[1];
        assert_relative_eq!(s.distance(mid, p0), s.distance(mid, p1), epsilon = 1e-6);

        let c = (PI / 4.0).cos();
        assert_relative_eq!(mid.x, c, epsilon = 1e-5);
        assert_relative_eq!(mid.w, c, epsilon = 1e-5);
        assert!(mid.x > 0.5, "slerp midpoint must bulge off the chord");
    }

    #[test]
    fn slerp_antipode_produces_finite_unit_samples() {
        let p0 = Vec4::X;
        let p1 = -Vec4::X;
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 16, |p| {
            out.push(p)
        });
        assert_eq!(out.len(), 17);
        for p in &out {
            assert!(
                p.is_finite(),
                "antipodal slerp sample must be finite: {p:?}"
            );
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn slerp_near_antipode_samples_stay_on_sphere() {
        let p0 = Vec4::X;

        for delta in [1e-3_f32, 1e-5, GEODESIC_DIRECTION_MIN * 2.0] {
            let omega = PI - delta;
            let p1 = Vec4::new(omega.cos(), omega.sin(), 0.0, 0.0).normalize();
            let mut out = Vec::new();
            <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 16, |p| {
                out.push(p)
            });
            for p in &out[1..out.len() - 1] {
                assert!(p.is_finite(), "near-antipode sample must be finite: {p:?}");
                assert!(
                    (p.length() - 1.0).abs() < 1e-4,
                    "near-antipode (omega = PI - {delta:e}) sample off-sphere: |p| = {}",
                    p.length()
                );
            }
        }
    }

    #[test]
    fn slerp_consecutive_arc_sum_equals_total() {
        let s = s3();
        let p0 = Vec4::new(0.2, 0.1, -0.3, 0.9).normalize();
        let p1 = Vec4::new(-0.1, 0.4, 0.2, 0.8).normalize();
        let total = s.distance(p0, p1);
        for samples in [2usize, 3, 8, 17] {
            let mut out = Vec::new();
            <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(
                p0,
                p1,
                samples,
                |p| out.push(p),
            );
            let arc_sum: f32 = out.windows(2).map(|w| s.distance(w[0], w[1])).sum();
            assert_relative_eq!(arc_sum, total, epsilon = 1e-6);
        }
    }

    #[test]
    fn slerp_near_coincident_falls_back_on_sphere() {
        let p0 = Vec4::new(0.1, -0.2, 0.3, 0.9).normalize();
        let nudge = GEODESIC_DIRECTION_MIN * 0.01;
        let p1 = (p0 + Vec4::new(nudge, 0.0, -nudge, 0.0)).normalize();
        let mut out = Vec::new();
        <SphericalS3Embedded as RasterizableSpace<4>>::tessellate_segment(p0, p1, 8, |p| {
            out.push(p)
        });
        for p in &out {
            assert!(p.is_finite(), "coincident sample must be finite: {p:?}");
            assert_relative_eq!(p.length(), 1.0, epsilon = 1e-6);
            assert!(
                s3().distance(*p, p0) < 1e-4,
                "coincident-arc sample should stay at p0, dist {}",
                s3().distance(*p, p0)
            );
        }
    }

    #[test]
    fn project_point_stereographic_is_conformal_map_not_drop_w() {
        let p = Vec4::new(0.5, 0.5, 0.5, 0.5);
        let proj = Projection::Stereographic { pole: Vec4::W };
        let got = <SphericalS3Embedded as RasterizableSpace<4>>::project_point(p, &proj);
        let want = glam::Vec3::new(p.x, p.y, p.z) / (1.0 - p.w);
        assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, want.z, epsilon = 1e-6);
        let drop_w = glam::Vec3::new(p.x, p.y, p.z);
        assert!(
            (got - drop_w).length() > 1e-3,
            "stereographic must scale by 1/(1-w), not pass through drop-w; got {got:?}"
        );
    }
}
