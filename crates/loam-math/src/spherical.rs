//! Upper-hemisphere chart `(p, sqrt(1 - |p|²))` with Vec3 points.
//! Crossing the equator loses the hemisphere sign; use SphericalS3Embedded for full coverage.

use std::borrow::Cow;

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::space::{IsometryGroup, Space, WgslSpace};

const SPHERE_R2_MAX: f32 = 1.0 - 1e-6;

const EXP_TANGENT_MIN_SQ: f32 = 1e-14;

const LOG_PERP_MIN: f32 = 1e-7;

const ISO_TRANSLATION_MIN_ARC: f32 = 1e-7;

fn clamp_to_hemisphere(p: Vec3) -> Vec3 {
    let r2 = p.length_squared();
    if r2 <= SPHERE_R2_MAX {
        p
    } else {
        #[cfg(debug_assertions)]
        tracing::warn!("SphericalS3: point outside upper hemisphere clamped (|p|²={r2:.4})");
        p * (SPHERE_R2_MAX.sqrt() / r2.sqrt())
    }
}

fn to_sphere(p: Vec3) -> Vec4 {
    let r2 = p.length_squared().min(SPHERE_R2_MAX);
    Vec4::new(p.x, p.y, p.z, (1.0 - r2).sqrt())
}

fn from_sphere(q: Vec4) -> Vec3 {
    #[cfg(debug_assertions)]
    if q.w < 0.0 {
        tracing::warn!(
            "SphericalS3: iso_apply moved point to lower hemisphere (w={:.4}); \
             result will be out-of-domain",
            q.w
        );
    }
    q.truncate()
}

/// SO(4) acting on the ambient embedding of S³.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso4 {
    /// Column-major orthogonal matrix; inverse uses the transpose without validation.
    pub matrix: Mat4,
}

impl Iso4 {
    pub const IDENTITY: Self = Self {
        matrix: Mat4::IDENTITY,
    };

    /// Fixes the north pole.
    pub fn from_rotation(rotation: Quat) -> Self {
        let r = Mat3::from_quat(rotation);
        Self {
            matrix: Mat4::from_cols(
                r.col(0).extend(0.0),
                r.col(1).extend(0.0),
                r.col(2).extend(0.0),
                Vec4::W,
            ),
        }
    }

    /// Maps the north pole to `target`, clamped to the upper-hemisphere chart.
    pub fn from_translation(target: Vec3) -> Self {
        let qt = to_sphere(clamp_to_hemisphere(target));
        let c = qt.w;
        let s = qt.truncate().length();
        if s < ISO_TRANSLATION_MIN_ARC {
            return Self::IDENTITY;
        }
        let n = qt.truncate() / s;
        let k = c - 1.0;

        Self {
            matrix: Mat4::from_cols(
                Vec4::new(1.0 + k * n.x * n.x, k * n.x * n.y, k * n.x * n.z, -s * n.x),
                Vec4::new(k * n.y * n.x, 1.0 + k * n.y * n.y, k * n.y * n.z, -s * n.y),
                Vec4::new(k * n.z * n.x, k * n.z * n.y, 1.0 + k * n.z * n.z, -s * n.z),
                Vec4::new(s * n.x, s * n.y, s * n.z, c),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3;

impl Space for SphericalS3 {
    type Point = Vec3;
    type Vector = Vec3;

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        let qa = to_sphere(clamp_to_hemisphere(a));
        let qb = to_sphere(clamp_to_hemisphere(b));

        let half_chord = (qa - qb).length() * 0.5;
        2.0 * half_chord.clamp(0.0, 1.0).asin()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        let at = clamp_to_hemisphere(at);
        if v.length_squared() < EXP_TANGENT_MIN_SQ {
            return at;
        }
        let q = to_sphere(at);

        let vw = -v.dot(at) / q.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        let mag = v4.length();
        let result4 = (q * mag.cos() + v4 * (mag.sin() / mag)).normalize();
        clamp_to_hemisphere(result4.truncate())
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        let qf = to_sphere(clamp_to_hemisphere(from));
        let qt = to_sphere(clamp_to_hemisphere(to));
        let d_dot = qf.dot(qt).clamp(-1.0, 1.0);
        let perp4 = qt - d_dot * qf;
        let n = perp4.length();
        if n < LOG_PERP_MIN {
            return Vec3::ZERO;
        }
        let half_chord = (qt - qf).length() * 0.5;
        let d = 2.0 * half_chord.clamp(0.0, 1.0).asin();

        perp4.truncate() * (d / n)
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        let from = clamp_to_hemisphere(from);
        let to = clamp_to_hemisphere(to);
        let qf = to_sphere(from);
        let qt = to_sphere(to);
        let vw = -v.dot(from) / qf.w;
        let v4 = Vec4::new(v.x, v.y, v.z, vw);
        // do Carmo, Riemannian Geometry, ch. 2.
        let sum = qf + qt;
        let denom = sum.length_squared() * 0.5;
        let v4_transported = v4 - v4.dot(qt) / denom * sum;
        v4_transported.truncate()
    }
}

impl IsometryGroup for SphericalS3 {
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

    fn iso_apply(&self, iso: Iso4, p: Vec3) -> Vec3 {
        from_sphere(iso.matrix * to_sphere(clamp_to_hemisphere(p)))
    }

    fn iso_transport(&self, iso: Iso4, at: Vec3, v: Vec3) -> Vec3 {
        let target = self.exp(at, v);
        let m_at = self.iso_apply(iso, at);
        let m_target = self.iso_apply(iso, target);
        self.log(m_at, m_target)
    }
}

impl WgslSpace for SphericalS3 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Owned(format!(
            r#"

const LOAM_MAX_ARC: f32 = 1.5;
const LOAM_S3_R2_MAX: f32 = {SPHERE_R2_MAX};
const LOAM_S3_EXP_TANGENT_MIN_SQ: f32 = {EXP_TANGENT_MIN_SQ:e};
const LOAM_S3_LOG_PERP_MIN: f32 = {LOG_PERP_MIN:e};
{WGSL_FUNCTIONS}"#
        ))
    }
}

const WGSL_FUNCTIONS: &str = r#"
fn loam_s3_clamp(p: vec3<f32>) -> vec3<f32> {
    let r2 = dot(p, p);
    if (r2 <= LOAM_S3_R2_MAX) { return p; }
    return p * (sqrt(LOAM_S3_R2_MAX) / sqrt(r2));
}

fn loam_s3_lift(p: vec3<f32>) -> vec4<f32> {
    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return vec4<f32>(p.x, p.y, p.z, sqrt(1.0 - r2));
}

fn loam_origin_distance(p: vec3<f32>) -> f32 {

    let r2 = min(dot(p, p), LOAM_S3_R2_MAX);
    return asin(sqrt(r2));
}

fn loam_distance(a: vec3<f32>, b: vec3<f32>) -> f32 {
    let qa = loam_s3_lift(loam_s3_clamp(a));
    let qb = loam_s3_lift(loam_s3_clamp(b));
    let half_chord = length(qa - qb) * 0.5;
    return 2.0 * asin(clamp(half_chord, 0.0, 1.0));
}

fn loam_exp(at: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let p = loam_s3_clamp(at);
    let n2 = dot(v, v);
    if (n2 < LOAM_S3_EXP_TANGENT_MIN_SQ) { return p; }
    let q = loam_s3_lift(p);
    let vw = -dot(v, p) / q.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);
    let mag = length(v4);
    let result4 = normalize(q * cos(mag) + v4 * (sin(mag) / mag));
    return loam_s3_clamp(result4.xyz);
}

fn loam_log(p_from: vec3<f32>, p_to: vec3<f32>) -> vec3<f32> {
    let qf = loam_s3_lift(loam_s3_clamp(p_from));
    let qt = loam_s3_lift(loam_s3_clamp(p_to));
    let d_dot = clamp(dot(qf, qt), -1.0, 1.0);
    let perp4 = qt - d_dot * qf;
    let n = length(perp4);
    if (n < LOAM_S3_LOG_PERP_MIN) { return vec3<f32>(0.0, 0.0, 0.0); }
    let half_chord = length(qt - qf) * 0.5;
    let d = 2.0 * asin(clamp(half_chord, 0.0, 1.0));
    return perp4.xyz * (d / n);
}

fn loam_parallel_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
    let pf = loam_s3_clamp(p_from);
    let pt = loam_s3_clamp(p_to);
    let qf = loam_s3_lift(pf);
    let qt = loam_s3_lift(pt);
    let vw = -dot(v, pf) / qf.w;
    let v4 = vec4<f32>(v.x, v.y, v.z, vw);

    let sum = qf + qt;
    let denom = dot(sum, sum) * 0.5;
    let v4t = v4 - (dot(v4, qt) / denom) * sum;
    return v4t.xyz;
}
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn s3() -> SphericalS3 {
        SphericalS3
    }

    #[test]
    fn to_sphere_from_sphere_round_trip() {
        let p = Vec3::new(0.2, -0.3, 0.1);
        let q = to_sphere(p);
        assert_relative_eq!(q.length(), 1.0, epsilon = 1e-6);
        assert_relative_eq!(q.w, (1.0 - p.length_squared()).sqrt(), epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).x, p.x, epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).y, p.y, epsilon = 1e-6);
        assert_relative_eq!(from_sphere(q).z, p.z, epsilon = 1e-6);
    }

    #[test]
    fn distance_at_origin_matches_arc_length() {
        let s = s3();

        let r = 0.4;
        let p = Vec3::new(r, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), r.asin(), epsilon = 1e-5);
    }

    #[test]
    fn exp_tiny_vector_clamps_out_of_domain_basepoint() {
        let s = s3();
        let at = Vec3::new(2.0, 0.0, 0.0);
        let tiny = Vec3::new(EXP_TANGENT_MIN_SQ.sqrt() * 0.1, 0.0, 0.0);
        let got = s.exp(at, tiny);
        let want = clamp_to_hemisphere(at);
        assert_relative_eq!(got.x, want.x, epsilon = 1e-6);
        assert_relative_eq!(got.y, want.y, epsilon = 1e-6);
        assert_relative_eq!(got.z, want.z, epsilon = 1e-6);
    }

    #[test]
    fn iso_translation_moves_origin_to_target() {
        let s = s3();
        let target = Vec3::new(0.2, -0.1, 0.15);
        let iso = Iso4::from_translation(target);
        let moved = s.iso_apply(iso, Vec3::ZERO);
        assert_relative_eq!(moved.x, target.x, epsilon = 1e-5);
        assert_relative_eq!(moved.y, target.y, epsilon = 1e-5);
        assert_relative_eq!(moved.z, target.z, epsilon = 1e-5);
    }

    #[test]
    fn parallel_transport_preserves_norm_near_antipode() {
        let s = s3();
        let lifted_norm = |p: Vec3, v: Vec3| {
            let vw = -v.dot(p) / to_sphere(p).w;
            Vec4::new(v.x, v.y, v.z, vw).length()
        };
        for w in [5e-3_f32, 2e-3, 1.2e-3] {
            let b = w;
            let a = (1.0 - w * w - b * b).sqrt();
            let from = Vec3::new(a, b, 0.0);
            let to = Vec3::new(-a, b, 0.0);

            let v = Vec3::X;
            let vt = s.parallel_transport(from, to, v);
            let norm_from = lifted_norm(from, v);
            assert_relative_eq!(lifted_norm(to, vt), norm_from, max_relative = 1e-5);
            assert!(
                (vt - v).length() > 0.5 * norm_from,
                "transport should rotate an in-plane vector, got {vt:?}"
            );
        }
    }

    #[test]
    fn small_scale_distance_matches_euclidean() {
        let s = s3();
        let eps = 1e-3;
        let p = Vec3::new(eps, 0.0, 0.0);
        assert_relative_eq!(s.distance(Vec3::ZERO, p), eps, epsilon = 1e-6);
    }

    #[test]
    fn angle_excess_in_small_triangle_scales_with_area() {
        let s = s3();
        let l = 0.05_f32;
        let a = Vec3::ZERO;
        let b = s.exp(a, Vec3::new(l, 0.0, 0.0));
        let c = s.exp(a, Vec3::new(l * 0.5, l * 3.0_f32.sqrt() * 0.5, 0.0));

        let angle_at = |p: Vec3, q: Vec3, r: Vec3| -> f32 {
            let u3 = s.log(p, q);
            let w3 = s.log(p, r);

            let qp = to_sphere(p);
            let u4 = Vec4::new(u3.x, u3.y, u3.z, -u3.dot(p) / qp.w);
            let w4 = Vec4::new(w3.x, w3.y, w3.z, -w3.dot(p) / qp.w);
            (u4.dot(w4) / (u4.length() * w4.length()))
                .clamp(-1.0, 1.0)
                .acos()
        };

        let alpha = angle_at(a, b, c);
        let beta = angle_at(b, a, c);
        let gamma = angle_at(c, a, b);
        let excess = (alpha + beta + gamma) - std::f32::consts::PI;
        let expected_area = 3.0_f32.sqrt() / 4.0 * l * l;

        assert!(
            excess > 0.0,
            "spherical triangle should have positive angle excess, got {excess}"
        );
        assert_relative_eq!(excess, expected_area, epsilon = 5e-4);
    }

    const PARITY_DIR: Vec3 = Vec3::new(0.6, -0.48, 0.64);

    fn parity_points() -> [Vec3; 11] {
        let shell = SPHERE_R2_MAX.sqrt();
        [
            Vec3::ZERO,
            Vec3::new(LOG_PERP_MIN * 0.5, 0.0, 0.0),
            Vec3::new(LOG_PERP_MIN * 1.5, 0.0, 0.0),
            Vec3::new(LOG_PERP_MIN * 100.0, 0.0, 0.0),
            Vec3::new(0.2, -0.3, 0.1),
            Vec3::new(-0.45, 0.5, 0.35),
            PARITY_DIR * 0.95,
            PARITY_DIR * shell,
            -PARITY_DIR * shell,
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(2.0, -1.0, 0.5),
        ]
    }

    #[test]
    fn lift_floors_w_at_the_shell_even_where_the_clamp_overshoots_it() {
        let floor = (1.0 - SPHERE_R2_MAX).sqrt();
        let mut worst_w = f32::INFINITY;
        for i in 0..6 {
            for j in 0..6 {
                for k in 0..6 {
                    let dir = Vec3::new(i as f32 - 2.5, j as f32 - 2.5, k as f32 - 2.5).normalize();
                    for scale in [1.0 + 1e-6, 1.01, 2.0, 1e3, 1e18] {
                        let raw = dir * scale;
                        let clamped = clamp_to_hemisphere(raw);
                        worst_w = worst_w.min(to_sphere(raw).w).min(to_sphere(clamped).w);
                    }
                }
            }
        }
        assert!(
            worst_w >= floor,
            "lift w reached {worst_w:e}, under the shell floor {floor:e}"
        );

        assert!(to_sphere(Vec3::splat(f32::NAN)).w >= floor);
        assert!(to_sphere(Vec3::splat(f32::INFINITY)).w >= floor);
    }

    #[test]
    fn transport_denominator_is_bounded_below_by_the_saturation_shell() {
        let chart_min = 2.0 * (1.0 - SPHERE_R2_MAX);
        let mut worst = f32::INFINITY;
        for from in parity_points() {
            for to in parity_points() {
                let sum = to_sphere(clamp_to_hemisphere(from)) + to_sphere(clamp_to_hemisphere(to));
                worst = worst.min(sum.length_squared() * 0.5);
            }
        }
        assert!(
            worst >= chart_min,
            "denominator reached {worst:e}, under the shell bound {chart_min:e}"
        );

        assert!(
            worst <= chart_min * 1.5,
            "closest approach {worst:e} is not the chart minimum {chart_min:e}"
        );
    }

    #[test]
    fn translation_guard_separates_degenerate_targets_from_representable_ones() {
        let s = s3();
        let below = Vec3::new(ISO_TRANSLATION_MIN_ARC * 0.5, 0.0, 0.0);
        assert_eq!(Iso4::from_translation(below).matrix, Mat4::IDENTITY);

        let above = Vec3::new(ISO_TRANSLATION_MIN_ARC * 1.5, 0.0, 0.0);
        let moved = s.iso_apply(Iso4::from_translation(above), Vec3::ZERO);
        assert_relative_eq!(moved.x, above.x, max_relative = 1e-5);
        assert_eq!(moved.y, 0.0);
        assert_eq!(moved.z, 0.0);
    }
}
