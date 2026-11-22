//! Upper-hemisphere chart `(p, sqrt(1 - |p|²))` with Vec3 points.
//! Crossing the equator loses the hemisphere sign; use SphericalS3Embedded for full coverage.

use std::borrow::Cow;

use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::space::{IsometryGroup, Space, WgslAccuracy, WgslSpace};

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

    pub fn left_translation(point: Vec4) -> Self {
        let (x, y, z, w) = (point.x, point.y, point.z, point.w);
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(w, z, -y, -x),
                Vec4::new(-z, w, x, -y),
                Vec4::new(y, -x, w, -z),
                point,
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SphericalS3;

impl Space for SphericalS3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Frame = Iso4;

    fn frame_at(&self, at: Vec3) -> Iso4 {
        Iso4::from_translation(at)
    }

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

    fn chart_envelope(&self) -> f32 {
        std::f32::consts::FRAC_PI_2
    }

    fn valid_point(&self, p: Vec3) -> bool {
        p.is_finite() && p.length_squared() <= SPHERE_R2_MAX
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

    fn wgsl_accuracy(&self) -> WgslAccuracy {
        WgslAccuracy::Exact
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

fn loam_s3_transport(p_from: vec3<f32>, p_to: vec3<f32>, v: vec3<f32>) -> vec3<f32> {
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

struct LoamGeodesicStep { p: vec3<f32>, v: vec3<f32> }

fn loam_geodesic_step(p: vec3<f32>, v: vec3<f32>, s: f32) -> LoamGeodesicStep {
    let next = loam_exp(p, v * s);
    return LoamGeodesicStep(next, loam_s3_transport(p, next, v));
}
"#;
