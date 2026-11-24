//! Poincaré-ball points satisfy `|p| < 1`; isometries act on the Lorentz embedding.

use glam::{Mat4, Quat, Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::space::{IsometryGroup, Space};

pub const POINCARE_R2_MAX: f32 = 1.0 - 1e-7;

/// H³ hits at least `H3_DEPTH_SEPARATION` apart stay ordered within this hyperbolic distance of an eye at most `H3_EYE_CHART_REACH` from the chart origin, and `valid_point` refuses points farther than this from the origin.
pub const H3_DEPTH_ENVELOPE: f32 = 6.0;
pub const H3_DEPTH_SEPARATION: f32 = 0.05;
pub const H3_EYE_CHART_REACH: f32 = 1.0;

/// False past the radius where the chart's f32 arithmetic is clamped.
pub fn in_poincare_ball(p: Vec3) -> bool {
    p.is_finite() && p.length_squared() <= POINCARE_R2_MAX
}

fn clamp_to_ball(p: Vec3) -> Vec3 {
    let r2 = p.length_squared();
    if r2 <= POINCARE_R2_MAX {
        return p;
    }
    #[cfg(debug_assertions)]
    tracing::warn!("HyperbolicH3: point outside Poincaré ball clamped (|p|²={r2:.4})");
    let mut q = p * (POINCARE_R2_MAX.sqrt() / r2.sqrt());
    while q.length_squared() > POINCARE_R2_MAX {
        q *= 1.0 - f32::EPSILON;
    }
    q
}

/// SO⁺(3,1) acting on `(x, y, z, w)`, with `w` time-like.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso3H {
    pub matrix: Mat4,
}

impl Iso3H {
    pub const IDENTITY: Self = Self {
        matrix: Mat4::IDENTITY,
    };

    /// Fixes the ball origin.
    pub fn from_rotation(rotation: Quat) -> Self {
        let r = glam::Mat3::from_quat(rotation);
        let c0 = r.col(0);
        let c1 = r.col(1);
        let c2 = r.col(2);
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(c0.x, c0.y, c0.z, 0.0),
                Vec4::new(c1.x, c1.y, c1.z, 0.0),
                Vec4::new(c2.x, c2.y, c2.z, 0.0),
                Vec4::new(0.0, 0.0, 0.0, 1.0),
            ),
        }
    }

    /// Maps the ball origin to `target`, with rapidity capped at the chart boundary.
    pub fn from_translation(target: Vec3) -> Self {
        let r2 = target.length_squared();
        if r2 < 1e-14 {
            return Self::IDENTITY;
        }
        let r = r2.sqrt();
        let dir = target / r;

        let rapidity = 2.0 * artanh(r.min(POINCARE_R2_MAX.sqrt()));
        let ch = rapidity.cosh();
        let sh = rapidity.sinh();
        let k = ch - 1.0;
        let (dx, dy, dz) = (dir.x, dir.y, dir.z);
        Self {
            matrix: Mat4::from_cols(
                Vec4::new(1.0 + k * dx * dx, k * dx * dy, k * dx * dz, sh * dx),
                Vec4::new(k * dy * dx, 1.0 + k * dy * dy, k * dy * dz, sh * dy),
                Vec4::new(k * dz * dx, k * dz * dy, 1.0 + k * dz * dz, sh * dz),
                Vec4::new(sh * dx, sh * dy, sh * dz, ch),
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HyperbolicH3;

impl Space for HyperbolicH3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Frame = Iso3H;

    fn frame_at(&self, at: Vec3) -> Iso3H {
        Iso3H::from_translation(at)
    }

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        let a = clamp_to_ball(a);
        let b = clamp_to_ball(b);
        let n = mobius_add(-a, b).length();
        2.0 * artanh(n.min(POINCARE_R2_MAX.sqrt()))
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        let n = v.length();
        if n < 1e-7 {
            return at;
        }
        let at = clamp_to_ball(at);
        let lambda = 2.0 / (1.0 - at.length_squared());
        let dir = v / n;
        let scale = (lambda * n * 0.5).tanh();
        mobius_add(at, scale * dir)
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        let from = clamp_to_ball(from);
        let to = clamp_to_ball(to);
        let d = mobius_add(-from, to);
        let n = d.length();
        if n < 1e-7 {
            return Vec3::ZERO;
        }
        let lambda = 2.0 / (1.0 - from.length_squared());
        let mag = (2.0 / lambda) * artanh(n.min(POINCARE_R2_MAX.sqrt()));
        mag * d / n
    }

    fn parallel_transport(&self, from: Vec3, to: Vec3, v: Vec3) -> Vec3 {
        let from = clamp_to_ball(from);
        let to = clamp_to_ball(to);
        let conformal = (1.0 - to.length_squared()) / (1.0 - from.length_squared());
        conformal * gyr_apply(to, -from, v)
    }

    fn chart_envelope(&self) -> f32 {
        H3_DEPTH_ENVELOPE
    }

    fn valid_point(&self, p: Vec3) -> bool {
        in_poincare_ball(p) && self.distance(Vec3::ZERO, p) <= H3_DEPTH_ENVELOPE
    }
}

impl IsometryGroup for HyperbolicH3 {
    type Iso = Iso3H;

    fn iso_identity(&self) -> Iso3H {
        Iso3H::IDENTITY
    }

    fn iso_compose(&self, a: Iso3H, b: Iso3H) -> Iso3H {
        Iso3H {
            matrix: a.matrix * b.matrix,
        }
    }

    fn iso_inverse(&self, a: Iso3H) -> Iso3H {
        let mt = a.matrix.transpose().to_cols_array_2d();
        let mut out = [[0.0f32; 4]; 4];
        for col in 0..4 {
            for row in 0..4 {
                let sign = if (row == 3) ^ (col == 3) { -1.0 } else { 1.0 };
                out[col][row] = sign * mt[col][row];
            }
        }
        Iso3H {
            matrix: Mat4::from_cols_array_2d(&out),
        }
    }

    fn iso_apply(&self, iso: Iso3H, p: Vec3) -> Vec3 {
        let h = poincare_to_hyperboloid(p);
        let h2 = iso.matrix * h;
        hyperboloid_to_poincare(h2)
    }

    fn iso_transport(&self, iso: Iso3H, at: Vec3, v: Vec3) -> Vec3 {
        let at = clamp_to_ball(at);
        let h = poincare_to_hyperboloid(at);
        let dh = poincare_to_hyperboloid_tangent(at, v);
        hyperboloid_to_poincare_tangent(iso.matrix * h, iso.matrix * dh)
    }
}

fn artanh(x: f32) -> f32 {
    0.5 * ((1.0 + x) / (1.0 - x)).ln()
}

// Ungar, From Möbius to Gyrogroups, 2008, §4, Def. 3.
fn mobius_add(a: Vec3, b: Vec3) -> Vec3 {
    let ab = a.dot(b);
    let aa = a.length_squared();
    let bb = b.length_squared();
    let num = (1.0 + 2.0 * ab + bb) * a + (1.0 - aa) * b;
    let den = 1.0 + 2.0 * ab + aa * bb;
    if den.abs() < 1e-12 {
        Vec3::ZERO
    } else {
        num / den
    }
}

// Ungar, From Möbius to Gyrogroups, 2008, §4, Def. 4.
fn gyr_apply(a: Vec3, b: Vec3, v: Vec3) -> Vec3 {
    let scalar = 1.0 + a.dot(b);
    let axis = -a.cross(b);
    let norm2 = scalar * scalar + axis.length_squared();
    v + (2.0 / norm2) * (scalar * axis.cross(v) + axis.cross(axis.cross(v)))
}

// Cannon, Floyd, Kenyon, Parry, Hyperbolic Geometry, 1997, §7: the Lorentz embedding of the ball.
pub fn poincare_to_hyperboloid(p: Vec3) -> Vec4 {
    let r2 = p.length_squared().min(POINCARE_R2_MAX);
    let den = 1.0 - r2;
    Vec4::new(
        2.0 * p.x / den,
        2.0 * p.y / den,
        2.0 * p.z / den,
        (1.0 + r2) / den,
    )
}

pub fn hyperboloid_to_poincare(h: Vec4) -> Vec3 {
    let den = (1.0 + h.w).max(1e-7);
    Vec3::new(h.x / den, h.y / den, h.z / den)
}

// Cannon, Floyd, Kenyon, Parry, Hyperbolic Geometry, 1997, §7: Klein is the hyperboloid seen from the origin.
pub fn hyperboloid_to_klein(h: Vec4) -> Vec3 {
    Vec3::new(h.x, h.y, h.z) / h.w.max(1.0)
}

pub fn klein_to_hyperboloid(k: Vec3) -> Vec4 {
    let w = 1.0 / (1.0 - k.length_squared().min(POINCARE_R2_MAX)).sqrt();
    Vec4::new(k.x * w, k.y * w, k.z * w, w)
}

pub fn poincare_to_klein(p: Vec3) -> Vec3 {
    hyperboloid_to_klein(poincare_to_hyperboloid(p))
}

pub fn klein_to_poincare(k: Vec3) -> Vec3 {
    hyperboloid_to_poincare(klein_to_hyperboloid(k))
}

/// Klein point at hyperbolic distance `t` from the origin along unit `dir`; geodesics from the origin are straight.
pub fn klein_ray(dir: Vec3, t: f32) -> Vec3 {
    dir * t.tanh()
}

fn poincare_to_hyperboloid_tangent(p: Vec3, v: Vec3) -> Vec4 {
    let r2 = p.length_squared().min(POINCARE_R2_MAX);
    let den = 1.0 - r2;
    let radial = 4.0 * p.dot(v) / (den * den);
    let space = (2.0 / den) * v + radial * p;
    Vec4::new(space.x, space.y, space.z, radial)
}

fn hyperboloid_to_poincare_tangent(h: Vec4, dh: Vec4) -> Vec3 {
    let den = (1.0 + h.w).max(1e-7);
    let space = Vec3::new(h.x, h.y, h.z);
    let d_space = Vec3::new(dh.x, dh.y, dh.z);
    d_space / den - space * (dh.w / (den * den))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn iso_translation_moves_origin_to_target() {
        let target = Vec3::new(0.3, -0.1, 0.2);
        let iso = Iso3H::from_translation(target);
        let moved = HyperbolicH3.iso_apply(iso, Vec3::ZERO);
        assert_relative_eq!(moved.x, target.x, epsilon = 1e-5);
        assert_relative_eq!(moved.y, target.y, epsilon = 1e-5);
        assert_relative_eq!(moved.z, target.z, epsilon = 1e-5);
    }

    fn ball_sweep() -> Vec<Vec3> {
        let radii = [
            0.0f32, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999,
        ];
        let directions = [
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(-1.0, 0.0, 0.0),
            Vec3::new(0.577, 0.577, 0.577),
            Vec3::new(0.3, -0.9, 0.2),
        ];
        let mut out = Vec::new();
        for r in radii {
            for d in directions {
                out.push(d.normalize_or_zero() * r);
            }
        }
        out
    }

    #[test]
    fn a_point_past_the_declared_envelope_is_refused() {
        let s = HyperbolicH3;
        let at = |d: f32| Vec3::X * (0.5 * d).tanh();
        assert!(s.valid_point(at(H3_DEPTH_ENVELOPE - 1.0)));
        assert!(!s.valid_point(at(H3_DEPTH_ENVELOPE + 1.0)));
        assert!(!s.valid_point(Vec3::X));
        assert!(!s.valid_point(Vec3::splat(f32::NAN)));
    }

    #[test]
    fn the_poincare_klein_and_hyperboloid_charts_agree() {
        for p in ball_sweep() {
            let expected = 2.0 * p / (1.0 + p.length_squared());
            let k = poincare_to_klein(p);
            assert!(
                (k - expected).length() <= 1e-6,
                "Klein of {p:?} is {k:?}, closed form {expected:?}"
            );
        }

        let p = Vec3::new(0.2, -0.3, 0.1);
        let h = poincare_to_hyperboloid(p);
        let lorentz = -h.x * h.x - h.y * h.y - h.z * h.z + h.w * h.w;
        assert_relative_eq!(lorentz, 1.0, epsilon = 1e-5);
        let p2 = hyperboloid_to_poincare(h);
        assert_relative_eq!(p2.x, p.x, epsilon = 1e-6);
        assert_relative_eq!(p2.y, p.y, epsilon = 1e-6);
        assert_relative_eq!(p2.z, p.z, epsilon = 1e-6);

        let dir = Vec3::new(0.6, -0.48, 0.64);
        for t in [0.1_f32, 1.0, 3.0, 6.0] {
            let p = klein_to_poincare(klein_ray(dir, t));
            assert_relative_eq!(HyperbolicH3.distance(Vec3::ZERO, p), t, epsilon = 2e-4 * t);
        }
    }
}
