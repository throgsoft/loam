//! Rigid motions of R⁴ use rotors and translations; [`crate::Iso4`] acts on S³ instead.

use std::borrow::Cow;

use glam::Vec4;
use serde::{Deserialize, Serialize};

use crate::bivector::{Rotor, Rotor4};
use crate::space::{IsometryGroup, Space, WgslSpace};

/// Rigid motion of R⁴; callers must keep the rotor unit length.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso4Flat {
    /// Applied before translation; composition does not renormalize.
    pub rotation: Rotor4,
    /// Offset added after the rotation, in the target frame's coordinates.
    pub translation: Vec4,
}

impl Iso4Flat {
    pub const IDENTITY: Self = Self {
        rotation: Rotor4::IDENTITY,
        translation: Vec4::ZERO,
    };

    pub fn from_rotation(rotation: Rotor4) -> Self {
        Self {
            rotation,
            translation: Vec4::ZERO,
        }
    }

    pub fn from_translation(translation: Vec4) -> Self {
        Self {
            rotation: Rotor4::IDENTITY,
            translation,
        }
    }
}

impl Default for Iso4Flat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR4;

impl Space for EuclideanR4 {
    type Point = Vec4;
    type Vector = Vec4;

    fn distance(&self, a: Vec4, b: Vec4) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec4, v: Vec4) -> Vec4 {
        at + v
    }

    fn log(&self, from: Vec4, to: Vec4) -> Vec4 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec4, _to: Vec4, v: Vec4) -> Vec4 {
        v
    }

    fn is_chart_flat(&self) -> bool {
        true
    }
}

impl IsometryGroup for EuclideanR4 {
    type Iso = Iso4Flat;

    fn iso_identity(&self) -> Iso4Flat {
        Iso4Flat::IDENTITY
    }

    fn iso_compose(&self, a: Iso4Flat, b: Iso4Flat) -> Iso4Flat {
        Iso4Flat {
            rotation: b.rotation * a.rotation,
            translation: a.rotation.apply(b.translation) + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso4Flat) -> Iso4Flat {
        let inv_rot = a.rotation.inverse();
        Iso4Flat {
            rotation: inv_rot,
            translation: inv_rot.apply(-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso4Flat, p: Vec4) -> Vec4 {
        iso.rotation.apply(p) + iso.translation
    }

    fn iso_transport(&self, iso: Iso4Flat, _at: Vec4, v: Vec4) -> Vec4 {
        iso.rotation.apply(v)
    }
}

impl WgslSpace for EuclideanR4 {
    fn wgsl_impl(&self) -> Cow<'static, str> {
        Cow::Borrowed(WGSL_IMPL)
    }
}

const WGSL_IMPL: &str = r#"
// loam-math :: EuclideanR4 (v0 Space WGSL ABI)
const LOAM_MAX_ARC: f32 = 1e9;
fn loam_distance(a: vec4<f32>, b: vec4<f32>) -> f32 { return length(a - b); }
fn loam_origin_distance(p: vec4<f32>) -> f32 { return length(p); }
fn loam_exp(at: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return at + v; }
fn loam_log(p_from: vec4<f32>, p_to: vec4<f32>) -> vec4<f32> { return p_to - p_from; }
fn loam_parallel_transport(p_from: vec4<f32>, p_to: vec4<f32>, v: vec4<f32>) -> vec4<f32> { return v; }
"#;

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn r4() -> EuclideanR4 {
        EuclideanR4
    }

    #[test]
    fn iso_transport_ignores_translation() {
        let s = r4();
        let iso = Iso4Flat::from_translation(Vec4::new(100.0, -50.0, 7.0, 12.0));
        let v = Vec4::new(1.0, 0.0, 0.0, 0.0);
        let transported = s.iso_transport(iso, Vec4::ZERO, v);
        assert_relative_eq!(transported.x, v.x);
        assert_relative_eq!(transported.y, v.y);
        assert_relative_eq!(transported.z, v.z);
        assert_relative_eq!(transported.w, v.w);
    }
}
