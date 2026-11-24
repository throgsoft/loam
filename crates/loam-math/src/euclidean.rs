use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::space::{IsometryGroup, Space};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso3 {
    pub rotation: Quat,
    pub translation: Vec3,
}

impl Iso3 {
    pub const IDENTITY: Self = Self {
        rotation: Quat::IDENTITY,
        translation: Vec3::ZERO,
    };

    pub fn from_rotation(rotation: Quat) -> Self {
        Self {
            rotation,
            translation: Vec3::ZERO,
        }
    }

    pub fn from_translation(translation: Vec3) -> Self {
        Self {
            rotation: Quat::IDENTITY,
            translation,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR3;

impl Space for EuclideanR3 {
    type Point = Vec3;
    type Vector = Vec3;
    type Frame = Quat;

    fn frame_at(&self, _at: Vec3) -> Quat {
        Quat::IDENTITY
    }

    fn distance(&self, a: Vec3, b: Vec3) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec3, v: Vec3) -> Vec3 {
        at + v
    }

    fn log(&self, from: Vec3, to: Vec3) -> Vec3 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec3, _to: Vec3, v: Vec3) -> Vec3 {
        v
    }

    fn is_chart_flat(&self) -> bool {
        true
    }

    fn chart_envelope(&self) -> f32 {
        f32::INFINITY
    }

    fn valid_point(&self, p: Vec3) -> bool {
        p.is_finite()
    }
}

impl IsometryGroup for EuclideanR3 {
    type Iso = Iso3;

    fn iso_identity(&self) -> Iso3 {
        Iso3::IDENTITY
    }

    fn iso_compose(&self, a: Iso3, b: Iso3) -> Iso3 {
        Iso3 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation * b.translation + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso3) -> Iso3 {
        let inv_rot = a.rotation.inverse();
        Iso3 {
            rotation: inv_rot,
            translation: inv_rot * (-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso3, p: Vec3) -> Vec3 {
        iso.rotation * p + iso.translation
    }

    fn iso_transport(&self, iso: Iso3, _at: Vec3, v: Vec3) -> Vec3 {
        iso.rotation * v
    }
}
