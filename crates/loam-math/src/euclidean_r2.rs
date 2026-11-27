use glam::Vec2;
use serde::{Deserialize, Serialize};

use crate::bivector::{Rotor, Rotor2};
use crate::space::{IsometryGroup, Space};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Iso2 {
    pub rotation: Rotor2,
    pub translation: Vec2,
}

impl Iso2 {
    pub const IDENTITY: Self = Self {
        rotation: Rotor2::IDENTITY,
        translation: Vec2::ZERO,
    };

    pub fn from_rotation(rotation: Rotor2) -> Self {
        Self {
            rotation,
            translation: Vec2::ZERO,
        }
    }

    pub fn from_translation(translation: Vec2) -> Self {
        Self {
            rotation: Rotor2::IDENTITY,
            translation,
        }
    }
}

impl Default for Iso2 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Serialize for Rotor2 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        (self.a, self.b).serialize(s)
    }
}

impl<'de> Deserialize<'de> for Rotor2 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let (a, b) = <(f32, f32)>::deserialize(d)?;
        Ok(Self { a, b })
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EuclideanR2;

impl Space for EuclideanR2 {
    type Point = Vec2;
    type Vector = Vec2;
    type Frame = Rotor2;

    fn frame_at(&self, _at: Vec2) -> Rotor2 {
        Rotor2::IDENTITY
    }

    fn distance(&self, a: Vec2, b: Vec2) -> f32 {
        (a - b).length()
    }

    fn exp(&self, at: Vec2, v: Vec2) -> Vec2 {
        at + v
    }

    fn log(&self, from: Vec2, to: Vec2) -> Vec2 {
        to - from
    }

    fn parallel_transport(&self, _from: Vec2, _to: Vec2, v: Vec2) -> Vec2 {
        v
    }

    fn is_chart_flat(&self) -> bool {
        true
    }

    fn chart_envelope(&self) -> f32 {
        f32::INFINITY
    }

    fn valid_point(&self, p: Vec2) -> bool {
        p.is_finite()
    }
}

impl IsometryGroup for EuclideanR2 {
    type Iso = Iso2;

    fn iso_identity(&self) -> Iso2 {
        Iso2::IDENTITY
    }

    fn iso_compose(&self, a: Iso2, b: Iso2) -> Iso2 {
        Iso2 {
            rotation: a.rotation * b.rotation,
            translation: a.rotation.apply(b.translation) + a.translation,
        }
    }

    fn iso_inverse(&self, a: Iso2) -> Iso2 {
        let inv = a.rotation.inverse();
        Iso2 {
            rotation: inv,
            translation: inv.apply(-a.translation),
        }
    }

    fn iso_apply(&self, iso: Iso2, p: Vec2) -> Vec2 {
        iso.rotation.apply(p) + iso.translation
    }

    fn iso_transport(&self, iso: Iso2, _at: Vec2, v: Vec2) -> Vec2 {
        iso.rotation.apply(v)
    }
}
