use std::f32::consts::TAU;

pub(crate) const BODY_SIZE: f32 = 0.7;

pub(crate) const BODY_Y: f32 = 0.9;

// Over `2 × BODY_SIZE`, so rotated bodies stay clear of their neighbours.
pub(crate) const BODY_X_SPACING: f32 = 1.8;

pub(crate) const W_RANGE: f32 = 1.5;

pub(crate) const W_SCRUB_RATE: f32 = 0.5;

pub(crate) const BASE_ROTATION_RATE: f32 = TAU * 0.3;

// Shared by the marched half-space and the background ground, or the swap shows a seam.
pub(crate) const FLOOR_Y: f32 = 0.0;

pub(crate) const ARENA_HALF: f32 = 3.0;

pub(crate) const GRAVITY: f32 = 9.8;
