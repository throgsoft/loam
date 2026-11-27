//! Space-free: nothing here depends on `PhysicsSpace`.

pub mod epa;
pub mod epa_r4;
pub mod gjk;
pub mod gjk_r4;
mod minkowski;
mod simplex_r4;

pub use crate::space::VectorOps;
pub use epa::epa;
pub use epa_r4::epa_r4;
pub use gjk::gjk_intersect;
pub use gjk_r4::gjk_intersect_r4;
pub use minkowski::{
    minkowski_support, ContactInfo, ConvexHull, GjkResult, MinkowskiPoint, PosedHull, Sphere,
    SupportFn,
};
