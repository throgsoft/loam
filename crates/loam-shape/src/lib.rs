//! Pose is extrinsic: shapes are defined in a local frame and positioned by
//! the caller's transform. [`Shape::Sphere`] and [`Shape::HyperSphere4D`] are
//! the exceptions, carrying a `center` that physics ignores.

pub mod field;
pub mod field_shape;
pub mod isovolume;
pub mod polytope;
pub mod polytope_geom;
pub mod projected_edges;
pub mod shape;
pub mod visualizable;

pub use field::{DistanceField, FieldKind};
pub use field_shape::FieldShape;
pub use isovolume::Isovolume;
pub use shape::{Shape, ShapeKind};
pub use visualizable::{LineMesh, NotVisualizable, PointMesh, TriangleMesh, Visualizable};
