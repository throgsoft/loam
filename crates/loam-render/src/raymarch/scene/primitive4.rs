use glam::Vec4;
use loam_shape::shape::{half_space4, hyper_sphere4};
use loam_shape::Shape;

use super::shapes;
use super::SENTINEL_DISTANCE;

pub trait Primitive4 {
    fn to_wgsl_4d(&self, name: &str) -> String;

    fn eval_4d(&self, p: Vec4) -> f32;
}

impl Primitive4 for Shape {
    fn to_wgsl_4d(&self, name: &str) -> String {
        match self {
            Shape::HyperSphere4D { center, radius } => {
                shapes::hyper_sphere4::wgsl_4d(name, *center, *radius)
            }
            Shape::ConvexPolytope4D { .. } => shapes::convex_polytope4::wgsl_4d(name),
            Shape::HalfSpace4D { normal, offset } => {
                shapes::half_space4::wgsl_4d(name, *normal, *offset)
            }
            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => shapes::sentinel_wgsl_4d(name),
        }
    }

    fn eval_4d(&self, p: Vec4) -> f32 {
        match self {
            Shape::HyperSphere4D { center, radius } => hyper_sphere4::distance(p, *center, *radius),
            Shape::ConvexPolytope4D { .. } => SENTINEL_DISTANCE,
            Shape::HalfSpace4D { normal, offset } => half_space4::distance(p, *normal, *offset),
            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => SENTINEL_DISTANCE,
        }
    }
}
