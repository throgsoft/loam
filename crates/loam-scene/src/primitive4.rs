use glam::Vec4;
use loam_shape::Shape;

use crate::literal::wgsl_f32;
use crate::SENTINEL_DISTANCE;

/// CPU evaluation and named vec4 WGSL functions share primitive support rules.
pub trait Primitive4 {
    fn to_wgsl_4d(&self, name: &str) -> String;

    fn eval_4d(&self, p: Vec4) -> f32;
}

impl Primitive4 for Shape {
    fn to_wgsl_4d(&self, name: &str) -> String {
        match self {
            Shape::HyperSphere4D { center, radius } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn length(p - vec4<f32>({cx}, {cy}, {cz}, {cw})) - ({radius});\n\
                }}\n",
                name = name,
                cx = wgsl_f32(center.x),
                cy = wgsl_f32(center.y),
                cz = wgsl_f32(center.z),
                cw = wgsl_f32(center.w),
                radius = wgsl_f32(*radius),
            ),

            Shape::ConvexPolytope4D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \t// ConvexPolytope4D: half-space emit lives in the\n\
                \t// render node's per-frame uniform path, not here.\n\
                \t// See Hyperslice4DNode.\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),

            Shape::HalfSpace4D { normal, offset } => format!(
                "fn {name}(p: vec4<f32>) -> f32 {{\n\
                \treturn dot(p, vec4<f32>({nx}, {ny}, {nz}, {nw})) - ({offset});\n\
                }}\n",
                name = name,
                nx = wgsl_f32(normal.x),
                ny = wgsl_f32(normal.y),
                nz = wgsl_f32(normal.z),
                nw = wgsl_f32(normal.w),
                offset = wgsl_f32(*offset),
            ),

            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => format!(
                "fn {name}(_p: vec4<f32>) -> f32 {{\n\
                \treturn {SENTINEL_DISTANCE:e};\n\
                }}\n",
            ),
        }
    }

    fn eval_4d(&self, p: Vec4) -> f32 {
        match self {
            Shape::HyperSphere4D { center, radius } => (p - *center).length() - *radius,

            Shape::ConvexPolytope4D { .. } => SENTINEL_DISTANCE,

            Shape::HalfSpace4D { normal, offset } => p.dot(*normal) - *offset,

            Shape::Sphere { .. }
            | Shape::HalfSpace { .. }
            | Shape::Box3 { .. }
            | Shape::Polygon2D { .. }
            | Shape::ConvexPolytope3D { .. } => SENTINEL_DISTANCE,
        }
    }
}
