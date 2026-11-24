//! The WGSL form of each shape's distance function. The CPU form is in `loam_shape::shape`.

pub(super) mod box3;
pub(super) mod convex_polytope4;
pub(super) mod half_space;
pub(super) mod half_space4;
pub(super) mod hyper_sphere4;
pub(super) mod sphere;

use super::SENTINEL_DISTANCE;

pub(super) fn sentinel_wgsl(name: &str) -> String {
    format!("fn {name}(_p: vec3<f32>) -> f32 {{\n\treturn {SENTINEL_DISTANCE:e};\n}}\n",)
}

pub(super) fn sentinel_wgsl_4d(name: &str) -> String {
    format!(
        "fn {name}(_p: vec4<f32>) -> f32 {{\n\
        \treturn {SENTINEL_DISTANCE:e};\n\
        }}\n",
    )
}
