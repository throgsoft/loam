use std::boxed::Box;

use glam::{Vec3, Vec4};
use serde::{Deserialize, Serialize};

use crate::literal::wgsl_f32;
use crate::primitive4::Primitive4;
use crate::SENTINEL_DISTANCE;
pub use loam_shape::Shape;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneNode4 {
    Leaf(Shape),
    Union(Box<SceneNode4>, Box<SceneNode4>),
    Intersection(Box<SceneNode4>, Box<SceneNode4>),
    /// `max(left, −right)`: `right` is carved out of `left`.
    Difference(Box<SceneNode4>, Box<SceneNode4>),
}

impl SceneNode4 {
    pub fn hypersphere(center: Vec4, radius: f32) -> Self {
        SceneNode4::Leaf(Shape::HyperSphere4D { center, radius })
    }

    pub fn halfspace(normal: Vec4, offset: f32) -> Self {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset })
    }

    /// The static `Primitive4` emit returns a sentinel; polytope leaves are invisible.
    pub fn polytope(vertices: Vec<Vec4>) -> Self {
        SceneNode4::Leaf(Shape::ConvexPolytope4D { vertices })
    }

    pub fn union(self, other: SceneNode4) -> Self {
        SceneNode4::Union(Box::new(self), Box::new(other))
    }

    pub fn intersect(self, other: SceneNode4) -> Self {
        SceneNode4::Intersection(Box::new(self), Box::new(other))
    }

    pub fn subtract(self, other: SceneNode4) -> Self {
        SceneNode4::Difference(Box::new(self), Box::new(other))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scene4 {
    pub root: SceneNode4,
}

impl Scene4 {
    pub fn new(root: SceneNode4) -> Self {
        Self { root }
    }

    /// Emits `fn loam_scene_sdf_4d(p: vec4<f32>) -> f32`.
    pub fn to_wgsl_4d(&self) -> String {
        let mut helpers = String::new();
        let mut body = String::new();
        let mut counter = 0u32;
        let (d_root, _k_root) =
            emit_node_4d(&self.root, &mut counter, &mut helpers, &mut body, None);
        let kind_consts = scene_kind_constants();
        format!(
            "// ---- loam-scene scene4 (native 4D) ----\n\
             {kind_consts}\
             {helpers}\
             fn loam_scene_sdf_4d(p: vec4<f32>) -> f32 {{\n\
             {body}\
             \treturn {d_root};\n\
             }}\n"
        )
    }

    /// Emits scene distance, primitive kind and march bounds at the supplied WGSL slice expression.
    pub fn to_hyperslice_wgsl(&self, w_slice_expr: &str) -> String {
        emit_hyperslice(self, w_slice_expr, None)
    }

    /// Halfspaces return [`SENTINEL_DISTANCE`] when the supplied WGSL f32 expression is below 0.5.
    pub fn to_hyperslice_wgsl_gated(
        &self,
        w_slice_expr: &str,
        halfspace_gate_expr: &str,
    ) -> String {
        emit_hyperslice(self, w_slice_expr, Some(halfspace_gate_expr))
    }

    /// Evaluates a slice; disabling `halfspace_gate` replaces halfspaces with [`SENTINEL_DISTANCE`].
    pub fn eval_at(&self, p3: Vec3, w_slice: f32, halfspace_gate: bool) -> (f32, u32) {
        eval_node_4d(&self.root, p3.extend(w_slice), halfspace_gate)
    }

    pub fn eval(&self, p3: Vec3, w_slice: f32, halfspace_gate: bool) -> f32 {
        self.eval_at(p3, w_slice, halfspace_gate).0
    }
}

pub const PRIM_KIND_HYPERSPHERE4D: u32 = 0;
pub const PRIM_KIND_HALFSPACE4D: u32 = 1;
/// A `Difference` node, which has no single owning primitive, or a leaf with no 4D closed form.
pub const PRIM_KIND_OTHER: u32 = 255;

fn scene_kind_constants() -> String {
    format!(
        "const LOAM_PRIM_HYPERSPHERE4D: u32 = {PRIM_KIND_HYPERSPHERE4D}u;\n\
         const LOAM_PRIM_HALFSPACE4D: u32 = {PRIM_KIND_HALFSPACE4D}u;\n\
         const LOAM_PRIM_OTHER: u32 = {PRIM_KIND_OTHER}u;\n"
    )
}

fn emit_hyperslice(
    scene: &Scene4,
    w_slice_expr: &str,
    halfspace_gate_expr: Option<&str>,
) -> String {
    let mut helpers = String::new();
    let mut body = String::new();
    let mut counter = 0u32;
    let (d_root, k_root) = emit_node_4d(
        &scene.root,
        &mut counter,
        &mut helpers,
        &mut body,
        halfspace_gate_expr,
    );
    let kind_consts = scene_kind_constants();
    let max_t_body = emit_max_t_body(&scene.root, w_slice_expr, halfspace_gate_expr);

    format!(
        "// ---- loam-scene scene4 (hyperslice at w = {w_slice_expr}) ----\n\
         {kind_consts}\
         struct LoamSceneHit {{ dist: f32, kind: u32 }}\n\
         {helpers}\
         fn loam_scene_at(p3: vec3<f32>) -> LoamSceneHit {{\n\
         \tlet p = vec4<f32>(p3, {w_slice_expr});\n\
         {body}\
         \treturn LoamSceneHit({d_root}, {k_root});\n\
         }}\n\
         fn loam_scene_sdf(p3: vec3<f32>) -> f32 {{\n\
         \treturn loam_scene_at(p3).dist;\n\
         }}\n\
         fn loam_scene_max_t(ro: vec3<f32>, rd: vec3<f32>) -> f32 {{\n\
         \tvar t_max: f32 = {SENTINEL_DISTANCE:e};\n\
         {max_t_body}\
         \treturn t_max;\n\
         }}\n"
    )
}

fn emit_max_t_body(
    node: &SceneNode4,
    w_slice_expr: &str,
    halfspace_gate_expr: Option<&str>,
) -> String {
    let mut body = String::new();
    walk_max_t(node, &mut body, w_slice_expr, halfspace_gate_expr);
    body
}

fn walk_max_t(
    node: &SceneNode4,
    body: &mut String,
    w_slice_expr: &str,
    halfspace_gate_expr: Option<&str>,
) {
    match node {
        SceneNode4::Leaf(Shape::HalfSpace4D { normal, offset }) => {
            let inner = format!(
                "\t\tlet n = vec3<f32>({nx}, {ny}, {nz});\n\
                 \t\tlet dr = dot(rd, n);\n\
                 \t\tif (dr < -1.0e-4) {{\n\
                 \t\t\tlet t = (({offset}) - ({nw}) * ({w_slice_expr}) - dot(ro, n)) / dr;\n\
                 \t\t\tif (t > 0.0 && t < t_max) {{ t_max = t; }}\n\
                 \t\t}}\n",
                nw = wgsl_f32(normal.w),
                nx = wgsl_f32(normal.x),
                ny = wgsl_f32(normal.y),
                nz = wgsl_f32(normal.z),
                offset = wgsl_f32(*offset),
            );
            match halfspace_gate_expr {
                None => {
                    body.push_str("\t{\n");
                    body.push_str(&inner);
                    body.push_str("\t}\n");
                }
                Some(gate) => {
                    body.push_str(&format!("\tif ({gate} >= 0.5) {{\n"));
                    body.push_str(&inner);
                    body.push_str("\t}\n");
                }
            }
        }
        SceneNode4::Leaf(_) | SceneNode4::Intersection(..) | SceneNode4::Difference(..) => {}
        SceneNode4::Union(l, r) => {
            walk_max_t(l, body, w_slice_expr, halfspace_gate_expr);
            walk_max_t(r, body, w_slice_expr, halfspace_gate_expr);
        }
    }
}

fn primitive_kind(shape: &Shape) -> (&'static str, u32) {
    match shape {
        Shape::HyperSphere4D { .. } => ("LOAM_PRIM_HYPERSPHERE4D", PRIM_KIND_HYPERSPHERE4D),
        Shape::HalfSpace4D { .. } => ("LOAM_PRIM_HALFSPACE4D", PRIM_KIND_HALFSPACE4D),
        _ => ("LOAM_PRIM_OTHER", PRIM_KIND_OTHER),
    }
}

fn emit_node_4d(
    node: &SceneNode4,
    counter: &mut u32,
    helpers: &mut String,
    body: &mut String,
    halfspace_gate_expr: Option<&str>,
) -> (String, String) {
    let idx = *counter;
    *counter += 1;
    match node {
        SceneNode4::Leaf(prim) => {
            let fn_name = format!("sdf4_p{idx}");
            helpers.push_str(&prim.to_wgsl_4d(&fn_name));
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            let (kind, _kind_value) = primitive_kind(prim);
            let gated = matches!(prim, Shape::HalfSpace4D { .. }) && halfspace_gate_expr.is_some();
            if gated {
                let gate = halfspace_gate_expr.expect("gated branch implies Some");
                body.push_str(&format!("\tlet {d_var}_raw = {fn_name}(p);\n"));
                body.push_str(&format!(
                    "\tlet {d_var} = select({SENTINEL_DISTANCE:e}, {d_var}_raw, {gate} >= 0.5);\n"
                ));
            } else {
                body.push_str(&format!("\tlet {d_var} = {fn_name}(p);\n"));
            }
            body.push_str(&format!("\tlet {k_var}: u32 = {kind};\n"));
            (d_var, k_var)
        }
        SceneNode4::Union(left, right) => {
            let (ld, lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = min({ld}, {rd});\n"));
            body.push_str(&format!(
                "\tlet {k_var}: u32 = select({rk}, {lk}, {ld} <= {rd});\n"
            ));
            (d_var, k_var)
        }
        SceneNode4::Intersection(left, right) => {
            let (ld, lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = max({ld}, {rd});\n"));
            body.push_str(&format!(
                "\tlet {k_var}: u32 = select({rk}, {lk}, {ld} >= {rd});\n"
            ));
            (d_var, k_var)
        }
        SceneNode4::Difference(left, right) => {
            let (ld, _lk) = emit_node_4d(left, counter, helpers, body, halfspace_gate_expr);
            let (rd, _rk) = emit_node_4d(right, counter, helpers, body, halfspace_gate_expr);
            let d_var = format!("d{idx}");
            let k_var = format!("k{idx}");
            body.push_str(&format!("\tlet {d_var} = max({ld}, -({rd}));\n"));

            body.push_str(&format!("\tlet {k_var}: u32 = LOAM_PRIM_OTHER;\n"));
            (d_var, k_var)
        }
    }
}

fn eval_node_4d(node: &SceneNode4, p: Vec4, halfspace_gate: bool) -> (f32, u32) {
    match node {
        SceneNode4::Leaf(prim) => {
            let (_kind_name, kind) = primitive_kind(prim);
            let gated = matches!(prim, Shape::HalfSpace4D { .. }) && !halfspace_gate;
            let dist = if gated {
                SENTINEL_DISTANCE
            } else {
                prim.eval_4d(p)
            };
            (dist, kind)
        }
        SceneNode4::Union(left, right) => {
            let (ld, lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.min(rd), if ld <= rd { lk } else { rk })
        }
        SceneNode4::Intersection(left, right) => {
            let (ld, lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.max(rd), if ld >= rd { lk } else { rk })
        }
        SceneNode4::Difference(left, right) => {
            let (ld, _lk) = eval_node_4d(left, p, halfspace_gate);
            let (rd, _rk) = eval_node_4d(right, p, halfspace_gate);
            (ld.max(-rd), PRIM_KIND_OTHER)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ron_round_trip_4d() {
        let scene = Scene4::new(
            SceneNode4::hypersphere(Vec4::ZERO, 0.3).union(SceneNode4::halfspace(Vec4::Y, -0.4)),
        );
        let ron_str = scene.to_ron().expect("serialize");
        let recovered = Scene4::from_ron("<round trip>", &ron_str).expect("deserialize");
        assert_eq!(scene.to_wgsl_4d(), recovered.to_wgsl_4d());
    }
}
