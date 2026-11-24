use super::super::SENTINEL_DISTANCE;

pub(in super::super) fn wgsl_4d(name: &str) -> String {
    format!(
        "fn {name}(_p: vec4<f32>) -> f32 {{\n\
        \t// ConvexPolytope4D: half-space emit lives in the\n\
        \t// render node's per-frame uniform path, not here.\n\
        \t// See Hyperslice4DNode.\n\
        \treturn {SENTINEL_DISTANCE:e};\n\
        }}\n",
    )
}
