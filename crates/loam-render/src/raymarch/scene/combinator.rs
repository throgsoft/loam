use super::literal::wgsl_f32;

/// Quilez polynomial smooth-minimum; `k` is the blend radius in Space distance units.
pub fn smooth_min_fn(name: &str, k: f32) -> String {
    let k = wgsl_f32(k);
    format!(
        "fn {name}(a: f32, b: f32) -> f32 {{\n\
         \tlet h = clamp(0.5 + 0.5 * (b - a) / ({k}), 0.0, 1.0);\n\
         \treturn mix(b, a, h) - ({k}) * h * (1.0 - h);\n\
         }}\n",
    )
}
