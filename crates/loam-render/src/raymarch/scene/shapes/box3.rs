use glam::Vec3;

use super::super::literal::wgsl_f32;

pub(in super::super) fn wgsl(name: &str, half_extents: Vec3) -> String {
    format!(
        "fn {name}(p: vec3<f32>) -> f32 {{\n\
         \tlet b = vec3<f32>({hx}, {hy}, {hz});\n\
         \tlet q = abs(p) - b;\n\
         \treturn length(max(q, vec3<f32>(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);\n\
         }}\n",
        name = name,
        hx = wgsl_f32(half_extents.x),
        hy = wgsl_f32(half_extents.y),
        hz = wgsl_f32(half_extents.z),
    )
}
