use glam::Vec3;

use super::super::literal::wgsl_f32;

pub(in super::super) fn wgsl(name: &str, normal: Vec3, offset: f32) -> String {
    format!(
        "fn {name}(p: vec3<f32>) -> f32 {{\n\
         \treturn dot(p, vec3<f32>({nx}, {ny}, {nz})) - ({d});\n\
         }}\n",
        name = name,
        nx = wgsl_f32(normal.x),
        ny = wgsl_f32(normal.y),
        nz = wgsl_f32(normal.z),
        d = wgsl_f32(offset),
    )
}
