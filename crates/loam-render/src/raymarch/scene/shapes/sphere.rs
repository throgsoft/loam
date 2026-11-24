use glam::Vec3;

use super::super::literal::wgsl_f32;

pub(in super::super) fn wgsl(name: &str, center: Vec3, radius: f32) -> String {
    format!(
        "fn {name}(p: vec3<f32>) -> f32 {{\n\
         \treturn loam_distance(p, vec3<f32>({cx}, {cy}, {cz})) - ({r});\n\
         }}\n",
        name = name,
        cx = wgsl_f32(center.x),
        cy = wgsl_f32(center.y),
        cz = wgsl_f32(center.z),
        r = wgsl_f32(radius),
    )
}
