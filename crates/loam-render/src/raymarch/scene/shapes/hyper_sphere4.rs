use glam::Vec4;

use super::super::literal::wgsl_f32;

pub(in super::super) fn wgsl_4d(name: &str, center: Vec4, radius: f32) -> String {
    format!(
        "fn {name}(p: vec4<f32>) -> f32 {{\n\
        \treturn length(p - vec4<f32>({cx}, {cy}, {cz}, {cw})) - ({radius});\n\
        }}\n",
        name = name,
        cx = wgsl_f32(center.x),
        cy = wgsl_f32(center.y),
        cz = wgsl_f32(center.z),
        cw = wgsl_f32(center.w),
        radius = wgsl_f32(radius),
    )
}
