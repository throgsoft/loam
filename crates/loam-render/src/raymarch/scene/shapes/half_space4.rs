use glam::Vec4;

use super::super::literal::wgsl_f32;

pub(in super::super) fn wgsl_4d(name: &str, normal: Vec4, offset: f32) -> String {
    format!(
        "fn {name}(p: vec4<f32>) -> f32 {{\n\
        \treturn dot(p, vec4<f32>({nx}, {ny}, {nz}, {nw})) - ({offset});\n\
        }}\n",
        name = name,
        nx = wgsl_f32(normal.x),
        ny = wgsl_f32(normal.y),
        nz = wgsl_f32(normal.z),
        nw = wgsl_f32(normal.w),
        offset = wgsl_f32(offset),
    )
}

// Lowers `t_max` to where a ray in the slice at `w_slice_expr` enters the half space.
pub(in super::super) fn max_t_wgsl(normal: Vec4, offset: f32, w_slice_expr: &str) -> String {
    format!(
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
        offset = wgsl_f32(offset),
    )
}
