//! Convex distance bounds for unit-circumradius polytopes.

use loam_shape::polytope::Polytope4;
use std::fmt::Write;

/// Include the stubs only when neither extended polytope appears in the scene.
pub fn polytope_stub_sdfs_wgsl() -> &'static str {
    "fn cell120_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n\
     fn cell600_sdf_local(p: vec4<f32>) -> f32 { return 1.0e9; }\n"
}

/// Exact signed distance inside; a conservative distance bound outside.
pub fn polytope_extended_sdfs_wgsl() -> String {
    let mut source = String::with_capacity(48 * 1024);
    for (shape, name) in [
        (Polytope4::Cell120, "cell120"),
        (Polytope4::Cell600, "cell600"),
    ] {
        let (normals, inradius) = shape.face_planes();
        let count = normals.len();
        let _ = writeln!(
            source,
            "const {name}_planes: array<vec4<f32>, {count}> = array("
        );
        for normal in normals {
            let _ = writeln!(
                source,
                "    vec4<f32>({:.10}, {:.10}, {:.10}, {:.10}),",
                normal.x, normal.y, normal.z, normal.w
            );
        }
        source.push_str(");\n");
        // Hart (1996), Sphere Tracing: the maximum of unit-plane distances is 1-Lipschitz.
        let _ = writeln!(
            source,
            r#"
fn {name}_sdf_local(p: vec4<f32>) -> f32 {{
    var distance = -1.0e9;
    for (var i = 0u; i < {count}u; i += 1u) {{
        distance = max(distance, dot({name}_planes[i], p) - {inradius:.10});
    }}
    return distance;
}}"#
        );
    }
    source
}
