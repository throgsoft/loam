use loam_render::raymarch::polytope_extended_sdfs_wgsl;
use loam_shape::polytope::Polytope4;

mod support;
use support::{dispatch, request_device};

#[test]
#[ignore = "requires a working wgpu adapter"]
fn generated_facets_bound_the_shape_geometry_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    for (shape, name) in [
        (Polytope4::Cell120, "cell120"),
        (Polytope4::Cell600, "cell600"),
    ] {
        let vertices = shape.topology().vertices;
        let (_, inradius) = shape.face_planes();
        let mut points = vec![[0.0; 4]];
        points.extend(vertices.iter().map(|v| v.to_array()));
        points.extend(vertices.iter().map(|v| (*v * 1.5).to_array()));
        let source = format!(
            r#"{}
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> distances: array<f32>;
@compute @workgroup_size(64)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {{
    if id.x >= arrayLength(&points) {{ return; }}
    distances[id.x] = {name}_sdf_local(points[id.x]);
}}
"#,
            polytope_extended_sdfs_wgsl()
        );
        let distances: Vec<f32> = dispatch(&device, &queue, &source, "probe", &points);
        assert!((distances[0] + inradius).abs() < 2e-5);
        for (index, vertex) in vertices.iter().enumerate() {
            assert!(
                distances[1 + index].abs() < 2e-5,
                "{shape:?} facet misses vertex {vertex}"
            );
            let outside = distances[1 + vertices.len() + index];
            assert!(
                outside > 0.0 && outside <= 0.5 * vertex.length() + 2e-5,
                "{shape:?} oversteps at {vertex}: {outside}"
            );
        }
    }
}
