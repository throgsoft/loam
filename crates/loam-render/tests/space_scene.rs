use loam_render::shader::validate_wgsl;

fn assemble_source(space: &str, user: &str) -> String {
    assemble_source_with_scene(space, None, user)
}

fn assemble_source_with_scene(space: &str, scene: Option<&str>, user: &str) -> String {
    format!("{space}\n{}\n{user}", scene.unwrap_or_default())
}

use bytemuck::{Pod, Zeroable};
use glam::{Vec3, Vec4};
use loam_math::{
    BlendedSpace, EuclideanR3, EuclideanR4, HyperbolicH3, LinearBlendX, Space, SphericalS3,
    WgslSpace,
};

const ABI_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec3<f32>(0.1, 0.2, 0.3);
    let b = vec3<f32>(0.2, -0.1, 0.05);
    let v = vec3<f32>(0.01, 0.02, -0.03);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

const ABI_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let a = vec4<f32>(0.1, 0.2, 0.3, 0.0);
    let b = vec4<f32>(0.2, -0.1, 0.05, 0.4);
    let v = vec4<f32>(0.01, 0.02, -0.03, 0.05);
    _ = loam_distance(a, b);
    _ = loam_origin_distance(a);
    _ = loam_exp(a, v);
    _ = loam_log(a, b);
    _ = loam_parallel_transport(a, b, v);
    _ = LOAM_MAX_ARC;
}
"#;

#[test]
fn euclidean_space_prelude_validates_against_abi_probe() {
    let src = assemble_source(&EuclideanR3.wgsl_impl(), ABI_PROBE);
    validate_wgsl(&src).expect("EuclideanR3 WGSL prelude should validate");
}

#[test]
fn hyperbolic_space_prelude_validates_against_abi_probe() {
    let src = assemble_source(&HyperbolicH3.wgsl_impl(), ABI_PROBE);
    validate_wgsl(&src).expect("HyperbolicH3 WGSL prelude should validate");
}

#[test]
fn spherical_space_prelude_validates_against_abi_probe() {
    let src = assemble_source(&SphericalS3.wgsl_impl(), ABI_PROBE);
    validate_wgsl(&src).expect("SphericalS3 WGSL prelude should validate");
}

#[test]
fn euclidean_r4_space_prelude_validates_against_abi_probe() {
    let src = assemble_source(&EuclideanR4.wgsl_impl(), ABI_PROBE_VEC4);
    validate_wgsl(&src).expect("EuclideanR4 WGSL prelude should validate");
}

const KERNEL_SCENE: &str = r#"
fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    return loam_distance(p, vec3<f32>(0.0, 0.0, 0.0)) - 0.25;
}
"#;

const KERNEL_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main() {
    let ro = vec3<f32>(0.0, 0.0, 2.0);
    let rd = vec3<f32>(0.0, 0.0, -1.0);
    _ = loam_march_geodesic(ro, rd, 0.2);
    _ = loam_estimate_normal(vec3<f32>(0.0, 0.0, 0.0), 0.2);
    _ = loam_safe_normalize(vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(0.0, 1.0, 0.0));
}
"#;

fn assemble_geodesic_probe(space_wgsl: &str) -> String {
    assemble_source_with_scene(
        space_wgsl,
        Some(&format!(
            "{KERNEL_SCENE}{}",
            loam_render::shader::GEODESIC_MARCH_KERNEL
        )),
        KERNEL_PROBE,
    )
}

#[test]
fn euclidean_geodesic_kernel_validates() {
    let src = assemble_geodesic_probe(&EuclideanR3.wgsl_impl());
    validate_wgsl(&src).expect("EuclideanR3 + geodesic kernel should validate");
}

#[test]
fn hyperbolic_geodesic_kernel_validates() {
    let src = assemble_geodesic_probe(&HyperbolicH3.wgsl_impl());
    validate_wgsl(&src).expect("HyperbolicH3 + geodesic kernel should validate");
}

#[test]
fn spherical_geodesic_kernel_validates() {
    let src = assemble_geodesic_probe(&SphericalS3.wgsl_impl());
    validate_wgsl(&src).expect("SphericalS3 + geodesic kernel should validate");
}

#[test]
fn blended_e3_h3_prelude_validates_against_abi_probe() {
    let bs = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-2.0, 2.0).unwrap(),
    );
    let src = assemble_source(&bs.wgsl_impl(), ABI_PROBE);
    validate_wgsl(&src).expect("BlendedSpace<E3,H3,LinearBlendX> WGSL prelude should validate");
}

#[test]
fn blended_e3_h3_geodesic_kernel_validates() {
    let bs = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-2.0, 2.0).unwrap(),
    );
    let src = assemble_geodesic_probe(&bs.wgsl_impl());
    validate_wgsl(&src)
        .expect("BlendedSpace<E3,H3,LinearBlendX> + geodesic kernel should validate");
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuCase {
    a: [f32; 4],
    b: [f32; 4],
    v: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct GpuOut {
    scalars: [f32; 4],
    exp_point: [f32; 4],
    log_vec: [f32; 4],
    transported: [f32; 4],
}

const PROBE_IO: &str = r#"
struct Case {
    a: vec4<f32>,
    b: vec4<f32>,
    v: vec4<f32>,
};

struct ProbeOut {
    scalars: vec4<f32>,
    exp_point: vec4<f32>,
    log_vec: vec4<f32>,
    transported: vec4<f32>,
};

@group(0) @binding(0) var<storage, read> cases: array<Case>;
@group(0) @binding(1) var<storage, read_write> out: array<ProbeOut>;
"#;

const GPU_PROBE: &str = r#"
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    let a = c.a.xyz;
    let b = c.b.xyz;
    let v = c.v.xyz;
    out[i].scalars = vec4<f32>(
        loam_distance(a, b),
        loam_origin_distance(a),
        loam_origin_distance(b),
        0.0);
    out[i].exp_point = vec4<f32>(loam_exp(a, v), 0.0);
    out[i].log_vec = vec4<f32>(loam_log(a, b), 0.0);
    out[i].transported = vec4<f32>(loam_parallel_transport(a, b, v), 0.0);
}
"#;

const GPU_PROBE_VEC4: &str = r#"
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let c = cases[i];
    out[i].scalars = vec4<f32>(
        loam_distance(c.a, c.b),
        loam_origin_distance(c.a),
        loam_origin_distance(c.b),
        0.0);
    out[i].exp_point = loam_exp(c.a, c.v);
    out[i].log_vec = loam_log(c.a, c.b);
    out[i].transported = loam_parallel_transport(c.a, c.b, c.v);
}
"#;

struct ParityCase {
    corner: &'static str,
    a: Vec3,
    b: Vec3,
    v: Vec3,
}

fn corner(corner: &'static str, a: Vec3, b: Vec3, v: Vec3) -> ParityCase {
    ParityCase { corner, a, b, v }
}

fn flat_corners() -> Vec<ParityCase> {
    vec![
        corner(
            "coincident at the origin",
            Vec3::ZERO,
            Vec3::ZERO,
            Vec3::new(0.01, 0.02, -0.03),
        ),
        corner(
            "separation below one coordinate ulp",
            Vec3::new(1.0, 1.0, 1.0),
            Vec3::new(1.0 + 1e-8, 1.0, 1.0),
            Vec3::new(1e-8, 0.0, 0.0),
        ),
        corner(
            "small radius",
            Vec3::new(1e-4, 0.0, 0.0),
            Vec3::new(0.0, -1e-4, 0.0),
            Vec3::new(1e-5, 0.0, 2e-5),
        ),
        corner(
            "generic interior",
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.5, -0.1, 0.0),
            Vec3::new(0.01, 0.02, -0.03),
        ),
        corner(
            "wide separation, tangent past the unit ball",
            Vec3::new(-2.0, 4.0, 0.5),
            Vec3::new(1.5, 0.25, -3.0),
            Vec3::new(30.0, -12.0, 7.0),
        ),
    ]
}

fn hemisphere_corners() -> Vec<ParityCase> {
    vec![
        corner(
            "at the pole",
            Vec3::ZERO,
            Vec3::new(0.2, -0.1, 0.05),
            Vec3::new(0.01, 0.02, -0.03),
        ),
        corner(
            "small radius",
            Vec3::new(1e-4, 0.0, 0.0),
            Vec3::new(0.0, 1e-4, 0.0),
            Vec3::new(1e-5, -2e-5, 0.0),
        ),
        corner(
            "generic interior",
            Vec3::new(0.1, 0.2, 0.3),
            Vec3::new(0.2, -0.1, 0.05),
            Vec3::new(0.01, 0.02, -0.03),
        ),
        corner(
            "near-antipodal across the equator",
            Vec3::new(0.9999, 0.0, 0.0),
            Vec3::new(-0.9999, 0.0, 0.0),
            Vec3::new(0.02, 0.03, -0.01),
        ),
        corner(
            "near-antipodal, oblique",
            Vec3::new(0.7, 0.7, 0.1),
            Vec3::new(-0.7, -0.7, -0.1),
            Vec3::new(0.05, -0.02, 0.03),
        ),
    ]
}

fn ball_corners() -> Vec<ParityCase> {
    vec![
        corner(
            "at the ball centre",
            Vec3::ZERO,
            Vec3::new(0.2, -0.1, 0.08),
            Vec3::new(0.01, 0.02, -0.015),
        ),
        corner(
            "small radius",
            Vec3::new(1e-4, 0.0, 0.0),
            Vec3::new(0.0, 1e-4, 0.0),
            Vec3::new(1e-5, -2e-5, 0.0),
        ),
        corner(
            "generic interior",
            Vec3::new(0.1, 0.2, 0.05),
            Vec3::new(0.2, -0.1, 0.08),
            Vec3::new(0.01, 0.02, -0.015),
        ),
        corner(
            "near-antipodal across the ideal boundary",
            Vec3::new(0.99, 0.0, 0.0),
            Vec3::new(-0.99, 0.0, 0.0),
            Vec3::new(0.02, 0.03, -0.01),
        ),
        corner(
            "both endpoints at r = 0.9999",
            Vec3::new(0.9999, 0.0, 0.0),
            Vec3::new(0.0, -0.9999, 0.0),
            Vec3::new(0.02, -0.015, 0.01),
        ),
    ]
}

fn out_of_domain_corners() -> Vec<ParityCase> {
    let directions = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, -1.0, 0.0),
        Vec3::new(0.6, 0.8, 0.0),
        Vec3::new(0.5773503, 0.5773503, 0.5773503),
        Vec3::new(0.26726124, -0.5345225, 0.8017837),
        Vec3::new(-0.35856858, 0.5976143, -0.71713716),
    ];
    let radii = [1.0_f32, 1.0000001, 1.5, 8.0, 1e4];
    let partners = [
        Vec3::new(0.3, 0.1, 0.0),
        Vec3::ZERO,
        Vec3::new(0.0, 0.0, 1.0),
        Vec3::new(-2.0, 0.5, 0.25),
    ];
    let tangents = [
        Vec3::new(0.01, 0.02, -0.03),
        Vec3::new(2.0, -1.0, 0.5),
        Vec3::ZERO,
    ];
    let mut cases = Vec::new();
    for (i, direction) in directions.iter().enumerate() {
        for (j, radius) in radii.iter().enumerate() {
            let outside = *direction * *radius;
            let partner = partners[(i + j) % partners.len()];
            let tangent = tangents[(i + j) % tangents.len()];
            cases.push(corner("out of domain, source", outside, partner, tangent));
            cases.push(corner("out of domain, target", partner, outside, tangent));
            cases.push(corner("out of domain, both", outside, -outside, tangent));
        }
    }
    cases
}

fn gpu_case(a: Vec3, b: Vec3, v: Vec3) -> GpuCase {
    GpuCase {
        a: a.extend(0.0).to_array(),
        b: b.extend(0.0).to_array(),
        v: v.extend(0.0).to_array(),
    }
}

async fn run_gpu_probe<S: WgslSpace>(space: &S, cases: &[GpuCase]) -> Result<Vec<GpuOut>, String> {
    run_probe_body(space, GPU_PROBE, cases).await
}

async fn run_probe_body<S: WgslSpace>(
    space: &S,
    body: &str,
    cases: &[GpuCase],
) -> Result<Vec<GpuOut>, String> {
    run_compute_probe(
        &assemble_source(&space.wgsl_impl(), &format!("{PROBE_IO}{body}")),
        "loam-space-gpu-probe",
        cases,
    )
    .await
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn euclidean_r3_wgsl_matches_the_rust_space_at_the_domain_corners_gpu_probe() {
    assert_prelude_matches_cpu(&EuclideanR3, &flat_corners(), 1e-6);
}

const FLAT_CORNER_W: [[f32; 3]; 5] = [
    [0.0, 0.0, 0.125],
    [1.0, 1.0 + 1e-8, 1e-8],
    [1e-4, -1e-4, 2e-5],
    [-0.75, 2.5, 0.125],
    [-6.0, 9.0, 40.0],
];

#[test]
#[ignore = "requires a working wgpu adapter"]
fn euclidean_r4_wgsl_matches_the_rust_space_at_the_domain_corners_gpu_probe() {
    let space = EuclideanR4;
    let corners = flat_corners();
    assert_eq!(corners.len(), FLAT_CORNER_W.len());
    let cases: Vec<GpuCase> = corners
        .iter()
        .zip(FLAT_CORNER_W)
        .map(|(c, w)| GpuCase {
            a: c.a.extend(w[0]).to_array(),
            b: c.b.extend(w[1]).to_array(),
            v: c.v.extend(w[2]).to_array(),
        })
        .collect();
    let rows = pollster::block_on(run_probe_body(&space, GPU_PROBE_VEC4, &cases))
        .expect("EuclideanR4 GPU probe");
    for ((corner, case), row) in corners.iter().zip(&cases).zip(&rows) {
        let (a, b, v) = (
            Vec4::from_array(case.a),
            Vec4::from_array(case.b),
            Vec4::from_array(case.v),
        );
        let at = |what| format!("{} :: {what}", corner.corner);
        assert_near(&at("distance"), row.scalars[0], space.distance(a, b), 1e-6);
        assert_near(
            &at("origin_distance(a)"),
            row.scalars[1],
            space.distance(Vec4::ZERO, a),
            1e-6,
        );
        assert_near(
            &at("origin_distance(b)"),
            row.scalars[2],
            space.distance(Vec4::ZERO, b),
            1e-6,
        );
        assert_vec_near(&at("exp"), row.exp_point, space.exp(a, v), 1e-6);
        assert_vec_near(&at("log"), row.log_vec, space.log(a, b), 1e-6);
        assert_vec_near(
            &at("parallel_transport"),
            row.transported,
            space.parallel_transport(a, b, v),
            1e-6,
        );
    }
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn spherical_s3_wgsl_matches_the_rust_space_at_the_domain_corners_gpu_probe() {
    assert_prelude_matches_cpu(&SphericalS3, &hemisphere_corners(), 2e-4);
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn hyperbolic_h3_wgsl_matches_the_rust_space_at_the_domain_corners_gpu_probe() {
    assert_prelude_matches_cpu(&HyperbolicH3, &ball_corners(), 2e-4);
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn spherical_s3_wgsl_degrades_finitely_outside_the_hemisphere_gpu_probe() {
    assert_prelude_survives_out_of_domain(&SphericalS3, "SphericalS3");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn hyperbolic_h3_wgsl_degrades_finitely_outside_the_ball_gpu_probe() {
    assert_prelude_survives_out_of_domain(&HyperbolicH3, "HyperbolicH3");
}

async fn run_compute_probe<In: Pod, Out: Pod>(
    source: &str,
    label: &str,
    inputs: &[In],
) -> Result<Vec<Out>, String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter failed: {e}"))?;

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some(label),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))?;

    validate_wgsl(source).map_err(|e| e.to_string())?;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.to_owned().into()),
    });

    let input_size = std::mem::size_of_val(inputs) as u64;
    let output_size = (inputs.len() * std::mem::size_of::<Out>()) as u64;

    let input = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}-input")),
        size: input_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: true,
    });
    input
        .slice(..)
        .get_mapped_range_mut()
        .copy_from_slice(bytemuck::cast_slice(inputs));
    input.unmap();

    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}-output")),
        size: output_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(&format!("{label}-staging")),
        size: output_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label}-bgl")),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label}-bg")),
        layout: &bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: input.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label}-layout")),
        bind_group_layouts: &[&bgl],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(&format!("{label}-pipeline")),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some(&format!("{label}-encoder")),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(&format!("{label}-pass")),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(inputs.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &staging, 0, output_size);
    queue.submit(Some(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |res| {
        tx.send(res).expect("map callback receiver should exist");
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| e.to_string())?;
    rx.recv()
        .map_err(|e| e.to_string())?
        .map_err(|e| e.to_string())?;

    let data = slice.get_mapped_range();
    let rows = bytemuck::cast_slice::<u8, Out>(&data).to_vec();
    drop(data);
    staging.unmap();
    Ok(rows)
}

fn assert_prelude_matches_cpu<S>(space: &S, cases: &[ParityCase], eps: f32)
where
    S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
{
    let gpu: Vec<GpuCase> = cases.iter().map(|c| gpu_case(c.a, c.b, c.v)).collect();
    let rows = pollster::block_on(run_gpu_probe(space, &gpu)).expect("GPU probe");
    assert_eq!(cases.len(), rows.len());
    for (case, row) in cases.iter().zip(&rows) {
        let (a, b, v) = (case.a, case.b, case.v);
        let at = |what| format!("{} :: {what}", case.corner);
        assert_near(&at("distance"), row.scalars[0], space.distance(a, b), eps);
        assert_near(
            &at("origin_distance(a)"),
            row.scalars[1],
            space.distance(Vec3::ZERO, a),
            eps,
        );
        assert_near(
            &at("origin_distance(b)"),
            row.scalars[2],
            space.distance(Vec3::ZERO, b),
            eps,
        );
        assert_vec_near(&at("exp"), row.exp_point, space.exp(a, v).extend(0.0), eps);
        assert_vec_near(&at("log"), row.log_vec, space.log(a, b).extend(0.0), eps);
        assert_vec_near(
            &at("parallel_transport"),
            row.transported,
            space.parallel_transport(a, b, v).extend(0.0),
            eps,
        );
    }
}

fn assert_prelude_survives_out_of_domain<S>(space: &S, label: &str)
where
    S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
{
    let cases = out_of_domain_corners();
    let gpu: Vec<GpuCase> = cases.iter().map(|c| gpu_case(c.a, c.b, c.v)).collect();
    let rows = pollster::block_on(run_gpu_probe(space, &gpu)).expect("GPU probe");
    let mut worst = 0.0_f32;
    let mut worst_at = String::new();
    for (case, row) in cases.iter().zip(&rows) {
        let (a, b, v) = (case.a, case.b, case.v);
        let cpu = [
            Vec4::new(
                space.distance(a, b),
                space.distance(Vec3::ZERO, a),
                space.distance(Vec3::ZERO, b),
                0.0,
            ),
            space.exp(a, v).extend(0.0),
            space.log(a, b).extend(0.0),
            space.parallel_transport(a, b, v).extend(0.0),
        ];
        let gpu = [row.scalars, row.exp_point, row.log_vec, row.transported];
        let names = ["scalars", "exp", "log", "parallel_transport"];
        for ((name, cpu), gpu) in names.iter().zip(cpu).zip(gpu) {
            let where_ = || format!("{label}/{} a={a:?} b={b:?} v={v:?} {name}", case.corner);
            for (lane, (cpu, gpu)) in cpu.to_array().iter().zip(gpu).enumerate() {
                assert!(
                    gpu.is_finite(),
                    "{}: GPU lane {lane} is {gpu}, not finite",
                    where_()
                );
                assert!(
                    cpu.is_finite(),
                    "{}: CPU lane {lane} is {cpu}, not finite",
                    where_()
                );
                let divergence = (gpu - cpu).abs() / cpu.abs().max(1.0);
                if divergence > worst {
                    worst = divergence;
                    worst_at = format!("{} lane {lane}", where_());
                }
            }
        }
        let exp_gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
        assert!(
            v == Vec3::ZERO || exp_gpu.length_squared() <= 1.0,
            "{label}/{}: loam_exp({a:?}, {v:?}) returned {exp_gpu:?}, outside the chart",
            case.corner,
        );
    }
    println!("{label} out-of-domain CPU/GPU divergence: worst {worst} at {worst_at}");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn blended_e3_h3_gpu_probe_exp_matches_cpu() {
    let space = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-0.5, 0.5).unwrap(),
    );
    let cases = [
        gpu_case(
            Vec3::new(-1.0, 0.05, 0.0),
            Vec3::new(-0.8, 0.05, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
        ),
        gpu_case(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(0.1, 0.05, 0.0),
            Vec3::new(0.05, 0.0, 0.0),
        ),
        gpu_case(
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(0.71, 0.05, 0.0),
            Vec3::new(0.02, 0.02, 0.0),
        ),
        gpu_case(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(1e-7, 0.0, 0.0),
        ),
        gpu_case(
            Vec3::new(-2.0, 0.3, 0.1),
            Vec3::new(-1.6, 0.3, 0.1),
            Vec3::new(0.4, 0.05, 0.0),
        ),
    ];
    let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

    for (case, row) in cases.iter().zip(&out) {
        let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
        let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
        let cpu = space.exp(a, v);
        let gpu = Vec3::new(row.exp_point[0], row.exp_point[1], row.exp_point[2]);
        let diff = (cpu - gpu).length();
        assert!(
            diff < 5e-3,
            "BlendedSpace exp parity failed at a={a:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
        );
    }
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn blended_e3_h3_gpu_probe_transport_matches_cpu() {
    let space = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-0.5, 0.5).unwrap(),
    );
    let cases = [
        gpu_case(
            Vec3::new(-1.0, 0.05, 0.0),
            Vec3::new(-0.8, 0.05, 0.0),
            Vec3::new(0.1, 0.0, 0.0),
        ),
        gpu_case(
            Vec3::new(-0.6, 0.0, 0.0),
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(0.5, 0.5, 0.0),
        ),
        gpu_case(
            Vec3::new(0.7, 0.0, 0.0),
            Vec3::new(0.72, 0.05, 0.0),
            Vec3::new(0.02, 0.02, 0.0),
        ),
        gpu_case(
            Vec3::new(0.0, 0.05, 0.0),
            Vec3::new(1e-7, 0.05, 0.0),
            Vec3::new(0.05, -0.02, 0.01),
        ),
        gpu_case(
            Vec3::new(-2.0, 0.3, 0.1),
            Vec3::new(-1.6, 0.3, 0.1),
            Vec3::new(0.4, 0.05, 0.0),
        ),
    ];
    let out = pollster::block_on(run_gpu_probe(&space, &cases)).expect("BlendedSpace GPU probe");

    for (case, row) in cases.iter().zip(&out) {
        let a = Vec3::from_array([case.a[0], case.a[1], case.a[2]]);
        let b = Vec3::from_array([case.b[0], case.b[1], case.b[2]]);
        let v = Vec3::from_array([case.v[0], case.v[1], case.v[2]]);
        let cpu = space.parallel_transport(a, b, v);
        let gpu = Vec3::new(row.transported[0], row.transported[1], row.transported[2]);
        let diff = (cpu - gpu).length();
        assert!(
            diff < 5e-3,
            "BlendedSpace transport parity failed at a={a:?} b={b:?} v={v:?}: cpu={cpu:?} gpu={gpu:?} diff={diff}",
        );
    }
}

const SCENE_SDF_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    out[i] = vec4<f32>(loam_scene_sdf(points[i].xyz), 0.0, 0.0, 0.0);
}
"#;

const SCENE4_HIT_PROBE: &str = r#"
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let hit = loam_scene_at(points[i].xyz);
    out[i] = vec4<f32>(hit.dist, f32(hit.kind), 0.0, 0.0);
}
"#;

const PROBE_BALL_A: (Vec3, f32) = (Vec3::new(0.10, 0.00, 0.05), 0.22);
const PROBE_BALL_B: (Vec3, f32) = (Vec3::new(-0.15, 0.08, 0.00), 0.18);
const PROBE_BOX_HALF_EXTENTS: Vec3 = Vec3::new(0.20, 0.15, 0.25);
const PROBE_PLANE_OFFSET: f32 = -0.30;
const PROBE_SMOOTH_K: [f32; 2] = [0.12, 0.012];

fn probe_scenes() -> Vec<(&'static str, loam_scene::Scene)> {
    use loam_scene::{Scene, SceneNode};
    let ball_a = || SceneNode::sphere(PROBE_BALL_A.0, PROBE_BALL_A.1);
    let ball_b = || SceneNode::sphere(PROBE_BALL_B.0, PROBE_BALL_B.1);
    let box3 = || SceneNode::box_(PROBE_BOX_HALF_EXTENTS);
    let plane = || SceneNode::plane(Vec3::Y, PROBE_PLANE_OFFSET);
    vec![
        ("sphere", Scene::new(ball_a())),
        ("sphere union plane", Scene::new(ball_a().union(plane()))),
        (
            "sphere intersect box",
            Scene::new(ball_a().intersect(box3())),
        ),
        (
            "sphere minus sphere",
            Scene::new(ball_a().subtract(ball_b())),
        ),
        (
            "smooth union k=0.12",
            Scene::new(ball_a().smooth_union(ball_b(), PROBE_SMOOTH_K[0])),
        ),
        (
            "smooth union k=0.012",
            Scene::new(ball_a().smooth_union(ball_b(), PROBE_SMOOTH_K[1])),
        ),
        (
            "three-deep nested tree",
            Scene::new(
                ball_a()
                    .smooth_union(box3(), PROBE_SMOOTH_K[0])
                    .union(ball_b().subtract(plane()))
                    .intersect(SceneNode::cube(0.6)),
            ),
        ),
    ]
}

fn scene_probe_points(extent: f32) -> Vec<[f32; 4]> {
    let mut points: Vec<Vec3> = vec![
        Vec3::ZERO,
        PROBE_BALL_A.0,
        PROBE_BALL_B.0,
        PROBE_BALL_A.0 + Vec3::X * PROBE_BALL_A.1,
        PROBE_BALL_A.0 - Vec3::Y * PROBE_BALL_A.1,
        PROBE_BALL_B.0 + Vec3::Z * PROBE_BALL_B.1,
        (PROBE_BALL_A.0 + PROBE_BALL_B.0) * 0.5,
        PROBE_BOX_HALF_EXTENTS,
        PROBE_BOX_HALF_EXTENTS * Vec3::new(1.0, -1.0, 1.0),
        Vec3::new(0.0, PROBE_PLANE_OFFSET, 0.0),
    ];
    let mut state: u32 = 0x517E_5DF0;
    let mut next_f32 = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        (state as f32 / u32::MAX as f32) * 2.0 - 1.0
    };
    for _ in 0..118 {
        points.push(Vec3::new(
            next_f32() * extent,
            next_f32() * extent,
            next_f32() * extent,
        ));
    }
    points
        .into_iter()
        .map(|p| p.extend(0.0).to_array())
        .collect()
}

fn assert_scene_parity<S>(space: &S, label: &str, extent: f32, tolerance: f32) -> f32
where
    S: WgslSpace + Space<Point = Vec3, Vector = Vec3>,
{
    let points = scene_probe_points(extent);
    let mut worst = 0.0_f32;
    for (name, scene) in probe_scenes() {
        let source = assemble_source_with_scene(
            &space.wgsl_impl(),
            Some(&scene.to_wgsl(space)),
            SCENE_SDF_PROBE,
        );
        let rows: Vec<[f32; 4]> =
            pollster::block_on(run_compute_probe(&source, "loam-scene-gpu-probe", &points))
                .expect("scene GPU probe");
        for (point, row) in points.iter().zip(&rows) {
            let p = Vec3::new(point[0], point[1], point[2]);
            let cpu = scene.eval(space, p);
            let residual = (cpu - row[0]).abs();
            worst = worst.max(residual);
            assert!(
                residual <= tolerance,
                "{label}/{name}: CPU {cpu} vs GPU {} at {p:?} differ by {residual} \
                     (tolerance {tolerance})",
                row[0],
            );
        }
    }
    worst
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene_sdf_gpu_probe_matches_cpu_in_euclidean_r3() {
    let worst = assert_scene_parity(&EuclideanR3, "EuclideanR3", 0.9, 1e-5);
    println!("EuclideanR3 scene parity: worst residual {worst}");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene_sdf_gpu_probe_matches_cpu_in_hyperbolic_h3() {
    let worst = assert_scene_parity(&HyperbolicH3, "HyperbolicH3", 0.30, 2e-4);
    println!("HyperbolicH3 scene parity: worst residual {worst}");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene_sdf_gpu_probe_matches_cpu_in_spherical_s3() {
    let worst = assert_scene_parity(&SphericalS3, "SphericalS3", 0.30, 2e-4);
    println!("SphericalS3 scene parity: worst residual {worst}");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene_sdf_gpu_probe_bounds_blended_space_error() {
    let space = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-0.5, 0.5).unwrap(),
    );
    let worst = assert_scene_parity(&space, "BlendedSpace<E3,H3>", 0.30, 5e-2);
    println!("BlendedSpace<E3,H3> scene parity: worst residual {worst}");
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene4_hyperslice_gpu_probe_matches_cpu() {
    use glam::Vec4;
    use loam_scene::{Scene4, SceneNode4};

    const W_SLICE: f32 = 0.25;
    let scene = Scene4::new(
        SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.05, 0.0), 0.5)
            .union(SceneNode4::halfspace(Vec4::Y, -0.4))
            .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.1, 0.0, 0.1), 0.2))
            .intersect(SceneNode4::hypersphere(Vec4::ZERO, 1.2)),
    );
    let source = assemble_source_with_scene(
        &EuclideanR3.wgsl_impl(),
        Some(&scene.to_hyperslice_wgsl(&format!("{W_SLICE}"))),
        SCENE4_HIT_PROBE,
    );
    let points = scene_probe_points(0.9);
    let rows: Vec<[f32; 4]> =
        pollster::block_on(run_compute_probe(&source, "loam-scene4-gpu-probe", &points))
            .expect("scene4 GPU probe");

    let mut worst = 0.0_f32;
    for (point, row) in points.iter().zip(&rows) {
        let p = Vec3::new(point[0], point[1], point[2]);
        let (cpu_dist, cpu_kind) = scene.eval_at(p, W_SLICE, true);
        let residual = (cpu_dist - row[0]).abs();
        worst = worst.max(residual);
        assert!(
            residual <= 1e-5,
            "scene4: CPU {cpu_dist} vs GPU {} at {p:?} differ by {residual}",
            row[0],
        );
        assert_eq!(
            cpu_kind, row[1] as u32,
            "scene4: kind mismatch at {p:?} (CPU {cpu_kind}, GPU {})",
            row[1],
        );
    }
    println!("Scene4 hyperslice parity: worst residual {worst}");
}

fn assert_vec_near(what: &str, actual: [f32; 4], expected: Vec4, eps: f32) {
    for (lane, (actual, expected)) in actual.iter().zip(expected.to_array()).enumerate() {
        assert_near(&format!("{what}[{lane}]"), *actual, expected, eps);
    }
}

const PARITY_ABSOLUTE_FLOOR: f32 = 1e-6;

fn assert_near(what: &str, actual: f32, expected: f32, eps: f32) {
    let budget = eps * expected.abs() + PARITY_ABSOLUTE_FLOOR;
    assert!(
        (actual - expected).abs() <= budget,
        "{what}: GPU {actual} differs from CPU {expected} by more than {budget}",
    );
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn hyperslice_march_bound_respects_boolean_geometry_gpu_probe() {
    use loam_scene::{Scene4, SceneNode4};
    let plane = || SceneNode4::halfspace(Vec4::new(0.0, 0.6, 0.0, 0.8), -0.4);
    let sphere = || SceneNode4::hypersphere(Vec4::ZERO, 0.5);
    for (root, finite) in [
        (plane(), true),
        (sphere().union(plane()), true),
        (sphere().intersect(plane()), false),
        (sphere().subtract(plane()), false),
    ] {
        let scene = Scene4::new(root).to_hyperslice_wgsl("0.5");
        let source = format!(
            r#"{scene}
@group(0) @binding(0) var<storage, read> origins: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> bounds: array<f32>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    bounds[id.x] = loam_scene_max_t(origins[id.x].xyz, vec3<f32>(0.0, -1.0, 0.0));
}}
"#
        );
        let result: Vec<f32> = pollster::block_on(run_compute_probe(
            &source,
            "slice-bound",
            &[[0.0_f32, 2.0, 0.0, 0.0]],
        ))
        .expect("GPU probe");
        if finite {
            assert!(
                (result[0] - 10.0 / 3.0).abs() < 1e-5,
                "tilted floor bound: {}",
                result[0]
            );
        } else {
            assert!(
                result[0] > 1e8,
                "conditional floor clipped scene: {}",
                result[0]
            );
        }
    }
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn tiny_scene_constants_survive_shader_execution_gpu_probe() {
    use loam_scene::{Scene, SceneNode};
    let centre = Vec3::new(3.7e-7, -1.25e-7, 0.0);
    let sphere = Scene::new(SceneNode::sphere(centre, 1e-7));
    let source = assemble_source_with_scene(
        &EuclideanR3.wgsl_impl(),
        Some(&sphere.to_wgsl(&EuclideanR3)),
        SCENE_SDF_PROBE,
    );
    let points = [
        centre.extend(0.0).to_array(),
        (centre + Vec3::X * 3e-7).extend(0.0).to_array(),
    ];
    let rows: Vec<[f32; 4]> =
        pollster::block_on(run_compute_probe(&source, "tiny-sphere", &points)).expect("GPU probe");
    assert!((rows[0][0] + 1e-7).abs() < 1e-12);
    assert!((rows[1][0] - 2e-7).abs() < 1e-12);

    let k = 1e-20;
    let smooth = Scene::new(
        SceneNode::sphere(Vec3::ZERO, 0.0).smooth_union(SceneNode::sphere(Vec3::ZERO, 0.0), k),
    );
    let source = assemble_source_with_scene(
        &EuclideanR3.wgsl_impl(),
        Some(&smooth.to_wgsl(&EuclideanR3)),
        SCENE_SDF_PROBE,
    );
    let rows: Vec<[f32; 4]> =
        pollster::block_on(run_compute_probe(&source, "tiny-blend", &[[0.0_f32; 4]]))
            .expect("GPU probe");
    assert!((rows[0][0] + k * 0.25).abs() < k * 1e-5);
}

#[test]
#[ignore = "requires a working wgpu adapter"]
fn hyperslice_gate_changes_floor_distance_kind_and_march_bound_gpu_probe() {
    use loam_scene::{Scene4, SceneNode4};
    use loam_shape::Shape;
    let root = SceneNode4::hypersphere(Vec4::new(0.0, 2.0, 0.0, 0.5), 0.5)
        .union(SceneNode4::halfspace(Vec4::Y, 0.0))
        .union(SceneNode4::Leaf(Shape::sphere_at(Vec3::ZERO, 10.0)))
        .union(SceneNode4::Leaf(Shape::ConvexPolytope4D {
            vertices: vec![Vec4::ZERO; 5],
        }));
    for gate in [0.0_f32, 1.0] {
        let scene = Scene4::new(root.clone()).to_hyperslice_wgsl_gated("0.5", &format!("{gate:?}"));
        let source = format!(
            r#"{scene}
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> out: array<vec4<f32>>;
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
    let hit = loam_scene_at(points[id.x].xyz);
    out[id.x] = vec4<f32>(hit.dist, f32(hit.kind), loam_scene_max_t(vec3<f32>(0.0, 3.0, 0.0), vec3<f32>(0.0, -1.0, 0.0)), 0.0);
}}
"#
        );
        let rows: Vec<[f32; 4]> = pollster::block_on(run_compute_probe(
            &source,
            "floor-gate",
            &[[0.0_f32, -1.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]],
        ))
        .expect("GPU probe");
        assert_eq!(rows[1][0], -0.5);
        assert_eq!(rows[1][1], 0.0);
        if gate == 1.0 {
            assert_eq!(&rows[0][..3], &[-1.0, 1.0, 3.0]);
        } else {
            assert_eq!(&rows[0][..2], &[2.5, 0.0]);
            assert!(rows[0][2] > 1e8);
        }
    }
}
