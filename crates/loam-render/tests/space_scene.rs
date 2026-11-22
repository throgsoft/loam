use loam_render::shader::{assemble_wgsl, validate_wgsl};

use bytemuck::Pod;
use glam::{Vec3, Vec4};
use loam_math::{EuclideanR3, WgslSpace};

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

#[test]
#[ignore = "requires a working wgpu adapter"]
fn scene4_hyperslice_gpu_probe_matches_cpu() {
    use loam_render::raymarch::scene::{Scene4, SceneNode4};

    const W_SLICE: f32 = 0.25;
    let scene = Scene4::new(
        SceneNode4::hypersphere(Vec4::new(0.1, 0.0, -0.05, 0.0), 0.5)
            .union(SceneNode4::halfspace(Vec4::Y, -0.4))
            .subtract(SceneNode4::hypersphere(Vec4::new(0.3, 0.1, 0.0, 0.1), 0.2))
            .intersect(SceneNode4::hypersphere(Vec4::ZERO, 1.2)),
    );
    let prelude = EuclideanR3.wgsl_impl();
    let scene_source = scene.to_hyperslice_wgsl(&format!("{W_SLICE}"));
    let source = assemble_wgsl(&[prelude.as_ref(), scene_source.as_str(), SCENE4_HIT_PROBE]);
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

#[test]
#[ignore = "requires a working wgpu adapter"]
fn hyperslice_march_bound_respects_boolean_geometry_and_the_floor_gate_gpu_probe() {
    use loam_render::raymarch::scene::{Scene4, SceneNode4};
    use loam_shape::Shape;
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
