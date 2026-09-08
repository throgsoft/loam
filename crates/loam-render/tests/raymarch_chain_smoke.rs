//! The 3D raymarch chain, `loam_scene::Scene` -> WGSL -> `ShaderDb` ->
//! `RayMarchNode`: nothing else in the workspace instantiates it. Headless
//! except the `gpu_probe`.

use glam::Vec3;
use loam_math::EuclideanR3;
use loam_render::shader::ShaderDb;
use loam_render::{RayMarchNode, RayMarchUniforms};
use loam_scene::{Scene, SceneNode};
use wgpu::{Device, TextureFormat};

fn probe_scene() -> Scene {
    Scene::new(SceneNode::sphere(Vec3::ZERO, 0.5).union(SceneNode::plane(Vec3::Y, -0.5)))
}

// Mirror of `RayMarchUniforms`; WGSL's `vec3` alignment matches the hand padding.
const UNIFORMS_WGSL: &str = r#"
struct RayMarchUniforms {
    camera_pos: vec3<f32>,
    camera_forward: vec3<f32>,
    camera_right: vec3<f32>,
    camera_up: vec3<f32>,
    fov_y_tan: f32,
    resolution: vec2<f32>,
    time: f32,
    tick: f32,
    params: vec4<f32>,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

fn ray_direction(pos: vec4<f32>) -> vec3<f32> {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    return u.camera_forward
        + u.camera_right * ndc.x * u.fov_y_tan
        + u.camera_up * ndc.y * u.fov_y_tan;
}
"#;

const GEODESIC_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let dir = loam_safe_normalize(ray_direction(pos), vec3<f32>(0.0, 0.0, -1.0));
    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    if hit.w < 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }
    let n = loam_estimate_normal(hit.xyz, 1.0);
    return vec4<f32>(n * 0.5 + vec3<f32>(0.5), 1.0);
}
"#;

const SCENE_SHADING_WGSL: &str = r#"
@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    let p = u.camera_pos + ray_direction(pos);
    let lit = step(loam_scene_sdf(p), 0.0);
    return vec4<f32>(vec3<f32>(lit), 1.0);
}
"#;

fn geodesic_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{GEODESIC_SHADING_WGSL}")
}

fn scene_user_shader() -> String {
    format!("{UNIFORMS_WGSL}{SCENE_SHADING_WGSL}")
}

fn build_geodesic_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<RayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_geodesic_scene(
        ShaderDb::ROOT_OWNER,
        shader_path,
        &scene.to_wgsl(&EuclideanR3),
        &EuclideanR3,
    )?;
    Ok(RayMarchNode::new(device, surface_format, db.module(id), 1))
}

fn build_raymarch_node(
    device: &Device,
    surface_format: TextureFormat,
    shader_path: &std::path::Path,
    scene: &Scene,
) -> anyhow::Result<RayMarchNode> {
    let mut db = ShaderDb::new(device.clone());
    let id = db.load_with_scene(
        ShaderDb::ROOT_OWNER,
        shader_path,
        &scene.to_wgsl(&EuclideanR3),
        &EuclideanR3,
    )?;
    Ok(RayMarchNode::new(device, surface_format, db.module(id), 1))
}

fn frame_uniforms() -> RayMarchUniforms {
    RayMarchUniforms {
        resolution: [64.0, 64.0],
        camera_pos: [0.0, 0.0, 1.0],
        ..Default::default()
    }
}

async fn request_device() -> Result<(wgpu::Device, wgpu::Queue), String> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .map_err(|e| format!("request_adapter failed: {e}"))?;
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("raymarch-chain-smoke"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn scene_and_geodesic_shaders_draw_through_shader_db_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
    let surface_format = TextureFormat::Rgba8UnormSrgb;
    let scene = probe_scene();

    let dir = tempfile::tempdir().expect("tempdir");

    let geodesic_path = dir.path().join("geodesic.wgsl");
    std::fs::write(&geodesic_path, geodesic_user_shader()).expect("write geodesic shader");
    let mut geodesic = build_geodesic_node(&device, surface_format, &geodesic_path, &scene)
        .expect("geodesic chain should build a pipeline");
    geodesic.set_uniforms(&queue, frame_uniforms());
    let pixel = center_pixel(&device, &queue, &mut geodesic);
    assert!(
        pixel[2] > 200 && pixel[2] > pixel[0],
        "sphere normal pixel: {pixel:?}"
    );

    let scene_path = dir.path().join("scene.wgsl");
    std::fs::write(&scene_path, scene_user_shader()).expect("write scene shader");
    let mut plain = build_raymarch_node(&device, surface_format, &scene_path, &scene)
        .expect("scene chain should build a pipeline");
    plain.set_uniforms(&queue, frame_uniforms());
    assert_eq!(center_pixel(&device, &queue, &mut plain), [255; 4]);
}

fn center_pixel(device: &Device, queue: &wgpu::Queue, node: &mut RayMarchNode) -> [u8; 4] {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("raymarch output"),
        size: wgpu::Extent3d {
            width: 64,
            height: 64,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&Default::default());
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("raymarch readback"),
        size: 256,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    node.record_in_viewport(&mut encoder, &view, loam_render::Viewport::full([64, 64]));
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d { x: 32, y: 32, z: 0 },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(256),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("readback");
    let bytes = readback.slice(..).get_mapped_range();
    [bytes[0], bytes[1], bytes[2], bytes[3]]
}
