use glam::{Vec2, Vec3};
use loam_math::Rotor4;
use loam_render::view;
use loam_render::{DepthConvention, DepthMode, LineRasterStaticR4Node};
use loam_shape::LineMesh;
use wgpu::*;

const SIZE: u32 = 64;
const NEAR: f32 = 0.05;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const WIDTH_PX: f32 = 6.0;
const FRONT: Vec3 = Vec3::new(0.5, 0.0, -2.0);
const BEHIND: Vec3 = Vec3::new(-0.1, 0.0, 0.34);
const INK_MIN: u8 = 120;
const BACKGROUND_MAX: u8 = 10;

fn crossing_column() -> u32 {
    let t = (-NEAR - FRONT.z) / (BEHIND.z - FRONT.z);
    let crossing = FRONT + (BEHIND - FRONT) * t;
    let ndc_x = (1.0 / (FOV_Y * 0.5).tan()) * crossing.x / -crossing.z;
    ((ndc_x + 1.0) * 0.5 * SIZE as f32).round() as u32
}

async fn request_device() -> (Device, Queue) {
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("request_adapter");
    adapter
        .request_device(&DeviceDescriptor {
            label: Some("near-plane-clip"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("request_device")
}

fn extent() -> Extent3d {
    Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    }
}

fn render_crossing_segment() -> Vec<u8> {
    let (device, queue) = pollster::block_on(request_device());
    let mut node = LineRasterStaticR4Node::new(
        &device,
        TextureFormat::Rgba8Unorm,
        DepthMode::Off,
        DepthConvention::ReversedZ,
    );
    let mut mesh = LineMesh::<4>::default();
    mesh.segments
        .push((FRONT.extend(0.0).to_array(), BEHIND.extend(0.0).to_array()));
    mesh.colors.push(([1.0; 4], [1.0; 4]));
    mesh.widths.push(WIDTH_PX);
    node.upload_mesh(&device, &queue, &mesh);
    node.set_transform(
        &queue,
        Rotor4::IDENTITY,
        view::root_projection(FOV_Y, 1.0, NEAR),
        Vec2::splat(SIZE as f32),
        2.0,
    );

    let color = device.create_texture(&TextureDescriptor {
        label: Some("near-plane-clip color"),
        size: extent(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("near-plane-clip"),
    });
    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("near-plane-clip clear"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    node.record(&mut encoder, &view, None, None);

    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("near-plane-clip readback"),
        size: (SIZE * SIZE * 4) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &color,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &readback,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: None,
            },
        },
        extent(),
    );
    queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("readback poll");
    readback.slice(..).get_mapped_range().to_vec()
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_reversed_z_segment_ends_on_the_near_plane_at_full_coverage_gpu_probe() {
    let pixels = render_crossing_segment();
    let row = SIZE / 2;
    let red = |column: u32| pixels[((row * SIZE + column) * 4) as usize];
    let crossing = crossing_column();

    assert!(
        red(crossing - 1) <= BACKGROUND_MAX,
        "column {} past the crossing shows {}, so the segment reaches beyond the near plane",
        crossing - 1,
        red(crossing - 1)
    );
    assert!(
        red(crossing + 2) >= INK_MIN,
        "column {} shows {}, so the segment fades out before the near plane",
        crossing + 2,
        red(crossing + 2)
    );
}
