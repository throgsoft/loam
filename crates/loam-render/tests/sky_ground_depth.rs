use glam::Vec3;
use loam_render::view::{self, DEPTH_CLEAR, DEPTH_FORMAT};
use loam_render::{DepthConvention, Ground, SkyGroundNode, SkyGroundUniforms, Viewport};
use wgpu::*;

const SIZE: u32 = 64;
const NEAR: f32 = 0.05;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const GROUND_Y: f32 = -1.0;
const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const SKY_PIXEL: (u32, u32) = (32, 8);
const GROUND_PIXEL: (u32, u32) = (32, 48);

fn ndc(pixel: (u32, u32)) -> (f32, f32) {
    (
        ((pixel.0 as f32 + 0.5) / SIZE as f32) * 2.0 - 1.0,
        1.0 - ((pixel.1 as f32 + 0.5) / SIZE as f32) * 2.0,
    )
}

fn ground_depth(pixel: (u32, u32)) -> f32 {
    let focal = 1.0 / (FOV_Y * 0.5).tan();
    let (x, y) = ndc(pixel);
    let origin = Vec3::new(x * NEAR / focal, y * NEAR / focal, -NEAR);
    let direction = Vec3::new(x / focal, y / focal, -1.0).normalize();
    let t = (GROUND_Y - origin.y) / direction.y;
    assert!(t > 0.0, "{pixel:?} looks away from the ground");
    NEAR / -(origin + direction * t).z
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
            label: Some("sky-ground-depth"),
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

fn attachment(device: &Device, format: TextureFormat, label: &str) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(label),
        size: extent(),
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
}

fn reversed_z_depths() -> Vec<f32> {
    let (device, queue) = pollster::block_on(request_device());
    let color = attachment(&device, TARGET_FORMAT, "sky-ground-depth color");
    let depth = attachment(&device, DEPTH_FORMAT, "sky-ground-depth depth");
    let color_view = color.create_view(&TextureViewDescriptor::default());
    let depth_view = depth.create_view(&TextureViewDescriptor::default());

    let node = SkyGroundNode::new(
        &device,
        TARGET_FORMAT,
        DEPTH_FORMAT,
        DepthConvention::ReversedZ,
    );
    node.set_uniforms(
        &queue,
        &SkyGroundUniforms::new(
            view::root_projection(FOV_Y, 1.0, NEAR),
            Viewport::full([SIZE, SIZE]),
            Ground {
                y: GROUND_Y,
                dark: loam_render::sky_ground::GROUND_DARK_GREY,
                light: loam_render::sky_ground::GROUND_LIGHT_GREY,
                fog_per_unit: 0.0,
                visible: true,
            },
        ),
    );

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("sky-ground-depth"),
    });
    node.record(&mut encoder, &color_view, &depth_view, None);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("sky-ground-depth readback"),
        size: (SIZE * SIZE * 4) as u64,
        usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &depth,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::DepthOnly,
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
    bytemuck::cast_slice::<u8, f32>(&readback.slice(..).get_mapped_range()).to_vec()
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn reversed_z_background_clears_far_and_the_ground_writes_its_projective_depth_gpu_probe() {
    let depths = reversed_z_depths();
    let at = |(x, y): (u32, u32)| depths[(y * SIZE + x) as usize];

    assert_eq!(
        at(SKY_PIXEL),
        DEPTH_CLEAR,
        "the reversed-Z background at {SKY_PIXEL:?} must sit at the far plane"
    );
    let expected = ground_depth(GROUND_PIXEL);
    let written = at(GROUND_PIXEL);
    assert!(
        (written - expected).abs() <= 1.0e-5,
        "the ground at {GROUND_PIXEL:?} wrote {written}, not the projective depth {expected}"
    );
}
