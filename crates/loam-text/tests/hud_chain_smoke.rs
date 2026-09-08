//! Only the wgpu half needs a device, and only it is `#[ignore]`d. The
//! `gpu_probe` suffix is what CI's software-adapter job selects on.

use loam_text::TextRenderer;
use wgpu::{Device, Queue, TextureFormat, TextureView};

const TARGET_FORMAT: TextureFormat = TextureFormat::Rgba8UnormSrgb;
const VIEWPORT: [f32; 2] = [256.0, 64.0];

fn draw_hud_frame(
    device: &Device,
    queue: &Queue,
    view: &TextureView,
    font_bytes: &[u8],
) -> anyhow::Result<()> {
    let mut text = TextRenderer::new(device, queue, TARGET_FORMAT, font_bytes, 48.0, 1)?;
    text.queue("fps 240", [16.0, 16.0], 32.0, [1.0, 1.0, 1.0, 1.0]);
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("hud-chain-smoke frame"),
    });
    text.record(device, queue, &mut encoder, view, VIEWPORT);
    queue.submit(Some(encoder.finish()));
    Ok(())
}

async fn request_device() -> Result<(Device, Queue), String> {
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
            label: Some("hud-chain-smoke"),
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
fn hud_frame_renders_into_an_offscreen_target_gpu_probe() {
    let font_bytes = include_bytes!("../../hero/fonts/lmroman10-bold.otf");
    let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");

    let target = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("hud-chain-smoke target"),
        size: wgpu::Extent3d {
            width: VIEWPORT[0] as u32,
            height: VIEWPORT[1] as u32,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: TARGET_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = target.create_view(&wgpu::TextureViewDescriptor::default());

    draw_hud_frame(&device, &queue, &view, font_bytes).expect("HUD frame should render");
    let bytes_per_row = VIEWPORT[0] as u32 * 4;
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("HUD pixels"),
        size: (bytes_per_row * VIEWPORT[1] as u32) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &target,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        target.size(),
    );
    queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .expect("readback");
    let pixels = readback.slice(..).get_mapped_range();
    assert!(
        pixels.chunks_exact(4).filter(|p| p[0] > 100).count() > 100,
        "glyph ink is missing"
    );
    assert!(
        pixels[..bytes_per_row as usize].iter().all(|&b| b == 0),
        "glyphs escaped the text box"
    );
}
