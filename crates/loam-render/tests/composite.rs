use loam_render::composite::CompositeNode;
use wgpu::{
    Color, CommandEncoderDescriptor, Device, Extent3d, LoadOp, MapMode, Operations, Origin3d,
    PollType, Queue, RenderPassColorAttachment, RenderPassDescriptor, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages, TextureView, TextureViewDescriptor,
};

const SIZE: u32 = 64;

const SWAP_FORMAT: TextureFormat = TextureFormat::Bgra8Unorm;
const SCENE_FORMAT: TextureFormat = TextureFormat::Bgra8UnormSrgb;

// IEC 61966-2-1; expected bytes below are BGRA after sRGB encoding.
const SCENE_LINEAR: [f64; 3] = [0.25, 0.5, 0.75];

const SCENE_LINEAR_AFTER_UI: [f64; 3] = [0.9, 0.1, 0.4];

const TOLERANCE: u8 = 1;

struct Probe {
    device: Device,
    queue: Queue,
    scene_view: TextureView,
    swap: Texture,
    swap_view: TextureView,
    composite: CompositeNode,
}

impl Probe {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(request_device()).expect("wgpu device");
        let scene = create_target(&device, SCENE_FORMAT, "scene");
        let swap = create_target(&device, SWAP_FORMAT, "swap");
        let scene_view = scene.create_view(&TextureViewDescriptor::default());
        let swap_view = swap.create_view(&TextureViewDescriptor::default());
        let mut composite = CompositeNode::new(&device, SWAP_FORMAT);
        composite.set_scene_view(&device, &scene_view);
        Self {
            device,
            queue,
            scene_view,
            swap,
            swap_view,
            composite,
        }
    }

    fn fill(&self, view: &TextureView, color: Color) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("capture-order probe fill"),
            });
        encoder.begin_render_pass(&RenderPassDescriptor {
            label: Some("capture-order probe fill"),
            color_attachments: &[Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: Operations {
                    load: LoadOp::Clear(color),
                    store: StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
        self.queue.submit(Some(encoder.finish()));
    }

    fn fill_scene(&self, linear: [f64; 3]) {
        self.fill(
            &self.scene_view,
            Color {
                r: linear[0],
                g: linear[1],
                b: linear[2],
                a: 1.0,
            },
        );
    }

    fn record_scene(&self) {
        self.fill(&self.swap_view, unwritten_color());
        self.fill_scene(SCENE_LINEAR);
    }

    fn run_composite(&self) {
        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("capture-order probe composite"),
            });
        self.composite.run(&mut encoder, &self.swap_view);
        self.queue.submit(Some(encoder.finish()));
    }
}

fn unwritten_color() -> Color {
    Color {
        r: 1.0,
        g: 0.0,
        b: 1.0,
        a: 1.0,
    }
}

fn create_target(device: &Device, format: TextureFormat, tag: &str) -> Texture {
    device.create_texture(&TextureDescriptor {
        label: Some(&format!("capture-order probe {tag}")),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format,
        usage: TextureUsages::RENDER_ATTACHMENT
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::COPY_SRC,
        view_formats: &[],
    })
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
            label: Some("capture-tap-swapchain-order"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .map_err(|e| format!("request_device failed: {e}"))
}

fn read_back(probe: &Probe, texture: &Texture) -> Vec<u8> {
    let bytes_per_row = SIZE * 4;
    let buffer = probe.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("capture-order probe readback"),
        size: (bytes_per_row * SIZE) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = probe
        .device
        .create_command_encoder(&CommandEncoderDescriptor {
            label: Some("capture-order probe readback"),
        });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(bytes_per_row),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    probe.queue.submit(Some(encoder.finish()));
    let slice = buffer.slice(..);
    slice.map_async(MapMode::Read, |_| {});
    probe
        .device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let data = slice.get_mapped_range().to_vec();
    buffer.unmap();
    data
}

fn assert_every_texel(pixels: &[u8], expected: [u8; 4], tolerance: u8, what: &str) {
    for (index, texel) in pixels.chunks_exact(4).enumerate() {
        for channel in 0..4 {
            let delta = texel[channel].abs_diff(expected[channel]);
            assert!(
                delta <= tolerance,
                "{what}: texel {index} channel {channel} is {}, expected {} \
                 (delta {delta} > {tolerance}); full texel {texel:?}",
                texel[channel],
                expected[channel],
            );
        }
    }
}

#[test]
#[ignore = "requires a working wgpu adapter; CI runs software-adapter probes"]
fn composite_encodes_channels_and_refreshes_after_scene_changes_gpu_probe() {
    let probe = Probe::new();
    probe.record_scene();
    probe.run_composite();
    assert_every_texel(
        &read_back(&probe, &probe.swap),
        [225, 188, 137, 255],
        TOLERANCE,
        "initial composite",
    );
    probe.fill_scene(SCENE_LINEAR_AFTER_UI);
    probe.run_composite();
    assert_every_texel(
        &read_back(&probe, &probe.swap),
        [170, 89, 243, 255],
        TOLERANCE,
        "updated composite",
    );
}
