use glam::Vec2;
use loam_render::view::{root_view_projection, DEPTH_CLEAR, DEPTH_FORMAT};
use loam_render::work::{BulkBuffers, ComputeWork, Readbacks};
use loam_render::{DepthConvention, DepthMode, PointInstance, PointRasterNode};
use loam_runtime::{
    BulkSpec, Eye, Input, Phase, Readback, Schedule, Session, SimConfig, SnapshotPolicy, WorkItem,
};
use wgpu::{
    Color, Extent3d, LoadOp, MapMode, Operations, Origin3d, PollType, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, StoreOp, TexelCopyBufferInfo,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureUsages,
};

const SIZE: u32 = 64;
const REACH: f32 = 13.0;

const ADVANCE_WGSL: &str = r#"
struct Particle {
    pos: vec3<f32>,
    radius_px: f32,
    color: vec4<f32>,
};
@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&particles)) {
        return;
    }
    particles[id.x].pos.z = particles[id.x].pos.z - 13.0;
}
"#;

async fn request_device() -> (wgpu::Device, wgpu::Queue) {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter");
    adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("bulk work probe"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bare {}
}

fn seeded() -> [PointInstance; 2] {
    [
        PointInstance {
            pos: [0.0, 0.0, REACH - 3.0],
            radius_px: 12.0,
            color: [1.0, 0.0, 0.0, 1.0],
        },
        PointInstance {
            pos: [0.0, 0.0, REACH - 1.0],
            radius_px: 12.0,
            color: [0.0, 1.0, 0.0, 1.0],
        },
    ]
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_compute_stepped_bulk_store_reaches_the_point_raster_without_an_entity_per_element_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let seed = seeded();
    let element_size = std::mem::size_of::<PointInstance>() as u32;

    let mut session = Session::new(Bare::default(), SimConfig::default());
    let particles = session.register_bulk(BulkSpec {
        name: "particles",
        element_size,
        count: seed.len() as u32,
        readback: Readback::Optional,
        snapshot: SnapshotPolicy::Reinitializable,
        schedule: Schedule::InStep,
    });
    session.work(
        Phase::Simulation,
        WorkItem::new("advance", Schedule::InStep, Readback::Optional).writes(particles),
    );
    let growth = session.boundary(Input::default()).unwrap();
    assert_eq!(growth.bulk_elements, seed.len());

    let mut buffers = BulkBuffers::default();
    for (id, spec) in session.bulk().iter() {
        buffers.ensure(&device, id, spec);
    }
    buffers.write(&queue, particles, bytemuck::cast_slice(&seed));
    let bulk = buffers
        .get(particles)
        .expect("the bulk buffer exists")
        .clone();
    let advance = ComputeWork::new(&device, "advance", ADVANCE_WGSL, "main", &[&bulk]);

    let mut readbacks = Readbacks::default();
    let mut encoder = device.create_command_encoder(&Default::default());
    session.tick().unwrap();
    let issued = session.issue_work(|order| {
        advance.record(&mut encoder, "advance", 1);
        readbacks.request(
            &device,
            &mut encoder,
            order.request,
            &bulk,
            u64::from(element_size) * seed.len() as u64,
        );
    });
    assert_eq!(issued, 1);

    let color = device.create_texture(&TextureDescriptor {
        label: Some("bulk work color"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba8Unorm,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let color_view = color.create_view(&Default::default());
    let depth = device.create_texture(&TextureDescriptor {
        label: Some("bulk work depth"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let depth_view = depth.create_view(&Default::default());

    let mut node = PointRasterNode::new(
        &device,
        TextureFormat::Rgba8Unorm,
        DepthMode::ReadWrite {
            format: DEPTH_FORMAT,
        },
        DepthConvention::ReversedZ,
        1,
    );
    let eye = Eye::default();
    node.set_camera(&queue, root_view_projection(&eye), Vec2::splat(SIZE as f32));
    node.draw_buffer(bulk.clone(), seed.len() as u32);

    encoder.begin_render_pass(&RenderPassDescriptor {
        label: Some("bulk work clear"),
        color_attachments: &[Some(RenderPassColorAttachment {
            view: &color_view,
            depth_slice: None,
            resolve_target: None,
            ops: Operations {
                load: LoadOp::Clear(Color::BLACK),
                store: StoreOp::Store,
            },
        })],
        depth_stencil_attachment: Some(RenderPassDepthStencilAttachment {
            view: &depth_view,
            depth_ops: Some(Operations {
                load: LoadOp::Clear(DEPTH_CLEAR),
                store: StoreOp::Store,
            }),
            stencil_ops: None,
        }),
        timestamp_writes: None,
        occlusion_query_set: None,
    });
    node.record(&mut encoder, &color_view, Some(&depth_view), None);

    let pixels = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("bulk work pixels"),
        size: u64::from(SIZE * SIZE * 4),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
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
            buffer: &pixels,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(SIZE * 4),
                rows_per_image: None,
            },
        },
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    readbacks.after_submit();

    pixels.slice(..).map_async(MapMode::Read, |_| {});
    device.poll(PollType::wait_indefinitely()).expect("poll");
    let image = pixels.slice(..).get_mapped_range().to_vec();
    let middle = ((SIZE / 2 * SIZE + SIZE / 2) * 4) as usize;
    let centre = [image[middle], image[middle + 1], image[middle + 2]];
    assert_eq!(
        centre,
        [0, 255, 0],
        "the compute step or the reversed-Z compare did not put the nearer particle at the centre"
    );

    let mut landed = Vec::new();
    readbacks.poll(&device, |request, rows| {
        session.land_readback(request, rows);
        landed.push(request);
    });
    assert_eq!(landed.len(), 1);
    let stepped: Vec<[f32; 3]> = session
        .readbacks()
        .flat_map(|result| {
            bytemuck::cast_slice::<u8, PointInstance>(result.rows)
                .iter()
                .map(|row| row.pos)
                .collect::<Vec<_>>()
        })
        .collect();
    assert_eq!(stepped, [[0.0, 0.0, -3.0], [0.0, 0.0, -1.0]]);
    assert_eq!(session.entities().len(), 0);
}
