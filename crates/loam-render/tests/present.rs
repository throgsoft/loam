use glam::{Vec2, Vec3, Vec4};
use loam_math::{EuclideanR4, HyperbolicH3, Iso3};
use loam_render::present::Presenter;
use loam_runtime::{
    BridgeSpec, DomainBuilder, Entity, Instance, Klein, LogCapacity, Material, Placement, Pose,
    PreparedGeometry, Projection4, Publication, Rigid, Section4, Session, SimConfig, SpawnBundle,
    ViewSpec,
};
use wgpu::*;

const SIZE: u32 = 128;
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;
const BACKGROUND: Color = Color {
    r: 0.02,
    g: 0.02,
    b: 0.03,
    a: 1.0,
};
const INK_MIN: u8 = 180;
const BACKGROUND_MAX: u8 = 60;
const FOCAL: f32 = 2.0;
const REACH: f32 = 0.3;
const LANDMARK_R4: Vec4 = Vec4::new(1.5, 0.0, -4.0, 0.0);
const LANDMARK_H3: Vec3 = Vec3::new(-0.12, 0.0, -0.55);
const EMPTY: (u32, u32) = (10, 118);
const BRIDGE_SCALE: f32 = 0.25;
const BRIDGE_SHIFT: f32 = 0.3;
const SECTION_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);
const PLACED_NDC: f32 = 0.519_615_25;
const TIP_NDC: f32 = 0.649_519;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Landmarks {
        seen: Store<u32>,
    }
}

fn cross4(reach: f32) -> PreparedGeometry {
    PreparedGeometry::Lines4 {
        segments: (0..3)
            .map(|axis| {
                let mut arm = [0.0; 4];
                arm[axis] = reach;
                let mut back = [0.0; 4];
                back[axis] = -reach;
                [back, arm]
            })
            .collect(),
    }
}

fn cross3(reach: f32) -> PreparedGeometry {
    PreparedGeometry::Lines3 {
        segments: (0..3)
            .map(|axis| {
                let mut arm = [0.0; 3];
                arm[axis] = reach;
                let mut back = [0.0; 3];
                back[axis] = -reach;
                [back, arm]
            })
            .collect(),
    }
}

fn twospace() -> (Session<Landmarks>, [Entity; 2]) {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let h3 = session
        .register_domain(DomainBuilder::new("h3", HyperbolicH3).tracked(LogCapacity::default()));
    let edges4 = session.prepare(cross4(REACH));
    let edges3 = session.prepare(cross3(REACH * 0.5));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 1.0], 3.0));
    let root = session.views().root();
    let landmarks = session.dispatch(|d| {
        let landmark4 = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(LANDMARK_R4))
                    .instance(Instance::new(edges4, white)),
            )
            .unwrap();
        let landmark3 = d
            .spawn(
                SpawnBundle::new()
                    .at(h3, Pose::at(LANDMARK_H3))
                    .instance(Instance::new(edges3, white)),
            )
            .unwrap();
        let walker4 = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        let walker3 = d
            .spawn(SpawnBundle::new().at(h3, Pose::at(Vec3::ZERO)))
            .unwrap();
        d.domains.typed(r4).unwrap().add_view(ViewSpec::new(
            root,
            walker4,
            Projection4 { focal: FOCAL },
        ));
        d.domains
            .typed(h3)
            .unwrap()
            .add_view(ViewSpec::new(root, walker3, Klein));
        [landmark4, landmark3]
    });
    (session, landmarks)
}

async fn request_device() -> (Device, Queue) {
    let instance = wgpu::Instance::default();
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
            label: Some("present"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("request_device")
}

fn render(publication: &Publication<Landmarks>) -> Vec<u8> {
    let (device, queue) = pollster::block_on(request_device());
    let mut presenter = Presenter::new(COLOR_FORMAT, 1);
    let color = device.create_texture(&TextureDescriptor {
        label: Some("present color"),
        size: Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: COLOR_FORMAT,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&TextureViewDescriptor::default());
    presenter.upload(
        &device,
        &queue,
        &loam_runtime::Eye::default(),
        Vec2::splat(SIZE as f32),
        &publication.views,
    );
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("present"),
    });
    presenter.record(&device, &mut encoder, &view, (SIZE, SIZE), BACKGROUND);
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("present readback"),
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
        Extent3d {
            width: SIZE,
            height: SIZE,
            depth_or_array_layers: 1,
        },
    );
    queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::wait_indefinitely())
        .expect("readback poll");
    readback.slice(..).get_mapped_range().to_vec()
}

fn pixel(ndc: [f32; 2]) -> (u32, u32) {
    (
        ((ndc[0] + 1.0) * 0.5 * SIZE as f32) as u32,
        ((1.0 - ndc[1]) * 0.5 * SIZE as f32) as u32,
    )
}

fn texel(pixels: &[u8], (x, y): (u32, u32)) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn published_landmarks_are_absent_from_their_own_ndc_positions_gpu_probe() {
    let (mut session, landmarks) = twospace();
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();

    let pixels = render(&publication);
    for landmark in landmarks {
        let image_point = publication
            .views
            .iter()
            .flat_map(|view| view.records.instances.rows())
            .find(|record| record.entity == landmark)
            .map(|record| record.image_point)
            .expect("landmark published");
        let [x, y] = session.views().ndc(image_point).expect("landmark in front");
        let pixel = (
            ((x + 1.0) * 0.5 * SIZE as f32) as u32,
            ((1.0 - y) * 0.5 * SIZE as f32) as u32,
        );
        let drawn = texel(&pixels, pixel);
        assert!(
            drawn[0] >= INK_MIN,
            "{landmark:?} at {pixel:?} shows {drawn:?}, not its lines"
        );
    }
    let empty = texel(&pixels, EMPTY);
    assert!(
        empty[0] <= BACKGROUND_MAX,
        "the pixel at {EMPTY:?} shows {empty:?}, not the background"
    );
}

fn bridged() -> Session<Landmarks> {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let edges = session.prepare(cross4(REACH));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 1.0], 3.0));
    let root = session.views().root();
    let (section, anchor) = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        let anchor = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(SECTION_AT))
                    .instance(Instance::new(edges, white)),
            )
            .unwrap();
        let section =
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }));
        (section, anchor)
    });
    session
        .bridge(BridgeSpec {
            anchor,
            into: root,
            source: r4.id(),
            view: section,
            placement: Placement::Rigid(Rigid {
                pose: Iso3::from_translation(Vec3::new(BRIDGE_SHIFT, 0.0, 0.0)),
                scale: BRIDGE_SCALE,
            }),
        })
        .unwrap();
    session
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_bridged_view_rasterizes_where_its_placement_did_not_put_it_gpu_probe() {
    let mut session = bridged();
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let pixels = render(&publication);

    for (name, ndc) in [("center", PLACED_NDC), ("tip", TIP_NDC)] {
        let at = pixel([ndc, 0.0]);
        let drawn = texel(&pixels, at);
        assert!(
            drawn[0] >= INK_MIN,
            "the placed {name} at {at:?} shows {drawn:?}, not its lines"
        );
    }
    let unplaced = pixel([0.0, 0.0]);
    let drawn = texel(&pixels, unplaced);
    assert!(
        drawn[0] <= BACKGROUND_MAX,
        "the unplaced center at {unplaced:?} shows {drawn:?}, so the placement was dropped"
    );
}
