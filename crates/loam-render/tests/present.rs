use glam::{Vec2, Vec3, Vec4};
use loam_math::{EuclideanR4, HyperbolicH3, Iso3};
use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::pass::FramePass;
use loam_render::present::Presenter;
use loam_render::SkyGroundPass;
use loam_runtime::{
    Backdrop, BridgeSpec, DomainBuilder, Entity, Eye, Ground, Instance, Klein, LogCapacity,
    Material, Phase, Placement, PointRecord, Pose, PreparedGeometry, Projection4, Publication,
    Rigid, Section4, SegmentRecord, Session, SimConfig, Sky, SpawnBundle, ViewSpec,
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
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, walker4, Projection4 { focal: FOCAL }))
            .unwrap();
        d.domains
            .typed(h3)
            .unwrap()
            .add_view(ViewSpec::new(root, walker3, Klein))
            .unwrap();
        [landmark4, landmark3]
    });
    (session, landmarks)
}

fn render(publication: &Publication) -> Vec<u8> {
    render_with(publication, Vec::new())
}

fn render_with(publication: &Publication, passes: Vec<Box<dyn FramePass>>) -> Vec<u8> {
    let gpu = pollster::block_on(GpuContext::new(
        wgpu::Instance::default(),
        FeatureRequest::default(),
        None,
    ))
    .expect("gpu context");
    let (device, queue) = (&gpu.device, &gpu.queue);
    let mut presenter = Presenter::new(COLOR_FORMAT).expect("presenter");
    for pass in passes {
        presenter.register_pass(pass).expect("registered the pass");
    }
    presenter.attach(&gpu).expect("attached the passes");
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
        device,
        queue,
        &loam_runtime::Eye::default(),
        Vec2::splat(SIZE as f32),
        &publication.views,
    );
    presenter.upload_overlay(&Eye::default(), &publication.overlay);
    presenter.upload_backdrop(&Eye::default(), publication.backdrop);
    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("present"),
    });
    presenter
        .record_scene(device, &mut encoder, &view, (SIZE, SIZE), BACKGROUND)
        .expect("recorded the scene");
    presenter
        .record_overlays(&mut encoder, &view, (SIZE, SIZE))
        .expect("recorded the overlays");
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
        let section = d
            .domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
            .unwrap();
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

const OVERLAY_LINE_AT: [f32; 3] = [0.0, 0.5, -4.0];
const OVERLAY_POINT_AT: [f32; 3] = [0.0, -0.5, -4.0];
const SKY_PIXEL: (u32, u32) = (64, 8);
const RED_SKY: Sky = Sky {
    below: [0.9, 0.1, 0.1],
    above: [0.9, 0.1, 0.1],
};
const HIDDEN_GROUND: Ground = Ground {
    y: -1.0,
    dark: [0.0; 3],
    light: [0.0; 3],
    fog_per_unit: 0.0,
    fog_start: 0.0,
    visible: false,
};

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn an_overlay_line_and_point_from_a_publication_system_are_drawn_where_they_stand_gpu_probe() {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    session.system(Phase::Publication, "marks", |ctx| {
        let [x, y, z] = OVERLAY_LINE_AT;
        ctx.overlay.lines.push(SegmentRecord {
            start: [x - 0.5, y, z],
            end: [x + 0.5, y, z],
            start_color: [1.0; 4],
            end_color: [1.0; 4],
            width_px: 3.0,
            ..SegmentRecord::default()
        });
        ctx.overlay.points.push(PointRecord {
            position: OVERLAY_POINT_AT,
            radius_px: 6.0,
            color: [1.0; 4],
        });
        Ok(())
    });
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();

    let pixels = render(&publication);
    for (name, at) in [("line", OVERLAY_LINE_AT), ("point", OVERLAY_POINT_AT)] {
        let at = pixel(session.views().ndc(at).expect("in front of the eye"));
        let drawn = texel(&pixels, at);
        assert!(
            drawn[0] >= INK_MIN,
            "the overlay {name} at {at:?} shows {drawn:?}"
        );
    }
    let empty = texel(&pixels, EMPTY);
    assert!(
        empty[0] <= BACKGROUND_MAX,
        "the pixel at {EMPTY:?} shows {empty:?}, not the background"
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_root_backdrop_paints_its_sky_over_the_background_gpu_probe() {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    session.views_mut().root_mut().backdrop = Some(Backdrop {
        sky: RED_SKY,
        ground: HIDDEN_GROUND,
    });
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();

    let sky = texel(&render(&publication), SKY_PIXEL);
    assert!(
        sky[0] >= INK_MIN,
        "the pixel at {SKY_PIXEL:?} shows {sky:?}, not the sky"
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn a_registered_sky_pass_paints_over_the_background_when_no_backdrop_is_published_gpu_probe() {
    let own = SkyGroundPass::new(HIDDEN_GROUND);
    own.publish(&Eye::default(), RED_SKY, HIDDEN_GROUND);

    let pixels = render_with(&Publication::default(), vec![Box::new(own)]);
    let sky = texel(&pixels, SKY_PIXEL);
    assert!(
        sky[0] >= INK_MIN,
        "the pixel at {SKY_PIXEL:?} shows {sky:?}, so the background clear won over the sky"
    );
}
