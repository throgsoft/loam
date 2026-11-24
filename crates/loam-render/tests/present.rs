use glam::{Vec2, Vec4};
use loam_math::EuclideanR4;
use loam_render::device::{FeatureRequest, GpuContext};
use loam_render::pass::{ColorLoad, FramePass, PublishedFrame};
use loam_render::present::Presenter;
use loam_render::SkyGroundPass;
use loam_runtime::{
    Backdrop, DomainBuilder, Entity, Eye, Ground, Instance, LogCapacity, Material, Overlay, Phase,
    PointRecord, Pose, PreparedGeometry, Projection4, Publication, SegmentRecord, Session,
    SimConfig, Sky, SpawnBundle, ViewSpec,
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
const EMPTY: (u32, u32) = (10, 118);

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

fn landmark_scene() -> (Session<Landmarks>, [Entity; 1]) {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let edges4 = session.prepare(cross4(REACH));
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 1.0], 3.0));
    let root = session.views().root_id();
    let landmarks = session.dispatch(|d| {
        let landmark4 = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(LANDMARK_R4))
                    .instance(Instance::new(edges4, white)),
            )
            .unwrap();
        let walker4 = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, walker4, Projection4 { focal: FOCAL }))
            .unwrap();
        [landmark4]
    });
    (session, landmarks)
}

fn render(publication: &Publication<()>) -> Vec<u8> {
    render_with(publication, Vec::new())
}

fn render_with(publication: &Publication<()>, passes: Vec<Box<dyn FramePass>>) -> Vec<u8> {
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
        &PublishedFrame {
            eye: Eye::default(),
            viewport_px: Vec2::splat(SIZE as f32),
            pixels_per_point: 1.0,
            text_origin_px: [0.0; 2],
            views: &publication.views,
            overlay: &publication.overlay,
            backdrop: publication.backdrop,
        },
    );
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
fn landmarks_gpu_probe() {
    let (mut session, landmarks) = landmark_scene();
    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();

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
fn overlay_line_and_point_gpu_probe() {
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
    session.publish(&mut publication, 1.0).unwrap();

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
fn root_backdrop_gpu_probe() {
    let mut session = Session::new(Landmarks::default(), SimConfig::default());
    session.views_mut().root_mut().backdrop = Some(Backdrop {
        sky: RED_SKY,
        ground: HIDDEN_GROUND,
    });
    let mut publication = Publication::default();
    session.publish(&mut publication, 1.0).unwrap();

    let sky = texel(&render(&publication), SKY_PIXEL);
    assert!(
        sky[0] >= INK_MIN,
        "the pixel at {SKY_PIXEL:?} shows {sky:?}, not the sky"
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn sky_pass_without_backdrop_gpu_probe() {
    let own = SkyGroundPass::new(HIDDEN_GROUND);
    own.publish(&Eye::default(), RED_SKY, HIDDEN_GROUND);

    let pixels = render_with(&Publication::default(), vec![Box::new(own)]);
    let sky = texel(&pixels, SKY_PIXEL);
    assert!(
        sky[0] >= INK_MIN,
        "the pixel at {SKY_PIXEL:?} shows {sky:?}, so the background clear won over the sky"
    );
}

#[test]
fn owner_fed_sky_pass() {
    let mut own = SkyGroundPass::new(HIDDEN_GROUND);
    own.publish(&Eye::default(), RED_SKY, HIDDEN_GROUND);
    let red = own.clear_color();
    let overlay = Overlay::default();
    let frame = |backdrop| PublishedFrame {
        eye: Eye::default(),
        viewport_px: Vec2::splat(SIZE as f32),
        pixels_per_point: 1.0,
        text_origin_px: [0.0; 2],
        views: &[],
        overlay: &overlay,
        backdrop,
    };

    own.read(&frame(None));
    assert_eq!(
        own.color_load(),
        ColorLoad::Clear,
        "a frame with no backdrop took the pass's own sky away"
    );
    own.read(&frame(Some(Backdrop {
        sky: Sky {
            below: [0.1, 0.1, 0.9],
            above: [0.1, 0.1, 0.9],
        },
        ground: HIDDEN_GROUND,
    })));
    assert_eq!(
        own.clear_color(),
        red,
        "the frame's backdrop replaced the sky the owner published"
    );
}
