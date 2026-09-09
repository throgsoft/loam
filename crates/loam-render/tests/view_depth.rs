use glam::{Mat4, Quat, Vec2, Vec3, Vec4};
use loam_math::{HyperbolicH3, Iso3, Iso3H, IsometryGroup, Rotor4};
use loam_render::shader::ShaderDb;
use loam_render::view::{
    self, DEPTH_FORMAT, H3_DEPTH_ENVELOPE, H3_DEPTH_SEPARATION, H3_EYE_CHART_REACH,
};
use loam_render::{
    DepthConvention, DepthMode, LineRasterStaticR4Node, RayMarchNode, RayMarchUniforms, Viewport,
};
use loam_shape::LineMesh;
use wgpu::*;

const SIZE: u32 = 64;
const NEAR: f32 = 0.05;
const FOV_Y: f32 = std::f32::consts::FRAC_PI_3;
const FOCAL: f32 = 2.0;
const LINE_WIDTH_PX: f32 = 6.0;
const COLOR_FORMAT: TextureFormat = TextureFormat::Rgba8Unorm;

const EYE_YAW: f32 = 0.26;
const LINE_W: (f32, f32) = (0.3, 0.7);

const LINE_RED_MIN: u8 = 200;
const MARCH_BLUE_MIN: u8 = 200;
const MARCH_RED_MAX: u8 = 60;

const SCENE_WGSL: &str = r#"
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
    eye_inverse: mat4x4<f32>,
    near: f32,
};
@group(0) @binding(0) var<uniform> u: RayMarchUniforms;

fn loam_scene_sdf(p: vec3<f32>) -> f32 {
    if u.params.x > 1.5 {
        return abs(loam_origin_distance(p) - u.params.y);
    }
    if u.params.x > 0.5 {
        return abs(asinh(loam_poincare_to_hyperboloid(p).z));
    }
    return 1.0e9;
}
"#;

const USER_WGSL: &str = r#"
struct Fragment {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@vertex
fn vs_fullscreen(@builtin(vertex_index) vid: u32) -> @builtin(position) vec4<f32> {
    let x = f32(i32(vid) / 2) * 4.0 - 1.0;
    let y = f32(i32(vid) & 1) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> Fragment {
    let ndc = pos.xy / u.resolution * 2.0 - vec2<f32>(1.0, 1.0);
    let aspect = u.resolution.x / u.resolution.y;
    let dir = u.camera_forward
        + u.camera_right * (ndc.x * u.fov_y_tan * aspect)
        - u.camera_up * (ndc.y * u.fov_y_tan);
    let hit = loam_march_geodesic(u.camera_pos, dir, 1.0);
    var out: Fragment;
    if hit.w < 0.0 {
        out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        out.depth = 0.0;
        return out;
    }
    let image = loam_hyperboloid_to_klein(u.eye_inverse * loam_poincare_to_hyperboloid(hit.xyz));
    out.color = vec4<f32>(0.0, 0.0, 1.0, 1.0);
    out.depth = loam_projective_depth(image, u.near);
    return out;
}
"#;

#[derive(Clone, Copy)]
enum Scene {
    Empty,
    PlaneThroughOrigin,
    SphereAboutOrigin(f32),
}

impl Scene {
    fn params(self) -> [f32; 4] {
        match self {
            Scene::Empty => [0.0; 4],
            Scene::PlaneThroughOrigin => [1.0, 0.0, 0.0, 0.0],
            Scene::SphereAboutOrigin(radius) => [2.0, radius, 0.0, 0.0],
        }
    }
}

fn fov_tan() -> f32 {
    (FOV_Y * 0.5).tan()
}

fn pixel_ray((x, y): (u32, u32)) -> Vec3 {
    let nx = (x as f32 + 0.5) / SIZE as f32 * 2.0 - 1.0;
    let ny = 1.0 - (y as f32 + 0.5) / SIZE as f32 * 2.0;
    Vec3::new(nx * fov_tan(), ny * fov_tan(), -1.0)
}

fn h3_eye(distance: f32, yaw: f32) -> Iso3H {
    let translate = Iso3H::from_translation(Vec3::Z * (distance * 0.5).tanh());
    let turn = Iso3H::from_rotation(Quat::from_rotation_y(yaw));
    HyperbolicH3.iso_compose(translate, turn)
}

fn r3_eye() -> Iso3 {
    Iso3 {
        rotation: Quat::from_rotation_y(0.17),
        translation: Vec3::new(0.1, -0.05, 0.2),
    }
}

fn plane_hit(eye_inverse: &Iso3H, pixel: (u32, u32)) -> Vec3 {
    let n = eye_inverse.matrix * Vec4::Z;
    let d = pixel_ray(pixel);
    let s = n.w / d.dot(Vec3::new(n.x, n.y, n.z));
    assert!(s > 0.0 && s < 1.0, "pixel {pixel:?} misses the plane");
    d * s
}

fn sphere_hit(eye_inverse: &Iso3H, radius: f32, pixel: (u32, u32)) -> Vec3 {
    let c = eye_inverse.matrix * Vec4::W;
    let d = pixel_ray(pixel).normalize();
    let m = d.dot(Vec3::new(c.x, c.y, c.z));
    let cosh_r = radius.cosh();
    let a = m * m + cosh_r * cosh_r;
    let b = -2.0 * c.w * m;
    let k = c.w * c.w - cosh_r * cosh_r;
    let s = (-b + (b * b - 4.0 * a * k).sqrt()) / (2.0 * a);
    assert!(s > 0.0 && s < 1.0, "pixel {pixel:?} misses the sphere");
    d * s
}

fn along_ray(image: Vec3, delta: f32) -> Vec3 {
    let r = image.length() as f64;
    let moved = (r.atanh() + delta as f64).tanh();
    image * (moved / r) as f32
}

fn r4_point(image: Vec3, w: f32) -> [f32; 4] {
    let eye = r3_eye();
    let q3 = Mat4::from_rotation_translation(eye.rotation, eye.translation).transform_point3(image)
        * ((FOCAL - w) / FOCAL);
    [q3.x, q3.y, q3.z, w]
}

fn line_through(p1: Vec3, p2: Vec3) -> ([f32; 4], [f32; 4]) {
    let reach = (p2 - p1) * 0.25;
    (
        r4_point(p1 - reach, LINE_W.0),
        r4_point(p2 + reach, LINE_W.1),
    )
}

fn line_mesh(segment: Option<([f32; 4], [f32; 4])>) -> LineMesh<4> {
    let mut mesh = LineMesh::<4>::default();
    if let Some((a, b)) = segment {
        mesh.segments.push((a, b));
        mesh.colors.push(([1.0; 4], [1.0; 4]));
        mesh.widths.push(LINE_WIDTH_PX);
    }
    mesh
}

struct Frame {
    h3_eye: Iso3H,
    line: Option<([f32; 4], [f32; 4])>,
    scene: Scene,
}

struct Gpu {
    device: Device,
    queue: Queue,
    march: RayMarchNode,
    lines: LineRasterStaticR4Node,
    _shader_dir: tempfile::TempDir,
}

impl Gpu {
    fn new() -> Self {
        let (device, queue) = pollster::block_on(request_device());
        let shader_dir = tempfile::tempdir().expect("tempdir");
        let path = shader_dir.path().join("h3_view.wgsl");
        std::fs::write(&path, USER_WGSL).expect("write shader");
        let mut db = ShaderDb::new(device.clone());
        let id = db
            .load_geodesic_scene(ShaderDb::ROOT_OWNER, &path, SCENE_WGSL, &HyperbolicH3)
            .expect("H3 view shader");
        let depth = DepthMode::ReadWrite {
            format: DEPTH_FORMAT,
        };
        let march = RayMarchNode::with_depth(&device, COLOR_FORMAT, db.module(id), depth, 1);
        let lines = LineRasterStaticR4Node::new(
            &device,
            COLOR_FORMAT,
            depth,
            DepthConvention::ReversedZ,
            1,
        );
        Self {
            device,
            queue,
            march,
            lines,
            _shader_dir: shader_dir,
        }
    }

    fn render(&mut self, frame: &Frame) -> (Vec<u8>, Vec<f32>) {
        let color = self.texture(COLOR_FORMAT, "view-depth color");
        let depth = self.texture(DEPTH_FORMAT, "view-depth depth");
        let color_view = color.create_view(&TextureViewDescriptor::default());
        let depth_view = depth.create_view(&TextureViewDescriptor::default());

        let space = HyperbolicH3;
        let eye = frame.h3_eye;
        let chart = |image_dir: Vec3| space.iso_transport(eye, Vec3::ZERO, image_dir).to_array();
        self.march.set_uniforms(
            &self.queue,
            RayMarchUniforms {
                camera_pos: space.iso_apply(eye, Vec3::ZERO).to_array(),
                camera_forward: chart(-Vec3::Z),
                camera_right: chart(Vec3::X),
                camera_up: chart(Vec3::Y),
                fov_y_tan: fov_tan(),
                resolution: [SIZE as f32; 2],
                params: frame.scene.params(),
                eye_inverse: space.iso_inverse(eye).matrix.to_cols_array_2d(),
                near: NEAR,
                ..Default::default()
            },
        );

        let view_projection =
            view::root_projection(FOV_Y, 1.0, NEAR) * view::eye_relative(r3_eye());
        self.lines.set_transform(
            &self.queue,
            Rotor4::IDENTITY,
            view_projection,
            Vec2::splat(SIZE as f32),
            FOCAL,
        );
        self.lines
            .upload_mesh(&self.device, &self.queue, &line_mesh(frame.line));

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor {
                label: Some("view-depth"),
            });
        self.march.record(
            &mut encoder,
            &color_view,
            Some(&depth_view),
            Viewport::full([SIZE, SIZE]),
        );
        self.lines
            .record(&mut encoder, &color_view, Some(&depth_view), None);

        let color_buf = self.readback_buffer("view-depth color readback");
        let depth_buf = self.readback_buffer("view-depth depth readback");
        encoder.copy_texture_to_buffer(
            copy_source(&color, TextureAspect::All),
            copy_target(&color_buf),
            frame_extent(),
        );
        encoder.copy_texture_to_buffer(
            copy_source(&depth, TextureAspect::DepthOnly),
            copy_target(&depth_buf),
            frame_extent(),
        );
        self.queue.submit(Some(encoder.finish()));

        color_buf.slice(..).map_async(MapMode::Read, |_| {});
        depth_buf.slice(..).map_async(MapMode::Read, |_| {});
        self.device
            .poll(PollType::wait_indefinitely())
            .expect("readback poll");
        let pixels = color_buf.slice(..).get_mapped_range().to_vec();
        let depths =
            bytemuck::cast_slice::<u8, f32>(&depth_buf.slice(..).get_mapped_range()).to_vec();
        (pixels, depths)
    }

    fn texture(&self, format: TextureFormat, label: &str) -> Texture {
        self.device.create_texture(&TextureDescriptor {
            label: Some(label),
            size: frame_extent(),
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    fn readback_buffer(&self, label: &str) -> Buffer {
        self.device.create_buffer(&BufferDescriptor {
            label: Some(label),
            size: (SIZE * SIZE * 4) as u64,
            usage: BufferUsages::COPY_DST | BufferUsages::MAP_READ,
            mapped_at_creation: false,
        })
    }
}

fn frame_extent() -> Extent3d {
    Extent3d {
        width: SIZE,
        height: SIZE,
        depth_or_array_layers: 1,
    }
}

fn copy_source(texture: &Texture, aspect: TextureAspect) -> TexelCopyTextureInfo<'_> {
    TexelCopyTextureInfo {
        texture,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect,
    }
}

fn copy_target(buffer: &Buffer) -> TexelCopyBufferInfo<'_> {
    TexelCopyBufferInfo {
        buffer,
        layout: TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SIZE * 4),
            rows_per_image: None,
        },
    }
}

fn texel(pixels: &[u8], (x, y): (u32, u32)) -> [u8; 4] {
    let i = ((y * SIZE + x) * 4) as usize;
    [pixels[i], pixels[i + 1], pixels[i + 2], pixels[i + 3]]
}

fn depth_at(depths: &[f32], (x, y): (u32, u32)) -> f32 {
    depths[(y * SIZE + x) as usize]
}

fn assert_line(pixels: &[u8], pixel: (u32, u32), what: &str) {
    let t = texel(pixels, pixel);
    assert!(
        t[0] >= LINE_RED_MIN,
        "{what}: pixel {pixel:?} must show the line, got {t:?}"
    );
}

fn assert_marched(pixels: &[u8], pixel: (u32, u32), what: &str) {
    let t = texel(pixels, pixel);
    assert!(
        t[2] >= MARCH_BLUE_MIN && t[0] <= MARCH_RED_MAX,
        "{what}: pixel {pixel:?} must show the marched surface, got {t:?}"
    );
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
            label: Some("view-depth"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("request_device")
}

fn crossing(
    gpu: &mut Gpu,
    h3_eye: Iso3H,
    scene: Scene,
    (near_pixel, in_front): ((u32, u32), Vec3),
    (far_pixel, behind): ((u32, u32), Vec3),
    what: &str,
) -> Vec<f32> {
    let line = Some(line_through(in_front, behind));
    let (pixels, depths) = gpu.render(&Frame {
        h3_eye,
        line,
        scene,
    });
    assert_line(&pixels, near_pixel, what);
    assert_marched(&pixels, far_pixel, what);

    let (pixels, _) = gpu.render(&Frame {
        h3_eye,
        line,
        scene: Scene::Empty,
    });
    assert_line(&pixels, far_pixel, "control without the marched surface");
    depths
}

#[test]
fn raster_shaders_write_no_fragment_depth() {
    for (name, source) in [
        ("line", include_str!("../src/line_raster.wgsl")),
        ("line_r4", include_str!("../src/line_raster_static_r4.wgsl")),
        ("point", include_str!("../src/point_raster.wgsl")),
        ("triangle", include_str!("../src/triangle_raster.wgsl")),
    ] {
        let module =
            naga::front::wgsl::parse_str(source).unwrap_or_else(|error| panic!("{name}: {error}"));
        for entry in &module.entry_points {
            if entry.stage != naga::ShaderStage::Fragment {
                continue;
            }
            let Some(result) = &entry.function.result else {
                continue;
            };
            let bindings: Vec<Option<naga::Binding>> = match &module.types[result.ty].inner {
                naga::TypeInner::Struct { members, .. } => {
                    members.iter().map(|m| m.binding.clone()).collect()
                }
                _ => vec![result.binding.clone()],
            };
            let writes_depth = bindings
                .iter()
                .any(|b| matches!(b, Some(naga::Binding::BuiltIn(naga::BuiltIn::FragDepth))));
            assert!(
                !writes_depth,
                "{name}::{}: a standard raster shader leaves depth to the hardware",
                entry.name
            );
        }
    }
}

#[test]
fn klein_tessellation_stays_within_its_declared_error() {
    let p0 = Vec3::new(-0.7, 0.2, 0.1);
    let p1 = Vec3::new(0.5, -0.6, 0.3);
    let exact = |s: f32| {
        let p = p0.lerp(p1, s);
        2.0 * p / (1.0 + p.length_squared())
    };
    for samples in [1usize, 2, 4, 8, 16, 64] {
        let mut polyline = Vec::new();
        view::klein_tessellate_segment(p0, p1, samples, |k| polyline.push(k));
        let declared = view::klein_tessellation_error(p0, p1, samples);
        let mut worst = 0.0_f32;
        for i in 0..=4096 {
            let s = i as f32 / 4096.0;
            let u = s * samples as f32;
            let chord = (u.floor() as usize).min(samples - 1);
            let on_polyline = polyline[chord].lerp(polyline[chord + 1], u - chord as f32);
            worst = worst.max((exact(s) - on_polyline).length());
        }
        assert!(
            worst <= declared,
            "{samples} chords: gap {worst} exceeds the declared {declared}"
        );
    }
}

#[test]
fn declared_h3_depth_envelope_is_inside_the_f32_ordering_limit() {
    let eye_inverse = HyperbolicH3.iso_inverse(h3_eye(H3_EYE_CHART_REACH, 0.0));
    let on_axis = |from_eye: f32| Vec3::Z * -((from_eye - H3_EYE_CHART_REACH) * 0.5).tanh();
    let mut t = H3_EYE_CHART_REACH + 0.5;
    let lost = loop {
        assert!(t < 12.0, "f32 must lose the order somewhere below 12");
        let hit = view::projective_depth(view::h3_image_of(&eye_inverse, on_axis(t)), NEAR);
        let past = view::projective_depth(
            view::h3_image_of(&eye_inverse, on_axis(t + H3_DEPTH_SEPARATION)),
            NEAR,
        );
        if hit <= past {
            break (t, hit, past);
        }
        t += 0.01;
    };
    println!(
        "H3 depth order for separation {H3_DEPTH_SEPARATION} with the eye {H3_EYE_CHART_REACH} \
         from the chart origin is lost at hyperbolic distance {}: depths {} and {}",
        lost.0, lost.1, lost.2
    );
    assert!(
        H3_DEPTH_ENVELOPE < lost.0,
        "the declared envelope {H3_DEPTH_ENVELOPE} reaches past the measured limit {}",
        lost.0
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn known_point_writes_the_same_depth_from_both_paths_gpu_probe() {
    let mut gpu = Gpu::new();
    let pixel = (40, 28);
    let h3_eye = h3_eye(1.0, EYE_YAW);
    let point = plane_hit(&HyperbolicH3.iso_inverse(h3_eye), pixel);
    let expected = NEAR / -point.z;
    let line = Some((
        r4_point(point - Vec3::X * 0.3, LINE_W.0),
        r4_point(point + Vec3::X * 0.3, LINE_W.1),
    ));

    let (pixels, depths) = gpu.render(&Frame {
        h3_eye,
        line,
        scene: Scene::Empty,
    });
    assert_line(&pixels, pixel, "line alone");
    let raster = depth_at(&depths, pixel);
    assert!(
        (raster - expected).abs() <= 1e-5 * expected,
        "hardware depth {raster} is not the projective depth {expected}"
    );

    let (pixels, depths) = gpu.render(&Frame {
        h3_eye,
        line: None,
        scene: Scene::PlaneThroughOrigin,
    });
    assert_marched(&pixels, pixel, "plane alone");
    let marched = depth_at(&depths, pixel);
    assert!(
        (marched - expected).abs() <= 2e-3 * expected,
        "marched depth {marched} is not the projective depth {expected}"
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn crossing_line_and_marched_plane_occlude_each_other_gpu_probe() {
    let mut gpu = Gpu::new();
    let h3_eye = h3_eye(1.0, EYE_YAW);
    let eye_inverse = HyperbolicH3.iso_inverse(h3_eye);
    let (near_pixel, far_pixel) = ((18, 32), (46, 32));
    crossing(
        &mut gpu,
        h3_eye,
        Scene::PlaneThroughOrigin,
        (
            near_pixel,
            along_ray(plane_hit(&eye_inverse, near_pixel), -0.25),
        ),
        (
            far_pixel,
            along_ray(plane_hit(&eye_inverse, far_pixel), 0.25),
        ),
        "crossing",
    );
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn moved_eye_keeps_the_order_at_the_declared_envelope_gpu_probe() {
    let mut gpu = Gpu::new();
    let h3_eye = h3_eye(H3_EYE_CHART_REACH, EYE_YAW);
    let eye_inverse = HyperbolicH3.iso_inverse(h3_eye);
    let radius = H3_DEPTH_ENVELOPE - H3_EYE_CHART_REACH;
    let (near_pixel, far_pixel) = ((18, 30), (46, 35));
    let near_hit = sphere_hit(&eye_inverse, radius, near_pixel);
    let far_hit = sphere_hit(&eye_inverse, radius, far_pixel);
    println!(
        "hits at hyperbolic distances {} and {} from the eye",
        near_hit.length().atanh(),
        far_hit.length().atanh()
    );
    let depths = crossing(
        &mut gpu,
        h3_eye,
        Scene::SphereAboutOrigin(radius),
        (near_pixel, along_ray(near_hit, -H3_DEPTH_SEPARATION)),
        (far_pixel, along_ray(far_hit, H3_DEPTH_SEPARATION)),
        "at the envelope",
    );
    println!(
        "depths at the envelope: line {} in front, sphere {} in front",
        depth_at(&depths, near_pixel),
        depth_at(&depths, far_pixel)
    );
}
