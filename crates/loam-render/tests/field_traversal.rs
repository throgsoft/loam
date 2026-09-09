use glam::Vec3;
use loam_math::{EuclideanR3, Iso3};
use loam_runtime::domain::{Domain, DomainBuilder, DomainHandle, Field, FieldKind, Pose};
use loam_runtime::entity::Entity;
use loam_runtime::field::{
    evaluate_bounded_counted, evaluate_counted, FieldCost, FieldCounts, FieldOp, FieldPrimitive,
    OP_SPHERE, OP_UNION,
};
use loam_runtime::{FieldProgram, Session, SimConfig, SpawnBundle, DEFAULT_LOG_CAPACITY};
use loam_scene::{Scene, SceneNode};
use wgpu::*;

const RADIUS: f32 = 0.35;
const HIT_EPS: f32 = 1e-3;
const MIN_STEP: f32 = 1e-4;
const MAX_T: f32 = 60.0;
const RAYS: usize = 16;

fn centers(count: usize) -> Vec<Vec3> {
    (0..count)
        .map(|i| {
            let t = i as f32;
            Vec3::new(
                (t * 0.37).sin() * 4.0,
                (t * 0.71).cos() * 4.0,
                -3.0 - (t * 0.11).sin().abs() * 8.0,
            )
        })
        .collect()
}

const SPREAD_RADIUS: f32 = 4.0;

fn spread_centers() -> Vec<Vec3> {
    let step = 100.0 / 9.0;
    (0..1000)
        .map(|i| {
            let axis = |shift: usize| (i / shift) % 10;
            Vec3::new(
                axis(1) as f32 * step - 50.0,
                axis(10) as f32 * step - 50.0,
                axis(100) as f32 * step - 100.0,
            )
        })
        .collect()
}

fn identity_frame() -> [[f32; 4]; 4] {
    let mut frame = [[0.0; 4]; 4];
    for (axis, column) in frame.iter_mut().enumerate() {
        column[axis] = 1.0;
    }
    frame
}

fn interpreted(centers: &[Vec3]) -> FieldProgram {
    let mut program = FieldProgram {
        primitives: centers
            .iter()
            .map(|c| FieldPrimitive {
                frame: identity_frame(),
                translation: [c.x, c.y, c.z, 0.0],
                params: [RADIUS, 0.0, 0.0, 0.0],
            })
            .collect(),
        program: Vec::new(),
        nodes: Vec::new(),
        stack: 2,
        kind: FieldKind::ExactDistance,
    };
    for index in 0..centers.len() as u32 {
        program.program.extend([OP_SPHERE, index]);
        if index > 0 {
            program.program.extend([OP_UNION, 0]);
        }
    }
    program
}

fn specialized(centers: &[Vec3]) -> Scene {
    let mut root = SceneNode::sphere(centers[0], RADIUS);
    for &c in &centers[1..] {
        root = root.union(SceneNode::sphere(c, RADIUS));
    }
    Scene::new(root)
}

fn march(mut sdf: impl FnMut(Vec3) -> f32, ro: Vec3, rd: Vec3) -> (Option<f32>, u64) {
    let mut t = 0.0f32;
    let mut steps = 0u64;
    for _ in 0..256 {
        steps += 1;
        let d = sdf(ro + rd * t);
        if d < HIT_EPS {
            return (Some(t), steps);
        }
        t += d.max(MIN_STEP);
        if t > MAX_T {
            break;
        }
    }
    (None, steps)
}

fn ray(pixel: usize) -> Vec3 {
    let angle = pixel as f32 * 0.19;
    Vec3::new(angle.sin() * 0.4, angle.cos() * 0.4, -1.0).normalize()
}

#[test]
fn the_interpreter_and_the_specialized_emit_march_the_same_ray_alike() {
    let centers = centers(1000);
    let program = interpreted(&centers);
    let scene = specialized(&centers);
    let ro = Vec3::ZERO;

    let mut agreed = 0;
    let mut interpreter_steps = 0u64;
    let mut interpreter_counts = FieldCounts::default();
    let mut emit_steps = 0u64;
    for pixel in 0..RAYS {
        let rd = ray(pixel);

        let mut counts = FieldCounts::default();
        let (interpreted_hit, steps) = march(
            |p| {
                evaluate_counted(
                    &program.program,
                    &program.primitives,
                    [p.x, p.y, p.z, 0.0],
                    &mut counts,
                )
                .expect("well-formed program")
            },
            ro,
            rd,
        );
        interpreter_steps += steps;
        interpreter_counts.primitive_evals += counts.primitive_evals;
        interpreter_counts.instructions += counts.instructions;

        let (emit_hit, steps) = march(|p| scene.eval(&EuclideanR3, p), ro, rd);
        emit_steps += steps;

        match (interpreted_hit, emit_hit) {
            (Some(a), Some(b)) => {
                assert!(
                    (a - b).abs() < 1e-4,
                    "ray {pixel}: interpreter hit at {a}, specialized emit at {b}"
                );
                agreed += 1;
            }
            (None, None) => agreed += 1,
            (a, b) => panic!("ray {pixel}: interpreter {a:?}, specialized emit {b:?}"),
        }
    }
    assert_eq!(agreed, RAYS);
    assert_eq!(interpreter_steps, emit_steps);
    println!(
        "1000 primitives, {RAYS} rays: steps {interpreter_steps} interpreter, {emit_steps} \
         specialized; primitive evaluations {} interpreter, {} specialized; \
         instructions {}",
        interpreter_counts.primitive_evals,
        emit_steps * centers.len() as u64,
        interpreter_counts.instructions,
    );
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Bench {
        tags: Store<u8>,
    }
}

struct Compiled {
    session: Session<Bench>,
    domain: DomainHandle<EuclideanR3>,
    cost: FieldCost,
}

impl Compiled {
    fn program(&self) -> &FieldProgram {
        self.session
            .domains()
            .iter()
            .find(|d| d.id() == self.domain.id())
            .expect("field domain")
            .field_program()
    }
}

fn compiled(centers: &[Vec3], radius: f32) -> Compiled {
    let mut session = Session::new(Bench::default(), SimConfig::default());
    let domain = session.register_domain(
        DomainBuilder::new("field bench", EuclideanR3)
            .tracked(DEFAULT_LOG_CAPACITY)
            .fields(),
    );
    let place = |session: &mut Session<Bench>, at: Vec3| {
        session
            .dispatch(|d| d.spawn(SpawnBundle::new().at(domain, Pose(Iso3::from_translation(at)))))
            .expect("spawn")
    };
    let leaves: Vec<Entity> = centers.iter().map(|&c| place(&mut session, c)).collect();
    let mut level = leaves.clone();
    let mut pairs: Vec<(Entity, Vec<Entity>)> = Vec::new();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| {
                if pair.len() == 1 {
                    pair[0]
                } else {
                    let node = place(&mut session, Vec3::ZERO);
                    pairs.push((node, pair.to_vec()));
                    node
                }
            })
            .collect();
    }

    let typed = session.domains_mut().typed(domain).expect("typed domain");
    let fields = typed.fields_mut().expect("field store");
    for leaf in &leaves {
        fields
            .insert(
                *leaf,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::Sphere { radius },
                    operands: Vec::new(),
                },
            )
            .expect("field row");
    }
    for (node, operands) in pairs {
        fields
            .insert(
                node,
                Field {
                    kind: FieldKind::ExactDistance,
                    op: FieldOp::Union,
                    operands,
                },
            )
            .expect("field row");
    }
    let cost = typed.compile_fields().expect("compile");
    Compiled {
        session,
        domain,
        cost,
    }
}

fn traverse(program: &FieldProgram, culled: bool) -> (u64, FieldCounts, usize) {
    let mut steps = 0u64;
    let mut counts = FieldCounts::default();
    let mut hits = 0usize;
    for pixel in 0..RAYS {
        let (hit, taken) = march(
            |p| {
                let point = [p.x, p.y, p.z, 0.0];
                if culled {
                    evaluate_bounded_counted(
                        &program.program,
                        &program.primitives,
                        &program.nodes,
                        point,
                        HIT_EPS,
                        &mut counts,
                    )
                } else {
                    evaluate_counted(&program.program, &program.primitives, point, &mut counts)
                }
                .expect("well-formed program")
            },
            Vec3::ZERO,
            ray(pixel),
        );
        steps += taken;
        hits += usize::from(hit.is_some());
    }
    (steps, counts, hits)
}

#[test]
fn the_hierarchy_marches_every_ray_to_the_same_hit_as_the_unculled_program() {
    for (name, centers, radius) in [
        ("balanced", centers(1000), RADIUS),
        ("100-unit cube", spread_centers(), SPREAD_RADIUS),
    ] {
        let built = compiled(&centers, radius);
        let (program, cost) = (built.program(), built.cost);
        assert_eq!(program.primitives.len(), centers.len());
        assert!(!program.nodes.is_empty(), "{name}: no hierarchy was built");

        for pixel in 0..RAYS {
            let rd = ray(pixel);
            let flat = march(
                |p| {
                    program
                        .evaluate([p.x, p.y, p.z, 0.0])
                        .expect("well-formed program")
                        .0
                },
                Vec3::ZERO,
                rd,
            );
            let culled = march(
                |p| {
                    program
                        .evaluate_bounded([p.x, p.y, p.z, 0.0], HIT_EPS)
                        .expect("well-formed program")
                        .0
                },
                Vec3::ZERO,
                rd,
            );
            assert_eq!(
                flat.0, culled.0,
                "{name} ray {pixel}: unculled {:?}, hierarchy {:?}",
                flat.0, culled.0
            );
        }

        let (flat_steps, flat_counts, hits) = traverse(program, false);
        let (culled_steps, culled_counts, culled_hits) = traverse(program, true);
        assert_eq!(hits, culled_hits);
        println!(
            "{name}, 1000 primitives, {RAYS} rays, {hits} hits: \
             unculled {} steps/ray {} evals/ray; hierarchy {} steps/ray {} evals/ray \
             {} visits/ray {} skips/ray; nodes {}; compile {cost:?}",
            flat_steps as f64 / RAYS as f64,
            flat_counts.primitive_evals as f64 / RAYS as f64,
            culled_steps as f64 / RAYS as f64,
            culled_counts.primitive_evals as f64 / RAYS as f64,
            culled_counts.node_visits as f64 / RAYS as f64,
            culled_counts.node_skips as f64 / RAYS as f64,
            program.nodes.len(),
        );
    }
}

const PROBE: u32 = 32;

async fn request_device() -> (Device, Queue) {
    let instance = Instance::default();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: PowerPreference::LowPower,
            compatible_surface: None,
            force_fallback_adapter: false,
        })
        .await
        .expect("wgpu adapter");
    adapter
        .request_device(&DeviceDescriptor {
            label: Some("field traversal probe"),
            required_features: Features::empty(),
            required_limits: Limits::default(),
            memory_hints: MemoryHints::default(),
            trace: Trace::Off,
            experimental_features: Default::default(),
        })
        .await
        .expect("wgpu device")
}

fn gpu_counts(
    device: &Device,
    queue: &Queue,
    program: &FieldProgram,
    specialize: bool,
) -> [f64; 4] {
    let mut node = loam_render::FieldMarchNode::counting(device, 1);
    node.uniforms_mut().resolution = [PROBE as f32; 2];
    node.uniforms_mut().fov_y_tan = 0.4;
    node.set_program(queue, program);
    if specialize {
        node.specialize_after(1);
        node.boundary();
        assert!(
            node.is_specialized(),
            "the counting node did not specialize"
        );
    }

    let size = Extent3d {
        width: PROBE,
        height: PROBE,
        depth_or_array_layers: 1,
    };
    let color = device.create_texture(&TextureDescriptor {
        label: Some("field counts"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba32Float,
        usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = color.create_view(&TextureViewDescriptor::default());
    let mut encoder = device.create_command_encoder(&Default::default());
    node.record(
        &mut encoder,
        &view,
        None,
        loam_render::Viewport::full([PROBE, PROBE]),
    );
    let readback = device.create_buffer(&BufferDescriptor {
        label: Some("field counts readback"),
        size: (PROBE * PROBE * 16) as u64,
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
                bytes_per_row: Some(PROBE * 16),
                rows_per_image: None,
            },
        },
        size,
    );
    queue.submit(Some(encoder.finish()));
    readback.slice(..).map_async(MapMode::Read, |_| {});
    device
        .poll(PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .expect("readback poll");
    let pixels = bytemuck::cast_slice::<u8, f32>(&readback.slice(..).get_mapped_range()).to_vec();

    let mut sums = [0.0f64; 3];
    let mut hits = 0.0f64;
    for texel in pixels.chunks_exact(4) {
        if texel[0] == 0.0 && texel[2] == 0.0 {
            continue;
        }
        hits += 1.0;
        for (slot, value) in sums.iter_mut().zip(texel) {
            *slot += *value as f64;
        }
    }
    let scale = hits.max(1.0);
    [sums[0] / scale, sums[1] / scale, sums[2] / scale, hits]
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_gpu_hierarchy_evaluates_fewer_primitives_than_the_unculled_kernel_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    for (name, centers, radius) in [
        ("balanced", centers(1000), RADIUS),
        ("100-unit cube", spread_centers(), SPREAD_RADIUS),
    ] {
        let built = compiled(&centers, radius);
        let culled = built.program();
        let mut flat = culled.clone();
        flat.nodes.clear();

        let unculled = gpu_counts(&device, &queue, &flat, false);
        let hierarchy = gpu_counts(&device, &queue, culled, false);
        assert!(unculled[3] > 0.0, "{name}: no ray hit the field");
        assert_eq!(
            hierarchy[3], unculled[3],
            "{name}: the hierarchy changed which pixels hit the field"
        );
        assert!(
            hierarchy[2] < unculled[2],
            "{name}: the hierarchy evaluated {} primitives per ray against {} unculled",
            hierarchy[2],
            unculled[2]
        );
        println!(
            "{name}, 1000 primitives, GPU {PROBE}x{PROBE}, {} hit rays: \
             unculled {} evals/ray; hierarchy {} evals/ray {} visits/ray {} skips/ray",
            unculled[3], unculled[2], hierarchy[2], hierarchy[0], hierarchy[1],
        );
    }
}

#[test]
#[ignore = "requires a working wgpu adapter; run with --include-ignored"]
fn the_specialized_kernel_keeps_the_hierarchys_culling_gpu_probe() {
    let (device, queue) = pollster::block_on(request_device());
    let built = compiled(&centers(1000), RADIUS);
    let culled = built.program();
    let mut flat = culled.clone();
    flat.nodes.clear();

    let unculled = gpu_counts(&device, &queue, &flat, false);
    let interpreted = gpu_counts(&device, &queue, culled, false);
    let specialized = gpu_counts(&device, &queue, culled, true);
    assert_eq!(
        specialized[3], interpreted[3],
        "the specialized kernel changed which pixels hit the field"
    );
    assert!(
        (specialized[2] - interpreted[2]).abs() <= 0.05 * interpreted[2],
        "the specialized kernel evaluated {} primitives per ray, the hierarchy interpreter {}",
        specialized[2],
        interpreted[2]
    );
    assert!(
        specialized[2] * 4.0 < unculled[2],
        "the specialized kernel evaluated {} primitives per ray against {} unculled",
        specialized[2],
        unculled[2]
    );
    println!(
        "balanced, 1000 primitives, GPU {PROBE}x{PROBE}, {} hit rays: \
         unculled {} evals/ray; hierarchy interpreter {} evals/ray; \
         specialized {} evals/ray {} visits/ray {} skips/ray",
        interpreted[3], unculled[2], interpreted[2], specialized[2], specialized[0], specialized[1],
    );
}
