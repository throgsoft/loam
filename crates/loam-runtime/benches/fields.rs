use std::hint::black_box;
use std::time::Instant;

use loam_math::{EuclideanR3, Space};
use loam_physics::euclidean_r3::{halfspace_body_r3, sphere_body_r3};
use loam_physics::field_contact::sphere_against_field;
use loam_physics::{BodyId, FieldContact, FieldRefusal, World};
use loam_runtime::{
    ChartId, ChartPose, Domain, DomainBuilder, DomainError, DomainHandle, DomainSpace, Entity,
    Field, FieldCost, FieldKind, FieldOp, FieldProgram, LogCapacity, Pose, Session, SimConfig,
    SpawnBundle,
};
use loam_shape::field::DistanceField;

const BATCHES: usize = 9;
const QUERY_REPS: u32 = 20_000;
const CONTACT_REPS: u32 = 500;
const IDLE_COMPILE_REPS: u32 = 2_000;
const EDIT_COMPILE_REPS: u32 = 100;
const FAILED_COMPILE_REPS: u32 = 500;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Empty {}
}

type Point3 = <EuclideanR3 as Space>::Point;

struct Runtime {
    session: Session<Empty>,
    domain: DomainHandle<EuclideanR3>,
}

impl Runtime {
    fn new() -> Self {
        let mut session = Session::new(Empty::default(), SimConfig::default());
        let domain = session.register_domain(
            DomainBuilder::new("r3", EuclideanR3)
                .tracked(LogCapacity::default())
                .fields(),
        );
        Self { session, domain }
    }

    fn add(&mut self, pose: Pose<EuclideanR3>, op: FieldOp, operands: Vec<Entity>) -> Entity {
        let domain = self.domain;
        self.session.dispatch(|dispatch| {
            let field = Field {
                kind: FieldKind::ExactDistance,
                op,
                operands,
            };
            let entity = dispatch
                .spawn(SpawnBundle::new().at(domain, pose))
                .expect("field entity");
            dispatch
                .attach_field(domain, entity, field)
                .expect("field row");
            entity
        })
    }

    fn compile_result(&mut self) -> Result<FieldCost, DomainError> {
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain")
            .compile_fields()
    }

    fn compile(&mut self) -> FieldCost {
        self.compile_result().expect("field compile")
    }

    fn program(&self) -> &FieldProgram {
        self.session
            .domains()
            .read(self.domain)
            .expect("R3 domain")
            .field_program()
    }

    fn move_to(&mut self, entity: Entity, at: [f32; 3]) {
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain")
            .set_point(entity, point(at))
            .expect("field pose");
    }

    fn set_operands(&mut self, entity: Entity, operands: Vec<Entity>) {
        self.session
            .domains_mut()
            .typed(self.domain)
            .expect("R3 domain")
            .field_mut(entity)
            .expect("field row")
            .operands = operands;
    }
}

type Case = (
    &'static str,
    Pose<EuclideanR3>,
    FieldOp,
    [f32; 4],
    f32,
    fn([f32; 4]) -> f64,
);

fn point(p: [f32; 3]) -> Point3 {
    EuclideanR3.local_point([p[0], p[1], p[2], 0.0])
}

fn sphere_world(at: [f32; 4], radius: f32) -> (World<EuclideanR3>, BodyId) {
    let mut world = World::new(EuclideanR3);
    let body = world.push_body(
        sphere_body_r3(point([at[0], at[1], at[2]]), point([0.0; 3]), radius, 1.0)
            .expect("sphere body"),
    );
    (world, body)
}

fn contact(
    world: &World<EuclideanR3>,
    body: BodyId,
    field: &dyn DistanceField,
) -> Result<FieldContact<EuclideanR3>, FieldRefusal> {
    sphere_against_field(&world.bodies()[body], world.geometry(), field, &EuclideanR3)
}

fn rotated_box_pose() -> Pose<EuclideanR3> {
    EuclideanR3
        .pose_from_chart(&ChartPose {
            chart: ChartId(0),
            coordinates: [2.0, -1.0, 0.5, 0.0],
            frame: [
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        })
        .expect("valid pose")
}

fn norm3(p: [f32; 4], center: [f64; 3]) -> f64 {
    (0..3)
        .map(|i| (p[i] as f64 - center[i]).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn box_rotated(p: [f32; 4]) -> f64 {
    let local = [p[1] as f64 + 1.0, 2.0 - p[0] as f64, p[2] as f64 - 0.5];
    let q = [0, 1, 2].map(|i| local[i].abs() - [2.0, 1.0, 0.5][i]);
    q.into_iter()
        .map(|v| v.max(0.0).powi(2))
        .sum::<f64>()
        .sqrt()
        + q.into_iter().fold(f64::NEG_INFINITY, f64::max).min(0.0)
}

fn median<T>(reps: u32, mut body: impl FnMut() -> T) -> f64 {
    let mut batches = [0.0; BATCHES];
    for batch in &mut batches {
        let start = Instant::now();
        for _ in 0..reps {
            black_box(body());
        }
        *batch = start.elapsed().as_nanos() as f64 / f64::from(reps);
    }
    batches.sort_unstable_by(f64::total_cmp);
    batches[BATCHES / 2]
}

fn median_failed_rebuild(
    runtime: &mut Runtime,
    operator: Entity,
    cyclic: [Entity; 2],
    valid: [Entity; 2],
) -> (f64, f64, f64) {
    let mut failures = [0.0; BATCHES];
    let mut retries = [0.0; BATCHES];
    let mut rebuilds = [0.0; BATCHES];
    for batch in 0..BATCHES {
        let mut failed_ns = 0u128;
        let mut retry_ns = 0u128;
        let mut rebuild_ns = 0u128;
        for _ in 0..FAILED_COMPILE_REPS {
            runtime.set_operands(operator, cyclic.to_vec());
            let start = Instant::now();
            assert_eq!(
                runtime.compile_result(),
                Err(DomainError::FieldCycle(operator))
            );
            failed_ns += start.elapsed().as_nanos();

            let start = Instant::now();
            assert_eq!(
                runtime.compile_result(),
                Err(DomainError::FieldCycle(operator))
            );
            retry_ns += start.elapsed().as_nanos();

            runtime.set_operands(operator, valid.to_vec());
            let start = Instant::now();
            assert!(runtime.compile().full_rebuild);
            rebuild_ns += start.elapsed().as_nanos();
        }
        failures[batch] = failed_ns as f64 / f64::from(FAILED_COMPILE_REPS);
        retries[batch] = retry_ns as f64 / f64::from(FAILED_COMPILE_REPS);
        rebuilds[batch] = rebuild_ns as f64 / f64::from(FAILED_COMPILE_REPS);
    }
    failures.sort_unstable_by(f64::total_cmp);
    retries.sort_unstable_by(f64::total_cmp);
    rebuilds.sort_unstable_by(f64::total_cmp);
    (
        failures[BATCHES / 2],
        retries[BATCHES / 2],
        rebuilds[BATCHES / 2],
    )
}

fn run_exact((name, pose, op, query_point, body_radius, oracle): Case) {
    let mut runtime = Runtime::new();
    runtime.add(pose, op, Vec::new());
    let layout = runtime.compile().program_layout;
    let program = runtime.program();
    assert_eq!(program.field_kind(), FieldKind::ExactDistance);
    let expected = oracle(query_point);
    let query_error = program.error_at(query_point);
    let query_residual = (f64::from(program.distance(query_point)) - expected).abs();
    assert!(query_residual <= f64::from(query_error));

    let (world, body) = sphere_world(query_point, body_radius);
    let query = || contact(&world, body, program).expect("admitted contact");
    let contact = query();
    let expected_separation = expected - f64::from(body_radius);
    let separation_residual = (f64::from(contact.separation) - expected_separation).abs();
    let witness = EuclideanR3.chart_point(contact.witness).coordinates;
    let witness_residual = oracle(witness).abs();
    assert!(separation_residual <= f64::from(contact.error));
    assert!(witness_residual <= f64::from(contact.error));

    let query_ns = median(QUERY_REPS, || program.distance(query_point));
    let contact_ns = median(CONTACT_REPS, query);
    println!(
        "exact {} {:?} {layout} {query_ns:.1} {contact_ns:.1} {expected:.9} {expected_separation:.9} {query_error:.9} {query_residual:.9} {:.9} {separation_residual:.9} {witness_residual:.9}",
        name,
        program.field_kind(),
        contact.error,
    );
}

fn run_boolean(leaves: usize) {
    let mut runtime = Runtime::new();
    let first_x = -(leaves as f32 - 1.0) * 1.25;
    let mut level = (0..leaves)
        .map(|i| {
            runtime.add(
                Pose::at(point([first_x + i as f32 * 2.5, 0.0, 0.0])),
                FieldOp::Sphere { radius: 0.75 },
                Vec::new(),
            )
        })
        .collect::<Vec<_>>();
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| runtime.add(Pose::at(point([0.0; 3])), FieldOp::Union, pair.to_vec()))
            .collect();
    }
    let layout = runtime.compile().program_layout;
    let program = runtime.program();
    assert_eq!(program.field_kind(), FieldKind::ConservativeBound);
    for (case, query) in [
        ("near", [first_x + 1.0, 0.0, 0.0, 0.0]),
        ("far", [0.0, 12.0, 0.0, 0.0]),
    ] {
        let value = program.distance(query);
        let query_ns = median(QUERY_REPS, || program.distance(query));
        println!(
            "boolean {leaves} {} {:?} {layout} {case} {query_ns:.1} {value:.9}",
            leaves * 2 - 1,
            program.field_kind(),
        );
    }
    let (world, body) = sphere_world([first_x + 1.0, 0.0, 0.0, 0.0], 0.1);
    let refusal = contact(&world, body, program)
        .err()
        .expect("Boolean contact refusal");
    assert_eq!(refusal, FieldRefusal::Kind(FieldKind::ConservativeBound));
    println!("boolean_contact {leaves} {refusal:?}");
}

fn run_compile_cost(leaves: usize) {
    let mut runtime = Runtime::new();
    let first_x = -(leaves as f32 - 1.0) * 1.25;
    let mut level = (0..leaves)
        .map(|i| {
            runtime.add(
                Pose::at(point([first_x + i as f32 * 2.5, 0.0, 0.0])),
                FieldOp::Sphere { radius: 0.75 },
                Vec::new(),
            )
        })
        .collect::<Vec<_>>();
    let moved = level[0];
    while level.len() > 1 {
        level = level
            .chunks(2)
            .map(|pair| runtime.add(Pose::at(point([0.0; 3])), FieldOp::Union, pair.to_vec()))
            .collect();
    }
    let rows = leaves * 2 - 1;
    assert!(runtime.compile().full_rebuild);
    let idle_ns = median(IDLE_COMPILE_REPS, || {
        let cost = runtime.compile();
        assert!(cost.is_idle());
        cost
    });
    let mut shifted = false;
    let pose_edit_ns = median(EDIT_COMPILE_REPS, || {
        shifted = !shifted;
        runtime.move_to(moved, [first_x + if shifted { 1.0 } else { 0.0 }, 0.0, 0.0]);
        let cost = runtime.compile();
        assert_eq!(cost.changed_inputs, 1);
        assert!(!cost.full_rebuild);
        cost
    });
    println!("compile {leaves} {rows} {idle_ns:.1} {pose_edit_ns:.1}");
}

fn run_failed_rebuild_cost() {
    let mut runtime = Runtime::new();
    let leaf = runtime.add(
        Pose::at(point([0.0; 3])),
        FieldOp::Sphere { radius: 1.0 },
        Vec::new(),
    );
    let up = runtime.add(Pose::at(point([0.0; 3])), FieldOp::Union, vec![leaf, leaf]);
    let down = runtime.add(Pose::at(point([0.0; 3])), FieldOp::Union, vec![up, leaf]);
    assert!(runtime.compile().full_rebuild);
    let valid = runtime.program().evaluate([3.0, 0.0, 0.0, 0.0]).unwrap();
    let (failed_ns, retry_ns, rebuild_ns) =
        median_failed_rebuild(&mut runtime, up, [down, leaf], [leaf, leaf]);
    assert_eq!(
        runtime.program().evaluate([3.0, 0.0, 0.0, 0.0]).unwrap(),
        valid
    );
    println!("failed_rebuild 3 {failed_ns:.1} {retry_ns:.1} {rebuild_ns:.1}");
}

fn run_fixed_step_support(name: &str, field_center: [f32; 3], field_radius: f32, body_radius: f32) {
    const SOLVER_SLOP: f64 = 0.005;
    const SUPPORT_BAND: f64 = 0.001;

    let mut runtime = Runtime::new();
    runtime.add(
        Pose::at(point(field_center)),
        FieldOp::Sphere {
            radius: field_radius,
        },
        Vec::new(),
    );
    runtime.compile();
    let program = runtime.program().clone();
    assert_eq!(program.field_kind(), FieldKind::ExactDistance);

    let start = [
        field_center[0],
        field_center[1] + field_radius + body_radius + 2.0,
        field_center[2],
    ];
    let expected = norm3(
        [start[0], start[1], start[2], 0.0],
        field_center.map(f64::from),
    ) - f64::from(field_radius);
    let residual =
        (f64::from(program.distance([start[0], start[1], start[2], 0.0])) - expected).abs();
    assert!(residual <= f64::from(program.error_at([start[0], start[1], start[2], 0.0])));

    let mut world = World::new(EuclideanR3);
    loam_physics::field_contact::register_field_contacts(&mut world.field_narrowphase);
    world
        .set_gravity(Some(point([0.0, -9.8, 0.0])))
        .expect("gravity");
    let anchor = world.push_body(halfspace_body_r3(point([0.0, 1.0, 0.0]), 0.0).expect("anchor"));
    let field = world
        .insert_field(anchor, Box::new(program.clone()))
        .expect("compiled field");
    let body = world.push_body(
        sphere_body_r3(point(start), point([0.0; 3]), body_radius, 1.0).expect("sphere body"),
    );
    world.bind_field(body, field).expect("field binding");
    for _ in 0..600 {
        world.step(1.0 / 240.0).expect("world step");
    }

    let settled = world.bodies()[body].position;
    let oracle_separation = norm3(
        [settled.x, settled.y, settled.z, 0.0],
        field_center.map(f64::from),
    ) - f64::from(field_radius)
        - f64::from(body_radius);
    let measured = contact(&world, body, &program).expect("settled contact");
    let velocity = world.bodies()[body].velocity;
    let coordinate_rounding = f64::from(settled.y.next_up() - settled.y);
    let witness = EuclideanR3.chart_point(measured.witness).coordinates;
    let witness_residual =
        (norm3(witness, field_center.map(f64::from)) - f64::from(field_radius)).abs();
    println!(
        "support_probe {name} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9} {:.9}",
        settled.x,
        settled.y,
        settled.z,
        velocity.y,
        oracle_separation,
        measured.separation,
        coordinate_rounding,
        witness_residual
    );
    assert!(
        (f64::from(measured.separation) - oracle_separation).abs() <= f64::from(measured.error)
    );
    assert!(oracle_separation <= f64::from(measured.error));
    assert!(oracle_separation >= -SOLVER_SLOP - SUPPORT_BAND - coordinate_rounding);
    assert_eq!(measured.normal, point([0.0, 1.0, 0.0]));
    assert!(witness_residual <= f64::from(measured.error));
    println!(
        "fixed_step_support {name} {} {} {} {} {} {:.9} {:.9} {:.9}",
        field_center[0],
        field_center[1],
        field_center[2],
        field_radius,
        body_radius,
        oracle_separation,
        measured.error,
        coordinate_rounding
    );
}

fn main() {
    println!("section case kind instruction_pairs_plus_primitive_records query_ns contact_ns expected_distance expected_separation query_error query_residual contact_error separation_residual witness_residual");
    for case in [
        (
            "sphere_origin",
            Pose::at(point([0.0; 3])),
            FieldOp::Sphere { radius: 1.0 },
            [1.75, 0.0, 0.0, 0.0],
            0.25,
            (|p| norm3(p, [0.0; 3]) - 1.0) as fn(_) -> _,
        ),
        (
            "sphere_translated_scaled",
            Pose::at(point([100.0, -50.0, 7.0])),
            FieldOp::Sphere { radius: 3.5 },
            [103.75, -50.0, 7.0, 0.0],
            0.5,
            (|p| norm3(p, [100.0, -50.0, 7.0]) - 3.5) as fn(_) -> _,
        ),
        (
            "box_rotated_z90",
            rotated_box_pose(),
            FieldOp::Box {
                half_extents: [2.0, 1.0, 0.5],
            },
            [2.0, 1.4, 0.5, 0.0],
            0.25,
            box_rotated as fn(_) -> _,
        ),
    ] {
        run_exact(case);
    }
    println!("section operands field_rows kind instruction_pairs_plus_primitive_records query_case query_ns value");
    for leaves in [2, 8, 32] {
        run_boolean(leaves);
    }
    println!("section leaves field_rows idle_compile_ns pose_edit_compile_ns");
    run_compile_cost(1024);
    println!("section field_rows failed_compile_ns retry_failed_compile_ns recovery_rebuild_ns");
    run_failed_rebuild_cost();
    println!("section case supported_x supported_y supported_z velocity_y oracle_separation measured_separation coordinate_rounding witness_residual");
    println!("section case center_x center_y center_z field_radius body_radius oracle_separation contact_error coordinate_rounding");
    for (name, center, field_radius, body_radius) in [
        ("origin_unit", [0.0, 0.0, 0.0], 1.0, 0.125),
        ("origin_small", [0.0, 0.0, 0.0], 0.125, 0.03125),
        ("translated_mid", [1024.0, 512.0, -256.0], 4.0, 0.5),
        ("translated_large", [8192.0, 4096.0, -2048.0], 64.0, 0.125),
    ] {
        run_fixed_step_support(name, center, field_radius, body_radius);
    }
}
