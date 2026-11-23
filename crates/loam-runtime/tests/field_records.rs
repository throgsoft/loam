use loam_math::{EuclideanR4, Rotor, Rotor4, Space};
use loam_runtime::{
    ChartId, Ctx, DomainBuilder, DomainError, DomainHandle, Entity, FieldRecord, Instance,
    LogCapacity, Material, MaterialId, Phase, Pose, PreparedGeometry, Projection4, Publication,
    Section4, Session, SimConfig, SpawnBundle, ViewId, ViewSpec, ViewStyle,
};
use loam_shape::polytope::Polytope4;
use loam_shape::FieldShape;

type Vec4 = <EuclideanR4 as Space>::Point;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u8>,
    }
}

const CUT: f32 = 0.25;
const TESSERACT_SCALE: f32 = 0.5;
const BALL_SIZE: f32 = 0.75;
const BALL_COLOR: [f32; 4] = [0.1, 0.6, 0.9, 1.0];
const FILLED_AT: Vec4 = Vec4::new(-2.0, 0.0, -4.0, 0.0);
const HOLLOW_AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);
const BALL_AT: Vec4 = Vec4::new(2.0, 0.0, -4.0, 0.0);
const WIRE_AT: Vec4 = Vec4::new(0.0, 2.0, -4.0, 0.0);
const TOLERANCE: f32 = 1e-5;

struct Stage {
    session: Session<Probe>,
    r4: DomainHandle<EuclideanR4>,
    view: ViewId,
    filled: Entity,
    hollow: Entity,
    ball: Entity,
    ball_material: MaterialId,
}

fn stage(eye: Pose<EuclideanR4>) -> Stage {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: TESSERACT_SCALE,
    });
    let sphere = session.prepare(PreparedGeometry::Field {
        shape: FieldShape::ThreeSphere,
        size: BALL_SIZE,
    });
    let stub = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.5, 0.0, 0.0, 0.0], [-0.5, 0.0, 0.0, 0.0]]],
    });
    let edge = session.add_material(Material::lines([1.0; 4], 2.0));
    let ball_material = session.add_material(Material::flat(BALL_COLOR));
    let root = session.views().root();
    let (view, filled, hollow, ball) = session.dispatch(|d| {
        let eye = d.spawn(SpawnBundle::new().at(r4, eye)).unwrap();
        let mut body = |at: Vec4, instance: Instance| {
            d.spawn(SpawnBundle::new().at(r4, Pose::at(at)).instance(instance))
                .unwrap()
        };
        let filled = body(FILLED_AT, Instance::new(tesseract, edge).sectioned(edge));
        let hollow = body(HOLLOW_AT, Instance::new(tesseract, edge));
        let ball = body(
            BALL_AT,
            Instance::new(sphere, ball_material).sectioned(edge),
        );
        body(WIRE_AT, Instance::new(stub, edge));
        let view = d
            .domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Section4 { w: CUT }))
            .unwrap();
        (view, filled, hollow, ball)
    });
    Stage {
        session,
        r4,
        view,
        filled,
        hollow,
        ball,
        ball_material,
    }
}

impl Stage {
    fn style(&mut self, change: impl FnOnce(&mut ViewStyle<EuclideanR4>)) {
        let (r4, view) = (self.r4, self.view);
        let domain = self.session.domains_mut().typed(r4).unwrap();
        change(domain.view_mut(view).unwrap());
    }

    fn fields(&mut self) -> (Vec<FieldRecord>, Option<f32>) {
        let mut publication = Publication::default();
        self.session.publish(&mut publication).unwrap();
        let records = &publication.views[0].records;
        (records.fields().to_vec(), records.field_cut())
    }
}

fn entities(fields: &[FieldRecord]) -> Vec<Entity> {
    let mut entities: Vec<Entity> = fields.iter().map(|field| field.entity).collect();
    entities.sort_unstable();
    entities
}

fn sorted(mut entities: Vec<Entity>) -> Vec<Entity> {
    entities.sort_unstable();
    entities
}

fn of(fields: &[FieldRecord], entity: Entity) -> FieldRecord {
    *fields
        .iter()
        .find(|field| field.entity == entity)
        .expect("the entity has a field record")
}

fn near(got: [f32; 4], want: [f32; 4]) -> bool {
    got.iter()
        .zip(want)
        .all(|(g, w)| (g - w).abs() <= TOLERANCE)
}

fn same_turn(got: Rotor4, want: Rotor4) -> bool {
    got.to_mat4()
        .into_iter()
        .zip(want.to_mat4())
        .all(|(g, w)| near(g, w))
}

#[test]
fn a_view_draws_as_fields_only_the_solids_its_section_does_not_already_fill() {
    let mut stage = stage(Pose::at(Vec4::ZERO));
    let (filled, hollow, ball) = (stage.filled, stage.hollow, stage.ball);

    let (fields, cut) = stage.fields();
    assert!(
        fields.is_empty() && cut.is_none(),
        "a view that never asked for fields published {fields:?} at {cut:?}"
    );

    stage.style(|style| style.fields = true);
    let (fields, cut) = stage.fields();
    assert_eq!(cut, Some(CUT));
    assert_eq!(
        entities(&fields),
        sorted(vec![hollow, ball]),
        "the filled tesseract would be drawn twice, or a solid without a fill was dropped"
    );
    let record = of(&fields, ball);
    assert_eq!(
        (record.shape, record.size, record.material, record.color),
        (
            FieldShape::ThreeSphere,
            BALL_SIZE,
            stage.ball_material,
            BALL_COLOR
        )
    );
    let record = of(&fields, hollow);
    assert_eq!(
        (record.shape, record.size),
        (FieldShape::Polytope(Polytope4::Tesseract), TESSERACT_SCALE)
    );

    stage.style(|style| style.section_faces = false);
    let (fields, _) = stage.fields();
    assert_eq!(
        entities(&fields),
        sorted(vec![filled, hollow, ball]),
        "a view that fills no section left a tesseract with nothing solid"
    );

    stage.style(|style| {
        style.set_mapping(Section4 { w: -0.5 });
    });
    assert_eq!(stage.fields().1, Some(-0.5));

    stage.style(|style| {
        style.set_mapping(Projection4 { focal: 2.0 });
    });
    let (fields, cut) = stage.fields();
    assert!(
        fields.is_empty() && cut.is_none(),
        "a projection has no cut, yet it published {fields:?} at {cut:?}"
    );
}

#[test]
fn a_field_record_is_posed_in_the_frame_of_a_moved_and_turned_eye() {
    // The eye's x axis points along world z, so world (dx, dy, dz, dw) reads (dz, dy, -dx, dw).
    let turn = Rotor4::from_rotation_arc(Vec4::X, Vec4::Z);
    let eye = Pose {
        point: BALL_AT - Vec4::new(4.0, 2.0, 1.0, 0.25),
        frame: turn,
    };
    let mut stage = stage(eye);
    stage.style(|style| style.fields = true);
    let (fields, _) = stage.fields();
    let record = of(&fields, stage.ball);
    assert!(
        near(record.position, [1.0, 2.0, -4.0, 0.25]),
        "the ball sits at {:?} in the eye frame",
        record.position
    );
    assert!(
        same_turn(record.frame, turn.inverse()),
        "the ball's frame {:?} is not the eye's turn undone",
        record.frame
    );
}

#[test]
fn a_patched_publication_holds_the_same_blended_field_pose_as_a_fresh_one() {
    const EYE_AT: Vec4 = Vec4::new(0.5, 0.0, 0.0, 0.0);
    const STEP: Vec4 = Vec4::new(2.0, 0.0, 0.0, 1.0);
    let mut stage = stage(Pose::at(EYE_AT));
    stage.style(|style| style.fields = true);
    let (r4, ball) = (stage.r4, stage.ball);
    stage.session.system(
        Phase::Simulation,
        "move",
        move |ctx: Ctx<'_, Probe>| -> Result<(), DomainError> {
            if ctx.step.tick.0 != 0 {
                return Ok(());
            }
            ctx.domains.typed(r4)?.set_pose(
                ball,
                Pose {
                    point: BALL_AT + STEP,
                    frame: Rotor4::from_rotation_arc(Vec4::X, Vec4::W),
                },
            )
        },
    );

    let mut held = Publication::default();
    stage.session.publish(&mut held).unwrap();
    stage.session.tick().unwrap();
    stage.session.publish_at(&mut held, 0.5).unwrap();
    let mut fresh = Publication::default();
    stage.session.publish_at(&mut fresh, 0.5).unwrap();

    let drawn = stage
        .session
        .domains()
        .read(r4)
        .unwrap()
        .drawn_pose(ball, 0.5)
        .unwrap();
    let midway = (BALL_AT + STEP * 0.5 - EYE_AT).to_array();
    let patched = held.views[0].records.fields();
    let rebuilt = fresh.views[0].records.fields();
    for (path, fields) in [("patched", patched), ("rebuilt", rebuilt)] {
        let record = of(fields, ball);
        assert!(
            near(record.position, midway),
            "the {path} ball sits at {:?}, not halfway at {midway:?}",
            record.position
        );
        assert!(
            same_turn(record.frame, drawn.frame),
            "the {path} ball's frame {:?} is not the drawn frame {:?}",
            record.frame,
            drawn.frame
        );
    }
    assert_eq!(patched.len(), rebuilt.len());
    for (patched, rebuilt) in patched.iter().zip(rebuilt) {
        assert_eq!(patched.entity, rebuilt.entity);
        assert!(
            near(patched.position, rebuilt.position) && same_turn(patched.frame, rebuilt.frame),
            "the patch left {patched:?} where a rebuild gives {rebuilt:?}"
        );
    }
}

#[test]
fn a_field_solid_publishes_no_lines_or_faces_and_is_picked_at_its_surface() {
    const SIZE: f32 = 0.5;
    const AT: Vec4 = Vec4::new(0.0, 0.0, -4.0, 0.0);
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let sphere = session.prepare(PreparedGeometry::Field {
        shape: FieldShape::ThreeSphere,
        size: SIZE,
    });
    let material = session.add_material(Material::lines([1.0; 4], 2.0));
    let root = session.views().root();
    let ball = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))
            .unwrap();
        d.spawn(
            SpawnBundle::new()
                .at(r4, Pose::at(AT))
                .instance(Instance::new(sphere, material).sectioned(material)),
        )
        .unwrap()
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let records = &publication.views[0].records;
    assert_eq!(records.instances.rows().len(), 1);
    assert!(
        records.segments().is_empty() && records.triangles().is_empty(),
        "a solid with no edges published lines or faces"
    );

    let pick = session.pick([0.0, 0.0]).expect("the ball is under the ray");
    assert_eq!(pick.entity, ball);
    let hit = pick.hit.expect("a section lifts its ray");
    assert_eq!(hit.chart, ChartId(0));
    assert!(
        near(hit.coordinates, [0.0, 0.0, AT.z + SIZE, 0.0]),
        "the ray entered at {:?}, off the ball's surface",
        hit.coordinates
    );
}
