use loam_math::blended::{BlendedSpace, LinearBlendX};
use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Iso3H, IsometryGroup, Space};
use loam_runtime::{
    ChartId, ChartPose, DepthEnvelope, DomainBuilder, DomainError, DomainHandle, DomainRay,
    DomainSpace, Entity, ImageRay, Instance, Klein, LogCapacity, Material, Pose, PreparedGeometry,
    Projection4, Publication, RefusalSource, Session, SimConfig, SpawnBundle, ViewMapping,
    ViewSpec,
};

type Vec3 = <EuclideanR3 as Space>::Point;
type Vec4 = <EuclideanR4 as Space>::Point;
type Blend = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u8>,
    }
}

const IDENTITY_FRAME: [[f32; 4]; 4] = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
];

fn spin_z(angle: f32) -> Iso3H {
    let mut spin = Iso3H::IDENTITY;
    spin.matrix.x_axis = Vec4::new(angle.cos(), angle.sin(), 0.0, 0.0);
    spin.matrix.y_axis = Vec4::new(-angle.sin(), angle.cos(), 0.0, 0.0);
    spin
}

fn h3_walker() -> (Session<Probe>, DomainHandle<HyperbolicH3>, Entity) {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let h3 = session.register_domain(DomainBuilder::new("h3", HyperbolicH3));
    let walker = session
        .dispatch(|d| d.spawn(SpawnBundle::new().at(h3, Pose::at(Vec3::ZERO))))
        .unwrap();
    (session, h3, walker)
}

fn blend() -> Blend {
    BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-0.5, 0.5).expect("a blend zone with width"),
    )
}

struct ChartIdentity;

impl ViewMapping<Blend> for ChartIdentity {
    fn name(&self) -> &'static str {
        "chart identity"
    }

    fn image_point(&self, _eye: &Pose<Blend>, point: Vec3) -> Option<[f32; 3]> {
        Some(point.to_array())
    }

    fn lift(&self, _eye: &Pose<Blend>, _ray: &ImageRay) -> Option<DomainRay<Blend>> {
        None
    }

    fn ray_lift(&self) -> bool {
        false
    }

    fn depth_envelope(&self) -> DepthEnvelope {
        DepthEnvelope {
            near: 0.0,
            far: f32::INFINITY,
        }
    }
}

#[test]
fn a_blended_local_answers_past_its_metric_error_budget() {
    let space = blend();
    let pose = Pose::new(&space, Vec3::ZERO);
    assert!(space.local(&pose, Vec3::X * 0.98).is_ok());
    assert_eq!(
        space.local(&pose, Vec3::X * 0.995),
        Err(DomainError::ErrorBudget)
    );
}

#[test]
fn a_blended_local_answers_from_a_path_past_its_error_budget() {
    let space = blend();
    let pose = Pose::new(&space, Vec3::X * 0.8);
    assert_eq!(
        space.local(&pose, Vec3::new(-0.8, 0.1, 0.0)),
        Err(DomainError::ErrorBudget)
    );
    assert!(space.local(&pose, Vec3::new(0.7, 0.05, 0.0)).is_ok());
}

#[test]
fn blended_walk_refuses_an_incomplete_geodesic_step() {
    let space = blend();
    let pose = Pose::new(&space, Vec3::ZERO);
    assert_eq!(
        space.walk(&pose, Vec3::splat(1.0e20), 1.0).err(),
        Some(DomainError::InvalidCoordinate("geodesic"))
    );
}

#[test]
fn partial_blended_placement_is_refused_during_publication() {
    let space = blend();
    let pose = Pose::new(&space, Vec3::ZERO);
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let blend =
        session.register_domain(DomainBuilder::new("blend", space).tracked(LogCapacity::default()));
    let geometry = session.prepare(PreparedGeometry::Lines3 {
        segments: vec![[Vec3::splat(1.0e20).to_array(), [0.0; 3]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let object = session.dispatch(|dispatch| {
        let eye = dispatch.spawn(SpawnBundle::new().at(blend, pose)).unwrap();
        let object = dispatch
            .spawn(
                SpawnBundle::new()
                    .at(blend, pose)
                    .instance(Instance::new(geometry, material)),
            )
            .unwrap();
        dispatch
            .domains
            .typed(blend)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, ChartIdentity))
            .unwrap();
        object
    });

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let records = &publication.views[0].records;
    let refusals = records.refusals();
    assert!(records.segments().is_empty());
    assert_eq!(refusals.count, 1);
    let refusal = refusals.first.unwrap();
    assert_eq!(refusals.last, Some(refusal));
    assert_eq!(refusal.entity, object);
    assert_eq!(refusal.error, DomainError::InvalidCoordinate("geodesic"));
    assert_eq!(refusal.source, RefusalSource::Segment);
}

#[test]
fn chart_pose_outside_the_ball_or_with_a_non_finite_coordinate_is_accepted() {
    let chart = |coordinates: [f32; 4]| ChartPose {
        chart: ChartId(0),
        coordinates,
        frame: IDENTITY_FRAME,
    };
    let reject = |coordinates| HyperbolicH3.pose_from_chart(&chart(coordinates)).err();
    assert_eq!(
        reject([1.0, 0.0, 0.0, 0.0]),
        Some(DomainError::ChartBoundary)
    );
    assert_eq!(
        reject([0.2, f32::NAN, 0.0, 0.0]),
        Some(DomainError::InvalidCoordinate("y"))
    );
    assert_eq!(
        reject([0.2, 0.0, f32::INFINITY, 0.0]),
        Some(DomainError::InvalidCoordinate("z"))
    );
    let mut tilted = chart([0.2, 0.0, 0.0, 0.0]);
    tilted.frame[1] = [0.0, 0.5, 0.0, 0.0];
    assert_eq!(
        HyperbolicH3.pose_from_chart(&tilted).err(),
        Some(DomainError::InvalidFrame)
    );

    let pose = Pose::from(HyperbolicH3.iso_compose(
        Iso3H::from_translation(Vec3::new(0.3, -0.1, 0.2)),
        spin_z(0.4),
    ));
    let data = HyperbolicH3.chart_pose(&pose);
    let back = HyperbolicH3.pose_from_chart(&data).unwrap();
    let placed = |pose: &Pose<HyperbolicH3>, local| {
        HyperbolicH3
            .place(&HyperbolicH3.prepare(pose), local)
            .unwrap()
    };
    for local in [Vec3::X * 0.1, Vec3::Y * 0.1, Vec3::Z * 0.1] {
        let (there, back_there) = (placed(&pose, local), placed(&back, local));
        assert!(
            (there - back_there).length() <= 1e-5,
            "the round trip put {local} at {back_there} rather than {there}"
        );
    }
    let position = Vec3::from_slice(&data.coordinates[..3]);
    assert!((position - Vec3::new(0.3, -0.1, 0.2)).length() <= 1e-5);
    assert!((back.point - pose.point).length() <= 1e-5);
}

#[test]
fn klein_image_point_disagrees_with_the_lorentz_embedding_by_hand() {
    let (reach, theta) = (0.3_f32, 0.5_f32);
    let eye = HyperbolicH3.iso_compose(
        Iso3H::from_translation(Vec3::new(reach, 0.0, 0.0)),
        spin_z(theta),
    );
    let p = Vec3::new(0.1, 0.2, -0.3);

    let r2 = p.length_squared();
    let (hx, hy, hz, hw) = (
        2.0 * p.x / (1.0 - r2),
        2.0 * p.y / (1.0 - r2),
        2.0 * p.z / (1.0 - r2),
        (1.0 + r2) / (1.0 - r2),
    );
    let rapidity = 2.0 * reach.atanh();
    let (bx, bw) = (
        rapidity.cosh() * hx - rapidity.sinh() * hw,
        -rapidity.sinh() * hx + rapidity.cosh() * hw,
    );
    let (rx, ry) = (
        theta.cos() * bx + theta.sin() * hy,
        -theta.sin() * bx + theta.cos() * hy,
    );
    let expected = [rx / bw, ry / bw, hz / bw];

    let image = Klein.image_point(&Pose::from(eye), p).unwrap();
    for (got, want) in image.iter().zip(expected) {
        assert!((got - want).abs() <= 1e-5, "{image:?} against {expected:?}");
    }
}

#[test]
fn pick_returns_the_wrong_domains_entity_or_a_projection_claims_a_hit_point() {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let h3 = session
        .register_domain(DomainBuilder::new("h3", HyperbolicH3).tracked(LogCapacity::default()));
    let stub4 = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[0.05, 0.0, 0.0, 0.0], [-0.05, 0.0, 0.0, 0.0]]],
    });
    let stub3 = session.prepare(PreparedGeometry::Lines3 {
        segments: vec![[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let center3 = Vec3::new(-0.2, 0.0, -0.5);
    let (object4, object3) = session.dispatch(|d| {
        let eye4 = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        let eye3 = d
            .spawn(SpawnBundle::new().at(h3, Pose::at(Vec3::ZERO)))
            .unwrap();
        d.domains
            .typed(r4)
            .unwrap()
            .add_view(ViewSpec::new(root, eye4, Projection4 { focal: 2.0 }))
            .unwrap();
        d.domains
            .typed(h3)
            .unwrap()
            .add_view(ViewSpec::new(root, eye3, Klein))
            .unwrap();
        let object4 = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(Vec4::new(1.0, 0.0, -4.0, 0.0)))
                    .instance(Instance::new(stub4, material)),
            )
            .unwrap();
        let object3 = d
            .spawn(
                SpawnBundle::new()
                    .at(h3, Pose::at(center3))
                    .instance(Instance::new(stub3, material)),
            )
            .unwrap();
        (object4, object3)
    });
    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let image_of = |entity| {
        publication
            .views
            .iter()
            .flat_map(|view| view.records.instances.rows())
            .find(|record| record.entity == entity)
            .unwrap()
            .image_point
    };
    let ndc4 = session.views().ndc(image_of(object4)).unwrap();
    let ndc3 = session.views().ndc(image_of(object3)).unwrap();

    let pick4 = session.pick(ndc4).unwrap();
    assert_eq!(
        (pick4.entity, pick4.domain, pick4.hit),
        (object4, r4.id(), None)
    );
    let pick3 = session.pick(ndc3).unwrap();
    assert_eq!((pick3.entity, pick3.domain), (object3, h3.id()));
    let hit = Vec3::from_slice(&pick3.hit.unwrap().coordinates[..3]);
    let radius = 2.0 * 0.1_f32.atanh();
    assert!((HyperbolicH3.distance(hit, center3) - radius).abs() <= 1e-3);
    assert_eq!(session.pick([0.0, 0.9]), None);

    let image3 = Vec3::from(image_of(object3));
    let mut place4 = |scale: f32| {
        let point = image3 * scale;
        session
            .domains_mut()
            .typed(r4)
            .unwrap()
            .set_point(object4, point.extend(0.0))
            .unwrap();
        session.pick(ndc3).map(|pick| pick.entity)
    };
    assert_eq!(place4(0.5), Some(object4));
    assert_eq!(place4(2.0), Some(object3));
}

#[test]
fn the_h3_chart_reach_stays_finite_past_the_ball() {
    use loam_runtime::DomainSpace;

    let at_bound =
        HyperbolicH3.chart_reach(Vec3::ZERO, loam_math::hyperbolic::POINCARE_R2_MAX.sqrt());
    for radius in [1.0_f32, 1.5, 40.0] {
        let reach = HyperbolicH3.chart_reach(Vec3::ZERO, radius);
        assert!(reach.is_finite(), "radius {radius} gave reach {reach}");
        assert_eq!(
            reach, at_bound,
            "radius {radius} was not clamped to the ball"
        );
    }
    assert!(HyperbolicH3.chart_reach(Vec3::ZERO, 0.5) < at_bound);
}

#[test]
fn the_klein_depth_envelope_shrinks_as_the_eye_leaves_the_origin() {
    use loam_runtime::view::klein_depth_envelope;

    let assumed = klein_depth_envelope(1.0).far;
    assert!((assumed - 6.0).abs() <= 1e-6, "{assumed}");
    let admitted = klein_depth_envelope(HyperbolicH3.chart_envelope()).far;
    assert!((admitted - 1.0).abs() <= 1e-6, "{admitted}");
    assert!(klein_depth_envelope(0.5).far >= klein_depth_envelope(2.0).far);
    assert!(klein_depth_envelope(2.0).far >= klein_depth_envelope(4.0).far);
}

#[test]
fn move_to_parallel_transports_around_an_h3_geodesic_triangle() {
    let origin = Vec3::ZERO;
    let a = HyperbolicH3.exp(origin, Vec3::X * 0.5);
    let b = HyperbolicH3.exp(origin, Vec3::Y * 0.5);
    let path = [origin, a, b, origin];
    let (mut session, h3, walker) = h3_walker();
    let domain = session.domains_mut().typed(h3).unwrap();

    for index in 1..path.len() {
        domain
            .move_to(walker, HyperbolicH3.chart_point(path[index]))
            .unwrap();
        let pose = *domain.poses().get(walker).unwrap();
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            let got = HyperbolicH3.carry(&pose, Vec3::ZERO, axis).unwrap();
            let want = HyperbolicH3.parallel_transport_along(&path[..=index], axis);
            assert!((got - want).length() <= 1e-4, "{got:?} is not {want:?}");
        }
    }

    let pose = *domain.poses().get(walker).unwrap();
    let carried = HyperbolicH3.carry(&pose, Vec3::ZERO, Vec3::X).unwrap();
    assert!(pose.point.length() <= 1e-5);
    assert!((carried - Vec3::X).length() > 0.1);
}

#[test]
fn h3_admission_and_movement_obey_the_numerical_envelope() {
    let (mut session, h3, walker) = h3_walker();
    let domain = session.domains_mut().typed(h3).unwrap();
    let outside = Vec3::new(0.997, 0.0, 0.0);
    assert_eq!(HyperbolicH3.check(outside), Err(DomainError::ErrorBudget));
    assert_eq!(
        domain.move_to(walker, HyperbolicH3.chart_point(outside)),
        Err(DomainError::ErrorBudget)
    );
    assert_eq!(domain.poses().get(walker).unwrap().point, Vec3::ZERO);

    let inside = Vec3::new(0.99, 0.0, 0.0);
    assert_eq!(HyperbolicH3.check(inside), Ok(()));
    assert_eq!(
        domain.move_to(walker, HyperbolicH3.chart_point(inside)),
        Ok(())
    );
}
