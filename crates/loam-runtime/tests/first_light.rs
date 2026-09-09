use std::f32::consts::FRAC_PI_2;

use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Iso3H, IsometryGroup, Space};
use loam_runtime::{
    ChartId, ChartPose, DomainBuilder, DomainError, DomainHandle, DomainSpace, Entity, Eye,
    Instance, Klein, LogCapacity, Material, Orbit, Pose, PreparedGeometry, Projection4,
    Publication, Session, SimConfig, SpawnBundle, ViewMapping, ViewSpec,
};

type Vec3 = <EuclideanR3 as Space>::Point;
type Vec4 = <EuclideanR4 as Space>::Point;

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
    let placed =
        |pose: &Pose<HyperbolicH3>, local| HyperbolicH3.place(&HyperbolicH3.prepare(pose), local);
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
        d.domains.typed(r4).unwrap().add_view(ViewSpec::new(
            root,
            eye4,
            Projection4 { focal: 2.0 },
        ));
        d.domains
            .typed(h3)
            .unwrap()
            .add_view(ViewSpec::new(root, eye3, Klein));
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
            .poses
            .get_mut(object4)
            .unwrap()
            .point = point.extend(0.0);
        session.pick(ndc3).map(|pick| pick.entity)
    };
    assert_eq!(place4(0.5), Some(object4));
    assert_eq!(place4(2.0), Some(object3));
}

#[test]
fn transport_has_the_wrong_signed_holonomy_or_loses_normalization_on_a_geodesic_triangle() {
    let (a, b) = (1.0_f32, 1.0_f32);
    // Gauss-Bonnet: holonomy = K · area, with the right triangle at the origin giving tan β = tanh b / sinh a, tan γ = tanh a / sinh b, area = π/2 - β - γ.
    let beta = (b.tanh() / a.sinh()).atan();
    let gamma = (a.tanh() / b.sinh()).atan();
    let c = (a.cosh() * b.cosh()).acosh();
    let area = FRAC_PI_2 - beta - gamma;
    let turn = beta + gamma;
    let forward = [
        ([1.0, 0.0], a),
        ([-beta.cos(), beta.sin()], c),
        ([turn.cos(), -turn.sin()], b),
    ];
    let reverse = [
        ([0.0, 1.0], b),
        ([gamma.sin(), -gamma.cos()], c),
        ([-turn.sin(), turn.cos()], a),
    ];
    let retrace = [([1.0, 0.0], a), ([-1.0, 0.0], a)];
    for (legs, expected) in [
        (&forward[..], -area),
        (&reverse[..], area),
        (&retrace[..], 0.0),
    ] {
        let (mut session, h3, walker) = h3_walker();
        let domain = session.domains_mut().typed(h3).unwrap();
        for ([dx, dy], length) in legs {
            // The chart tangent at the origin has metric length 2|v|.
            let velocity = Vec3::new(*dx, *dy, 0.0) * (length / 2.0);
            domain.walk(walker, velocity, 1.0).unwrap();
        }
        let pose = *domain.poses.get(walker).unwrap();
        let position = pose.point;
        assert!(position.length() <= 1e-3, "loop ended at {position:?}");
        let carried = [Vec3::X, Vec3::Y, Vec3::Z]
            .map(|axis| HyperbolicH3.carry(&pose, Vec3::ZERO, axis).unwrap());
        let angle = carried[0].y.atan2(carried[0].x);
        assert!(
            (angle - expected).abs() <= 1e-3,
            "holonomy {angle} against {expected} for {legs:?}"
        );
        for (i, column) in carried.iter().enumerate() {
            assert!((column.length() - 1.0).abs() <= 1e-4);
            assert!(column.dot(carried[(i + 1) % 3]).abs() <= 1e-4);
        }
    }
}

#[test]
fn walk_leaving_the_envelope_is_reported_as_within_it() {
    let (mut session, h3, walker) = h3_walker();
    let domain = session.domains_mut().typed(h3).unwrap();
    assert_eq!(
        domain.walk(walker, Vec3::X * 10.0, 1.0),
        Err(DomainError::ChartBoundary)
    );
    let held = *domain.poses.get(walker).unwrap();
    assert_eq!(held.point, Vec3::ZERO);
    assert_eq!(HyperbolicH3.carry(&held, Vec3::ZERO, Vec3::X), Ok(Vec3::X));
    assert_eq!(domain.walk(walker, Vec3::X * 8.0, 1.0), Ok(()));
}

#[test]
fn looking_at_mirrors_the_eye_basis_or_the_orbit_sits_on_the_wrong_side_of_its_target() {
    let close = |got: [f32; 3], want: [f32; 3]| {
        assert!(
            got.iter().zip(want).all(|(g, w)| (g - w).abs() <= 1e-6),
            "{got:?} is not {want:?}"
        );
    };

    let eye = Eye::looking_at([0.0, 0.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    close(eye.forward, [0.0, 0.0, -1.0]);
    close(eye.right, [1.0, 0.0, 0.0]);
    close(eye.up, [0.0, 1.0, 0.0]);

    let orbit = Orbit {
        target: [1.0, 2.0, 3.0],
        yaw: FRAC_PI_2,
        pitch: 0.0,
        distance: 4.0,
    };
    let turned = orbit.eye();
    close(turned.position, [5.0, 2.0, 3.0]);
    close(turned.forward, [-1.0, 0.0, 0.0]);
    close(turned.right, [0.0, 0.0, -1.0]);
    close(turned.up, [0.0, 1.0, 0.0]);
}
