use std::fmt::Debug;
use std::ops::Mul;

use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Space};
use loam_runtime::{
    DomainBuilder, DomainSpace, Identity3, Instance, Klein, Material, Pose, PreparedGeometry,
    Projection4, Publication, Session, SimConfig, SpawnBundle, ViewMapping, ViewSpec,
};

type Vec3 = <EuclideanR3 as Space>::Point;
type Vec4 = <EuclideanR4 as Space>::Point;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u8>,
    }
}

struct Trace<S: Space> {
    tangent: S::Vector,
    target: S::Point,
    eye: S::Point,
}

fn image_of(publication: &Publication, entity: loam_runtime::Entity) -> [f32; 3] {
    publication
        .views
        .iter()
        .flat_map(|view| view.records.instances.rows())
        .find(|record| record.entity == entity)
        .map(|record| record.image_point)
        .expect("the object was published")
}

fn assert_image(label: &str, got: [f32; 3], want: [f32; 3]) {
    for (g, w) in got.iter().zip(want) {
        assert!(
            (g - w).abs() <= 1e-5,
            "{label}: published {got:?}, the mapping by hand gives {want:?}"
        );
    }
}

fn trace<S, M>(space: S, label: &str, geometry: PreparedGeometry, mapping: M, trace: Trace<S>)
where
    S: DomainSpace + Clone,
    S::Point: Copy + Debug,
    S::Vector: Copy + Mul<f32, Output = S::Vector>,
    M: ViewMapping<S> + 'static,
{
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let handle = session.register_domain(DomainBuilder::new("traced", space.clone()));
    let geometry = session.prepare(geometry);
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let origin = space.origin();
    let eye_pose = Pose::new(&space, trace.eye);
    let image_at_target = mapping.image_point(&eye_pose, trace.target).unwrap();
    let image_at_origin = mapping.image_point(&eye_pose, origin).unwrap();
    let object = session.dispatch(|d| {
        let object = d
            .spawn(
                SpawnBundle::new()
                    .at(handle, Pose::new(&space, origin))
                    .instance(Instance::new(geometry, material)),
            )
            .unwrap();
        let eye = d
            .spawn(SpawnBundle::new().at(handle, Pose::new(&space, trace.eye)))
            .unwrap();
        d.domains
            .typed(handle)
            .unwrap()
            .add_view(ViewSpec::new(root, eye, mapping))
            .unwrap();
        object
    });
    let before = session.snapshot().unwrap();

    let domain = session.domains_mut().typed(handle).unwrap();
    domain.walk(object, trace.tangent, 0.5).unwrap();
    let walked = domain.poses().get(object).unwrap().point;
    let expected = space.exp(origin, trace.tangent * 0.5);
    assert!(
        space.distance(walked, expected) <= 1e-5,
        "{label}: a walk from the origin landed at {walked:?}, exp gives {expected:?}"
    );

    domain
        .move_to(object, space.chart_point(trace.target))
        .unwrap();
    let moved = domain.poses().get(object).unwrap().point;
    assert!(
        space.distance(moved, trace.target) <= 1e-5,
        "{label}: move_to landed at {moved:?}, not {:?}",
        trace.target
    );

    let mut publication = Publication::default();
    session.publish(&mut publication).unwrap();
    let image = image_of(&publication, object);
    assert_image(label, image, image_at_target);

    let ndc = session.views().ndc(image).unwrap();
    let pick = session.pick(ndc).expect("the published object is picked");
    assert_eq!((pick.entity, pick.domain), (object, handle.id()), "{label}");

    session.restore(&before).unwrap();
    let object = session
        .domains()
        .read(handle)
        .unwrap()
        .poses()
        .iter()
        .find(|(_, pose)| space.distance(pose.point, origin) <= 1e-6)
        .map(|(entity, _)| entity)
        .unwrap_or_else(|| panic!("{label}: restore left no object at the origin"));
    session.publish(&mut publication).unwrap();
    assert_image(label, image_of(&publication, object), image_at_origin);
}

#[test]
fn every_adapter_walks_moves_publishes_picks_and_restores_the_same_way() {
    trace(
        EuclideanR3,
        "r3",
        PreparedGeometry::Lines3 {
            segments: vec![[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]],
        },
        Identity3,
        Trace {
            tangent: Vec3::new(0.2, 0.1, 0.0),
            target: Vec3::new(0.3, -0.2, -1.5),
            eye: Vec3::new(0.1, 0.0, 0.5),
        },
    );
    trace(
        HyperbolicH3,
        "h3",
        PreparedGeometry::Lines3 {
            segments: vec![[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]],
        },
        Klein,
        Trace {
            tangent: Vec3::new(0.1, 0.05, 0.0),
            target: Vec3::new(0.1, 0.2, -0.3),
            eye: Vec3::new(0.2, 0.0, 0.1),
        },
    );
    trace(
        EuclideanR4,
        "r4",
        PreparedGeometry::Lines4 {
            segments: vec![[[0.05, 0.0, 0.0, 0.0], [-0.05, 0.0, 0.0, 0.0]]],
        },
        Projection4 { focal: 2.0 },
        Trace {
            tangent: Vec4::new(0.2, 0.1, 0.0, 0.3),
            target: Vec4::new(0.3, -0.2, -3.0, 0.4),
            eye: Vec4::ZERO,
        },
    );
}
