use loam_math::{BlendedSpace, EuclideanR3, HyperbolicH3, LinearBlendX, Mat3, Space};
use loam_runtime::{
    ChartCommand, ChartId, ChartPose, ChartTangent, Command, DomainBuilder, Outcome, Pose, Session,
    SimConfig, SpawnBundle,
};

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Empty {}
}

type Point = <EuclideanR3 as Space>::Point;

const METRIC_SCALE: f32 = 2.090_200_4;
const ERROR_LIMIT: f32 = 5.0e-5;

fn point(coordinates: [f32; 3]) -> Point {
    Point::from_array(coordinates)
}

fn metric_error(actual: Point, expected: [f32; 3]) -> f32 {
    METRIC_SCALE * (actual - point(expected)).length()
}

#[test]
fn blended_chart_place_keeps_a_positive_orthonormal_gauge_near_the_hyperbolic_edge() {
    let space = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-2.0, -1.0).expect("blend interval"),
    );
    let mut session = Session::new(Empty::default(), SimConfig::default());
    let domain = session.register_domain(DomainBuilder::new("blend", space));
    let entity = session
        .dispatch(|dispatch| {
            dispatch.spawn(SpawnBundle::new().at_chart(
                domain.id(),
                ChartPose {
                    chart: ChartId(0),
                    coordinates: [0.99, 0.0, 0.0, 0.0],
                    frame: [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            ))
        })
        .expect("chart placement");
    let frame = session
        .domains()
        .read(domain)
        .expect("blended domain")
        .poses()
        .get(entity)
        .expect("placed pose")
        .frame;
    for (actual, expected) in [
        (frame.x_axis, point([0.009_95, 0.0, 0.0])),
        (frame.y_axis, point([0.0, 0.009_95, 0.0])),
        (frame.z_axis, point([0.0, 0.0, 0.009_95])),
    ] {
        assert!((actual - expected).length() <= 1.0e-6);
    }
}

#[test]
fn blended_chart_walk_drifts_from_the_variational_path_and_frame() {
    let space = BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(-0.5, 0.5).expect("blend interval"),
    );
    let mut session = Session::new(Empty::default(), SimConfig::default());
    let domain = session.register_domain(DomainBuilder::new("blend", space));

    let scale = 0.850_117_44;
    let frame = Mat3::from_cols(
        point([0.0, scale, 0.0]),
        point([-scale, 0.0, 0.0]),
        point([0.0, 0.0, scale]),
    );
    let entity = session
        .dispatch(|dispatch| {
            dispatch.spawn(SpawnBundle::new().at(
                domain,
                Pose {
                    point: point([-0.25, 0.125, 0.0]),
                    frame,
                },
            ))
        })
        .expect("typed spawn");

    let outcome = session
        .dispatch(|dispatch| {
            dispatch.apply(Command::Chart(
                domain.id(),
                ChartCommand::Walk {
                    entity,
                    tangent: ChartTangent {
                        chart: ChartId(0),
                        vector: [0.0, -0.806_588_9, 0.0, 0.0],
                    },
                    dt: 1.0,
                },
            ))
        })
        .expect("blended chart walk");
    assert_eq!(outcome, Outcome::Done);

    let pose = *session
        .domains()
        .read(domain)
        .expect("blended domain")
        .poses()
        .get(entity)
        .expect("stored pose");
    let endpoint_error = metric_error(pose.point, [0.25, 0.143_538_88, 0.0]);
    assert!(
        endpoint_error <= ERROR_LIMIT,
        "endpoint metric error is {endpoint_error}"
    );

    for (actual, expected) in [
        (pose.frame.x_axis, [-0.040_076_16, 0.476_741_55, 0.0]),
        (pose.frame.y_axis, [-0.476_741_55, -0.040_076_16, 0.0]),
        (pose.frame.z_axis, [0.0, 0.0, 0.478_423_03]),
    ] {
        let column_error = metric_error(actual, expected);
        assert!(
            column_error <= ERROR_LIMIT,
            "frame column metric error is {column_error}"
        );
    }
}
