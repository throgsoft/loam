use loam_math::{EuclideanR3, EuclideanR4, Iso3, Space};
use loam_runtime::{
    BridgeError, BridgeSpec, ChartId, ChartPose, Command, Ctx, DomainBuilder, DomainError,
    DomainHandle, DomainSpace, DragError, Entity, Eye, ImageSpaceId, Input, Instance, LogCapacity,
    Material, Phase, Placement, Pose, PreparedGeometry, Projection4, Publication, Rigid, Section4,
    Session, SimConfig, SpawnBundle, ViewId, ViewMapping, ViewSpec,
};

type Vec3 = <EuclideanR3 as Space>::Point;
type Vec4 = <EuclideanR4 as Space>::Point;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Probe {
        tags: Store<u8>,
    }
}

const OBJECT: Vec4 = Vec4::new(1.0, 0.0, -4.0, 0.0);
const RADIUS: f32 = 1.0;
const SCALE: f32 = 0.25;
const SHIFT: f32 = 0.25;
const GRAB_NDC: f32 = 0.866_025_4;
const HIT: Vec4 = Vec4::new(0.552_786_4, 0.0, -3.105_572_8, 0.0);
const ROOT_POINT: [f32; 3] = [0.388_196_6, 0.0, -0.776_393_2];
const DEPTH: f32 = 0.064_400_4;
const DRAG_NDC: f32 = 0.2;
const DRAGGED_X: f32 = 1.358_600_7;
const DRAG_SPEED: f32 = 0.717_201_3;

struct Stage {
    session: Session<Probe>,
    r4: DomainHandle<EuclideanR4>,
    eye: Entity,
    object: Entity,
    anchor: Entity,
    root: ImageSpaceId,
}

fn stage(fields: bool) -> Stage {
    let mut session = Session::new(Probe::default(), SimConfig::default());
    let mut builder = DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default());
    if fields {
        builder = builder.fields();
    }
    let r4 = session.register_domain(builder);
    let stub = session.prepare(PreparedGeometry::Lines4 {
        segments: vec![[[RADIUS, 0.0, 0.0, 0.0], [-RADIUS, 0.0, 0.0, 0.0]]],
    });
    let material = session.add_material(Material::flat([1.0; 4]));
    let root = session.views().root();
    let (eye, object, anchor) = session.dispatch(|d| {
        let eye = d
            .spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))
            .unwrap();
        let object = d
            .spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(OBJECT))
                    .instance(Instance::new(stub, material)),
            )
            .unwrap();
        let anchor = d.spawn(SpawnBundle::new()).unwrap();
        (eye, object, anchor)
    });
    Stage {
        session,
        r4,
        eye,
        object,
        anchor,
        root,
    }
}

impl Stage {
    fn view(&mut self, mapping: impl ViewMapping<EuclideanR4>) -> ViewId {
        let (r4, eye, root) = (self.r4, self.eye, self.root);
        self.session.dispatch(|d| {
            d.domains
                .typed(r4)
                .unwrap()
                .add_view(ViewSpec::new(root, eye, mapping))
        })
    }

    fn bridge(
        &mut self,
        into: ImageSpaceId,
        view: ViewId,
        placement: Placement,
    ) -> Result<ImageSpaceId, BridgeError> {
        let spec = BridgeSpec {
            anchor: self.anchor,
            into,
            source: self.r4.id(),
            view,
            placement,
        };
        let link = self.session.bridge(spec)?;
        Ok(self.session.bridges().get(link).unwrap().data.image)
    }

    fn at(&mut self) -> Vec4 {
        self.session
            .domains()
            .read(self.r4)
            .unwrap()
            .poses()
            .get(self.object)
            .unwrap()
            .point
    }
}

fn shrunk() -> Placement {
    Placement::Rigid(Rigid {
        pose: Iso3::from_translation(Vec3::new(SHIFT, 0.0, 0.0)),
        scale: SCALE,
    })
}

fn turn_y(quarter: bool) -> Pose<EuclideanR3> {
    let axes: [Vec3; 3] = if quarter {
        [
            Vec3::new(0.0, 0.0, -1.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
        ]
    } else {
        [Vec3::X, Vec3::Y, Vec3::Z]
    };
    let mut frame = [[0.0; 4]; 4];
    for (slot, axis) in frame.iter_mut().zip(axes) {
        *slot = axis.extend(0.0).to_array();
    }
    frame[3] = [0.0, 0.0, 0.0, 1.0];
    EuclideanR3
        .pose_from_chart(&ChartPose {
            chart: ChartId(0),
            coordinates: [0.0; 4],
            frame,
        })
        .unwrap()
}

fn close(got: [f32; 3], want: [f32; 3], tolerance: f32) {
    assert!(
        got.iter()
            .zip(want)
            .all(|(g, w)| (g - w).abs() <= tolerance),
        "{got:?} is not {want:?}"
    );
}

#[test]
fn a_nonlinear_placement_bridges_a_raster_view() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    assert_eq!(
        stage.bridge(root, section, Placement::Nonlinear("tessellated")),
        Err(BridgeError::Nonlinear("section4"))
    );
    assert!(stage.session.bridges().is_empty());
}

#[test]
fn a_stale_bridge_eye_changes_no_view_or_link() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let eye = stage.eye;
    stage
        .session
        .dispatch(|dispatch| dispatch.despawn(eye))
        .unwrap();
    let root = stage.root;

    assert_eq!(
        stage.bridge(root, section, shrunk()),
        Err(BridgeError::Domain(DomainError::Stale(eye)))
    );
    assert!(stage.session.bridges().is_empty());
    assert_eq!(targets(&stage, section), (stage.root, stage.root));
}

#[test]
fn two_hops_compose_to_a_different_depth_or_ndc_than_the_single_placement() {
    let mut stage = stage(false);
    let outer = Rigid {
        pose: Iso3 {
            rotation: turn_y(true).frame,
            translation: Vec3::new(0.0, 0.0, -6.0),
        },
        scale: 2.0,
    };
    let inner = Rigid {
        pose: Iso3::from_translation(Vec3::X),
        scale: 0.5,
    };
    let first = stage.view(Section4 { w: 0.0 });
    let second = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    let middle = stage.bridge(root, first, Placement::Rigid(outer)).unwrap();
    let leaf = stage
        .bridge(middle, second, Placement::Rigid(inner))
        .unwrap();

    let chain = stage
        .session
        .views()
        .to_root(leaf)
        .unwrap()
        .rigid()
        .unwrap();
    let point = [2.0, 1.0, 0.0];
    close(chain.apply(point), [0.0, 1.0, -10.0], 1e-5);
    close(chain.apply(point), outer.apply(inner.apply(point)), 1e-5);
    let views = stage.session.views();
    let depth = views.depth(chain.apply(point)).unwrap();
    assert!((depth - 0.005).abs() <= 1e-7, "depth {depth}");
    let [x, y] = views.ndc(chain.apply(point)).unwrap();
    assert!(
        x.abs() <= 1e-6 && (y - 0.173_205_08).abs() <= 1e-6,
        "{x} {y}"
    );
}

#[test]
fn retargeted_view_publishes_and_picks_the_same_image() {
    let mut stage = stage(false);
    let projection = stage.view(Projection4 { focal: 2.0 });
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    let r4 = stage.r4;
    stage.session.dispatch(|d| {
        let settings = d.domains.typed(r4).unwrap().view_mut(section).unwrap();
        settings.set_mapping(Section4 { w: 0.0 });
    });
    assert_eq!(targets(&stage, section), (root, root));

    let image = stage.bridge(root, section, shrunk()).unwrap();
    assert_eq!(targets(&stage, section), (image, image));

    let mut publication = Publication::default();
    stage.session.publish(&mut publication).unwrap();
    let published = publication
        .views
        .iter()
        .find(|published| published.target.view == section)
        .unwrap();
    assert_eq!(published.target.image, image);

    let pick = stage.session.pick([GRAB_NDC, 0.0]).unwrap();
    assert_ne!(pick.view, projection);
    assert_eq!(
        (pick.entity, pick.domain, pick.view, pick.image),
        (stage.object, stage.r4.id(), section, image)
    );
    close(pick.image_point, ROOT_POINT, 1e-5);
    assert!((pick.depth - DEPTH).abs() <= 1e-5, "depth {}", pick.depth);
}

#[test]
fn a_section_pick_through_a_bridge_misses_the_analytic_r4_hit() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    stage.bridge(root, section, shrunk()).unwrap();
    let hit = stage.session.pick([GRAB_NDC, 0.0]).unwrap().hit.unwrap();
    assert_eq!(hit.chart, ChartId(0));
    for (got, want) in hit.coordinates.iter().zip(HIT.to_array()) {
        assert!(
            (got - want).abs() <= 1e-5,
            "{:?} is not {HIT:?}",
            hit.coordinates
        );
    }
}

#[test]
fn a_drag_through_a_projection_moves_the_entity_instead_of_naming_the_view() {
    let mut stage = stage(false);
    let projection = stage.view(Projection4 { focal: 2.0 });
    let root = stage.root;
    stage.bridge(root, projection, shrunk()).unwrap();
    assert_eq!(
        stage.session.grab([GRAB_NDC, 0.0], 0.0),
        Err(DragError::NoPick)
    );
    assert_eq!(
        stage.session.drag([GRAB_NDC, 0.0], 0.1),
        Err(DragError::NotGrabbed)
    );
    stage.session.boundary(Input::default()).unwrap();
    assert_eq!(stage.at(), OBJECT);
}

#[test]
fn a_drag_whose_ray_runs_parallel_to_its_constraint_moves_the_entity() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    stage.bridge(root, section, shrunk()).unwrap();
    stage.session.grab([GRAB_NDC, 0.0], 0.0).unwrap();
    stage.session.views_mut().root_mut().eye =
        Eye::looking_at([0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
    assert_eq!(
        stage.session.drag([0.0, 0.0], 0.5),
        Err(DragError::Ambiguous("section4"))
    );
    stage.session.boundary(Input::default()).unwrap();
    assert_eq!(stage.at(), OBJECT);
}

#[test]
fn a_drag_through_a_section_lands_off_the_analytic_point() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    stage.bridge(root, section, shrunk()).unwrap();
    stage.session.grab([GRAB_NDC, 0.0], 0.0).unwrap();
    stage.session.drag([GRAB_NDC + DRAG_NDC, 0.0], 0.5).unwrap();
    assert_eq!(stage.at(), OBJECT);

    stage.session.boundary(Input::default()).unwrap();
    let moved = stage.at();
    let want = Vec4::new(DRAGGED_X, 0.0, -4.0, 0.0);
    assert!((moved - want).length() <= 1e-4, "{moved:?} is not {want:?}");
    let release = stage.session.release().unwrap();
    assert!(
        (release.velocity[0] - DRAG_SPEED).abs() <= 1e-4,
        "{:?} against {DRAG_SPEED}",
        release.velocity
    );
}

#[test]
fn restore_keeps_a_bridge_made_after_the_snapshot() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let snapshot = stage.session.snapshot().unwrap();
    let root = stage.root;
    let image = stage.bridge(root, section, shrunk()).unwrap();
    assert_eq!(stage.session.bridges().len(), 1);
    assert!(stage.session.views().placed(image));
    assert_eq!(targets(&stage, section), (image, image));

    stage.session.restore(&snapshot).unwrap();
    assert!(stage.session.bridges().is_empty());
    assert!(!stage.session.views().placed(image));
    assert_eq!(targets(&stage, section), (root, root));
}

fn targets(stage: &Stage, view: ViewId) -> (ImageSpaceId, ImageSpaceId) {
    let domain = stage.session.domains().get(stage.r4.id()).unwrap();
    (
        domain.views()[view.index()].image,
        domain.view(view).unwrap().image,
    )
}

#[test]
fn a_bridge_survives_its_anchor_despawned_by_a_deferred_command() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    let image = stage.bridge(root, section, shrunk()).unwrap();
    let anchor = stage.anchor;
    stage
        .session
        .system(Phase::Dispatch, "retire", move |ctx: Ctx<'_, Probe>| {
            ctx.commands.submit(Command::Despawn(anchor));
            Ok(())
        });
    stage.session.boundary(Input::default()).unwrap();
    assert!(stage.session.bridges().is_empty());
    assert!(!stage.session.views().placed(image));
    assert_eq!(stage.session.pick([GRAB_NDC, 0.0]), None);
}

#[test]
fn a_dropped_bridge_leaks_its_image_space_slot() {
    let mut stage = stage(false);
    let section = stage.view(Section4 { w: 0.0 });
    let root = stage.root;
    let first = stage.bridge(root, section, shrunk()).unwrap();
    let retired = stage.anchor;
    stage.anchor = stage.session.dispatch(|d| {
        d.despawn(retired).unwrap();
        d.spawn(SpawnBundle::new()).unwrap()
    });
    let second = stage.bridge(root, section, shrunk()).unwrap();
    assert_eq!(second.index(), first.index());
    assert_ne!(second, first);
    assert!(!stage.session.views().placed(first));
    assert!(stage.session.views().placed(second));
    assert_eq!(stage.session.views().to_root(first), None);
}
