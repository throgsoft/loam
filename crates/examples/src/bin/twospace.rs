use glam::{Vec3, Vec4};
use loam_math::{EuclideanR4, HyperbolicH3, Iso3H, Iso4Flat};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, Bindings, Ctx, DomainBuilder, Domains, Input, Instance, Key, Klein,
    LogCapacity, Material, Phase, Pick, Pose, PreparedGeometry, Projection4, Rejection, Session,
    SimConfig, SpawnBundle, Step, ViewSpec,
};
use loam_shape::polytope::Polytope4;

const FORWARD: ActionId = ActionId(0);
const BACK: ActionId = ActionId(1);
const LEFT: ActionId = ActionId(2);
const RIGHT: ActionId = ActionId(3);
const WALK_SPEED: f32 = 1.5;
const FOCAL_DISTANCE: f32 = 2.0;
const LANDMARK_SCALE: f32 = 0.15;

#[derive(Clone, Copy)]
struct Player {
    speed: f32,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct TwoSpaceStores {
        players: Store<Player>,
        last_pick: Value<Option<Pick>>,
    }
}

fn heading(input: &Input) -> [f32; 2] {
    let axis = |negative, positive| {
        f32::from(input.is_held(positive)) - f32::from(input.is_held(negative))
    };
    [axis(LEFT, RIGHT), axis(BACK, FORWARD)]
}

fn tetrahedron_edges() -> PreparedGeometry {
    let corners = [
        Vec3::new(1.0, 1.0, 1.0),
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
    ]
    .map(|corner| (corner * LANDMARK_SCALE).to_array());
    let mut segments = Vec::new();
    for a in 0..4 {
        for b in (a + 1)..4 {
            segments.push([corners[a], corners[b]]);
        }
    }
    PreparedGeometry::Lines3 { segments }
}

fn main() -> Result<(), HostError> {
    let mut session = Session::new(TwoSpaceStores::default(), SimConfig::default());
    let r4 = session
        .register_domain(DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default()));
    let h3 = session
        .register_domain(DomainBuilder::new("h3", HyperbolicH3).tracked(LogCapacity::default()));
    let topology = Polytope4::Tesseract.topology();
    let edges4 = session.prepare(PreparedGeometry::edges_of(topology, 1.0));
    let edges3 = session.prepare(tetrahedron_edges());
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));
    let root = session.views().root();

    session.dispatch(|d| -> Result<(), Rejection> {
        let far4 = Pose(Iso4Flat::from_translation(Vec4::new(0.0, 0.0, -4.0, 0.0)));
        d.spawn(
            SpawnBundle::new()
                .at(r4, far4)
                .instance(Instance::new(edges4, white)),
        )?;
        let far3 = Pose(Iso3H::from_translation(Vec3::new(0.0, 0.0, -0.4)));
        d.spawn(
            SpawnBundle::new()
                .at(h3, far3)
                .instance(Instance::new(edges3, white)),
        )?;
        let player = Player { speed: WALK_SPEED };
        let walker4 = d.spawn(
            SpawnBundle::new()
                .at(r4, Pose(Iso4Flat::IDENTITY))
                .row(player),
        )?;
        let walker3 = d.spawn(SpawnBundle::new().at(h3, Pose(Iso3H::IDENTITY)).row(player))?;
        let projection = Projection4 {
            focal: FOCAL_DISTANCE,
        };
        d.domains
            .typed(r4)?
            .add_view(ViewSpec::new(root, walker4, projection));
        d.domains
            .typed(h3)?
            .add_view(ViewSpec::new(root, walker3, Klein));
        Ok(())
    })?;

    session.system(
        Phase::Simulation,
        "walk",
        Access::new()
            .reads::<Player>()
            .domain(r4.id())
            .domain(h3.id()),
        move |app: &mut TwoSpaceStores, domains: &mut Domains, input: &Input, step: Step| {
            let [strafe, advance] = heading(input);
            for (entity, player) in app.players.iter() {
                if let Ok(r4) = domains.typed(r4) {
                    if r4.poses.contains(entity) {
                        let velocity = Vec4::new(strafe, 0.0, -advance, 0.0) * player.speed;
                        let _ = r4.walk(entity, velocity, step.dt);
                        continue;
                    }
                }
                if let Ok(h3) = domains.typed(h3) {
                    if h3.poses.contains(entity) {
                        let velocity = Vec3::new(strafe, 0.0, -advance) * player.speed;
                        let _ = h3.walk(entity, velocity, step.dt);
                    }
                }
            }
        },
    );

    session.system(
        Phase::Dispatch,
        "pick",
        Access::new().writes::<Option<Pick>>(),
        |ctx: Ctx<'_, TwoSpaceStores>| {
            let Some(pointer) = ctx.input.began() else {
                return;
            };
            let pick = ctx.pick(pointer.ndc);
            ctx.app.last_pick.set(pick);
        },
    );

    session.set_initial()?;
    let bindings = Bindings::new()
        .key(Key::Letter('w'), FORWARD)
        .key(Key::Letter('s'), BACK)
        .key(Key::Letter('a'), LEFT)
        .key(Key::Letter('d'), RIGHT);
    host::run(session, HostConfig::new("twospace", bindings))
}
