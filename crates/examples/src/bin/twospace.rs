use std::ops::Range;
use std::process::ExitCode;

use glam::{Vec3, Vec4};
use loam_math::{EuclideanR4, HyperbolicH3, Iso3H, Iso4Flat};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, Bindings, Ctx, DomainBuilder, DomainHandle, Domains, Entity, Input, Instance,
    Key, Klein, LogCapacity, Material, Phase, Pick, Pose, PreparedGeometry, Projection4,
    Publication, Rejection, Session, SimConfig, SpawnBundle, Step, ViewSpec,
};
use loam_shape::polytope::Polytope4;

const FORWARD: ActionId = ActionId(0);
const BACK: ActionId = ActionId(1);
const LEFT: ActionId = ActionId(2);
const RIGHT: ActionId = ActionId(3);
const WALK_SPEED: f32 = 1.5;
const FOCAL_DISTANCE: f32 = 2.0;
const LANDMARK_SCALE: f32 = 0.15;
const LANDMARK_R4: Vec4 = Vec4::new(1.5, 0.0, -4.0, 0.0);
const LANDMARK_H3: Vec3 = Vec3::new(-0.12, 0.0, -0.55);
const HEADLESS_WALK: Range<u32> = 0..6;

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

struct Scene {
    r4: DomainHandle<EuclideanR4>,
    h3: DomainHandle<HyperbolicH3>,
    landmark4: Entity,
    landmark3: Entity,
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

fn bindings() -> Bindings {
    Bindings::new()
        .key(Key::Letter('w'), FORWARD)
        .key(Key::Letter('s'), BACK)
        .key(Key::Letter('a'), LEFT)
        .key(Key::Letter('d'), RIGHT)
}

fn build() -> Result<(Session<TwoSpaceStores>, Scene), HostError> {
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

    let (landmark4, landmark3) = session.dispatch(|d| -> Result<(Entity, Entity), Rejection> {
        let landmark4 = d.spawn(
            SpawnBundle::new()
                .at(r4, Pose(Iso4Flat::from_translation(LANDMARK_R4)))
                .instance(Instance::new(edges4, white)),
        )?;
        let landmark3 = d.spawn(
            SpawnBundle::new()
                .at(h3, Pose(Iso3H::from_translation(LANDMARK_H3)))
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
        Ok((landmark4, landmark3))
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
            if strafe == 0.0 && advance == 0.0 {
                return;
            }
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
    let scene = Scene {
        r4,
        h3,
        landmark4,
        landmark3,
    };
    Ok((session, scene))
}

fn published_point(publication: &Publication<TwoSpaceStores>, entity: Entity) -> Option<[f32; 3]> {
    publication
        .views
        .iter()
        .flat_map(|view| view.records.instances.rows())
        .find(|record| record.entity == entity)
        .map(|record| record.image_point)
}

fn headless(
    session: &mut Session<TwoSpaceStores>,
    scene: &Scene,
    steps: u32,
) -> Result<bool, HostError> {
    let config = HostConfig::new("twospace", bindings());
    let holds = [(Key::Letter('w'), HEADLESS_WALK)];
    let publication = host::run_headless(session, &config, steps, &holds)?;
    let mut all_matched = true;
    for (name, landmark, domain) in [
        ("r4", scene.landmark4, scene.r4.id()),
        ("h3", scene.landmark3, scene.h3.id()),
    ] {
        let ndc =
            published_point(&publication, landmark).and_then(|point| session.views().ndc(point));
        let pick = ndc.and_then(|ndc| session.pick(ndc));
        match (ndc, pick) {
            (Some([x, y]), Some(pick)) => {
                let picked = session
                    .domains()
                    .iter()
                    .find(|candidate| candidate.id() == pick.domain)
                    .map_or("none", |candidate| candidate.name());
                let key = pick.entity.key();
                println!(
                    "{name} landmark at ({x:.3}, {y:.3}): pick returned entity {}.{} in domain {picked}, hit {:?}",
                    key.slot(),
                    key.generation(),
                    pick.hit.map(|hit| hit.coordinates)
                );
                all_matched &= pick.entity == landmark && pick.domain == domain;
            }
            _ => {
                println!("{name} landmark: no pick");
                all_matched = false;
            }
        }
    }
    Ok(all_matched)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let steps = args
        .iter()
        .position(|arg| arg == "--headless")
        .map(|at| args.get(at + 1).and_then(|steps| steps.parse::<u32>().ok()));
    let outcome = match steps {
        Some(None) => {
            eprintln!("twospace: --headless needs a step count");
            return ExitCode::FAILURE;
        }
        Some(Some(steps)) => {
            build().and_then(|(mut session, scene)| headless(&mut session, &scene, steps))
        }
        None => build().and_then(|(session, _)| {
            host::run(session, HostConfig::new("twospace", bindings())).map(|()| true)
        }),
    };
    match outcome {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => {
            eprintln!("twospace: a pick returned the wrong entity");
            ExitCode::FAILURE
        }
        Err(error) => {
            eprintln!("twospace: {error:?}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reset_keeps_relations_but_a_handle_from_before_it_still_resolves() {
        let (mut session, scene) = build().unwrap();
        let config = HostConfig::new("twospace", bindings());
        let holds = [(Key::Letter('w'), HEADLESS_WALK)];
        host::run_headless(&mut session, &config, HEADLESS_WALK.end, &holds).unwrap();
        let players = |session: &Session<TwoSpaceStores>| -> Vec<Entity> {
            session
                .app
                .players
                .iter()
                .map(|(entity, _)| entity)
                .collect()
        };
        let walked = |session: &mut Session<TwoSpaceStores>| {
            let players = players(session);
            let domains = session.domains_mut();
            let r4 = domains.typed(scene.r4).unwrap();
            let moved4 = players
                .iter()
                .filter_map(|player| r4.poses.get(*player))
                .any(|pose| pose.0.translation != Vec4::ZERO);
            let h3 = domains.typed(scene.h3).unwrap();
            let moved3 = players
                .iter()
                .filter_map(|player| h3.poses.get(*player))
                .any(|pose| pose.0 != Iso3H::IDENTITY);
            moved4 || moved3
        };
        assert!(walked(&mut session));

        session.reset().unwrap();
        assert_eq!(session.entities().resolve(scene.landmark4), None);
        assert_eq!(session.entities().resolve(scene.landmark3), None);
        assert!(!walked(&mut session));
        let players = players(&session);
        assert_eq!(players.len(), 2);
        for player in players {
            assert_eq!(session.entities().resolve(player), Some(player.key()));
            let domains = session.domains_mut();
            let placed = domains.typed(scene.r4).unwrap().poses.contains(player)
                || domains.typed(scene.h3).unwrap().poses.contains(player);
            assert!(placed, "{player:?} lost its pose");
        }
        let mut publication = Publication::default();
        session.publish(&mut publication).unwrap();
        let landmarks: Vec<_> = publication
            .views
            .iter()
            .map(|view| {
                let [record] = view.records.instances.rows() else {
                    panic!(
                        "{} records in one view",
                        view.records.instances.rows().len()
                    );
                };
                (record.entity.scene(), record.entity.key())
            })
            .collect();
        assert_eq!(
            landmarks,
            [
                (session.scene(), scene.landmark4.key()),
                (session.scene(), scene.landmark3.key()),
            ]
        );
    }
}
