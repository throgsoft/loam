use std::ops::Range;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use glam::{Vec3, Vec4};
use loam_app::session::run;
use loam_math::blended::{BlendedSpace, LinearBlendX};
use loam_math::{EuclideanR3, EuclideanR4, HyperbolicH3, Iso3, Mat3};
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, register_default_narrowphase, sphere_body_r4,
};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionEvent, ActionId, Bindings, BridgeSpec, Ctx, DepthEnvelope, DomainBuilder,
    DomainError, DomainHandle, DomainRay, DomainSpace, Domains, Entity, ImageRay, Input, Instance,
    Key, Klein, LogCapacity, Material, Phase, PhysicsConfig, Pick, Placement, Pose,
    PreparedGeometry, Projection4, Publication, Rejection, Rigid, Section4, Session, SimConfig,
    SpawnBundle, Step, ViewId, ViewMapping, ViewSpec,
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
const BRIDGE_POSITION: Vec3 = Vec3::new(0.5, 0.0, -1.2);
const BRIDGE_SCALE: f32 = 0.2;
const SECTION_W: f32 = 0.0;
const DRAG_NDC: f32 = 0.15;
const DRAG_SECONDS: f64 = 0.25;
const EDIT: ActionId = ActionId(4);
const EDIT_STEP: f32 = 0.05;
const LATENCY_SAMPLES: usize = 128;
const BALL_RADIUS: f32 = 0.5;
const BALL_MASS: f32 = 1.0;
const BALL_SPAWN: Vec4 = Vec4::new(0.0, 0.8, 0.0, 0.0);
const GRAVITY: f32 = 1.0;
const FLOOR_HEIGHT: f32 = 0.0;
const BLEND_START: f32 = -0.5;
const BLEND_END: f32 = 0.5;
const BLEND_EYE: Vec3 = Vec3::new(-0.7, 0.0, 0.0);
const BLEND_TURN: f32 = 0.3;
const BLEND_IMAGE: Vec3 = Vec3::new(0.0, -0.0866, -0.25);
const BLEND_SPAN: f32 = 0.05;
const BLEND_STEP: f32 = 0.5;
const PUBLISH_SAMPLES: usize = 64;

#[derive(Clone, Copy)]
struct Player {
    speed: f32,
}

#[derive(Clone, Copy, Default)]
struct Edit {
    step: u32,
    submitted: Option<Instant>,
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct TwoSpaceStores {
        players: Store<Player>,
        last_pick: Value<Option<Pick>>,
        edits: Value<Edit>,
    }
}

struct Scene {
    r4: DomainHandle<EuclideanR4>,
    h3: DomainHandle<HyperbolicH3>,
    blend: DomainHandle<Blend>,
    landmark4: Entity,
    landmark3: Entity,
    walker4: Entity,
    blend_eye: Entity,
    blend_mark: Entity,
    blend_view: ViewId,
    ball: Option<Entity>,
}

type Blend = BlendedSpace<EuclideanR3, HyperbolicH3, LinearBlendX>;

fn blend_space() -> Option<Blend> {
    Some(BlendedSpace::new(
        EuclideanR3,
        HyperbolicH3,
        LinearBlendX::new(BLEND_START, BLEND_END)?,
    ))
}

struct ChartRelative;

impl<S: DomainSpace<Point = Vec3, Frame = Mat3>> ViewMapping<S> for ChartRelative {
    fn name(&self) -> &'static str {
        "chart-relative"
    }

    fn image_point(&self, eye: &Pose<S>, point: Vec3) -> Option<[f32; 3]> {
        let inverse = eye.frame.inverse();
        inverse
            .is_finite()
            .then(|| (inverse * (point - eye.point)).to_array())
    }

    fn lift(&self, _eye: &Pose<S>, _ray: &ImageRay) -> Option<DomainRay<S>> {
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
        .key(Key::Letter('e'), EDIT)
}

fn build(physics: bool) -> Result<(Session<TwoSpaceStores>, Scene), HostError> {
    let mut session = Session::new(TwoSpaceStores::default(), SimConfig::default());
    let flat = DomainBuilder::new("r4", EuclideanR4).tracked(LogCapacity::default());
    let r4 = session.register_domain(if physics {
        flat.physics(
            PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::NEG_Y * GRAVITY),
        )
    } else {
        flat
    });
    let h3 = session
        .register_domain(DomainBuilder::new("h3", HyperbolicH3).tracked(LogCapacity::default()));
    let missing = || HostError::Host("the blend zone has no width".into());
    let blend = session.register_domain(
        DomainBuilder::new("blend", blend_space().ok_or_else(missing)?)
            .tracked(LogCapacity::default())
            .fields()
            .marched(),
    );
    let space = blend_space().ok_or_else(missing)?;
    let topology = Polytope4::Tesseract.topology();
    let edges4 = session.prepare(PreparedGeometry::edges_of(topology, 1.0));
    let edges3 = session.prepare(tetrahedron_edges());
    let mark_edges = session.prepare(PreparedGeometry::Lines3 {
        segments: vec![[[BLEND_SPAN, 0.0, 0.0], [-BLEND_SPAN, 0.0, 0.0]]],
    });
    let white = session.add_material(Material::lines([1.0, 1.0, 1.0, 0.95], 1.6));
    let root = session.views().root();

    type Built = (
        Entity,
        Entity,
        Entity,
        Entity,
        Entity,
        ViewId,
        Option<Entity>,
    );
    let (landmark4, landmark3, walker4, blend_eye, blend_mark, blend_view, ball) = session
        .dispatch(|d| -> Result<Built, Rejection> {
            let landmark4 = d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose::at(LANDMARK_R4))
                    .instance(Instance::new(edges4, white)),
            )?;
            let landmark3 = d.spawn(
                SpawnBundle::new()
                    .at(h3, Pose::at(LANDMARK_H3))
                    .instance(Instance::new(edges3, white)),
            )?;
            let player = Player { speed: WALK_SPEED };
            let walker4 = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)).row(player))?;
            let walker3 = d.spawn(SpawnBundle::new().at(h3, Pose::at(Vec3::ZERO)).row(player))?;
            let projection = Projection4 {
                focal: FOCAL_DISTANCE,
            };
            d.domains
                .typed(r4)?
                .add_view(ViewSpec::new(root, walker4, projection));
            d.domains
                .typed(h3)?
                .add_view(ViewSpec::new(root, walker3, Klein));
            let turn = Mat3::from_rotation_y(BLEND_TURN);
            let blend_eye = d.spawn(SpawnBundle::new().at(
                blend,
                Pose {
                    point: BLEND_EYE,
                    frame: turn,
                },
            ))?;
            let blend_mark = d.spawn(
                SpawnBundle::new()
                    .at(blend, Pose::new(&space, BLEND_EYE + turn * BLEND_IMAGE))
                    .instance(Instance::new(mark_edges, white)),
            )?;
            let blend_view =
                d.domains
                    .typed(blend)?
                    .add_view(ViewSpec::new(root, blend_eye, ChartRelative));
            let ball = match physics {
                false => None,
                true => {
                    let ball = d.spawn(SpawnBundle::new().at(r4, Pose::at(BALL_SPAWN)))?;
                    let world = d
                        .domains
                        .typed(r4)?
                        .physics_mut()
                        .ok_or(Rejection::Unsupported("physics on r4"))?;
                    let floor = halfspace4_body_r4(Vec4::Y, FLOOR_HEIGHT)
                        .ok_or(Rejection::Unsupported("half-space floor"))?;
                    world.world_mut().push_body(floor);
                    let sphere = sphere_body_r4(BALL_SPAWN, Vec4::ZERO, BALL_RADIUS, BALL_MASS)
                        .ok_or(Rejection::Unsupported("hypersphere body"))?;
                    world.spawn(ball, sphere);
                    Some(ball)
                }
            };
            Ok((
                landmark4, landmark3, walker4, blend_eye, blend_mark, blend_view, ball,
            ))
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

    session.system(
        Phase::Dispatch,
        "edit",
        Access::new().writes::<Edit>().commands().domain(r4.id()),
        move |ctx: Ctx<'_, TwoSpaceStores>| {
            if !ctx.input.is_held(EDIT) {
                return;
            }
            let step = ctx.app.edits.get().step + 1;
            ctx.commands.app_fn("place-landmark", move |dispatch| {
                if let Ok(domain) = dispatch.domains.typed(r4) {
                    if let Some(pose) = domain.poses.get_mut(landmark4) {
                        pose.point = LANDMARK_R4 + Vec4::X * (step as f32 * EDIT_STEP);
                    }
                }
            });
            ctx.app.edits.set(Edit {
                step,
                submitted: Some(Instant::now()),
            });
        },
    );

    session.set_initial()?;
    let scene = Scene {
        r4,
        h3,
        blend,
        landmark4,
        landmark3,
        walker4,
        blend_eye,
        blend_mark,
        blend_view,
        ball,
    };
    Ok((session, scene))
}

fn ball_height(session: &mut Session<TwoSpaceStores>, scene: &Scene) -> Option<f32> {
    let ball = scene.ball?;
    let domain = session.domains_mut().typed(scene.r4).ok()?;
    Some(domain.poses.get(ball)?.point.y)
}

fn blended(session: &mut Session<TwoSpaceStores>, scene: &Scene) -> Result<bool, HostError> {
    let lost = |what: &'static str| HostError::Host(what.into());
    let mut publication = Publication::default();
    session.publish(&mut publication)?;
    let record = published_point(&publication, scene.blend_mark, Some(scene.blend_view))
        .ok_or_else(|| lost("the blended mark published no record"))?;
    let ndc = session
        .views()
        .ndc(record)
        .ok_or_else(|| lost("the blended record left the root frustum"))?;
    let pick = session
        .pick(ndc)
        .ok_or_else(|| lost("the blended mark took no pick"))?;
    let placed = Vec3::from(record) - BLEND_IMAGE;
    let apart = (Vec3::from(pick.image_point) - Vec3::from(record)).length();
    println!(
        "blend mark at ({:.3}, {:.3}): record ({:.4}, {:.4}, {:.4}) off the eye frame by {:.6}, pick {} entry {:.4} ahead of it",
        ndc[0],
        ndc[1],
        record[0],
        record[1],
        record[2],
        placed.length(),
        u32::from(pick.entity == scene.blend_mark),
        apart
    );

    let compiled = session
        .domains_mut()
        .facade(scene.blend.id())
        .ok_or_else(|| lost("the blended domain went missing"))?
        .compile_fields();
    println!("blend compile_fields: {compiled:?}");

    let walked = session.domains_mut().typed(scene.blend)?.walk(
        scene.blend_eye,
        Vec3::NEG_X * BLEND_STEP,
        1.0,
    );
    println!("blend walk past the ball: {walked:?}");

    let mut moved_blend = Vec::with_capacity(PUBLISH_SAMPLES);
    let mut moved_r4 = Vec::with_capacity(PUBLISH_SAMPLES);
    let mut still = Vec::with_capacity(PUBLISH_SAMPLES);
    let (mark, landmark4) = (scene.blend_mark, scene.landmark4);
    for _ in 0..PUBLISH_SAMPLES {
        let _ = session
            .domains_mut()
            .typed(scene.blend)?
            .poses
            .get_mut(mark);
        let start = Instant::now();
        session.publish(&mut publication)?;
        moved_blend.push(start.elapsed());
        let _ = session
            .domains_mut()
            .typed(scene.r4)?
            .poses
            .get_mut(landmark4);
        let start = Instant::now();
        session.publish(&mut publication)?;
        moved_r4.push(start.elapsed());
        let start = Instant::now();
        session.publish(&mut publication)?;
        still.push(start.elapsed());
    }
    println!(
        "blend publish over {PUBLISH_SAMPLES} samples: median {:.1} us blend moved, {:.1} us r4 moved, {:.1} us unchanged",
        median(moved_blend).as_secs_f64() * 1e6,
        median(moved_r4).as_secs_f64() * 1e6,
        median(still).as_secs_f64() * 1e6
    );

    Ok(pick.entity == scene.blend_mark
        && pick.domain == scene.blend.id()
        && placed.length() <= 1e-5
        && (apart - BLEND_SPAN).abs() <= 1e-4
        && matches!(
            compiled,
            Err(DomainError::Unsupported("curved field chart"))
        )
        && walked == Err(DomainError::ChartBoundary))
}

fn published_point(
    publication: &Publication<TwoSpaceStores>,
    entity: Entity,
    through: Option<ViewId>,
) -> Option<[f32; 3]> {
    publication
        .views
        .iter()
        .filter(|view| through.is_none_or(|id| view.target.view == id))
        .flat_map(|view| {
            view.records
                .instances
                .rows()
                .iter()
                .map(|record| (record.entity, view.placement.apply(record.image_point)))
        })
        .find(|(row, _)| *row == entity)
        .map(|(_, point)| point)
}

fn bridged(session: &mut Session<TwoSpaceStores>, scene: &Scene) -> Result<bool, HostError> {
    let refused = |what: &str, error: String| HostError::Host(format!("{what} refused: {error}"));
    let root = session.views().root();
    let section = session.dispatch(|d| -> Result<ViewId, Rejection> {
        Ok(d.domains.typed(scene.r4)?.add_view(ViewSpec::new(
            root,
            scene.walker4,
            Section4 { w: SECTION_W },
        )))
    })?;
    session
        .bridge(BridgeSpec {
            anchor: scene.landmark3,
            into: root,
            source: scene.r4.id(),
            view: section,
            placement: Placement::Rigid(Rigid {
                pose: Iso3::from_translation(BRIDGE_POSITION),
                scale: BRIDGE_SCALE,
            }),
        })
        .map_err(|error| refused("bridge", format!("{error:?}")))?;

    let mut publication = Publication::default();
    session.publish(&mut publication)?;
    let placed = published_point(&publication, scene.landmark4, Some(section));
    let Some([x, y]) = placed.and_then(|point| session.views().ndc(point)) else {
        println!("r4 landmark through the bridge: not published");
        return Ok(false);
    };

    let pick = session
        .grab([x, y], 0.0)
        .map_err(|error| refused("grab", format!("{error:?}")))?;
    let key = pick.entity.key();
    println!(
        "r4 landmark through the bridge at ({x:.3}, {y:.3}): pick returned entity {}.{} in image space {}, hit {:?}",
        key.slot(),
        key.generation(),
        pick.image.index(),
        pick.hit.map(|hit| hit.coordinates)
    );
    session
        .drag([x + DRAG_NDC, y], DRAG_SECONDS)
        .map_err(|error| refused("drag", format!("{error:?}")))?;
    session.boundary(Input::default())?;
    let at = session
        .domains_mut()
        .typed(scene.r4)?
        .poses
        .get(scene.landmark4)
        .map(|pose| pose.point)
        .ok_or_else(|| HostError::Host("the landmark lost its pose".into()))?;
    println!(
        "r4 landmark dragged by ndc ({DRAG_NDC:.3}, 0.000): position ({:.4}, {:.4}, {:.4}, {:.4})",
        at.x, at.y, at.z, at.w
    );
    Ok(pick.entity == scene.landmark4 && pick.view == section && pick.image != root)
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
        let ndc = published_point(&publication, landmark, None)
            .and_then(|point| session.views().ndc(point));
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
    let matched = all_matched && bridged(session, scene)? && blended(session, scene)?;
    if scene.ball.is_some() {
        let y = ball_height(session, scene)
            .ok_or_else(|| HostError::Host("the r4 ball lost its pose".into()))?;
        println!("r4 ball after {steps} ticks: y = {y:.4}");
    }
    Ok(matched)
}

fn edit_input() -> Input {
    Input {
        actions: vec![ActionEvent {
            action: EDIT,
            pressed: true,
        }],
        held: vec![EDIT],
        ..Input::default()
    }
}

fn median<T: Copy + Ord>(mut samples: Vec<T>) -> T {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn edit_latency(
    session: &mut Session<TwoSpaceStores>,
    scene: &Scene,
    samples: usize,
) -> Result<(), HostError> {
    let mut publication = Publication::default();
    session.publish(&mut publication)?;
    let mut ticks_seen: Vec<u32> = Vec::with_capacity(samples);
    let mut wall_seen: Vec<Duration> = Vec::with_capacity(samples);

    for _ in 0..samples {
        let before = published_point(&publication, scene.landmark4, None);
        session.boundary(edit_input())?;
        let mut ticks = 0;
        loop {
            session.tick()?;
            session.publish(&mut publication)?;
            if published_point(&publication, scene.landmark4, None) != before {
                break;
            }
            ticks += 1;
            session.boundary(Input::default())?;
        }
        let submitted = session
            .app
            .edits
            .get()
            .submitted
            .ok_or_else(|| HostError::Host("the edit system submitted no command".into()))?;
        wall_seen.push(submitted.elapsed());
        ticks_seen.push(ticks);
    }

    println!(
        "twospace edit-to-result over {samples} samples: median {} ticks, {:.4} ms",
        median(ticks_seen),
        median(wall_seen).as_secs_f64() * 1000.0
    );
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|arg| arg == "--edit-latency") {
        let outcome = build(false).and_then(|(mut session, scene)| {
            edit_latency(&mut session, &scene, LATENCY_SAMPLES).map(|()| true)
        });
        return match outcome {
            Ok(_) => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("twospace: {error:?}");
                ExitCode::FAILURE
            }
        };
    }
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
            build(true).and_then(|(mut session, scene)| headless(&mut session, &scene, steps))
        }
        None => build(false).and_then(|(session, _)| {
            run(session, HostConfig::new("twospace", bindings())).map(|()| true)
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

    const SETTLE_TICKS: u32 = 120;
    const HAND_REST_Y: f32 = 0.493_611_1;

    #[test]
    fn the_r4_ball_settles_at_the_hand_derived_rest_height() {
        let (mut session, scene) = build(true).unwrap();
        let config = HostConfig::new("twospace", bindings());
        let holds = [(Key::Letter('w'), HEADLESS_WALK)];
        host::run_headless(&mut session, &config, SETTLE_TICKS, &holds).unwrap();
        let y = ball_height(&mut session, &scene).unwrap();
        assert!(
            (y - HAND_REST_Y).abs() < 1e-5,
            "the ball rested at {y}, not {HAND_REST_Y}"
        );
    }

    #[test]
    fn reset_keeps_relations_but_a_handle_from_before_it_still_resolves() {
        let (mut session, scene) = build(false).unwrap();
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
                .any(|pose| pose.point != Vec4::ZERO);
            let h3 = domains.typed(scene.h3).unwrap();
            let moved3 = players
                .iter()
                .filter_map(|player| h3.poses.get(*player))
                .any(|pose| pose.point != Vec3::ZERO);
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
                (session.scene(), scene.blend_mark.key()),
            ]
        );
    }
}
