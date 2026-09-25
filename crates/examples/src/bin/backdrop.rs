use glam::{Vec3, Vec4};
use loam::app::environment::Environment;
use loam::app::session::{launch_with, look, FrameHook, SessionApp};
use loam::app::{LaunchMode, WasmConfig};
use loam::math::{Bivector, Bivector4, EuclideanR4, Plane4, Rotor, Rotor4};
use loam::physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam::physics::manifold::PENETRATION_SLOP;
use loam::physics::EditError;
use loam::render::view::root_view_projection;
use loam::render::SkyGroundPass;
use loam::runtime::host::{HostConfig, HostError};
use loam::runtime::{
    Bindings, Ctx, DomainBuilder, DomainError, Entity, Eye, GrabConfig, Instance, Material,
    Outcome, Phase, Physics, PhysicsConfig, PointerButton, PointerPhase, Pose, PreparedGeometry,
    Rejection, Section4, Session, SimConfig, SpawnBundle, ViewSpec,
};
use loam::shape::polytope::Polytope4;

const TOY_COLOR: [f32; 4] = [0.30, 0.55, 0.95, 1.0];
const EDGE_WIDTH_PX: f32 = 1.8;
const SECTION_COLOR: [f32; 4] = [1.0, 0.85, 0.35, 1.0];
const SECTION_WIDTH_PX: f32 = 2.0;
const FLOOR_Y: f32 = 0.0;
const GRAVITY: f32 = 9.8;
const BODY_SIZE: f32 = 0.45;
const BODY_MASS: f32 = 1.0;
const BODY_RESTITUTION: f32 = 0.05;
const WALL_RESTITUTION: f32 = 0.4;
const ARENA_HALF: f32 = 3.6;
const ARENA_TOP: f32 = FLOOR_Y + 2.0 * ARENA_HALF;
const PHYSICS_FLOOR_Y: f32 = FLOOR_Y + 2.0 * PENETRATION_SLOP;
const SPAWN_CLEARANCE: f32 = 0.20;
const TICK_DT: f32 = 1.0 / 60.0;
const BASE_SUBSTEPS: u32 = 4;
const MAX_SUBSTEPS: u32 = 16;
const STEP_TRAVEL_BUDGET: f32 = 0.135;
const GRAB_STIFFNESS: f32 = 40.0;
const MAX_CARRY_SPEED: f32 = 35.0;
const MAX_GRAB_ACCELERATION: f32 = 800.0;
const RELEASE_GAIN: f32 = 0.3;
const MAX_RELEASE_SPEED: f32 = 0.5 * STEP_TRAVEL_BUDGET / (TICK_DT / MAX_SUBSTEPS as f32);
const RELEASE_SPIN_GAIN: f32 = 0.35;
const MAX_ANGULAR_SPEED: f32 =
    0.5 * STEP_TRAVEL_BUDGET / (BODY_SIZE * (TICK_DT / MAX_SUBSTEPS as f32));
const ANGULAR_DAMPING: f32 = 1.2;
const REST_TRAVEL: f32 = 0.02;
const REST_WINDOW: f32 = 30.0 * TICK_DT;
const REST_SPEED: f32 = 0.15;
const REST_ANGULAR_SPEED: f32 = 0.3;
const EYE_HEIGHT: f32 = 0.55;
const EYE_BACK: f32 = 2.6;
const LOOK_HEIGHT: f32 = 0.3;
const FOV_Y_DEGREES: f32 = 35.0;
const SUBJECT_NDC_X: f32 = -0.42;
const SCROLL_DROP: f32 = 2.4;
const REST_YAW: f32 = -0.4;

#[derive(Clone, Copy, Debug, Default)]
struct Rest {
    anchor: Vec4,
    time: f32,
}

loam::runtime::stores! {
    #[derive(Default)]
    pub struct BackdropStores {
        scroll: Value<f32>,
        rest: Value<Rest>,
    }
}

fn motion_speed(body: &loam::physics::RigidBody<EuclideanR4>) -> f32 {
    body.velocity.length() + body.angular_velocity.magnitude() * BODY_SIZE
}

fn clamp_target_to_arena(current: Vec4, target: Vec4) -> Vec4 {
    let reach = ARENA_HALF - BODY_SIZE;
    Vec4::new(
        target.x.clamp(-reach, reach),
        target
            .y
            .clamp(PHYSICS_FLOOR_Y + BODY_SIZE, ARENA_TOP - BODY_SIZE),
        target.z.clamp(-reach, reach),
        current.w,
    )
}

fn physics_config() -> PhysicsConfig<EuclideanR4> {
    PhysicsConfig::new(register_default_narrowphase)
        .gravity(Vec4::NEG_Y * GRAVITY)
        .grab(
            GrabConfig::new(GRAB_STIFFNESS, MAX_CARRY_SPEED, MAX_GRAB_ACCELERATION)
                .anchor_point(|body: Vec4, picked: Vec4| {
                    Vec4::new(picked.x, picked.y, picked.z, body.w)
                })
                .constrain_target(clamp_target_to_arena),
        )
        .adaptive_substeps(
            BASE_SUBSTEPS,
            MAX_SUBSTEPS,
            STEP_TRAVEL_BUDGET,
            motion_speed,
        )
}

fn arena() -> [(Vec4, f32, f32); 8] {
    [
        (Vec4::Y, PHYSICS_FLOOR_Y, BODY_RESTITUTION),
        (-Vec4::Y, -ARENA_TOP, WALL_RESTITUTION),
        (Vec4::X, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::X, -ARENA_HALF, WALL_RESTITUTION),
        (Vec4::Z, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::Z, -ARENA_HALF, WALL_RESTITUTION),
        (Vec4::W, -ARENA_HALF, WALL_RESTITUTION),
        (-Vec4::W, -ARENA_HALF, WALL_RESTITUTION),
    ]
}

fn resting_pose() -> Pose<EuclideanR4> {
    let topology = Polytope4::Tesseract.topology();
    let centroid: Vec4 = topology.cells[0]
        .iter()
        .map(|index| topology.vertices[*index as usize])
        .sum();
    let frame = Rotor4::from_rotation_arc(centroid.normalize(), -Vec4::Y)
        * (Plane4::Xz.unit_bivector() * REST_YAW).exp();
    let lowest = topology
        .vertices
        .iter()
        .map(|vertex| frame.apply(*vertex * BODY_SIZE).y)
        .fold(f32::INFINITY, f32::min);
    Pose {
        point: Vec4::new(0.0, SPAWN_CLEARANCE - lowest, 0.0, 0.0),
        frame,
    }
}

fn release(
    physics: &mut Physics<EuclideanR4>,
    entity: Entity,
    pointer: [f32; 3],
) -> Result<(), EditError> {
    let body = *physics
        .world()
        .body(physics.body(entity).ok_or(EditError::StaleHandle)?)
        .ok_or(EditError::StaleHandle)?;
    let flick = Vec4::new(pointer[0], pointer[1], pointer[2], 0.0);
    let throw = flick * RELEASE_GAIN;
    let linear = throw * (MAX_RELEASE_SPEED / throw.length().max(MAX_RELEASE_SPEED))
        + Vec4::W * body.velocity.w;
    let mut angular = body.angular_velocity;
    if let Some(lever) = physics
        .released_anchor(entity)
        .and_then(|handle| (handle - body.position).truncate().try_normalize())
    {
        angular =
            angular + Bivector4::wedge(lever.extend(0.0), flick) * (RELEASE_SPIN_GAIN / BODY_SIZE);
    }
    let speed = angular.magnitude();
    if speed > MAX_ANGULAR_SPEED {
        angular = angular * (MAX_ANGULAR_SPEED / speed);
    }
    physics.set_velocity(entity, linear, angular)
}

fn build(
    args: loam::app::args::Args,
) -> Result<(Session<BackdropStores>, SessionApp<BackdropStores>), HostError> {
    let mut session = Session::new(BackdropStores::default(), SimConfig::default());
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .physics(physics_config())
            .map_err(|error| HostError::Setup(Rejection::Edit(error)))?,
    );
    let tesseract = session.prepare(PreparedGeometry::Polytope4 {
        polytope: Polytope4::Tesseract,
        scale: BODY_SIZE,
    });
    let material = session.add_material(Material::lines(TOY_COLOR, EDGE_WIDTH_PX));
    let cut = session.add_material(Material::lines(SECTION_COLOR, SECTION_WIDTH_PX));
    let root = session.views().root();

    let toy = session.dispatch(|d| -> Result<Entity, Rejection> {
        let pose = resting_pose();
        let toy = d.spawn(
            SpawnBundle::new()
                .at(r4, pose)
                .instance(Instance::new(tesseract, material).sectioned(cut)),
        )?;
        let vertices = Polytope4::Tesseract
            .topology()
            .vertices
            .iter()
            .map(|vertex| *vertex * BODY_SIZE)
            .collect();
        let body = polytope_body_r4(pose.point, Vec4::ZERO, vertices, BODY_MASS)
            .ok_or(Rejection::Unsupported("invalid toy body"))?;
        let domain = d.domains.typed(r4)?;
        domain.spawn_body(toy, body)?;
        let physics = domain
            .physics_mut()
            .ok_or(Rejection::Unsupported("the domain has no physics"))?;
        physics.set_mass_properties(
            toy,
            BODY_MASS,
            regular_polytope4_inertia(Polytope4::Tesseract, BODY_MASS, BODY_SIZE),
        )?;
        physics.set_restitution(toy, BODY_RESTITUTION)?;
        for (normal, offset, restitution) in arena() {
            let wall = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
            let plane = halfspace4_body_r4(normal, offset)
                .ok_or(Rejection::Unsupported("invalid arena plane"))?;
            let domain = d.domains.typed(r4)?;
            domain.spawn_body(wall, plane)?;
            domain
                .physics_mut()
                .ok_or(Rejection::Unsupported("the domain has no physics"))?
                .set_restitution(wall, restitution)?;
        }
        let eye = d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)))?;
        let mut spec = ViewSpec::new(root, eye, Section4 { w: 0.0 });
        spec.edges = false;
        spec.section_edges = false;
        spec.section_faces = true;
        d.domains.typed(r4)?.add_view(spec)?;
        Ok(toy)
    })?;

    session.system(
        Phase::Dispatch,
        "page and pointer",
        move |mut ctx: Ctx<'_, BackdropStores>| {
            if let Some(scroll) = ctx
                .input
                .host
                .latest("scroll")
                .and_then(|values| values.first())
                .filter(|value| value.is_finite())
            {
                ctx.app.scroll.set(scroll.clamp(0.0, 1.0));
            }
            let drop = *ctx.app.scroll.get() * SCROLL_DROP;
            let aspect = ctx.views.root_mut().eye.aspect;
            let fov_y = FOV_Y_DEGREES.to_radians();
            let shift = -SUBJECT_NDC_X * EYE_BACK * (0.5 * fov_y).tan() * aspect;
            look(
                ctx.views,
                Eye {
                    fov_y,
                    ..Eye::looking_at(
                        [shift, EYE_HEIGHT - drop, EYE_BACK],
                        [shift, LOOK_HEIGHT - drop, 0.0],
                        [0.0, 1.0, 0.0],
                    )
                },
            );
            for index in 0..ctx.input.pointers.len() {
                let pointer = ctx.input.pointers[index];
                if pointer.button != Some(PointerButton::Primary) {
                    continue;
                }
                match pointer.phase {
                    PointerPhase::Began => {
                        let _ = ctx.grab(pointer.ndc, pointer.time);
                    }
                    PointerPhase::Moved
                        if ctx.dragging().is_some()
                            && ctx.drag(pointer.ndc, pointer.time).is_err() =>
                    {
                        ctx.cancel_drag();
                    }
                    PointerPhase::Ended => {
                        if let Some(release_at) = ctx.release_at(pointer.time) {
                            let pointer = release_at.velocity;
                            ctx.commands.try_app_fn("throw", move |d| {
                                let physics =
                                    d.domains.typed(r4)?.physics_mut().ok_or(
                                        Rejection::Unsupported("the domain has no physics"),
                                    )?;
                                release(physics, toy, pointer).map_err(Rejection::Edit)?;
                                Ok(Outcome::Done)
                            });
                        }
                    }
                    PointerPhase::Cancelled => {
                        ctx.cancel_drag();
                    }
                    _ => {}
                }
            }
            Ok(())
        },
    );

    session.system(
        Phase::Simulation,
        "settle",
        move |ctx: Ctx<'_, BackdropStores>| -> Result<(), DomainError> {
            let physics = ctx
                .domains
                .typed(r4)?
                .physics_mut()
                .ok_or(DomainError::Unsupported("physics on r4"))?;
            let Some(id) = physics.body(toy) else {
                return Ok(());
            };
            let Some(mut body) = physics.world().body(id).copied() else {
                return Ok(());
            };
            let angular = body.angular_velocity * (-ANGULAR_DAMPING * ctx.step.dt).exp();
            if angular != body.angular_velocity {
                physics.set_velocity(toy, body.velocity, angular)?;
                body.angular_velocity = angular;
            }
            let rest = ctx.app.rest.get_mut();
            let moving = body.velocity.length() > REST_SPEED
                || body.angular_velocity.magnitude() > REST_ANGULAR_SPEED;
            if physics.is_held(toy)
                || moving
                || (body.position - rest.anchor).length() > REST_TRAVEL
                || (!body.is_sleeping() && rest.time >= REST_WINDOW)
            {
                rest.anchor = body.position;
                rest.time = 0.0;
                return Ok(());
            }
            rest.time += ctx.step.dt;
            if rest.time >= REST_WINDOW {
                physics.sleep_body(toy)?;
            }
            Ok(())
        },
    );

    session.set_initial()?;
    let sky = SkyGroundPass::new(Environment::default().ground(FLOOR_Y, true));
    let sky_for_frame = sky.clone();
    let mut posted: Option<[f32; 4]> = None;
    let app = SessionApp::with_args(HostConfig::new("backdrop", Bindings::new()), args)
        .debug_layer(false)
        .pass(Box::new(sky))
        .on_frame(move |hook: &mut FrameHook<'_, BackdropStores>| {
            let environment = Environment::default();
            let root = hook.session.views().root();
            if let Some(view) = hook.session.views().get(root) {
                sky_for_frame.publish(
                    &view.eye,
                    environment.sky,
                    environment.ground(FLOOR_Y, true),
                );
            }
            if let Some(rect) = screen_rect(hook) {
                if posted != Some(rect) {
                    posted = Some(rect);
                    hook.post("rect", &rect);
                }
            }
        });
    Ok((session, app))
}

fn screen_rect(hook: &FrameHook<'_, BackdropStores>) -> Option<[f32; 4]> {
    let root = hook.session.views().root();
    let clip = root_view_projection(&hook.session.views().get(root)?.eye);
    let mut rect = [1.0_f32, 1.0, 0.0, 0.0];
    let mut seen = false;
    for view in &hook.published.views {
        for triangle in view.records.triangles() {
            for corner in triangle.vertices {
                let point = clip * Vec3::from(view.placement.apply(corner)).extend(1.0);
                if point.w <= 0.0 {
                    continue;
                }
                let x = (0.5 + 0.5 * point.x / point.w).clamp(0.0, 1.0);
                let y = (0.5 - 0.5 * point.y / point.w).clamp(0.0, 1.0);
                rect = [
                    rect[0].min(x),
                    rect[1].min(y),
                    rect[2].max(x),
                    rect[3].max(y),
                ];
                seen = true;
            }
        }
    }
    seen.then_some(rect)
}

fn main() -> Result<(), HostError> {
    launch_with(
        WasmConfig {
            mode: LaunchMode::Background,
            max_pixels: Some(1920 * 1080),
            ..Default::default()
        },
        build,
    )
}
