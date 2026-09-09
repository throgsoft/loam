use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;

use ab_glyph::FontRef;
use glam::{Vec2, Vec3, Vec4};
use loam_app::args::Args;
use loam_app::capture::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};
use loam_app::environment::Environment;
use loam_app::session::{launch, FrameHook, SessionApp};
use loam_math::{Bivector4, EuclideanR4, Iso4Flat, Rotor, WPlane};
use loam_physics::body::MASK_ALL;
use loam_physics::euclidean_r4::{
    halfspace4_body_r4, polytope_body_r4, register_default_narrowphase, regular_polytope4_inertia,
};
use loam_physics::BodyId;
use loam_render::{FragmentShading, TriangleFeed};
use loam_runtime::host::{self, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, Bindings, Command, Commands, Ctx, Dispatch, DomainBuilder, DomainHandle,
    Domains, Entity, Input, Key, LogCapacity, Orbit, Order, Phase, Physics, PhysicsConfig, Pose,
    Rejection, Rigid, Session, SimConfig, SpawnBundle, Views, DOMAIN_STEP,
};
use loam_shape::polytope::{polytope_section_faces_append, Polytope4, SectionScratch};
use loam_shape::{Shape, TriangleMesh};
use loam_text::glyph::{append_field_prism, layout_word, DistanceField2D, GlyphParams, GlyphSolid};
use loam_time::director::{Ease, Track};

const WORD: &str = "LOAM";
const FONT: &[u8] = include_bytes!("../../fonts/lmroman10-bold.otf");

const TICK_HZ: u32 = 60;
const SUBSTEPS: u32 = 8;
const GRAVITY: f32 = -9.8;
const PILE_PGS_ITERS: usize = 20;
const RESTITUTION: f32 = 0.0;

const GROUP_SCENERY: u32 = 1 << 0;
const GROUP_FALLING: u32 = 1 << 1;
const GROUP_LANDED: u32 = 1 << 2;
const MASK_FALLING: u32 = GROUP_SCENERY | GROUP_LANDED;

const ASSEMBLE_FRAMES: u32 = 90;
const LETTER_STAGGER_FRAMES: u32 = 12;
const LETTER_SLIDE_FRAMES: u32 = 36;
const SETTLE_FRAMES: u32 = 120;
const RAIN_START_FRAME: u32 = ASSEMBLE_FRAMES + SETTLE_FRAMES;
const PHYSICS_FRAMES: u32 = 360;
const FREEZE_FRAME: u32 = RAIN_START_FRAME + PHYSICS_FRAMES;
const SWEEP_FRAMES: u32 = 300;
const SEQUENCE_FRAMES: u32 = FREEZE_FRAME + SWEEP_FRAMES;

const W_ENTRY_SPAN: f32 = 0.6;
const RELEASE_CLEARANCE: f32 = 0.20;
const LETTER_MASS: f32 = 1.0;
const LETTER_COLOR: [f32; 4] = [0.92, 0.90, 0.86, 1.0];
const FLOOR_Y: f32 = 0.0;
const W_SLICE: f32 = 0.0;
const W_PER_LETTERFORM: f32 = 0.3;
const SLICE_SWEEP_RANGE: f32 = 4.0 * W_PER_LETTERFORM;
const MORPH_PAD_EM: f32 = 0.25;

const RAIN_CAP: usize = 64;
const RAIN_INTERVAL_JITTER: u32 = 4;
const RAIN_INTERVAL_MIN: u32 = 7 - RAIN_INTERVAL_JITTER;
const RAIN_SIZE: f32 = 0.30;
const RAIN_MASS: f32 = 0.75;
const RAIN_HEIGHT: (f32, f32) = (2.6, 3.6);
const RAIN_ENTRY_SPEED: f32 = 1.0;
const RAIN_W_SPREAD: f32 = 0.10;
const RAIN_Z_SPREAD: f32 = 0.20;
const RAIN_TUMBLE: f32 = 6.0;
const RAIN_SHAPES: [Polytope4; 6] = [
    Polytope4::Cell24,
    Polytope4::Pentatope,
    Polytope4::Cell600,
    Polytope4::Cell16,
    Polytope4::Tesseract,
    Polytope4::Cell120,
];

const DEFAULT_SEED: u64 = 0x10a3_5eed;
const BOOT_ORBIT_DISTANCE: f32 = 4.0;
const BOOT_ORBIT_PITCH: f32 = -0.12;

const PAUSE: ActionId = ActionId(0);
const RESEED: ActionId = ActionId(1);

#[derive(Clone)]
struct Letter {
    index: usize,
    hull: Vec<Vec4>,
    mark: Vec4,
    entry: Vec4,
    slide: Track<Vec4>,
    w_before: f32,
}

struct Scene {
    r4: DomainHandle<EuclideanR4>,
    morph: MorphField,
    half_depth: f32,
}

#[derive(Clone, Copy)]
struct Drop {
    polytope: Polytope4,
    color: [f32; 3],
}

#[derive(Clone, Copy)]
struct Held {
    body: BodyId,
    velocity: Vec4,
    spin: Bivector4,
}

#[derive(Clone, Copy)]
struct Stage {
    tick: u32,
    next_spawn: u32,
    rng: u64,
    seed: u64,
    paused: bool,
    released: bool,
    frozen: bool,
}

impl Stage {
    fn seeded(seed: u64) -> Self {
        Self {
            tick: 0,
            next_spawn: RAIN_START_FRAME,
            rng: (seed ^ 0x9e37_79b9_7f4a_7c15).max(1),
            seed,
            paused: false,
            released: false,
            frozen: false,
        }
    }

    fn frame(&self) -> u32 {
        self.tick / SUBSTEPS
    }

    fn slice(&self) -> f32 {
        let Some(since) = self.tick.checked_sub(FREEZE_FRAME * SUBSTEPS) else {
            return W_SLICE;
        };
        let phase = std::f32::consts::TAU * since as f32 / (SWEEP_FRAMES * SUBSTEPS) as f32;
        W_SLICE + SLICE_SWEEP_RANGE * phase.sin()
    }

    fn draw(&mut self) -> u64 {
        self.rng ^= self.rng >> 12;
        self.rng ^= self.rng << 25;
        self.rng ^= self.rng >> 27;
        self.rng.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
}

impl Default for Stage {
    fn default() -> Self {
        Self::seeded(DEFAULT_SEED)
    }
}

loam_runtime::stores! {
    #[derive(Default)]
    pub struct HeroStores {
        letters: Store<Letter>,
        drops: Store<Drop>,
        stage: Value<Stage>,
        held: Value<Vec<Held>>,
        environment: Value<Environment>,
        mesh: Value<TriangleMesh<3>>,
    }
}

fn unit(draw: u64) -> f32 {
    ((draw >> 40) as u32) as f32 * (1.0 / 16_777_216.0)
}

fn signed_unit(draw: u64) -> f32 {
    2.0 * unit(draw) - 1.0
}

fn lerp(a: f32, b: f32, u: f32) -> f32 {
    a + (b - a) * u
}

fn fps() -> NonZeroU32 {
    NonZeroU32::new(TICK_HZ).unwrap_or(NonZeroU32::MIN)
}

fn refused(what: impl std::fmt::Display) -> HostError {
    HostError::Host(what.to_string())
}

#[derive(Clone)]
struct MorphField {
    letters: Vec<Vec<f32>>,
    blended: DistanceField2D,
}

impl MorphField {
    fn new(solids: &[GlyphSolid], cell: f32) -> Option<Self> {
        let inked: Vec<&GlyphSolid> = solids.iter().filter(|s| !s.is_blank()).collect();
        let mut half = Vec2::ZERO;
        for solid in &inked {
            let field = solid.field()?;
            let (nx, ny) = field.sample_counts();
            let lo = field.sample_position(0, 0);
            let hi = field.sample_position(nx - 1, ny - 1);
            let centre = 0.5 * (lo + hi);
            half = half.max((hi - centre).abs());
        }
        half += Vec2::splat(MORPH_PAD_EM);
        let counts = (
            (2.0 * half.x / cell).ceil() as usize + 1,
            (2.0 * half.y / cell).ceil() as usize + 1,
        );
        let origin = -half;

        let mut letters = Vec::with_capacity(inked.len());
        for solid in &inked {
            let field = solid.field()?;
            let centre = solid
                .rigid_hull_4d()
                .map_or(Vec2::ZERO, |(c, _)| Vec2::new(c.x, c.y));
            let mut grid = Vec::with_capacity(counts.0 * counts.1);
            for j in 0..counts.1 {
                for i in 0..counts.0 {
                    let p = origin + Vec2::new(i as f32, j as f32) * cell;
                    grid.push(field.sample(p + centre));
                }
            }
            letters.push(grid);
        }
        if letters.is_empty() {
            return None;
        }
        Some(Self {
            blended: DistanceField2D::from_samples(
                origin,
                cell,
                counts.0,
                counts.1,
                vec![0.0; counts.0 * counts.1],
            )?,
            letters,
        })
    }

    fn blend_at(&mut self, u: f32) -> &DistanceField2D {
        let n = self.letters.len();
        let wrapped = u.rem_euclid(n as f32);
        let lo = wrapped.floor() as usize % n;
        let t = wrapped - wrapped.floor();
        let (a, b) = (&self.letters[lo], &self.letters[(lo + 1) % n]);
        for (out, (x, y)) in self
            .blended
            .samples_mut()
            .iter_mut()
            .zip(a.iter().zip(b.iter()))
        {
            *out = x + (y - x) * t;
        }
        &self.blended
    }
}

fn letters_of(solids: &[GlyphSolid]) -> Option<Vec<Letter>> {
    let seconds = |frames: u32| frames as f32 / TICK_HZ as f32;
    let mut letters = Vec::new();
    for (index, solid) in solids.iter().filter(|s| !s.is_blank()).enumerate() {
        let (centre, shape) = solid.rigid_hull_4d()?;
        let Shape::ConvexPolytope4D { vertices } = shape else {
            return None;
        };
        let lowest = vertices.iter().fold(f32::INFINITY, |m, v| m.min(v.y));
        let mark = Vec4::new(centre.x, RELEASE_CLEARANCE - lowest, centre.z, centre.w);
        let side = if index.is_multiple_of(2) { -1.0 } else { 1.0 };
        letters.push(Letter {
            index,
            hull: vertices,
            mark,
            entry: mark + Vec4::new(0.0, 0.0, 0.0, side * W_ENTRY_SPAN),
            slide: Track::new(),
            w_before: 0.0,
        });
    }
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for letter in &letters {
        for v in &letter.hull {
            lo = lo.min(letter.mark.x + v.x);
            hi = hi.max(letter.mark.x + v.x);
        }
    }
    let shift = -0.5 * (lo + hi);
    for letter in &mut letters {
        letter.mark.x += shift;
        letter.entry.x += shift;
        let start = LETTER_STAGGER_FRAMES * letter.index as u32;
        letter.slide = Track::new()
            .key(seconds(start), letter.entry, Ease::Linear)
            .key(
                seconds(start + LETTER_SLIDE_FRAMES),
                letter.mark,
                Ease::InOutCubic,
            );
    }
    Some(letters)
}

fn word_centre(letters: &[Letter]) -> Vec3 {
    let (mut lo, mut hi) = (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));
    for letter in letters {
        for v in &letter.hull {
            let world = letter.mark.truncate() + v.truncate();
            lo = lo.min(world);
            hi = hi.max(world);
        }
    }
    0.5 * (lo + hi)
}

fn word_span(letters: &[Letter]) -> (f32, f32) {
    let lo = letters.iter().fold(f32::INFINITY, |m, l| m.min(l.mark.x));
    let hi = letters
        .iter()
        .fold(f32::NEG_INFINITY, |m, l| m.max(l.mark.x));
    (lo - RAIN_SIZE, hi + RAIN_SIZE)
}

fn drop_color(polytope: Polytope4) -> [f32; 3] {
    match polytope {
        Polytope4::Pentatope => [0.95, 0.55, 0.30],
        Polytope4::Tesseract => [0.30, 0.55, 0.95],
        Polytope4::Cell16 => [0.55, 0.95, 0.40],
        Polytope4::Cell24 => [0.95, 0.45, 0.85],
        Polytope4::Cell120 => [0.40, 0.85, 0.85],
        Polytope4::Cell600 => [0.95, 0.85, 0.40],
    }
}

fn record_request(args: &Args) -> Option<CaptureRequest> {
    let dir = args.get("record").map(PathBuf::from);
    if dir.is_none() && !args.has_bare_flag("record") {
        return None;
    }
    Some(CaptureRequest::StartSequence {
        format: CaptureFormat::Apng,
        stage: CaptureStage::Pre,
        dir,
        name: Some("hero".into()),
        fps: Some(TICK_HZ as u16),
        scale: None,
        palette: PaletteMode::default(),
    })
}

fn bindings() -> Bindings {
    Bindings::new()
        .key(Key::Space, PAUSE)
        .key(Key::Letter('n'), RESEED)
}

fn physics_of(
    domains: &mut Domains,
    r4: DomainHandle<EuclideanR4>,
) -> Result<&mut Physics<EuclideanR4>, Rejection> {
    domains
        .typed(r4)?
        .physics_mut()
        .ok_or(Rejection::Unsupported("physics on r4"))
}

fn hold(d: &mut Dispatch<'_, HeroStores>, r4: DomainHandle<EuclideanR4>, asleep: bool) {
    let Ok(physics) = physics_of(d.domains, r4) else {
        return;
    };
    if !asleep {
        for held in d.app.held.get() {
            if physics.world_mut().wake_body(held.body).is_ok() {
                let body = &mut physics.world_mut().bodies[held.body];
                body.velocity = held.velocity;
                body.angular_velocity = held.spin;
            }
        }
        d.app.held.get_mut().clear();
        return;
    }
    let world = physics.world_mut();
    let mut held = Vec::new();
    for dense in 0..world.bodies.len() {
        let id = world.bodies.id_at(dense);
        let body = &world.bodies[id];
        if body.is_static() {
            continue;
        }
        held.push(Held {
            body: id,
            velocity: body.velocity,
            spin: body.angular_velocity,
        });
    }
    for row in &held {
        let _ = world.sleep_body(row.body);
    }
    d.app.held.set(held);
}

fn spawn_drop(
    d: &mut Dispatch<'_, HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    polytope: Polytope4,
    position: Vec4,
    tumble: Bivector4,
) -> Result<Entity, Rejection> {
    let entity = d.spawn(
        SpawnBundle::new()
            .at(r4, Pose(Iso4Flat::from_translation(position)))
            .row(Drop {
                polytope,
                color: drop_color(polytope),
            }),
    )?;
    let vertices: Vec<Vec4> = polytope
        .topology()
        .vertices
        .iter()
        .map(|v| RAIN_SIZE * *v)
        .collect();
    let physics = physics_of(d.domains, r4)?;
    let body = polytope_body_r4(
        position,
        Vec4::new(0.0, -RAIN_ENTRY_SPEED, 0.0, 0.0),
        vertices,
        RAIN_MASS,
    )
    .ok_or(Rejection::Unsupported("polytope rain body"))?;
    let id = physics.spawn(entity, body);
    let body = &mut physics.world_mut().bodies[id];
    body.restitution = RESTITUTION;
    body.angular_velocity = tumble;
    body.collision_group = GROUP_FALLING;
    body.collision_mask = MASK_FALLING;
    body.inertia = regular_polytope4_inertia(polytope, RAIN_MASS, RAIN_SIZE);
    Ok(entity)
}

fn compose(
    d: &mut Dispatch<'_, HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    morph: &mut MorphField,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    half_depth: f32,
) {
    let stage = *d.app.stage.get();
    let Ok(domain) = d.domains.typed(r4) else {
        return;
    };
    let slice = stage.slice();
    let mesh = d.app.mesh.get_mut();
    mesh.vertices.clear();
    mesh.colors.clear();
    mesh.indices.clear();
    for (entity, letter) in d.app.letters.iter() {
        let Some(pose) = domain.poses.get(entity) else {
            continue;
        };
        let base = mesh.vertices.len();
        let u = letter.index as f32 - (pose.0.translation.w - slice) / W_PER_LETTERFORM;
        if !append_field_prism(morph.blend_at(u), half_depth, LETTER_COLOR, mesh) {
            continue;
        }
        let translate = pose.0.translation.truncate();
        for v in &mut mesh.vertices[base..] {
            let posed = pose.0.rotation.apply(Vec4::new(v[0], v[1], v[2], 0.0));
            *v = (posed.truncate() + translate).to_array();
        }
    }
    for (entity, drop) in d.app.drops.iter() {
        let Some(pose) = domain.poses.get(entity) else {
            continue;
        };
        let topology = drop.polytope.topology();
        local.clear();
        local.extend(
            topology
                .vertices
                .iter()
                .map(|v| RAIN_SIZE * pose.0.rotation.apply(*v) + Vec4::W * pose.0.translation.w),
        );
        let [r, g, b] = drop.color;
        let base = mesh.vertices.len();
        polytope_section_faces_append(
            topology.edges,
            topology.cells,
            local,
            WPlane::new(slice),
            [r, g, b, 1.0],
            scratch,
            mesh,
        );
        let translate = pose.0.translation.truncate();
        for v in &mut mesh.vertices[base..] {
            v[0] += translate.x;
            v[1] += translate.y;
            v[2] += translate.z;
        }
    }
}

fn build() -> Result<(Session<HeroStores>, Scene), HostError> {
    let font = FontRef::try_from_slice(FONT).map_err(refused)?;
    let params = GlyphParams::default();
    let solids = layout_word(&font, WORD, &params).map_err(refused)?;
    let letters = letters_of(&solids).ok_or_else(|| refused("a letter has no convex hull"))?;
    let morph = MorphField::new(&solids, params.em_size / params.resolution as f32)
        .ok_or_else(|| refused("the word laid out with no ink to morph"))?;
    let centre = word_centre(&letters);
    let half_depth = 0.5 * params.depth;

    let mut session = Session::new(
        HeroStores::default(),
        SimConfig {
            fixed_hz: TICK_HZ * SUBSTEPS,
            max_ticks_per_frame: 2 * SUBSTEPS,
            ..SimConfig::default()
        },
    );
    let r4 = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::Y * GRAVITY)),
    );
    session.dispatch(|d| -> Result<(), Rejection> {
        for letter in letters {
            let entry = letter.entry;
            d.spawn(
                SpawnBundle::new()
                    .at(r4, Pose(Iso4Flat::from_translation(entry)))
                    .row(letter),
            )?;
        }
        let world = physics_of(d.domains, r4)?.world_mut();
        world.pgs_iters = PILE_PGS_ITERS;
        let floor = halfspace4_body_r4(Vec4::Y, FLOOR_Y)
            .ok_or(Rejection::Unsupported("half-space floor"))?;
        let floor = world.push_body(floor);
        world.bodies[floor].restitution = RESTITUTION;
        Ok(())
    })?;

    let mut orbit = Orbit::around(centre.to_array(), BOOT_ORBIT_DISTANCE);
    orbit.pitch = BOOT_ORBIT_PITCH;
    session.system(
        Phase::Dispatch,
        "orbit",
        Access::new().views(),
        move |input: &Input, views: &mut Views| {
            orbit.drag(input.drag());
            views.root_mut().eye = orbit.eye();
        },
    );

    session.system(
        Phase::Dispatch,
        "keys",
        Access::new().commands(),
        move |app: &mut HeroStores, input: &Input, commands: &mut Commands<HeroStores>| {
            if input.pressed(PAUSE) {
                let paused = !app.stage.get().paused;
                app.stage.get_mut().paused = paused;
                commands.app_fn("pause", move |d: &mut Dispatch<'_, HeroStores>| {
                    hold(d, r4, paused);
                });
            }
            if input.pressed(RESEED) {
                let seed = app.stage.get().seed.wrapping_add(1);
                commands.submit(Command::Reset);
                commands.app_fn("reseed", move |d: &mut Dispatch<'_, HeroStores>| {
                    d.app.stage.set(Stage::seeded(seed));
                });
            }
        },
    );

    session.system_at(
        Phase::Simulation,
        Order::Before(DOMAIN_STEP),
        "assemble",
        Access::new().writes::<Letter>().domain(r4.id()),
        move |app: &mut HeroStores, domains: &mut Domains| {
            let stage = *app.stage.get();
            if stage.paused {
                return;
            }
            let Ok(domain) = domains.typed(r4) else {
                return;
            };
            if !stage.released {
                for (entity, letter) in app.letters.iter() {
                    let Some(at) = letter.slide.sample(stage.frame(), fps()) else {
                        continue;
                    };
                    if let Some(pose) = domain.poses.get_mut(entity) {
                        pose.0.translation = at;
                    }
                }
                if stage.frame() < ASSEMBLE_FRAMES {
                    return;
                }
                let Some(physics) = domain.physics_mut() else {
                    return;
                };
                for (entity, letter) in app.letters.iter() {
                    let Some(body) =
                        polytope_body_r4(letter.mark, Vec4::ZERO, letter.hull.clone(), LETTER_MASS)
                    else {
                        continue;
                    };
                    let id = physics.spawn(entity, body);
                    physics.world_mut().bodies[id].restitution = RESTITUTION;
                }
                app.stage.get_mut().released = true;
                return;
            }
            let Some(physics) = domain.physics_mut() else {
                return;
            };
            for (entity, letter) in app.letters.iter_mut() {
                letter.w_before = physics
                    .body(entity)
                    .map_or(0.0, |id| physics.world().bodies[id].velocity.w);
            }
        },
    );

    session.system_at(
        Phase::Simulation,
        Order::After(DOMAIN_STEP),
        "settle",
        Access::new()
            .reads::<Letter>()
            .reads::<Drop>()
            .commands()
            .domain(r4.id()),
        move |ctx: Ctx<'_, HeroStores>| {
            let mut stage = *ctx.app.stage.get();
            if stage.paused || stage.frame() >= SEQUENCE_FRAMES {
                return;
            }
            if let Ok(physics) = physics_of(ctx.domains, r4) {
                for (entity, letter) in ctx.app.letters.iter() {
                    let Some(id) = physics.body(entity) else {
                        continue;
                    };
                    let scenery_only = (physics.world().manifolds.iter())
                        .filter(|(key, manifold)| {
                            !manifold.points.is_empty() && (key.0 == id || key.1 == id)
                        })
                        .all(|(key, _)| {
                            let other = if key.0 == id { key.1 } else { key.0 };
                            physics.world().bodies[other].inv_mass() == 0.0
                        });
                    let body = &mut physics.world_mut().bodies[id];
                    body.angular_velocity.xw = 0.0;
                    body.angular_velocity.yw = 0.0;
                    body.angular_velocity.zw = 0.0;
                    if scenery_only && body.velocity.w.abs() > letter.w_before.abs() {
                        body.velocity.w = letter.w_before;
                    }
                }
                for (entity, _) in ctx.app.drops.iter() {
                    let Some(id) = physics.body(entity) else {
                        continue;
                    };
                    if physics.world().bodies[id].collision_group != GROUP_FALLING {
                        continue;
                    }
                    let touched = (physics.world().manifolds.iter()).any(|(key, manifold)| {
                        (key.0 == id || key.1 == id) && !manifold.points.is_empty()
                    });
                    if touched {
                        let body = &mut physics.world_mut().bodies[id];
                        body.collision_group = GROUP_LANDED;
                        body.collision_mask = MASK_ALL;
                    }
                }
            }

            stage.tick += 1;
            if stage.frame() >= FREEZE_FRAME {
                if !stage.frozen {
                    stage.frozen = true;
                    ctx.commands
                        .app_fn("freeze", move |d: &mut Dispatch<'_, HeroStores>| {
                            hold(d, r4, true);
                        });
                }
            } else if stage.frame() >= stage.next_spawn && ctx.app.drops.len() < RAIN_CAP {
                let polytope = RAIN_SHAPES[ctx.app.drops.len() % RAIN_SHAPES.len()];
                let span = word_span(ctx.app.letters.rows());
                let position = Vec4::new(
                    lerp(span.0, span.1, unit(stage.draw())),
                    lerp(RAIN_HEIGHT.0, RAIN_HEIGHT.1, unit(stage.draw())),
                    RAIN_Z_SPREAD * signed_unit(stage.draw()),
                    RAIN_W_SPREAD * signed_unit(stage.draw()),
                );
                let tumble = Bivector4::new(
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                    RAIN_TUMBLE * signed_unit(stage.draw()),
                );
                let jitter = (stage.draw() % (2 * RAIN_INTERVAL_JITTER as u64 + 1)) as u32;
                stage.next_spawn = stage.frame() + RAIN_INTERVAL_MIN + jitter;
                ctx.commands
                    .app_fn("rain", move |d: &mut Dispatch<'_, HeroStores>| {
                        if let Err(error) = spawn_drop(d, r4, polytope, position, tumble) {
                            tracing::error!("hero: the rain was refused: {error:?}");
                        }
                    });
            }
            ctx.app.stage.set(stage);
        },
    );

    session.set_initial()?;
    Ok((
        session,
        Scene {
            r4,
            morph,
            half_depth,
        },
    ))
}

fn host(scene: Scene, record: Option<CaptureRequest>) -> SessionApp<HeroStores> {
    let feed = TriangleFeed::default();
    let Scene {
        r4,
        mut morph,
        half_depth,
    } = scene;
    let mut local: Vec<Vec4> = Vec::new();
    let mut scratch = SectionScratch::default();
    let mut composed: Option<u32> = None;
    let mut start = record;
    let mut recording = start.is_some();
    SessionApp::new(HostConfig::new("loam", bindings()))
        .pass(feed.pass(FragmentShading::FaceNormalLambert))
        .target_fps(TICK_HZ as f32)
        .on_frame(move |hook: &mut FrameHook<'_, HeroStores>| {
            let root = hook.session.views().root();
            if let Some(image) = hook.session.views().get(root) {
                feed.set_view(&image.eye, Rigid::IDENTITY);
            }
            let environment = *hook.session.app.environment.get();
            feed.set_ground(Some(environment.ground(FLOOR_Y, environment.floor_visible)));
            let tick = hook.session.app.stage.get().tick;
            if composed != Some(tick) {
                composed = Some(tick);
                hook.session
                    .dispatch(|d| compose(d, r4, &mut morph, &mut local, &mut scratch, half_depth));
                feed.edit(|into| std::mem::swap(into, hook.session.app.mesh.get_mut()));
            }
            if let Some(request) = start.take() {
                hook.capture.start(request);
            }
            if recording && hook.session.app.stage.get().frame() >= SEQUENCE_FRAMES {
                hook.capture.stop();
                recording = false;
            }
        })
        .command(
            "ground",
            "checker colours and fog: dark|light <r> <g> <b>, fog <density>, reset",
            |args, submit, out| {
                Environment::default().apply(args)?;
                let owned: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                submit.app_fn("ground", move |d: &mut Dispatch<'_, HeroStores>| {
                    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
                    match d.app.environment.get_mut().apply(&refs) {
                        Ok(line) => tracing::info!("{line}"),
                        Err(error) => tracing::error!("{error}"),
                    }
                });
                out.line("ground: applied at the next boundary");
                Ok(())
            },
        )
        .command(
            "floor",
            "the ground plane: on | off (bare flips)",
            |args, submit, out| {
                let next = match args.first().copied() {
                    None => None,
                    Some("on") => Some(true),
                    Some("off") => Some(false),
                    Some(other) => {
                        out.line(format!("floor: unknown arg `{other}` (try on|off)"));
                        return Ok(());
                    }
                };
                submit.app_fn("floor", move |d: &mut Dispatch<'_, HeroStores>| {
                    let visible = d.app.environment.get().floor_visible;
                    d.app.environment.get_mut().floor_visible = next.unwrap_or(!visible);
                });
                out.line("floor: applied at the next boundary");
                Ok(())
            },
        )
}

fn letter_height(
    session: &mut Session<HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    index: usize,
) -> Option<f32> {
    let entity = session
        .app
        .letters
        .iter()
        .find(|(_, letter)| letter.index == index)
        .map(|(entity, _)| entity)?;
    let pose = session.domains_mut().typed(r4).ok()?.poses.get(entity)?;
    Some(pose.0.translation.y)
}

fn headless(
    session: &mut Session<HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    steps: u32,
) -> Result<(), HostError> {
    host::run_headless(session, &HostConfig::new("hero", bindings()), steps, &[])?;
    let y = letter_height(session, r4, 0).ok_or_else(|| refused("the first letter has no pose"))?;
    let frame = session.app.stage.get().frame();
    println!("hero letter 0 after {steps} ticks (frame {frame}): y = {y:.4}");
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let steps = args
        .iter()
        .position(|arg| arg == "--headless")
        .map(|at| args.get(at + 1).and_then(|steps| steps.parse::<u32>().ok()));
    let outcome = match steps {
        Some(None) => {
            eprintln!("hero: --headless needs a tick count");
            return ExitCode::FAILURE;
        }
        Some(Some(steps)) => {
            build().and_then(|(mut session, scene)| headless(&mut session, scene.r4, steps))
        }
        None => build().and_then(|(session, scene)| {
            launch(session, host(scene, record_request(&Args::current())))
        }),
    };
    match outcome {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("hero: {error:?}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use loam_math::Rotor4;
    use loam_physics::manifold::PENETRATION_SLOP;
    use loam_shape::Visualizable;

    use super::*;

    const SETTLE_TICKS: u32 = (ASSEMBLE_FRAMES + 60) * SUBSTEPS;

    const REST_WINDOW_FRAMES: u32 = 60;
    const REST_SPREAD: f32 = 0.02;
    const TUNNEL_DEPTH: f32 = 0.075;

    fn solids() -> Vec<GlyphSolid> {
        let font = FontRef::try_from_slice(FONT).expect("the vendored font parses");
        layout_word(&font, WORD, &GlyphParams::default()).expect("the word lays out")
    }

    fn morph_field() -> MorphField {
        let params = GlyphParams::default();
        MorphField::new(&solids(), params.em_size / params.resolution as f32)
            .expect("a morph field")
    }

    fn seeded(seed: u64) -> (Session<HeroStores>, Scene) {
        let (mut session, scene) = build().expect("hero builds");
        session.app.stage.set(Stage::seeded(seed));
        (session, scene)
    }

    fn frames(session: &mut Session<HeroStores>, count: u32) {
        for _ in 0..count {
            session.boundary(Input::default()).expect("boundary");
            for _ in 0..SUBSTEPS {
                session.tick().expect("tick");
            }
        }
    }

    fn rows(session: &Session<HeroStores>) -> Vec<Letter> {
        let mut letters: Vec<Letter> = session
            .app
            .letters
            .iter()
            .map(|(_, letter)| letter.clone())
            .collect();
        letters.sort_by_key(|letter| letter.index);
        letters
    }

    fn poses(session: &mut Session<HeroStores>, r4: DomainHandle<EuclideanR4>) -> Vec<Vec4> {
        let mut found: Vec<(usize, Entity)> = session
            .app
            .letters
            .iter()
            .map(|(entity, letter)| (letter.index, entity))
            .collect();
        found.sort_by_key(|(index, _)| *index);
        let domain = session.domains_mut().typed(r4).expect("the r4 domain");
        found
            .iter()
            .filter_map(|(_, entity)| domain.poses.get(*entity).map(|pose| pose.0.translation))
            .collect()
    }

    fn letter_rotors(
        session: &mut Session<HeroStores>,
        r4: DomainHandle<EuclideanR4>,
    ) -> Vec<Rotor4> {
        let mut found: Vec<(usize, Entity)> = session
            .app
            .letters
            .iter()
            .map(|(entity, letter)| (letter.index, entity))
            .collect();
        found.sort_by_key(|(index, _)| *index);
        let domain = session.domains_mut().typed(r4).expect("the r4 domain");
        found
            .iter()
            .filter_map(|(_, entity)| domain.poses.get(*entity).map(|pose| pose.0.rotation))
            .collect()
    }

    fn drop_poses(session: &mut Session<HeroStores>, r4: DomainHandle<EuclideanR4>) -> Vec<Vec4> {
        let entities: Vec<Entity> = session.app.drops.iter().map(|(entity, _)| entity).collect();
        let domain = session.domains_mut().typed(r4).expect("the r4 domain");
        entities
            .iter()
            .filter_map(|entity| domain.poses.get(*entity).map(|pose| pose.0.translation))
            .collect()
    }

    fn bounds_of(mesh: &TriangleMesh<3>) -> Option<(Vec3, Vec3)> {
        mesh.vertices
            .iter()
            .map(|v| (Vec3::from_array(*v), Vec3::from_array(*v)))
            .reduce(|(lo, hi), (l, h)| (lo.min(l), hi.max(h)))
    }

    fn section_bounds(morph: &mut MorphField, u: f32) -> (Vec3, Vec3) {
        let mut mesh = TriangleMesh::<3>::default();
        assert!(
            append_field_prism(
                morph.blend_at(u),
                0.5 * GlyphParams::default().depth,
                LETTER_COLOR,
                &mut mesh,
            ),
            "the blend at {u} cut an empty section"
        );
        bounds_of(&mesh).expect("a non-empty section has bounds")
    }

    fn blend_of(letter: &Letter, at: Vec4) -> f32 {
        letter.index as f32 - (at.w - W_SLICE) / W_PER_LETTERFORM
    }

    fn composed(session: &mut Session<HeroStores>, scene: &mut Scene) -> TriangleMesh<3> {
        let (mut local, mut scratch) = (Vec::new(), SectionScratch::default());
        let half_depth = scene.half_depth;
        let (r4, morph) = (scene.r4, &mut scene.morph);
        session.dispatch(|d| compose(d, r4, morph, &mut local, &mut scratch, half_depth));
        session.app.mesh.get().clone()
    }

    fn deepest_dynamic_point(
        session: &mut Session<HeroStores>,
        r4: DomainHandle<EuclideanR4>,
    ) -> f32 {
        let world = physics_of(session.domains_mut(), r4)
            .expect("physics")
            .world();
        let mut deepest = f32::INFINITY;
        for body in world.bodies.iter() {
            let Some(Shape::ConvexPolytope4D { vertices }) = world.collider(body) else {
                continue;
            };
            for v in vertices {
                deepest = deepest.min(body.orientation.rotation.apply(*v).y + body.position.y);
            }
        }
        deepest
    }

    const ONE_SUBSTEP_FALL: f32 = -GRAVITY / ((TICK_HZ * SUBSTEPS) * (TICK_HZ * SUBSTEPS)) as f32;

    fn run(steps: u32) -> (Session<HeroStores>, Scene) {
        let (mut session, scene) = build().expect("hero builds");
        host::run_headless(
            &mut session,
            &HostConfig::new("hero", bindings()),
            steps,
            &[],
        )
        .expect("headless");
        (session, scene)
    }

    #[test]
    fn a_released_letter_falls_its_clearance_onto_the_floor() {
        let (mut session, scene) = run(SETTLE_TICKS);
        let r4 = scene.r4;
        let mark = session
            .app
            .letters
            .iter()
            .find(|(_, letter)| letter.index == 0)
            .map(|(_, letter)| letter.mark.y)
            .expect("the first letter");
        let y = letter_height(&mut session, r4, 0).expect("a pose");
        assert!(
            (y - (mark - RELEASE_CLEARANCE)).abs() < 2.0 * PENETRATION_SLOP,
            "the letter rests at {y}, not at its mark {mark} less the {RELEASE_CLEARANCE} clearance"
        );
    }

    #[test]
    fn the_letters_are_directed_until_the_solver_owns_them() {
        let (mut session, scene) = run(ASSEMBLE_FRAMES * SUBSTEPS);
        let r4 = scene.r4;
        let bodies = |session: &mut Session<HeroStores>| {
            let physics = physics_of(session.domains_mut(), r4).expect("physics");
            physics.world().bodies.len()
        };
        assert_eq!(bodies(&mut session), 1, "a letter became a body early");
        let directed = letter_height(&mut session, r4, 0).expect("a pose");

        host::run_headless(&mut session, &HostConfig::new("hero", bindings()), 1, &[])
            .expect("headless");
        assert_eq!(
            bodies(&mut session),
            session.app.letters.len() + 1,
            "the release left a letter without a body"
        );
        let handed = letter_height(&mut session, r4, 0).expect("a pose");
        assert!(
            (handed - directed).abs() < 2.0 * ONE_SUBSTEP_FALL,
            "the release put the letter at {handed}, not where the director left it at {directed}"
        );
    }

    #[test]
    fn record_takes_a_directory_or_the_default_and_is_off_otherwise() {
        assert!(record_request(&Args::from_argv(["--fps=60"])).is_none());
        let bare = record_request(&Args::from_argv(["--record"])).expect("a bare flag records");
        assert!(matches!(
            bare,
            CaptureRequest::StartSequence { dir: None, .. }
        ));
        let named = record_request(&Args::from_argv(["--record=out/hero"])).expect("a named dir");
        let CaptureRequest::StartSequence { dir, .. } = named else {
            panic!("--record started something other than a sequence");
        };
        assert_eq!(dir, Some(PathBuf::from("out/hero")));
    }

    #[test]
    fn morph_wraps_across_both_ends_of_the_word() {
        let mut morph = morph_field();
        let cycle = morph.letters.len() as f32;
        let point = Vec2::new(0.13, 0.27);
        for position in [-2.25_f32, -0.25, 0.0, 0.25, 3.5] {
            let expected = morph.blend_at(position).sample(point);
            let wrapped = morph.blend_at(position + 2.0 * cycle).sample(point);
            assert!((expected - wrapped).abs() < 1e-6);
        }
    }

    #[test]
    fn at_its_mark_every_letter_cuts_its_own_letterform() {
        let (mut session, scene) = seeded(DEFAULT_SEED);
        let r4 = scene.r4;
        frames(&mut session, ASSEMBLE_FRAMES);
        let letters = rows(&session);
        let at = poses(&mut session, r4);
        let mut morph = morph_field();
        let baked = solids();
        let cell = GlyphParams::default().em_size / GlyphParams::default().resolution as f32;
        for letter in &letters {
            let (lo, hi) = section_bounds(&mut morph, blend_of(letter, at[letter.index]));
            let solid = baked
                .iter()
                .filter(|solid| !solid.is_blank())
                .nth(letter.index)
                .expect("a solid per letter");
            let own = bounds_of(&Visualizable::<3>::to_triangles(solid).expect("glyph mesh"))
                .expect("baked mesh");
            let (want, got) = (own.1 - own.0, hi - lo);
            assert!(
                (want.x - got.x).abs() < 2.0 * cell && (want.y - got.y).abs() < 2.0 * cell,
                "letter {} settled on a section {got:?} against its own {want:?}",
                letter.index
            );
        }
    }

    #[test]
    fn a_letter_mid_approach_is_a_different_letterform_and_not_a_scaled_copy() {
        let mut morph = morph_field();
        let (mut settled, scene) = seeded(DEFAULT_SEED);
        let r4 = scene.r4;
        frames(&mut settled, ASSEMBLE_FRAMES);
        let own = section_bounds(
            &mut morph,
            blend_of(&rows(&settled)[0], poses(&mut settled, r4)[0]),
        );

        let (mut approaching, scene) = seeded(DEFAULT_SEED);
        let r4 = scene.r4;
        frames(&mut approaching, LETTER_SLIDE_FRAMES / 2);
        let u = blend_of(&rows(&approaching)[0], poses(&mut approaching, r4)[0]);
        let mid = section_bounds(&mut morph, u);
        let nearest = section_bounds(&mut morph, u.floor());
        assert_ne!(
            mid,
            nearest,
            "the section at {u} equals the letterform at {}, so the morph snaps",
            u.floor()
        );

        let ratio = |(lo, hi): (Vec3, Vec3)| (hi.x - lo.x) / (hi.y - lo.y);
        assert!(
            (ratio(mid) - ratio(own)).abs() > 0.05,
            "mid-approach aspect {} matches its own {}, so the section is only scaling",
            ratio(mid),
            ratio(own)
        );
        assert!(mid.1.y - mid.0.y > 0.1, "the approach drew almost nothing");
    }

    #[test]
    fn the_assembly_slides_every_letter_onto_its_mark_before_the_release() {
        let (mut session, scene) = seeded(DEFAULT_SEED);
        let r4 = scene.r4;
        let letters = rows(&session);
        let entries = poses(&mut session, r4);
        for letter in &letters {
            assert_eq!(entries[letter.index], letter.entry);
            let offset = letter.entry - letter.mark;
            assert!(
                offset.w.abs() > W_PER_LETTERFORM,
                "letter {} starts less than a letterform away, so it never morphs",
                letter.index
            );
            assert!(
                offset.truncate().length() < 1e-6,
                "letter {} enters by a 3D translation of {:?}",
                letter.index,
                offset.truncate()
            );
        }
        frames(&mut session, LETTER_STAGGER_FRAMES);
        assert_ne!(poses(&mut session, r4), entries);

        frames(&mut session, ASSEMBLE_FRAMES - LETTER_STAGGER_FRAMES);
        let marks: Vec<Vec4> = letters.iter().map(|letter| letter.mark).collect();
        assert_eq!(poses(&mut session, r4), marks, "a letter missed its mark");
    }

    #[test]
    fn the_assembled_word_is_centred_on_what_the_camera_aims_at() {
        let (session, _) = seeded(DEFAULT_SEED);
        let letters = rows(&session);
        let centre = word_centre(&letters);
        assert!(
            centre.x.abs() < 1e-5,
            "the word centres at x {}, so it hangs off one side of frame",
            centre.x
        );
        let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
        for letter in &letters {
            for v in &letter.hull {
                lo = lo.min(letter.mark.x + v.x);
                hi = hi.max(letter.mark.x + v.x);
            }
        }
        assert!(
            (lo + hi).abs() < 1e-5,
            "the word spans [{lo}, {hi}] em, which is not symmetric about the aim"
        );
        assert!(
            centre.y > 0.0 && centre.y < 1.0,
            "the word centres at y {}, outside a one-em letter's own height",
            centre.y
        );
    }

    #[test]
    fn a_settled_letter_is_drawn_standing_on_the_floor_within_the_covers_margin() {
        let (mut session, mut scene) = seeded(DEFAULT_SEED);
        frames(&mut session, RAIN_START_FRAME);
        assert!(session.app.drops.is_empty(), "the rain started early");
        let margin = solids()
            .iter()
            .map(GlyphSolid::collider_margin)
            .fold(0.0f32, f32::max);

        let mesh = composed(&mut session, &mut scene);
        assert!(!mesh.vertices.is_empty(), "the settled word drew nothing");
        let drawn = mesh.vertices.iter().fold(f32::INFINITY, |m, v| m.min(v[1]));
        assert!(
            drawn > -2.0 * PENETRATION_SLOP,
            "the word is drawn {drawn} below the floor"
        );
        assert!(
            drawn < margin + 2.0 * PENETRATION_SLOP,
            "the word floats {drawn} above the floor, past the {margin} margin"
        );
    }

    #[test]
    fn the_seed_moves_the_rain_and_leaves_the_letters_landing_untouched() {
        let settled = |seed: u64| {
            let (mut session, scene) = seeded(seed);
            let r4 = scene.r4;
            frames(&mut session, RAIN_START_FRAME);
            let letters = poses(&mut session, r4);
            frames(&mut session, REST_WINDOW_FRAMES);
            (letters, drop_poses(&mut session, r4))
        };
        let (letters_a, drops_a) = settled(DEFAULT_SEED);
        let (letters_b, drops_b) = settled(DEFAULT_SEED ^ 0xdead_beef);
        assert_eq!(
            letters_a, letters_b,
            "the seed reached the letters' landing"
        );
        assert!(
            drops_a.len() != drops_b.len() || drops_a.iter().zip(&drops_b).any(|(a, b)| a != b),
            "two seeds rained identically, so the seed is decoration"
        );
    }

    #[test]
    fn release_rain_and_freeze_preserve_the_sequence_boundaries() {
        let (mut session, mut scene) = seeded(DEFAULT_SEED);
        let r4 = scene.r4;
        let letters = rows(&session);
        let mut rest_start: Vec<Vec4> = Vec::new();
        let mut frozen: Vec<(Vec4, Rotor4)> = Vec::new();
        let mut falling_seen = false;
        let mut slice_sweep = 0.0_f32;
        for frame in 1..=SEQUENCE_FRAMES {
            frames(&mut session, 1);
            let deepest = deepest_dynamic_point(&mut session, r4);
            assert!(
                deepest > -TUNNEL_DEPTH,
                "floor depth {deepest} at frame {frame}"
            );

            let at = poses(&mut session, r4);
            if frame == RAIN_START_FRAME - REST_WINDOW_FRAMES {
                rest_start = at.clone();
            }
            for letter in &letters {
                let pose = at[letter.index];
                if (ASSEMBLE_FRAMES..=RAIN_START_FRAME).contains(&frame) {
                    assert!(
                        (pose.w - letter.mark.w).abs() < 1e-6,
                        "the floor moved letter {} through w",
                        letter.index
                    );
                }
                if (RAIN_START_FRAME - REST_WINDOW_FRAMES..=RAIN_START_FRAME).contains(&frame) {
                    assert!(
                        pose.distance(rest_start[letter.index]) < REST_SPREAD,
                        "letter {} has not settled",
                        letter.index
                    );
                }
            }

            for (index, rotor) in letter_rotors(&mut session, r4).iter().enumerate() {
                let determinant = rotor.apply(Vec4::X).truncate().dot(
                    rotor
                        .apply(Vec4::Y)
                        .truncate()
                        .cross(rotor.apply(Vec4::Z).truncate()),
                );
                assert!(
                    (determinant - 1.0).abs() < 1e-3,
                    "letter {index} left the draw's rotation plane at frame {frame}"
                );
            }
            let world = physics_of(session.domains_mut(), r4)
                .expect("physics")
                .world();
            for (key, manifold) in &world.manifolds {
                if manifold.points.is_empty() {
                    continue;
                }
                let falling = |id: BodyId| world.bodies[id].collision_group == GROUP_FALLING;
                assert!(
                    !(falling(key.0) && falling(key.1)),
                    "airborne drops collided"
                );
            }
            falling_seen |= (0..world.bodies.len())
                .map(|dense| world.bodies.id_at(dense))
                .any(|id| world.bodies[id].collision_group == GROUP_FALLING);

            if frame == RAIN_START_FRAME {
                assert!(session.app.drops.is_empty(), "the rain started early");
            }
            if frame == FREEZE_FRAME {
                let drops = session.app.drops.len();
                let world = physics_of(session.domains_mut(), r4)
                    .expect("physics")
                    .world();
                let landed = (0..world.bodies.len())
                    .map(|dense| world.bodies.id_at(dense))
                    .filter(|id| world.bodies[*id].collision_group == GROUP_LANDED)
                    .count();
                assert!(
                    falling_seen && landed * 2 > drops,
                    "{landed} of {drops} drops had landed by the freeze"
                );
                frozen = (0..world.bodies.len())
                    .map(|dense| world.bodies.id_at(dense))
                    .map(|id| {
                        (
                            world.bodies[id].position,
                            world.bodies[id].orientation.rotation,
                        )
                    })
                    .collect();
            }
            if frame > FREEZE_FRAME {
                let stage = *session.app.stage.get();
                let world = physics_of(session.domains_mut(), r4)
                    .expect("physics")
                    .world();
                let now: Vec<(Vec4, Rotor4)> = (0..world.bodies.len())
                    .map(|dense| world.bodies.id_at(dense))
                    .map(|id| {
                        (
                            world.bodies[id].position,
                            world.bodies[id].orientation.rotation,
                        )
                    })
                    .collect();
                assert_eq!(
                    now, frozen,
                    "the pile moved after the freeze at frame {frame}"
                );
                slice_sweep = slice_sweep.max((stage.slice() - W_SLICE).abs());
            }
        }
        assert!(
            slice_sweep > 0.9 * SLICE_SWEEP_RANGE,
            "the slice swept {slice_sweep}, not {SLICE_SWEEP_RANGE}"
        );

        let mesh = composed(&mut session, &mut scene);
        assert_eq!(mesh.colors.len(), mesh.vertices.len());
        let count = mesh.vertices.len() as u32;
        assert!(count > 4, "the mesh holds almost nothing");
        for tri in &mesh.indices {
            for index in tri {
                assert!(*index < count, "index {index} past {count} vertices");
            }
        }
        for v in &mesh.vertices {
            assert!(v.iter().all(|c| c.is_finite()), "non-finite vertex {v:?}");
        }
    }
}
