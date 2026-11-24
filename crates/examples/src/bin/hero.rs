use std::fmt::Write as _;
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::ExitCode;

use loam::app::capture::{CaptureFormat, CaptureRequest, CaptureStage, PaletteMode};
use loam::app::environment::Environment;
use loam::app::script::{driver_from_args, ScriptDriver};
use loam::app::session::{launch_or_headless, run_script};
use loam::physics::body::MASK_ALL;
use loam::physics::euclidean_r4::{polytope_body_r4, register_default_narrowphase};
use loam::physics::{BodyDef, BodyId};
use loam::prelude::*;
use loam::render::TriangleFeed;
use loam::runtime::host;
use loam::shape::polytope::{polytope_section_faces_append, SectionScratch};
use loam::text::ab_glyph::FontRef;
use loam::text::glyph::{
    append_field_prism, layout_word, DistanceField2D, GlyphParams, GlyphSolid,
};
use loam::time::director::{Ease, Track};

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

const PAUSE: ActionId = ActionId("pause");
const RESEED: ActionId = ActionId("reseed");

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

    // xorshift64*, Vigna 2016 §4.
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

stores! {
    #[derive(Default)]
    pub struct HeroStores {
        letters: Store<Letter>,
        drops: Store<Drop>,
        stage: Value<Stage>,
        held: Value<Vec<Held>>,
        mesh: Value<TriangleMesh<3>>,
    }
    publishes Shown;
}

#[derive(Default)]
pub struct Shown {
    mesh: TriangleMesh<3>,
    version: Option<Version>,
    frame: u32,
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

fn hold(
    d: &mut Dispatch<'_, HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    asleep: bool,
) -> Result<(), Rejection> {
    let physics = d.domains.physics(r4)?;
    if !asleep {
        for held in d.app.held.get() {
            if let Some(entity) = physics.entity(held.body) {
                physics.set_velocity(entity, held.velocity, held.spin)?;
            }
        }
        d.app.held.get_mut().clear();
        return Ok(());
    }
    let mut held = Vec::new();
    for dense in 0..physics.world().bodies().len() {
        let id = physics.world().bodies().id_at(dense);
        let Some(body) = physics.world().body(id) else {
            continue;
        };
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
        if let Some(entity) = physics.entity(row.body) {
            physics.sleep_body(entity)?;
        }
    }
    d.app.held.set(held);
    Ok(())
}

fn spawn_drop(
    d: &mut Dispatch<'_, HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    polytope: Polytope4,
    position: Vec4,
    tumble: Bivector4,
) -> Result<Entity, Rejection> {
    let body = BodyDef::regular_polytope(polytope, RAIN_SIZE, RAIN_MASS)?
        .restitution(RESTITUTION)?
        .velocity(
            Vec4::new(0.0, -RAIN_ENTRY_SPEED, 0.0, 0.0),
            tumble,
            &EuclideanR4,
        )?
        .collision_filter(GROUP_FALLING, MASK_FALLING);
    d.spawn(
        SpawnBundle::new()
            .at(r4, Pose::at(position))
            .body(body)
            .row(Drop {
                polytope,
                color: drop_color(polytope),
            }),
    )
}

fn compose(
    app: &mut HeroStores,
    domains: &mut Domains,
    r4: DomainHandle<EuclideanR4>,
    morph: &mut MorphField,
    local: &mut Vec<Vec4>,
    scratch: &mut SectionScratch,
    half_depth: f32,
) {
    let stage = *app.stage.get();
    let Ok(domain) = domains.read(r4) else {
        return;
    };
    let slice = stage.slice();
    let mesh = app.mesh.get_mut();
    mesh.vertices.clear();
    mesh.colors.clear();
    mesh.indices.clear();
    for (entity, letter) in app.letters.iter() {
        let Some(pose) = domain.poses().get(entity) else {
            continue;
        };
        let base = mesh.vertices.len();
        let u = letter.index as f32 - (pose.point.w - slice) / W_PER_LETTERFORM;
        if !append_field_prism(morph.blend_at(u), half_depth, LETTER_COLOR, mesh) {
            continue;
        }
        let translate = pose.point.truncate();
        for v in &mut mesh.vertices[base..] {
            let posed = pose.frame.apply(Vec4::new(v[0], v[1], v[2], 0.0));
            *v = (posed.truncate() + translate).to_array();
        }
    }
    for (entity, drop) in app.drops.iter() {
        let Some(pose) = domain.poses().get(entity) else {
            continue;
        };
        let topology = drop.polytope.topology();
        local.clear();
        local.extend(
            topology
                .vertices
                .iter()
                .map(|v| RAIN_SIZE * pose.frame.apply(*v) + Vec4::W * pose.point.w),
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
        let translate = pose.point.truncate();
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
    let mut morph = MorphField::new(&solids, params.em_size / params.resolution as f32)
        .ok_or_else(|| refused("the word laid out with no ink to morph"))?;
    let centre = word_centre(&letters);
    let half_depth = 0.5 * params.depth;

    let mut session = Session::new(
        HeroStores::default(),
        SimConfig {
            fixed_hz: TICK_HZ * SUBSTEPS,
            max_ticks_per_frame: 2 * SUBSTEPS,
        },
    );
    let physics = PhysicsConfig::new(register_default_narrowphase).gravity(Vec4::Y * GRAVITY);
    let r4 = session.register_domain(DomainBuilder::new("r4", EuclideanR4).physics(physics)?);
    session.dispatch(|d| -> Result<(), Rejection> {
        for letter in letters {
            let entry = letter.entry;
            d.spawn(SpawnBundle::new().at(r4, Pose::at(entry)).row(letter))?;
        }
        let floor = BodyDef::halfspace(Vec4::Y, FLOOR_Y)?.restitution(RESTITUTION)?;
        d.spawn(SpawnBundle::new().at(r4, Pose::at(Vec4::ZERO)).body(floor))?;
        d.domains.physics(r4)?.set_solver_iterations(PILE_PGS_ITERS);
        Ok(())
    })?;

    session.set_rig(Rig::Orbit(Orbit {
        pitch: BOOT_ORBIT_PITCH,
        ..Orbit::around(centre.to_array(), BOOT_ORBIT_DISTANCE)
    }));

    session.on_action(PAUSE, move |d| {
        let paused = !d.app.stage.paused;
        hold(d, r4, paused)?;
        d.app.stage.get_mut().paused = paused;
        Ok(())
    });
    // The seed is read at the press, because the reset lands before the reseed.
    session.system(Phase::Dispatch, "keys", |ctx| {
        if ctx.input.pressed(RESEED) {
            let seed = ctx.app.stage.seed.wrapping_add(1);
            ctx.commands.submit(Command::Reset);
            ctx.commands.try_app_fn("reseed", move |d| {
                d.app.stage.set(Stage::seeded(seed));
                Ok(Outcome::Done)
            });
        }
        Ok(())
    });

    session.system_at(
        Phase::Simulation,
        Order::Before(DOMAIN_STEP),
        "assemble",
        move |ctx| {
            let stage = *ctx.app.stage;
            if stage.paused {
                return Ok(());
            }
            let domain = ctx.domains.typed(r4)?;
            if !stage.released {
                for (entity, letter) in ctx.app.letters.iter() {
                    let Some(at) = letter.slide.sample(stage.frame(), fps()) else {
                        continue;
                    };
                    domain.set_point(entity, at)?;
                }
                if stage.frame() < ASSEMBLE_FRAMES {
                    return Ok(());
                }
                for (entity, letter) in ctx.app.letters.iter() {
                    let body =
                        polytope_body_r4(letter.mark, Vec4::ZERO, letter.hull.clone(), LETTER_MASS)
                            .ok_or(DomainError::Unsupported("letter collider"))?
                            .restitution(RESTITUTION)?;
                    domain.spawn_body(entity, body)?;
                }
                ctx.app.stage.get_mut().released = true;
                return Ok(());
            }
            let physics = ctx.domains.physics(r4)?;
            for (entity, letter) in ctx.app.letters.iter_mut() {
                letter.w_before = physics
                    .body(entity)
                    .and_then(|id| physics.world().body(id))
                    .map_or(0.0, |body| body.velocity.w);
            }
            Ok(())
        },
    );

    session.system_at(
        Phase::Simulation,
        Order::After(DOMAIN_STEP),
        "settle",
        move |ctx| {
            let mut stage = *ctx.app.stage;
            if stage.paused || stage.frame() >= SEQUENCE_FRAMES {
                return Ok(());
            }
            {
                let physics = ctx.domains.physics(r4)?;
                for (entity, letter) in ctx.app.letters.iter() {
                    let Some(id) = physics.body(entity) else {
                        continue;
                    };
                    let scenery_only = physics
                        .world()
                        .manifolds()
                        .filter(|(key, manifold)| {
                            !manifold.points.is_empty() && (key.0 == id || key.1 == id)
                        })
                        .all(|(key, _)| {
                            let other = if key.0 == id { key.1 } else { key.0 };
                            physics
                                .world()
                                .body(other)
                                .is_some_and(|body| body.inv_mass() == 0.0)
                        });
                    let Some(body) = physics.world().body(id).copied() else {
                        continue;
                    };
                    let mut angular = body.angular_velocity;
                    angular.xw = 0.0;
                    angular.yw = 0.0;
                    angular.zw = 0.0;
                    let mut velocity = body.velocity;
                    if scenery_only && velocity.w.abs() > letter.w_before.abs() {
                        velocity.w = letter.w_before;
                    }
                    if velocity != body.velocity || angular != body.angular_velocity {
                        physics.set_velocity(entity, velocity, angular)?;
                    }
                }
                for (entity, _) in ctx.app.drops.iter() {
                    let Some(id) = physics.body(entity) else {
                        continue;
                    };
                    if physics
                        .world()
                        .body(id)
                        .is_none_or(|body| body.collision_group != GROUP_FALLING)
                    {
                        continue;
                    }
                    let touched = physics.world().manifolds().any(|(key, manifold)| {
                        (key.0 == id || key.1 == id) && !manifold.points.is_empty()
                    });
                    if touched {
                        physics.set_collision_filter(entity, GROUP_LANDED, MASK_ALL)?;
                    }
                }
            }

            stage.tick += 1;
            if stage.frame() >= FREEZE_FRAME {
                if !stage.frozen {
                    stage.frozen = true;
                    ctx.commands
                        .try_app_fn(
                            "freeze",
                            move |d: &mut Dispatch<'_, HeroStores>| match hold(d, r4, true) {
                                Ok(()) => Ok(Outcome::Done),
                                Err(error) => {
                                    d.app.stage.get_mut().frozen = false;
                                    Err(error)
                                }
                            },
                        );
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
                    .try_app_fn("rain", move |d: &mut Dispatch<'_, HeroStores>| {
                        spawn_drop(d, r4, polytope, position, tumble).map(Outcome::Spawned)
                    });
            }
            ctx.app.stage.set(stage);
            Ok(())
        },
    );

    let mut local = Vec::new();
    let mut scratch = SectionScratch::default();
    session.system_on_change(
        Phase::Publication,
        "compose",
        |app: &HeroStores| (app.stage.tick, app.mesh.version()),
        move |ctx| {
            compose(
                ctx.app,
                ctx.domains,
                r4,
                &mut morph,
                &mut local,
                &mut scratch,
                half_depth,
            );
            Ok(())
        },
    );
    session.publish_value("shown", |ctx, shown| {
        let version = ctx.app.mesh.version();
        if shown.version != Some(version) {
            let mesh = ctx.app.mesh.get();
            shown.mesh.vertices.clone_from(&mesh.vertices);
            shown.mesh.colors.clone_from(&mesh.colors);
            shown.mesh.indices.clone_from(&mesh.indices);
            shown.version = Some(version);
        }
        shown.frame = ctx.app.stage.frame();
        Ok(())
    });
    session.views_mut().root_mut().backdrop = Some(Environment::default().backdrop(FLOOR_Y));

    // Taken here so a reset straight after build has a state to return to.
    session.set_initial()?;
    Ok((session, Scene { r4 }))
}

fn host(args: Args, record: Option<CaptureRequest>) -> SessionApp<HeroStores> {
    let feed = TriangleFeed::default();
    let mut start = record;
    let mut recording = start.is_some();
    SessionApp::with_args(HostConfig::new("loam", bindings()), args)
        .recover_on_fault(RESEED)
        .pass(feed.pass(FragmentShading::FaceNormalLambert))
        .target_fps(TICK_HZ as f32)
        .on_published(move |hook: &mut PublishedHook<'_, HeroStores>| {
            let shown = &hook.published.value;
            feed.set_view(&hook.eye, Rigid::IDENTITY);
            if let Some(version) = shown.version {
                feed.set_mesh(version, &shown.mesh);
            }
            if let Some(request) = start.take() {
                if let Err(error) = hook.capture.start(request) {
                    tracing::error!("hero capture start refused: {error}");
                    recording = false;
                }
            }
            if recording && shown.frame >= SEQUENCE_FRAMES {
                if let Err(error) = hook.capture.stop() {
                    tracing::error!("hero capture stop refused: {error}");
                }
                recording = false;
            }
        })
        .command(
            "ground",
            "checker colors and fog: dark|light <r> <g> <b>, fog <density>, reset",
            |args, submit, out| {
                Environment::default().apply(args)?;
                let owned: Vec<String> = args.iter().map(|arg| arg.to_string()).collect();
                submit.app_fn("ground", move |d: &mut Dispatch<'_, HeroStores>| {
                    let refs: Vec<&str> = owned.iter().map(String::as_str).collect();
                    let root = d.views.root_mut();
                    let mut environment = root
                        .backdrop
                        .map_or_else(Environment::default, |backdrop| Environment::of(&backdrop));
                    match environment.apply(&refs) {
                        Ok(line) => {
                            root.backdrop = Some(environment.backdrop(FLOOR_Y));
                            tracing::info!("{line}");
                        }
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
                    if let Some(backdrop) = d.views.root_mut().backdrop.as_mut() {
                        let ground = &mut backdrop.ground;
                        ground.visible = next.unwrap_or(!ground.visible);
                    }
                });
                out.line("floor: applied at the next boundary");
                Ok(())
            },
        )
}

fn letter_height(
    session: &Session<HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    index: usize,
) -> Option<f32> {
    let entity = session
        .app
        .letters
        .iter()
        .find(|(_, letter)| letter.index == index)
        .map(|(entity, _)| entity)?;
    let pose = session.domains().read(r4).ok()?.poses().get(entity)?;
    Some(pose.point.y)
}

fn headless(
    session: &mut Session<HeroStores>,
    r4: DomainHandle<EuclideanR4>,
    steps: u32,
    records: bool,
) -> Result<(), HostError> {
    let config = HostConfig::new("hero", bindings());
    if records {
        let mut dump = String::new();
        host::write_headless(
            &mut dump,
            session,
            &config,
            steps,
            1.0,
            RECORD_STRIDE,
            |_, session, out| match out {
                Some(out) => write_mesh(out, session.app.mesh.get())
                    .map_err(|_| refused("the record dump failed")),
                None => Ok(()),
            },
        )?;
        print!("{dump}");
        return Ok(());
    }
    host::run_headless(session, &config, steps, &[], 1.0, |_, _, _| Ok(()))?;
    let y = letter_height(session, r4, 0).ok_or_else(|| refused("the first letter has no pose"))?;
    let frame = session.app.stage.get().frame();
    println!("hero letter 0 after {steps} ticks (frame {frame}): y = {y:.4}");
    Ok(())
}

fn scripted(
    session: &mut Session<HeroStores>,
    args: Args,
    driver: ScriptDriver,
) -> Result<(), HostError> {
    let mut app = host(args, None);
    let mut dump = String::new();
    run_script(
        session,
        &mut app,
        driver,
        1.0,
        |step, session, publication| {
            if host::write_stride(&mut dump, step, RECORD_STRIDE, session, publication)? {
                write_mesh(&mut dump, session.app.mesh.get())
                    .map_err(|_| refused("the record dump failed"))?;
            }
            Ok(())
        },
    )?;
    print!("{dump}");
    Ok(())
}

// A record dump keeps every 20th step so it stays short enough to diff.
const RECORD_STRIDE: NonZeroU32 = NonZeroU32::new(20).expect("the record stride is nonzero");

fn write_mesh(out: &mut String, mesh: &TriangleMesh<3>) -> std::fmt::Result {
    for (vertex, color) in mesh.vertices.iter().zip(&mesh.colors) {
        writeln!(out, "vertex {vertex:?} color {color:?}")?;
    }
    for face in &mesh.indices {
        writeln!(out, "face {face:?}")?;
    }
    Ok(())
}

fn main() -> ExitCode {
    let outcome = launch_or_headless(
        |args| {
            let (session, _) = build()?;
            let record = record_request(&args);
            Ok((session, host(args, record)))
        },
        |args| {
            let driver = driver_from_args(&args).map_err(|error| refused(format!("{error:#}")))?;
            if let Some(driver) = driver {
                if !args.has_bare_flag("records") || args.flag_value("headless").is_some() {
                    return Err(refused(
                        "--script runs as --headless --records with no tick count",
                    ));
                }
                let (mut session, _) = build()?;
                return scripted(&mut session, args, driver);
            }
            let steps = args
                .flag_value("headless")
                .and_then(|count| count.parse::<u32>().ok())
                .ok_or_else(|| refused("--headless needs a tick count"))?;
            let (mut session, scene) = build()?;
            headless(&mut session, scene.r4, steps, args.has_bare_flag("records"))
        },
    );
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
    use loam::runtime::Publication;

    use super::*;

    #[test]
    fn reset_at_the_same_tick_rebuilds_the_published_mesh() {
        let (mut session, _) = build().expect("hero builds");
        session.app.stage.set(Stage::seeded(DEFAULT_SEED));
        let mut publication = Publication::default();
        session.publish(&mut publication).expect("publication");
        let before = session.app.mesh.get().clone();

        session.reset().expect("reset");
        session.publish(&mut publication).expect("publication");
        let after = session.app.mesh.get().clone();

        assert_eq!(
            (after.vertices, after.indices, after.colors),
            (before.vertices, before.indices, before.colors)
        );
    }

    fn publish(session: &mut Session<HeroStores>, into: &mut Publication<Shown>) {
        session.publish(into).expect("publication");
        let mesh = session.app.mesh.get();
        assert!(
            into.value.mesh.vertices == mesh.vertices
                && into.value.mesh.indices == mesh.indices
                && into.value.mesh.colors == mesh.colors,
            "the published mesh is not the session's mesh"
        );
    }

    #[test]
    fn the_published_mesh_follows_the_sessions_mesh_through_ticks_and_a_reset() {
        let (mut session, _) = build().expect("hero builds");
        let mut publication = Publication::default();
        publish(&mut session, &mut publication);
        let first = publication.value.mesh.vertices.clone();

        for _ in 0..TICK_HZ {
            session.tick().expect("tick");
        }
        publish(&mut session, &mut publication);
        assert!(
            publication.value.mesh.vertices != first,
            "a second of the scene left the mesh as it was, so the copy was never exercised"
        );

        session.reset().expect("reset");
        publish(&mut session, &mut publication);
    }

    #[test]
    fn a_reset_restores_the_sky_and_ground_the_scene_was_built_with() {
        let (mut session, _) = build().expect("hero builds");
        let built = session.views_mut().root_mut().backdrop;
        assert!(built.is_some(), "hero was built with no backdrop");

        session.views_mut().root_mut().backdrop = None;
        session.reset().expect("reset");

        assert!(
            session.views_mut().root_mut().backdrop == built,
            "the reset lost the backdrop"
        );
    }
}
