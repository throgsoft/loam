use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::Vec4;
use loam_app::args::Args;
use loam_app::session::{launch, FrameHook, SessionApp};
use loam_math::{Bivector, EuclideanR4, Iso4Flat};
use loam_render::pass::{FramePass, PassOrder, PassSchedule};
use loam_render::raymarch::BodyUniform;
use loam_render::{DepthConvention, HyperslicePass, LinePass, PointPass, SkyGroundPass};
use loam_runtime::host::{run_headless, HostConfig, HostError};
use loam_runtime::{
    Access, ActionId, Bindings, Command, Commands, Ctx, DomainBuilder, DomainHandle, Domains,
    Entity, Eye, Input, Instance, Key, LogCapacity, Material, MaterialId, Orbit, Phase,
    PhysicsConfig, Pointer, PointerPhase, Pose, PreparedGeometry, PreparedId, Rejection, Section4,
    SegmentRecord, Session, SimConfig, SpawnBundle, Step, ViewId, ViewSpec,
};

#[cfg(test)]
mod alloc_probe {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;

    thread_local! {
        static CALLS: Cell<usize> = const { Cell::new(0) };
    }

    pub struct Counting;

    // SAFETY: Methods preserve System contracts; const TLS and wrapping Cell updates cannot unwind.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            let _ = CALLS.try_with(|calls| calls.set(calls.get().wrapping_add(1)));
            // SAFETY: The caller supplies a valid nonzero allocation layout.
            unsafe { System.alloc(layout) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: The caller supplies a live System allocation and its original layout.
            unsafe { System.dealloc(ptr, layout) }
        }

        unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
            let _ = CALLS.try_with(|calls| calls.set(calls.get().wrapping_add(1)));
            // SAFETY: The caller supplies a live System allocation, its layout, and a valid new size.
            unsafe { System.realloc(ptr, layout, new_size) }
        }
    }

    pub fn allocations_in(body: impl FnOnce()) -> usize {
        let before = CALLS.with(Cell::get);
        body();
        CALLS.with(Cell::get).wrapping_sub(before)
    }
}

#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: alloc_probe::Counting = alloc_probe::Counting;

mod catalog;
mod color;
mod composer;
mod consts;
mod gimbal;
mod mode;
mod points;
mod projection;
mod scene;
mod section;
mod toy;
mod ui;

use catalog::ShapeEntry;
use color::{ColorMode, Shades};
use composer::{Composer, Term};
use consts::{BODY_SIZE, BODY_X_SPACING, BODY_Y, GRAVITY, W_SCRUB_RATE};
use gimbal::Gimbal;
use mode::{
    ClearComposer, ClearDraft, CommitDraft, DraftPlane, DropTerm, Mode, PushTerm, SetActive,
    SetColorMode, SetMode, SetProjection, SetRunning, SetScrub, SetShape, SetSlice, Spin,
    ToggleGimbal, TogglePlane, TogglePoints, TurnRow,
};
use projection::Family;

const SPIN: ActionId = ActionId(0);
const SLICE_UP: ActionId = ActionId(1);
const SLICE_DOWN: ActionId = ActionId(2);
const NEXT_MODE: ActionId = ActionId(3);
const RESET: ActionId = ActionId(4);
const GIMBAL: ActionId = ActionId(5);
const PLANE: [ActionId; 6] = [
    ActionId(10),
    ActionId(11),
    ActionId(12),
    ActionId(13),
    ActionId(14),
    ActionId(15),
];

const SECTION_COLOR: [f32; 4] = [1.0, 0.85, 0.35, 1.0];
const SECTION_WIDTH_PX: f32 = 2.0;
const EDGE_WIDTH_PX: f32 = 1.4;
const HEADLESS_STEPS: u32 = 8;
const HEADLESS_FORMULA: &str = "90deg (xy + zw)";
const HEADLESS_SCRUB: f32 = 0.7;

#[derive(Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) index: usize,
    pub(crate) entry: ShapeEntry,
    pub(crate) rest: Vec4,
}

#[derive(Clone, Copy)]
pub(crate) struct Wall;

loam_runtime::stores! {
    #[derive(Default)]
    pub struct Playground {
        slots: Store<Slot>,
        walls: Store<Wall>,
        mode: Value<Mode>,
        spin: Value<Spin>,
        composer: Value<Composer>,
        active: Value<usize>,
        slice: Value<f32>,
        projection: Value<Family>,
        pointer: Value<Option<Pointer>>,
        gimbal: Value<bool>,
        points: Value<bool>,
        color: Value<ColorMode>,
        floor: Value<bool>,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Intent {
    Mode(Mode),
    Active(usize),
    Slice(f32),
    Plane(usize),
    Running(bool),
    Projection(Family),
    Term(Term),
    DropTerm(usize),
    Draft(usize),
    CommitDraft,
    ClearDraft,
    ClearTerms,
    Scrub(f32),
    Gimbal,
    Turn(loam_math::Rotor4),
    Color(ColorMode),
    Points,
    Shape(usize, usize),
}

pub(crate) type Intents = Arc<Mutex<Vec<Intent>>>;

pub(crate) fn push(intents: &Intents, intent: Intent) {
    intents
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(intent);
}

pub(crate) struct Boot {
    pub(crate) session: Session<Playground>,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

fn rest_of(index: usize, len: usize) -> Vec4 {
    let centre = (len.max(1) - 1) as f32 * 0.5;
    Vec4::new((index as f32 - centre) * BODY_X_SPACING, BODY_Y, 0.0, 0.0)
}

pub(crate) fn boot(row: &[ShapeEntry], intents: &Intents) -> Result<Boot, HostError> {
    let mut session = Session::new(Playground::default(), SimConfig::default());
    let domain = session.register_domain(
        DomainBuilder::new("r4", EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(
                PhysicsConfig::new(loam_physics::euclidean_r4::register_default_narrowphase)
                    .gravity(Vec4::NEG_Y * GRAVITY),
            ),
    );
    let root = session.views().root();
    let cards = prepare_catalog(&mut session);

    let layers = session.dispatch(|d| -> Result<Layers, Rejection> {
        for (index, entry) in row.iter().enumerate() {
            let rest = rest_of(index, row.len());
            let mut bundle = SpawnBundle::new()
                .at(domain, Pose(Iso4Flat::from_translation(rest)))
                .row(Slot {
                    index,
                    entry: *entry,
                    rest,
                });
            if let Some(card) = card_of(entry) {
                if let Some(geometry) = cards[card].geometry {
                    bundle = bundle.instance(Instance::new(geometry, cards[card].material));
                }
            }
            d.spawn(bundle)?;
        }
        let eye = d.spawn(SpawnBundle::new().at(domain, Pose(Iso4Flat::IDENTITY)))?;
        let r4 = d.domains.typed(domain)?;
        let section = r4.add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }));
        let projection = r4.add_view(ViewSpec::new(root, eye, Family::default().mapping(None, 0)));
        Ok(Layers {
            section,
            projection,
        })
    })?;
    session.views_mut().root_mut().eye =
        Eye::looking_at([0.0, 3.0, 9.0], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
    session.app.floor.set(true);

    install_systems(&mut session, domain, layers, cards, intents);
    session.set_initial()?;
    Ok(Boot { session, domain })
}

#[derive(Clone, Copy)]
struct Layers {
    section: ViewId,
    projection: ViewId,
}

/// One prepared wireframe, material, and palette set per catalogue entry, built once at boot.
#[derive(Clone, Copy)]
struct Card {
    geometry: Option<PreparedId>,
    material: MaterialId,
    shades: Option<Shades>,
}

fn card_of(entry: &ShapeEntry) -> Option<usize> {
    catalog::SHAPE_CATALOG.iter().position(|held| held == entry)
}

fn prepare_catalog(session: &mut Session<Playground>) -> Vec<Card> {
    catalog::SHAPE_CATALOG
        .iter()
        .map(|entry| {
            let [r, g, b] = entry.body_color;
            let material = session.add_material(Material::lines([r, g, b, 0.9], EDGE_WIDTH_PX));
            let polytope = entry.shape.polytope4();
            Card {
                geometry: polytope.map(|polytope| {
                    session.prepare(PreparedGeometry::edges_of(polytope.topology(), BODY_SIZE))
                }),
                material,
                shades: polytope.map(|polytope| {
                    let topology = polytope.topology();
                    Shades {
                        gradient: session.add_palette(color::vertex_gradient_colors(topology)),
                        unique: session.add_palette(color::unique_edge_colors(topology.edges)),
                        extent: color::w_extent(topology, BODY_SIZE),
                    }
                }),
            }
        })
        .collect()
}

fn install_systems(
    session: &mut Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    layers: Layers,
    cards: Vec<Card>,
    intents: &Intents,
) {
    let queued = intents.clone();
    let mut drained: Vec<Intent> = Vec::new();
    let submitted = cards.clone();
    session.system(
        Phase::Dispatch,
        "intents",
        Access::new().commands(),
        move |_input: &Input, commands: &mut Commands<Playground>| {
            {
                let mut held = queued.lock().unwrap_or_else(|error| error.into_inner());
                if held.is_empty() {
                    return;
                }
                std::mem::swap(&mut *held, &mut drained);
            }
            for intent in drained.drain(..) {
                submit(commands, domain, &submitted, intent);
            }
        },
    );

    session.system(
        Phase::Dispatch,
        "controls",
        Access::new().writes::<Option<Pointer>>().commands(),
        move |ctx: Ctx<'_, Playground>| {
            ctx.app.pointer.set(ctx.input.pointers.last().copied());
            if ctx.input.pressed(SPIN) {
                let running = ctx.app.spin.get().running;
                ctx.commands.app(SetRunning { running: !running });
            }
            if ctx.input.pressed(NEXT_MODE) {
                let next = match *ctx.app.mode.get() {
                    Mode::Rotate => Mode::Compose,
                    Mode::Compose => Mode::Toybox,
                    Mode::Toybox => Mode::Rotate,
                };
                ctx.commands.app(SetMode { mode: next, domain });
            }
            if ctx.input.pressed(RESET) {
                ctx.commands.submit(Command::Reset);
            }
            if ctx.input.pressed(GIMBAL) {
                ctx.commands.app(ToggleGimbal);
            }
            for (index, action) in PLANE.into_iter().enumerate() {
                if ctx.input.pressed(action) {
                    ctx.commands.app(TogglePlane { plane: index });
                }
            }
            let scrub =
                f32::from(ctx.input.is_held(SLICE_UP)) - f32::from(ctx.input.is_held(SLICE_DOWN));
            if scrub != 0.0 {
                let w = *ctx.app.slice.get() + scrub * W_SCRUB_RATE * ctx.step.dt;
                ctx.commands.app(SetSlice { w });
            }
        },
    );

    let mut applied = f32::NAN;
    session.system(
        Phase::Dispatch,
        "slice view",
        Access::new().domain(domain.id()),
        move |app: &mut Playground, domains: &mut Domains| {
            let w = *app.slice.get();
            if w == applied {
                return;
            }
            let Ok(r4) = domains.typed(domain) else {
                return;
            };
            let Some(spec) = r4.view_mut(layers.section) else {
                return;
            };
            spec.mapping = Box::new(Section4 { w });
            applied = w;
        },
    );

    let mut painted: Option<ColorMode> = None;
    session.system(
        Phase::Dispatch,
        "shading",
        Access::new().reads::<Slot>().domain(domain.id()),
        move |app: &mut Playground, domains: &mut Domains| {
            let mode = *app.color.get();
            if painted == Some(mode) {
                return;
            }
            let Ok(r4) = domains.typed(domain) else {
                return;
            };
            for (entity, slot) in app.slots.iter() {
                let Some(shades) = card_of(&slot.entry).and_then(|card| cards[card].shades) else {
                    continue;
                };
                if let Some(instance) = r4.instances.get_mut(entity) {
                    instance.shading = shades.of(mode);
                }
            }
            painted = Some(mode);
        },
    );

    let mut shown: Option<(Family, Option<loam_shape::polytope::Polytope4>)> = None;
    session.system(
        Phase::Dispatch,
        "projection view",
        Access::new().domain(domain.id()),
        move |app: &mut Playground, domains: &mut Domains| {
            let family = *app.projection.get();
            let active = *app.active.get();
            let subject = app
                .slots
                .iter()
                .find(|(_, slot)| slot.index == active)
                .and_then(|(_, slot)| slot.entry.shape.polytope4());
            if shown == Some((family, subject)) {
                return;
            }
            let Ok(r4) = domains.typed(domain) else {
                return;
            };
            let Some(spec) = r4.view_mut(layers.projection) else {
                return;
            };
            spec.mapping = Box::new(family.mapping(subject, 0));
            shown = Some((family, subject));
        },
    );

    session.system(
        Phase::Simulation,
        "spin",
        Access::new().reads::<Slot>().domain(domain.id()),
        move |app: &mut Playground, domains: &mut Domains, step: Step| {
            let omega = match *app.mode.get() {
                Mode::Rotate => app.spin.get().omega(),
                Mode::Compose if app.spin.get().running => app.composer.get().angular_velocity(),
                _ => return,
            };
            let Ok(r4) = domains.typed(domain) else {
                return;
            };
            for (entity, _) in app.slots.iter() {
                if let Some(pose) = r4.poses.get_mut(entity) {
                    pose.0.rotation = ((omega * step.dt).exp() * pose.0.rotation).normalize();
                }
            }
        },
    );
}

fn submit(
    commands: &mut Commands<Playground>,
    domain: DomainHandle<EuclideanR4>,
    cards: &[Card],
    intent: Intent,
) {
    match intent {
        Intent::Mode(mode) => commands.app(SetMode { mode, domain }),
        Intent::Active(slot) => commands.app(SetActive { slot }),
        Intent::Slice(w) => commands.app(SetSlice { w }),
        Intent::Plane(plane) => commands.app(TogglePlane { plane }),
        Intent::Running(running) => commands.app(SetRunning { running }),
        Intent::Projection(family) => commands.app(SetProjection { family }),
        Intent::Term(term) => commands.app(PushTerm { term }),
        Intent::DropTerm(index) => commands.app(DropTerm { index }),
        Intent::Draft(plane) => commands.app(DraftPlane { plane }),
        Intent::CommitDraft => commands.app(CommitDraft),
        Intent::ClearDraft => commands.app(ClearDraft),
        Intent::ClearTerms => commands.app(ClearComposer),
        Intent::Scrub(scrub) => commands.app(SetScrub { scrub, domain }),
        Intent::Gimbal => commands.app(ToggleGimbal),
        Intent::Turn(rotor) => commands.app(TurnRow { rotor, domain }),
        Intent::Color(mode) => commands.app(SetColorMode { mode }),
        Intent::Points => commands.app(TogglePoints),
        Intent::Shape(slot, card) => {
            let Some(entry) = catalog::SHAPE_CATALOG.get(card) else {
                return;
            };
            commands.app(SetShape {
                slot,
                entry: *entry,
                geometry: cards[card].geometry,
                material: cards[card].material,
                shades: cards[card].shades,
                domain,
            })
        }
    };
}

pub(crate) fn bindings() -> Bindings {
    let mut bound = Bindings::new()
        .key(Key::Space, SPIN)
        .key(Key::Letter('t'), SPIN)
        .key(Key::Letter('m'), NEXT_MODE)
        .key(Key::Letter('r'), RESET)
        .key(Key::Letter('g'), GIMBAL)
        .key(Key::Letter('e'), SLICE_UP)
        .key(Key::Letter('q'), SLICE_DOWN);
    for (index, action) in PLANE.into_iter().enumerate() {
        bound = bound.key(Key::Digit(index as u8 + 1), action);
    }
    bound
}

pub(crate) struct Frame {
    sky: SkyGroundPass,
    hyperslice: HyperslicePass,
    cut: LinePass,
    rings: LinePass,
    cloud: PointPass,
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self {
            sky: SkyGroundPass::new(scene::ground(true)),
            hyperslice: HyperslicePass::new(scene::shader_source()),
            cut: LinePass::new("section"),
            rings: LinePass::new("gimbal"),
            cloud: PointPass::new("points"),
        }
    }

    fn passes(&self) -> Vec<Box<dyn FramePass>> {
        vec![
            Box::new(self.sky.clone()),
            Box::new(self.hyperslice.clone()),
            Box::new(self.cut.clone()),
            Box::new(self.rings.clone()),
            Box::new(self.cloud.clone()),
        ]
    }
}

pub(crate) fn frame_sections(frame: &Frame) -> Vec<&'static str> {
    let mut schedule = PassSchedule::new(DepthConvention::ReversedZ);
    for pass in frame.passes() {
        if schedule.register(pass).is_err() {
            return Vec::new();
        }
    }
    let before = frame
        .passes()
        .iter()
        .filter(|pass| pass.order() == PassOrder::BeforeScene)
        .count();
    let ordered: Vec<&'static str> = schedule.names().collect();
    let mut names: Vec<&'static str> = vec!["present-clear"];
    names.extend(ordered.iter().take(before).copied());
    names.push("present-draw");
    names.extend(ordered.iter().skip(before).copied());
    names
}

struct Scratch {
    center: glam::Vec3,
    slots: Vec<(Entity, ShapeEntry)>,
    bodies: Vec<BodyUniform>,
    segments: Vec<SegmentRecord>,
    cutter: section::Cutter,
    cloud: points::Cloud,
    anchors: Vec<(usize, &'static str, glam::Vec3)>,
}

impl Scratch {
    fn new(row: &[ShapeEntry]) -> Self {
        Self {
            center: glam::Vec3::ZERO,
            slots: Vec::new(),
            bodies: Vec::new(),
            segments: Vec::new(),
            cutter: section::Cutter::new(SECTION_COLOR, SECTION_WIDTH_PX),
            cloud: points::Cloud::new(row.iter().filter_map(|entry| entry.shape.polytope4())),
            anchors: Vec::new(),
        }
    }
}

fn collect(
    session: &mut Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    slice: f32,
    scratch: &mut Scratch,
) {
    let mode = *session.app.color.get();
    let cloud_on = *session.app.points.get();
    scratch.slots.clear();
    scratch.slots.extend(
        session
            .app
            .slots
            .iter()
            .map(|(entity, slot)| (entity, slot.entry)),
    );
    scratch.anchors.clear();
    scratch.anchors.extend(
        session
            .app
            .slots
            .iter()
            .map(|(_, slot)| (slot.index, slot.entry.label, glam::Vec3::ZERO)),
    );
    scratch.bodies.clear();
    scratch.segments.clear();
    scratch.cloud.clear();
    let mut sum = glam::Vec3::ZERO;
    let Ok(r4) = session.domains_mut().typed(domain) else {
        return;
    };
    for (at, (entity, entry)) in scratch.slots.iter().enumerate() {
        let Some(pose) = r4.poses.get(*entity) else {
            continue;
        };
        let center = pose.0.translation.truncate();
        sum += center;
        if let Some(anchor) = scratch.anchors.get_mut(at) {
            anchor.2 = center;
        }
        scratch.bodies.push(scene::body_of(entry, &pose.0));
        let Some(polytope) = entry.shape.polytope4() else {
            continue;
        };
        scratch.cutter.cut(
            polytope,
            pose.0.rotation,
            pose.0.translation,
            slice,
            &mut scratch.segments,
        );
        if cloud_on {
            scratch
                .cloud
                .append(polytope, pose.0.rotation, pose.0.translation, mode);
        }
    }
    scratch.center = sum / scratch.slots.len().max(1) as f32;
}

fn main() -> Result<(), HostError> {
    let args = Args::current();
    let row = catalog::parse_row(&args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let intents: Intents = Intents::default();
    let mut booted = boot(&row, &intents)?;
    let frame = Frame::new();
    let config = HostConfig::new("polytope playground", bindings());
    if args.has_bare_flag("headless") {
        return headless(&mut booted, &frame, &config);
    }

    let grabbed = Arc::new(AtomicBool::new(false));
    let sky = frame.sky.clone();
    let hyperslice = frame.hyperslice.clone();
    let cut = frame.cut.clone();
    let rings = frame.rings.clone();
    let cloud = frame.cloud.clone();
    let turn_intents = intents.clone();
    let mut gimbal = Gimbal::default();
    let ui_intents = intents.clone();
    let mode_intents = intents.clone();
    let slice_intents = intents.clone();
    let spin_intents = intents.clone();
    let shape_intents = intents.clone();
    let projection_intents = intents.clone();
    let color_intents = intents.clone();
    let formula_intents = intents.clone();
    let scrub_intents = intents.clone();
    let domain = booted.domain;
    let mut scratch = Scratch::new(&row);
    let mut panel = ui::Panel::default();
    let mut orbit = Orbit::around([0.0, BODY_Y, 0.0], 9.0);
    orbit.pitch = -0.25;

    let mut app = SessionApp::new(config);
    for pass in frame.passes() {
        app = app.pass(pass);
    }
    app = app
        .command(
            "mode",
            "set the playground mode (rotate | toybox)",
            move |args, _submit, out| {
                match args.first().copied().and_then(Mode::from_token) {
                    Some(mode) => {
                        push(&mode_intents, Intent::Mode(mode));
                        out.line(format!("mode: {} requested", mode.name()));
                    }
                    None => out.line("usage: mode rotate | toybox"),
                }
                Ok(())
            },
        )
        .command(
            "slice",
            "set the w hyperplane the marcher and the cut share",
            move |args, _submit, out| {
                match args.first().and_then(|token| token.parse::<f32>().ok()) {
                    Some(w) => {
                        push(&slice_intents, Intent::Slice(w));
                        out.line(format!("slice: {w:.3} requested"));
                    }
                    None => out.line("usage: slice <w>"),
                }
                Ok(())
            },
        )
        .command(
            "spin",
            "run or pause the row's rotation, or toggle one of its six planes",
            move |args, _submit, out| {
                match args.first().copied() {
                    Some("on") => push(&spin_intents, Intent::Running(true)),
                    Some("off") => push(&spin_intents, Intent::Running(false)),
                    Some(token) => match token.parse::<usize>() {
                        Ok(plane) if (1..=6).contains(&plane) => {
                            push(&spin_intents, Intent::Plane(plane - 1));
                        }
                        _ => out.line("usage: spin on | off | <1..6>"),
                    },
                    None => out.line("usage: spin on | off | <1..6>"),
                }
                Ok(())
            },
        )
        .command(
            "shape",
            "make one slot of the row the active polytope",
            move |args, _submit, out| {
                match args.first().and_then(|token| token.parse::<usize>().ok()) {
                    Some(slot) => {
                        push(&shape_intents, Intent::Active(slot));
                        out.line(format!("shape: slot {slot} requested"));
                    }
                    None => out.line("usage: shape <slot>"),
                }
                Ok(())
            },
        )
        .command(
            "project",
            "choose the projection the second layer draws (perspective | stereographic | schlegel)",
            move |args, _submit, out| {
                match args.first().copied().and_then(Family::from_token) {
                    Some(family) => {
                        push(&projection_intents, Intent::Projection(family));
                        out.line(format!("projection: {} requested", family.name()));
                    }
                    None => out.line("usage: project perspective | stereographic | schlegel"),
                }
                Ok(())
            },
        )
        .command(
            "colour",
            "choose how edges and points are coloured (vertex | edge | w-depth)",
            move |args, _submit, out| {
                match args.first().copied().and_then(ColorMode::from_token) {
                    Some(mode) => {
                        push(&color_intents, Intent::Color(mode));
                        out.line(format!("colour: {} requested", mode.name()));
                    }
                    None => out.line("usage: colour vertex | edge | w-depth"),
                }
                Ok(())
            },
        )
        .command(
            "formula",
            "add a term to the composer sequence, or clear it, or drop one",
            move |args, _submit, out| {
                match args {
                    ["clear"] => push(&formula_intents, Intent::ClearTerms),
                    ["drop", index] => match index.parse::<usize>() {
                        Ok(index) => push(&formula_intents, Intent::DropTerm(index)),
                        Err(_) => out.line("usage: formula drop <index>"),
                    },
                    [] => out.line("usage: formula <angle> (<plane> + ...) | clear | drop <index>"),
                    terms => match composer::parse_term(&terms.join(" ")) {
                        Ok(term) => {
                            push(&formula_intents, Intent::Term(term));
                            out.line("formula: term requested");
                        }
                        Err(error) => out.line(format!("formula: {error}")),
                    },
                }
                Ok(())
            },
        )
        .command(
            "scrub",
            "set the row's turn along the composer's bivector, in degrees",
            move |args, _submit, out| {
                match args.first().and_then(|token| token.parse::<f32>().ok()) {
                    Some(degrees) => {
                        push(&scrub_intents, Intent::Scrub(degrees.to_radians()));
                        out.line(format!("scrub: {degrees:.1} degrees requested"));
                    }
                    None => out.line("usage: scrub <degrees>"),
                }
                Ok(())
            },
        )
        .on_frame(move |hook: &mut FrameHook<'_, Playground>| {
            let wants_pointer = hook.ui.is_some_and(|context| context.wants_pointer_input());
            gimbal.enabled = *hook.session.app.gimbal.get();
            let turning = drive_gimbal(
                hook.session,
                wants_pointer,
                &mut gimbal,
                scratch.center,
                &turn_intents,
            );
            if !turning {
                drive_pointer(hook.session, wants_pointer, &grabbed);
            }
            if !turning && !grabbed.load(Ordering::Relaxed) && !wants_pointer {
                orbit.drag(pointer_drag(hook.session));
            }
            hook.session.views_mut().root_mut().eye = orbit.eye();
            let eye = orbit.eye();

            let slice = *hook.session.app.slice.get();
            let floor = *hook.session.app.floor.get();
            collect(hook.session, domain, slice, &mut scratch);
            sky.publish(&eye, scene::ground(floor));
            hyperslice.publish(scene::uniforms(&eye, slice, floor), &scratch.bodies);
            cut.publish(&eye, &scratch.segments);
            rings.publish(&eye, gimbal.rings(scratch.center));
            cloud.publish(&eye, scratch.cloud.records());
            if let Some(context) = hook.ui {
                ui::draw(context, hook.session, &mut panel, &ui_intents);
                ui::callouts(context, hook.session, &scratch.anchors);
            }
        });
    launch(booted.session, app)
}

fn pointer_drag(session: &Session<Playground>) -> [f32; 2] {
    match *session.app.pointer.get() {
        Some(pointer) if pointer.phase == PointerPhase::Moved => pointer.delta,
        _ => [0.0; 2],
    }
}

/// True while the gimbal owns the pointer, so the body grab and the orbit stay out of the drag.
fn drive_gimbal(
    session: &mut Session<Playground>,
    wants_pointer: bool,
    gimbal: &mut Gimbal,
    center: glam::Vec3,
    intents: &Intents,
) -> bool {
    let root = session.views().root();
    let pointer = *session.app.pointer.get();
    let ray = pointer.and_then(|pointer| session.views().ray(root, pointer.ndc));
    if !gimbal.enabled {
        gimbal.release();
        return false;
    }
    gimbal.aim(ray.as_ref(), center);
    let (Some(pointer), Some(ray)) = (pointer, ray) else {
        return gimbal.held();
    };
    match pointer.phase {
        PointerPhase::Began if !wants_pointer => gimbal.press(&ray, center),
        PointerPhase::Moved if gimbal.held() => {
            if let Some(rotor) = gimbal.turn(&ray) {
                push(intents, Intent::Turn(rotor));
            }
            true
        }
        PointerPhase::Ended | PointerPhase::Cancelled => {
            let held = gimbal.held();
            gimbal.release();
            held
        }
        _ => gimbal.held(),
    }
}

fn drive_pointer(session: &mut Session<Playground>, wants_pointer: bool, grabbed: &AtomicBool) {
    let Some(pointer) = *session.app.pointer.get() else {
        return;
    };
    match pointer.phase {
        PointerPhase::Began if !wants_pointer => {
            let taken = session.grab(pointer.ndc, pointer.time).is_ok();
            grabbed.store(taken, Ordering::Relaxed);
        }
        PointerPhase::Moved
            if grabbed.load(Ordering::Relaxed)
                && session.drag(pointer.ndc, pointer.time).is_err() =>
        {
            session.release();
            grabbed.store(false, Ordering::Relaxed);
        }
        PointerPhase::Ended | PointerPhase::Cancelled => {
            session.release();
            grabbed.store(false, Ordering::Relaxed);
        }
        _ => {}
    }
}

fn report(booted: &mut Boot, frame: &Frame, config: &HostConfig) -> Result<Vec<String>, HostError> {
    let publication = run_headless(&mut booted.session, config, HEADLESS_STEPS, &[])?;
    let active = *booted.session.app.active.get();
    let entry = booted
        .session
        .app
        .slots
        .iter()
        .find(|(_, slot)| slot.index == active)
        .map(|(_, slot)| slot.entry);
    let edges = entry
        .and_then(|entry| entry.shape.polytope4())
        .map_or(0, |polytope| polytope.edge_count());
    Ok(vec![
        composer_line(),
        format!(
            "active: {} with {edges} edges",
            entry.map_or("none", |entry| entry.label)
        ),
        format!(
            "published segments: {}",
            publication
                .views
                .iter()
                .map(|view| view.records.segments().len())
                .sum::<usize>()
        ),
        format!("sections: {}", frame_sections(frame).join(", ")),
    ])
}

fn composer_line() -> String {
    use loam_math::{Bivector, Plane4, Rotor};

    let mut probe = Composer::default();
    probe.push(composer::parse_term(HEADLESS_FORMULA).unwrap_or_default());
    let mut text = String::new();
    probe.write(&mut text);
    let turned = probe.axis().map_or(loam_math::Bivector4::ZERO, |axis| {
        (axis * HEADLESS_SCRUB).exp().log()
    });
    format!(
        "composer: {text} scrubbed to {HEADLESS_SCRUB:.3} turns xy {:.4} zw {:.4}",
        turned.component(Plane4::Xy),
        turned.component(Plane4::Zw)
    )
}

fn headless(booted: &mut Boot, frame: &Frame, config: &HostConfig) -> Result<(), HostError> {
    for line in report(booted, frame, config)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use loam_render::raymarch::RaymarchShape;
    use loam_runtime::{AppCommand, Records};
    use loam_shape::polytope::Polytope4;

    use super::*;

    const CELL24: ShapeEntry = ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
    };

    const EYE_BACK: f32 = 5.0;
    const HALF_FOV_TAN: f32 = 0.577_350_26;

    fn one_slot() -> (Boot, Intents) {
        let intents = Intents::default();
        let mut booted = boot(&[CELL24], &intents).expect("the session boots");
        booted.session.views_mut().root_mut().eye =
            Eye::looking_at([0.0, BODY_Y, EYE_BACK], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
        booted.session.app.spin.get_mut().running = false;
        (booted, intents)
    }

    fn slot_entity(booted: &Boot) -> Entity {
        booted
            .session
            .app
            .slots
            .iter()
            .map(|(entity, _)| entity)
            .next()
            .expect("the row has a slot")
    }

    fn published<R>(
        session: &mut Session<Playground>,
        records: &mut Records<Playground>,
        read: impl FnOnce(&loam_runtime::Publication<Playground>) -> R,
    ) -> R {
        records.publish(session).expect("published");
        let publication = records.lend().expect("the buffer is free");
        let value = read(&publication);
        records.release(publication);
        value
    }

    #[test]
    fn a_mode_command_lands_in_the_store_at_the_next_boundary() {
        let (mut booted, intents) = one_slot();
        push(&intents, Intent::Mode(Mode::Toybox));
        assert_eq!(
            *booted.session.app.mode.get(),
            Mode::Rotate,
            "an intent changed the store before any boundary applied it"
        );

        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");

        assert_eq!(*booted.session.app.mode.get(), Mode::Toybox);
        assert_eq!(
            booted.session.results().len(),
            1,
            "the boundary applied {} commands, not the one mode change",
            booted.session.results().len()
        );
        assert!(booted.session.results()[0].outcome.is_ok());
    }

    #[test]
    fn the_published_wireframe_carries_every_edge_of_the_24_cell() {
        let (mut booted, _intents) = one_slot();
        let mut records = Records::default();
        let counts = published(&mut booted.session, &mut records, |publication| {
            publication
                .views
                .iter()
                .map(|view| view.records.segments().len())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            counts,
            [96, 96],
            "the 24-cell has 96 edges in each of the section and projection layers"
        );
    }

    #[test]
    fn an_unmoved_row_under_an_unchanged_view_is_not_republished() {
        let (mut booted, intents) = one_slot();
        let mut records = Records::default();
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let built = |session: &mut Session<Playground>, records: &mut Records<Playground>| {
            published(session, records, |publication| {
                publication.views[0].records.built()
            })
        };
        built(&mut booted.session, &mut records);
        let first = built(&mut booted.session, &mut records);
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let idle = built(&mut booted.session, &mut records);
        assert_eq!(first, idle, "publication rebuilt records nothing changed");

        push(&intents, Intent::Slice(0.4));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let moved = built(&mut booted.session, &mut records);
        assert_ne!(
            idle, moved,
            "a new slice left the view's records at their old build"
        );
    }

    #[test]
    fn a_toybox_body_carries_its_slots_pose_down_after_a_step() {
        let (mut booted, intents) = one_slot();
        push(&intents, Intent::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let entity = slot_entity(&booted);
        for _ in 0..10 {
            booted.session.tick().expect("the tick ran");
        }
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let height = r4
            .poses
            .get(entity)
            .expect("the slot has a pose")
            .0
            .translation
            .y;
        assert!(
            height < BODY_Y - 1e-3,
            "the facility stepped its world without writing the pose back: y is still {height}"
        );
    }

    #[test]
    fn a_drag_moves_the_grabbed_slot_to_where_the_pointer_ray_meets_its_grab_plane() {
        let (mut booted, _intents) = one_slot();
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        booted
            .session
            .grab([0.0, 0.0], 0.0)
            .expect("the ray through the slot centre picks it");

        const NDC_X: f32 = 0.5;
        booted
            .session
            .drag([NDC_X, 0.0], 1.0)
            .expect("the drag meets its plane");
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary applied the move");

        let entity = slot_entity(&booted);
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let at = r4.poses.get(entity).expect("pose").0.translation;
        let expected = (EYE_BACK - BODY_SIZE) * NDC_X * HALF_FOV_TAN;
        assert!(
            (at.x - expected).abs() < 1e-4,
            "the drag put the slot at x {} rather than the analytic {expected}",
            at.x
        );
        assert!(
            (at.y - BODY_Y).abs() < 1e-4 && at.z.abs() < 1e-4 && at.w.abs() < 1e-4,
            "the drag left the grab plane: {at:?}"
        );
    }

    #[test]
    fn a_warmed_frame_asks_the_allocator_for_nothing() {
        let (mut booted, _intents) = one_slot();
        let mut records = Records::default();
        let mut scratch = Scratch::new(&[CELL24]);
        let frame =
            |booted: &mut Boot, records: &mut Records<Playground>, scratch: &mut Scratch| {
                booted
                    .session
                    .boundary(Input::default())
                    .expect("the boundary ran");
                booted.session.tick().expect("the tick ran");
                records.publish(&mut booted.session).expect("published");
                let publication = records.lend().expect("the buffer is free");
                records.release(publication);
                collect(&mut booted.session, booted.domain, 0.0, scratch);
            };
        for _ in 0..16 {
            frame(&mut booted, &mut records, &mut scratch);
        }

        let calls = alloc_probe::allocations_in(|| {
            for _ in 0..16 {
                frame(&mut booted, &mut records, &mut scratch);
            }
        });
        assert_eq!(
            calls, 0,
            "sixteen warmed frames of boundary, tick, publication, and collection asked the allocator {calls} times"
        );
    }

    #[test]
    fn a_drag_in_toybox_survives_the_step_that_follows_it() {
        let (mut booted, intents) = one_slot();
        push(&intents, Intent::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        booted
            .session
            .grab([0.0, 0.0], 0.0)
            .expect("the ray through the slot centre picks it");

        const NDC_X: f32 = 0.5;
        booted
            .session
            .drag([NDC_X, 0.0], 1.0)
            .expect("the drag meets its plane");
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary applied the move");
        let dragged = (EYE_BACK - BODY_SIZE) * NDC_X * HALF_FOV_TAN;

        let entity = slot_entity(&booted);
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let body = r4
            .physics()
            .and_then(|physics| physics.body(entity))
            .expect("the slot has a body");
        let placed = r4
            .physics()
            .and_then(|physics| physics.world().bodies.get(body))
            .expect("the body is live")
            .position;
        assert!(
            (placed.x - dragged).abs() < 1e-4,
            "the world never took the drag: the body sits at x {} rather than {dragged}",
            placed.x
        );

        booted.session.tick().expect("the tick ran");
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let pose = r4.poses.get(entity).expect("pose").0.translation;
        assert!(
            (pose.x - dragged).abs() < 1e-3,
            "the step overwrote the drag from a body that never moved: pose x {}",
            pose.x
        );
    }

    #[test]
    fn a_shape_card_respawns_the_slot_in_place_with_the_new_polytopes_edges() {
        let (mut booted, intents) = one_slot();
        let rest = booted
            .session
            .app
            .slots
            .iter()
            .map(|(_, slot)| slot.rest)
            .next()
            .expect("the row has a slot");

        push(&intents, Intent::Shape(0, 0));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");

        let held: Vec<Slot> = booted.session.app.slots.iter().map(|(_, s)| *s).collect();
        assert_eq!(held.len(), 1, "the swap left {} slots", held.len());
        assert_eq!(held[0].entry.label, "5-cell");
        assert_eq!(held[0].index, 0);
        assert_eq!(
            held[0].rest, rest,
            "the replacement moved off the slot's rest"
        );

        let mut records = Records::default();
        let counts = published(&mut booted.session, &mut records, |publication| {
            publication
                .views
                .iter()
                .map(|view| view.records.segments().len())
                .collect::<Vec<_>>()
        });
        assert_eq!(
            counts,
            [10, 10],
            "the swapped slot still publishes the old polytope's edges"
        );
    }

    #[test]
    fn the_scrub_turns_each_slot_by_its_own_angle_along_the_sequences_bivector() {
        use loam_math::{Bivector, Bivector4, Rotor};

        let (mut booted, intents) = one_slot();
        push(&intents, Intent::Mode(Mode::Compose));
        push(
            &intents,
            Intent::Term(composer::parse_term("90deg (xy + zw)").expect("the formula parses")),
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");

        const SCRUB: f32 = 0.7;
        let start = loam_math::Plane4::Xz.unit_bivector() * 0.5
            + loam_math::Plane4::Xy.unit_bivector() * 0.3;
        let entity = slot_entity(&booted);
        booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain")
            .poses
            .get_mut(entity)
            .expect("pose")
            .0
            .rotation = start.exp();
        push(&intents, Intent::Scrub(SCRUB));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");

        let axis = booted
            .session
            .app
            .composer
            .get()
            .axis()
            .expect("the sequence names a bivector");
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let log: Bivector4 = r4.poses.get(entity).expect("pose").0.rotation.log();
        assert!(
            (log.dot(axis) - SCRUB).abs() < 1e-4,
            "the scrub turned the slot by {} along the sequence, not {SCRUB}",
            log.dot(axis)
        );
        let across = log + axis * -log.dot(axis);
        let kept = start + axis * -start.dot(axis);
        assert!(
            (across + kept * -1.0).magnitude() < 1e-4,
            "the scrub disturbed the turn across its own bivector: {across:?} rather than {kept:?}"
        );
    }

    #[test]
    fn the_gimbal_rotor_turns_the_row_by_the_angle_its_ring_names() {
        use loam_math::{Bivector, Plane4, Rotor};

        let (mut booted, _intents) = one_slot();
        const ANGLE: f32 = 0.4;
        let domain = booted.domain;
        booted.session.dispatch(|d| {
            TurnRow {
                rotor: (Plane4::Xw.unit_bivector() * ANGLE).exp(),
                domain,
            }
            .apply(d)
            .expect("the turn applies")
        });

        let entity = slot_entity(&booted);
        let r4 = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("the r4 domain");
        let turned = r4
            .poses
            .get(entity)
            .expect("pose")
            .0
            .rotation
            .apply(Vec4::X);
        let expected = Vec4::new(ANGLE.cos(), 0.0, 0.0, ANGLE.sin());
        assert!(
            (turned - expected).length() < 1e-5,
            "the ring's rotor sent x to {turned} rather than the analytic {expected}"
        );
    }

    #[test]
    fn a_cancelled_pointer_releases_a_live_grab() {
        let (mut booted, _intents) = one_slot();
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        booted
            .session
            .grab([0.0, 0.0], 0.0)
            .expect("the ray through the slot centre picks it");
        let grabbed = AtomicBool::new(true);
        booted.session.app.pointer.set(Some(Pointer {
            id: 0,
            ndc: [0.0, 0.0],
            delta: [0.0; 2],
            phase: PointerPhase::Cancelled,
            time: 1.0,
        }));

        drive_pointer(&mut booted.session, false, &grabbed);

        assert!(!grabbed.load(Ordering::Relaxed));
        assert!(
            booted.session.drag([0.5, 0.0], 2.0).is_err(),
            "the grab survived the focus change that cancelled the pointer"
        );
    }

    #[test]
    fn the_headless_report_names_the_active_polytope_and_the_frames_sections() {
        let (mut booted, _intents) = one_slot();
        let frame = Frame::new();
        let config = HostConfig::new("polytope playground", bindings());
        let lines = report(&mut booted, &frame, &config).expect("the headless run");
        assert!(
            lines[1].contains("24-cell") && lines[1].contains("96 edges"),
            "the report does not name the active polytope and its edges: {}",
            lines[1]
        );
        assert!(
            lines[3].contains("present-clear")
                && lines[3].contains("sky-ground")
                && lines[3].contains("present-draw")
                && lines[3].contains("hyperslice")
                && lines[3].contains("section"),
            "the report does not list the frame's sections: {}",
            lines[3]
        );
        assert!(
            lines[0].contains("xy 0.4950") && lines[0].contains("zw 0.4950"),
            "the composer line does not carry the scrubbed turn: {}",
            lines[0]
        );
    }
}
