use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use glam::Vec4;
use loam_app::args::Args;
use loam_app::session::{launch, FrameHook, SessionApp};
use loam_math::{Bivector, EuclideanR4, Iso4Flat};
use loam_render::pass::{FramePass, PassOrder, PassSchedule};
use loam_render::raymarch::BodyUniform;
use loam_render::{DepthConvention, HyperslicePass, LinePass, SkyGroundPass};
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
mod consts;
mod mode;
mod projection;
mod scene;
mod section;
mod toy;
mod ui;

use catalog::ShapeEntry;
use consts::{BODY_SIZE, BODY_X_SPACING, BODY_Y, GRAVITY, W_SCRUB_RATE};
use mode::{Mode, SetActive, SetMode, SetProjection, SetRunning, SetSlice, Spin, TogglePlane};
use projection::Family;

const SPIN: ActionId = ActionId(0);
const SLICE_UP: ActionId = ActionId(1);
const SLICE_DOWN: ActionId = ActionId(2);
const NEXT_MODE: ActionId = ActionId(3);
const RESET: ActionId = ActionId(4);
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
        active: Value<usize>,
        slice: Value<f32>,
        projection: Value<Family>,
        pointer: Value<Option<Pointer>>,
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
    let prepared: Vec<Option<PreparedId>> = row
        .iter()
        .map(|entry| {
            entry.shape.polytope4().map(|polytope| {
                session.prepare(PreparedGeometry::edges_of(polytope.topology(), BODY_SIZE))
            })
        })
        .collect();
    let materials: Vec<MaterialId> = row
        .iter()
        .map(|entry| {
            let [r, g, b] = entry.body_color;
            session.add_material(Material::lines([r, g, b, 0.9], EDGE_WIDTH_PX))
        })
        .collect();

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
            if let Some(geometry) = prepared[index] {
                bundle = bundle.instance(Instance::new(geometry, materials[index]));
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

    install_systems(&mut session, domain, layers, intents);
    session.set_initial()?;
    Ok(Boot { session, domain })
}

#[derive(Clone, Copy)]
struct Layers {
    section: ViewId,
    projection: ViewId,
}

fn install_systems(
    session: &mut Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    layers: Layers,
    intents: &Intents,
) {
    let queued = intents.clone();
    let mut drained: Vec<Intent> = Vec::new();
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
                submit(commands, domain, intent);
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
                    Mode::Rotate => Mode::Toybox,
                    Mode::Toybox => Mode::Rotate,
                };
                ctx.commands.app(SetMode { mode: next, domain });
            }
            if ctx.input.pressed(RESET) {
                ctx.commands.submit(Command::Reset);
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
            if *app.mode.get() != Mode::Rotate {
                return;
            }
            let omega = app.spin.get().omega();
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

fn submit(commands: &mut Commands<Playground>, domain: DomainHandle<EuclideanR4>, intent: Intent) {
    match intent {
        Intent::Mode(mode) => commands.app(SetMode { mode, domain }),
        Intent::Active(slot) => commands.app(SetActive { slot }),
        Intent::Slice(w) => commands.app(SetSlice { w }),
        Intent::Plane(plane) => commands.app(TogglePlane { plane }),
        Intent::Running(running) => commands.app(SetRunning { running }),
        Intent::Projection(family) => commands.app(SetProjection { family }),
    };
}

pub(crate) fn bindings() -> Bindings {
    let mut bound = Bindings::new()
        .key(Key::Space, SPIN)
        .key(Key::Letter('t'), SPIN)
        .key(Key::Letter('m'), NEXT_MODE)
        .key(Key::Letter('r'), RESET)
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
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self {
            sky: SkyGroundPass::new(scene::ground(true)),
            hyperslice: HyperslicePass::new(scene::shader_source()),
            cut: LinePass::new("section"),
        }
    }

    fn passes(&self) -> Vec<Box<dyn FramePass>> {
        vec![
            Box::new(self.sky.clone()),
            Box::new(self.hyperslice.clone()),
            Box::new(self.cut.clone()),
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
    slots: Vec<(Entity, ShapeEntry)>,
    bodies: Vec<BodyUniform>,
    segments: Vec<SegmentRecord>,
    cutter: section::Cutter,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            slots: Vec::new(),
            bodies: Vec::new(),
            segments: Vec::new(),
            cutter: section::Cutter::new(SECTION_COLOR, SECTION_WIDTH_PX),
        }
    }
}

fn collect(
    session: &mut Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    slice: f32,
    scratch: &mut Scratch,
) {
    scratch.slots.clear();
    scratch.slots.extend(
        session
            .app
            .slots
            .iter()
            .map(|(entity, slot)| (entity, slot.entry)),
    );
    scratch.bodies.clear();
    scratch.segments.clear();
    let Ok(r4) = session.domains_mut().typed(domain) else {
        return;
    };
    for (entity, entry) in &scratch.slots {
        let Some(pose) = r4.poses.get(*entity) else {
            continue;
        };
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
    }
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
    let ui_intents = intents.clone();
    let mode_intents = intents.clone();
    let slice_intents = intents.clone();
    let spin_intents = intents.clone();
    let shape_intents = intents.clone();
    let projection_intents = intents.clone();
    let domain = booted.domain;
    let mut scratch = Scratch::default();
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
        .on_frame(move |hook: &mut FrameHook<'_, Playground>| {
            let wants_pointer = hook.ui.is_some_and(|context| context.wants_pointer_input());
            drive_pointer(hook.session, wants_pointer, &grabbed);
            if !grabbed.load(Ordering::Relaxed) && !wants_pointer {
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
            if let Some(context) = hook.ui {
                ui::draw(context, hook.session, &ui_intents);
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

fn headless(booted: &mut Boot, frame: &Frame, config: &HostConfig) -> Result<(), HostError> {
    for line in report(booted, frame, config)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use loam_render::raymarch::RaymarchShape;
    use loam_runtime::Records;
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
        let mut scratch = Scratch::default();
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
    fn the_headless_report_names_the_active_polytope_and_the_frames_sections() {
        let (mut booted, _intents) = one_slot();
        let frame = Frame::new();
        let config = HostConfig::new("polytope playground", bindings());
        let lines = report(&mut booted, &frame, &config).expect("the headless run");
        assert!(
            lines[0].contains("24-cell") && lines[0].contains("96 edges"),
            "the report does not name the active polytope and its edges: {}",
            lines[0]
        );
        assert!(
            lines[2].contains("present-clear")
                && lines[2].contains("sky-ground")
                && lines[2].contains("present-draw")
                && lines[2].contains("hyperslice")
                && lines[2].contains("section"),
            "the report does not list the frame's sections: {}",
            lines[2]
        );
    }
}
