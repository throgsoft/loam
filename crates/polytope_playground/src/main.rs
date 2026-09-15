use std::sync::Arc;

use glam::Vec4;
use loam::app::args::Args;
use loam::app::session::{launch_or_headless, look, FrameHook, Orbit, SessionApp};
use loam::math::{Bivector, Bivector4, EuclideanR4};
use loam::render::pass::FramePass;
use loam::render::raymarch::BodyUniform;
use loam::render::{HyperslicePass, LinePass, PointPass, SkyGroundPass};
use loam::runtime::host::run_headless;
use loam::runtime::host::{HostConfig, HostError};
#[cfg(test)]
use loam::runtime::Input;
use loam::runtime::{
    ActionId, AppCommand, Bindings, Ctx, Dispatch, DomainBuilder, DomainError, DomainHandle,
    Domains, Entity, Eye, Instance, Key, LogCapacity, Material, MaterialId, Outcome, Phase,
    Pointer, PointerButton, PointerPhase, Pose, PreparedGeometry, PreparedId, Rejection, Section4,
    Session, SimConfig, SpawnBundle, ViewId, ViewSpec,
};

#[cfg(test)]
#[global_allocator]
static COUNTING_ALLOCATOR: loam_time::alloc::CountingAllocator<std::alloc::System> =
    loam_time::alloc::CountingAllocator::new(std::alloc::System);

mod camera;
mod catalog;
mod color;
mod console;
mod consts;
mod display;
mod gimbal;
mod guides;
mod hud;
mod mode;
mod projection;
mod row;
mod scene;
mod strip;
mod toy;
mod ui;

use catalog::ShapeEntry;
use color::{ColorMode, Shades};
use consts::{BODY_SIZE, BODY_X_SPACING, BODY_Y, W_SCRUB_RATE};
use display::{Display, Surface};
use gimbal::Gimbal;
use mode::{Mode, Spin};
use projection::Family;
use strip::{Cell, Strip};

const SPIN: ActionId = ActionId(0);
const SLICE_UP: ActionId = ActionId(1);
const SLICE_DOWN: ActionId = ActionId(2);
const NEXT_MODE: ActionId = ActionId(3);
const RESET: ActionId = ActionId(4);
const GIMBAL: ActionId = ActionId(5);
const STRIP: ActionId = ActionId(6);
const CONTROLS: ActionId = ActionId(7);
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
const EDGE_WIDTH_PX: f32 = display::DEFAULT_WIREFRAME_WIDTH_PX;
const HEADLESS_STEPS: u32 = 8;
const HEADLESS_FRAME: (u32, u32) = (1280, 720);
const CAMERA_DISTANCE: f32 = 8.0;
const DOMAIN_NAME: &str = "r4";

#[derive(Clone, Copy)]
pub(crate) struct Slot {
    pub(crate) index: usize,
    pub(crate) entry: ShapeEntry,
    pub(crate) rest: Vec4,
}

#[derive(Clone, Copy)]
pub(crate) struct Wall;

#[derive(Clone, Copy)]
pub(crate) struct Toy {
    pub(crate) rest_anchor: Vec4,
    pub(crate) rest_time: f32,
}

#[derive(Clone, Copy)]
pub(crate) struct HiddenSlot {
    pub(crate) slot: Slot,
    pub(crate) instance: Option<Instance>,
}

#[derive(Clone, Copy)]
pub(crate) struct Control {
    orbit: Orbit,
    gimbal: Gimbal,
    latest: Option<Pointer>,
    center: glam::Vec3,
    camera_focus: Option<(bool, usize, Mode)>,
    filmstrip_return: Option<Orbit>,
}

impl Default for Control {
    fn default() -> Self {
        let mut orbit = Orbit::around([0.0; 3], CAMERA_DISTANCE);
        orbit.pitch = -0.25;
        Self {
            orbit,
            gimbal: Gimbal::default(),
            latest: None,
            center: glam::Vec3::ZERO,
            camera_focus: None,
            filmstrip_return: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(crate) struct Card {
    geometry: Option<PreparedId>,
    toy_geometry: Option<PreparedId>,
    material: MaterialId,
    cut: MaterialId,
    shades: Option<Shades>,
}

impl Card {
    fn dressed(&self, geometry: PreparedId) -> Instance {
        Instance::new(geometry, self.material).sectioned(self.cut)
    }

    pub(crate) fn body(&self) -> Option<Instance> {
        self.geometry.map(|geometry| self.dressed(geometry))
    }

    pub(crate) fn toy(&self) -> Option<Instance> {
        self.toy_geometry.map(|geometry| self.dressed(geometry))
    }
}

#[derive(Clone, Default)]
pub(crate) struct Catalog {
    cards: Arc<[Card]>,
}

loam::runtime::stores! {
    #[derive(Default)]
    pub struct Playground {
        slots: Store<Slot>,
        walls: Store<Wall>,
        toys: Store<Toy>,
        hidden: Store<HiddenSlot>,
        mode: Value<Mode>,
        spin: Value<Spin>,
        active: Value<usize>,
        slice: Value<f32>,
        projection: Value<Family>,
        gimbal: Value<bool>,
        hud: Value<bool>,
        color: Value<ColorMode>,
        strip: Value<Strip>,
        environment: Value<loam::app::environment::Environment>,
        display: Value<Display>,
        time: Value<f32>,
        angles: Value<[f32; 6]>,
        controls: Value<bool>,
        formula: Value<bool>,
        camera: Value<camera::Camera>,
        control: Value<Control>,
        catalog: Value<Catalog>,
    }
}

#[derive(Clone, Copy)]
pub(crate) enum Action {
    Mode(Mode),
    Active(usize),
    Slice(f32),
    ExactSlice(f32),
    Plane(usize),
    Gimbal(Option<bool>),
    Running(Option<bool>),
    Projection(Family),
    NextProjection,
    Color(ColorMode),
    NextColor,
    Hud(Option<bool>),
    Shape(usize, usize),
    AddShape(usize),
    RemoveShape(usize),
    ReorderShape { from: usize, to: usize },
    Controls(Option<bool>),
    Formula(Option<bool>),
    Strip(Strip),
    Rate(f32),
    Display(Display),
    Reset,
    Time(f32),
    PlaneAngle(usize, f32),
    Throw(Entity, [f32; 3]),
}

pub(crate) struct Boot {
    pub(crate) session: Session<Playground>,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

fn rest_of(index: usize, len: usize) -> Vec4 {
    let center = (len.max(1) - 1) as f32 * 0.5;
    Vec4::new((index as f32 - center) * BODY_X_SPACING, BODY_Y, 0.0, 0.0)
}

pub(crate) fn boot(row: &[ShapeEntry]) -> Result<Boot, HostError> {
    let mut session = Session::new(
        Playground {
            slots: loam::runtime::Store::tracked(LogCapacity::default()),
            ..Playground::default()
        },
        SimConfig::default(),
    );
    let domain = session.register_domain(
        DomainBuilder::new(DOMAIN_NAME, EuclideanR4)
            .tracked(LogCapacity::default())
            .physics(toy::physics_config())
            .map_err(|error| HostError::Setup(Rejection::Edit(error)))?,
    );
    let root = session.views().root();

    let layers = session.dispatch(|d| -> Result<Layers, Rejection> {
        let cut = d.add_material(Material::lines(SECTION_COLOR, SECTION_WIDTH_PX));
        let cards: Arc<[Card]> = prepare_catalog(d, cut).into();
        for (index, entry) in row.iter().enumerate() {
            let rest = rest_of(index, row.len());
            let mut bundle = SpawnBundle::new().at(domain, Pose::at(rest)).row(Slot {
                index,
                entry: *entry,
                rest,
            });
            if let Some(instance) = card_of(entry).and_then(|card| cards[card].body()) {
                bundle = bundle.instance(instance);
            }
            d.spawn(bundle)?;
        }
        let eye = d.spawn(SpawnBundle::new().at(domain, Pose::at(Vec4::ZERO)))?;
        let r4 = d.domains.typed(domain)?;
        let section = r4.add_view(ViewSpec::new(root, eye, Section4 { w: 0.0 }))?;
        let projection = r4.add_view(ViewSpec::new(root, eye, Family::default().mapping(0.0)))?;
        d.app.catalog.set(Catalog { cards });
        Ok(Layers {
            section,
            projection,
        })
    })?;
    session.views_mut().root_mut().eye =
        Eye::looking_at([0.0, 3.0, 9.0], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
    session.app.controls.set(true);

    let cards = session.app.catalog.get().cards.clone();
    install_systems(&mut session, domain, layers, cards);
    session.set_initial()?;
    Ok(Boot { session, domain })
}

#[derive(Clone, Copy)]
struct Layers {
    section: ViewId,
    projection: ViewId,
}

fn card_of(entry: &ShapeEntry) -> Option<usize> {
    catalog::SHAPE_CATALOG.iter().position(|held| held == entry)
}

fn prepare_catalog(dispatch: &mut Dispatch<'_, Playground>, cut: MaterialId) -> Vec<Card> {
    catalog::SHAPE_CATALOG
        .iter()
        .map(|entry| {
            let [r, g, b] = entry.body_color;
            let material = dispatch.add_material(Material::lines([r, g, b, 1.0], EDGE_WIDTH_PX));
            let polytope = entry.shape.polytope4();
            Card {
                geometry: polytope.map(|polytope| {
                    dispatch.prepare(PreparedGeometry::Polytope4 {
                        polytope,
                        scale: BODY_SIZE,
                    })
                }),
                toy_geometry: polytope.map(|polytope| {
                    dispatch.prepare(PreparedGeometry::Polytope4 {
                        polytope,
                        scale: toy::BODY_SIZE,
                    })
                }),
                material,
                cut,
                shades: polytope.map(|polytope| {
                    let topology = polytope.topology();
                    Shades {
                        gradient: dispatch.add_palette(color::vertex_gradient_colors(topology)),
                        unique: dispatch.add_palette(color::unique_edge_colors(topology.edges)),
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
    cards: Arc<[Card]>,
) {
    session.system(
        Phase::Dispatch,
        "controls",
        move |mut ctx: Ctx<'_, Playground>| {
            control_camera(&mut ctx, domain);
            control_primary(&mut ctx, domain);
            if ctx.input.pressed(SPIN) && *ctx.app.mode.get() == Mode::Rotate {
                let running = ctx.app.spin.get().running;
                ctx.commands.app(Action::Running(Some(!running)));
            }
            if ctx.input.pressed(NEXT_MODE) {
                let next = match *ctx.app.mode.get() {
                    Mode::Rotate => Mode::Toybox,
                    Mode::Toybox => Mode::Rotate,
                };
                ctx.commands.app(Action::Mode(next));
            }
            if ctx.input.pressed(RESET) {
                ctx.commands.app(Action::Reset);
            }
            if ctx.input.pressed(GIMBAL) {
                ctx.commands.app(Action::Gimbal(None));
            }
            if ctx.input.pressed(CONTROLS) {
                ctx.commands.app(Action::Controls(None));
            }
            if ctx.input.pressed(STRIP) {
                let mut strip = *ctx.app.strip.get();
                strip.on = !strip.on;
                ctx.commands.app(Action::Strip(strip));
            }
            for (index, action) in PLANE.into_iter().enumerate() {
                if ctx.input.pressed(action) {
                    ctx.commands.app(Action::Plane(index));
                }
            }
            Ok(())
        },
    );

    let mut applied = None;
    session.system(
        Phase::Publication,
        "slice view",
        move |ctx: Ctx<'_, Playground>| -> Result<(), DomainError> {
            let w = *ctx.app.slice.get();
            let revision = ctx.app.slice.version();
            if Some(revision) == applied {
                return Ok(());
            }
            let r4 = ctx.domains.typed(domain)?;
            let spec = r4
                .view_mut(layers.section)
                .ok_or(DomainError::Unsupported("section view"))?;
            spec.set_mapping(Section4 { w });
            applied = Some(revision);
            Ok(())
        },
    );

    let mut styled = None;
    let mut styled_slots = loam::runtime::Cursor::default();
    session.system(
        Phase::Publication,
        "shading",
        move |ctx: Ctx<'_, Playground>| -> Result<(), DomainError> {
            let app = &mut *ctx.app;
            let revision = (
                app.color.version(),
                app.strip.version(),
                app.display.version(),
                app.active.version(),
            );
            let slots_changed = app.slots.changed_since(&mut styled_slots);
            if !slots_changed && styled == Some(revision) {
                return Ok(());
            }
            let mode = *app.color.get();
            let strip = app.strip.get().on;
            let display = *app.display.get();
            let active = *app.active.get();
            let subject = if display.single {
                app.slots
                    .iter()
                    .find(|(_, slot)| slot.index == active)
                    .map(|(entity, _)| entity)
            } else {
                None
            };
            let r4 = ctx.domains.typed(domain)?;
            r4.set_view_subject(layers.section, subject)?;
            r4.set_view_subject(layers.projection, subject)?;
            let spec = r4
                .view_mut(layers.section)
                .ok_or(DomainError::Unsupported("section view"))?;
            spec.enabled = !strip;
            spec.edges = false;
            spec.section_edges = display.wireframe && display.section_perimeter;
            spec.section_faces = display.surface == Surface::Raster;
            let spec = r4
                .view_mut(layers.projection)
                .ok_or(DomainError::Unsupported("projection view"))?;
            spec.enabled = display.wireframe && !strip;
            spec.section_edges = false;
            spec.section_faces = false;
            for (entity, slot) in app.slots.iter() {
                let Some(card) = card_of(&slot.entry).map(|index| cards[index]) else {
                    continue;
                };
                let Some(shades) = card.shades else {
                    continue;
                };
                if let Some(instance) = r4.instance_mut(entity) {
                    instance.shading = shades.of(mode);
                    instance.section = (!strip).then_some(card.cut);
                    instance.line_width_px = Some(display.wireframe_width_px);
                    instance.line_opacity = Some(display.wireframe_opacity);
                }
            }
            styled = Some(revision);
            app.slots.catch_up(&mut styled_slots);
            Ok(())
        },
    );

    let mut shown = None;
    session.system(
        Phase::Publication,
        "projection view",
        move |ctx: Ctx<'_, Playground>| -> Result<(), DomainError> {
            let family = *ctx.app.projection.get();
            let slice = *ctx.app.slice.get();
            let revision = (ctx.app.projection.version(), ctx.app.slice.version());
            if shown == Some(revision) {
                return Ok(());
            }
            let r4 = ctx.domains.typed(domain)?;
            let spec = r4
                .view_mut(layers.projection)
                .ok_or(DomainError::Unsupported("projection view"))?;
            spec.set_mapping(family.mapping(slice));
            shown = Some(revision);
            Ok(())
        },
    );

    session.system(
        Phase::Simulation,
        "slice scrub",
        |ctx: Ctx<'_, Playground>| {
            if ctx.app.camera.get().mode != camera::CameraMode::Orbit {
                return Ok(());
            }
            let scrub =
                f32::from(ctx.input.is_held(SLICE_UP)) - f32::from(ctx.input.is_held(SLICE_DOWN));
            if scrub == 0.0 {
                return Ok(());
            }
            let next = *ctx.app.slice.get() + scrub * W_SCRUB_RATE * ctx.step.dt;
            let next = match *ctx.app.mode.get() {
                Mode::Rotate => next.clamp(-consts::W_RANGE, consts::W_RANGE),
                Mode::Toybox => next,
            };
            ctx.app.slice.set(next);
            Ok(())
        },
    );
    session.system(
        Phase::Simulation,
        "spin",
        move |ctx: Ctx<'_, Playground>| -> Result<(), DomainError> {
            if !ctx.app.spin.get().running || *ctx.app.mode.get() == Mode::Toybox {
                return Ok(());
            }
            let time = *ctx.app.time.get() + ctx.step.dt * ctx.app.spin.get().rate;
            display::seek(ctx.app, ctx.domains, domain, time)?;
            Ok(())
        },
    );
    session.system(
        Phase::Simulation,
        "free camera",
        |ctx: Ctx<'_, Playground>| {
            if ctx.app.camera.get().mode == camera::CameraMode::Freecam && ctx.input.cursor_locked {
                let axes = [
                    f32::from(ctx.input.is_held(camera::RIGHT))
                        - f32::from(ctx.input.is_held(camera::LEFT)),
                    f32::from(ctx.input.is_held(SLICE_UP))
                        - f32::from(ctx.input.is_held(SLICE_DOWN)),
                    f32::from(ctx.input.is_held(camera::FORWARD))
                        - f32::from(ctx.input.is_held(camera::BACKWARD)),
                ];
                let camera = ctx.app.camera.get_mut();
                camera.free.travel(axes, camera.speed * ctx.step.dt);
                look(ctx.views, camera.free.eye);
            }
            Ok(())
        },
    );
    session.system(
        Phase::Simulation,
        "toy settle",
        move |ctx: Ctx<'_, Playground>| -> Result<(), DomainError> {
            toy::settle(ctx.app, ctx.domains, domain, ctx.step)
        },
    );
}

impl AppCommand<Playground> for Action {
    fn name(&self) -> &'static str {
        match self {
            Self::Mode(mode) => mode.name(),
            Self::Active(_) => "active",
            Self::Slice(_) | Self::ExactSlice(_) => "slice",
            Self::Plane(_) => "plane",
            Self::Gimbal(None) => "gimbal",
            Self::Gimbal(Some(_)) => "handles",
            Self::Running(_) => "spin",
            Self::Projection(family) => family.name(),
            Self::NextProjection => "wireframe perspective",
            Self::Color(mode) => mode.name(),
            Self::NextColor => "wireframe color",
            Self::Hud(_) => "hud",
            Self::Shape(_, card) => catalog::SHAPE_CATALOG
                .get(*card)
                .map_or("shape", |entry| entry.label),
            Self::AddShape(_) => "add shape",
            Self::RemoveShape(_) => "remove shape",
            Self::ReorderShape { .. } => "reorder shape",
            Self::Controls(_) => "controls",
            Self::Formula(_) => "formula",
            Self::Strip(_) => "strip",
            Self::Rate(_) => "rate",
            Self::Display(_) => "display",
            Self::Reset => "reset",
            Self::Time(_) => "time",
            Self::PlaneAngle(_, _) => "plane angle",
            Self::Throw(_, _) => "throw toy",
        }
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let domain = dispatch.domains.named::<EuclideanR4>(DOMAIN_NAME)?;
        let Catalog { cards } = dispatch.app.catalog.get().clone();
        match *self {
            Action::Mode(mode) => mode::set_mode(dispatch, domain, &cards, mode)?,
            Action::Active(slot) => {
                if slot >= dispatch.app.slots.len() {
                    return Err(Rejection::Unsupported("no such slot"));
                }
                dispatch.app.active.set(slot);
            }
            Action::ReorderShape { from, to } => {
                row::reorder_shape(dispatch, domain, from, to)?;
            }
            Action::Slice(w) => {
                if !w.is_finite() {
                    return Err(Rejection::Unsupported("slice is not finite"));
                }
                let w = if *dispatch.app.mode.get() == Mode::Toybox {
                    w
                } else {
                    w.clamp(-consts::W_RANGE, consts::W_RANGE)
                };
                dispatch.app.slice.set(w);
            }
            Action::ExactSlice(w) => {
                if *dispatch.app.mode.get() == Mode::Rotate
                    && !(-consts::W_RANGE..=consts::W_RANGE).contains(&w)
                {
                    return Err(Rejection::Unsupported("slice is outside the rotate range"));
                }
                if !w.is_finite() {
                    return Err(Rejection::Unsupported("slice is not finite"));
                }
                dispatch.app.slice.set(w);
            }
            Action::Plane(plane) => mode::toggle_plane(dispatch, plane)?,
            Action::Gimbal(setting) => {
                let shown = *dispatch.app.gimbal.get();
                dispatch.app.gimbal.set(setting.unwrap_or(!shown));
            }
            Action::Running(setting) => {
                if *dispatch.app.mode.get() == Mode::Toybox {
                    return Err(Rejection::Unsupported("rotation belongs to Rotate"));
                }
                let running = dispatch.app.spin.get_mut();
                running.running = setting.unwrap_or(!running.running);
            }
            Action::Projection(family) => {
                dispatch.app.projection.set(family);
                if family == Family::Stereographic {
                    dispatch.app.display.get_mut().wireframe = true;
                }
            }
            Action::NextProjection => {
                let held = *dispatch.app.projection.get();
                let at = Family::ALL
                    .iter()
                    .position(|family| *family == held)
                    .unwrap_or(0);
                let family = Family::ALL[(at + 1) % Family::ALL.len()];
                dispatch.app.projection.set(family);
                if family == Family::Stereographic {
                    dispatch.app.display.get_mut().wireframe = true;
                }
            }
            Action::Color(mode) => dispatch.app.color.set(mode),
            Action::NextColor => {
                let held = *dispatch.app.color.get();
                let at = ColorMode::ALL
                    .iter()
                    .position(|mode| *mode == held)
                    .unwrap_or(0);
                dispatch
                    .app
                    .color
                    .set(ColorMode::ALL[(at + 1) % ColorMode::ALL.len()]);
            }
            Action::Hud(setting) => {
                let shown = *dispatch.app.hud.get();
                dispatch.app.hud.set(setting.unwrap_or(!shown));
            }
            Action::Strip(strip) => mode::set_strip(dispatch.app, strip)?,
            Action::Rate(rate) => {
                if !rate.is_finite() {
                    return Err(Rejection::Unsupported("the rate is not finite"));
                }
                dispatch.app.spin.get_mut().rate = rate.clamp(0.0, consts::MAX_RATE);
            }
            Action::Display(display) => display::set(dispatch.app, display)?,
            Action::Reset => mode::reset(dispatch, domain, &cards)?,
            Action::Controls(setting) => {
                let visible = *dispatch.app.controls.get();
                dispatch.app.controls.set(setting.unwrap_or(!visible));
            }
            Action::Formula(setting) => {
                let visible = *dispatch.app.formula.get();
                dispatch.app.formula.set(setting.unwrap_or(!visible));
            }
            Action::Time(time) => {
                display::seek(dispatch.app, dispatch.domains, domain, time)
                    .map_err(Rejection::Domain)?;
            }
            Action::PlaneAngle(plane, angle) => {
                display::set_plane_angle(dispatch, domain, plane, angle)?;
            }
            Action::Shape(slot, card) => {
                let Some(entry) = catalog::SHAPE_CATALOG.get(card) else {
                    return Ok(Outcome::Done);
                };
                mode::set_shape(dispatch, domain, slot, *entry, cards[card])?;
            }
            Action::AddShape(card) => {
                let Some(entry) = catalog::SHAPE_CATALOG.get(card) else {
                    return Ok(Outcome::Done);
                };
                row::add_shape(dispatch, domain, *entry, cards[card])?;
            }
            Action::RemoveShape(slot) => row::remove_shape(dispatch, domain, slot)?,
            Action::Throw(entity, velocity) => {
                mode::throw_toy(dispatch, domain, entity, velocity)?;
            }
        }
        Ok(Outcome::Done)
    }
}

pub(crate) fn bindings() -> Bindings {
    let mut bound = Bindings::new()
        .key(Key::Space, SPIN)
        .key(Key::Letter('t'), SPIN)
        .key(Key::Letter('m'), NEXT_MODE)
        .key(Key::Letter('r'), RESET)
        .key(Key::Letter('g'), GIMBAL)
        .key(Key::Letter('f'), STRIP)
        .key(Key::Letter('h'), CONTROLS)
        .key(Key::Letter('d'), camera::RIGHT)
        .key(Key::Letter('a'), camera::LEFT)
        .key(Key::Letter('w'), camera::FORWARD)
        .key(Key::Letter('s'), camera::BACKWARD)
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
    rings: LinePass,
    guides: LinePass,
    grab_point: PointPass,
    hud: loam::text::TextPass,
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self {
            sky: SkyGroundPass::new(scene::ground(true)),
            hyperslice: HyperslicePass::new(scene::shader_source()),
            rings: LinePass::new("gimbal"),
            guides: LinePass::new("toybox-guides"),
            grab_point: PointPass::new("grab-point"),
            hud: hud::pass(),
        }
    }

    fn passes(&self) -> Vec<Box<dyn FramePass>> {
        vec![
            Box::new(self.sky.clone()),
            Box::new(self.hyperslice.clone()),
            Box::new(self.rings.clone()),
            Box::new(self.guides.clone()),
            Box::new(self.grab_point.clone()),
            Box::new(self.hud.clone()),
        ]
    }
}

struct Scratch {
    center: glam::Vec3,
    slots: Vec<(Entity, ShapeEntry, f32)>,
    bodies: Vec<BodyUniform>,
    cells: Vec<Cell>,
    strip: Vec<(loam::render::Viewport, f32, BodyUniform)>,
    subject: loam::math::Rotor4,
    anchors: Vec<(usize, &'static str, glam::Vec3)>,
    depths: Vec<toy::DepthBand>,
}

impl Scratch {
    fn new() -> Self {
        Self {
            center: glam::Vec3::ZERO,
            slots: Vec::new(),
            bodies: Vec::new(),
            cells: Vec::new(),
            strip: Vec::new(),
            subject: loam::math::Rotor4::IDENTITY,
            anchors: Vec::new(),
            depths: Vec::with_capacity(5),
        }
    }
}

fn collect(
    session: &Session<Playground>,
    domain: DomainHandle<EuclideanR4>,
    scratch: &mut Scratch,
) {
    let active = *session.app.active.get();
    let display = *session.app.display.get();
    scratch.slots.clear();
    scratch.slots.extend(
        session
            .app
            .slots
            .iter()
            .filter(|(_, slot)| !display.single || slot.index == active)
            .map(|(entity, slot)| {
                let size = if session.app.toys.get(entity).is_some() {
                    toy::BODY_SIZE
                } else {
                    BODY_SIZE
                };
                (entity, slot.entry, size)
            }),
    );
    scratch.anchors.clear();
    scratch.anchors.extend(
        session
            .app
            .slots
            .iter()
            .filter(|(_, slot)| !display.single || slot.index == active)
            .map(|(_, slot)| (slot.index, slot.entry.label, glam::Vec3::ZERO)),
    );
    scratch.bodies.clear();
    scratch.subject = loam::math::Rotor4::IDENTITY;
    let mut sum = glam::Vec3::ZERO;
    let Ok(r4) = session.domains().read(domain) else {
        return;
    };
    for (at, (entity, entry, size)) in scratch.slots.iter().enumerate() {
        let Some(pose) = r4.poses().get(*entity) else {
            continue;
        };
        if scratch.anchors.get(at).is_some_and(|held| held.0 == active) {
            scratch.subject = pose.frame;
        }
        let center = pose.point.truncate();
        sum += center;
        if let Some(anchor) = scratch.anchors.get_mut(at) {
            anchor.2 = center;
        }
        if display.surface == Surface::Sdf
            || (display.surface == Surface::Raster && entry.shape.polytope4().is_none())
        {
            scratch.bodies.push(scene::body_of(entry, pose, *size));
        }
    }
    scratch.center = sum / scratch.slots.len().max(1) as f32;
}

fn main() -> Result<(), HostError> {
    launch_or_headless(interactive, headless)
}

fn interactive(args: Args) -> Result<(Session<Playground>, SessionApp<Playground>), HostError> {
    let row = catalog::parse_row(&args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let booted = boot(&row)?;
    let frame = Frame::new();
    let config = HostConfig::new("polytope playground", bindings());

    let sky = frame.sky.clone();
    let hyperslice = frame.hyperslice.clone();
    let rings = frame.rings.clone();
    let guide_pass = frame.guides.clone();
    let grab_point = frame.grab_point.clone();
    let mut guides = guides::Guides::default();
    let readout = frame.hud.clone();
    let mut lines = String::new();
    let mut gimbal_renderer = gimbal::GimbalRenderer::default();
    let domain = booted.domain;
    let mut scratch = Scratch::new();
    let mut panel = ui::Panel::default();
    let mut app = SessionApp::with_args(config, args).recover_on_fault(RESET);
    for pass in frame.passes() {
        app = app.pass(pass);
    }
    app = console::install(app).on_frame(move |hook: &mut FrameHook<'_, Playground>| {
        let camera = *hook.session.app.camera.get();
        hook.capture_cursor(
            camera.mode == camera::CameraMode::Freecam
                && !hook
                    .ui
                    .is_some_and(|context| context.wants_keyboard_input()),
            camera.cursor_policy,
        );
        collect(hook.session, domain, &mut scratch);
        let eye = hook
            .session
            .views()
            .get(hook.session.views().root())
            .map_or(Eye::default(), |root| root.eye);
        let strip = *hook.session.app.strip.get();

        let slice = *hook.session.app.slice.get();
        let environment = *hook.session.app.environment.get();
        let floor = environment.floor_visible;
        sky.publish(&eye, environment.ground(consts::FLOOR_Y, floor));
        hyperslice.set_enabled(strip.on || !scratch.bodies.is_empty());
        hyperslice.publish(scene::uniforms(&eye, slice, floor), &scratch.bodies);
        fill_strip(
            &strip,
            turn_of(hook.session),
            hook.size,
            slice,
            &mut scratch,
        );
        hyperslice.publish_strip(&scratch.strip);
        rings.publish(
            &eye,
            gimbal_renderer.rings(&hook.session.app.control.get().gimbal, scratch.center),
        );
        guides.update(hook.session, domain);
        guide_pass.publish(&eye, &guides.lines);
        grab_point.publish(&eye, &guides.points);
        let shown = *hook.session.app.hud.get();
        let seat = hook.ui.map_or(hud::Seat::default(), |context| {
            hud::Seat::in_panel(context.available_rect(), context.pixels_per_point())
        });
        hud::publish(
            &readout,
            shown.then(|| readout_of(hook.session)).as_ref(),
            seat,
            &mut lines,
        );
        if let Some(context) = hook.ui {
            toy::fill_depth_bands(hook.session, domain, &mut scratch.depths);
            ui::draw(
                context,
                hook.session,
                &mut panel,
                hook.sender,
                scratch.subject,
                &scratch.depths,
            );
            if panel.show_callouts && !strip.on {
                ui::callouts(context, hook.session, &scratch.anchors);
            }
            ui::strip_labels(context, hook.session, &scratch.cells);
        }
    });
    Ok((booted.session, app))
}

fn focus_camera(
    orbit: &mut Orbit,
    camera_focus: &mut Option<(bool, usize, Mode)>,
    filmstrip_return: &mut Option<Orbit>,
    strip: Strip,
    focus: (bool, usize, Mode),
    center: glam::Vec3,
) {
    if strip.on {
        if filmstrip_return.is_none() {
            let mut centered = Orbit::around([0.0; 3], orbit.distance);
            centered.yaw = orbit.yaw;
            centered.pitch = orbit.pitch;
            *filmstrip_return = Some(std::mem::replace(orbit, centered));
            *camera_focus = Some(focus);
        }
        orbit.target[1] = if strip.w && strip.t { BODY_Y } else { 0.0 };
        return;
    }
    if let Some(returned) = filmstrip_return.take() {
        *orbit = returned;
    }
    if *camera_focus == Some(focus) {
        return;
    }
    let reset =
        camera_focus.is_some_and(|previous| previous.2 != focus.2 || previous.0 && !focus.0);
    let target = [center.x, 0.0, center.z];
    if reset {
        *orbit = Orbit::around(target, CAMERA_DISTANCE);
        orbit.pitch = -0.25;
    } else {
        orbit.target = target;
    }
    *camera_focus = Some(focus);
}

fn readout_of(session: &Session<Playground>) -> hud::Readout {
    hud::Readout {
        slice: *session.app.slice.get(),
        rate: session.app.spin.get().rate,
        bodies: session.app.slots.len(),
        planes: session.app.spin.get().planes,
    }
}

fn turn_of(session: &Session<Playground>) -> Bivector4 {
    match *session.app.mode.get() {
        Mode::Rotate => session.app.spin.get().omega(),
        Mode::Toybox => Bivector4::ZERO,
    }
}

fn fill_strip(
    strip: &Strip,
    omega: Bivector4,
    frame: (u32, u32),
    slice: f32,
    scratch: &mut Scratch,
) {
    scratch.strip.clear();
    if !strip.on {
        scratch.cells.clear();
        return;
    }
    strip.cells([frame.0, frame.1], slice, BODY_SIZE, &mut scratch.cells);
    let entry = catalog::SHAPE_CATALOG[strip.subject()];
    for cell in &scratch.cells {
        let rotor = ((omega * cell.t).exp() * scratch.subject).normalize();
        scratch.strip.push((
            cell.viewport,
            cell.w,
            BodyUniform::polytope_with_rotor(
                [0.0, BODY_Y, 0.0, 0.0],
                entry.shape.shape_id(),
                BODY_SIZE,
                rotor,
                entry.body_color,
            ),
        ));
    }
}

fn scene_center(
    app: &Playground,
    domains: &Domains,
    domain: DomainHandle<EuclideanR4>,
) -> glam::Vec3 {
    let active = *app.active.get();
    let single = app.display.get().single;
    let Ok(r4) = domains.read(domain) else {
        return glam::Vec3::ZERO;
    };
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0;
    for (entity, slot) in app.slots.iter() {
        if single && slot.index != active {
            continue;
        }
        let Some(pose) = r4.poses().get(entity) else {
            continue;
        };
        sum += pose.point.truncate();
        count += 1;
    }
    sum / count.max(1) as f32
}

fn control_camera(ctx: &mut Ctx<'_, Playground>, domain: DomainHandle<EuclideanR4>) {
    let center = scene_center(ctx.app, ctx.domains, domain);
    let focus = (
        ctx.app.display.get().single,
        *ctx.app.active.get(),
        *ctx.app.mode.get(),
    );
    let strip = *ctx.app.strip.get();
    {
        let control = ctx.app.control.get_mut();
        for pointer in &ctx.input.pointers {
            control.latest = Some(*pointer);
        }
        control.center = center;
        focus_camera(
            &mut control.orbit,
            &mut control.camera_focus,
            &mut control.filmstrip_return,
            strip,
            focus,
            center,
        );
    }
    let mode = ctx.app.camera.get().mode;
    let eye = match mode {
        camera::CameraMode::Orbit => {
            let control = ctx.app.control.get_mut();
            control.orbit.apply(ctx.input);
            control.orbit.eye()
        }
        camera::CameraMode::Freecam => {
            let camera = ctx.app.camera.get_mut();
            if ctx.input.cursor_locked {
                camera.free.look(ctx.input.look);
            }
            camera.free.eye
        }
    };
    look(ctx.views, eye);
}

fn control_primary(ctx: &mut Ctx<'_, Playground>, domain: DomainHandle<EuclideanR4>) {
    if ctx.dragging().is_some_and(|drag| {
        *ctx.app.mode.get() != Mode::Toybox || !ctx.app.toys.contains(drag.entity)
    }) {
        ctx.cancel_drag();
    }
    let (latest, center, mut gimbal) = {
        let control = ctx.app.control.get();
        (control.latest, control.center, control.gimbal)
    };
    gimbal.enabled =
        *ctx.app.gimbal.get() && *ctx.app.mode.get() == Mode::Rotate && !ctx.app.strip.get().on;
    let root = ctx.views.root();
    let ray = latest.and_then(|pointer| ctx.views.ray(root, pointer.ndc));
    if gimbal.enabled {
        gimbal.aim(ray.as_ref(), center);
    } else {
        gimbal.release();
    }
    for index in 0..ctx.input.pointers.len() {
        let pointer = ctx.input.pointers[index];
        if pointer.button != Some(PointerButton::Primary) {
            continue;
        }
        match pointer.phase {
            PointerPhase::Began => {
                let ring = gimbal.enabled
                    && ctx
                        .views
                        .ray(root, pointer.ndc)
                        .is_some_and(|ray| gimbal.press(&ray, center));
                if !ring && *ctx.app.mode.get() == Mode::Toybox && !ctx.app.strip.get().on {
                    let _ = ctx.grab(pointer.ndc, pointer.time);
                }
            }
            PointerPhase::Moved if gimbal.held() => {
                let turn = ctx
                    .views
                    .ray(root, pointer.ndc)
                    .and_then(|ray| gimbal.turn(&ray));
                if let Some(rotor) = turn {
                    let _ = mode::turn_row(ctx.app, ctx.domains, domain, rotor);
                }
            }
            PointerPhase::Moved
                if ctx.dragging().is_some() && ctx.drag(pointer.ndc, pointer.time).is_err() =>
            {
                ctx.cancel_drag();
            }
            PointerPhase::Ended if gimbal.held() => gimbal.release(),
            PointerPhase::Ended if ctx.dragging().is_some() => {
                if let Some(release) = ctx.release_at(pointer.time) {
                    ctx.commands
                        .app(Action::Throw(release.entity, release.velocity));
                }
            }
            PointerPhase::Cancelled => {
                gimbal.release();
                ctx.cancel_drag();
            }
            _ => {}
        }
    }
    ctx.app.control.get_mut().gimbal = gimbal;
}

fn report(booted: &mut Boot, config: &HostConfig) -> Result<Vec<String>, HostError> {
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
        format!(
            "section fills: {} triangles",
            publication
                .views
                .iter()
                .map(|view| view.records.triangles().len())
                .sum::<usize>()
        ),
        strip_line(&mut Scratch::new(), *booted.session.app.strip.get()),
    ])
}

fn strip_line(scratch: &mut Scratch, strip: Strip) -> String {
    let mut shown = strip;
    shown.on = true;
    fill_strip(&shown, Bivector4::ZERO, HEADLESS_FRAME, 0.0, scratch);
    let (cols, rows, _) = shown.grid();
    format!("filmstrip: {} cells, {cols} by {rows}", scratch.strip.len())
}

fn headless(args: Args) -> Result<(), HostError> {
    let row = catalog::parse_row(&args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let mut booted = boot(&row)?;
    let config = HostConfig::new("polytope playground", bindings());
    for line in report(&mut booted, &config)? {
        println!("{line}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use loam::math::Rotor;
    use loam::render::raymarch::RaymarchShape;
    use loam::runtime::{AppCommand, ChartCommand, ChartPoint, Command, Records};
    use loam::shape::polytope::Polytope4;

    use super::*;

    const CELL24: ShapeEntry = ShapeEntry {
        shape: RaymarchShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
        category: catalog::Category::RegularPolychoron,
    };

    const EYE_BACK: f32 = 5.0;

    fn one_slot() -> Boot {
        let mut booted = boot(&[CELL24]).expect("the session boots");
        booted.session.views_mut().root_mut().eye =
            Eye::looking_at([0.0, BODY_Y, EYE_BACK], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
        booted.session.app.spin.get_mut().running = false;
        booted
    }

    pub(crate) fn send(booted: &mut Boot, action: Action) {
        booted.session.submit(Command::App(Box::new(action)));
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
        records: &mut Records,
        read: impl FnOnce(&loam::runtime::Publication) -> R,
    ) -> R {
        records.publish(session).expect("published");
        let publication = records.lend().expect("the buffer is free");
        let value = read(&publication);
        records.release(publication);
        value
    }

    #[test]
    fn a_reset_before_any_action_leaves_every_action_working() {
        let mut booted = one_slot();
        booted.session.submit(Command::Reset);
        booted
            .session
            .boundary(Input::default())
            .expect("the reset applied");
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");

        assert_eq!(*booted.session.app.mode.get(), Mode::Toybox);
        assert!(
            booted.session.results()[0].outcome.is_ok(),
            "the first action after a reset was refused: {:?}",
            booted.session.results()[0].outcome
        );
    }

    #[test]
    fn spin_is_refused_in_toybox_instead_of_reporting_done() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        send(&mut booted, Action::Running(Some(true)));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let outcome = &booted.session.results()[0].outcome;
        assert!(
            matches!(outcome, Err(Rejection::Unsupported(_))),
            "spin in Toybox reported {outcome:?}"
        );
    }

    #[test]
    fn a_mode_command_lands_in_the_store_at_the_next_boundary() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
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
    fn wireframe_and_perimeter_toggles_control_separate_edge_layers() {
        let mut booted = one_slot();
        booted.session.boundary(Input::default()).expect("boundary");
        let mut records = Records::default();
        for reset in [false, true] {
            if reset {
                send(&mut booted, Action::Reset);
                booted.session.boundary(Input::default()).expect("reset");
            }
            published(&mut booted.session, &mut records, |publication| {
                assert!(publication
                    .views
                    .iter()
                    .all(|view| view.records.segments().is_empty()));
                assert!(!publication.views[0].records.triangles().is_empty());
            });
        }
        send(
            &mut booted,
            Action::Display(Display {
                wireframe: true,
                ..Display::default()
            }),
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let counts = published(&mut booted.session, &mut records, |publication| {
            publication
                .views
                .iter()
                .map(|view| view.records.segments().len())
                .collect::<Vec<_>>()
        });
        assert!(counts[0] > 0);
        assert_eq!(counts[1], 96);
        let mut display = *booted.session.app.display.get();
        display.section_perimeter = false;
        send(&mut booted, Action::Display(display));
        booted
            .session
            .boundary(Input::default())
            .expect("perimeter off");
        published(&mut booted.session, &mut records, |publication| {
            assert!(publication.views[0].records.segments().is_empty());
            assert_eq!(publication.views[1].records.segments().len(), 96);
        });
    }

    #[test]
    fn an_unmoved_row_under_an_unchanged_view_is_not_republished() {
        let mut booted = one_slot();
        let mut records = Records::default();
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let built = |session: &mut Session<Playground>, records: &mut Records| {
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

        send(&mut booted, Action::Slice(0.4));
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
    fn toybox_mode_parks_the_rotation_row_and_spawns_five_floor_cleared_toys() {
        let mut booted = one_slot();
        let original = slot_entity(&booted);
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        assert!(booted.session.app.hidden.contains(original));
        assert!(!booted.session.app.slots.contains(original));
        assert_eq!(booted.session.app.slots.len(), 5);
        assert_eq!(booted.session.app.toys.len(), 5);
        assert_eq!(booted.session.app.walls.len(), 5);
        let expected = [
            Polytope4::Cell24,
            Polytope4::Tesseract,
            Polytope4::Pentatope,
            Polytope4::Cell16,
            Polytope4::Tesseract,
        ];
        let slots: Vec<(Entity, Slot)> = booted
            .session
            .app
            .slots
            .iter()
            .map(|(entity, slot)| (entity, *slot))
            .collect();
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        for &(entity, slot) in &slots {
            assert_eq!(slot.entry.collider_polytope(), Some(expected[slot.index]));
            let pose = r4.poses().get(entity).expect("the toy has a pose");
            let polytope = expected[slot.index];
            let lowest = polytope
                .topology()
                .vertices
                .iter()
                .map(|vertex| (pose.point + pose.frame.apply(*vertex * toy::BODY_SIZE)).y)
                .fold(f32::INFINITY, f32::min);
            assert!((lowest - consts::FLOOR_Y - 0.20).abs() < 1e-5);
            assert!(r4
                .physics()
                .and_then(|physics| physics.body(entity))
                .is_some());
        }
        for _ in 0..600 {
            booted.session.tick().expect("the tick ran");
        }
        let toys: Vec<Entity> = booted
            .session
            .app
            .toys
            .iter()
            .map(|(entity, _)| entity)
            .collect();
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let physics = r4.physics().expect("the domain has physics");
        assert!(toys.iter().all(|entity| physics
            .body(*entity)
            .and_then(|body| physics.world().body(body))
            .is_some_and(|body| body.is_sleeping())));
        for (entity, slot) in slots {
            let body = physics
                .world()
                .body(physics.body(entity).expect("body"))
                .expect("body row");
            let lowest = slot
                .entry
                .collider_polytope()
                .expect("polytope")
                .topology()
                .vertices
                .iter()
                .map(|vertex| {
                    (body.position + body.orientation.rotation.apply(*vertex * toy::BODY_SIZE)).y
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                (consts::FLOOR_Y..=consts::FLOOR_Y + loam::physics::manifold::PENETRATION_SLOP)
                    .contains(&lowest),
                "{} rests at y={lowest}",
                slot.entry.label
            );
        }
    }

    #[test]
    fn row_edits_preserve_rotation_and_reset_keeps_authored_shapes() {
        let mut booted = one_slot();
        send(&mut booted, Action::PlaneAngle(0, 0.4));
        send(&mut booted, Action::AddShape(4));
        send(&mut booted, Action::AddShape(6));
        send(&mut booted, Action::RemoveShape(0));
        booted.session.boundary(Input::default()).expect("edit row");
        assert!(booted.session.results().iter().all(|r| r.outcome.is_ok()));
        let active = booted
            .session
            .app
            .slots
            .iter()
            .find(|(_, slot)| slot.index == *booted.session.app.active.get())
            .expect("active shape")
            .0;
        let frame = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain")
            .poses()
            .get(active)
            .expect("pose")
            .frame;
        send(&mut booted, Action::ReorderShape { from: 0, to: 2 });
        booted.session.boundary(Input::default()).expect("reorder");
        assert!(booted.session.results().iter().all(|r| r.outcome.is_ok()));
        assert_eq!(booted.session.app.active.get(), &0);
        assert_eq!(
            booted
                .session
                .app
                .slots
                .get(active)
                .expect("active shape")
                .index,
            0
        );
        let rows: Vec<_> = booted
            .session
            .app
            .slots
            .iter()
            .map(|(entity, slot)| (entity, *slot))
            .collect();
        assert_eq!(rows.len(), 2);
        let domain = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain");
        for (entity, slot) in &rows {
            assert_eq!(slot.rest, rest_of(slot.index, 2));
            let pose = domain.poses().get(*entity).expect("pose");
            assert_ne!(pose.frame, loam::math::Rotor4::IDENTITY);
            assert_eq!(pose.frame, frame);
            assert_eq!(pose.point, slot.rest);
            assert_eq!(domain.instances().contains(*entity), slot.index == 1);
        }
        send(&mut booted, Action::Reset);
        booted.session.boundary(Input::default()).expect("reset");
        assert_eq!(booted.session.app.slots.len(), 2);
        let domain = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain");
        for (entity, slot) in rows {
            let pose = domain.poses().get(entity).expect("same entity");
            assert_eq!(pose.point, slot.rest);
            assert_eq!(pose.frame, loam::math::Rotor4::IDENTITY);
        }
    }

    #[test]
    fn rotate_pointer_drag_cannot_pick_or_translate_a_shape() {
        let mut booted = one_slot();
        booted.session.boundary(Input::default()).expect("boundary");
        let entity = slot_entity(&booted);
        let before = *booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain")
            .poses()
            .get(entity)
            .expect("pose");
        for (phase, ndc) in [
            (PointerPhase::Began, [0.0, 0.0]),
            (PointerPhase::Moved, [0.5, 0.0]),
            (PointerPhase::Ended, [0.5, 0.0]),
        ] {
            let pointer = Pointer {
                id: 0,
                button: Some(PointerButton::Primary),
                ndc,
                delta: [0.5, 0.0],
                phase,
                time: 1.0,
            };
            booted
                .session
                .boundary(Input {
                    pointers: vec![pointer],
                    ..Input::default()
                })
                .expect("boundary");
        }
        assert!(booted.session.dragging().is_none());
        let after = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain")
            .poses()
            .get(entity)
            .expect("pose");
        assert_eq!(after.point, before.point);
        assert_eq!(after.frame, before.frame);
    }

    #[test]
    fn a_warmed_frame_with_every_overlay_on_asks_the_allocator_for_nothing() {
        let mut booted = one_slot();
        booted
            .session
            .dispatch(|dispatch| Action::Gimbal(None).apply(dispatch))
            .expect("handles");
        booted
            .session
            .dispatch(|dispatch| Action::Hud(None).apply(dispatch))
            .expect("readout");
        send(&mut booted, Action::Color(ColorMode::WDepth));
        send(
            &mut booted,
            Action::Strip(Strip {
                on: true,
                w: true,
                t: true,
                ..Strip::default()
            }),
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        assert!(booted.session.results().iter().all(|r| r.outcome.is_ok()));

        let mut records = Records::default();
        let mut scratch = Scratch::new();
        let mut gimbal = Gimbal::default();
        gimbal.enabled = true;
        let mut gimbal_renderer = gimbal::GimbalRenderer::default();
        let mut lines = String::new();
        let frame = |booted: &mut Boot,
                     records: &mut Records,
                     scratch: &mut Scratch,
                     gimbal_renderer: &mut gimbal::GimbalRenderer,
                     lines: &mut String| {
            booted
                .session
                .boundary(Input::default())
                .expect("the boundary ran");
            booted.session.tick().expect("the tick ran");
            records.publish(&mut booted.session).expect("published");
            let publication = records.lend().expect("the buffer is free");
            records.release(publication);
            collect(&booted.session, booted.domain, scratch);
            fill_strip(
                &{ *booted.session.app.strip.get() },
                turn_of(&booted.session),
                HEADLESS_FRAME,
                0.0,
                scratch,
            );
            gimbal_renderer.rings(&gimbal, scratch.center);
            hud::write_readout(lines, &readout_of(&booted.session));
        };
        for _ in 0..16 {
            frame(
                &mut booted,
                &mut records,
                &mut scratch,
                &mut gimbal_renderer,
                &mut lines,
            );
        }

        let bytes = loam_time::alloc::bytes_allocated_by(|| {
            for _ in 0..16 {
                frame(
                    &mut booted,
                    &mut records,
                    &mut scratch,
                    &mut gimbal_renderer,
                    &mut lines,
                );
            }
        })
        .expect("the counting allocator is installed");
        assert_eq!(
            bytes, 0,
            "sixteen warmed frames of boundary, tick, publication, collection, strip, gimbal, and readout asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn reset_restyles_replacement_toys_even_when_the_row_count_does_not_change() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        send(&mut booted, Action::Color(ColorMode::UniqueEdge));
        send(
            &mut booted,
            Action::Display(Display {
                wireframe: true,
                wireframe_width_px: 4.0,
                wireframe_opacity: 0.2,
                ..Display::default()
            }),
        );
        booted
            .session
            .boundary(Input::default())
            .expect("style toys");
        let mut records = Records::default();
        published(&mut booted.session, &mut records, |_| ());
        let instances = |booted: &Boot| {
            booted
                .session
                .domains()
                .read(booted.domain)
                .expect("domain")
                .instances()
                .iter()
                .map(|(_, instance)| *instance)
                .collect::<Vec<_>>()
        };
        let before = instances(&booted);
        assert_eq!(before.len(), 5);
        assert!(before
            .iter()
            .all(|i| i.line_width_px == Some(4.0) && i.line_opacity == Some(0.2)));
        send(&mut booted, Action::Reset);
        booted
            .session
            .boundary(Input::default())
            .expect("reset toys");
        published(&mut booted.session, &mut records, |_| ());
        assert_eq!(instances(&booted), before);
    }

    #[test]
    fn a_stationary_pointer_release_clears_the_spring_carry_velocity() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted.session.boundary(Input::default()).expect("toybox");
        let entity = booted
            .session
            .app
            .slots
            .iter()
            .find(|(_, slot)| slot.index == 2)
            .map(|(entity, _)| entity)
            .expect("middle toy");
        let center = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain")
            .poses()
            .get(entity)
            .expect("pose")
            .point
            .truncate();
        booted.session.views_mut().root_mut().eye = Eye::looking_at(
            [center.x, center.y, EYE_BACK],
            center.to_array(),
            [0.0, 1.0, 0.0],
        );
        booted.session.grab([0.0; 2], 0.0).expect("grab");
        booted.session.boundary(Input::default()).expect("hold");
        let physics = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("domain")
            .physics_mut()
            .expect("physics");
        physics
            .set_velocity(entity, Vec4::Y, Bivector4::ZERO)
            .expect("carry velocity");
        let pointer = Pointer {
            id: 0,
            button: Some(PointerButton::Primary),
            ndc: [0.0; 2],
            delta: [0.0; 2],
            phase: PointerPhase::Ended,
            time: 1.0,
        };
        booted
            .session
            .boundary(Input {
                pointers: vec![pointer],
                ..Input::default()
            })
            .expect("drop");
        let physics = booted
            .session
            .domains_mut()
            .typed(booted.domain)
            .expect("domain")
            .physics()
            .expect("physics");
        let body = physics.body(entity).expect("body");
        assert_eq!(
            physics.world().body(body).expect("body row").velocity,
            Vec4::ZERO
        );
    }

    #[test]
    fn a_primary_grab_does_not_block_secondary_orbit_or_wheel_zoom() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted.session.boundary(Input::default()).expect("toybox");
        let center = toy::pose_at(Polytope4::Pentatope, 0.0).point.truncate();
        booted.session.app.control.get_mut().orbit = Orbit::around(center.to_array(), EYE_BACK);
        let mut input = loam::app::session::input::InputMap::default();
        input.resize(800, 600, 1.0);
        input.button([0.0; 2], PointerButton::Primary, true);
        booted.session.boundary(input.take()).expect("grab");
        let picked = booted.session.dragging().expect("grab");
        booted.session.boundary(Input::default()).expect("hold");

        input.button([0.0; 2], PointerButton::Secondary, true);
        input.moved([0.05, 0.05]);
        input.wheel([0.0, 1.0]);
        let movement = input.take();
        booted.session.boundary(movement).expect("move");
        let orbit = booted.session.app.control.get().orbit;
        let root = booted.session.views().root();
        let eye = booted.session.views().get(root).expect("root view").eye;
        assert!(orbit.yaw < -0.1 && orbit.pitch > 0.08);
        assert!(orbit.distance < EYE_BACK);
        let held = booted.session.dragging().expect("still grabbed");
        assert_eq!(held.entity, picked.entity);
        let clip = loam::render::view::root_view_projection(&eye)
            * glam::Vec3::from(held.image_point()).extend(1.0);
        assert!((clip.x / clip.w - 0.05).abs() < 1e-5);
        assert!((clip.y / clip.w - 0.05).abs() < 1e-5);
        let physics = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain");
        assert!(physics.physics().expect("physics").is_held(picked.entity));
    }

    #[test]
    fn a_toybox_drag_stays_held_between_samples_and_reports_its_guides() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let entity = booted
            .session
            .app
            .slots
            .iter()
            .find(|(_, slot)| slot.index == 2)
            .map(|(entity, _)| entity)
            .expect("the center toy is live");
        let center = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain")
            .poses()
            .get(entity)
            .expect("the center toy has a pose")
            .point
            .truncate();
        booted.session.views_mut().root_mut().eye = Eye::looking_at(
            [center.x, center.y, EYE_BACK],
            center.to_array(),
            [0.0, 1.0, 0.0],
        );
        let picked = booted
            .session
            .grab([0.0, 0.0], 0.0)
            .expect("the ray through the slot center picks it");
        assert_eq!(picked.entity, entity);
        booted.session.boundary(Input::default()).expect("hold");
        let mut guides = guides::Guides::default();
        guides.update(&booted.session, booted.domain);
        assert_eq!(guides.points.len(), 1);
        assert_eq!(guides.lines.len(), 8);
        let anchor = guides.points[0].position;

        let target = booted
            .session
            .drag([0.0, 0.28], 1.0)
            .expect("the drag meets its plane");
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary applied the move");
        guides.update(&booted.session, booted.domain);
        assert_eq!(guides.points[0].position, anchor);
        let refusal = booted
            .session
            .dispatch(|dispatch| {
                dispatch.apply(Command::Chart(
                    booted.domain.id(),
                    ChartCommand::Move {
                        entity: picked.entity,
                        point: ChartPoint {
                            chart: target.chart,
                            coordinates: [f32::NAN, 0.0, 0.0, 0.0],
                        },
                    },
                ))
            })
            .expect_err("a held body accepted a nonfinite target");
        assert!(matches!(
            refusal,
            Rejection::Domain(loam::runtime::DomainError::InvalidCoordinate("x"))
        ));
        for _ in 0..120 {
            booted.session.tick().expect("the tick ran");
        }
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let pose = r4.poses().get(picked.entity).expect("pose").point;
        assert!(
            (pose.y - target.coordinates[1]).abs() < 0.1,
            "the held body stopped following at y {} before target {}",
            pose.y,
            target.coordinates[1]
        );

        let release = booted.session.release_at(1.05).expect("the drag was live");
        guides.update(&booted.session, booted.domain);
        assert!(guides.points.is_empty());
        assert_eq!(guides.lines.len(), 4);
        send(&mut booted, Action::Throw(release.entity, release.velocity));
        booted
            .session
            .boundary(Input::default())
            .expect("the release and throw landed");
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let physics = r4.physics().expect("the domain has physics");
        let body = physics.body(release.entity).expect("the toy has a body");
        assert!(physics.world().body(body).expect("body row").velocity.y > 0.0);
        for _ in 0..600 {
            booted.session.tick().expect("the tick ran");
        }
        let polytope = booted
            .session
            .app
            .slots
            .get(release.entity)
            .and_then(|slot| slot.entry.collider_polytope())
            .expect("the released toy has a polytope");
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let physics = r4.physics().expect("the domain has physics");
        let body = physics.body(release.entity).expect("the toy has a body");
        let body = physics.world().body(body).expect("body row");
        let slice_reach = polytope
            .topology()
            .vertices
            .iter()
            .map(|vertex| {
                (body.orientation.rotation.apply(*vertex * toy::BODY_SIZE))
                    .w
                    .abs()
            })
            .fold(0.0f32, f32::max);
        assert!(
            body.position.w.abs() <= slice_reach,
            "the released toy left its slice reach {slice_reach} at w {} with velocity {}",
            body.position.w,
            body.velocity.w
        );
        assert!(body.is_sleeping());
    }

    #[test]
    fn a_shape_card_respawns_the_slot_in_place_with_the_new_polytopes_edges() {
        let mut booted = one_slot();
        send(
            &mut booted,
            Action::Display(Display {
                wireframe: true,
                ..Display::default()
            }),
        );
        send(&mut booted, Action::Slice(consts::W_RANGE));
        let rest = booted
            .session
            .app
            .slots
            .iter()
            .map(|(_, slot)| slot.rest)
            .next()
            .expect("the row has a slot");

        send(&mut booted, Action::Shape(0, 0));
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
            [0, 10],
            "the swapped slot still publishes the old polytope's edges"
        );
    }

    #[test]
    fn held_slice_motion_uses_fixed_ticks_while_authored_spin_is_paused() {
        const TICKS: usize = 8;

        let scrubbed = |boundaries: usize| {
            let mut booted = one_slot();
            booted.session.app.spin.get_mut().running = false;
            for boundary in 0..boundaries {
                let held = if boundary == 0 || boundaries > 1 {
                    vec![SLICE_UP]
                } else {
                    Vec::new()
                };
                booted
                    .session
                    .boundary(Input {
                        held,
                        ..Input::default()
                    })
                    .expect("the boundary ran");
                for _ in 0..TICKS / boundaries {
                    booted.session.tick().expect("the tick ran");
                }
            }
            *booted.session.app.slice.get()
        };

        assert!((scrubbed(1) - scrubbed(TICKS)).abs() < 1e-6);
        assert!(scrubbed(1) > 0.0);
    }

    #[test]
    fn the_gimbal_rotor_turns_the_row_by_the_angle_its_ring_names() {
        use loam::math::{Bivector, Plane4, Rotor};

        let mut booted = one_slot();
        const ANGLE: f32 = 0.4;
        let domain = booted.domain;
        booted.session.dispatch(|d| {
            mode::turn_row(
                d.app,
                d.domains,
                domain,
                (Plane4::Xw.unit_bivector() * ANGLE).exp(),
            )
            .expect("the turn applies")
        });

        let entity = slot_entity(&booted);
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let turned = r4.poses().get(entity).expect("pose").frame.apply(Vec4::X);
        let expected = Vec4::new(ANGLE.cos(), 0.0, 0.0, ANGLE.sin());
        assert!(
            (turned - expected).length() < 1e-5,
            "the ring's rotor sent x to {turned} rather than the analytic {expected}"
        );
    }

    #[test]
    fn a_press_on_a_ring_turns_the_row_only_while_the_filmstrip_is_off() {
        fn press_and_drag(booted: &mut Boot, ndc: [f32; 2]) -> bool {
            let mut at = |phase, ndc, time| {
                let pointer = Pointer {
                    id: 0,
                    button: Some(PointerButton::Primary),
                    ndc,
                    delta: [0.0; 2],
                    phase,
                    time,
                };
                booted
                    .session
                    .boundary(Input {
                        pointers: vec![pointer],
                        ..Input::default()
                    })
                    .expect("pointer boundary");
                booted.session.app.control.get().gimbal.held()
            };
            let taken = at(PointerPhase::Began, ndc, 0.0);
            at(PointerPhase::Moved, [ndc[0] + 0.2, ndc[1] + 0.2], 1.0);
            taken
        }

        let mut booted = one_slot();
        booted
            .session
            .dispatch(|dispatch| Action::Gimbal(None).apply(dispatch))
            .expect("handles");
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let on_ring = gimbal::widget(glam::Vec3::new(0.0, BODY_Y, 0.0)).rings()[0].point(0.0);
        let ndc = booted
            .session
            .views()
            .ndc(on_ring.to_array())
            .expect("the ring is in front of the eye");

        assert!(
            press_and_drag(&mut booted, ndc),
            "the press missed the ring, so the strip has nothing to mask"
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let entity = slot_entity(&booted);
        let turned = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain")
            .poses()
            .get(entity)
            .expect("pose")
            .frame;
        assert_ne!(
            turned,
            loam::math::Rotor4::IDENTITY,
            "the ring drag never turned the row, so the check below proves nothing"
        );

        send(
            &mut booted,
            Action::Strip(Strip {
                on: true,
                ..Strip::default()
            }),
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        assert!(
            !press_and_drag(&mut booted, ndc),
            "the gimbal took a press while the filmstrip covered it"
        );
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let held = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain")
            .poses()
            .get(entity)
            .expect("pose")
            .frame;
        assert_eq!(
            held, turned,
            "an invisible gimbal turned the row under the filmstrip"
        );
    }

    #[test]
    fn a_cancelled_pointer_releases_a_live_grab() {
        let mut booted = one_slot();
        send(&mut booted, Action::Mode(Mode::Toybox));
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary ran");
        let entity = booted
            .session
            .app
            .slots
            .iter()
            .find(|(_, slot)| slot.index == 2)
            .map(|(entity, _)| entity)
            .expect("middle toy");
        let center = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("domain")
            .poses()
            .get(entity)
            .expect("pose")
            .point
            .truncate();
        booted.session.views_mut().root_mut().eye = Eye::looking_at(
            [center.x, center.y, EYE_BACK],
            center.to_array(),
            [0.0, 1.0, 0.0],
        );
        booted
            .session
            .grab([0.0, 0.0], 0.0)
            .expect("the ray through the slot center picks it");
        let pointer = Pointer {
            id: 0,
            button: Some(PointerButton::Primary),
            ndc: [0.0, 0.0],
            delta: [0.0; 2],
            phase: PointerPhase::Cancelled,
            time: 1.0,
        };
        let hover = Pointer {
            button: None,
            phase: PointerPhase::Moved,
            ..pointer
        };
        let input = Input {
            pointers: vec![pointer, hover],
            ..Input::default()
        };
        booted.session.boundary(input).expect("cancel boundary");

        assert!(booted.session.dragging().is_none());
        assert!(
            booted.session.drag([0.5, 0.0], 2.0).is_err(),
            "the grab survived the focus change that cancelled the pointer"
        );
    }

    #[test]
    fn a_raster_cut_is_duplicated_in_the_projection_layer() {
        const TESSERACT: ShapeEntry = ShapeEntry {
            shape: RaymarchShape::Polytope(Polytope4::Tesseract),
            body_color: [0.30, 0.55, 0.95],
            label: "8-cell",
            long_name: "tesseract",
            category: catalog::Category::RegularPolychoron,
        };
        let mut booted = boot(&[TESSERACT]).expect("the session boots");
        booted.session.boundary(Input::default()).expect("boundary");
        let mut records = Records::default();
        published(&mut booted.session, &mut records, |publication| {
            assert_eq!(publication.views[0].records.triangles().len(), 24);
            assert!(publication.views[1].records.triangles().is_empty());
        });
    }

    #[test]
    fn filmstrip_recenters_after_single_and_restores_the_single_camera() {
        let focus = (true, 0, Mode::Rotate);
        let single_target = [-2.4, 0.0, 0.0];
        let mut orbit = Orbit::around(single_target, 5.0);
        orbit.yaw = 0.3;
        orbit.pitch = -0.4;
        let mut camera_focus = Some(focus);
        let mut filmstrip_return = None;
        let strip = Strip {
            on: true,
            ..Strip::default()
        };

        focus_camera(
            &mut orbit,
            &mut camera_focus,
            &mut filmstrip_return,
            strip,
            focus,
            glam::Vec3::from(single_target),
        );
        assert_eq!(orbit.target, [0.0; 3]);
        assert_eq!((orbit.yaw, orbit.pitch, orbit.distance), (0.3, -0.4, 5.0));

        focus_camera(
            &mut orbit,
            &mut camera_focus,
            &mut filmstrip_return,
            Strip { t: true, ..strip },
            focus,
            glam::Vec3::from(single_target),
        );
        assert_eq!(orbit.target, [0.0, BODY_Y, 0.0]);

        focus_camera(
            &mut orbit,
            &mut camera_focus,
            &mut filmstrip_return,
            Strip { on: false, ..strip },
            focus,
            glam::Vec3::from(single_target),
        );
        assert_eq!(orbit.target, single_target);
        assert_eq!((orbit.yaw, orbit.pitch, orbit.distance), (0.3, -0.4, 5.0));
    }

    #[test]
    fn the_marcher_takes_one_strip_cell_per_grid_rectangle_and_none_once_the_strip_is_off() {
        let mut scratch = Scratch::new();
        let strip = Strip {
            on: true,
            w: true,
            t: true,
            count_w: 5,
            count_t: 3,
            ..Strip::default()
        };
        fill_strip(&strip, Bivector4::ZERO, HEADLESS_FRAME, 0.25, &mut scratch);
        assert_eq!(scratch.strip.len(), 15, "a 5 by 3 grid is fifteen draws");
        let covered: u64 = scratch
            .strip
            .iter()
            .map(|(viewport, _, _)| u64::from(viewport.width) * u64::from(viewport.height))
            .sum();
        assert_eq!(
            covered,
            u64::from(HEADLESS_FRAME.0) * u64::from(HEADLESS_FRAME.1),
            "the cells the marcher draws leave a gap or overlap"
        );
        let slices: Vec<f32> = scratch.strip.iter().map(|(_, w, _)| *w).collect();
        assert!(
            (slices[0] - (0.25 - BODY_SIZE)).abs() < 1e-6
                && (slices[14] - (0.25 + BODY_SIZE)).abs() < 1e-6,
            "the strip does not span the body around the slider: {slices:?}"
        );

        fill_strip(
            &Strip { on: false, ..strip },
            Bivector4::ZERO,
            HEADLESS_FRAME,
            0.25,
            &mut scratch,
        );
        assert!(
            scratch.strip.is_empty(),
            "a stale strip kept the filmstrip on screen after it was switched off"
        );
    }
}
