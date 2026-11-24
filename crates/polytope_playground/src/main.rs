use std::sync::Arc;

use loam::app::environment::Environment;
use loam::prelude::*;
use loam::render::raymarch::{shape_id, BodyUniform};

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
mod shown;
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
use shown::Shown;
use strip::{Cell, Strip};

const SPIN: ActionId = ActionId("spin");
const NEXT_MODE: ActionId = ActionId("next mode");
const RESET: ActionId = ActionId("reset");
const GIMBAL: ActionId = ActionId("gimbal");
const STRIP: ActionId = ActionId("strip");
const CONTROLS: ActionId = ActionId("controls");
const PLANE: [ActionId; 6] = [
    ActionId("plane xy"),
    ActionId("plane xz"),
    ActionId("plane xw"),
    ActionId("plane yz"),
    ActionId("plane yw"),
    ActionId("plane zw"),
];

const SECTION_COLOR: [f32; 4] = [1.0, 0.85, 0.35, 1.0];
const SECTION_WIDTH_PX: f32 = 2.0;
const EDGE_WIDTH_PX: f32 = display::DEFAULT_WIREFRAME_WIDTH_PX;
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
pub(crate) struct Toy;

#[derive(Clone, Copy)]
pub(crate) struct HiddenSlot {
    pub(crate) slot: Slot,
    pub(crate) instance: Option<Instance>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct Control {
    gimbal: Gimbal,
    latest: Option<Pointer>,
    center: glam::Vec3,
    camera_focus: Option<(bool, usize, Mode)>,
    filmstrip_return: Option<Orbit>,
}

#[derive(Clone, Copy)]
pub(crate) struct Card {
    geometry: PreparedId,
    toy_geometry: Option<PreparedId>,
    material: MaterialId,
    cut: MaterialId,
    shades: Option<Shades>,
}

impl Card {
    fn dressed(&self, geometry: PreparedId) -> Instance {
        Instance::new(geometry, self.material).sectioned(self.cut)
    }

    pub(crate) fn body(&self) -> Instance {
        self.dressed(self.geometry)
    }

    pub(crate) fn toy(&self) -> Option<Instance> {
        self.toy_geometry.map(|geometry| self.dressed(geometry))
    }
}

#[derive(Clone, Default)]
pub(crate) struct Catalog {
    cards: Arc<[Card]>,
}

stores! {
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
        display: Value<Display>,
        time: Value<f32>,
        angles: Value<[f32; 6]>,
        controls: Value<bool>,
        formula: Value<bool>,
        labels: Value<bool>,
        guides: Value<bool>,
        toybox_debug: Value<bool>,
        rope: Value<bool>,
        control: Value<Control>,
        catalog: Value<Catalog>,
    }
    publishes Shown;
}

#[derive(Clone, Copy)]
pub(crate) struct Toggle {
    pub(crate) name: &'static str,
    pub(crate) setting: Setting<Playground>,
}

pub(crate) const SHOW_CONTROLS: Toggle = Toggle {
    name: "controls",
    setting: |app| app.controls.get_mut(),
};
pub(crate) const SHOW_FORMULA: Toggle = Toggle {
    name: "formula",
    setting: |app| app.formula.get_mut(),
};
pub(crate) const SHOW_GUIDES: Toggle = Toggle {
    name: "guides",
    setting: |app| app.guides.get_mut(),
};
pub(crate) const SHOW_GIMBAL: Toggle = Toggle {
    name: "gimbal",
    setting: |app| app.gimbal.get_mut(),
};
pub(crate) const TOYBOX_DEBUG: Toggle = Toggle {
    name: "toybox debug",
    setting: |app| app.toybox_debug.get_mut(),
};

#[derive(Clone, Copy)]
pub(crate) enum Action {
    Mode(Mode),
    Active(usize),
    Slice(f32),
    ExactSlice(f32),
    Plane(usize),
    Running(Option<bool>),
    Projection(Family),
    NextProjection,
    Color(ColorMode),
    NextColor,
    Shape(usize, usize),
    AddShape(usize),
    RemoveShape(usize),
    ReorderShape { from: usize, to: usize },
    Strip(Strip),
    Rate(f32),
    Display(Display),
    Reset,
    Time(f32),
    PlaneAngle(usize, f32),
}

fn rest_of(index: usize, len: usize) -> Vec4 {
    let center = (len.max(1) - 1) as f32 * 0.5;
    Vec4::new((index as f32 - center) * BODY_X_SPACING, BODY_Y, 0.0, 0.0)
}

pub(crate) fn boot(
    row: &[ShapeEntry],
) -> Result<(Session<Playground>, DomainHandle<EuclideanR4>), HostError> {
    let mut session = Session::new(
        Playground {
            slots: Store::tracked(LogCapacity::default()),
            ..Playground::default()
        },
        SimConfig::default(),
    );
    let domain = session.register_domain(
        DomainBuilder::new(DOMAIN_NAME, EuclideanR4)
            .physics(toy::physics_config())
            .map_err(Rejection::from)?,
    );
    let root = session.views().root_id();

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
            if let Some(card) = card_of(entry) {
                bundle = bundle.instance(cards[card].body());
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
    let root = session.views_mut().root_mut();
    root.eye = Eye::looking_at([0.0, 3.0, 9.0], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
    root.backdrop = Some(Environment::default().backdrop(consts::FLOOR_Y));
    session.set_rig(Rig::Pair(CameraPair::new(
        home([0.0; 3]),
        FreeCamera::default(),
    )));
    session.app.controls.set(true);
    session.app.guides.set(true);

    let cards = session.app.catalog.cards.clone();
    install_systems(&mut session, domain, layers, cards);
    Ok((session, domain))
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
                geometry: dispatch.prepare(match polytope {
                    Some(polytope) => PreparedGeometry::Polytope4 {
                        polytope,
                        scale: BODY_SIZE,
                    },
                    None => PreparedGeometry::Field {
                        shape: entry.shape,
                        size: BODY_SIZE,
                    },
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
    session
        .system_at(
            Phase::Dispatch,
            Order::Before(CAMERA_RIG),
            "camera focus",
            move |ctx| {
                let center = scene_center(ctx.app, ctx.domains, domain, ctx.alpha);
                let focus = (ctx.app.display.single, *ctx.app.active, *ctx.app.mode);
                let strip = *ctx.app.strip;
                let control = ctx.app.control.get_mut();
                for pointer in &ctx.input.pointers {
                    control.latest = Some(*pointer);
                }
                control.center = center;
                let orbit = ctx
                    .views
                    .root_mut()
                    .rig
                    .orbit_mut()
                    .expect("the playground's rig is a pair");
                focus_camera(
                    orbit,
                    &mut control.camera_focus,
                    &mut control.filmstrip_return,
                    strip,
                    focus,
                    center,
                );
                Ok(())
            },
        )
        .expect("Session::new registers the camera rig");
    // Runs before the pointer, so a release throws with the hold the rope setting names in the same boundary.
    session.system(Phase::Dispatch, "grab hold", move |ctx| {
        let hold = if *ctx.app.rope.get() {
            GrabHold::Anchor
        } else {
            GrabHold::Center
        };
        ctx.domains.physics(domain)?.set_grab_hold(hold);
        Ok(())
    });
    session.system(Phase::Dispatch, "pointer", move |mut ctx| {
        control_primary(&mut ctx, domain);
        Ok(())
    });
    session.on_action(SPIN, |d| {
        if *d.app.mode == Mode::Rotate {
            Action::Running(None).apply(d)?;
        }
        Ok(())
    });
    // One system, so the strip toggle reads the strip before the mode change in the same boundary clears it.
    session.system(Phase::Dispatch, "keys", |ctx| {
        let strip = *ctx.app.strip;
        if ctx.input.pressed(NEXT_MODE) {
            let next = match *ctx.app.mode {
                Mode::Rotate => Mode::Toybox,
                Mode::Toybox => Mode::Rotate,
            };
            ctx.commands.app(Action::Mode(next));
        }
        if ctx.input.pressed(RESET) {
            ctx.commands.app(Action::Reset);
        }
        if ctx.input.pressed(STRIP) {
            ctx.commands.app(Action::Strip(Strip {
                on: !strip.on,
                ..strip
            }));
        }
        Ok(())
    });
    let mut rings = gimbal::GimbalRenderer::default();
    let mut lines = String::new();
    session.system(Phase::Publication, "overlay", move |mut ctx| {
        let center = scene_center(ctx.app, ctx.domains, domain, ctx.alpha);
        rings.rings(
            &ctx.app.control.get().gimbal,
            center,
            &mut ctx.overlay.lines,
        );
        if *ctx.app.hud.get() {
            hud::draw(ctx.overlay, &readout_of(ctx.app), &mut lines);
        }
        guides::draw(&mut ctx, domain);
        Ok(())
    });
    session.publish_value("shown", move |ctx, shown: &mut Shown| {
        shown.write(ctx.app, ctx.domains, ctx.views, domain, ctx.alpha);
        Ok(())
    });
    session.toggle_on(GIMBAL, SHOW_GIMBAL.setting);
    session.toggle_on(CONTROLS, SHOW_CONTROLS.setting);
    for (index, action) in PLANE.into_iter().enumerate() {
        session.on_action(action, move |d| Action::Plane(index).apply(d).map(drop));
    }

    session.system_on_change(
        Phase::Publication,
        "slice view",
        |app: &Playground| app.slice.version(),
        move |ctx| {
            let w = *ctx.app.slice.get();
            let r4 = ctx.domains.typed(domain)?;
            let spec = r4
                .view_mut(layers.section)
                .ok_or(DomainError::Unsupported("section view"))?;
            spec.set_mapping(Section4 { w });
            Ok(())
        },
    );

    session.system_on_change(
        Phase::Publication,
        "shading",
        |app: &Playground| {
            (
                app.color.version(),
                app.strip.version(),
                app.display.version(),
                app.active.version(),
                app.slots.revision(),
            )
        },
        move |ctx| {
            let app = &*ctx.app;
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
            spec.fields = display.surface != Surface::Off;
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
            Ok(())
        },
    );

    session.system_on_change(
        Phase::Publication,
        "projection view",
        |app: &Playground| (app.projection.version(), app.slice.version()),
        move |ctx| {
            let family = *ctx.app.projection.get();
            let slice = *ctx.app.slice.get();
            let r4 = ctx.domains.typed(domain)?;
            let spec = r4
                .view_mut(layers.projection)
                .ok_or(DomainError::Unsupported("projection view"))?;
            spec.set_mapping(family.mapping(slice));
            Ok(())
        },
    );

    session.system(Phase::Simulation, "slice scrub", |ctx| {
        if ctx.views.root_mut().rig.flying() {
            return Ok(());
        }
        // E and Q move the free camera, so the slice takes them only while the orbit holds the eye.
        let scrub = ctx.input.axis(MOVE_DOWN, MOVE_UP);
        if scrub == 0.0 {
            return Ok(());
        }
        let next = *ctx.app.slice + scrub * W_SCRUB_RATE * ctx.step.dt;
        let reach = slice_reach(*ctx.app.mode);
        ctx.app.slice.set(next.clamp(-reach, reach));
        Ok(())
    });
    session.system(Phase::Simulation, "spin", move |ctx| {
        if !ctx.app.spin.running || *ctx.app.mode == Mode::Toybox {
            return Ok(());
        }
        let time = *ctx.app.time + ctx.step.dt * ctx.app.spin.rate;
        display::seek(ctx.app, ctx.domains, domain, time)?;
        Ok(())
    });
}

impl AppCommand<Playground> for Action {
    fn name(&self) -> &'static str {
        match self {
            Self::Mode(mode) => mode.name(),
            Self::Active(_) => "active",
            Self::Slice(_) | Self::ExactSlice(_) => "slice",
            Self::Plane(_) => "plane",
            Self::Running(_) => "spin",
            Self::Projection(family) => family.name(),
            Self::NextProjection => "wireframe perspective",
            Self::Color(mode) => mode.name(),
            Self::NextColor => "wireframe color",
            Self::Shape(_, card) => catalog::SHAPE_CATALOG
                .get(*card)
                .map_or("shape", |entry| entry.label),
            Self::AddShape(_) => "add shape",
            Self::RemoveShape(_) => "remove shape",
            Self::ReorderShape { .. } => "reorder shape",
            Self::Strip(_) => "strip",
            Self::Rate(_) => "rate",
            Self::Display(_) => "display",
            Self::Reset => "reset",
            Self::Time(_) => "time",
            Self::PlaneAngle(_, _) => "plane angle",
        }
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let domain = dispatch.domains.named::<EuclideanR4>(DOMAIN_NAME)?;
        let cards = dispatch.app.catalog.cards.clone();
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
                let reach = slice_reach(*dispatch.app.mode);
                dispatch.app.slice.set(w.clamp(-reach, reach));
            }
            Action::ExactSlice(w) => {
                if !w.is_finite() {
                    return Err(Rejection::Unsupported("slice is not finite"));
                }
                let reach = slice_reach(*dispatch.app.mode);
                if !(-reach..=reach).contains(&w) {
                    return Err(Rejection::Unsupported(
                        "slice is outside the mode's w range",
                    ));
                }
                dispatch.app.slice.set(w);
            }
            Action::Plane(plane) => mode::toggle_plane(dispatch, plane)?,
            Action::Running(setting) => {
                if *dispatch.app.mode == Mode::Toybox {
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
                let held = *dispatch.app.projection;
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
                let held = *dispatch.app.color;
                let at = ColorMode::ALL
                    .iter()
                    .position(|mode| *mode == held)
                    .unwrap_or(0);
                dispatch
                    .app
                    .color
                    .set(ColorMode::ALL[(at + 1) % ColorMode::ALL.len()]);
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
        .key(Key::Letter('d'), MOVE_RIGHT)
        .key(Key::Letter('a'), MOVE_LEFT)
        .key(Key::Letter('w'), MOVE_FORWARD)
        .key(Key::Letter('s'), MOVE_BACKWARD)
        .key(Key::Letter('e'), MOVE_UP)
        .key(Key::Letter('q'), MOVE_DOWN);
    for (index, action) in PLANE.into_iter().enumerate() {
        bound = bound.key(Key::Digit(index as u8 + 1), action);
    }
    bound
}

pub(crate) struct Frame {
    hyperslice: HyperslicePass,
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self {
            hyperslice: HyperslicePass::with_ground(),
        }
    }

    fn passes(&self) -> Vec<Box<dyn FramePass>> {
        vec![Box::new(self.hyperslice.clone())]
    }
}

#[derive(Default)]
struct Scratch {
    cells: Vec<Cell>,
    strip: Vec<(loam::render::Viewport, f32, BodyUniform)>,
}

fn main() -> Result<(), HostError> {
    launch(interactive)
}

fn interactive(args: Args) -> Result<(Session<Playground>, SessionApp<Playground>), HostError> {
    let row = catalog::parse_row(&args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let (mut session, _) = boot(&row)?;
    let surface = match args.flag_value("surface") {
        None => None,
        Some("raster") => Some(Surface::Raster),
        Some("sdf") => Some(Surface::Sdf),
        Some("off") => Some(Surface::Off),
        Some(other) => {
            return Err(HostError::Host(format!(
                "--surface {other:?} is not raster, sdf or off"
            )))
        }
    };
    if let Some(surface) = surface {
        session.app.display.get_mut().surface = surface;
    }
    let frame = Frame::new();
    let config = HostConfig::new("polytope playground", bindings());

    let hyperslice = frame.hyperslice.clone();
    let mut scratch = Scratch::default();
    let mut panel = ui::Panel::default();
    let mut app = SessionApp::with_args(config, args).recover_on_fault(RESET);
    for pass in frame.passes() {
        app = app.pass(pass);
    }
    app = console::install(app).on_published(move |hook: &mut PublishedHook<'_, Playground>| {
        let shown = &hook.published.value;
        fill_strip(
            &shown.strip,
            shown.turn,
            shown.subject,
            hook.size,
            shown.slice,
            &mut scratch,
        );
        hyperslice.publish_strip(&scratch.strip);
        if let Some(context) = hook.ui {
            ui::draw(context, shown, &mut panel, hook.sender);
            if shown.labels && !shown.strip.on {
                ui::callouts(context, &shown.callouts);
            }
            ui::strip_labels(context, shown.strip, &scratch.cells);
        }
    });
    Ok((session, app))
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
        *orbit = home(target);
    } else {
        orbit.target = target;
    }
    *camera_focus = Some(focus);
}

fn home(target: [f32; 3]) -> Orbit {
    Orbit {
        pitch: -0.25,
        ..Orbit::around(target, CAMERA_DISTANCE)
    }
}

fn slice_reach(mode: Mode) -> f32 {
    match mode {
        Mode::Rotate => consts::W_RANGE,
        Mode::Toybox => toy::ARENA_HALF,
    }
}

fn readout_of(app: &Playground) -> hud::Readout {
    hud::Readout {
        slice: *app.slice.get(),
        rate: app.spin.get().rate,
        bodies: app.slots.len(),
        planes: app.spin.get().planes,
        toybox: *app.mode.get() == Mode::Toybox,
    }
}

fn fill_strip(
    strip: &Strip,
    omega: Bivector4,
    subject: loam::math::Rotor4,
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
        let rotor = ((omega * cell.t).exp() * subject).normalize();
        scratch.strip.push((
            cell.viewport,
            cell.w,
            BodyUniform::polytope_with_rotor(
                [0.0, BODY_Y, 0.0, 0.0],
                shape_id(entry.shape),
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
    alpha: f32,
) -> glam::Vec3 {
    let active = *app.active;
    let single = app.display.single;
    let Ok(r4) = domains.read(domain) else {
        return glam::Vec3::ZERO;
    };
    let mut sum = glam::Vec3::ZERO;
    let mut count = 0;
    for (entity, slot) in app.slots.iter() {
        if single && slot.index != active {
            continue;
        }
        let Some(pose) = r4.drawn_pose(entity, alpha) else {
            continue;
        };
        sum += pose.point.truncate();
        count += 1;
    }
    sum / count.max(1) as f32
}

fn control_primary(ctx: &mut Ctx<'_, Playground>, domain: DomainHandle<EuclideanR4>) {
    // Toybox mode keeps the filmstrip off and the gimbal hidden, so the press checks only the mode and the toy.
    ctx.carry(PointerButton::Primary, |app, entity| {
        *app.mode.get() == Mode::Toybox && app.toys.contains(entity)
    });
    let (latest, center, mut gimbal) = {
        let control = ctx.app.control.get();
        (control.latest, control.center, control.gimbal)
    };
    gimbal.enabled =
        *ctx.app.gimbal.get() && *ctx.app.mode.get() == Mode::Rotate && !ctx.app.strip.get().on;
    let root = ctx.views.root_id();
    let ray = latest.and_then(|pointer| ctx.views.ray(root, pointer.ndc));
    if gimbal.enabled {
        gimbal.aim(ray.as_ref(), center);
    } else {
        gimbal.release();
    }
    for pointer in &ctx.input.pointers {
        if pointer.button != Some(PointerButton::Primary) {
            continue;
        }
        match pointer.phase {
            PointerPhase::Began if gimbal.enabled => {
                if let Some(ray) = ctx.views.ray(root, pointer.ndc) {
                    gimbal.press(&ray, center);
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
            PointerPhase::Ended | PointerPhase::Cancelled => gimbal.release(),
            _ => {}
        }
    }
    ctx.app.control.get_mut().gimbal = gimbal;
}

#[cfg(test)]
mod tests {
    use loam::math::Rotor;
    use loam::runtime::{Command, Publication};
    use loam::shape::polytope::Polytope4;
    use loam::shape::FieldShape;
    use loam::time::par::Sequential;

    use super::*;

    const CELL24: ShapeEntry = ShapeEntry {
        shape: FieldShape::Polytope(Polytope4::Cell24),
        body_color: [0.95, 0.45, 0.85],
        label: "24-cell",
        long_name: "icositetrachoron",
        category: catalog::Category::RegularPolychoron,
    };

    const EYE_BACK: f32 = 5.0;

    pub(crate) struct Boot {
        pub(crate) session: Session<Playground>,
        pub(crate) domain: DomainHandle<EuclideanR4>,
    }

    fn one_slot() -> Boot {
        let (session, domain) = boot(&[CELL24]).expect("the session boots");
        let mut booted = Boot { session, domain };
        booted.session.views_mut().root_mut().eye =
            Eye::looking_at([0.0, BODY_Y, EYE_BACK], [0.0, BODY_Y, 0.0], [0.0, 1.0, 0.0]);
        booted.session.app.spin.get_mut().running = false;
        booted
    }

    pub(crate) fn send(booted: &mut Boot, action: Action) {
        booted.session.submit(Command::App(Box::new(action)));
    }

    fn published<R>(
        session: &mut Session<Playground>,
        publication: &mut Publication<Shown>,
        read: impl FnOnce(&Publication<Shown>) -> R,
    ) -> R {
        session.publish(publication, 1.0).expect("published");
        read(publication)
    }

    #[test]
    fn edit_and_reset() {
        let mut booted = one_slot();
        let mut publication = Publication::default();
        booted.session.boundary(Input::default()).expect("boundary");
        published(&mut booted.session, &mut publication, |publication| {
            assert!(!publication.views[0].records.triangles().is_empty());
        });

        send(&mut booted, Action::PlaneAngle(0, 0.4));
        send(&mut booted, Action::AddShape(4));
        booted.session.boundary(Input::default()).expect("edit row");
        assert!(booted.session.results().iter().all(|r| r.outcome.is_ok()));
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
        for (entity, _) in &rows {
            let pose = domain.poses().get(*entity).expect("pose");
            assert_ne!(pose.frame, loam::math::Rotor4::IDENTITY);
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
        published(&mut booted.session, &mut publication, |publication| {
            assert!(!publication.views[0].records.triangles().is_empty());
        });
    }

    #[test]
    fn grab_drag_and_throw() {
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
        let anchor = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain")
            .physics()
            .and_then(|physics| physics.held_anchor(picked.entity))
            .expect("the toy is held");

        let target = booted
            .session
            .drag([0.0, 0.28], 1.0)
            .expect("the drag meets its plane");
        booted
            .session
            .boundary(Input::default())
            .expect("the boundary applied the move");
        for _ in 0..120 {
            booted.session.tick(&Sequential).expect("the tick ran");
        }
        let r4 = booted
            .session
            .domains()
            .read(booted.domain)
            .expect("the r4 domain");
        let grip = r4
            .physics()
            .and_then(|physics| physics.held_anchor(picked.entity))
            .expect("the toy is held");
        let goal = anchor.y + target.coordinates[1] - center.y;
        assert!(
            (grip.y - goal).abs() < 0.1,
            "the grab point stopped following at y {} before its goal {goal}",
            grip.y
        );

        let release = booted.session.release_at(1.05).expect("the drag was live");
        booted
            .session
            .boundary(Input::default())
            .expect("the release and throw landed");
        for _ in 0..600 {
            booted.session.tick(&Sequential).expect("the tick ran");
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
    fn held_slice_while_spin_paused() {
        const TICKS: usize = 8;

        let scrubbed = |boundaries: usize| {
            let mut booted = one_slot();
            booted.session.app.spin.get_mut().running = false;
            for boundary in 0..boundaries {
                let held = if boundary == 0 || boundaries > 1 {
                    vec![MOVE_UP]
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
                    booted.session.tick(&Sequential).expect("the tick ran");
                }
            }
            *booted.session.app.slice.get()
        };

        assert!((scrubbed(1) - scrubbed(TICKS)).abs() < 1e-6);
        assert!(scrubbed(1) > 0.0);
    }

    #[test]
    fn cancelled_pointer() {
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
}
