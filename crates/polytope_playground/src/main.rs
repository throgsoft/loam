use std::sync::Arc;

use loam::prelude::*;

mod action;
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
mod shown;
mod strip;
mod toy;
mod ui;

use action::Action;
use camera::{focus_camera, home, scene_center};
use catalog::ShapeEntry;
use color::{ColorMode, Shades};
use consts::{BODY_SIZE, BODY_X_SPACING, BODY_Y, W_SCRUB_RATE};
use display::{Display, Surface};
use gimbal::Gimbal;
use mode::{slice_reach, Mode, Spin};
use projection::Family;
use shown::Shown;
use strip::{fill_strip, Scratch, Strip};

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

pub(crate) const SHOW_CONTROLS: Setting<Playground, bool> =
    Setting::flag("controls", |app| app.controls.get_mut());
pub(crate) const SHOW_FORMULA: Setting<Playground, bool> =
    Setting::flag("formula", |app| app.formula.get_mut());
pub(crate) const SHOW_GUIDES: Setting<Playground, bool> =
    Setting::flag("guides", |app| app.guides.get_mut());
pub(crate) const SHOW_GIMBAL: Setting<Playground, bool> =
    Setting::flag("handles", |app| app.gimbal.get_mut());
pub(crate) const TOYBOX_DEBUG: Setting<Playground, bool> =
    Setting::flag("toybox debug", |app| app.toybox_debug.get_mut());
pub(crate) const SHOW_HUD: Setting<Playground, bool> =
    Setting::flag("hud", |app| app.hud.get_mut());
pub(crate) const SHOW_LABELS: Setting<Playground, bool> =
    Setting::flag("labels", |app| app.labels.get_mut());
pub(crate) const ROPE_CARRY: Setting<Playground, bool> =
    Setting::flag("rope carry", |app| app.rope.get_mut());

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
    session.toggle_on(GIMBAL, SHOW_GIMBAL);
    session.toggle_on(CONTROLS, SHOW_CONTROLS);
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

fn main() -> Result<(), HostError> {
    launch(interactive)
}

fn interactive(args: Args) -> Result<(Session<Playground>, SessionApp<Playground>), HostError> {
    let row = catalog::parse_row(&args).map_err(|error| HostError::Host(format!("{error:#}")))?;
    let (mut session, _) = boot(&row)?;
    if let Some(token) = args.flag_value("surface") {
        session.app.display.get_mut().surface = display::SURFACE.parse(token).ok_or_else(|| {
            HostError::Host(format!("--surface {token:?} is not raster, sdf or off"))
        })?;
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

fn readout_of(app: &Playground) -> hud::Readout {
    hud::Readout {
        slice: *app.slice.get(),
        rate: app.spin.get().rate,
        bodies: app.slots.len(),
        planes: app.spin.get().planes,
        toybox: *app.mode.get() == Mode::Toybox,
    }
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
            assert_ne!(pose.frame, Rotor4::IDENTITY);
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
            assert_eq!(pose.frame, Rotor4::IDENTITY);
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

    #[test]
    fn refused_display_edits() {
        let mut booted = one_slot();
        booted.session.submit(display::WIREFRAME_WIDTH.set(3.0));
        booted.session.boundary(Input::default()).expect("boundary");
        let accepted = *booted.session.app.display.get();
        assert_eq!(
            accepted.wireframe_width_px, 3.0,
            "an in range width was refused"
        );

        let refused = [
            display::WIREFRAME_WIDTH.set(0.0),
            display::WIREFRAME_WIDTH.set(2.0 * display::MAX_WIREFRAME_WIDTH_PX),
            display::WIREFRAME_WIDTH.set(f32::NAN),
            display::WIREFRAME_OPACITY.set(0.0),
            display::WIREFRAME_OPACITY.set(1.5),
            display::WIREFRAME_OPACITY.set(f32::NAN),
        ];
        let count = refused.len();
        for command in refused {
            booted.session.submit(command);
        }
        booted.session.boundary(Input::default()).expect("boundary");

        let rejected = booted
            .session
            .results()
            .iter()
            .filter(|result| result.outcome.is_err())
            .count();
        assert_eq!(rejected, count, "an out of range edit was accepted");
        assert_eq!(*booted.session.app.display.get(), accepted);
    }
}
