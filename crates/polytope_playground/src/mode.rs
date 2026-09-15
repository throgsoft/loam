use loam::math::{Bivector4, EuclideanR4, Plane4, Rotor4};
use loam::runtime::{
    Dispatch, DomainHandle, Domains, EdgeShading, Entity, Instance, Pose, Rejection, SpawnBundle,
};

use crate::catalog::ShapeEntry;
use crate::consts::BASE_ROTATION_RATE;
use crate::strip::{Strip, MAX_CELLS, MAX_T_EXTENT, MIN_CELLS, MIN_T_EXTENT};
use crate::toy;
use crate::{Card, Playground, Toy};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Rotate,
    Toybox,
}

impl Mode {
    pub(crate) const ALL: [Mode; 2] = [Mode::Rotate, Mode::Toybox];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Rotate => "rotate",
            Mode::Toybox => "toybox",
        }
    }

    pub(crate) fn from_token(token: &str) -> Option<Self> {
        Mode::ALL.into_iter().find(|mode| mode.name() == token)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Spin {
    pub(crate) planes: [bool; 6],
    pub(crate) rate: f32,
    pub(crate) running: bool,
}

impl Default for Spin {
    fn default() -> Self {
        Self {
            planes: [false, false, true, false, false, false],
            rate: 1.0,
            running: false,
        }
    }
}

impl Spin {
    pub(crate) fn omega(&self) -> Bivector4 {
        Plane4::ALL
            .into_iter()
            .enumerate()
            .filter(|(index, _)| self.planes[*index])
            .fold(Bivector4::ZERO, |sum, (_, plane)| {
                sum + plane.unit_bivector() * (BASE_ROTATION_RATE * self.rate)
            })
    }
}

pub(crate) fn set_mode(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    cards: &[Card],
    mode: Mode,
) -> Result<(), Rejection> {
    let held = *dispatch.app.mode.get();
    if held == mode {
        return Ok(());
    }
    match (held, mode) {
        (_, Mode::Toybox) => toy::populate(dispatch, domain, cards)?,
        (Mode::Toybox, _) => toy::clear(dispatch, domain)?,
        _ => {}
    }
    dispatch.app.strip.get_mut().on = false;
    dispatch.app.display.get_mut().single = false;
    dispatch.app.active.set(0);
    dispatch.app.mode.set(mode);
    dispatch.app.spin.get_mut().running = false;
    Ok(())
}

pub(crate) fn reset(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    cards: &[Card],
) -> Result<(), Rejection> {
    if *dispatch.app.mode.get() == Mode::Toybox {
        toy::reset(dispatch, domain, cards)?;
    } else {
        let r4 = dispatch.domains.typed(domain)?;
        for (entity, slot) in dispatch.app.slots.iter() {
            if r4.poses().contains(entity) {
                r4.set_pose(entity, Pose::at(slot.rest))?;
            }
        }
    }
    dispatch.app.time.set(0.0);
    dispatch.app.angles.set([0.0; 6]);
    dispatch.app.spin.set(Spin::default());
    dispatch.app.slice.set(0.0);
    Ok(())
}

pub(crate) fn throw_toy(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    entity: Entity,
    velocity: [f32; 3],
) -> Result<(), Rejection> {
    if !dispatch.app.toys.contains(entity) {
        return Ok(());
    }
    let physics = dispatch
        .domains
        .typed(domain)?
        .physics_mut()
        .ok_or(Rejection::Unsupported("the domain has no physics"))?;
    toy::release(physics, entity, velocity).map_err(Rejection::Edit)
}

pub(crate) fn toggle_plane(
    dispatch: &mut Dispatch<'_, Playground>,
    plane: usize,
) -> Result<(), Rejection> {
    let held = *dispatch
        .app
        .spin
        .get()
        .planes
        .get(plane)
        .ok_or(Rejection::Unsupported("no such plane"))?;
    let elapsed = *dispatch.app.time.get() * BASE_ROTATION_RATE;
    let angle = dispatch
        .app
        .angles
        .get_mut()
        .get_mut(plane)
        .ok_or(Rejection::Unsupported("no such plane"))?;
    *angle += if held { elapsed } else { -elapsed };
    dispatch.app.spin.get_mut().planes[plane] = !held;
    Ok(())
}

pub(crate) fn turn_row(
    app: &Playground,
    domains: &mut Domains,
    domain: DomainHandle<EuclideanR4>,
    rotor: Rotor4,
) -> Result<(), Rejection> {
    if *app.mode.get() != Mode::Rotate {
        return Err(Rejection::Unsupported("rotation handles belong to Rotate"));
    }
    let entities = app.slots.iter().map(|(entity, _)| entity);
    let r4 = domains.typed(domain)?;
    for entity in entities {
        if let Some(mut pose) = r4.poses().get(entity).copied() {
            pose.frame = (rotor * pose.frame).normalize();
            r4.set_pose(entity, pose)?;
        }
    }
    Ok(())
}

pub(crate) fn set_shape(
    dispatch: &mut Dispatch<'_, Playground>,
    domain: DomainHandle<EuclideanR4>,
    slot: usize,
    entry: ShapeEntry,
    card: Card,
) -> Result<(), Rejection> {
    let held = dispatch
        .app
        .slots
        .iter()
        .find(|(_, held)| held.index == slot)
        .map(|(entity, held)| (entity, *held))
        .ok_or(Rejection::Unsupported("no such slot"))?;
    let (entity, mut row) = held;
    if row.entry == entry {
        return Ok(());
    }
    let is_toy = dispatch.app.toys.contains(entity);
    let polytope = if is_toy {
        Some(
            entry
                .collider_polytope()
                .ok_or(Rejection::Unsupported("the toy has no collider"))?,
        )
    } else {
        None
    };
    let geometry = if is_toy {
        Some(
            card.toy_geometry
                .ok_or(Rejection::Unsupported("the toy has no prepared geometry"))?,
        )
    } else {
        card.geometry
    };
    let frame = dispatch
        .domains
        .typed(domain)?
        .poses()
        .get(entity)
        .map_or(Rotor4::IDENTITY, |pose| pose.frame);
    let pose = polytope.map_or_else(
        || Pose {
            point: row.rest,
            frame,
        },
        |polytope| toy::pose_at(polytope, row.rest.x),
    );
    row.entry = entry;
    row.rest = pose.point;
    dispatch.despawn(entity)?;
    let mut bundle = SpawnBundle::new().at(domain, pose).row(row);
    if is_toy {
        bundle = bundle.row(Toy {
            rest_anchor: pose.point,
            rest_time: 0.0,
        });
    }
    if let Some(geometry) = geometry {
        let display = *dispatch.app.display.get();
        let mode = *dispatch.app.color.get();
        let shading = card
            .shades
            .map_or(EdgeShading::Material, |shades| shades.of(mode));
        bundle = bundle.instance(
            Instance::new(geometry, card.material)
                .shaded(shading)
                .sectioned(card.cut)
                .line_style(display.wireframe_width_px, display.wireframe_opacity),
        );
    }
    let spawned = dispatch.spawn(bundle)?;
    if let Some(polytope) = polytope {
        toy::add_body(dispatch, domain, spawned, polytope, pose)?;
    }
    Ok(())
}

pub(crate) fn set_strip(app: &mut Playground, mut strip: Strip) -> Result<(), Rejection> {
    if strip.on && *app.mode.get() == Mode::Toybox {
        return Err(Rejection::Unsupported("filmstrip belongs to Rotate"));
    }
    if !strip.t_extent.is_finite() {
        return Err(Rejection::Unsupported("the t extent is not finite"));
    }
    if !strip.w && !strip.t {
        return Err(Rejection::Unsupported("the strip needs one axis"));
    }
    strip.count_w = strip.count_w.clamp(MIN_CELLS, MAX_CELLS);
    strip.count_t = strip.count_t.clamp(MIN_CELLS, MAX_CELLS);
    strip.t_extent = strip.t_extent.clamp(MIN_T_EXTENT, MAX_T_EXTENT);
    app.strip.set(strip);
    Ok(())
}
