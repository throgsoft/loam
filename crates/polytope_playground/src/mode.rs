use loam_math::Iso4Flat;
use loam_math::{Bivector, Bivector4, EuclideanR4, Plane4, Rotor, Rotor4};
use loam_runtime::{
    AppCommand, Dispatch, DomainHandle, EdgeShading, Instance, MaterialId, Outcome, Pose,
    PreparedId, Rejection, SpawnBundle,
};

use crate::catalog::ShapeEntry;
use crate::color::{ColorMode, Shades};
use crate::composer::Term;
use crate::consts::{BASE_ROTATION_RATE, W_RANGE};
use crate::projection::Family;
use crate::toy;
use crate::Playground;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Rotate,
    Compose,
    Toybox,
}

impl Mode {
    pub(crate) const ALL: [Mode; 3] = [Mode::Rotate, Mode::Compose, Mode::Toybox];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Mode::Rotate => "rotate",
            Mode::Compose => "compose",
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
            planes: [false, false, false, false, false, true],
            rate: 1.0,
            running: true,
        }
    }
}

impl Spin {
    pub(crate) fn omega(&self) -> Bivector4 {
        if !self.running {
            return Bivector4::ZERO;
        }
        Plane4::ALL
            .into_iter()
            .enumerate()
            .filter(|(index, _)| self.planes[*index])
            .fold(Bivector4::ZERO, |sum, (_, plane)| {
                sum + plane.unit_bivector() * (BASE_ROTATION_RATE * self.rate)
            })
    }
}

pub(crate) struct SetMode {
    pub(crate) mode: Mode,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

impl AppCommand<Playground> for SetMode {
    fn name(&self) -> &'static str {
        self.mode.name()
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let held = *dispatch.app.mode.get();
        if held == self.mode {
            return Ok(Outcome::Done);
        }
        match (held, self.mode) {
            (_, Mode::Toybox) => toy::populate(dispatch, self.domain)?,
            (Mode::Toybox, _) => toy::clear(dispatch, self.domain)?,
            _ => {}
        }
        dispatch.app.mode.set(self.mode);
        Ok(Outcome::Done)
    }
}

pub(crate) struct SetActive {
    pub(crate) slot: usize,
}

impl AppCommand<Playground> for SetActive {
    fn name(&self) -> &'static str {
        "active"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if self.slot >= dispatch.app.slots.len() {
            return Err(Rejection::Unsupported("no such slot"));
        }
        dispatch.app.active.set(self.slot);
        Ok(Outcome::Done)
    }
}

pub(crate) struct SetSlice {
    pub(crate) w: f32,
}

impl AppCommand<Playground> for SetSlice {
    fn name(&self) -> &'static str {
        "slice"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if !self.w.is_finite() {
            return Err(Rejection::Unsupported("slice is not finite"));
        }
        dispatch.app.slice.set(self.w.clamp(-W_RANGE, W_RANGE));
        Ok(Outcome::Done)
    }
}

pub(crate) struct TogglePlane {
    pub(crate) plane: usize,
}

impl AppCommand<Playground> for TogglePlane {
    fn name(&self) -> &'static str {
        "plane"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let spin = dispatch.app.spin.get_mut();
        let held = spin
            .planes
            .get_mut(self.plane)
            .ok_or(Rejection::Unsupported("no such plane"))?;
        *held = !*held;
        Ok(Outcome::Done)
    }
}

pub(crate) struct SetRunning {
    pub(crate) running: bool,
}

impl AppCommand<Playground> for SetRunning {
    fn name(&self) -> &'static str {
        "spin"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        dispatch.app.spin.get_mut().running = self.running;
        Ok(Outcome::Done)
    }
}

pub(crate) struct SetProjection {
    pub(crate) family: Family,
}

impl AppCommand<Playground> for SetProjection {
    fn name(&self) -> &'static str {
        self.family.name()
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        dispatch.app.projection.set(self.family);
        Ok(Outcome::Done)
    }
}

pub(crate) struct PushTerm {
    pub(crate) term: Term,
}

impl AppCommand<Playground> for PushTerm {
    fn name(&self) -> &'static str {
        "term"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if dispatch.app.composer.get_mut().push(self.term) {
            Ok(Outcome::Done)
        } else {
            Err(Rejection::Unsupported("the sequence is empty or full"))
        }
    }
}

pub(crate) struct DropTerm {
    pub(crate) index: usize,
}

impl AppCommand<Playground> for DropTerm {
    fn name(&self) -> &'static str {
        "drop term"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if dispatch.app.composer.get_mut().remove(self.index) {
            Ok(Outcome::Done)
        } else {
            Err(Rejection::Unsupported("no such term"))
        }
    }
}

pub(crate) struct DraftPlane {
    pub(crate) plane: usize,
}

impl AppCommand<Playground> for DraftPlane {
    fn name(&self) -> &'static str {
        "draft"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let composer = dispatch.app.composer.get_mut();
        let slot = composer
            .draft
            .get_mut(self.plane)
            .ok_or(Rejection::Unsupported("no such plane"))?;
        *slot = slot.saturating_add(1);
        Ok(Outcome::Done)
    }
}

pub(crate) struct ClearDraft;

impl AppCommand<Playground> for ClearDraft {
    fn name(&self) -> &'static str {
        "clear draft"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        dispatch.app.composer.get_mut().draft = [0; 6];
        Ok(Outcome::Done)
    }
}

pub(crate) struct CommitDraft;

impl AppCommand<Playground> for CommitDraft {
    fn name(&self) -> &'static str {
        "commit"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if dispatch.app.composer.get_mut().commit_draft() {
            Ok(Outcome::Done)
        } else {
            Err(Rejection::Unsupported(
                "the draft is empty or the sequence is full",
            ))
        }
    }
}

pub(crate) struct ClearComposer;

impl AppCommand<Playground> for ClearComposer {
    fn name(&self) -> &'static str {
        "clear"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        dispatch.app.composer.get_mut().clear();
        Ok(Outcome::Done)
    }
}

/// Moves each slot's rotor along the sequence's unit bivector and leaves the components across it alone.
pub(crate) struct SetScrub {
    pub(crate) scrub: f32,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

impl AppCommand<Playground> for SetScrub {
    fn name(&self) -> &'static str {
        "scrub"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        if !self.scrub.is_finite() {
            return Err(Rejection::Unsupported("scrub is not finite"));
        }
        let axis = dispatch
            .app
            .composer
            .get()
            .axis()
            .ok_or(Rejection::Unsupported("the sequence names no bivector"))?;
        dispatch.app.composer.get_mut().scrub = self.scrub;
        let entities = dispatch.app.slots.iter().map(|(entity, _)| entity);
        let r4 = dispatch.domains.typed(self.domain)?;
        for entity in entities {
            if let Some(pose) = r4.poses.get_mut(entity) {
                let held = pose.0.rotation.log();
                let turned = held + axis * (self.scrub - held.dot(axis));
                pose.0.rotation = turned.exp().normalize();
            }
        }
        Ok(Outcome::Done)
    }
}

pub(crate) struct TurnRow {
    pub(crate) rotor: Rotor4,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

impl AppCommand<Playground> for TurnRow {
    fn name(&self) -> &'static str {
        "turn"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let entities = dispatch.app.slots.iter().map(|(entity, _)| entity);
        let r4 = dispatch.domains.typed(self.domain)?;
        for entity in entities {
            if let Some(pose) = r4.poses.get_mut(entity) {
                pose.0.rotation = (self.rotor * pose.0.rotation).normalize();
            }
        }
        Ok(Outcome::Done)
    }
}

pub(crate) struct ToggleGimbal;

impl AppCommand<Playground> for ToggleGimbal {
    fn name(&self) -> &'static str {
        "gimbal"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let shown = *dispatch.app.gimbal.get();
        dispatch.app.gimbal.set(!shown);
        Ok(Outcome::Done)
    }
}

pub(crate) struct SetColorMode {
    pub(crate) mode: ColorMode,
}

impl AppCommand<Playground> for SetColorMode {
    fn name(&self) -> &'static str {
        self.mode.name()
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        dispatch.app.color.set(self.mode);
        Ok(Outcome::Done)
    }
}

pub(crate) struct TogglePoints;

impl AppCommand<Playground> for TogglePoints {
    fn name(&self) -> &'static str {
        "points"
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let shown = *dispatch.app.points.get();
        dispatch.app.points.set(!shown);
        Ok(Outcome::Done)
    }
}

/// Despawns the row's slot and spawns its replacement in place, with the toybox body when that mode is live.
pub(crate) struct SetShape {
    pub(crate) slot: usize,
    pub(crate) entry: ShapeEntry,
    pub(crate) geometry: Option<PreparedId>,
    pub(crate) material: MaterialId,
    pub(crate) shades: Option<Shades>,
    pub(crate) cut: MaterialId,
    pub(crate) domain: DomainHandle<EuclideanR4>,
}

impl AppCommand<Playground> for SetShape {
    fn name(&self) -> &'static str {
        self.entry.label
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, Playground>) -> Result<Outcome, Rejection> {
        let held = dispatch
            .app
            .slots
            .iter()
            .find(|(_, slot)| slot.index == self.slot)
            .map(|(entity, slot)| (entity, *slot))
            .ok_or(Rejection::Unsupported("no such slot"))?;
        let (entity, mut row) = held;
        if row.entry == self.entry {
            return Ok(Outcome::Done);
        }
        row.entry = self.entry;
        dispatch.despawn(entity)?;
        let mut bundle = SpawnBundle::new()
            .at(self.domain, Pose(Iso4Flat::from_translation(row.rest)))
            .row(row);
        if let Some(geometry) = self.geometry {
            let mode = *dispatch.app.color.get();
            let shading = self
                .shades
                .map_or(EdgeShading::Material, |shades| shades.of(mode));
            bundle = bundle.instance(
                Instance::new(geometry, self.material)
                    .shaded(shading)
                    .sectioned(self.cut),
            );
        }
        let spawned = dispatch.spawn(bundle)?;
        if *dispatch.app.mode.get() == Mode::Toybox {
            if let Some(polytope) = self.entry.collider_polytope() {
                toy::add_body(dispatch, self.domain, spawned, polytope, row.rest)?;
            }
        }
        Ok(Outcome::Done)
    }
}
