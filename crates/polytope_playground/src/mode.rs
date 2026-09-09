use loam_math::{Bivector4, EuclideanR4, Plane4};
use loam_runtime::{AppCommand, Dispatch, DomainHandle, Outcome, Rejection};

use crate::consts::{BASE_ROTATION_RATE, W_RANGE};
use crate::toy;
use crate::Playground;

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
            planes: [false, false, false, false, false, true],
            rate: 1.0,
            running: true,
        }
    }
}

impl Spin {
    /// `Plane4::ALL` product order, summed; the row turns by its exponential each step.
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
        if *dispatch.app.mode.get() == self.mode {
            return Ok(Outcome::Done);
        }
        match self.mode {
            Mode::Rotate => toy::clear(dispatch, self.domain)?,
            Mode::Toybox => toy::populate(dispatch, self.domain)?,
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
