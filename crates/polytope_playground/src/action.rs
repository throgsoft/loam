use loam::prelude::*;

use crate::display;
use crate::mode::{self, slice_reach, Mode};
use crate::projection::Family;
use crate::strip::Strip;
use crate::{catalog, row, Playground, DOMAIN_NAME};

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
    Shape(usize, usize),
    AddShape(usize),
    RemoveShape(usize),
    ReorderShape { from: usize, to: usize },
    Strip(Strip),
    View { single: bool },
    Reset,
    Time(f32),
    PlaneAngle(usize, f32),
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
            Self::Shape(_, card) => catalog::SHAPE_CATALOG
                .get(*card)
                .map_or("shape", |entry| entry.label),
            Self::AddShape(_) => "add shape",
            Self::RemoveShape(_) => "remove shape",
            Self::ReorderShape { .. } => "reorder shape",
            Self::Strip(_) => "strip",
            Self::View { .. } => "view",
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
            Action::Strip(strip) => mode::set_strip(dispatch.app, strip)?,
            Action::View { single } => {
                dispatch.app.display.get_mut().single = single;
                let strip = *dispatch.app.strip.get();
                if strip.on {
                    mode::set_strip(dispatch.app, Strip { on: false, ..strip })?;
                }
            }
            Action::Reset => mode::reset(dispatch, domain, &cards)?,
            Action::Time(time) => {
                if !time.is_finite() || time < 0.0 {
                    return Err(Rejection::Unsupported(
                        "playback time must be finite and nonnegative",
                    ));
                }
                if *dispatch.app.mode.get() == Mode::Toybox {
                    return Err(Rejection::Unsupported("playback belongs to Rotate"));
                }
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
