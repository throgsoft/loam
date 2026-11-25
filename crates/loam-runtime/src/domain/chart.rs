use std::sync::Arc;

use super::{put_row, DomainError, DomainSpace, Instance, TypedDomain};
use crate::command::{Outcome, Rejection};
use crate::entity::Entity;
use crate::store::{Owner, StoreError};
use crate::view::ViewId;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChartId(pub u16);

/// Coordinates and a frame in one chart of the domain; unused entries are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartPose {
    pub chart: ChartId,
    pub coordinates: [f32; 4],
    /// Column-major, matching glam's `Mat4` and WGSL's `mat4x4<f32>`.
    pub frame: [[f32; 4]; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartTangent {
    pub chart: ChartId,
    pub vector: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ChartPoint {
    pub chart: ChartId,
    pub coordinates: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ChartCommand {
    Place {
        entity: Entity,
        pose: ChartPose,
    },
    Attach {
        entity: Entity,
        instance: Instance,
    },
    Walk {
        entity: Entity,
        tangent: ChartTangent,
        dt: f32,
    },
    Move {
        entity: Entity,
        point: ChartPoint,
    },
    Grab {
        entity: Entity,
        point: ChartPoint,
    },
    Release {
        entity: Entity,
    },
    /// Sets the part of the body's velocity that `view` shows from `velocity` under the grab's throw configuration.
    Throw {
        entity: Entity,
        view: ViewId,
        velocity: ChartTangent,
    },
}

impl ChartCommand {
    pub(crate) fn entity(&self) -> Entity {
        match *self {
            Self::Place { entity, .. }
            | Self::Attach { entity, .. }
            | Self::Walk { entity, .. }
            | Self::Move { entity, .. }
            | Self::Grab { entity, .. }
            | Self::Release { entity }
            | Self::Throw { entity, .. } => entity,
        }
    }
}

pub(super) fn single_chart(chart: ChartId) -> Result<(), DomainError> {
    (chart == ChartId(0))
        .then_some(())
        .ok_or(DomainError::UnknownChart(chart))
}

pub(super) fn finite(values: [f32; 4]) -> Result<(), DomainError> {
    for (value, axis) in values.into_iter().zip(["x", "y", "z", "w"]) {
        if !value.is_finite() {
            return Err(DomainError::InvalidCoordinate(axis));
        }
    }
    Ok(())
}

impl<S: DomainSpace> TypedDomain<S> {
    pub(super) fn apply_command(&mut self, command: &ChartCommand) -> Result<Outcome, Rejection> {
        let owned = self.poses.contains(command.entity());
        if !owned && !matches!(command, ChartCommand::Place { .. }) {
            return Err(StoreError::Missing(command.entity()).into());
        }
        match command {
            ChartCommand::Place { pose, .. } => {
                self.space.pose_from_chart(pose)?;
            }
            ChartCommand::Walk { tangent, .. } => {
                self.space.tangent_from_chart(tangent)?;
            }
            ChartCommand::Move { point, .. } | ChartCommand::Grab { point, .. } => {
                self.space.point_from_chart(point)?;
            }
            _ => {}
        }
        if owned {
            for facility in &mut self.facilities {
                if let Some(outcome) = facility.apply(command, &mut self.poses, Owner::new()) {
                    return outcome;
                }
            }
        }
        match command {
            ChartCommand::Place { entity, pose } => {
                let placed = self.space.pose_from_chart(pose)?;
                put_row(&mut self.poses, *entity, placed)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Attach { entity, instance } => {
                put_row(&mut self.instances, *entity, *instance)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Walk {
                entity,
                tangent,
                dt,
            } => {
                let velocity = self.space.tangent_from_chart(tangent)?;
                self.walk(*entity, velocity, *dt)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Move { entity, point } => {
                self.move_to(*entity, *point)?;
                Ok(Outcome::Done)
            }
            ChartCommand::Throw {
                entity,
                view,
                velocity,
            } => {
                let velocity = self.space.tangent_from_chart(velocity)?;
                let spec = self
                    .views
                    .get(*view)
                    .ok_or(DomainError::UnknownView(*view))?;
                let mapping = Arc::clone(&spec.style.mapping);
                let eye = self
                    .drawn(spec.eye, self.blend.presented())
                    .ok_or(DomainError::Stale(spec.eye))?;
                for facility in &mut self.facilities {
                    if let Some(outcome) = facility.throw(*entity, velocity, mapping.as_ref(), &eye)
                    {
                        return outcome;
                    }
                }
                Ok(Outcome::Done)
            }
            ChartCommand::Grab { .. } | ChartCommand::Release { .. } => Ok(Outcome::Done),
        }
    }
}
