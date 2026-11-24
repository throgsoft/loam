use std::any::Any;

use loam_time::par::Executor;

use super::{ChartCommand, DomainError, DomainSpace, Pose};
use crate::command::{Outcome, Rejection};
use crate::entity::{Entity, SceneId};
use crate::phase::Step;
use crate::session::RestoreError;
use crate::store::{Owner, Store};
use crate::view::ViewMapping;

/// Engine-owned work inside one domain: `step` runs in the simulation phase's domain-step entry, `release` at dispatch before the stores forget the entity, and `apply` gets first offer of a chart command.
pub trait Facility<S: DomainSpace>: Any + Send + 'static {
    fn name(&self) -> &'static str;

    fn bind(&mut self, _scene: SceneId, _owner: Owner) {}

    fn step(
        &mut self,
        poses: &mut Store<Pose<S>>,
        step: Step,
        executor: &dyn Executor,
        owner: Owner,
    ) -> Result<(), DomainError>;

    fn set_pose(
        &mut self,
        _entity: Entity,
        _pose: Pose<S>,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        None
    }

    fn move_to(
        &mut self,
        _entity: Entity,
        _point: S::Point,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<(), DomainError>> {
        None
    }

    fn synchronize(&mut self, _poses: &mut Store<Pose<S>>, _owner: Owner) {}

    fn snapshot(&self, owner: Owner) -> Box<dyn Any + Send>;

    /// Refuses whatever `restore` would refuse, changing nothing; the session calls it on every facility before the first one restores.
    fn check_restore(&self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError>;

    fn restore(&mut self, from: &(dyn Any + Send), owner: Owner) -> Result<(), RestoreError>;

    fn release(&mut self, _entity: Entity, _owner: Owner) {}

    /// `Some` claims the command with its outcome; `None` leaves it to the domain.
    fn apply(
        &mut self,
        _command: &ChartCommand,
        _poses: &mut Store<Pose<S>>,
        _owner: Owner,
    ) -> Option<Result<Outcome, Rejection>> {
        None
    }

    /// `Some` claims a [`ChartCommand::Throw`] made through the view that `mapping` draws from `eye`.
    fn throw(
        &mut self,
        _entity: Entity,
        _velocity: S::Vector,
        _mapping: &dyn ViewMapping<S>,
        _eye: &Pose<S>,
    ) -> Option<Result<Outcome, Rejection>> {
        None
    }
}
