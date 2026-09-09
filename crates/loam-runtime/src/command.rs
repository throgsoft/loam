use crate::domain::{
    ChartCommand, ChartPose, DomainError, DomainHandle, DomainId, DomainSpace, Domains, Instance,
    Pose,
};
use crate::entity::{Entities, EntitiesSnapshot, Entity, SceneId};
use crate::session::RestoreError;
use crate::store::StoreError;
use crate::stores::{HasStore, Stores};
use crate::view::Views;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct RequestId(u64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Done,
    Spawned(Entity),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejection {
    Stale(Entity),
    Reserved(Entity),
    Capacity,
    Domain(DomainError),
    Store(StoreError),
    Restore(RestoreError),
    Cancelled,
    Unsupported(&'static str),
}

impl From<DomainError> for Rejection {
    fn from(error: DomainError) -> Self {
        Self::Domain(error)
    }
}

impl From<StoreError> for Rejection {
    fn from(error: StoreError) -> Self {
        Self::Store(error)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CommandResult {
    pub request: RequestId,
    pub outcome: Result<Outcome, Rejection>,
}

/// A simulation-time spawn; the entity resolves only after the boundary commits it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reservation {
    pub request: RequestId,
    pub entity: Entity,
}

/// App-defined; UI and scripts reach the session through these and the engine commands.
pub trait AppCommand<A>: Send + 'static {
    fn name(&self) -> &'static str;

    fn apply(&mut self, dispatch: &mut Dispatch<'_, A>) -> Result<Outcome, Rejection>;
}

pub enum Command<A> {
    Spawn(SpawnBundle<A>),
    Despawn(Entity),
    Chart(DomainId, ChartCommand),
    App(Box<dyn AppCommand<A>>),
    Reset,
}

pub struct Request<A> {
    pub id: RequestId,
    pub command: Command<A>,
}

/// Deferred to the next boundary; a result arrives before the next entry that can observe it.
pub struct Commands<A> {
    queue: Vec<Request<A>>,
    entities: Entities,
    next: u64,
}

impl<A: Stores> Commands<A> {
    pub(crate) fn new(scene: SceneId) -> Self {
        Self {
            queue: Vec::new(),
            entities: Entities::new(scene),
            next: 0,
        }
    }

    pub(crate) fn entities(&self) -> &Entities {
        &self.entities
    }

    pub(crate) fn entities_mut(&mut self) -> &mut Entities {
        &mut self.entities
    }

    pub(crate) fn drain_into(&mut self, into: &mut Vec<Request<A>>) {
        into.append(&mut self.queue);
    }

    pub(crate) fn cancel_into(&mut self, results: &mut Vec<CommandResult>) {
        for request in self.queue.drain(..) {
            if let Command::Spawn(SpawnBundle {
                reserved: Some(entity),
                ..
            }) = &request.command
            {
                self.entities.release(*entity);
            }
            results.push(CommandResult {
                request: request.id,
                outcome: Err(Rejection::Cancelled),
            });
        }
    }

    pub(crate) fn next_request(&self) -> RequestId {
        RequestId(self.next)
    }

    pub(crate) fn restore(&mut self, entities: &EntitiesSnapshot, next: RequestId) {
        self.entities.restore(entities);
        self.next = next.0;
    }

    pub fn submit(&mut self, command: Command<A>) -> RequestId {
        let id = RequestId(self.next);
        self.next += 1;
        self.queue.push(Request { id, command });
        id
    }

    pub fn app(&mut self, command: impl AppCommand<A>) -> RequestId {
        self.submit(Command::App(Box::new(command)))
    }

    /// A command with no data of its own; one that carries data implements `AppCommand`.
    pub fn app_fn(
        &mut self,
        name: &'static str,
        apply: impl FnMut(&mut Dispatch<'_, A>) + Send + 'static,
    ) -> RequestId {
        self.app(FnCommand { name, apply })
    }

    pub fn spawn(&mut self, mut bundle: SpawnBundle<A>) -> Result<Reservation, Rejection> {
        let entity = self.entities.reserve();
        bundle.reserved = Some(entity);
        let request = self.submit(Command::Spawn(bundle));
        Ok(Reservation { request, entity })
    }

    pub fn pending(&self) -> &[Request<A>] {
        &self.queue
    }

    pub fn len(&self) -> usize {
        self.queue.len()
    }

    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

struct FnCommand<F> {
    name: &'static str,
    apply: F,
}

impl<A, F> AppCommand<A> for FnCommand<F>
where
    F: FnMut(&mut Dispatch<'_, A>) + Send + 'static,
{
    fn name(&self) -> &'static str {
        self.name
    }

    fn apply(&mut self, dispatch: &mut Dispatch<'_, A>) -> Result<Outcome, Rejection> {
        (self.apply)(dispatch);
        Ok(Outcome::Done)
    }
}

trait Place: Send {
    fn place(
        &self,
        domains: &mut Domains,
        entity: Entity,
        instance: Option<Instance>,
    ) -> Result<DomainId, Rejection>;
}

struct TypedPlace<S: DomainSpace> {
    domain: DomainHandle<S>,
    pose: Pose<S>,
}

impl<S: DomainSpace> Place for TypedPlace<S> {
    fn place(
        &self,
        domains: &mut Domains,
        entity: Entity,
        instance: Option<Instance>,
    ) -> Result<DomainId, Rejection> {
        let domain = domains.typed(self.domain)?;
        domain.poses.insert(entity, self.pose)?;
        if let Some(instance) = instance {
            domain.instances.insert(entity, instance)?;
        }
        Ok(self.domain.id())
    }
}

struct ChartPlace {
    domain: DomainId,
    pose: ChartPose,
}

impl Place for ChartPlace {
    fn place(
        &self,
        domains: &mut Domains,
        entity: Entity,
        instance: Option<Instance>,
    ) -> Result<DomainId, Rejection> {
        let domain = domains
            .facade(self.domain)
            .ok_or(DomainError::UnknownDomain(self.domain))?;
        domain.apply(&ChartCommand::Place {
            entity,
            pose: self.pose,
        })?;
        if let Some(instance) = instance {
            domain.apply(&ChartCommand::Attach { entity, instance })?;
        }
        Ok(self.domain)
    }
}

trait Attach<A>: Send {
    fn attach(self: Box<Self>, app: &mut A, entity: Entity) -> Result<(), StoreError>;
}

struct Row<T>(T);

impl<A: HasStore<T>, T: Send + 'static> Attach<A> for Row<T> {
    fn attach(self: Box<Self>, app: &mut A, entity: Entity) -> Result<(), StoreError> {
        app.store_mut().insert(entity, self.0)
    }
}

/// One entity and its initial attachments, accepted or rejected together.
pub struct SpawnBundle<A> {
    placement: Option<Box<dyn Place>>,
    instance: Option<Instance>,
    rows: Vec<Box<dyn Attach<A>>>,
    reserved: Option<Entity>,
}

impl<A: Stores> Default for SpawnBundle<A> {
    fn default() -> Self {
        Self::new()
    }
}

impl<A: Stores> SpawnBundle<A> {
    pub fn new() -> Self {
        Self {
            placement: None,
            instance: None,
            rows: Vec::new(),
            reserved: None,
        }
    }

    pub fn at<S: DomainSpace>(mut self, domain: DomainHandle<S>, pose: Pose<S>) -> Self {
        self.placement = Some(Box::new(TypedPlace { domain, pose }));
        self
    }

    pub fn at_chart(mut self, domain: DomainId, pose: ChartPose) -> Self {
        self.placement = Some(Box::new(ChartPlace { domain, pose }));
        self
    }

    pub fn instance(mut self, instance: Instance) -> Self {
        self.instance = Some(instance);
        self
    }

    pub fn row<T: Send + 'static>(mut self, row: T) -> Self
    where
        A: HasStore<T>,
    {
        self.rows.push(Box::new(Row(row)));
        self
    }
}

/// Immediate application at a boundary; every handle it returns is live.
pub struct Dispatch<'a, A> {
    pub app: &'a mut A,
    pub domains: &'a mut Domains,
    pub views: &'a mut Views,
    entities: &'a mut Entities,
}

impl<'a, A: Stores> Dispatch<'a, A> {
    pub(crate) fn new(
        app: &'a mut A,
        domains: &'a mut Domains,
        views: &'a mut Views,
        entities: &'a mut Entities,
    ) -> Self {
        Self {
            app,
            domains,
            views,
            entities,
        }
    }

    pub fn spawn(&mut self, bundle: SpawnBundle<A>) -> Result<Entity, Rejection> {
        let entity = match bundle.reserved {
            Some(entity) if self.entities.is_reserved(entity) => entity,
            Some(entity) => return Err(Rejection::Stale(entity)),
            None => self.entities.reserve(),
        };
        match self.attach(entity, bundle) {
            Ok(()) => {
                self.entities.commit(entity);
                Ok(entity)
            }
            Err(rejection) => {
                self.detach(entity);
                self.entities.release(entity);
                Err(rejection)
            }
        }
    }

    fn attach(&mut self, entity: Entity, bundle: SpawnBundle<A>) -> Result<(), Rejection> {
        if let Some(placement) = &bundle.placement {
            placement.place(self.domains, entity, bundle.instance)?;
        }
        for row in bundle.rows {
            row.attach(self.app, entity)?;
        }
        Ok(())
    }

    fn detach(&mut self, entity: Entity) {
        self.app.release(entity);
        for domain in self.domains.iter_mut() {
            domain.release(entity);
        }
    }

    pub fn despawn(&mut self, entity: Entity) -> Result<(), Rejection> {
        if self.entities.is_reserved(entity) {
            return Err(Rejection::Reserved(entity));
        }
        if self.entities.resolve(entity).is_none() {
            return Err(Rejection::Stale(entity));
        }
        self.detach(entity);
        self.entities.despawn(entity)?;
        Ok(())
    }

    pub fn apply(&mut self, command: Command<A>) -> Result<Outcome, Rejection> {
        match command {
            Command::Spawn(bundle) => self.spawn(bundle).map(Outcome::Spawned),
            Command::Despawn(entity) => self.despawn(entity).map(|()| Outcome::Done),
            Command::Chart(domain, command) => self
                .domains
                .facade(domain)
                .ok_or(DomainError::UnknownDomain(domain))?
                .apply(&command),
            Command::App(mut command) => command.apply(self),
            Command::Reset => Err(Rejection::Unsupported(
                "reset applies at a session boundary",
            )),
        }
    }

    pub fn entities(&self) -> &Entities {
        self.entities
    }
}
