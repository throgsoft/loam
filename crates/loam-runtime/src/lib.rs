//! One session's CPU state: entities, typed stores, domains behind one facade, phases, commands.

pub mod command;
pub mod domain;
pub mod entity;
pub mod host;
pub mod input;
pub mod phase;
pub mod relation;
pub mod session;
pub mod store;
pub mod stores;
pub mod value;
pub mod view;

pub use command::{
    AppCommand, Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
    Reservation, SpawnBundle,
};
pub use domain::{
    ChartCommand, ChartId, ChartPose, ChartTangent, Domain, DomainBuilder, DomainError,
    DomainHandle, DomainId, DomainSnapshot, DomainSpace, Domains, Facility, Field, FieldKind,
    FieldProgram, Instance, Pose, TypedDomain,
};
pub use entity::{Entities, EntitiesSnapshot, Entity, EntityKey, Epoch, RuntimeId, SceneId};
pub use host::{HostConfig, HostError};
pub use input::{ActionEvent, ActionId, Bindings, Input, Key, Pointer, PointerPhase};
pub use phase::{
    Access, Ctx, Entry, EntryId, Phase, Readback, Schedule, Step, StoreId, System, SystemEntry,
    Tick, WorkItem,
};
pub use relation::{Endpoints, Link, LinkId, Relation, RelationSnapshot};
pub use session::{
    Material, MaterialId, PreparedGeometry, PreparedId, Publication, PublishedView, RestoreError,
    Session, SessionSnapshot, SimConfig, Stamp,
};
pub use store::{
    Change, Changes, Cursor, ErasedStore, LogCapacity, Part, Partition, PartitionError, Parts,
    Publish, RecordBuffer, Removal, SchemaId, Store, StoreError, StoreField, StoreSnapshot,
    Version, DEFAULT_LOG_CAPACITY,
};
pub use stores::{HasRelation, HasStore, Stores};
pub use value::Value;
pub use view::{
    DepthEnvelope, DomainRay, Eye, Identity3, ImageRay, ImageSpace, ImageSpaceId, InstanceRecord,
    Klein, Pick, Projection4, Section4, ViewId, ViewMapping, ViewRecords, ViewSpec, ViewTarget,
    Views,
};
