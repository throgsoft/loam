//! One session's CPU state: entities, typed stores, domains behind one facade, phases, commands, bulk stores, work orders, checkpoints, and with the `physics` feature a world per domain.

pub mod bridge;
pub mod bulk;
pub mod command;
pub mod domain;
pub mod entity;
pub mod field;
pub mod host;
pub mod input;
pub mod phase;
#[cfg(feature = "physics")]
pub mod physics;
pub mod relation;
pub mod session;
pub mod store;
pub mod stores;
pub mod value;
pub mod view;

pub use bridge::{Bridge, BridgeError, BridgeSpec, Drag, DragError, DragRelease};
pub use bulk::{
    Bulk, BulkAction, BulkCheckpoint, BulkError, BulkId, BulkSnapshot, BulkSpec, Landed, Landing,
    SnapshotPolicy, Wait, WorkOrder, WorkStats,
};
pub use command::{
    AppCommand, Command, CommandResult, Commands, Dispatch, Outcome, Rejection, Request, RequestId,
    Reservation, SpawnBundle,
};
pub use domain::{
    ChartCommand, ChartId, ChartPoint, ChartPose, ChartTangent, Domain, DomainBuilder, DomainError,
    DomainHandle, DomainId, DomainSnapshot, DomainSpace, Domains, Facility, Field, FieldKind,
    FieldProgram, Instance, Pose, TypedDomain,
};
pub use entity::{Entities, EntitiesSnapshot, Entity, EntityKey, Epoch, RuntimeId, SceneId};
pub use field::{
    evaluate, evaluate_bounded, evaluate_bounded_counted, evaluate_counted, FieldCompiler,
    FieldCost, FieldCounts, FieldError, FieldNode, FieldOp, FieldPrimitive, FIELD_FAR,
    FIELD_PROGRAM_ERROR, MAX_POSE_DEPTH, MAX_STACK, OP_BOX, OP_HALFSPACE, OP_HALFSPACE4,
    OP_HYPERSPHERE, OP_INTERSECTION, OP_POP_POSE, OP_PUSH_POSE, OP_SMOOTH_UNION, OP_SPHERE,
    OP_SUBTRACTION, OP_UNION,
};
pub use host::{HostConfig, HostError};
pub use input::{ActionEvent, ActionId, Bindings, Input, Key, Pointer, PointerPhase};
pub use phase::{
    Access, Ctx, Entry, EntryId, Order, Phase, Readback, Schedule, Step, StoreId, System,
    SystemEntry, Tick, WorkItem,
};
#[cfg(feature = "physics")]
pub use physics::{Physics, PhysicsConfig};
pub use relation::{Endpoints, Link, LinkId, Relation, RelationSnapshot};
pub use session::{
    Growth, Library, Material, MaterialId, PreparedGeometry, PreparedId, Publication, PublishError,
    PublishedView, Records, RestoreError, Session, SessionSnapshot, SimConfig, Stamp, DOMAIN_STEP,
};
pub use store::{
    Change, Changes, Cursor, ErasedStore, LogCapacity, Part, Partition, PartitionError, Parts,
    Publish, Published, RecordBuffer, Removal, SchemaId, Store, StoreError, StoreField,
    StoreSnapshot, Version, DEFAULT_LOG_CAPACITY,
};
pub use stores::{HasRelation, HasStore, Stores};
pub use value::Value;
pub use view::{
    DepthEnvelope, DomainRay, Eye, Identity3, ImageRay, ImageSpace, ImageSpaceId, InstanceRecord,
    Klein, Orbit, Pick, Placement, Projection4, Rigid, Section4, SegmentRecord, ViewId,
    ViewMapping, ViewRecords, ViewSpec, ViewSummary, ViewTarget, Views, ViewsSnapshot,
};
