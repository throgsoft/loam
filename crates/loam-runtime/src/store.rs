use std::any::type_name;
use std::marker::PhantomData;

use crate::entity::{Entity, EntityKey, SceneId};
use crate::session::Stamp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreError {
    Foreign(Entity),
    Occupied(Entity),
    Missing(Entity),
    Capacity,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Version(u64);

impl Version {
    pub fn bump(self) -> Self {
        Self(self.0.wrapping_add(1))
    }
}

/// Retained events; independent of live row count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LogCapacity {
    pub dirty: usize,
    pub removals: usize,
}

pub const DEFAULT_LOG_CAPACITY: LogCapacity = LogCapacity {
    dirty: 4096,
    removals: 1024,
};

impl Default for LogCapacity {
    fn default() -> Self {
        DEFAULT_LOG_CAPACITY
    }
}

/// A consumer's position in a store's logs; the default cursor is stale and forces a resync.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    position: u64,
}

impl Cursor {
    pub fn position(self) -> u64 {
        self.position
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Removal {
    pub entity: Entity,
    pub version: Version,
}

pub enum Change<'a, T> {
    Row(Entity, &'a T),
    Removed(Removal),
}

/// Deltas since the cursor, or every live row and retained removal after a resync.
pub struct Changes<'a, T> {
    resync: bool,
    rows: PhantomData<&'a T>,
}

impl<T> Changes<'_, T> {
    pub fn is_resync(&self) -> bool {
        self.resync
    }
}

impl<'a, T> Iterator for Changes<'a, T> {
    type Item = Change<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

/// Disjoint mutable row ranges of one store, checked once when built.
pub struct Partition<'a, T> {
    rows: &'a mut [T],
    cuts: &'a [usize],
}

impl<'a, T> Partition<'a, T> {
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    pub fn part_count(&self) -> usize {
        self.cuts.len() + 1
    }

    pub fn parts(self) -> Parts<'a, T> {
        todo!()
    }
}

pub struct Part<'a, T> {
    pub first: usize,
    pub rows: &'a mut [T],
}

pub struct Parts<'a, T> {
    rows: PhantomData<&'a mut T>,
}

impl<'a, T> Iterator for Parts<'a, T> {
    type Item = Part<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        todo!()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionError {
    Unsorted { at: usize },
    OutOfRange { cut: usize, len: usize },
}

pub trait Publish: Sized + Send + 'static {
    type Record: Copy + Send + 'static;

    fn record(&self, entity: Entity) -> Self::Record;
}

/// One consumer's copy of a store's records, caught up through its own cursor.
pub struct RecordBuffer<R> {
    rows: Vec<R>,
    cursor: Cursor,
    stamp: Stamp,
}

impl<R> RecordBuffer<R> {
    pub fn rows(&self) -> &[R] {
        &self.rows
    }

    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    pub fn stamp(&self) -> Stamp {
        self.stamp
    }
}

impl<R> Default for RecordBuffer<R> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            cursor: Cursor::default(),
            stamp: Stamp::default(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SchemaId(&'static str);

impl SchemaId {
    pub fn of<T: 'static>() -> Self {
        Self(type_name::<T>())
    }

    pub fn name(self) -> &'static str {
        self.0
    }
}

/// Off the frame path: serialization and tooling see a store without its row type.
pub trait ErasedStore: Send {
    fn schema(&self) -> SchemaId;

    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn is_tracked(&self) -> bool;
}

/// What every field of a `stores!` struct implements.
pub trait StoreField: Send + 'static {
    type Snapshot: Send + 'static;

    fn bind(&mut self, scene: SceneId);

    fn snapshot(&self) -> Self::Snapshot;

    fn restore(&mut self, from: &Self::Snapshot, scene: SceneId);

    fn erased(&mut self) -> &mut dyn ErasedStore;
}

pub struct StoreSnapshot<T> {
    rows: Vec<T>,
    keys: Vec<EntityKey>,
}

impl<T> StoreSnapshot<T> {
    pub fn rows(&self) -> &[T] {
        &self.rows
    }

    pub fn keys(&self) -> &[EntityKey] {
        &self.keys
    }
}

#[derive(Clone, Copy)]
struct SparseEntry {
    generation: u32,
    dense: u32,
}

struct Tracking {
    capacity: LogCapacity,
}

impl Tracking {
    fn touch(&mut self, _dense: usize) {
        todo!()
    }

    fn touch_all(&mut self) {
        todo!()
    }

    fn remove(&mut self, _key: EntityKey) {
        todo!()
    }
}

/// A `Store<T>` that `stores!` publishes; `T: Publish` is checked at the generated call.
pub type Published<T> = Store<T>;

/// Dense rows with a sparse index; a tracked store also keeps versions and bounded logs.
pub struct Store<T> {
    scene: SceneId,
    dense: Vec<T>,
    keys: Vec<EntityKey>,
    sparse: Vec<Option<SparseEntry>>,
    tracking: Option<Tracking>,
}

impl<T> Default for Store<T> {
    fn default() -> Self {
        Self::untracked()
    }
}

impl<T> Store<T> {
    pub fn untracked() -> Self {
        Self::with_tracking(None)
    }

    pub fn tracked(capacity: LogCapacity) -> Self {
        Self::with_tracking(Some(Tracking { capacity }))
    }

    fn with_tracking(tracking: Option<Tracking>) -> Self {
        Self {
            scene: SceneId::UNBOUND,
            dense: Vec::new(),
            keys: Vec::new(),
            sparse: Vec::new(),
            tracking,
        }
    }

    pub fn bind(&mut self, scene: SceneId) {
        self.scene = scene;
    }

    pub fn is_tracked(&self) -> bool {
        self.tracking.is_some()
    }

    pub fn log_capacity(&self) -> Option<LogCapacity> {
        self.tracking.as_ref().map(|tracking| tracking.capacity)
    }

    pub fn scene(&self) -> SceneId {
        self.scene
    }

    pub fn len(&self) -> usize {
        self.dense.len()
    }

    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    pub fn contains(&self, entity: Entity) -> bool {
        self.dense_index(entity).is_some()
    }

    /// Stable between structural boundaries and meaningless across them.
    pub fn dense_index(&self, entity: Entity) -> Option<usize> {
        if entity.scene() != self.scene {
            return None;
        }
        let key = entity.key();
        let entry = (*self.sparse.get(key.slot() as usize)?)?;
        (entry.generation == key.generation()).then_some(entry.dense as usize)
    }

    pub fn entity_at(&self, dense: usize) -> Option<Entity> {
        self.keys
            .get(dense)
            .map(|&key| Entity::new(self.scene, key))
    }

    pub fn get(&self, entity: Entity) -> Option<&T> {
        self.dense_index(entity).map(|dense| &self.dense[dense])
    }

    pub fn get_mut(&mut self, entity: Entity) -> Option<&mut T> {
        let dense = self.dense_index(entity)?;
        if let Some(tracking) = &mut self.tracking {
            tracking.touch(dense);
        }
        Some(&mut self.dense[dense])
    }

    /// Two disjoint mutable rows of one store; equal handles yield `None`.
    pub fn pair_mut(&mut self, _a: Entity, _b: Entity) -> Option<(&mut T, &mut T)> {
        todo!()
    }

    pub fn rows(&self) -> &[T] {
        &self.dense
    }

    pub fn iter(&self) -> impl Iterator<Item = (Entity, &T)> {
        let scene = self.scene;
        self.keys
            .iter()
            .zip(&self.dense)
            .map(move |(&key, row)| (Entity::new(scene, key), row))
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (Entity, &mut T)> {
        if let Some(tracking) = &mut self.tracking {
            tracking.touch_all();
        }
        let scene = self.scene;
        self.keys
            .iter()
            .zip(&mut self.dense)
            .map(move |(&key, row)| (Entity::new(scene, key), row))
    }

    pub fn insert(&mut self, entity: Entity, row: T) -> Result<(), StoreError> {
        if entity.scene() != self.scene {
            return Err(StoreError::Foreign(entity));
        }
        let key = entity.key();
        let slot = key.slot() as usize;
        if slot >= self.sparse.len() {
            self.sparse.resize(slot + 1, None);
        }
        if self.sparse[slot].is_some() {
            return Err(StoreError::Occupied(entity));
        }
        let dense = self.dense.len();
        let dense32 = u32::try_from(dense).map_err(|_| StoreError::Capacity)?;
        self.sparse[slot] = Some(SparseEntry {
            generation: key.generation(),
            dense: dense32,
        });
        self.dense.push(row);
        self.keys.push(key);
        if let Some(tracking) = &mut self.tracking {
            tracking.touch(dense);
        }
        Ok(())
    }

    pub fn remove(&mut self, entity: Entity) -> Result<T, StoreError> {
        let dense = self
            .dense_index(entity)
            .ok_or(StoreError::Missing(entity))?;
        let key = entity.key();
        self.sparse[key.slot() as usize] = None;
        let row = self.dense.swap_remove(dense);
        self.keys.swap_remove(dense);
        if let Some(&moved) = self.keys.get(dense) {
            if let Some(entry) = &mut self.sparse[moved.slot() as usize] {
                entry.dense = dense as u32;
            }
        }
        if let Some(tracking) = &mut self.tracking {
            tracking.remove(key);
        }
        Ok(row)
    }

    /// `cuts` must be strictly ascending and within `len`.
    pub fn partition<'a>(
        &'a mut self,
        _cuts: &'a [usize],
    ) -> Result<Partition<'a, T>, PartitionError> {
        todo!()
    }

    pub fn version(&self, _entity: Entity) -> Option<Version> {
        todo!()
    }

    pub fn changes(&self, _cursor: &mut Cursor) -> Changes<'_, T> {
        todo!()
    }
}

impl<T: Publish> Store<T> {
    pub fn publish(&self, _into: &mut RecordBuffer<T::Record>, _stamp: Stamp) {
        todo!()
    }
}

impl<T: Clone + Send + 'static> StoreField for Store<T> {
    type Snapshot = StoreSnapshot<T>;

    fn bind(&mut self, scene: SceneId) {
        Store::bind(self, scene);
    }

    fn snapshot(&self) -> StoreSnapshot<T> {
        todo!()
    }

    fn restore(&mut self, _from: &StoreSnapshot<T>, _scene: SceneId) {
        todo!()
    }

    fn erased(&mut self) -> &mut dyn ErasedStore {
        self
    }
}

impl<T: Send + 'static> ErasedStore for Store<T> {
    fn schema(&self) -> SchemaId {
        SchemaId::of::<T>()
    }

    fn len(&self) -> usize {
        self.dense.len()
    }

    fn is_tracked(&self) -> bool {
        self.tracking.is_some()
    }
}
