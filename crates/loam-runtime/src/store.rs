use std::any::type_name;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::entity::{Entities, Entity, EntityKey, SceneId};
use crate::relation::LinkId;
use crate::session::Stamp;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoreError {
    Foreign(Entity),
    Stale(Entity),
    Occupied(Entity),
    Missing(Entity),
    Unlinked(LinkId),
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

const CURSOR_EXPIRY_BOUNDARIES: u64 = 8;

/// A consumer's position in a store's logs; the default cursor is stale and forces a resync.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Cursor {
    position: u64,
    removed: u64,
    boundary: u64,
    synced: bool,
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
    store: &'a Store<T>,
    resync: bool,
    removals: Range<u64>,
    dirty: Range<u64>,
    rows: Range<usize>,
}

impl<T> Changes<'_, T> {
    pub fn is_resync(&self) -> bool {
        self.resync
    }
}

impl<'a, T> Iterator for Changes<'a, T> {
    type Item = Change<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let store = self.store;
        if let Some(tracking) = &store.tracking {
            for seq in &mut self.removals {
                if let Some(removal) = tracking.removals.get(seq) {
                    return Some(Change::Removed(removal));
                }
            }
            for seq in &mut self.dirty {
                let Some(key) = tracking.dirty.get(seq) else {
                    continue;
                };
                let Some(dense) = store.dense_of(key) else {
                    continue;
                };
                if tracking.logged[dense] == seq {
                    return Some(Change::Row(
                        Entity::new(store.scene, key),
                        &store.dense[dense],
                    ));
                }
            }
        }
        let dense = self.rows.next()?;
        Some(Change::Row(
            Entity::new(store.scene, store.keys[dense]),
            &store.dense[dense],
        ))
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
        Parts {
            rows: self.rows,
            cuts: self.cuts,
            first: 0,
            done: false,
        }
    }
}

pub struct Part<'a, T> {
    pub first: usize,
    pub rows: &'a mut [T],
}

pub struct Parts<'a, T> {
    rows: &'a mut [T],
    cuts: &'a [usize],
    first: usize,
    done: bool,
}

impl<'a, T> Iterator for Parts<'a, T> {
    type Item = Part<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let rows = std::mem::take(&mut self.rows);
        let first = self.first;
        match self.cuts.split_first() {
            Some((&cut, rest)) => {
                let (head, tail) = rows.split_at_mut(cut - first);
                self.rows = tail;
                self.cuts = rest;
                self.first = cut;
                Some(Part { first, rows: head })
            }
            None => {
                self.done = true;
                Some(Part { first, rows })
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PartitionError {
    Unsorted { at: usize },
    OutOfRange { cut: usize, len: usize },
}

const NO_RECORD: u32 = u32::MAX;

/// Stamped records keyed by entity, kept dense with a sparse position index.
pub struct RecordBuffer<R> {
    rows: Vec<R>,
    entities: Vec<Entity>,
    positions: Vec<u32>,
    stamp: Stamp,
}

impl<R> RecordBuffer<R> {
    pub fn rows(&self) -> &[R] {
        &self.rows
    }

    pub fn stamp(&self) -> Stamp {
        self.stamp
    }

    fn position(&self, entity: Entity) -> Option<usize> {
        let position = *self.positions.get(entity.key().slot() as usize)?;
        (position != NO_RECORD && self.entities[position as usize] == entity)
            .then_some(position as usize)
    }

    fn upsert(&mut self, entity: Entity, record: R) {
        if let Some(position) = self.position(entity) {
            self.rows[position] = record;
            return;
        }
        let slot = entity.key().slot() as usize;
        if slot >= self.positions.len() {
            self.positions.resize(slot + 1, NO_RECORD);
        }
        self.positions[slot] = self.rows.len() as u32;
        self.rows.push(record);
        self.entities.push(entity);
    }

    fn clear(&mut self) {
        self.rows.clear();
        self.entities.clear();
        self.positions.fill(NO_RECORD);
    }

    pub(crate) fn restamp(&mut self, stamp: Stamp) {
        self.stamp = stamp;
    }

    pub(crate) fn update(&mut self, entity: Entity, record: R) -> bool {
        let Some(position) = self.position(entity) else {
            return false;
        };
        self.rows[position] = record;
        true
    }

    pub(crate) fn replace(&mut self, records: impl Iterator<Item = (Entity, R)>, stamp: Stamp) {
        self.clear();
        for (entity, record) in records {
            self.upsert(entity, record);
        }
        self.stamp = stamp;
    }
}

impl<R> Default for RecordBuffer<R> {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            entities: Vec::new(),
            positions: Vec::new(),
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

#[derive(Clone, Copy)]
pub struct Owner {
    _private: (),
}

impl Owner {
    pub(crate) fn new() -> Self {
        Self { _private: () }
    }
}

/// A storage-field contract; only `Stores` implementations call its lifetime hooks.
pub trait StoreField: Send + 'static {
    type Snapshot: Send + 'static;

    fn bind(&mut self, scene: SceneId, owner: Owner);

    fn snapshot(&self) -> Self::Snapshot;

    fn restore(&mut self, from: &Self::Snapshot, scene: SceneId, owner: Owner);

    fn boundary(&mut self, _owner: Owner) {}

    fn release(&mut self, _entity: Entity, _owner: Owner) {}
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

struct Ring<E> {
    entries: Vec<E>,
    capacity: usize,
    pushed: u64,
}

impl<E: Copy> Ring<E> {
    fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity),
            capacity,
            pushed: 0,
        }
    }

    fn push(&mut self, entry: E) -> u64 {
        if self.entries.len() < self.capacity {
            self.entries.push(entry);
        } else if self.capacity > 0 {
            self.entries[(self.pushed % self.capacity as u64) as usize] = entry;
        }
        self.pushed += 1;
        self.pushed
    }

    fn get(&self, seq: u64) -> Option<E> {
        let index = seq.checked_sub(1)? % self.capacity.max(1) as u64;
        self.entries.get(index as usize).copied()
    }

    fn retained(&self) -> Range<u64> {
        self.pushed - self.entries.len() as u64 + 1..self.pushed + 1
    }

    fn lost(&self, consumed: u64) -> bool {
        consumed + 1 < self.retained().start
    }

    fn reset(&mut self) {
        self.entries.clear();
        let stride = self.capacity.max(1) as u64;
        self.pushed = self.pushed / stride * stride + stride;
    }
}

struct Tracking {
    capacity: LogCapacity,
    versions: Vec<Version>,
    logged: Vec<u64>,
    dirty: Ring<EntityKey>,
    removals: Ring<Removal>,
    last_read: AtomicU64,
    boundary: u64,
}

impl Tracking {
    fn new(capacity: LogCapacity) -> Self {
        Self {
            capacity,
            versions: Vec::new(),
            logged: Vec::new(),
            dirty: Ring::new(capacity.dirty),
            removals: Ring::new(capacity.removals),
            last_read: AtomicU64::new(0),
            boundary: 0,
        }
    }

    fn push_row(&mut self) {
        self.versions.push(Version::default());
        self.logged.push(0);
    }

    fn touch(&mut self, dense: usize, key: EntityKey) {
        self.versions[dense] = self.versions[dense].bump();
        if self.logged[dense] > self.last_read.load(Ordering::Relaxed) {
            return;
        }
        self.logged[dense] = self.dirty.push(key);
    }

    fn touch_all(&mut self, keys: &[EntityKey]) {
        let last_read = self.last_read.load(Ordering::Relaxed);
        for (dense, &key) in keys.iter().enumerate() {
            self.versions[dense] = self.versions[dense].bump();
            if self.logged[dense] <= last_read {
                self.logged[dense] = self.dirty.push(key);
            }
        }
    }

    fn remove(&mut self, dense: usize, entity: Entity) {
        let version = self.versions.swap_remove(dense);
        self.logged.swap_remove(dense);
        self.removals.push(Removal { entity, version });
    }

    fn stale(&self, cursor: &Cursor) -> bool {
        !cursor.synced
            || self.boundary.saturating_sub(cursor.boundary) > CURSOR_EXPIRY_BOUNDARIES
            || self.dirty.lost(cursor.position)
            || self.removals.lost(cursor.removed)
    }

    fn reset(&mut self, rows: usize) {
        self.versions.clear();
        self.versions.resize(rows, Version::default());
        self.logged.clear();
        self.logged.resize(rows, 0);
        self.dirty.reset();
        self.removals.reset();
    }
}

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
        Self::with_tracking(Some(Tracking::new(capacity)))
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

    pub(crate) fn bind(&mut self, scene: SceneId) {
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
        self.dense_of(entity.key())
    }

    fn dense_of(&self, key: EntityKey) -> Option<usize> {
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
            tracking.touch(dense, self.keys[dense]);
        }
        Some(&mut self.dense[dense])
    }

    /// Two disjoint mutable rows of one store; equal handles yield `None`.
    pub fn pair_mut(&mut self, a: Entity, b: Entity) -> Option<(&mut T, &mut T)> {
        let i = self.dense_index(a)?;
        let j = self.dense_index(b)?;
        if i == j {
            return None;
        }
        if let Some(tracking) = &mut self.tracking {
            tracking.touch(i, self.keys[i]);
            tracking.touch(j, self.keys[j]);
        }
        let (low, high) = self.dense.split_at_mut(i.max(j));
        if i < j {
            Some((&mut low[i], &mut high[0]))
        } else {
            Some((&mut high[0], &mut low[j]))
        }
    }

    pub fn rows(&self) -> &[T] {
        &self.dense
    }

    pub(crate) fn rows_mut_untracked(&mut self) -> &mut [T] {
        &mut self.dense
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
            tracking.touch_all(&self.keys);
        }
        let scene = self.scene;
        self.keys
            .iter()
            .zip(&mut self.dense)
            .map(move |(&key, row)| (Entity::new(scene, key), row))
    }

    pub fn insert(
        &mut self,
        entities: &Entities,
        entity: Entity,
        row: T,
    ) -> Result<(), StoreError> {
        if entity.scene() != entities.scene() {
            return Err(StoreError::Foreign(entity));
        }
        if entities.resolve(entity).is_none() {
            return Err(StoreError::Stale(entity));
        }
        if self.scene == SceneId::UNBOUND {
            self.scene = entities.scene();
        }
        self.insert_raw(entity, row)
    }

    pub(crate) fn insert_raw(&mut self, entity: Entity, row: T) -> Result<(), StoreError> {
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
            tracking.push_row();
            tracking.touch(dense, key);
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
            tracking.remove(dense, entity);
        }
        Ok(row)
    }

    /// `cuts` must be strictly ascending and within `len`.
    pub fn partition<'a>(
        &'a mut self,
        cuts: &'a [usize],
    ) -> Result<Partition<'a, T>, PartitionError> {
        let len = self.dense.len();
        let mut previous = None;
        for (at, &cut) in cuts.iter().enumerate() {
            if cut > len {
                return Err(PartitionError::OutOfRange { cut, len });
            }
            if previous.is_some_and(|previous| cut <= previous) {
                return Err(PartitionError::Unsorted { at });
            }
            previous = Some(cut);
        }
        if let Some(tracking) = &mut self.tracking {
            tracking.touch_all(&self.keys);
        }
        Ok(Partition {
            rows: &mut self.dense,
            cuts,
        })
    }

    /// Counts toward cursor expiry; the session calls it once per boundary.
    pub fn boundary(&mut self) {
        if let Some(tracking) = &mut self.tracking {
            tracking.boundary += 1;
        }
    }

    /// `None` for an untracked store or an entity it does not hold.
    pub fn version(&self, entity: Entity) -> Option<Version> {
        let dense = self.dense_index(entity)?;
        self.tracking
            .as_ref()
            .map(|tracking| tracking.versions[dense])
    }

    pub fn changes(&self, cursor: &mut Cursor) -> Changes<'_, T> {
        let Some(tracking) = &self.tracking else {
            *cursor = Cursor {
                synced: true,
                ..Cursor::default()
            };
            return Changes {
                store: self,
                resync: true,
                removals: 0..0,
                dirty: 0..0,
                rows: 0..self.dense.len(),
            };
        };
        let resync = tracking.stale(cursor);
        let removals = if resync {
            tracking.removals.retained()
        } else {
            cursor.removed + 1..tracking.removals.pushed + 1
        };
        let dirty = if resync {
            0..0
        } else {
            cursor.position + 1..tracking.dirty.pushed + 1
        };
        let rows = if resync { 0..self.dense.len() } else { 0..0 };
        *cursor = Cursor {
            position: tracking.dirty.pushed,
            removed: tracking.removals.pushed,
            boundary: tracking.boundary,
            synced: true,
        };
        tracking
            .last_read
            .fetch_max(tracking.dirty.pushed, Ordering::Relaxed);
        Changes {
            store: self,
            resync,
            removals,
            dirty,
            rows,
        }
    }

    /// An untracked store or an expired cursor counts as changed; a clean check renews the cursor's expiry.
    pub fn changed_since(&self, cursor: &mut Cursor) -> bool {
        let Some(tracking) = &self.tracking else {
            return true;
        };
        if tracking.stale(cursor)
            || cursor.position != tracking.dirty.pushed
            || cursor.removed != tracking.removals.pushed
        {
            return true;
        }
        cursor.boundary = tracking.boundary;
        false
    }

    pub fn catch_up(&self, cursor: &mut Cursor) {
        let _ = self.changes(cursor);
    }
}

impl<T: Clone + Send + 'static> StoreField for Store<T> {
    type Snapshot = StoreSnapshot<T>;

    fn bind(&mut self, scene: SceneId, _owner: Owner) {
        Store::bind(self, scene);
    }

    fn snapshot(&self) -> StoreSnapshot<T> {
        StoreSnapshot {
            rows: self.dense.clone(),
            keys: self.keys.clone(),
        }
    }

    fn restore(&mut self, from: &StoreSnapshot<T>, scene: SceneId, _owner: Owner) {
        self.scene = scene;
        self.dense.clone_from(&from.rows);
        self.keys.clone_from(&from.keys);
        let slots = self
            .keys
            .iter()
            .map(|key| key.slot() as usize + 1)
            .max()
            .unwrap_or(0);
        self.sparse.clear();
        self.sparse.resize(slots, None);
        for (dense, key) in self.keys.iter().enumerate() {
            self.sparse[key.slot() as usize] = Some(SparseEntry {
                generation: key.generation(),
                dense: dense as u32,
            });
        }
        if let Some(tracking) = &mut self.tracking {
            tracking.reset(self.dense.len());
        }
    }

    fn boundary(&mut self, _owner: Owner) {
        Store::boundary(self);
    }

    fn release(&mut self, entity: Entity, _owner: Owner) {
        let _ = Store::remove(self, entity);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::alloc::System;
    use std::collections::BTreeMap;

    use loam_time::alloc::{bytes_allocated_by, CountingAllocator};

    use super::*;
    use crate::entity::{Entities, Epoch, RuntimeId};
    use crate::relation::Relation;

    #[global_allocator]
    static COUNTING_ALLOCATOR: CountingAllocator<System> = CountingAllocator::new(System);

    const SMALL: LogCapacity = LogCapacity {
        dirty: 4,
        removals: 2,
    };

    fn entities() -> Entities {
        Entities::new(SceneId {
            runtime: RuntimeId::allocate(),
            epoch: Epoch::default(),
        })
    }

    fn filled(count: u32, capacity: Option<LogCapacity>) -> (Entities, Store<u32>, Vec<Entity>) {
        let entities = entities();
        let mut store = capacity.map_or_else(Store::untracked, Store::tracked);
        StoreField::bind(&mut store, entities.scene(), Owner::new());
        let mut entities = entities;
        let spawned: Vec<Entity> = (0..count)
            .map(|value| {
                let entity = entities.spawn();
                store.insert(&entities, entity, value).unwrap();
                entity
            })
            .collect();
        (entities, store, spawned)
    }

    #[derive(Default)]
    struct Mirror {
        rows: BTreeMap<Entity, u32>,
        cursor: Cursor,
    }

    impl Mirror {
        fn sync(&mut self, store: &Store<u32>) -> (bool, Vec<Entity>) {
            let changes = store.changes(&mut self.cursor);
            let resync = changes.is_resync();
            if resync {
                self.rows.clear();
            }
            let mut seen = Vec::new();
            for change in changes {
                match change {
                    Change::Row(entity, &value) => {
                        seen.push(entity);
                        self.rows.insert(entity, value);
                    }
                    Change::Removed(removal) => {
                        self.rows.remove(&removal.entity);
                    }
                }
            }
            (resync, seen)
        }
    }

    fn live(store: &Store<u32>) -> BTreeMap<Entity, u32> {
        store
            .iter()
            .map(|(entity, &value)| (entity, value))
            .collect()
    }

    #[test]
    fn pair_mut_and_partition_never_alias_and_never_refuse_distinct_rows() {
        let (_, mut store, e) = filled(5, None);
        assert!(store.pair_mut(e[1], e[1]).is_none());
        let (a, b) = store.pair_mut(e[3], e[1]).unwrap();
        *a += 10;
        *b += 20;
        assert_eq!(store.rows(), [0, 21, 2, 13, 4]);

        assert_eq!(
            store.partition(&[1, 6]).err(),
            Some(PartitionError::OutOfRange { cut: 6, len: 5 })
        );
        assert_eq!(
            store.partition(&[2, 2]).err(),
            Some(PartitionError::Unsorted { at: 1 })
        );
        let partition = store.partition(&[1, 3]).unwrap();
        assert_eq!(partition.part_count(), 3);
        let mut covered = Vec::new();
        for (index, part) in partition.parts().enumerate() {
            for (offset, row) in part.rows.iter_mut().enumerate() {
                *row = index as u32;
                covered.push(part.first + offset);
            }
        }
        assert_eq!(covered, [0, 1, 2, 3, 4]);
        assert_eq!(store.rows(), [0, 1, 1, 2, 2]);
    }

    #[test]
    fn swap_remove_keeps_every_survivor_once_in_iteration_and_the_sparse_index() {
        let (_, mut store, e) = filled(5, None);
        assert_eq!(store.remove(e[1]), Ok(1));
        assert_eq!(store.remove(e[1]), Err(StoreError::Missing(e[1])));
        let mut seen: Vec<u32> = store.iter().map(|(_, &value)| value).collect();
        seen.sort_unstable();
        assert_eq!(seen, [0, 2, 3, 4]);
        for dense in 0..store.len() {
            let entity = store.entity_at(dense).unwrap();
            assert_eq!(store.dense_index(entity), Some(dense));
            assert_eq!(store.get(entity), Some(&store.rows()[dense]));
        }
        assert_eq!(store.get(e[4]), Some(&4));
        assert!(!store.contains(e[1]));
    }

    #[test]
    fn deleting_the_last_rows_keeps_their_removals() {
        let retaining = LogCapacity {
            dirty: SMALL.dirty,
            removals: 3,
        };
        let (_, mut store, e) = filled(3, Some(retaining));
        let mut mirror = Mirror::default();
        mirror.sync(&store);
        assert_eq!(mirror.rows, live(&store));
        for &entity in [e[2], e[0], e[1]].iter() {
            store.remove(entity).unwrap();
        }
        let mut cursor = mirror.cursor;
        let changes = store.changes(&mut cursor);
        assert!(!changes.is_resync());
        let mut removed: Vec<Entity> = changes
            .map(|change| match change {
                Change::Removed(removal) => removal.entity,
                Change::Row(entity, _) => panic!("{entity:?} was published after its removal"),
            })
            .collect();
        removed.sort();
        assert_eq!(removed, e);
        assert!(store.is_empty());
    }

    #[test]
    fn churn_or_cursor_overflow_forces_a_resync_that_rebuilds_the_consumer() {
        let (mut entities, mut store, e) = filled(3, Some(SMALL));
        let mut mirror = Mirror::default();
        assert!(mirror.sync(&store).0);
        for _ in 0..3 {
            store.boundary();
        }
        *store.get_mut(e[1]).unwrap() = 11;
        *store.get_mut(e[1]).unwrap() = 12;
        let (resync, seen) = mirror.sync(&store);
        assert!(!resync);
        assert_eq!(seen, [e[1]]);
        assert_eq!(mirror.rows, live(&store));

        for value in 0..SMALL.dirty as u32 + 1 {
            let entity = entities.spawn();
            store.insert(&entities, entity, 100 + value).unwrap();
        }
        store.remove(e[0]).unwrap();
        assert!(mirror.sync(&store).0);
        assert_eq!(mirror.rows, live(&store));

        for _ in 0..=CURSOR_EXPIRY_BOUNDARIES {
            store.boundary();
        }
        assert!(mirror.sync(&store).0);
        store.boundary();
        assert!(!mirror.sync(&store).0);

        for &entity in &e[1..] {
            store.remove(entity).unwrap();
        }
        store.remove(store.entity_at(0).unwrap()).unwrap();
        assert!(mirror.sync(&store).0);
        assert_eq!(mirror.rows, live(&store));
    }

    #[test]
    fn old_generation_removal_never_deletes_a_reused_slots_new_object() {
        let (mut entities, mut store, e) = filled(1, Some(SMALL));
        let mut mirror = Mirror::default();
        mirror.sync(&store);
        store.remove(e[0]).unwrap();
        entities.despawn(e[0]).unwrap();
        let reused = entities.spawn();
        assert_eq!(reused.key().slot(), e[0].key().slot());
        store.insert(&entities, reused, 7).unwrap();

        let mut cursor = mirror.cursor;
        let removals: Vec<Removal> = store
            .changes(&mut cursor)
            .filter_map(|change| match change {
                Change::Removed(removal) => Some(removal),
                Change::Row(..) => None,
            })
            .collect();
        assert_eq!(removals.len(), 1);
        assert_eq!(removals[0].entity, e[0]);
        assert_ne!(removals[0].entity, reused);
        mirror.sync(&store);
        assert_eq!(mirror.rows.get(&reused), Some(&7));
        assert_eq!(mirror.rows.len(), 1);
    }

    #[test]
    fn warm_read_iterate_pair_and_partition_paths_allocate_nothing() {
        let (_, mut store, e) = filled(64, Some(LogCapacity::default()));
        let mut cursor = Cursor::default();
        let mut warm = |store: &mut Store<u32>| {
            for (_, row) in store.iter_mut() {
                *row += 1;
            }
            store.changes(&mut cursor).count()
        };
        warm(&mut store);
        assert_eq!(warm(&mut store), 64);

        let bytes = bytes_allocated_by(|| {
            for _ in 0..16 {
                let mut sum = 0u32;
                for &entity in &e {
                    sum = sum.wrapping_add(*store.get(entity).unwrap());
                }
                for (_, row) in store.iter() {
                    sum = sum.wrapping_add(*row);
                }
                let (a, b) = store.pair_mut(e[0], e[63]).unwrap();
                *a = sum;
                *b += 1;
                for part in store.partition(&[16, 48]).unwrap().parts() {
                    for row in part.rows {
                        *row += 1;
                    }
                }
                warm(&mut store);
            }
        });
        assert_eq!(
            bytes, 0,
            "16 warmed passes asked the allocator for {bytes} bytes"
        );
    }

    #[test]
    fn edits_and_removals_after_a_restore_reach_a_resynchronized_cursor() {
        const RETAINING: LogCapacity = LogCapacity {
            dirty: 3,
            removals: 2,
        };
        let (mut entities, mut store, e) = filled(3, Some(RETAINING));
        let mut mirror = Mirror::default();
        mirror.sync(&store);
        let (snapshot, rows) = (entities.snapshot(), store.snapshot());

        entities.restore(&snapshot);
        let scene = entities.scene();
        StoreField::restore(&mut store, &rows, scene, Owner::new());
        let rebased: Vec<Entity> = e
            .iter()
            .map(|entity| Entity::new(scene, entity.key()))
            .collect();
        assert!(mirror.sync(&store).0, "the restore never forced a resync");

        *store.get_mut(rebased[0]).unwrap() = 40;
        *store.get_mut(rebased[1]).unwrap() = 41;
        store.remove(rebased[2]).unwrap();
        let (resync, seen) = mirror.sync(&store);
        assert!(!resync, "a second resync hid the lost log entries");
        assert_eq!(seen, [rebased[0], rebased[1]]);
        assert_eq!(mirror.rows, live(&store));
    }

    #[test]
    fn restore_advances_the_epoch_so_old_handles_fail_and_relations_survive() {
        let (mut entities, mut store, e) = filled(2, Some(SMALL));
        let mut relation = Relation::<u8>::new();
        StoreField::bind(&mut relation, entities.scene(), Owner::new());
        let link = relation.link(&entities, e[0], e[1], 5).unwrap();
        let mut mirror = Mirror::default();
        mirror.sync(&store);
        let (snapshot, rows, links) = (entities.snapshot(), store.snapshot(), relation.snapshot());

        let extra = entities.spawn();
        store.insert(&entities, extra, 9).unwrap();
        relation.unlink(link).unwrap();
        *store.get_mut(e[0]).unwrap() = 8;

        entities.restore(&snapshot);
        let scene = entities.scene();
        assert_eq!(scene.epoch(), Epoch::default().advance());
        StoreField::restore(&mut store, &rows, scene, Owner::new());
        StoreField::restore(&mut relation, &links, scene, Owner::new());

        assert_eq!(store.scene(), scene);
        assert_eq!(store.get(e[0]), None);
        assert_eq!(store.get(extra), None);
        assert_eq!(relation.outgoing(e[0]).count(), 0);
        assert!(relation.get(link).is_none());

        let rebased: Vec<Entity> = e
            .iter()
            .map(|entity| Entity::new(scene, entity.key()))
            .collect();
        assert_eq!(store.get(rebased[0]), Some(&0));
        assert_eq!(store.len(), 2);
        let survivor = relation.outgoing(rebased[0]).next().unwrap();
        let survivor = relation.get(survivor).unwrap();
        assert_eq!(
            (survivor.from(), survivor.to(), survivor.data),
            (rebased[0], rebased[1], 5)
        );
        assert_eq!(store.get(survivor.to()), Some(&1));
        assert!(mirror.sync(&store).0);
        assert_eq!(mirror.rows, live(&store));
    }
}
