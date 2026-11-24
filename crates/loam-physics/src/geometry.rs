use std::collections::hash_map::{DefaultHasher, Entry as MapEntry};
use std::collections::HashMap;
use std::hash::Hasher;

use loam_shape::{Shape, ShapeKind};

use crate::error::EditError;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct GeometryId(u32);

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryRef {
    id: GeometryId,
    version: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderRef {
    kind: ShapeKind,
    geometry: GeometryRef,
}

impl ColliderRef {
    pub fn kind(self) -> ShapeKind {
        self.kind
    }

    pub fn geometry(self) -> GeometryRef {
        self.geometry
    }
}

#[derive(Clone)]
struct Entry {
    version: u32,
    uses: u32,
    hash: u64,
    shape: Option<Shape>,
}

#[derive(Clone, Default)]
pub struct GeometryStore {
    entries: Vec<Entry>,
    free: Vec<u32>,
    index: HashMap<u64, Vec<GeometryId>>,
    buckets: Vec<Vec<GeometryId>>,
}

impl GeometryStore {
    /// A shape equal to one already held shares its entry.
    pub fn prepare(&mut self, shape: Shape) -> ColliderRef {
        let kind = shape.kind();
        let hash = shape_hash(&shape);
        let entries = &mut self.entries;
        let shared = self.index.get(&hash).and_then(|bucket| {
            bucket.iter().copied().find(|id| {
                entries[id.0 as usize]
                    .shape
                    .as_ref()
                    .is_some_and(|prepared| *prepared == shape)
            })
        });
        if let Some(id) = shared {
            let entry = &mut self.entries[id.0 as usize];
            entry.uses += 1;
            let version = entry.version;
            return ColliderRef {
                kind,
                geometry: GeometryRef { id, version },
            };
        }

        let slot = match self.free.pop() {
            Some(slot) => {
                let entry = &mut self.entries[slot as usize];
                entry.uses = 1;
                entry.hash = hash;
                entry.shape = Some(shape);
                slot
            }
            None => {
                self.entries.push(Entry {
                    version: 0,
                    uses: 1,
                    hash,
                    shape: Some(shape),
                });
                (self.entries.len() - 1) as u32
            }
        };
        let id = GeometryId(slot);
        let spare = self.buckets.pop().unwrap_or_default();
        match self.index.entry(hash) {
            MapEntry::Occupied(mut held) => {
                held.get_mut().push(id);
                self.buckets.push(spare);
            }
            MapEntry::Vacant(empty) => {
                let mut bucket = spare;
                bucket.push(id);
                empty.insert(bucket);
            }
        }
        ColliderRef {
            kind,
            geometry: GeometryRef {
                id,
                version: self.entries[slot as usize].version,
            },
        }
    }

    pub fn release(&mut self, collider: ColliderRef) {
        let slot = collider.geometry.id.0 as usize;
        let Some(entry) = self.entries.get_mut(slot) else {
            return;
        };
        if entry.version != collider.geometry.version || entry.uses == 0 {
            return;
        }
        entry.uses -= 1;
        if entry.uses > 0 {
            return;
        }
        entry.shape = None;
        let hash = entry.hash;
        if let Some(bucket) = self.index.get_mut(&hash) {
            bucket.retain(|held| *held != collider.geometry.id);
            if bucket.is_empty() {
                if let Some(spare) = self.index.remove(&hash) {
                    self.buckets.push(spare);
                }
            }
        }
        let entry = &mut self.entries[slot];
        if let Some(next) = entry.version.checked_add(1) {
            entry.version = next;
            self.free.push(slot as u32);
        }
    }

    pub fn get(&self, collider: ColliderRef) -> Option<&Shape> {
        let entry = self.entries.get(collider.geometry.id.0 as usize)?;
        if entry.version != collider.geometry.version {
            return None;
        }
        entry.shape.as_ref()
    }

    pub fn prepared(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.shape.is_some())
            .count()
    }

    pub(crate) fn validate(
        &self,
        colliders: impl Iterator<Item = ColliderRef>,
    ) -> Result<(), EditError> {
        if self.buckets.iter().any(|bucket| !bucket.is_empty()) {
            return Err(EditError::InvalidGeometry);
        }
        let mut free = vec![false; self.entries.len()];
        for &slot in &self.free {
            let Some(entry) = self.entries.get(slot as usize) else {
                return Err(EditError::InvalidGeometry);
            };
            if free[slot as usize] || entry.shape.is_some() || entry.uses != 0 {
                return Err(EditError::InvalidGeometry);
            }
            free[slot as usize] = true;
        }
        let mut indexed = vec![false; self.entries.len()];
        for (&hash, bucket) in &self.index {
            if bucket.is_empty() {
                return Err(EditError::InvalidGeometry);
            }
            for &id in bucket {
                let Some(entry) = self.entries.get(id.0 as usize) else {
                    return Err(EditError::InvalidGeometry);
                };
                if indexed[id.0 as usize]
                    || entry.hash != hash
                    || entry.shape.is_none()
                    || entry
                        .shape
                        .as_ref()
                        .is_some_and(|shape| shape_hash(shape) != hash)
                {
                    return Err(EditError::InvalidGeometry);
                }
                indexed[id.0 as usize] = true;
            }
        }
        let mut uses = vec![0_u32; self.entries.len()];
        for collider in colliders {
            let Some(entry) = self.entries.get(collider.geometry.id.0 as usize) else {
                return Err(EditError::InvalidGeometry);
            };
            if entry.version != collider.geometry.version
                || entry.shape.as_ref().map(Shape::kind) != Some(collider.kind)
            {
                return Err(EditError::InvalidGeometry);
            }
            let Some(count) = uses[collider.geometry.id.0 as usize].checked_add(1) else {
                return Err(EditError::InvalidGeometry);
            };
            uses[collider.geometry.id.0 as usize] = count;
        }
        for (slot, entry) in self.entries.iter().enumerate() {
            if entry.shape.is_some() {
                if free[slot] || !indexed[slot] || entry.uses == 0 || uses[slot] != entry.uses {
                    return Err(EditError::InvalidGeometry);
                }
            } else if entry.uses != 0
                || indexed[slot]
                || uses[slot] != 0
                || (entry.version != u32::MAX && !free[slot])
            {
                return Err(EditError::InvalidGeometry);
            }
        }
        Ok(())
    }
}

fn shape_hash(shape: &Shape) -> u64 {
    let mut hasher = DefaultHasher::new();
    shape.hash_into(&mut hasher);
    hasher.finish()
}
