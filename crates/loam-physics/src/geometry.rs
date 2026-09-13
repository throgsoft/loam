use std::collections::hash_map::{DefaultHasher, Entry as MapEntry};
use std::collections::HashMap;
use std::hash::Hasher;

use crate::collider::{Collider, ColliderKind};
use crate::edit::EditError;

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct GeometryId(u32);

impl GeometryId {
    pub fn index(self) -> u32 {
        self.0
    }
}

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GeometryRef {
    id: GeometryId,
    version: u32,
}

impl GeometryRef {
    pub fn id(self) -> GeometryId {
        self.id
    }

    pub fn version(self) -> u32 {
        self.version
    }
}

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ColliderRef {
    kind: ColliderKind,
    geometry: GeometryRef,
}

impl ColliderRef {
    pub fn kind(self) -> ColliderKind {
        self.kind
    }

    pub fn geometry(self) -> GeometryRef {
        self.geometry
    }
}

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
struct Entry {
    version: u32,
    uses: u32,
    hash: u64,
    shape: Option<Collider>,
}

const RELEASED_BUFFERS: usize = 2;

#[cfg_attr(feature = "persist", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, Default)]
pub struct GeometryStore {
    entries: Vec<Entry>,
    free: Vec<u32>,
    released: Vec<Collider>,
    index: HashMap<u64, Vec<GeometryId>>,
    buckets: Vec<Vec<GeometryId>>,
}

impl GeometryStore {
    /// A shape equal to one already held shares its entry, and the duplicate joins the released pool.
    pub fn prepare(&mut self, shape: Collider) -> ColliderRef {
        let kind = shape.kind();
        let hash = shape_hash(&shape);
        let entries = &mut self.entries;
        let shared = self.index.get(&hash).and_then(|bucket| {
            bucket.iter().copied().find(|id| {
                entries[id.0 as usize]
                    .shape
                    .as_ref()
                    .is_some_and(|prepared| same_shape(prepared, &shape))
            })
        });
        if let Some(id) = shared {
            let entry = &mut self.entries[id.0 as usize];
            entry.uses += 1;
            let version = entry.version;
            self.stash(shape);
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
        let freed = entry.shape.take();
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
        if let Some(freed) = freed {
            self.stash(freed);
        }
    }

    pub(crate) fn stash(&mut self, shape: Collider) {
        if self.released.len() == RELEASED_BUFFERS {
            self.released.remove(0);
        }
        self.released.push(shape);
    }

    pub fn take_released(&mut self) -> Option<Collider> {
        self.released.pop()
    }

    pub fn get(&self, collider: ColliderRef) -> Option<&Collider> {
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
        if self.released.len() > RELEASED_BUFFERS
            || self.buckets.iter().any(|bucket| !bucket.is_empty())
        {
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
                || entry.shape.as_ref().map(Collider::kind) != Some(collider.kind)
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

fn shape_hash(shape: &Collider) -> u64 {
    let mut hasher = DefaultHasher::new();
    hasher.write_u8(kind_tag(shape));
    match shape {
        Collider::Sphere { center, radius } => {
            hash_f32s(&mut hasher, center.as_ref());
            hasher.write_u32(radius.to_bits());
        }
        Collider::HyperSphere4D { center, radius } => {
            hash_f32s(&mut hasher, center.as_ref());
            hasher.write_u32(radius.to_bits());
        }
        Collider::HalfSpace { normal, offset } => {
            hash_f32s(&mut hasher, normal.as_ref());
            hasher.write_u32(offset.to_bits());
        }
        Collider::HalfSpace4D { normal, offset } => {
            hash_f32s(&mut hasher, normal.as_ref());
            hasher.write_u32(offset.to_bits());
        }
        Collider::Box3 { half_extents } => hash_f32s(&mut hasher, half_extents.as_ref()),
        Collider::Polygon2D { vertices } => {
            for v in vertices {
                hash_f32s(&mut hasher, v.as_ref());
            }
        }
        Collider::ConvexPolytope3D { vertices } => {
            for v in vertices {
                hash_f32s(&mut hasher, v.as_ref());
            }
        }
        Collider::ConvexPolytope4D { vertices } => {
            for v in vertices {
                hash_f32s(&mut hasher, v.as_ref());
            }
        }
    }
    hasher.finish()
}

fn hash_f32s(hasher: &mut DefaultHasher, values: &[f32]) {
    for value in values {
        hasher.write_u32(value.to_bits());
    }
}

fn kind_tag(shape: &Collider) -> u8 {
    match shape.kind() {
        ColliderKind::Sphere => 0,
        ColliderKind::HalfSpace => 1,
        ColliderKind::HalfSpace4D => 2,
        ColliderKind::Box3 => 3,
        ColliderKind::Polygon2D => 4,
        ColliderKind::ConvexPolytope3D => 5,
        ColliderKind::ConvexPolytope4D => 6,
        ColliderKind::HyperSphere4D => 7,
    }
}

fn same_shape(a: &Collider, b: &Collider) -> bool {
    match (a, b) {
        (
            Collider::Sphere {
                center: ca,
                radius: ra,
            },
            Collider::Sphere {
                center: cb,
                radius: rb,
            },
        ) => ca == cb && ra == rb,
        (
            Collider::HyperSphere4D {
                center: ca,
                radius: ra,
            },
            Collider::HyperSphere4D {
                center: cb,
                radius: rb,
            },
        ) => ca == cb && ra == rb,
        (
            Collider::HalfSpace {
                normal: na,
                offset: oa,
            },
            Collider::HalfSpace {
                normal: nb,
                offset: ob,
            },
        ) => na == nb && oa == ob,
        (
            Collider::HalfSpace4D {
                normal: na,
                offset: oa,
            },
            Collider::HalfSpace4D {
                normal: nb,
                offset: ob,
            },
        ) => na == nb && oa == ob,
        (Collider::Box3 { half_extents: a }, Collider::Box3 { half_extents: b }) => a == b,
        (Collider::Polygon2D { vertices: a }, Collider::Polygon2D { vertices: b }) => a == b,
        (
            Collider::ConvexPolytope3D { vertices: a },
            Collider::ConvexPolytope3D { vertices: b },
        ) => a == b,
        (
            Collider::ConvexPolytope4D { vertices: a },
            Collider::ConvexPolytope4D { vertices: b },
        ) => a == b,
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decoded_geometry_with_a_wrong_reference_count_is_rejected() {
        let mut store = GeometryStore::default();
        let collider = store.prepare(Collider::sphere_at_origin(1.0));
        assert!(store.validate(std::iter::once(collider)).is_ok());

        store.entries[0].uses = 2;
        assert_eq!(
            store.validate(std::iter::once(collider)),
            Err(EditError::InvalidGeometry)
        );
    }

    #[test]
    fn a_maximum_version_entry_is_reused_once_then_retired() {
        let mut store = GeometryStore::default();
        let first = store.prepare(Collider::sphere_at_origin(1.0));
        store.entries[0].version = u32::MAX - 1;
        let first = ColliderRef {
            kind: first.kind,
            geometry: GeometryRef {
                id: first.geometry.id,
                version: u32::MAX - 1,
            },
        };

        store.release(first);
        assert!(store.validate(std::iter::empty()).is_ok());
        let last = store.prepare(Collider::sphere_at_origin(2.0));
        assert_eq!(last.geometry.version, u32::MAX);
        store.release(last);
        assert!(store.validate(std::iter::empty()).is_ok());

        let fresh = store.prepare(Collider::sphere_at_origin(3.0));
        assert_ne!(fresh.geometry.id, last.geometry.id);
    }
}
